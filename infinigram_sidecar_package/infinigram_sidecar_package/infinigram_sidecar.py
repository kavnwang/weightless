from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from heapq import heappop, heappush, heapreplace
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view

MASK64 = (1 << 64) - 1
DEFAULT_HASH_BASE1 = 1_315_423_911
DEFAULT_HASH_BASE2 = 2_654_435_761
DEFAULT_SLOT_DTYPE = np.dtype(
    [
        ("occupied", "u1"),
        ("order", "u1"),
        ("length", "u2"),
        ("total", "u4"),
        ("top_count", "u4"),
        ("offset", "u8"),
        ("h1", "u8"),
        ("h2", "u8"),
    ],
    align=True,
)
DEFAULT_PAYLOAD_DTYPE = np.dtype(
    [
        ("token", "u4"),
        ("count", "u4"),
    ],
    align=True,
)
RAW_RECORD_DTYPE = np.dtype(
    [
        ("order", "u1"),
        ("h1", "u8"),
        ("h2", "u8"),
        ("next", "u4"),
    ]
)
REDUCED_RECORD_DTYPE = np.dtype(
    [
        ("order", "u1"),
        ("h1", "u8"),
        ("h2", "u8"),
        ("next", "u4"),
        ("count", "u4"),
    ]
)


def _next_power_of_two(x: int) -> int:
    if x < 1:
        return 1
    return 1 << (x - 1).bit_length()



def _mix_key(order: int, h1: int, h2: int) -> int:
    x = (int(h1) ^ ((int(h2) << 1) & MASK64) ^ ((int(order) + 1) * 0x9E3779B97F4A7C15)) & MASK64
    x ^= x >> 33
    x = (x * 0xFF51AFD7ED558CCD) & MASK64
    x ^= x >> 33
    x = (x * 0xC4CEB9FE1A85EC53) & MASK64
    x ^= x >> 33
    return x & MASK64



def _u64_weights(order: int, base: int) -> np.ndarray:
    weights = np.empty(order, dtype=np.uint64)
    cur = 1
    for i in range(order - 1, -1, -1):
        weights[i] = np.uint64(cur)
        cur = (cur * int(base)) & MASK64
    return weights



def _hash_windows_uint64(windows: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if windows.size == 0:
        return np.empty((0,), dtype=np.uint64)
    windows_u64 = windows.astype(np.uint64, copy=False) + np.uint64(1)
    return (windows_u64 * weights.reshape(1, -1)).sum(axis=1, dtype=np.uint64)



def _sort_reduce_records(records: np.ndarray) -> np.ndarray:
    if records.size == 0:
        return np.empty((0,), dtype=REDUCED_RECORD_DTYPE)

    order = np.argsort(records, order=("order", "h1", "h2", "next"), kind="mergesort")
    sorted_records = records[order]

    boundaries = np.empty(sorted_records.shape[0], dtype=bool)
    boundaries[0] = True
    boundaries[1:] = (
        (sorted_records["order"][1:] != sorted_records["order"][:-1])
        | (sorted_records["h1"][1:] != sorted_records["h1"][:-1])
        | (sorted_records["h2"][1:] != sorted_records["h2"][:-1])
        | (sorted_records["next"][1:] != sorted_records["next"][:-1])
    )
    starts = np.flatnonzero(boundaries)
    counts = np.diff(np.append(starts, sorted_records.shape[0])).astype(np.uint32, copy=False)

    out = np.empty(starts.shape[0], dtype=REDUCED_RECORD_DTYPE)
    out["order"] = sorted_records["order"][starts]
    out["h1"] = sorted_records["h1"][starts]
    out["h2"] = sorted_records["h2"][starts]
    out["next"] = sorted_records["next"][starts]
    out["count"] = counts
    return out



def _strip_state_dict_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        if new_key.startswith("_orig_mod."):
            new_key = new_key[len("_orig_mod.") :]
        out[new_key] = value
    return out


@dataclass(frozen=True)
class SidecarBuildConfig:
    orders: tuple[int, ...] = (4, 6, 8)
    topk: int = 8
    min_count: int = 2
    min_store_confidence: float = 0.0
    load_factor: float = 0.70
    chunk_records: int = 5_000_000
    hash_base1: int = DEFAULT_HASH_BASE1
    hash_base2: int = DEFAULT_HASH_BASE2
    keep_temporary_chunks: bool = False
    pad_token_id: int = 50256

    def __post_init__(self):
        orders = tuple(sorted(set(int(x) for x in self.orders)))
        if not orders:
            raise ValueError("orders cannot be empty")
        if min(orders) < 1:
            raise ValueError(f"orders must be >= 1, got {orders}")
        if max(orders) > 255:
            raise ValueError(f"largest supported order is 255, got {max(orders)}")
        if self.topk < 1:
            raise ValueError(f"topk must be >= 1, got {self.topk}")
        if self.min_count < 1:
            raise ValueError(f"min_count must be >= 1, got {self.min_count}")
        if not (0.1 <= self.load_factor < 0.95):
            raise ValueError(f"load_factor must be in [0.1, 0.95), got {self.load_factor}")
        object.__setattr__(self, "orders", orders)


@dataclass(frozen=True)
class SidecarRuntimeConfig:
    min_order: int = 4
    max_order: int | None = None
    sidecar_weight: float = 0.70
    sidecar_temperature: float = 1.0
    min_model_prob: float = 0.02
    model_topk_agree: int = 8
    require_argmax_agreement: bool = False
    min_sidecar_confidence: float = 0.55
    min_count: int = 2
    max_sidecar_bytes_per_token: int = 256
    apply_to_last_token_only: bool = True

    def __post_init__(self):
        if self.min_order < 1:
            raise ValueError(f"min_order must be >= 1, got {self.min_order}")
        if self.max_order is not None and self.max_order < self.min_order:
            raise ValueError(
                f"max_order must be >= min_order, got max_order={self.max_order}, min_order={self.min_order}"
            )
        if not (0.0 < self.sidecar_weight < 1.0):
            raise ValueError(f"sidecar_weight must be in (0, 1), got {self.sidecar_weight}")
        if self.sidecar_temperature <= 0:
            raise ValueError(f"sidecar_temperature must be > 0, got {self.sidecar_temperature}")
        if self.min_model_prob < 0:
            raise ValueError(f"min_model_prob must be >= 0, got {self.min_model_prob}")
        if self.model_topk_agree < 1:
            raise ValueError(f"model_topk_agree must be >= 1, got {self.model_topk_agree}")
        if self.min_count < 1:
            raise ValueError(f"min_count must be >= 1, got {self.min_count}")
        if self.max_sidecar_bytes_per_token < 0:
            raise ValueError(
                f"max_sidecar_bytes_per_token must be >= 0, got {self.max_sidecar_bytes_per_token}"
            )


@dataclass(frozen=True)
class ContextHit:
    order: int
    total_count: int
    top_count: int
    tokens: np.ndarray
    counts: np.ndarray
    probes: int
    bytes_read: int

    @property
    def top_token(self) -> int:
        return int(self.tokens[0])

    @property
    def top_confidence(self) -> float:
        if self.total_count <= 0:
            return 0.0
        return float(self.top_count) / float(self.total_count)

    def probabilities(self, temperature: float = 1.0) -> np.ndarray:
        probs = self.counts.astype(np.float64, copy=False)
        if temperature != 1.0:
            probs = np.power(probs, 1.0 / temperature, dtype=np.float64)
        probs_sum = probs.sum()
        if probs_sum <= 0:
            return np.full_like(probs, 1.0 / max(1, probs.size), dtype=np.float64)
        return probs / probs_sum


@dataclass
class SidecarStats:
    lookup_attempts: int = 0
    lookup_hits: int = 0
    fused: int = 0
    bytes_read: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


class CompactNgramSidecarBuilder:
    """Builds a compact exact n-gram index with external-sort style chunking.

    The stored structure is deliberately small at inference time:
      - one open-addressed hash-table probe sequence over fixed-size slots
      - one contiguous read of a short candidate list (token, count) pairs

    That keeps the extra bytes-per-token read low compared to suffix-array search.
    """

    def __init__(self, output_dir: str | os.PathLike[str], config: SidecarBuildConfig):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_dir = self.output_dir / "_tmp_chunks"
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self._weights1 = {order: _u64_weights(order, config.hash_base1) for order in config.orders}
        self._weights2 = {order: _u64_weights(order, config.hash_base2) for order in config.orders}
        self._pending: list[np.ndarray] = []
        self._pending_records = 0
        self._chunk_paths: list[Path] = []
        self._chunk_index = 0
        self._stats: dict[str, int] = {
            "sequences": 0,
            "raw_records": 0,
            "reduced_chunk_records": 0,
        }

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def add_batch(self, batch: dict[str, torch.Tensor | np.ndarray]) -> None:
        input_ids = batch["input_ids"]
        labels = batch.get("labels")
        attention_mask = batch.get("attention_mask")

        if isinstance(input_ids, torch.Tensor):
            input_ids_np = input_ids.detach().cpu().numpy()
        else:
            input_ids_np = np.asarray(input_ids)
        labels_np = None
        if labels is not None:
            if isinstance(labels, torch.Tensor):
                labels_np = labels.detach().cpu().numpy()
            else:
                labels_np = np.asarray(labels)
        attention_np = None
        if attention_mask is not None:
            if isinstance(attention_mask, torch.Tensor):
                attention_np = attention_mask.detach().cpu().numpy()
            else:
                attention_np = np.asarray(attention_mask)

        batch_size = int(input_ids_np.shape[0])
        for i in range(batch_size):
            valid_len = int(attention_np[i].sum()) if attention_np is not None else int(input_ids_np.shape[1])
            if valid_len <= 0:
                continue
            row_input = np.asarray(input_ids_np[i, :valid_len], dtype=np.int64)
            row_labels = None if labels_np is None else np.asarray(labels_np[i, :valid_len], dtype=np.int64)
            self.add_sequence(row_input, row_labels)

    def add_sequence(self, input_ids: np.ndarray, labels: np.ndarray | None = None) -> None:
        self._stats["sequences"] += 1
        if labels is None:
            if input_ids.shape[0] < (min(self.config.orders) + 1):
                # Need at least one extra token beyond the shortest context order.
                return
            effective_input = input_ids[:-1]
            next_tokens_full = input_ids[1:].astype(np.uint32, copy=False)
            label_mask_full = np.ones_like(next_tokens_full, dtype=bool)
        else:
            if input_ids.shape[0] < min(self.config.orders):
                return
            effective_input = input_ids
            next_tokens_full = labels.astype(np.int64, copy=False)
            label_mask_full = next_tokens_full != self.config.pad_token_id

        for order in self.config.orders:
            if effective_input.shape[0] < order:
                continue

            contexts = sliding_window_view(effective_input, order)
            next_tokens = next_tokens_full[order - 1 :]
            valid_mask = label_mask_full[order - 1 :]
            if not np.any(valid_mask):
                continue

            contexts = contexts[valid_mask]
            next_tokens = next_tokens[valid_mask]
            if contexts.shape[0] == 0:
                continue

            h1 = _hash_windows_uint64(contexts, self._weights1[order])
            h2 = _hash_windows_uint64(contexts, self._weights2[order])
            records = np.empty(contexts.shape[0], dtype=RAW_RECORD_DTYPE)
            records["order"] = order
            records["h1"] = h1
            records["h2"] = h2
            records["next"] = next_tokens.astype(np.uint32, copy=False)
            self._pending.append(records)
            self._pending_records += int(records.shape[0])
            self._stats["raw_records"] += int(records.shape[0])

            if self._pending_records >= self.config.chunk_records:
                self.flush_chunk()

    def flush_chunk(self) -> None:
        if not self._pending:
            return
        records = np.concatenate(self._pending, axis=0)
        reduced = _sort_reduce_records(records)
        chunk_path = self.chunk_dir / f"chunk_{self._chunk_index:06d}.npy"
        np.save(chunk_path, reduced)
        self._chunk_paths.append(chunk_path)
        self._chunk_index += 1
        self._stats["reduced_chunk_records"] += int(reduced.shape[0])
        self._pending.clear()
        self._pending_records = 0

    def finalize(self) -> dict[str, int | float]:
        self.flush_chunk()
        if not self._chunk_paths:
            raise RuntimeError("No sidecar records were emitted; nothing to finalize.")

        n_contexts, payload_len = self._measure_output_size()
        if n_contexts == 0:
            raise RuntimeError(
                "All contexts were filtered out; try lowering min_count or min_store_confidence."
            )
        table_size = _next_power_of_two(math.ceil(n_contexts / self.config.load_factor))
        mask = table_size - 1

        slot_path = self.output_dir / "slots.npy"
        payload_path = self.output_dir / "payload.npy"
        slots = np.lib.format.open_memmap(
            slot_path,
            mode="w+",
            dtype=DEFAULT_SLOT_DTYPE,
            shape=(table_size,),
        )
        payload = np.lib.format.open_memmap(
            payload_path,
            mode="w+",
            dtype=DEFAULT_PAYLOAD_DTYPE,
            shape=(payload_len,),
        )
        slots[:] = 0

        offset = 0
        occupied = 0
        max_probe_len = 0
        max_payload_len = 0

        for order, h1, h2, total_count, top_count, cand_tokens, cand_counts in self._iter_context_summaries():
            if total_count < self.config.min_count:
                continue
            confidence = float(top_count) / float(total_count)
            if confidence < self.config.min_store_confidence:
                continue
            length = int(len(cand_tokens))
            if length < 1:
                continue

            idx = _mix_key(order, h1, h2) & mask
            probe_len = 1
            while int(slots["occupied"][idx]) != 0:
                idx = (idx + 1) & mask
                probe_len += 1
            max_probe_len = max(max_probe_len, probe_len)
            max_payload_len = max(max_payload_len, length)

            slots["occupied"][idx] = 1
            slots["order"][idx] = order
            slots["length"][idx] = length
            slots["total"][idx] = total_count
            slots["top_count"][idx] = top_count
            slots["offset"][idx] = offset
            slots["h1"][idx] = np.uint64(h1)
            slots["h2"][idx] = np.uint64(h2)

            payload[offset : offset + length]["token"] = np.asarray(cand_tokens, dtype=np.uint32)
            payload[offset : offset + length]["count"] = np.asarray(cand_counts, dtype=np.uint32)
            offset += length
            occupied += 1

        metadata = {
            "orders": list(self.config.orders),
            "topk": self.config.topk,
            "min_count": self.config.min_count,
            "min_store_confidence": self.config.min_store_confidence,
            "hash_base1": self.config.hash_base1,
            "hash_base2": self.config.hash_base2,
            "load_factor": self.config.load_factor,
            "pad_token_id": self.config.pad_token_id,
            "slot_dtype": DEFAULT_SLOT_DTYPE.descr,
            "payload_dtype": DEFAULT_PAYLOAD_DTYPE.descr,
            "slot_itemsize": int(DEFAULT_SLOT_DTYPE.itemsize),
            "payload_itemsize": int(DEFAULT_PAYLOAD_DTYPE.itemsize),
            "n_contexts": int(occupied),
            "payload_len": int(offset),
            "table_size": int(table_size),
            "max_probe_len": int(max_probe_len),
            "max_payload_len": int(max_payload_len),
            "max_lookup_bytes": int(max_probe_len * DEFAULT_SLOT_DTYPE.itemsize + max_payload_len * DEFAULT_PAYLOAD_DTYPE.itemsize),
            "stats": self.stats,
        }
        with open(self.output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        if not self.config.keep_temporary_chunks:
            for chunk_path in self._chunk_paths:
                try:
                    chunk_path.unlink()
                except FileNotFoundError:
                    pass
            try:
                self.chunk_dir.rmdir()
            except OSError:
                pass

        return {
            **self.stats,
            "contexts": int(occupied),
            "payload_len": int(offset),
            "table_size": int(table_size),
            "max_probe_len": int(max_probe_len),
            "max_payload_len": int(max_payload_len),
            "max_lookup_bytes": int(metadata["max_lookup_bytes"]),
        }

    def _measure_output_size(self) -> tuple[int, int]:
        n_contexts = 0
        payload_len = 0
        for _, _, _, total_count, top_count, cand_tokens, _ in self._iter_context_summaries():
            if total_count < self.config.min_count:
                continue
            confidence = float(top_count) / float(total_count)
            if confidence < self.config.min_store_confidence:
                continue
            if len(cand_tokens) == 0:
                continue
            n_contexts += 1
            payload_len += len(cand_tokens)
        return n_contexts, payload_len

    def _iter_context_summaries(
        self,
    ) -> Iterator[tuple[int, int, int, int, int, list[int], list[int]]]:
        current_ctx: tuple[int, int, int] | None = None
        total_count = 0
        heap: list[tuple[int, int, int]] = []

        def flush_current() -> tuple[int, int, int, int, int, list[int], list[int]] | None:
            nonlocal current_ctx, total_count, heap
            if current_ctx is None:
                return None
            candidates = [(tok, count) for count, _neg_tok, tok in heap]
            candidates.sort(key=lambda x: (-x[1], x[0]))
            if not candidates:
                out = None
            else:
                top_count = int(candidates[0][1])
                out = (
                    int(current_ctx[0]),
                    int(current_ctx[1]),
                    int(current_ctx[2]),
                    int(total_count),
                    top_count,
                    [int(tok) for tok, _ in candidates],
                    [int(count) for _tok, count in candidates],
                )
            current_ctx = None
            total_count = 0
            heap = []
            return out

        for order, h1, h2, next_token, count in self._iter_merged_records():
            ctx = (order, h1, h2)
            if current_ctx != ctx:
                flushed = flush_current()
                if flushed is not None:
                    yield flushed
                current_ctx = ctx
            total_count += count
            item = (int(count), -int(next_token), int(next_token))
            if len(heap) < self.config.topk:
                heappush(heap, item)
            elif item > heap[0]:
                heapreplace(heap, item)

        flushed = flush_current()
        if flushed is not None:
            yield flushed

    def _iter_merged_records(self) -> Iterator[tuple[int, int, int, int, int]]:
        arrays = [np.load(path, mmap_mode="r") for path in self._chunk_paths]
        heap: list[tuple[tuple[int, int, int, int], int]] = []
        positions = [0 for _ in arrays]

        for idx, arr in enumerate(arrays):
            if arr.shape[0] == 0:
                continue
            key = (
                int(arr["order"][0]),
                int(arr["h1"][0]),
                int(arr["h2"][0]),
                int(arr["next"][0]),
            )
            heappush(heap, (key, idx))

        while heap:
            key, arr_idx = heappop(heap)
            pos = positions[arr_idx]
            arr = arrays[arr_idx]
            total = int(arr["count"][pos])
            pos += 1
            positions[arr_idx] = pos
            if pos < arr.shape[0]:
                next_key = (
                    int(arr["order"][pos]),
                    int(arr["h1"][pos]),
                    int(arr["h2"][pos]),
                    int(arr["next"][pos]),
                )
                heappush(heap, (next_key, arr_idx))

            while heap and heap[0][0] == key:
                _, arr_idx2 = heappop(heap)
                pos2 = positions[arr_idx2]
                arr2 = arrays[arr_idx2]
                total += int(arr2["count"][pos2])
                pos2 += 1
                positions[arr_idx2] = pos2
                if pos2 < arr2.shape[0]:
                    next_key2 = (
                        int(arr2["order"][pos2]),
                        int(arr2["h1"][pos2]),
                        int(arr2["h2"][pos2]),
                        int(arr2["next"][pos2]),
                    )
                    heappush(heap, (next_key2, arr_idx2))

            yield key[0], key[1], key[2], key[3], total


class CompactNgramIndex:
    def __init__(self, index_dir: str | os.PathLike[str]):
        self.index_dir = Path(index_dir)
        metadata_path = self.index_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing sidecar metadata at {metadata_path}")
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.orders = tuple(int(x) for x in self.metadata["orders"])
        self.topk = int(self.metadata["topk"])
        self.min_count = int(self.metadata["min_count"])
        self.hash_base1 = int(self.metadata["hash_base1"])
        self.hash_base2 = int(self.metadata["hash_base2"])
        self.table_size = int(self.metadata["table_size"])
        self.mask = self.table_size - 1
        self.slot_itemsize = int(self.metadata["slot_itemsize"])
        self.payload_itemsize = int(self.metadata["payload_itemsize"])
        self.max_lookup_bytes = int(self.metadata.get("max_lookup_bytes", 0))
        self.max_probe_len = int(self.metadata.get("max_probe_len", 0))
        self.max_payload_len = int(self.metadata.get("max_payload_len", 0))

        self.slots = np.load(self.index_dir / "slots.npy", mmap_mode="r")
        self.payload = np.load(self.index_dir / "payload.npy", mmap_mode="r")

    def _hash_context(self, context_tokens: Sequence[int], order: int) -> tuple[int, int]:
        h1 = 0
        h2 = 0
        for tok in context_tokens[-order:]:
            x = int(tok) + 1
            h1 = ((h1 * self.hash_base1) + x) & MASK64
            h2 = ((h2 * self.hash_base2) + x) & MASK64
        return h1, h2

    def lookup(
        self,
        context_tokens: Sequence[int],
        *,
        min_order: int = 1,
        max_order: int | None = None,
        max_bytes: int | None = None,
    ) -> ContextHit | None:
        if not context_tokens:
            return None
        max_order_eff = len(context_tokens) if max_order is None else min(int(max_order), len(context_tokens))
        candidate_orders = [o for o in self.orders if min_order <= o <= max_order_eff]
        candidate_orders.sort(reverse=True)
        for order in candidate_orders:
            hit = self._lookup_exact_order(context_tokens, order, max_bytes=max_bytes)
            if hit is not None:
                return hit
        return None

    def _lookup_exact_order(
        self,
        context_tokens: Sequence[int],
        order: int,
        *,
        max_bytes: int | None = None,
    ) -> ContextHit | None:
        h1, h2 = self._hash_context(context_tokens, order)
        idx = _mix_key(order, h1, h2) & self.mask
        probes = 0
        bytes_read = 0
        while probes < self.table_size:
            probes += 1
            bytes_read += self.slot_itemsize
            if max_bytes is not None and bytes_read > max_bytes:
                return None

            slot = self.slots[idx]
            if int(slot["occupied"]) == 0:
                return None
            if (
                int(slot["order"]) == order
                and int(slot["h1"]) == h1
                and int(slot["h2"]) == h2
            ):
                offset = int(slot["offset"])
                length = int(slot["length"])
                bytes_read += length * self.payload_itemsize
                if max_bytes is not None and bytes_read > max_bytes:
                    return None
                payload = self.payload[offset : offset + length]
                return ContextHit(
                    order=order,
                    total_count=int(slot["total"]),
                    top_count=int(slot["top_count"]),
                    tokens=np.asarray(payload["token"], dtype=np.int64),
                    counts=np.asarray(payload["count"], dtype=np.int64),
                    probes=probes,
                    bytes_read=bytes_read,
                )
            idx = (idx + 1) & self.mask
        return None

    def summary(self) -> dict[str, int | list[int]]:
        return {
            "orders": list(self.orders),
            "topk": self.topk,
            "n_contexts": int(self.metadata["n_contexts"]),
            "payload_len": int(self.metadata["payload_len"]),
            "table_size": self.table_size,
            "max_probe_len": self.max_probe_len,
            "max_payload_len": self.max_payload_len,
            "max_lookup_bytes": self.max_lookup_bytes,
        }


class InfinigramSidecarWrapper(nn.Module):
    """Eval-time sidecar that blends a base model with a compact n-gram index.

    It is fully separate from the checkpointed model: the base model is unchanged,
    and the sidecar can be added or removed at evaluation time.
    """

    def __init__(
        self,
        base_model: nn.Module,
        index: CompactNgramIndex,
        runtime_config: SidecarRuntimeConfig,
    ):
        super().__init__()
        self.base_model = base_model
        self.index = index
        self.runtime_config = runtime_config
        self.stats = SidecarStats()
        # Eval outputs are normalized log-probabilities, not raw logits.
        self.returns_log_probs = True

    def reset_stats(self) -> None:
        self.stats = SidecarStats()

    def count_parameters(self, count_zeros: bool = False):
        if hasattr(self.base_model, "count_parameters"):
            return self.base_model.count_parameters(count_zeros=count_zeros)
        if count_zeros:
            return sum(p.numel() for p in self.base_model.parameters())
        return sum((p != 0).sum().item() for p in self.base_model.parameters())

    def get_inference_profile(self, *args, **kwargs):
        if hasattr(self.base_model, "get_inference_profile"):
            return self.base_model.get_inference_profile(*args, **kwargs)
        raise AttributeError("base model does not expose get_inference_profile")

    @property
    def extra_bytes_per_token_infer(self) -> int:
        budget = self.runtime_config.max_sidecar_bytes_per_token
        if budget <= 0:
            return 0
        if self.index.max_lookup_bytes <= 0:
            return budget
        return min(budget, self.index.max_lookup_bytes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        logits = self.base_model(input_ids, attention_mask)
        if self.training:
            return logits
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        if self.runtime_config.apply_to_last_token_only:
            return self._blend_last_position(input_ids, log_probs, attention_mask)
        return self._blend_all_positions(input_ids, log_probs, attention_mask)

    def _blend_last_position(
        self,
        input_ids: torch.Tensor,
        log_probs: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        input_ids_cpu = input_ids.detach().cpu()
        attention_cpu = None if attention_mask is None else attention_mask.detach().cpu()
        out = log_probs
        for b in range(input_ids.shape[0]):
            valid_len = int(attention_cpu[b].sum().item()) if attention_cpu is not None else int(input_ids.shape[1])
            if valid_len < self.runtime_config.min_order:
                continue
            context = input_ids_cpu[b, :valid_len].tolist()
            self.stats.lookup_attempts += 1
            hit = self.index.lookup(
                context,
                min_order=self.runtime_config.min_order,
                max_order=self.runtime_config.max_order,
                max_bytes=self.runtime_config.max_sidecar_bytes_per_token,
            )
            if hit is None:
                continue
            self.stats.lookup_hits += 1
            self.stats.bytes_read += hit.bytes_read
            row_idx = valid_len - 1
            fused_row = self._maybe_fuse_row(log_probs[b, row_idx], hit)
            if fused_row is not None:
                if out is log_probs:
                    out = log_probs.clone()
                out[b, row_idx] = fused_row
                self.stats.fused += 1
        return out

    def _blend_all_positions(
        self,
        input_ids: torch.Tensor,
        log_probs: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        input_ids_cpu = input_ids.detach().cpu()
        attention_cpu = None if attention_mask is None else attention_mask.detach().cpu()
        out = log_probs
        for b in range(input_ids.shape[0]):
            valid_len = int(attention_cpu[b].sum().item()) if attention_cpu is not None else int(input_ids.shape[1])
            if valid_len < self.runtime_config.min_order:
                continue
            seq = input_ids_cpu[b, :valid_len].tolist()
            for t in range(self.runtime_config.min_order - 1, valid_len):
                context = seq[: t + 1]
                self.stats.lookup_attempts += 1
                hit = self.index.lookup(
                    context,
                    min_order=self.runtime_config.min_order,
                    max_order=self.runtime_config.max_order,
                    max_bytes=self.runtime_config.max_sidecar_bytes_per_token,
                )
                if hit is None:
                    continue
                self.stats.lookup_hits += 1
                self.stats.bytes_read += hit.bytes_read
                fused_row = self._maybe_fuse_row(log_probs[b, t], hit)
                if fused_row is not None:
                    if out is log_probs:
                        out = log_probs.clone()
                    out[b, t] = fused_row
                    self.stats.fused += 1
        return out

    def _maybe_fuse_row(self, row_log_probs: torch.Tensor, hit: ContextHit) -> torch.Tensor | None:
        if hit.total_count < self.runtime_config.min_count:
            return None
        if hit.top_confidence < self.runtime_config.min_sidecar_confidence:
            return None

        work = row_log_probs.float()
        top_token = int(hit.top_token)
        if self.runtime_config.require_argmax_agreement:
            if int(torch.argmax(work).item()) != top_token:
                return None
        else:
            k = min(int(self.runtime_config.model_topk_agree), int(work.shape[-1]))
            if k >= 1:
                topk_idx = torch.topk(work, k=k, dim=-1).indices
                if not bool((topk_idx == top_token).any().item()):
                    return None

        if self.runtime_config.min_model_prob > 0:
            model_logp = work[top_token]
            if float(model_logp.item()) < math.log(self.runtime_config.min_model_prob):
                return None
        return self._fuse_row(work, row_log_probs.dtype, hit)

    def _fuse_row(self, row_log_probs: torch.Tensor, out_dtype: torch.dtype, hit: ContextHit) -> torch.Tensor:
        candidate_ids = torch.as_tensor(hit.tokens, device=row_log_probs.device, dtype=torch.long)
        side_probs = torch.as_tensor(
            hit.probabilities(temperature=self.runtime_config.sidecar_temperature),
            device=row_log_probs.device,
            dtype=row_log_probs.dtype,
        ).clamp_min(1e-12)
        lam = float(self.runtime_config.sidecar_weight)
        base_log_coeff = math.log1p(-lam)
        side_log_coeff = math.log(lam)

        fused = row_log_probs.clone()
        fused = fused + base_log_coeff
        fused[candidate_ids] = torch.logaddexp(
            fused[candidate_ids],
            torch.log(side_probs) + side_log_coeff,
        )
        return fused.to(dtype=out_dtype)


@torch.no_grad()
def load_model_checkpoint(
    checkpoint_path: str | os.PathLike[str],
    model: nn.Module,
    *,
    strict: bool = True,
) -> nn.Module:
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format at {checkpoint_path}")
    cleaned = _strip_state_dict_prefixes(state)
    model.load_state_dict(cleaned, strict=strict)
    return model
