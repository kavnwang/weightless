"""Build an eval-time InfiniGram sidecar index from training data."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import shutil

from data import get_dataloader
from infinigram_sidecar import CompactNgramSidecarBuilder, SidecarBuildConfig


def _parse_orders(values: list[str]) -> tuple[int, ...]:
    pieces: list[str] = []
    for value in values:
        pieces.extend(part.strip() for part in str(value).split(",") if part.strip())
    orders = tuple(sorted({int(x) for x in pieces}))
    if not orders:
        raise argparse.ArgumentTypeError("orders must contain at least one integer")
    return orders


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a compact eval-time InfiniGram-style sidecar index. "
            "This does not modify model checkpoints."
        )
    )
    parser.add_argument("--out_dir", "--output_dir", dest="out_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--streaming", action="store_true", default=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--no_pin_memory", action="store_true")
    parser.add_argument("--no_persistent_workers", action="store_true")
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument(
        "--orders",
        nargs="+",
        default=["4", "6", "8"],
        help="Context orders to store, e.g. --orders 4 6 8 or --orders 4,6,8",
    )
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--min_count", type=int, default=2)
    parser.add_argument("--min_store_confidence", type=float, default=0.0)
    parser.add_argument("--load_factor", type=float, default=0.70)
    parser.add_argument("--chunk_records", type=int, default=5_000_000)
    parser.add_argument("--pad_token_id", type=int, default=50256)
    parser.add_argument("--keep_temporary_chunks", action="store_true")
    parser.add_argument(
        "--world_size",
        type=int,
        default=None,
        help="Override distributed world size (defaults to WORLD_SIZE env or 1).",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Override distributed rank (defaults to RANK env or 0).",
    )
    parser.add_argument(
        "--dist_timeout_s",
        type=int,
        default=7200,
        help="Seconds rank0 waits for all ranks to finish chunking.",
    )
    return parser


def _resolve_dist(args: argparse.Namespace) -> tuple[int, int]:
    import os

    world_size = int(
        args.world_size if args.world_size is not None else os.environ.get("WORLD_SIZE", "1")
    )
    rank = int(args.rank if args.rank is not None else os.environ.get("RANK", "0"))
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    return rank, world_size


def _wait_for_markers(marker_dir: Path, world_size: int, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    expected = [marker_dir / f"rank_{r:03d}.done" for r in range(world_size)]
    while True:
        if all(path.exists() for path in expected):
            return
        if time.time() > deadline:
            missing = [str(path.name) for path in expected if not path.exists()]
            raise TimeoutError(f"Timed out waiting for ranks to finish: missing {missing}")
        time.sleep(2.0)


def main() -> None:
    args = build_arg_parser().parse_args()
    rank, world_size = _resolve_dist(args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = SidecarBuildConfig(
        orders=_parse_orders(args.orders),
        topk=args.topk,
        min_count=args.min_count,
        min_store_confidence=args.min_store_confidence,
        load_factor=args.load_factor,
        chunk_records=args.chunk_records,
        keep_temporary_chunks=args.keep_temporary_chunks,
        pad_token_id=args.pad_token_id,
    )

    loader = get_dataloader(
        split=args.split,
        batch_size=args.batch_size,
        streaming=args.streaming,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        persistent_workers=not args.no_persistent_workers,
        rank=rank,
        world_size=world_size,
    )

    if world_size == 1:
        rank_out_dir = out_dir
    else:
        rank_out_dir = out_dir / "_dist_parts" / f"rank_{rank:03d}"
        rank_out_dir.mkdir(parents=True, exist_ok=True)

    builder = CompactNgramSidecarBuilder(output_dir=rank_out_dir, config=config)

    start = time.time()
    n_batches = 0
    for batch in loader:
        builder.add_batch(batch)
        n_batches += 1
        if args.log_every > 0 and n_batches % args.log_every == 0:
            elapsed = time.time() - start
            stats = builder.stats
            prefix = f"[rank {rank}/{world_size}]"
            print(
                f"{prefix} batches={n_batches} sequences={stats['sequences']} "
                f"raw_records={stats['raw_records']} reduced_chunk_records={stats['reduced_chunk_records']} "
                f"elapsed_s={elapsed:.1f}"
            )
        if args.max_batches is not None and n_batches >= args.max_batches:
            break

    if world_size == 1:
        summary = builder.finalize()
        elapsed = time.time() - start
        print("\n[build] done")
        print(json.dumps(summary, indent=2, sort_keys=True))
        print(f"[build] wrote index to {out_dir}")
        print(f"[build] elapsed_s={elapsed:.1f}")
        return

    # Distributed path: each rank flushes chunks, rank0 merges all chunk files once.
    builder.flush_chunk()

    sync_dir = out_dir / "_dist_sync"
    sync_dir.mkdir(parents=True, exist_ok=True)
    stats_path = sync_dir / f"rank_{rank:03d}.stats.json"
    done_path = sync_dir / f"rank_{rank:03d}.done"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(builder.stats, f, sort_keys=True)
    done_path.write_text("ok\n", encoding="utf-8")

    if rank != 0:
        print(
            f"[rank {rank}/{world_size}] chunking complete; waiting for rank0 to finalize into {out_dir}"
        )
        return

    _wait_for_markers(sync_dir, world_size=world_size, timeout_s=args.dist_timeout_s)

    merged_builder = CompactNgramSidecarBuilder(output_dir=out_dir, config=config)
    chunk_paths = sorted((out_dir / "_dist_parts").glob("rank_*/_tmp_chunks/chunk_*.npy"))
    if not chunk_paths:
        raise RuntimeError("No distributed chunk files found to merge.")
    merged_builder._chunk_paths = list(chunk_paths)

    aggregated_stats = {"sequences": 0, "raw_records": 0, "reduced_chunk_records": 0}
    for r in range(world_size):
        rank_stats_path = sync_dir / f"rank_{r:03d}.stats.json"
        with open(rank_stats_path, "r", encoding="utf-8") as f:
            rs = json.load(f)
        for key in aggregated_stats:
            aggregated_stats[key] += int(rs.get(key, 0))
    merged_builder._stats = aggregated_stats

    summary = merged_builder.finalize()
    elapsed = time.time() - start
    print("\n[build] done (distributed)")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[build] wrote index to {out_dir}")
    print(f"[build] elapsed_s={elapsed:.1f}")

    if not args.keep_temporary_chunks:
        shutil.rmtree(out_dir / "_dist_parts", ignore_errors=True)
        shutil.rmtree(sync_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

