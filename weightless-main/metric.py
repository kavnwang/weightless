"""bytes_per_token_infer: the core metric for weightless.

Measures total bytes read from memory to produce one token during
single-token autoregressive decode (batch=1, seq_len=1 incremental).

The metric is broken down by component so candidates can see exactly
where their bytes are going and which parts of the architecture to attack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# InferenceProfile dataclass
# ---------------------------------------------------------------------------

@dataclass
class InferenceProfile:
    """Byte-level breakdown for single-token decode inference.

    Every field ending in ``_bytes`` counts bytes read (or written, for
    KV-cache writes) from/to memory for **one token** of autoregressive
    decode.

    Supplementary fields (``unique_param_bytes``, ``unique_opt_state_bytes``)
    are not part of the per-token score but are useful context.
    """

    # -- model identity --
    model_name: str = "unknown"

    # -- architecture dims (for display) --
    d_model: int = 0
    n_layers: int = 0
    n_heads: int = 0
    n_kv_heads: int = 0  # equals n_heads for MHA
    head_dim: int = 0
    d_ff: int = 0
    vocab_size: int = 0

    # -- measurement config --
    seq_len: int = 512
    batch_size: int = 1
    weight_dtype_bytes: float = 2  # bf16=2, fp8=1, int4/fp4=0.5
    kv_dtype_bytes: float = 2
    cold_cache: bool = True
    count_reuse: bool = False  # tied weights counted once

    # -- byte breakdown (the score components) --
    embedding_bytes: float = 0
    attn_q_bytes: float = 0
    attn_k_bytes: float = 0
    attn_v_bytes: float = 0
    attn_o_bytes: float = 0
    ffn_bytes: float = 0
    norm_bytes: float = 0
    lm_head_bytes: float = 0  # 0 when tied & count_reuse=False
    kv_cache_read_bytes: float = 0
    kv_cache_write_bytes: float = 0

    # -- supplementary --
    unique_param_bytes: float = 0
    unique_opt_state_bytes: float = 0  # AdamW bf16 mixed-prec: 12 B/param

    # -- optional per-component detail --
    ffn_detail: Optional[dict] = field(default=None, repr=False)
    notes: str = ""

    # -- derived properties --

    @property
    def attn_proj_bytes(self) -> int:
        """Total attention projection weight bytes (Q+K+V+O) across all layers."""
        return self.attn_q_bytes + self.attn_k_bytes + self.attn_v_bytes + self.attn_o_bytes

    @property
    def kv_cache_bytes(self) -> int:
        return self.kv_cache_read_bytes + self.kv_cache_write_bytes

    @property
    def weight_bytes(self) -> int:
        """Total weight bytes (embedding + attn + ffn + norms + lm_head)."""
        return (
            self.embedding_bytes
            + self.attn_proj_bytes
            + self.ffn_bytes
            + self.norm_bytes
            + self.lm_head_bytes
        )

    @property
    def total_bytes(self) -> int:
        """The score: total bytes touched per token."""
        return self.weight_bytes + self.kv_cache_bytes

    @property
    def bytes_per_token(self) -> int:
        """Alias for total_bytes (batch=1, one token)."""
        return self.total_bytes

    def breakdown_dict(self) -> dict[str, float]:
        """Ordered dict of component name -> bytes for the breakdown."""
        return {
            "Embeddings": self.embedding_bytes,
            "Attn Q proj": self.attn_q_bytes,
            "Attn K proj": self.attn_k_bytes,
            "Attn V proj": self.attn_v_bytes,
            "Attn O proj": self.attn_o_bytes,
            "FFN": self.ffn_bytes,
            "Norms": self.norm_bytes,
            "LM head": self.lm_head_bytes,
            "KV cache read": self.kv_cache_read_bytes,
            "KV cache write": self.kv_cache_write_bytes,
        }

    def supplementary_dict(self) -> dict[str, float]:
        return {
            "unique_param_bytes": self.unique_param_bytes,
            "unique_opt_state_bytes": self.unique_opt_state_bytes,
        }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_bytes(b: float) -> str:
    """Human-readable byte count."""
    if b < 1024:
        return f"{b:.0f} B" if b == int(b) else f"{b:.1f} B"
    elif b < 1024 ** 2:
        return f"{b / 1024:.1f} KB"
    elif b < 1024 ** 3:
        return f"{b / 1024**2:.2f} MB"
    else:
        return f"{b / 1024**3:.2f} GB"


def print_profile(profile: InferenceProfile) -> str:
    """Pretty-print an InferenceProfile and return the string."""
    bd = profile.breakdown_dict()
    total = profile.total_bytes

    lines: list[str] = []
    lines.append("")
    lines.append(f"  bytes_per_token_infer Breakdown  (seq_len={profile.seq_len})")
    lines.append("  " + "=" * 58)
    lines.append(f"  {'Component':<22} {'Bytes':>12}  {'%':>6}   Bar")
    lines.append("  " + "-" * 58)

    max_val = max(bd.values()) if any(v > 0 for v in bd.values()) else 1
    bar_width = 20

    for name, val in bd.items():
        pct = (val / total * 100) if total > 0 else 0.0
        bar_len = int(val / max_val * bar_width) if max_val > 0 else 0
        bar = "\u2588" * bar_len
        lines.append(f"  {name:<22} {_fmt_bytes(val):>12}  {pct:5.1f}%   {bar}")

    lines.append("  " + "-" * 58)
    lines.append(f"  {'TOTAL (score)':<22} {_fmt_bytes(total):>12}  100.0%")
    lines.append("")

    # Supplementary
    lines.append("  Supplementary:")
    lines.append(f"    unique_param_bytes:     {_fmt_bytes(profile.unique_param_bytes)}")
    lines.append(f"    unique_opt_state_bytes: {_fmt_bytes(profile.unique_opt_state_bytes)}")
    lines.append("")

    # Assumptions
    lines.append("  Assumptions:")
    lines.append(f"    batch={profile.batch_size}, single-token decode, seq_len={profile.seq_len}")
    cache_mode = "cold cache" if profile.cold_cache else "warm cache"
    reuse_mode = "tied weights counted once" if not profile.count_reuse else "tied weights counted per use"
    lines.append(f"    {cache_mode}, {reuse_mode}")
    wdt = {0.5: "4-bit", 1: "fp8/int8", 2: "bf16", 4: "fp32"}.get(profile.weight_dtype_bytes, f"{profile.weight_dtype_bytes}B")
    kdt = {0.5: "4-bit", 1: "fp8/int8", 2: "bf16", 4: "fp32"}.get(profile.kv_dtype_bytes, f"{profile.kv_dtype_bytes}B")
    lines.append(f"    weight dtype: {wdt} ({profile.weight_dtype_bytes}B), KV dtype: {kdt} ({profile.kv_dtype_bytes}B)")

    if profile.notes:
        lines.append(f"    notes: {profile.notes}")
    lines.append("")

    text = "\n".join(lines)
    print(text)
    return text


# ---------------------------------------------------------------------------
# Auto-profiler (fallback introspection)
# ---------------------------------------------------------------------------

def auto_profile(
    model: nn.Module,
    seq_len: int = 512,
    weight_dtype_bytes: int = 2,
    kv_dtype_bytes: int = 2,
    count_reuse: bool = False,
    model_name: str = "auto",
) -> InferenceProfile:
    """Best-effort profiling via module introspection.

    Walks the model's named modules and classifies them by type and name
    into embedding, attention, FFN, norm, and head categories.  Works for
    standard transformer architectures; candidates with exotic designs
    should implement ``get_inference_profile()`` on their model.
    """
    profile = InferenceProfile(
        model_name=model_name,
        seq_len=seq_len,
        weight_dtype_bytes=weight_dtype_bytes,
        kv_dtype_bytes=kv_dtype_bytes,
        count_reuse=count_reuse,
    )

    # Gather dims from model attributes (best effort)
    d_model = getattr(model, "d_model", 0)
    profile.d_model = d_model

    # Detect weight tying
    head_weight_id = None
    emb_weight_id = None

    # Collect all parameter data_ptrs to detect tying
    param_ids_seen: set[int] = set()
    unique_param_numel = 0

    embedding_bytes = 0
    attn_q_bytes = 0
    attn_k_bytes = 0
    attn_v_bytes = 0
    attn_o_bytes = 0
    ffn_bytes = 0
    norm_bytes = 0
    lm_head_bytes = 0

    # Track n_layers, n_heads, head_dim, d_ff, vocab_size
    n_layers = 0
    n_heads = 0
    n_kv_heads = 0
    head_dim = 0
    d_ff = 0
    vocab_size = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            vocab_size = module.num_embeddings
            for p in module.parameters(recurse=False):
                pid = p.data_ptr()
                b = p.numel() * weight_dtype_bytes
                embedding_bytes += b
                emb_weight_id = pid
                if pid not in param_ids_seen:
                    param_ids_seen.add(pid)
                    unique_param_numel += p.numel()

        elif isinstance(module, (nn.LayerNorm,)):
            for p in module.parameters(recurse=False):
                pid = p.data_ptr()
                norm_bytes += p.numel() * weight_dtype_bytes
                if pid not in param_ids_seen:
                    param_ids_seen.add(pid)
                    unique_param_numel += p.numel()

        elif isinstance(module, nn.Linear):
            for p in module.parameters(recurse=False):
                pid = p.data_ptr()
                b = p.numel() * weight_dtype_bytes

                # Classify by name
                name_lower = name.lower()
                if "head" in name_lower or "lm_head" in name_lower:
                    head_weight_id = pid
                    # LM head always reads the full vocab x d_model matrix.
                    # When tied with embedding, the embedding row is a subset
                    # of this read, so we zero out embedding_bytes instead.
                    lm_head_bytes += b
                    if not count_reuse and pid == emb_weight_id:
                        embedding_bytes = 0  # subsumed by lm_head
                elif any(k in name_lower for k in ["qkv", "q_proj", "k_proj", "v_proj",
                                                      "wq", "wk", "wv"]):
                    # Fused QKV or individual projections
                    if "qkv" in name_lower:
                        third = b // 3
                        attn_q_bytes += third
                        attn_k_bytes += third
                        attn_v_bytes += b - 2 * third
                    elif "q" in name_lower.split(".")[-1] or "wq" in name_lower:
                        attn_q_bytes += b
                    elif "k" in name_lower.split(".")[-1] or "wk" in name_lower:
                        attn_k_bytes += b
                    elif "v" in name_lower.split(".")[-1] or "wv" in name_lower:
                        attn_v_bytes += b
                elif any(k in name_lower for k in ["proj", "o_proj", "wo"]):
                    attn_o_bytes += b
                elif any(k in name_lower for k in ["w1", "w2", "w3", "up", "down",
                                                      "gate", "ff", "mlp", "ffn"]):
                    ffn_bytes += b
                else:
                    # Unknown linear, add to FFN as fallback
                    ffn_bytes += b

                if pid not in param_ids_seen:
                    param_ids_seen.add(pid)
                    unique_param_numel += p.numel()

    # Try to infer architecture dims
    n_layers = getattr(model, "n_layers", len([m for n, m in model.named_modules()
                                                if "layers." in n and n.count(".") == 1]))
    if hasattr(model, "layers"):
        n_layers = len(model.layers)

    # Try to get attention config
    for name, module in model.named_modules():
        if hasattr(module, "n_heads"):
            n_heads = module.n_heads
            head_dim = getattr(module, "head_dim", d_model // n_heads if n_heads > 0 else 0)
            n_kv_heads = getattr(module, "n_kv_heads", n_heads)
            break

    for name, module in model.named_modules():
        if hasattr(module, "d_ff"):
            d_ff = module.d_ff
            break

    # KV cache bytes
    if n_kv_heads > 0 and head_dim > 0 and n_layers > 0:
        kv_cache_read = 2 * n_kv_heads * head_dim * seq_len * n_layers * kv_dtype_bytes
        kv_cache_write = 2 * n_kv_heads * head_dim * n_layers * kv_dtype_bytes
    else:
        kv_cache_read = 0
        kv_cache_write = 0

    profile.d_model = d_model
    profile.n_layers = n_layers
    profile.n_heads = n_heads
    profile.n_kv_heads = n_kv_heads
    profile.head_dim = head_dim
    profile.d_ff = d_ff
    profile.vocab_size = vocab_size

    profile.embedding_bytes = embedding_bytes
    profile.attn_q_bytes = attn_q_bytes
    profile.attn_k_bytes = attn_k_bytes
    profile.attn_v_bytes = attn_v_bytes
    profile.attn_o_bytes = attn_o_bytes
    profile.ffn_bytes = ffn_bytes
    profile.norm_bytes = norm_bytes
    profile.lm_head_bytes = lm_head_bytes
    profile.kv_cache_read_bytes = kv_cache_read
    profile.kv_cache_write_bytes = kv_cache_write

    profile.unique_param_bytes = unique_param_numel * weight_dtype_bytes
    # AdamW mixed-precision: master weights (fp32=4B) + mean (fp32=4B) + var (fp32=4B) = 12B/param
    profile.unique_opt_state_bytes = unique_param_numel * 12

    profile.notes = "auto-profiled via module introspection"

    return profile


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute bytes_per_token_infer for a model")
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "baseline_plus"],
                        help="Model variant to profile")
    parser.add_argument("--seq_len", type=int, default=512, help="Context length for KV cache")
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--d_ff", type=int, default=None)
    args = parser.parse_args()

    from model import create_model

    kwargs = {}
    if args.d_model is not None:
        kwargs["d_model"] = args.d_model
    if args.n_layers is not None:
        kwargs["n_layers"] = args.n_layers
    if args.n_heads is not None:
        kwargs["n_heads"] = args.n_heads
    if args.d_ff is not None:
        kwargs["d_ff"] = args.d_ff

    model = create_model(variant=args.model, **kwargs)

    if hasattr(model, "get_inference_profile"):
        profile = model.get_inference_profile(seq_len=args.seq_len)
    else:
        profile = auto_profile(model, seq_len=args.seq_len, model_name=args.model)

    print_profile(profile)
