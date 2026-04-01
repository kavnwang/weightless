"""Visualize bytes_per_token_infer breakdowns.

Usage:
    python visualize.py                         # compare baseline vs baseline+
    python visualize.py --models baseline       # single model breakdown
    python visualize.py --seq_len 1024          # custom context length
"""

from __future__ import annotations

import argparse
from typing import Dict

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; works headless
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from metric import InferenceProfile, _fmt_bytes, print_profile


# ---------------------------------------------------------------------------
# Color palette (tab10 inspired, deterministic)
# ---------------------------------------------------------------------------

COMPONENT_COLORS = {
    "Embeddings":     "#4e79a7",
    "Attn Q proj":    "#f28e2b",
    "Attn K proj":    "#e15759",
    "Attn V proj":    "#76b7b2",
    "Attn O proj":    "#59a14f",
    "FFN":            "#edc948",
    "Norms":          "#b07aa1",
    "LM head":        "#ff9da7",
    "KV cache read":  "#9c755f",
    "KV cache write": "#bab0ac",
}


def _bytes_formatter(x, _pos):
    """Axis formatter: bytes -> human-readable."""
    if x >= 1024 ** 3:
        return f"{x / 1024**3:.1f} GB"
    elif x >= 1024 ** 2:
        return f"{x / 1024**2:.1f} MB"
    elif x >= 1024:
        return f"{x / 1024:.0f} KB"
    else:
        return f"{int(x)} B"


# ---------------------------------------------------------------------------
# Single-model breakdown
# ---------------------------------------------------------------------------

def plot_breakdown(
    profile: InferenceProfile,
    ax: plt.Axes | None = None,
    save_path: str | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Horizontal stacked bar chart showing where bytes go."""
    bd = profile.breakdown_dict()
    components = list(bd.keys())
    values = list(bd.values())
    colors = [COMPONENT_COLORS.get(c, "#888888") for c in components]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(12, 3))
    else:
        fig = ax.figure

    # Single horizontal stacked bar
    left = 0
    for comp, val, color in zip(components, values, colors):
        ax.barh(0, val, left=left, color=color, edgecolor="white", linewidth=0.5,
                label=f"{comp} ({_fmt_bytes(val)})")
        left += val

    ax.set_yticks([])
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_bytes_formatter))
    ax.set_xlabel("Bytes per token (inference)")

    if title is None:
        title = f"{profile.model_name}  —  bytes_per_token = {_fmt_bytes(profile.total_bytes)}  (seq_len={profile.seq_len})"
    ax.set_title(title, fontsize=11, fontweight="bold")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35),
              ncol=5, fontsize=8, frameon=False)

    if standalone:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Multi-model comparison
# ---------------------------------------------------------------------------

def plot_comparison(
    profiles: Dict[str, InferenceProfile],
    save_path: str | None = None,
    title: str = "bytes_per_token_infer Comparison",
) -> plt.Figure:
    """Grouped horizontal bars comparing multiple models side-by-side.

    Each model gets a row; components are stacked within the row.
    """
    names = list(profiles.keys())
    n = len(names)

    fig, ax = plt.subplots(figsize=(12, 1.5 + 1.2 * n))

    components = list(next(iter(profiles.values())).breakdown_dict().keys())
    y_positions = np.arange(n)

    for i, name in enumerate(names):
        bd = profiles[name].breakdown_dict()
        left = 0
        for comp in components:
            val = bd.get(comp, 0)
            color = COMPONENT_COLORS.get(comp, "#888888")
            label = comp if i == 0 else None  # legend only once
            ax.barh(i, val, left=left, color=color, edgecolor="white",
                    linewidth=0.5, label=label)
            left += val
        # Annotate total
        total = profiles[name].total_bytes
        ax.text(left + left * 0.01, i, f" {_fmt_bytes(total)}",
                va="center", fontsize=9, fontweight="bold")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_bytes_formatter))
    ax.set_xlabel("Bytes per token (inference)")
    ax.set_title(title, fontsize=12, fontweight="bold")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=5, fontsize=8, frameon=False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Detailed component comparison table
# ---------------------------------------------------------------------------

def print_comparison(profiles: Dict[str, InferenceProfile]) -> str:
    """Print a side-by-side comparison table of profiles."""
    names = list(profiles.keys())
    components = list(next(iter(profiles.values())).breakdown_dict().keys())

    lines: list[str] = []
    lines.append("")
    lines.append("  bytes_per_token_infer Comparison")
    lines.append("  " + "=" * (28 + 16 * len(names)))

    header = f"  {'Component':<26}"
    for name in names:
        header += f" {name:>14}"
    if len(names) == 2:
        header += f" {'Delta':>14}"
    lines.append(header)
    lines.append("  " + "-" * (28 + 16 * len(names)))

    for comp in components:
        row = f"  {comp:<26}"
        vals = []
        for name in names:
            val = profiles[name].breakdown_dict()[comp]
            vals.append(val)
            row += f" {_fmt_bytes(val):>14}"
        if len(names) == 2 and vals[0] > 0:
            delta = vals[1] - vals[0]
            pct = delta / vals[0] * 100
            sign = "+" if delta >= 0 else ""
            row += f" {sign}{pct:.1f}%".rjust(14)
        lines.append(row)

    lines.append("  " + "-" * (28 + 16 * len(names)))

    total_row = f"  {'TOTAL (score)':<26}"
    totals = []
    for name in names:
        t = profiles[name].total_bytes
        totals.append(t)
        total_row += f" {_fmt_bytes(t):>14}"
    if len(names) == 2 and totals[0] > 0:
        delta = totals[1] - totals[0]
        pct = delta / totals[0] * 100
        sign = "+" if delta >= 0 else ""
        total_row += f" {sign}{pct:.1f}%".rjust(14)
    lines.append(total_row)
    lines.append("")

    text = "\n".join(lines)
    print(text)
    return text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize bytes_per_token_infer")
    parser.add_argument("--models", nargs="+", default=["baseline", "baseline_plus"],
                        choices=["baseline", "baseline_plus"],
                        help="Model variants to profile and compare")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--out", type=str, default="bytes_breakdown.png",
                        help="Output image file path")
    args = parser.parse_args()

    from model import create_model
    from metric import auto_profile

    kwargs = {}
    if args.d_model is not None:
        kwargs["d_model"] = args.d_model
    if args.n_layers is not None:
        kwargs["n_layers"] = args.n_layers
    if args.n_heads is not None:
        kwargs["n_heads"] = args.n_heads
    if args.d_ff is not None:
        kwargs["d_ff"] = args.d_ff

    profiles: Dict[str, InferenceProfile] = {}
    for variant in args.models:
        model = create_model(variant=variant, **kwargs)
        if hasattr(model, "get_inference_profile"):
            profile = model.get_inference_profile(seq_len=args.seq_len)
        else:
            profile = auto_profile(model, seq_len=args.seq_len, model_name=variant)
        profiles[variant] = profile
        print_profile(profile)

    if len(profiles) == 1:
        profile = next(iter(profiles.values()))
        plot_breakdown(profile, save_path=args.out)
    else:
        print_comparison(profiles)
        plot_comparison(profiles, save_path=args.out)

    print(f"\n  Visualization saved to {args.out}")


if __name__ == "__main__":
    main()
