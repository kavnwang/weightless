"""Evaluation script for trained models.

Usage:
    python eval.py                                      # evaluate random-init baseline (quick check)
    python eval.py --checkpoint model.pt                # evaluate a saved checkpoint
    python eval.py --checkpoint model.pt --visualize    # + save breakdown chart
"""

import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data import get_dataloader
from model import create_model
from metric import print_profile

GOAL_VAL_LOSS = 3.5


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate model and return metrics."""
    model.eval()
    model.to(device)
    
    total_loss = 0.0
    total_hot_loss = 0.0
    total_cold_loss = 0.0
    total_tokens = 0
    total_hot_tokens = 0
    total_cold_tokens = 0
    n_batches = 0
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids, attention_mask)
        
        token_losses = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction="none",
        )

        flat_labels = labels.reshape(-1)
        valid_mask = attention_mask.reshape(-1) > 0
        n_tokens = valid_mask.sum().item()
        total_loss += token_losses[valid_mask].sum().item()
        total_tokens += n_tokens
        raw_model = model.module if hasattr(model, "module") else model
        if hasattr(raw_model, "_orig_mod"):
            raw_model = raw_model._orig_mod
        if hasattr(raw_model, "token_partition_masks"):
            hot_mask, cold_mask = raw_model.token_partition_masks(labels)
            if hot_mask is not None and cold_mask is not None:
                hot_mask = hot_mask.reshape(-1) & valid_mask
                cold_mask = cold_mask.reshape(-1) & valid_mask
                if hot_mask.any():
                    total_hot_loss += F.cross_entropy(
                        logits.reshape(-1, logits.size(-1))[hot_mask],
                        flat_labels[hot_mask],
                        reduction="sum",
                    ).item()
                    total_hot_tokens += hot_mask.sum().item()
                if cold_mask.any():
                    total_cold_loss += F.cross_entropy(
                        logits.reshape(-1, logits.size(-1))[cold_mask],
                        flat_labels[cold_mask],
                        reduction="sum",
                    ).item()
                    total_cold_tokens += cold_mask.sum().item()
        n_batches += 1
        
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    # Parameters
    raw_model = model.module if hasattr(model, "module") else model
    if hasattr(raw_model, "_orig_mod"):
        raw_model = raw_model._orig_mod
    total_params = raw_model.count_parameters(count_zeros=True)
    nonzero_params = raw_model.count_parameters(count_zeros=False)
    sparsity = 1.0 - (nonzero_params / total_params) if total_params > 0 else 0.0
    
    return {
        "val_loss": avg_loss,
        "hot_loss": (total_hot_loss / total_hot_tokens) if total_hot_tokens > 0 else None,
        "cold_loss": (total_cold_loss / total_cold_tokens) if total_cold_tokens > 0 else None,
        "perplexity": perplexity,
        "total_params": total_params,
        "nonzero_params": nonzero_params,
        "sparsity": sparsity,
        "n_batches": n_batches,
        "n_tokens": total_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on FineWeb")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model", type=str, default="baseline",
                        choices=["baseline", "gqa_only", "topk_only", "baseline_plus", "mla", "hotcold_mla", "hotcold_svd", "twostage_svd", "mla_twostage_svd_mem12_monarch", "loop_top4x3_attnres", "mla_hybrid_loop12", "mla_hybrid_loop12_monarch"])
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--kv_lora_rank", type=int, default=None,
                        help="MLA KV latent rank (d_c); used for --model mla/hotcold_mla")
    parser.add_argument("--q_lora_rank", type=int, default=None,
                        help="MLA query latent rank (d'_c); used for --model mla/hotcold_mla")
    parser.add_argument("--qk_nope_head_dim", type=int, default=None,
                        help="MLA non-RoPE per-head Q/K dim; used for --model mla/hotcold_mla")
    parser.add_argument("--qk_rope_head_dim", type=int, default=None,
                        help="MLA RoPE per-head Q/K dim (must be even); used for --model mla/hotcold_mla")
    parser.add_argument("--v_head_dim", type=int, default=None,
                        help="MLA per-head V dim; used for --model mla/hotcold_mla")
    parser.add_argument("--hot_token_k", type=int, default=2000,
                        help="Number of dense hot tokens; used for --model hotcold_svd/hotcold_mla/twostage_svd")
    parser.add_argument("--cold_latent_dim", type=int, default=128,
                        help="Rank for cold-token SVD factors; used for --model hotcold_svd/hotcold_mla/twostage_svd")
    parser.add_argument("--hot_token_cache_path", type=str, default="cache/hot_tokens_train1p3b_top2000.pt",
                        help="Path to cached hot tokens from build_hot_token_cache.py")
    parser.add_argument("--svd_switch_fraction", type=float, default=None,
                        help="For twostage_svd/hotcold_mla/mla_twostage_svd_mem12_monarch/mla_hybrid_loop12: fraction of train steps before dense -> hot/cold switch")
    parser.add_argument("--monarch_block_size", type=int, default=32)
    parser.add_argument("--memory_layers", type=int, default=12)
    parser.add_argument("--mem_n_keys", type=int, default=384)
    parser.add_argument("--mem_heads", type=int, default=4)
    parser.add_argument("--mem_knn", type=int, default=32)
    parser.add_argument("--mem_k_dim", type=int, default=None)
    parser.add_argument("--mem_v_dim", type=int, default=None)
    parser.add_argument("--mem_q_rank", type=int, default=None)
    parser.add_argument("--no_mem_share_values", action="store_true")
    parser.add_argument("--qk_norm", action="store_true")
    parser.add_argument("--seq_len", type=int, default=512,
                        help="Context length for bytes_per_token_infer metric")
    parser.add_argument("--visualize", action="store_true",
                        help="Save a breakdown chart as PNG")
    args = parser.parse_args()
    if args.model in {"mla_hybrid_loop12", "mla_hybrid_loop12_monarch"} and args.n_layers == 8:
        # Keep CLI ergonomic: these variants are fixed to 12 layers.
        args.n_layers = 12
    if args.model == "mla_hybrid_loop12_monarch" and args.d_model == 768:
        args.d_model = 1024
    if args.model == "mla_hybrid_loop12_monarch" and args.d_ff == 2048:
        args.d_ff = args.d_model
    if args.svd_switch_fraction is None:
        args.svd_switch_fraction = (1.0 / 3.0) if args.model == "mla_hybrid_loop12" else 0.5
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    print("  Loading model...")
    model = create_model(
        variant=args.model,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        kv_lora_rank=args.kv_lora_rank,
        q_lora_rank=args.q_lora_rank,
        qk_nope_head_dim=args.qk_nope_head_dim,
        qk_rope_head_dim=args.qk_rope_head_dim,
        v_head_dim=args.v_head_dim,
        hot_token_k=args.hot_token_k,
        cold_latent_dim=args.cold_latent_dim,
        hot_token_cache_path=args.hot_token_cache_path,
        svd_switch_fraction=args.svd_switch_fraction,
        monarch_block_size=args.monarch_block_size,
        memory_layers=args.memory_layers,
        mem_n_keys=args.mem_n_keys,
        mem_heads=args.mem_heads,
        mem_knn=args.mem_knn,
        mem_k_dim=args.mem_k_dim,
        mem_v_dim=args.mem_v_dim,
        mem_q_rank=args.mem_q_rank,
        mem_share_values=not args.no_mem_share_values,
        qk_norm=args.qk_norm,
    )
    
    if args.checkpoint is not None:
        state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"  Loaded checkpoint: {args.checkpoint}")
    else:
        print("  WARNING: No checkpoint specified -- evaluating random-init model.")
        print("  Use --checkpoint model.pt to evaluate a trained model.")
    
    model.to(device)
    
    # Data
    print("  Loading validation data...")
    val_loader = get_dataloader(split="test", batch_size=args.batch_size, streaming=True)
    
    # Evaluate
    print("  Evaluating...")
    metrics = evaluate(model, val_loader, device)
    
    # Print results
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Validation Loss:    {metrics['val_loss']:.4f}")
    if metrics["hot_loss"] is not None:
        print(f"  Hot Token Loss:     {metrics['hot_loss']:.4f}")
    if metrics["cold_loss"] is not None:
        print(f"  Cold Token Loss:    {metrics['cold_loss']:.4f}")
    print(f"  Perplexity:         {metrics['perplexity']:.2f}")
    print(f"  Total Parameters:   {metrics['total_params']:,}")
    print(f"  Non-zero Params:    {metrics['nonzero_params']:,}")
    print(f"  Sparsity:           {metrics['sparsity']:.2%}")
    print(f"  Evaluated Tokens:   {metrics['n_tokens']:,}")
    
    # bytes_per_token_infer breakdown
    profile = model.get_inference_profile(seq_len=args.seq_len)
    print_profile(profile)
    
    # Goal check
    print("=" * 60)
    if metrics['val_loss'] < GOAL_VAL_LOSS:
        print(f"  GOAL ACHIEVED: val_loss={metrics['val_loss']:.4f} < {GOAL_VAL_LOSS}")
        print(f"  bytes_per_token_infer = {profile.total_bytes:,} bytes")
    else:
        print(f"  Goal not yet achieved: val_loss={metrics['val_loss']:.4f} >= {GOAL_VAL_LOSS}")
    print("=" * 60)
    
    # Optional visualization
    if args.visualize:
        from visualize import plot_breakdown
        out_path = "eval_breakdown.png"
        plot_breakdown(profile, save_path=out_path)
        print(f"  Breakdown chart saved to {out_path}")
    
    return metrics


if __name__ == "__main__":
    main()
