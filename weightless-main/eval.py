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
    total_tokens = 0
    n_batches = 0
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids, attention_mask)
        
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction="sum",
        )
        
        n_tokens = attention_mask.sum().item()
        total_loss += loss.item()
        total_tokens += n_tokens
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
                        choices=["baseline", "baseline_plus"])
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--seq_len", type=int, default=512,
                        help="Context length for bytes_per_token_infer metric")
    parser.add_argument("--visualize", action="store_true",
                        help="Save a breakdown chart as PNG")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    print("  Loading model...")
    model = create_model(
        variant=args.model,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
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
