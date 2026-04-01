"""Training script with wandb logging and optional DDP support."""

import argparse
import os
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Peak TFLOPS for MFU calculation (BF16 tensor core ops)
# H100 SXM: 990 TFLOPS BF16, A100: 312 TFLOPS BF16
GPU_PEAK_TFLOPS = 990

from data import get_dataloader
from model import create_model
from metric import print_profile

PAD_TOKEN_ID = 50256  # GPT-2 <|endoftext|> used as pad token
GOAL_VAL_LOSS = 3.5


# ---------------------------------------------------------------------------
# DDP helpers (optional -- works without torchrun)
# ---------------------------------------------------------------------------

def setup_ddp():
    """Initialize DDP if launched via torchrun. Returns (local_rank, world_size, use_ddp)."""
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, False
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: F811
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size(), True


def is_main(use_ddp: bool = False):
    if not use_ddp:
        return True
    import torch.distributed as dist
    return dist.get_rank() == 0


def get_world_size(use_ddp: bool = False):
    if not use_ddp:
        return 1
    import torch.distributed as dist
    return dist.get_world_size()


def get_rank(use_ddp: bool = False):
    if not use_ddp:
        return 0
    import torch.distributed as dist
    return dist.get_rank()


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def compute_loss(model, batch, device):
    """Compute cross-entropy loss for a batch using BF16 autocast."""
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids, attention_mask)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=PAD_TOKEN_ID,
        )
    return loss


def get_lr(step: int, warmup_steps: int, total_steps: int, max_lr: float, min_lr: float) -> float:
    """Linear warmup then linear decay."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    return max_lr - (max_lr - min_lr) * decay_ratio


@torch.no_grad()
def evaluate(model, val_loader, device, max_batches: int = 20):
    """Evaluate model on validation set (lightweight, uses BF16)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    for batch in val_loader:
        loss = compute_loss(model, batch, device)
        total_loss += loss.item()
        n_batches += 1
        if n_batches >= max_batches:
            break
    
    model.train()
    return total_loss / n_batches


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_steps: int = 5000,
    eval_every: int = 50,
    max_lr: float = 1e-3,
    warmup_steps: int = 200,
    use_wandb: bool = True,
    use_ddp: bool = False,
):
    """Main training loop with logging every eval_every steps."""
    model.train()
    raw_model = model.module if hasattr(model, "module") else model
    num_params = raw_model.count_parameters(count_zeros=True)
    min_lr = max_lr * 0.1

    train_iter = iter(train_loader)
    running_loss = 0.0
    total_tokens = 0
    epoch = 0
    t0 = time.time()
    world_size = get_world_size(use_ddp)

    pbar = tqdm(range(num_steps), desc="Training", disable=not is_main(use_ddp))
    for step in pbar:
        lr = get_lr(step, warmup_steps, num_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
            epoch += 1

        B, T = batch["input_ids"].shape
        tokens_this_step = B * T * world_size
        total_tokens += tokens_this_step

        optimizer.zero_grad()
        loss = compute_loss(model, batch, device)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (step + 1) % eval_every == 0:
            torch.cuda.synchronize()
            dt = time.time() - t0
            tokens_interval = tokens_this_step * eval_every
            mfu = 6 * num_params * tokens_interval / (GPU_PEAK_TFLOPS * 1e12 * dt * world_size)

            total_flops = 6 * num_params * total_tokens
            train_loss = running_loss / eval_every
            val_loss = evaluate(model, val_loader, device)
            nonzero_params = raw_model.count_parameters(count_zeros=False)

            if is_main(use_ddp):
                pbar.set_postfix({"train": f"{train_loss:.3f}", "val": f"{val_loss:.3f}", "mfu": f"{mfu:.1%}"})
                if use_wandb:
                    import wandb
                    wandb.log({
                        "train/loss": train_loss,
                        "train/lr": lr,
                        "train/total_tokens": total_tokens,
                        "train/total_flops": total_flops,
                        "train/epoch": epoch,
                        "val/loss": val_loss,
                        "params/nonzero": nonzero_params,
                        "mfu": mfu,
                        "step": step + 1,
                    })
                if val_loss < GOAL_VAL_LOSS:
                    print(f"\n  Goal achieved! val_loss={val_loss:.4f} < {GOAL_VAL_LOSS} with {nonzero_params:,} non-zero params")

            running_loss = 0.0
            t0 = time.time()

    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_lr", type=float, default=6e-4)
    parser.add_argument("--num_steps", type=int, default=55000)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--model", type=str, default="baseline",
                        choices=["baseline", "baseline_plus"],
                        help="Model variant: baseline (dense) or baseline_plus (GQA + top-k FFN)")
    parser.add_argument("--wandb_project", type=str, default="weightless")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Name for this run (used for checkpoint filename)")
    parser.add_argument("--save_checkpoint", action="store_true",
                        help="Save model checkpoint at end of training")
    args = parser.parse_args()

    # Autoscale max_lr: args.max_lr is calibrated at d_model=768
    # Wider models use lower LR (muP-style sqrt scaling)
    base_lr = args.max_lr
    args.max_lr = base_lr * (768 / args.d_model) ** 0.5

    # DDP setup (optional -- works with plain `python train.py`)
    local_rank, world_size, use_ddp = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    use_wandb = not args.no_wandb

    if is_main(use_ddp):
        if args.d_model != 768:
            print(f"  Autoscaled max_lr from {base_lr:.2e} to {args.max_lr:.2e} (d_model={args.d_model})")
        else:
            print(f"  max_lr={args.max_lr:.2e} (d_model={args.d_model})")
        if use_ddp:
            print(f"  DDP: rank {local_rank}, world_size {world_size}")
        else:
            print(f"  Single-GPU mode (use torchrun for DDP)")

    # Enable flash attention and bf16 optimizations
    torch.backends.cuda.enable_flash_sdp(True)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

    # Run name
    if args.run_name is None:
        args.run_name = f"{args.model}_d{args.d_model}_L{args.n_layers}"

    # wandb
    if use_wandb and is_main(use_ddp):
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config={**vars(args), "model_variant": args.model},
        )

    # Data
    rank = get_rank(use_ddp)
    if is_main(use_ddp):
        print("  Setting up data loaders...")
    train_loader = get_dataloader(split="train", batch_size=args.batch_size,
                                  streaming=True, rank=rank, world_size=world_size)
    val_loader = get_dataloader(split="test", batch_size=args.batch_size,
                                streaming=True, rank=rank, world_size=world_size)

    # Model
    if is_main(use_ddp):
        print(f"  Creating model (variant={args.model}, BF16 + torch.compile)...")
    model = create_model(
        variant=args.model,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
    )
    model.to(device)
    model = torch.compile(model)

    if use_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank])

    raw_model = model.module if hasattr(model, "module") else model
    # Handle torch.compile wrapper
    if hasattr(raw_model, "_orig_mod"):
        raw_model = raw_model._orig_mod

    total_params = raw_model.count_parameters(count_zeros=True)
    nonzero_params = raw_model.count_parameters(count_zeros=False)

    if is_main(use_ddp):
        print(f"  Total parameters:    {total_params:,}")
        print(f"  Non-zero parameters: {nonzero_params:,}")

        # Show the bytes_per_token_infer breakdown
        profile = raw_model.get_inference_profile()
        print_profile(profile)

        if use_wandb:
            import wandb
            bd = profile.breakdown_dict()
            wandb.log({
                "params/total": total_params,
                "params/nonzero": nonzero_params,
                "metric/bytes_per_token_infer": profile.total_bytes,
                **{f"metric/{k}": v for k, v in bd.items()},
                "metric/unique_param_bytes": profile.unique_param_bytes,
                "metric/unique_opt_state_bytes": profile.unique_opt_state_bytes,
            })

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(0.9, 0.95),
        weight_decay=0.3,
    )

    # Train
    model = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_steps=args.num_steps,
        eval_every=args.eval_every,
        max_lr=args.max_lr,
        use_wandb=use_wandb,
        use_ddp=use_ddp,
    )

    # End-of-training summary
    if is_main(use_ddp):
        raw_model_final = model.module if hasattr(model, "module") else model
        if hasattr(raw_model_final, "_orig_mod"):
            raw_model_final = raw_model_final._orig_mod
        final_val_loss = evaluate(model, val_loader, device)
        profile = raw_model_final.get_inference_profile()
        print("\n" + "=" * 60)
        print("  TRAINING COMPLETE")
        print("=" * 60)
        print(f"  Final val_loss:         {final_val_loss:.4f}  (target < {GOAL_VAL_LOSS})")
        print(f"  bytes_per_token_infer:  {profile.total_bytes:,} bytes")
        print_profile(profile)
        if final_val_loss < GOAL_VAL_LOSS:
            print(f"  GOAL ACHIEVED!")
        else:
            print(f"  Goal not yet reached (val_loss {final_val_loss:.4f} >= {GOAL_VAL_LOSS})")
        print()

        # Save checkpoint
        if args.save_checkpoint:
            import os
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = f"checkpoints/{args.run_name}.pt"
            torch.save(raw_model_final.state_dict(), ckpt_path)
            print(f"  Checkpoint saved to {ckpt_path}")

    if use_wandb and is_main(use_ddp):
        import wandb
        wandb.finish()
    if use_ddp:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
