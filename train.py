"""Training script with wandb logging and optional DDP support."""

import argparse
import json
import math
import os
import time
import shutil
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Peak TFLOPS for MFU calculation (BF16 tensor core ops)
# H100 SXM: 990 TFLOPS BF16, A100: 312 TFLOPS BF16
GPU_PEAK_TFLOPS = 990
DATASET_TOKENS_PER_EPOCH = 1_300_000_000

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


def unwrap_model(model):
    raw = model.module if hasattr(model, "module") else model
    if hasattr(raw, "_orig_mod"):
        raw = raw._orig_mod
    return raw


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(max(0, num_bytes))
    unit = units[0]
    for candidate in units:
        unit = candidate
        if value < 1024.0 or candidate == units[-1]:
            break
        value /= 1024.0
    return f"{value:.2f} {unit}"


def _estimate_state_dict_bytes(state_dict: dict[str, torch.Tensor]) -> int:
    total = 0
    for tensor in state_dict.values():
        if not isinstance(tensor, torch.Tensor):
            continue
        total += tensor.numel() * tensor.element_size()
    return int(total)


def _safe_save_checkpoint(
    state_dict: dict[str, torch.Tensor],
    ckpt_path: str,
    fallback_dir: str | None = None,
) -> str:
    estimate = _estimate_state_dict_bytes(state_dict)
    parent_dirs: list[str] = []
    primary_dir = os.path.dirname(ckpt_path) or "."
    parent_dirs.append(primary_dir)
    if fallback_dir:
        parent_dirs.append(fallback_dir)

    attempted: list[str] = []
    for out_dir in parent_dirs:
        os.makedirs(out_dir, exist_ok=True)
        final_path = os.path.join(out_dir, os.path.basename(ckpt_path))
        attempted.append(final_path)

        usage = shutil.disk_usage(out_dir)
        reserve = max(2 * 1024**3, int(estimate * 0.15))
        required = estimate + reserve
        if usage.free < required:
            print(
                f"  Checkpoint skip at {final_path}: free={_format_bytes(usage.free)} "
                f"< required~{_format_bytes(required)} (estimate={_format_bytes(estimate)})"
            )
            continue

        tmp_path = f"{final_path}.tmp-{os.getpid()}"
        try:
            torch.save(state_dict, tmp_path)
            os.replace(tmp_path, final_path)
            return final_path
        except (RuntimeError, OSError) as e:
            msg = str(e)
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            if "file write failed" in msg or "unexpected pos" in msg or "No space left on device" in msg:
                print(f"  Checkpoint write failed at {final_path}: {msg}")
                continue
            raise

    raise RuntimeError(
        "Failed to save checkpoint in all candidate locations: "
        + ", ".join(attempted)
        + ". Set --checkpoint_dir to a volume with enough space."
    )


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def compute_loss(model, batch, device):
    """Compute total/hot/cold cross-entropy losses for a batch."""
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids, attention_mask)
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_labels = labels.reshape(-1)
        valid_mask = (attention_mask.reshape(-1) > 0) & (flat_labels != PAD_TOKEN_ID)
        token_losses = F.cross_entropy(
            flat_logits,
            flat_labels,
            reduction="none",
        )

    if valid_mask.any():
        total_loss = token_losses[valid_mask].mean()
    else:
        total_loss = token_losses.mean()

    raw_model = model.module if hasattr(model, "module") else model
    if hasattr(raw_model, "_orig_mod"):
        raw_model = raw_model._orig_mod

    hot_loss = None
    cold_loss = None
    if hasattr(raw_model, "token_partition_masks"):
        hot_mask, cold_mask = raw_model.token_partition_masks(labels)
        if hot_mask is not None and cold_mask is not None:
            hot_mask = hot_mask.reshape(-1) & valid_mask
            cold_mask = cold_mask.reshape(-1) & valid_mask
            if hot_mask.any():
                hot_loss = token_losses[hot_mask].mean()
            if cold_mask.any():
                cold_loss = token_losses[cold_mask].mean()

    return total_loss, hot_loss, cold_loss


def get_lr(step: int, warmup_steps: int, total_steps: int, max_lr: float, min_lr: float) -> float:
    """Linear warmup then cosine decay (hits min_lr at final scheduled step)."""
    if total_steps <= 1:
        return min_lr

    warmup_steps = max(0, min(warmup_steps, total_steps - 1))
    if warmup_steps > 0 and step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    # Ensure final scheduled step reaches min_lr exactly with cosine anneal.
    decay_den = max(1, total_steps - warmup_steps - 1)
    decay_step = min(max(step - warmup_steps, 0), decay_den)
    decay_ratio = decay_step / decay_den
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + (max_lr - min_lr) * cosine


@torch.no_grad()
def evaluate(model, val_loader, device, max_batches: int = 20):
    """Evaluate model on validation set (lightweight, uses BF16)."""
    model.eval()
    total_loss = 0.0
    total_hot_loss = 0.0
    total_cold_loss = 0.0
    n_batches = 0
    n_hot_batches = 0
    n_cold_batches = 0
    
    for batch in val_loader:
        loss, hot_loss, cold_loss = compute_loss(model, batch, device)
        total_loss += loss.item()
        if hot_loss is not None:
            total_hot_loss += hot_loss.item()
            n_hot_batches += 1
        if cold_loss is not None:
            total_cold_loss += cold_loss.item()
            n_cold_batches += 1
        n_batches += 1
        if n_batches >= max_batches:
            break
    
    model.train()
    return {
        "loss": total_loss / n_batches,
        "hot_loss": (total_hot_loss / n_hot_batches) if n_hot_batches > 0 else None,
        "cold_loss": (total_cold_loss / n_cold_batches) if n_cold_batches > 0 else None,
    }


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
    warmup_steps: int = 100,
    grad_clip: float = 1.0,
    use_wandb: bool = True,
    use_ddp: bool = False,
    first_epoch_steps: int | None = None,
    hold_min_after_first_epoch: bool = False,
):
    """Main training loop with train logging every eval_every steps."""
    model.train()
    raw_model = unwrap_model(model)
    num_params = raw_model.count_parameters(count_zeros=True)
    min_lr = max_lr * 0.2

    train_iter = iter(train_loader)
    running_loss = 0.0
    running_hot_loss = 0.0
    running_cold_loss = 0.0
    total_tokens = 0
    epoch = 0
    t0 = time.time()
    world_size = get_world_size(use_ddp)
    switched_to_svd = False
    svd_switch_step = None
    reset_lr_on_structured_switch = False
    is_lora_structured_model = hasattr(raw_model, "lora_ffn_layers")
    if hasattr(raw_model, "convert_full_to_hotcold_svd"):
        switch_frac = float(getattr(raw_model, "svd_switch_fraction", 0.5))
        if is_lora_structured_model:
            # LoRA hybrid path: force a dense warmup phase before enabling LoRA.
            switch_frac = 0.20
            reset_lr_on_structured_switch = True
        switch_frac = min(max(switch_frac, 0.0), 1.0)
        svd_switch_step = min(num_steps - 1, max(0, int(num_steps * switch_frac)))

    hot_batches = 0
    cold_batches = 0

    pbar = tqdm(range(num_steps), desc="Training", disable=not is_main(use_ddp))
    for step in pbar:
        if (not switched_to_svd) and svd_switch_step is not None and step >= svd_switch_step:
            raw_model.convert_full_to_hotcold_svd()
            switched_to_svd = True
            if is_main(use_ddp):
                if is_lora_structured_model:
                    print(f"\n  Switched LoRA regime at step {step}: dense-only -> dense+LoRA")
                    print("  Resetting LR schedule from switch step for remaining training")
                else:
                    print(f"\n  Switched embedding regime at step {step}: dense -> hot/cold SVD")

        if reset_lr_on_structured_switch and switched_to_svd and svd_switch_step is not None:
            # Restart warmup+decay schedule when LoRA gets activated.
            phase_step = max(0, step - svd_switch_step)
            phase_total_steps = max(1, num_steps - svd_switch_step)
            lr = get_lr(phase_step, warmup_steps, phase_total_steps, max_lr, min_lr)
        else:
            if hold_min_after_first_epoch and first_epoch_steps is not None:
                schedule_steps = max(1, min(num_steps, first_epoch_steps))
                if step >= schedule_steps:
                    lr = min_lr
                else:
                    lr = get_lr(step, warmup_steps, schedule_steps, max_lr, min_lr)
            else:
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
        loss, hot_loss, cold_loss = compute_loss(model, batch, device)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        running_loss += loss.item()
        if hot_loss is not None:
            running_hot_loss += hot_loss.item()
            hot_batches += 1
        if cold_loss is not None:
            running_cold_loss += cold_loss.item()
            cold_batches += 1

        if (step + 1) % eval_every == 0:
            torch.cuda.synchronize()
            dt = time.time() - t0
            tokens_interval = tokens_this_step * eval_every
            mfu = 6 * num_params * tokens_interval / (GPU_PEAK_TFLOPS * 1e12 * dt * world_size)

            total_flops = 6 * num_params * total_tokens
            train_loss = running_loss / eval_every
            train_hot_loss = (running_hot_loss / hot_batches) if hot_batches > 0 else None
            train_cold_loss = (running_cold_loss / cold_batches) if cold_batches > 0 else None
            raw_model = unwrap_model(model)
            nonzero_params = raw_model.count_parameters(count_zeros=False)

            if is_main(use_ddp):
                postfix = {"train": f"{train_loss:.3f}", "mfu": f"{mfu:.1%}"}
                if train_hot_loss is not None:
                    postfix["train_hot"] = f"{train_hot_loss:.3f}"
                if train_cold_loss is not None:
                    postfix["train_cold"] = f"{train_cold_loss:.3f}"
                pbar.set_postfix(postfix)
                if use_wandb:
                    import wandb
                    log_payload = {
                        "train/loss": train_loss,
                        "train/lr": lr,
                        "train/total_tokens": total_tokens,
                        "train/total_flops": total_flops,
                        "train/epoch": epoch,
                        "params/nonzero": nonzero_params,
                        "mfu": mfu,
                        "step": step + 1,
                    }
                    if train_hot_loss is not None:
                        log_payload["train/hot_loss"] = train_hot_loss
                    if train_cold_loss is not None:
                        log_payload["train/cold_loss"] = train_cold_loss
                    wandb.log(log_payload)

            running_loss = 0.0
            running_hot_loss = 0.0
            running_cold_loss = 0.0
            hot_batches = 0
            cold_batches = 0
            t0 = time.time()

    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--no_pin_memory", action="store_true")
    parser.add_argument("--no_persistent_workers", action="store_true")
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument(
        "--dataset_epochs",
        type=int,
        default=1,
        help="Number of dataset epochs to train in one continuous run (no LR reset)",
    )
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--model", type=str, default="baseline",
                        choices=["baseline", "gqa_only", "topk_only", "baseline_plus", "mla", "hotcold_mla", "hotcold_svd", "twostage_svd", "mla_twostage_svd_mem12_monarch", "mla_twostage_svd_mem12_binarydp", "dp_shared_memory", "loop_top4x3_attnres", "mla_hybrid_loop12", "mla_hybrid_loop12_monarch", "mla_hybrid_loop12_monarch_attn_svd_ffn", "mla_hybrid_loop12_monarch_attn_svd_ffn_binarydp", "mla_hybrid_loop12_monarch_attn_lora_ffn", "mla_hybrid_loop12_monarch_attn_lora_ffn_binarydp"],
                        help="Model variant")
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
                        help="For twostage_svd/hotcold_mla/mla_twostage_svd_mem12_monarch/mla_twostage_svd_mem12_binarydp/mla_hybrid_loop12/mla_hybrid_loop12_monarch/mla_hybrid_loop12_monarch_attn_svd_ffn/mla_hybrid_loop12_monarch_attn_svd_ffn_binarydp/mla_hybrid_loop12_monarch_attn_lora_ffn/mla_hybrid_loop12_monarch_attn_lora_ffn_binarydp: fraction of total steps before switching dense -> structured phase")
    parser.add_argument("--monarch_block_size", type=int, default=32,
                        help="Monarch block size for MLA O-proj in mla_twostage_svd_mem12_monarch/mla_twostage_svd_mem12_binarydp")
    parser.add_argument("--memory_layers", type=int, default=12,
                        help="Number of memory layers (must be 12 for mla_twostage_svd_mem12_monarch/mla_twostage_svd_mem12_binarydp)")
    parser.add_argument("--mem_n_keys", type=int, default=256,
                        help="Memory key table size per axis for mla_twostage_svd_mem12_monarch/mla_twostage_svd_mem12_binarydp")
    parser.add_argument("--mem_heads", type=int, default=4,
                        help="Memory heads for mla_twostage_svd_mem12_monarch/mla_twostage_svd_mem12_binarydp")
    parser.add_argument("--mem_knn", type=int, default=32,
                        help="Memory k-NN lookup count for mla_twostage_svd_mem12_monarch/mla_twostage_svd_mem12_binarydp")
    parser.add_argument("--mem_k_dim", type=int, default=None,
                        help="Memory key dimension for mla_twostage_svd_mem12_monarch/mla_twostage_svd_mem12_binarydp")
    parser.add_argument("--mem_v_dim", type=int, default=None,
                        help="Memory value dimension for mla_twostage_svd_mem12_monarch/mla_twostage_svd_mem12_binarydp")
    parser.add_argument("--mem_q_rank", type=int, default=None,
                        help="Low-rank query latent dim for memory lookups in mla_twostage_svd_mem12_monarch/mla_twostage_svd_mem12_binarydp")
    parser.add_argument("--no_mem_share_values", action="store_true",
                        help="Disable shared value table across memory layers")
    parser.add_argument("--qk_norm", action="store_true",
                        help="Enable RMS q/k normalization in memory lookups")
    parser.add_argument("--wandb_project", type=str, default="weightless")
    parser.add_argument("--wandb_entity", type=str, default="kavn",
                        help="W&B workspace/entity to log runs to")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Name for this run (used for checkpoint filename)")
    parser.set_defaults(save_checkpoint=True)
    parser.add_argument("--save_checkpoint", dest="save_checkpoint", action="store_true",
                        help="Save model checkpoint at end of training (default: enabled)")
    parser.add_argument("--no_save_checkpoint", dest="save_checkpoint", action="store_false",
                        help="Disable end-of-training checkpoint save")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Primary directory for end-of-training checkpoint files")
    parser.add_argument("--checkpoint_fallback_dir", type=str, default=None,
                        help="Optional fallback directory if primary checkpoint write fails")
    args = parser.parse_args()
    if args.model in {
        "mla_hybrid_loop12",
        "mla_hybrid_loop12_monarch",
        "mla_hybrid_loop12_monarch_attn_svd_ffn",
        "mla_hybrid_loop12_monarch_attn_svd_ffn_binarydp",
        "mla_hybrid_loop12_monarch_attn_lora_ffn",
        "mla_hybrid_loop12_monarch_attn_lora_ffn_binarydp",
    } and args.n_layers == 8:
        # Keep CLI ergonomic: these variants are fixed to 12 layers.
        args.n_layers = 12
    if args.model == "mla_hybrid_loop12_monarch" and args.d_ff == 2048:
        # Monarch FFN path in this variant is square: d_ff must match d_model.
        args.d_ff = args.d_model
    if args.model in {
        "mla_hybrid_loop12_monarch_attn_svd_ffn",
        "mla_hybrid_loop12_monarch_attn_svd_ffn_binarydp",
        "mla_hybrid_loop12_monarch_attn_lora_ffn",
        "mla_hybrid_loop12_monarch_attn_lora_ffn_binarydp",
    } and args.d_ff == 2048:
        # Keep this hybrid path at d_ff=1024 unless user explicitly overrides.
        args.d_ff = 1024
    if args.dataset_epochs < 1:
        raise ValueError(f"--dataset_epochs must be >= 1, got {args.dataset_epochs}")
    user_set_num_steps = args.num_steps is not None

    # Autoscale max_lr: args.max_lr is calibrated at d_model=768
    # Wider models use lower LR (muP-style sqrt scaling)
    base_lr = args.max_lr
    args.max_lr = base_lr * (768 / args.d_model) ** 0.5

    # DDP setup (optional -- works with plain `python train.py`)
    local_rank, world_size, use_ddp = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    use_wandb = not args.no_wandb
    tokens_per_step = args.batch_size * 512 * world_size
    steps_per_epoch = (DATASET_TOKENS_PER_EPOCH + tokens_per_step - 1) // tokens_per_step
    if args.num_steps is None:
        total_tokens_target = DATASET_TOKENS_PER_EPOCH * args.dataset_epochs
        args.num_steps = (total_tokens_target + tokens_per_step - 1) // tokens_per_step
    if args.eval_every is None:
        # Keep eval cadence tied to a single epoch, even for multi-epoch continuous runs.
        args.eval_every = max(1, int(0.01 * steps_per_epoch))
    if args.svd_switch_fraction is None:
        if args.model in {
            "mla_hybrid_loop12_monarch_attn_lora_ffn",
            "mla_hybrid_loop12_monarch_attn_lora_ffn_binarydp",
        }:
            args.svd_switch_fraction = 0.3
        else:
            args.svd_switch_fraction = (
                1.0 / 3.0
                if args.model in {
                    "mla_hybrid_loop12",
                    "mla_hybrid_loop12_monarch",
                    "mla_hybrid_loop12_monarch_attn_svd_ffn",
                    "mla_hybrid_loop12_monarch_attn_svd_ffn_binarydp",
                }
                else 0.5
            )

    if is_main(use_ddp):
        if args.d_model != 768:
            print(f"  Autoscaled max_lr from {base_lr:.2e} to {args.max_lr:.2e} (d_model={args.d_model})")
        else:
            print(f"  max_lr={args.max_lr:.2e} (d_model={args.d_model})")
        if user_set_num_steps:
            print(f"  dataset_epochs={args.dataset_epochs} (num_steps manually set)")
        else:
            print(f"  dataset_epochs={args.dataset_epochs} (steps auto-derived)")
        if use_ddp:
            print(f"  DDP: rank {local_rank}, world_size {world_size}")
        else:
            print(f"  Single-GPU mode (use torchrun for DDP)")
        if args.dataset_epochs > 1:
            print("  LR schedule: decay to min_lr in epoch 1, then hold min_lr for remaining epochs")

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
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.run_name,
            config={**vars(args), "model_variant": args.model},
        )

    # Data
    rank = get_rank(use_ddp)
    if is_main(use_ddp):
        print("  Setting up data loaders...")
    train_loader = get_dataloader(split="train", batch_size=args.batch_size,
                                  streaming=True, num_workers=args.num_workers,
                                  pin_memory=not args.no_pin_memory,
                                  persistent_workers=not args.no_persistent_workers,
                                  rank=rank, world_size=world_size)
    val_loader = get_dataloader(split="test", batch_size=args.batch_size,
                                streaming=True, num_workers=args.num_workers,
                                pin_memory=not args.no_pin_memory,
                                persistent_workers=not args.no_persistent_workers,
                                rank=rank, world_size=world_size)

    # Model
    if is_main(use_ddp):
        print(f"  Creating model (variant={args.model}, BF16 + torch.compile)...")
    model_kwargs = dict(
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
    model = create_model(**model_kwargs)
    model.to(device)
    model = torch.compile(model)

    if use_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        ddp_find_unused = args.model in {
            "twostage_svd",
            "hotcold_mla",
            "mla_twostage_svd_mem12_monarch",
            "mla_twostage_svd_mem12_binarydp",
            "mla_hybrid_loop12",
            "mla_hybrid_loop12_monarch",
            "mla_hybrid_loop12_monarch_attn_svd_ffn",
            "mla_hybrid_loop12_monarch_attn_svd_ffn_binarydp",
            "mla_hybrid_loop12_monarch_attn_lora_ffn",
            "mla_hybrid_loop12_monarch_attn_lora_ffn_binarydp",
        }
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=ddp_find_unused,
        )
        if is_main(use_ddp) and ddp_find_unused:
            print("  DDP: find_unused_parameters=True (required for two-stage dense->structured switch)")

    raw_model = unwrap_model(model)

    total_params = raw_model.count_parameters(count_zeros=True)
    nonzero_params = raw_model.count_parameters(count_zeros=False)

    profile_count_reuse = hasattr(raw_model, "lora_ffn_layers")

    if is_main(use_ddp):
        print(f"  Total parameters:    {total_params:,}")
        print(f"  Non-zero parameters: {nonzero_params:,}")

        # Show the bytes_per_token_infer breakdown
        profile = raw_model.get_inference_profile(count_reuse=profile_count_reuse)
        if profile_count_reuse:
            print("  Profiling with count_reuse=True (shared LoRA FFN bases deduplicated).")
        print_profile(profile)
        post_switch_profile = None
        if hasattr(raw_model, "convert_full_to_hotcold_svd"):
            # Compute a projected post-switch profile at startup so we can compare
            # the eventual compressed inference footprint before training begins.
            try:
                preview_model = create_model(**model_kwargs)
                preview_model.convert_full_to_hotcold_svd()
                post_switch_profile = preview_model.get_inference_profile(
                    count_reuse=profile_count_reuse
                )
                print("  Projected post-switch inference profile (dense->structured):")
                print_profile(post_switch_profile)
                del preview_model
            except Exception as e:
                print(f"  Skipping projected post-switch profile: {e}")
        projected_final_profile = post_switch_profile or profile
        print(
            "  Projected final bytes_per_token_infer "
            f"(end-of-training architecture): {projected_final_profile.total_bytes:,} bytes"
        )

        if use_wandb:
            import wandb
            bd = profile.breakdown_dict()
            log_payload = {
                "params/total": total_params,
                "params/nonzero": nonzero_params,
                "metric/bytes_per_token_infer": profile.total_bytes,
                "metric/projected_final_bytes_per_token_infer": projected_final_profile.total_bytes,
                **{f"metric/{k}": v for k, v in bd.items()},
                "metric/unique_param_bytes": profile.unique_param_bytes,
                "metric/unique_opt_state_bytes": profile.unique_opt_state_bytes,
            }
            if post_switch_profile is not None:
                post_bd = post_switch_profile.breakdown_dict()
                log_payload.update({
                    "metric_postswitch/bytes_per_token_infer": post_switch_profile.total_bytes,
                    **{f"metric_postswitch/{k}": v for k, v in post_bd.items()},
                    "metric_postswitch/unique_param_bytes": post_switch_profile.unique_param_bytes,
                    "metric_postswitch/unique_opt_state_bytes": post_switch_profile.unique_opt_state_bytes,
                })
            wandb.log(log_payload)

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
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        use_wandb=use_wandb,
        use_ddp=use_ddp,
        first_epoch_steps=steps_per_epoch,
        hold_min_after_first_epoch=args.dataset_epochs > 1,
    )

    # End-of-training summary
    if is_main(use_ddp):
        raw_model_final = unwrap_model(model)
        final_val_metrics = evaluate(model, val_loader, device)
        final_val_loss = final_val_metrics["loss"]
        profile = raw_model_final.get_inference_profile(count_reuse=profile_count_reuse)
        print("\n" + "=" * 60)
        print("  TRAINING COMPLETE")
        print("=" * 60)
        print(f"  Final val_loss:         {final_val_loss:.4f}  (target < {GOAL_VAL_LOSS})")
        if final_val_metrics["hot_loss"] is not None:
            print(f"  Final val_hot_loss:     {final_val_metrics['hot_loss']:.4f}")
        if final_val_metrics["cold_loss"] is not None:
            print(f"  Final val_cold_loss:    {final_val_metrics['cold_loss']:.4f}")
        print(f"  post bytes_per_token_infer:  {profile.total_bytes:,} bytes")
        print_profile(profile)
        if final_val_loss < GOAL_VAL_LOSS:
            print(f"  GOAL ACHIEVED!")
        else:
            print(f"  Goal not yet reached (val_loss {final_val_loss:.4f} >= {GOAL_VAL_LOSS})")
        print()

        # Save checkpoint
        if args.save_checkpoint:
            ckpt_path = f"{args.checkpoint_dir}/{args.run_name}.pt"
            try:
                saved_path = _safe_save_checkpoint(
                    raw_model_final.state_dict(),
                    ckpt_path=ckpt_path,
                    fallback_dir=args.checkpoint_fallback_dir,
                )
                print(f"  Checkpoint saved to {saved_path}")
                config_path = os.path.splitext(saved_path)[0] + ".config.json"
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "model_variant": args.model,
                            "model_kwargs": model_kwargs,
                        },
                        f,
                        indent=2,
                    )
                print(f"  Checkpoint config saved to {config_path}")
            except Exception as e:
                # End-of-training checkpoint failure should not crash the whole run.
                print(f"  WARNING: checkpoint save failed: {e}")

    if use_wandb and is_main(use_ddp):
        import wandb
        wandb.finish()
    if use_ddp:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
