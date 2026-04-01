from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from infinigram_sidecar import (
    CompactNgramIndex,
    InfinigramSidecarWrapper,
    SidecarRuntimeConfig,
    load_model_checkpoint,
)

PAD_TOKEN_ID = 50256



def compute_loss(model, batch, device):
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    autocast_enabled = device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
        output = model(input_ids, attention_mask)
        flat_output = output.reshape(-1, output.size(-1))
        flat_labels = labels.reshape(-1)
        valid_mask = (attention_mask.reshape(-1) > 0) & (flat_labels != PAD_TOKEN_ID)
        if bool(getattr(model, "returns_log_probs", False)):
            token_losses = F.nll_loss(flat_output, flat_labels, reduction="none")
        else:
            token_losses = F.cross_entropy(flat_output, flat_labels, reduction="none")

    if valid_mask.any():
        return token_losses[valid_mask].mean()
    return token_losses.mean()


@torch.no_grad()
def evaluate(model, val_loader, device, max_batches: int = 20):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in val_loader:
        loss = compute_loss(model, batch, device)
        total_loss += float(loss.item())
        n_batches += 1
        if n_batches >= max_batches:
            break
    return total_loss / max(1, n_batches)



def _load_checkpoint_config(config_path: str | None, checkpoint_path: str) -> dict | None:
    candidates: list[Path] = []
    if config_path is not None:
        candidates.append(Path(config_path))
    ckpt = Path(checkpoint_path)
    candidates.extend([
        ckpt.with_suffix('.config.json'),
        ckpt.with_suffix('.meta.json'),
    ])
    for cand in candidates:
        if cand.exists():
            with open(cand, 'r', encoding='utf-8') as f:
                return json.load(f)
    return None


def _apply_checkpoint_config(args) -> None:
    payload = _load_checkpoint_config(getattr(args, 'checkpoint_config', None), args.checkpoint)
    if payload is None:
        return
    model_kwargs = payload.get('model_kwargs', payload)
    variant = payload.get('model_variant', model_kwargs.get('variant', payload.get('variant')))
    if variant is not None:
        args.model = variant
    field_map = {
        'd_model': 'd_model',
        'n_layers': 'n_layers',
        'n_heads': 'n_heads',
        'd_ff': 'd_ff',
        'kv_lora_rank': 'kv_lora_rank',
        'q_lora_rank': 'q_lora_rank',
        'qk_nope_head_dim': 'qk_nope_head_dim',
        'qk_rope_head_dim': 'qk_rope_head_dim',
        'v_head_dim': 'v_head_dim',
        'hot_token_k': 'hot_token_k',
        'cold_latent_dim': 'cold_latent_dim',
        'hot_token_cache_path': 'hot_token_cache_path',
        'svd_switch_fraction': 'svd_switch_fraction',
        'monarch_block_size': 'monarch_block_size',
        'memory_layers': 'memory_layers',
        'mem_n_keys': 'mem_n_keys',
        'mem_heads': 'mem_heads',
        'mem_knn': 'mem_knn',
        'mem_k_dim': 'mem_k_dim',
        'mem_v_dim': 'mem_v_dim',
        'mem_q_rank': 'mem_q_rank',
        'mem_share_values': 'no_mem_share_values',
        'qk_norm': 'qk_norm',
    }
    for src, dst in field_map.items():
        if src not in model_kwargs:
            continue
        value = model_kwargs[src]
        if src == 'mem_share_values':
            setattr(args, dst, not bool(value))
        else:
            setattr(args, dst, value)



def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--checkpoint_config", type=str, default=None)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--kv_lora_rank", type=int, default=None)
    parser.add_argument("--q_lora_rank", type=int, default=None)
    parser.add_argument("--qk_nope_head_dim", type=int, default=None)
    parser.add_argument("--qk_rope_head_dim", type=int, default=None)
    parser.add_argument("--v_head_dim", type=int, default=None)
    parser.add_argument("--hot_token_k", type=int, default=2000)
    parser.add_argument("--cold_latent_dim", type=int, default=128)
    parser.add_argument("--hot_token_cache_path", type=str, default="cache/hot_tokens_train1p3b_top2000.pt")
    parser.add_argument("--svd_switch_fraction", type=float, default=0.5)
    parser.add_argument("--monarch_block_size", type=int, default=32)
    parser.add_argument("--memory_layers", type=int, default=12)
    parser.add_argument("--mem_n_keys", type=int, default=256)
    parser.add_argument("--mem_heads", type=int, default=4)
    parser.add_argument("--mem_knn", type=int, default=32)
    parser.add_argument("--mem_k_dim", type=int, default=None)
    parser.add_argument("--mem_v_dim", type=int, default=None)
    parser.add_argument("--mem_q_rank", type=int, default=None)
    parser.add_argument("--no_mem_share_values", action="store_true")
    parser.add_argument("--qk_norm", action="store_true")



def build_model_from_args(args) -> torch.nn.Module:
    from model import create_model

    if args.model is None:
        raise ValueError("Model variant is unknown. Pass --model, or save/use checkpoint.config.json next to the checkpoint.")

    if args.model in {
        "mla_hybrid_loop12",
        "mla_hybrid_loop12_monarch",
        "mla_hybrid_loop12_monarch_attn_svd_ffn",
        "mla_hybrid_loop12_monarch_attn_svd_ffn_binarydp",
        "mla_hybrid_loop12_monarch_attn_svd_ffn_nobinarydp",
    } and args.n_layers == 8:
        args.n_layers = 12
    if args.model == "mla_hybrid_loop12_monarch" and args.d_ff == 2048:
        args.d_ff = args.d_model

    kwargs = dict(
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
    return create_model(**kwargs)



def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint with a separate Infini-gram sidecar.")
    add_model_args(parser)
    parser.add_argument("--sidecar_dir", type=str, required=True)
    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--no_pin_memory", action="store_true")
    parser.add_argument("--no_persistent_workers", action="store_true")
    parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compare_base", action="store_true")

    parser.add_argument("--min_order", type=int, default=4)
    parser.add_argument("--max_order", type=int, default=None)
    parser.add_argument("--sidecar_weight", type=float, default=0.70)
    parser.add_argument("--sidecar_temperature", type=float, default=1.0)
    parser.add_argument("--min_model_prob", type=float, default=0.02)
    parser.add_argument("--model_topk_agree", type=int, default=8)
    parser.add_argument("--require_argmax_agreement", action="store_true")
    parser.add_argument("--min_sidecar_confidence", type=float, default=0.55)
    parser.add_argument("--min_count", type=int, default=2)
    parser.add_argument("--max_sidecar_bytes_per_token", type=int, default=256)
    parser.add_argument("--apply_to_all_positions", action="store_true")

    args = parser.parse_args()
    _apply_checkpoint_config(args)

    device = torch.device(args.device)
    model = build_model_from_args(args)
    load_model_checkpoint(args.checkpoint, model, strict=True)
    model.to(device)
    model.eval()

    index = CompactNgramIndex(args.sidecar_dir)
    runtime_cfg = SidecarRuntimeConfig(
        min_order=args.min_order,
        max_order=args.max_order,
        sidecar_weight=args.sidecar_weight,
        sidecar_temperature=args.sidecar_temperature,
        min_model_prob=args.min_model_prob,
        model_topk_agree=args.model_topk_agree,
        require_argmax_agreement=args.require_argmax_agreement,
        min_sidecar_confidence=args.min_sidecar_confidence,
        min_count=args.min_count,
        max_sidecar_bytes_per_token=args.max_sidecar_bytes_per_token,
        apply_to_last_token_only=not args.apply_to_all_positions,
    )
    wrapper = InfinigramSidecarWrapper(model, index, runtime_cfg)
    wrapper.to(device)
    wrapper.eval()

    from data import get_dataloader

    val_loader = get_dataloader(
        split=args.eval_split,
        batch_size=args.batch_size,
        streaming=True,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        persistent_workers=not args.no_persistent_workers,
        rank=0,
        world_size=1,
    )

    base_profile = None
    base_total_bytes = None
    if hasattr(model, "get_inference_profile"):
        try:
            base_profile = model.get_inference_profile()
            base_total_bytes = int(getattr(base_profile, "total_bytes"))
        except Exception:
            base_profile = None
            base_total_bytes = None

    print("=" * 60)
    print("SIDEcar EVAL CONFIG")
    print("=" * 60)
    print(f"checkpoint:         {Path(args.checkpoint).resolve()}")
    print(f"sidecar_dir:        {Path(args.sidecar_dir).resolve()}")
    print(f"model:              {args.model}")
    print(f"device:             {device}")
    print(f"apply_to_last_only: {runtime_cfg.apply_to_last_token_only}")
    print(f"wrapper_outputs:    log_probs")
    print(f"orders_in_index:    {index.orders}")
    print(f"min_order:          {runtime_cfg.min_order}")
    print(f"max_order:          {runtime_cfg.max_order}")
    print(f"sidecar_weight:     {runtime_cfg.sidecar_weight}")
    print(f"min_model_prob:     {runtime_cfg.min_model_prob}")
    print(f"model_topk_agree:   {runtime_cfg.model_topk_agree}")
    print(f"min_sidecar_conf:   {runtime_cfg.min_sidecar_confidence}")
    print(f"max_sidecar_bytes:  {runtime_cfg.max_sidecar_bytes_per_token}")
    print(f"index_max_lookup:   {index.max_lookup_bytes}")
    if base_total_bytes is not None:
        print(f"base_bytes/token:   {base_total_bytes:,}")
        print(f"worst_case_combined:{base_total_bytes + wrapper.extra_bytes_per_token_infer:,}")

    base_loss = None
    if args.compare_base:
        base_loss = evaluate(model, val_loader, device, max_batches=args.max_batches)
        wrapper.reset_stats()
    val_loss = evaluate(wrapper, val_loader, device, max_batches=args.max_batches)
    print("=" * 60)
    print("SIDEcar EVAL RESULT")
    print("=" * 60)
    if base_loss is not None:
        print(f"base_val_loss:      {base_loss:.6f}")
        print(f"sidecar_val_loss:   {val_loss:.6f}")
        print(f"delta_loss:         {val_loss - base_loss:+.6f}")
    else:
        print(f"val_loss:           {val_loss:.6f}")
    stats = wrapper.stats.as_dict()
    print(f"lookup_attempts:    {stats['lookup_attempts']:,}")
    print(f"lookup_hits:        {stats['lookup_hits']:,}")
    print(f"fused:              {stats['fused']:,}")
    print(f"bytes_read_total:   {stats['bytes_read']:,}")
    if stats["lookup_attempts"] > 0:
        hit_rate = stats["lookup_hits"] / stats["lookup_attempts"]
        fuse_rate = stats["fused"] / stats["lookup_attempts"]
        print(f"hit_rate:           {hit_rate:.4%}")
        print(f"fuse_rate:          {fuse_rate:.4%}")
    if stats["lookup_hits"] > 0:
        avg_bytes = stats["bytes_read"] / stats["lookup_hits"]
        print(f"avg_bytes_per_hit:  {avg_bytes:.2f}")


if __name__ == "__main__":
    main()
