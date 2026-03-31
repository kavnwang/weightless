# weightless

How few bytes can we touch per token at inference time?

## The Score: `bytes_per_token_infer`

Total bytes read from memory to produce **one token** during single-token autoregressive decode.
Lower is better.

The metric is broken down by component so you can see exactly where your bytes are going, and what it counts: 

- Embeddings: 1 row lookup (`d_model * dtype_bytes`)
- Attn Q/K/V/O proj: Attention projection weight reads (all layers) 
- FFN: Feed-forward weight reads: w1 + w2 + w3 (all layers)
- Norms: LayerNorm/RMSNorm parameters
- LM head: Output projection (`vocab * d_model`; 0 if tied)
- KV cache read: Reading past K/V for attention
- KV cache write: Writing new K/V entry

What counts: Everything touched during the forward pass -- weights, KV cache traffic. Memory hierarchy treated as flat (cold-cache assumption).

Supplementary metrics: `unique_param_bytes` (model size) and `unique_opt_state_bytes` (optimizer memory).

### Assumptions

- `batch=1`, single-token decode, `seq_len=512` (configurable)
- Cold cache: all weights read from memory each token
- Tied weights counted once (configurable)
- Weight dtype: bf16 (2 bytes), KV cache dtype: bf16 (2 bytes)

## Your Task

Train a model that achieves **val_loss < 3.5** on a 1.3B slice of [FineWebEdu](https://huggingface.co/datasets/kushalt/fineweb-edu-gpt2) while **minimizing `bytes_per_token_infer`**.

The baseline model achieves this loss in about an hour with the default config. Your job is to reduce the byte score while maintaining (or improving) loss.

## Dataset

This repo uses the tokenized [FineWeb-edu-gpt2](https://huggingface.co/datasets/kushalt/fineweb-edu-gpt2) dataset (GPT-2 tokenizer, 513-token sequences, 1.31B token subset).

## Setup

Using pixi (recommended):
```bash
pixi install
```

Or with pip:
```bash
pip install -r requirements.txt
```

Log in to wandb (optional but recommended):
```bash
wandb login
```

## Files

- `data.py` - Data loading from HuggingFace streaming dataset
- `model.py` - Baseline, baseline+, and MLA transformer models (modify this!)
- `train.py` - Training loop with wandb logging
- `eval.py` - Evaluation script with metric breakdown
- `metric.py` - `bytes_per_token_infer` metric and `InferenceProfile`
- `visualize.py` - Visualization of metric breakdowns

## Training

```bash
# Single GPU (no torchrun needed)
python train.py

# With a specific model variant
python train.py --model baseline
python train.py --model gqa_only
python train.py --model topk_only
python train.py --model baseline_plus
python train.py --model mla
python train.py --model mla --kv_lora_rank 96 --q_lora_rank 96 --qk_rope_head_dim 48
python train.py --model hotcold_mla --hot_token_cache_path cache/hot_tokens_train1p3b_top2000.pt --svd_switch_fraction 0.5
python build_hot_token_cache.py  # cached top-2000 hot tokens from 2% of 1.3B train split
python train.py --model hotcold_svd --hot_token_cache_path cache/hot_tokens_train1p3b_top2000.pt
python train.py --model twostage_svd --hot_token_cache_path cache/hot_tokens_train1p3b_top2000.pt --svd_switch_fraction 0.5

# Custom config
python train.py --batch_size 64 --max_lr 8e-4 --num_steps 30000 --d_model 768 --n_layers 8

# Without wandb
python train.py --no_wandb

# Multi-GPU with DDP
torchrun --nproc_per_node=2 train.py
```

The training script prints the full `bytes_per_token_infer` breakdown at the start and end of training.
The default config (d_model=768, n_layers=8) reaches val_loss < 3.5 in ~55K steps (~60 min on H100).

## Evaluation

```bash
python eval.py --checkpoint model.pt
python eval.py --checkpoint model.pt --visualize  # + save breakdown chart
```

## Visualization

Compare variants (including ablations) side-by-side:

```bash
python visualize.py                                   # compare baseline vs baseline+ by default
python visualize.py --models baseline mla             # compare selected variants
python visualize.py --models baseline gqa_only topk_only baseline_plus
python visualize.py --models baseline                  # single model breakdown
python visualize.py --seq_len 1024                     # custom context length
```

## Metric CLI

Profile any model variant without training:

```bash
python metric.py --model baseline
python metric.py --model gqa_only --seq_len 1024
python metric.py --model topk_only --seq_len 1024
python metric.py --model baseline_plus --seq_len 1024
python metric.py --model mla --seq_len 1024
python metric.py --model hotcold_mla --hot_token_cache_path cache/hot_tokens_train1p3b_top2000.pt
python metric.py --model hotcold_svd --hot_token_cache_path cache/hot_tokens_train1p3b_top2000.pt
python metric.py --model twostage_svd --hot_token_cache_path cache/hot_tokens_train1p3b_top2000.pt
```

## Baselines

Five model variants are provided:

| Variant | Description | bytes_per_token (bf16) | Key changes |
|---|---|---|---|
| `baseline` | Dense transformer | **194 MB** | Standard MHA, full SwiGLU FFN |
| `gqa_only` | GQA ablation | Depends on `n_kv_heads` | Reduced KV heads, dense FFN |
| `topk_only` | FFN sparsity ablation | Depends on `ffn_top_k` | Full MHA, top-k activation sparsity in FFN |
| `baseline_plus` | GQA + top-k FFN | **153 MB** (-21%) | Fewer KV heads, activation sparsity in FFN |
| `mla` | DeepSeek-style latent attention | Depends on MLA dims | Low-rank Q/KV compression and reduced KV cache payload |
| `hotcold_mla` | MLA + hot/cold vocab SVD | Depends on MLA + rank | MLA attention plus hot dense and cold low-rank vocab factors |
| `hotcold_svd` | Hot dense + cold SVD vocab | Depends on hot_k/rank | Top-2000 tokens dense, cold tokens rank-128 factors (tied embed/unembed) |
| `twostage_svd` | Dense then SVD switch | Phase-dependent | Train dense for first half, then convert to hot/cold SVD and continue |

The `baseline` is your starting point. `gqa_only` and `topk_only` are ablations that isolate each optimization in `baseline_plus`, while `mla` is an alternative attention architecture. Run `python visualize.py` to compare per-component effects.

## Challenge

**Goal**: `val_loss < 3.5` with minimal `bytes_per_token_infer`.

Run `python visualize.py` to see where your bytes are going and compare against baselines.
