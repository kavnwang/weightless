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
- `model.py` - Baseline and baseline+ transformer models (modify this!)
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
python train.py --model baseline_plus

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

Compare baseline vs baseline+ (or any model variants):

```bash
python visualize.py                                   # compare baseline vs baseline+
python visualize.py --models baseline                  # single model breakdown
python visualize.py --seq_len 1024                     # custom context length
```

## Metric CLI

Profile any model variant without training:

```bash
python metric.py --model baseline
python metric.py --model baseline_plus --seq_len 1024
```

## Baselines

Two model variants are provided:

| Variant | Description | bytes_per_token (bf16) | Key changes |
|---|---|---|---|
| `baseline` | Dense transformer | **194 MB** | Standard MHA, full SwiGLU FFN |
| `baseline_plus` | GQA + top-k FFN | **153 MB** (-21%) | Fewer KV heads, activation sparsity in FFN |

The `baseline` is your starting point. The `baseline_plus` is included purely as an example to show how architectural changes move the byte metric -- it is not something you need to use or build on. Run `python visualize.py` to see the per-component comparison.

## Challenge

**Goal**: `val_loss < 3.5` with minimal `bytes_per_token_infer`.

Run `python visualize.py` to see where your bytes are going and compare against baselines.
