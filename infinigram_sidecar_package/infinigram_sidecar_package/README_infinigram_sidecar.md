# Infini-gram sidecar

This is an **eval-only sidecar**. It does **not** change the base model architecture or training loop.

The intended workflow is:

1. Train your normal model and save a checkpoint.
2. Build a compact n-gram sidecar index from the training split.
3. Load the checkpoint for evaluation/inference.
4. At eval time, the wrapper:
   - checks the longest stored suffix first (`max_order -> min_order` backoff),
   - rejects any lookup that exceeds the sidecar byte budget,
   - requires the base model to also assign enough probability to the retrieved continuation,
   - and only then fuses the retrieval distribution into the model output as an exact sparse probability mixture.
   - otherwise it falls back to the original model distribution unchanged.

## Files

- `infinigram_sidecar.py` — compact index builder, mmap index loader, and eval wrapper.
- `build_infinigram_sidecar.py` — builds the sidecar index from the train split.
- `eval_with_infinigram_sidecar.py` — evaluates a saved checkpoint with the sidecar attached.

## Why this is fast enough

The sidecar is not a suffix array. Instead it stores exact fixed-order contexts in a compact hash table:

- fixed-size slot reads for lookup,
- one contiguous payload read for top-k next-token counts,
- mmap-backed arrays, so no Python dict overhead in the serving path,
- optional `apply_to_last_token_only` path for fast generation.

That keeps the extra bytes-per-token read small and predictable.

## Checkpoint loading note

The patched `train.py` now saves both:

- `checkpoints/<run_name>.pt` — the raw `state_dict`, and
- `checkpoints/<run_name>.config.json` — the model construction config.

`eval_with_infinigram_sidecar.py` will automatically use the adjacent `.config.json` file if it exists, so evaluation can be run from the checkpoint path directly. If that config file is missing, pass the normal model architecture flags (`--model`, `--d_model`, `--n_layers`, and any variant-specific args).

## Example build

```bash
python build_infinigram_sidecar.py \
  --out_dir sidecars/train_ngram_468 \
  --split train \
  --orders 4,6,8 \
  --topk 8 \
  --min_count 2 \
  --batch_size 32
```

## Example eval

```bash
python eval_with_infinigram_sidecar.py \
  --checkpoint checkpoints/my_model.pt \
  --sidecar_dir sidecars/train_ngram_468 \
  --min_order 4 \
  --sidecar_weight 0.70 \
  --min_model_prob 0.02 \
  --model_topk_agree 8 \
  --min_sidecar_confidence 0.55 \
  --max_sidecar_bytes_per_token 256
```

## Important defaults

- `--apply_to_all_positions` is **off** by default.
  - Default behavior only adjusts the **last position** for speed.
  - Turn it on if you want full-sequence eval / perplexity with the sidecar.
- `sidecar_weight` is a **mixture weight**, not a hard overwrite.
- `model_topk_agree` gates the retrieval so the sidecar only fires when the base model already finds the same continuation plausible.

## Implementation notes

The fusion is a sparse probability mixture:

- if the sidecar does **not** fire, output = base model log-probabilities
- if it **does** fire, only the retrieved candidate log-probabilities are boosted
- non-candidate log-probabilities are shifted by the common mixture constant

This gives the effect you asked for: the continuation becomes much more likely when both the retrieval path and the model agree, but the model stays unchanged otherwise.
