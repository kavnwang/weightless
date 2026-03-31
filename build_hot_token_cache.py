"""Build and cache top-K hot tokens from a 2% sweep of the 1.3B train split.

Usage:
    python build_hot_token_cache.py
    python build_hot_token_cache.py --top_k 2000 --sample_frac 0.02
"""

from __future__ import annotations

import argparse
import math
import os
import random

import torch
from huggingface_hub import HfFileSystem
from tqdm import tqdm

from data import DATASET_REPO, SUBSET, TRAIN_SPLIT_1P3B, get_parquet_files
from model import VOCAB_SIZE


DEFAULT_CACHE_PATH = "cache/hot_tokens_train1p3b_top2000.pt"


def _iter_input_ids(file_paths: list[str]):
    import pyarrow.parquet as pq

    fs = HfFileSystem()
    for file_path in file_paths:
        with fs.open(file_path, "rb") as f:
            table = pq.read_table(f, columns=["input_ids"])
            col = table["input_ids"]
            for i in range(len(table)):
                yield col[i].as_py()


def build_hot_token_cache(
    split: str,
    top_k: int,
    sample_frac: float,
    seed: int,
    cache_path: str,
) -> dict:
    if split != TRAIN_SPLIT_1P3B:
        raise ValueError(
            f"This cache builder is restricted to the 1.3B train split: "
            f"expected split='{TRAIN_SPLIT_1P3B}', got split='{split}'."
        )
    if not (0.0 < sample_frac <= 1.0):
        raise ValueError(f"sample_frac must be in (0, 1], got {sample_frac}")
    if not (1 <= top_k <= VOCAB_SIZE):
        raise ValueError(f"top_k must be in [1, {VOCAB_SIZE}], got {top_k}")

    files = get_parquet_files(split=split)
    if not files:
        raise RuntimeError(f"No parquet files found for split='{split}'.")
    n_sample_files = max(1, math.ceil(len(files) * sample_frac))
    rng = random.Random(seed)
    sampled_files = sorted(rng.sample(files, n_sample_files))

    counts = torch.zeros(VOCAB_SIZE, dtype=torch.int64)
    total_sequences = 0

    for input_ids in tqdm(_iter_input_ids(sampled_files), desc="Sweeping tokens"):
        token_tensor = torch.tensor(input_ids, dtype=torch.long)
        counts += torch.bincount(token_tensor, minlength=VOCAB_SIZE)
        total_sequences += 1

    hot_counts, hot_token_ids = torch.topk(counts, k=top_k)

    payload = {
        "hot_token_ids": hot_token_ids.cpu(),
        "hot_token_counts": hot_counts.cpu(),
        "counts": counts.cpu(),
        "top_k": top_k,
        "sample_frac": sample_frac,
        "seed": seed,
        "dataset_repo": DATASET_REPO,
        "subset": SUBSET,
        "split": split,
        "total_files_in_split": len(files),
        "sampled_files_count": n_sample_files,
        "sampled_files": sampled_files,
        "total_sequences_swept": total_sequences,
    }

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    torch.save(payload, cache_path)
    return payload


def main():
    parser = argparse.ArgumentParser(description="Build cached top-K hot tokens.")
    parser.add_argument("--split", type=str, default=TRAIN_SPLIT_1P3B)
    parser.add_argument("--top_k", type=int, default=2000)
    parser.add_argument("--sample_frac", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--cache_path", type=str, default=DEFAULT_CACHE_PATH)
    args = parser.parse_args()

    print(
        f"Building hot-token cache from dataset={DATASET_REPO}/{SUBSET}, "
        f"split={args.split}, sample_frac={args.sample_frac:.2%}, top_k={args.top_k}"
    )
    payload = build_hot_token_cache(
        split=args.split,
        top_k=args.top_k,
        sample_frac=args.sample_frac,
        seed=args.seed,
        cache_path=args.cache_path,
    )
    print(f"Saved cache to: {args.cache_path}")
    print(
        f"Sampled files: {payload['sampled_files_count']}/{payload['total_files_in_split']}, "
        f"sequences swept: {payload['total_sequences_swept']:,}"
    )


if __name__ == "__main__":
    main()
