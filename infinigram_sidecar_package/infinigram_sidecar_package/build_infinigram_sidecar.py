from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from infinigram_sidecar import CompactNgramSidecarBuilder, SidecarBuildConfig


def _parse_orders(values: list[str]) -> tuple[int, ...]:
    pieces: list[str] = []
    for value in values:
        pieces.extend(part.strip() for part in str(value).split(",") if part.strip())
    orders = tuple(sorted({int(x) for x in pieces}))
    if not orders:
        raise argparse.ArgumentTypeError("orders must contain at least one integer")
    return orders


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an eval-time InfiniGram-style sidecar index from the training data. "
            "The base model checkpoint is not touched."
        )
    )
    parser.add_argument("--out_dir", "--output_dir", dest="out_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--streaming", action="store_true", default=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--no_pin_memory", action="store_true")
    parser.add_argument("--no_persistent_workers", action="store_true")
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=50)

    parser.add_argument(
        "--orders",
        nargs="+",
        default=["4", "6", "8"],
        help="Context orders to store, e.g. --orders 4 6 8 or --orders 4,6,8",
    )
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--min_count", type=int, default=2)
    parser.add_argument("--min_store_confidence", type=float, default=0.0)
    parser.add_argument("--load_factor", type=float, default=0.70)
    parser.add_argument("--chunk_records", type=int, default=5_000_000)
    parser.add_argument("--pad_token_id", type=int, default=50256)
    parser.add_argument("--keep_temporary_chunks", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    from data import get_dataloader  # local import so the library stays drop-in

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = SidecarBuildConfig(
        orders=_parse_orders(args.orders),
        topk=args.topk,
        min_count=args.min_count,
        min_store_confidence=args.min_store_confidence,
        load_factor=args.load_factor,
        chunk_records=args.chunk_records,
        keep_temporary_chunks=args.keep_temporary_chunks,
        pad_token_id=args.pad_token_id,
    )

    loader = get_dataloader(
        split=args.split,
        batch_size=args.batch_size,
        streaming=args.streaming,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        persistent_workers=not args.no_persistent_workers,
        rank=0,
        world_size=1,
    )

    builder = CompactNgramSidecarBuilder(output_dir=out_dir, config=config)

    start = time.time()
    n_batches = 0
    for batch in loader:
        builder.add_batch(batch)
        n_batches += 1
        if args.log_every > 0 and n_batches % args.log_every == 0:
            elapsed = time.time() - start
            stats = builder.stats
            print(
                f"[build] batches={n_batches} sequences={stats['sequences']} "
                f"raw_records={stats['raw_records']} reduced_chunk_records={stats['reduced_chunk_records']} "
                f"elapsed_s={elapsed:.1f}"
            )
        if args.max_batches is not None and n_batches >= args.max_batches:
            break

    summary = builder.finalize()
    elapsed = time.time() - start

    print("\n[build] done")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[build] wrote index to {out_dir}")
    print(f"[build] elapsed_s={elapsed:.1f}")


if __name__ == "__main__":
    main()
