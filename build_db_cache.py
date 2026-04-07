#!/usr/bin/env python3
import argparse

from src.data import load_samples
from src.project_paths import (
    default_data_root,
    default_database_table,
    default_db_cache,
    resolve_checkpoint,
)
from src.runtime import build_database_cache


def parse_args():
    parser = argparse.ArgumentParser(description="Build a CPU-friendly database descriptor cache.")
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Path to the trained BEVPlace2 checkpoint. If omitted, auto-detect the latest bevplace checkpoint under ../BEVPlace2/runs.",
    )
    parser.add_argument(
        "--database-table",
        default=default_database_table(),
        help="Path to database_samples.csv.",
    )
    parser.add_argument(
        "--data-root",
        default=default_data_root(),
        help="Data root used to resolve bev_rel_path.",
    )
    parser.add_argument(
        "--output-cache",
        default=default_db_cache(),
        help="Output npz path for cached global descriptors.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Device used for descriptor extraction.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for database descriptor extraction.")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of dataloader workers.")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    print(f"[INFO] Using checkpoint: {checkpoint_path}")
    print(f"[INFO] Loading database sample table: {args.database_table}")
    samples = load_samples(args.database_table, data_root=args.data_root)
    print(f"[INFO] Database sample count: {len(samples)}")
    print(f"[INFO] Output cache path: {args.output_cache}")
    cache_path = build_database_cache(
        checkpoint_path=checkpoint_path,
        samples=samples,
        cache_path=args.output_cache,
        device_arg=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"[INFO] Cached database descriptors: {cache_path}")


if __name__ == "__main__":
    raise SystemExit(main())
