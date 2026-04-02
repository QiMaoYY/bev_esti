#!/usr/bin/env python3
import argparse
from pathlib import Path

from bev_esti.data import load_samples
from bev_esti.runtime import build_database_cache


def parse_args():
    parser = argparse.ArgumentParser(description="Build a CPU-friendly database descriptor cache for bev_esti.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the trained BEVPlace2 checkpoint, e.g. model_best.pth.tar.",
    )
    parser.add_argument(
        "--database-table",
        default="/home/qimao/grad_ws/data/bevplace_tables/database_samples.csv",
        help="Path to database_samples.csv.",
    )
    parser.add_argument(
        "--data-root",
        default="/home/qimao/grad_ws/data",
        help="Data root used to resolve bev_rel_path.",
    )
    parser.add_argument(
        "--output-cache",
        default="/home/qimao/grad_ws/bev_esti/database_cache.npz",
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
    print(f"[INFO] Loading database sample table: {args.database_table}")
    samples = load_samples(args.database_table, data_root=args.data_root)
    print(f"[INFO] Database sample count: {len(samples)}")
    print(f"[INFO] Output cache path: {args.output_cache}")
    cache_path = build_database_cache(
        checkpoint_path=args.checkpoint,
        samples=samples,
        cache_path=args.output_cache,
        device_arg=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"[INFO] Cached database descriptors: {cache_path}")


if __name__ == "__main__":
    raise SystemExit(main())
