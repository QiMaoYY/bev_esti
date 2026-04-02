#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Optional

from tqdm import tqdm

from bev_esti.data import load_samples
from bev_esti.runtime import estimate_pose_for_query, load_database_cache


def parse_args():
    parser = argparse.ArgumentParser(description="Batch evaluate coarse and refined 3DoF estimation on local BEV queries.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the trained BEVPlace2 checkpoint.",
    )
    parser.add_argument(
        "--database-table",
        default="/home/qimao/grad_ws/data/bevplace_tables/database_samples.csv",
        help="Path to database_samples.csv.",
    )
    parser.add_argument(
        "--query-table",
        default="/home/qimao/grad_ws/data/bevplace_tables/query_samples.csv",
        help="Path to query_samples.csv.",
    )
    parser.add_argument(
        "--data-root",
        default="/home/qimao/grad_ws/data",
        help="Data root used to resolve relative paths.",
    )
    parser.add_argument(
        "--db-cache",
        default="/home/qimao/grad_ws/bev_esti/database_cache.npz",
        help="Database descriptor cache path.",
    )
    parser.add_argument("--topk", type=int, default=5, help="Number of coarse retrieval candidates.")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Device used for inference.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="How many query samples to evaluate from the start of query_samples.csv.",
    )
    parser.add_argument(
        "--output-csv",
        default="/home/qimao/grad_ws/bev_esti/debug_outputs/batch_eval_top20.csv",
        help="Per-query result CSV path.",
    )
    parser.add_argument(
        "--output-json",
        default="/home/qimao/grad_ws/bev_esti/debug_outputs/batch_eval_top20_summary.json",
        help="Summary JSON path.",
    )
    return parser.parse_args()


def yaw_error_deg(pred_rad: float, gt_rad: float) -> float:
    delta = pred_rad - gt_rad
    delta = math.atan2(math.sin(delta), math.cos(delta))
    return abs(math.degrees(delta))


def summarize(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"count": 0, "mean": None, "median": None}
    return {"count": len(values), "mean": mean(values), "median": median(values)}


def rate(values: List[float], threshold: float) -> float:
    if not values:
        return 0.0
    return sum(v < threshold for v in values) / float(len(values))


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()

    print(f"[INFO] Loading database samples: {args.database_table}")
    db_samples = load_samples(args.database_table, data_root=args.data_root)
    print(f"[INFO] Database samples: {len(db_samples)}")

    print(f"[INFO] Loading query samples: {args.query_table}")
    query_samples = load_samples(args.query_table, data_root=args.data_root)
    query_samples = query_samples[: args.limit]
    print(f"[INFO] Query samples selected: {len(query_samples)}")

    db_cache = load_database_cache(args.db_cache) if args.db_cache else None
    if args.db_cache:
        print(f"[INFO] Using database cache: {args.db_cache}")

    rows: List[Dict[str, object]] = []
    coarse_xy_errors: List[float] = []
    coarse_yaw_errors: List[float] = []
    refined_xy_errors: List[float] = []
    refined_yaw_errors: List[float] = []
    refined_available = 0

    iterator = tqdm(query_samples, desc="Batch evaluating queries")
    for query_sample in iterator:
        result = estimate_pose_for_query(
            checkpoint_path=args.checkpoint,
            database_samples=db_samples,
            query_image_path=str(query_sample.bev_path),
            db_cache=db_cache,
            device_arg=args.device,
            topk=args.topk,
            show_progress=False,
        )

        best = result["best"]
        coarse = result["candidates"][0]["coarse_pose"]
        refined = best["estimated_pose"]

        coarse_xy = math.hypot(coarse["x"] - query_sample.anchor_x, coarse["y"] - query_sample.anchor_y)
        coarse_yaw = yaw_error_deg(coarse["yaw_rad"], query_sample.anchor_yaw_rad)
        refined_xy = math.hypot(refined["x"] - query_sample.anchor_x, refined["y"] - query_sample.anchor_y)
        refined_yaw = yaw_error_deg(refined["yaw_rad"], query_sample.anchor_yaw_rad)

        coarse_xy_errors.append(coarse_xy)
        coarse_yaw_errors.append(coarse_yaw)
        refined_xy_errors.append(refined_xy)
        refined_yaw_errors.append(refined_yaw)

        if best.get("mode") == "refined":
            refined_available += 1

        rows.append(
            {
                "query_index": query_sample.sample_index,
                "query_key": query_sample.sample_key,
                "query_bev": str(query_sample.bev_path),
                "gt_x": query_sample.anchor_x,
                "gt_y": query_sample.anchor_y,
                "gt_yaw_deg": query_sample.anchor_yaw_deg,
                "coarse_db_key": result["candidates"][0]["db_sample_key"],
                "coarse_x": coarse["x"],
                "coarse_y": coarse["y"],
                "coarse_yaw_deg": coarse["yaw_deg"],
                "coarse_xy_error_m": coarse_xy,
                "coarse_yaw_error_deg": coarse_yaw,
                "best_rank": best["rank"],
                "best_mode": best.get("mode", "coarse_only"),
                "best_db_key": best["db_sample_key"],
                "best_x": refined["x"],
                "best_y": refined["y"],
                "best_yaw_deg": refined["yaw_deg"],
                "best_xy_error_m": refined_xy,
                "best_yaw_error_deg": refined_yaw,
                "feature_sq_l2": best["feature_sq_l2"],
                "inlier_count": best.get("relative_pose", {}).get("inlier_count", ""),
                "inlier_ratio": best.get("relative_pose", {}).get("inlier_ratio", ""),
            }
        )

    summary = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "database_table": str(Path(args.database_table).expanduser().resolve()),
        "query_table": str(Path(args.query_table).expanduser().resolve()),
        "db_cache": str(Path(args.db_cache).expanduser().resolve()) if args.db_cache else "",
        "device": args.device,
        "topk": args.topk,
        "query_count": len(query_samples),
        "refined_available_count": refined_available,
        "coarse": {
            "xy": summarize(coarse_xy_errors),
            "yaw_deg": summarize(coarse_yaw_errors),
            "xy_lt_2m_rate": rate(coarse_xy_errors, 2.0),
            "yaw_lt_5deg_rate": rate(coarse_yaw_errors, 5.0),
        },
        "refined": {
            "xy": summarize(refined_xy_errors),
            "yaw_deg": summarize(refined_yaw_errors),
            "xy_lt_2m_rate": rate(refined_xy_errors, 2.0),
            "yaw_lt_5deg_rate": rate(refined_yaw_errors, 5.0),
        },
    }

    output_csv = Path(args.output_csv).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    write_csv(output_csv, rows)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[INFO] Per-query CSV saved to: {output_csv}")
    print(f"[INFO] Summary JSON saved to: {output_json}")
    print(
        "[INFO] Coarse summary: "
        f"xy_mean={summary['coarse']['xy']['mean']:.3f} m, "
        f"yaw_mean={summary['coarse']['yaw_deg']['mean']:.3f} deg"
    )
    print(
        "[INFO] Refined summary: "
        f"xy_mean={summary['refined']['xy']['mean']:.3f} m, "
        f"yaw_mean={summary['refined']['yaw_deg']['mean']:.3f} deg"
    )


if __name__ == "__main__":
    raise SystemExit(main())
