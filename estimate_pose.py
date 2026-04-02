#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from bev_esti.data import find_sample_by_index, find_sample_by_key, load_samples
from bev_esti.runtime import estimate_pose_for_query, load_database_cache, result_to_jsonable
from bev_esti.visualization import export_pose_visualizations


def parse_args():
    parser = argparse.ArgumentParser(description="Estimate 3DoF pose for a single BEV image on CPU.")
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
        "--db-cache",
        default="",
        help="Optional database descriptor cache built by build_db_cache.py.",
    )
    parser.add_argument(
        "--query-image",
        default="",
        help="Path to a single query BEV image. If empty, use --query-table with --query-index or --query-key.",
    )
    parser.add_argument(
        "--query-table",
        default="/home/qimao/grad_ws/data/bevplace_tables/query_samples.csv",
        help="Path to query_samples.csv for evaluation-style inference.",
    )
    parser.add_argument("--query-index", type=int, default=-1, help="Query sample_index in query_samples.csv.")
    parser.add_argument("--query-key", default="", help="Query sample_key in query_samples.csv.")
    parser.add_argument("--topk", type=int, default=5, help="Number of coarse retrieval candidates.")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Device used for inference.",
    )
    parser.add_argument(
        "--resolution-m",
        type=float,
        default=None,
        help="Optional override for BEV resolution. If omitted, use database sample resolution.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to save the result JSON.",
    )
    parser.add_argument(
        "--visualize-dir",
        default="",
        help="Optional directory to export Top-K transformation and local feature visualizations.",
    )
    parser.add_argument(
        "--visualize-match-limit",
        type=int,
        default=80,
        help="Maximum number of local matches drawn per candidate panel.",
    )
    return parser.parse_args()


def _resolve_query(args) -> Tuple[str, Optional[Dict[str, Any]]]:
    if args.query_image:
        return str(Path(args.query_image).expanduser().resolve()), None

    query_samples = load_samples(args.query_table, data_root=args.data_root)
    if args.query_key:
        sample = find_sample_by_key(query_samples, args.query_key)
    elif args.query_index >= 0:
        sample = find_sample_by_index(query_samples, args.query_index)
    else:
        raise ValueError("Either --query-image or (--query-table with --query-index/--query-key) must be provided.")

    return str(sample.bev_path), {
        "sample_key": sample.sample_key,
        "anchor_x": sample.anchor_x,
        "anchor_y": sample.anchor_y,
        "anchor_yaw_rad": sample.anchor_yaw_rad,
        "anchor_yaw_deg": sample.anchor_yaw_deg,
    }


def _yaw_error_deg(pred_rad: float, gt_rad: float) -> float:
    delta = pred_rad - gt_rad
    delta = math.atan2(math.sin(delta), math.cos(delta))
    return abs(math.degrees(delta))


def main():
    args = parse_args()

    print(f"[INFO] Loading database sample table: {args.database_table}")
    database_samples = load_samples(args.database_table, data_root=args.data_root)
    print(f"[INFO] Database sample count: {len(database_samples)}")
    query_image_path, query_gt = _resolve_query(args)
    print(f"[INFO] Resolved query image: {query_image_path}")

    db_cache = load_database_cache(args.db_cache) if args.db_cache else None
    if args.db_cache:
        print(f"[INFO] Using database cache: {args.db_cache}")
    result = estimate_pose_for_query(
        checkpoint_path=args.checkpoint,
        database_samples=database_samples,
        query_image_path=query_image_path,
        db_cache=db_cache,
        device_arg=args.device,
        topk=args.topk,
        resolution_override_m=args.resolution_m,
        show_progress=True,
    )

    if query_gt is not None:
        est = result["best"]["estimated_pose"]
        xy_err = math.hypot(est["x"] - query_gt["anchor_x"], est["y"] - query_gt["anchor_y"])
        yaw_err = _yaw_error_deg(est["yaw_rad"], query_gt["anchor_yaw_rad"])
        result["query_ground_truth"] = query_gt
        result["best"]["estimation_error"] = {
            "xy_error_m": xy_err,
            "yaw_error_deg": yaw_err,
        }
        print(
            "[INFO] Best estimation error: "
            f"xy={xy_err:.3f} m, yaw={yaw_err:.3f} deg"
        )

    if args.visualize_dir:
        print(f"[INFO] Exporting visualizations to: {args.visualize_dir}")
        visualization_outputs = export_pose_visualizations(
            checkpoint_path=args.checkpoint,
            database_samples=database_samples,
            query_image_path=query_image_path,
            output_dir=args.visualize_dir,
            db_cache=db_cache,
            device_arg=args.device,
            topk=args.topk,
            resolution_override_m=args.resolution_m,
            max_matches=args.visualize_match_limit,
        )
        result["visualization"] = visualization_outputs
        print(f"[INFO] Visualization summary saved to: {visualization_outputs['summary_image']}")

    output_text = json.dumps(result_to_jsonable(result), indent=2, ensure_ascii=False)
    #print(output_text)

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text + "\n", encoding="utf-8")
        print(f"[INFO] Result JSON saved to: {output_path}")


if __name__ == "__main__":
    raise SystemExit(main())
