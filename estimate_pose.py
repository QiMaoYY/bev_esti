#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.data import find_sample_by_index, find_sample_by_key, load_samples
from src.project_paths import (
    default_data_root,
    default_database_table,
    default_db_cache,
    default_estimate_output_dir,
    default_query_table,
    resolve_checkpoint,
)
from src.runtime import (
    PoseEstimator,
    ensure_cache_matches_checkpoint,
    get_cache_checkpoint_path,
    load_database_cache,
    result_to_jsonable,
)
from src.visualization import export_pose_visualizations_with_estimator


def parse_args():
    parser = argparse.ArgumentParser(description="Estimate 3DoF pose for a single BEV image on CPU.")
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Path to the trained BEVPlace2 checkpoint. If omitted, prefer the checkpoint recorded in db-cache; otherwise auto-detect the latest bevplace checkpoint under ../BEVPlace2/runs.",
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
        "--db-cache",
        default=default_db_cache(),
        help="Optional database descriptor cache built by build_db_cache.py.",
    )
    parser.add_argument(
        "--query-image",
        default="",
        help="Path to a single query BEV image. If empty, use --query-table with --query-index or --query-key.",
    )
    parser.add_argument(
        "--query-table",
        default=default_query_table(),
        help="Path to query_samples.csv for evaluation-style inference.",
    )
    parser.add_argument(
        "--query-index",
        type=int,
        default=0,
        help="Query sample_index in query_samples.csv. Ignored when --query-image or --query-key is provided.",
    )
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
        help="Optional explicit path to save the result JSON. Defaults to debug_outputs/estimate_pose/<query>/result.json.",
    )
    parser.add_argument(
        "--no-save-json",
        action="store_true",
        help="Do not save result JSON to disk.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Export Top-K transformation and local feature visualizations to the default query output directory.",
    )
    parser.add_argument(
        "--visualize-dir",
        default="",
        help="Optional explicit directory to export Top-K transformation and local feature visualizations.",
    )
    parser.add_argument(
        "--visualize-match-limit",
        type=int,
        default=15,
        help="Maximum number of local matches drawn per candidate panel.",
    )
    return parser.parse_args()


def _resolve_query(args) -> Tuple[str, Optional[Dict[str, Any]], str]:
    if args.query_image:
        query_path = Path(args.query_image).expanduser().resolve()
        return str(query_path), None, query_path.stem

    query_samples = load_samples(args.query_table, data_root=args.data_root)
    if args.query_key:
        sample = find_sample_by_key(query_samples, args.query_key)
    else:
        sample = find_sample_by_index(query_samples, args.query_index)

    return str(sample.bev_path), {
        "sample_key": sample.sample_key,
        "anchor_x": sample.anchor_x,
        "anchor_y": sample.anchor_y,
        "anchor_yaw_rad": sample.anchor_yaw_rad,
        "anchor_yaw_deg": sample.anchor_yaw_deg,
    }, sample.sample_key


def _resolve_output_json_path(args, query_label: str) -> Optional[Path]:
    if args.no_save_json:
        return None
    if args.output_json:
        return Path(args.output_json).expanduser().resolve()
    return default_estimate_output_dir(query_label) / "result.json"


def _resolve_visualize_dir(args, query_label: str) -> Optional[Path]:
    if args.visualize_dir:
        return Path(args.visualize_dir).expanduser().resolve()
    if args.visualize:
        return default_estimate_output_dir(query_label) / "viz"
    return None


def _yaw_error_deg(pred_rad: float, gt_rad: float) -> float:
    delta = pred_rad - gt_rad
    delta = math.atan2(math.sin(delta), math.cos(delta))
    return abs(math.degrees(delta))


def main():
    args = parse_args()
    db_cache = load_database_cache(args.db_cache) if args.db_cache else None
    if args.db_cache:
        print(f"[INFO] Using database cache: {args.db_cache}")
    cache_checkpoint_path = get_cache_checkpoint_path(db_cache)

    if args.checkpoint:
        checkpoint_path = resolve_checkpoint(args.checkpoint)
    elif cache_checkpoint_path:
        checkpoint_path = resolve_checkpoint(cache_checkpoint_path)
    elif db_cache is not None:
        raise ValueError(
            "The selected database cache does not record which checkpoint created it. "
            "To avoid corrupted Top-K retrieval, please provide --checkpoint explicitly once "
            "or rebuild the cache with the current build_db_cache.py."
        )
    else:
        checkpoint_path = resolve_checkpoint(args.checkpoint)
    print(f"[INFO] Using checkpoint: {checkpoint_path}")
    ensure_cache_matches_checkpoint(db_cache, checkpoint_path, cache_path=args.db_cache)

    print(f"[INFO] Loading database sample table: {args.database_table}")
    database_samples = load_samples(args.database_table, data_root=args.data_root)
    print(f"[INFO] Database sample count: {len(database_samples)}")
    query_image_path, query_gt, query_label = _resolve_query(args)
    print(f"[INFO] Resolved query image: {query_image_path}")

    output_json_path = _resolve_output_json_path(args, query_label)
    visualize_dir = _resolve_visualize_dir(args, query_label)

    estimator = PoseEstimator(
        checkpoint_path=checkpoint_path,
        database_samples=database_samples,
        db_cache=db_cache,
        device_arg=args.device,
        show_progress=True,
    )
    result = estimator.estimate_pose_for_query(
        query_image_path=query_image_path,
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
        print("[INFO] Best estimation error: " f"xy={xy_err:.3f} m, yaw={yaw_err:.3f} deg")

    if visualize_dir is not None:
        print(f"[INFO] Exporting visualizations to: {visualize_dir}")
        visualization_outputs = export_pose_visualizations_with_estimator(
            estimator=estimator,
            query_image_path=query_image_path,
            output_dir=str(visualize_dir),
            topk=args.topk,
            resolution_override_m=args.resolution_m,
            max_matches=args.visualize_match_limit,
        )
        result["visualization"] = visualization_outputs
        print(f"[INFO] Visualization summary saved to: {visualization_outputs['summary_image']}")

    output_text = json.dumps(result_to_jsonable(result), indent=2, ensure_ascii=False)

    if output_json_path is not None:
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(output_text + "\n", encoding="utf-8")
        print(f"[INFO] Result JSON saved to: {output_json_path}")


if __name__ == "__main__":
    raise SystemExit(main())
