#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Optional, Sequence

from tqdm import tqdm

from src.data import load_samples
from src.project_paths import (
    default_batch_output_dir,
    default_data_root,
    default_database_table,
    default_db_cache,
    default_query_table,
    resolve_checkpoint,
)
from src.runtime import (
    PoseEstimator,
    ensure_cache_matches_checkpoint,
    get_cache_checkpoint_path,
    load_database_cache,
)
from src.visualization import export_pose_visualizations_with_estimator


def parse_args():
    parser = argparse.ArgumentParser(description="Batch evaluate coarse and refined 3DoF estimation on local BEV queries.")
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
        "--query-table",
        default=default_query_table(),
        help="Path to query_samples.csv.",
    )
    parser.add_argument(
        "--data-root",
        default=default_data_root(),
        help="Data root used to resolve relative paths.",
    )
    parser.add_argument(
        "--db-cache",
        default=default_db_cache(),
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
        "--resolution-m",
        type=float,
        default=None,
        help="Optional override for BEV resolution. If omitted, use database sample resolution.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start query sample_index in query_samples.csv.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="How many query samples to evaluate from --start-index. Use <=0 to evaluate all remaining samples.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional root directory for generated outputs. Defaults to debug_outputs/batch_evaluate/range_xxxx_yyyy.",
    )
    parser.add_argument(
        "--output-csv",
        default="",
        help="Optional explicit per-query result CSV path. Defaults to <output-dir>/results.csv.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional explicit summary JSON path. Defaults to <output-dir>/summary.json.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Export per-query visualization panels under <output-dir>/visualizations/<query_key>/.",
    )
    parser.add_argument(
        "--visualize-root",
        default="",
        help="Optional explicit root directory for per-query visualizations.",
    )
    parser.add_argument(
        "--visualize-match-limit",
        type=int,
        default=15,
        help="Maximum number of local matches drawn per candidate panel.",
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _select_query_samples(all_query_samples: Sequence, start_index: int, limit: int):
    if start_index < 0 or start_index >= len(all_query_samples):
        raise IndexError(f"start-index out of range: {start_index}")

    selected = list(all_query_samples[start_index:])
    if limit > 0:
        selected = selected[:limit]
    if not selected:
        raise ValueError("No query samples selected.")
    return selected


def _resolve_output_dir(args, query_samples) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    if args.output_csv:
        return Path(args.output_csv).expanduser().resolve().parent
    if args.output_json:
        return Path(args.output_json).expanduser().resolve().parent
    return default_batch_output_dir(query_samples[0].sample_index, query_samples[-1].sample_index)


def _resolve_output_csv_path(args, output_dir: Path) -> Path:
    if args.output_csv:
        return Path(args.output_csv).expanduser().resolve()
    return output_dir / "results.csv"


def _resolve_output_json_path(args, output_dir: Path) -> Path:
    if args.output_json:
        return Path(args.output_json).expanduser().resolve()
    return output_dir / "summary.json"


def _resolve_visualize_root(args, output_dir: Path) -> Optional[Path]:
    if args.visualize_root:
        return Path(args.visualize_root).expanduser().resolve()
    if args.visualize:
        return output_dir / "visualizations"
    return None


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

    print(f"[INFO] Loading database samples: {args.database_table}")
    db_samples = load_samples(args.database_table, data_root=args.data_root)
    print(f"[INFO] Database samples: {len(db_samples)}")

    print(f"[INFO] Loading query samples: {args.query_table}")
    all_query_samples = load_samples(args.query_table, data_root=args.data_root)
    query_samples = _select_query_samples(all_query_samples, args.start_index, args.limit)
    print(
        "[INFO] Query samples selected: "
        f"{len(query_samples)} (sample_index {query_samples[0].sample_index} -> {query_samples[-1].sample_index})"
    )

    output_dir = _resolve_output_dir(args, query_samples)
    output_csv = _resolve_output_csv_path(args, output_dir)
    output_json = _resolve_output_json_path(args, output_dir)
    visualize_root = _resolve_visualize_root(args, output_dir)

    if visualize_root is not None:
        print(f"[INFO] Batch visualizations will be exported to: {visualize_root}")

    estimator = PoseEstimator(
        checkpoint_path=checkpoint_path,
        database_samples=db_samples,
        db_cache=db_cache,
        device_arg=args.device,
        show_progress=True,
    )

    rows: List[Dict[str, object]] = []
    coarse_xy_errors: List[float] = []
    coarse_yaw_errors: List[float] = []
    refined_xy_errors: List[float] = []
    refined_yaw_errors: List[float] = []
    refined_available = 0

    iterator = tqdm(query_samples, desc="Batch evaluating queries")
    for query_sample in iterator:
        result = estimator.estimate_pose_for_query(
            query_image_path=str(query_sample.bev_path),
            topk=args.topk,
            resolution_override_m=args.resolution_m,
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

        row = {
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
            "visualization_dir": "",
            "visualization_summary_image": "",
        }

        if visualize_root is not None:
            query_viz_dir = visualize_root / query_sample.sample_key
            viz_outputs = export_pose_visualizations_with_estimator(
                estimator=estimator,
                query_image_path=str(query_sample.bev_path),
                output_dir=str(query_viz_dir),
                topk=args.topk,
                resolution_override_m=args.resolution_m,
                max_matches=args.visualize_match_limit,
            )
            row["visualization_dir"] = viz_outputs["output_dir"]
            row["visualization_summary_image"] = viz_outputs["summary_image"]

        rows.append(row)

    summary = {
        "checkpoint": checkpoint_path,
        "database_table": str(Path(args.database_table).expanduser().resolve()),
        "query_table": str(Path(args.query_table).expanduser().resolve()),
        "db_cache": str(Path(args.db_cache).expanduser().resolve()) if args.db_cache else "",
        "device": args.device,
        "topk": args.topk,
        "resolution_m": args.resolution_m,
        "start_index": args.start_index,
        "limit": args.limit,
        "query_count": len(query_samples),
        "query_index_range": [query_samples[0].sample_index, query_samples[-1].sample_index],
        "refined_available_count": refined_available,
        "output_dir": str(output_dir),
        "output_csv": str(output_csv),
        "output_json": str(output_json),
        "visualize_root": "" if visualize_root is None else str(visualize_root),
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
