from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .data import Sample, read_bev_grayscale
from .runtime import (
    PoseEstimator,
    estimate_relative_pose_from_match_debug,
    extract_features_batch,
    extract_match_debug,
    extract_query_features,
    l2_topk,
    matrix_to_pose_2d,
    pose_to_matrix_2d,
)


def _gray_to_bgr(gray_image: np.ndarray) -> np.ndarray:
    if gray_image.ndim == 2:
        return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    return gray_image.copy()


def _resize_with_padding(image: np.ndarray, size: Tuple[int, int], fill_value: int = 0) -> np.ndarray:
    target_w, target_h = size
    canvas = np.full((target_h, target_w, 3), fill_value, dtype=np.uint8)
    image = _gray_to_bgr(image)
    if image.size == 0:
        return canvas

    h, w = image.shape[:2]
    scale = min(target_w / max(1, w), target_h / max(1, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2
    canvas[offset_y : offset_y + new_h, offset_x : offset_x + new_w] = resized
    return canvas


def _add_title(image: np.ndarray, title: str) -> np.ndarray:
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 28), (0, 0, 0), thickness=-1)
    cv2.putText(canvas, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _stack_row(images: Sequence[np.ndarray], gap: int = 12) -> np.ndarray:
    if not images:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    gap_img = np.zeros((images[0].shape[0], gap, 3), dtype=np.uint8)
    row: List[np.ndarray] = []
    for idx, image in enumerate(images):
        if idx > 0:
            row.append(gap_img)
        row.append(image)
    return np.hstack(row)


def _stack_col(images: Sequence[np.ndarray], gap: int = 12) -> np.ndarray:
    if not images:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    gap_img = np.zeros((gap, images[0].shape[1], 3), dtype=np.uint8)
    col: List[np.ndarray] = []
    for idx, image in enumerate(images):
        if idx > 0:
            col.append(gap_img)
        col.append(image)
    return np.vstack(col)


def _fit_width(image: np.ndarray, width: int, max_height: int) -> np.ndarray:
    image = _gray_to_bgr(image)
    if image.size == 0:
        return np.zeros((max_height, width, 3), dtype=np.uint8)

    h, w = image.shape[:2]
    scale = min(width / max(1, w), max_height / max(1, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((max_height, width, 3), dtype=np.uint8)
    offset_x = (width - new_w) // 2
    offset_y = (max_height - new_h) // 2
    canvas[offset_y : offset_y + new_h, offset_x : offset_x + new_w] = resized
    return canvas


def _pad_to_width(image: np.ndarray, width: int) -> np.ndarray:
    if image.shape[1] >= width:
        return image
    canvas = np.zeros((image.shape[0], width, 3), dtype=np.uint8)
    canvas[:, : image.shape[1]] = image
    return canvas


def _text_block(lines: Sequence[str], width: int, height: int) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y = 28
    for line in lines:
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        y += 28
        if y > height - 10:
            break
    return canvas


def feature_pseudocolor(local_feat_hwc: np.ndarray) -> np.ndarray:
    channel_groups = np.array_split(np.arange(local_feat_hwc.shape[2]), 3)
    projections: List[np.ndarray] = []
    for channel_indices in channel_groups:
        group_map = np.mean(np.abs(local_feat_hwc[:, :, channel_indices]), axis=2)
        if float(np.max(group_map) - np.min(group_map)) < 1e-6:
            normalized = np.zeros_like(group_map, dtype=np.uint8)
        else:
            normalized = cv2.normalize(group_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        projections.append(normalized)
    return np.stack(projections, axis=2)


def draw_points(gray_image: np.ndarray, points_px: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    canvas = _gray_to_bgr(gray_image)
    for point in np.asarray(points_px, dtype=np.float32):
        x, y = int(round(point[0])), int(round(point[1]))
        cv2.circle(canvas, (x, y), 2, color, thickness=-1, lineType=cv2.LINE_AA)
    return canvas


def create_overlay(
    query_gray: np.ndarray,
    db_gray: np.ndarray,
    affine_query_to_db: Optional[np.ndarray] = None,
) -> np.ndarray:
    query_image = query_gray
    if affine_query_to_db is not None:
        query_image = cv2.warpAffine(
            query_gray,
            affine_query_to_db,
            (db_gray.shape[1], db_gray.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    overlay = np.zeros((db_gray.shape[0], db_gray.shape[1], 3), dtype=np.uint8)
    overlay[:, :, 1] = db_gray
    overlay[:, :, 2] = query_image
    return overlay


def estimate_query_to_db_affine(
    matched_query_points_px: np.ndarray,
    matched_db_points_px: np.ndarray,
    inlier_mask: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    query_points = np.asarray(matched_query_points_px, dtype=np.float32)
    db_points = np.asarray(matched_db_points_px, dtype=np.float32)
    if inlier_mask is not None:
        mask = np.asarray(inlier_mask).astype(bool).reshape(-1)
        query_points = query_points[mask]
        db_points = db_points[mask]
    if len(query_points) < 2 or len(db_points) < 2:
        return None

    affine, _ = cv2.estimateAffinePartial2D(
        query_points,
        db_points,
        method=cv2.LMEDS,
        refineIters=10,
    )
    if affine is None:
        return None
    return np.asarray(affine, dtype=np.float32)


def draw_match_image(
    query_gray: np.ndarray,
    db_gray: np.ndarray,
    matched_query_points_px: np.ndarray,
    matched_db_points_px: np.ndarray,
    inlier_mask: Optional[np.ndarray],
    max_matches: int = 80,
) -> np.ndarray:
    query_bgr = _gray_to_bgr(query_gray)
    db_bgr = _gray_to_bgr(db_gray)
    canvas = np.hstack((query_bgr, db_bgr))
    offset_x = query_bgr.shape[1]

    query_points = np.asarray(matched_query_points_px, dtype=np.float32)
    db_points = np.asarray(matched_db_points_px, dtype=np.float32)
    if len(query_points) == 0 or len(db_points) == 0:
        return _add_title(canvas, "Local Matches (no valid correspondences)")

    mask = np.zeros((len(query_points),), dtype=bool)
    if inlier_mask is not None and len(inlier_mask) == len(query_points):
        mask = np.asarray(inlier_mask).astype(bool).reshape(-1)

    inlier_indices = np.flatnonzero(mask)
    outlier_indices = np.flatnonzero(~mask)
    ordered_indices = np.concatenate((inlier_indices, outlier_indices))
    if len(ordered_indices) > max_matches:
        sample_positions = np.linspace(0, len(ordered_indices) - 1, num=max_matches, dtype=np.int32)
        ordered_indices = ordered_indices[sample_positions]

    for idx in ordered_indices.tolist():
        qx, qy = query_points[idx]
        dx, dy = db_points[idx]
        color = (0, 255, 0) if mask[idx] else (0, 0, 255)
        q_pt = (int(round(qx)), int(round(qy)))
        d_pt = (int(round(dx + offset_x)), int(round(dy)))
        cv2.circle(canvas, q_pt, 2, color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, d_pt, 2, color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.line(canvas, q_pt, d_pt, color, thickness=1, lineType=cv2.LINE_AA)

    title = f"Local Matches ({len(ordered_indices)} / {len(query_points)}, green=inlier)"
    return _add_title(canvas, title)


def build_candidate_panel(
    query_gray: np.ndarray,
    db_gray: np.ndarray,
    query_local_feat: np.ndarray,
    db_local_feat: np.ndarray,
    match_debug: Dict[str, object],
    pose_result: Optional[Dict[str, object]],
    metadata: Dict[str, object],
    tile_size: Tuple[int, int] = (280, 280),
    max_matches: int = 80,
) -> np.ndarray:
    query_keypoints_img = draw_points(query_gray, np.asarray(match_debug["query_keypoints_px"]), (0, 255, 0))
    db_keypoints_img = draw_points(db_gray, np.asarray(match_debug["db_keypoints_px"]), (0, 255, 0))
    query_feat_img = feature_pseudocolor(query_local_feat)
    db_feat_img = feature_pseudocolor(db_local_feat)

    affine = None
    if pose_result is not None:
        affine = estimate_query_to_db_affine(
            np.asarray(match_debug["matched_query_points_px"]),
            np.asarray(match_debug["matched_db_points_px"]),
            np.asarray(pose_result.get("inlier_mask", []), dtype=bool),
        )

    row1 = _stack_row(
        [
            _add_title(_resize_with_padding(query_gray, tile_size), "Input Query BEV"),
            _add_title(_resize_with_padding(db_gray, tile_size), f"Top-{metadata['rank']} Candidate DB"),
            _add_title(_resize_with_padding(create_overlay(query_gray, db_gray), tile_size), "Overlay Before"),
            _add_title(_resize_with_padding(create_overlay(query_gray, db_gray, affine_query_to_db=affine), tile_size), "Overlay After"),
        ]
    )

    row2 = _stack_row(
        [
            _add_title(_resize_with_padding(query_keypoints_img, tile_size), "Query FAST Keypoints"),
            _add_title(_resize_with_padding(db_keypoints_img, tile_size), "DB FAST Keypoints"),
            _add_title(_resize_with_padding(query_feat_img, tile_size), "Query Local Feature RGB"),
            _add_title(_resize_with_padding(db_feat_img, tile_size), "DB Local Feature RGB"),
        ]
    )

    match_img = draw_match_image(
        query_gray=query_gray,
        db_gray=db_gray,
        matched_query_points_px=np.asarray(match_debug["matched_query_points_px"]),
        matched_db_points_px=np.asarray(match_debug["matched_db_points_px"]),
        inlier_mask=None if pose_result is None else np.asarray(pose_result.get("inlier_mask", []), dtype=bool),
        max_matches=max_matches,
    )
    row_width = row1.shape[1]
    row3 = _fit_width(match_img, width=row_width, max_height=420)

    lines = [
        (
            f"Rank {metadata['rank']} | {metadata['db_sample_key']} | "
            f"source={metadata['pose_source']} | sq_l2={metadata['feature_sq_l2']:.4f}"
        ),
        (
            "retrieval anchor: "
            f"x={metadata['retrieval_anchor_pose']['x']:.3f}, "
            f"y={metadata['retrieval_anchor_pose']['y']:.3f}, "
            f"yaw={metadata['retrieval_anchor_pose']['yaw_deg']:.2f} deg"
        ),
    ]
    if pose_result is not None:
        lines.append(
            "BEVPlace++ 3DoF delta: "
            f"dx={pose_result['relative_tx_m']:.3f} m, dy={pose_result['relative_ty_m']:.3f} m, "
            f"dyaw={pose_result['relative_yaw_deg']:.2f} deg"
        )
        lines.append(
            f"matches={pose_result['match_count']}, inliers={pose_result['inlier_count']}, "
            f"ratio={pose_result['inlier_ratio']:.3f}"
        )
    else:
        lines.append("BEVPlace++ 3DoF: unavailable, fallback uses retrieval anchor pose")

    header = _text_block(lines, width=row_width, height=120)
    return _stack_col([header, row1, row2, row3], gap=12)


def build_summary_canvas(
    query_gray: np.ndarray,
    query_local_feat: np.ndarray,
    query_keypoints_px: np.ndarray,
    candidate_cards: Sequence[Dict[str, object]],
    tile_size: Tuple[int, int] = (220, 220),
) -> np.ndarray:
    query_row = _stack_row(
        [
            _add_title(_resize_with_padding(query_gray, tile_size), "Input Query BEV"),
            _add_title(_resize_with_padding(draw_points(query_gray, query_keypoints_px, (0, 255, 0)), tile_size), "Query FAST Keypoints"),
            _add_title(_resize_with_padding(feature_pseudocolor(query_local_feat), tile_size), "Query Local Feature RGB"),
        ]
    )

    rows = [query_row]
    for card in candidate_cards:
        text_lines = [
            f"rank={card['rank']} | {card['db_sample_key']}",
            f"source={card['pose_source']} | l2={card['feature_sq_l2']:.4f}",
            f"inliers={card['inlier_count']} | ratio={card['inlier_ratio']:.3f}",
            f"dyaw={card['relative_yaw_deg']:.2f} deg",
            f"dx={card['relative_tx_m']:.3f} m | dy={card['relative_ty_m']:.3f} m",
        ]
        rows.append(
            _stack_row(
                [
                    _add_title(_resize_with_padding(card["db_gray"], tile_size), f"Top-{card['rank']} Candidate"),
                    _add_title(_resize_with_padding(card["overlay_after"], tile_size), "Overlay After"),
                    _text_block(text_lines, width=340, height=tile_size[1]),
                ]
            )
        )

    max_width = max(row.shape[1] for row in rows)
    rows = [_pad_to_width(row, max_width) for row in rows]
    return _stack_col(rows, gap=12)


def export_pose_visualizations_with_estimator(
    estimator: PoseEstimator,
    query_image_path: str,
    output_dir: str,
    topk: int = 5,
    resolution_override_m: Optional[float] = None,
    max_matches: int = 80,
) -> Dict[str, str]:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    query_image_path = Path(query_image_path).expanduser().resolve()
    query_gray = read_bev_grayscale(query_image_path)
    query_local_feat, query_global_desc = extract_query_features(estimator.model, query_gray, device=estimator.device)
    top_indices, top_sq_dists = l2_topk(query_global_desc, estimator.database_descs, topk=topk)

    retrieved_candidates: List[Tuple[int, int, float, Sample, np.ndarray]] = []
    for rank, (db_index, feature_sq_l2) in enumerate(zip(top_indices.tolist(), top_sq_dists.tolist()), start=1):
        db_sample = estimator.database_samples[db_index]
        db_gray = read_bev_grayscale(db_sample.bev_path)
        retrieved_candidates.append((rank, db_index, float(feature_sq_l2), db_sample, db_gray))

    candidate_batch_size = max(1, min(len(retrieved_candidates), estimator.batch_size_for_db, 5))
    candidate_cards: List[Dict[str, object]] = []
    query_keypoints_px: Optional[np.ndarray] = None

    for batch_start in range(0, len(retrieved_candidates), candidate_batch_size):
        batch_entries = retrieved_candidates[batch_start : batch_start + candidate_batch_size]
        batch_grays = [entry[4] for entry in batch_entries]
        batch_local_feats, _ = extract_features_batch(estimator.model, batch_grays, device=estimator.device)

        for (rank, db_index, feature_sq_l2, db_sample, db_gray), db_local_feat in zip(batch_entries, batch_local_feats):
            match_debug = extract_match_debug(
                query_gray=query_gray,
                db_gray=db_gray,
                query_local_feat=query_local_feat,
                db_local_feat=db_local_feat,
            )
            if query_keypoints_px is None:
                query_keypoints_px = np.asarray(match_debug["query_keypoints_px"], dtype=np.float32)

            resolution_m = resolution_override_m or db_sample.bev_resolution_m
            pose_result = estimate_relative_pose_from_match_debug(
                match_debug=match_debug,
                db_image_shape=db_gray.shape[:2],
                resolution_m=resolution_m,
            )

            retrieval_anchor_pose = {
                "x": db_sample.anchor_x,
                "y": db_sample.anchor_y,
                "yaw_rad": db_sample.anchor_yaw_rad,
                "yaw_deg": db_sample.anchor_yaw_deg,
            }

            pose_source = "retrieval_anchor_only"
            bevplace_3dof_pose = retrieval_anchor_pose
            inlier_count = 0
            inlier_ratio = 0.0
            relative_yaw_deg = 0.0
            relative_tx_m = 0.0
            relative_ty_m = 0.0
            inlier_mask = None

            if pose_result is not None:
                db_pose = pose_to_matrix_2d(db_sample.anchor_x, db_sample.anchor_y, db_sample.anchor_yaw_rad)
                relative_h = np.asarray(pose_result["relative_matrix_3x3"], dtype=np.float64)
                bevplace_3dof_pose = matrix_to_pose_2d(db_pose @ relative_h)
                pose_source = "bevplace_3dof"
                inlier_count = int(pose_result["inlier_count"])
                inlier_ratio = float(pose_result["inlier_ratio"])
                relative_yaw_deg = float(pose_result["relative_yaw_deg"])
                relative_tx_m = float(pose_result["relative_tx_m"])
                relative_ty_m = float(pose_result["relative_ty_m"])
                inlier_mask = np.asarray(pose_result.get("inlier_mask", []), dtype=bool)

            affine = estimate_query_to_db_affine(
                np.asarray(match_debug["matched_query_points_px"]),
                np.asarray(match_debug["matched_db_points_px"]),
                inlier_mask,
            )
            overlay_after = create_overlay(query_gray, db_gray, affine_query_to_db=affine)

            metadata = {
                "rank": rank,
                "db_sample_key": db_sample.sample_key,
                "feature_sq_l2": float(feature_sq_l2),
                "retrieval_anchor_pose": retrieval_anchor_pose,
                "pose_source": pose_source,
                "bevplace_3dof_pose": bevplace_3dof_pose,
            }

            panel = build_candidate_panel(
                query_gray=query_gray,
                db_gray=db_gray,
                query_local_feat=query_local_feat,
                db_local_feat=db_local_feat,
                match_debug=match_debug,
                pose_result=pose_result,
                metadata=metadata,
                max_matches=max_matches,
            )

            panel_name = f"rank_{rank:02d}_{db_sample.sample_key}_panel.png"
            cv2.imwrite(str(output_path / panel_name), panel)

            candidate_cards.append(
                {
                    "rank": rank,
                    "db_sample_key": db_sample.sample_key,
                    "feature_sq_l2": float(feature_sq_l2),
                    "pose_source": pose_source,
                    "inlier_count": inlier_count,
                    "inlier_ratio": inlier_ratio,
                    "relative_yaw_deg": relative_yaw_deg,
                    "relative_tx_m": relative_tx_m,
                    "relative_ty_m": relative_ty_m,
                    "db_gray": db_gray,
                    "overlay_after": overlay_after,
                    "panel_path": panel_name,
                }
            )

    if query_keypoints_px is None:
        query_keypoints_px = np.empty((0, 2), dtype=np.float32)

    summary = build_summary_canvas(
        query_gray=query_gray,
        query_local_feat=query_local_feat,
        query_keypoints_px=query_keypoints_px,
        candidate_cards=candidate_cards,
    )
    summary_name = "topk_summary.png"
    cv2.imwrite(str(output_path / summary_name), summary)

    return {
        "output_dir": str(output_path),
        "summary_image": str(output_path / summary_name),
        "query_image": str(query_image_path),
    }


def export_pose_visualizations(
    checkpoint_path: str,
    database_samples: Sequence[Sample],
    query_image_path: str,
    output_dir: str,
    db_cache: Optional[Dict[str, np.ndarray]] = None,
    device_arg: str = "cpu",
    topk: int = 5,
    batch_size_for_db: int = 32,
    num_workers: int = 0,
    resolution_override_m: Optional[float] = None,
    max_matches: int = 80,
) -> Dict[str, str]:
    estimator = PoseEstimator(
        checkpoint_path=checkpoint_path,
        database_samples=database_samples,
        db_cache=db_cache,
        device_arg=device_arg,
        batch_size_for_db=batch_size_for_db,
        num_workers=num_workers,
        show_progress=False,
    )
    return export_pose_visualizations_with_estimator(
        estimator=estimator,
        query_image_path=query_image_path,
        output_dir=output_dir,
        topk=topk,
        resolution_override_m=resolution_override_m,
        max_matches=max_matches,
    )
