import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import BEVImageDataset, Sample, bev_to_model_tensor, read_bev_grayscale
from .model import REIN
from .ransac import rigid_ransac


def choose_device(device_arg: str = "cpu") -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> REIN:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    model = REIN().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def extract_global_descriptors(
    model: REIN,
    samples: Sequence[Sample],
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 0,
    show_progress: bool = True,
) -> np.ndarray:
    dataset = BEVImageDataset(samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    outputs: List[np.ndarray] = []

    iterator = tqdm(loader, desc="Extracting descriptors") if show_progress else loader
    with torch.no_grad():
        for images, _ in iterator:
            images = images.to(device)
            _, _, global_desc = model(images)
            outputs.append(global_desc.detach().cpu().numpy())

    return np.concatenate(outputs, axis=0).astype(np.float32)


def save_database_cache(output_path: str, samples: Sequence[Sample], descriptors: np.ndarray) -> None:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        descriptors=np.asarray(descriptors, dtype=np.float32),
        sample_index=np.asarray([sample.sample_index for sample in samples], dtype=np.int32),
        sample_key=np.asarray([sample.sample_key for sample in samples]),
        bev_path=np.asarray([str(sample.bev_path) for sample in samples]),
        anchor_x=np.asarray([sample.anchor_x for sample in samples], dtype=np.float64),
        anchor_y=np.asarray([sample.anchor_y for sample in samples], dtype=np.float64),
        anchor_yaw_rad=np.asarray([sample.anchor_yaw_rad for sample in samples], dtype=np.float64),
        bev_resolution_m=np.asarray([sample.bev_resolution_m for sample in samples], dtype=np.float64),
    )


def load_database_cache(cache_path: str) -> Dict[str, np.ndarray]:
    cache_path = Path(cache_path).expanduser().resolve()
    with np.load(cache_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def build_database_cache(
    checkpoint_path: str,
    samples: Sequence[Sample],
    cache_path: str,
    device_arg: str = "cpu",
    batch_size: int = 32,
    num_workers: int = 0,
) -> str:
    device = choose_device(device_arg)
    model = load_model_from_checkpoint(checkpoint_path, device=device)
    descriptors = extract_global_descriptors(
        model,
        samples,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        show_progress=True,
    )
    save_database_cache(cache_path, samples, descriptors)
    return str(Path(cache_path).expanduser().resolve())


def extract_query_features(model: REIN, gray_image: np.ndarray, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        tensor = bev_to_model_tensor(gray_image).unsqueeze(0).to(device)
        _, local_feat, global_desc = model(tensor)

    local_feat = local_feat[0].detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
    global_desc = global_desc[0].detach().cpu().numpy().astype(np.float32)
    return local_feat, global_desc


def l2_topk(query_desc: np.ndarray, database_descs: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    diff = database_descs - query_desc.reshape(1, -1)
    sq_dist = np.sum(diff * diff, axis=1)
    top_indices = np.argsort(sq_dist)[:topk]
    return top_indices.astype(np.int64), sq_dist[top_indices].astype(np.float32)


def sample_local_descriptors(local_feat_hwc: np.ndarray, keypoints: Sequence[cv2.KeyPoint]):
    height, width, channels = local_feat_hwc.shape
    valid_kps: List[cv2.KeyPoint] = []
    descriptors: List[np.ndarray] = []

    for keypoint in keypoints:
        x = int(keypoint.pt[0])
        y = int(keypoint.pt[1])
        if 0 <= x < width and 0 <= y < height:
            valid_kps.append(keypoint)
            descriptors.append(local_feat_hwc[y, x].astype(np.float32))

    if not descriptors:
        return [], np.empty((0, channels), dtype=np.float32)
    return valid_kps, np.stack(descriptors, axis=0)


def estimate_relative_pose(
    query_gray: np.ndarray,
    db_gray: np.ndarray,
    query_local_feat: np.ndarray,
    db_local_feat: np.ndarray,
    resolution_m: float,
    ransac_iters: int = 1000,
    inlier_threshold_m: float = 0.5,
) -> Optional[Dict[str, object]]:
    fast = cv2.FastFeatureDetector_create()
    query_kps = fast.detect(query_gray, None)
    db_kps = fast.detect(db_gray, None)

    query_kps, query_desc = sample_local_descriptors(query_local_feat, query_kps)
    db_kps, db_desc = sample_local_descriptors(db_local_feat, db_kps)

    if len(query_kps) < 2 or len(db_kps) < 2:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.knnMatch(query_desc, db_desc, k=2)
    primary_matches = [pair[0] for pair in matches if len(pair) >= 1]

    if len(primary_matches) < 2:
        return None

    query_points = np.float32([query_kps[m.queryIdx].pt for m in primary_matches])
    db_points = np.float32([db_kps[m.trainIdx].pt for m in primary_matches])

    center = np.array([[db_gray.shape[1] // 2, db_gray.shape[0] // 2]], dtype=np.float32)
    query_metric = (center - query_points) * float(resolution_m)
    db_metric = (center - db_points) * float(resolution_m)

    try:
        relative_mat, inlier_mask, consensus = rigid_ransac(
            query_metric,
            db_metric,
            iterations=ransac_iters,
            inlier_threshold=inlier_threshold_m,
        )
    except Exception:
        return None

    relative_h = np.vstack((relative_mat, np.array([[0.0, 0.0, 1.0]], dtype=np.float64)))
    yaw_rad = math.atan2(relative_h[0, 1], relative_h[0, 0])

    inlier_mask = np.asarray(inlier_mask).astype(bool).reshape(-1)
    return {
        "relative_matrix_2x3": relative_mat.tolist(),
        "relative_matrix_3x3": relative_h.tolist(),
        "relative_yaw_rad": yaw_rad,
        "relative_yaw_deg": math.degrees(yaw_rad),
        "relative_tx_m": float(relative_h[0, 2]),
        "relative_ty_m": float(relative_h[1, 2]),
        "query_keypoints": len(query_kps),
        "db_keypoints": len(db_kps),
        "match_count": len(primary_matches),
        "inlier_count": int(consensus),
        "inlier_ratio": float(consensus / max(1, len(primary_matches))),
        "inlier_mask": inlier_mask.tolist(),
    }


def pose_to_matrix_2d(x: float, y: float, yaw_rad: float) -> np.ndarray:
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return np.array(
        [
            [c, s, x],
            [-s, c, y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def matrix_to_pose_2d(matrix: np.ndarray) -> Dict[str, float]:
    yaw_rad = math.atan2(matrix[0, 1], matrix[0, 0])
    return {
        "x": float(matrix[0, 2]),
        "y": float(matrix[1, 2]),
        "yaw_rad": yaw_rad,
        "yaw_deg": math.degrees(yaw_rad),
    }


def estimate_pose_for_query(
    checkpoint_path: str,
    database_samples: Sequence[Sample],
    query_image_path: str,
    db_cache: Optional[Dict[str, np.ndarray]] = None,
    device_arg: str = "cpu",
    topk: int = 5,
    batch_size_for_db: int = 32,
    num_workers: int = 0,
    resolution_override_m: Optional[float] = None,
    show_progress: bool = True,
) -> Dict[str, object]:
    device = choose_device(device_arg)
    if show_progress:
        print(f"[INFO] Using device: {device}")
        print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    model = load_model_from_checkpoint(checkpoint_path, device=device)

    if db_cache is None:
        if show_progress:
            print("[INFO] No database cache provided. Extracting database descriptors on the fly.")
        database_descs = extract_global_descriptors(
            model,
            database_samples,
            device=device,
            batch_size=batch_size_for_db,
            num_workers=num_workers,
            show_progress=show_progress,
        )
    else:
        if show_progress:
            print("[INFO] Loading database descriptors from cache.")
        database_descs = np.asarray(db_cache["descriptors"], dtype=np.float32)
        if len(database_descs) != len(database_samples):
            raise ValueError("Database cache size does not match database sample table size.")

    query_image_path = Path(query_image_path).expanduser().resolve()
    if show_progress:
        print(f"[INFO] Query image: {query_image_path}")
    query_gray = read_bev_grayscale(query_image_path)
    query_local_feat, query_global_desc = extract_query_features(model, query_gray, device=device)

    top_indices, top_sq_dists = l2_topk(query_global_desc, database_descs, topk=topk)
    if show_progress:
        print(f"[INFO] Retrieved Top-{topk} coarse candidates.")

    candidates: List[Dict[str, object]] = []
    best_refined: Optional[Dict[str, object]] = None
    best_refined_score: Optional[Tuple[int, float]] = None

    candidate_iter = zip(top_indices.tolist(), top_sq_dists.tolist())
    if show_progress and topk > 1:
        candidate_iter = tqdm(
            candidate_iter,
            total=topk,
            desc="Evaluating Top-K candidates",
            leave=False,
        )

    for rank, (db_index, feature_sq_l2) in enumerate(candidate_iter, start=1):
        db_sample = database_samples[db_index]
        db_gray = read_bev_grayscale(db_sample.bev_path)
        db_local_feat, _ = extract_query_features(model, db_gray, device=device)

        resolution_m = resolution_override_m or db_sample.bev_resolution_m
        pose_result = estimate_relative_pose(
            query_gray=query_gray,
            db_gray=db_gray,
            query_local_feat=query_local_feat,
            db_local_feat=db_local_feat,
            resolution_m=resolution_m,
        )

        coarse_pose = {
            "x": db_sample.anchor_x,
            "y": db_sample.anchor_y,
            "yaw_rad": db_sample.anchor_yaw_rad,
            "yaw_deg": db_sample.anchor_yaw_deg,
        }

        candidate: Dict[str, object] = {
            "rank": rank,
            "db_index": db_index,
            "db_sample_key": db_sample.sample_key,
            "db_bev_path": str(db_sample.bev_path),
            "feature_sq_l2": float(feature_sq_l2),
            "coarse_pose": coarse_pose,
            "mode": "coarse_only",
        }

        if pose_result is not None:
            db_pose = pose_to_matrix_2d(db_sample.anchor_x, db_sample.anchor_y, db_sample.anchor_yaw_rad)
            relative_h = np.asarray(pose_result["relative_matrix_3x3"], dtype=np.float64)
            estimated_pose = matrix_to_pose_2d(db_pose @ relative_h)
            candidate["mode"] = "refined"
            candidate["relative_pose"] = pose_result
            candidate["estimated_pose"] = estimated_pose

            score = (int(pose_result["inlier_count"]), -float(feature_sq_l2))
            if best_refined is None or score > best_refined_score:
                best_refined = candidate
                best_refined_score = score

        candidates.append(candidate)

    if best_refined is not None:
        best = best_refined
    else:
        best = dict(candidates[0])
        best["estimated_pose"] = best["coarse_pose"]

    if show_progress:
        mode = best.get("mode", "unknown")
        db_key = best.get("db_sample_key", "unknown")
        print(f"[INFO] Best candidate: {db_key} ({mode})")

    result = {
        "query_image": str(query_image_path),
        "device": str(device),
        "topk": topk,
        "best": best,
        "candidates": candidates,
    }
    return result


def result_to_jsonable(result: Dict[str, object]) -> Dict[str, object]:
    return json.loads(json.dumps(result))
