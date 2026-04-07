import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _safe_float(value: str, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: str, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass
class Sample:
    sample_index: int
    sample_key: str
    dataset_tag: str
    bev_path: Path
    pcd_path: Optional[Path]
    anchor_stamp: Optional[float]
    anchor_x: float
    anchor_y: float
    anchor_z: float
    anchor_yaw_rad: float
    anchor_yaw_deg: float
    bev_width: int
    bev_height: int
    bev_resolution_m: float
    bev_origin_mode: str


def load_samples(csv_path: str, data_root: Optional[str] = None) -> List[Sample]:
    csv_path = Path(csv_path).expanduser().resolve()
    if data_root is None:
        data_root = csv_path.parents[1]
    data_root = Path(data_root).expanduser().resolve()

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    rows = sorted(rows, key=lambda row: int(row["sample_index"]))
    samples: List[Sample] = []
    for expected_index, row in enumerate(rows):
        sample_index = int(row["sample_index"])
        if sample_index != expected_index:
            raise ValueError(
                f"Unexpected sample_index order in {csv_path}: got {sample_index}, expected {expected_index}"
            )

        bev_path = (data_root / row["bev_rel_path"]).resolve()
        pcd_rel_path = (row.get("pcd_rel_path") or "").strip()
        pcd_path = (data_root / pcd_rel_path).resolve() if pcd_rel_path else None

        samples.append(
            Sample(
                sample_index=sample_index,
                sample_key=row["sample_key"],
                dataset_tag=row["dataset_tag"],
                bev_path=bev_path,
                pcd_path=pcd_path,
                anchor_stamp=_safe_float(row.get("anchor_stamp")),
                anchor_x=float(row["anchor_x"]),
                anchor_y=float(row["anchor_y"]),
                anchor_z=float(row.get("anchor_z", 0.0) or 0.0),
                anchor_yaw_rad=float(row["anchor_yaw_rad"]),
                anchor_yaw_deg=float(row["anchor_yaw_deg"]),
                bev_width=int(row["bev_width"]),
                bev_height=int(row["bev_height"]),
                bev_resolution_m=float(row["bev_resolution_m"]),
                bev_origin_mode=row.get("bev_origin_mode", ""),
            )
        )

    return samples


def find_sample_by_index(samples: Sequence[Sample], sample_index: int) -> Sample:
    if sample_index < 0 or sample_index >= len(samples):
        raise IndexError(f"sample_index out of range: {sample_index}")
    return samples[sample_index]


def find_sample_by_key(samples: Sequence[Sample], sample_key: str) -> Sample:
    for sample in samples:
        if sample.sample_key == sample_key:
            return sample
    raise KeyError(f"sample_key not found: {sample_key}")


def read_bev_grayscale(image_path: str) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Failed to read BEV image: {image_path}")
    return image


def bev_to_model_tensor(gray_image: np.ndarray) -> torch.Tensor:
    image = gray_image.astype(np.float32) / 256.0
    image = np.repeat(image[np.newaxis, :, :], 3, axis=0)
    return torch.from_numpy(image)


class BEVImageDataset(Dataset):
    def __init__(self, samples: Sequence[Sample]):
        self.samples = list(samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        gray = read_bev_grayscale(sample.bev_path)
        tensor = bev_to_model_tensor(gray)
        return tensor, index

    def __len__(self):
        return len(self.samples)
