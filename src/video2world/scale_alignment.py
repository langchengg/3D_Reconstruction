from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from video2world.colmap_io import ImageRecord, Point3D


@dataclass(frozen=True)
class AlignmentResult:
    scale: float
    shift: float
    num_inliers: int
    success: bool
    reason: str = ""


def fit_scale_shift(
    predicted_depth: np.ndarray,
    sparse_depth: np.ndarray,
    *,
    min_points: int = 3,
    trim_quantile: float = 0.1,
) -> AlignmentResult:
    """Fit sparse_depth ~= scale * predicted_depth + shift with simple robust trimming."""
    pred = np.asarray(predicted_depth, dtype=np.float64).reshape(-1)
    sparse = np.asarray(sparse_depth, dtype=np.float64).reshape(-1)
    if len(pred) != len(sparse):
        raise ValueError("predicted_depth and sparse_depth must have the same length")

    valid = np.isfinite(pred) & np.isfinite(sparse)
    pred = pred[valid]
    sparse = sparse[valid]
    if len(pred) < min_points:
        return AlignmentResult(1.0, 0.0, 0, False, "too few finite samples")

    order = np.argsort(pred)
    trim = int(np.floor(len(order) * trim_quantile))
    if trim > 0 and len(order) - 2 * trim >= min_points:
        order = order[trim : len(order) - trim]
    x = pred[order]
    y = sparse[order]

    scale, shift = _least_squares_line(x, y)
    residual = np.abs((scale * x + shift) - y)
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
    if mad > 0:
        threshold = median + 3.0 * 1.4826 * mad
        inlier_mask = residual <= threshold
    else:
        inlier_mask = np.ones_like(residual, dtype=bool)

    if int(inlier_mask.sum()) < min_points:
        return AlignmentResult(scale, shift, int(inlier_mask.sum()), False, "too few inliers after robust fit")

    scale, shift = _least_squares_line(x[inlier_mask], y[inlier_mask])
    return AlignmentResult(float(scale), float(shift), int(inlier_mask.sum()), True)


def _least_squares_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    design = np.stack([x, np.ones_like(x)], axis=1)
    scale, shift = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(scale), float(shift)


def collect_alignment_samples(
    image: ImageRecord,
    points3d: dict[int, Point3D],
    predicted_depth: np.ndarray,
    *,
    invert_prediction: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect predicted pixel depth and COLMAP camera-frame sparse depth for one image."""
    depth_map = np.asarray(predicted_depth, dtype=np.float64)
    height, width = depth_map.shape[:2]
    pred_values: list[float] = []
    sparse_values: list[float] = []

    for xy, point_id in zip(image.xys, image.point3d_ids, strict=False):
        if point_id < 0 or point_id not in points3d:
            continue
        u = int(round(float(xy[0])))
        v = int(round(float(xy[1])))
        if u < 0 or v < 0 or u >= width or v >= height:
            continue
        pred = float(depth_map[v, u])
        if not np.isfinite(pred):
            continue
        if invert_prediction:
            if pred <= 1e-8:
                continue
            pred = 1.0 / pred
        sparse_z = image.world_to_camera_depth(points3d[point_id].xyz)
        if sparse_z <= 0 or not np.isfinite(sparse_z):
            continue
        pred_values.append(pred)
        sparse_values.append(sparse_z)

    return np.array(pred_values, dtype=np.float64), np.array(sparse_values, dtype=np.float64)


def align_depth_map(depth_map: np.ndarray, result: AlignmentResult) -> np.ndarray:
    if not result.success:
        raise ValueError(f"Cannot align depth map with failed result: {result.reason}")
    aligned = result.scale * np.asarray(depth_map, dtype=np.float64) + result.shift
    return aligned.astype(np.float32)

