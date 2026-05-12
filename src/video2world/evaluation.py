from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from video2world.colmap_io import read_colmap_text_model
from video2world.scale_alignment import collect_alignment_samples


def build_evaluation_report(run_dir: str | Path) -> dict:
    root = Path(run_dir)
    world_model_path = root / "world_model" / "world_model.json"
    world_model = _read_json(world_model_path, default={})
    alignment_report = _read_json(root / "depth_aligned" / "alignment_report.json", default={})
    semantic_summary = _read_json(root / "world_model" / "semantic_objects.json", default={})

    num_keyframes = int(world_model.get("num_keyframes", _count_files(root / "frames", "*.jpg")))
    num_registered_frames = int(world_model.get("num_registered_frames", _count_colmap_images(root / "colmap" / "text_model")))
    num_sparse_points = int(world_model.get("num_sparse_points", _count_colmap_points(root / "colmap" / "text_model")))
    num_dense_points_raw = int(world_model.get("num_dense_points_raw", 0))
    num_dense_points_cleaned = int(world_model.get("num_dense_points_cleaned", 0))

    median_residual = _median_reported_alignment_residual(alignment_report)
    median_residual_colmap_scale = None
    if median_residual is None:
        predicted, sparse = collect_aligned_depth_samples(root / "colmap" / "text_model", root / "depth_aligned")
        residual_stats = summarize_alignment_residuals(predicted, sparse)
        median_residual = residual_stats["median_alignment_residual"]
        median_residual_colmap_scale = residual_stats["median_alignment_residual_colmap_scale"]

    report = {
        "num_keyframes": num_keyframes,
        "num_registered_frames": num_registered_frames,
        "registration_ratio": _safe_ratio(num_registered_frames, num_keyframes),
        "num_sparse_points": num_sparse_points,
        "num_dense_points_raw": num_dense_points_raw,
        "num_dense_points_cleaned": num_dense_points_cleaned,
        "alignment_success_rate": _alignment_success_rate(alignment_report),
        "median_alignment_residual": median_residual,
        "median_alignment_residual_colmap_scale": median_residual_colmap_scale,
        "outlier_removed_ratio": _safe_ratio(num_dense_points_raw - num_dense_points_cleaned, num_dense_points_raw),
        "has_camera_trajectory": (root / "visualizations" / "camera_trajectory.png").exists(),
        "has_cleaned_pointcloud": (root / "pointclouds" / "cleaned_scene.ply").exists(),
        "has_semantic_scene": (root / "pointclouds" / "semantic_scene.ply").exists(),
        "has_world_model_json": world_model_path.exists(),
        "semantic_coverage": semantic_summary.get("semantic_coverage"),
        "num_semantic_labeled_points": semantic_summary.get("num_labeled_points"),
    }
    return report


def save_evaluation_report(run_dir: str | Path, output_path: str | Path | None = None) -> Path:
    root = Path(run_dir)
    path = Path(output_path) if output_path is not None else root / "evaluation" / "evaluation_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_evaluation_report(root), indent=2, sort_keys=False), encoding="utf-8")
    return path


def collect_aligned_depth_samples(model_dir: str | Path, depth_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    model_path = Path(model_dir)
    depth_path = Path(depth_dir)
    if not model_path.exists() or not depth_path.exists():
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    model = read_colmap_text_model(model_path)
    predicted_values: list[np.ndarray] = []
    sparse_values: list[np.ndarray] = []
    for image in model.images.values():
        depth_file = depth_path / f"{Path(image.name).stem}.npy"
        if not depth_file.exists():
            continue
        aligned_depth = np.load(depth_file)
        predicted, sparse = collect_alignment_samples(image, model.points3d, aligned_depth)
        if len(predicted):
            predicted_values.append(predicted)
            sparse_values.append(sparse)

    if not predicted_values:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    return np.concatenate(predicted_values).astype(np.float64), np.concatenate(sparse_values).astype(np.float64)


def compute_alignment_residuals(model_dir: str | Path, depth_dir: str | Path) -> np.ndarray:
    predicted, sparse = collect_aligned_depth_samples(model_dir, depth_dir)
    stats = summarize_alignment_residuals(predicted, sparse)
    value = stats["median_alignment_residual"]
    return np.array([], dtype=np.float64) if value is None else np.array([value], dtype=np.float64)


def summarize_alignment_residuals(predicted_depth: np.ndarray, sparse_depth: np.ndarray) -> dict:
    predicted = np.asarray(predicted_depth, dtype=np.float64).reshape(-1)
    sparse = np.asarray(sparse_depth, dtype=np.float64).reshape(-1)
    valid = np.isfinite(predicted) & np.isfinite(sparse) & (np.abs(sparse) > 1e-8)
    if not np.any(valid):
        return {
            "median_alignment_residual": None,
            "median_alignment_residual_colmap_scale": None,
        }
    absolute = np.abs(predicted[valid] - sparse[valid])
    relative = absolute / np.abs(sparse[valid])
    return {
        "median_alignment_residual": float(np.median(relative)),
        "median_alignment_residual_colmap_scale": float(np.median(absolute)),
    }


def _read_json(path: Path, *, default: dict) -> dict:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _count_files(path: Path, pattern: str) -> int:
    return len(list(path.glob(pattern))) if path.exists() else 0


def _count_colmap_images(model_dir: Path) -> int:
    images_path = model_dir / "images.txt"
    if not images_path.exists():
        return 0
    lines = [line for line in images_path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    return len(lines) // 2


def _count_colmap_points(model_dir: Path) -> int:
    points_path = model_dir / "points3D.txt"
    if not points_path.exists():
        return 0
    return len([line for line in points_path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")])


def _alignment_success_rate(alignment_report: dict) -> float | None:
    if not alignment_report:
        return None
    successes = [bool(entry.get("success", False)) for entry in alignment_report.values()]
    return _safe_ratio(sum(successes), len(successes))


def _median_reported_alignment_residual(alignment_report: dict) -> float | None:
    residuals = [
        float(entry["median_residual"])
        for entry in alignment_report.values()
        if entry.get("success") and entry.get("median_residual") is not None
    ]
    if not residuals:
        return None
    return float(np.median(np.array(residuals, dtype=np.float64)))


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)
