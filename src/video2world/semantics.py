from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np

from video2world.fusion import PointCloud


LABEL_UNKNOWN = 0
LABEL_SUPPORT_PLANE = 1
LABEL_OBSTACLE = 2

LABEL_NAMES = {
    LABEL_UNKNOWN: "unknown",
    LABEL_SUPPORT_PLANE: "support_plane",
    LABEL_OBSTACLE: "obstacle",
}

LABEL_COLORS = {
    LABEL_UNKNOWN: np.array([140, 140, 140], dtype=np.uint8),
    LABEL_SUPPORT_PLANE: np.array([82, 190, 128], dtype=np.uint8),
    LABEL_OBSTACLE: np.array([231, 86, 74], dtype=np.uint8),
}


@dataclass(frozen=True)
class PlaneModel:
    normal: np.ndarray
    offset: float
    num_inliers: int


@dataclass(frozen=True)
class SemanticResult:
    labels: np.ndarray
    semantic_cloud: PointCloud
    plane: PlaneModel | None
    summary: dict


def derive_geometry_semantics(
    cloud: PointCloud,
    *,
    distance_threshold: float = 0.08,
    obstacle_height: float | None = 0.25,
    ransac_iterations: int = 300,
    random_seed: int = 13,
) -> SemanticResult:
    points = np.asarray(cloud.points, dtype=np.float64).reshape((-1, 3))
    labels = np.full(len(points), LABEL_UNKNOWN, dtype=np.uint8)
    if len(points) < 3:
        return _semantic_result(cloud, labels, None, "too_few_points")

    plane = fit_dominant_plane(
        points,
        distance_threshold=distance_threshold,
        ransac_iterations=ransac_iterations,
        random_seed=random_seed,
    )
    signed_distance = points @ plane.normal + plane.offset
    resolved_obstacle_height = _resolve_obstacle_height(signed_distance, distance_threshold, obstacle_height)
    support_mask = np.abs(signed_distance) <= distance_threshold
    obstacle_mask = signed_distance >= resolved_obstacle_height

    labels[support_mask] = LABEL_SUPPORT_PLANE
    labels[obstacle_mask] = LABEL_OBSTACLE
    return _semantic_result(
        cloud,
        labels,
        plane,
        "geometry_derived_plane_and_height_labels",
        obstacle_height=resolved_obstacle_height,
    )


def fit_dominant_plane(
    points: np.ndarray,
    *,
    distance_threshold: float,
    ransac_iterations: int,
    random_seed: int,
) -> PlaneModel:
    pts = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    if len(pts) < 3:
        raise ValueError("At least three points are required to fit a plane")

    rng = np.random.default_rng(random_seed)
    best_mask = np.zeros(len(pts), dtype=bool)
    best_median_distance = float("inf")
    for _ in range(max(1, ransac_iterations)):
        sample = pts[rng.choice(len(pts), size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-9:
            continue
        normal = normal / norm
        offset = -float(normal @ sample[0])
        distances = np.abs(pts @ normal + offset)
        mask = distances <= distance_threshold
        median_distance = float(np.median(distances[mask])) if np.any(mask) else float("inf")
        if int(mask.sum()) > int(best_mask.sum()) or (int(mask.sum()) == int(best_mask.sum()) and median_distance < best_median_distance):
            best_mask = mask
            best_median_distance = median_distance

    if int(best_mask.sum()) >= 3:
        normal, offset = _fit_plane_svd(pts[best_mask])
    else:
        normal, offset = _fit_plane_svd(pts)
        best_mask = np.abs(pts @ normal + offset) <= distance_threshold

    if normal[2] < 0:
        normal = -normal
        offset = -offset
    return PlaneModel(normal=normal.astype(np.float64), offset=float(offset), num_inliers=int(best_mask.sum()))


def write_semantic_ply_ascii(result: SemanticResult, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cloud = result.semantic_cloud
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(cloud.points)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("property uchar semantic_label\n")
        handle.write("end_header\n")
        for point, color, label in zip(cloud.points, cloud.colors, result.labels, strict=False):
            handle.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} {int(label)}\n"
            )
    return path


def save_semantic_summary(result: SemanticResult, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.summary, indent=2, sort_keys=False), encoding="utf-8")
    return path


def attach_semantics_to_world_model(world_model_path: str | Path, summary: dict, outputs: dict[str, str]) -> None:
    path = Path(world_model_path)
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("outputs", {}).update(outputs)
    data["semantics"] = {
        "enabled": True,
        "label_space": ["support_plane", "obstacle", "unknown"],
        "num_labeled_points": summary.get("num_labeled_points", 0),
        "semantic_coverage": summary.get("semantic_coverage", 0.0),
        "method": summary.get("method", "geometry_derived_plane_and_height_labels"),
    }
    robot_cues = data.setdefault("robot_relevant_cues", {})
    robot_cues["floor_estimated"] = summary.get("label_counts", {}).get("support_plane", 0) > 0
    robot_cues["obstacle_regions_estimated"] = summary.get("label_counts", {}).get("obstacle", 0) > 0
    path.write_text(json.dumps(data, indent=2, sort_keys=False), encoding="utf-8")


def _semantic_result(
    cloud: PointCloud,
    labels: np.ndarray,
    plane: PlaneModel | None,
    method: str,
    *,
    obstacle_height: float | None = None,
) -> SemanticResult:
    colors = np.stack([LABEL_COLORS[int(label)] for label in labels], axis=0).astype(np.uint8) if len(labels) else np.empty((0, 3), dtype=np.uint8)
    semantic_cloud = PointCloud(points=cloud.points.copy(), colors=colors, frame_ids=list(cloud.frame_ids))
    counts = {name: int(np.sum(labels == label)) for label, name in LABEL_NAMES.items()}
    num_labeled = counts["support_plane"] + counts["obstacle"]
    summary = {
        "enabled": True,
        "method": method,
        "label_space": ["support_plane", "obstacle", "unknown"],
        "num_points": int(len(labels)),
        "num_labeled_points": int(num_labeled),
        "semantic_coverage": float(num_labeled / len(labels)) if len(labels) else 0.0,
        "label_counts": counts,
    }
    if obstacle_height is not None:
        summary["obstacle_height"] = float(obstacle_height)
    if plane is not None:
        summary["support_plane"] = {
            "normal": plane.normal.tolist(),
            "offset": plane.offset,
            "num_inliers": plane.num_inliers,
        }
    return SemanticResult(labels=labels, semantic_cloud=semantic_cloud, plane=plane, summary=summary)


def _resolve_obstacle_height(signed_distance: np.ndarray, distance_threshold: float, obstacle_height: float | None) -> float:
    if obstacle_height is not None:
        return float(obstacle_height)
    signed = np.asarray(signed_distance, dtype=np.float64)
    positive = signed[np.isfinite(signed) & (signed > distance_threshold)]
    if len(positive) == 0:
        return float(distance_threshold * 3.0)
    return float(max(distance_threshold * 3.0, np.percentile(positive, 60)))


def _fit_plane_svd(points: np.ndarray) -> tuple[np.ndarray, float]:
    centroid = np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vh[-1]
    normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
    offset = -float(normal @ centroid)
    return normal.astype(np.float64), float(offset)
