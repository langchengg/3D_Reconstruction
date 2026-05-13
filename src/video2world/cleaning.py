from __future__ import annotations

from pathlib import Path

import numpy as np

from video2world.fusion import PointCloud, write_ply_ascii


def _filter_points(cloud: PointCloud, mask: np.ndarray) -> PointCloud:
    mask = np.asarray(mask, dtype=bool)
    if len(mask) != len(cloud.points):
        raise ValueError("mask must match point count")
    frame_ids = [frame_id for frame_id, keep in zip(cloud.frame_ids, mask.tolist(), strict=False) if keep]
    return PointCloud(
        points=cloud.points[mask],
        colors=cloud.colors[mask],
        frame_ids=frame_ids,
    )


def filter_finite_points(cloud: PointCloud) -> PointCloud:
    if len(cloud.points) == 0:
        return cloud
    finite = np.isfinite(cloud.points).all(axis=1)
    if bool(np.all(finite)):
        return cloud
    return _filter_points(cloud, finite)


def robust_crop_mad(
    cloud: PointCloud,
    *,
    mad_scale: float = 12.0,
    min_keep_ratio: float = 0.25,
) -> PointCloud:
    """Remove extreme fused-depth outliers without assuming metric scale."""
    if len(cloud.points) < 8 or mad_scale <= 0:
        return cloud

    points = cloud.points.astype(np.float64, copy=False)
    median = np.median(points, axis=0)
    mad = np.median(np.abs(points - median), axis=0)
    spread = np.ptp(points, axis=0)
    scale = np.maximum(mad, spread * 1e-6)
    scale = np.maximum(scale, 1e-12)
    keep = np.all(np.abs(points - median) <= mad_scale * scale, axis=1)
    keep_ratio = float(np.mean(keep)) if len(keep) else 0.0
    if keep_ratio < min_keep_ratio or not bool(np.any(keep)):
        return cloud
    return _filter_points(cloud, keep)


def adaptive_voxel_size(points: np.ndarray, requested_voxel_size: float, min_scene_ratio: float = 1e-5) -> float:
    if requested_voxel_size <= 0 or len(points) == 0:
        return requested_voxel_size
    bounds = np.ptp(points.astype(np.float64, copy=False), axis=0)
    diagonal = float(np.linalg.norm(bounds))
    if not np.isfinite(diagonal) or diagonal <= 0:
        return requested_voxel_size
    return max(float(requested_voxel_size), diagonal * float(min_scene_ratio))


def voxel_downsample_numpy(cloud: PointCloud, voxel_size: float) -> PointCloud:
    if voxel_size <= 0 or len(cloud.points) == 0:
        return cloud
    voxel_index = np.floor(cloud.points / voxel_size).astype(np.int64)
    unique_voxels, inverse = np.unique(voxel_index, axis=0, return_inverse=True)
    points = np.zeros((len(unique_voxels), 3), dtype=np.float64)
    colors = np.zeros((len(unique_voxels), 3), dtype=np.float64)
    counts = np.bincount(inverse)
    np.add.at(points, inverse, cloud.points)
    np.add.at(colors, inverse, cloud.colors.astype(np.float64))
    points /= counts[:, None]
    colors = np.clip(colors / counts[:, None], 0, 255).astype(np.uint8)
    return PointCloud(points=points, colors=colors, frame_ids=[0] * len(points))


def clean_point_cloud(
    cloud: PointCloud,
    *,
    voxel_size: float,
    remove_outliers: bool,
    statistical_nb_neighbors: int,
    statistical_std_ratio: float,
    radius_outlier_removal: bool,
    radius: float,
    radius_nb_points: int,
    robust_crop: bool = True,
    robust_crop_mad_scale: float = 12.0,
    robust_crop_min_keep_ratio: float = 0.25,
    adaptive_voxel_min_scene_ratio: float = 1e-5,
) -> PointCloud:
    if len(cloud.points) == 0:
        return cloud

    cloud = filter_finite_points(cloud)
    if robust_crop:
        cloud = robust_crop_mad(
            cloud,
            mad_scale=robust_crop_mad_scale,
            min_keep_ratio=robust_crop_min_keep_ratio,
        )
    voxel_size = adaptive_voxel_size(
        cloud.points,
        voxel_size,
        min_scene_ratio=adaptive_voxel_min_scene_ratio,
    )

    try:
        import open3d as o3d
    except ImportError:
        return voxel_downsample_numpy(cloud, voxel_size)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud.points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector((cloud.colors.astype(np.float64) / 255.0).clip(0.0, 1.0))
    if voxel_size > 0:
        try:
            pcd = pcd.voxel_down_sample(voxel_size)
        except RuntimeError:
            return voxel_downsample_numpy(cloud, voxel_size)
    if remove_outliers and len(pcd.points) > statistical_nb_neighbors:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=statistical_nb_neighbors,
            std_ratio=statistical_std_ratio,
        )
    if radius_outlier_removal and len(pcd.points) > radius_nb_points:
        pcd, _ = pcd.remove_radius_outlier(nb_points=radius_nb_points, radius=radius)

    colors = np.asarray(pcd.colors)
    if len(colors) == 0:
        rgb = np.empty((0, 3), dtype=np.uint8)
    else:
        rgb = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    points = np.asarray(pcd.points, dtype=np.float64).reshape((-1, 3))
    return PointCloud(points=points, colors=rgb.reshape((-1, 3)), frame_ids=[0] * len(points))


def clean_ply_file(input_ply: str | Path, output_ply: str | Path, config: dict) -> Path:
    from video2world.fusion import read_ply_ascii

    cloud = read_ply_ascii(input_ply)
    cleaned = clean_point_cloud(
        cloud,
        voxel_size=float(config.get("voxel_size", 0.03)),
        remove_outliers=bool(config.get("remove_outliers", True)),
        statistical_nb_neighbors=int(config.get("statistical_nb_neighbors", 20)),
        statistical_std_ratio=float(config.get("statistical_std_ratio", 2.0)),
        radius_outlier_removal=bool(config.get("radius_outlier_removal", True)),
        radius=float(config.get("radius", 0.08)),
        radius_nb_points=int(config.get("radius_nb_points", 12)),
        robust_crop=bool(config.get("robust_crop", True)),
        robust_crop_mad_scale=float(config.get("robust_crop_mad_scale", 12.0)),
        robust_crop_min_keep_ratio=float(config.get("robust_crop_min_keep_ratio", 0.25)),
        adaptive_voxel_min_scene_ratio=float(config.get("adaptive_voxel_min_scene_ratio", 1e-5)),
    )
    return write_ply_ascii(cleaned, output_ply)
