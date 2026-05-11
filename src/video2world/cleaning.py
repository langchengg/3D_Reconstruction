from __future__ import annotations

from pathlib import Path

import numpy as np

from video2world.fusion import PointCloud, write_ply_ascii


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
) -> PointCloud:
    if len(cloud.points) == 0:
        return cloud

    try:
        import open3d as o3d
    except ImportError:
        return voxel_downsample_numpy(cloud, voxel_size)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud.points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector((cloud.colors.astype(np.float64) / 255.0).clip(0.0, 1.0))
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
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
    )
    return write_ply_ascii(cleaned, output_ply)

