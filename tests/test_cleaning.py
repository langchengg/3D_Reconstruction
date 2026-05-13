from __future__ import annotations

import numpy as np

from video2world.cleaning import adaptive_voxel_size, clean_point_cloud, robust_crop_mad
from video2world.fusion import PointCloud


def test_robust_crop_mad_removes_extreme_fusion_outliers():
    cluster = np.array([[float(i), 0.1 * float(i), 2.0] for i in range(20)], dtype=np.float64)
    outliers = np.array([[1e8, 0.0, 0.0], [0.0, -1e8, 0.0]], dtype=np.float64)
    points = np.vstack([cluster, outliers])
    cloud = PointCloud(
        points=points,
        colors=np.full((len(points), 3), 128, dtype=np.uint8),
        frame_ids=list(range(len(points))),
    )

    cropped = robust_crop_mad(cloud, mad_scale=12.0, min_keep_ratio=0.25)

    assert len(cropped.points) == len(cluster)
    assert np.max(np.abs(cropped.points)) < 20.0


def test_adaptive_voxel_size_tracks_scene_scale():
    points = np.array([[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0]], dtype=np.float64)

    assert adaptive_voxel_size(points, 0.001, min_scene_ratio=0.01) == 10.0
    assert adaptive_voxel_size(points, 20.0, min_scene_ratio=0.01) == 20.0


def test_clean_point_cloud_handles_outlier_scale_before_open3d_voxelization():
    cluster = np.array([[float(i), 0.0, 1.0] for i in range(30)], dtype=np.float64)
    points = np.vstack([cluster, np.array([[0.0, 1e9, 0.0]], dtype=np.float64)])
    cloud = PointCloud(
        points=points,
        colors=np.full((len(points), 3), 200, dtype=np.uint8),
        frame_ids=[0] * len(points),
    )

    cleaned = clean_point_cloud(
        cloud,
        voxel_size=0.001,
        remove_outliers=False,
        statistical_nb_neighbors=20,
        statistical_std_ratio=2.0,
        radius_outlier_removal=False,
        radius=0.08,
        radius_nb_points=12,
    )

    assert len(cleaned.points) > 0
    assert np.max(np.abs(cleaned.points)) < 100.0
