from __future__ import annotations

import numpy as np

from video2world.fusion import PointCloud
from video2world.semantics import LABEL_OBSTACLE, LABEL_SUPPORT_PLANE, derive_geometry_semantics


def test_derive_geometry_semantics_labels_support_plane_and_obstacles():
    floor_points = np.array(
        [[x, y, 0.0] for x in np.linspace(-1.0, 1.0, 5) for y in np.linspace(-1.0, 1.0, 5)],
        dtype=np.float64,
    )
    obstacle_points = np.array([[0.2, 0.1, 0.45], [-0.4, 0.3, 0.75]], dtype=np.float64)
    points = np.vstack([floor_points, obstacle_points])
    cloud = PointCloud(
        points=points,
        colors=np.full((len(points), 3), 180, dtype=np.uint8),
        frame_ids=[0] * len(points),
    )

    result = derive_geometry_semantics(
        cloud,
        distance_threshold=0.03,
        obstacle_height=0.20,
        ransac_iterations=80,
        random_seed=7,
    )

    assert int(np.sum(result.labels == LABEL_SUPPORT_PLANE)) >= len(floor_points)
    assert int(np.sum(result.labels == LABEL_OBSTACLE)) == len(obstacle_points)
    assert result.summary["method"] == "geometry_derived_plane_and_height_labels"
    assert result.summary["semantic_coverage"] == 1.0
    assert result.semantic_cloud.colors.shape == cloud.colors.shape


def test_derive_geometry_semantics_uses_adaptive_obstacle_height_when_unspecified():
    floor_points = np.array([[float(x), float(y), 0.0] for x in range(4) for y in range(4)], dtype=np.float64)
    obstacle_points = np.array([[1.0, 1.0, 4.0], [2.0, 2.0, 5.0]], dtype=np.float64)
    points = np.vstack([floor_points, obstacle_points])
    cloud = PointCloud(
        points=points,
        colors=np.full((len(points), 3), 180, dtype=np.uint8),
        frame_ids=[0] * len(points),
    )

    result = derive_geometry_semantics(
        cloud,
        distance_threshold=0.03,
        obstacle_height=None,
        ransac_iterations=80,
        random_seed=11,
    )

    assert result.summary["obstacle_height"] > 0.03
    assert int(np.sum(result.labels == LABEL_OBSTACLE)) <= len(obstacle_points)
