import numpy as np

from video2world.fusion import PointCloud
from video2world.presentation import crop_cloud_to_anchor_bounds, render_point_cloud_camera_view, render_point_cloud_splat


def test_crop_cloud_to_anchor_bounds_removes_far_depth_outliers(tmp_path):
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [100.0, 100.0, 100.0],
        ],
        dtype=np.float64,
    )
    colors = np.full((5, 3), 120, dtype=np.uint8)
    cloud = PointCloud(points=points, colors=colors, frame_ids=[0] * 5)
    anchors = points[:4]

    cropped, report = crop_cloud_to_anchor_bounds(
        cloud,
        anchors,
        lower_quantile=0.0,
        upper_quantile=1.0,
        margin_ratio=0.25,
        min_margin=0.1,
    )

    assert len(cropped.points) == 4
    assert report["input_points"] == 5
    assert report["output_points"] == 4


def test_render_point_cloud_splat_writes_nonempty_png(tmp_path):
    points = np.array(
        [[x, y, x + y] for x in np.linspace(0.0, 1.0, 10) for y in np.linspace(0.0, 1.0, 10)],
        dtype=np.float64,
    )
    colors = np.tile(np.array([[30, 120, 220]], dtype=np.uint8), (len(points), 1))
    cloud = PointCloud(points=points, colors=colors, frame_ids=[0] * len(points))
    output = tmp_path / "render.png"

    path = render_point_cloud_splat(cloud, output, width=320, height=240, point_radius=2)

    assert path.exists()
    assert path.stat().st_size > 1000


def test_render_point_cloud_camera_view_projects_visible_points(tmp_path):
    points = np.array(
        [
            [-0.5, -0.5, 2.0],
            [0.5, -0.5, 2.0],
            [0.5, 0.5, 2.0],
            [-0.5, 0.5, 2.0],
        ],
        dtype=np.float64,
    )
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]], dtype=np.uint8)
    cloud = PointCloud(points=points, colors=colors, frame_ids=[0] * len(points))
    intrinsics = np.array([[100.0, 0.0, 80.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]])

    output = tmp_path / "camera_view.png"
    path = render_point_cloud_camera_view(
        cloud,
        intrinsics,
        np.eye(3),
        np.zeros(3),
        output,
        width=160,
        height=120,
        point_radius=3,
    )

    assert path.exists()
    assert path.stat().st_size > 500
