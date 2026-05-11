from pathlib import Path

import numpy as np

from video2world.fusion import PointCloud, fuse_rgbd_frames, write_ply_ascii


def test_fuse_rgbd_frames_unprojects_and_colors_points():
    image = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )
    depth = np.ones((2, 2), dtype=np.float64)
    intrinsics = np.eye(3)
    camera_to_world = np.eye(4)

    cloud = fuse_rgbd_frames(
        frames=[("frame_000001.jpg", image, depth, intrinsics, camera_to_world)],
        pixel_stride=1,
        min_depth=0.1,
        max_depth=3.0,
    )

    assert cloud.points.shape == (4, 3)
    assert cloud.colors.shape == (4, 3)
    assert cloud.frame_ids == [0, 0, 0, 0]
    np.testing.assert_allclose(cloud.points[3], np.array([1.0, 1.0, 1.0]))
    np.testing.assert_array_equal(cloud.colors[0], np.array([255, 0, 0], dtype=np.uint8))


def test_write_ply_ascii_creates_valid_header(tmp_path: Path):
    cloud = PointCloud(
        points=np.array([[1.0, 2.0, 3.0]]),
        colors=np.array([[10, 20, 30]], dtype=np.uint8),
        frame_ids=[2],
    )
    output = tmp_path / "scene.ply"

    write_ply_ascii(cloud, output)

    text = output.read_text(encoding="utf-8")
    assert "element vertex 1" in text
    assert "1.000000 2.000000 3.000000 10 20 30" in text
