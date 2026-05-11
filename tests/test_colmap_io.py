from pathlib import Path

import numpy as np

from video2world.colmap_io import read_colmap_text_model


def test_read_colmap_text_model_parses_cameras_images_and_points(tmp_path: Path):
    (tmp_path / "cameras.txt").write_text(
        "# Camera list\n"
        "1 SIMPLE_PINHOLE 640 480 500 320 240\n",
        encoding="utf-8",
    )
    (tmp_path / "images.txt").write_text(
        "# Image list\n"
        "1 1 0 0 0 0 0 0 1 frame_000001.jpg\n"
        "10 20 7 30 40 -1\n",
        encoding="utf-8",
    )
    (tmp_path / "points3D.txt").write_text(
        "# Point list\n"
        "7 0 0 2 255 0 0 0.1 1 0\n",
        encoding="utf-8",
    )

    model = read_colmap_text_model(tmp_path)

    assert model.cameras[1].model == "SIMPLE_PINHOLE"
    np.testing.assert_allclose(
        model.cameras[1].intrinsics,
        np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]),
    )
    assert model.images[1].name == "frame_000001.jpg"
    assert model.images[1].point3d_ids == [7, -1]
    np.testing.assert_allclose(model.points3d[7].xyz, np.array([0.0, 0.0, 2.0]))
