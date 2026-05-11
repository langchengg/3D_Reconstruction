import numpy as np

from video2world.geometry import camera_to_world_from_colmap, transform_points, unproject_pixels


def test_unproject_pixels_uses_intrinsics_and_depth():
    pixels = np.array([[2.0, 3.0], [4.0, 3.0]], dtype=np.float64)
    depth = np.array([2.0, 4.0], dtype=np.float64)
    k = np.array(
        [
            [2.0, 0.0, 2.0],
            [0.0, 2.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    points = unproject_pixels(pixels, depth, k)

    np.testing.assert_allclose(
        points,
        np.array(
            [
                [0.0, 2.0, 2.0],
                [4.0, 4.0, 4.0],
            ]
        ),
    )


def test_camera_to_world_from_colmap_inverts_world_to_camera_pose():
    rotation_world_to_camera = np.eye(3)
    translation_world_to_camera = np.array([1.0, 2.0, 3.0])

    transform = camera_to_world_from_colmap(rotation_world_to_camera, translation_world_to_camera)
    world_origin = transform_points(np.array([[0.0, 0.0, 0.0]]), transform)

    np.testing.assert_allclose(world_origin, np.array([[-1.0, -2.0, -3.0]]))
