from __future__ import annotations

import numpy as np


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert a COLMAP qvec in [qw, qx, qy, qz] order to a rotation matrix."""
    q = np.asarray(qvec, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"qvec must have shape (4,), got {q.shape}")

    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("qvec must be non-zero")
    qw, qx, qy, qz = q / norm

    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )


def camera_to_world_from_colmap(rotation_world_to_camera: np.ndarray, translation_world_to_camera: np.ndarray) -> np.ndarray:
    """Return a 4x4 camera-to-world matrix from COLMAP world-to-camera pose."""
    r_cw = np.asarray(rotation_world_to_camera, dtype=np.float64)
    t_cw = np.asarray(translation_world_to_camera, dtype=np.float64).reshape(3)
    if r_cw.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3, 3), got {r_cw.shape}")

    r_wc = r_cw.T
    t_wc = -r_wc @ t_cw
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = r_wc
    transform[:3, 3] = t_wc
    return transform


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to Nx3 points."""
    pts = np.asarray(points, dtype=np.float64)
    tfm = np.asarray(transform, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}")
    if tfm.shape != (4, 4):
        raise ValueError(f"transform must have shape (4, 4), got {tfm.shape}")
    homogeneous = np.concatenate([pts, np.ones((len(pts), 1), dtype=np.float64)], axis=1)
    transformed = homogeneous @ tfm.T
    return transformed[:, :3]


def unproject_pixels(pixels: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Unproject Nx2 image pixels and N depths into camera-frame XYZ points."""
    px = np.asarray(pixels, dtype=np.float64)
    z = np.asarray(depth, dtype=np.float64).reshape(-1)
    k = np.asarray(intrinsics, dtype=np.float64)
    if px.ndim != 2 or px.shape[1] != 2:
        raise ValueError(f"pixels must have shape (N, 2), got {px.shape}")
    if len(px) != len(z):
        raise ValueError("pixels and depth must contain the same number of samples")
    if k.shape != (3, 3):
        raise ValueError(f"intrinsics must have shape (3, 3), got {k.shape}")

    fx = k[0, 0]
    fy = k[1, 1]
    cx = k[0, 2]
    cy = k[1, 2]
    if fx == 0 or fy == 0:
        raise ValueError("camera focal length must be non-zero")

    x = (px[:, 0] - cx) * z / fx
    y = (px[:, 1] - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def sample_pixel_grid(height: int, width: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Return meshgrid x/y pixel coordinates sampled at a fixed stride."""
    if stride <= 0:
        raise ValueError("stride must be positive")
    ys = np.arange(0, height, stride, dtype=np.int64)
    xs = np.arange(0, width, stride, dtype=np.int64)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return grid_x.reshape(-1), grid_y.reshape(-1)

