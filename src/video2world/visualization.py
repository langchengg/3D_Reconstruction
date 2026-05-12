from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from video2world.colmap_io import ColmapModel
from video2world.fusion import PointCloud


def _setup_matplotlib():
    cache_dir = Path.cwd() / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_camera_trajectory(model: ColmapModel, output_path: str | Path) -> Path:
    plt = _setup_matplotlib()
    centers = []
    for image in model.images.values():
        centers.append(image.camera_to_world[:3, 3])
    centers_arr = np.array(centers, dtype=np.float64)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    if len(centers_arr):
        ax.plot(centers_arr[:, 0], centers_arr[:, 1], centers_arr[:, 2], "-o", markersize=3, linewidth=1.5)
        ax.scatter(centers_arr[0, 0], centers_arr[0, 1], centers_arr[0, 2], c="green", label="start")
        ax.scatter(centers_arr[-1, 0], centers_arr[-1, 1], centers_arr[-1, 2], c="red", label="end")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("COLMAP camera trajectory")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_pointcloud_preview(cloud: PointCloud, output_path: str | Path, *, max_points: int = 50000) -> Path:
    plt = _setup_matplotlib()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    points = cloud.points
    colors = cloud.colors.astype(np.float64) / 255.0
    if len(points) > max_points:
        idx = np.linspace(0, len(points) - 1, max_points).astype(np.int64)
        points = points[idx]
        colors = colors[idx]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    if len(points):
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.2, c=colors)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Reconstructed scene preview")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path
