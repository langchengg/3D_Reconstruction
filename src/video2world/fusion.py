from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from video2world.geometry import sample_pixel_grid, transform_points, unproject_pixels


@dataclass(frozen=True)
class PointCloud:
    points: np.ndarray
    colors: np.ndarray
    frame_ids: list[int]

    def __post_init__(self) -> None:
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        if self.colors.ndim != 2 or self.colors.shape[1] != 3:
            raise ValueError("colors must have shape (N, 3)")
        if len(self.points) != len(self.colors) or len(self.points) != len(self.frame_ids):
            raise ValueError("points, colors, and frame_ids must have matching lengths")

    @classmethod
    def empty(cls) -> "PointCloud":
        return cls(
            points=np.empty((0, 3), dtype=np.float64),
            colors=np.empty((0, 3), dtype=np.uint8),
            frame_ids=[],
        )


RgbdFrame = tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def fuse_rgbd_frames(
    frames: Iterable[RgbdFrame],
    *,
    pixel_stride: int,
    min_depth: float,
    max_depth: float,
    max_points_per_frame: int | None = None,
) -> PointCloud:
    """Fuse registered RGB-D frames into a colored point cloud."""
    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    all_frame_ids: list[int] = []

    for frame_index, (_name, image, depth, intrinsics, camera_to_world) in enumerate(frames):
        rgb = np.asarray(image)
        z = np.asarray(depth, dtype=np.float64)
        if rgb.shape[:2] != z.shape[:2]:
            raise ValueError("RGB image and depth map must have the same height and width")

        height, width = z.shape[:2]
        xs, ys = sample_pixel_grid(height, width, pixel_stride)
        sampled_depth = z[ys, xs]
        valid = np.isfinite(sampled_depth) & (sampled_depth >= min_depth) & (sampled_depth <= max_depth)
        xs = xs[valid]
        ys = ys[valid]
        sampled_depth = sampled_depth[valid]

        if max_points_per_frame is not None and len(sampled_depth) > max_points_per_frame:
            keep = np.linspace(0, len(sampled_depth) - 1, max_points_per_frame).astype(np.int64)
            xs = xs[keep]
            ys = ys[keep]
            sampled_depth = sampled_depth[keep]

        if len(sampled_depth) == 0:
            continue

        pixels = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
        camera_points = unproject_pixels(pixels, sampled_depth, intrinsics)
        world_points = transform_points(camera_points, camera_to_world)
        colors = rgb[ys, xs, :3].astype(np.uint8)

        all_points.append(world_points)
        all_colors.append(colors)
        all_frame_ids.extend([frame_index] * len(world_points))

    if not all_points:
        return PointCloud.empty()

    return PointCloud(
        points=np.concatenate(all_points, axis=0),
        colors=np.concatenate(all_colors, axis=0),
        frame_ids=all_frame_ids,
    )


def concatenate_clouds(clouds: Iterable[PointCloud]) -> PointCloud:
    points: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    frame_ids: list[int] = []
    for cloud in clouds:
        points.append(cloud.points)
        colors.append(cloud.colors)
        frame_ids.extend(cloud.frame_ids)
    if not points:
        return PointCloud.empty()
    return PointCloud(np.concatenate(points, axis=0), np.concatenate(colors, axis=0), frame_ids)


def write_ply_ascii(cloud: PointCloud, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(cloud.points)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(cloud.points, cloud.colors, strict=False):
            handle.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
    return path


def read_ply_ascii(path: str | Path) -> PointCloud:
    ply_path = Path(path)
    lines = ply_path.read_text(encoding="utf-8").splitlines()
    end_header = lines.index("end_header")
    points: list[list[float]] = []
    colors: list[list[int]] = []
    for line in lines[end_header + 1 :]:
        if not line.strip():
            continue
        x, y, z, r, g, b = line.split()[:6]
        points.append([float(x), float(y), float(z)])
        colors.append([int(r), int(g), int(b)])
    return PointCloud(
        points=np.array(points, dtype=np.float64).reshape((-1, 3)),
        colors=np.array(colors, dtype=np.uint8).reshape((-1, 3)),
        frame_ids=[0] * len(points),
    )

