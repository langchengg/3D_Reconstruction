from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from video2world.geometry import camera_to_world_from_colmap, qvec_to_rotmat


@dataclass(frozen=True)
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    @property
    def intrinsics(self) -> np.ndarray:
        model = self.model.upper()
        p = self.params.astype(np.float64)
        if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
            fx = fy = p[0]
            cx = p[1]
            cy = p[2]
        elif model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"}:
            fx = p[0]
            fy = p[1]
            cx = p[2]
            cy = p[3]
        else:
            raise ValueError(f"Unsupported COLMAP camera model: {self.model}")
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


@dataclass(frozen=True)
class ImageRecord:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray
    point3d_ids: list[int]

    @property
    def rotation_world_to_camera(self) -> np.ndarray:
        return qvec_to_rotmat(self.qvec)

    @property
    def camera_to_world(self) -> np.ndarray:
        return camera_to_world_from_colmap(self.rotation_world_to_camera, self.tvec)

    def world_to_camera_depth(self, xyz_world: np.ndarray) -> float:
        xyz = np.asarray(xyz_world, dtype=np.float64).reshape(3)
        xyz_camera = self.rotation_world_to_camera @ xyz + self.tvec
        return float(xyz_camera[2])


@dataclass(frozen=True)
class Point3D:
    point3d_id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float


@dataclass(frozen=True)
class ColmapModel:
    cameras: dict[int, Camera]
    images: dict[int, ImageRecord]
    points3d: dict[int, Point3D]

    @property
    def images_by_name(self) -> dict[str, ImageRecord]:
        return {image.name: image for image in self.images.values()}


def _data_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


def read_cameras_text(path: Path) -> dict[int, Camera]:
    cameras: dict[int, Camera] = {}
    for line in _data_lines(path):
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"Malformed camera line: {line}")
        camera_id = int(parts[0])
        cameras[camera_id] = Camera(
            camera_id=camera_id,
            model=parts[1],
            width=int(parts[2]),
            height=int(parts[3]),
            params=np.array([float(value) for value in parts[4:]], dtype=np.float64),
        )
    return cameras


def read_images_text(path: Path) -> dict[int, ImageRecord]:
    lines = _data_lines(path)
    if len(lines) % 2 != 0:
        raise ValueError("COLMAP images.txt should contain pairs of image and points2D lines")

    images: dict[int, ImageRecord] = {}
    for idx in range(0, len(lines), 2):
        meta = lines[idx].split(maxsplit=9)
        if len(meta) != 10:
            raise ValueError(f"Malformed image metadata line: {lines[idx]}")
        image_id = int(meta[0])
        qvec = np.array([float(value) for value in meta[1:5]], dtype=np.float64)
        tvec = np.array([float(value) for value in meta[5:8]], dtype=np.float64)
        camera_id = int(meta[8])
        name = meta[9]

        point_tokens = lines[idx + 1].split()
        if len(point_tokens) % 3 != 0:
            raise ValueError(f"Malformed image points2D line for image {image_id}")
        xys: list[tuple[float, float]] = []
        point3d_ids: list[int] = []
        for offset in range(0, len(point_tokens), 3):
            xys.append((float(point_tokens[offset]), float(point_tokens[offset + 1])))
            point3d_ids.append(int(point_tokens[offset + 2]))

        images[image_id] = ImageRecord(
            image_id=image_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,
            name=name,
            xys=np.array(xys, dtype=np.float64).reshape((-1, 2)),
            point3d_ids=point3d_ids,
        )
    return images


def read_points3d_text(path: Path) -> dict[int, Point3D]:
    points: dict[int, Point3D] = {}
    for line in _data_lines(path):
        parts = line.split()
        if len(parts) < 8:
            raise ValueError(f"Malformed points3D line: {line}")
        point_id = int(parts[0])
        points[point_id] = Point3D(
            point3d_id=point_id,
            xyz=np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64),
            rgb=np.array([int(parts[4]), int(parts[5]), int(parts[6])], dtype=np.uint8),
            error=float(parts[7]),
        )
    return points


def read_colmap_text_model(model_dir: str | Path) -> ColmapModel:
    root = Path(model_dir)
    cameras_path = root / "cameras.txt"
    images_path = root / "images.txt"
    points_path = root / "points3D.txt"
    missing = [str(path) for path in [cameras_path, images_path, points_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing COLMAP text model files: {', '.join(missing)}")
    return ColmapModel(
        cameras=read_cameras_text(cameras_path),
        images=read_images_text(images_path),
        points3d=read_points3d_text(points_path),
    )

