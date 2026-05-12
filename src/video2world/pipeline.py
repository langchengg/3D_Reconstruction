from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from video2world.cleaning import clean_point_cloud
from video2world.colmap_io import ColmapModel, read_colmap_text_model
from video2world.colmap_runner import convert_model_to_text, run_sparse_reconstruction
from video2world.config import deep_update, load_config
from video2world.depth import estimate_depth_for_frames, save_depth_preview
from video2world.fusion import PointCloud, fuse_rgbd_frames, read_ply_ascii, write_ply_ascii
from video2world.scale_alignment import align_depth_map, collect_alignment_samples, fit_scale_shift
from video2world.video import extract_keyframes
from video2world.visualization import save_camera_trajectory, save_pointcloud_preview
from video2world.world_model import WorldModelSummary, save_world_model


def find_sparse_model_dir(sparse_root: str | Path) -> Path:
    root = Path(sparse_root)
    if (root / "cameras.bin").exists() or (root / "cameras.txt").exists():
        return root
    candidates = sorted([path for path in root.iterdir() if _looks_like_sparse_model(path)]) if root.exists() else []
    if not candidates:
        raise FileNotFoundError(f"No COLMAP sparse model directory found under {root}")
    return max(candidates, key=_sparse_model_score)


def _looks_like_sparse_model(path: Path) -> bool:
    return path.is_dir() and ((path / "cameras.bin").exists() or (path / "cameras.txt").exists())


def _sparse_model_score(path: Path) -> tuple[int, int, int]:
    """Prefer reconstructions with more registered image data and more sparse structure."""
    images_size = _file_size(path / "images.bin") + _file_size(path / "images.txt")
    points_size = _file_size(path / "points3D.bin") + _file_size(path / "points3D.txt")
    return (images_size, points_size, -len(path.name))


def _file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def list_frame_paths(frame_dir: str | Path) -> list[Path]:
    root = Path(frame_dir)
    frames = sorted(list(root.glob("*.jpg")) + list(root.glob("*.png")) + list(root.glob("*.jpeg")))
    if not frames:
        raise FileNotFoundError(f"No image frames found in {root}")
    return frames


def align_depth_directory(
    model: ColmapModel,
    depth_dir: str | Path,
    output_dir: str | Path,
    config: dict,
) -> dict[str, dict]:
    depth_root = Path(depth_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict] = {}
    min_points = int(config.get("min_sparse_points_per_frame", 50))
    trim_quantile = float(config.get("trim_quantile", 0.1))
    invert_prediction = bool(config.get("invert_prediction", False))

    for image in model.images.values():
        depth_path = depth_root / f"{Path(image.name).stem}.npy"
        if not depth_path.exists():
            continue
        depth = np.load(depth_path)
        predicted, sparse = collect_alignment_samples(
            image,
            model.points3d,
            depth,
            invert_prediction=invert_prediction,
        )
        result = fit_scale_shift(predicted, sparse, min_points=min_points, trim_quantile=trim_quantile)
        if result.success:
            aligned = align_depth_map(depth, result)
        else:
            aligned = depth.astype(np.float32)
        np.save(output / depth_path.name, aligned)
        save_depth_preview(aligned, output / f"{depth_path.stem}.png")
        report[Path(image.name).name] = {
            "success": result.success,
            "scale": result.scale,
            "shift": result.shift,
            "num_inliers": result.num_inliers,
            "num_sparse_samples": int(len(predicted)),
            "reason": result.reason,
        }

    (output / "alignment_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_registered_rgbd_frames(
    model: ColmapModel,
    frame_dir: str | Path,
    depth_dir: str | Path,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    import cv2

    frame_root = Path(frame_dir)
    depth_root = Path(depth_dir)
    frames = []
    for image in sorted(model.images.values(), key=lambda item: item.image_id):
        image_name = Path(image.name).name
        frame_path = frame_root / image_name
        depth_path = depth_root / f"{Path(image_name).stem}.npy"
        if not frame_path.exists() or not depth_path.exists():
            continue
        rgb_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            continue
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        depth = np.load(depth_path)
        camera = model.cameras[image.camera_id]
        frames.append((image_name, rgb, depth, camera.intrinsics, image.camera_to_world))
    if not frames:
        raise RuntimeError("No registered RGB-D frames could be matched between COLMAP, frames, and depth outputs.")
    return frames


def count_ply_vertices(path: str | Path) -> int:
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("element vertex "):
            return int(line.split()[-1])
    return 0


def run_full_pipeline(
    video_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    config_override: dict | None = None,
) -> dict[str, Path]:
    config = load_config(config_path)
    if config_override:
        config = deep_update(config, config_override)
    output = Path(output_dir)
    frames_dir = output / "frames"
    colmap_dir = output / "colmap"
    text_model_dir = colmap_dir / "text_model"
    depth_dir = output / "depth"
    aligned_depth_dir = output / "depth_aligned"
    pointcloud_dir = output / "pointclouds"
    vis_dir = output / "visualizations"
    world_dir = output / "world_model"

    extracted = extract_keyframes(
        video_path,
        frames_dir,
        frame_rate=float(config["preprocessing"]["frame_rate"]),
        max_frames=int(config["preprocessing"]["max_frames"]),
        resize_width=int(config["preprocessing"]["resize_width"]),
        blur_threshold=float(config["preprocessing"]["blur_threshold"]),
    )

    run_sparse_reconstruction(
        frames_dir,
        colmap_dir,
        matcher=str(config["colmap"]["matcher"]),
        camera_model=str(config["colmap"]["camera_model"]),
    )
    sparse_model = find_sparse_model_dir(colmap_dir / "sparse")
    convert_model_to_text(sparse_model, text_model_dir)
    model = read_colmap_text_model(text_model_dir)

    estimate_depth_for_frames([frame.path for frame in extracted], depth_dir, config["depth"])
    active_depth_dir = depth_dir
    scale_aligned = False
    if bool(config["scale_alignment"]["enabled"]):
        report = align_depth_directory(model, depth_dir, aligned_depth_dir, config["scale_alignment"])
        scale_aligned = any(item.get("success") for item in report.values())
        active_depth_dir = aligned_depth_dir

    rgbd_frames = load_registered_rgbd_frames(model, frames_dir, active_depth_dir)
    raw_cloud = fuse_rgbd_frames(
        rgbd_frames,
        pixel_stride=int(config["fusion"]["pixel_stride"]),
        min_depth=float(config["fusion"]["min_depth"]),
        max_depth=float(config["fusion"]["max_depth"]),
        max_points_per_frame=int(config["fusion"]["max_points_per_frame"]),
    )
    raw_path = write_ply_ascii(raw_cloud, pointcloud_dir / "raw_scene.ply")

    cleaned = clean_point_cloud(
        raw_cloud,
        voxel_size=float(config["fusion"]["voxel_size"]),
        remove_outliers=bool(config["cleaning"]["remove_outliers"]),
        statistical_nb_neighbors=int(config["cleaning"]["statistical_nb_neighbors"]),
        statistical_std_ratio=float(config["cleaning"]["statistical_std_ratio"]),
        radius_outlier_removal=bool(config["cleaning"]["radius_outlier_removal"]),
        radius=float(config["cleaning"]["radius"]),
        radius_nb_points=int(config["cleaning"]["radius_nb_points"]),
    )
    cleaned_path = write_ply_ascii(cleaned, pointcloud_dir / "cleaned_scene.ply")

    trajectory_path = save_camera_trajectory(model, vis_dir / "camera_trajectory.png")
    scene_preview_path = save_pointcloud_preview(cleaned, vis_dir / "scene_preview.png")

    summary = WorldModelSummary(
        scene_id=Path(video_path).stem,
        input_video=str(video_path),
        num_keyframes=len(extracted),
        num_registered_frames=len(model.images),
        num_sparse_points=len(model.points3d),
        num_dense_points_raw=len(raw_cloud.points),
        num_dense_points_cleaned=len(cleaned.points),
        scale_aligned_with_colmap=scale_aligned,
        floor_estimated=False,
        obstacle_regions_estimated=True,
        outputs={
            "raw_pointcloud": str(raw_path),
            "cleaned_pointcloud": str(cleaned_path),
            "camera_trajectory": str(trajectory_path),
            "scene_preview": str(scene_preview_path),
        },
    )
    world_model_path = save_world_model(summary, world_dir / "world_model.json")
    return {
        "raw_pointcloud": raw_path,
        "cleaned_pointcloud": cleaned_path,
        "camera_trajectory": trajectory_path,
        "scene_preview": scene_preview_path,
        "world_model": world_model_path,
    }


def build_world_model_from_outputs(
    *,
    scene_id: str,
    input_video: str,
    frame_dir: str | Path,
    text_model_dir: str | Path,
    raw_ply: str | Path,
    cleaned_ply: str | Path,
    output_json: str | Path,
    scale_aligned: bool,
) -> Path:
    model = read_colmap_text_model(text_model_dir)
    frame_count = len(list_frame_paths(frame_dir))
    summary = WorldModelSummary(
        scene_id=scene_id,
        input_video=input_video,
        num_keyframes=frame_count,
        num_registered_frames=len(model.images),
        num_sparse_points=len(model.points3d),
        num_dense_points_raw=count_ply_vertices(raw_ply),
        num_dense_points_cleaned=count_ply_vertices(cleaned_ply),
        scale_aligned_with_colmap=scale_aligned,
        floor_estimated=False,
        obstacle_regions_estimated=True,
        outputs={
            "raw_pointcloud": str(raw_ply),
            "cleaned_pointcloud": str(cleaned_ply),
        },
    )
    return save_world_model(summary, output_json)
