from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "project": {
        "name": "Video2World-Lite",
        "description": "Monocular video to robot-centric 3D world model",
    },
    "input": {
        "video_path": "data/raw/input_video.mp4",
        "frame_dir": "data/frames",
    },
    "preprocessing": {
        "frame_rate": 2.0,
        "max_frames": 60,
        "resize_width": 1024,
        "blur_threshold": 80.0,
    },
    "colmap": {
        "workspace": "outputs/colmap",
        "matcher": "sequential",
        "camera_model": "SIMPLE_PINHOLE",
        "data_type": "VIDEO",
        "quality": "MEDIUM",
    },
    "depth": {
        "model": "depth-anything-v2-small",
        "mode": "depth-anything-v2",
        "encoder": "vits",
        "checkpoint": "",
        "device": "mps",
        "input_size": 518,
        "save_numpy": True,
        "save_visualization": True,
        "heuristic_min_depth": 0.5,
        "heuristic_max_depth": 4.0,
    },
    "scale_alignment": {
        "enabled": True,
        "method": "robust_linear_fit",
        "min_sparse_points_per_frame": 50,
        "trim_quantile": 0.1,
        "invert_prediction": False,
    },
    "fusion": {
        "pixel_stride": 6,
        "min_depth": 0.2,
        "max_depth": 8.0,
        "voxel_size": 0.03,
        "max_points_per_frame": 80000,
    },
    "cleaning": {
        "remove_outliers": True,
        "statistical_nb_neighbors": 20,
        "statistical_std_ratio": 2.0,
        "radius_outlier_removal": True,
        "radius": 0.08,
        "radius_nb_points": 12,
    },
    "world_model": {
        "estimate_floor": True,
        "estimate_obstacles": True,
        "save_json": True,
    },
    "visualization": {
        "save_camera_trajectory": True,
        "save_scene_preview": True,
        "save_demo_video": False,
    },
}


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return deepcopy(DEFAULT_CONFIG)
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    return deep_update(DEFAULT_CONFIG, loaded)

