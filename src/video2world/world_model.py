from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json


DEFAULT_LIMITATIONS = [
    "Monocular video has scale ambiguity.",
    "Reflective, transparent, and textureless surfaces may reduce reconstruction quality.",
    "Dynamic objects are not explicitly handled.",
    "The coordinate frame is inherited from COLMAP and is not gravity-aligned by default.",
]


@dataclass(frozen=True)
class WorldModelSummary:
    scene_id: str
    input_video: str
    num_keyframes: int
    num_registered_frames: int
    num_sparse_points: int
    num_dense_points_raw: int
    num_dense_points_cleaned: int
    scale_aligned_with_colmap: bool
    support_plane_estimated: bool
    obstacle_regions_estimated: bool
    outputs: dict[str, str]
    coordinate_frame: str = "COLMAP_world"
    limitations: list[str] = field(default_factory=lambda: list(DEFAULT_LIMITATIONS))


def world_model_dict(summary: WorldModelSummary) -> dict:
    data = asdict(summary)
    robot_cues = {
        "support_plane_estimated": data.pop("support_plane_estimated"),
        "obstacle_regions_estimated": data.pop("obstacle_regions_estimated"),
        "scale_aligned_with_colmap": data.pop("scale_aligned_with_colmap"),
    }
    data["robot_relevant_cues"] = robot_cues
    return data


def save_world_model(summary: WorldModelSummary, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(world_model_dict(summary), indent=2, sort_keys=False), encoding="utf-8")
    return path
