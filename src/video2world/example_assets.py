from __future__ import annotations

from pathlib import Path
import json


REQUIRED_EXAMPLE_ASSETS = [
    "hero_camera_view.png",
    "hero_scene.png",
    "depth_grid.png",
    "camera_trajectory.png",
    "semantic_scene.png",
    "evaluation_report.json",
    "world_model.json",
    "semantic_objects.json",
]


def check_example_assets(example_dir: str | Path) -> dict:
    root = Path(example_dir)
    missing = [name for name in REQUIRED_EXAMPLE_ASSETS if not (root / name).exists()]
    metrics = {}
    evaluation_path = root / "evaluation_report.json"
    if evaluation_path.exists():
        metrics = json.loads(evaluation_path.read_text(encoding="utf-8"))
    return {
        "example_dir": str(root),
        "ready": len(missing) == 0,
        "missing": missing,
        "metrics": metrics,
    }
