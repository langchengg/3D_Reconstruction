from __future__ import annotations

import json

from video2world.example_assets import check_example_assets


def test_check_example_assets_reports_missing_files(tmp_path):
    result = check_example_assets(tmp_path)

    assert "hero_camera_view.png" in result["missing"]
    assert result["ready"] is False


def test_check_example_assets_reads_metrics_when_present(tmp_path):
    for name in [
        "hero_camera_view.png",
        "hero_scene.png",
        "depth_grid.png",
        "camera_trajectory.png",
        "semantic_scene.png",
        "world_model.json",
        "semantic_objects.json",
    ]:
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "evaluation_report.json").write_text(
        json.dumps({"registration_ratio": 1.0, "semantic_coverage": 0.25}),
        encoding="utf-8",
    )

    result = check_example_assets(tmp_path)

    assert result["ready"] is True
    assert result["missing"] == []
    assert result["metrics"]["registration_ratio"] == 1.0
    assert result["metrics"]["semantic_coverage"] == 0.25
