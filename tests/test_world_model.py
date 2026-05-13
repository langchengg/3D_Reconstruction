import json
from pathlib import Path

from video2world.world_model import WorldModelSummary, save_world_model


def test_save_world_model_writes_robot_relevant_contract(tmp_path: Path):
    summary = WorldModelSummary(
        scene_id="demo",
        input_video="data/raw/input_video.mp4",
        num_keyframes=4,
        num_registered_frames=3,
        num_sparse_points=10,
        num_dense_points_raw=100,
        num_dense_points_cleaned=70,
        scale_aligned_with_colmap=True,
        support_plane_estimated=False,
        obstacle_regions_estimated=True,
        outputs={"cleaned_pointcloud": "outputs/pointclouds/cleaned_scene.ply"},
    )

    path = save_world_model(summary, tmp_path / "world_model.json")

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["scene_id"] == "demo"
    assert data["robot_relevant_cues"]["scale_aligned_with_colmap"] is True
    assert data["outputs"]["cleaned_pointcloud"] == "outputs/pointclouds/cleaned_scene.ply"
    assert "Monocular video has scale ambiguity." in data["limitations"]
