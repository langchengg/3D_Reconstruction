from __future__ import annotations

import json

import numpy as np

from video2world.evaluation import build_evaluation_report, summarize_alignment_residuals


def test_build_evaluation_report_summarizes_run_artifacts(tmp_path):
    run_dir = tmp_path / "demo"
    (run_dir / "world_model").mkdir(parents=True)
    (run_dir / "depth_aligned").mkdir()
    (run_dir / "visualizations").mkdir()
    (run_dir / "pointclouds").mkdir()

    (run_dir / "world_model" / "world_model.json").write_text(
        json.dumps(
            {
                "num_keyframes": 4,
                "num_registered_frames": 3,
                "num_sparse_points": 12,
                "num_dense_points_raw": 100,
                "num_dense_points_cleaned": 75,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "depth_aligned" / "alignment_report.json").write_text(
        json.dumps(
            {
                "frame_000001.jpg": {"success": True, "median_residual": 0.10},
                "frame_000002.jpg": {"success": False, "reason": "too few samples"},
                "frame_000003.jpg": {"success": True, "median_residual": 0.30},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "visualizations" / "camera_trajectory.png").write_bytes(b"png")
    (run_dir / "pointclouds" / "cleaned_scene.ply").write_text("ply\nend_header\n", encoding="utf-8")
    (run_dir / "pointclouds" / "semantic_scene.ply").write_text("ply\nend_header\n", encoding="utf-8")
    (run_dir / "world_model" / "semantic_objects.json").write_text(
        json.dumps({"semantic_coverage": 0.6, "num_labeled_points": 45}),
        encoding="utf-8",
    )

    report = build_evaluation_report(run_dir)

    assert report["registration_ratio"] == 0.75
    assert report["alignment_success_rate"] == 2 / 3
    assert report["median_alignment_residual"] == 0.20
    assert report["outlier_removed_ratio"] == 0.25
    assert report["has_camera_trajectory"] is True
    assert report["has_cleaned_pointcloud"] is True
    assert report["has_semantic_scene"] is True
    assert report["has_world_model_json"] is True
    assert report["semantic_coverage"] == 0.6
    assert report["num_semantic_labeled_points"] == 45


def test_summarize_alignment_residuals_reports_relative_and_colmap_scale_values():
    predicted = np.array([9.0, 22.0], dtype=np.float64)
    sparse = np.array([10.0, 20.0], dtype=np.float64)

    stats = summarize_alignment_residuals(predicted, sparse)

    assert stats["median_alignment_residual"] == 0.1
    assert stats["median_alignment_residual_colmap_scale"] == 1.5
