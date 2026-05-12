#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from video2world.fusion import read_ply_ascii
from video2world.presentation import render_point_cloud_splat
from video2world.semantics import (
    attach_semantics_to_world_model,
    derive_geometry_semantics,
    save_semantic_summary,
    write_semantic_ply_ascii,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add geometry-derived semantic labels to a reconstructed point cloud.")
    parser.add_argument("--run-dir", required=True, help="Pipeline output directory, e.g. outputs/demo_room")
    parser.add_argument("--input", default=None, help="Input PLY. Defaults to presentation_scene.ply, then cleaned_scene.ply.")
    parser.add_argument("--output", default=None, help="Output semantic PLY path")
    parser.add_argument("--summary", default=None, help="Output semantic JSON path")
    parser.add_argument("--preview", default=None, help="Output semantic preview PNG path")
    parser.add_argument("--distance-threshold", type=float, default=0.08)
    parser.add_argument(
        "--obstacle-height",
        type=float,
        default=None,
        help="Signed distance above the support plane for obstacle labels. Defaults to an adaptive point-cloud-scale threshold.",
    )
    parser.add_argument("--ransac-iterations", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    input_path = Path(args.input) if args.input else _default_input(run_dir)
    output_path = Path(args.output) if args.output else run_dir / "pointclouds" / "semantic_scene.ply"
    summary_path = Path(args.summary) if args.summary else run_dir / "world_model" / "semantic_objects.json"
    preview_path = Path(args.preview) if args.preview else run_dir / "visualizations" / "semantic_scene.png"

    cloud = read_ply_ascii(input_path)
    result = derive_geometry_semantics(
        cloud,
        distance_threshold=args.distance_threshold,
        obstacle_height=args.obstacle_height,
        ransac_iterations=args.ransac_iterations,
    )
    write_semantic_ply_ascii(result, output_path)
    save_semantic_summary(result, summary_path)
    render_point_cloud_splat(result.semantic_cloud, preview_path, point_radius=2)
    attach_semantics_to_world_model(
        run_dir / "world_model" / "world_model.json",
        result.summary,
        {
            "semantic_pointcloud": str(output_path),
            "semantic_preview": str(preview_path),
            "semantic_summary": str(summary_path),
        },
    )

    print(f"Wrote semantic point cloud to {output_path}")
    print(f"Wrote semantic summary to {summary_path}")
    print(f"Wrote semantic preview to {preview_path}")
    print(json.dumps(result.summary, indent=2, sort_keys=False))


def _default_input(run_dir: Path) -> Path:
    presentation = run_dir / "pointclouds" / "presentation_scene.ply"
    if presentation.exists():
        return presentation
    return run_dir / "pointclouds" / "cleaned_scene.ply"


if __name__ == "__main__":
    main()
