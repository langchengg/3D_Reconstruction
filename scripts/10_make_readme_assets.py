from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from video2world.colmap_io import read_colmap_text_model
from video2world.config import load_config
from video2world.fusion import read_ply_ascii
from video2world.presentation import (
    create_poisson_mesh_from_cloud,
    crop_cloud_to_anchor_bounds,
    render_point_cloud_splat,
    render_point_cloud_camera_view,
    sample_mesh_preview_cloud,
    save_depth_grid,
    save_presentation_cloud,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate README-grade presentation assets from an existing run.")
    parser.add_argument("--run-dir", default="outputs/img9574_full")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--skip-mesh", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config = load_config(args.config)
    model = read_colmap_text_model(run_dir / "colmap" / "text_model")
    cleaned = read_ply_ascii(run_dir / "pointclouds" / "cleaned_scene.ply")
    anchors = np.array([point.xyz for point in model.points3d.values()], dtype=np.float64).reshape((-1, 3))

    presentation_cloud, crop_report = crop_cloud_to_anchor_bounds(
        cleaned,
        anchors,
        lower_quantile=float(config["presentation"]["lower_quantile"]),
        upper_quantile=float(config["presentation"]["upper_quantile"]),
        margin_ratio=float(config["presentation"]["margin_ratio"]),
        min_margin=float(config["presentation"]["min_margin"]),
    )

    pointcloud_dir = run_dir / "pointclouds"
    vis_dir = run_dir / "visualizations"
    mesh_dir = run_dir / "meshes"
    save_presentation_cloud(presentation_cloud, pointcloud_dir / "presentation_scene.ply")
    render_point_cloud_splat(
        presentation_cloud,
        vis_dir / "hero_scene.png",
        width=int(config["presentation"]["render_width"]),
        height=int(config["presentation"]["render_height"]),
        max_points=int(config["presentation"]["render_max_points"]),
        point_radius=int(config["presentation"]["render_point_radius"]),
    )
    camera_view_image = sorted(model.images.values(), key=lambda item: item.image_id)[len(model.images) // 2]
    camera_view_camera = model.cameras[camera_view_image.camera_id]
    render_point_cloud_camera_view(
        presentation_cloud,
        camera_view_camera.intrinsics,
        camera_view_image.rotation_world_to_camera,
        camera_view_image.tvec,
        vis_dir / "hero_camera_view.png",
        width=camera_view_camera.width,
        height=camera_view_camera.height,
        max_points=int(config["presentation"]["render_max_points"]),
        point_radius=int(config["presentation"]["render_point_radius"]),
    )

    registered = [run_dir / "frames" / Path(image.name).name for image in sorted(model.images.values(), key=lambda item: item.image_id)]
    depth_previews = [run_dir / "depth_aligned" / f"{path.stem}.png" for path in registered]
    save_depth_grid(
        registered,
        depth_previews,
        vis_dir / "depth_grid.png",
        columns=int(config["presentation"]["depth_grid_columns"]),
        max_items=int(config["presentation"]["depth_grid_max_items"]),
    )
    vis_dir.mkdir(parents=True, exist_ok=True)
    (vis_dir / "presentation_report.json").write_text(json.dumps(crop_report, indent=2), encoding="utf-8")

    if bool(config.get("mesh", {}).get("enabled", False)) and not args.skip_mesh:
        mesh_path, mesh_report = create_poisson_mesh_from_cloud(
            presentation_cloud,
            mesh_dir / "scene_mesh.ply",
            max_points=int(config["mesh"]["max_points"]),
            poisson_depth=int(config["mesh"]["poisson_depth"]),
            density_quantile=float(config["mesh"]["density_quantile"]),
        )
        mesh_dir.mkdir(parents=True, exist_ok=True)
        (mesh_dir / "mesh_report.json").write_text(json.dumps(mesh_report, indent=2), encoding="utf-8")
        if mesh_path is not None:
            mesh_preview_cloud = sample_mesh_preview_cloud(mesh_path)
            render_point_cloud_splat(mesh_preview_cloud, vis_dir / "mesh_preview.png")

    print(f"Wrote README assets under {run_dir}")


if __name__ == "__main__":
    main()
