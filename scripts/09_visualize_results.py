from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from video2world.colmap_io import read_colmap_text_model
from video2world.fusion import read_ply_ascii
from video2world.visualization import save_camera_trajectory, save_pointcloud_preview


def main() -> None:
    parser = argparse.ArgumentParser(description="Save camera trajectory and point cloud preview images.")
    parser.add_argument("--model", default="outputs/colmap/text_model")
    parser.add_argument("--cloud", default="outputs/pointclouds/cleaned_scene.ply")
    parser.add_argument("--trajectory-output", default="outputs/visualizations/camera_trajectory.png")
    parser.add_argument("--scene-output", default="outputs/visualizations/scene_preview.png")
    args = parser.parse_args()

    model = read_colmap_text_model(args.model)
    cloud = read_ply_ascii(args.cloud)
    print(f"Wrote {save_camera_trajectory(model, args.trajectory_output)}")
    print(f"Wrote {save_pointcloud_preview(cloud, args.scene_output)}")


if __name__ == "__main__":
    main()

