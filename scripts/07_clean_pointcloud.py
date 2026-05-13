from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from video2world.cleaning import clean_point_cloud
from video2world.config import load_config
from video2world.fusion import read_ply_ascii, write_ply_ascii


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean a fused point cloud.")
    parser.add_argument("--input", default="outputs/pointclouds/raw_scene.ply")
    parser.add_argument("--output", default="outputs/pointclouds/cleaned_scene.ply")
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    cloud = read_ply_ascii(args.input)
    cleaned = clean_point_cloud(
        cloud,
        voxel_size=float(config["fusion"]["voxel_size"]),
        remove_outliers=bool(config["cleaning"]["remove_outliers"]),
        statistical_nb_neighbors=int(config["cleaning"]["statistical_nb_neighbors"]),
        statistical_std_ratio=float(config["cleaning"]["statistical_std_ratio"]),
        radius_outlier_removal=bool(config["cleaning"]["radius_outlier_removal"]),
        radius=float(config["cleaning"]["radius"]),
        radius_nb_points=int(config["cleaning"]["radius_nb_points"]),
        robust_crop=bool(config["cleaning"].get("robust_crop", True)),
        robust_crop_mad_scale=float(config["cleaning"].get("robust_crop_mad_scale", 12.0)),
        robust_crop_min_keep_ratio=float(config["cleaning"].get("robust_crop_min_keep_ratio", 0.25)),
        adaptive_voxel_min_scene_ratio=float(config["cleaning"].get("adaptive_voxel_min_scene_ratio", 1e-5)),
    )
    path = write_ply_ascii(cleaned, args.output)
    print(f"Wrote {len(cleaned.points)} cleaned points to {path}")


if __name__ == "__main__":
    main()
