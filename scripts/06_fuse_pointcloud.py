from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from video2world.colmap_io import read_colmap_text_model
from video2world.config import load_config
from video2world.fusion import fuse_rgbd_frames, write_ply_ascii
from video2world.pipeline import load_registered_rgbd_frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse registered RGB-D frames into a point cloud.")
    parser.add_argument("--model", default="outputs/colmap/text_model")
    parser.add_argument("--frames", default="data/frames")
    parser.add_argument("--depth", default="outputs/depth_aligned")
    parser.add_argument("--output", default="outputs/pointclouds/raw_scene.ply")
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    model = read_colmap_text_model(args.model)
    frames = load_registered_rgbd_frames(model, args.frames, args.depth)
    cloud = fuse_rgbd_frames(
        frames,
        pixel_stride=int(config["fusion"]["pixel_stride"]),
        min_depth=float(config["fusion"]["min_depth"]),
        max_depth=float(config["fusion"]["max_depth"]),
        max_points_per_frame=int(config["fusion"]["max_points_per_frame"]),
    )
    path = write_ply_ascii(cloud, args.output)
    print(f"Wrote {len(cloud.points)} points to {path}")


if __name__ == "__main__":
    main()

