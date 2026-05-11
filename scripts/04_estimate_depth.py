from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from video2world.config import load_config
from video2world.depth import estimate_depth_for_frames
from video2world.pipeline import list_frame_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate dense depth maps for extracted frames.")
    parser.add_argument("--frames", default=None)
    parser.add_argument("--output", default="outputs/depth")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--depth-mode", choices=["depth-anything-v2", "heuristic"], default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    depth_config = dict(config["depth"])
    if args.depth_mode:
        depth_config["mode"] = args.depth_mode
    frame_dir = Path(args.frames or config["input"]["frame_dir"])
    predictions = estimate_depth_for_frames(list_frame_paths(frame_dir), args.output, depth_config)
    print(f"Estimated {len(predictions)} depth maps into {args.output}")


if __name__ == "__main__":
    main()

