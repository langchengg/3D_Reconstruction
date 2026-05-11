from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from video2world.config import load_config
from video2world.pipeline import run_full_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full Video2World-Lite reconstruction pipeline.")
    parser.add_argument("--video", default=None, help="Input video path.")
    parser.add_argument("--output", default="outputs/demo_room", help="Output run directory.")
    parser.add_argument("--config", default="config/default.yaml", help="YAML config path.")
    parser.add_argument(
        "--depth-mode",
        choices=["depth-anything-v2", "heuristic"],
        default=None,
        help="Override depth.mode from config. Use heuristic only for smoke tests.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    video = args.video or config["input"]["video_path"]
    config_path = args.config
    override = {"depth": {"mode": args.depth_mode}} if args.depth_mode is not None else None

    outputs = run_full_pipeline(video, args.output, config_path, config_override=override)
    print("Pipeline outputs:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
