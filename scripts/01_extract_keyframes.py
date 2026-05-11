from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from video2world.config import load_config
from video2world.video import extract_keyframes


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract sharp keyframes from a video.")
    parser.add_argument("--video", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    video = args.video or config["input"]["video_path"]
    output = args.output or config["input"]["frame_dir"]
    frames = extract_keyframes(
        video,
        output,
        frame_rate=float(config["preprocessing"]["frame_rate"]),
        max_frames=int(config["preprocessing"]["max_frames"]),
        resize_width=int(config["preprocessing"]["resize_width"]),
        blur_threshold=float(config["preprocessing"]["blur_threshold"]),
    )
    print(f"Extracted {len(frames)} keyframes to {output}")


if __name__ == "__main__":
    main()

