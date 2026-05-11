from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from video2world.pipeline import build_world_model_from_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Export structured robot-centric world_model.json.")
    parser.add_argument("--scene-id", default="small_room_demo")
    parser.add_argument("--input-video", default="data/raw/input_video.mp4")
    parser.add_argument("--frames", default="data/frames")
    parser.add_argument("--model", default="outputs/colmap/text_model")
    parser.add_argument("--raw", default="outputs/pointclouds/raw_scene.ply")
    parser.add_argument("--cleaned", default="outputs/pointclouds/cleaned_scene.ply")
    parser.add_argument("--output", default="outputs/world_model/world_model.json")
    parser.add_argument("--scale-aligned", action="store_true")
    args = parser.parse_args()

    path = build_world_model_from_outputs(
        scene_id=args.scene_id,
        input_video=args.input_video,
        frame_dir=args.frames,
        text_model_dir=args.model,
        raw_ply=args.raw,
        cleaned_ply=args.cleaned,
        output_json=args.output,
        scale_aligned=args.scale_aligned,
    )
    print(f"Wrote world model to {Path(path)}")


if __name__ == "__main__":
    main()

