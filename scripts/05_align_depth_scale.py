from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from video2world.colmap_io import read_colmap_text_model
from video2world.config import load_config
from video2world.pipeline import align_depth_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Align monocular depth maps to COLMAP sparse geometry.")
    parser.add_argument("--model", default="outputs/colmap/text_model")
    parser.add_argument("--depth", default="outputs/depth")
    parser.add_argument("--output", default="outputs/depth_aligned")
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    model = read_colmap_text_model(args.model)
    report = align_depth_directory(model, args.depth, args.output, config["scale_alignment"])
    successes = sum(1 for item in report.values() if item["success"])
    print(f"Aligned {successes}/{len(report)} depth maps into {args.output}")


if __name__ == "__main__":
    main()

