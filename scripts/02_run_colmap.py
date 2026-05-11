from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from video2world.colmap_runner import run_sparse_reconstruction
from video2world.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run COLMAP sparse reconstruction.")
    parser.add_argument("--frames", default=None)
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    run_sparse_reconstruction(
        args.frames or config["input"]["frame_dir"],
        args.workspace or config["colmap"]["workspace"],
        matcher=str(config["colmap"]["matcher"]),
        camera_model=str(config["colmap"]["camera_model"]),
    )


if __name__ == "__main__":
    main()

