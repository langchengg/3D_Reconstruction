from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from video2world.colmap_runner import convert_model_to_text
from video2world.config import load_config
from video2world.pipeline import find_sparse_model_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a COLMAP sparse model to text format.")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    workspace = Path(config["colmap"]["workspace"])
    input_model = Path(args.input) if args.input else find_sparse_model_dir(workspace / "sparse")
    output_model = args.output or workspace / "text_model"
    convert_model_to_text(input_model, output_model)
    print(f"Converted COLMAP model to {output_model}")


if __name__ == "__main__":
    main()

