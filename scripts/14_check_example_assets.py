#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from video2world.example_assets import check_example_assets


def main() -> None:
    parser = argparse.ArgumentParser(description="Check packaged README/example outputs without external model weights.")
    parser.add_argument("--example-dir", default="assets/example_outputs")
    args = parser.parse_args()

    result = check_example_assets(args.example_dir)
    print(json.dumps(result, indent=2, sort_keys=False))
    if not result["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
