#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from video2world.evaluation import build_evaluation_report, save_evaluation_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Video2World-Lite reconstruction run.")
    parser.add_argument("--run-dir", required=True, help="Pipeline output directory, e.g. outputs/demo_room")
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = save_evaluation_report(args.run_dir, args.output)
    report = build_evaluation_report(args.run_dir)
    print(f"Wrote evaluation report to {output_path}")
    print(json.dumps(report, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
