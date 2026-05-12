#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${RUN_DIR:-outputs/demo_room}"
VIDEO="${VIDEO:-data/raw/input_video.mp4}"
CONFIG="${CONFIG:-config/default.yaml}"
PYTHON_BIN="${PYTHON:-python}"

"${PYTHON_BIN}" scripts/00_check_environment.py

"${PYTHON_BIN}" scripts/run_pipeline.py \
  --video "${VIDEO}" \
  --output "${RUN_DIR}" \
  --config "${CONFIG}"

"${PYTHON_BIN}" scripts/10_make_readme_assets.py \
  --run-dir "${RUN_DIR}" \
  --config "${CONFIG}" \
  --skip-mesh

"${PYTHON_BIN}" scripts/12_add_semantics.py \
  --run-dir "${RUN_DIR}"

"${PYTHON_BIN}" scripts/11_evaluate_reconstruction.py \
  --run-dir "${RUN_DIR}"
