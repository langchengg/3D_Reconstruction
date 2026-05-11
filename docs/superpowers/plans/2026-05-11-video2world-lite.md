# Video2World-Lite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable monocular video-to-3D reconstruction codebase using COLMAP, Depth Anything V2, depth scale alignment, Open3D fusion/cleaning, and robot-centric world model export.

**Architecture:** The project is a Python package under `src/video2world` with thin stage scripts under `scripts`. External systems are isolated behind wrappers, while geometry and file-format logic are tested with deterministic synthetic data.

**Tech Stack:** Python 3.10+, NumPy, OpenCV, PyYAML, Matplotlib, Open3D, PyTorch/Depth Anything V2 adapter, COLMAP CLI.

---

### Task 1: Project Skeleton and Configuration

**Files:**
- Create: `README.md`
- Create: `requirements.txt`
- Create: `environment.yml`
- Create: `config/default.yaml`
- Create: `src/video2world/__init__.py`
- Create: `src/video2world/config.py`

- [x] Write configuration dataclasses with defaults matching Route A.
- [x] Add YAML loading with recursive overrides.
- [x] Add README quick start, output contract, design choices, and limitations.

### Task 2: Core Geometry and COLMAP Parsing

**Files:**
- Create: `src/video2world/geometry.py`
- Create: `src/video2world/colmap_io.py`
- Test: `tests/test_colmap_io.py`
- Test: `tests/test_geometry.py`

- [x] Parse `cameras.txt`, `images.txt`, and `points3D.txt`.
- [x] Convert COLMAP quaternion/tvec camera poses into camera-to-world transforms.
- [x] Implement unprojection and rigid transform helpers.
- [x] Verify with focused synthetic tests.

### Task 3: Video, Depth, and Alignment

**Files:**
- Create: `src/video2world/video.py`
- Create: `src/video2world/depth.py`
- Create: `src/video2world/scale_alignment.py`
- Test: `tests/test_scale_alignment.py`

- [x] Extract sharp keyframes with blur filtering and resizing.
- [x] Add Depth Anything V2 adapter with explicit dependency errors.
- [x] Add deterministic heuristic depth mode for tests and local smoke runs.
- [x] Fit robust scale and shift from sparse COLMAP observations.

### Task 4: Fusion, Cleaning, World Model, Visualization

**Files:**
- Create: `src/video2world/fusion.py`
- Create: `src/video2world/cleaning.py`
- Create: `src/video2world/world_model.py`
- Create: `src/video2world/visualization.py`
- Test: `tests/test_fusion.py`
- Test: `tests/test_world_model.py`

- [x] Fuse sampled RGB-D frames into a colored global point cloud.
- [x] Save ASCII PLY without requiring Open3D.
- [x] Use Open3D cleaning when available and NumPy voxel downsampling otherwise.
- [x] Export a structured `world_model.json`.
- [x] Generate trajectory, depth, and point cloud preview images.

### Task 5: Stage Scripts and Verification

**Files:**
- Create: `scripts/00_check_environment.py`
- Create: `scripts/01_extract_keyframes.py`
- Create: `scripts/02_run_colmap.py`
- Create: `scripts/03_convert_colmap_model.py`
- Create: `scripts/04_estimate_depth.py`
- Create: `scripts/05_align_depth_scale.py`
- Create: `scripts/06_fuse_pointcloud.py`
- Create: `scripts/07_clean_pointcloud.py`
- Create: `scripts/08_build_world_model.py`
- Create: `scripts/09_visualize_results.py`
- Create: `scripts/run_pipeline.py`

- [x] Add single-stage CLIs for debugging and demonstration.
- [x] Add full pipeline CLI with clear failure modes.
- [x] Run `python -m pytest`.
- [x] Run environment check.

### Self-Review

- No placeholder behavior is required to understand or run the project.
- External dependencies are explicit and isolated.
- Tests do not require COLMAP, model weights, or GPU hardware.
- The README states the difference between real Depth Anything V2 inference and heuristic smoke mode.
