# Video2World-Lite Design

## Goal

Build a lightweight engineering project for the Humanoid intern challenge: convert a short monocular indoor video into a robot-centric 3D world model. The project should be runnable from the command line, readable as a GitHub submission, and honest about monocular reconstruction limits.

## Architecture

The system follows the Route A pipeline supplied by the user:

1. Extract sharp keyframes from a phone video.
2. Estimate camera intrinsics, poses, and sparse geometry with COLMAP.
3. Predict dense monocular depth for each registered frame.
4. Align dense depth to COLMAP sparse geometry to reduce scale ambiguity.
5. Fuse RGB-D observations into a global point cloud.
6. Clean the point cloud and export a structured world model.
7. Save visual outputs for the README.

The implementation keeps each step isolated in `src/video2world/`, with thin scripts in `scripts/` so users can run the full pipeline or individual stages.

## Components

- `video.py`: frame extraction, blur scoring, resizing.
- `colmap_runner.py`: subprocess wrappers for COLMAP CLI stages.
- `colmap_io.py`: COLMAP text model parsing and camera pose utilities.
- `depth.py`: Depth Anything V2 adapter plus deterministic fallback for tests and smoke runs.
- `scale_alignment.py`: robust linear scale/shift fitting from COLMAP sparse point observations.
- `geometry.py`: pixel unprojection, camera/world transforms, and sampling helpers.
- `fusion.py`: multi-view colored point cloud fusion.
- `cleaning.py`: Open3D-based cleaning with a NumPy fallback.
- `world_model.py`: JSON export containing statistics, outputs, limitations, and robot-relevant cues.
- `visualization.py`: trajectory, depth preview, and point cloud preview images.

## Data Flow

`input_video.mp4` is converted into `data/frames/*.jpg`. COLMAP reads those frames and writes a sparse model under `outputs/colmap/sparse`, which is converted to text. The text model is parsed into camera intrinsics, registered image poses, and sparse points. Depth inference writes `outputs/depth/*.npy` and preview PNGs. Alignment uses each image's COLMAP 2D-to-3D observations to fit `z_aligned = a * z_pred + b`. Fusion unprojects sampled pixels with aligned depth and transforms them into COLMAP world coordinates, producing `raw_scene.ply`, `cleaned_scene.ply`, visual previews, and `world_model.json`.

## Error Handling

External tools are checked explicitly. Missing COLMAP, Depth Anything V2, PyTorch, or Open3D produce actionable messages instead of silent fallback in the real pipeline. The heuristic depth mode exists only for local smoke tests and demonstrations where model weights are unavailable. Frame extraction rejects unreadable videos and fails if no sufficiently sharp frames are found.

## Testing

Tests focus on deterministic core behavior rather than external binaries or model weights:

- COLMAP text parsing and quaternion pose conversion.
- Robust depth scale/shift fitting.
- Pixel unprojection and camera-to-world transforms.
- Point cloud fusion from small synthetic RGB-D inputs.
- World model JSON contract.

## Scope Boundaries

The first version does not train models, run real-time reconstruction, or implement semantic segmentation. Semantic projection is documented as future work because the challenge prioritizes geometric reconstruction.
