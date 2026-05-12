# Video2World-Lite

Monocular phone video to a robot-centric 3D world model.

This project is built for the Humanoid intern challenge, "From Video to 3D Reconstruction". It combines classical geometry with pretrained monocular depth:

```text
Phone video
-> sharp keyframes
-> COLMAP camera poses and sparse points
-> Depth Anything V2 dense depth
-> sparse-geometry depth scale alignment
-> multi-view point cloud fusion
-> cleaned robot-centric world_model.json
```

The core output is not only a visual point cloud. The pipeline exports camera trajectory, sparse/dense geometry, cleaned scene points, and a structured world model that can be consumed by downstream robot navigation or scene understanding code.

## Example Results

The sample below was generated from a short indoor phone video with lateral camera motion.

| Camera-view reconstruction | Oblique point-cloud reconstruction |
| --- | --- |
| ![Camera-view reconstruction](assets/example_outputs/hero_camera_view.png) | ![Oblique point-cloud reconstruction](assets/example_outputs/hero_scene.png) |

| Depth predictions | COLMAP camera trajectory |
| --- | --- |
| ![Depth prediction grid](assets/example_outputs/depth_grid.png) | ![Camera trajectory](assets/example_outputs/camera_trajectory.png) |

Sample run statistics:

- 22 extracted keyframes
- 22 COLMAP-registered frames
- 9,382 sparse SfM points
- 1,143,648 fused dense points before cleaning
- 1,101,661 cleaned dense points
- 909,540 cropped presentation points

## Quick Start

Install Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install external tools:

- COLMAP must be available as `colmap` on `PATH`.
- Depth Anything V2 must be available from the official repository.
- The default config expects the official repo at `third_party/Depth-Anything-V2` and the small checkpoint at `third_party/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth`.

Check the environment:

```bash
python scripts/00_check_environment.py
```

Place a short indoor phone video at:

```text
data/raw/input_video.mp4
```

Run the full pipeline:

```bash
python scripts/run_pipeline.py \
  --video data/raw/input_video.mp4 \
  --output outputs/demo_room \
  --config config/default.yaml
```

For a dependency-light smoke run of the depth stage only, use the deterministic fallback:

```bash
python scripts/04_estimate_depth.py \
  --frames data/frames \
  --output outputs/depth_smoke \
  --depth-mode heuristic
```

The heuristic mode is not the intended reconstruction method. It exists so the code path can be tested without model weights.

## Step-by-Step Usage

```bash
python scripts/01_extract_keyframes.py --video data/raw/input_video.mp4 --output data/frames
python scripts/02_run_colmap.py --frames data/frames --workspace outputs/colmap
python scripts/03_convert_colmap_model.py --input outputs/colmap/sparse/0 --output outputs/colmap/text_model
python scripts/04_estimate_depth.py --frames data/frames --output outputs/depth
python scripts/05_align_depth_scale.py --model outputs/colmap/text_model --depth outputs/depth --output outputs/depth_aligned
python scripts/06_fuse_pointcloud.py --model outputs/colmap/text_model --frames data/frames --depth outputs/depth_aligned --output outputs/pointclouds/raw_scene.ply
python scripts/07_clean_pointcloud.py --input outputs/pointclouds/raw_scene.ply --output outputs/pointclouds/cleaned_scene.ply
python scripts/08_build_world_model.py --scale-aligned
python scripts/09_visualize_results.py
```

## Outputs

For a run directory such as `outputs/demo_room`, the main files are:

```text
frames/
colmap/text_model/cameras.txt
colmap/text_model/images.txt
colmap/text_model/points3D.txt
depth/*.npy
depth_aligned/*.npy
pointclouds/raw_scene.ply
pointclouds/cleaned_scene.ply
pointclouds/presentation_scene.ply
visualizations/camera_trajectory.png
visualizations/scene_preview.png
visualizations/hero_camera_view.png
visualizations/hero_scene.png
visualizations/depth_grid.png
meshes/scene_mesh.ply
world_model/world_model.json
```

`world_model.json` includes:

- input video and scene id
- number of keyframes and registered frames
- sparse and dense point counts
- output artifact paths
- whether dense depth was aligned to COLMAP sparse geometry
- robot-relevant cues and known limitations

## Method

1. **Keyframe extraction**: sample a compact set of frames, filter blurry images using Laplacian variance, and resize to a controlled width.
2. **COLMAP SfM**: use sequential matching because the input is a video. COLMAP estimates camera poses and sparse 3D points.
3. **Dense depth**: run Depth Anything V2 per registered frame.
4. **Scale alignment**: monocular depth is ambiguous up to scale. The pipeline fits a global robust linear alignment from Depth Anything predictions to COLMAP sparse point depths, with an automatic direct-vs-inverse depth direction check.
5. **Fusion**: sample pixels, unproject with camera intrinsics and aligned depth, transform into COLMAP world coordinates, and write a colored PLY.
6. **Cleaning**: voxel downsample and remove outliers with Open3D when available.
7. **Presentation crop and rendering**: crop the dense cloud around robust COLMAP sparse bounds, then render README-grade camera-view and oblique point-splat previews.
8. **World model export**: package geometry, trajectory, statistics, and robot-facing cues.

To regenerate the README-grade images from an existing run:

```bash
python scripts/10_make_readme_assets.py \
  --run-dir outputs/demo_room \
  --config config/default.yaml
```

## Design Choices

- **COLMAP for poses**: reliable classical SfM gives camera trajectory and sparse anchors instead of relying entirely on a learned model.
- **Depth Anything V2 for density**: pretrained dense depth fills surfaces that sparse SfM cannot represent.
- **Global scale alignment before fusion**: predicted monocular depth is not treated as metric truth. Sparse COLMAP geometry provides global geometric anchors, and the code can automatically choose direct or inverse depth prediction direction.
- **Permissive demo alignment**: `min_sparse_points_per_frame` defaults to `3` so short videos can exercise the alignment path; increase it for stricter experiments.
- **COLMAP-scale fusion**: `fusion.max_depth` is intentionally wide because aligned depth is in COLMAP's arbitrary reconstruction scale, not guaranteed meters.
- **README rendering path**: Open3D is used for point-cloud processing and mesh generation. For reproducible GitHub screenshots on macOS headless environments, the repo also includes a deterministic software point-splat renderer.
- **Point cloud first**: PLY is simple, inspectable, and compatible with common robotics/3D tooling.
- **Mesh is optional**: Poisson mesh reconstruction is generated as an experimental artifact, but the main reconstruction deliverable remains the aligned dense point cloud.
- **No training required**: the project is designed to run on consumer hardware, including Apple Silicon, without fine-tuning.

## Failure Cases

- Textureless white walls can reduce COLMAP registration quality.
- Motion blur and fast camera motion can break feature matching.
- Reflective or transparent surfaces can produce unstable depth.
- Dynamic people or moving objects are not explicitly filtered.
- COLMAP world coordinates are not automatically gravity-aligned.

## Tests

The deterministic tests cover the core contracts that do not need external binaries or model weights:

```bash
python -m pytest
```

Covered behavior includes COLMAP text parsing, quaternion pose conversion, depth scale alignment, RGB-D fusion, PLY writing, and world model JSON export.

## Future Work

- Project 2D segmentation masks into 3D for semantic point clouds.
- Estimate a gravity-aligned floor plane and navigable free-space regions.
- Add a browser-based viewer for inspecting camera trajectory and point cloud artifacts.
- Add dynamic object filtering before fusion.
