# Design Choices

## Geometry-First Pipeline

The challenge asks for geometric reconstruction, so the system prioritizes camera pose estimation, depth alignment, and point cloud quality before semantics.

## Sparse Anchors Plus Dense Depth

COLMAP gives sparse but geometrically grounded structure. Depth Anything V2 gives dense but scale-ambiguous depth. Combining them creates a stronger engineering system than using either alone.

## Explicit Scale Ambiguity Handling

The implementation aligns predicted dense depth to COLMAP sparse points before fusion. This is a deliberate response to a central monocular reconstruction failure mode.

## Lightweight Outputs

PLY and JSON are easy to inspect and easy to reuse. They are also better for a challenge submission than opaque notebook state.

## Optional Semantics

Semantic labels are left as an extension because the core requirement is geometric coherence. A robust next step is to run 2D segmentation, project masks into the fused cloud, and aggregate labels per 3D point or object region.

