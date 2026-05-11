# Method

Video2World-Lite combines sparse multi-view geometry with dense monocular depth.

COLMAP supplies camera intrinsics, camera extrinsics, and sparse 3D points. Depth Anything V2 supplies dense per-frame depth. Because monocular depth has scale ambiguity, the system does not fuse raw predictions directly. For each registered frame, it finds COLMAP 2D observations with valid 3D point ids, computes the sparse point depth in the camera frame, samples the predicted depth at the same pixels, and fits:

```text
z_colmap = scale * z_pred + shift
```

The aligned depth map is then unprojected:

```text
X_camera = depth(u, v) * K^-1 [u, v, 1]^T
X_world = T_world_camera * X_camera
```

The final point cloud is voxel-downsampled, filtered for outliers, and exported with a structured JSON summary.

