from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from video2world.fusion import PointCloud, write_ply_ascii


def crop_cloud_to_anchor_bounds(
    cloud: PointCloud,
    anchors: np.ndarray,
    *,
    lower_quantile: float,
    upper_quantile: float,
    margin_ratio: float,
    min_margin: float,
) -> tuple[PointCloud, dict]:
    anchor_points = np.asarray(anchors, dtype=np.float64).reshape((-1, 3))
    if len(cloud.points) == 0 or len(anchor_points) == 0:
        return cloud, {"input_points": len(cloud.points), "output_points": len(cloud.points), "reason": "empty"}

    low = np.quantile(anchor_points, lower_quantile, axis=0)
    high = np.quantile(anchor_points, upper_quantile, axis=0)
    extent = np.maximum(high - low, min_margin)
    margin = np.maximum(extent * margin_ratio, min_margin)
    crop_low = low - margin
    crop_high = high + margin

    mask = np.all((cloud.points >= crop_low) & (cloud.points <= crop_high), axis=1)
    cropped = PointCloud(
        points=cloud.points[mask],
        colors=cloud.colors[mask],
        frame_ids=[frame_id for frame_id, keep in zip(cloud.frame_ids, mask, strict=False) if bool(keep)],
    )
    report = {
        "input_points": int(len(cloud.points)),
        "output_points": int(len(cropped.points)),
        "anchor_points": int(len(anchor_points)),
        "crop_low": crop_low.tolist(),
        "crop_high": crop_high.tolist(),
    }
    return cropped, report


def render_point_cloud_splat(
    cloud: PointCloud,
    output_path: str | Path,
    *,
    width: int = 1400,
    height: int = 1000,
    max_points: int = 180000,
    point_radius: int = 1,
    background: tuple[int, int, int] = (24, 26, 30),
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(cloud.points) == 0:
        image = Image.new("RGB", (width, height), background)
        image.save(path)
        return path

    points = cloud.points.astype(np.float64)
    colors = cloud.colors.astype(np.uint8)
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points).astype(np.int64)
        points = points[indices]
        colors = colors[indices]

    projected, depth_order = _project_points_for_readme(points, width, height)
    colors = colors[depth_order]
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image, "RGBA")
    for (x, y), color in zip(projected, colors, strict=False):
        boosted = np.clip(color.astype(np.int16) + 24, 0, 255).astype(np.uint8)
        fill = (int(boosted[0]), int(boosted[1]), int(boosted[2]), 235)
        if point_radius <= 1:
            draw.point((int(x), int(y)), fill=fill)
        else:
            draw.ellipse(
                (
                    int(x) - point_radius,
                    int(y) - point_radius,
                    int(x) + point_radius,
                    int(y) + point_radius,
                ),
                fill=fill,
            )
    image.save(path)
    return path


def render_point_cloud_camera_view(
    cloud: PointCloud,
    intrinsics: np.ndarray,
    rotation_world_to_camera: np.ndarray,
    translation_world_to_camera: np.ndarray,
    output_path: str | Path,
    *,
    width: int,
    height: int,
    max_points: int = 250000,
    point_radius: int = 2,
    background: tuple[int, int, int] = (18, 20, 24),
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (width, height), background)
    if len(cloud.points) == 0:
        image.save(path)
        return path

    points = cloud.points.astype(np.float64)
    colors = cloud.colors.astype(np.uint8)
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points).astype(np.int64)
        points = points[indices]
        colors = colors[indices]

    r = np.asarray(rotation_world_to_camera, dtype=np.float64).reshape((3, 3))
    t = np.asarray(translation_world_to_camera, dtype=np.float64).reshape(3)
    k = np.asarray(intrinsics, dtype=np.float64).reshape((3, 3))
    camera_points = (r @ points.T).T + t
    z = camera_points[:, 2]
    valid = np.isfinite(z) & (z > 1e-6)
    camera_points = camera_points[valid]
    colors = colors[valid]
    z = z[valid]
    if len(z) == 0:
        image.save(path)
        return path

    u = k[0, 0] * (camera_points[:, 0] / z) + k[0, 2]
    v = k[1, 1] * (camera_points[:, 1] / z) + k[1, 2]
    inside = (u >= 0) & (v >= 0) & (u < width) & (v < height)
    u = u[inside]
    v = v[inside]
    z = z[inside]
    colors = colors[inside]
    order = np.argsort(z)[::-1]

    draw = ImageDraw.Draw(image, "RGBA")
    for x, y, color in zip(u[order], v[order], colors[order], strict=False):
        boosted = np.clip(color.astype(np.int16) + 28, 0, 255).astype(np.uint8)
        fill = (int(boosted[0]), int(boosted[1]), int(boosted[2]), 230)
        draw.ellipse(
            (
                int(x) - point_radius,
                int(y) - point_radius,
                int(x) + point_radius,
                int(y) + point_radius,
            ),
            fill=fill,
        )
    image.save(path)
    return path


def _project_points_for_readme(points: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(points, axis=0)
    centered = points - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    basis = eigvecs[:, order]
    coords = centered @ basis

    screen = np.column_stack(
        [
            coords[:, 0] - 0.40 * coords[:, 1],
            -coords[:, 2] - 0.25 * coords[:, 1],
        ]
    )
    depth = coords[:, 1]
    depth_order = np.argsort(depth)
    screen = screen[depth_order]
    lo = np.percentile(screen, 1, axis=0)
    hi = np.percentile(screen, 99, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    normalized = (screen - lo) / span
    margin = 0.08
    normalized = normalized * (1.0 - 2.0 * margin) + margin
    pixels = np.column_stack([normalized[:, 0] * width, normalized[:, 1] * height])
    pixels[:, 0] = np.clip(pixels[:, 0], 0, width - 1)
    pixels[:, 1] = np.clip(pixels[:, 1], 0, height - 1)
    return pixels, depth_order


def save_presentation_cloud(cloud: PointCloud, output_path: str | Path) -> Path:
    return write_ply_ascii(cloud, output_path)


def save_depth_grid(
    frame_paths: list[Path],
    depth_preview_paths: list[Path],
    output_path: str | Path,
    *,
    columns: int,
    max_items: int,
    tile_width: int = 360,
) -> Path:
    pairs = [(frame, depth) for frame, depth in zip(frame_paths, depth_preview_paths, strict=False) if frame.exists() and depth.exists()]
    pairs = pairs[:max_items]
    if not pairs:
        raise FileNotFoundError("No RGB/depth preview pairs found for depth grid")

    tiles: list[Image.Image] = []
    for frame_path, depth_path in pairs:
        rgb = Image.open(frame_path).convert("RGB")
        depth = Image.open(depth_path).convert("RGB")
        rgb = _resize_to_width(rgb, tile_width)
        depth = depth.resize(rgb.size)
        tile = Image.new("RGB", (rgb.width * 2, rgb.height), (255, 255, 255))
        tile.paste(rgb, (0, 0))
        tile.paste(depth, (rgb.width, 0))
        tiles.append(tile)

    columns = max(1, columns)
    rows = int(np.ceil(len(tiles) / columns))
    tile_w = max(tile.width for tile in tiles)
    tile_h = max(tile.height for tile in tiles)
    canvas = Image.new("RGB", (columns * tile_w, rows * tile_h), (248, 249, 250))
    for idx, tile in enumerate(tiles):
        row, col = divmod(idx, columns)
        canvas.paste(tile, (col * tile_w, row * tile_h))

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


def create_poisson_mesh_from_cloud(
    cloud: PointCloud,
    output_path: str | Path,
    *,
    max_points: int,
    poisson_depth: int,
    density_quantile: float,
) -> tuple[Path | None, dict]:
    if len(cloud.points) < 100:
        return None, {"success": False, "reason": "too few points", "input_points": len(cloud.points)}

    try:
        import open3d as o3d
    except ImportError:
        return None, {"success": False, "reason": "open3d not installed", "input_points": len(cloud.points)}

    points = cloud.points
    colors = cloud.colors.astype(np.float64) / 255.0
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points).astype(np.int64)
        points = points[indices]
        colors = colors[indices]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.clip(0.0, 1.0))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(20)

    try:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=poisson_depth)
    except RuntimeError as exc:
        return None, {"success": False, "reason": str(exc), "input_points": len(cloud.points)}
    density_values = np.asarray(densities)
    if len(density_values):
        threshold = np.quantile(density_values, density_quantile)
        mesh.remove_vertices_by_mask(density_values < threshold)
    mesh = mesh.crop(pcd.get_axis_aligned_bounding_box())
    mesh.compute_vertex_normals()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=True)
    return path, {
        "success": True,
        "input_points": int(len(cloud.points)),
        "mesh_points_used": int(len(points)),
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.triangles)),
    }


def sample_mesh_preview_cloud(mesh_path: str | Path, *, number_of_points: int = 120000) -> PointCloud:
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    pcd = mesh.sample_points_uniformly(number_of_points=min(number_of_points, max(1, len(mesh.triangles) * 2)))
    points = np.asarray(pcd.points, dtype=np.float64)
    colors = np.full((len(points), 3), 185, dtype=np.uint8)
    return PointCloud(points=points.reshape((-1, 3)), colors=colors.reshape((-1, 3)), frame_ids=[0] * len(points))


def _resize_to_width(image: Image.Image, width: int) -> Image.Image:
    if image.width == width:
        return image
    height = int(round(image.height * (width / image.width)))
    return image.resize((width, height))
