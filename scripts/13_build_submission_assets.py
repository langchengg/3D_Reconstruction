#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser(description="Build static README/submission assets.")
    parser.add_argument("--example-dir", default="assets/example_outputs")
    parser.add_argument("--assets-dir", default="assets")
    args = parser.parse_args()

    example_dir = Path(args.example_dir)
    assets_dir = Path(args.assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)

    build_pipeline_diagram(assets_dir / "pipeline.png")
    build_teaser_gif(example_dir, assets_dir / "teaser.gif")
    print(f"Wrote {assets_dir / 'pipeline.png'}")
    print(f"Wrote {assets_dir / 'teaser.gif'}")


def build_pipeline_diagram(output_path: Path) -> None:
    width, height = 1800, 520
    image = Image.new("RGB", (width, height), (248, 249, 250))
    draw = ImageDraw.Draw(image)
    title_font = _font(44)
    label_font = _font(28)
    small_font = _font(20)

    draw.text((70, 48), "Video2World-Lite Pipeline", fill=(22, 27, 34), font=title_font)
    draw.text(
        (70, 104),
        "Phone video -> geometry -> aligned dense scene -> robot-centric world model",
        fill=(82, 92, 105),
        font=small_font,
    )

    steps = [
        ("Video", "phone capture"),
        ("Keyframes", "sharp frames"),
        ("COLMAP", "poses + anchors"),
        ("Depth", "Depth Anything V2"),
        ("Alignment", "global scale fit"),
        ("Fusion", "colored dense PLY"),
        ("Semantics", "support + obstacle"),
        ("World Model", "JSON + metrics"),
    ]
    box_w, box_h = 190, 118
    gap = 24
    x0, y0 = 64, 250
    colors = [
        (55, 120, 180),
        (70, 150, 130),
        (105, 120, 170),
        (175, 120, 75),
        (180, 85, 85),
        (80, 145, 95),
        (180, 110, 120),
        (60, 70, 84),
    ]
    for idx, (name, desc) in enumerate(steps):
        x = x0 + idx * (box_w + gap)
        color = colors[idx]
        draw.rounded_rectangle((x, y0, x + box_w, y0 + box_h), radius=14, fill=color)
        draw.text((x + 18, y0 + 26), name, fill=(255, 255, 255), font=label_font)
        draw.text((x + 18, y0 + 70), desc, fill=(235, 241, 247), font=small_font)
        if idx < len(steps) - 1:
            ax = x + box_w + 4
            ay = y0 + box_h // 2
            draw.line((ax, ay, ax + gap - 8, ay), fill=(85, 95, 110), width=4)
            draw.polygon([(ax + gap - 8, ay - 8), (ax + gap + 4, ay), (ax + gap - 8, ay + 8)], fill=(85, 95, 110))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def build_teaser_gif(example_dir: Path, output_path: Path) -> None:
    frames: list[Image.Image] = []
    sources = [
        ("Camera-view reconstruction", example_dir / "hero_camera_view.png"),
        ("Depth predictions", example_dir / "depth_grid.png"),
        ("Dense 3D world model", example_dir / "hero_scene.png"),
        ("Geometry-derived semantics", example_dir / "semantic_scene.png"),
    ]
    for title, path in sources:
        if path.exists():
            frames.append(_slide(path, title))
    if not frames:
        frames.append(Image.new("RGB", (1100, 720), (24, 26, 30)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=1200, loop=0)


def _slide(path: Path, title: str) -> Image.Image:
    canvas = Image.new("RGB", (1100, 720), (18, 20, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((40, 30), title, fill=(244, 246, 248), font=_font(34))
    image = Image.open(path).convert("RGB")
    image.thumbnail((1020, 610), Image.Resampling.LANCZOS)
    x = (canvas.width - image.width) // 2
    y = 96 + (590 - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def _font(size: int) -> ImageFont.ImageFont:
    for name in ["Arial.ttf", "Helvetica.ttc", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


if __name__ == "__main__":
    main()
