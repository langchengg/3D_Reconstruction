from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ExtractedFrame:
    path: Path
    source_frame_index: int
    timestamp_sec: float
    blur_score: float


def blur_score_laplacian(image_bgr: np.ndarray) -> float:
    import cv2

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def resize_to_width(image_bgr: np.ndarray, width: int) -> np.ndarray:
    import cv2

    if width <= 0 or image_bgr.shape[1] <= width:
        return image_bgr
    scale = width / image_bgr.shape[1]
    height = int(round(image_bgr.shape[0] * scale))
    return cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_AREA)


def extract_keyframes(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    frame_rate: float,
    max_frames: int,
    resize_width: int,
    blur_threshold: float,
) -> list[ExtractedFrame]:
    import cv2

    source = Path(video_path)
    if not source.exists():
        raise FileNotFoundError(f"Input video not found: {source}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for old_frame in output.glob("frame_*.jpg"):
        old_frame.unlink()

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {source}")

    native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    step = max(1, int(round(native_fps / frame_rate)))
    extracted: list[ExtractedFrame] = []
    frame_index = 0

    while len(extracted) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % step == 0:
            score = blur_score_laplacian(frame)
            if score >= blur_threshold:
                resized = resize_to_width(frame, resize_width)
                path = output / f"frame_{len(extracted) + 1:06d}.jpg"
                cv2.imwrite(str(path), resized)
                extracted.append(
                    ExtractedFrame(
                        path=path,
                        source_frame_index=frame_index,
                        timestamp_sec=frame_index / native_fps,
                        blur_score=score,
                    )
                )
        frame_index += 1

    capture.release()
    if not extracted:
        raise RuntimeError(
            "No keyframes were extracted. Lower preprocessing.blur_threshold or check that the input video is readable."
        )
    return extracted

