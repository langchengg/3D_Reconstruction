from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DepthPrediction:
    frame_path: Path
    depth: np.ndarray


class HeuristicDepthEstimator:
    """Deterministic fallback for smoke tests when model weights are unavailable."""

    def __init__(self, min_depth: float = 0.5, max_depth: float = 4.0) -> None:
        self.min_depth = min_depth
        self.max_depth = max_depth

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        import cv2

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        height_weight = np.linspace(1.0, 0.0, gray.shape[0], dtype=np.float32)[:, None]
        relative = 0.65 * (1.0 - gray) + 0.35 * height_weight
        relative = (relative - relative.min()) / max(float(relative.max() - relative.min()), 1e-6)
        depth = self.min_depth + relative * (self.max_depth - self.min_depth)
        return depth.astype(np.float32)


class DepthAnythingV2Estimator:
    """Adapter for the official Depth Anything V2 repository code."""

    MODEL_CONFIGS = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }

    def __init__(self, *, encoder: str, checkpoint: str | Path, device: str, input_size: int = 518) -> None:
        if encoder not in self.MODEL_CONFIGS:
            raise ValueError(f"Unsupported Depth Anything V2 encoder: {encoder}")
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                "Depth Anything V2 checkpoint not found. Download a checkpoint and set depth.checkpoint in config."
            )

        try:
            import torch
            from depth_anything_v2.dpt import DepthAnythingV2
        except ImportError as exc:
            raise ImportError(
                "Depth Anything V2 mode requires torch and the official depth_anything_v2 package. "
                "Install the official repository or use depth.mode=heuristic for a smoke run."
            ) from exc

        self.torch = torch
        self.device = _resolve_device(torch, device)
        self.input_size = input_size
        self.model = DepthAnythingV2(**self.MODEL_CONFIGS[encoder])
        state = torch.load(str(checkpoint_path), map_location="cpu")
        self.model.load_state_dict(state)
        self.model = self.model.to(self.device).eval()

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            depth = self.model.infer_image(image_bgr, self.input_size)
        return np.asarray(depth, dtype=np.float32)


def _resolve_device(torch_module, requested: str) -> str:
    if requested == "mps" and getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
        return "mps"
    if requested == "cuda" and torch_module.cuda.is_available():
        return "cuda"
    return "cpu"


def create_depth_estimator(config: dict) -> HeuristicDepthEstimator | DepthAnythingV2Estimator:
    mode = str(config.get("mode", "depth-anything-v2")).lower()
    if mode == "heuristic":
        return HeuristicDepthEstimator(
            min_depth=float(config.get("heuristic_min_depth", 0.5)),
            max_depth=float(config.get("heuristic_max_depth", 4.0)),
        )
    if mode == "depth-anything-v2":
        return DepthAnythingV2Estimator(
            encoder=str(config.get("encoder", "vits")),
            checkpoint=str(config.get("checkpoint", "")),
            device=str(config.get("device", "cpu")),
            input_size=int(config.get("input_size", 518)),
        )
    raise ValueError(f"Unknown depth mode: {mode}")


def estimate_depth_for_frames(frame_paths: list[Path], output_dir: str | Path, config: dict) -> list[DepthPrediction]:
    import cv2

    estimator = create_depth_estimator(config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions: list[DepthPrediction] = []
    for frame_path in frame_paths:
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read frame: {frame_path}")
        depth = estimator.predict(image)
        npy_path = output / f"{frame_path.stem}.npy"
        np.save(npy_path, depth.astype(np.float32))
        if bool(config.get("save_visualization", True)):
            save_depth_preview(depth, output / f"{frame_path.stem}.png")
        predictions.append(DepthPrediction(frame_path=frame_path, depth=depth))
    return predictions


def save_depth_preview(depth: np.ndarray, output_path: str | Path) -> Path:
    import cv2

    arr = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        normalized = np.zeros_like(arr, dtype=np.uint8)
    else:
        lo = float(np.percentile(arr[finite], 2))
        hi = float(np.percentile(arr[finite], 98))
        normalized = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        normalized = (normalized * 255).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), colored)
    return path

