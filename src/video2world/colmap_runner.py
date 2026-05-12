from __future__ import annotations

from pathlib import Path
import shutil
import subprocess


def ensure_colmap_available() -> str:
    executable = shutil.which("colmap")
    if executable is None:
        raise FileNotFoundError("COLMAP executable not found on PATH. Install COLMAP before running reconstruction.")
    return executable


def run_colmap_stage(args: list[str]) -> None:
    executable = ensure_colmap_available()
    command = [executable, *args]
    subprocess.run(command, check=True)


def feature_extractor_args(database_path: str | Path, image_path: str | Path, camera_model: str) -> list[str]:
    return [
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_path),
        "--ImageReader.camera_model",
        camera_model,
        "--ImageReader.single_camera",
        "1",
        "--FeatureExtraction.use_gpu",
        "0",
    ]


def matcher_args(matcher: str, database_path: str | Path) -> list[str]:
    if matcher == "sequential":
        command = "sequential_matcher"
    elif matcher == "exhaustive":
        command = "exhaustive_matcher"
    else:
        raise ValueError(f"Unsupported COLMAP matcher: {matcher}")
    return [
        command,
        "--database_path",
        str(database_path),
        "--FeatureMatching.use_gpu",
        "0",
    ]


def run_sparse_reconstruction(frame_dir: str | Path, workspace: str | Path, *, matcher: str, camera_model: str) -> None:
    image_path = Path(frame_dir)
    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    sparse_path = workspace_path / "sparse"
    sparse_path.mkdir(parents=True, exist_ok=True)
    database_path = workspace_path / "database.db"

    run_colmap_stage(feature_extractor_args(database_path, image_path, camera_model))
    run_colmap_stage(matcher_args(matcher, database_path))
    run_colmap_stage(
        [
            "mapper",
            "--database_path",
            str(database_path),
            "--image_path",
            str(image_path),
            "--output_path",
            str(sparse_path),
        ]
    )


def convert_model_to_text(input_model: str | Path, output_model: str | Path) -> None:
    output = Path(output_model)
    output.mkdir(parents=True, exist_ok=True)
    run_colmap_stage(
        [
            "model_converter",
            "--input_path",
            str(input_model),
            "--output_path",
            str(output),
            "--output_type",
            "TXT",
        ]
    )
