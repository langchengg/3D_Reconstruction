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


def run_sparse_reconstruction(frame_dir: str | Path, workspace: str | Path, *, matcher: str, camera_model: str) -> None:
    image_path = Path(frame_dir)
    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    sparse_path = workspace_path / "sparse"
    sparse_path.mkdir(parents=True, exist_ok=True)
    database_path = workspace_path / "database.db"

    run_colmap_stage(
        [
            "feature_extractor",
            "--database_path",
            str(database_path),
            "--image_path",
            str(image_path),
            "--ImageReader.camera_model",
            camera_model,
        ]
    )
    if matcher == "sequential":
        run_colmap_stage(["sequential_matcher", "--database_path", str(database_path)])
    elif matcher == "exhaustive":
        run_colmap_stage(["exhaustive_matcher", "--database_path", str(database_path)])
    else:
        raise ValueError(f"Unsupported COLMAP matcher: {matcher}")
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

