from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import sys

import _bootstrap  # noqa: F401


def package_status(module_name: str) -> str:
    return "OK" if importlib.util.find_spec(module_name) is not None else "MISSING"


def main() -> int:
    required = ["numpy", "yaml", "cv2", "matplotlib"]
    optional = ["open3d", "torch", "depth_anything_v2"]
    print("Video2World-Lite environment check")
    print("Required Python packages:")
    missing_required = []
    for name in required:
        status = package_status(name)
        print(f"  {name}: {status}")
        if status != "OK":
            missing_required.append(name)
    print("Optional Python packages:")
    for name in optional:
        print(f"  {name}: {package_status(name)}")
    repo = Path("third_party/Depth-Anything-V2")
    checkpoint = repo / "checkpoints" / "depth_anything_v2_vits.pth"
    print(f"Depth Anything V2 official repo: {repo if repo.exists() else 'MISSING'}")
    print(f"Depth Anything V2 Small checkpoint: {checkpoint if checkpoint.exists() else 'MISSING'}")
    print(f"COLMAP executable: {shutil.which('colmap') or 'MISSING'}")
    if missing_required:
        print(f"Missing required packages: {', '.join(missing_required)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
