from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEPTH_ANYTHING_ROOT = PROJECT_ROOT / "third_party" / "Depth-Anything-V2"
if DEPTH_ANYTHING_ROOT.exists() and str(DEPTH_ANYTHING_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPTH_ANYTHING_ROOT))
