from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path
