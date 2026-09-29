#!/usr/bin/env python3

"""validate this collision-physics model with the shared analysis in validate_colphys.py"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from validate_colphys import analyze

__all__ = ["analyze"]
