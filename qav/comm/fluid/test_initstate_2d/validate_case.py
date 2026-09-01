#!/usr/bin/env python3

"""Use the shared independent reference for the 2D production initialization state"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"test_common"))
from initstate_validate import analyze

__all__ = ["analyze"]
