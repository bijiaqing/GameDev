#!/usr/bin/env python3

"""Reuse the dimension-aware radial outflow validator"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_source = Path(__file__).resolve().parents[1]/"test_y_outflow_2d"/"validate_case.py"
_spec = importlib.util.spec_from_file_location("qav_y_outflow_shared", _source)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"cannot load radial outflow validator: {_source}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
analyze = _module.analyze

