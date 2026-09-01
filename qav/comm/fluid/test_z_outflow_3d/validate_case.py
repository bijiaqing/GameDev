#!/usr/bin/env python3

"""Reuse the shared exact polar-boundary validator"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_source = Path(__file__).resolve().parents[1]/"test_common"/"polar_boundary_validate.py"
_spec = importlib.util.spec_from_file_location("qav_polar_boundary_shared", _source)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"cannot load polar-boundary validator: {_source}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
analyze = _module.analyze
