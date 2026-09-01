#!/usr/bin/env python3

"""Reuse the shared non-full-disk periodic validator"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_source = Path(__file__).resolve().parents[1]/"test_common"/"wedge_periodic_validate.py"
_spec = importlib.util.spec_from_file_location("qav_wedge_periodic_shared", _source)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"cannot load wedge-periodic validator: {_source}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
analyze = _module.analyze
