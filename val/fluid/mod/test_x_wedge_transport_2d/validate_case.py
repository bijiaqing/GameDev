#!/usr/bin/env python3

"""reuse the shared non-full-disk periodic validator"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

_source = Path(__file__).resolve().parents[2] / "src" / "wedge_periodic_validate.py"
_spec = importlib.util.spec_from_file_location("val_wedge_periodic_shared", _source)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"cannot load wedge-periodic validator: {_source}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
analyze = _module.analyze
