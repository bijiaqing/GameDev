#!/usr/bin/env python3

"""run the production custom-kernel fragmentation-chain case"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from run_chain import run

run(Path(__file__).resolve().parent.name)
