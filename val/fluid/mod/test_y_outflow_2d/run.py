#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True
import os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"src"))
from run_model import run

run(Path(__file__).resolve().parent.name)

