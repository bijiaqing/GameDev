#!/usr/bin/env python3
import sys

sys.dont_write_bytecode = True
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"test_common"))
from run_model import run

run(Path(__file__).resolve().parent.name)
