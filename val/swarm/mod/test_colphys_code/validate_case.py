#!/usr/bin/env python3


import sys
sys.dont_write_bytecode = True
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"src"))
from validate_colphys import analyze
