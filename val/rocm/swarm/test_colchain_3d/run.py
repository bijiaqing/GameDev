#!/usr/bin/env python3

"""Run the production full-three-dimensional collision-chain case"""

from pathlib import Path
import sys

sys.dont_write_bytecode = True

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"test_common"))

from run_chain import run


run(Path(__file__).resolve().parent.name)
