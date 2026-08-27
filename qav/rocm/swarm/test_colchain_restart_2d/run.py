#!/usr/bin/env python3

"""Run the ROCm production collision-chain checkpoint and restart qualification"""

from pathlib import Path
import sys

model_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(model_dir.parents[2]/"comm"/"swarm"/"test_common"))
sys.path.insert(0, str(model_dir.parent/"test_common"))

from run_chain_restart import run


run(model_dir.name, "rocm")
