#!/usr/bin/env python3

"""Run the CUDA collision-search geometry-epoch qualification"""

from pathlib import Path
import sys

model_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(model_dir.parents[2]/"comm"/"swarm"/"test_common"))

from run_geometry_reuse import run


run(model_dir.name, "cuda")
