#!/usr/bin/env python3


import sys
sys.dont_write_bytecode = True
import os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import json

from pathlib import Path

VAL_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(VAL_ROOT))
from val_config import model_output

from val_config import BACKEND

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"src"))
from run_model import run


model = Path(__file__).resolve().parent.name
run(model)

# changing TEST_RES changes only the simulation polar mesh for this case
# the continuous initializer must therefore produce identical host results
if "--build-only" not in sys.argv:
    project_root = Path(__file__).resolve().parents[4]
    scope = os.environ.get("VAL_SCOPE", "manual")
    out_dir = model_output(project_root/"val", "swarm", model, BACKEND, scope=scope)
    manifest_path = out_dir/"manifest.json"
    manifest = json.loads(manifest_path.read_text())
    resolutions = manifest["resolutions"]
    reference = None
    identical = True
    compared_files = []
    for resolution in resolutions:
        for name in ("initial", "mass_bank", "mass_summary"):
            path = out_dir/f"{name}_N{resolution}.dat"
            values = np.fromfile(path, dtype=np.float64)
            key = (name, values.shape)
            if reference is None:
                reference = {}
            if key not in reference:
                reference[key] = values
            else:
                identical = identical and np.array_equal(values, reference[key])
            compared_files.append(path.name)
    polar_independent = bool(identical) if len(resolutions) > 1 else None
    manifest["polar_resolution_independent"] = polar_independent
    manifest["polar_comparison_files"] = compared_files
    manifest["passed"] = bool(manifest["passed"] and (polar_independent is not False))
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if polar_independent is False:
        raise SystemExit("continuous initialization changed with simulation N_Z")
