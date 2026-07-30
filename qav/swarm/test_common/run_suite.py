#!/usr/bin/env python3

"""Run the analytical swarm verification matrix"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


GROUPS = {
    "grid": ["test_grid_2d", "test_grid_3d"],
    "transport": ["test_orbit_2d", "test_drag_2d"],
    "diffusion": ["test_diffusion_2d", "test_diffusion_3d"],
    "radiation": ["test_radiation_2d", "test_prdrag_2d"],
    "collision": ["test_collision_2d", "test_collision_3d"],
    "knn": ["test_knn"],
}

# Rebuilding one-step algebra checks at several TEST_RES values adds no
# coverage.  Spatial, temporal, and ensemble tests retain every requested N.
FIXED_RESOLUTION = {
    "test_drag_2d",
    "test_radiation_2d",
    "test_prdrag_2d",
    "test_collision_2d",
    "test_collision_3d",
    "test_knn",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=("all", *GROUPS), default="all")
    parser.add_argument("--res", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--knn-full", action="store_true")
    args = parser.parse_args()

    resolutions = args.res[:2] if args.quick else args.res
    root = Path(__file__).resolve().parents[1]

    # Preserve the order in GROUPS so terminal output follows the progression
    # from grid primitives to coupled physical operators.
    models = [model for values in GROUPS.values() for model in values] if args.group == "all" else GROUPS[args.group]

    for model in models:
        model_resolutions = resolutions[:1] if model in FIXED_RESOLUTION else resolutions
        command = [sys.executable, str(root/model/"run.py"), "--res", *(str(value) for value in model_resolutions)]
        if args.build_only:
            command.append("--build-only")
        if model == "test_knn" and args.knn_full:
            command.append("--full")
        print(f"\n=== {model} ===", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
