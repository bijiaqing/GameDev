#!/usr/bin/env python3

"""Build and execute the KNN correctness and performance matrix"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--res", nargs="+", type=int, default=[32],
        help="accepted for consistency with the common suite; KNN sizes are controlled by --full",
    )
    parser.add_argument("--arch", default="sm_80")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument(
        "--full", action="store_true",
        help="also benchmark one million and ten million particles",
    )
    args = parser.parse_args()

    test_dir = Path(__file__).resolve().parent
    project_root = test_dir.parents[2]
    python = sys.executable

    # KNN capacities and backend headers are compile-time inputs that Make
    # cannot infer from a previously linked executable, so always rebuild the
    # standalone drivers for a clean, reproducible benchmark run
    subprocess.run(
        ["make", "-C", str(test_dir), "clean"], check=True
    )

    # Compile the four independent CUDA drivers once.  The ordinary and wedge
    # programs time production search paths, while the edge and periodic
    # programs concentrate on exact topology at difficult geometric boundaries.
    subprocess.run(
        [
            "make", "-C", str(test_dir), "suite", f"ARCH={args.arch}", "K=200",
        ],
        check=True,
    )

    # Compile and link the real collision translation units with each backend.
    # Distinct executable names prevent Make from mistaking an executable linked
    # against one backend's object directory for the other backend's result.
    for backend in ("kdtree", "morton"):
        executable = test_dir/"bin"/f"production_{backend}"
        build_arguments = [
            "make", "-C", str(project_root),
            "MODEL=test_collision_2d", "RES=32",
            f"COLLISION_SEARCH={backend}", f"EXEC={executable}",
        ]
        subprocess.run([*build_arguments, "clean"], check=True)
        subprocess.run(
            build_arguments,
            check=True,
        )
    if args.build_only:
        return

    bin_dir = test_dir/"bin"
    subprocess.run([str(bin_dir/"knn_edge_tests")], check=True)
    subprocess.run([str(bin_dir/"knn_periodic_tests")], check=True)

    particles = [100_000, 1_000_000, 10_000_000] if args.full else [100_000]
    common = ["--particles", *(str(value) for value in particles), "--arch", args.arch]
    subprocess.run([python, str(test_dir/"run_benchmarks.py"), *common], check=True)
    subprocess.run([python, str(test_dir/"run_wedge_benchmarks.py"), *common], check=True)
    subprocess.run([python, str(test_dir/"analyze_results.py")], check=True)
    subprocess.run([python, str(test_dir/"analyze_wedge.py")], check=True)

    ordinary = json.loads((test_dir/"out"/"manifest.json").read_text())
    wedge = json.loads((test_dir/"out"/"wedge"/"manifest.json").read_text())
    passed = ordinary["all_quality_passed"] and wedge["all_quality_passed"]
    print(
        f"\nknn: {'PASS' if passed else 'FAIL'}  "
        f"ordinary={ordinary['cases']}  wedge={wedge['cases']}  edge=PASS  periodic=PASS"
    )
    if not passed:
        raise SystemExit("one or more KNN benchmark cases failed")


if __name__ == "__main__":
    main()
