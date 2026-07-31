#!/usr/bin/env python3

"""Build, run, and compare matched thread and block fluid sweep models"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from compare_sweeps import compare_pair
from run_model import capture


def build_and_run(
    project_root: Path,
    test_root: Path,
    model: str,
    sweep: str,
    resolution: int,
    save_max: int,
    output_time: float,
) -> float:
    """Clean-build one model, save its logs, execute it, and return wall seconds"""

    model_dir = test_root/model
    out_dir = test_root/"out"/sweep/model
    out_dir.mkdir(parents=True, exist_ok=True)

    # Model constants are compile-time values, so clean before applying a new benchmark configuration
    subprocess.run([
        "make", "-C", str(project_root), f"MODEL={model}", f"FLUID_SWEEP={sweep}", "clean"
    ], check=True)
    build = subprocess.run(
        [
            "make", "-C", str(project_root), f"MODEL={model}",
            f"FLUID_SWEEP={sweep}",
            f"RES={resolution}", f"SAVE={save_max}", f"OUT_TIME={output_time:.17g}",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (out_dir/"build.txt").write_text(build.stdout)
    print(build.stdout, end="", flush=True)

    environment = {
        "nvcc": capture(["nvcc", "--version"], project_root),
        "nvidia_smi": capture(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"], project_root
        ),
    }
    (out_dir/"environment.txt").unlink(missing_ok=True)
    (out_dir/"environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n"
    )

    # Keep the verbose simulation stream in run.txt while reporting a compact elapsed time here
    start = time.perf_counter()
    with (out_dir/"run.txt").open("w") as run_log:
        subprocess.run([str(model_dir/"gamedev")], cwd=project_root, check=True,
                       stdout=run_log, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - start
    (out_dir/"timing.json").write_text(json.dumps({"wall_seconds": elapsed}, indent=2) + "\n")
    print(f"{model}: {elapsed:.3f} wall seconds", flush=True)
    return elapsed


def run_pair(
    project_root: Path,
    test_root: Path,
    dimension: str,
    resolution: int,
    save_max: int,
    output_time: float,
) -> None:
    """Run one dimensional pair and apply the automatic equivalence checks"""

    thread_model = f"test_sweep_thread_{dimension}"
    block_model = f"test_sweep_block_{dimension}"
    print(f"\n=== sweep comparison {dimension} at N={resolution} ===", flush=True)

    thread_time = build_and_run(
        project_root, test_root, thread_model, "thread", resolution, save_max, output_time
    )
    block_time = build_and_run(
        project_root, test_root, block_model, "block", resolution, save_max, output_time
    )

    if dimension == "2d":
        nx, ny, nz = resolution, resolution, 1
        z_min = z_max = 0.5*3.141592653589793
    else:
        nx = ny = nz = resolution
        z_min = 1.4207963267948966
        z_max = 1.7207963267948966

    result = compare_pair(
        test_root/"out"/"thread"/thread_model,
        test_root/"out"/"block"/block_model,
        save_max,
        nx, ny, nz, z_min, z_max,
    )
    result["thread_wall_seconds"] = thread_time
    result["block_wall_seconds"] = block_time
    result["block_over_thread_wall"] = block_time/thread_time

    result_path = test_root/"out"/f"sweep_comparison_{dimension}.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wall-time ratio block/thread: {block_time/thread_time:.4f}")
    print(f"saved comparison: {result_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="run matched fluid sweep implementation tests")
    parser.add_argument("--dimension", choices=("all", "2d", "3d"), default="all")
    parser.add_argument("--res-2d", type=int, default=1024)
    parser.add_argument("--res-3d", type=int, default=128)
    parser.add_argument("--save", type=int, default=10)
    parser.add_argument("--out-time", type=float, default=2.0*3.141592653589793)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    # Quick mode is a compile-and-execute smoke test rather than a performance benchmark
    if args.quick:
        args.res_2d = 128
        args.res_3d = 32
        args.save = 1
        args.out_time = 0.1

    project_root = Path(__file__).resolve().parents[3]
    test_root = Path(__file__).resolve().parents[1]
    if args.dimension in {"all", "2d"}:
        run_pair(project_root, test_root, "2d", args.res_2d, args.save, args.out_time)
    if args.dimension in {"all", "3d"}:
        run_pair(project_root, test_root, "3d", args.res_3d, args.save, args.out_time)


if __name__ == "__main__":
    main()
