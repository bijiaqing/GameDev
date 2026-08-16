#!/usr/bin/env python3

"""Build, run, and compare matched thread and block fluid sweep models"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

from compare_sweeps import compare_pair
from run_model import capture


def write_json(path: Path, record: dict[str, object]) -> None:
    """Write one deterministic JSON record with a final newline"""

    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def parse_step(line: str) -> tuple[int, float, float, float] | None:
    """Parse one accepted-step row from the native executable stream"""

    fields = line.split()
    if len(fields) != 4:
        return None
    try:
        return int(fields[0]), float(fields[1]), float(fields[2]), float(fields[3])
    except ValueError:
        return None


def convert_variables(out_dir: Path) -> None:
    """Convert the production parameter report to a structured sweep artifact"""

    source = out_dir/"variables.txt"
    if not source.is_file():
        return

    parameters: dict[str, int | float | str] = {}
    for line in source.read_text().splitlines():
        if "=" not in line:
            continue
        name, value = (part.strip() for part in line.split("=", 1))
        if re.fullmatch(r"[+-]?\d+", value):
            parameters[name] = int(value)
            continue
        try:
            parameters[name] = float(value)
        except ValueError:
            parameters[name] = value

    write_json(out_dir/"variables.json", {"schema": 1, "parameters": parameters})
    source.unlink()


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

    out_dir = project_root/"qav"/"logs"/"fluid"/"cuda"/sweep/model
    out_dir.mkdir(parents=True, exist_ok=True)
    for legacy_name in ("build.txt", "run.txt", "variables.txt"):
        (out_dir/legacy_name).unlink(missing_ok=True)

    # Model constants are compile-time values, so clean before applying a new benchmark configuration
    subprocess.run([
        "make", "-C", str(project_root), f"MODEL={model}", "GPU_BACKEND=cuda", f"FLUID_SWEEP={sweep}", "clean"
    ], check=True)
    build_command = [
        "make", "-C", str(project_root), f"MODEL={model}", "GPU_BACKEND=cuda",
        f"FLUID_SWEEP={sweep}",
        f"RES={resolution}", f"SAVE={save_max}", f"OUT_TIME={output_time:.17g}",
    ]
    build = subprocess.run(
        build_command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    write_json(out_dir/"build.json", {
        "schema": 1,
        "command": build_command,
        "return_code": build.returncode,
        "output": build.stdout,
    })
    print(build.stdout, end="", flush=True)
    build.check_returncode()

    environment = {
        "nvcc": capture(["nvcc", "--version"], project_root),
        "nvidia_smi": capture(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"], project_root
        ),
    }
    write_json(out_dir/"environment.json", environment)

    # Retain the native stream and accepted-step records inside one JSON log
    start = time.perf_counter()
    run_command = [str(project_root/"bin"/model/"cuda"/"gamedev")]
    simulation = subprocess.Popen(
        run_command, cwd=project_root, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
    )
    if simulation.stdout is None:
        raise RuntimeError("failed to capture simulation output")
    run_lines = []
    step_records: list[dict[str, int | float]] = []
    for line in simulation.stdout:
        print(line, end="", flush=True)
        run_lines.append(line)
        parsed = parse_step(line)
        if parsed is not None:
            frame, dt, clock_out, clock_sim = parsed
            step_records.append({
                "step": len(step_records) + 1,
                "frame": frame,
                "dt": dt,
                "clock_out": clock_out,
                "clock_sim": clock_sim,
            })
    return_code = simulation.wait()
    elapsed = time.perf_counter() - start
    write_json(out_dir/"run.json", {
        "schema": 1,
        "command": run_command,
        "return_code": return_code,
        "wall_seconds": elapsed,
        "accepted_steps": len(step_records),
        "last_step": step_records[-1] if step_records else None,
        "steps": step_records,
        "output": "".join(run_lines),
    })
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, run_command)

    convert_variables(out_dir)
    write_json(out_dir/"timing.json", {"wall_seconds": elapsed})
    print(
        f"{model}: {elapsed:.3f} wall seconds, {len(step_records)} accepted steps",
        flush=True,
    )
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
        project_root/"qav"/"logs"/"fluid"/"cuda"/"thread"/thread_model,
        project_root/"qav"/"logs"/"fluid"/"cuda"/"block"/block_model,
        save_max,
        nx, ny, nz, z_min, z_max,
    )
    result["thread_wall_seconds"] = thread_time
    result["block_wall_seconds"] = block_time
    result["block_over_thread_wall"] = block_time/thread_time

    result_path = project_root/"qav"/"logs"/"fluid"/"cuda"/f"sweep_comparison_{dimension}.json"
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

    project_root = Path(__file__).resolve().parents[4]
    test_root = Path(__file__).resolve().parents[1]
    if args.dimension in {"all", "2d"}:
        run_pair(project_root, test_root, "2d", args.res_2d, args.save, args.out_time)
    if args.dimension in {"all", "3d"}:
        run_pair(project_root, test_root, "3d", args.res_3d, args.save, args.out_time)


if __name__ == "__main__":
    main()
