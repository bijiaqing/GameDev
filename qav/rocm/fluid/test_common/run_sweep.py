#!/usr/bin/env python3

"""Build, run, and compare matched thread and block fluid sweep models"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import threading
import time
from pathlib import Path

from compare_sweeps import compare_pair
from run_model import environment_record, model_executable


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
        raise FileNotFoundError(f"missing simulation parameter report: {source}")

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


def sweep_tag(resolution: int, save_max: int, output_time: float) -> str:
    """Name a sweep configuration so quick and full records cannot collide"""

    time_tag = f"{output_time:.8g}".replace("-", "m").replace("+", "").replace(".", "p")
    return f"N{resolution}_save{save_max}_tout{time_tag}"


def build_and_run(
    project_root: Path,
    test_root: Path,
    model: str,
    sweep: str,
    resolution: int,
    save_max: int,
    output_time: float,
    target: str,
    progress_seconds: float,
    tag: str,
) -> float:
    """Clean-build one model, save its logs, execute it, and return wall seconds"""

    out_dir = project_root/"qav"/"logs"/"fluid"/"rocm"/sweep/model/tag
    out_dir.mkdir(parents=True, exist_ok=True)
    for legacy_name in ("build.txt", "run.txt", "variables.txt"):
        (out_dir/legacy_name).unlink(missing_ok=True)

    # model constants are compile-time values, so clean before applying a new benchmark configuration
    subprocess.run([
        "make", "-C", str(project_root), f"MODEL={model}", "GPU_BACKEND=rocm", f"FLUID_SWEEP={sweep}",
        f"GPU_TARGET={target}", "QAV_SCOPE=all", f"OUT_TAG={tag}", "clean"
    ], check=True)
    build_command = [
        "make", "-C", str(project_root), f"MODEL={model}", "GPU_BACKEND=rocm",
        f"FLUID_SWEEP={sweep}",
        f"GPU_TARGET={target}", "QAV_SCOPE=all",
        f"RES={resolution}", f"SAVE={save_max}", f"OUT_TIME={output_time:.17g}",
        f"OUT_TAG={tag}",
    ]
    build = subprocess.Popen(
        build_command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    if build.stdout is None:
        raise RuntimeError("failed to capture compiler output")
    build_lines = []
    for line in build.stdout:
        print(line, end="", flush=True)
        build_lines.append(line)
    return_code = build.wait()
    write_json(out_dir/"build.json", {
        "schema": 1,
        "command": build_command,
        "return_code": return_code,
        "output": "".join(build_lines),
    })
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, build_command)

    environment = environment_record(project_root)
    environment["gpu_target"] = target
    write_json(out_dir/"environment.json", environment)

    # retain the native stream in run.json while reporting a throttled live heartbeat
    start = time.perf_counter()
    run_command = [str(model_executable(project_root, model, "rocm", "fluid"))]
    step_records: list[dict[str, int | float]] = []
    latest_step: tuple[int, float, float, float] | None = None
    next_report = start + progress_seconds
    print(f"[RUN] model={model} sweep={sweep}", flush=True)

    simulation = subprocess.Popen(
        run_command, cwd=project_root, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
    )
    if simulation.stdout is None:
        raise RuntimeError("failed to capture simulation output")

    output_queue: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        """Drain the child pipe independently so heartbeat timing cannot block"""

        for line in simulation.stdout:
            output_queue.put(line)
        output_queue.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    run_lines = []
    stream_finished = False
    while not stream_finished:
        try:
            line = output_queue.get(timeout=0.5)
            if line is None:
                stream_finished = True
            else:
                run_lines.append(line)
                parsed = parse_step(line)
                if parsed is not None:
                    latest_step = parsed
                    frame, dt, clock_out, clock_sim = parsed
                    step_records.append({
                        "step": len(step_records) + 1,
                        "frame": frame,
                        "dt": dt,
                        "clock_out": clock_out,
                        "clock_sim": clock_sim,
                    })
        except queue.Empty:
            pass

        now = time.perf_counter()
        if progress_seconds > 0.0 and now >= next_report:
            elapsed = now - start
            if latest_step is None:
                print(
                    f"[PROGRESS] model={model} wall={elapsed:.1f}s "
                    "waiting for first accepted step",
                    flush=True,
                )
            else:
                frame, dt, clock_out, clock_sim = latest_step
                print(
                    f"[PROGRESS] model={model} wall={elapsed:.1f}s "
                    f"step={len(step_records)} frame={frame} dt={dt:.4e} "
                    f"t_out={clock_out:.4e} t_sim={clock_sim:.4e}",
                    flush=True,
                )
            next_report = now + progress_seconds

    reader.join()
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
    (out_dir/"timing.json").write_text(json.dumps({"wall_seconds": elapsed}, indent=2) + "\n")
    print(
        f"[DONE] model={model} wall={elapsed:.3f}s accepted_steps={len(step_records)}",
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
    target: str,
    progress_seconds: float,
) -> None:
    """Run one dimensional pair and apply the automatic equivalence checks"""

    thread_model = f"test_sweep_thread_{dimension}"
    block_model = f"test_sweep_block_{dimension}"
    tag = sweep_tag(resolution, save_max, output_time)
    print(f"\n=== sweep comparison {dimension} at N={resolution} ===", flush=True)

    thread_time = build_and_run(
        project_root, test_root, thread_model, "thread", resolution, save_max, output_time, target,
        progress_seconds, tag
    )
    block_time = build_and_run(
        project_root, test_root, block_model, "block", resolution, save_max, output_time, target,
        progress_seconds, tag
    )

    if dimension == "2d":
        nx, ny, nz = resolution, resolution, 1
        z_min = z_max = 0.5*3.141592653589793
    else:
        nx = ny = nz = resolution
        z_min = 1.4207963267948966
        z_max = 1.7207963267948966

    result = compare_pair(
        project_root/"qav"/"logs"/"fluid"/"rocm"/"thread"/thread_model/tag,
        project_root/"qav"/"logs"/"fluid"/"rocm"/"block"/block_model/tag,
        save_max,
        nx, ny, nz, z_min, z_max,
    )
    result["thread_wall_seconds"] = thread_time
    result["block_wall_seconds"] = block_time
    result["block_over_thread_wall"] = block_time/thread_time

    result["configuration"] = tag
    result_path = project_root/"qav"/"logs"/"fluid"/"rocm"/f"sweep_comparison_{dimension}_{tag}.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wall-time ratio block/thread: {block_time/thread_time:.4f}")
    print(f"saved comparison: {result_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="run matched fluid sweep implementation tests")
    parser.add_argument("--dimension", choices=("all", "2d", "3d"), default="all")
    parser.add_argument("--res-2d", type=int, default=512)
    parser.add_argument("--res-3d", type=int, default=128)
    parser.add_argument("--save", type=int, default=10)
    parser.add_argument("--out-time", type=float, default=2.0*3.141592653589793)
    parser.add_argument("--target", default=os.environ.get("AMDGPU_TARGET", "gfx942"))
    parser.add_argument(
        "--progress-seconds", type=float, default=10.0,
        help="wall-clock interval between live simulation heartbeats; use 0 to disable",
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    # quick mode is a compile-and-execute smoke test rather than a performance benchmark
    if args.quick:
        args.res_2d = 128
        args.res_3d = 32
        args.save = 1
        args.out_time = 0.1

    project_root = Path(__file__).resolve().parents[4]
    test_root = Path(__file__).resolve().parents[1]
    if args.dimension in {"all", "2d"}:
        run_pair(
            project_root, test_root, "2d", args.res_2d, args.save, args.out_time,
            args.target, args.progress_seconds
        )
    if args.dimension in {"all", "3d"}:
        run_pair(
            project_root, test_root, "3d", args.res_3d, args.save, args.out_time,
            args.target, args.progress_seconds
        )


if __name__ == "__main__":
    main()
