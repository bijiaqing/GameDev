#!/usr/bin/env python3

"""Run repeatable uninstrumented ROCm performance campaigns"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import shutil
import statistics
import subprocess
import threading
import time


def utc_now() -> str:
    """Return one machine-readable UTC timestamp"""

    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, record: dict) -> None:
    """Write one JSON artifact atomically"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def capture(command: list[str], cwd: Path) -> str:
    """Capture an optional environment diagnostic without aborting the campaign"""

    try:
        result = subprocess.run(
            command, cwd=cwd, check=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        return result.stdout.strip()
    except OSError as error:
        return f"unavailable: {error}"


def capture_optional(command: list[str], cwd: Path) -> str | None:
    """Capture optional provenance and return null when the command is unavailable"""

    try:
        result = subprocess.run(
            command, cwd=cwd, check=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except OSError:
        return None


def source_fingerprint(project_root: Path) -> dict:
    """Hash build-relevant text files when a standalone cluster copy has no Git metadata"""

    digest = hashlib.sha256()
    count = 0
    suffixes = {".h", ".hpp", ".cuh", ".hip", ".py", ".mk"}
    for path in sorted(project_root.rglob("*")):
        if not path.is_file() or any(
            part in {"obj", "out", "bin", "__pycache__"} for part in path.parts
        ):
            continue
        if path.name != "Makefile" and path.suffix not in suffixes:
            continue
        relative = path.relative_to(project_root).as_posix().encode()
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
        count += 1
    return {"algorithm": "sha256", "digest": digest.hexdigest(), "file_count": count}


def environment_record(project_root: Path, target: str) -> dict:
    """Record the toolchain and accelerator associated with timing results"""

    return {
        "target": target,
        "hipcc": capture(["hipcc", "--version"], project_root),
        "hipconfig": capture(["hipconfig", "--full"], project_root),
        "amd_smi": capture(["amd-smi", "static", "--json"], project_root),
        "rocprofv3": capture(["rocprofv3", "--version"], project_root),
        "rocprof_compute": capture(["rocprof-compute", "--version"], project_root),
        "git_revision": capture_optional(["git", "rev-parse", "HEAD"], project_root),
        "git_status": capture_optional(["git", "status", "--short"], project_root),
        "source_fingerprint": source_fingerprint(project_root),
    }


def percentile(values: list[float], fraction: float) -> float:
    """Return a linearly interpolated percentile for a nonempty sample"""

    ordered = sorted(values)
    location = fraction*(len(ordered) - 1)
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    weight = location - lower
    return ordered[lower]*(1.0 - weight) + ordered[upper]*weight


def parse_bytes(value: object) -> int | None:
    """Convert one AMD SMI byte, KiB, MiB, or GiB value to bytes"""

    if isinstance(value, (int, float)):
        return int(value)
    if not isinstance(value, str):
        return None
    match = re.fullmatch(
        r"\s*([0-9]+(?:\.[0-9]+)?)\s*(B|KB|KIB|MB|MIB|GB|GIB)?\s*",
        value.upper(),
    )
    if match is None:
        return None
    scale = {
        None: 1,
        "B": 1,
        "KB": 1000,
        "KIB": 1024,
        "MB": 1000**2,
        "MIB": 1024**2,
        "GB": 1000**3,
        "GIB": 1024**3,
    }[match.group(2)]
    return int(float(match.group(1))*scale)


def memory_candidates(value: object, path: tuple[str, ...] = ()) -> list[int]:
    """Extract process-memory values from the version-dependent AMD SMI JSON schema"""

    result: list[int] = []
    if isinstance(value, dict):
        for key, child in value.items():
            result.extend(memory_candidates(child, (*path, str(key).lower())))
    elif isinstance(value, list):
        for child in value:
            result.extend(memory_candidates(child, path))
    elif path and (
        path[-1] == "mem"
        or any(token in "_".join(path) for token in ("vram_mem", "gtt_mem", "memory_usage"))
    ):
        parsed = parse_bytes(value)
        if parsed is not None:
            result.append(parsed)
    return result


def monitor_memory(pid: int, stop: threading.Event, interval: float, record: dict) -> None:
    """Sample process memory during the untimed warm-up only"""

    command = ["amd-smi", "process", "--pid", str(pid), "--general", "--json"]
    record.update({"command": command, "samples": 0, "parsed_samples": 0, "peak_bytes": None})
    peak = 0
    while not stop.is_set():
        try:
            sample = subprocess.run(
                command, check=False, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=max(2.0, 2.0*interval),
            )
            record["samples"] += 1
            if sample.returncode == 0:
                values = memory_candidates(json.loads(sample.stdout))
                if values:
                    peak = max(peak, max(values))
                    record["parsed_samples"] += 1
                elif "first_unparsed_sample" not in record:
                    record["first_unparsed_sample"] = sample.stdout.strip()
            elif "first_error" not in record:
                record["first_error"] = sample.stdout.strip()
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
            if "first_error" not in record:
                record["first_error"] = str(error)
        stop.wait(interval)
    record["peak_bytes"] = peak or None
    record["available"] = record["parsed_samples"] > 0
    if record["samples"] > 0 and not record["available"] and "first_error" not in record:
        record["first_error"] = (
            "amd-smi returned no process-memory value for this MI300A process"
        )


def parse_step(line: str, component: str) -> dict | None:
    """Parse one accepted fluid step or completed collision interval"""

    fields = line.split()
    try:
        if component == "fluid" and len(fields) == 4:
            return {
                "frame": int(fields[0]),
                "dt": float(fields[1]),
                "clock_out": float(fields[2]),
                "clock_sim": float(fields[3]),
            }
        if component == "swarm" and len(fields) == 5:
            return {
                "frame": int(fields[0]),
                "clock_sim": float(fields[1]),
                "clock_out": float(fields[2]),
                "collision_batches": int(fields[3]),
                "dt_collision": float(fields[4]),
            }
    except ValueError:
        return None
    return None


def run_executable(
    executable: Path,
    project_root: Path,
    component: str,
    progress_seconds: float,
    monitor: bool,
) -> dict:
    """Execute one workload and separate initialization, evolution, and final-output time"""

    command = [str(executable)]
    started = time.perf_counter()
    process = subprocess.Popen(
        command, cwd=project_root, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
    )
    if process.stdout is None:
        raise RuntimeError("failed to capture benchmark output")

    memory: dict = {"enabled": monitor}
    memory_stop = threading.Event()
    memory_thread = None
    if monitor and shutil.which("amd-smi"):
        memory_thread = threading.Thread(
            target=monitor_memory, args=(process.pid, memory_stop, 0.5, memory), daemon=True,
        )
        memory_thread.start()
    elif monitor:
        memory["first_error"] = "amd-smi is unavailable"

    output_queue: queue.Queue[tuple[str, float] | None] = queue.Queue()

    def read_output() -> None:
        """Drain native output without blocking heartbeat timing"""

        for line in process.stdout:
            output_queue.put((line, time.perf_counter()))
        output_queue.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    output_lines: list[str] = []
    step_records: list[dict] = []
    initialization_done = None
    last_step_time = None
    stream_finished = False
    next_report = started + progress_seconds
    while not stream_finished:
        try:
            item = output_queue.get(timeout=0.5)
            if item is None:
                stream_finished = True
            else:
                line, received = item
                output_lines.append(line)
                if initialization_done is None and "finished on" in line:
                    initialization_done = received
                step = parse_step(line, component)
                if step is not None:
                    step_records.append(step)
                    last_step_time = received
        except queue.Empty:
            pass

        now = time.perf_counter()
        if progress_seconds > 0.0 and now >= next_report:
            if step_records:
                last = step_records[-1]
                simulated = last["clock_sim"]
                count = len(step_records) if component == "fluid" else last["collision_batches"]
                print(
                    f"[PROGRESS] wall={now-started:.1f}s work={count} "
                    f"t_sim={simulated:.6e} dt={last.get('dt', last.get('dt_collision')):.6e}",
                    flush=True,
                )
            elif initialization_done is not None:
                print(
                    f"[PROGRESS] wall={now-started:.1f}s evolution awaiting first completed record",
                    flush=True,
                )
            else:
                print(f"[PROGRESS] wall={now-started:.1f}s initializing", flush=True)
            next_report = now + progress_seconds

    reader.join()
    return_code = process.wait()
    finished = time.perf_counter()
    memory_stop.set()
    if memory_thread is not None:
        memory_thread.join()

    accepted = len(step_records) if component == "fluid" else (
        step_records[-1]["collision_batches"] if step_records else 0
    )
    simulated_time = step_records[-1]["clock_sim"] if step_records else 0.0
    initialization_seconds = (
        initialization_done - started if initialization_done is not None else None
    )
    evolution_seconds = (
        last_step_time - initialization_done
        if initialization_done is not None and last_step_time is not None else None
    )
    output_seconds = finished - last_step_time if last_step_time is not None else None
    return {
        "command": command,
        "return_code": return_code,
        "wall_seconds": finished - started,
        "initialization_seconds": initialization_seconds,
        "evolution_seconds": evolution_seconds,
        "output_seconds": output_seconds,
        "accepted_work_units": accepted,
        "completed_simulated_time": simulated_time,
        "last_step": step_records[-1] if step_records else None,
        "output": "".join(output_lines),
        "memory": memory,
        "passed": (
            return_code == 0
            and accepted >= 100
            and initialization_seconds is not None
            and evolution_seconds is not None
            and output_seconds is not None
        ),
    }


def configurations(group: str, scale: str) -> list[dict]:
    """Construct the requested production-like performance matrix"""

    scales = ("baseline", "production") if scale == "all" else (scale,)
    records: list[dict] = []
    if group in {"all", "fluid"}:
        table = {
            "baseline": (("2d", 512), ("3d", 64)),
            "production": (("2d", 1024), ("3d", 128)),
            "large": (("2d", 2048), ("3d", 256)),
        }
        for name in scales:
            for dimension, resolution in table[name]:
                records.append({
                    "case": f"fluid_{dimension}_{name}",
                    "component": "fluid",
                    "model": f"test_sweep_block_{dimension}",
                    "sweep": "block",
                    "resolution": resolution,
                    "particles": None,
                    "backend": None,
                    "duration": 4.0*math.pi if dimension == "2d" else 10.0,
                    "cells": resolution**2 if dimension == "2d" else resolution**3,
                    "grid": [resolution, resolution, 1 if dimension == "2d" else resolution],
                    "cfl": 0.45,
                    "physics": ["transport"] if dimension == "2d" else ["transport", "diffusion"],
                })
    if group in {"all", "swarm"}:
        table = {"baseline": 100_000, "production": 1_000_000, "large": 10_000_000}
        durations = {"baseline": 2.0e-3, "production": 2.0e-5, "large": 3.0e-6}
        for name in scales:
            for backend in ("kdtree", "morton"):
                records.append({
                    "case": f"swarm_collision_{backend}_{name}",
                    "component": "swarm",
                    "model": "test_perf_collision_2d",
                    "sweep": None,
                    "resolution": None,
                    "particles": table[name],
                    "backend": backend,
                    "duration": durations[name],
                    "cells": None,
                    "grid": [100, 100, 1],
                    "cfl": None,
                    "collision_cfl": 0.01,
                    "neighbors": 200,
                    "physics": ["collision"],
                })
    return records


def build_case(project_root: Path, case: dict, target: str, output_dir: Path) -> tuple[Path, Path]:
    """Compile one configuration once for its warm-up and measured repetitions"""

    model = case["model"]
    model_root = project_root/"qav"/("fluid" if case["component"] == "fluid" else "swarm")/model
    output_tag = f"perf/{case['case']}"
    base = ["make", "-C", str(project_root), f"MODEL={model}", f"AMDGPU_TARGET={target}"]
    if case["sweep"]:
        base.append(f"FLUID_SWEEP={case['sweep']}")
    if case["backend"]:
        base.append(f"COLLISION_SEARCH={case['backend']}")
    if case["resolution"]:
        base.append(f"RES={case['resolution']}")
    if case["cfl"] is not None:
        base.append(f"CFL={case['cfl']:.17g}")
    if case["particles"]:
        base.append(f"PARTICLES={case['particles']}")
    base.extend(["SAVE=1", f"OUT_TIME={case['duration']:.17g}", f"OUT_TAG={output_tag}"])

    if case["component"] == "fluid":
        raw_output = project_root/"qav"/"fluid"/"out"/"block"/model/output_tag
    else:
        raw_output = project_root/"qav"/"swarm"/"out"/model/output_tag
    if raw_output.exists():
        shutil.rmtree(raw_output)

    clean = subprocess.run(
        [*base, "clean"], check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    print(clean.stdout, end="", flush=True)
    resolved = subprocess.run(
        [*base, "-n"], check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    build = subprocess.run(
        base, check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    print(build.stdout, end="", flush=True)
    write_json(output_dir/"build.json", {
        "command": base,
        "clean_return_code": clean.returncode,
        "clean_output": clean.stdout,
        "resolved_build_return_code": resolved.returncode,
        "resolved_build_commands": resolved.stdout,
        "build_return_code": build.returncode,
        "build_output": build.stdout,
        "passed": clean.returncode == 0 and resolved.returncode == 0 and build.returncode == 0,
    })
    if clean.returncode != 0 or resolved.returncode != 0 or build.returncode != 0:
        raise RuntimeError(f"{case['case']} did not compile; see {output_dir/'build.json'}")

    return model_root/"gamedev", raw_output


def summarize(case: dict, repetitions: list[dict], memory: dict) -> dict:
    """Calculate robust timing statistics and normalized evolution costs"""

    walls = [record["wall_seconds"] for record in repetitions]
    evolutions = [record["evolution_seconds"] for record in repetitions]
    median_wall = statistics.median(walls)
    median_evolution = statistics.median(evolutions)
    q1 = percentile(walls, 0.25)
    q3 = percentile(walls, 0.75)
    work_values = [record["accepted_work_units"] for record in repetitions]
    simulated_values = [record["completed_simulated_time"] for record in repetitions]
    work_units = work_values[0]
    simulated_time = simulated_values[0]
    work_consistent = len(set(work_values)) == 1
    simulated_consistent = all(
        math.isclose(value, simulated_time, rel_tol=0.0, abs_tol=1.0e-14)
        for value in simulated_values
    )
    normalized: dict[str, float | None] = {
        "simulated_time_per_wall_second": statistics.median([
            record["completed_simulated_time"]/record["wall_seconds"]
            for record in repetitions
        ]),
        "seconds_per_cell_step": None,
        "seconds_per_particle_batch": None,
    }
    if case["component"] == "fluid":
        normalized["seconds_per_cell_step"] = statistics.median([
            record["evolution_seconds"]/(record["accepted_work_units"]*case["cells"])
            for record in repetitions
        ])
    else:
        normalized["seconds_per_particle_batch"] = statistics.median([
            record["evolution_seconds"]/(record["accepted_work_units"]*case["particles"])
            for record in repetitions
        ])

    spread = (max(walls) - min(walls))/median_wall
    return {
        "case": case,
        "repeat_count": len(repetitions),
        "median_wall_seconds": median_wall,
        "wall_q1_seconds": q1,
        "wall_q3_seconds": q3,
        "wall_iqr_seconds": q3 - q1,
        "wall_spread_fraction": spread,
        "noisy": spread > 0.05,
        "median_evolution_seconds": median_evolution,
        "accepted_work_units": work_units,
        "accepted_work_units_by_repeat": work_values,
        "accepted_work_units_consistent": work_consistent,
        "completed_simulated_time": simulated_time,
        "completed_simulated_time_by_repeat": simulated_values,
        "completed_simulated_time_consistent": simulated_consistent,
        "normalized": normalized,
        "warmup_memory": memory,
        "passed": (
            all(record["passed"] for record in repetitions)
            and work_consistent
            and simulated_consistent
        ),
    }


def main() -> None:
    all_cases = {
        case["case"]: case
        for scale in ("baseline", "production", "large")
        for case in configurations("all", scale)
    }
    parser = argparse.ArgumentParser(description="run unprofiled ROCm performance benchmarks")
    parser.add_argument("--group", choices=("all", "fluid", "swarm"), default="all")
    parser.add_argument("--scale", choices=("baseline", "production", "large", "all"), default="baseline")
    parser.add_argument("--target", default=os.environ.get("AMDGPU_TARGET", "gfx942"))
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--progress-seconds", type=float, default=10.0)
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument(
        "--case", action="append", choices=tuple(all_cases),
        help="run one named case; repeat the option to select multiple cases",
    )
    parser.add_argument(
        "--duration", type=float,
        help="override simulated duration when exactly one --case is selected",
    )
    args = parser.parse_args()
    if args.repeat < 1:
        raise SystemExit("--repeat must be positive")
    if args.duration is not None and (args.case is None or len(args.case) != 1):
        raise SystemExit("--duration requires exactly one --case")
    if args.duration is not None and args.duration <= 0.0:
        raise SystemExit("--duration must be positive")

    project_root = Path(__file__).resolve().parents[2]
    perf_root = Path(__file__).resolve().parent
    out_root = perf_root/"out"
    out_root.mkdir(parents=True, exist_ok=True)
    started_utc = utc_now()
    environment = environment_record(project_root, args.target)
    write_json(out_root/"environment.json", environment)
    summaries = []

    selected_cases = (
        [dict(all_cases[name]) for name in args.case]
        if args.case is not None else configurations(args.group, args.scale)
    )
    if args.duration is not None:
        selected_cases[0]["duration"] = args.duration

    for case in selected_cases:
        print(f"\n=== {case['case']} ===", flush=True)
        case_dir = out_root/case["case"]
        if case_dir.exists():
            shutil.rmtree(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)
        executable, raw_output = build_case(project_root, case, args.target, case_dir)
        memory: dict = {"enabled": False, "peak_bytes": None}
        if not args.skip_warmup:
            print("[WARMUP] excluded from timing statistics", flush=True)
            warmup = run_executable(
                executable, project_root, case["component"], args.progress_seconds, True
            )
            memory = warmup["memory"]
            write_json(case_dir/"warmup.json", warmup)
            if not warmup["passed"]:
                raise RuntimeError(f"{case['case']} warm-up failed or completed fewer than 100 work units")

        repetitions = []
        for index in range(1, args.repeat + 1):
            print(f"[MEASURE] repetition={index}/{args.repeat}", flush=True)
            record = run_executable(
                executable, project_root, case["component"], args.progress_seconds, False
            )
            record["repetition"] = index
            write_json(case_dir/f"repeat_{index:02d}.json", record)
            if not record["passed"]:
                raise RuntimeError(
                    f"{case['case']} repetition {index} failed or completed fewer than 100 work units"
                )
            repetitions.append(record)

        summary = summarize(case, repetitions, memory)
        summary["raw_output"] = str(raw_output)
        write_json(case_dir/"summary.json", summary)
        summaries.append(summary)
        print(
            f"[RESULT] median={summary['median_wall_seconds']:.6f}s "
            f"IQR={summary['wall_iqr_seconds']:.6f}s spread={summary['wall_spread_fraction']:.3%}",
            flush=True,
        )

    passed = all(summary["passed"] for summary in summaries)
    manifest = {
        "suite": "rocm_performance",
        "instrumented": False,
        "target": args.target,
        "group": args.group,
        "scale": args.scale,
        "repeat": args.repeat,
        "warmup": not args.skip_warmup,
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "environment": "environment.json",
        "cases": [summary["case"]["case"] for summary in summaries],
        "summaries": [f"{summary['case']['case']}/summary.json" for summary in summaries],
        "passed": passed,
    }
    campaign_name = (
        "selected" if args.case is not None else f"{args.group}_{args.scale}"
    )
    campaign_manifest = out_root/f"manifest_{campaign_name}.json"
    write_json(campaign_manifest, manifest)
    write_json(out_root/"manifest.json", manifest)
    inventory_cases = []
    for summary_path in sorted(out_root.glob("*/summary.json")):
        summary_record = json.loads(summary_path.read_text())
        inventory_cases.append({
            "case": summary_path.parent.name,
            "summary": str(summary_path.relative_to(out_root)),
            "passed": summary_record.get("passed", False),
            "noisy": summary_record.get("noisy"),
        })
    write_json(out_root/"inventory.json", {
        "suite": "rocm_performance_inventory",
        "updated_utc": utc_now(),
        "cases": inventory_cases,
        "complete_cases": len(inventory_cases),
        "all_complete_cases_passed": all(case["passed"] for case in inventory_cases),
    })
    print(
        f"ROCm performance suite: {'PASS' if passed else 'FAIL'}  "
        f"manifest={campaign_manifest}"
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
