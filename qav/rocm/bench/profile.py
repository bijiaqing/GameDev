#!/usr/bin/env python3

"""Prepare and run one profiler-instrumented ROCm workload"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess

from run_suite import build_case, capture, configurations, write_json


def artifacts(root: Path) -> list[dict]:
    """List profiler-native artifacts without embedding their contents"""

    result = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            result.append({"path": str(path), "bytes": path.stat().st_size})
    return result


def available_cases() -> dict[str, dict]:
    """Return every baseline, production, and large benchmark configuration"""

    cases = []
    for scale in ("baseline", "production", "large"):
        cases.extend(configurations("all", scale))
    return {case["case"]: case for case in cases}


def main() -> None:
    cases = available_cases()
    parser = argparse.ArgumentParser(description="run one ROCm profiler collection")
    parser.add_argument("--case", choices=tuple(cases), required=True)
    parser.add_argument("--tool", choices=("trace", "compute"), default="trace")
    parser.add_argument("--kernel", help="kernel-name substring required by --tool compute")
    parser.add_argument("--target", default=os.environ.get("AMDGPU_TARGET", "gfx942"))
    parser.add_argument(
        "--duration", type=float,
        help="short simulated duration used only for profiler collection",
    )
    args = parser.parse_args()
    if args.tool == "compute" and not args.kernel:
        raise SystemExit("--kernel is required with --tool compute")

    project_root = Path(__file__).resolve().parents[3]
    case = dict(cases[args.case])
    if args.duration is not None:
        case["duration"] = args.duration
    elif args.tool == "trace":
        case["duration"] = (
            1.0 if case["component"] == "fluid" else min(case["duration"], 5.0e-5)
        )
    else:
        case["duration"] = (
            0.1 if case["component"] == "fluid" else case["duration"]/100.0
        )
    kernel_tag = None if args.kernel is None else "".join(
        character if character.isalnum() or character in "_-" else "_"
        for character in args.kernel
    )
    name = args.case if args.tool == "trace" else f"{args.case}_{kernel_tag}"
    output_dir = project_root/"qav"/"logs"/"bench"/"rocm"/"profiles"/name
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    executable, raw_output = build_case(project_root, case, args.target, output_dir)

    if args.tool == "trace":
        tool = shutil.which("rocprofv3")
        if tool is None:
            raise SystemExit("rocprofv3 is unavailable")
        native_dir = output_dir/"trace"
        native_dir.mkdir(parents=True, exist_ok=True)
        help_text = capture([tool, "--help"], project_root)
        summary_option = (
            "--stats" if "--stats" in help_text
            else "--summary" if "--summary" in help_text
            else None
        )
        command = [
            tool,
            "--runtime-trace",
        ]
        if summary_option is not None:
            command.append(summary_option)
        command.extend([
            "--output-format", "json",
            "--output-directory", str(native_dir),
            "--output-file", args.case,
            "--", str(executable),
        ])
    else:
        tool = shutil.which("rocprof-compute")
        if tool is None:
            raise SystemExit("rocprof-compute is unavailable")
        command = [
            tool, "profile",
            "--name", name,
            "--no-roof",
            "--kernel", args.kernel,
            "--", str(executable),
        ]

    started = datetime.now(timezone.utc).isoformat()
    environment = os.environ.copy()
    environment["ROCPROFCOMPUTE_COLOR"] = "0"
    result = subprocess.run(
        command, cwd=output_dir, check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=environment,
    )
    print(result.stdout, end="", flush=True)
    record = {
        "suite": "rocm_profile",
        "tier": "qualification",
        "instrumented": True,
        "timing_is_baseline": False,
        "case": case,
        "target": args.target,
        "tool": args.tool,
        "tool_version": capture([tool, "--version"], project_root),
        "kernel_filter": args.kernel,
        "command": command,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "return_code": result.returncode,
        "output": result.stdout,
        "raw_simulation_output": str(raw_output),
        "artifacts": artifacts(output_dir),
        "passed": result.returncode == 0,
    }
    write_json(output_dir/"manifest.json", record)
    print(f"profiler manifest: {output_dir/'manifest.json'}")
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
