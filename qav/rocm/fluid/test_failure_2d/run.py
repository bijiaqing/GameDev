#!/usr/bin/env python3

"""Exercise controlled fluid NaN and infinity rejection in subprocesses"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


STATE_FIELDS = ("rhod", "mx", "my", "mz", "lx", "vy", "lz", "optdepth")
CFL_FIELDS = ("rhod", "mx", "my", "mz", "lx", "vy", "lz")
BAD_VALUES = ("nan", "inf")


def utc_now() -> str:
    """Return one machine-readable UTC timestamp"""

    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, record: dict) -> None:
    """Write one result atomically below the ignored QA output tree"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_probe(
    executable: Path,
    mode: str,
    field: str,
    value: str,
    expected_diagnostic: str,
    expect_failure: bool,
) -> dict:
    """Run one isolated probe and check its process-level failure contract"""

    command = [str(executable), mode, field, value]
    timed_out = False
    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30.0,
        )
        return_code = result.returncode
        output = result.stdout
    except subprocess.TimeoutExpired as error:
        timed_out = True
        return_code = None
        output = error.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")

    output_lower = output.lower()
    runtime_fault = any(
        message in output_lower
        for message in ("hip error", "illegal memory access", "memory access fault")
    )
    if expect_failure:
        passed = (
            not timed_out
            and return_code not in (None, 0)
            and expected_diagnostic in output
            and not runtime_fault
        )
    else:
        passed = (
            not timed_out
            and return_code == 0
            and expected_diagnostic in output
            and not runtime_fault
        )

    return {
        "mode": mode,
        "field": field,
        "value": value,
        "command": command,
        "expected_failure": expect_failure,
        "expected_diagnostic": expected_diagnostic,
        "return_code": return_code,
        "timed_out": timed_out,
        "runtime_fault": runtime_fault,
        "output": output,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=os.environ.get("AMDGPU_TARGET", "gfx942"))
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    started_utc = utc_now()

    model_dir = Path(__file__).resolve().parent
    fluid_root = model_dir.parent
    project_root = fluid_root.parents[2]
    output_dir = project_root/"qav"/"logs"/"fluid"/"rocm"/"failure"/model_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    executable = model_dir/"gamedev"

    build_command = [
        "make", "-C", str(project_root), f"MODEL={model_dir.name}",
        "GPU_BACKEND=rocm", f"GPU_TARGET={args.target}",
    ]
    subprocess.run([*build_command, "clean"], check=True)
    build = subprocess.run(
        build_command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(build.stdout, end="", flush=True)
    write_json(
        output_dir/"build.json",
        {
            "command": build_command,
            "return_code": build.returncode,
            "output": build.stdout,
            "gpu_target": args.target,
        },
    )
    build.check_returncode()

    if args.build_only:
        print("fluid failure-path build: PASS", flush=True)
        return

    cases = [
        run_probe(executable, "state", "rhod", "finite", "clean state accepted", False),
        run_probe(executable, "cfl", "rhod", "finite", "clean CFL state accepted", False),
    ]

    for field in STATE_FIELDS:
        for value in BAD_VALUES:
            cases.append(
                run_probe(
                    executable,
                    "state",
                    field,
                    value,
                    "Error: non-finite simulation state at cell (2,1,0)",
                    True,
                )
            )

    for field in CFL_FIELDS:
        for value in BAD_VALUES:
            cases.append(
                run_probe(
                    executable,
                    "cfl",
                    field,
                    value,
                    "Error: non-finite dust state detected by CFL validation at cell (0,1,0)",
                    True,
                )
            )

    passed = all(case["passed"] for case in cases)
    manifest = {
        "suite": "fluid_failure",
        "tier": "qualification",
        "model": model_dir.name,
        "gpu_target": args.target,
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "cases": len(cases),
        "clean_controls": 2,
        "expected_failures": len(cases) - 2,
        "passed_cases": sum(case["passed"] for case in cases),
        "passed": passed,
        "records": cases,
    }
    write_json(output_dir/"manifest.json", manifest)

    for case in cases:
        status = "PASS" if case["passed"] else "FAIL"
        print(f"{status}  {case['mode']:5s} {case['field']:8s} {case['value']}")
    print(
        f"fluid failure paths: {'PASS' if passed else 'FAIL'}  "
        f"cases={manifest['passed_cases']}/{manifest['cases']}  "
        f"manifest={output_dir/'manifest.json'}",
        flush=True,
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
