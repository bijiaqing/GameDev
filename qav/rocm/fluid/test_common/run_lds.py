#!/usr/bin/env python3

"""Build and execute the native large-LDS fluid qualification cases"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess


POSITIVE_MODELS = ("test_lds_x", "test_lds_y", "test_lds_z")
REJECT_MODEL = "test_lds_reject"


def utc_now() -> str:
    """Return one machine-readable UTC timestamp"""

    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, record: dict) -> None:
    """Write one result atomically below the ignored QA output tree"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def build_model(project_root: Path, model: str, target: str, output_dir: Path) -> tuple[Path, dict]:
    """Clean and build one model while preserving compiler output as JSON"""

    executable = project_root/"qav"/"rocm"/"fluid"/model/"gamedev"
    base = [
        "make", "-C", str(project_root), f"MODEL={model}", "GPU_BACKEND=rocm",
        f"GPU_TARGET={target}", "QAV_SCOPE=all",
    ]
    clean = subprocess.run(
        [*base, "clean"], check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    print(clean.stdout, end="", flush=True)
    build = subprocess.run(
        base, check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    print(build.stdout, end="", flush=True)
    record = {
        "model": model,
        "gpu_target": target,
        "clean_command": [*base, "clean"],
        "clean_return_code": clean.returncode,
        "clean_output": clean.stdout,
        "build_command": base,
        "build_return_code": build.returncode,
        "build_output": build.stdout,
        "passed": clean.returncode == 0 and build.returncode == 0,
    }
    write_json(output_dir/f"build_{model}.json", record)
    if not record["passed"]:
        raise RuntimeError(f"{model} did not compile; see {output_dir/f'build_{model}.json'}")
    return executable, record


def run_model(executable: Path, timeout: float) -> dict:
    """Run one model and capture its process-level launch result"""

    timed_out = False
    try:
        result = subprocess.run(
            [str(executable)], check=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout,
        )
        output = result.stdout
        return_code = result.returncode
    except subprocess.TimeoutExpired as error:
        timed_out = True
        return_code = None
        output = error.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
    print(output, end="", flush=True)
    return {
        "command": [str(executable)],
        "return_code": return_code,
        "timed_out": timed_out,
        "output": output,
    }


def build_summary(build: dict, record_path: Path) -> dict:
    """Reference the complete build JSON without duplicating compiler streams"""

    return {
        "record": str(record_path),
        "clean_return_code": build["clean_return_code"],
        "build_return_code": build["build_return_code"],
        "passed": build["passed"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=os.environ.get("AMDGPU_TARGET", "gfx942"))
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()

    common = Path(__file__).resolve().parent
    project_root = common.parents[3]
    output_dir = project_root/"qav"/"logs"/"fluid"/"rocm"/"lds"
    started_utc = utc_now()
    records = []

    for model in (*POSITIVE_MODELS, REJECT_MODEL):
        print(f"\n=== {model} ===", flush=True)
        executable, build = build_model(project_root, model, args.target, output_dir)
        build_record = output_dir/f"build_{model}.json"
        if args.build_only:
            records.append({
                "model": model,
                "build": build_summary(build, build_record),
                "passed": True,
            })
            continue

        metrics_path = project_root/"qav"/"logs"/"fluid"/"rocm"/"block"/model/"metrics.json"
        if model in POSITIVE_MODELS:
            metrics_path.unlink(missing_ok=True)
        process = run_model(executable, args.timeout)
        runtime_fault = any(
            message in process["output"].lower()
            for message in ("hip error", "illegal memory access", "memory access fault")
        )
        if model in POSITIVE_MODELS:
            metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else None
            passed = (
                not process["timed_out"] and process["return_code"] == 0
                and not runtime_fault and metrics is not None and metrics.get("passed", False)
            )
        else:
            metrics = None
            expected = "Error: diffusion_ybl requires 65568 dynamic LDS bytes"
            passed = (
                not process["timed_out"] and process["return_code"] not in (None, 0)
                and expected in process["output"] and not runtime_fault
            )

        records.append({
            "model": model,
            "expected_rejection": model == REJECT_MODEL,
            "build": build_summary(build, build_record),
            "process": process,
            "runtime_fault": runtime_fault,
            "metrics": metrics,
            "passed": passed,
        })
        print(f"{model}: {'PASS' if passed else 'FAIL'}", flush=True)

    passed = all(record["passed"] for record in records)
    manifest = {
        "suite": "fluid_large_lds",
        "tier": "qualification",
        "gpu_target": args.target,
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "build_only": args.build_only,
        "positive_launches": len(POSITIVE_MODELS),
        "expected_rejections": 1,
        "passed_cases": sum(record["passed"] for record in records),
        "cases": len(records),
        "records": records,
        "passed": passed,
    }
    write_json(output_dir/"manifest.json", manifest)
    print(
        f"fluid large-LDS qualification: {'PASS' if passed else 'FAIL'}  "
        f"cases={manifest['passed_cases']}/{manifest['cases']}  "
        f"manifest={output_dir/'manifest.json'}",
        flush=True,
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
