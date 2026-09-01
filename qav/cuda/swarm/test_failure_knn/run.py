#!/usr/bin/env python3

"""Verify deterministic CUDA rejection of nonfinite collision-search inputs"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import subprocess
from pathlib import Path


PARTICLE_FIELDS = (
    "position_x", "position_y", "position_z",
    "velocity_x", "velocity_y", "velocity_z",
    "par_size", "par_numr",
)
RESULT_FIELDS = ("col_rate", "col_dist")
VALUES = ("nan", "inf")
IDX_BAD = 7


def utc_now() -> str:
    """Return an ISO UTC timestamp for the machine-readable record"""

    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict) -> None:
    """Atomically write one JSON artifact"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_command(command: list[str], cwd: Path, timeout: int = 30) -> dict:
    """Run one bounded subprocess and retain its complete combined stream"""

    started = utc_now()
    try:
        result = subprocess.run(
            command, cwd=cwd, text=True, timeout=timeout,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        return {
            "command": command,
            "return_code": result.returncode,
            "timed_out": False,
            "output": result.stdout,
            "started_utc": started,
            "finished_utc": utc_now(),
        }
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        return {
            "command": command,
            "return_code": None,
            "timed_out": True,
            "output": output,
            "started_utc": started,
            "finished_utc": utc_now(),
        }


def assess(record: dict, expected_return: int, diagnostic: str) -> bool:
    """Apply the expected-failure contract to one subprocess"""

    output = record["output"]
    forbidden = ("illegal memory access", "CUDA error", "unspecified launch failure")
    return (
        not record["timed_out"]
        and record["return_code"] == expected_return
        and diagnostic in output
        and not any(message.lower() in output.lower() for message in forbidden)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", nargs="+", type=int, default=[32])
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--target", default=os.environ.get("CUDA_ARCH", "sm_80"))
    args = parser.parse_args()

    model_dir = Path(__file__).resolve().parent
    project_root = model_dir.parents[3]
    archive_root = project_root/"qav"/"logs"/"swarm"/"cuda"
    scope = os.environ.get("QAV_SCOPE", "manual")
    scope_root = archive_root if scope == "all" else archive_root/"groups"/scope
    out_dir = scope_root/model_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    executable = model_dir/"gamedev"

    manifest = {
        "model": model_dir.name,
        "tier": "qualification",
        "gpu_target": args.target,
        "injected_particle": IDX_BAD,
        "backends": {},
        "passed": True,
        "started_utc": utc_now(),
        "finished_utc": None,
    }

    for backend in ("kdtree", "morton"):
        clean = run_command(
            ["make", "clean", f"MODEL={model_dir.name}", "GPU_BACKEND=cuda", f"COLLISION_SEARCH={backend}",
             f"GPU_TARGET={args.target}"],
            project_root, timeout=120,
        )
        build = run_command(
            ["make", f"MODEL={model_dir.name}", "GPU_BACKEND=cuda", f"COLLISION_SEARCH={backend}",
             f"GPU_TARGET={args.target}"],
            project_root, timeout=600,
        )
        build_passed = clean["return_code"] == 0 and build["return_code"] == 0
        backend_record = {
            "build_passed": build_passed,
            "clean": clean,
            "build": build,
            "cases": [],
        }
        manifest["backends"][backend] = backend_record
        if not build_passed:
            manifest["passed"] = False
            continue
        if args.build_only:
            continue

        controls = (
            ("site", "none", "finite", "clean collision search input accepted"),
            ("result", "none", "finite", "clean collision result accepted"),
        )
        for mode, field, value, diagnostic in controls:
            record = run_command([str(executable), mode, field, value], project_root)
            record.update({"mode": mode, "field": field, "value": value})
            record["passed"] = assess(record, 0, diagnostic)
            backend_record["cases"].append(record)

        for field in PARTICLE_FIELDS:
            for value in VALUES:
                diagnostic = (
                    f"Error: non-finite particle state before collision search at particle {IDX_BAD}"
                )
                record = run_command([str(executable), "site", field, value], project_root)
                record.update({"mode": "site", "field": field, "value": value})
                record["passed"] = assess(record, 1, diagnostic)
                backend_record["cases"].append(record)

        for field in RESULT_FIELDS:
            for value in VALUES:
                diagnostic = f"Error: non-finite collision result at particle {IDX_BAD}"
                record = run_command([str(executable), "result", field, value], project_root)
                record.update({"mode": "result", "field": field, "value": value})
                record["passed"] = assess(record, 1, diagnostic)
                backend_record["cases"].append(record)

        backend_record["passed"] = all(case["passed"] for case in backend_record["cases"])
        manifest["passed"] = manifest["passed"] and backend_record["passed"]

    manifest["finished_utc"] = utc_now()
    write_json(out_dir/"manifest.json", manifest)

    if args.build_only:
        print(
            f"{model_dir.name}: {'BUILD PASS' if manifest['passed'] else 'BUILD FAIL'} "
            f"backends={len(manifest['backends'])}",
            flush=True,
        )
    else:
        case_count = sum(len(value["cases"]) for value in manifest["backends"].values())
        print(
            f"{model_dir.name}: {'PASS' if manifest['passed'] else 'FAIL'} "
            f"backends={len(manifest['backends'])} cases={case_count}",
            flush=True,
        )
    if not manifest["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
