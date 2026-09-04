#!/usr/bin/env python3

"""Build and wall-time every linear-kernel coagulation model."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import subprocess
import time
from pathlib import Path


GROUP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = GROUP_DIR.parents[2]
TEMP_DIR = PROJECT_ROOT/"val"/"temp"/"coag"/"test_linear"
SUMMARY_PATH = PROJECT_ROOT/"val"/"logs"/"coagulation"/"test_linear"/"wall_time_summary.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_summary(summary: dict[str, object]) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = SUMMARY_PATH.with_name(f".{SUMMARY_PATH.name}.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(SUMMARY_PATH)


def main() -> None:
    models = sorted(path.parent.name for path in GROUP_DIR.glob("*/flags.mk"))
    records: list[dict[str, object]] = [
        {"model": model, "status": "pending"} for model in models
    ]
    summary: dict[str, object] = {
        "schema": 1,
        "suite": "coag/test_linear",
        "timing_scope": "wall time of each gamedev process; compilation excluded",
        "models_expected": len(models),
        "models_completed": 0,
        "status": "running",
        "started_utc": utc_now(),
        "finished_utc": None,
        "total_wall_seconds": 0.0,
        "models": records,
    }
    write_summary(summary)

    for index, model in enumerate(models):
        executable = TEMP_DIR/"bin"/model/"gamedev"
        print(f"\n=== build {model} ===", flush=True)
        build = subprocess.run(
            ["make", "-C", str(GROUP_DIR), f"MODEL={model}"],
            check=False,
        )

        record = records[index]
        record["executable"] = str(executable)
        if build.returncode != 0:
            record["status"] = "build_failed"
            record["build_return_code"] = build.returncode
            summary["status"] = "failed"
            summary["finished_utc"] = utc_now()
            write_summary(summary)
            raise SystemExit(build.returncode)

        record["status"] = "running"
        record["started_utc"] = utc_now()
        write_summary(summary)

        print(f"\n=== run {model} ===", flush=True)
        start_ns = time.perf_counter_ns()
        result = subprocess.run([str(executable)], cwd=PROJECT_ROOT, check=False)
        wall_seconds = (time.perf_counter_ns() - start_ns)/1.0e9

        record["finished_utc"] = utc_now()
        record["wall_seconds"] = wall_seconds
        record["return_code"] = result.returncode
        record["status"] = "passed" if result.returncode == 0 else "failed"
        summary["total_wall_seconds"] = sum(
            float(item.get("wall_seconds", 0.0)) for item in records
        )
        if result.returncode != 0:
            summary["status"] = "failed"
            summary["finished_utc"] = utc_now()
            write_summary(summary)
            raise SystemExit(result.returncode)

        summary["models_completed"] = index + 1
        write_summary(summary)

    summary["status"] = "passed"
    summary["finished_utc"] = utc_now()
    write_summary(summary)
    print(f"\nTiming summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
