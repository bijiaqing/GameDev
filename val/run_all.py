#!/usr/bin/env python3

"""run the complete native validation matrix for one backend and prepare or compare its archive"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from val_config import fluid_archive_sweep, model_output, portable, source_fingerprint

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    """return one ISO-formatted UTC timestamp for the campaign record"""

    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, record: dict[str, Any]) -> None:
    """atomically update the campaign record after each command"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    portable_record = portable(record, PROJECT_ROOT)
    temporary.write_text(json.dumps(portable_record, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_stage(
    name: str,
    command: list[str],
    project_root: Path,
    environment: dict[str, str],
    records: list[tuple[dict[str, Any], Path]],
) -> None:
    """stream one child command while preserving its complete output in every affected record"""

    stage: dict[str, Any] = {
        "name": name,
        "command": command,
        "status": "running",
        "started_utc": utc_now(),
        "finished_utc": None,
        "return_code": None,
        "output": "",
    }
    for campaign, path in records:
        campaign["stages"].append(stage)
        write_json(path, campaign)
    print(f"\n=== {name} ===", flush=True)

    process = subprocess.Popen(
        command,
        cwd=project_root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    if process.stdout is None:
        raise RuntimeError(f"failed to capture output for {name}")
    lines = []
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    return_code = process.wait()
    stage["output"] = "".join(lines)
    stage["return_code"] = return_code
    stage["finished_utc"] = utc_now()
    stage["status"] = "passed" if return_code == 0 else "failed"
    for campaign, path in records:
        write_json(path, campaign)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def check_counterpart(
    parser: argparse.ArgumentParser,
    val_root: Path,
    counterpart: str,
    sweep: str,
    fingerprint: str,
) -> None:
    """require a complete copied counterpart campaign for one fluid sweep from this source tree"""

    archive = model_output(
        val_root, "fluid", "_suite", counterpart, fluid_archive_sweep(counterpart, sweep)
    )
    if not archive.is_dir():
        parser.error(
            f"copied {counterpart.upper()} archive does not exist: {archive}; "
            f"copy val/fluid/out, val/swarm/out and val/*.json from the {counterpart.upper()} "
            "system or omit --compare"
        )
    path = val_root / f"run_all_{counterpart}_{sweep}.json"
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        parser.error(f"invalid copied {counterpart.upper()} campaign {path}: {error}")
    if not isinstance(record, dict) or (
        record.get("backend") != counterpart
        or record.get("fluid_sweep") != sweep
        or record.get("passed") is not True
        or record.get("source_sha256") != fingerprint
        or record.get("final_source_sha256") != fingerprint
    ):
        parser.error(
            f"copied {counterpart.upper()} {sweep} campaign is incomplete or was produced from "
            "a different source tree"
        )


def main() -> None:
    """run the fluid matrix for each requested sweep, the swarm matrices once, and archive checks"""

    val_root = Path(__file__).resolve().parent
    project_root = val_root.parent
    parser = argparse.ArgumentParser(
        description=(
            "run the native fluid matrix for each fluid sweep and the swarm matrices once, "
            "and optionally compare a copied counterpart"
        ),
    )
    parser.add_argument("--backend", choices=("cuda", "rocm"), required=True)
    parser.add_argument("--res", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--target", help="GPU target; defaults to sm_80 or gfx942")
    parser.add_argument(
        "--fluid-sweep",
        choices=("thread", "block", "both"),
        default="both",
        help="fluid line solver or solvers to validate; the swarm matrices run once in any case",
    )
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument(
        "--compare",
        action="store_true",
        help="require the copied counterpart archive and run cross-backend comparison",
    )
    comparison.add_argument(
        "--native-only",
        dest="compare",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(compare=False)
    args = parser.parse_args()

    sweeps = ["thread", "block"] if args.fluid_sweep == "both" else [args.fluid_sweep]
    target = args.target or ("sm_80" if args.backend == "cuda" else "gfx942")
    fingerprint, source_files = source_fingerprint(project_root)
    counterpart = "rocm" if args.backend == "cuda" else "cuda"
    if args.compare:
        for sweep in sweeps:
            check_counterpart(parser, val_root, counterpart, sweep, fingerprint)

    base_environment = os.environ.copy()
    base_environment["CUDA_ARCH" if args.backend == "cuda" else "AMDGPU_TARGET"] = target
    base_environment["GPU_BACKEND"] = args.backend
    base_environment["VAL_SCOPE"] = "all"

    # one campaign record per fluid sweep; the shared swarm stages are recorded in every one
    records: dict[str, tuple[dict[str, Any], Path]] = {}
    for sweep in sweeps:
        path = val_root / f"run_all_{args.backend}_{sweep}.json"
        campaign: dict[str, Any] = {
            "schema": 1,
            "backend": args.backend,
            "gpu_target": target,
            "fluid_sweep": sweep,
            "fluid_sweeps_in_run": sweeps,
            "python_version": platform.python_version(),
            "source_sha256": fingerprint,
            "source_files": source_files,
            "requested_resolutions": args.res,
            "quick": args.quick,
            "native_only": not args.compare,
            "comparison_requested": args.compare,
            "counterpart_backend": counterpart if args.compare else None,
            "campaign_tier": "publication",
            "included_tiers": ["publication"],
            "status": "running",
            "passed": None,
            "started_utc": utc_now(),
            "finished_utc": None,
            "stages": [],
        }
        write_json(path, campaign)
        records[sweep] = (campaign, path)

    resolution_arguments = ["--res", *(str(value) for value in args.res)]
    quick_argument = ["--quick"] if args.quick else []
    target_argument = ["--target", target]
    partial_argument = ["--allow-partial"] if args.quick else []

    # stage name, command, fluid sweep for the environment, and the sweeps whose records include it
    stages: list[tuple[str, list[str], str, list[str]]] = []
    if args.compare:
        for sweep in sweeps:
            stages.append(
                (
                    f"copied {counterpart.upper()} {sweep} archive completeness",
                    [
                        sys.executable,
                        str(val_root / "check_archive.py"),
                        "--backend",
                        counterpart,
                        "--component",
                        "all",
                        "--fluid-sweep",
                        fluid_archive_sweep(counterpart, sweep),
                        *partial_argument,
                    ],
                    sweep,
                    [sweep],
                )
            )
    for sweep in sweeps:
        stages.append(
            (
                f"fluid common matrix ({sweep})",
                [
                    sys.executable,
                    str(val_root / "fluid" / "src" / "run_suite.py"),
                    "--group",
                    "all",
                    *resolution_arguments,
                    *quick_argument,
                    *target_argument,
                ],
                sweep,
                [sweep],
            )
        )
    stages.append(
        (
            "swarm common matrix",
            [
                sys.executable,
                str(val_root / "swarm" / "src" / "run_suite.py"),
                "--group",
                "all",
                *resolution_arguments,
                *quick_argument,
                *target_argument,
            ],
            sweeps[0],
            sweeps,
        )
    )
    stages.append(
        (
            "swarm collision chain",
            [
                sys.executable,
                str(val_root / "swarm" / "src" / "run_suite.py"),
                "--group",
                "chain",
                *target_argument,
            ],
            sweeps[0],
            sweeps,
        )
    )
    for sweep in sweeps:
        stages.append(
            (
                f"archive completeness ({sweep})",
                [
                    sys.executable,
                    str(val_root / "check_archive.py"),
                    "--backend",
                    args.backend,
                    "--component",
                    "all",
                    "--fluid-sweep",
                    fluid_archive_sweep(args.backend, sweep),
                    *partial_argument,
                ],
                sweep,
                [sweep],
            )
        )
    if args.compare:
        for sweep in sweeps:
            stages.append(
                (
                    f"complete backend comparison ({sweep})",
                    [
                        sys.executable,
                        str(val_root / "compare_backends.py"),
                        "--cuda-root",
                        str(val_root),
                        "--rocm-root",
                        str(val_root),
                        "--cuda-sweep",
                        fluid_archive_sweep("cuda", sweep),
                        "--rocm-sweep",
                        sweep,
                        "--component",
                        "all",
                        *partial_argument,
                    ],
                    sweep,
                    [sweep],
                )
            )

    def finish(status: str, passed: bool) -> None:
        for campaign, path in records.values():
            campaign["status"] = status
            campaign["passed"] = passed
            campaign["finished_utc"] = utc_now()
            write_json(path, campaign)

    try:
        for name, command, env_sweep, owners in stages:
            environment = {**base_environment, "FLUID_SWEEP": env_sweep}
            run_stage(name, command, project_root, environment, [records[s] for s in owners])
    except (KeyboardInterrupt, subprocess.CalledProcessError) as error:
        finish("interrupted" if isinstance(error, KeyboardInterrupt) else "failed", False)
        if isinstance(error, KeyboardInterrupt):
            raise
        raise SystemExit(error.returncode) from error

    final_fingerprint, final_source_files = source_fingerprint(project_root)
    for campaign, _ in records.values():
        campaign["final_source_sha256"] = final_fingerprint
        campaign["final_source_files"] = final_source_files
    if final_fingerprint != fingerprint or final_source_files != source_files:
        finish("failed", False)
        raise SystemExit("validation source changed while the campaign was running")

    finish("passed", True)
    listed = ", ".join(str(path) for _, path in records.values())
    print(
        f"\nCOMPLETE VALIDATION: PASS  backend={args.backend}  sweeps={','.join(sweeps)}  "
        f"campaigns={listed}"
    )


if __name__ == "__main__":
    main()
