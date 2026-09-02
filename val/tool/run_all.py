#!/usr/bin/env python3

"""Run one complete native validation matrix and prepare or compare its archive"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

sys.dont_write_bytecode = True

from val_config import source_fingerprint


def utc_now() -> str:
    """Return one ISO-formatted UTC timestamp for the campaign record"""

    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, record: dict[str, Any]) -> None:
    """Atomically update the campaign record after each command"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_stage(
    name: str,
    command: list[str],
    project_root: Path,
    environment: dict[str, str],
    campaign: dict[str, Any],
    campaign_path: Path,
) -> None:
    """Stream one child command while preserving its complete output in JSON"""

    stage: dict[str, Any] = {
        "name": name,
        "command": command,
        "status": "running",
        "started_utc": utc_now(),
        "finished_utc": None,
        "return_code": None,
        "output": "",
    }
    campaign["stages"].append(stage)
    write_json(campaign_path, campaign)
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
    write_json(campaign_path, campaign)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def main() -> None:
    val_root = Path(__file__).resolve().parents[1]
    project_root = val_root.parent
    parser = argparse.ArgumentParser(
        description=(
            "run one native fluid and swarm matrix and optionally compare a copied counterpart"
        ),
    )
    parser.add_argument("--backend", choices=("cuda", "rocm"), required=True)
    parser.add_argument("--res", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--target", help="GPU target; defaults to sm_80 or gfx942")
    parser.add_argument(
        "--fluid-sweep", choices=("thread", "block"), default="thread",
        help="fluid line solver used consistently by the native and cross-backend stages",
    )
    parser.add_argument("--skip-static", action="store_true")
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument(
        "--compare", action="store_true",
        help="require the copied counterpart archive and run cross-backend comparison",
    )
    comparison.add_argument(
        "--native-only", dest="compare", action="store_false", help=argparse.SUPPRESS,
    )
    parser.set_defaults(compare=False)
    args = parser.parse_args()

    target = args.target or ("sm_80" if args.backend == "cuda" else "gfx942")
    fingerprint, source_files = source_fingerprint(project_root)
    counterpart = "rocm" if args.backend == "cuda" else "cuda"
    counterpart_archive = val_root/"logs"/"fluid"/counterpart/args.fluid_sweep
    if args.compare and not counterpart_archive.is_dir():
        parser.error(
            f"copied {counterpart.upper()} archive does not exist: {counterpart_archive}; "
            f"copy val/logs from the {counterpart.upper()} system or omit --compare"
        )
    if args.compare:
        counterpart_campaign_path = val_root/"logs"/f"run_all_{counterpart}.json"
        try:
            counterpart_campaign = json.loads(counterpart_campaign_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            parser.error(
                f"invalid copied {counterpart.upper()} campaign "
                f"{counterpart_campaign_path}: {error}"
            )
        if not isinstance(counterpart_campaign, dict) or (
            counterpart_campaign.get("backend") != counterpart
            or counterpart_campaign.get("passed") is not True
            or counterpart_campaign.get("source_sha256") != fingerprint
            or counterpart_campaign.get("final_source_sha256") != fingerprint
        ):
            parser.error(
                f"copied {counterpart.upper()} campaign is incomplete or was produced from "
                "a different source tree"
            )
    environment = os.environ.copy()
    environment["CUDA_ARCH" if args.backend == "cuda" else "AMDGPU_TARGET"] = target
    environment["FLUID_SWEEP"] = args.fluid_sweep
    environment["VAL_SCOPE"] = "all"
    campaign_path = val_root/"logs"/f"run_all_{args.backend}.json"
    campaign: dict[str, Any] = {
        "schema": 1,
        "backend": args.backend,
        "gpu_target": target,
        "fluid_sweep": args.fluid_sweep,
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
    write_json(campaign_path, campaign)

    resolution_arguments = ["--res", *(str(value) for value in args.res)]
    quick_argument = ["--quick"] if args.quick else []
    target_argument = ["--target", target]
    backend_root = val_root/args.backend

    stages: list[tuple[str, list[str]]] = []
    if not args.skip_static:
        stages.append(("static", [sys.executable, str(val_root/"tool"/"static_check.py")]))
    if args.compare:
        stages.append((
            f"copied {counterpart.upper()} archive completeness",
            [
                sys.executable, str(val_root/"tool"/"check_archive.py"),
                "--backend", counterpart, "--component", "all",
                "--fluid-sweep", args.fluid_sweep,
                *(["--allow-partial"] if args.quick else []),
            ],
        ))
    stages.extend([
        (
            "fluid common matrix",
            [
                sys.executable, str(backend_root/"fluid"/"test_common"/"run_suite.py"),
                "--group", "all", *resolution_arguments, *quick_argument, *target_argument,
            ],
        ),
        (
            "swarm common matrix",
            [
                sys.executable, str(backend_root/"swarm"/"test_common"/"run_suite.py"),
                "--group", "all", *resolution_arguments, *quick_argument, *target_argument,
            ],
        ),
        (
            "swarm collision chain",
            [
                sys.executable, str(backend_root/"swarm"/"test_common"/"run_suite.py"),
                "--group", "chain", *target_argument,
            ],
        ),
    ])

    stages.append((
        "archive completeness",
        [
            sys.executable, str(val_root/"tool"/"check_archive.py"),
            "--backend", args.backend, "--component", "all",
            "--fluid-sweep", args.fluid_sweep,
            *(["--allow-partial"] if args.quick else []),
        ],
    ))

    if args.compare:
        stages.append((
            "complete backend comparison",
            [
                sys.executable, str(val_root/"tool"/"compare_backends.py"),
                "--cuda-root", str(val_root),
                "--rocm-root", str(val_root),
                "--cuda-sweep", args.fluid_sweep,
                "--rocm-sweep", args.fluid_sweep,
                "--component", "all", *(["--allow-partial"] if args.quick else []),
            ],
        ))

    try:
        for name, command in stages:
            run_stage(name, command, project_root, environment, campaign, campaign_path)
    except (KeyboardInterrupt, subprocess.CalledProcessError) as error:
        campaign["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        campaign["passed"] = False
        campaign["finished_utc"] = utc_now()
        write_json(campaign_path, campaign)
        if isinstance(error, KeyboardInterrupt):
            raise
        raise SystemExit(error.returncode)

    final_fingerprint, final_source_files = source_fingerprint(project_root)
    campaign["final_source_sha256"] = final_fingerprint
    campaign["final_source_files"] = final_source_files
    if final_fingerprint != fingerprint or final_source_files != source_files:
        campaign["status"] = "failed"
        campaign["passed"] = False
        campaign["finished_utc"] = utc_now()
        write_json(campaign_path, campaign)
        raise SystemExit("validation source changed while the campaign was running")

    campaign["status"] = "passed"
    campaign["passed"] = True
    campaign["finished_utc"] = utc_now()
    write_json(campaign_path, campaign)
    print(f"\nCOMPLETE VALIDATION: PASS  backend={args.backend}  campaign={campaign_path}")


if __name__ == "__main__":
    main()
