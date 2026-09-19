#!/usr/bin/env python3

"""Select verification cases and invoke each model-local runner in sequence

This is the top-level entry point used by ``--group all`` and the smaller
physics groups.  It does not compile or validate results itself.  Instead, it
builds a list of model/parameter combinations and delegates each combination
to the short ``run.py`` wrapper inside that model directory.
"""

# store type annotations without evaluating them at import time; this keeps
# annotations such as ``list[tuple[str, list[str]]]`` as documentation and has
# no effect on the numerical tests
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
import os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
from datetime import datetime, timezone
import json
import subprocess
from pathlib import Path


VAL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(VAL_ROOT))
from val_config import model_output

from val_config import BACKEND, TARGET_ENV, DEFAULT_TARGET, fluid_archive_sweep

from val_config import FLUID_GROUPS, fluid_case_tier, fluid_cases, fluid_metric_tiers


def utc_now() -> str:
    """Return one ISO-formatted UTC timestamp for suite manifests"""

    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, record: dict[str, object]) -> None:
    """Atomically update one machine-readable suite record"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    # ``nargs="+"`` accepts one or more values after --res, for example
    # ``--res 32 64 128 256``; --quick keeps only the first two of those
    # resolutions, which is useful for checking the workflow before a full run
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--quick", action="store_true", help="Use only the two coarsest requested resolutions")
    parser.add_argument(
        "--group", choices=("all", *FLUID_GROUPS),
        default="all",
    )
    parser.add_argument("--target", default=os.environ.get(TARGET_ENV, DEFAULT_TARGET))
    args = parser.parse_args()

    run_environment = os.environ.copy()
    run_environment["GPU_BACKEND"] = BACKEND
    run_environment[TARGET_ENV] = args.target
    run_environment["VAL_SCOPE"] = args.group

    resolutions = args.res[:2] if args.quick else args.res

    common = Path(__file__).resolve().parent
    model_root = common.parent/"mod"
    project_root = common.parents[2]

    commands = fluid_cases(args.group)
    selected_sweep = os.environ.get("FLUID_SWEEP", "thread")
    archive_sweep = fluid_archive_sweep(BACKEND, selected_sweep)
    scope_root = model_output(project_root/"val", "fluid", "_suite", BACKEND, archive_sweep, args.group)
    manifest_path = scope_root/("manifest_all.json" if args.group == "all" else "manifest.json")
    entries = [
        {
            "model": model,
            "arguments": list(extra),
            "tier": fluid_case_tier(model, extra),
            "status": "pending",
        }
        for model, extra in commands
    ]
    metric_tiers = fluid_metric_tiers(commands, len(resolutions))
    included_tiers = [tier for tier, count in metric_tiers.items() if count > 0]
    manifest: dict[str, object] = {
        "schema": 1,
        "suite": "fluid",
        "backend": BACKEND,
        "group": args.group,
        "sweep": archive_sweep,
        "requested_resolutions": args.res,
        "effective_resolutions": resolutions,
        "math_mode": "precise",
        "gpu_target": args.target,
        "campaign_tier": "publication",
        "included_tiers": included_tiers,
        "metric_tiers": metric_tiers,
        "cases_expected": len(entries),
        "cases_completed": 0,
        "status": "running",
        "passed": None,
        "started_utc": utc_now(),
        "finished_utc": None,
        "cases": entries,
    }
    write_json(manifest_path, manifest)

    for index, (model, extra) in enumerate(commands):
        # sys.executable reuses the Python interpreter that launched this suite,
        # avoiding accidental changes of environment between the two scripts
        command = [
            sys.executable, str(model_root/model/"run.py"),
        ]

        # fixed-resolution source cases override the suite-wide convergence grid
        if "--res" not in extra:
            command += ["--res", *(str(value) for value in resolutions)]
        command += list(extra)
        print("\n===", model, " ".join(extra), "===", flush=True)

        # check=True stops the suite immediately if compilation, execution, or
        # validation of any case fails, so later output cannot hide that failure
        entries[index]["status"] = "running"
        write_json(manifest_path, manifest)
        result = subprocess.run(command, check=False, env=run_environment)
        if result.returncode != 0:
            entries[index]["status"] = "failed"
            entries[index]["return_code"] = result.returncode
            manifest["status"] = "failed"
            manifest["passed"] = False
            manifest["finished_utc"] = utc_now()
            write_json(manifest_path, manifest)
            raise SystemExit(result.returncode)
        entries[index]["status"] = "passed"
        entries[index]["return_code"] = 0
        manifest["cases_completed"] = index + 1
        write_json(manifest_path, manifest)

    manifest["status"] = "passed"
    manifest["passed"] = True
    manifest["finished_utc"] = utc_now()
    write_json(manifest_path, manifest)
    print(f"\nFLUID TEST SUITE: PASS  cases={len(entries)}/{len(entries)}  manifest={manifest_path}")


if __name__ == "__main__":
    main()
