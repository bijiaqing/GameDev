#!/usr/bin/env python3

"""build and execute the publication KNN correctness matrix"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

VAL_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(VAL_ROOT))
from val_config import (
    BACKEND,
    DEFAULT_TARGET,
    PUBLICATION_TIER,
    TARGET_ENV,
    backend_environment,
    model_output,
)


def write_json(path: Path, record: dict) -> None:
    """write one machine-readable component result"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def capture(command: list[str], cwd: Path) -> str:
    """capture optional environment information without hiding test failures"""

    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return result.stdout.strip()
    except OSError as error:
        return f"unavailable: {error}"


def run_adversarial(command: list[str], component: str, output: Path) -> dict:
    """run a small adversarial executable and preserve every labeled check"""

    result = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(result.stdout, end="")
    checks = [
        {"case": name, "passed": status == "PASS"}
        for status, name in re.findall(r"^(PASS|FAIL)\s+(\S+)\s*$", result.stdout, re.MULTILINE)
    ]
    record = {
        "component": component,
        "tier": PUBLICATION_TIER,
        "cases": len(checks),
        "passed_cases": sum(check["passed"] for check in checks),
        "checks": checks,
        "passed": result.returncode == 0
        and bool(checks)
        and all(check["passed"] for check in checks),
    }
    write_json(output, record)
    if not record["passed"]:
        result.check_returncode()
        raise RuntimeError(f"{component} did not report a complete passing check list")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", nargs="+", type=int, default=[32], help=argparse.SUPPRESS)
    parser.add_argument("--target", default=os.environ.get(TARGET_ENV, DEFAULT_TARGET))
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()

    test_dir = Path(__file__).resolve().parent
    project_root = test_dir.parents[3]
    build_root = project_root / "val" / "swarm" / "obj" / "test_knn" / BACKEND
    scope = os.environ.get("VAL_SCOPE", "manual")
    output_root = model_output(project_root / "val", "swarm", "test_knn", BACKEND, scope=scope)
    result_root = output_root
    python = sys.executable

    if not args.build_only:
        if result_root.exists():
            shutil.rmtree(result_root)
        result_root.mkdir(parents=True)
        write_json(
            result_root / "environment.json",
            {**backend_environment(capture, project_root), "gpu_target": args.target, "k": 200},
        )

    # KNN capacities and backend headers are compile-time inputs that Make
    # cannot infer from a previously linked executable, so always rebuild the
    # standalone correctness drivers
    subprocess.run(
        [
            "make",
            "-C",
            str(test_dir),
            f"GPU_BACKEND={BACKEND}",
            f"GPU_TARGET={args.target}",
            "clean",
        ],
        check=True,
    )

    subprocess.run(
        [
            "make",
            "-C",
            str(test_dir),
            "suite",
            f"GPU_TARGET={args.target}",
            f"GPU_BACKEND={BACKEND}",
            "K=200",
        ],
        check=True,
    )
    if args.build_only:
        print("\nknn build: PASS  standalone=4/4")
        return

    edge = run_adversarial(
        [str(build_root / "knn_edge_tests")],
        "edge",
        result_root / "edge_results.json",
    )
    periodic = run_adversarial(
        [str(build_root / "knn_periodic_tests")],
        "periodic",
        result_root / "periodic_results.json",
    )

    particles = [100_000]
    common = ["--particles", *(str(value) for value in particles), "--target", args.target]
    subprocess.run([python, str(test_dir / "run_benchmarks.py"), *common], check=True)
    analyze_root = project_root / "val" / "swarm" / "mod" / "test_knn"
    analyze_command = [python, str(analyze_root / "analyze_results.py"), "--backend", BACKEND]
    subprocess.run(analyze_command, check=True)

    ordinary = json.loads((result_root / "manifest.json").read_text())
    subprocess.run([python, str(test_dir / "run_wedge_benchmarks.py"), *common], check=True)
    subprocess.run(
        [python, str(analyze_root / "analyze_wedge.py"), "--backend", BACKEND], check=True
    )
    wedge = json.loads((output_root / "wedge" / "manifest.json").read_text())
    passed = ordinary["passed"] and edge["passed"] and periodic["passed"] and wedge["passed"]
    suite = {
        "component": "knn",
        "tier": PUBLICATION_TIER,
        "particles": particles,
        "environment": "environment.json",
        "standalone_builds_passed": True,
        "ordinary": {"cases": ordinary["cases"], "passed": ordinary["passed"]},
        "edge": {"cases": edge["cases"], "passed": edge["passed"]},
        "periodic": {
            "cases": periodic["cases"],
            "passed": periodic["passed"],
        },
        "wedge": {
            "cases": wedge["cases"],
            "passed": wedge["passed"],
        },
        "passed": passed,
    }
    write_json(result_root / "suite_manifest.json", suite)

    components = [
        f"ordinary={ordinary['cases']}/{ordinary['cases']}",
        f"edge={edge['passed_cases']}/{edge['cases']}",
        f"periodic={periodic['passed_cases']}/{periodic['cases']}",
        f"wedge={wedge['cases']}/{wedge['cases']}",
    ]
    print(f"\nknn: {'PASS' if passed else 'FAIL'}  " + "  ".join(components))
    if not passed:
        raise SystemExit("one or more KNN benchmark cases failed")


if __name__ == "__main__":
    main()
