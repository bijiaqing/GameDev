#!/usr/bin/env python3

"""Build and execute the KNN correctness and performance matrix"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

QAV_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(QAV_ROOT/"tool"))

from qav_config import PUBLICATION_TIER, QUALIFICATION_TIER


def write_json(path: Path, record: dict) -> None:
    """Write one machine-readable component result"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def capture(command: list[str], cwd: Path) -> str:
    """Capture optional environment information without hiding test failures"""

    try:
        result = subprocess.run(
            command, cwd=cwd, check=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        return result.stdout.strip()
    except OSError as error:
        return f"unavailable: {error}"


def run_adversarial(command: list[str], component: str, output: Path) -> dict:
    """Run a small adversarial executable and preserve every labeled check"""

    result = subprocess.run(
        command, check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
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
        "passed": result.returncode == 0 and bool(checks) and all(
            check["passed"] for check in checks
        ),
    }
    write_json(output, record)
    if not record["passed"]:
        result.check_returncode()
        raise RuntimeError(f"{component} did not report a complete passing check list")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--res", nargs="+", type=int, default=[32],
        help="accepted for consistency with the common suite; KNN sizes are controlled by --full",
    )
    parser.add_argument("--arch", default="sm_80")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument(
        "--radial-only", action="store_true",
        help="run only the physical 1D radial search checks",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="also benchmark one million and ten million particles",
    )
    args = parser.parse_args()

    test_dir = Path(__file__).resolve().parent
    project_root = test_dir.parents[3]
    archive_root = project_root/"qav"/"logs"/"swarm"/"cuda"
    scope = os.environ.get("QAV_SCOPE", "manual")
    scope_root = archive_root if scope == "all" else archive_root/"groups"/scope
    output_root = scope_root/"test_knn"
    result_root = output_root/"radial" if args.radial_only and scope != "radial" \
        else output_root
    python = sys.executable

    if not args.build_only:
        if result_root.exists():
            shutil.rmtree(result_root)
        result_root.mkdir(parents=True)
        write_json(
            result_root/"environment.json",
            {
                "nvcc": capture(["nvcc", "--version"], project_root),
                "nvidia_smi": capture(
                    ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"],
                    project_root,
                ),
                "gpu_target": args.arch,
                "k": 200,
            },
        )

    # KNN capacities and backend headers are compile-time inputs that Make
    # cannot infer from a previously linked executable, so always rebuild the
    # standalone drivers for a clean, reproducible benchmark run
    subprocess.run(
        ["make", "-C", str(test_dir), "clean"], check=True
    )

    # The radial group needs only the ordinary benchmark and focused edge
    # driver; the general KNN group also builds periodic and wedge drivers.
    make_targets = ["all", "edge"] if args.radial_only else ["suite"]
    subprocess.run(
        [
            "make", "-C", str(test_dir), *make_targets, f"ARCH={args.arch}", "K=200",
        ],
        check=True,
    )

    # Compile and link the real collision translation units with each backend.
    # Distinct executable names prevent Make from mistaking an executable linked
    # against one backend's object directory for the other backend's result.
    production_models = (
        ("test_collision_1d",)
        if args.radial_only
        else ("test_collision_1d", "test_collision_2d", "test_collision_3d")
    )
    production_links = []
    for model in production_models:
        for backend in ("kdtree", "morton"):
            executable = test_dir/"bin"/f"production_{model}_{backend}"
            build_arguments = [
                "make", "-C", str(project_root),
                f"MODEL={model}", "GPU_BACKEND=cuda", "RES=32",
                f"COLLISION_SEARCH={backend}", f"EXEC={executable}",
            ]
            clean_result = subprocess.run([*build_arguments, "clean"], check=False)
            build_result = subprocess.run(build_arguments, check=False)
            link = {
                "model": model,
                "backend": backend,
                "executable": str(executable.relative_to(project_root)),
                "passed": clean_result.returncode == 0 and build_result.returncode == 0,
            }
            production_links.append(link)
            if not args.build_only:
                write_json(
                    result_root/"production_links.json",
                    {
                        "component": "production_links",
                        "tier": PUBLICATION_TIER,
                        "cases": len(production_links),
                        "links": production_links,
                        "passed": all(record["passed"] for record in production_links),
                    },
                )
            clean_result.check_returncode()
            build_result.check_returncode()
    if args.build_only:
        standalone_count = 2 if args.radial_only else 4
        print(
            f"\nknn build: PASS  standalone={standalone_count}/{standalone_count}  "
            f"production links={len(production_links)}/{len(production_links)}"
        )
        return

    bin_dir = test_dir/"bin"
    edge_command = [str(bin_dir/"knn_edge_tests")]
    if args.radial_only:
        edge_command.append("--radial-only")
    edge = run_adversarial(edge_command, "edge", result_root/"edge_results.json")
    periodic = None
    if not args.radial_only:
        periodic = run_adversarial(
            [str(bin_dir/"knn_periodic_tests")],
            "periodic", result_root/"periodic_results.json",
        )

    particles = [100_000, 1_000_000, 10_000_000] if args.full else [100_000]
    common = ["--particles", *(str(value) for value in particles), "--arch", args.arch]
    if args.radial_only and scope != "radial":
        common.extend(("--distribution", "radial", "--dim", "2", "--output-subdir", "radial"))
    elif args.radial_only:
        common.extend(("--distribution", "radial", "--dim", "2"))
    subprocess.run([python, str(test_dir/"run_benchmarks.py"), *common], check=True)
    analyze_root = project_root/"qav"/"comm"/"swarm"/"test_knn"
    analyze_command = [python, str(analyze_root/"analyze_results.py"), "--backend", "cuda"]
    if args.radial_only and scope != "radial":
        analyze_command.extend(("--output-subdir", "radial"))
    subprocess.run(analyze_command, check=True)

    ordinary = json.loads((result_root/"manifest.json").read_text())
    wedge = None
    passed = ordinary["passed"] and edge["passed"]
    if args.radial_only:
        pass
    else:
        subprocess.run([python, str(test_dir/"run_wedge_benchmarks.py"), *common], check=True)
        subprocess.run([python, str(analyze_root/"analyze_wedge.py"), "--backend", "cuda"], check=True)
        wedge = json.loads((output_root/"wedge"/"manifest.json").read_text())
        passed = passed and periodic["passed"] and wedge["passed"]

    production_passed = all(record["passed"] for record in production_links)
    passed = passed and production_passed
    suite = {
        "component": "knn",
        "tier": PUBLICATION_TIER,
        "extended_tier": QUALIFICATION_TIER if args.full else None,
        "scope": "radial" if args.radial_only else "general",
        "particles": particles,
        "environment": "environment.json",
        "standalone_builds_passed": True,
        "ordinary": {"cases": ordinary["cases"], "passed": ordinary["passed"]},
        "edge": {"cases": edge["cases"], "passed": edge["passed"]},
        "periodic": None if periodic is None else {
            "cases": periodic["cases"], "passed": periodic["passed"],
        },
        "wedge": None if wedge is None else {
            "cases": wedge["cases"], "passed": wedge["passed"],
        },
        "production_links": {
            "cases": len(production_links), "passed": production_passed,
        },
        "passed": passed,
    }
    write_json(result_root/"suite_manifest.json", suite)

    label = "knn radial" if args.radial_only else "knn"
    components = [
        f"ordinary={ordinary['cases']}/{ordinary['cases']}",
        f"edge={edge['passed_cases']}/{edge['cases']}",
    ]
    if periodic is not None:
        components.append(f"periodic={periodic['passed_cases']}/{periodic['cases']}")
    if wedge is not None:
        components.append(f"wedge={wedge['cases']}/{wedge['cases']}")
    components.append(f"production links={len(production_links)}/{len(production_links)}")
    print(f"\n{label}: {'PASS' if passed else 'FAIL'}  " + "  ".join(components))
    if not passed:
        raise SystemExit("one or more KNN benchmark cases failed")


if __name__ == "__main__":
    main()
