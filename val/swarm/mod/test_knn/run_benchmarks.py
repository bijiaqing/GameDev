#!/usr/bin/env python3

"""Build and run the QA KD-tree versus Morton KNN experiment"""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True
import os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
import json
import subprocess
from pathlib import Path

VAL_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(VAL_ROOT))
from val_config import model_output

from val_config import BACKEND, TARGET_ENV, DEFAULT_TARGET, backend_environment


def capture(command: list[str], cwd: Path) -> str:
    """Capture optional cluster information without hiding benchmark failures"""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--particles", nargs="+", type=int,
        default=[100_000, 1_000_000, 10_000_000],
    )
    parser.add_argument("--distribution", nargs="+", choices=("smooth", "ring", "clump", "radial"), default=None)
    parser.add_argument("--dim", nargs="+", type=int, choices=(2, 3), default=[2, 3])
    parser.add_argument("--queries", type=int, default=4096)
    parser.add_argument("--brute-queries", type=int, default=32)
    parser.add_argument("--radius", type=float, default=0.1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--leaf-target", type=int, default=128)
    parser.add_argument("--max-level", type=int, default=20)
    parser.add_argument("--max-leaf-scan", type=int, default=4096)
    parser.add_argument("--target", default=os.environ.get(TARGET_ENV, DEFAULT_TARGET))
    parser.add_argument("--k", type=int, default=200)
    parser.add_argument(
        "--output-subdir", default="",
        help="store a selected matrix below a separate output directory",
    )
    args = parser.parse_args()

    test_root = Path(__file__).resolve().parent
    project_root = test_root.parents[3]
    executable = project_root/"val"/"swarm"/"obj"/"test_knn"/BACKEND/"knn_benchmark"
    scope = os.environ.get("VAL_SCOPE", "manual")
    result_root = model_output(project_root/"val", "swarm", "test_knn", BACKEND, scope=scope)
    if args.output_subdir:
        output_subdir = Path(args.output_subdir)
        if output_subdir.is_absolute() or ".." in output_subdir.parts:
            parser.error("--output-subdir must remain below the test_knn result directory")
        result_root = result_root/output_subdir
    result_root.mkdir(parents=True, exist_ok=True)

    distributions = args.distribution or ["smooth", "ring", "clump", "radial"]

    # k is a compile-time top-K capacity for both backends, so compile once for
    # the complete matrix and then vary only runtime particle distributions
    subprocess.run(
        [
            "make", "-C", str(test_root), f"GPU_TARGET={args.target}", f"GPU_BACKEND={BACKEND}", f"K={args.k}",
        ],
        check=True,
    )

    environment = {**backend_environment(capture, test_root), "gpu_target": args.target, "k": args.k}
    (result_root / "environment.txt").unlink(missing_ok=True)
    (result_root / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n"
    )

    completed = []
    for dimension in args.dim:
        for distribution in distributions:
            if distribution == "radial" and dimension != 2:
                continue
            for particles in args.particles:
                physical_dimension = 1 if distribution == "radial" else dimension
                name = f"{distribution}_{physical_dimension}d_N{particles}"
                output = result_root / f"{name}.json"
                command = [
                    str(executable),
                    "--particles", str(particles),
                    "--queries", str(min(args.queries, particles)),
                    "--brute-queries", str(min(args.brute_queries, args.queries, particles)),
                    "--repeat", str(args.repeat),
                    "--dim", str(dimension),
                    "--distribution", distribution,
                    "--radius", str(args.radius),
                    "--leaf-target", str(args.leaf_target),
                    "--max-level", str(args.max_level),
                    "--max-leaf-scan", str(args.max_leaf_scan),
                    "--output", str(output),
                ]
                print(f"\n=== {name} ===", flush=True)
                output.unlink(missing_ok=True)
                result = subprocess.run(command, cwd=test_root.parent, check=False)
                if not output.exists():
                    result.check_returncode()
                    raise RuntimeError(f"ordinary benchmark did not write {output}")
                record = json.loads(output.read_text())
                if (
                    record.get("distribution") != distribution
                    or record.get("particles") != particles
                    or record.get("dimension") != dimension
                    or record.get("physical_dimension") != physical_dimension
                    or record.get("k") != args.k
                    or record.get("passed") != record.get("quality_passed")
                ):
                    raise RuntimeError(
                        f"ordinary benchmark labels in {output} do not match case {name}"
                    )
                record["tier"] = "publication" if particles == 100_000 else "qualification"
                output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
                completed.append((name, record))
                print(
                    f"pass={record['passed']}  "
                    f"query batch: KD={record['kd_query_ms']:.3f} ms  "
                    f"Morton={record['morton_query_ms']:.3f} ms  "
                    f"KD/Morton={record['query_time_ratio_kd_morton']:.3f}"
                )
                if result.returncode != 0:
                    print(
                        "failure counters: "
                        f"KD/brute={record['kd_brute_mismatches']}  "
                        f"Morton/brute={record['morton_brute_mismatches']}  "
                        f"KD/disagreements={record['kd_disagreement_brute_mismatches']}  "
                        f"Morton/disagreements={record['morton_disagreement_brute_mismatches']}  "
                        f"Morton/records={record['morton_record_mismatches']}  "
                        f"records/geometry={record['record_geometry_mismatches']}  "
                        f"first-query={record['first_failure_query']}  "
                        f"overflows={record['stack_overflows']}  "
                        f"max-error={record['maximum_distance_error']:.3e}",
                        flush=True,
                    )
                    result.check_returncode()

    passed = all(record["passed"] for _, record in completed)
    manifest = {
        "component": "ordinary",
        "tier": "publication",
        "extended_tier": "qualification" if any(
            particles != 100_000 for particles in args.particles
        ) else None,
        "cases": len(completed),
        "all_quality_passed": passed,
        "passed": passed,
        "environment": "environment.json",
        "files": [f"{name}.json" for name, _ in completed],
    }
    (result_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
