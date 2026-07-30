#!/usr/bin/env python3

"""Build and run the QA KD-tree versus Morton KNN experiment"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


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
    parser.add_argument("--particles", nargs="+", type=int, default=[100_000, 1_000_000])
    parser.add_argument("--distribution", nargs="+", choices=("smooth", "ring", "clump"), default=None)
    parser.add_argument("--dim", nargs="+", type=int, choices=(2, 3), default=[2, 3])
    parser.add_argument("--queries", type=int, default=4096)
    parser.add_argument("--brute-queries", type=int, default=32)
    parser.add_argument("--radius", type=float, default=0.1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--leaf-target", type=int, default=128)
    parser.add_argument("--max-level", type=int, default=20)
    parser.add_argument("--max-leaf-scan", type=int, default=4096)
    parser.add_argument("--arch", default="sm_80")
    parser.add_argument("--k", type=int, default=200)
    args = parser.parse_args()

    test_root = Path(__file__).resolve().parent
    executable = test_root / "bin" / "knn_benchmark"
    result_root = test_root / "out"
    result_root.mkdir(parents=True, exist_ok=True)

    distributions = args.distribution or ["smooth", "ring", "clump"]

    # K is a compile-time top-K capacity for both backends, so compile once for
    # the complete matrix and then vary only runtime particle distributions.
    subprocess.run(
        ["make", "-C", str(test_root), f"ARCH={args.arch}", f"K={args.k}"],
        check=True,
    )

    environment = {
        "nvcc": capture(["nvcc", "--version"], test_root),
        "nvidia_smi": capture(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"],
            test_root,
        ),
        "arch": args.arch,
        "k": args.k,
    }
    (result_root / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n"
    )

    completed = []
    for dimension in args.dim:
        for distribution in distributions:
            for particles in args.particles:
                name = f"{distribution}_{dimension}d_N{particles}"
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
                subprocess.run(command, cwd=test_root.parent, check=True)
                record = json.loads(output.read_text())
                completed.append(record)
                print(
                    f"quality={record['quality_passed']}  "
                    f"KD={record['kd_query_ms']:.3f} ms  "
                    f"Morton={record['morton_query_ms']:.3f} ms  "
                    f"speedup={record['query_speedup']:.3f}"
                )

    manifest = {
        "cases": len(completed),
        "all_quality_passed": all(record["quality_passed"] for record in completed),
        "files": [
            f"{record['distribution']}_{record['dimension']}d_N{record['particles']}.json"
            for record in completed
        ],
    }
    (result_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
