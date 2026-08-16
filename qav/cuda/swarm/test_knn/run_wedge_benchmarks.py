#!/usr/bin/env python3

"""Run the periodic-wedge comparison between KD-tree and Morton methods"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--particles", nargs="+", type=int,
        default=[100_000, 1_000_000, 10_000_000],
    )
    parser.add_argument("--dim", nargs="+", type=int, choices=(2, 3), default=[2, 3])
    parser.add_argument(
        "--distribution",
        nargs="+",
        choices=("smooth", "ring", "interior_clump", "seam_clump"),
        default=None,
    )
    parser.add_argument("--queries", type=int, default=4096)
    parser.add_argument("--brute-queries", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--radius", type=float, default=0.1)
    parser.add_argument("--leaf-target", type=int, default=128)
    parser.add_argument("--arch", default="sm_80")
    parser.add_argument("--k", type=int, default=200)
    args = parser.parse_args()

    test_root = Path(__file__).resolve().parent
    executable = test_root / "bin" / "knn_wedge_benchmark"
    project_root = test_root.parents[3]
    archive_root = project_root/"qav"/"logs"/"swarm"/"cuda"
    scope = os.environ.get("QAV_SCOPE", "manual")
    scope_root = archive_root if scope == "all" else archive_root/"groups"/scope
    result_root = scope_root/"test_knn"/"wedge"
    result_root.mkdir(parents=True, exist_ok=True)
    distributions = args.distribution or ["smooth", "ring", "interior_clump", "seam_clump"]

    # K is compiled into the candidate-list types used by both search backends.
    subprocess.run(
        [
            "make", "-C", str(test_root), "wedge", f"ARCH={args.arch}", f"K={args.k}",
        ],
        check=True,
    )

    cases = [(distribution, None) for distribution in distributions]
    cases.append(("seam_clump", 0.2))
    completed = []
    for dimension in args.dim:
        for distribution, wedge_width in cases:
            for particles in args.particles:
                prefix = "narrow_" if wedge_width is not None else ""
                name = f"{prefix}{distribution}_{dimension}d_N{particles}"
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
                    "--output", str(output),
                ]
                if wedge_width is not None:
                    command.extend(("--x-min", str(-0.5*wedge_width), "--x-max", str(0.5*wedge_width)))
                print(f"\n=== {name} ===", flush=True)
                output.unlink(missing_ok=True)
                result = subprocess.run(command, cwd=test_root.parent, check=False)
                if not output.exists():
                    result.check_returncode()
                    raise RuntimeError(f"wedge benchmark did not write {output}")
                record = json.loads(output.read_text())
                if (
                    record.get("distribution") != distribution
                    or record.get("particles") != particles
                    or record.get("dimension") != dimension
                    or record.get("passed") != record.get("quality_passed")
                ):
                    raise RuntimeError(
                        f"wedge benchmark labels in {output} do not match case {name}"
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
                        f"overflows={record['stack_overflows']}  "
                        f"max-error={record['maximum_distance_error']:.3e}  "
                        f"ties: KD={record['kd_tie_equivalent_neighbors']} "
                        f"Morton={record['morton_tie_equivalent_neighbors']}",
                        flush=True,
                    )
                    result.check_returncode()

    passed = all(record["passed"] for _, record in completed)
    manifest = {
        "component": "wedge",
        "tier": "publication",
        "extended_tier": "qualification" if any(
            particles != 100_000 for particles in args.particles
        ) else None,
        "cases": len(completed),
        "all_quality_passed": passed,
        "passed": passed,
        "environment": "../environment.json",
        "files": [f"{name}.json" for name, _ in completed],
    }
    (result_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
