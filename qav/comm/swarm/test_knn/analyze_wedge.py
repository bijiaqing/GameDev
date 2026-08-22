#!/usr/bin/env python3

"""Summarize production KD-tree and compact-ghost Morton wedge benchmarks"""

from __future__ import annotations

import json
import os
from pathlib import Path


def load_records(test_root: Path, backend: str) -> dict[str, dict]:
    """Load the completed periodic-wedge matrix"""

    project_root = test_root.parents[3]
    archive_root = project_root/"qav"/"logs"/"swarm"/backend
    scope = os.environ.get("QAV_SCOPE", "manual")
    scope_root = archive_root if scope == "all" else archive_root/"groups"/scope
    result_root = scope_root/"test_knn"/"wedge"
    manifest_path = result_root / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit("missing wedge manifest")
    manifest = json.loads(manifest_path.read_text())
    return {
        Path(name).stem: json.loads((result_root / name).read_text())
        for name in manifest["files"]
    }


def print_records(records: dict[str, dict]) -> None:
    """Print the compact-ghost Morton comparison against the KD-tree reference"""

    print(
        f"{'case':34s} {'pass':>5s} {'KD ms':>9s} {'Morton ms':>10s} "
        f"{'time KD/M':>10s} {'mem M/KD':>10s} {'records/NP':>10s} {'dedup':>6s}"
    )
    for name, record in records.items():
        # accept the two retained development schemas when reviewing old files,
        # while new runs use direction-explicit ratio names
        if "record_ratio_morton_per_particle" in record:
            morton_ms = record["morton_query_ms"]
            speedup = record["query_time_ratio_kd_morton"]
            memory = record["memory_ratio_morton_kd"]
            records_ratio = record["record_ratio_morton_per_particle"]
        elif "record_ratio" in record:
            morton_ms = record["morton_query_ms"]
            speedup = record["query_speedup"]
            memory = record["persistent_memory_ratio"]
            records_ratio = record["record_ratio"]
        else:
            morton_ms = record["ghost_query_ms"]
            speedup = record["ghost_query_speedup"]
            memory = record["ghost_memory_ratio"]
            records_ratio = record["ghost_record_ratio"]
        print(
            f"{name:34s} {str(record.get('passed', record['quality_passed'])):>5s} "
            f"{record['kd_query_ms']:9.3f} {morton_ms:10.3f} "
            f"{speedup:10.3f} {memory:10.3f} "
            f"{records_ratio:10.3f} {str(record.get('kd_deduplicate', True)):>6s}"
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("cuda", "rocm"), required=True)
    args = parser.parse_args()
    print_records(load_records(Path(__file__).resolve().parent, args.backend))


if __name__ == "__main__":
    main()
