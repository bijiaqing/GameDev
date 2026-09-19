#!/usr/bin/env python3

"""Summarize ordinary KD-tree and Morton KNN benchmark output"""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import json
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from val_config import model_output


def load_records(test_root: Path, backend: str, include_all: bool, output_subdir: str = "") -> dict[str, dict]:
    """Load the canonical matrix without mixing in stale experiments"""

    project_root = test_root.parents[3]
    scope = os.environ.get("VAL_SCOPE", "manual")
    result_root = model_output(project_root/"val", "swarm", "test_knn", backend, scope=scope)
    if output_subdir:
        result_root = result_root / output_subdir
    manifest_path = result_root / "manifest.json"
    if manifest_path.exists() and not include_all:
        manifest = json.loads(manifest_path.read_text())
        result_paths = [result_root / name for name in manifest["files"]]
    else:
        result_paths = sorted(result_root.glob("*.json"))

    records = {}
    for path in result_paths:
        if path.name in {"environment.json", "manifest.json", "suite_manifest.json"}:
            continue
        if not path.exists():
            raise SystemExit(f"manifest references missing result: {path}")
        record = json.loads(path.read_text())
        if "passed" not in record or "kd_query_ms" not in record:
            continue
        records[path.stem] = record
    if not records:
        raise SystemExit("no ordinary benchmark JSON files found")
    return records


def print_records(records: dict[str, dict]) -> None:
    """Print the KD-tree and sorted-Morton comparison"""

    print(
        f"{'case':32s} {'pass':>5s} {'KD ms':>10s} {'Morton ms':>11s} "
        f"{'time KD/M':>10s} {'mem M/KD':>10s} {'p99 leaf':>9s} {'max leaf':>9s} "
        f"{'leaves/q':>9s} {'cand/q':>10s}"
    )
    for name, record in records.items():
        occupancy = record["leaf_occupancy"]
        leaves = record.get("mean_leaves_visited", record.get("mean_cells_visited", float("nan")))
        print(
            f"{name:32s} {str(record['passed']):>5s} "
            f"{record['kd_query_ms']:10.3f} {record['morton_query_ms']:11.3f} "
            f"{record['query_time_ratio_kd_morton']:10.3f} "
            f"{record['memory_ratio_morton_kd']:10.3f} "
            f"{occupancy['p99']:9d} {occupancy['maximum']:9d} "
            f"{leaves:9.2f} {record['mean_candidates_examined']:10.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("cuda", "rocm"), required=True)
    parser.add_argument(
        "--all",
        action="store_true",
        help="include exploratory JSON files not listed in the latest manifest",
    )
    parser.add_argument("--output-subdir", default="")
    args = parser.parse_args()
    print_records(load_records(Path(__file__).resolve().parent, args.backend, args.all, args.output_subdir))


if __name__ == "__main__":
    main()
