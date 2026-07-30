#!/usr/bin/env python3

"""Summarize ordinary KD-tree and Morton KNN benchmark output"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_records(test_root: Path, include_all: bool) -> dict[str, dict]:
    """Load the canonical matrix without mixing in stale experiments"""

    result_root = test_root / "out"
    manifest_path = result_root / "manifest.json"
    if manifest_path.exists() and not include_all:
        manifest = json.loads(manifest_path.read_text())
        result_paths = [result_root / name for name in manifest["files"]]
    else:
        result_paths = sorted(result_root.glob("*.json"))

    records = {}
    for path in result_paths:
        if path.name in {"environment.json", "manifest.json"}:
            continue
        if not path.exists():
            raise SystemExit(f"manifest references missing result: {path}")
        records[path.stem] = json.loads(path.read_text())
    if not records:
        raise SystemExit("no ordinary benchmark JSON files found")
    return records


def print_records(records: dict[str, dict]) -> None:
    """Print the KD-tree and sorted-Morton comparison"""

    print(
        f"{'case':32s} {'ok':>4s} {'KD ms':>10s} {'Morton ms':>11s} "
        f"{'speedup':>9s} {'mem ratio':>10s} {'p99 leaf':>9s} {'max leaf':>9s} "
        f"{'leaves/q':>9s} {'cand/q':>10s}"
    )
    for name, record in records.items():
        occupancy = record["leaf_occupancy"]
        leaves = record.get("mean_leaves_visited", record.get("mean_cells_visited", float("nan")))
        print(
            f"{name:32s} {str(record['quality_passed']):>4s} "
            f"{record['kd_query_ms']:10.3f} {record['morton_query_ms']:11.3f} "
            f"{record['query_speedup']:9.3f} {record['persistent_memory_ratio']:10.3f} "
            f"{occupancy['p99']:9d} {occupancy['maximum']:9d} "
            f"{leaves:9.2f} {record['mean_candidates_examined']:10.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all",
        action="store_true",
        help="include exploratory JSON files not listed in the latest manifest",
    )
    args = parser.parse_args()
    print_records(load_records(Path(__file__).resolve().parent, args.all))


if __name__ == "__main__":
    main()
