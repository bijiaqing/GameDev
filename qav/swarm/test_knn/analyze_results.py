#!/usr/bin/env python3

"""Summarize correctness, speed, memory, and clumping behavior from KNN QA output"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all",
        action="store_true",
        help="include exploratory JSON files not listed in the latest manifest",
    )
    args = parser.parse_args()

    result_root = Path(__file__).resolve().parent / "out"
    manifest_path = result_root / "manifest.json"

    # A completed matrix records its canonical files in the manifest. Reading
    # that list avoids mixing old failed prototypes into the current summary.
    if manifest_path.exists() and not args.all:
        manifest = json.loads(manifest_path.read_text())
        result_paths = [result_root / name for name in manifest["files"]]
    else:
        result_paths = sorted(result_root.glob("*.json"))

    records = []
    for path in result_paths:
        if path.name in {"environment.json", "manifest.json"}:
            continue
        if not path.exists():
            raise SystemExit(f"manifest references missing result: {path}")
        record = json.loads(path.read_text())
        records.append((path.name, record))

    if not records:
        raise SystemExit("no benchmark JSON files found under qav/swarm/test_knn/out")

    print(
        f"{'case':32s} {'ok':>4s} {'KD ms':>10s} {'Morton ms':>11s} "
        f"{'speedup':>9s} {'mem ratio':>10s} {'p99 leaf':>9s} {'max leaf':>9s} "
        f"{'leaves/q':>9s} {'cand/q':>10s}"
    )
    for name, record in records:
        occupancy = record["leaf_occupancy"]
        leaves_visited = record.get("mean_leaves_visited", record.get("mean_cells_visited", float("nan")))
        print(
            f"{name.removesuffix('.json'):32s} "
            f"{str(record['quality_passed']):>4s} "
            f"{record['kd_query_ms']:10.3f} "
            f"{record['morton_query_ms']:11.3f} "
            f"{record['query_speedup']:9.3f} "
            f"{record['persistent_memory_ratio']:10.3f} "
            f"{occupancy['p99']:9d} "
            f"{occupancy['maximum']:9d} "
            f"{leaves_visited:9.2f} "
            f"{record['mean_candidates_examined']:10.2f}"
        )


if __name__ == "__main__":
    main()
