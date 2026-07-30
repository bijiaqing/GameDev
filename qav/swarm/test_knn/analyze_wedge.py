#!/usr/bin/env python3

"""Summarize the latest periodic-wedge benchmark matrix"""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    result_root = Path(__file__).resolve().parent / "out" / "wedge"
    manifest_path = result_root / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit("missing qav/swarm/test_knn/out/wedge/manifest.json")
    manifest = json.loads(manifest_path.read_text())

    print(
        f"{'case':34s} {'ok':>4s} {'KD ms':>9s} {'Query ms':>10s} {'Ghost ms':>10s} "
        f"{'Q speed':>8s} {'G speed':>8s} {'Q memory':>9s} {'G memory':>9s} {'ghosts':>8s} {'dedup':>6s}"
    )
    for name in manifest["files"]:
        path = result_root / name
        record = json.loads(path.read_text())
        print(
            f"{path.stem:34s} "
            f"{str(record['quality_passed']):>4s} "
            f"{record['kd_query_ms']:9.3f} "
            f"{record['morton_query_ms']:10.3f} "
            f"{record['ghost_query_ms']:10.3f} "
            f"{record['query_speedup']:8.3f} "
            f"{record['ghost_query_speedup']:8.3f} "
            f"{record['persistent_memory_ratio']:9.3f} "
            f"{record['ghost_memory_ratio']:9.3f} "
            f"{record['ghost_record_ratio']:8.3f} "
            f"{str(record.get('kd_deduplicate', True)):>6s}"
        )


if __name__ == "__main__":
    main()
