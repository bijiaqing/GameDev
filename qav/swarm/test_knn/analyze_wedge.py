#!/usr/bin/env python3

"""Summarize production KD-tree and compact-ghost Morton wedge benchmarks"""

from __future__ import annotations

import json
from pathlib import Path


def load_records(test_root: Path) -> dict[str, dict]:
    """Load the completed periodic-wedge matrix"""

    result_root = test_root / "out" / "wedge"
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
        f"{'case':34s} {'ok':>4s} {'KD ms':>9s} {'Morton ms':>10s} "
        f"{'speedup':>8s} {'memory':>9s} {'records':>8s} {'dedup':>6s}"
    )
    for name, record in records.items():
        # Current files call the production compact-ghost result simply
        # "morton".  Retained pre-promotion files recorded the same result
        # under "ghost", so accept that schema without reporting the obsolete
        # query-image timing as the production comparison.
        current_schema = "record_ratio" in record
        morton_ms = record["morton_query_ms"] if current_schema else record["ghost_query_ms"]
        speedup = record["query_speedup"] if current_schema else record["ghost_query_speedup"]
        memory = (
            record["persistent_memory_ratio"] if current_schema else record["ghost_memory_ratio"]
        )
        records_ratio = record["record_ratio"] if current_schema else record["ghost_record_ratio"]
        print(
            f"{name:34s} {str(record['quality_passed']):>4s} "
            f"{record['kd_query_ms']:9.3f} {morton_ms:10.3f} "
            f"{speedup:8.3f} {memory:9.3f} "
            f"{records_ratio:8.3f} {str(record.get('kd_deduplicate', True)):>6s}"
        )


def main() -> None:
    print_records(load_records(Path(__file__).resolve().parent))


if __name__ == "__main__":
    main()
