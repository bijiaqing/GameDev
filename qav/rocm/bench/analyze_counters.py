#!/usr/bin/env python3

"""Summarize merged ROCm Compute Profiler counter tables as JSON"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median


RAW_COUNTERS = (
    "SQ_LEVEL_WAVES",
    "SQ_BUSY_CU_CYCLES",
    "SQ_THREAD_CYCLES_VALU",
    "SQ_INSTS_VALU",
    "SQ_INSTS_SALU",
    "SQ_INSTS_VMEM",
    "SQ_INSTS_LDS",
    "SQ_LDS_BANK_CONFLICT",
    "TCC_HIT_sum",
    "TCC_MISS_sum",
    "TCC_EA0_RDREQ_sum",
    "TCC_EA0_RDREQ_32B_sum",
    "TCC_EA0_WRREQ_sum",
    "TCC_EA0_WRREQ_64B_sum",
)

INTEGER_METADATA = (
    "Grid_Size",
    "Workgroup_Size",
    "LDS_Per_Workgroup",
    "Scratch_Per_Workitem",
    "Arch_VGPR",
    "Accum_VGPR",
    "SGPR",
    "Wave_Size",
)


def load_object(path: Path) -> dict[str, object]:
    """Read one JSON object and reject malformed input"""

    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def finite_number(row: dict[str, str], name: str) -> float | None:
    """Read one finite numeric CSV value while tolerating unavailable counters"""

    text = row.get(name, "").strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def median_value(rows: list[dict[str, str]], name: str) -> float | None:
    """Return the median available value of one column"""

    values = [value for row in rows if (value := finite_number(row, name)) is not None]
    return median(values) if values else None


def median_ratio(
    rows: list[dict[str, str]], numerator: str, denominator: str,
) -> float | None:
    """Return the median per-dispatch ratio without mixing different replay rows"""

    values = []
    for row in rows:
        value_n = finite_number(row, numerator)
        value_d = finite_number(row, denominator)
        if value_n is not None and value_d is not None and value_d > 0.0:
            values.append(value_n/value_d)
    return median(values) if values else None


def find_counter_table(profile_dir: Path) -> Path:
    """Locate the one merged counter table belonging to a profile archive"""

    tables = sorted(profile_dir.glob("workloads/**/pmc_perf.csv"))
    if len(tables) != 1:
        raise ValueError(
            f"{profile_dir}: expected one workloads/**/pmc_perf.csv, found {len(tables)}"
        )
    return tables[0]


def analyze_profile(profile_dir: Path, max_waves_per_cu: float) -> dict[str, object]:
    """Analyze one compute-profile directory and retain its provenance"""

    manifest_path = profile_dir/"manifest.json"
    manifest = load_object(manifest_path)
    if manifest.get("passed") is not True or manifest.get("tool") != "compute":
        raise ValueError(f"not a passing compute-profile manifest: {manifest_path}")

    kernel_filter = str(manifest.get("kernel_filter") or "")
    if not kernel_filter:
        raise ValueError(f"missing kernel_filter: {manifest_path}")

    table = find_counter_table(profile_dir)
    with table.open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = [row for row in reader if kernel_filter in row.get("Kernel_Name", "")]
        fieldnames = set(reader.fieldnames or ())
    if not rows:
        raise ValueError(f"no rows matching {kernel_filter!r}: {table}")

    metadata: dict[str, int | None] = {}
    inconsistent_metadata = []
    for name in INTEGER_METADATA:
        values = {
            int(value)
            for row in rows
            if (value := finite_number(row, name)) is not None
        }
        if len(values) > 1:
            inconsistent_metadata.append(name)
        metadata[name] = next(iter(values)) if len(values) == 1 else None
    missing_metadata = [name for name, value in metadata.items() if value is None]

    counters = {name: median_value(rows, name) for name in RAW_COUNTERS}
    missing_counters = [name for name, value in counters.items() if value is None]

    duration_ms_values = []
    external_bytes_values = []
    bandwidth_values = []
    for row in rows:
        time_0 = finite_number(row, "Start_Timestamp")
        time_1 = finite_number(row, "End_Timestamp")
        read_count = finite_number(row, "TCC_EA0_RDREQ_sum")
        read_count_32 = finite_number(row, "TCC_EA0_RDREQ_32B_sum")
        write_count = finite_number(row, "TCC_EA0_WRREQ_sum")
        write_count_64 = finite_number(row, "TCC_EA0_WRREQ_64B_sum")
        duration_s = None
        if time_0 is not None and time_1 is not None and time_1 > time_0:
            duration_s = (time_1 - time_0)*1.0e-9
            duration_ms_values.append(duration_s*1.0e3)
        if (
            read_count is not None and read_count_32 is not None
            and write_count is not None and write_count_64 is not None
        ):
            # total reads contain 32- and 64-byte requests, whereas the read subtype counts 32 B
            # total writes contain 32- and 64-byte requests, whereas the write subtype counts 64 B
            read_bytes = 64.0*read_count - 32.0*read_count_32
            write_bytes = 32.0*write_count + 32.0*write_count_64
            external_bytes = read_bytes + write_bytes
            external_bytes_values.append(external_bytes)
            if duration_s is not None:
                bandwidth_values.append(external_bytes/duration_s/1.0e9)

    hit_fraction = None
    miss_fraction = None
    hit = counters["TCC_HIT_sum"]
    miss = counters["TCC_MISS_sum"]
    if hit is not None and miss is not None and hit + miss > 0.0:
        hit_fraction = hit/(hit + miss)
        miss_fraction = miss/(hit + miss)

    resident_waves = median_ratio(rows, "SQ_LEVEL_WAVES", "SQ_BUSY_CU_CYCLES")
    active_workitems = median_ratio(rows, "SQ_THREAD_CYCLES_VALU", "SQ_INSTS_VALU")
    derived = {
        "duration_ms": median(duration_ms_values) if duration_ms_values else None,
        "resident_waves_per_cu": resident_waves,
        "resident_wave_fraction": (
            resident_waves/max_waves_per_cu if resident_waves is not None else None
        ),
        "active_workitems_per_valu_instruction": active_workitems,
        "active_workitem_fraction": (
            active_workitems/metadata["Wave_Size"]
            if active_workitems is not None and metadata["Wave_Size"]
            else None
        ),
        "l2_hit_fraction": hit_fraction,
        "l2_miss_fraction": miss_fraction,
        "external_bytes": median(external_bytes_values) if external_bytes_values else None,
        "external_bandwidth_gb_s": median(bandwidth_values) if bandwidth_values else None,
    }

    return {
        "profile": profile_dir.name,
        "kernel_filter": kernel_filter,
        "manifest": str(manifest_path.resolve()),
        "counter_table": str(table.resolve()),
        "dispatches": len(rows),
        "metadata": metadata,
        "missing_metadata": missing_metadata,
        "inconsistent_metadata": inconsistent_metadata,
        "counters": counters,
        "missing_counters": missing_counters,
        "derived": derived,
        "passed": not inconsistent_metadata and not missing_metadata and not missing_counters,
        "available_columns": len(fieldnames),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="summarize one or more ROCm Compute Profiler archives",
    )
    parser.add_argument("--profile-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--max-waves-per-cu", type=float, default=32.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    profiles = [
        analyze_profile(path.resolve(), args.max_waves_per_cu)
        for path in args.profile_dir
    ]
    report = {
        "schema": 1,
        "max_waves_per_cu": args.max_waves_per_cu,
        "profiles": profiles,
        "passed": all(bool(profile["passed"]) for profile in profiles),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(
        f"{'profile':48s} {'rows':>6s} {'WG':>5s} {'VGPR':>6s} {'scratch':>8s} "
        f"{'ms':>10s} {'lanes':>9s} {'L2 hit':>9s} {'GB/s':>10s}"
    )
    for profile in profiles:
        metadata = profile["metadata"]
        derived = profile["derived"]

        def shown(value: object, fmt: str) -> str:
            return "-" if value is None else format(value, fmt)

        print(
            f"{profile['profile']:48s} {profile['dispatches']:6d} "
            f"{shown(metadata['Workgroup_Size'], 'd'):>5s} "
            f"{shown(metadata['Arch_VGPR'], 'd'):>6s} "
            f"{shown(metadata['Scratch_Per_Workitem'], 'd'):>8s} "
            f"{shown(derived['duration_ms'], '.3f'):>10s} "
            f"{shown(derived['active_workitems_per_valu_instruction'], '.3f'):>9s} "
            f"{shown(derived['l2_hit_fraction'], '.3f'):>9s} "
            f"{shown(derived['external_bandwidth_gb_s'], '.1f'):>10s}"
        )
    print(f"counter analysis: {'PASS' if report['passed'] else 'FAIL'}")
    print(f"report: {args.output}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
