#!/usr/bin/env python3


from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--quick", action="store_true", help="Use only the two coarsest requested resolutions")
    parser.add_argument(
        "--group", choices=("all", "transport", "diffusion", "source", "radiation", "ring"), default="all"
    )
    args = parser.parse_args()

    resolutions = args.res[:2] if args.quick else args.res
    common = Path(__file__).resolve().parent
    model_root = common.parent

    commands: list[tuple[str, list[str]]] = []
    if args.group in {"all", "transport"}:
        for shift in (3.0, 3.25, 3.5, 3.75):
            commands.append(("verify_x_transport_2d", ["--shift", str(shift)]))
        commands.extend([
            ("verify_y_transport_cyl", ["--cfl", "0.05"]),
            ("verify_y_transport_cyl", ["--cfl", "0.5"]),
            ("verify_y_transport_sph", ["--cfl", "0.05"]),
            ("verify_y_transport_sph", ["--cfl", "0.5"]),
            ("verify_z_transport_3d", ["--cfl", "0.05"]),
            ("verify_z_transport_3d", ["--cfl", "0.5"]),
        ])
    if args.group in {"all", "diffusion"}:
        commands.extend([
            ("verify_x_diffusion_2d", []),
            ("verify_y_diffusion_cyl", []),
            ("verify_y_diffusion_sph", []),
            ("verify_z_diffusion_3d", []),
        ])
    if args.group in {"all", "source"}:
        commands.append(("verify_source_drag", ["--res", "8"]))
    if args.group in {"all", "radiation"}:
        for power in (0.0, -1.0, 1.0):
            commands.append(("verify_optdepth", ["--power", str(power)]))
    if args.group in {"all", "ring"}:
        commands.extend([
            ("verify_ring_transport_2d", []),
            ("verify_ring_diffusion_2d", []),
            ("verify_ring_radiation_2d", []),
            ("verify_ring_all_2d", []),
        ])

    for model, extra in commands:
        command = [sys.executable, str(model_root/model/"run.py")]
        if "--res" not in extra:
            command += ["--res", *(str(value) for value in resolutions)]
        command += extra
        print("\n===", model, " ".join(extra), "===", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
