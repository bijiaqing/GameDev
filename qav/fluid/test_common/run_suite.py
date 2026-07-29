#!/usr/bin/env python3

"""Select verification cases and invoke each model-local runner in sequence

This is the top-level entry point used by ``--group all`` and the smaller
physics groups.  It does not compile or validate results itself.  Instead, it
builds a list of model/parameter combinations and delegates each combination
to the short ``run.py`` wrapper inside that model directory.
"""

# Store type annotations without evaluating them at import time.  This keeps
# annotations such as ``list[tuple[str, list[str]]]`` as documentation and has
# no effect on the numerical tests.
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    # ``nargs="+"`` accepts one or more values after --res, for example
    # ``--res 32 64 128 256``.  --quick keeps only the first two of those
    # resolutions, which is useful for checking the workflow before a full run.
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--quick", action="store_true", help="Use only the two coarsest requested resolutions")
    parser.add_argument(
        "--group", choices=("all", "transport", "diffusion", "source", "radiation", "ring", "sweep"),
        default="all",
    )
    parser.add_argument("--sweep-dim", choices=("all", "2d", "3d"), default="all")
    parser.add_argument("--sweep-res-2d", type=int, default=1024)
    parser.add_argument("--sweep-res-3d", type=int, default=128)
    args = parser.parse_args()

    resolutions = args.res[:2] if args.quick else args.res

    # __file__ is this script.  Its parent is test_common, and the next parent
    # is the directory containing both test_common and every model directory.
    common = Path(__file__).resolve().parent
    model_root = common.parent

    # Sweep cross-validation is a fixed-work implementation benchmark rather than a convergence sequence.  It remains a
    # separate branch because the full 128^3 pair is substantially more expensive than the 85 analytical configurations.
    if args.group == "sweep":
        command = [
            sys.executable, str(common/"run_sweep.py"),
            "--dimension", args.sweep_dim,
            "--res-2d", str(args.sweep_res_2d),
            "--res-3d", str(args.sweep_res_3d),
        ]
        if args.quick:
            command.append("--quick")
        subprocess.run(command, check=True)
        return

    # Each entry contains the model directory name and model-specific command
    # line options.  Keeping variants as separate entries gives each one an
    # independent compilation, run, validation, and convergence report.
    commands: list[tuple[str, list[str]]] = []
    if args.group in {"all", "transport"}:
        # Exercise integer and fractional FARGO shifts because the fractional
        # remap is the part that contributes interpolation error.
        for shift in (3.0, 3.25, 3.5, 3.75):
            commands.append(("test_x_transport_2d", ["--shift", str(shift)]))

        # Radial and polar transport are each tested at a small CFL number and
        # at the production CFL number to separate spatial and temporal effects.
        commands.extend([
            ("test_y_transport_cyl", ["--cfl", "0.05"]),
            ("test_y_transport_cyl", ["--cfl", "0.5"]),
            ("test_y_transport_sph", ["--cfl", "0.05"]),
            ("test_y_transport_sph", ["--cfl", "0.5"]),
            ("test_z_transport_3d", ["--cfl", "0.05"]),
            ("test_z_transport_3d", ["--cfl", "0.5"]),
        ])
    if args.group in {"all", "diffusion"}:
        # Cover periodic azimuthal diffusion, cylindrical radial diffusion,
        # spherical radial diffusion, and polar diffusion separately.
        commands.extend([
            ("test_x_diffusion_2d", []),
            ("test_y_diffusion_cyl", []),
            ("test_y_diffusion_sph", []),
            ("test_z_diffusion_3d", []),
        ])
    if args.group in {"all", "source"}:
        # The source test stores eight drag stiffnesses in eight x cells, so it
        # uses one fixed resolution instead of a spatial convergence sequence.
        commands.append(("test_source_drag", ["--res", "8"]))
    if args.group in {"all", "radiation"}:
        # These density powers exercise both the logarithmic p=-1 optical-depth
        # integral and the ordinary power-law antiderivative.
        for power in (0.0, -1.0, 1.0):
            commands.append(("test_optdepth", ["--power", str(power)]))
    if args.group in {"all", "ring"}:
        # The ring cases test operator combinations rather than isolated kernels.
        commands.extend([
            ("test_ring_transport_2d", []),
            ("test_ring_diffusion_2d", []),
            ("test_ring_radiation_2d", []),
            ("test_ring_all_2d", []),
        ])

    for model, extra in commands:
        # sys.executable reuses the Python interpreter that launched this suite,
        # avoiding accidental changes of environment between the two scripts.
        command = [sys.executable, str(model_root/model/"run.py")]

        # A model-specific --res, currently only the source test, takes priority
        # over the suite-wide resolution list.
        if "--res" not in extra:
            command += ["--res", *(str(value) for value in resolutions)]
        command += extra
        print("\n===", model, " ".join(extra), "===", flush=True)

        # check=True stops the suite immediately if compilation, execution, or
        # validation of any case fails, so later output cannot hide that failure.
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
