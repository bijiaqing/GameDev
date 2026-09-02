# Fluid verification case index

The canonical model inventory, exact initial conditions, analytical solutions, finite-volume
measurements, acceptance criteria, archive contract, and coverage limits are maintained in
[`doc/fluid_testset.md`](../../../../doc/fluid_testset.md).

The executable matrix itself is defined in `val/tool/val_config.py`. Keep model selection there and
use this directory only for implementation shared by several cases; test-specific flags and source
overrides belong to the corresponding `val/comm/fluid/test_*` directory.
