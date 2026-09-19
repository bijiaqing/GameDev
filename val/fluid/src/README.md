# Shared fluid verification implementation

This directory contains the backend-neutral verification driver, Python analysis routines, common
constants, and the physical-helper override used by multiple fluid test models. Backend-local
wrappers compile these files as CUDA or HIP without duplicating the analytical definitions.

The constant-diffusivity cases replace only `_get_nu` and `_get_alpha` through the local
`param_phys.cuh`; production PPM, HLL, geometry, vacuum recovery, and remaining helpers stay active.
`test_source_drag` is the sole case that replaces `source_update`, because its prescribed constant
gas velocities and force endpoints cannot be expressed through the production disk interface.

Every resolution is cleaned and rebuilt because grid sizes and control values are compile-time
constants. Variant tags keep CFL, FARGO-shift, optical-depth-power, sweep, and CUDA arithmetic-mode
outputs in distinct directories.

See [`doc/fluid_testset.md`](../../../../doc/fluid_testset.md) for the model inventory, equations,
commands, pass criteria, archive contract, and documented coverage limits.
