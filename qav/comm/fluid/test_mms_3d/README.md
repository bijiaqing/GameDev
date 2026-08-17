# Full-3D MMS placeholder

This directory is intentionally not a buildable model. It has no `flags.mk`, so the Makefile
rejects accidental use rather than running an ordinary disk model under a verification label.

The required manufactured equations, four physics configurations, generated artifacts, boundary
conditions, and acceptance prerequisites are specified in the planned-MMS section of
[`doc/fluid_testset.md`](../../../../doc/fluid_testset.md). Add source here only when that complete
harness is implemented.
