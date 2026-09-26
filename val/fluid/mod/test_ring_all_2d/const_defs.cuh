// include fragment: select this model's parameters, then include the shared constants
// select the analytical branch; the included test constants intentionally replace production model parameters
#define VERIFY_RING
#define VERIFY_RING_DIFFUSION
#define VERIFY_RING_RADIATION
#include "../../src/const_defs.cuh"
