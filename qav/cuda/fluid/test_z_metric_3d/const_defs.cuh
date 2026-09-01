// select the fixed-polar, radial-metric probe and its test-owned constants
#define VERIFY_Z_METRIC
#include "../test_common/const_defs.cuh"

static_assert(VERIFY_RES >= 32 && VERIFY_RES % 8 == 0,
    "VERIFY_Z_METRIC requires VERIFY_RES=8*N_Y with N_Y at least four");
