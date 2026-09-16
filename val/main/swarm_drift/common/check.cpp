// Host-only check of model setup; this does not execute or qualify GPU transport.
#include <cassert>
#include <cmath>
struct double3 { double x, y, z; };
struct index3 { int x; };
index3 threadIdx{0}, blockIdx{0}, blockDim{64};
#define __global__
#include "particle_init.cu"
#include "dyn_rate_calc.cu"
int main() {
    swarm p{};
    particle_init(&p, nullptr, nullptr, nullptr);
    assert(N_P == 1 && p.position.y == 1.0);
    assert(p.velocity.y < 0.0 && p.velocity.x > 0.0 && p.velocity.x < 1.0);
    assert(p.velocity.z == 0.0);
    const double gas_ratio2 = 1.0 + (IDX_P + 0.5*IDX_Q - 1.5)*ASPR_0*ASPR_0;
    assert(std::abs(gas_ratio2 - (1.0 - 0.05*0.05)) < 1e-15);
    const double leading_vr = -0.05*0.05*STOKES_0/(1.0 + STOKES_0*STOKES_0);
    assert(std::abs(p.velocity.y/leading_vr - 1.0) < 0.003);
    double rate = 0;
    dyn_rate_calc(&rate, &p);
    assert(rate > 0 && std::abs(rate*DT_MAX - 1) < 1e-14);
    assert(DT_OUT >= DT_MAX);
    assert(std::abs(DT_OUT/DT_MAX - std::round(DT_OUT/DT_MAX)) < 1e-7);
    assert(SAVE_MAX*DT_OUT == 10.0*std::fmax(1.0, STOKES_0));
    threadIdx.x = 1;
    dyn_rate_calc(&rate, &p); // inactive launch lanes must not write beyond the one-particle buffer
}
