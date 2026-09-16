// Host setup check, not a GPU integration test.
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
    assert(N_P == 1 && p.position.y == 0.5 && p.position.x == 0.0);
    assert(p.velocity.y == 0.0 && p.velocity.z == 0.0);
    const double r = p.position.y, vphi = p.velocity.x/r;
    assert(std::abs(vphi - std::sqrt(3.0)) < 1e-15);
    assert(std::abs(0.5*vphi*vphi - G*M_S/r + 0.5) < 1e-15);
    assert(Y_MIN < 0.5 && Y_MAX > 1.5 && N_X > 1 && N_Z == 1);
    assert(std::abs(SAVE_MAX*DT_OUT/(2*M_PI) - 100.0) < 1e-12);
    assert(std::abs(DT_OUT/DT_MAX - ORBIT_STEPS_PER_PERIOD/20.0) < 1e-12);
    assert(std::abs(SAVE_MAX*DT_OUT/DT_MAX - 100*ORBIT_STEPS_PER_PERIOD) < 1e-8);
    double rate;
    dyn_rate_calc(&rate, &p);
    assert(std::abs(rate*DT_MAX - 1) < 1e-15);
    threadIdx.x = 1;
    particle_init(&p, nullptr, nullptr, nullptr);
    dyn_rate_calc(&rate, &p);
}
