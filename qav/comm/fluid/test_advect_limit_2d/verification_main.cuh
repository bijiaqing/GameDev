#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <_transport.cuh>
#include <device_api.cuh>
#include <fluid_kern.cuh>
#include <param_grid.cuh>

// test-only driver for the production azimuthal advection kernel
// a sharp counter-streaming state stresses positivity and primitive bounds, while a separate controlled correction forces
// the same invariant-scale helper used by the kernel into its active branch and exposes its exact result to the validator

namespace
{
const std::string output_path = PATH_OUT;

void write_binary (const std::string &name, const std::vector<real> &values)
{
    std::ofstream file(output_path + name + "_N" + std::to_string(VERIFY_RES) + ".dat", std::ios::binary);
    if (!file) throw std::runtime_error("cannot open advection-limiter output");
    file.write(reinterpret_cast<const char *>(values.data()), sizeof(real)*values.size());
    if (!file) throw std::runtime_error("cannot write advection-limiter output");
}
}

__global__
void qav_limit_probe (real *dev_probe)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    constexpr real rhod = 1.0;
    constexpr real mx = 0.2;
    constexpr real my = -0.1;
    constexpr real mz = 0.05;
    constexpr real corr_rhod = -1.2;
    constexpr real corr_mx = 0.7;
    constexpr real corr_my = 0.0;
    constexpr real corr_mz = 0.0;

    real scale = _invariant_scale(
        rhod, mx, my, mz,
        corr_rhod, corr_mx, corr_my, corr_mz,
        -0.5, 0.5, -0.4, 0.4, -0.2, 0.2
    );
    dev_probe[0] = scale;
    dev_probe[1] = rhod + scale*corr_rhod;
    dev_probe[2] = mx + scale*corr_mx;
    dev_probe[3] = my + scale*corr_my;
    dev_probe[4] = mz + scale*corr_mz;
}

int main ()
{
    static_assert(N_Y == 1 && N_Z == 1, "advection limiter qualification uses one periodic ring");

    const real dx = _get_dx();
    const real y = sqrt(Y_MIN*Y_MAX);
    const real R = y;
    constexpr real omega_stream = 0.45;
    const real dt = 0.4*dx / omega_stream;

    std::vector<real> initial(4*N_G), final(4*N_G);
    for (int ix = 0; ix < N_X; ix++)
    {
        int phase = ix % 16;
        real rhod = (phase == 0) ? 1.0e-14 : ((phase < 8) ? 1.0 : 0.05);
        real omega = (phase < 8) ? omega_stream : -omega_stream;
        real lx = R*R*omega;
        real vy = (phase < 4 || phase >= 12) ? 0.35 : -0.35;
        real lz = (phase % 4 < 2) ? 0.12 : -0.12;

        initial[ix] = rhod;
        initial[N_G + ix] = rhod*lx;
        initial[2*N_G + ix] = rhod*vy;
        initial[3*N_G + ix] = rhod*lz;
    }

    real *dev_rhod, *dev_mx, *dev_my, *dev_mz;
    qav_malloc(&dev_rhod, N_G, "allocate limiter density");
    qav_malloc(&dev_mx, N_G, "allocate limiter x momentum");
    qav_malloc(&dev_my, N_G, "allocate limiter y momentum");
    qav_malloc(&dev_mz, N_G, "allocate limiter z momentum");
    qav_copy_h2d(dev_rhod, initial.data(), N_G, "upload limiter density");
    qav_copy_h2d(dev_mx, initial.data() + N_G, N_G, "upload limiter x momentum");
    qav_copy_h2d(dev_my, initial.data() + 2*N_G, N_G, "upload limiter y momentum");
    qav_copy_h2d(dev_mz, initial.data() + 3*N_G, N_G, "upload limiter z momentum");

    #ifdef FLUID_BLOCK_SWEEP
    real *dev_adv_work;
    qav_malloc(&dev_adv_work, BLOCK_ADV_FIELDS*N_G, "allocate limiter block workspace");
    advection_xbl <<< N_Y*N_Z, TPB_BLOCK >>> (dev_rhod, dev_mx, dev_my, dev_mz, dev_adv_work, dt);
    #else  // THREAD_SWEEP
    advection_xth <<< NB_X, TPB >>> (dev_rhod, dev_mx, dev_my, dev_mz, dt);
    #endif // FLUID_BLOCK_SWEEP
    qav_kernel_check("near-vacuum counter-stream advection");

    qav_copy_d2h(final.data(), dev_rhod, N_G, "copy limiter density");
    qav_copy_d2h(final.data() + N_G, dev_mx, N_G, "copy limiter x momentum");
    qav_copy_d2h(final.data() + 2*N_G, dev_my, N_G, "copy limiter y momentum");
    qav_copy_d2h(final.data() + 3*N_G, dev_mz, N_G, "copy limiter z momentum");

    real *dev_probe;
    qav_malloc(&dev_probe, 5, "allocate invariant-scale probe");
    qav_limit_probe <<< 1, 1 >>> (dev_probe);
    qav_kernel_check("controlled invariant-scale activation");
    std::vector<real> probe(5);
    qav_copy_d2h(probe.data(), dev_probe, probe.size(), "copy invariant-scale probe");

    write_binary("state_initial", initial);
    write_binary("state_final", final);
    write_binary("limit_probe", probe);

    std::ofstream meta(output_path + "meta_N" + std::to_string(VERIFY_RES) + ".json");
    meta << std::setprecision(17)
         << "{\n"
         << "  \"case\": \"advect_limit_2d\",\n"
         << "  \"resolution\": " << VERIFY_RES << ",\n"
         << "  \"nx\": " << N_X << ",\n"
         << "  \"radius\": " << R << ",\n"
         << "  \"dt\": " << dt << ",\n"
         << "  \"rho_vac\": " << RHO_VAC << ",\n"
         << "  \"production_advection\": true,\n"
         << "  \"controlled_limiter_probe\": true\n"
         << "}\n";

    qav_free(dev_rhod, "free limiter density");
    qav_free(dev_mx, "free limiter x momentum");
    qav_free(dev_my, "free limiter y momentum");
    qav_free(dev_mz, "free limiter z momentum");
    qav_free(dev_probe, "free invariant-scale probe");
    #ifdef FLUID_BLOCK_SWEEP
    qav_free(dev_adv_work, "free limiter block workspace");
    #endif // FLUID_BLOCK_SWEEP

    std::cout << "fluid advection limiter case completed at N=" << VERIFY_RES << std::endl;
    return 0;
}
