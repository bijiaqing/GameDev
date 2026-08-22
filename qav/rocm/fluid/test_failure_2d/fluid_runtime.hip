#include <cmath>            // std::isfinite, std::isinf
#include <cstdlib>          // EXIT_FAILURE, EXIT_SUCCESS
#include <iostream>         // std::cerr, std::cout, std::endl
#include <limits>           // std::numeric_limits
#include <string>           // std::string
#include <vector>           // std::vector

#include <fluid_host.cuh>
#include <fluid_kern.cuh>

// ROCm failure-path driver rather than a physical evolution model
// inject exactly one NaN or Inf into an otherwise finite production state and require the state and CFL guards to identify
// the correct cell or ring while finite controls prevent a test that passes merely by rejecting every input

namespace
{

constexpr int INJECT_IX = 2;
constexpr int INJECT_IY = 1;
constexpr int INJECT_IZ = 0;
constexpr int INJECT_CELL = INJECT_IX + INJECT_IY*N_X + INJECT_IZ*N_X*N_Y;

real *select_field (
    const std::string &field,
    std::vector<real> &rhod,
    std::vector<real> &mx, std::vector<real> &my, std::vector<real> &mz,
    std::vector<real> &lx, std::vector<real> &vy, std::vector<real> &lz,
    std::vector<real> &optdepth
)
{
    if (field == "rhod") return rhod.data();
    if (field == "mx") return mx.data();
    if (field == "my") return my.data();
    if (field == "mz") return mz.data();
    if (field == "lx") return lx.data();
    if (field == "vy") return vy.data();
    if (field == "lz") return lz.data();
    if (field == "optdepth") return optdepth.data();
    return nullptr;
}

void report_bad_cell (const char *message, int idx_cell)
{
    int ix = idx_cell % N_X;
    int iy = (idx_cell / N_X) % N_Y;
    int iz = idx_cell / (N_X*N_Y);

    std::cerr << message << "(" << ix << "," << iy << "," << iz << ")" << std::endl;
}

} // namespace

int main (int argc, char **argv)
{
    if (argc != 4)
    {
        std::cerr << "usage: gamedev state|cfl FIELD finite|nan|inf" << std::endl;
        return 2;
    }

    const std::string mode = argv[1];
    const std::string field = argv[2];
    const std::string value_name = argv[3];

    if (mode != "state" && mode != "cfl")
    {
        std::cerr << "invalid failure-test mode: " << mode << std::endl;
        return 2;
    }

    std::vector<real> rhod(N_G, 1.0);
    std::vector<real> mx(N_G, 0.0);
    std::vector<real> my(N_G, 0.0);
    std::vector<real> mz(N_G, 0.0);
    std::vector<real> lx(N_G, 1.0);
    std::vector<real> vy(N_G, 0.0);
    std::vector<real> lz(N_G, 0.0);
    std::vector<real> optdepth(N_G, 0.0);

    if (value_name != "finite")
    {
        // corrupt one known interior cell so the expected one-based failure flag and CFL ring are deterministic
        real *target = select_field(field, rhod, mx, my, mz, lx, vy, lz, optdepth);
        if (target == nullptr || (mode == "cfl" && field == "optdepth"))
        {
            std::cerr << "invalid failure-test field: " << field << std::endl;
            return 2;
        }

        if (value_name == "nan")
            target[INJECT_CELL] = std::numeric_limits<real>::quiet_NaN();
        else if (value_name == "inf")
            target[INJECT_CELL] = std::numeric_limits<real>::infinity();
        else
        {
            std::cerr << "invalid failure-test value: " << value_name << std::endl;
            return 2;
        }
    }

    real *dev_rhod, *dev_mx, *dev_my, *dev_mz;
    real *dev_lx, *dev_vy, *dev_lz, *dev_optdepth, *dev_cfl_rate;
    int *dev_bad_cell;

    HIP_CHECK(hipMalloc((void**)&dev_rhod, sizeof(real)*N_G));
    HIP_CHECK(hipMalloc((void**)&dev_mx, sizeof(real)*N_G));
    HIP_CHECK(hipMalloc((void**)&dev_my, sizeof(real)*N_G));
    HIP_CHECK(hipMalloc((void**)&dev_mz, sizeof(real)*N_G));
    HIP_CHECK(hipMalloc((void**)&dev_lx, sizeof(real)*N_G));
    HIP_CHECK(hipMalloc((void**)&dev_vy, sizeof(real)*N_G));
    HIP_CHECK(hipMalloc((void**)&dev_lz, sizeof(real)*N_G));
    HIP_CHECK(hipMalloc((void**)&dev_optdepth, sizeof(real)*N_G));
    HIP_CHECK(hipMalloc((void**)&dev_cfl_rate, sizeof(real)*N_G));
    HIP_CHECK(hipMalloc((void**)&dev_bad_cell, sizeof(int)));

    HIP_CHECK(hipMemcpy(dev_rhod, rhod.data(), sizeof(real)*N_G, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_mx, mx.data(), sizeof(real)*N_G, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_my, my.data(), sizeof(real)*N_G, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_mz, mz.data(), sizeof(real)*N_G, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_lx, lx.data(), sizeof(real)*N_G, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_vy, vy.data(), sizeof(real)*N_G, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_lz, lz.data(), sizeof(real)*N_G, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_optdepth, optdepth.data(), sizeof(real)*N_G, hipMemcpyHostToDevice));

    if (mode == "state")
    {
        // exercise the production all-field finite-state kernel without allowing the host harness to pre-screen the value
        HIP_CHECK(hipMemset(dev_bad_cell, 0, sizeof(int)));
        inf_cell_flag <<< NB_G, TPB >>> (
            dev_rhod, dev_mx, dev_my, dev_mz, dev_lx, dev_vy, dev_lz, dev_optdepth, dev_bad_cell
        );
        HIP_KERNEL_CHECK("inf_cell_flag");

        int bad_cell = 0;
        HIP_CHECK(hipMemcpy(&bad_cell, dev_bad_cell, sizeof(int), hipMemcpyDeviceToHost));
        if (value_name == "finite")
        {
            if (bad_cell != 0)
            {
                std::cerr << "clean state was rejected" << std::endl;
                return 2;
            }
            std::cout << "clean state accepted" << std::endl;
            return EXIT_SUCCESS;
        }

        if (bad_cell != INJECT_CELL + 1)
        {
            std::cerr << "state guard reported cell " << bad_cell - 1
                      << " instead of " << INJECT_CELL << std::endl;
            return 2;
        }

        report_bad_cell("Error: non-finite simulation state at cell ", INJECT_CELL);
        return EXIT_FAILURE;
    }

    // a non-finite primitive or momentum must poison the complete x ring reduced by the production CFL kernel
    cfl_rate_calc <<< NB_X, TPB >>> (
        dev_cfl_rate, dev_rhod, dev_mx, dev_my, dev_mz, dev_lx, dev_vy, dev_lz
    );
    HIP_KERNEL_CHECK("cfl_rate_calc");

    std::vector<real> cfl_rate(N_G);
    HIP_CHECK(hipMemcpy(cfl_rate.data(), dev_cfl_rate, sizeof(real)*N_G, hipMemcpyDeviceToHost));

    if (value_name == "finite")
    {
        for (real rate : cfl_rate)
        {
            if (!std::isfinite(rate))
            {
                std::cerr << "clean CFL state produced a non-finite rate" << std::endl;
                return 2;
            }
        }
        real dt = get_dt_cfl(dev_cfl_rate, dev_lx, dev_vy, dev_lz, false);
        if (!std::isfinite(dt) || dt <= 0.0)
        {
            std::cerr << "clean CFL state produced an invalid timestep" << std::endl;
            return 2;
        }
        std::cout << "clean CFL state accepted" << std::endl;
        return EXIT_SUCCESS;
    }

    for (int iz = 0; iz < N_Z; iz++)
    {
        for (int iy = 0; iy < N_Y; iy++)
        {
            for (int ix = 0; ix < N_X; ix++)
            {
                int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
                bool expected_inf = iy == INJECT_IY && iz == INJECT_IZ;
                if (std::isinf(cfl_rate[idx_cell]) != expected_inf)
                {
                    std::cerr << "CFL ring rejection pattern is incorrect at cell "
                              << idx_cell << std::endl;
                    return 2;
                }
            }
        }
    }

    get_dt_cfl(dev_cfl_rate, dev_lx, dev_vy, dev_lz, false);
    std::cerr << "non-finite CFL state unexpectedly returned a timestep" << std::endl;
    return 2;
}
