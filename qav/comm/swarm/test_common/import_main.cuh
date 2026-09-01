#include <cmath>       // std::floor, std::log, std::pow, std::sqrt
#include <fstream>     // std::ofstream
#include <limits>      // std::numeric_limits
#include <random>      // std::mt19937
#include <stdexcept>   // std::runtime_error
#include <string>      // std::string, std::to_string
#include <vector>      // std::vector

#include <device_api.cuh>
#include <param_grid.cuh>
#include <param_phys.cuh>
#include <swarm_kern.cuh>

// include the production host sampler explicitly so this test cannot silently exercise its QAV copy
#ifdef GAMEDEV_CUDA
#include "../../../../inc/cuda/swarm/swarm_host.cuh"
#else  // GAMEDEV_ROCM
#include "../../../../inc/rocm/swarm/swarm_host.cuh"
#endif // GAMEDEV_CUDA

std::mt19937 rand_generator;

namespace
{

constexpr int query_count = 4;
const std::string output_path = PATH_OUT;

std::string suffix ()
{ return "_N" + std::to_string(VERIFY_RES) + ".dat"; }

void write_binary (const std::string &name, const std::vector<real> &values)
{
    std::string path = output_path + name + suffix();
    std::ofstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open output file: " + path);
    file.write(reinterpret_cast<const char *>(values.data()), sizeof(real)*values.size());
    if (!file) throw std::runtime_error("cannot write output file: " + path);
}

// probe trilinear interpolation and the imported-density Stokes closure at prescribed continuous grid coordinates
__global__
void import_probe (real *result, const real *query, const real *gas_dens,
    const real *gas_velx, const real *gas_vely, const real *gas_velz)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= query_count) return;

    real loc_x = query[idx];
    real loc_y = query[query_count + idx];
    real loc_z = query[2*query_count + idx];
    real x = X_MIN + loc_x*_get_dx();
    real y = Y_MIN*pow(_get_dy(), loc_y);
    real z = Z_MIN + loc_z*_get_dz();
    real R = _get_cyl_R(y, z);
    real Z = _get_cyl_Z(y, z);
    real h_g = _get_hg(R);

    result[idx] = _interp_field(gas_dens, loc_x, loc_y, loc_z);
    result[query_count + idx] = _interp_field(gas_velx, loc_x, loc_y, loc_z);
    result[2*query_count + idx] = _interp_field(gas_vely, loc_x, loc_y, loc_z);
    result[3*query_count + idx] = _interp_field(gas_velz, loc_x, loc_y, loc_z);
    result[4*query_count + idx] = _get_stokes(R, Z, h_g, S_0, x, y, z, gas_dens);
}

// isolate the analytical reference-density anchor and its response to a tenfold depleted gap
__global__
void anchor_probe (real *result, const real *gas_anchor, const real *gas_gap)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    constexpr real x = 0.0;
    constexpr real y = R_0;
    constexpr real z = 0.5*M_PI;
    real h_g = _get_hg(R_0);
    real loc_x = _get_loc_x(x);
    real loc_y = _get_loc_y(y);
    real loc_z = _get_loc_z(z);

    result[0] = _interp_field(gas_anchor, loc_x, loc_y, loc_z);
    result[1] = _get_stokes(R_0, 0.0, h_g, S_0, x, y, z, gas_anchor);
    result[2] = _interp_field(gas_gap, loc_x, loc_y, loc_z);
    result[3] = _get_stokes(R_0, 0.0, h_g, S_0, x, y, z, gas_gap);
}

bool profile_rejected (std::vector<real> &gas_dens, std::vector<real> &epsilon)
{
    real x, y, z;
    try
    {
        rand_from_file(&x, &y, &z, 1, gas_dens.data(), epsilon.data());
    }
    catch (const std::runtime_error &)
    {
        return true;
    }
    return false;
}

} // namespace

int main ()
{
    std::vector<real> gas_dens(N_G);
    std::vector<real> gas_velx(N_G);
    std::vector<real> gas_vely(N_G);
    std::vector<real> gas_velz(N_G);
    std::vector<real> gas_dens_next(N_G);
    std::vector<real> gas_velx_next(N_G);
    std::vector<real> gas_vely_next(N_G);
    std::vector<real> gas_velz_next(N_G);

    // use nonseparable affine fields so every directional stencil weight affects the reference value
    for (int iz = 0; iz < N_Z; iz++)
    {
        for (int iy = 0; iy < N_Y; iy++)
        {
            for (int ix = 0; ix < N_X; ix++)
            {
                int idx_cell = ix + iy*N_X + iz*N_X*N_Y;
                gas_dens[idx_cell] = 2.0 + 0.03*ix + 0.04*iy + 0.02*iz;
                gas_velx[idx_cell] = -0.2 + 0.01*ix - 0.02*iy + 0.03*iz;
                gas_vely[idx_cell] = 0.4 - 0.04*ix + 0.015*iy + 0.025*iz;
                gas_velz[idx_cell] = -0.1 + 0.02*ix + 0.03*iy - 0.01*iz;
                gas_dens_next[idx_cell] = 1.3*gas_dens[idx_cell] + 0.2;
                gas_velx_next[idx_cell] = 0.7*gas_velx[idx_cell] - 0.05;
                gas_vely_next[idx_cell] = 1.2*gas_vely[idx_cell] + 0.08;
                gas_velz_next[idx_cell] = 0.8*gas_velz[idx_cell] - 0.03;
            }
        }
    }

    std::vector<real> query = {
        1.5, 1.75, 3.75, 0.25,
        2.0, 2.35, 3.8, 0.1,
        1.5, 1.7, 2.2, 0.2,
    };
    real dy = _get_dy();
    real mesh_dim = _get_mesh_dim();
    real ref_y = std::log(
        (mesh_dim / (mesh_dim + 1.0))*(std::pow(dy, mesh_dim + 1.0) - 1.0)
        / (std::pow(dy, mesh_dim) - 1.0)
    ) / std::log(dy);
    query[query_count    ] += ref_y;
    query[query_count + 1] += ref_y;
    query[query_count + 2] += ref_y;

    real *dev_query = nullptr;
    real *dev_result = nullptr;
    real *dev_gas_dens = nullptr;
    real *dev_gas_velx = nullptr;
    real *dev_gas_vely = nullptr;
    real *dev_gas_velz = nullptr;
    real *dev_gas_dens_next = nullptr;
    real *dev_gas_velx_next = nullptr;
    real *dev_gas_vely_next = nullptr;
    real *dev_gas_velz_next = nullptr;
    qav_malloc(&dev_query, query.size(), "allocate import queries");
    qav_malloc(&dev_result, 5*query_count, "allocate import results");
    qav_malloc(&dev_gas_dens, N_G, "allocate gas density");
    qav_malloc(&dev_gas_velx, N_G, "allocate gas velocity x");
    qav_malloc(&dev_gas_vely, N_G, "allocate gas velocity y");
    qav_malloc(&dev_gas_velz, N_G, "allocate gas velocity z");
    qav_malloc(&dev_gas_dens_next, N_G, "allocate next gas density");
    qav_malloc(&dev_gas_velx_next, N_G, "allocate next gas velocity x");
    qav_malloc(&dev_gas_vely_next, N_G, "allocate next gas velocity y");
    qav_malloc(&dev_gas_velz_next, N_G, "allocate next gas velocity z");
    qav_copy_h2d(dev_query, query.data(), query.size(), "upload import queries");
    qav_copy_h2d(dev_gas_dens, gas_dens.data(), N_G, "upload gas density");
    qav_copy_h2d(dev_gas_velx, gas_velx.data(), N_G, "upload gas velocity x");
    qav_copy_h2d(dev_gas_vely, gas_vely.data(), N_G, "upload gas velocity y");
    qav_copy_h2d(dev_gas_velz, gas_velz.data(), N_G, "upload gas velocity z");
    qav_copy_h2d(dev_gas_dens_next, gas_dens_next.data(), N_G, "upload next gas density");
    qav_copy_h2d(dev_gas_velx_next, gas_velx_next.data(), N_G, "upload next gas velocity x");
    qav_copy_h2d(dev_gas_vely_next, gas_vely_next.data(), N_G, "upload next gas velocity y");
    qav_copy_h2d(dev_gas_velz_next, gas_velz_next.data(), N_G, "upload next gas velocity z");

    import_probe <<< 1, query_count >>> (
        dev_result, dev_query, dev_gas_dens, dev_gas_velx, dev_gas_vely, dev_gas_velz
    );
    qav_kernel_check("import_probe");
    std::vector<real> result(5*query_count);
    qav_copy_d2h(result.data(), dev_result, result.size(), "copy import results");
    write_binary("spatial", result);

    // incremental blends at fractions 0.25 and 0.70 must equal direct interpolation from the original pair of frames
    gas_lerp_calc <<< NB_G, TPB >>> (
        dev_gas_dens, dev_gas_velx, dev_gas_vely, dev_gas_velz,
        dev_gas_dens_next, dev_gas_velx_next, dev_gas_vely_next, dev_gas_velz_next, 0.25
    );
    qav_kernel_check("gas_lerp_calc at fraction 0.25");
    std::vector<real> temporal(4*N_G);
    qav_copy_d2h(temporal.data(), dev_gas_dens, N_G, "copy gas density at fraction 0.25");
    qav_copy_d2h(temporal.data() + N_G, dev_gas_velx, N_G, "copy gas velocity x at fraction 0.25");
    qav_copy_d2h(temporal.data() + 2*N_G, dev_gas_vely, N_G, "copy gas velocity y at fraction 0.25");
    qav_copy_d2h(temporal.data() + 3*N_G, dev_gas_velz, N_G, "copy gas velocity z at fraction 0.25");
    write_binary("temporal_25", temporal);

    gas_lerp_calc <<< NB_G, TPB >>> (
        dev_gas_dens, dev_gas_velx, dev_gas_vely, dev_gas_velz,
        dev_gas_dens_next, dev_gas_velx_next, dev_gas_vely_next, dev_gas_velz_next, 0.6
    );
    qav_kernel_check("gas_lerp_calc at fraction 0.70");
    qav_copy_d2h(temporal.data(), dev_gas_dens, N_G, "copy gas density at fraction 0.70");
    qav_copy_d2h(temporal.data() + N_G, dev_gas_velx, N_G, "copy gas velocity x at fraction 0.70");
    qav_copy_d2h(temporal.data() + 2*N_G, dev_gas_vely, N_G, "copy gas velocity y at fraction 0.70");
    qav_copy_d2h(temporal.data() + 3*N_G, dev_gas_velz, N_G, "copy gas velocity z at fraction 0.70");
    write_binary("temporal_70", temporal);

    real H_g0 = ASPR_0*R_0;
    real rhog_0 = SIGMA_0 / (std::sqrt(2.0*M_PI)*H_g0);
    std::vector<real> gas_anchor(N_G, rhog_0);
    std::vector<real> gas_gap(N_G, 0.1*rhog_0);
    qav_copy_h2d(dev_gas_dens, gas_anchor.data(), N_G, "upload reference gas density");
    qav_copy_h2d(dev_gas_dens_next, gas_gap.data(), N_G, "upload depleted gas density");
    anchor_probe <<< 1, 1 >>> (dev_result, dev_gas_dens, dev_gas_dens_next);
    qav_kernel_check("anchor_probe");
    std::vector<real> anchor(4);
    qav_copy_d2h(anchor.data(), dev_result, anchor.size(), "copy Stokes anchor results");
    write_binary("anchor", anchor);

    // sparse positive mass confirms that empty cells remain valid while invalid profiles are rejected before sampling
    std::vector<real> gas_host(N_G, 0.0);
    std::vector<real> epsilon(N_G, 0.0);
    constexpr int ix_active = 1;
    constexpr int iy_active = 3;
    constexpr int iz_active = 2;
    constexpr int idx_active = ix_active + iy_active*N_X + iz_active*N_X*N_Y;
    gas_host[idx_active] = 2.0;
    epsilon[idx_active] = 0.5;
    std::vector<real> randposx(N_P), randposy(N_P), randposz(N_P);
    rand_generator.seed(73);
    rand_from_file(randposx.data(), randposy.data(), randposz.data(), N_P, gas_host.data(), epsilon.data());
    bool support_valid = true;
    for (int idx = 0; idx < N_P; idx++)
    {
        int ix = static_cast<int>(std::floor((randposx[idx] - X_MIN) / _get_dx()));
        int iy = static_cast<int>(std::floor(std::log(randposy[idx] / Y_MIN) / std::log(_get_dy())));
        int iz = static_cast<int>(std::floor((randposz[idx] - Z_MIN) / _get_dz()));
        support_valid = support_valid && ix == ix_active && iy == iy_active && iz == iz_active;
    }

    std::vector<real> checks(7);
    checks[0] = support_valid ? 1.0 : 0.0;
    gas_host.assign(N_G, 0.0);
    epsilon.assign(N_G, 1.0);
    checks[1] = profile_rejected(gas_host, epsilon) ? 1.0 : 0.0;
    gas_host[0] = -1.0;
    checks[2] = profile_rejected(gas_host, epsilon) ? 1.0 : 0.0;
    gas_host[0] = 1.0;
    epsilon[0] = -1.0;
    checks[3] = profile_rejected(gas_host, epsilon) ? 1.0 : 0.0;
    epsilon[0] = 1.0;
    gas_host[0] = std::numeric_limits<real>::quiet_NaN();
    checks[4] = profile_rejected(gas_host, epsilon) ? 1.0 : 0.0;
    gas_host[0] = 1.0;
    epsilon[0] = std::numeric_limits<real>::infinity();
    checks[5] = profile_rejected(gas_host, epsilon) ? 1.0 : 0.0;
    gas_host[0] = std::numeric_limits<real>::max();
    epsilon[0] = std::numeric_limits<real>::max();
    checks[6] = profile_rejected(gas_host, epsilon) ? 1.0 : 0.0;
    write_binary("profile_checks", checks);

    std::ofstream meta(output_path + "meta_N" + std::to_string(VERIFY_RES) + ".json");
    meta << "{\n"
         << "  \"case\": \"import_3d\",\n"
         << "  \"resolution\": " << VERIFY_RES << ",\n"
         << "  \"np\": " << N_P << ",\n"
         << "  \"nx\": " << N_X << ",\n"
         << "  \"ny\": " << N_Y << ",\n"
         << "  \"nz\": " << N_Z << "\n"
         << "}\n";
    if (!meta) throw std::runtime_error("cannot write imported-gas metadata");

    qav_free(dev_query, "free import queries");
    qav_free(dev_result, "free import results");
    qav_free(dev_gas_dens, "free gas density");
    qav_free(dev_gas_velx, "free gas velocity x");
    qav_free(dev_gas_vely, "free gas velocity y");
    qav_free(dev_gas_velz, "free gas velocity z");
    qav_free(dev_gas_dens_next, "free next gas density");
    qav_free(dev_gas_velx_next, "free next gas velocity x");
    qav_free(dev_gas_vely_next, "free next gas velocity y");
    qav_free(dev_gas_velz_next, "free next gas velocity z");

    return 0;
}
