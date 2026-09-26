#pragma once
#define get_total_dust_mass _disk_total_dust_mass
#define rand_disk_mono _disk_rand_mono
#define save_variable _production_save_variable
#include "../../../../../inc/swarm/swarm_host.cuh"
#undef get_total_dust_mass
#undef rand_disk_mono
#undef save_variable
inline real get_total_dust_mass (const std::vector<real>&) { return BENCHMARK_MASS; }
inline __host__
void rand_disk_mono (
    real *randposx, real *randposy, real *randposz, real, int count
)
{
    const real radial_span = Y_MAX - Y_MIN;
    const real azimuth_span = X_MAX - X_MIN;
    int radial_count = std::max(1, static_cast<int>(std::round(std::sqrt(
        static_cast<real>(count)*radial_span / (R_0*azimuth_span)
    ))));

    int azimuth_base = count / radial_count;
    int azimuth_extra = count % radial_count;
    std::mt19937 position_generator(SEED + 1);
    std::uniform_real_distribution <real> jitter(-0.25, 0.25);

    int idx = 0;
    for (int idx_radial = 0; idx_radial < radial_count; idx_radial++)
    {
        int azimuth_count = azimuth_base + static_cast<int>(idx_radial < azimuth_extra);
        for (int idx_azimuth = 0; idx_azimuth < azimuth_count; idx_azimuth++, idx++)
        {
            randposx[idx] = X_MIN + azimuth_span
                *(static_cast<real>(idx_azimuth) + 0.5 + jitter(position_generator))
                / static_cast<real>(azimuth_count);
            randposy[idx] = Y_MIN + radial_span
                *(static_cast<real>(idx_radial) + 0.5 + jitter(position_generator))
                / static_cast<real>(radial_count);
            randposz[idx] = 0.5*M_PI;
        }
    }
}

inline bool save_variable (const std::string &path, real mass)
{
    if (!_production_save_variable(path, mass)) return false;
    std::ofstream file(path, std::ios::app);
    file << "\n[CAMPAIGN]\nSEED = " << SEED
         << "\nPOSITION_SEED = " << SEED + 1 << "\nCOLLISION_SEED = " << SEED + 1
         << "\nUNIT_VOLUME = 1\nGEOMETRY_REUSE = 1"
            "\nSIZE_BIN_POLICY = moving_per_group\nSIZE_MIN_FACTOR = 0.5\nSIZE_MAX_FACTOR = 8\n";
    return static_cast<bool>(file);
}
