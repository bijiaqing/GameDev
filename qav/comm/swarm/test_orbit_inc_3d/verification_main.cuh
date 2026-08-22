#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <_transport.cuh>
#include <device_api.cuh>
#include <swarm_kern.cuh>

// test-only driver for an inclined drag-free orbit
// DIFFUSION satisfies the full-3D configuration contract but the zero-diffusivity test closure is not scheduled

namespace
{
const std::string output_path = PATH_OUT;

real eccentric_anomaly (real mean_anomaly, real eccentricity)
{
    real anomaly = mean_anomaly;
    for (int iteration = 0; iteration < 20; iteration++)
    {
        anomaly -= (anomaly - eccentricity*sin(anomaly) - mean_anomaly)/(1.0 - eccentricity*cos(anomaly));
    }
    return anomaly;
}

real3 rotate_orbit (real xp, real yp, real node, real inclination, real periapsis)
{
    real x1 = cos(periapsis)*xp - sin(periapsis)*yp;
    real y1 = sin(periapsis)*xp + cos(periapsis)*yp;
    real y2 = cos(inclination)*y1;
    real z2 = sin(inclination)*y1;
    return make_double3(cos(node)*x1 - sin(node)*y2, sin(node)*x1 + cos(node)*y2, z2);
}

void write_binary (const std::string &name, const std::vector<real> &values)
{
    std::ofstream file(output_path + name + "_N" + std::to_string(VERIFY_RES) + ".dat", std::ios::binary);
    if (!file) throw std::runtime_error("cannot open inclined-orbit output");
    file.write(reinterpret_cast<const char *>(values.data()), sizeof(real)*values.size());
    if (!file) throw std::runtime_error("cannot write inclined-orbit output");
}
}

int main ()
{
    constexpr real semimajor = 1.0;
    constexpr real eccentricity = 0.2;
    constexpr real inclination = 0.3;
    constexpr real node = 0.4;
    constexpr real periapsis = 0.5;
    constexpr real time_end = 1.3;
    const real mean_initial[N_P] = {0.1, 0.7, 1.4, 2.2};

    std::vector<swarm> particle(N_P);
    std::vector<real> mean_anomaly(N_P);
    for (int idx = 0; idx < N_P; idx++)
    {
        real E = eccentric_anomaly(mean_initial[idx], eccentricity);
        real denom = 1.0 - eccentricity*cos(E);
        real dE_dt = sqrt(G*M_S/(semimajor*semimajor*semimajor)) / denom;
        real3 pos = rotate_orbit(semimajor*(cos(E) - eccentricity),
            semimajor*sqrt(1.0 - eccentricity*eccentricity)*sin(E), node, inclination, periapsis);
        real3 vel = rotate_orbit(-semimajor*sin(E)*dE_dt,
            semimajor*sqrt(1.0 - eccentricity*eccentricity)*cos(E)*dE_dt, node, inclination, periapsis);
        real y = sqrt(pos.x*pos.x + pos.y*pos.y + pos.z*pos.z);
        real x = atan2(pos.y, pos.x);
        real z = acos(pos.z/y);
        real sinz = sin(z), cosz = cos(z), sinx = sin(x), cosx = cos(x);
        real vy = vel.x*sinz*cosx + vel.y*sinz*sinx + vel.z*cosz;
        real vphi = -vel.x*sinx + vel.y*cosx;
        real vtheta = vel.x*cosz*cosx + vel.y*cosz*sinx - vel.z*sinz;
        particle[idx].position = make_double3(x, y, z);
        particle[idx].velocity = make_double3(y*sinz*vphi, vy, y*vtheta);
        mean_anomaly[idx] = mean_initial[idx];
    }

    swarm *dev_particle;
    qav_malloc(&dev_particle, N_P, "allocate inclined-orbit particles");
    qav_copy_h2d(dev_particle, particle.data(), N_P, "upload inclined-orbit particles");
    const real dt = time_end/static_cast<real>(VERIFY_RES);
    for (int step = 0; step < VERIFY_RES; step++)
    {
        ssa_transport <<< NB_P, TPB >>> (dev_particle, dt);
    }
    qav_kernel_check("inclined Kepler transport");
    qav_copy_d2h(particle.data(), dev_particle, N_P, "copy inclined-orbit particles");

    std::vector<real> state(6*N_P);
    for (int idx = 0; idx < N_P; idx++)
    {
        state[idx] = particle[idx].position.x;
        state[N_P + idx] = particle[idx].position.y;
        state[2*N_P + idx] = particle[idx].position.z;
        state[3*N_P + idx] = particle[idx].velocity.x;
        state[4*N_P + idx] = particle[idx].velocity.y;
        state[5*N_P + idx] = particle[idx].velocity.z;
    }
    write_binary("state", state);
    write_binary("mean_anomaly", mean_anomaly);

    std::ofstream meta(output_path + "meta_N" + std::to_string(VERIFY_RES) + ".json");
    meta << std::setprecision(17)
         << "{\n"
         << "  \"case\": \"orbit_inc_3d\",\n"
         << "  \"resolution\": " << VERIFY_RES << ",\n"
         << "  \"np\": " << N_P << ",\n"
         << "  \"time\": " << time_end << ",\n"
         << "  \"semimajor\": " << semimajor << ",\n"
         << "  \"eccentricity\": " << eccentricity << ",\n"
         << "  \"inclination\": " << inclination << ",\n"
         << "  \"node\": " << node << ",\n"
         << "  \"periapsis\": " << periapsis << ",\n"
         << "  \"zero_drag_specialization\": " << (QAV_ZERO_DRAG_ACTIVE ? "true" : "false") << ",\n"
         << "  \"zero_diffusivity_specialization\": " << (NU == 0.0 ? "true" : "false") << ",\n"
         << "  \"transport_only_schedule\": true\n"
         << "}\n";

    qav_free(dev_particle, "free inclined-orbit particles");
    std::cout << "swarm inclined orbit completed at N=" << VERIFY_RES << std::endl;
    return 0;
}
