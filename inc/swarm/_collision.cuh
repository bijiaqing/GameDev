#ifndef GAMEDEV_SWARM_COLLISION_CUH
#define GAMEDEV_SWARM_COLLISION_CUH

#include <_col_image.cuh>
#include <const_defs.cuh>

#ifdef COLLISION

#include <cassert>      // assert

#include <param_grid.cuh>
#include <param_phys.cuh>
#include <swarm_grid.cuh>

#ifdef COLLISION_KDTREE
#include <kdtree/index_heap.cuh> // idx_old_heap
#include <kdtree/knn.h>          // kdtree::cct::knn

using kdtree_heap = idx_old_heap<N_K, kdtree_node>;
#endif // COLLISION_KDTREE

enum kernel_type
{
    CONSTANT_KERNEL = 0,
    LINEAR_KERNEL   = 1,
    PRODUCT_KERNEL  = 2,
    CUSTOM_KERNEL   = 3,
};

// =====================================================================================================================
// resolved velocity and KNN geometry
// =====================================================================================================================

// reconstruct one particle's physical Cartesian velocity from its spherical stored variables
__device__ __forceinline__
real3 _get_cart_vel (const swarm &particle, int image = 0)
{
    real x = particle.position.x;
    if constexpr (X_WEDGE) x += _get_col_image_shift(image)*(X_MAX - X_MIN);
    real y = particle.position.y;
    real z = particle.position.z;

    real R = _get_cyl_R(y, z);
    real vx = particle.velocity.x / R;
    real vy = particle.velocity.y;
    real vz = particle.velocity.z / y;

    real3 v_cart;

    if constexpr (N_Z == 1)
    {
        v_cart.x = vy*cos(x) - vx*sin(x);
        v_cart.y = vy*sin(x) + vx*cos(x);
        v_cart.z = 0.0;
        return v_cart;
    }

    v_cart.x = vy*sin(z)*cos(x) + vz*cos(z)*cos(x) - vx*sin(x);
    v_cart.y = vy*sin(z)*sin(x) + vz*cos(z)*sin(x) + vx*cos(x);
    v_cart.z = vy*cos(z)        - vz*sin(z);

    return v_cart;
}

// approximate the part of a local KNN ball lying inside radial and polar domain boundaries
__device__ __forceinline__
real _get_ball_measure (real y, real z, real radius)
{
    if (radius <= 0.0) return 0.0;

    if constexpr (N_X == 1 && N_Z == 1)
    {
        real R_lo = fmax(Y_MIN, y - radius);
        real R_hi = fmin(Y_MAX, y + radius);

        return (R_hi > R_lo) ? M_PI*(R_hi*R_hi - R_lo*R_lo) : 0.0;
    }

    int dim = 1 + static_cast<int>(N_X > 1) + static_cast<int>(N_Z > 1);
    real measure = (dim == 1) ? 2.0*radius : (dim == 2) ? M_PI*radius*radius : 4.0*M_PI*radius*radius*radius / 3.0;

    real distances[4] = {y - Y_MIN, Y_MAX - y, 1.0e+100, 1.0e+100};
    if (N_Z > 1)
    {
        distances[2] = y*(z - Z_MIN);
        distances[3] = y*(Z_MAX - z);
    }

    int bounds = (N_Z > 1) ? 4 : 2;
    for (int idx_bound = 0; idx_bound < bounds; idx_bound++)
    {
        real d = fmax(0.0, distances[idx_bound]);
        if (d >= radius) continue;

        if (dim == 1)
        {
            measure *= 0.5*(1.0 + d / radius);
        }
        else if (dim == 2)
        {
            real cap = radius*radius*acos(d / radius) - d*sqrt(radius*radius - d*d);
            measure *= 1.0 - cap / (M_PI*radius*radius);
        }
        else
        {
            real cap = M_PI*(radius - d)*(radius - d)*(2.0*radius + d) / 3.0;
            measure *= 1.0 - cap / (4.0*M_PI*radius*radius*radius / 3.0);
        }
    }

    // revolve a reduced-dimensional axisymmetric neighborhood around the complete ring
    if (N_X == 1) measure *= 2.0*M_PI*_get_cyl_R(y, z);

    return measure;
}

#ifndef CODE_UNIT // physical units
// calculate the Brownian relative speed and cap it at the sound speed
__device__ __forceinline__
real _get_vrel_b (real R, real size_i, real size_j, real h_g)
{
    // Brownian motion-induced relative velocity v = sqrt(8*k_B*T*(m_i+m_j) / (pi*m_i*m_j))
    // here we take c_s^2 = k_B*T / mmw_gas

    real c_s = _get_cs(R, h_g);

    real m_i = _get_grain_mass(size_i);
    real m_j = _get_grain_mass(size_j);

    real vrel_b = sqrt(8.0*c_s*c_s*M_MOL*(m_i + m_j) / (M_PI*m_i*m_j));

    return min(vrel_b, c_s);
}
#endif // NOT CODE_UNIT

// calculate the inverse square root of the turbulent Reynolds number
__device__ __forceinline__
real _get_re_inv_sqrt (real R, real alpha, real sigma_g)
{
    real reynolds = 1.0;

    #ifdef CODE_UNIT
    real alpha_0 = _get_alpha(R_0, ASPR_0);
    reynolds = REYNOLDS_0*(alpha / alpha_0)*(sigma_g / SIGMA_0);
    #else  // PHYSICAL_UNIT
    reynolds = 0.5*alpha*sigma_g*X_SEC / M_MOL;
    #endif // CODE_UNIT

    return 1.0 / sqrt(reynolds);
}

// calculate the turbulence-induced relative speed using the Ormel-Cuzzi regimes
__device__ __forceinline__
real _get_vrel_t (real R, real stokes_i, real stokes_j, real h_g, real sigma_g)
{
    // turbulence-induced relative velocity based on Ormel & Cuzzi (2007), A&A, 466, 413
    // part of code adapted from DustPy (Stammler & Birnstiel 2022, ApJ, 935, 35), also see:
    // https://github.com/stammler/dustpy/blob/48e6c05b2b9c2a91ca35a0945f3138dc3aa34685/dustpy/std/dust.f90#L1402
    // important concepts mentioned in OC07 and used (but not necessarily coded) here include:
    // (0)  Class I  eddies (t_k > t_stop): particles follow eddy motions systematically
    //      Class II eddies (t_k < t_stop): eddies fluctuate too fast, act as random kicks to particles
    // (1)  t_large: turnover timescale of the largest  eddies (integral scale)
    //      t_large = omega_K^(-1)                                          (page 413, section 2)
    // (2)  t_small: turnover timescale of the smallest eddies (Kolmogorov scale)
    //      t_small = t_eta = Re^(-1/2)*t_large                             (page 414, section 2)
    // (3)  t_k: turnover timescale of arbitrary eddy k with spatial scale l = 1 / k and velocity V(k)
    //      t_k = l / V(k) = (k*V(k))^(-1)                                  (page 413, section 2)
    // (4)  t_cross: eddy crossing timescale due to particle-eddy relative velocity
    //      t_cross = l / V_rel = (k*V_rel(k))^(-1)                         (page 414, section 2)
    // (5)  t_stop: particle stopping time (friction timescale)
    //      t_stop = St*t_large = St*omega_K^(-1)                           (page 413, section 2)
    // (6)  t_star: boundary eddy turnover time separating Class I and Class II eddies
    //      t_star = 1.6*t_stop for St << 1                                 (eq. 21d)
    //      t_star = (t_stop^(-1) - t_cross^(-1))^(-1)                      (eq. 3)
    // (7)  v_large: turbulent velocity at the largest  eddy (integral scale)
    //      v_large = c_s*alpha^(1/2)                                       (page 416, section 3.3)
    // (8)  v_small: turbulent velocity at the smallest eddy (Kolmogorov scale)
    //      v_small = Re^(-1/4)*v_large                                     (page 417, section 3.4.1)

    real c_s = _get_cs(R, h_g);
    real alpha = _get_alpha(R, h_g);
    real re_inv_sqrt = _get_re_inv_sqrt(R, alpha, sigma_g);

    // comes from normalizing the power spectrum                            (page 415, section 3.2)
    real vg_sq = 1.5*alpha*c_s*c_s;

    real stokes_large, stokes_small, eps;

    if (stokes_i >= stokes_j)
    {
        stokes_large = stokes_i;
        stokes_small = stokes_j;
    }
    else
    {
        stokes_large = stokes_j;
        stokes_small = stokes_i;
    }

    eps = stokes_small / stokes_large;

    // y_a = t_star / t_stop = 1.6 is the solution to y_star when St << 1
    // y_s is an empirical polynomial fit to the exact solution of y_star (eq. 21d)
    real y_a = 1.6;
    real y_s = 1.6015125;

    // taken from DustPy
    y_s += -0.63119577*stokes_large;
    y_s +=  0.32938936*stokes_large*stokes_large;
    y_s += -0.29847604*stokes_large*stokes_large*stokes_large;

    real vrel_sq = 0.0;

    if (stokes_large < 0.2*re_inv_sqrt)
    {
        // regime 1: very small particles (t_stop_large << t_small) following eq. 27

        vrel_sq = vg_sq*(stokes_large - stokes_small)*(stokes_large - stokes_small) / re_inv_sqrt;
    }
    else if (stokes_large < re_inv_sqrt / y_a)
    {
        // regime 2: transition near t_small boundary (t_stop_large ~ t_small) following eq. 26

        vrel_sq = vg_sq*(stokes_large - stokes_small) / (stokes_large + stokes_small);
        vrel_sq *= (stokes_large / (1.0 + re_inv_sqrt / stokes_large) - stokes_small / (1.0 + re_inv_sqrt
            / stokes_small));
    }
    else if (stokes_large < 5.0*re_inv_sqrt)
    {
        // regime 3: intermediate coupling (t_small < t_stop_large < 5*t_small)

        real coeff = 0.0;
        // coefficient of delta_VI^2  following eq. 17
        coeff  = (stokes_large - stokes_small) / (stokes_large + stokes_small);
        coeff *= (stokes_large / (1.0 + y_a) - stokes_small*stokes_small / (stokes_small + y_a*stokes_large));
        // coefficient of delta_VII^2 following eq. 18
        coeff += 2.0*(y_a*stokes_large - re_inv_sqrt) + stokes_large / (1.0 + y_a);
        coeff -= stokes_large*stokes_large / (stokes_large + re_inv_sqrt);
        coeff += stokes_small*stokes_small / (y_a*stokes_large + stokes_small);
        coeff -= stokes_small*stokes_small / (stokes_small + re_inv_sqrt);

        vrel_sq = vg_sq*coeff;
    }
    else if (stokes_large < 0.2)
    {
        // regime 4: fully intermediate regime (5t_small < t_stop_large < 0.2t_large) following eq. 28

        vrel_sq = vg_sq*stokes_large;
        vrel_sq *= (2.0*y_a - (1.0 + eps) + 2.0 / (1.0 + eps)*(1.0 / (1.0 + y_a) + eps*eps*eps / (y_a + eps)));
    }
    else if (stokes_large < 1.0)
    {
        // regime 5: transition near t_large boundary (0.2t_large < t_stop_large < t_large)
        // following eq. 28, but uses the empirical y_s fit instead of the fixed y_a = 1.6

        vrel_sq = vg_sq*stokes_large;
        vrel_sq *= (2.0*y_s - (1.0 + eps) + 2.0 / (1.0 + eps)*(1.0 / (1.0 + y_s) + eps*eps*eps / (y_s + eps)));
    }
    else
    {
        // regime 6: heavy particles (t_stop_large >= t_large) following eq. 29

        vrel_sq = vg_sq*(1.0 / (1.0 + stokes_large) + 1.0 / (1.0 + stokes_small));
    }

    if (vrel_sq < 0.0)
    {
        printf("ERROR: negative vrel_sq in _get_vrel_t\n");
        assert(false);
    }

    return sqrt(vrel_sq);
}

// combine query-local drift, settling, Brownian, and turbulent relative speeds for two explicit grain sizes
// both Stokes numbers use the owner's position; partner position, image, and instantaneous velocities do not enter
__device__ __forceinline__
real _get_vrel_pair (const swarm *dev_particle, real size_i, real size_j,
    int idx_old_i, int idx_old_j, int image_j
    #ifdef IMPORTGAS
    , const real *dev_gas_dens
    #endif // IMPORTGAS
)
{
    real y = dev_particle[idx_old_i].position.y;
    real z = dev_particle[idx_old_i].position.z;
    real R = _get_cyl_R(y, z);
    real Z = _get_cyl_Z(y, z);
    real h = _get_hg(R);
    real omega = _get_omegaK(R);
    real si = _get_stokes(R, Z, h, size_i
        #ifdef IMPORTGAS
        , dev_particle[idx_old_i].position.x, y, z, dev_gas_dens
        #endif // IMPORTGAS
    );
    real sj = _get_stokes(R, Z, h, size_j
        #ifdef IMPORTGAS
        , dev_particle[idx_old_i].position.x, y, z, dev_gas_dens
        #endif // IMPORTGAS
    );
    // use the analytic pressure-drift speed vn = (dP/dR)/(2 rho Omega), as in mcdust
    real vn = -_get_eta(R, Z, h)*R*omega;
    real fi = 1.0 / (1.0 + si*si);
    real fj = 1.0 / (1.0 + sj*sj);
    real dvr = 2.0*vn*(si*fi - sj*fj); // benchmark gas radial velocity is zero
    real dvphi = vn*(fi - fj);
    // cap the terminal settling speed at St = 0.5, beyond which grains oscillate about the midplane
    real dvz = Z*omega*(fmin(si, 0.5) - fmin(sj, 0.5));
    // use the effective local column so Re = sqrt(pi/2)*alpha*rho*cs*sigma/(Omega*m_mol)
    real sigma_local = _get_sigma_g(R)*_get_gas_strat(R, Z, h);
    #ifdef IMPORTGAS
    real density = _interp_field(dev_gas_dens,
        _get_loc_x(dev_particle[idx_old_i].position.x), _get_loc_y(y), _get_loc_z(z));
    if constexpr (N_Z == 1) sigma_local = density;
    else sigma_local = sqrt(2.0*M_PI)*density*h*R;
    #endif // IMPORTGAS
    real vt = _get_vrel_t(R, si, sj, h, sigma_local);
    // omit Brownian motion in code units, which provide no molecular mass in simulation mass units
    real vb = 0.0;
    #ifndef CODE_UNIT
    vb = _get_vrel_b(R, size_i, size_j, h);
    #endif // !CODE_UNIT
    return sqrt(dvr*dvr + dvphi*dvphi + dvz*dvz + vt*vt + vb*vb);

}

// combine query-local relative speeds for two frozen grain sizes
__device__ __forceinline__
real _get_vrel (const swarm *dev_particle, const real *dev_size_old,
    int idx_old_i, int idx_old_j, int image_j
    #ifdef IMPORTGAS
    , const real *dev_gas_dens
    #endif // IMPORTGAS
)
{
    return _get_vrel_pair(
        dev_particle, dev_size_old[idx_old_i], dev_size_old[idx_old_j], idx_old_i, idx_old_j, image_j
        #ifdef IMPORTGAS
        , dev_gas_dens
        #endif // IMPORTGAS
    );
}

// =====================================================================================================================
// pair collision propensity
// =====================================================================================================================

// calculate the pair-propensity numerator before division by the local KNN measure
template <kernel_type kernel> __device__ __forceinline__
real _get_col_rate_ij (const swarm *dev_particle, const real *dev_size_old, const real *dev_numr_old,
    #ifdef IMPORTGAS
    const real *dev_gas_dens,
    #endif // IMPORTGAS
    int idx_old_i, int idx_old_j, int image_j, real lambda_0
)
{
    // text mainly from Drazkowska et al. 2013:
    // we assume that a limited number n representative particles represent all N physical particles
    // each representative particle i describes a swarm of N_i identical physical particles
    // as n << N, we only need to consider the collisions between representative and non-representative particles
    // the probability of a physical collision between particles i and j is determined as
    // lambda_ij = N_j * K_ij / V, where K_ij is the coagulation kernel and V is the local measure
    // synthetic kernels instead multiply their normalized kernel shape by the supplied lambda_0

    // include the owner's own swarm when i == j, using the large-number approximation N_i - 1 ~= N_i
    real numr_j = dev_numr_old[idx_old_j];

    if constexpr (kernel == CONSTANT_KERNEL)
    {
        return lambda_0*numr_j;
    }
    else if constexpr (kernel == LINEAR_KERNEL)
    {
        real size_i = dev_size_old[idx_old_i];
        real size_j = dev_size_old[idx_old_j];

        // m_i + m_j
        return lambda_0*numr_j*(_get_grain_mass(size_i) + _get_grain_mass(size_j));
    }
    else if constexpr (kernel == PRODUCT_KERNEL)
    {
        real size_i = dev_size_old[idx_old_i];
        real size_j = dev_size_old[idx_old_j];

        // m_i * m_j
        return lambda_0*numr_j*_get_grain_mass(size_i)*_get_grain_mass(size_j);
    }
    else if constexpr (kernel == CUSTOM_KERNEL)
    {
        // use K_ij = sigma_ij delta_v_ij for the physical collision kernel

        real size_i = dev_size_old[idx_old_i];
        real size_j = dev_size_old[idx_old_j];

        real vrel_ij = _get_vrel(dev_particle, dev_size_old, idx_old_i, idx_old_j, image_j
            #ifdef IMPORTGAS
            , dev_gas_dens
            #endif // IMPORTGAS
        );
        real sigma_ij = M_PI*(size_i + size_j)*(size_i + size_j) / 4.0;

        real rate_numer = numr_j*vrel_ij*sigma_ij;
        if (N_Z == 1)
        {
            // convert the vertically integrated neighbor area to an effective pair volume
            real R_i = _get_cyl_R(
                dev_particle[idx_old_i].position.y, dev_particle[idx_old_i].position.z
            );
            real R_j = _get_cyl_R(
                dev_particle[idx_old_j].position.y, dev_particle[idx_old_j].position.z
            );
            real H_gi = R_i*_get_hg(R_i);
            real H_gj = R_j*_get_hg(R_j);

            // assume every species shares the gas vertical profile in a vertically integrated disk
            rate_numer /= sqrt(2.0*M_PI*(H_gi*H_gi + H_gj*H_gj));
        }

        return rate_numer;
    }
    else
    {
        // kernel is a compile-time constant
        if (threadIdx.x == 0 && blockIdx.x == 0)
        {
            printf("ERROR: Invalid COAG_KERNEL value = %d\n", static_cast<int>(kernel));
        }

        assert(false);
        return 0.0; // unreachable, but prevents compiler warning
    }
}

#endif // COLLISION

// =====================================================================================================================

#endif // GAMEDEV_SWARM_COLLISION_CUH
