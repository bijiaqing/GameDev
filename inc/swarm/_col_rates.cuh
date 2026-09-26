#ifndef GAMEDEV_SWARM_COL_RATES_CUH
#define GAMEDEV_SWARM_COL_RATES_CUH

// device pair-rate evaluation, uniform draws, and atomic maxima for the collision chain

#ifdef COLLISION
#include <_col_image.cuh>
#include <_col_types.cuh>
#include <_collision.cuh>

#ifdef COL_QUERY_ENV_CACHE
// positions and gas are fixed throughout one collision half-operator
__device__ __forceinline__ query_environment _cache_query_environment (const swarm &p)
{
    real R = _get_cyl_R(p.position.y, p.position.z);
    real Z = _get_cyl_Z(p.position.y, p.position.z);
    real h = _get_hg(R);
    real omega = _get_omegaK(R);
    real cs = _get_cs(R, h);
    real alpha = _get_alpha(R, h);
    real strat = _get_gas_strat(R, Z, h);
    return {Z, omega, -_get_eta(R, Z, h)*R*omega, pow(R / R_0, IDX_P), strat, cs,
        _get_re_inv_sqrt(R, alpha, _get_sigma_g(R)*strat), 1.5*alpha*cs*cs};
}
// reproduce _get_stokes for analytic gas from the cached radial and vertical scalings
__device__ __forceinline__ real _cached_stokes (const query_environment &e, real size)
{
    real st = STOKES_0*(size / S_0);
    st /= e.radial;
    st /= e.strat;
    return st;
}
// reproduce the _get_vrel_t regime algebra with precomputed gas coefficients
__device__ __forceinline__ real _cached_turbulence (const query_environment &e, real stokes_i, real stokes_j)
{
    real re_inv_sqrt = e.re_inv_sqrt;
    real vg_sq = e.vg_sq;
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

// reproduce _get_vrel_pair with drift, settling, turbulence, and Brownian terms from the cached environment
__device__ __forceinline__ real _cached_pair_velocity (const query_environment &e, real size_i, real size_j)
{
    real si = _cached_stokes(e, size_i);
    real sj = _cached_stokes(e, size_j);
    real fi = 1.0 / (1.0 + si*si);
    real fj = 1.0 / (1.0 + sj*sj);
    real dvr = 2.0*e.vn*(si*fi - sj*fj);
    real dvphi = e.vn*(fi - fj);
    real dvz = e.Z*e.omega*(fmin(si, 0.5) - fmin(sj, 0.5));
    real vt = _cached_turbulence(e, si, sj);
    real mi = _get_grain_mass(size_i);
    real mj = _get_grain_mass(size_j);
    real vb = fmin(sqrt(8.0*e.cs*e.cs*M_MOL*(mi + mj) / (M_PI*mi*mj)), e.cs);
    return sqrt(dvr*dvr + dvphi*dvphi + dvz*dvz + vt*vt + vb*vb);
}
#endif // COL_QUERY_ENV_CACHE

// draw a uniform deviate strictly below one so -log(U) and inverse-CDF selection stay finite
__device__ __forceinline__
real _get_col_uniform (curs *rngstate)
{
    return fmin(gpuRandUniformDouble(rngstate), nextafter(1.0, 0.0));
}

// atomically raise a double to value with a compare-and-swap loop
__device__ __forceinline__
void _col_atomic_max (real *address, real value)
{
    auto integer = reinterpret_cast<unsigned long long *>(address);
    unsigned long long old = *integer;
    while (value > __longlong_as_double(static_cast<long long>(old)))
    {
        unsigned long long assumed = old;
        old = atomicCAS(integer, assumed,
            static_cast<unsigned long long>(__double_as_longlong(value)));
        if (old == assumed) break;
    }
}

// evaluate one pair with the same synthetic or physical normalization as _get_col_rate_ij
template <kernel_type kernel> __device__ __forceinline__
real _get_col_chain_rate (const swarm *dev_particle, real size_i,
    const real *dev_size_old, const real *dev_numr_old,
    #ifdef IMPORTGAS
    const real *dev_gas_dens,
    #endif // IMPORTGAS
    int idx_old_i, int idx_old_j, int image_j, real lambda_0, real &vrel,
    const query_environment *environment)
{
    vrel = 0.0;
    real size_j = dev_size_old[idx_old_j];
    // include the owner's own swarm when i == j, using the large-number approximation N_i - 1 ~= N_i
    real numr_j = dev_numr_old[idx_old_j];
    if constexpr (kernel == CONSTANT_KERNEL)
    {
        return lambda_0*numr_j;
    }
    else if constexpr (kernel == LINEAR_KERNEL)
    {
        return lambda_0*numr_j*(_get_grain_mass(size_i) + _get_grain_mass(size_j));
    }
    else if constexpr (kernel == PRODUCT_KERNEL)
    {
        return lambda_0*numr_j*_get_grain_mass(size_i)*_get_grain_mass(size_j);
    }
    else if constexpr (kernel == CUSTOM_KERNEL)
    {
        #ifdef COL_QUERY_ENV_CACHE
        vrel = _cached_pair_velocity(environment[idx_old_i], size_i, size_j);
        #else  // !COL_QUERY_ENV_CACHE
        vrel = _get_vrel_pair(dev_particle, size_i, size_j, idx_old_i, idx_old_j, image_j
            #ifdef IMPORTGAS
            , dev_gas_dens
            #endif // IMPORTGAS
        );
        #endif // COL_QUERY_ENV_CACHE
        real rate = numr_j*vrel*M_PI*(size_i + size_j)*(size_i + size_j) / 4.0;
        if constexpr (N_Z == 1)
        {
            real R_i = _get_cyl_R(
                dev_particle[idx_old_i].position.y, dev_particle[idx_old_i].position.z
            );
            real R_j = _get_cyl_R(
                dev_particle[idx_old_j].position.y, dev_particle[idx_old_j].position.z
            );
            real H_gi = R_i*_get_hg(R_i);
            real H_gj = R_j*_get_hg(R_j);
            rate /= sqrt(2.0*M_PI*(H_gi*H_gi + H_gj*H_gj));
        }
        return rate;
    }
    else
    {
        assert(false);
        return 0.0;
    }
}

#endif // COLLISION

#endif // GAMEDEV_SWARM_COL_RATES_CUH
