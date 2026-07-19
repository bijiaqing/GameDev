#ifdef DIFFUSION

#include <graffiti_kern.cuh>
#include <helpers.cuh>

// ---- X direction (azimuthal) -- Cyclic Thomas / Sherman-Morrison (CN implicit) ------
// Periodic BC requires a cyclic tridiagonal solve (Sherman-Morrison decomposition).
// Within an analytic-gas ring all cells share the same Dx and rho_g (same Rc and z), so
// turbulent diffusion of q=rho_d/rho_g has uniform off-diagonal coefficients.
//
// Sherman-Morrison: A = A' + u*v^T
//   A  = cyclic tridiagonal (cn_wrap entries a[0] = c[N-1] = -cn_coeff)
//   A' = A with corners zeroed, diagonals b'[0] = 2*cn_diag, b'[N-1] = cn_diag + cn_coeff^2/cn_diag
//   u  = [sm_gamma, 0, ..., 0, -cn_coeff]^T    sm_gamma = -cn_diag
//   v  = [1,    0, ..., 0,  cn_coeff/cn_diag]^T
//
// Solve A'*y = rhs  and  A'*z = u  simultaneously (shared upper_work).
// Final solution:  x = y - (v.y)/(1 + v.z) * z
//
// CN is linearly stable for all dt, but positivity additionally requires the explicit-side
// centre coefficient to remain nonnegative.  The solve is therefore subcycled below so that
// dt_sub*Dx/dx_len^2 <= POS_LIMIT.
//
// Minimum conservative momentum closure: reconstruct the time-centred CN mass flux at every
// face and let it carry the upwind cell's specific momentum.  This conserves each stored
// momentum component face by face, but is only the interim closure documented in the audit;
// it is not the complete Galilean-invariant Huang--Bai diffusion-momentum formulation.
//
// TODO(performance): replace the one-thread-per-ring Thomas/Sherman-Morrison solve with a
// one-block-per-ring cyclic Parallel Cyclic Reduction (PCR) solver if X diffusion becomes
// performance-relevant.  This is the preferred library-free CUDA-native design; benchmark it
// against batched cuFFT before removing the present reference implementation.
//
// Proposed PCR implementation:
//   1. Launch one CUDA block per (iy,iz) ring and one thread per azimuthal cell.  For N_X larger
//      than the device's maximum threads per block, process multiple cells per thread or use a
//      hybrid CR/PCR decomposition.
//   2. Load the cyclic CN system directly into shared memory: lower=-cn_coeff,
//      diagonal=1+2*cn_coeff, upper=-cn_coeff, and the CN ratio RHS.  Preserve the periodic
//      neighbours of cells 0 and N_X-1; a direct cyclic PCR solve removes the need for the
//      Sherman-Morrison correction vectors.
//   3. Apply log2(N_X) PCR elimination stages.  At stage s, each thread couples to neighbours
//      at offsets +/-2^s, using periodic modular indices.  Read the previous stage before
//      writing the next values and synchronize the block between stages.
//   4. Store the solved ratio only after all stages finish, then recover rho_d=q*rho_g.
//      Avoid an independent per-cell positivity clamp unless a conservative positivity
//      treatment is added.
//   5. Keep coefficients in registers when possible and use shared memory only for the arrays
//      that change between PCR stages.  Check shared-memory use and occupancy for double
//      precision; four N_X=1024 arrays require about 32 KiB per block.
//
// Required validation before switching solvers:
//   - compare PCR against this implementation for random positive rings over a wide range of
//     cn_coeff, including very small and very large values;
//   - verify periodicity, constant-state preservation, and ring-mass conservation to roundoff;
//   - test N_X values that are and are not powers of two, or explicitly restrict supported N_X;
//   - measure residual ||A*q_new-rhs|| and benchmark wall time against both this solver and
//     batched cuFFT on the target GPU.

__global__
void f_diffusion_x (real *dev_dustdens, real *dev_dustmomx, real *dev_dustmomy,
    real *dev_dustmomz, real dt)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_Y*N_Z) return;
    if (dt <= 0.0) return;

    int iy = idx % N_Y;
    int iz = idx / N_Y;

    real dx = _get_dx();
    real dy = _get_dy();
    real dz = _get_dz();

    real yc = Y_MIN*pow(dy, iy + 0.5);
    real zc = Z_MIN + (iz + 0.5)*dz;

    real Rc = yc*sin(zc);
    real Zc = yc*cos(zc);

    real dx_len = Rc*dx;

    real h_g  = _get_hg(Rc);
    real rhog = _get_rhog(Rc, Zc, h_g);
    real Dx   = _get_nu(Rc, h_g) / SC_X;

    // recover the current dust-to-gas ratio in this ring
    real ratio[N_X];
    for (int ix = 0; ix < N_X; ix++)
    {
        int ic = ix + iy*N_X + iz*N_X*N_Y;
        ratio[ix] = dev_dustdens[ic] / rhog;
    }

    // Crank–Nicolson is not unconditionally positivity-preserving for this uniform stencil
    // a sufficient condition is 1 - 2*cn_coeff >= 0 on the explicit side
    // use a margin below that bound for floating-point robustness and repeat equal CN substeps
    // this retains the second-order CN update while avoiding a non-conservative post-solve density clamp
    int n_sub = static_cast<int>(ceil(dt*Dx / (dx_len*dx_len) / POS_LIMIT));
    if (n_sub < 1) n_sub = 1;

    // Crank–Nicolson matrix coefficients (uniform within a ring and across all substeps)
    real dt_sub   = dt / static_cast<real>(n_sub);
    real cn_coeff = 0.5*dt_sub*Dx / (dx_len*dx_len);   // CN half-step coefficient
    real cn_diag  = 1.0 + 2.0*cn_coeff;                 // diagonal entry
    real cn_wrap  = -cn_coeff;                          // A[0,N-1] = A[N-1,0] = -c

    // Sherman–Morrison decomposition parameters
    real sm_gamma = -cn_diag;                           // standard choice
    real sm_vlast = cn_wrap / sm_gamma;                 // = cn_coeff / cn_diag  (v[N-1])
    real sm_diag0 = cn_diag - sm_gamma;                 // = 2*cn_diag
    real sm_diagN = cn_diag - cn_wrap*sm_vlast;         // = cn_diag + cn_coeff^2/cn_diag

    // CN ratio RHS, Sherman–Morrison correction RHS, and modified Thomas upper diagonal
    // these arrays are rebuilt in place for every substep, avoiding additional local storage
    real ratio_work[N_X], cycle_work[N_X], upper_work[N_X];
    for (int i_sub = 0; i_sub < n_sub; i_sub++)
    {
        for (int ix = 0; ix < N_X; ix++)
        {
            int ixm1 = (ix - 1 + N_X) % N_X;
            int ixp1 = (ix + 1)       % N_X;

            ratio_work[ix] = cn_coeff*ratio[ixm1] + (1.0 - 2.0*cn_coeff)*ratio[ix] + cn_coeff*ratio[ixp1];
            cycle_work[ix] = (ix == 0) ? sm_gamma : (ix == N_X - 1) ? cn_wrap : 0.0;
        }

        // simultaneous Thomas solves A'*y=rhs and A'*z=u
        real diag_cur  = sm_diag0;
        upper_work[0]  = -cn_coeff / diag_cur;
        ratio_work[0] /= diag_cur;
        cycle_work[0] /= diag_cur;

        for (int ix = 1; ix < N_X; ix++)
        {
            diag_cur = (ix < N_X - 1) ? cn_diag : sm_diagN;
            real pivot = diag_cur + cn_coeff*upper_work[ix - 1];
            upper_work[ix] = (ix < N_X - 1) ? (-cn_coeff / pivot) : 0.0;
            ratio_work[ix] = (ratio_work[ix] + cn_coeff*ratio_work[ix - 1]) / pivot;
            cycle_work[ix] = (cycle_work[ix] + cn_coeff*cycle_work[ix - 1]) / pivot;
        }

        for (int ix = N_X - 2; ix >= 0; ix--)
        {
            ratio_work[ix] -= upper_work[ix]*ratio_work[ix + 1];
            cycle_work[ix] -= upper_work[ix]*cycle_work[ix + 1];
        }

        // Sherman–Morrison correction
        // keep ratio as q_old and store q_new in ratio_work
        // until the matching mass and momentum fluxes have both been applied
        real base_proj  = ratio_work[0] + sm_vlast*ratio_work[N_X - 1];
        real corr_proj  = cycle_work[0] + sm_vlast*cycle_work[N_X - 1];
        real corr_scale = base_proj / (1.0 + corr_proj);

        for (int ix = 0; ix < N_X; ix++)
        {
            ratio_work[ix] -= corr_scale*cycle_work[ix];
        }

        // time-centred CN dust-mass flux at face ix+1/2, positive toward increasing ix
        // upper_work is no longer needed by the linear solve, so reuse it for the face flux
        for (int ix = 0; ix < N_X; ix++)
        {
            int ixp1 = (ix + 1) % N_X;
            upper_work[ix] = -0.5*Dx*rhog*((ratio[ixp1] - ratio[ix]) + (ratio_work[ixp1] - ratio_work[ix])) / dx_len;
        }

        // reuse cycle_work for each momentum face flux
        // all face fluxes are formed before updating a component,
        // so every upwind state belongs to the start of this substep
        for (int ix = 0; ix < N_X; ix++)
        {
            int ix_up = (upper_work[ix] >= 0.0) ? ix : (ix + 1) % N_X;
            real dens_up = rhog*ratio[ix_up];

            int ic_up = ix_up + iy*N_X + iz*N_X*N_Y;
            real velx_up = (dens_up >= RHO_VAC) ? dev_dustmomx[ic_up] / dens_up : sqrt(G*M_S*fmax(Rc, 0.0));

            cycle_work[ix] = upper_work[ix]*velx_up;
        }

        for (int ix = 0; ix < N_X; ix++)
        {
            int ixm1 = (ix - 1 + N_X) % N_X;
            int ic = ix + iy*N_X + iz*N_X*N_Y;

            dev_dustmomx[ic] -= dt_sub*(cycle_work[ix] - cycle_work[ixm1]) / dx_len;
        }

        for (int ix = 0; ix < N_X; ix++)
        {
            int ix_up = (upper_work[ix] >= 0.0) ? ix : (ix + 1) % N_X;
            real dens_up = rhog*ratio[ix_up];

            int ic_up = ix_up + iy*N_X + iz*N_X*N_Y;
            real vely_up = (dens_up >= RHO_VAC) ? dev_dustmomy[ic_up] / dens_up : 0.0;

            cycle_work[ix] = upper_work[ix]*vely_up;
        }

        for (int ix = 0; ix < N_X; ix++)
        {
            int ixm1 = (ix - 1 + N_X) % N_X;
            int ic = ix + iy*N_X + iz*N_X*N_Y;

            dev_dustmomy[ic] -= dt_sub*(cycle_work[ix] - cycle_work[ixm1]) / dx_len;
        }

        for (int ix = 0; ix < N_X; ix++)
        {
            int ix_up = (upper_work[ix] >= 0.0) ? ix : (ix + 1) % N_X;
            real dens_up = rhog*ratio[ix_up];

            int ic_up = ix_up + iy*N_X + iz*N_X*N_Y;
            real velz_up = (dens_up >= RHO_VAC) ? dev_dustmomz[ic_up] / dens_up : 0.0;

            cycle_work[ix] = upper_work[ix]*velz_up;
        }

        for (int ix = 0; ix < N_X; ix++)
        {
            int ixm1 = (ix - 1 + N_X) % N_X;
            int ic = ix + iy*N_X + iz*N_X*N_Y;

            dev_dustmomz[ic] -= dt_sub*(cycle_work[ix] - cycle_work[ixm1]) / dx_len;
            ratio[ix] = ratio_work[ix];
        }
    }

    for (int ix = 0; ix < N_X; ix++)
    {
        int ic = ix + iy*N_X + iz*N_X*N_Y;
        dev_dustdens[ic] = rhog*ratio[ix];
    }
}

// =========================================================================================================================

#endif // DIFFUSION
