#include <fluid_kern.cuh>

// test-local replacement for the production source kernel
//
// replace disk-dependent gas velocity, stopping time, and force that prevent an independent closed-form comparison
// while calling the production exponential drag quadrature with fixed endpoint forces and eight prescribed
// stiffnesses so every coefficient can be checked analytically
__global__
void source_update (real *dev_dustvelx, real *dev_dustvely, real *dev_dustvelz,
    const real *dev_dustdens, real dt)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_G) return;
    if (dev_dustdens[idx] < RHO_VAC) return;

    // use x as a parameter index rather than a spatial coordinate to cover non-stiff through strongly stiff drag in one
    // launch
    const real stiffness[8] = {1.0e-6, 1.0e-3, 0.1, 1.0, 10.0, 1.0e2, 1.0e4, 1.0e6};
    real drag_h = stiffness[idx % 8];
    real ts = dt / drag_h;
    real drag_relax, drag_decay, force_weight_n, force_weight_new;
    _get_drag_weights(dt, ts, drag_relax, drag_decay, force_weight_n, force_weight_new);

    // prescribe gas velocity and linearly varying force endpoints instead of evaluating disk-dependent helpers
    const real gas_x = 0.4,  gas_y = -0.2, gas_z = 0.1;
    const real fn_x  = -0.3, fn_y  = 0.7,  fn_z  = -0.5;
    const real f1_x  = 0.2,  f1_y  = -0.1, f1_z  = 0.9;

    dev_dustvelx[idx] = drag_decay*dev_dustvelx[idx] + drag_relax*gas_x
                       + force_weight_n*fn_x + force_weight_new*f1_x;
    dev_dustvely[idx] = drag_decay*dev_dustvely[idx] + drag_relax*gas_y
                       + force_weight_n*fn_y + force_weight_new*f1_y;
    dev_dustvelz[idx] = drag_decay*dev_dustvelz[idx] + drag_relax*gas_z
                       + force_weight_n*fn_z + force_weight_new*f1_z;
}
