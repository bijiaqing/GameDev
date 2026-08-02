#include <fluid_kern.cuh>

// isolate the cancellation-safe exact drag-force update over eight stiffness decades
__global__
void source_update (real *dev_dustvelx, real *dev_dustvely, real *dev_dustvelz,
    const real *dev_dustdens, real dt)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= N_G) return;
    if (dev_dustdens[idx] < RHO_VAC) return;

    const real stiffness[8] = {1.0e-6, 1.0e-3, 0.1, 1.0, 10.0, 1.0e2, 1.0e4, 1.0e6};
    real drag_h = stiffness[idx % 8];
    real ts = dt/drag_h;
    real drag_relax = -expm1(-drag_h);
    real drag_decay = 1.0 - drag_relax;

    real force_weight_n, force_weight_new;
    if (drag_h < 1.0e-4)
    {
        real drag_h2 = drag_h*drag_h;
        real drag_h3 = drag_h2*drag_h;
        force_weight_n   = dt*(0.5 - drag_h/3.0 + drag_h2/8.0  - drag_h3/30.0);
        force_weight_new = dt*(0.5 - drag_h/6.0 + drag_h2/24.0 - drag_h3/120.0);
    }
    else
    {
        force_weight_new = ts*(drag_h - drag_relax)/drag_h;
        force_weight_n = ts*drag_relax - force_weight_new;
    }

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
