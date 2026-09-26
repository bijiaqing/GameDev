DUST_REPR := fluid

GPU_FLAGS += -DDIFFUSION
# constant nu makes the cyclic CN amplification factor independent of disk thermodynamics
GPU_FLAGS += -DCONST_NU
