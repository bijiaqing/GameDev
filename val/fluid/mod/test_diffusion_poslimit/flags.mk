DUST_REPR := fluid

# constant nu makes the cyclic CN amplification factor independent of disk thermodynamics
GPU_FLAGS += -DDIFFUSION -DCONST_NU
