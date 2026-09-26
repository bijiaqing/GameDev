DUST_REPR := fluid

# isolate azimuthal density diffusion with a spatially constant coefficient
GPU_FLAGS += -DDIFFUSION
GPU_FLAGS += -DCONST_NU
