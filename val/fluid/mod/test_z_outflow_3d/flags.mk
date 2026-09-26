DUST_REPR := fluid

# satisfy the production 3D guard while the test advances only polar advection
GPU_FLAGS += -DDIFFUSION
GPU_FLAGS += -DCONST_NU
