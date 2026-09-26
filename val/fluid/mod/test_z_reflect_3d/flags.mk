DUST_REPR := fluid

# activate the production midplane wall; diffusion is required structurally for every resolved-polar model
GPU_FLAGS += -DDIFFUSION
GPU_FLAGS += -DCONST_NU
GPU_FLAGS += -DHALF_DISK
