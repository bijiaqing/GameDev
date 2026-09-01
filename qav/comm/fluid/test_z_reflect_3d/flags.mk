DUST_REPR := fluid

# activate the production midplane wall; diffusion is required structurally for every resolved-polar model
GPU_FLAGS += -DDIFFUSION -DCONST_NU -DHALF_DISK
