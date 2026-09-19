DUST_REPR := swarm

# enable the full-3D contract with zero diffusivity while scheduling only drag-free production transport
GPU_FLAGS += -DTRANSPORT -DDIFFUSION -DCONST_NU
