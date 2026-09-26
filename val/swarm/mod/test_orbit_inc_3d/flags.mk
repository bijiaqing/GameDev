DUST_REPR := swarm

# enable the full-3D contract with zero diffusivity while scheduling only drag-free production transport
GPU_FLAGS += -DTRANSPORT
GPU_FLAGS += -DDIFFUSION
GPU_FLAGS += -DCONST_NU
