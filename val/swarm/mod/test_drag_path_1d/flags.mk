DUST_REPR := swarm

GPU_FLAGS += -DTRANSPORT
# MULTISIZE lets three particles carry distinct controlled stopping-time labels in par_size
GPU_FLAGS += -DMULTISIZE
