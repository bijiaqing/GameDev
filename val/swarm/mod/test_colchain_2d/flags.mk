DUST_REPR := swarm

GPU_FLAGS += -DCOLLISION
GPU_FLAGS += -DMULTISIZE
GPU_FLAGS += -DCODE_UNIT
# audit the production local collision controller during this short test
GPU_FLAGS += -DCOL_DIAGNOSTICS
