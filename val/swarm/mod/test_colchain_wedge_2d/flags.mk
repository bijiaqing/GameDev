DUST_REPR := swarm

GPU_FLAGS += -DCOLLISION -DMULTISIZE -DCODE_UNIT

# Audit the current production local collision controller during this short test.
GPU_FLAGS += -DCOL_DIAGNOSTICS
