DUST_REPR := swarm

# exercise the code-unit Reynolds closure and the physical custom kernel without Brownian motion
GPU_FLAGS += -DCOLLISION -DBERNOULLI -DKNN_CACHE -DMULTISIZE -DCODE_UNIT
