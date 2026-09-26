DUST_REPR := swarm

# omit CODE_UNIT deliberately so the molecular Reynolds closure and Brownian velocity are compiled
GPU_FLAGS += -DCOLLISION
GPU_FLAGS += -DMULTISIZE
