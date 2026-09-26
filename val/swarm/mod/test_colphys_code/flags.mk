DUST_REPR := swarm

# exercise the code-unit Reynolds closure and the physical custom kernel without Brownian motion
GPU_FLAGS += -DCOLLISION
GPU_FLAGS += -DMULTISIZE
GPU_FLAGS += -DCODE_UNIT
