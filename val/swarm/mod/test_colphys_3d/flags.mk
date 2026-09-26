DUST_REPR := swarm

# exercise the three-dimensional periodic-image velocity in the physical custom kernel
GPU_FLAGS += -DCOLLISION
GPU_FLAGS += -DMULTISIZE
GPU_FLAGS += -DCODE_UNIT
