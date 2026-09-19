DUST_REPR := swarm

# exercise the three-dimensional periodic-image velocity in the physical custom kernel
GPU_FLAGS += -DCOLLISION -DBERNOULLI -DKNN_CACHE -DMULTISIZE -DCODE_UNIT
