DUST_REPR := swarm
MODEL_INCLUDE_DIRS := $(VAL_ROOT)/swarm/src

# exercise the three-dimensional periodic-image velocity in the physical custom kernel
GPU_FLAGS += -DCOLLISION -DBERNOULLI -DKNN_CACHE -DMULTISIZE -DCODE_UNIT
