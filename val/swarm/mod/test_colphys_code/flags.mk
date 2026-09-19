DUST_REPR := swarm
MODEL_INCLUDE_DIRS := $(VAL_ROOT)/swarm/src

# exercise the code-unit Reynolds closure and the physical custom kernel without Brownian motion
GPU_FLAGS += -DCOLLISION -DBERNOULLI -DKNN_CACHE -DMULTISIZE -DCODE_UNIT
