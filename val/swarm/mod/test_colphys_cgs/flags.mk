DUST_REPR := swarm
MODEL_INCLUDE_DIRS := $(VAL_ROOT)/swarm/src

# omit CODE_UNIT deliberately so the molecular Reynolds closure and Brownian velocity are compiled
GPU_FLAGS += -DCOLLISION -DBERNOULLI -DKNN_CACHE -DMULTISIZE
