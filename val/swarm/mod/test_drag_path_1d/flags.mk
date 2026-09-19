DUST_REPR := swarm
MODEL_INCLUDE_DIRS := $(VAL_ROOT)/swarm/src

# MULTISIZE lets three particles carry distinct controlled stopping-time labels in par_size
GPU_FLAGS += -DTRANSPORT -DMULTISIZE
