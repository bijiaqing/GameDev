DUST_REPR := swarm
MODEL_INCLUDE_DIRS := $(VAL_SWARM_DIR)/test_common

# MULTISIZE lets three particles carry distinct controlled stopping-time labels in par_size
GPU_FLAGS += -DTRANSPORT -DMULTISIZE
