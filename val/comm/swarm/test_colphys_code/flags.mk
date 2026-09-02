DUST_REPR := swarm
MODEL_INCLUDE_DIRS := $(VAL_SWARM_DIR)/test_common

# exercise the code-unit Reynolds closure and the physical custom kernel without Brownian motion
GPU_FLAGS += -DCOLLISION -DMULTISIZE -DCODE_UNIT
