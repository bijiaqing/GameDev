DUST_REPR := swarm
MODEL_INCLUDE_DIRS := $(QAV_SWARM_DIR)/test_common

# omit CODE_UNIT deliberately so the molecular Reynolds closure and Brownian velocity are compiled
GPU_FLAGS += -DCOLLISION -DMULTISIZE
