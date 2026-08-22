DUST_REPR := swarm
MODEL_INCLUDE_DIRS := $(QAV_SWARM_DIR)/test_common

# build the production transport kernel around the model-local drag-free second substep
GPU_FLAGS += -DTRANSPORT
