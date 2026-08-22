DUST_REPR := swarm
MODEL_INCLUDE_DIRS := $(QAV_SWARM_DIR)/test_common

# retain the production split radiation transport around the model-local drag-free force update
GPU_FLAGS += -DTRANSPORT -DRADIATION -DMULTISIZE
