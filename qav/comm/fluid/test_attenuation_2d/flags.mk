DUST_REPR := fluid
MODEL_INCLUDE_DIRS := $(QAV_FLUID_DIR)/test_common

# enable only the production source and optical-depth path isolated by this test
GPU_FLAGS += -DRADIATION
