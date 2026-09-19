DUST_REPR := fluid
MODEL_INCLUDE_DIRS := $(VAL_ROOT)/fluid/src

# enable only the production source and optical-depth path isolated by this test
GPU_FLAGS += -DRADIATION
