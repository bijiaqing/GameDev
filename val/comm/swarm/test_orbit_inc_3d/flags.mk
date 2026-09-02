DUST_REPR := swarm
MODEL_INCLUDE_DIRS := $(VAL_SWARM_DIR)/test_common

# enable the full-3D contract with zero diffusivity while scheduling only drag-free production transport
GPU_FLAGS += -DTRANSPORT -DDIFFUSION -DCONST_NU
