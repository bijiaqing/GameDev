DUST_REPR := swarm
MODEL_INCLUDE_DIRS := $(QAV_SWARM_DIR)/test_common

# retain the production diffusion kernel while supplying a local linear settling map in the verification driver
GPU_FLAGS += -DTRANSPORT -DDIFFUSION -DCONST_NU
