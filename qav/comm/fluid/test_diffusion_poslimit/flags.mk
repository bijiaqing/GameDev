DUST_REPR := fluid
MODEL_INCLUDE_DIRS := $(QAV_FLUID_DIR)/test_common

# constant nu makes the cyclic CN amplification factor independent of disk thermodynamics
GPU_FLAGS += -DDIFFUSION -DCONST_NU
