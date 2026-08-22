DUST_REPR := fluid
MODEL_INCLUDE_DIRS := $(QAV_FLUID_DIR)/test_common

# the production configuration requires diffusion support whenever the polar dimension is active
GPU_FLAGS += -DDIFFUSION -DCONST_NU

