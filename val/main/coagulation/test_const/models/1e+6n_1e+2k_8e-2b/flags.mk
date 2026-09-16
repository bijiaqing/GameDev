MODEL_PARENT := models_common
include $(dir $(lastword $(MAKEFILE_LIST)))../models_common/common_flags.mk

GPU_FLAGS += -DSWEEP_N_P=1000000
GPU_FLAGS += -DSWEEP_N_K=100
GPU_FLAGS += -DSWEEP_COL_BATH_EPS=8.0e-2
