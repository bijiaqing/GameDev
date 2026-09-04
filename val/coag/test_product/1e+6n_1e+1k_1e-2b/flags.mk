MODEL_PARENT := test_common
include $(dir $(lastword $(MAKEFILE_LIST)))../test_common/common_flags.mk

GPU_FLAGS += -DSWEEP_N_P=1000000
GPU_FLAGS += -DSWEEP_N_K=10
GPU_FLAGS += -DSWEEP_COL_BATH_EPS=1.0e-2
