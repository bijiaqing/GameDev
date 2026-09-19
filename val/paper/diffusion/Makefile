# Reuse production kernels; src/ contains only the test-specific overrides.
PAPER_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
override MOD_ROOT := $(PAPER_ROOT)/mod
override OUT_ROOT := $(PAPER_ROOT)/out
override OUT_DIR = $(OUT_ROOT)/$(MODEL)/$(GPU_BACKEND)
override OBJ_ROOT := $(PAPER_ROOT)/obj
override EXEC = $(OBJ_ROOT)/$(MODEL)/$(GPU_BACKEND)/gamedev
override MODEL_EXECUTABLES = $(wildcard $(OBJ_ROOT)/*/*/gamedev)
include $(PAPER_ROOT)/../../../Makefile
