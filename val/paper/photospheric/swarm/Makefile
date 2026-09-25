# Reuse the root build, treating src/ as this formulation's model overrides.
PAPER_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
override MODEL := src
override MOD_ROOT := $(PAPER_ROOT)
override OBJ_ROOT := $(PAPER_ROOT)/obj
override OBJ_DIR = $(OBJ_ROOT)/$(GPU_BACKEND)/$(GPU_TARGET)
override EXEC = $(OBJ_DIR)/gamedev
override OUT_ROOT := $(PAPER_ROOT)/out
override OUT_DIR = $(OUT_ROOT)/$(GPU_BACKEND)
override MODEL_EXECUTABLES = $(wildcard $(OBJ_ROOT)/*/*/gamedev)
include $(PAPER_ROOT)/../../../../Makefile
