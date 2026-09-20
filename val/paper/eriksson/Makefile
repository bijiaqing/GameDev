# All numerical kernels and backend handling come from the root build.
PAPER_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
GPU_BACKEND ?= cuda
GPU_TARGET ?= $(if $(filter rocm,$(GPU_BACKEND)),gfx942,sm_80)
COLLISION_SEARCH ?= kdtree
override MOD_ROOT := $(PAPER_ROOT)/mod
override OUT_ROOT := $(PAPER_ROOT)/out
override OUT_DIR = $(OUT_ROOT)/$(MODEL)/$(GPU_BACKEND)/$(COLLISION_SEARCH)
override OBJ_ROOT := $(PAPER_ROOT)/obj
override EXEC = $(OBJ_ROOT)/$(MODEL)/$(GPU_BACKEND)/$(COLLISION_SEARCH)/gamedev
override MODEL_EXECUTABLES = $(wildcard $(OBJ_ROOT)/*/*/*/gamedev)
include $(PAPER_ROOT)/../../../Makefile
