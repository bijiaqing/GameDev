PAPER_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
GPU_BACKEND := cuda
GPU_TARGET ?= sm_80
SEED ?= 0
ifneq ($(GPU_BACKEND),cuda)
$(error This campaign uses CUDA only)
endif
ifneq ($(filter-out kdtree,$(COLLISION_SEARCH)),)
$(error This campaign uses KD-tree only)
endif
override MOD_ROOT := $(PAPER_ROOT)/mod
override OUT_ROOT := $(PAPER_ROOT)/out
override OUT_DIR = $(if $(filter 0,$(SEED)),$(OUT_ROOT)/$(MODEL)/seed0,$(OBJ_ROOT)/$(MODEL)/out)
override OBJ_ROOT := $(PAPER_ROOT)/obj/seed$(SEED)
override EXEC = $(OBJ_ROOT)/$(MODEL)/gamedev
override MODEL_EXECUTABLES = $(wildcard $(PAPER_ROOT)/obj/seed*/*/gamedev)
include $(PAPER_ROOT)/../../../../Makefile
