# Production build rules; model outputs and temporary products have separate roots.
COAG_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
GPU_BACKEND ?= cuda
GPU_TARGET ?= $(if $(filter rocm,$(GPU_BACKEND)),gfx942,sm_80)
COLLISION_SEARCH ?= morton
override MOD_ROOT := $(COAG_ROOT)/models
override OUT_ROOT := $(COAG_ROOT)/outputs
override OUT_DIR = $(OUT_ROOT)/$(MODEL)/$(GPU_BACKEND)/$(COLLISION_SEARCH)
COAG_TEMP := $(COAG_ROOT)/obj
override OBJ_ROOT := $(COAG_TEMP)/obj
override EXEC = $(COAG_TEMP)/bin/$(MODEL)/$(GPU_BACKEND)/$(COLLISION_SEARCH)/gamedev
override MODEL_EXECUTABLES = $(wildcard $(COAG_TEMP)/bin/*/*/*/gamedev)
include $(COAG_ROOT)/../../../Makefile
