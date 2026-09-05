GROUP_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT_DIR  := $(abspath $(GROUP_DIR)/../../..)
TEMP_DIR  := $(ROOT_DIR)/val/temp/coag/test_product
MODEL_DIR := $(GROUP_DIR)/models
OUTPUT_ROOT ?= $(ROOT_DIR)/out/test_product/seed_000

GPU_BACKEND ?= cuda
GPU_TARGET  ?= sm_80
CUDA_MATH   ?= fast

ifneq ($(GPU_BACKEND),cuda)
$(error coag/test_product currently supports only GPU_BACKEND=cuda)
endif

ifndef MODEL
$(error MODEL is not defined. Usage: make -C val/coag/test_product MODEL=case_name)
endif

EXEC := $(TEMP_DIR)/bin/$(MODEL)/gamedev

ROOT_MAKE_ARGS = \
	MODEL=$(MODEL) \
	GPU_BACKEND=$(GPU_BACKEND) \
	GPU_TARGET=$(GPU_TARGET) \
	CUDA_MATH=$(CUDA_MATH) \
	MOD_ROOT=$(MODEL_DIR) \
	OUT_ROOT=$(OUTPUT_ROOT) \
	OBJ_ROOT=$(TEMP_DIR)/obj \
	EXEC=$(EXEC)

.PHONY: all run clean

all:
	@$(MAKE) -C $(ROOT_DIR) $(ROOT_MAKE_ARGS)

run: all
	@$(EXEC)

clean:
	@$(MAKE) -C $(ROOT_DIR) $(ROOT_MAKE_ARGS) clean
