# Reuse production build rules with this test set's model, output, and build roots.
DIFFUSION_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
override MOD_ROOT := $(DIFFUSION_ROOT)/models
override OUT_ROOT := $(DIFFUSION_ROOT)/outputs
DIFFUSION_TEMP := $(abspath $(DIFFUSION_ROOT)/../../temp/swarm_diffusion)
override OBJ_ROOT := $(DIFFUSION_TEMP)/obj
override EXEC = $(DIFFUSION_TEMP)/bin/$(MODEL)/gamedev
override MODEL_EXECUTABLES = $(wildcard $(DIFFUSION_TEMP)/bin/*/gamedev)
include $(DIFFUSION_ROOT)/../../../Makefile
