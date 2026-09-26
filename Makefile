ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

.DEFAULT_GOAL := all

GPU_BACKEND ?= cuda
ifneq ($(words $(GPU_BACKEND)),1)
$(error GPU_BACKEND must be cuda or rocm)
endif
ifeq ($(filter cuda rocm,$(GPU_BACKEND)),)
$(error GPU_BACKEND must be cuda or rocm)
endif

MOD_ROOT        = $(ROOT_DIR)/mod
OBJ_ROOT        = $(ROOT_DIR)/obj
OUT_ROOT        = $(ROOT_DIR)/out
VAL_ROOT        = $(ROOT_DIR)/val
VAL_FLUID_DIR   = $(VAL_ROOT)/fluid/mod
VAL_SWARM_DIR   = $(VAL_ROOT)/swarm/mod
INC_DIR    = $(ROOT_DIR)/inc
SRC_DIR    = $(ROOT_DIR)/src
MODEL_EXEC_DIRS = $(dir $(wildcard $(MOD_ROOT)/*/flags.mk))
MODEL_EXECUTABLES = $(addsuffix gamedev,$(MODEL_EXEC_DIRS))

ifeq ($(GPU_BACKEND),cuda)
GPU_COMPILER ?= nvcc
GPU_TARGET ?= sm_80
GPU_BASE_FLAGS = -arch=$(GPU_TARGET) -O2 -std=c++17 --diag-suppress 177,550
GPU_DEVICE_FLAG = --device-c
GPU_LANGUAGE_FLAG =
GPU_LINK_FLAGS =
BACKEND_EXT = cu
BACKEND_DEFINE = -DGAMEDEV_CUDA


else
GPU_COMPILER ?= hipcc
GPU_TARGET ?= gfx942
GPU_BASE_FLAGS = --offload-arch=$(GPU_TARGET) -O2 -std=c++17 -fgpu-rdc \
    -include hip/hip_runtime.h -Wno-unused-command-line-argument
GPU_DEVICE_FLAG = -c
GPU_LANGUAGE_FLAG = -x hip
GPU_LINK_FLAGS = -fgpu-rdc
BACKEND_EXT = hip
BACKEND_DEFINE = -DGAMEDEV_ROCM
ifneq ($(strip $(RESOURCE_REPORT)),)
GPU_BASE_FLAGS += -Rpass-analysis=kernel-resource-usage
endif
endif

ifneq ($(MAKECMDGOALS),clean)
ifndef MODEL
$(error MODEL is not defined. Usage: make MODEL=model_name GPU_BACKEND=cuda|rocm)
endif
endif

ifdef MODEL
PROD_MODEL_MATCH := $(wildcard $(MOD_ROOT)/$(MODEL))
VAL_FLUID_MATCH := $(wildcard $(VAL_FLUID_DIR)/$(MODEL))
VAL_SWARM_MATCH := $(wildcard $(VAL_SWARM_DIR)/$(MODEL))
VAL_MODEL_MATCH := $(VAL_FLUID_MATCH) $(VAL_SWARM_MATCH)

ifneq ($(words $(PROD_MODEL_MATCH) $(VAL_MODEL_MATCH)),1)
ifneq ($(words $(PROD_MODEL_MATCH) $(VAL_MODEL_MATCH)),0)
$(error MODEL=$(MODEL) is ambiguous: $(PROD_MODEL_MATCH) $(VAL_MODEL_MATCH))
endif
endif
ifeq ($(strip $(PROD_MODEL_MATCH) $(VAL_MODEL_MATCH)),)
$(error MODEL=$(MODEL) was not found under mod/, val/fluid/mod/, or val/swarm/mod/)
endif

MODEL_DIR := $(firstword $(PROD_MODEL_MATCH) $(VAL_MODEL_MATCH))
MODEL_FLAG_FILE := $(MODEL_DIR)/flags.mk
ifeq ($(wildcard $(MODEL_FLAG_FILE)),)
$(error Model flag file for MODEL=$(MODEL) does not exist)
endif

include $(MODEL_FLAG_FILE)

ifeq ($(GPU_BACKEND),cuda)
GPU_FLAGS += $(CUDA_FLAGS)
else
GPU_FLAGS += $(ROCM_FLAGS)
endif

ifndef DUST_REPR
$(error DUST_REPR is not defined in $(MODEL_FLAG_FILE))
endif
ifneq ($(words $(DUST_REPR)),1)
$(error DUST_REPR must be fluid or swarm)
endif
ifeq ($(filter fluid swarm,$(DUST_REPR)),)
$(error DUST_REPR must be fluid or swarm)
endif

IS_VAL := $(strip $(VAL_MODEL_MATCH))
ifneq ($(IS_VAL),)
ifneq ($(strip $(RES)),)
GPU_FLAGS += -DTEST_RES=$(RES)
endif
ifneq ($(strip $(CFL)),)
GPU_FLAGS += -DTEST_CFL=$(CFL)
endif
ifneq ($(strip $(POWER)),)
GPU_FLAGS += -DTEST_POWER=$(POWER)
endif
ifneq ($(strip $(SHIFT)),)
GPU_FLAGS += -DTEST_SHIFT=$(SHIFT)
endif
ifneq ($(strip $(SAVE)),)
GPU_FLAGS += -DTEST_SAVE_MAX=$(SAVE)
endif
ifneq ($(strip $(OUT_TIME)),)
GPU_FLAGS += -DTEST_DT_OUT=$(OUT_TIME)
endif
ifneq ($(strip $(CHAIN_CAP)),)
GPU_FLAGS += -DTEST_CHAIN_CAP=$(CHAIN_CAP)
endif
ifneq ($(strip $(PARTICLES)),)
GPU_FLAGS += -DPERF_PARTICLES=$(PARTICLES)
endif
ifneq ($(strip $(DIRECTION)),)
ifeq ($(filter x y z,$(DIRECTION)),)
$(error DIRECTION must be x, y, or z)
endif
GPU_FLAGS += $(if $(filter x,$(DIRECTION)),-DTEST_DIRECTION_X,$(if $(filter y,$(DIRECTION)),-DTEST_DIRECTION_Y,-DTEST_DIRECTION_Z))
endif
endif

VAL_REPR_SRC_DIR = $(VAL_ROOT)/$(DUST_REPR)/src

MODEL_SOURCE_DIRS := $(MODEL_DIR)
MODEL_HEADER_DIRS := $(MODEL_SOURCE_DIRS)
ifneq ($(IS_VAL),)
# allow validation models to replace complete files without adding test branches to production files
MODEL_SOURCE_DIRS += $(VAL_REPR_SRC_DIR)
MODEL_HEADER_DIRS += $(VAL_REPR_SRC_DIR)
endif
MODEL_HEADER_DIRS += $(MODEL_INCLUDE_DIRS)
MODEL_INCLUDE_FLAGS := $(addprefix -I ,$(MODEL_HEADER_DIRS))

ifneq ($(strip $(MODEL_PARENT)),)
MODEL_PARENT_DIR := $(MOD_ROOT)/$(MODEL_PARENT)
ifeq ($(wildcard $(MODEL_PARENT_DIR)),)
$(error MODEL_PARENT=$(MODEL_PARENT) was not found under mod/)
endif
MODEL_SOURCE_DIRS += $(MODEL_PARENT_DIR)
MODEL_HEADER_DIRS += $(MODEL_PARENT_DIR)
MODEL_INCLUDE_FLAGS += -I $(MODEL_PARENT_DIR)
endif

MODEL_CONST := $(firstword $(foreach dir,$(MODEL_HEADER_DIRS),$(wildcard $(dir)/const_defs.cuh)))

ifeq ($(IS_VAL),)
EXEC = $(dir $(MODEL_FLAG_FILE))gamedev
else
# keep generated validation build products out of the source and production trees
OBJ_ROOT = $(VAL_ROOT)/$(DUST_REPR)/obj
EXEC = $(OBJ_ROOT)/$(MODEL)/$(GPU_BACKEND)/gamedev
endif

ifeq ($(IS_VAL),)
OUT_DIR = $(OUT_ROOT)/$(MODEL)
else
OUT_TAG_DIR = $(if $(strip $(OUT_TAG)),/$(OUT_TAG))
VAL_SCOPE ?= all
VAL_SCOPE_DIR = $(if $(filter all,$(VAL_SCOPE)),,/groups/$(VAL_SCOPE))
ifeq ($(DUST_REPR),fluid)
VAL_SWEEP ?= $(FLUID_SWEEP)$(if $(filter cuda,$(GPU_BACKEND)),_precise)
OUT_DIR = $(VAL_ROOT)/fluid/out/$(MODEL)/$(GPU_BACKEND)/$(VAL_SWEEP)$(VAL_SCOPE_DIR)$(OUT_TAG_DIR)
else
OUT_DIR = $(VAL_ROOT)/swarm/out/$(MODEL)/$(GPU_BACKEND)$(VAL_SCOPE_DIR)$(OUT_TAG_DIR)
endif
endif
endif

_OBJ_FLUID_THREAD = \
    advection_xth.o \
    advection_yth.o \
    advection_zth.o \
    diffusion_xth.o \
    diffusion_yth.o \
    diffusion_zth.o

_OBJ_FLUID_BLOCK = \
    advection_xbl.o \
    advection_ybl.o \
    advection_zbl.o \
    diffusion_xbl.o \
    diffusion_ybl.o \
    diffusion_zbl.o

_OBJ_FLUID = \
    cfl_rate_calc.o \
    fluid_runtime.o \
    inf_cell_flag.o \
    init_rho_calc.o \
    init_vel_calc.o \
    momentum_getv.o \
    momentum_setv.o \
    optdepth_calc.o \
    optdepth_csum.o \
    source_update.o

_OBJ_SWARM = \
    col_audit_bin.o \
    col_bath_init.o \
    col_bath_rate.o \
    col_chain_run.o \
    col_comp_zero.o \
    col_count_bin.o \
    col_dep_graph.o \
    col_env_cache.o \
    col_event_sum.o \
    col_rate_bins.o \
    col_site_init.o \
    col_size_bnds.o \
    col_size_scan.o \
    col_size_zero.o \
    col_skip_scan.o \
    col_space_bin.o \
    diffusion_pos.o \
    dustdens_calc.o \
    dustdens_depo.o \
    dustdens_init.o \
    dyn_rate_calc.o \
    gas_lerp_calc.o \
    optdepth_calc.o \
    optdepth_csum.o \
    optdepth_depo.o \
    optdepth_init.o \
    optdepth_mean.o \
    particle_init.o \
    rngstate_init.o \
    ssa_substep_1.o \
    ssa_substep_2.o \
    ssa_transport.o \
    swarm_runtime.o

ifdef MODEL
RELINK_TRIGGER := FORCE
INC_BRANCH_DIR = $(INC_DIR)/$(DUST_REPR)
SRC_BRANCH_DIR = $(SRC_DIR)/$(DUST_REPR)

ifeq ($(DUST_REPR),fluid)
FLUID_SWEEP ?= $(if $(filter rocm,$(GPU_BACKEND)),block,thread)
ifneq ($(words $(FLUID_SWEEP)),1)
$(error FLUID_SWEEP must be thread or block)
endif
ifeq ($(filter thread block,$(FLUID_SWEEP)),)
$(error FLUID_SWEEP must be thread or block)
endif
GPU_FLAGS += -DDUST_FLUID
ifeq ($(FLUID_SWEEP),block)
GPU_FLAGS += -DFLUID_BLOCK_SWEEP
_OBJ_FLUID += $(_OBJ_FLUID_BLOCK)
else
_OBJ_FLUID += $(_OBJ_FLUID_THREAD)
endif
OBJ_DIR = $(OBJ_ROOT)/$(MODEL)/fluid/$(GPU_BACKEND)/$(FLUID_SWEEP)/$(GPU_TARGET)
_OBJ = $(_OBJ_FLUID)
else
ifneq ($(filter -DCOLLISION,$(GPU_FLAGS)),)
COLLISION_SEARCH ?= $(if $(filter rocm,$(GPU_BACKEND)),morton,kdtree)
ifneq ($(words $(COLLISION_SEARCH)),1)
$(error COLLISION_SEARCH must be kdtree or morton)
endif
ifeq ($(filter kdtree morton,$(COLLISION_SEARCH)),)
$(error COLLISION_SEARCH must be kdtree or morton)
endif
ifeq ($(COLLISION_SEARCH),morton)
GPU_FLAGS += -DCOLLISION_MORTON
else
GPU_FLAGS += -DCOLLISION_KDTREE
endif
OBJ_DIR = $(OBJ_ROOT)/$(MODEL)/swarm/$(GPU_BACKEND)/$(COLLISION_SEARCH)/$(GPU_TARGET)
else
OBJ_DIR = $(OBJ_ROOT)/$(MODEL)/swarm/$(GPU_BACKEND)/$(GPU_TARGET)
endif
_OBJ = $(_OBJ_SWARM)
endif

ifneq ($(IS_VAL),)
OBJ_DIR = $(OBJ_ROOT)/$(MODEL)/$(GPU_BACKEND)$(if $(filter fluid,$(DUST_REPR)),/$(FLUID_SWEEP),$(if $(strip $(COLLISION_SEARCH)),/$(COLLISION_SEARCH)))/$(GPU_TARGET)
endif

_OBJ += $(_OBJ_MOD)
OBJ = $(foreach file,$(strip $(_OBJ)),$(OBJ_DIR)/$(strip $(file)))
SOURCE_SEARCH_DIRS = $(MODEL_SOURCE_DIRS) $(SRC_BRANCH_DIR)
INC_SEARCH_FLAGS = -I $(INC_DIR) $(MODEL_INCLUDE_FLAGS) -I $(INC_BRANCH_DIR)
BUILD_CONFIG = $(OBJ_DIR)/.build_config

INC_HEADER_PATHS := $(INC_DIR)/gpu.cuh $(shell find $(INC_BRANCH_DIR) -type f \
    \( -name '*.cuh' -o -name '*.h' -o -name '*.hpp' \) 2>/dev/null)
INC_HEADER_NAMES := $(sort $(notdir $(INC_HEADER_PATHS)))
HEADER_OVERRIDE_PATHS := $(sort $(foreach header,$(INC_HEADER_NAMES),\
    $(firstword $(foreach directory,$(MODEL_HEADER_DIRS),$(wildcard $(directory)/$(header))))))

# choose the highest-priority directory before applying the backend's extension preference
define resolve_source
$(firstword $(foreach directory,$(SOURCE_SEARCH_DIRS),\
    $(if $(filter cuda,$(GPU_BACKEND)),\
        $(wildcard $(directory)/$(1).cu),\
        $(firstword $(wildcard $(directory)/$(1).hip) $(wildcard $(directory)/$(1).cu))\
    )\
))
endef

OBJ_NAMES := $(basename $(notdir $(OBJ)))
$(foreach name,$(OBJ_NAMES),$(eval SOURCE_$(name) := $(call resolve_source,$(name))))
MISSING_SOURCES := $(foreach name,$(OBJ_NAMES),$(if $(SOURCE_$(name)),,$(name)))
ifneq ($(strip $(MISSING_SOURCES)),)
$(error No source was found for objects: $(MISSING_SOURCES))
endif
RESOLVED_SOURCE_PATHS := $(foreach name,$(OBJ_NAMES),$(SOURCE_$(name)))

# import dependencies only when they belong to the source selected for this object
define source_identity_matches
$(and $(wildcard $(patsubst %.o,%.d,$(1))),\
    $(wildcard $(patsubst %.o,%.source,$(1))),\
    $(filter $(SOURCE_$(2)),$(strip $(shell sed -n '1p' $(patsubst %.o,%.source,$(1))))))
endef

VALID_DEP_FILES := $(foreach object,$(OBJ),\
    $(if $(call source_identity_matches,$(object),$(basename $(notdir $(object)))),\
        $(patsubst %.o,%.d,$(object))))
SOURCE_MISMATCH_OBJ := $(foreach object,$(OBJ),\
    $(if $(call source_identity_matches,$(object),$(basename $(notdir $(object)))),,$(object)))

$(info Using GPU backend: $(GPU_BACKEND))
$(info Using GPU target: $(GPU_TARGET))
$(info Using model setup: $(MODEL) from $(MODEL_DIR))
ifeq ($(strip $(MODEL_CONST)),)
$(info Using representation constants: $(INC_BRANCH_DIR)/const_defs.cuh)
endif
$(foreach header,$(HEADER_OVERRIDE_PATHS),$(info Using header override: $(header)))
$(info Using dust representation: $(DUST_REPR))
ifeq ($(DUST_REPR),fluid)
$(info Using fluid sweep: $(FLUID_SWEEP))
else ifneq ($(filter -DCOLLISION,$(GPU_FLAGS)),)
$(info Using collision search: $(COLLISION_SEARCH))
endif
endif

.PHONY: all clean FORCE

all: $(EXEC)

$(OUT_DIR):
	@mkdir -p $@

$(EXEC): $(OBJ) $(MODEL_FLAG_FILE) $(MODEL_CONST) $(RELINK_TRIGGER) | $(OUT_DIR)
	@mkdir -p $(dir $@)
	@printf "%-12s %-20s %s\n" "Linking" "$@" "from $(words $(OBJ)) $(GPU_BACKEND) objects"
	@$(GPU_COMPILER) $(GPU_BASE_FLAGS) $(GPU_FLAGS) $(GPU_LINK_FLAGS) -o $@ $(OBJ)

FORCE:

# preserve an unchanged timestamp for identical builds and invalidate every object when any
# effective compiler, include-search, source-selection, or output-path input changes
$(BUILD_CONFIG): FORCE
	@mkdir -p $(dir $@)
	@{ \
		printf '%s\n' \
			'GPU_COMPILER=$(GPU_COMPILER)' \
			'GPU_BASE_FLAGS=$(GPU_BASE_FLAGS)' \
			'GPU_FLAGS=$(GPU_FLAGS)' \
			'GPU_LANGUAGE_FLAG=$(GPU_LANGUAGE_FLAG)' \
			'GPU_DEVICE_FLAG=$(GPU_DEVICE_FLAG)' \
			'GPU_LINK_FLAGS=$(GPU_LINK_FLAGS)' \
			'BACKEND_DEFINE=$(BACKEND_DEFINE)' \
			'INC_SEARCH_FLAGS=$(INC_SEARCH_FLAGS)' \
			'SOURCE_SEARCH_DIRS=$(SOURCE_SEARCH_DIRS)' \
			'RESOLVED_SOURCE_PATHS=$(RESOLVED_SOURCE_PATHS)' \
			'HEADER_OVERRIDE_PATHS=$(HEADER_OVERRIDE_PATHS)' \
			'PATH_OUT=$(abspath $(OUT_DIR))'; \
	} > "$@.tmp"
	@if ! cmp -s "$@.tmp" "$@"; then \
		mv "$@.tmp" "$@"; \
	else \
		rm -f "$@.tmp"; \
	fi
define object_rule
$(1): $(SOURCE_$(2)) $(MODEL_FLAG_FILE) $(MODEL_CONST) $(BUILD_CONFIG)
	@mkdir -p $$(dir $$@)
	@printf "%-12s %60s -> %s\n" "Compiling" "$$(patsubst $(ROOT_DIR)/%,%,$$<)" "$$(notdir $$@)"
	@$$(GPU_COMPILER) $$(GPU_BASE_FLAGS) $$(GPU_FLAGS) $$(GPU_LANGUAGE_FLAG) $$(GPU_DEVICE_FLAG) -o $$@ $$< \
		$$(INC_SEARCH_FLAGS) $$(BACKEND_DEFINE) \
		-DPATH_OUT=\"$$(abspath $$(OUT_DIR))/\" \
		-MMD -MP -MF $$(patsubst %.o,%.d,$$@)
	@printf '%s\n' '$(SOURCE_$(2))' > $$(patsubst %.o,%.source,$$@).tmp
	@mv $$(patsubst %.o,%.source,$$@).tmp $$(patsubst %.o,%.source,$$@)
endef

$(foreach object,$(OBJ),\
    $(eval $(call object_rule,$(object),$(basename $(notdir $(object))))))

# rebuild an object instead of importing dependencies recorded for a different primary source
$(SOURCE_MISMATCH_OBJ): FORCE

clean:
ifdef MODEL
	@printf "%-12s %s\n" "Cleaning" "$(EXEC)"
	@rm -f $(EXEC)
	@printf "%-12s %s\n" "Cleaning" "$(OBJ_DIR)"
	@rm -rf $(OBJ_DIR)
else
	@printf "%-12s %s\n" "Cleaning" "all object and model executable files"
	@rm -rf $(OBJ_ROOT)/*
	@rm -rf $(VAL_ROOT)/fluid/obj $(VAL_ROOT)/swarm/obj $(VAL_ROOT)/paper/*/obj $(VAL_ROOT)/paper/coagulation/*/obj
	@rm -f $(MODEL_EXECUTABLES)
endif

-include $(VALID_DEP_FILES)
