ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
empty :=
space := $(empty) $(empty)

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
BIN_ROOT        = $(ROOT_DIR)/bin
QAV_ROOT        = $(ROOT_DIR)/qav
QAV_COMM_DIR    = $(QAV_ROOT)/comm
QAV_FLUID_DIR   = $(QAV_COMM_DIR)/fluid
QAV_SWARM_DIR   = $(QAV_COMM_DIR)/swarm
INC_COMM_DIR    = $(ROOT_DIR)/inc/comm
INC_BACKEND_DIR = $(ROOT_DIR)/inc/$(GPU_BACKEND)
SRC_COMM_DIR    = $(ROOT_DIR)/src/comm
SRC_BACKEND_DIR = $(ROOT_DIR)/src/$(GPU_BACKEND)

ifeq ($(GPU_BACKEND),cuda)
GPU_COMPILER ?= nvcc
GPU_TARGET ?= sm_80
GPU_BASE_FLAGS = -arch=$(GPU_TARGET) -O2 -std=c++17 --diag-suppress 177,550
GPU_DEVICE_FLAG = --device-c
GPU_LANGUAGE_FLAG =
GPU_LINK_FLAGS =
BACKEND_EXT = cu
BACKEND_DEFINE = -DGAMEDEV_CUDA

CUDA_MATH ?= fast
ifneq ($(words $(CUDA_MATH)),1)
$(error CUDA_MATH must be fast or precise)
endif
ifeq ($(filter fast precise,$(CUDA_MATH)),)
$(error CUDA_MATH must be fast or precise)
endif
ifeq ($(CUDA_MATH),fast)
GPU_BASE_FLAGS += --use_fast_math
endif
MATH_PATH = /$(CUDA_MATH)
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
MATH_PATH =
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
QAV_FLUID_MATCH := $(wildcard $(QAV_FLUID_DIR)/$(MODEL))
QAV_SWARM_MATCH := $(wildcard $(QAV_SWARM_DIR)/$(MODEL))
QAV_COMM_MATCH := $(QAV_FLUID_MATCH) $(QAV_SWARM_MATCH)
QAV_FLUID_OVERLAY := $(wildcard $(QAV_ROOT)/$(GPU_BACKEND)/fluid/$(MODEL))
QAV_SWARM_OVERLAY := $(wildcard $(QAV_ROOT)/$(GPU_BACKEND)/swarm/$(MODEL))
QAV_BACKEND_MATCH := $(QAV_FLUID_OVERLAY) $(QAV_SWARM_OVERLAY)

ifneq ($(words $(PROD_MODEL_MATCH) $(QAV_COMM_MATCH)),1)
ifneq ($(words $(PROD_MODEL_MATCH) $(QAV_COMM_MATCH)),0)
$(error MODEL=$(MODEL) is ambiguous: $(PROD_MODEL_MATCH) $(QAV_COMM_MATCH))
endif
endif
ifeq ($(strip $(PROD_MODEL_MATCH) $(QAV_COMM_MATCH) $(QAV_BACKEND_MATCH)),)
$(error MODEL=$(MODEL) was not found under mod/, qav/comm/fluid/, qav/comm/swarm/, or qav/$(GPU_BACKEND)/)
endif
ifneq ($(words $(QAV_BACKEND_MATCH)),0)
ifneq ($(words $(QAV_BACKEND_MATCH)),1)
$(error MODEL=$(MODEL) has ambiguous backend overlays: $(QAV_BACKEND_MATCH))
endif
endif

MODEL_DIR := $(firstword $(PROD_MODEL_MATCH) $(QAV_COMM_MATCH) $(QAV_BACKEND_MATCH))
MODEL_BACKEND_DIR := $(firstword $(QAV_BACKEND_MATCH))
MODEL_FLAG_FILE := $(firstword $(wildcard $(MODEL_DIR)/flags.mk) $(wildcard $(MODEL_BACKEND_DIR)/flags.mk))
ifeq ($(strip $(MODEL_FLAG_FILE)),)
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

IS_QAV := $(strip $(QAV_COMM_MATCH) $(QAV_BACKEND_MATCH))
ifneq ($(IS_QAV),)
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
ifneq ($(strip $(PARTICLES)),)
GPU_FLAGS += -DPERF_PARTICLES=$(PARTICLES)
endif
endif

MODEL_SOURCE_DIRS := $(strip $(MODEL_BACKEND_DIR) $(MODEL_DIR))
MODEL_HEADER_DIRS := $(MODEL_SOURCE_DIRS)
ifneq ($(IS_QAV),)
MODEL_HEADER_DIRS += $(QAV_ROOT)/$(GPU_BACKEND)/$(DUST_REPR)/test_common
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
EXEC = $(BIN_ROOT)/$(MODEL)/$(GPU_BACKEND)/gamedev

ifeq ($(IS_QAV),)
OUT_DIR = $(OUT_ROOT)/$(MODEL)/$(GPU_BACKEND)
else
OUT_TAG_DIR = $(if $(strip $(OUT_TAG)),/$(OUT_TAG))
ifeq ($(DUST_REPR),fluid)
QAV_SWEEP ?= $(FLUID_SWEEP)
OUT_DIR = $(QAV_ROOT)/logs/fluid/$(GPU_BACKEND)/$(QAV_SWEEP)/$(MODEL)$(OUT_TAG_DIR)
else
OUT_DIR = $(QAV_ROOT)/logs/swarm/$(GPU_BACKEND)/$(MODEL)$(OUT_TAG_DIR)
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
    col_event_run.o \
    col_rate_calc.o \
    col_snap_save.o \
    col_site_init.o \
    diffusion_pos.o \
    dyn_rate_calc.o \
    dustdens_calc.o \
    dustdens_depo.o \
    dustdens_init.o \
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
RELINK_TRIGGER :=
INC_BRANCH_DIR = $(INC_COMM_DIR)/$(DUST_REPR)
INC_BACKEND_BRANCH_DIR = $(INC_BACKEND_DIR)/$(DUST_REPR)
SRC_BRANCH_DIR = $(SRC_COMM_DIR)/$(DUST_REPR)
SRC_BACKEND_BRANCH_DIR = $(SRC_BACKEND_DIR)/$(DUST_REPR)

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
OBJ_DIR = $(OBJ_ROOT)/$(MODEL)/fluid/$(GPU_BACKEND)/$(FLUID_SWEEP)$(MATH_PATH)/$(GPU_TARGET)
_OBJ = $(_OBJ_FLUID)
else
ifneq ($(filter -DCOLLISION,$(GPU_FLAGS)),)
RELINK_TRIGGER := FORCE
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
OBJ_DIR = $(OBJ_ROOT)/$(MODEL)/swarm/$(GPU_BACKEND)/$(COLLISION_SEARCH)$(MATH_PATH)/$(GPU_TARGET)
else
OBJ_DIR = $(OBJ_ROOT)/$(MODEL)/swarm/$(GPU_BACKEND)$(MATH_PATH)/$(GPU_TARGET)
endif
_OBJ = $(_OBJ_SWARM)
endif

_OBJ += $(_OBJ_MOD)
OBJ = $(foreach file,$(strip $(_OBJ)),$(OBJ_DIR)/$(strip $(file)))
SOURCE_SEARCH_DIRS = $(MODEL_SOURCE_DIRS) $(SRC_BACKEND_BRANCH_DIR) $(SRC_BRANCH_DIR)
INC_SEARCH_FLAGS = $(MODEL_INCLUDE_FLAGS) -I $(INC_BACKEND_BRANCH_DIR) -I $(INC_BRANCH_DIR)

INC_HEADER_PATHS := $(shell find $(INC_BACKEND_BRANCH_DIR) $(INC_BRANCH_DIR) -type f \
    \( -name '*.cuh' -o -name '*.h' -o -name '*.hpp' \) 2>/dev/null)
INC_HEADER_NAMES := $(sort $(notdir $(INC_HEADER_PATHS)))
HEADER_OVERRIDE_PATHS := $(sort $(foreach header,$(INC_HEADER_NAMES),\
    $(firstword $(foreach directory,$(MODEL_HEADER_DIRS),$(wildcard $(directory)/$(header))))))

vpath %.hip $(subst $(space),:,$(SOURCE_SEARCH_DIRS))
vpath %.cu $(subst $(space),:,$(SOURCE_SEARCH_DIRS))

$(info Using GPU backend: $(GPU_BACKEND))
$(info Using GPU target: $(GPU_TARGET))
$(info Using model setup: $(MODEL) from $(MODEL_DIR))
ifneq ($(strip $(MODEL_BACKEND_DIR)),)
$(info Using backend overlay: $(MODEL_BACKEND_DIR))
endif
ifeq ($(strip $(MODEL_CONST)),)
$(info Using representation constants: $(INC_BACKEND_BRANCH_DIR)/const_defs.cuh)
endif
$(foreach header,$(HEADER_OVERRIDE_PATHS),$(info Using header override: $(header)))
$(info Using dust representation: $(DUST_REPR))
ifeq ($(GPU_BACKEND),cuda)
$(info Using CUDA math: $(CUDA_MATH))
endif
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

$(OBJ_DIR)/%.o: %.hip $(MODEL_FLAG_FILE) $(MODEL_CONST)
	@mkdir -p $(dir $@)
	@printf "%-12s %50s -> %s\n" "Compiling" "$(patsubst $(ROOT_DIR)/%,%,$<)" "$(notdir $@)"
	@$(GPU_COMPILER) $(GPU_BASE_FLAGS) $(GPU_FLAGS) $(GPU_LANGUAGE_FLAG) $(GPU_DEVICE_FLAG) -o $@ $< \
		$(INC_SEARCH_FLAGS) $(BACKEND_DEFINE) \
		-DPATH_OUT=\"$(abspath $(OUT_DIR))/\" \
		-MMD -MP -MF $(patsubst %.o,%.d,$@)

$(OBJ_DIR)/%.o: %.cu $(MODEL_FLAG_FILE) $(MODEL_CONST)
	@mkdir -p $(dir $@)
	@printf "%-12s %50s -> %s\n" "Compiling" "$(patsubst $(ROOT_DIR)/%,%,$<)" "$(notdir $@)"
	@$(GPU_COMPILER) $(GPU_BASE_FLAGS) $(GPU_FLAGS) $(GPU_LANGUAGE_FLAG) $(GPU_DEVICE_FLAG) -o $@ $< \
		$(INC_SEARCH_FLAGS) $(BACKEND_DEFINE) \
		-DPATH_OUT=\"$(abspath $(OUT_DIR))/\" \
		-MMD -MP -MF $(patsubst %.o,%.d,$@)

clean:
ifdef MODEL
	@printf "%-12s %s\n" "Cleaning" "$(EXEC)"
	@rm -f $(EXEC)
	@printf "%-12s %s\n" "Cleaning" "$(OBJ_DIR)"
	@rm -rf $(OBJ_DIR)
else
	@printf "%-12s %s\n" "Cleaning" "all object and executable files"
	@rm -rf $(OBJ_ROOT)/* $(BIN_ROOT)/*
endif

-include $(OBJ:.o=.d)
