ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

INC_SHARE_DIR = $(ROOT_DIR)/inc/share
INC_FLUID_DIR = $(ROOT_DIR)/inc/fluid
INC_SWARM_DIR = $(ROOT_DIR)/inc/swarm
MOD_ROOT       = $(ROOT_DIR)/mod
OBJ_ROOT       = $(ROOT_DIR)/obj
OUT_ROOT       = $(ROOT_DIR)/out
TST_ROOT       = $(ROOT_DIR)/tst
TST_FLUID_DIR  = $(TST_ROOT)/fluid
TST_SWARM_DIR  = $(TST_ROOT)/swarm
SRC_FLUID_DIR = $(ROOT_DIR)/src/fluid
SRC_SWARM_DIR = $(ROOT_DIR)/src/swarm

NVCC  = nvcc
NVCC += -arch=sm_80
NVCC += -O2 --use_fast_math
NVCC += -std=c++17
NVCC += --diag-suppress 177,550

ifneq ($(MAKECMDGOALS),clean)
ifndef MODEL
$(error MODEL is not defined. Usage: make MODEL=model_name)
endif
endif

ifdef MODEL
MODEL_MATCHES := $(wildcard \
    $(MOD_ROOT)/$(MODEL) \
    $(TST_FLUID_DIR)/$(MODEL) \
    $(TST_SWARM_DIR)/$(MODEL))

ifeq ($(strip $(MODEL_MATCHES)),)
$(error MODEL=$(MODEL) was not found under mod/, tst/fluid/, or tst/swarm/)
endif

ifneq ($(words $(MODEL_MATCHES)),1)
$(error MODEL=$(MODEL) is ambiguous: $(MODEL_MATCHES))
endif

MODEL_DIR := $(firstword $(MODEL_MATCHES))

ifeq ($(wildcard $(MODEL_DIR)/flags.mk),)
$(error Model flag file $(MODEL_DIR)/flags.mk does not exist. Please create it first.)
endif

include $(MODEL_DIR)/flags.mk

MODEL_CONST := $(wildcard $(MODEL_DIR)/const_defs.cuh)

ifneq ($(filter $(TST_ROOT)/%,$(MODEL_DIR)),)
ifneq ($(strip $(RES)),)
NVCC += -DTEST_RES=$(RES)
endif
ifneq ($(strip $(CFL)),)
NVCC += -DTEST_CFL=$(CFL)
endif
ifneq ($(strip $(POWER)),)
NVCC += -DTEST_POWER=$(POWER)
endif
ifneq ($(strip $(SHIFT)),)
NVCC += -DTEST_SHIFT=$(SHIFT)
endif
ifneq ($(strip $(SAVE)),)
NVCC += -DTEST_SAVE_MAX=$(SAVE)
endif
ifneq ($(strip $(OUT_TIME)),)
NVCC += -DTEST_DT_OUT=$(OUT_TIME)
endif
endif

ifneq ($(strip $(MODEL_PARENT)),)
MODEL_PARENT_DIR := $(MOD_ROOT)/$(MODEL_PARENT)
ifeq ($(wildcard $(MODEL_PARENT_DIR)),)
$(error MODEL_PARENT=$(MODEL_PARENT) was not found under mod/)
endif
MODEL_SOURCE_DIRS := $(MODEL_DIR):$(MODEL_PARENT_DIR)
MODEL_INCLUDE_FLAGS := -I $(MODEL_DIR) -I $(MODEL_PARENT_DIR) $(addprefix -I ,$(MODEL_INCLUDE_DIRS))
else
MODEL_SOURCE_DIRS := $(MODEL_DIR)
MODEL_INCLUDE_FLAGS := -I $(MODEL_DIR) $(addprefix -I ,$(MODEL_INCLUDE_DIRS))
endif

ifndef DUST_REPR
$(error DUST_REPR is not defined in $(MODEL_DIR)/flags.mk)
endif

ifneq ($(words $(DUST_REPR)),1)
$(error DUST_REPR must be fluid or swarm)
endif

ifeq ($(filter fluid swarm,$(DUST_REPR)),)
$(error DUST_REPR must be fluid or swarm)
endif

OBJ_DIR = $(OBJ_ROOT)/$(MODEL)/$(DUST_REPR)

ifneq ($(filter $(TST_ROOT)/%,$(MODEL_DIR)),)
OUT_TAG_DIR = $(if $(strip $(OUT_TAG)),/$(OUT_TAG))
ifeq ($(DUST_REPR),fluid)
OUT_DIR = $(TST_ROOT)/$(DUST_REPR)/out/$(FLUID_SWEEP)/$(MODEL)$(OUT_TAG_DIR)
else
OUT_DIR = $(TST_ROOT)/$(DUST_REPR)/out/$(MODEL)$(OUT_TAG_DIR)
endif
else
OUT_DIR = $(OUT_ROOT)/$(MODEL)
endif

EXEC = $(MODEL_DIR)/gamedev
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

_OBJ_FLUID =        \
    cfl_rate_calc.o  \
    inf_cell_flag.o \
    fluid_runtime.o    \
    momentum_getv.o \
    momentum_setv.o \
    init_rho_calc.o \
    optdepth_calc.o \
    optdepth_csum.o \
    source_update.o \
    init_vel_calc.o

_OBJ_SWARM =       \
    col_event_run.o  \
    col_rate_calc.o  \
    col_snap_save.o  \
    col_tree_init.o  \
    diffusion_pos.o  \
    dyn_rate_calc.o  \
    dustdens_calc.o  \
    dustdens_depo.o  \
    dustdens_init.o  \
    gas_lerp_calc.o  \
    optdepth_calc.o  \
    optdepth_csum.o  \
    optdepth_depo.o  \
    optdepth_init.o  \
    optdepth_mean.o  \
    particle_init.o  \
    rngstate_init.o  \
    ssa_substep_1.o  \
    ssa_substep_2.o  \
    ssa_transport.o  \
    swarm_runtime.o

ifdef MODEL
ifeq ($(DUST_REPR),fluid)
FLUID_SWEEP ?= thread
ifneq ($(words $(FLUID_SWEEP)),1)
$(error FLUID_SWEEP must be thread or block)
endif
ifeq ($(filter thread block,$(FLUID_SWEEP)),)
$(error FLUID_SWEEP must be thread or block)
endif

NVCC += -DDUST_FLUID
ifeq ($(FLUID_SWEEP),block)
NVCC += -DFLUID_BLOCK_SWEEP
_OBJ_FLUID += $(_OBJ_FLUID_BLOCK)
else
_OBJ_FLUID += $(_OBJ_FLUID_THREAD)
endif

OBJ_DIR = $(OBJ_ROOT)/$(MODEL)/$(DUST_REPR)/$(FLUID_SWEEP)
SRC_BRANCH_DIR = $(SRC_FLUID_DIR)
INC_BRANCH_DIR = $(INC_FLUID_DIR)
_OBJ = $(_OBJ_FLUID)
SRC_SEARCH_DIRS = $(MODEL_SOURCE_DIRS):$(SRC_BRANCH_DIR)
INC_SEARCH_FLAGS = $(MODEL_INCLUDE_FLAGS) -I $(INC_BRANCH_DIR) -I $(INC_SHARE_DIR)
else ifeq ($(DUST_REPR),swarm)
SRC_BRANCH_DIR = $(SRC_SWARM_DIR)
INC_BRANCH_DIR = $(INC_SWARM_DIR)
_OBJ = $(_OBJ_SWARM)
SRC_SEARCH_DIRS = $(MODEL_SOURCE_DIRS):$(SRC_BRANCH_DIR)
INC_SEARCH_FLAGS = $(MODEL_INCLUDE_FLAGS) -I $(INC_BRANCH_DIR)
endif

_OBJ += $(_OBJ_MOD)
OBJ = $(foreach file,$(strip $(_OBJ)),$(OBJ_DIR)/$(strip $(file)))

vpath %.cu $(SRC_SEARCH_DIRS)

$(info Using model setup: $(MODEL) from $(MODEL_DIR))
ifneq ($(strip $(MODEL_CONST)),)
$(info Using model constants: $(MODEL_DIR)/const_defs.cuh)
else
$(info Using representation constants: $(INC_BRANCH_DIR)/const_defs.cuh)
endif
$(info Using dust representation: $(DUST_REPR))
ifeq ($(DUST_REPR),fluid)
$(info Using fluid sweep: $(FLUID_SWEEP))
endif
endif

.PHONY: all clean

all: $(EXEC)

$(EXEC): $(OBJ) $(MODEL_DIR)/flags.mk $(MODEL_CONST)
	@mkdir -p $(OUT_DIR)
	@printf "%-12s %-20s %s\n" "Linking" "$@" "from $(words $(OBJ)) objects"
	@$(NVCC) -o $@ $(OBJ)

$(OBJ_DIR)/%.o: %.cu $(MODEL_DIR)/flags.mk $(MODEL_CONST)
	@mkdir -p $(dir $@)
	@printf "%-12s %40s -> %s\n" "Compiling" "$(patsubst $(ROOT_DIR)/%,%,$<)" "$(notdir $@)"
	@$(NVCC) --device-c -o $@ $< \
		$(INC_SEARCH_FLAGS) \
		-DPATH_OUT=\"$(abspath $(OUT_DIR))/\" \
		-MMD -MP -MF $(patsubst %.o,%.d,$@)

clean:
ifdef MODEL
	@printf "%-12s %s\n" "Cleaning" "$(EXEC)"
	@rm -f $(EXEC)
	@printf "%-12s %s\n" "Cleaning" "$(OBJ_DIR)"
	@rm -rf $(OBJ_DIR)
else
	@printf "%-12s %s\n" "Cleaning" "all object files"
	@rm -rf $(OBJ_ROOT)/*
endif

-include $(OBJ:.o=.d)
