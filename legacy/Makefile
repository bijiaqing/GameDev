# Project root directory (absolute path where Makefile is located)
ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

INC_DIR = $(ROOT_DIR)/inc
MOD_DIR = $(ROOT_DIR)/mod
OBJ_DIR = $(ROOT_DIR)/obj
OUT_DIR = $(ROOT_DIR)/out
SRC_DIR = $(ROOT_DIR)/src

# Allow clean targets to work without MODEL
ifneq ($(MAKECMDGOALS),clean)
ifndef MODEL
$(error MODEL is not defined. Usage: make MODEL=model_name)
endif
endif

ifdef MODEL
MOD_DIR := $(MOD_DIR)/$(MODEL)
OUT_DIR := $(OUT_DIR)/$(MODEL)
OBJ_DIR := $(OBJ_DIR)/$(MODEL)
endif

# NVCC base configuration (platform-specific flags)
NVCC  = nvcc
NVCC += -arch=sm_80
NVCC += -O2 --use_fast_math
NVCC += --diag-suppress 177,550

ifdef MODEL
ifeq ($(wildcard $(MOD_DIR)),)
$(error Model directory $(MOD_DIR) does not exist. Please create it first.)
endif

-include $(MOD_DIR)/flags.mk

$(info Using model setup: $(MODEL) from $(MOD_DIR))

EXEC = $(MOD_DIR)/graffiti
endif

_INC =						\
	const.cuh				\
	graffiti_host.cuh		\
	graffiti_kern.cuh		\
	helpers_collision.cuh	\
	helpers_diffusion.cuh	\
	helpers_interpval.cuh	\
	helpers_paramgrid.cuh	\
	helpers_paramphys.cuh	\
	helpers_scatfield.cuh	\
	helpers_transport.cuh

_OBJ =				\
	col_flag_calc.o	\
	col_proc_exec.o	\
	col_rate_calc.o	\
	col_rate_init.o	\
	col_tree_init.o	\
	diffusion_pos.o	\
	diffusion_vel.o	\
	dustdens_calc.o	\
	dustdens_init.o	\
	dustdens_scat.o	\
	graffiti_main.o	\
	optdepth_calc.o	\
	optdepth_csum.o	\
	optdepth_init.o	\
	optdepth_mean.o	\
	optdepth_scat.o	\
	particle_init.o	\
	rs_grids_init.o	\
	rs_swarm_init.o	\
	ssa_substep_1.o	\
	ssa_substep_2.o	\
	ssa_transport.o

# Append model-specific files if defined in flags.mk
_INC += $(_INC_MOD)
_OBJ += $(_OBJ_MOD)

# Build INC and OBJ lists with full paths
INC = $(foreach f,$(strip $(_INC)),$(INC_DIR)/$(strip $(f)))
OBJ = $(foreach f,$(strip $(_OBJ)),$(OBJ_DIR)/$(strip $(f)))

# Set search path for .cu files (model directory first, then src)
ifdef MODEL
vpath %.cu $(MOD_DIR):$(SRC_DIR)
else
vpath %.cu $(SRC_DIR)
endif

.PHONY: all clean

all: $(EXEC)

$(EXEC): $(OBJ) $(MOD_DIR)/flags.mk
	@mkdir -p $(OUT_DIR)
	@printf "%-12s %-20s %s\n" "Linking" "$@" "from $(words $(OBJ)) objects"
	@$(NVCC) -o $@ $(OBJ)

# Pattern rule for compiling .cu files
$(OBJ_DIR)/%.o: %.cu $(MOD_DIR)/flags.mk
	@mkdir -p $(dir $@)
	@printf "%-12s %40s -> %s\n" "Compiling" "$(patsubst $(ROOT_DIR)/%,%,$<)" "$(notdir $@)"
	@$(NVCC) --device-c -o $@ $< \
		$(if $(wildcard $(MOD_DIR)),-I $(MOD_DIR),) \
		-I $(INC_DIR) \
		-DPATH_OUT=\"$(abspath $(OUT_DIR))/\" \
		-MMD -MP -MF $(patsubst %.o,%.d,$@)

clean:
ifdef MODEL
	@printf "%-12s %s\n" "Cleaning" "$(EXEC)"
	@rm -f $(EXEC)
	@printf "%-12s %s\n" "Cleaning" "object files"
	@rm -rf $(OBJ_DIR)
else
	@printf "%-12s %s\n" "Cleaning" "all object files"
	@rm -rf $(OBJ_DIR)/*
endif

# Include dependency files at the very end
-include $(OBJ:.o=.d)