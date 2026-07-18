# =============================================================================
# Standalone Makefile for mod/rpi_fluid
# Run `make` from within this directory.
# =============================================================================

# Project directory (absolute path containing this Makefile).
ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

NVCC  = nvcc
NVCC += -arch=sm_80
NVCC += -O2 --use_fast_math
NVCC += -std=c++17
NVCC += --diag-suppress 177,550

EXEC    = $(ROOT_DIR)/graffiti
INC_DIR = $(ROOT_DIR)/inc
MOD_DIR = $(ROOT_DIR)/mod
SRC_DIR = $(ROOT_DIR)/src
OBJ_DIR = $(ROOT_DIR)/obj
OUT_DIR = $(ROOT_DIR)/out

# Prefer a model-specific flag file in mod/, while retaining the existing top-level flags.mk
# as the default when no override is present. Dust transport is always compiled.
FLAGS_FILE := $(firstword $(wildcard $(MOD_DIR)/flags.mk) $(wildcard $(ROOT_DIR)/flags.mk))
-include $(FLAGS_FILE)

# Headers placed in mod/ override same-named headers in inc/.
NVCC += -I$(MOD_DIR)
NVCC += -I$(INC_DIR)
NVCC += -DPATH_OUT=\"$(abspath $(OUT_DIR))/\"

_OBJ =              \
    cfl_rate_calc.o \
    f_advection_x.o \
    f_advection_y.o \
    f_advection_z.o \
    f_diffusion_x.o \
    f_diffusion_y.o \
    f_diffusion_z.o \
    f_fields_init.o \
    f_moment_recv.o \
    f_source_term.o \
    graffiti_main.o \
    optdepth_csum.o \
    optdepth_calc.o

# A selected flags.mk may append model-only translation units by defining _OBJ_MOD.
_OBJ += $(_OBJ_MOD)

OBJ = $(foreach f,$(_OBJ),$(OBJ_DIR)/$(f))

# Source overrides in mod/ take precedence over the default implementations in src/.
vpath %.cu $(MOD_DIR):$(SRC_DIR)

.PHONY: all clean

all: $(EXEC)

$(EXEC): $(OBJ) $(FLAGS_FILE)
	@mkdir -p $(OUT_DIR)
	@printf "%-12s %s\n" "Linking" "$@"
	@$(NVCC) -o $@ $(OBJ)

$(OBJ_DIR)/%.o: %.cu $(FLAGS_FILE)
	@mkdir -p $(OBJ_DIR)
	@printf "%-12s %40s -> %s\n" "Compiling" "$(patsubst $(ROOT_DIR)/%,%,$<)" "$(notdir $@)"
	@$(NVCC) --device-c -o $@ $< -MMD -MP -MF $(patsubst %.o,%.d,$@)

clean:
	@rm -f $(EXEC)
	@rm -rf $(OBJ_DIR)

-include $(OBJ:.o=.d)
