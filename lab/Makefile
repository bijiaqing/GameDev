ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

NVCC ?= nvcc
ARCH ?= sm_80
K ?= 200
PTXAS ?= 0

TARGET := $(ROOT_DIR)/bin/knn_benchmark
SOURCE := $(ROOT_DIR)/knn/knn_benchmark.cu
EDGE_TARGET := $(ROOT_DIR)/bin/knn_edge_tests
EDGE_SOURCE := $(ROOT_DIR)/knn/knn_edge_tests.cu
PERIODIC_TARGET := $(ROOT_DIR)/bin/knn_periodic_tests
PERIODIC_SOURCE := $(ROOT_DIR)/knn/knn_periodic_tests.cu
WEDGE_TARGET := $(ROOT_DIR)/bin/knn_wedge_benchmark
WEDGE_SOURCE := $(ROOT_DIR)/knn/knn_wedge_benchmark.cu

SWARM_MODEL ?= collision_disk_2d
COLLISION_SEARCH ?= kdtree
SWARM_MODEL_DIR := $(ROOT_DIR)/mod/$(SWARM_MODEL)
SWARM_SOURCE_DIR := $(ROOT_DIR)/src/swarm
SWARM_INCLUDE_DIR := $(ROOT_DIR)/inc/swarm
SWARM_OBJ_DIR := $(ROOT_DIR)/obj/$(SWARM_MODEL)/$(COLLISION_SEARCH)
SWARM_TARGET := $(ROOT_DIR)/bin/gamedev_$(SWARM_MODEL)_$(COLLISION_SEARCH)
SWARM_OUT_DIR := $(ROOT_DIR)/out/swarm/$(SWARM_MODEL)/$(COLLISION_SEARCH)

ifeq ($(wildcard $(SWARM_MODEL_DIR)/flags.mk),)
$(error laboratory swarm model $(SWARM_MODEL) does not exist)
endif

include $(SWARM_MODEL_DIR)/flags.mk

ifeq ($(COLLISION_SEARCH),kdtree)
SWARM_BACKEND_FLAG := -DCOLLISION_KDTREE
else ifeq ($(COLLISION_SEARCH),morton)
SWARM_BACKEND_FLAG := -DCOLLISION_MORTON
else
$(error COLLISION_SEARCH must be kdtree or morton)
endif

SWARM_NVCCFLAGS := -arch=$(ARCH) -O2 --use_fast_math -std=c++17
SWARM_NVCCFLAGS += --diag-suppress 177,550
SWARM_NVCCFLAGS += $(SWARM_FLAGS) $(SWARM_BACKEND_FLAG)
ifeq ($(PTXAS),1)
SWARM_NVCCFLAGS += -Xptxas=-v
endif
SWARM_INCLUDES := -I $(SWARM_MODEL_DIR) -I $(SWARM_INCLUDE_DIR) -I $(ROOT_DIR)/knn
SWARM_SOURCES := $(wildcard $(SWARM_SOURCE_DIR)/*.cu)
SWARM_OBJECTS := $(patsubst $(SWARM_SOURCE_DIR)/%.cu,$(SWARM_OBJ_DIR)/%.o,$(SWARM_SOURCES))

NVCCFLAGS := -arch=$(ARCH) -O3 --use_fast_math -std=c++17
NVCCFLAGS += -DLAB_KNN_K=$(K)
INCLUDES := -I $(ROOT_DIR)/knn -I $(ROOT_DIR)/inc/swarm

.PHONY: all edge periodic wedge swarm swarm-clean clean

all: $(TARGET)

edge: $(EDGE_TARGET)

periodic: $(PERIODIC_TARGET)

wedge: $(WEDGE_TARGET)

swarm: $(SWARM_TARGET)

$(TARGET): $(SOURCE) $(ROOT_DIR)/knn/knn_types.cuh $(ROOT_DIR)/knn/morton_types.cuh \
	$(ROOT_DIR)/knn/adaptive_morton.cuh
	@mkdir -p $(dir $@)
	@printf "%-12s %s\n" "Compiling" "$@"
	@$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $<

$(EDGE_TARGET): $(EDGE_SOURCE) $(ROOT_DIR)/knn/knn_types.cuh $(ROOT_DIR)/knn/morton_types.cuh \
	$(ROOT_DIR)/knn/adaptive_morton.cuh
	@mkdir -p $(dir $@)
	@printf "%-12s %s\n" "Compiling" "$@"
	@$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $<

$(PERIODIC_TARGET): $(PERIODIC_SOURCE) $(ROOT_DIR)/knn/knn_types.cuh \
	$(ROOT_DIR)/knn/morton_types.cuh $(ROOT_DIR)/knn/adaptive_morton.cuh \
	$(ROOT_DIR)/knn/periodic_query.cuh
	@mkdir -p $(dir $@)
	@printf "%-12s %s\n" "Compiling" "$@"
	@$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $<

$(WEDGE_TARGET): $(WEDGE_SOURCE) $(ROOT_DIR)/knn/knn_types.cuh \
	$(ROOT_DIR)/knn/morton_types.cuh $(ROOT_DIR)/knn/adaptive_morton.cuh \
	$(ROOT_DIR)/knn/periodic_query.cuh
	@mkdir -p $(dir $@)
	@printf "%-12s %s\n" "Compiling" "$@"
	@$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $<

$(SWARM_TARGET): $(SWARM_OBJECTS) $(SWARM_MODEL_DIR)/flags.mk $(SWARM_MODEL_DIR)/const_defs.cuh
	@mkdir -p $(dir $@) $(SWARM_OUT_DIR)
	@printf "%-12s %s\n" "Linking" "$@"
	@$(NVCC) $(SWARM_NVCCFLAGS) -o $@ $(SWARM_OBJECTS)

$(SWARM_OBJ_DIR)/%.o: $(SWARM_SOURCE_DIR)/%.cu $(SWARM_MODEL_DIR)/flags.mk $(SWARM_MODEL_DIR)/const_defs.cuh
	@mkdir -p $(dir $@) $(SWARM_OUT_DIR)
	@printf "%-12s %s -> %s\n" "Compiling" "$<" "$@"
	@$(NVCC) $(SWARM_NVCCFLAGS) --device-c -o $@ $< \
		$(SWARM_INCLUDES) \
		-DPATH_OUT=\"$(abspath $(SWARM_OUT_DIR))/\" \
		-MMD -MP -MF $(patsubst %.o,%.d,$@)

swarm-clean:
	@printf "%-12s %s\n" "Cleaning" "$(SWARM_OBJ_DIR)"
	@rm -rf $(SWARM_OBJ_DIR)
	@rm -f $(SWARM_TARGET)

clean:
	@printf "%-12s %s\n" "Cleaning" "$(ROOT_DIR)/bin"
	@rm -rf $(ROOT_DIR)/bin

-include $(SWARM_OBJECTS:.o=.d)
