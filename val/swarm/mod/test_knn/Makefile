TEST_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT_DIR := $(abspath $(TEST_DIR)/../../../..)

GPU_BACKEND ?= cuda
K ?= 200
ifeq ($(GPU_BACKEND),cuda)
GPU_COMPILER ?= nvcc
GPU_TARGET ?= sm_80
GPU_FLAGS := -arch=$(GPU_TARGET) -O3 -lineinfo -std=c++17 -DVAL_KNN_K=$(K) -DGAMEDEV_CUDA
else ifeq ($(GPU_BACKEND),rocm)
GPU_COMPILER ?= hipcc
GPU_TARGET ?= gfx942
GPU_FLAGS := --offload-arch=$(GPU_TARGET) -O3 -gline-tables-only -x hip \
    -include hip/hip_runtime.h -std=c++17 -DVAL_KNN_K=$(K) -DGAMEDEV_ROCM
ifneq ($(strip $(RESOURCE_REPORT)),)
GPU_FLAGS += -Rpass-analysis=kernel-resource-usage
endif
else
$(error GPU_BACKEND must be cuda or rocm)
endif

INCLUDES := -I $(TEST_DIR) -I $(ROOT_DIR)/inc -I $(ROOT_DIR)/inc/swarm

BUILD_DIR := $(ROOT_DIR)/val/swarm/obj/test_knn/$(GPU_BACKEND)
CONFIG_STAMP := $(BUILD_DIR)/.target_$(GPU_TARGET)_k_$(K)

TARGET          := $(BUILD_DIR)/knn_benchmark
EDGE_TARGET     := $(BUILD_DIR)/knn_edge_tests
PERIODIC_TARGET := $(BUILD_DIR)/knn_periodic_tests
WEDGE_TARGET    := $(BUILD_DIR)/knn_wedge_benchmark

TOPOLOGY_TARGET := $(BUILD_DIR)/knn_topology

.DEFAULT_GOAL := all

SEARCH_HEADERS := $(shell find $(ROOT_DIR)/inc/swarm/kdtree $(ROOT_DIR)/inc/swarm/morton -type f)
$(TARGET) $(EDGE_TARGET) $(PERIODIC_TARGET) $(WEDGE_TARGET) $(TOPOLOGY_TARGET): $(SEARCH_HEADERS)

.PHONY: all edge periodic wedge topology suite clean

all: $(TARGET)

edge: $(EDGE_TARGET)

periodic: $(PERIODIC_TARGET)

wedge: $(WEDGE_TARGET)

topology: $(TOPOLOGY_TARGET)

suite: all edge periodic wedge

$(CONFIG_STAMP):
	@mkdir -p $(dir $@)
	@touch $@

$(TARGET): $(TEST_DIR)/knn_benchmark.cu $(TEST_DIR)/knn_types.cuh \
	$(ROOT_DIR)/inc/swarm/kdtree/index_heap.cuh \
	$(ROOT_DIR)/inc/swarm/morton/morton_index.cuh \
	$(ROOT_DIR)/inc/swarm/morton/morton_types.cuh $(TEST_DIR)/morton/morton_index.cuh \
	$(CONFIG_STAMP) $(ROOT_DIR)/inc/gpu.cuh
	@printf "%-22s %s\n" "Using external header" "$(TEST_DIR)/knn_types.cuh"
	@printf "%-12s %s\n" "Compiling" "$@"
	@$(GPU_COMPILER) $(GPU_FLAGS) $(INCLUDES) -o $@ $<

$(EDGE_TARGET): $(TEST_DIR)/knn_edge_tests.cu $(TEST_DIR)/knn_types.cuh \
	$(ROOT_DIR)/inc/swarm/kdtree/builder.h $(ROOT_DIR)/inc/swarm/kdtree/index_heap.cuh \
	$(ROOT_DIR)/inc/swarm/kdtree/knn.h \
	$(ROOT_DIR)/inc/swarm/morton/morton_index.cuh \
	$(ROOT_DIR)/inc/swarm/morton/morton_types.cuh $(TEST_DIR)/morton/morton_index.cuh \
	$(CONFIG_STAMP) $(ROOT_DIR)/inc/gpu.cuh
	@printf "%-22s %s\n" "Using external header" "$(TEST_DIR)/knn_types.cuh"
	@printf "%-12s %s\n" "Compiling" "$@"
	@$(GPU_COMPILER) $(GPU_FLAGS) $(INCLUDES) -o $@ $<

$(PERIODIC_TARGET): $(TEST_DIR)/knn_periodic_tests.cu $(TEST_DIR)/periodic_query.cuh \
	$(ROOT_DIR)/inc/swarm/morton/morton_index.cuh \
	$(ROOT_DIR)/inc/swarm/morton/morton_types.cuh $(TEST_DIR)/morton/morton_index.cuh \
	$(CONFIG_STAMP) $(ROOT_DIR)/inc/gpu.cuh
	@printf "%-22s %s\n" "Using external header" "$(TEST_DIR)/periodic_query.cuh"
	@printf "%-12s %s\n" "Compiling" "$@"
	@$(GPU_COMPILER) $(GPU_FLAGS) $(INCLUDES) -o $@ $<

$(WEDGE_TARGET): $(TEST_DIR)/knn_wedge_benchmark.cu $(TEST_DIR)/knn_types.cuh \
	$(ROOT_DIR)/inc/swarm/kdtree/index_heap.cuh \
	$(ROOT_DIR)/inc/swarm/morton/morton_index.cuh \
	$(ROOT_DIR)/inc/swarm/morton/morton_types.cuh \
	$(TEST_DIR)/morton/morton_index.cuh \
	$(ROOT_DIR)/inc/swarm/morton/morton_ghost.cuh \
	$(ROOT_DIR)/inc/swarm/morton/morton_query.cuh $(CONFIG_STAMP) $(ROOT_DIR)/inc/gpu.cuh
	@printf "%-22s %s\n" "Using external header" "$(TEST_DIR)/knn_types.cuh"
	@printf "%-12s %s\n" "Compiling" "$@"
	@$(GPU_COMPILER) $(GPU_FLAGS) $(INCLUDES) -o $@ $<

$(TOPOLOGY_TARGET): $(TEST_DIR)/knn_topology.cu \
	$(ROOT_DIR)/val/swarm/mod/test_knn/topology.cuh \
	$(ROOT_DIR)/inc/swarm/morton/morton_index.cuh \
	$(ROOT_DIR)/inc/swarm/morton/morton_types.cuh \
	$(TEST_DIR)/morton/morton_index.cuh $(CONFIG_STAMP) $(ROOT_DIR)/inc/gpu.cuh
	@$(GPU_COMPILER) $(GPU_FLAGS) $(INCLUDES) -o $@ $<

clean:
	@printf "%-12s %s\n" "Cleaning" "$(BUILD_DIR)"
	@rm -rf $(BUILD_DIR)
