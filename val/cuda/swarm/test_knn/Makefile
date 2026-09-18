TEST_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT_DIR := $(abspath $(TEST_DIR)/../../../..)

NVCC ?= nvcc
ARCH ?= sm_80
K ?= 200

NVCC_FLAGS := -arch=$(ARCH) -O3 --use_fast_math -lineinfo -std=c++17 -DVAL_KNN_K=$(K) -DGAMEDEV_CUDA
INCLUDES := -I $(TEST_DIR) -I $(ROOT_DIR)/inc -I $(ROOT_DIR)/inc/swarm

BUILD_DIR := $(ROOT_DIR)/val/temp/cuda/swarm/test_knn
CONFIG_STAMP := $(BUILD_DIR)/.arch_$(ARCH)_k_$(K)

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
	$(CONFIG_STAMP) $(ROOT_DIR)/inc/gpu_compat.cuh
	@printf "%-22s %s\n" "Using external header" "$(TEST_DIR)/knn_types.cuh"
	@printf "%-12s %s\n" "Compiling" "$@"
	@$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

$(EDGE_TARGET): $(TEST_DIR)/knn_edge_tests.cu $(TEST_DIR)/knn_types.cuh \
	$(ROOT_DIR)/inc/swarm/kdtree/builder.h $(ROOT_DIR)/inc/swarm/kdtree/index_heap.cuh \
	$(ROOT_DIR)/inc/swarm/kdtree/knn.h \
	$(ROOT_DIR)/inc/swarm/morton/morton_index.cuh \
	$(ROOT_DIR)/inc/swarm/morton/morton_types.cuh $(TEST_DIR)/morton/morton_index.cuh \
	$(CONFIG_STAMP) $(ROOT_DIR)/inc/gpu_compat.cuh
	@printf "%-22s %s\n" "Using external header" "$(TEST_DIR)/knn_types.cuh"
	@printf "%-12s %s\n" "Compiling" "$@"
	@$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

$(PERIODIC_TARGET): $(TEST_DIR)/knn_periodic_tests.cu $(TEST_DIR)/periodic_query.cuh \
	$(ROOT_DIR)/inc/swarm/morton/morton_index.cuh \
	$(ROOT_DIR)/inc/swarm/morton/morton_types.cuh $(TEST_DIR)/morton/morton_index.cuh \
	$(CONFIG_STAMP) $(ROOT_DIR)/inc/gpu_compat.cuh
	@printf "%-22s %s\n" "Using external header" "$(TEST_DIR)/periodic_query.cuh"
	@printf "%-12s %s\n" "Compiling" "$@"
	@$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

$(WEDGE_TARGET): $(TEST_DIR)/knn_wedge_benchmark.cu $(TEST_DIR)/knn_types.cuh \
	$(ROOT_DIR)/inc/swarm/kdtree/index_heap.cuh \
	$(ROOT_DIR)/inc/swarm/morton/morton_index.cuh \
	$(ROOT_DIR)/inc/swarm/morton/morton_types.cuh \
	$(TEST_DIR)/morton/morton_index.cuh \
	$(ROOT_DIR)/inc/swarm/morton/morton_ghost.cuh \
	$(ROOT_DIR)/inc/swarm/morton/morton_query.cuh $(CONFIG_STAMP) $(ROOT_DIR)/inc/gpu_compat.cuh
	@printf "%-22s %s\n" "Using external header" "$(TEST_DIR)/knn_types.cuh"
	@printf "%-12s %s\n" "Compiling" "$@"
	@$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

$(TOPOLOGY_TARGET): $(TEST_DIR)/knn_topology.cu \
	$(ROOT_DIR)/val/comm/swarm/test_knn/topology.cuh \
	$(ROOT_DIR)/val/comm/swarm/test_common/device_api.cuh \
	$(ROOT_DIR)/inc/swarm/morton/morton_index.cuh \
	$(ROOT_DIR)/inc/swarm/morton/morton_types.cuh \
	$(TEST_DIR)/morton/morton_index.cuh $(CONFIG_STAMP) $(ROOT_DIR)/inc/gpu_compat.cuh
	@$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

clean:
	@printf "%-12s %s\n" "Cleaning" "$(BUILD_DIR)"
	@rm -rf $(BUILD_DIR)
