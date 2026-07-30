TEST_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT_DIR := $(abspath $(TEST_DIR)/../../..)
BIN_DIR  := $(TEST_DIR)/bin

NVCC ?= nvcc
ARCH ?= sm_80
K ?= 200

NVCC_FLAGS := -arch=$(ARCH) -O3 --use_fast_math -std=c++17 -DQAV_KNN_K=$(K)
INCLUDES := -I $(TEST_DIR) -I $(ROOT_DIR)/inc/swarm

TARGET          := $(BIN_DIR)/knn_benchmark
EDGE_TARGET     := $(BIN_DIR)/knn_edge_tests
PERIODIC_TARGET := $(BIN_DIR)/knn_periodic_tests
WEDGE_TARGET    := $(BIN_DIR)/knn_wedge_benchmark

.PHONY: all edge periodic wedge suite clean

all: $(TARGET)

edge: $(EDGE_TARGET)

periodic: $(PERIODIC_TARGET)

wedge: $(WEDGE_TARGET)

suite: all edge periodic wedge

$(TARGET): $(TEST_DIR)/knn_benchmark.cu $(TEST_DIR)/knn_types.cuh \
	$(ROOT_DIR)/inc/swarm/kdtree/index_heap.cuh \
	$(ROOT_DIR)/inc/swarm/morton/morton_types.cuh $(ROOT_DIR)/inc/swarm/morton/morton_index.cuh
	@mkdir -p $(BIN_DIR)
	@printf "%-12s %s\n" "Compiling" "$@"
	@$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

$(EDGE_TARGET): $(TEST_DIR)/knn_edge_tests.cu \
	$(ROOT_DIR)/inc/swarm/morton/morton_types.cuh $(ROOT_DIR)/inc/swarm/morton/morton_index.cuh
	@mkdir -p $(BIN_DIR)
	@printf "%-12s %s\n" "Compiling" "$@"
	@$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

$(PERIODIC_TARGET): $(TEST_DIR)/knn_periodic_tests.cu $(TEST_DIR)/periodic_query.cuh \
	$(ROOT_DIR)/inc/swarm/morton/morton_types.cuh $(ROOT_DIR)/inc/swarm/morton/morton_index.cuh
	@mkdir -p $(BIN_DIR)
	@printf "%-12s %s\n" "Compiling" "$@"
	@$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

$(WEDGE_TARGET): $(TEST_DIR)/knn_wedge_benchmark.cu $(TEST_DIR)/knn_types.cuh \
	$(ROOT_DIR)/inc/swarm/kdtree/index_heap.cuh \
	$(TEST_DIR)/periodic_query.cuh $(ROOT_DIR)/inc/swarm/morton/morton_types.cuh \
	$(ROOT_DIR)/inc/swarm/morton/morton_index.cuh \
	$(ROOT_DIR)/inc/swarm/morton/morton_ghost.cuh \
	$(ROOT_DIR)/inc/swarm/morton/morton_query.cuh
	@mkdir -p $(BIN_DIR)
	@printf "%-12s %s\n" "Compiling" "$@"
	@$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

clean:
	@printf "%-12s %s\n" "Cleaning" "$(BIN_DIR)"
	@rm -rf $(BIN_DIR)
