TEST_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT_DIR := $(abspath $(TEST_DIR)/../../../..)

HIPCC ?= hipcc
AMDGPU_TARGET ?= gfx942
K ?= 200

HIPFLAGS := --offload-arch=$(AMDGPU_TARGET) -O3 -gline-tables-only \
    -include hip/hip_runtime.h -std=c++17 -DQAV_KNN_K=$(K)

ifneq ($(strip $(RESOURCE_REPORT)),)
HIPFLAGS += -Rpass-analysis=kernel-resource-usage
endif
INCLUDES := -I $(TEST_DIR) -I $(ROOT_DIR)/inc/rocm/swarm -I $(ROOT_DIR)/inc/comm/swarm

CONFIG_STAMP := $(TEST_DIR)/.target_$(AMDGPU_TARGET)_k_$(K)

TARGET          := $(TEST_DIR)/knn_benchmark
EDGE_TARGET     := $(TEST_DIR)/knn_edge_tests
PERIODIC_TARGET := $(TEST_DIR)/knn_periodic_tests
WEDGE_TARGET    := $(TEST_DIR)/knn_wedge_benchmark

.PHONY: all edge periodic wedge suite clean

all: $(TARGET)

edge: $(EDGE_TARGET)

periodic: $(PERIODIC_TARGET)

wedge: $(WEDGE_TARGET)

suite: all edge periodic wedge

$(CONFIG_STAMP):
	@touch $@

$(TARGET): $(TEST_DIR)/knn_benchmark.hip $(TEST_DIR)/knn_types.cuh \
	$(ROOT_DIR)/inc/rocm/swarm/kdtree/index_heap.cuh \
	$(ROOT_DIR)/inc/rocm/swarm/morton/morton_types.cuh $(ROOT_DIR)/inc/rocm/swarm/morton/morton_index.cuh \
	$(CONFIG_STAMP)
	@printf "%-22s %s\n" "Using external header" "$(TEST_DIR)/knn_types.cuh"
	@printf "%-12s %s\n" "Compiling" "$@"
	@$(HIPCC) $(HIPFLAGS) $(INCLUDES) -o $@ $<

$(EDGE_TARGET): $(TEST_DIR)/knn_edge_tests.hip $(TEST_DIR)/knn_types.cuh \
	$(ROOT_DIR)/inc/rocm/swarm/kdtree/builder.h $(ROOT_DIR)/inc/rocm/swarm/kdtree/index_heap.cuh \
	$(ROOT_DIR)/inc/rocm/swarm/kdtree/knn.h \
	$(ROOT_DIR)/inc/rocm/swarm/morton/morton_types.cuh $(ROOT_DIR)/inc/rocm/swarm/morton/morton_index.cuh \
	$(CONFIG_STAMP)
	@printf "%-22s %s\n" "Using external header" "$(TEST_DIR)/knn_types.cuh"
	@printf "%-12s %s\n" "Compiling" "$@"
	@$(HIPCC) $(HIPFLAGS) $(INCLUDES) -o $@ $<

$(PERIODIC_TARGET): $(TEST_DIR)/knn_periodic_tests.hip $(TEST_DIR)/periodic_query.cuh \
	$(ROOT_DIR)/inc/rocm/swarm/morton/morton_types.cuh $(ROOT_DIR)/inc/rocm/swarm/morton/morton_index.cuh \
	$(CONFIG_STAMP)
	@printf "%-22s %s\n" "Using external header" "$(TEST_DIR)/periodic_query.cuh"
	@printf "%-12s %s\n" "Compiling" "$@"
	@$(HIPCC) $(HIPFLAGS) $(INCLUDES) -o $@ $<

$(WEDGE_TARGET): $(TEST_DIR)/knn_wedge_benchmark.hip $(TEST_DIR)/knn_types.cuh \
	$(ROOT_DIR)/inc/rocm/swarm/kdtree/index_heap.cuh \
	$(ROOT_DIR)/inc/rocm/swarm/morton/morton_types.cuh \
	$(ROOT_DIR)/inc/rocm/swarm/morton/morton_index.cuh \
	$(ROOT_DIR)/inc/rocm/swarm/morton/morton_ghost.cuh \
	$(ROOT_DIR)/inc/rocm/swarm/morton/morton_query.cuh $(CONFIG_STAMP)
	@printf "%-22s %s\n" "Using external header" "$(TEST_DIR)/knn_types.cuh"
	@printf "%-12s %s\n" "Compiling" "$@"
	@$(HIPCC) $(HIPFLAGS) $(INCLUDES) -o $@ $<

clean:
	@printf "%-12s %s\n" "Cleaning" "$(TEST_DIR) executables"
	@rm -f $(TARGET) $(EDGE_TARGET) $(PERIODIC_TARGET) $(WEDGE_TARGET) \
		$(TEST_DIR)/.target_* $(TEST_DIR)/production_test_collision_*
