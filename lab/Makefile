ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

NVCC ?= nvcc
ARCH ?= sm_80
K ?= 200

TARGET := $(ROOT_DIR)/bin/knn_benchmark
SOURCE := $(ROOT_DIR)/knn/knn_benchmark.cu
EDGE_TARGET := $(ROOT_DIR)/bin/knn_edge_tests
EDGE_SOURCE := $(ROOT_DIR)/knn/knn_edge_tests.cu
PERIODIC_TARGET := $(ROOT_DIR)/bin/knn_periodic_tests
PERIODIC_SOURCE := $(ROOT_DIR)/knn/knn_periodic_tests.cu
WEDGE_TARGET := $(ROOT_DIR)/bin/knn_wedge_benchmark
WEDGE_SOURCE := $(ROOT_DIR)/knn/knn_wedge_benchmark.cu

NVCCFLAGS := -arch=$(ARCH) -O3 --use_fast_math -std=c++17
NVCCFLAGS += -DLAB_KNN_K=$(K)
INCLUDES := -I $(ROOT_DIR)/knn -I $(ROOT_DIR)/inc/swarm

.PHONY: all edge periodic wedge clean

all: $(TARGET)

edge: $(EDGE_TARGET)

periodic: $(PERIODIC_TARGET)

wedge: $(WEDGE_TARGET)

$(TARGET): $(SOURCE) $(ROOT_DIR)/knn/knn_types.cuh $(ROOT_DIR)/knn/adaptive_morton.cuh
	@mkdir -p $(dir $@)
	@printf "%-12s %s\n" "Compiling" "$@"
	@$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $<

$(EDGE_TARGET): $(EDGE_SOURCE) $(ROOT_DIR)/knn/knn_types.cuh $(ROOT_DIR)/knn/adaptive_morton.cuh
	@mkdir -p $(dir $@)
	@printf "%-12s %s\n" "Compiling" "$@"
	@$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $<

$(PERIODIC_TARGET): $(PERIODIC_SOURCE) $(ROOT_DIR)/knn/knn_types.cuh \
	$(ROOT_DIR)/knn/adaptive_morton.cuh $(ROOT_DIR)/knn/periodic_query.cuh
	@mkdir -p $(dir $@)
	@printf "%-12s %s\n" "Compiling" "$@"
	@$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $<

$(WEDGE_TARGET): $(WEDGE_SOURCE) $(ROOT_DIR)/knn/knn_types.cuh \
	$(ROOT_DIR)/knn/adaptive_morton.cuh $(ROOT_DIR)/knn/periodic_query.cuh
	@mkdir -p $(dir $@)
	@printf "%-12s %s\n" "Compiling" "$@"
	@$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $<

clean:
	@printf "%-12s %s\n" "Cleaning" "$(ROOT_DIR)/bin"
	@rm -rf $(ROOT_DIR)/bin
