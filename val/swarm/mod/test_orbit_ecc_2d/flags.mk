DUST_REPR := swarm
MODEL_INCLUDE_DIRS := $(VAL_ROOT)/swarm/src

# build the production transport kernel around the model-local drag-free second substep
GPU_FLAGS += -DTRANSPORT
