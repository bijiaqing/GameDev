DUST_REPR := swarm
MODEL_INCLUDE_DIRS := $(VAL_ROOT)/swarm/src

# retain the production split radiation transport around the model-local drag-free force update
GPU_FLAGS += -DTRANSPORT -DRADIATION -DMULTISIZE
