DUST_REPR := swarm

# retain the production split radiation transport around the model-local drag-free force update
GPU_FLAGS += -DTRANSPORT -DRADIATION -DMULTISIZE
