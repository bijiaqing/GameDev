DUST_REPR := swarm
MODEL_INCLUDE_DIRS := $(VAL_SWARM_DIR)/test_common

# exercise the three-dimensional periodic-image velocity in the physical custom kernel
GPU_FLAGS += -DCOLLISION -DBERNOULLI -DKNN_CACHE -DMULTISIZE -DCODE_UNIT
