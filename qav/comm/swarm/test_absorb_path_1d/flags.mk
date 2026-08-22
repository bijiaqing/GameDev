DUST_REPR := swarm
MODEL_INCLUDE_DIRS := $(QAV_SWARM_DIR)/test_common

# SAVE_DENS verifies that the production particle-to-grid path excludes absorbed representatives
GPU_FLAGS += -DTRANSPORT -DRADIATION -DSAVE_DENS -DCOLLISION -DMULTISIZE -DCODE_UNIT
