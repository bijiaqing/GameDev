DUST_REPR := swarm
MODEL_INCLUDE_DIRS := $(QAV_SWARM_DIR)/test_common

# activate every analytic dynamics-rate family without imported-grid interpolation or radiation-size coupling
GPU_FLAGS += -DTRANSPORT -DDIFFUSION -DVISC_FLOW -DCONST_NU
