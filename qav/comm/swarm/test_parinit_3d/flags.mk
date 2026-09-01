DUST_REPR := swarm
MODEL_INCLUDE_DIRS := $(QAV_SWARM_DIR)/test_common

# initialize monodisperse particles with the resolved-vertical analytic gas and alpha-viscosity prescriptions
GPU_FLAGS += -DTRANSPORT -DDIFFUSION -DVISC_FLOW
