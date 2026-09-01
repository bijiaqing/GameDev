DUST_REPR := fluid

# exercise the complete resolved-polar production initializer with alpha viscosity and viscous gas inflow
GPU_FLAGS += -DDIFFUSION -DVISC_FLOW
