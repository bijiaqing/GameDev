DUST_REPR := fluid
FLUID_SWEEP := block

NVCC += -DDIFFUSION
NVCC += -lineinfo -Xptxas=-v
