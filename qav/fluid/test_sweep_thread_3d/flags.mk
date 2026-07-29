DUST_REPR := fluid
FLUID_SWEEP := thread

NVCC += -DDIFFUSION
NVCC += -lineinfo -Xptxas=-v
