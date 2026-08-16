DUST_REPR := fluid
FLUID_SWEEP := thread

GPU_FLAGS += -DDIFFUSION
CUDA_FLAGS += -lineinfo -Xptxas=-v
ROCM_FLAGS += -gline-tables-only
