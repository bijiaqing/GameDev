DUST_REPR := fluid
FLUID_SWEEP := block

GPU_FLAGS += -DDIFFUSION
CUDA_FLAGS += -lineinfo -Xptxas=-v
ROCM_FLAGS += -gline-tables-only
