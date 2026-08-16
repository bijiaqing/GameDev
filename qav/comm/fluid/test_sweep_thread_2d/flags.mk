DUST_REPR := fluid
FLUID_SWEEP := thread

CUDA_FLAGS += -lineinfo -Xptxas=-v
ROCM_FLAGS += -gline-tables-only
