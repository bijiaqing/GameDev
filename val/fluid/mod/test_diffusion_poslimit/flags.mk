DUST_REPR := fluid
MODEL_INCLUDE_DIRS := $(VAL_ROOT)/fluid/src

# constant nu makes the cyclic CN amplification factor independent of disk thermodynamics
GPU_FLAGS += -DDIFFUSION -DCONST_NU
