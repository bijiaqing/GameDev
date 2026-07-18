################################################################################
# Compile-time feature flags. Uncomment a line to enable the corresponding path.
################################################################################

# Radiation pressure, optical depth, and beta.
NVCC += -DRADIATION

# Turbulent dust diffusion. Required whenever N_Z > 1.
# NVCC += -DDIFFUSION

# Use constant kinematic viscosity NU instead of alpha viscosity.
# NVCC += -DCONST_NU

# Upper half-disk with Z_MAX=pi/2 and a reflecting midplane.
# Without HALFDISK, an N_Z > 1 domain must span the midplane and both polar boundaries are outflow.
# NVCC += -DHALFDISK
