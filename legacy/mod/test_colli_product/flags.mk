#################################################################################################
# Compile-time feature flags, uncomment lines below to enable specific physics modules			#
#################################################################################################
# physical features																				#
#																								#
# COLLISION: Enable dust collision and coagulation/fragmentation								#
NVCC += -DCOLLISION #																			#
#																								#
# TRANSPORT: Enable particle transport (position/velocity evolution) 							#
# NOTE: If TRANSPORT is off, RADIATION and DIFFUSION are inactive regardless of their flags		#
# NVCC += -DTRANSPORT #																			#
#																								#
# RADIATION: Enable radiation pressure calculations (optical depth, beta)						#
# NVCC += -DRADIATION #																			#
#																								#
# DIFFUSION: Enable turbulent diffusion of dust particles										#
# NVCC += -DDIFFUSION #																			#
#																								#
#################################################################################################
# extra constraints																				#
#																								#
# CONST_NU: Use constant kinematic viscosity NU instead of alpha-viscosity						#
# NVCC += -DCONST_NU #																			#
#																								#
# CONST_ST: Use constant Stokes number instead of constant physical size for dust particles		#
# NVCC += -DCONST_ST #																			#
#																								#
#################################################################################################
# file output features																			#
#																								#
# SAVE_DENS: Save dust density field to output files											#
# NVCC += -DSAVE_DENS #																			#
#																								#
# LOGTIMING: Use dynamic, logarithmic time stepping for the simulation evolution				#
# NVCC += -DLOGTIMING #																			#
#																								#
# LOGOUTPUT: Use logarithmic output intervals for particles with fixed output timesteps			#
NVCC += -DLOGOUTPUT #																			#
#																								#
#################################################################################################
# numercial features																			#
#																								#
# CODE_UNIT: Use code units instead of cgs units												#
NVCC += -DCODE_UNIT #																			#
#																								#
# IMPORTGAS: Import gas disk parameters from external file instead of using analytical profiles	#
# NVCC += -DIMPORTGAS #																			#
#																								#
#################################################################################################