# Reuse production build rules with this test set's model, output, and build roots.
ORBIT_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
override MOD_ROOT := $(ORBIT_ROOT)/models
override OUT_ROOT := $(ORBIT_ROOT)/outputs
override OBJ_ROOT := $(ORBIT_ROOT)/build
include $(ORBIT_ROOT)/../../../Makefile
