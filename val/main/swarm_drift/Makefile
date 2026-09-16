# Reuse production build rules with this test set's model, output, and build roots.
DRIFT_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
override MOD_ROOT := $(DRIFT_ROOT)/models
override OUT_ROOT := $(DRIFT_ROOT)/outputs
DRIFT_TEMP := $(abspath $(DRIFT_ROOT)/../../temp/swarm_drift)
override OBJ_ROOT := $(DRIFT_TEMP)/obj
override EXEC = $(DRIFT_TEMP)/bin/$(MODEL)/gamedev
override MODEL_EXECUTABLES = $(wildcard $(DRIFT_TEMP)/bin/*/gamedev)
include $(DRIFT_ROOT)/../../../Makefile
