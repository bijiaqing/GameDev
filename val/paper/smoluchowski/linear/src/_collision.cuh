#pragma once
// the well-mixed Smoluchowski problem uses unit volume, not the geometric neighbor measure
#define _get_ball_measure _geometric_ball_measure
#include "../../../../../inc/swarm/_collision.cuh"
#undef _get_ball_measure
__device__ __forceinline__
real _get_ball_measure(real, real, real radius) { return radius > 0.0 ? 1.0 : 0.0; }
