#include <gpu.cuh>

#ifdef COLLISION
#include <_col_rates.cuh>
#include <swarm_kern.cuh>

// COL_QUERY_ENV_CACHE is decided in _col_types.cuh, so test it only after the includes
#ifdef COL_QUERY_ENV_CACHE

// =====================================================================================================================
// kernel: col_env_cache
// cache every active owner's gas environment; absorbed particles at y=0 are skipped
// =====================================================================================================================

__global__
void col_env_cache (query_environment *env, const swarm *particle)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<N_P && particle[i].position.y>0.0) env[i] = _cache_query_environment(particle[i]);
}

#endif // COL_QUERY_ENV_CACHE
#endif // COLLISION
