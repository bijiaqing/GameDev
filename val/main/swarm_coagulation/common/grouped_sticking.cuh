#ifndef LAB_GROUPED_STICKING_CUH
#define LAB_GROUPED_STICKING_CUH
// Only tiny sticking projectiles; each packet adds at most 0.01% target mass.
__host__ __device__ inline real sticking_packet(real q, bool fragmentation) {
    return !fragmentation && q>0.0 && q<=1.e-6
        ? fmax(1.0,floor(1.e-4/q)) : 1.0;
}
__host__ __device__ inline real sticking_mass_ratio(real size_i, real size_j) {
    real ratio=size_j/size_i;
    return ratio*ratio*ratio;
}
// ponytail: packets preserve frozen-state mean mass growth but inflate variance;
// lower the 1e-4 packet bound if distribution comparisons show a bias.
#endif
