#ifndef LAB_EROSION_OUTCOME_CUH
#define LAB_EROSION_OUTCOME_CUH
#include "grouped_sticking.cuh"

// Return sampled-rate / physical-rate and conditional absolute log-diameter moments.
// Erosion is the superposition of remnant packets and ungrouped debris transitions.
__host__ __device__ inline real erosion_outcome_moments(real si, real sj,
    bool high_speed, real &mean, real &second, real &maximum) {
    real q=sticking_mass_ratio(si,sj);
    real G=sticking_packet(q,false);
    if (!high_speed) {
        mean=log1p(G*q)/3.0; second=mean*mean; maximum=mean;
        return 1.0/G;
    }
    if (q<=0.1) {
        real remnant=(1.0-q)/G, debris=q, factor=remnant+debris;
        real jr=-log1p(-G*q)/3.0, jd=-log(q)/3.0;
        mean=(remnant*jr+debris*jd)/factor;
        second=(remnant*jr*jr+debris*jd*jd)/factor;
        maximum=fmax(jr,jd);
        return factor;
    }
    // Fragment diameter: [sqrt(s_min)+U*(sqrt(si)-sqrt(s_min))]^2.
    // For L=log(si/s_min)/2, integrate -2 log(y) over y in [exp(-L),1].
    real L=0.5*log(si/INIT_SMIN);
    if (L<1.e-3) {
        // Series avoid cancellation when the target is near the monomer floor.
        mean=L-L*L/6.0+L*L*L*L/360.0;
        second=L*L*(4.0/3.0-L/3.0+L*L/90.0+L*L*L/180.0);
    } else {
        real tail=exp(-L)/(-expm1(-L));
        mean=2.0*(1.0-L*tail);
        second=8.0-(4.0*L*L+8.0*L)*tail;
    }
    maximum=2.0*L;
    return 1.0;
}

// Categories: four sticking q bins, fragmentation, remnant erosion, debris erosion.
// u is used only for high-speed events; the caller supplies one independent draw.
__host__ __device__ inline real sample_erosion_outcome(real si, real sj,
    bool high_speed, real u, int &category, real &log_mass) {
    real q=sticking_mass_ratio(si,sj);
    real G=sticking_packet(q,false);
    if (!high_speed) {
        category=q<=1.e-6?0:q<=1.e-4?1:q<=1.e-2?2:3;
        log_mass=log1p(G*q);
        return cbrt(si*si*si+G*sj*sj*sj);
    }
    if (q<=0.1) {
        real factor=(1.0-q)/G+q;
        if (u<q/factor) {
            category=6; log_mass=log(q);
            return sj;
        }
        category=5; log_mass=log1p(-G*q);
        return si*cbrt(1.0-G*q);
    }
    category=4;
    real lower=sqrt(INIT_SMIN);
    real root=lower+u*(sqrt(si)-lower);
    real size=fmin(si,fmax(INIT_SMIN,root*root));
    log_mass=3.0*log(size/si);
    return size;
}
#endif
