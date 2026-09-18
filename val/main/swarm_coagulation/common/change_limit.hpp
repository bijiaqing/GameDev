#ifndef CHANGE_REFRESH_LIMIT_HPP
#define CHANGE_REFRESH_LIMIT_HPP
#include <algorithm>
#include <cmath>
#include <stdexcept>

struct change_bound { double duration; int reason; };

// A = mass-weighted mean absolute log-size jump rate.
// B = mass-weighted second log-size jump moment rate (not variance of the mean).
// Frozen-rate estimates: mean accumulated absolute change = h*A;
// compound-Poisson fluctuation scale = sqrt(h*B). Bound each by epsilon.
inline change_bound change_limit(double A, double B, double epsilon, double horizon) {
    if (!std::isfinite(A) || A<0 || !std::isfinite(B) || B<0
        || !std::isfinite(epsilon) || !(epsilon>0)
        || !std::isfinite(horizon) || !(horizon>0))
        throw std::runtime_error("invalid change-based refresh inputs");
    change_bound result{horizon,0};
    if (A>0 && epsilon/A<result.duration) result={epsilon/A,1};
    if (B>0 && epsilon*epsilon/B<result.duration) result={epsilon*epsilon/B,2};
    if (!(result.duration>0)) throw std::runtime_error("change-based timestep underflow");
    return result;
}
#endif
