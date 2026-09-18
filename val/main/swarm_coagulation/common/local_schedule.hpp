#ifndef COLLISION_LOCAL_SCHEDULE_HPP
#define COLLISION_LOCAL_SCHEDULE_HPP
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
#include <utility>

// Integer endpoints avoid rounding drift between power-of-two timestep levels.
struct local_schedule {
    double horizon;
    std::vector<std::pair<int,int>> links;
    std::vector<int> level;
    std::vector<std::uint64_t> step, next;
    std::uint64_t end;

    local_schedule(double h, const std::vector<double>& requested,
                   const std::vector<unsigned int>& edges)
        : horizon(h), level(requested.size(), 0), step(requested.size()),
          next(requested.size(), 0) {
        if (!(h > 0) || !std::isfinite(h))
            throw std::runtime_error("invalid local collision horizon");
        const int n = static_cast<int>(level.size()), words = (n+31)/32;
        for (int c=0; c<n; ++c) {
            if (!(requested[c] > 0) || !std::isfinite(requested[c]))
                throw std::runtime_error("invalid local collision timestep");
            double dt = h;
            while (dt > requested[c]) {
                if (++level[c] > 52)
                    throw std::runtime_error("local collision timestep exceeds time resolution");
                dt *= 0.5;
            }
        }
        for (int c=0;c<n;++c) for (int d=c+1;d<n;++d)
            if ((edges[c*words+d/32] & (1u<<(d%32))) ||
                (edges[d*words+c/32] & (1u<<(c%32)))) links.emplace_back(c,d);
        // Symmetric compatibility on every directed KNN dependency: ratio <= 16.
        bool changed;
        do {
            changed = false;
            for (int c=0; c<n; ++c) for (int d=0; d<n; ++d) {
                if (!(edges[c*words+d/32] & (1u << (d%32)))) continue;
                if (level[c] < level[d]-4) { level[c]=level[d]-4; changed=true; }
                if (level[d] < level[c]-4) { level[d]=level[c]-4; changed=true; }
            }
        } while (changed);
        // A fixed lattice permits later refinement without moving pending endpoints.
        int finest = 52;
        end = std::uint64_t(1) << finest;
        for (int c=0; c<n; ++c) step[c] = std::uint64_t(1) << (finest-level[c]);
    }
    std::uint64_t time() const { return *std::min_element(next.begin(),next.end()); }
    double seconds(std::uint64_t tick) const { return horizon*(double(tick)/double(end)); }
    std::vector<int> due(std::uint64_t tick) const {
        std::vector<int> out;
        for (int c=0; c<int(next.size()); ++c) if (next[c]==tick) out.push_back(c);
        return out;
    }
    // Only due groups may change: other groups already hold computed endpoints.
    void adapt(std::uint64_t tick, const std::vector<int>& groups,
               const std::vector<double>& requested, const std::vector<bool>& passed) {
        const int n=level.size();
        std::vector<bool> active(n,false);
        for (int c:groups) active[c]=true;
        // Largest permitted level, propagated from immutable pending neighbors.
        std::vector<int> cap(n,52), candidate=level;
        for (int c=0;c<n;++c) if (!active[c]) cap[c]=level[c];
        bool changed;
        do {
            changed=false;
            for (const auto &edge:links) {
                int c=edge.first,d=edge.second;
                if (active[c] && cap[c]>cap[d]+4) {cap[c]=cap[d]+4;changed=true;}
                if (active[d] && cap[d]>cap[c]+4) {cap[d]=cap[c]+4;changed=true;}
            }
        } while (changed);
        for (int c:groups) {
            if (!(requested[c]>0) || !std::isfinite(requested[c]))
                throw std::runtime_error("invalid adaptive collision timestep");
            int want=0;
            double h=horizon;
            while (h>requested[c]) {
                if (++want>52) throw std::runtime_error("adaptive collision time resolution exceeded");
                h*=0.5;
            }
            // After a passing audit, recover directly; alignment and neighbors still constrain it.
            candidate[c]=passed[c] ? want : std::max(want,level[c]);
            while (tick % (std::uint64_t(1)<<(52-candidate[c]))) ++candidate[c];
            candidate[c]=std::min(candidate[c],cap[c]);
        }
        // Refine due groups until all directed dependencies satisfy ratio <= 16.
        // Caps above ensure this never requires changing an in-flight endpoint.
        do {
            changed=false;
            for (const auto &edge:links) {
                int c=edge.first,d=edge.second;
                if (active[c] && candidate[c]<candidate[d]-4) {
                    candidate[c]=candidate[d]-4;changed=true;
                }
                if (active[d] && candidate[d]<candidate[c]-4) {
                    candidate[d]=candidate[c]-4;changed=true;
                }
            }
        } while (changed);
        for (int c:groups) {
            level[c]=candidate[c];
            step[c]=std::uint64_t(1)<<(52-level[c]);
        }
    }
    void advance(const std::vector<int>& groups) {
        for (int c:groups) next[c] += step[c];
    }
};
#endif
