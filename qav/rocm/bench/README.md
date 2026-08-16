# ROCm performance qualification

This branch separates uninstrumented timing from profiler collection. Timing summaries must come
from `run_suite.py`; profiler-instrumented wall times are deliberately labelled as non-baseline
measurements because ROCm profiling can replay kernels and perturb execution.

## Uninstrumented benchmark matrix

Run one untimed warm-up followed by five measured repetitions:

```bash
python3 qav/rocm/bench/run_suite.py --group all --scale baseline --target gfx942 --repeat 5
python3 qav/rocm/bench/run_suite.py --group all --scale production --target gfx942 --repeat 5
```

The baseline matrix contains $512^2$ and $64^3$ block-fluid models and collision-only swarms with
$10^5$ particles using KD-tree and Morton search. The production matrix raises these to $1024^2$,
$128^3$, and $10^6$ particles. `--scale large` selects the optional $2048^2$, $256^3$, and
$10^7$-particle cases. Fluid performance runs use the production safety choice
`CFL_DYN = 0.45`; the ordinary sweep-equivalence tests retain their default `0.5` edge-case value.
Use `--group fluid` or `--group swarm` to run only one representation.

Collision duration is scale dependent because the fastest collision rate increases with particle
count: the baseline uses $2\times10^{-3}$, production uses $2\times10^{-5}$, and the optional large
case uses $3\times10^{-6}$ code-time units. This changes neither the collision kernel nor its CFL;
it prevents large cases from executing many thousands of statistically redundant batches. Every
run must still complete at least 100 batches. Select or repeat individual cases with `--case`, for
example:

```bash
python3 qav/rocm/bench/run_suite.py \
    --case swarm_collision_kdtree_production \
    --case swarm_collision_morton_production \
    --target gfx942 \
    --repeat 5
```

For a pilot measurement, one selected case may override the interval with `--duration`; such a run
has a different configuration and must not be combined with repeats using another duration.

Every case compiles once, runs an untimed warm-up, and then executes from the same deterministic
initial condition. The fluid cases advance for at least 100 accepted global steps; collision cases
complete at least 100 frozen-rate collision batches. Only the initial and final state are saved.
The warm-up samples process memory through `amd-smi`; measured repetitions do not run a memory
monitor. Each summary reports median wall and evolution time, interquartile range, full spread,
simulated-time throughput, and time per cell-step or particle-batch. A spread greater than five
percent is marked `noisy` and should be rerun on an idle APU.

All structured records are written below `qav/logs/bench/rocm/`. Each campaign writes a scale-specific
manifest such as `manifest_all_baseline.json` as well as `manifest.json` for the most recent run.
`inventory.json` indexes every completed case still present below `qav/logs/bench/rocm/`, including cases produced
by earlier selected campaigns.
Native final-state files remain in the ordinary ignored fluid or swarm QA output tree and are
referenced by the summary. Build streams, resolved HIP compile commands, native streams,
environment information, memory availability, and every repetition are JSON. A SHA-256 fingerprint
of all build-relevant text files supplies source provenance when the cluster copy has no `.git`
directory.

## Runtime trace and hardware counters

First collect a short runtime trace:

```bash
python3 qav/rocm/bench/profile.py \
    --case fluid_3d_production \
    --tool trace \
    --target gfx942
```

Rank kernels by total duration and call count in the profiler-native artifacts. Then collect focused
hardware counters for one measured hot kernel, for example:

```bash
ROCPROFCOMPUTE_COLOR=0 python3 qav/rocm/bench/profile.py \
    --case fluid_3d_production \
    --tool compute \
    --kernel advection_ybl \
    --target gfx942

ROCPROFCOMPUTE_COLOR=0 python3 qav/rocm/bench/profile.py \
    --case fluid_3d_production \
    --tool compute \
    --kernel advection_zbl \
    --target gfx942

python3 qav/rocm/bench/analyze_counters.py \
    --profile-dir \
        qav/logs/bench/rocm/profiles/fluid_3d_production_advection_ybl \
        qav/logs/bench/rocm/profiles/fluid_3d_production_advection_zbl \
    --output qav/logs/bench/rocm/counter_analysis_yz.json
```

The archived MI300A runtime trace identifies `advection_ybl` and `advection_zbl` as the dominant
3D kernels. `analyze_counters.py` requires each profile to contain its passing manifest and complete
merged `workloads/**/pmc_perf.csv`, rejects inconsistent per-dispatch resource metadata, and writes
one JSON summary of resource use, lane activity, residency, L2 behavior, and external-memory
traffic. Recompute the ranking after material source, compiler, grid, or physics changes rather
than treating these targets as fixed.

Equivalent swarm cases are named `swarm_collision_kdtree_baseline`,
`swarm_collision_morton_baseline`, and their `production` or `large` variants. Profiler-native JSON,
CSV, database, and workload directories remain untouched. The adjacent manifest records their paths,
sizes, tool version, command, kernel filter, and explicitly states that instrumented timing is not a
performance baseline. Trace defaults are long enough to sample several operator calls; focused
counter defaults are shorter because the profiler can replay the selected kernel. Override either
with `--duration` when the trace shows too few calls or counter collection is unnecessarily long.

## Recorded MI300A evidence

The 2026-08-12--16 gfx942 campaign provides a historical baseline; generated records are not
tracked and must be regenerated after material source, compiler, or machine changes. Five-repeat
median wall times were 4.604 s and 7.682 s for the $512^2$ and $1024^2$ fluid cases, 8.536 s and
78.755 s for $64^3$ and $128^3$, 60.838 s and 170.785 s for the $10^5$ and $10^6$ KD-tree swarm
cases, and 83.404 s and 203.565 s for the corresponding Morton cases. The $10^6$ KD-tree spread was
6.34% and is therefore noisy; the other seven cases remained below the five-percent threshold.

The production 3D runtime trace attributed 46.53% of aggregate kernel time to `advection_ybl` and
36.73% to `advection_zbl`. Focused counter collections for both kernels completed all required
replays. They reported 128 architected VGPRs per work-item, 68 B and 28 B of private scratch,
respectively, and only 2.26 and 3.09 active lanes per average VALU instruction despite 64-thread
workgroups. Estimated external bandwidth remained far below nominal HBM peak. The evidence points
to serial face correction and register/private-memory pressure rather than LDS conflicts or peak
memory bandwidth as the current block-advection bottleneck.

These values are attribution and baseline records, not promised performance. Use the commands
above to produce a new `qav/logs/bench/rocm/` archive before quoting results for another revision or
machine.
