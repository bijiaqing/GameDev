# Particle diffusion: density and concentration

Eight models use the production diffusion kernel, coordinate drift, Stokes-dependent
coefficient, boundaries, random-number generator, and two half-steps per runtime step.
Root source is unchanged. `src/` overrides only constants, the initial ring, the fixed
timestep, and deterministic transport (a no-op to isolate diffusion).

| Mode | St=0 | St=0.1 | St=1 | St=10 |
|---|---|---|---|---|
| Density | `ring_pilot` | `density_st0p1` | `density_st1` | `density_st10` |
| Concentration | `concentration_st0` | `concentration_st0p1` | `concentration_st1` | `concentration_st10` |

`CONST_ST` holds St fixed in space. Only concentration models define
`DIFFUSE_CONCENTRATION`. Each model has 2^20 equal-mass particles, initially
ln(R/R0) ~ Normal(0, 0.05²), with zero velocity. GM=R0=1.
The gas surface density is proportional to R^p with p=-1. Alpha=0.04,
h0=0.05, q=0.5 and Schmidt_R=1 give

    D(R) = a R²,   a = 1e-4/(1+St²).

## Analytical reference

Let chi=0 for density diffusion and chi=1 for concentration diffusion. Both obey

    ∂t Σd = (1/R) ∂R [R D (∂R Σd - chi Σd ∂R ln Σg)].

The production radial stochastic drift is (3+chi*p)*a*R; its noise amplitude is
sqrt(2*a)*R. Ito's formula therefore gives a Gaussian log radius:

    u = ln(R/R0)
    mean(u) = (2+chi*p)*a*t
    variance(u) = 0.05² + 2*a*t.

The surface density per total dust mass is

    Σd/Md = exp[-(u-mean)²/(2*variance)] / [2*pi*R²*sqrt(2*pi*variance)].

`analyze.py` compares empirical CDFs, log-radius moments, and area-averaged density
bins with this solution. `--mode` must match the flags used to build the model.
This is an unbounded-domain reference; the simulation reflects at exp(±3).
At the final diffusion age a*t=0.1, boundary tails are tiny, but timestep and
particle-number convergence remain necessary for a publication accuracy claim.
These cases test constant-St suppression and the gas-density-gradient term;
they do not test spatial derivatives of St or three-dimensional diffusion.

## Duration and commands

All models take 1,000 nominal steps and save the initial state plus ten outputs.
DT_MAX=1+St²; DT_OUT=100*(1+St²); final time=1000*(1+St²), in code time units.
This keeps the diffusion age and numerical resolution equal across St.

```sh
make -C val/paper/swarm_diffusion -j8 MODEL=concentration_st1 GPU_BACKEND=rocm GPU_TARGET=gfx942
val/paper/swarm_diffusion/obj/concentration_st1/rocm/gamedev
python3 val/paper/swarm_diffusion/analyze.py val/paper/swarm_diffusion/out/concentration_st1/rocm --mode concentration
```

For CUDA use `GPU_BACKEND=cuda GPU_TARGET=sm_80` and the `cuda` executable/output directory.
Models live in `mod/`, shared overrides in `src/`, results in `out/<model>/<backend>/`,
and build products in `obj/<model>/`. Existing downloaded outputs remain directly
under `out/<model>/`; they are historical results, not runs of this new matrix.

The retained historical `out/ring_pilot/` data predates St-dependent diffusivity.
Analyze it explicitly with `--legacy --mode density`; its saved St=1 did not
affect the old coefficient. Never use that option for the new models.
