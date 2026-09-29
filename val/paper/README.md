# Scientific campaigns

The campaigns in this directory produce the numerical experiments behind the GameDev code paper.
Each one runs the production kernels on a specific scientific problem, such as an analytical
diffusion solution, a convergence study, or a coagulation benchmark, and writes the outputs its
analysis needs. They complement the routine validation suites of [`../README.md`](../README.md): the
suites check each operator against a pass-or-fail criterion, whereas a campaign measures accuracy,
convergence, or agreement over a parameter range and leaves the judgment to the analysis.

## Contents

- [At a glance](#at-a-glance)
- [What the campaigns share](#what-the-campaigns-share)
- [Building and running](#building-and-running)
- [Relation to the validation suites](#relation-to-the-validation-suites)

## At a glance

| Campaign | Representation | Question | Scale |
|---|---|---|---|
| [`diffusion/`](diffusion/README.md) | swarm | Does stochastic diffusion reproduce the analytical log-normal density and concentration solutions at four Stokes numbers? | 8 models |
| [`ecc_orbit/`](ecc_orbit/README.md) | swarm | How does the trajectory integrator converge with timestep on an eccentric Kepler orbit without drag? | 4 models |
| [`eriksson/`](eriksson/README.md) | swarm | How much do fresh runs of monomer coagulation with full swarm physics differ between collision searches and GPU backends, from identical seeds? | 2 models, 8 runs |
| [`smoluchowski/`](smoluchowski/README.md) | swarm | Does the frozen-bath collision chain reproduce the analytical Smoluchowski solutions of the constant, additive, and product kernels? | 75 models, 750 planned runs |
| [`photospheric/`](photospheric/README.md) | swarm and fluid | How do the two representations compare on the same photospheric transport problem with radiation pressure? | 2 models |

The numerical methods under test are described in
[`../../doc/guide_swarm.md`](../../doc/guide_swarm.md),
[`../../doc/guide_fluid.md`](../../doc/guide_fluid.md), and the shared disk model in
[`../../doc/guide_basis.md`](../../doc/guide_basis.md). Project terms are defined in the
[glossary](../../doc/README.md#glossary).

## What the campaigns share

- **Production code.** Every campaign compiles the production runtime and kernels from `src/` and
  `inc/`. A campaign's local `src/` directory holds only its test constants (`const_defs.cuh`) and
  the few overrides its problem needs, such as a special initializer or scoring output; each
  README lists them under "Production code and overrides".
- **Model directories.** Each model or variant is a directory with its own `flags.mk`, which
  usually inherits the campaign `src/` through `MODEL_PARENT` (see
  [Source and header overrides](../../README.md#source-and-header-overrides)).
- **Own build tree.** Each campaign `Makefile` includes the root `Makefile` but redirects models,
  objects, and output into the campaign directory, so campaign builds never mix with production or
  validation builds. The exact output layout differs between campaigns and is given in each
  README.
- **Analysis.** The Smoluchowski campaigns include scoring scripts; the other campaigns describe
  the quantities to compute from their outputs.

## Building and running

Run every command from the repository root. A campaign with one `Makefile` builds one model at a
time, for example

```bash
make -C val/paper/diffusion MODEL=<model> GPU_BACKEND=cuda GPU_TARGET=sm_80
```

The photospheric campaign has one `Makefile` per representation and needs no `MODEL`, and the
Smoluchowski campaign has one `Makefile` per kernel with a `SEED` variable. Each README gives the
complete commands, the executable paths, and the expected output sizes.

`make clean` without `MODEL`, whether run at the repository root or inside any campaign directory,
removes every campaign `obj/` directory together with the validation objects; with `MODEL` it
removes only that model's executable and objects. Campaign `out/` directories, which hold all run
output, are never removed.

## Relation to the validation suites

The campaigns are not part of [`val/run_all.py`](../README.md#running-a-complete-campaign) and do
not enter the validation archive or its record counts. A campaign result shows what that campaign
measured for the source it ran on; it does not qualify the suites, and the suites do not replace a
campaign's convergence or seed studies. Their sources do count toward the validation source
fingerprint, so editing a campaign changes the fingerprint of later validation campaigns. The rules
for qualifying a result are in
[What qualifies a result](../README.md#what-qualifies-a-result).
