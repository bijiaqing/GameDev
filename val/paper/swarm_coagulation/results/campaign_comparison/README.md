# GameDev–MCDUST–cuDisc campaign comparison

GameDev and MCDUST downloaded times match to 0.01 yr or better. Strong turbulence: output 125, 12,500 yr; weak turbulence: output 250, 25,000 yr. These particle runs contain 1,048,576 representatives. Both start from grain radius 0.5 micrometres.

## Performance

Preserved snapshot modification times were confirmed by the user. Elapsed spans run from the initial snapshot to the final snapshot. They exclude initialization and may include filesystem delays or pauses. No profiler, scheduler, or hardware records accompany these downloads; backend timing differences are not isolated hardware-efficiency measurements.

| Configuration | Strong: hours | Weak: hours |
|---|---:|---:|
| CUDA KD-tree | 14.102 | 14.730 |
| CUDA Morton | 13.300 | 13.732 |
| ROCm KD-tree | 12.702 | 14.429 |
| ROCm Morton | 10.925 | 12.648 |
| MCDUST 36 cores | 6.141 | 8.414 |
| MCDUST 72 cores | 4.870 | 6.071 |

ROCm Morton is fastest among these GameDev runs. Relative to KD-tree, Morton reduces elapsed time by:
- strong, CUDA: 5.7%.
- strong, ROCm: 14.0%.
- weak, CUDA: 6.8%.
- weak, ROCm: 12.3%.
- strong: fastest GameDev takes 2.24 times the 72-core MCDUST elapsed time.
- weak: fastest GameDev takes 2.08 times the 72-core MCDUST elapsed time.

## Numerical agreement

Size CDF differences below are the largest absolute differences between normalized mass-weighted cumulative size distributions, measured on 320 logarithmic size bins. They are distribution discrepancies, not analytical errors or statistical pass thresholds.

| Region | Strong: GameDev mean radius [um] | Strong: MCDUST 72 mean [um] | Weak: GameDev mean [um] | Weak: MCDUST 72 mean [um] |
|---|---:|---:|---:|---:|
| inner | 424.29–425.30 | 483.39 | 3909.97–3922.26 | 4382.01 |
| outer | 5.39–5.46 | 10.36 | 14.42–14.87 | 30.50 |
- strong: GameDev vs CUDA KD-tree, maximum CDF difference 0.0023; GameDev outer disc vs MCDUST72: 0.2963–0.2976; MCDUST36 vs MCDUST72 outer disc: 0.0057.
- weak: GameDev vs CUDA KD-tree, maximum CDF difference 0.0033; GameDev outer disc vs MCDUST72: 0.3213–0.3229; MCDUST36 vs MCDUST72 outer disc: 0.0066.

The code-to-code outer-disc difference greatly exceeds backend/search variation and the MCDUST core-count variation. GameDev grows smaller grains there. Inner-disc mean sizes are about 10–12% smaller, while the outer-disc means are about a factor of two smaller. The outer vertical density profiles are much closer than the size distributions: GameDev RMS angles are about 6–7% larger. These observations point to a systematic inter-code difference; they do not identify a particular physical prescription or algorithm as its cause.

## Figure conventions and limits

The three columns follow Eriksson et al. Figures 14/16: vertical column density versus spherical radius, then radial integrals in r<15 au and r>25 au versus angle from the midplane. The colour limits are 1e-4–1e-1 g/cm2 and 1e-5–1 g/cm2 per published size bin. Grey curves are mass-weighted mean radii in each integration measure. No smoothing is applied.

The paper snapshots are 11,133 and 23,809 yr, whereas these reproductions use the available matched final times, 12,500 and 25,000 yr. No snapshots are interpolated. Source: https://arxiv.org/html/2603.22550v2#S5.SS2

All comparisons use the common spherical domain 5–50 au and |theta-pi/2|<=0.2. MCDUST uses cylindrical coordinates and has particles outside this selection even initially; differences in selected total mass are not automatically physical mass loss. GameDev particle sizes are diameters and are divided by two. Its swarm weights are particle multiplicity times physical grain mass. MCDUST weights use mass_of_swarm[g], because its stored number-of-particles-in-swarm field is zero in the inspected initial output.

## Files

- performance.png/.pdf: cumulative elapsed time and interval cost.
- agreement.png/.pdf: regional size distributions and outer vertical profiles.
- figure14_strong.png/.pdf and figure14_weak.png/.pdf: all seven configurations, one triplet per row.
- comparison.json: per-snapshot timing, mass, moments, quantiles, final regional CDF differences.
- distributions.npz: final regional mass histograms.

Reproduce with NumPy, Matplotlib and h5py installed:

```sh
python3 -B val/paper/swarm_coagulation/compare_campaign.py --root /Users/jiaqingbi/Scratch/gamedev
```

## cuDisc addition

cuDisc is included in agreement.png, both Figure-14-layout maps, comparison.json and distributions.npz. Strong uses dens_125.dat at 11,133.673 yr; weak uses dens_145.dat at 23,809.035 yr. GameDev/MCDUST remain at 12,500 and 25,000 yr. These are the available snapshots, without time interpolation. No cuDisc runtime is inferred: only one snapshot per model is available, and its download timestamp does not measure simulation runtime.

| Regime / region | GameDev mean radius [um] | MCDUST 72 mean [um] | cuDisc mean [um] |
|---|---:|---:|---:|
| Strong / inner | 424.29–425.30 | 483.39 | 344.77 |
| Strong / outer | 5.39–5.46 | 10.36 | 4.75 |
| Weak / inner | 3909.97–3922.26 | 4382.01 | 3123.72 |
| Weak / outer | 14.42–14.87 | 30.50 | 10.12 |

cuDisc and GameDev have much closer outer-disc cumulative size distributions than either has with MCDUST. Against CUDA KD-tree, cuDisc outer CDF differences are 0.0382 (strong) and 0.0185 (weak); against MCDUST72 they are 0.3191 and 0.3119. The weak outer mean still differs substantially despite its small CDF difference, because the mean is sensitive to the relatively low-mass large-grain tail. All three outer vertical mass profiles are close. Inner-disc cuDisc distributions are broader and have smaller mean sizes. These results do not establish which code is more accurate.

The supplied cuDisc C++ setups initialize an MRN distribution from 0.5 to 1 um; the current particle runs initialize 0.5 um monomers. This setup difference and the snapshot-time difference remain relevant when interpreting agreement.

### Grid handling and checks

The binary reader follows the upstream [cuDisc reader](https://github.com/cuDisc/cuDisc/blob/main/codes/python/fileIO.py) and [grid writer](https://github.com/cuDisc/cuDisc/blob/main/src/grid.cu). The supplied setup uses 200 radial by 100 angular cells, two ghost cells per boundary, and 127 size bins. Ghost cells are excluded. Density is integrated as piecewise constant in cylindrical-radius/angular cells into the same spherical domain and bins used for the particle data. Both hemispheres and azimuth are included in mass. Radial density integrals average the two symmetric hemispheres.

Four-point angular Gauss quadrature was checked against eight points: regional mass and mean differences are below 3e-9 relative, and map integrated absolute differences below 1.3e-8 relative. The loader asserts stored-cell volume consistency, finite nonnegative densities, conservation of mapped mass, and conservative size rebinning. This checks post-processing, not the underlying simulation accuracy.

The cuDisc grains.sizes file contains 128 **edges**, not centers, as confirmed by the upstream writer. This campaign's maps have been regenerated with those 127 native size bins for all codes. Earlier campaign maps incorrectly interpreted the entries as centers and shifted bin boundaries by half a bin. The independently binned GameDev/MCDUST agreement statistics were unaffected; older plots outside this campaign directory have not been regenerated.

For the agreement curves/CDFs, cuDisc bin masses are conservatively rebinned onto the existing 320-bin size grid, assuming constant mass per logarithmic size within each native bin. Means use cuDisc's mass-midpoint representative grain radius. Neither operation adds physical resolution.
