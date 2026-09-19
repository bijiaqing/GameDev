# Supplied MCDUST setup audit

Compared current `common/` and weak/strong model flags with
`/Users/jiaqingbi/Scratch/gamedev/mcdust/setups/alpha_1e-{3,4}_{36,72}cores/`
and the supplied `src/`. No simulation implementation was changed by this audit.
The current source is not proven to be the exact binary used for every downloaded snapshot.

## Matched settings

Both use alpha=1e-3/1e-4, 1,048,576 representatives, 0.5-micron-radius monomers,
material density 1 g/cm³, initial dust/gas ratio 0.01, fragmentation threshold
100 cm/s, surface-density normalization 1410.4014065096128 g/cm², temperature
normalization 209.7926358245702 K, and radial exponents -1 and -1/2.
Strong/weak final snapshot times are 12500/25000 years. MCDUST now writes every
100 years; files numbered 25 apart in the download represent 2500-year gaps,
not a different underlying output cadence.

For each alpha, the 36/72-core setup files differ only in the output directory;
their preprocessor flags are identical. The names do not themselves set the
OpenMP thread count. `INITSIZEDIST`, `NELSON2013`, and `LOGTIME` occur in the
setup flags but have no references in the supplied Fortran source. Therefore
those names do not establish an initial size spectrum, Nelson gas profile, or
logarithmic output schedule. Initial snapshots independently confirm monomers.

## Remaining physical and numerical differences

**Subsequent update:** GameDev now uses nu/(1+St²), including spatial
derivatives in both the diffusion drift and its timestep constraint. The audit
below describes the code before that change. MCDUST's noise-normalization and
missing radial drift terms remain different.

### 1. Gas profile and constants

MCDUST `src/discstruct.F90`, `z_exp`/`densg`, uses a Gaussian vertical profile,
with exp(-z²/(2H²)) floored at 0.01. GameDev uses
exp[(R/sqrt(R²+Z²)-1)/h²] without that floor. The latter is the spherical
hydrostatic profile, not the Gaussian approximation.

MCDUST's molecular mass is 2.3/N_A g; GameDev uses 2.34 times the proton mass.
Together with their constants, this gives MCDUST h(1 au)=0.02923445424 versus
GameDev 0.02888219943. MCDUST c_s² is about 2.48% larger at the same temperature.
G, solar mass and the definition of a year also differ slightly.
The MCDUST density floor is inconsistent with `ddensgdz`, which continues to
return -z*rho_g/H² in the floored region. Do not copy that derivative as though
it were the derivative of the floored profile.

### 2. Diffusion equation actually implemented

GameDev now uses concentration diffusion, with D=nu/Sc (Sc=1), Gaussian noise
variance 2D dt, and cylindrical drift dD/dR + D/R + D*dlnrho_g/dR; vertical
drift is D*dlnrho_g/dZ.

MCDUST `src/advection.F90`, `vel_rad`/`vel_ver`, instead uses
D_p=nu/(1+St²). Its Box–Muller draw is additionally divided by sqrt(2 ln 2),
so its actual displacement variance is 2D_p dt/(2 ln 2). Thus the effective
noise diffusivity is 0.7213475 D_p, while the density-gradient drift uses D_p.
It also does not explicitly include dD_p/dR or D_p/R in the radial drift.
These differences persist even for St << 1. Matching the phrase
"dust-to-gas diffusion" has not made the two implemented stochastic equations
identical. The extra noise factor is an implementation issue to assess, not an
assumption to import automatically.

### 3. Spatial support, boundaries and normalization

MCDUST initializes uniformly in cylindrical R between 5 and 50 au, with an
untruncated Gaussian Z/H. Its grid follows the particle extent. Advection can
cross the initial 5–50 au interval; particles stop advecting inside 0.99 au.
There is no corresponding fixed polar boundary in `mc_advection`.

GameDev initializes its exact hydrostatic distribution within spherical
r=5–50 au and theta=pi/2 ±0.2. Deterministic transport is absorbing at domain
exits; diffusion reflects there. Its dust mass is normalized to this finite
domain, whereas MCDUST normalizes with the full-column surface-density integral.
About 0.61–0.64% of the three available MCDUST initial populations already lie
outside GameDev's spherical/polar domain. This difference can grow during evolution.

### 4. Dynamics and stopping time

MCDUST uses prescribed radial drift and vertical settling velocities, the latter
-Z*Omega*St/(1+St²). GameDev integrates inertial motion and drag with SSA.
MCDUST supports a Stokes-drag branch above grain radius 2.25 mean free paths;
the GameDev analytic-gas configuration uses Epstein scaling only. This matters
only if grains reach that transition. GameDev initializes terminal drift/settling
velocities; MCDUST initializes stored velocities to zero and recomputes them
for advection. These are distinct from collision-relative settling, which uses
min(St,0.5) in both current implementations.

### 5. Collision environment, neighborhoods and grouping

MCDUST evaluates gas and relative-speed coefficients at adaptive cell centers;
GameDev uses the query particle's environment. MCDUST uses 256 representatives
per adaptive cell (128 radial ×32 vertical cells initially), while GameDev's
N_K=256 selects nearest neighbors for each query. These are different density
estimators and interaction neighborhoods, despite matching the number 256.

The main relative-speed ingredients now agree: Brownian motion, radial and
azimuthal differential drift, capped collision settling and Ormel–Cuzzi regimes.
Gas profiles, stopping-time laws and coefficient locations still differ.

Both implement high-speed erosion at target/projectile mass ratio >=10 and
fragmentation capped by the old target mass. The fragmentation draw
m_new=[m0^(1/6)+U*(m_old^(1/6)-m0^(1/6))]^6 is equivalent to GameDev's squared
interpolation in square-root diameter.

MCDUST uses dmmax=1e-3 and crossing-time-dependent grouped counts. GameDev keeps
narrow grouping (projectile/target <=1e-6, packet mass increment <=1e-4) and
separate remnant/debris rates for erosion. MCDUST's erosion applies the grouped
mass loss but tests the debris probability with the single-projectile ratio.
MCDUST refreshes affected matrix rows/columns after events; GameDev uses the
validated local frozen-neighbor scheduling. These deliberately remain different.
MCDUST additionally suppresses rates involving certain represented counts <=1.

## Downloaded-output caveats

`src/hdf5output.F90:swarm_unit_conversion` copies ID, mass, R, Z and St, but does
not initialize npar or either velocity in the temporary output array. The files
examined contain zeros for these omitted fields. Do not interpret those zeros
as physical values or use npar to mass-weight the comparison. Use the equal
represented mass `mass_of_swarm[g]`, or reconstruct npar=mswarm/m where needed.

Attributes labelled `time_between_outputs[yr]`, `maximum_time_of_simulation[yr]`
and `evaporation_radius_[AU]` are written from internal seconds/cm variables.
Use `/times/timesout` for snapshot time: the checked strong output 125 is 12500
and weak output 250 is 25000. Strong snapshot attributes nevertheless report
an internal tend corresponding to 25000 years; that disagrees with the supplied
strong setup and is a provenance/metadata inconsistency, not evidence to override
the snapshot time.

## Priority

The largest remaining setup issues are the diffusion prescription, gas vertical
profile, and spatial/boundary support. Align physical constants if an especially
close quantitative comparison is intended. Keep search/scheduling differences
explicit as numerical-method differences. Do not import MCDUST's noise factor,
inconsistent floor derivative, or uninitialized output fields merely to obtain
closer curves.
