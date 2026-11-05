# Cocoa Examples

Runnable configurations for the [Cocoa](https://github.com/zcobell/cocoa) coastal
ocean model. Each directory holds a mesh, one or more configuration files, and a
plotting script that produces the summary figure shown alongside it.

These are meant to be read as much as run: the configurations are commented, and
several directories carry more than one config so you can diff two files to see
exactly what a feature changes.

## Running an example

Run from inside the example's directory, since every path in a configuration is
relative to the working directory:

```bash
cd flow_channel
cocoa -i cocoa_config.yaml
python plot_results.py
```

On a GPU build, add `--kokkos-num-threads=1` so the host backend does not
oversubscribe. For MPI:

```bash
mpirun -np 4 cocoa -i cocoa_config.yaml --partition partition.nc
```

## The examples

| Directory | Demonstrates | Runtime |
|---|---|---|
| `flow_channel` | Specified-flow (type 22) inflow boundary against a tidal outlet | ~40 s |
| `flow_channel_radiation` | The same channel with a radiation outflow boundary | ~40 s |
| `internal_weir` | Internal weir (type 24) barrier boundary with overtopping | ~40 s |
| `global_case` | Global mesh, rotated pole, implicit solver, tidal potential + self-attraction loading | ~1 min |
| `global_sh_australia` | GAHM parametric cyclone on a rotated Southern Hemisphere mesh | ~1 min |
| `global_two_vortex` | Two simultaneous GAHM storms, to exercise multi-storm composition | ~1 min |
| `wnat` | Western North Atlantic with meteorological forcing, in three input formats | ~3 min |
| `tidal_beach` | Tidal wetting and drying on triangle, quadrilateral and hybrid meshes; station output; `include:` | ~20 s each |
| `wave_beach` | Cocoa's spectral wave model (KOMEN and ST6) coupled to the circulation; wave setup | ~5 s each |
| `katrina_waves` | Hurricane Katrina (2005): GAHM vortex driving the coupled wave model; station output | ~15 s |
| `water_level_correction` | A gridded data-assimilation correction, both methods, against a baseline | ~6 s each |
| `sea_ice` | Sea-ice drag attenuation from an ice-only source composed with `domains:` | ~6 s each |

Runtimes are single-threaded on a laptop CPU build, measured or extrapolated from a
short run; they are for picking an example to try, not a benchmark.

### flow_channel

A rectangular channel, 10 m deep, with a steady 10,000 m^3/s inflow on the
upstream boundary and a zero-amplitude tidal boundary at the outlet. Water level
rises through the 2-day ramp and settles into a steady gradient.

Three configurations share the mesh:

- `cocoa_config.yaml` -- the base case, default lumped/explicit GWCE.
- `cocoa_config_implicit.yaml` -- the same case on the consistent (implicit)
  GWCE. Diff it against the base to see the `numeric` block in isolation; it is
  the only example that sets `linear_solver`, since the explicit solver never
  performs a linear solve.
- `cocoa_config_restart_leg1.yaml` / `_leg2.yaml` -- the same 3 days split in
  two, to demonstrate checkpoint and restart:

  ```bash
  cocoa -i cocoa_config_restart_leg1.yaml   # runs to day 2, writes a checkpoint
  cocoa -i cocoa_config_restart_leg2.yaml   # resumes from it, runs to day 4
  ```

  Leg 1 writes `cocoa_checkpoint.20260102T000000.nc` -- the prefix from the
  config plus the checkpoint time. Leg 2 names that file in
  `checkpoint.restart_file`. A checkpoint carries full model state, not just
  elevation, so the two legs together reproduce the single 3-day run.

### flow_channel_radiation

The same channel geometry with the outlet switched to a radiation boundary, so
the outgoing wave leaves the domain instead of reflecting.

### internal_weir

A domain split by an internal weir (type 24). The tidal boundary drives water
over the barrier once the level exceeds the crest height, exercising the
overtopping formula and the paired-node bookkeeping either side of the weir.

### global_case

A global mesh run with the rotated-pole option, which moves the coordinate
singularity off the computational domain. Uses the implicit solver with a 12
minute timestep, tidal potential, and self-attraction/loading.

`cocoa_config_output_control.yaml` runs the same case while restricting output
to elevation and velocity and enabling compression -- the settings that matter
when a month of global output would otherwise be unwieldy. Diff it against
`cocoa_config.yaml` to see the `output` block on its own.

### global_sh_australia and global_two_vortex

Parametric tropical cyclone forcing with the GAHM vortex model, driven from an
ATCF best-track file rather than gridded meteorology.

`global_sh_australia` places a Southern Hemisphere cyclone on the rotated global
mesh; the track is synthesised by `make_australia_track.py`. `global_two_vortex`
runs the real Atlantic Katrina track alongside the same track relocated into the
Pacific, which is what exercises multi-storm composition -- the `tracks:` key
takes a list, and the storms are blended where their fields overlap.

Both reuse the mesh from `global_case/` via a relative path, so run them from
their own directory:

```bash
cd global_two_vortex
cocoa -i cocoa_config.yaml
```

### wnat

The western North Atlantic domain with meteorological forcing. Three
configurations run the identical simulation from three input formats:

- `cocoa_config_cf.yaml` -- CF-compliant NetCDF
- `cocoa_config_owi_netcdf.yaml` -- OWI NetCDF
- `cocoa_config_owi_ascii.yaml` -- OWI ASCII (the legacy `.pre` / `.win` pair)

They are the reference for what each format expects, and running more than one
shows that the choice of format does not change the answer.

### tidal_beach

A basin 8 km long whose bed rises from 10 m deep at the open boundary to 3 m
above still water at the far end, driven by a 1 m M2 tide, so the shoreline
moves about a kilometer up and down the beach twice a day. `make_meshes.py`
writes the same nodes three ways: all triangles, all quadrilaterals, and a
hybrid with quadrilaterals in the middle third. Cocoa reads all three
directly, and the three configurations differ only in the mesh and output
names:

```bash
cd tidal_beach
python make_meshes.py
cocoa -i cocoa_config_triangle.yaml
cocoa -i cocoa_config_quad.yaml
cocoa -i cocoa_config_hybrid.yaml
python plot_results.py
```

The tide boundary and the station list live in their own files,
`tide_boundary.yaml` and `stations.yaml`, pulled in with `include:`; it is
how a long per-node boundary block or station list is kept out of the main
configuration. Stations are written to `<mesh>_stations.nc` every 5 minutes
and report the fill value while their element is dry, which is why the
shallowest one's series is broken at low water.

### wave_beach

A steady 20 m/s onshore wind blows for 12 hours over a 20 km beach that
shoals from 15 m to dry land. Cocoa's spectral wave model runs on the
circulation mesh, grows the waves over the shelf, breaks them near the shore,
and hands the radiation-stress force back to the circulation.

- `cocoa_config.yaml` -- the wave model with every key at its default:
  SWAN's KOMEN source terms, 36 directions by 41 frequencies, coupled every
  10 minutes. The `wave` block needs nothing but `enabled: true`; a ramp is
  set only because the wave force's own ramp defaults to a day.
- `cocoa_config_st6.yaml` -- the same with `sources.physics: st6`.
- `cocoa_config_wind_only.yaml` -- no waves, so the difference in water level
  is the wave setdown offshore and setup at the shoreline.

```bash
cd wave_beach
python make_inputs.py      # beach.nc and the CF wind file wind.nc
cocoa -i cocoa_config.yaml
cocoa -i cocoa_config_st6.yaml
cocoa -i cocoa_config_wind_only.yaml
python plot_results.py
```

The mesh comes from `../tidal_beach/make_meshes.py`, which `make_inputs.py`
imports. Wave fields (`wave_hs`, `wave_dir`, `wave_tm01`, and the force on the
circulation, `wave_force`) are written to both the global and station files.

### katrina_waves

A hindcast of Hurricane Katrina (2005) on the WNAT mesh, from 26 August
12:00 to 30 August. The GAHM vortex, built from the NHC best track in
`katrina_bal122005.dat`, is the only forcing: `forcing.meteorological` has a
`vortex` block and no file. It drives the circulation and the spectral wave
model together, and the wave force feeds back into the water level. The wave
stop rule uses the SWAN+ADCIRC hurricane setting (`pass_fraction: 0.95`), and
the coupling interval is 20 minutes.

Station output at NDBC buoys and NOAA tide gauges (`stations.yaml`) carries
the water level, current, wind, pressure, and wave parameters every 10
minutes in `cocoa_station_output.nc`. The WNAT mesh is coarse along the coast,
so the water levels at the gauges illustrate the coupling rather than
reproduce the observed surge. The deep-water wave heights are closer: the
modeled peak at NDBC 42040 is 16 m.

### water_level_correction

The tide-driven WNAT case with a water level correction, the field data
assimilation produces to pull a model toward observed water levels. A
synthetic correction stands in for an assimilated one: `make_correction.py`
writes a CF longitude/latitude grid raising the northern Gulf of Mexico by
up to 0.25 m, with a small patch of fill values off Florida to show how nodes
next to missing data are treated. The correction's ramp holds it at zero for
two days while the tide spins up, then brings it in over one.

- `cocoa_config.yaml` -- `method: pressure_and_mass_adjust` (the default):
  the water level follows the correction at once, at the cost of mass
  conservation.
- `cocoa_config_pressure.yaml` -- `method: pressure`: mass is conserved, and
  the water reaches the raised level by flowing there, so it overshoots and
  oscillates about it.
- `cocoa_config_baseline.yaml` -- no correction, the reference for the other
  two.

```bash
cd water_level_correction
python make_correction.py
cocoa -i cocoa_config.yaml
cocoa -i cocoa_config_pressure.yaml
cocoa -i cocoa_config_baseline.yaml
python plot_results.py
```

The tide boundary is `include:`d from `tide_boundary.yaml`, the same
55-node M2 forcing as `../wnat`.

### sea_ice

The WNAT mesh under the CF wind and pressure from `../wnat`, with sea ice
from a second source: `make_ice.py` writes an ice-only CF file whose ice edge
moves south from 47 N to 39 N over five days. Ice changes the wind's drag on
the water (Lupkes et al.'s form drag), so the wind setup under the ice
differs from open water.

Two sources means the `domains:` list in place of the flat
`format`/`filename` form; each entry is self-contained, and between them they
supply wind, pressure and ice. Ice is opt-in (`ice: true`), and an ice-only
source must name its ice variable. `cocoa_config_no_ice.yaml` is the same
run without the ice, and the figure shows the difference: a few centimeters
of setdown in the Gulf of Maine once the ice arrives.

```bash
cd sea_ice
python make_ice.py
cocoa -i cocoa_config.yaml
cocoa -i cocoa_config_no_ice.yaml
python plot_results.py
```

## About the configuration files

The `physics` block in each example lists the wet/dry, friction and wind
parameters explicitly, **set to the model's own defaults**. They are spelled out
so the available knobs are visible in context; leaving any of them out changes
nothing. Values that are genuinely case-specific -- `manning_n`, `tau0`,
`cf_lower_limit` -- appear at the top of the block.

Cocoa's configuration schema is validated on load, and unknown keys are an
error rather than being ignored, so a typo fails immediately with the offending
key named.

## Input data

Meshes are committed directly; they are small. The meteorological data under
`wnat/example_met_data/` is the exception at roughly 68 MB.

New examples needing generated input should commit a Python generator script
rather than the resulting NetCDF, matching the convention Cocoa uses for its own
test fixtures.
