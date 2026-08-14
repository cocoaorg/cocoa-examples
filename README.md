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
