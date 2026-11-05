#!/usr/bin/env python3
"""
Write baroclinic_forcing.nc, the nodal baroclinic forcing for this example.

A real run builds the file from HYCOM temperature and salinity with
cocoa-ocean3d (`cocoa-ocean3d all --start ... --mesh ../wnat/wnat.nc ...`),
which needs network access and a few gigabytes of downloads. This script
runs the same compute, regrid and write steps on an analytic ocean instead,
so the example runs offline:

* temperature falls from a mixed layer 30 m deep through an exponential
  thermocline to about 2 C at depth, the same everywhere;
* a warm-core eddy in the Gulf of Mexico, like those the Loop Current
  sheds: up to 4 C warmer at its center, 150 km in radius, its anomaly
  fading over the top 300 m. Its density gradient is the baroclinic
  pressure gradient, and the sea surface rises over the eddy until its
  slope balances it, much as the sea surface stands high over a real one;
* salinity is a constant 35 PSU.

The anomaly is local because the forcing file carries the gradient as a
depth-averaged acceleration: a gradient spanning the whole basin would push
its geostrophic flow straight through the open boundary.

The file holds one snapshot, which Cocoa holds for the whole run, and carries
the mesh id of ../wnat/wnat.nc, which Cocoa checks at startup.

Requires the cocoa-tools package (pip install <cocoa>/utils/cocoa_tools).

Usage:
    python make_forcing.py
"""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from cocoa_ocean3d.compute import compute_snapshot
from cocoa_ocean3d.mesh import compute_mesh_fingerprint, read_mesh_nodes
from cocoa_ocean3d.pipeline import SnapshotRegridder
from cocoa_ocean3d.writer import NodalForcingWriter

HERE = Path(__file__).resolve().parent
MESH = HERE.parent / "wnat" / "wnat.nc"
OUTPUT = HERE / "baroclinic_forcing.nc"
SNAPSHOT_TIME = datetime(2026, 2, 1, tzinfo=timezone.utc)

# Source grid: the mesh footprint (lon -97.85..-60.04, lat 7.99..45.75) with
# margin. The regridder needs every node strictly inside a regional grid.
LON = np.arange(-101.0, -57.0 + 1e-9, 0.5)
LAT = np.arange(6.0, 47.0 + 1e-9, 0.5)
DEPTH = np.array(
    [0, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000,
     3000, 4000, 5000, 6000, 7000, 8000],
    dtype=float,
)
MIXED_LAYER_M = 30.0
SURFACE_TEMP_C = 26.0
EDDY_CENTER = (-88.0, 26.0)  # lon, lat
EDDY_RADIUS_KM = 150.0
EDDY_ANOMALY_C = 4.0  # at the surface, at the center
EDDY_DEPTH_M = 300.0  # e-folding depth of the anomaly


def temperature_profile(surface_temp: float) -> np.ndarray:
    below_ml = np.maximum(DEPTH - MIXED_LAYER_M, 0.0)
    thermocline = 22.0 * np.exp(-below_ml / 700.0)
    abyssal = 2.5 * np.exp(-DEPTH / 6000.0)
    return surface_temp - 22.0 - 2.5 + thermocline + abyssal


def main() -> None:
    gx, gy = np.meshgrid(LON, LAT)
    dx = np.radians(gx - EDDY_CENTER[0]) * 6371.0 * np.cos(np.radians(EDDY_CENTER[1]))
    dy = np.radians(gy - EDDY_CENTER[1]) * 6371.0
    shape = np.exp(-(dx**2 + dy**2) / (2.0 * EDDY_RADIUS_KM**2))
    anomaly = EDDY_ANOMALY_C * np.exp(-DEPTH / EDDY_DEPTH_M)
    temperature = (
        temperature_profile(SURFACE_TEMP_C)[:, None, None]
        + anomaly[:, None, None] * shape[None, :, :]
    )
    salinity = np.full_like(temperature, 35.0)

    snapshot = compute_snapshot(salinity, temperature, DEPTH, LON, LAT)
    mesh = read_mesh_nodes(str(MESH))
    products = SnapshotRegridder(mesh).regrid(snapshot)
    bpg = np.hypot(products.bpg_x, products.bpg_y)
    print(f"|BPG| max {bpg.max():.3e}, mean {bpg.mean():.3e} m/s^2")

    OUTPUT.unlink(missing_ok=True)
    with NodalForcingWriter(
        str(OUTPUT),
        num_nodes=mesh.num_nodes,
        mesh_id=compute_mesh_fingerprint(str(MESH)),
        source_dataset="analytic stratification (cocoa-examples)",
        generation_command="baroclinic/make_forcing.py",
    ) as writer:
        writer.append_snapshot(SNAPSHOT_TIME, products)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
