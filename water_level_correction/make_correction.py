#!/usr/bin/env python3
"""
Write correction.nc, the water level correction for this example.

Data assimilation produces a correction from observed-minus-modeled water
levels; here a made-up one stands in for it. It is a CF NetCDF field on a
regular longitude/latitude grid (format: cf_netcdf): a +0.25 m raised level
centered on the northern Gulf of Mexico, falling off with a 4-degree Gaussian
radius, with a small grid hole of fill values off Florida to show how
nodes next to missing data are handled. Two snapshots spanning the run hold
the same field, so the correction is steady and the ramp in the
configuration sets when it arrives.

Usage:
    python make_correction.py
"""

from pathlib import Path

import numpy as np
from netCDF4 import Dataset

HERE = Path(__file__).resolve().parent
PEAK_M = 0.25
CENTER = (-89.0, 28.0)  # lon, lat
RADIUS_DEG = 4.0
FILL = -9999.0


def main() -> None:
    lon = np.arange(-100.0, -55.0 + 1e-9, 0.25)
    lat = np.arange(5.0, 47.0 + 1e-9, 0.25)
    gx, gy = np.meshgrid(lon, lat)
    r2 = (gx - CENTER[0]) ** 2 + (gy - CENTER[1]) ** 2
    field = PEAK_M * np.exp(-r2 / (2.0 * RADIUS_DEG**2))
    hole = (np.abs(gx + 83.0) < 0.6) & (np.abs(gy - 25.0) < 0.6)
    field[hole] = FILL

    path = HERE / "correction.nc"
    with Dataset(path, "w") as nc:
        nc.Conventions = "CF-1.6"
        nc.title = "Synthetic water level correction for the cocoa-examples"
        nc.createDimension("time", 2)
        nc.createDimension("lat", lat.size)
        nc.createDimension("lon", lon.size)
        t = nc.createVariable("time", "f8", ("time",))
        t.units = "hours since 2026-01-15 00:00:00"
        t.calendar = "standard"
        t[:] = [0.0, 6 * 24.0]  # spans the six-day run
        x = nc.createVariable("lon", "f8", ("lon",))
        x.units = "degrees_east"
        x[:] = lon
        y = nc.createVariable("lat", "f8", ("lat",))
        y.units = "degrees_north"
        y[:] = lat
        v = nc.createVariable(
            "water_level_correction", "f4", ("time", "lat", "lon"), fill_value=FILL
        )
        v.units = "m"
        v.long_name = "water level correction"
        v[:] = np.stack([field, field]).astype(np.float32)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
