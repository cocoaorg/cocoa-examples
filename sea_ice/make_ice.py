#!/usr/bin/env python3
"""
Write ice.nc, an ice-only CF NetCDF source for the sea_ice example.

It carries one variable, sea-ice concentration (siconc, a fraction from 0 to
1), on a 0.5-degree grid covering the western North Atlantic mesh. An ice
edge 3 degrees wide moves south from 47 N to 39 N over the five days of the
run, so by the end the Gulf of Maine, the Bay of Fundy and the Scotian Shelf
are covered.

Usage:
    python make_ice.py
"""

from pathlib import Path

import numpy as np
from netCDF4 import Dataset

HERE = Path(__file__).resolve().parent
FRONT_START, FRONT_END = 47.0, 39.0  # latitude of the ice edge [deg N]
WIDTH_DEG = 3.0  # concentration rises 0 -> 1 across this band
HOURS = np.arange(0, 5 * 24 + 1, 3)


def main() -> None:
    lon = np.arange(-100.0, -55.0 + 1e-9, 0.5)
    lat = np.arange(5.0, 50.0 + 1e-9, 0.5)
    front = FRONT_START + (FRONT_END - FRONT_START) * HOURS / HOURS[-1]
    conc = np.clip((lat[None, :] - front[:, None]) / WIDTH_DEG, 0.0, 1.0)
    field = np.broadcast_to(conc[:, :, None], (HOURS.size, lat.size, lon.size))

    path = HERE / "ice.nc"
    with Dataset(path, "w") as nc:
        nc.Conventions = "CF-1.6"
        nc.title = "Synthetic southward-moving ice edge for the cocoa-examples"
        nc.createDimension("time", HOURS.size)
        nc.createDimension("lat", lat.size)
        nc.createDimension("lon", lon.size)
        t = nc.createVariable("time", "f8", ("time",))
        t.units = "hours since 2026-02-01 00:00:00"
        t[:] = HOURS
        x = nc.createVariable("lon", "f8", ("lon",))
        x.units = "degrees_east"
        x[:] = lon
        y = nc.createVariable("lat", "f8", ("lat",))
        y.units = "degrees_north"
        y[:] = lat
        v = nc.createVariable("siconc", "f4", ("time", "lat", "lon"))
        v.units = "1"
        v.long_name = "sea_ice_area_fraction"
        v[:] = field.astype(np.float32)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
