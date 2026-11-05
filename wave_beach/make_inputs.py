#!/usr/bin/env python3
"""
Write the inputs for the wave_beach example:

    beach.nc   a 20 km x 4 km beach whose bed rises from 15 m deep at the open
               western boundary to 2 m above still water at the eastern shore,
               in triangles (the generator is ../tidal_beach/make_meshes.py)
    wind.nc    CF NetCDF wind and pressure: a steady 20 m/s wind blowing
               onshore (toward the east) at 1013 hPa, hourly for one day

Usage:
    python make_inputs.py
"""

import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tidal_beach"))
from make_meshes import Beach, write_beach_mesh  # noqa: E402

BEACH = Beach(
    length_m=20000.0,
    width_m=4000.0,
    cell_m=200.0,
    depth_open_m=15.0,
    depth_land_m=-2.0,
)
WIND_SPEED_MS = 20.0
HOURS = np.arange(25)


def write_wind(path: Path) -> None:
    lon = np.arange(-90.10, -89.70 + 1e-9, 0.05)
    lat = np.arange(28.95, 29.10 + 1e-9, 0.05)
    shape = (HOURS.size, lat.size, lon.size)
    with Dataset(path, "w") as nc:
        nc.Conventions = "CF-1.6"
        nc.title = "Steady onshore wind for the wave_beach example"
        nc.createDimension("time", HOURS.size)
        nc.createDimension("lat", lat.size)
        nc.createDimension("lon", lon.size)
        t = nc.createVariable("time", "f8", ("time",))
        t.units = "hours since 2026-04-01 00:00:00"
        t[:] = HOURS
        x = nc.createVariable("lon", "f8", ("lon",))
        x.units = "degrees_east"
        x[:] = lon
        y = nc.createVariable("lat", "f8", ("lat",))
        y.units = "degrees_north"
        y[:] = lat
        for name, units, value in (
            ("wind_u", "m/s", WIND_SPEED_MS),
            ("wind_v", "m/s", 0.0),
            ("mslp", "mb", 1013.0),
        ):
            v = nc.createVariable(name, "f4", ("time", "lat", "lon"))
            v.units = units
            v[:] = np.full(shape, value, dtype=np.float32)


def main() -> None:
    write_beach_mesh(HERE / "beach.nc", BEACH, "triangle")
    print(f"wrote {HERE / 'beach.nc'}")
    write_wind(HERE / "wind.nc")
    print(f"wrote {HERE / 'wind.nc'}")


if __name__ == "__main__":
    main()
