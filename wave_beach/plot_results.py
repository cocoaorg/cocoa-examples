#!/usr/bin/env python3
"""
Plot the wave_beach example: the wave field, how it grows and breaks along
the beach under the KOMEN and ST6 packages, and the setup it adds to the
wind setup.

Run the three configurations first:
    python make_inputs.py
    cocoa -i cocoa_config.yaml
    cocoa -i cocoa_config_st6.yaml
    cocoa -i cocoa_config_wind_only.yaml
    python plot_results.py
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from netCDF4 import Dataset

HERE = Path(__file__).absolute().parent
EARTH_RADIUS_M = 6378206.4
LON0, LAT0 = -90.0, 29.0
RUNS = {
    "komen": ("KOMEN (default)", "tab:blue"),
    "st6": ("ST6", "tab:red"),
    "wind_only": ("no waves", "0.4"),
}
STATIONS = ("km8", "km14", "km18")


def to_km(lon, lat):
    x = np.radians(lon - LON0) * EARTH_RADIUS_M * math.cos(math.radians(LAT0))
    y = np.radians(lat - LAT0) * EARTH_RADIUS_M
    return x / 1000.0, y / 1000.0


def load(name):
    with Dataset(HERE / f"{name}_output.nc") as ds:
        x, y = to_km(ds["lon"][:], ds["lat"][:])
        out = {
            "x": x,
            "y": y,
            "depth": np.asarray(ds["depth"][:]),
            "elements": np.asarray(ds["element"][:]) - int(ds["element"].start_index),
            "zeta": np.ma.filled(ds["zeta"][-1], np.nan),
        }
        for key in ("wave_hs", "wave_dir"):
            if key in ds.variables:
                out[key] = np.ma.filled(ds[key][-1], np.nan)
    centre = np.isclose(out["y"], 2.0, atol=0.05)
    out["transect"] = np.flatnonzero(centre)[np.argsort(out["x"][centre])]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-o", "--output", default=HERE / "wave_beach_results.png")
    args = parser.parse_args()
    runs = {name: load(name) for name in RUNS}
    komen = runs["komen"]

    fig = plt.figure(figsize=(13, 11), layout="constrained")
    grid = fig.add_gridspec(4, 3, height_ratios=[1.0, 1.0, 1.0, 1.0])

    ax = fig.add_subplot(grid[0, :])
    tri = mtri.Triangulation(komen["x"], komen["y"], komen["elements"])
    tri.set_mask(np.isnan(komen["wave_hs"][komen["elements"]]).any(axis=1))
    tc = ax.tripcolor(tri, np.nan_to_num(komen["wave_hs"]), shading="gouraud",
                      cmap="viridis")
    step = slice(None, None, 37)
    # wave_dir is where the waves come from, nautical (clockwise from north).
    to = np.radians(komen["wave_dir"][step] + 180.0)
    ax.quiver(komen["x"][step], komen["y"][step], np.sin(to), np.cos(to),
              color="w", scale=40, width=0.002)
    fig.colorbar(tc, ax=ax, label="Hs (m)", shrink=0.9)
    ax.set_aspect("equal")
    ax.set_xlabel("distance from open boundary (km)")
    ax.set_ylabel("km")
    ax.set_title("Significant wave height and mean direction at 12:00 (KOMEN)",
                 fontsize=10)

    ax = fig.add_subplot(grid[1, :])
    for name in ("komen", "st6"):
        r = runs[name]
        t = r["transect"]
        ax.plot(r["x"][t], r["wave_hs"][t], color=RUNS[name][1], label=RUNS[name][0])
    bed = ax.twinx()
    t = komen["transect"]
    bed.fill_between(komen["x"][t], -komen["depth"][t], -16, color="tan", alpha=0.4)
    bed.set_ylim(-16, 6)
    bed.set_ylabel("bed elevation (m)", color="saddlebrown")
    ax.set_ylabel("Hs (m)")
    ax.set_title("Hs along the centerline: growth over the shelf, breaking near "
                 "the shore", fontsize=10)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_zorder(bed.get_zorder() + 1)
    ax.patch.set_visible(False)

    ax = fig.add_subplot(grid[2, :2])
    for name, (label, color) in RUNS.items():
        r = runs[name]
        t = r["transect"]
        ax.plot(r["x"][t], r["zeta"][t], color=color, label=label)
    ax.set_ylabel("water level (m)")
    ax.set_title("Water level along the centerline (wind setup dominates)",
                 fontsize=10)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(grid[2, 2])
    base = runs["wind_only"]
    for name in ("komen", "st6"):
        r = runs[name]
        t = r["transect"]
        ax.plot(r["x"][t], 100.0 * (r["zeta"][t] - base["zeta"][t]),
                color=RUNS[name][1], label=RUNS[name][0])
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_ylabel("cm")
    ax.set_title("Wave setdown and setup\n(water level minus no-waves run)",
                 fontsize=10)
    ax.grid(alpha=0.3)

    for k, name in enumerate(STATIONS):
        ax = fig.add_subplot(grid[3, k])
        for run in ("komen", "st6"):
            with Dataset(HERE / f"{run}_stations.nc") as ds:
                i = list(ds["station_name"][:]).index(name)
                hours = (ds["time"][:] - ds["time"][0]) / 3600.0
                ax.plot(hours, ds["wave_hs"][:, i], color=RUNS[run][1],
                        label=RUNS[run][0])
                bed = -float(ds["depth"][i])
        ax.set_title(f"station {name} (bed {bed:+.1f} m)", fontsize=10)
        ax.set_xlabel("hours")
        ax.grid(alpha=0.3)
        if k == 0:
            ax.set_ylabel("Hs (m)")

    fig.suptitle("Cocoa: wave_beach, a 20 m/s onshore wind over a sloping beach",
                 fontweight="bold")
    fig.savefig(args.output, dpi=130)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
