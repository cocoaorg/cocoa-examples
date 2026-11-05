#!/usr/bin/env python3
"""
Plot the sea_ice example: the ice edge as it moves south, how much the ice
changes the water level, and water level with and without ice at two points
on the northern shelf.

Usage:
    python make_ice.py
    cocoa -i cocoa_config.yaml
    cocoa -i cocoa_config_no_ice.yaml
    python plot_results.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from netCDF4 import Dataset

HERE = Path(__file__).absolute().parent
EXTENT = (-78.0, -60.0, 34.0, 46.0)
POINTS = {"Gulf of Maine": (-68.5, 43.0), "Scotian Shelf": (-63.0, 44.0)}


def map_axes(ax, title):
    ax.set_xlim(EXTENT[:2])
    ax.set_ylim(EXTENT[2:])
    ax.set_aspect("equal")
    ax.set_facecolor("0.85")
    ax.set_title(title, fontsize=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-o", "--output", default=HERE / "sea_ice_results.png")
    args = parser.parse_args()

    with Dataset(HERE / "ice_output.nc") as ice, Dataset(HERE / "no_ice_output.nc") as ref:
        lon, lat = ice["lon"][:], ice["lat"][:]
        tri = mtri.Triangulation(
            lon, lat, np.asarray(ice["element"][:]) - int(ice["element"].start_index)
        )
        hours = (ice["time"][:] - ice["time"][0]) / 3600.0
        conc = ice["ice_concentration"][:]
        zeta_ice = np.ma.filled(ice["zeta"][:], np.nan)
        zeta_ref = np.ma.filled(ref["zeta"][:], np.nan)
    change = zeta_ice - zeta_ref
    last_two_days = hours >= hours[-1] - 48.0
    mean_change = np.nanmean(change[last_two_days], axis=0)

    fig = plt.figure(figsize=(14, 9), layout="constrained")
    grid = fig.add_gridspec(2, 3)
    for k, day in enumerate((2, 5)):
        ax = fig.add_subplot(grid[0, k])
        t = int(np.argmin(np.abs(hours - 24 * day)))
        tc = ax.tripcolor(tri, conc[t], shading="gouraud", cmap="Blues_r",
                          vmin=0, vmax=1)
        map_axes(ax, f"Ice concentration, day {day}")
        fig.colorbar(tc, ax=ax, shrink=0.8, label="fraction")
    ax = fig.add_subplot(grid[0, 2])
    tc = ax.tripcolor(tri, np.nan_to_num(mean_change), shading="gouraud",
                      cmap="RdBu_r", vmin=-0.03, vmax=0.03)
    for name, (x, y) in POINTS.items():
        ax.plot(x, y, "o", mfc="none", mec="c", ms=8)
    map_axes(ax, "Mean water level change from the ice,\nlast two days")
    fig.colorbar(tc, ax=ax, shrink=0.8, label="m", extend="both")

    for (name, (x, y)), cols in zip(POINTS.items(), (slice(0, 2), slice(2, 3))):
        i = int(np.argmin((lon - x) ** 2 + (lat - y) ** 2))
        ax = fig.add_subplot(grid[1, cols])
        ax.plot(hours / 24.0, change[:, i], color="tab:blue", lw=1.0,
                label="water level, ice minus no ice")
        frac = ax.twinx()
        frac.fill_between(hours / 24.0, conc[:, i], color="tab:cyan", alpha=0.2,
                          label="ice concentration")
        frac.set_ylim(0, 1)
        frac.set_ylabel("ice concentration")
        ax.set_title(f"{name} ({lat[i]:.2f} N, {-lon[i]:.2f} W)", fontsize=10)
        ax.set_xlabel("days since 2026-02-01")
        ax.set_ylabel("m")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Cocoa: sea_ice, an ice edge moving south changes the wind "
                 "drag", fontweight="bold")
    fig.savefig(args.output, dpi=130)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
