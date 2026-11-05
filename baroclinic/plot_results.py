#!/usr/bin/env python3
"""
Plot the baroclinic example around the Gulf of Mexico eddy: the pressure
gradient in the forcing file, and the change in mean water level it makes
against the barotropic run.

Usage:
    python make_forcing.py
    cocoa -i cocoa_config.yaml
    cocoa -i cocoa_config_barotropic.yaml
    python plot_results.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from netCDF4 import Dataset

HERE = Path(__file__).absolute().parent
EXTENT = (-98.0, -80.0, 18.0, 31.0)
LAST_TWO_DAYS = 48  # hourly output


def frame(ax, title):
    ax.set_xlim(EXTENT[:2])
    ax.set_ylim(EXTENT[2:])
    ax.set_aspect("equal")
    ax.set_facecolor("0.85")
    ax.set_title(title, fontsize=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-o", "--output", default=HERE / "baroclinic_results.png")
    args = parser.parse_args()

    with Dataset(HERE / "baroclinic_output.nc") as bc, \
            Dataset(HERE / "barotropic_output.nc") as bt, \
            Dataset(HERE / "baroclinic_forcing.nc") as fc:
        lon, lat = bc["lon"][:], bc["lat"][:]
        tri = mtri.Triangulation(
            lon, lat, np.asarray(bc["element"][:]) - int(bc["element"].start_index)
        )
        hours = (bc["time"][:] - bc["time"][0]) / 3600.0
        dzeta = np.ma.filled(bc["zeta"][:] - bt["zeta"][:], np.nan)

        def mean_change(name):
            return np.ma.filled(
                bc[name][-LAST_TWO_DAYS:].mean(axis=0)
                - bt[name][-LAST_TWO_DAYS:].mean(axis=0), np.nan)

        dz = mean_change("zeta")
        bpg_x, bpg_y = fc["bpg_x"][0], fc["bpg_y"][0]

    inside = ((lon > EXTENT[0]) & (lon < EXTENT[1]) & (lat > EXTENT[2])
              & (lat < EXTENT[3]))
    arrows = np.flatnonzero(inside)[::4]

    fig = plt.figure(figsize=(12, 9), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.3, 1.0])

    ax = fig.add_subplot(grid[0, 0])
    tc = ax.tripcolor(tri, 1e6 * np.hypot(bpg_x, bpg_y), shading="gouraud",
                      cmap="viridis")
    ax.quiver(lon[arrows], lat[arrows], bpg_x[arrows], bpg_y[arrows], color="w",
              scale=2e-4, width=0.003)
    fig.colorbar(tc, ax=ax, shrink=0.8, label="1e-6 m s-2")
    frame(ax, "Baroclinic pressure gradient in the forcing file")

    ax = fig.add_subplot(grid[0, 1])
    tc = ax.tripcolor(tri, np.nan_to_num(dz), shading="gouraud", cmap="RdBu_r",
                      vmin=-0.4, vmax=0.4)
    fig.colorbar(tc, ax=ax, shrink=0.8, label="m")
    frame(ax, "Mean water level change, last two days")

    ax = fig.add_subplot(grid[1, :])
    centre = int(np.nanargmax(np.where(inside, dz, -np.inf)))
    ax.plot(hours / 24.0, dzeta[:, centre], lw=1.0)
    ax.set_xlabel("days since 2026-02-01")
    ax.set_ylabel("water level change (m)")
    ax.set_title(f"Over the eddy ({lat[centre]:.2f} N, {-lon[centre]:.2f} W): the "
                 "sea surface rises as the forcing ramps in over two days",
                 fontsize=10)
    ax.grid(alpha=0.3)

    fig.suptitle("Cocoa: baroclinic, a warm-core Gulf of Mexico eddy",
                 fontweight="bold")
    fig.savefig(args.output, dpi=130)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
