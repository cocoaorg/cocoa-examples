#!/usr/bin/env python3
"""
Plot the water_level_correction example: the correction on the mesh, how
much each method raised the mean water level over the last day, and the
water level change at the stations as the correction ramps in.

Usage:
    python make_correction.py
    cocoa -i cocoa_config.yaml
    cocoa -i cocoa_config_pressure.yaml
    cocoa -i cocoa_config_baseline.yaml
    python plot_results.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from netCDF4 import Dataset

HERE = Path(__file__).absolute().parent
METHODS = {
    "mass_adjust": ("pressure_and_mass_adjust (default)", "tab:blue"),
    "pressure": ("pressure", "tab:orange"),
}
LAST_DAY = 48  # half-hourly output: the last 24 hours


def triangulation(ds):
    return mtri.Triangulation(
        ds["lon"][:], ds["lat"][:],
        np.asarray(ds["element"][:]) - int(ds["element"].start_index),
    )


def mean_level(name):
    with Dataset(HERE / f"{name}_output.nc") as ds:
        return np.ma.filled(ds["zeta"][-LAST_DAY:], np.nan).mean(axis=0)


def draw(ax, fig, tri, values, title, vmax):
    v = np.asarray(values)
    t = mtri.Triangulation(tri.x, tri.y, tri.triangles)
    t.set_mask(np.isnan(v[tri.triangles]).any(axis=1))
    tc = ax.tripcolor(t, np.nan_to_num(v), shading="gouraud", cmap="RdBu_r",
                      vmin=-vmax, vmax=vmax)
    ax.set_aspect("equal")
    ax.set_facecolor("0.85")
    ax.set_title(title, fontsize=10)
    fig.colorbar(tc, ax=ax, label="m", shrink=0.8)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-o", "--output",
                        default=HERE / "water_level_correction_results.png")
    args = parser.parse_args()

    with Dataset(HERE / "mass_adjust_output.nc") as ds:
        tri = triangulation(ds)
        correction = np.ma.filled(ds["water_level_correction"][-1], np.nan)
    baseline = mean_level("baseline")

    fig = plt.figure(figsize=(15, 9), layout="constrained")
    grid = fig.add_gridspec(2, 3)
    draw(fig.add_subplot(grid[0, 0]), fig, tri, correction,
         "Applied correction", 0.3)
    for k, (name, (label, _)) in enumerate(METHODS.items(), start=1):
        draw(fig.add_subplot(grid[0, k]), fig, tri, mean_level(name) - baseline,
             f"Mean water level change, last day\n{label}", 0.3)

    with Dataset(HERE / "baseline_stations.nc") as ds:
        names = list(ds["station_name"][:])
        base = ds["zeta"][:]
    for i, name in enumerate(names):
        ax = fig.add_subplot(grid[1, i])
        for run, (label, color) in METHODS.items():
            with Dataset(HERE / f"{run}_stations.nc") as ds:
                hours = (ds["time"][:] - ds["time"][0]) / 3600.0
                ax.plot(hours / 24.0, ds["zeta"][:, i] - base[:, i], color=color,
                        lw=0.9, label=label)
                applied = ds["water_level_correction"][:, i]
        ax.plot(hours / 24.0, applied, "k--", lw=1.0, label="correction")
        ax.set_title(f"station {name}", fontsize=10)
        ax.set_xlabel("days since 2026-01-15")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.set_ylabel("water level minus baseline (m)")
            ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Cocoa: water_level_correction, a data-assimilation correction "
                 "ramped in after two days of tide", fontweight="bold")
    fig.savefig(args.output, dpi=130)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
