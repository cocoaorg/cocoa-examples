#!/usr/bin/env python3
"""
Plot the tidal_beach example: the three meshes, the shoreline each one moves
up and down the beach, and the water level at the stations.

Run the three configurations first:
    cocoa -i cocoa_config_triangle.yaml
    cocoa -i cocoa_config_quad.yaml
    cocoa -i cocoa_config_hybrid.yaml
    python plot_results.py
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from netCDF4 import Dataset, num2date

HERE = Path(__file__).absolute().parent
VARIANTS = ("triangle", "quad", "hybrid")
COLORS = {"triangle": "tab:blue", "quad": "tab:orange", "hybrid": "tab:green"}
STATIONS = ("km3", "km4", "km5", "km6")
EARTH_RADIUS_M = 6378206.4
LON0, LAT0 = -90.0, 29.0


def to_km(lon, lat):
    x = np.radians(lon - LON0) * EARTH_RADIUS_M * math.cos(math.radians(LAT0))
    y = np.radians(lat - LAT0) * EARTH_RADIUS_M
    return x / 1000.0, y / 1000.0


def polygons(ds):
    """Element outlines in km; quads keep four corners, triangles three."""
    x, y = to_km(ds["lon"][:], ds["lat"][:])
    elem = np.ma.filled(ds["element"][:], -1)
    start = int(ds["element"].start_index)
    return [
        np.column_stack([x[row[row >= start] - start], y[row[row >= start] - start]])
        for row in elem
    ]


def shoreline_km(ds):
    """Distance from the open boundary to the farthest wet node, per output time."""
    x, y = to_km(ds["lon"][:], ds["lat"][:])
    centre = np.isclose(y, y.max() / 2.0, atol=0.06)
    wet = ~np.ma.getmaskarray(ds["zeta"][:, centre])
    return np.where(wet, x[centre][None, :], 0.0).max(axis=1)


def hours_since_start(ds):
    t = ds["time"][:]
    return (t - t[0]) / 3600.0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-o", "--output", default=HERE / "tidal_beach_results.png")
    args = parser.parse_args()

    fig = plt.figure(figsize=(13, 10), layout="constrained")
    grid = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.1, 1.1])

    for k, variant in enumerate(VARIANTS):
        ax = fig.add_subplot(grid[0, k])
        with Dataset(HERE / f"{variant}_output.nc") as ds:
            polys = polygons(ds)
            depth = ds["depth"][:]
            elem = np.ma.filled(ds["element"][:], -1)
            start = int(ds["element"].start_index)
            elem_depth = [depth[row[row >= start] - start].mean() for row in elem]
        pc = PolyCollection(polys, array=-np.asarray(elem_depth), cmap="terrain",
                            edgecolors="k", linewidths=0.15)
        pc.set_clim(-10, 5)
        ax.add_collection(pc)
        # Zoom onto the hybrid mesh's switch from triangles to quadrilaterals.
        ax.set_xlim(2.0, 3.6)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal")
        ax.set_title(f"{variant} mesh ({len(polys)} elements)", fontsize=10)
        ax.set_xlabel("distance from open boundary (km)")
        if k == 0:
            ax.set_ylabel("km")
    cax = fig.add_subplot(grid[0, 3])
    cax.set_axis_off()
    fig.colorbar(pc, ax=cax, location="left", fraction=0.5, label="bed elevation (m)")

    ax = fig.add_subplot(grid[1, :])
    for variant in VARIANTS:
        with Dataset(HERE / f"{variant}_output.nc") as ds:
            ax.plot(hours_since_start(ds), shoreline_km(ds), color=COLORS[variant],
                    label=variant)
    ax.set_ylabel("shoreline (km from open boundary)")
    ax.set_xlabel("hours since 2026-03-01 00:00")
    ax.set_title("Wet extent along the centerline", fontsize=10)
    ax.legend(loc="lower center", ncols=3)
    ax.grid(alpha=0.3)

    for k, name in enumerate(STATIONS):
        ax = fig.add_subplot(grid[2, k])
        for variant in VARIANTS:
            with Dataset(HERE / f"{variant}_stations.nc") as ds:
                names = list(ds["station_name"][:])
                i = names.index(name)
                ax.plot(hours_since_start(ds), ds["zeta"][:, i], color=COLORS[variant],
                        lw=1.0)
                bed = -float(ds["depth"][i])
        ax.axhline(bed, color="saddlebrown", lw=0.8, ls="--")
        ax.set_title(f"station {name} (bed {bed:+.2f} m)", fontsize=10)
        ax.set_xlabel("hours")
        ax.grid(alpha=0.3)
        if k == 0:
            ax.set_ylabel("water level (m)")

    fig.suptitle("Cocoa: tidal_beach, a 1 m M2 tide on triangles, quadrilaterals "
                 "and a hybrid mesh", fontweight="bold")
    fig.savefig(args.output, dpi=130)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
