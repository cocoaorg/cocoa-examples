#!/usr/bin/env python3
"""
Plot the katrina_waves example: peak wind, peak water level and peak
significant wave height over the Gulf of Mexico with the storm track, and
water level and wave height at the stations.

Usage:
    cocoa -i cocoa_config.yaml
    python plot_results.py
"""

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from netCDF4 import Dataset, num2date

HERE = Path(__file__).absolute().parent
EXTENT = (-95.0, -80.0, 22.0, 31.5)
SERIES = ("ndbc_42001", "ndbc_42040", "grand_isle", "dauphin_island")


def read_track(path):
    """Best-track positions (lon, lat) from an ATCF b-deck, one per time."""
    seen, lon, lat = set(), [], []
    for line in path.read_text().splitlines():
        f = [c.strip() for c in line.split(",")]
        if f[2] in seen:
            continue
        seen.add(f[2])
        lat.append(float(f[6][:-1]) / 10.0 * (1 if f[6].endswith("N") else -1))
        lon.append(float(f[7][:-1]) / 10.0 * (-1 if f[7].endswith("W") else 1))
    return np.array(lon), np.array(lat)


def peak_map(ax, fig, tri, values, title, label, cmap, track):
    v = np.ma.filled(values, np.nan)
    t = mtri.Triangulation(tri.x, tri.y, tri.triangles)
    t.set_mask(np.isnan(v[tri.triangles]).any(axis=1))
    tc = ax.tripcolor(t, np.nan_to_num(v), shading="gouraud", cmap=cmap)
    ax.plot(*track, "k.-", lw=0.8, ms=3)
    ax.set_xlim(EXTENT[:2])
    ax.set_ylim(EXTENT[2:])
    ax.set_aspect("equal")
    ax.set_facecolor("0.85")
    ax.set_title(title, fontsize=10)
    fig.colorbar(tc, ax=ax, label=label, shrink=0.85)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-o", "--output", default=HERE / "katrina_waves_results.png")
    args = parser.parse_args()

    track = read_track(HERE / "katrina_bal122005.dat")
    with Dataset(HERE / "cocoa_output_peak.nc") as pk:
        tri = mtri.Triangulation(
            pk["lon"][:], pk["lat"][:],
            np.asarray(pk["element"][:]) - int(pk["element"].start_index),
        )
        wind = pk["wind_max"][:]
        zeta = pk["zeta_max"][:]
        hs = pk["wave_hs_max"][:]

    fig = plt.figure(figsize=(16, 11))
    grid = fig.add_gridspec(3, 12, height_ratios=[1.3, 1.0, 1.0], hspace=0.35,
                            wspace=1.6, left=0.05, right=0.97, top=0.92,
                            bottom=0.05)
    peak_map(fig.add_subplot(grid[0, 0:4]), fig, tri, wind, "Peak wind speed",
             "m/s", "YlOrRd", track)
    peak_map(fig.add_subplot(grid[0, 4:8]), fig, tri, zeta, "Peak water level",
             "m", "Blues", track)
    peak_map(fig.add_subplot(grid[0, 8:12]), fig, tri, hs,
             "Peak significant wave height", "m", "viridis", track)

    with Dataset(HERE / "cocoa_station_output.nc") as st:
        names = list(st["station_name"][:])
        times = num2date(st["time"][:], st["time"].units,
                         only_use_cftime_datetimes=False)
        lon, lat = st["lon"][:], st["lat"][:]
        for k, name in enumerate(SERIES):
            i = names.index(name)
            ax = fig.add_subplot(grid[1, 3 * k:3 * k + 3])
            ax.plot(times, st["zeta"][:, i], color="tab:blue")
            ax.set_title(f"{name}\n({lat[i]:.2f} N, {-lon[i]:.2f} W)", fontsize=9)
            ax.grid(alpha=0.3)
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.tick_params(axis="x", labelbottom=False)
            if k == 0:
                ax.set_ylabel("water level (m)")
            ax = fig.add_subplot(grid[2, 3 * k:3 * k + 3])
            ax.plot(times, st["wave_hs"][:, i], color="tab:green")
            ax.grid(alpha=0.3)
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            if k == 0:
                ax.set_ylabel("Hs (m)")
    fig.suptitle("Cocoa: Hurricane Katrina (2005), GAHM vortex coupled to the "
                 "wave model", fontweight="bold")
    fig.savefig(args.output, dpi=130)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
