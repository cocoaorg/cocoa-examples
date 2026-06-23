#!/usr/bin/env python3
"""
Plot Cocoa simulation results for the global test case.

Produces a figure with:
  - Peak elevation map (zeta_max) in Robinson projection
  - Peak velocity map (velocity_max) in Robinson projection
  - Time series at 4 probe locations (elevation + velocity)

Uses Robinson projection with antimeridian masking to avoid
rendering artifacts from elements that wrap around the globe.

Usage:
    python plot_results.py
    python plot_results.py --output cocoa_output.nc
"""

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from netCDF4 import Dataset

COCOA_DRY = -99990.0


def load_cocoa_output(output_path):
    """Load Cocoa output NetCDF and return a dict of arrays."""
    ds = Dataset(output_path, "r")
    data = {}
    data["x"] = np.array(ds.variables["x"][:])
    data["y"] = np.array(ds.variables["y"][:])
    data["depth"] = np.array(ds.variables["depth"][:])
    data["elements"] = np.array(ds.variables["element"][:]) - 1

    time_raw = np.array(ds.variables["time"][:])
    data["time_hours"] = (time_raw - time_raw[0]) / 3600.0

    data["zeta"] = np.array(ds.variables["zeta"][:])
    data["ucx"] = np.array(ds.variables["ucx"][:])
    data["ucy"] = np.array(ds.variables["ucy"][:])

    zeta_max = None
    vel_max = None
    if "zeta_max" in ds.variables:
        zeta_max = np.array(ds.variables["zeta_max"][:])
    if "velocity_max" in ds.variables:
        vel_max = np.array(ds.variables["velocity_max"][:])
    ds.close()

    if zeta_max is None or np.all(zeta_max <= COCOA_DRY):
        zeta_ts = np.where(data["zeta"] <= COCOA_DRY, -np.inf, data["zeta"])
        zeta_max = np.max(zeta_ts, axis=0)
        zeta_max[zeta_max == -np.inf] = COCOA_DRY - 1

    if vel_max is None or np.all(vel_max <= COCOA_DRY):
        speed = np.sqrt(data["ucx"] ** 2 + data["ucy"] ** 2)
        speed[data["zeta"] <= COCOA_DRY] = -np.inf
        vel_max = np.max(speed, axis=0)
        vel_max[vel_max == -np.inf] = COCOA_DRY - 1

    data["zeta_max"] = zeta_max
    data["velocity_max"] = vel_max
    return data


def compute_antimeridian_mask(x, elements, threshold=90.0):
    """Mask elements that cross the antimeridian (longitude jump > threshold)."""
    x_elem = x[elements]
    x_wrapped = np.column_stack([x_elem, x_elem[:, 0]])
    diffs = np.diff(x_wrapped, axis=1)
    return ~np.all(np.abs(diffs) < threshold, axis=1)


def build_robinson_triangulation(data, projection):
    """Build triangulation in Robinson projected coordinates with antimeridian masking."""
    x, y, elements = data["x"], data["y"], data["elements"]
    mask = compute_antimeridian_mask(x, elements)
    pts = projection.transform_points(ccrs.PlateCarree(), x, y)
    return mtri.Triangulation(pts[:, 0], pts[:, 1], elements, mask=mask)


def find_nearest_node(x, y, target_lon, target_lat):
    dist = (x - target_lon) ** 2 + (y - target_lat) ** 2
    return int(np.argmin(dist))


def plot_robinson_map(ax, fig, tri, values, label, cmap="RdYlBu_r",
                      fill_value=COCOA_DRY):
    """Plot a spatial map on Robinson projection axes."""
    vals = np.copy(values)
    vals[vals <= fill_value] = np.nan

    valid = ~np.isnan(vals)
    if not np.any(valid):
        return
    vmin, vmax = np.nanpercentile(vals[valid], [2, 98])

    tc = ax.tripcolor(tri, vals, shading="gouraud", cmap=cmap,
                      vmin=vmin, vmax=vmax, zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
    ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.5)
    ax.set_global()

    cb = fig.colorbar(tc, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8)
    cb.set_label(label)


def main():
    parser = argparse.ArgumentParser(
        description="Plot global_case simulation results"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "cocoa_output.nc",
        help="Path to output file (default: cocoa_output.nc in this directory)",
    )
    args = parser.parse_args()

    if not args.output.exists():
        print(f"Output file not found: {args.output}")
        print("Run the simulation first.")
        return

    print(f"Loading {args.output}")
    data = load_cocoa_output(args.output)

    n_times = data["zeta"].shape[0]
    x, y = data["x"], data["y"]
    print(
        f"  {len(x)} nodes, {len(data['elements'])} elements, "
        f"{n_times} output steps"
    )

    projection = ccrs.Robinson()
    tri = build_robinson_triangulation(data, projection)

    # Probe locations spread across the global ocean
    probes = [
        ("N. Atlantic", -30.0, 40.0),
        ("N. Pacific", -170.0, 35.0),
        ("S. Atlantic", -20.0, -30.0),
        ("Indian Ocean", 70.0, -15.0),
    ]
    probe_indices = [
        (name, find_nearest_node(x, y, tx, ty)) for name, tx, ty in probes
    ]

    fig = plt.figure(figsize=(18, 18))
    gs = gridspec.GridSpec(
        4, 4, height_ratios=[3, 3, 1, 1], hspace=0.30, wspace=0.30
    )
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]

    # Row 0: Peak elevation map (Robinson)
    ax_zmax = fig.add_subplot(gs[0, :], projection=projection)
    plot_robinson_map(ax_zmax, fig, tri, data["zeta_max"], "Peak elevation (m)")
    ax_zmax.set_title("Peak Elevation (zeta_max)", fontweight="bold")

    # Mark probe locations
    for i, (name, idx) in enumerate(probe_indices):
        ax_zmax.plot(
            x[idx], y[idx], "o", color=colors[i], markersize=7,
            markeredgecolor="black", markeredgewidth=0.8, zorder=10,
            transform=ccrs.PlateCarree(),
        )

    # Row 1: Peak velocity map (Robinson)
    ax_vmax = fig.add_subplot(gs[1, :], projection=projection)
    plot_robinson_map(
        ax_vmax, fig, tri, data["velocity_max"],
        "Peak velocity (m/s)", cmap="magma_r",
    )
    ax_vmax.set_title("Peak Velocity (velocity_max)", fontweight="bold")

    # Rows 2-3: Time series
    time_h = data["time_hours"]
    for i, (name, idx) in enumerate(probe_indices):
        ax_e = fig.add_subplot(gs[2, i])
        ax_e.plot(time_h, data["zeta"][:, idx], color=colors[i], linewidth=1.0)
        ax_e.set_title(
            f"{name}\n({y[idx]:.1f}\u00b0N, {abs(x[idx]):.1f}\u00b0"
            f"{'W' if x[idx] < 0 else 'E'})",
            fontsize=9, fontweight="bold",
        )
        ax_e.set_ylabel("Elev (m)", fontsize=8)
        ax_e.grid(True, alpha=0.3)

        ax_v = fig.add_subplot(gs[3, i])
        vel_ts = np.sqrt(data["ucx"][:, idx] ** 2 + data["ucy"][:, idx] ** 2)
        ax_v.plot(time_h, vel_ts, color=colors[i], linewidth=1.0)
        ax_v.set_ylabel("Speed (m/s)", fontsize=8)
        ax_v.set_xlabel("Time (hours)", fontsize=8)
        ax_v.grid(True, alpha=0.3)

    fig.suptitle(
        "Cocoa \u2014 global_case results", fontsize=14, fontweight="bold", y=0.98
    )

    out_file = args.output.parent / "global_case_results.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_file}")


if __name__ == "__main__":
    main()
