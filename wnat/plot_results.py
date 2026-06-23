#!/usr/bin/env python3
"""
Plot Cocoa simulation results for the WNAT (Western North Atlantic) case.

Produces a figure with:
  - Peak elevation map (zeta_max)
  - Peak velocity map (velocity_max)
  - Peak wind speed map (wind_max)
  - Minimum atmospheric pressure map (pressure_min)
  - Time series at 4 probe locations (elevation + velocity)

Usage:
    python plot_results.py
    python plot_results.py --output cocoa_output.nc
"""

import argparse
from pathlib import Path

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
    if "wind_max" in ds.variables:
        data["wind_max"] = np.array(ds.variables["wind_max"][:])
    if "pressure_min" in ds.variables:
        data["pressure_min"] = np.array(ds.variables["pressure_min"][:])
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


def build_triangulation(data, mask_field=None):
    """Build a Matplotlib Triangulation, masking elements where any vertex is dry."""
    tri = mtri.Triangulation(data["x"], data["y"], data["elements"])
    if mask_field is not None:
        elem_dry = np.any(mask_field[data["elements"]] <= COCOA_DRY, axis=1)
        tri.set_mask(elem_dry)
    return tri


def find_nearest_node(x, y, target_lon, target_lat):
    dist = (x - target_lon) ** 2 + (y - target_lat) ** 2
    return int(np.argmin(dist))


def plot_spatial_map(ax, fig, tri, values, label, cmap="RdYlBu_r",
                     fill_value=COCOA_DRY):
    vals = np.copy(values)
    vals[vals <= fill_value] = np.nan
    tc = ax.tripcolor(tri, vals, shading="gouraud", cmap=cmap)
    cb = fig.colorbar(tc, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(label)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")


def main():
    parser = argparse.ArgumentParser(
        description="Plot simple_test simulation results"
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
    tri = build_triangulation(data, mask_field=data["zeta_max"])

    n_times = data["zeta"].shape[0]
    print(
        f"  {len(data['x'])} nodes, {len(data['elements'])} elements, "
        f"{n_times} output steps"
    )

    x, y = data["x"], data["y"]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    probes = [
        ("Gulf of Mexico", x_min + 0.20 * x_range, y_min + 0.55 * y_range),
        ("Atlantic Coast", x_min + 0.70 * x_range, y_min + 0.75 * y_range),
        ("Caribbean West", x_min + 0.35 * x_range, y_min + 0.30 * y_range),
        ("Florida Straits", x_min + 0.50 * x_range, y_min + 0.50 * y_range),
    ]
    probe_indices = [
        (name, find_nearest_node(x, y, tx, ty)) for name, tx, ty in probes
    ]

    has_wind = "wind_max" in data
    has_pressure = "pressure_min" in data
    n_map_rows = 1 + (1 if has_wind or has_pressure else 0)

    fig = plt.figure(figsize=(18, 7 * n_map_rows + 7))
    height_ratios = [2] * n_map_rows + [1, 1]
    gs = gridspec.GridSpec(
        n_map_rows + 2, 4, height_ratios=height_ratios, hspace=0.32, wspace=0.30
    )
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]

    # Row 0: Peak elevation + peak velocity maps
    ax_zmax = fig.add_subplot(gs[0, 0:2])
    plot_spatial_map(ax_zmax, fig, tri, data["zeta_max"], "Peak elevation (m)")
    ax_zmax.set_title("Peak Elevation (zeta_max)", fontweight="bold")
    for i, (name, idx) in enumerate(probe_indices):
        ax_zmax.plot(
            x[idx], y[idx], "o", color=colors[i], markersize=7,
            markeredgecolor="black", markeredgewidth=0.8, zorder=10,
        )

    ax_vmax = fig.add_subplot(gs[0, 2:4])
    plot_spatial_map(
        ax_vmax, fig, tri, data["velocity_max"],
        "Peak velocity (m/s)", cmap="magma_r",
    )
    ax_vmax.set_title("Peak Velocity (velocity_max)", fontweight="bold")

    # Row 1 (if available): Peak wind speed + minimum pressure
    if has_wind or has_pressure:
        tri_unmasked = build_triangulation(data)
        if has_wind:
            ax_wind = fig.add_subplot(gs[1, 0:2])
            plot_spatial_map(
                ax_wind, fig, tri_unmasked, data["wind_max"],
                "Peak wind speed (m/s)", cmap="YlOrRd",
                fill_value=0.0,
            )
            ax_wind.set_title("Peak Wind Speed (wind_max)", fontweight="bold")

        if has_pressure:
            ax_pres = fig.add_subplot(gs[1, 2:4])
            pressure_mb = data["pressure_min"] / 100.0
            plot_spatial_map(
                ax_pres, fig, tri_unmasked, pressure_mb,
                "Min pressure (mb)", cmap="RdYlBu",
                fill_value=0.0,
            )
            ax_pres.set_title(
                "Minimum Atmospheric Pressure (pressure_min)", fontweight="bold"
            )

    # Time series rows
    ts_row0 = n_map_rows
    time_h = data["time_hours"]
    for i, (name, idx) in enumerate(probe_indices):
        ax_e = fig.add_subplot(gs[ts_row0, i])
        ax_e.plot(time_h, data["zeta"][:, idx], color=colors[i], linewidth=1.0)
        ax_e.set_title(
            f"{name}\n({y[idx]:.1f}\u00b0N, {abs(x[idx]):.1f}\u00b0W)",
            fontsize=9, fontweight="bold",
        )
        ax_e.set_ylabel("Elev (m)", fontsize=8)
        ax_e.grid(True, alpha=0.3)

        ax_v = fig.add_subplot(gs[ts_row0 + 1, i])
        vel_ts = np.sqrt(data["ucx"][:, idx] ** 2 + data["ucy"][:, idx] ** 2)
        ax_v.plot(time_h, vel_ts, color=colors[i], linewidth=1.0)
        ax_v.set_ylabel("Speed (m/s)", fontsize=8)
        ax_v.set_xlabel("Time (hours)", fontsize=8)
        ax_v.grid(True, alpha=0.3)

    fig.suptitle(
        "Cocoa \u2014 WNAT results", fontsize=14, fontweight="bold", y=0.98
    )

    out_file = args.output.parent / "wnat_results.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_file}")


if __name__ == "__main__":
    main()
