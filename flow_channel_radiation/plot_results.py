#!/usr/bin/env python3
"""
Plot Cocoa simulation results for the flow_channel_radiation test case.

Produces a figure with:
  - Peak elevation + velocity maps
  - Last-timestep elevation + velocity snapshots
  - Time series at 4 probe locations along the channel centerline

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
        description="Plot flow_channel_radiation simulation results"
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
    x, y = data["x"], data["y"]
    print(
        f"  {len(x)} nodes, {len(data['elements'])} elements, "
        f"{n_times} output steps"
    )

    # Probe locations along channel centerline
    x_min, x_max = x.min(), x.max()
    y_mid = (y.min() + y.max()) / 2.0
    probes = [
        ("Upstream", x_min + 0.10 * (x_max - x_min), y_mid),
        ("Quarter", x_min + 0.25 * (x_max - x_min), y_mid),
        ("Mid-channel", x_min + 0.50 * (x_max - x_min), y_mid),
        ("Downstream", x_min + 0.90 * (x_max - x_min), y_mid),
    ]
    probe_indices = [
        (name, find_nearest_node(x, y, tx, ty)) for name, tx, ty in probes
    ]

    fig = plt.figure(figsize=(18, 16))
    gs = gridspec.GridSpec(
        4, 4, height_ratios=[2, 2, 1, 1], hspace=0.35, wspace=0.30
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

    # Row 1: Last-timestep snapshots
    ax_zlast = fig.add_subplot(gs[1, 0:2])
    ax_vlast = fig.add_subplot(gs[1, 2:4])

    zeta_last = np.copy(data["zeta"][-1, :])
    zeta_last[zeta_last <= COCOA_DRY] = np.nan
    tc3 = ax_zlast.tripcolor(tri, zeta_last, shading="gouraud", cmap="RdYlBu_r")
    cb3 = fig.colorbar(tc3, ax=ax_zlast, shrink=0.85, pad=0.02)
    cb3.set_label("Elevation (m)")
    ax_zlast.set_title(
        f"Elevation at t = {data['time_hours'][-1]:.1f} h", fontweight="bold"
    )
    ax_zlast.set_xlabel("Longitude")
    ax_zlast.set_ylabel("Latitude")
    ax_zlast.set_aspect("equal")

    vel_last = np.sqrt(data["ucx"][-1, :] ** 2 + data["ucy"][-1, :] ** 2)
    vel_last[data["zeta"][-1, :] <= COCOA_DRY] = np.nan
    tc4 = ax_vlast.tripcolor(tri, vel_last, shading="gouraud", cmap="magma_r")
    cb4 = fig.colorbar(tc4, ax=ax_vlast, shrink=0.85, pad=0.02)
    cb4.set_label("Speed (m/s)")
    ax_vlast.set_title(
        f"Velocity at t = {data['time_hours'][-1]:.1f} h", fontweight="bold"
    )
    ax_vlast.set_xlabel("Longitude")
    ax_vlast.set_ylabel("Latitude")
    ax_vlast.set_aspect("equal")

    # Rows 2-3: Time series
    time_h = data["time_hours"]
    for i, (name, idx) in enumerate(probe_indices):
        ax_e = fig.add_subplot(gs[2, i])
        ax_e.plot(
            time_h, data["zeta"][:, idx] * 100, color=colors[i], linewidth=1.0
        )
        ax_e.set_title(
            f"{name}\n({y[idx]:.3f}\u00b0N, {abs(x[idx]):.3f}\u00b0W)",
            fontsize=9, fontweight="bold",
        )
        ax_e.set_ylabel("Elev (cm)", fontsize=8)
        ax_e.grid(True, alpha=0.3)

        ax_v = fig.add_subplot(gs[3, i])
        vel_ts = np.sqrt(data["ucx"][:, idx] ** 2 + data["ucy"][:, idx] ** 2)
        ax_v.plot(time_h, vel_ts, color=colors[i], linewidth=1.0)
        ax_v.set_ylabel("Speed (m/s)", fontsize=8)
        ax_v.set_xlabel("Time (hours)", fontsize=8)
        ax_v.grid(True, alpha=0.3)

    fig.suptitle(
        "Cocoa \u2014 flow_channel_radiation results",
        fontsize=14, fontweight="bold", y=0.99,
    )

    out_file = args.output.parent / "flow_channel_radiation_results.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_file}")


if __name__ == "__main__":
    main()
