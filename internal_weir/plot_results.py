#!/usr/bin/env python3
"""
Plot Cocoa simulation results for the internal_weir test case.

Produces a figure with:
  - Row 0: Pre-overtopping snapshots (t=0 h) and initial flooding (t=0.25, 0.5, 0.75 h)
  - Row 1: Steady-state tidal cycle snapshots (high tide -> low tide -> next high)
  - Row 2-3: Time series at 4 probes (elevation + velocity)

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
WEIR_CREST = 0.4  # meters


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
    ds.close()
    return data


def build_triangulation(data, mask_field=None):
    """Build a Matplotlib Triangulation, masking dry elements."""
    tri = mtri.Triangulation(data["x"], data["y"], data["elements"])
    if mask_field is not None:
        elem_dry = np.any(mask_field[data["elements"]] <= COCOA_DRY, axis=1)
        tri.set_mask(elem_dry)
    return tri


def find_nearest_node(x, y, target_lon, target_lat):
    dist = (x - target_lon) ** 2 + (y - target_lat) ** 2
    return int(np.argmin(dist))


def find_nearest_time_index(time_hours, target_hour):
    return int(np.argmin(np.abs(time_hours - target_hour)))


def detect_weir_longitude(mesh_path, y_coords):
    """Detect weir longitude from depth transition in mesh file."""
    if not mesh_path.exists():
        return None
    with Dataset(mesh_path, "r") as mds:
        mx = np.array(mds.variables["x"][:])
        md = np.array(mds.variables["depth"][:])
        my = np.array(mds.variables["y"][:])
        y_mid = (y_coords.min() + y_coords.max()) / 2.0
        center_mask = np.abs(my - y_mid) < 0.003
        if not np.any(center_mask):
            return None
        cx = mx[center_mask]
        cd = md[center_mask]
        order = np.argsort(cx)
        dd = np.abs(np.diff(cd[order]))
        if len(dd) > 0:
            return cx[order][np.argmax(dd)]
    return None


def plot_elevation_snapshot(ax, fig, data, time_idx, weir_lon, vmin, vmax,
                            cmap="RdYlBu_r", show_cbar=True, title=None):
    """Plot an elevation snapshot at a specific time index."""
    zeta_snap = data["zeta"][time_idx, :].copy()
    tri = build_triangulation(data, mask_field=zeta_snap)
    zeta_snap[zeta_snap <= COCOA_DRY] = np.nan

    tc = ax.tripcolor(tri, zeta_snap, shading="gouraud", cmap=cmap,
                       vmin=vmin, vmax=vmax)
    if show_cbar:
        cb = fig.colorbar(tc, ax=ax, shrink=0.8, pad=0.02)
        cb.set_label("Elevation (m)", fontsize=8)
        cb.ax.tick_params(labelsize=7)

    ax.axvline(weir_lon, color="black", linestyle="--", linewidth=1.0,
               alpha=0.7)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)

    if title is None:
        t_hr = data["time_hours"][time_idx]
        title = f"t = {t_hr:.2f} h"
    ax.set_title(title, fontweight="bold", fontsize=9)
    return tc


def main():
    parser = argparse.ArgumentParser(
        description="Plot internal_weir simulation results"
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).parent / "cocoa_output.nc",
    )
    parser.add_argument(
        "--mesh", type=Path,
        default=Path(__file__).parent / "internal_weir.nc",
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
    time_h = data["time_hours"]
    dt_out = time_h[1] - time_h[0] if n_times > 1 else 1.0
    print(f"  {len(x)} nodes, {len(data['elements'])} elements, "
          f"{n_times} steps, dt_out={dt_out:.4f} h")

    # Find weir location
    weir_lon = detect_weir_longitude(args.mesh, y)
    if weir_lon is None:
        weir_lon = x.min() + 2.0 / 3.0 * (x.max() - x.min())
    print(f"  Weir longitude: {weir_lon:.4f}")

    # =========================================================================
    # Select snapshot times
    # =========================================================================
    # Row 0: Early time showing head building up on left side
    row0_hours = [0.0, 6.0, 12.0, 24.0]
    row0_labels = [
        "t = 0 h (initial, flat)",
        "t = 6 h (ramp 25%)",
        "t = 12 h (ramp 50%, tide peak)",
        "t = 24 h (ramp complete)",
    ]

    # Row 1: Steady-state tidal cycle (after ramp, ~day 2)
    # M2 period ~12.42 h. Find a high tide peak after day 1.
    y_mid = (y.min() + y.max()) / 2.0
    probe_node = find_nearest_node(x, y, weir_lon - 0.02, y_mid)
    zeta_probe = data["zeta"][:, probe_node].copy()
    zeta_probe[zeta_probe <= COCOA_DRY] = np.nan

    # Search for a high tide peak between 24-48h
    search_start = find_nearest_time_index(time_h, 24.0)
    search_end = find_nearest_time_index(time_h, 48.0)
    search_slice = zeta_probe[search_start:search_end]
    peak_offset = np.nanargmax(search_slice)
    peak_idx = search_start + peak_offset
    peak_hr = time_h[peak_idx]
    print(f"  High tide peak at {peak_hr:.2f} h (elev={zeta_probe[peak_idx]:.3f} m)")

    # Snapshots: 3h before peak, 1.5h before, at peak, 1.5h after, 3h after
    m2_quarter = 12.42 / 4  # ~3.1 h
    row1_offsets = [-m2_quarter, -m2_quarter / 2, 0.0, m2_quarter / 2]
    row1_hours = [peak_hr + dt for dt in row1_offsets]
    row1_labels = [
        f"t = {row1_hours[0]:.1f} h (rising tide)",
        f"t = {row1_hours[1]:.1f} h (approaching crest)",
        f"t = {row1_hours[2]:.1f} h (high tide peak)",
        f"t = {row1_hours[3]:.1f} h (receding tide)",
    ]

    row0_indices = [find_nearest_time_index(time_h, t) for t in row0_hours]
    row1_indices = [find_nearest_time_index(time_h, t) for t in row1_hours]

    all_snap_indices = row0_indices + row1_indices

    print(f"  Row 0 hours: " +
          ", ".join(f"{time_h[i]:.2f}" for i in row0_indices))
    print(f"  Row 1 hours: " +
          ", ".join(f"{time_h[i]:.2f}" for i in row1_indices))

    # Color range: separate for each row to maximize contrast
    def compute_vlim(indices):
        all_wet = []
        for idx in indices:
            z = data["zeta"][idx, :]
            wet = z[z > COCOA_DRY]
            if len(wet) > 0:
                all_wet.append(wet)
        if all_wet:
            combined = np.concatenate(all_wet)
            vlim = max(abs(np.percentile(combined, 2)),
                       abs(np.percentile(combined, 98)))
            return -vlim, vlim
        return -0.5, 0.5

    vmin0, vmax0 = compute_vlim(row0_indices)
    vmin1, vmax1 = compute_vlim(row1_indices)

    # =========================================================================
    # Figure
    # =========================================================================
    fig = plt.figure(figsize=(20, 20))
    gs = gridspec.GridSpec(
        6, 4, height_ratios=[2, 0.15, 2, 0.15, 1, 1],
        hspace=0.35, wspace=0.25,
    )

    # Row 0: Initial + flooding snapshots
    for col in range(4):
        ax = fig.add_subplot(gs[0, col])
        show_cbar = (col == 3)
        plot_elevation_snapshot(ax, fig, data, row0_indices[col], weir_lon,
                                vmin0, vmax0, show_cbar=show_cbar,
                                title=row0_labels[col])
        if col > 0:
            ax.set_yticklabels([])

    # Row 2: Steady-state tidal cycle
    for col in range(4):
        ax = fig.add_subplot(gs[2, col])
        show_cbar = (col == 3)
        plot_elevation_snapshot(ax, fig, data, row1_indices[col], weir_lon,
                                vmin1, vmax1, show_cbar=show_cbar,
                                title=row1_labels[col])
        if col > 0:
            ax.set_yticklabels([])

    # Rows 4-5: Time series at 4 probes
    x_min, x_max = x.min(), x.max()
    probes = [
        ("Left basin", x_min + 0.15 * (x_max - x_min), y_mid),
        ("Before weir", weir_lon - 0.02, y_mid),
        ("After weir", weir_lon + 0.02, y_mid),
        ("Right basin", x_min + 0.85 * (x_max - x_min), y_mid),
    ]
    probe_indices = [
        (name, find_nearest_node(x, y, tx, ty)) for name, tx, ty in probes
    ]
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]

    for i, (name, idx) in enumerate(probe_indices):
        # Elevation
        ax_e = fig.add_subplot(gs[4, i])
        zeta_ts = data["zeta"][:, idx].copy()
        zeta_ts[zeta_ts <= COCOA_DRY] = np.nan
        ax_e.plot(time_h, zeta_ts * 100, color=colors[i], linewidth=0.8)
        ax_e.axhline(WEIR_CREST * 100, color="gray", linestyle=":",
                     linewidth=0.7, alpha=0.5, label="Weir crest")
        # Mark the tidal cycle window
        ax_e.axvspan(row1_hours[0], row1_hours[-1], alpha=0.08,
                     color="blue")
        ax_e.set_title(
            f"{name}\n({y[idx]:.3f}\u00b0N, {abs(x[idx]):.3f}\u00b0W)",
            fontsize=8, fontweight="bold",
        )
        ax_e.set_ylabel("Elev (cm)", fontsize=7)
        ax_e.tick_params(labelsize=7)
        ax_e.grid(True, alpha=0.3)
        if i == 0:
            ax_e.legend(fontsize=6, loc="upper right")

        # Velocity
        ax_v = fig.add_subplot(gs[5, i])
        vel_ts = np.sqrt(data["ucx"][:, idx] ** 2 + data["ucy"][:, idx] ** 2)
        vel_ts[data["zeta"][:, idx] <= COCOA_DRY] = np.nan
        ax_v.plot(time_h, vel_ts, color=colors[i], linewidth=0.8)
        ax_v.axvspan(row1_hours[0], row1_hours[-1], alpha=0.08,
                     color="blue")
        ax_v.set_ylabel("Speed (m/s)", fontsize=7)
        ax_v.set_xlabel("Time (hours)", fontsize=7)
        ax_v.tick_params(labelsize=7)
        ax_v.grid(True, alpha=0.3)

    fig.suptitle(
        "Cocoa \u2014 Internal Weir Overflow\n"
        f"Weir crest = {WEIR_CREST} m, M2 tide amplitude = 0.5 m, "
        f"1-day ramp",
        fontsize=13, fontweight="bold", y=0.995,
    )

    out_file = args.output.parent / "internal_weir_results.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_file}")


if __name__ == "__main__":
    main()
