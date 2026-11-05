#!/usr/bin/env python3
"""
Plot the two-vortex global test: peak wind, minimum pressure, peak water level.

Reproduces the Robinson-projection method used by global_case/plot_results.py
(antimeridian-masked triangulation + tripcolor) and renders the three
end-of-simulation peak fields on the globe, with zoomed insets over the two
storm basins so the Atlantic and Pacific vortices are both legible.

Usage:
    python plot_two_vortex.py                       # reads cocoa_output_peak.nc
    python plot_two_vortex.py --peak <file>_peak.nc
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib import gridspec
from netCDF4 import Dataset

if TYPE_CHECKING:
    from cartopy.crs import Projection
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

COCOA_DRY = -99990.0
PA_TO_MB = 0.01

# The two storm centers (peak-intensity longitude/latitude), for the insets.
# The Pacific center stays clear of the antimeridian so the inset extent
# (center +- INSET_HALF_WIDTH) does not wrap past -180 deg.
ATLANTIC = {"name": "Atlantic (Katrina)", "lon": -88.0, "lat": 27.5}
PACIFIC = {"name": "Pacific (relocated)", "lon": -166.0, "lat": 27.0}
INSET_HALF_WIDTH = 13.0  # degrees around each storm center


def load_peak(path: Path) -> dict[str, np.ndarray]:
    ds = Dataset(path, "r")
    data = {
        "x": np.array(ds.variables["lon"][:]),
        "y": np.array(ds.variables["lat"][:]),
        "elements": np.array(ds.variables["element"][:]) - 1,
        "zeta_max": np.array(ds.variables["zeta_max"][:]),
        "wind_max": np.array(ds.variables["wind_max"][:]),
        "pressure_min": np.array(ds.variables["pressure_min"][:]),
    }
    ds.close()
    return data


def antimeridian_mask(
    x: np.ndarray, elements: np.ndarray, threshold: float = 90.0
) -> np.ndarray:
    x_elem = x[elements]
    x_wrapped = np.column_stack([x_elem, x_elem[:, 0]])
    diffs = np.diff(x_wrapped, axis=1)
    return ~np.all(np.abs(diffs) < threshold, axis=1)


def triangulation(
    data: dict[str, np.ndarray], projection: Projection
) -> mtri.Triangulation:
    x, y, elements = data["x"], data["y"], data["elements"]
    mask = antimeridian_mask(x, elements)
    pts = projection.transform_points(ccrs.PlateCarree(), x, y)
    return mtri.Triangulation(pts[:, 0], pts[:, 1], elements, mask=mask)


def draw_map(
    ax: Axes,
    fig: Figure,
    tri: mtri.Triangulation,
    values: np.ndarray,
    label: str,
    cmap: str,
    vmin: float,
    vmax: float,
    *,
    global_view: bool = True,
) -> None:
    vals = np.copy(values)
    vals[vals <= COCOA_DRY] = np.nan
    tc = ax.tripcolor(
        tri, vals, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax, zorder=0
    )
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=2)
    ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.4)
    if global_view:
        ax.set_global()
    cb = fig.colorbar(tc, ax=ax, orientation="vertical", pad=0.02, shrink=0.85)
    cb.set_label(label, fontsize=9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--peak",
        type=Path,
        default=Path(__file__).parent / "cocoa_output_peak.nc",
        help="Path to the *_peak.nc output file",
    )
    args = parser.parse_args()
    if not args.peak.exists():
        msg = f"Peak file not found: {args.peak}. Run the simulation first."
        raise SystemExit(msg)

    print(f"Loading {args.peak}")
    data = load_peak(args.peak)
    pressure_mb = np.where(
        data["pressure_min"] <= COCOA_DRY,
        COCOA_DRY,
        data["pressure_min"] * PA_TO_MB,
    )

    globe = ccrs.Robinson(central_longitude=-120.0)  # centers both basins
    tri_globe = triangulation(data, globe)

    # Per-field color scaling. Wind and surge are ~0 over the quiet ocean and
    # peak at the storms; pressure is ~1013 mb background with deep lows at the
    # storms -- so pressure is scaled from its minimum up to the background.
    def hi(v: np.ndarray) -> float:
        m = v[v > COCOA_DRY]
        return float(np.nanpercentile(m, 99.9))

    fields = [
        (
            "wind_max",
            data["wind_max"],
            "Peak wind speed (m/s)",
            "viridis",
            0.0,
            hi(data["wind_max"]),
        ),
        (
            "pressure_min",
            pressure_mb,
            "Minimum pressure (mb)",
            "plasma_r",
            float(np.nanmin(pressure_mb[pressure_mb > COCOA_DRY])),
            1013.25,
        ),
        (
            "zeta_max",
            data["zeta_max"],
            "Peak water level (m)",
            "YlGnBu",
            0.0,
            hi(data["zeta_max"]),
        ),
    ]

    fig = plt.figure(figsize=(15, 16))
    gs = gridspec.GridSpec(3, 3, width_ratios=[2.0, 1.0, 1.0], hspace=0.18, wspace=0.16)

    for row, (_key, vals, label, cmap, vmin, vmax) in enumerate(fields):
        # Global map (left, wide).
        ax_g = fig.add_subplot(gs[row, 0], projection=globe)
        draw_map(ax_g, fig, tri_globe, vals, label, cmap, vmin, vmax)
        ax_g.set_title(label, fontweight="bold", fontsize=11)

        # Basin insets (PlateCarree) so both storms are legible.
        for col, storm in ((1, ATLANTIC), (2, PACIFIC)):
            proj = ccrs.PlateCarree()
            ax_i = fig.add_subplot(gs[row, col], projection=proj)
            ax_i.set_extent(
                [
                    storm["lon"] - INSET_HALF_WIDTH,
                    storm["lon"] + INSET_HALF_WIDTH,
                    storm["lat"] - INSET_HALF_WIDTH,
                    storm["lat"] + INSET_HALF_WIDTH,
                ],
                crs=proj,
            )
            tri_i = triangulation(data, proj)
            draw_map(ax_i, fig, tri_i, vals, label, cmap, vmin, vmax, global_view=False)
            if row == 0:
                ax_i.set_title(storm["name"], fontsize=10, fontweight="bold")

    fig.suptitle(
        "Cocoa — two-vortex global test (GAHM): Atlantic + Pacific storms",
        fontsize=15,
        fontweight="bold",
        y=0.92,
    )

    out = args.peak.parent / "two_vortex_results.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
