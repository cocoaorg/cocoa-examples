#!/usr/bin/env python3
"""
Plot the Southern-Hemisphere Australia cyclone test.

Shows peak wind, minimum pressure, and peak water level (global + NW-Australia
inset), plus a wind-vector panel at peak intensity that makes the CLOCKWISE
(Southern-Hemisphere) circulation directly visible -- the point of the test.

Reuses the Robinson-projection method from global_case/plot_results.py
(antimeridian-masked triangulation + tripcolor).

Usage:
    python plot_sh_australia.py
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

# NW-Australia inset extent [lon_min, lon_max, lat_min, lat_max].
INSET = [104.0, 134.0, -31.0, -5.0]


def antimeridian_mask(
    x: np.ndarray, elements: np.ndarray, threshold: float = 90.0
) -> np.ndarray:
    x_elem = x[elements]
    x_wrapped = np.column_stack([x_elem, x_elem[:, 0]])
    return ~np.all(np.abs(np.diff(x_wrapped, axis=1)) < threshold, axis=1)


def triangulation(
    x: np.ndarray, y: np.ndarray, elements: np.ndarray, projection: Projection
) -> mtri.Triangulation:
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


def peak_hi(v: np.ndarray) -> float:
    """Upper color limit: the 99.9th percentile of the wet values."""
    m = v[v > COCOA_DRY]
    return float(np.nanpercentile(m, 99.9))


def draw_wind_vectors(
    fig: Figure,
    ax: Axes,
    coords: tuple[np.ndarray, np.ndarray],
    vectors: tuple[np.ndarray, np.ndarray],
    speed: np.ndarray,
    wind_max: np.ndarray,
) -> None:
    """
    Hero panel: peak-intensity wind vectors, coloured by speed.

    A clean ocean background lets the arrows carry the message (clockwise sense),
    with a red curved annotation marking the sense of rotation.
    """
    x, y = coords
    plate = ccrs.PlateCarree()
    ax.set_extent(INSET, crs=plate)
    inbox = (x >= INSET[0]) & (x <= INSET[1]) & (y >= INSET[2]) & (y <= INSET[3])
    sel = inbox & (speed > 1.0)
    ax.add_feature(cfeature.OCEAN, facecolor="#eef4fb", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=2)
    q = ax.quiver(
        x[sel],
        y[sel],
        vectors[0][sel],
        vectors[1][sel],
        speed[sel],
        cmap="viridis",
        scale=650,
        width=0.0026,
        transform=plate,
        zorder=5,
    )
    ax.quiverkey(
        q, 0.90, 1.04, 30.0, "30 m/s", labelpos="E", fontproperties={"size": 9}
    )
    ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)
    cb = fig.colorbar(q, ax=ax, orientation="vertical", pad=0.02, shrink=0.85)
    cb.set_label("Wind speed at peak (m/s)", fontsize=9)
    ax.set_title(
        "Wind vectors at peak intensity — CLOCKWISE circulation (Southern Hemisphere)",
        fontweight="bold",
        fontsize=12,
    )
    ax.annotate(
        "",
        xy=(122.5, -17.0),
        xytext=(118.0, -13.0),
        xycoords=plate._as_mpl_transform(ax),
        arrowprops={
            "arrowstyle": "->,head_width=0.5",
            "color": "red",
            "lw": 2.5,
            "connectionstyle": "arc3,rad=-0.5",
        },
        zorder=8,
    )
    ax.text(
        123.0,
        -14.2,
        "clockwise",
        color="red",
        fontsize=12,
        fontweight="bold",
        transform=plate,
        zorder=8,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--peak", type=Path, default=Path(__file__).parent / "cocoa_output_peak.nc"
    )
    parser.add_argument(
        "--interval", type=Path, default=Path(__file__).parent / "cocoa_output.nc"
    )
    args = parser.parse_args()

    dp = Dataset(args.peak, "r")
    x = np.array(dp.variables["lon"][:])
    y = np.array(dp.variables["lat"][:])
    elements = np.array(dp.variables["element"][:]) - 1
    wind_max = np.array(dp.variables["wind_max"][:])
    zeta_max = np.array(dp.variables["zeta_max"][:])
    pressure_mb = np.where(
        dp.variables["pressure_min"][:] <= COCOA_DRY,
        COCOA_DRY,
        np.array(dp.variables["pressure_min"][:]) * PA_TO_MB,
    )
    dp.close()

    # Peak-intensity wind vectors from the interval file.
    di = Dataset(args.interval, "r")
    wu = np.array(di.variables["wind_u"][:])
    wv = np.array(di.variables["wind_v"][:])
    speed = np.hypot(wu, wv)
    kpeak = int(np.argmax(speed.max(axis=1)))
    di.close()
    print(f"Peak wind at step {kpeak} h, max |V| = {speed[kpeak].max():.1f} m/s")

    globe = ccrs.Robinson(central_longitude=120.0)
    plate = ccrs.PlateCarree()
    tri_globe = triangulation(x, y, elements, globe)
    tri_plate = triangulation(x, y, elements, plate)

    fields = [
        (wind_max, "Peak wind speed (m/s)", "viridis", 0.0, peak_hi(wind_max)),
        (
            pressure_mb,
            "Minimum pressure (mb)",
            "plasma_r",
            float(np.nanmin(pressure_mb[pressure_mb > COCOA_DRY])),
            1013.25,
        ),
        (zeta_max, "Peak water level (m)", "YlGnBu", 0.0, peak_hi(zeta_max)),
    ]

    fig = plt.figure(figsize=(15, 20))
    gs = gridspec.GridSpec(
        4,
        2,
        width_ratios=[1.7, 1.0],
        height_ratios=[1, 1, 1, 1.6],
        hspace=0.16,
        wspace=0.14,
    )

    for row, (vals, label, cmap, vmin, vmax) in enumerate(fields):
        ax_g = fig.add_subplot(gs[row, 0], projection=globe)
        draw_map(ax_g, fig, tri_globe, vals, label, cmap, vmin, vmax)
        ax_g.set_title(label + " -- global", fontweight="bold", fontsize=11)

        ax_i = fig.add_subplot(gs[row, 1], projection=plate)
        ax_i.set_extent(INSET, crs=plate)
        draw_map(ax_i, fig, tri_plate, vals, label, cmap, vmin, vmax, global_view=False)
        ax_i.set_title("NW Australia", fontsize=10, fontweight="bold")

    axq = fig.add_subplot(gs[3, :], projection=plate)
    draw_wind_vectors(fig, axq, (x, y), (wu[kpeak], wv[kpeak]), speed[kpeak], wind_max)

    fig.suptitle(
        "Cocoa — Southern-Hemisphere cyclone (GAHM) impacting NW Australia",
        fontsize=15,
        fontweight="bold",
        y=0.90,
    )

    out = args.peak.parent / "sh_australia_results.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
