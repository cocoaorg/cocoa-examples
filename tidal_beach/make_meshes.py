#!/usr/bin/env python3
"""
Write the sloping-beach meshes for the tidal_beach example.

The domain is a straight basin whose bed rises linearly from the open
boundary at its western end to dry land at its eastern end, so a tide moves
the shoreline back and forth across it. The same nodes are written three
times, as triangles, as quadrilaterals, and as a hybrid of the two:

    beach_triangle.nc   every cell split into two triangles
    beach_quad.nc       every cell a quadrilateral
    beach_hybrid.nc     quadrilaterals in the middle third of the basin,
                        triangles at either end

Usage:
    python make_meshes.py               # writes the three meshes here
    python make_meshes.py --outdir DIR

The wave_beach example imports write_beach_mesh() to build its own, longer
beach.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

EARTH_RADIUS_M = 6378206.4
FILL = -9999

OPEN_BOUNDARY = -1  # elevation-specified (ADCIRC IBTYPE -1)
LAND_BOUNDARY = 20  # mainland, natural (ADCIRC IBTYPE 20)


@dataclass(frozen=True)
class Beach:
    """A straight basin with a bed that rises linearly from west to east."""

    length_m: float = 8000.0  # west (open boundary) to east (land)
    width_m: float = 2000.0
    cell_m: float = 100.0
    depth_open_m: float = 10.0  # still-water depth at the open boundary
    depth_land_m: float = -3.0  # negative: 3 m above still water
    origin: tuple[float, float] = (-90.0, 29.0)  # lon, lat of the SW corner


def node_grid(beach: Beach) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Node positions (lon, lat) and depth, row-major: station i, then j."""
    ni = round(beach.length_m / beach.cell_m) + 1
    nj = round(beach.width_m / beach.cell_m) + 1
    xm = np.linspace(0.0, beach.length_m, ni)
    ym = np.linspace(0.0, beach.width_m, nj)
    xx, yy = np.meshgrid(xm, ym, indexing="ij")
    lon0, lat0 = beach.origin
    lat = lat0 + np.degrees(yy / EARTH_RADIUS_M)
    lon = lon0 + np.degrees(xx / (EARTH_RADIUS_M * math.cos(math.radians(lat0))))
    slope = (beach.depth_open_m - beach.depth_land_m) / beach.length_m
    depth = beach.depth_open_m - slope * xx
    return lon.ravel(), lat.ravel(), depth.ravel()


def connectivity(ni: int, nj: int, variant: str) -> np.ndarray:
    """Counter-clockwise element rows, padded to four columns with FILL."""
    ncell_i = ni - 1
    band = ncell_i // 3
    rows = []
    for i in range(ncell_i):
        quad = variant == "quad" or (variant == "hybrid" and band <= i < ncell_i - band)
        for j in range(nj - 1):
            c0 = i * nj + j
            c1 = (i + 1) * nj + j
            c2 = (i + 1) * nj + j + 1
            c3 = i * nj + j + 1
            if quad:
                rows.append([c0, c1, c2, c3])
            else:
                rows.append([c0, c1, c2, FILL])
                rows.append([c0, c2, c3, FILL])
    return np.asarray(rows, dtype=np.int32)


def boundaries(ni: int, nj: int) -> list[tuple[int, list[int]]]:
    """Open boundary across the west end; land walls on the other three sides."""
    west = [j for j in range(nj)]
    south = [i * nj for i in range(ni)]
    north = [i * nj + nj - 1 for i in range(ni)]
    east = [(ni - 1) * nj + j for j in range(nj)]
    return [
        (OPEN_BOUNDARY, west),
        (LAND_BOUNDARY, south),
        (LAND_BOUNDARY, north),
        (LAND_BOUNDARY, east),
    ]


def write_beach_mesh(path: Path, beach: Beach, variant: str) -> None:
    lon, lat, depth = node_grid(beach)
    ni = round(beach.length_m / beach.cell_m) + 1
    nj = round(beach.width_m / beach.cell_m) + 1
    rows = connectivity(ni, nj, variant)
    max_face_nodes = 3 if variant == "triangle" else 4

    with Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("nnode", lon.size)
        ds.createDimension("nface", rows.shape[0])
        ds.createDimension("max_face_nodes", max_face_nodes)
        ds.createVariable("x", "f8", ("nnode",))[:] = lon
        ds.createVariable("y", "f8", ("nnode",))[:] = lat
        ds.createVariable("depth", "f8", ("nnode",))[:] = depth
        conn = ds.createVariable(
            "mesh_face_nodes", "i4", ("nface", "max_face_nodes"), fill_value=FILL
        )
        conn.cf_role = "face_node_connectivity"
        conn.start_index = 0
        conn[:, :] = rows[:, :max_face_nodes]

        groups = boundaries(ni, nj)
        nodes = [n for _, ids in groups for n in ids]
        ends = np.cumsum([len(ids) for _, ids in groups])
        bg = ds.createGroup("boundaries")
        bg.createDimension("nboundary", len(groups))
        bg.createDimension("nboundary_node", len(nodes))
        bg.createVariable("boundary_types", "i4", ("nboundary",))[:] = [
            code for code, _ in groups
        ]
        bg.createVariable("boundary_start_index", "i4", ("nboundary",))[:] = (
            np.concatenate([[0], ends[:-1]])
        )
        bg.createVariable("boundary_end_index", "i4", ("nboundary",))[:] = ends
        bg.createVariable("boundary_nodes", "i4", ("nboundary_node",))[:] = nodes

        ds.Conventions = "UGRID-1.0"
        ds.title = f"Sloping beach, {variant} elements"
        ds.source = "cocoa-examples make_meshes.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--outdir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    beach = Beach()
    for variant in ("triangle", "quad", "hybrid"):
        path = args.outdir / f"beach_{variant}.nc"
        write_beach_mesh(path, beach, variant)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
