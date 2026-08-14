#!/usr/bin/env python3
"""
Author a synthetic Southern-Hemisphere tropical cyclone impacting NW Australia.

Hand-drawn best track for a Pilbara-coast landfalling cyclone (Timor Sea ->
North West Shelf -> coast near Port Hedland). The storm intensifies over the
warm shelf water to a strong system, then makes landfall and decays. Basin is
SH and every latitude is south, so the GAHM hemisphere handling (Coriolis sign,
clockwise tangential rotation) is exercised end-to-end.

The quadrant isotach radii are deliberately asymmetric (larger on the
equatorward / eastern flank) so the vortex is not radially symmetric and the
rotation sense is unambiguous in the wind field.

Emits standard ATCF b-deck lines (one per 34/50/64 kt isotach present at each
synoptic time) to australia_cyclone.dat.
"""

from __future__ import annotations

from pathlib import Path

DST = Path(__file__).parent / "australia_cyclone.dat"

# (YYYYMMDDHH, lat_south_deg, lon_east_deg, vmax_kt, mslp_mb, rmw_nmi,
#  {thresh_kt: (NE, SE, SW, NW) radii in nmi})
TRACK = [
    ("2024021000", 12.0, 122.0, 35, 1000, 30, {34: (90, 100, 60, 70)}),
    ("2024021006", 12.8, 121.3, 45, 994, 30, {34: (100, 110, 70, 80)}),
    (
        "2024021012",
        13.6,
        120.6,
        55,
        984,
        25,
        {34: (110, 120, 80, 90), 50: (50, 55, 35, 40)},
    ),
    (
        "2024021018",
        14.4,
        120.0,
        70,
        971,
        22,
        {34: (120, 130, 85, 95), 50: (60, 65, 40, 45), 64: (30, 35, 20, 25)},
    ),
    (
        "2024021100",
        15.2,
        119.4,
        90,
        953,
        18,
        {34: (130, 140, 90, 100), 50: (70, 75, 45, 55), 64: (40, 45, 28, 32)},
    ),
    (
        "2024021106",
        16.0,
        118.9,
        105,
        938,
        16,
        {34: (140, 150, 95, 110), 50: (80, 85, 50, 60), 64: (48, 52, 32, 38)},
    ),
    (
        "2024021112",
        16.9,
        118.5,
        115,
        928,
        14,
        {34: (150, 160, 100, 115), 50: (85, 90, 55, 65), 64: (52, 58, 35, 42)},
    ),
    (
        "2024021118",
        17.8,
        118.2,
        110,
        933,
        15,
        {34: (145, 155, 98, 112), 50: (82, 88, 52, 62), 64: (50, 55, 33, 40)},
    ),
    (
        "2024021200",
        18.7,
        118.0,
        95,
        949,
        18,
        {34: (135, 145, 92, 105), 50: (72, 78, 46, 56), 64: (42, 46, 29, 34)},
    ),
    (
        "2024021206",
        19.6,
        117.9,
        70,
        967,
        22,
        {34: (120, 130, 82, 95), 50: (58, 63, 38, 46)},
    ),
    ("2024021212", 20.6, 118.0, 45, 984, 28, {34: (100, 110, 68, 80)}),
    ("2024021218", 21.6, 118.2, 30, 995, 34, {}),
]


# Storm type by intensity, matching ATCF conventions (SH systems still use the
# generic TD/TS/HU/TY-agnostic intensity classes here for simplicity).
def storm_type(vmax: int) -> str:
    if vmax < 34:
        return "DB"
    if vmax < 64:
        return "TS"
    return "HU"


def fmt_lat(lat_south: float) -> str:
    return f"{round(lat_south * 10)}S"


def fmt_lon(lon_east: float) -> str:
    return f"{round(lon_east * 10)}E"


def emit_line(
    when: str,
    lat: float,
    lon: float,
    vmax: int,
    mslp: int,
    rmw: int,
    thresh: int,
    radii: tuple[int, int, int, int],
) -> str:
    ne, se, sw, nw = radii
    fields = [
        "SH",
        "05",
        when,
        "  ",
        "BEST",
        "  0",
        fmt_lat(lat).rjust(5),
        fmt_lon(lon).rjust(6),
        f"{vmax}".rjust(4),
        f"{mslp}".rjust(5),
        storm_type(vmax).rjust(3),
        f"{thresh}".rjust(4),
        "NEQ",
        f"{ne}".rjust(5),
        f"{se}".rjust(5),
        f"{sw}".rjust(5),
        f"{nw}".rjust(5),
        "1008".rjust(5),
        "200".rjust(5),
        f"{rmw}".rjust(4),
        "0".rjust(4),
        "0".rjust(4),
        " ",
        "0".rjust(4),
        "  ",
        "0".rjust(4),
        "0".rjust(4),
        "ANIKA".rjust(11),
        "S",
    ]
    return ", ".join(fields)


def main() -> None:
    lines = []
    for when, lat, lon, vmax, mslp, rmw, isotachs in TRACK:
        if not isotachs:
            # No closed isotach (weak system): emit a single 0-threshold line so
            # the snapshot still carries position/intensity/pressure/RMW.
            lines.append(emit_line(when, lat, lon, vmax, mslp, rmw, 0, (0, 0, 0, 0)))
            continue
        for thresh in sorted(isotachs):
            lines.append(
                emit_line(when, lat, lon, vmax, mslp, rmw, thresh, isotachs[thresh])
            )
    DST.write_text("\n".join(lines) + "\n")
    print(f"Wrote {DST} ({len(lines)} lines, {len(TRACK)} synoptic times)")


if __name__ == "__main__":
    main()
