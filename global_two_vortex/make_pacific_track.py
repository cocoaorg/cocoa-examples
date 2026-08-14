#!/usr/bin/env python3
"""
Create a Pacific storm track by relocating the Atlantic Katrina best track.

The storm scalars (intensity, pressure, isotach radii, timing) are preserved
verbatim -- only the geographic position is shifted 80 deg west (into the open
central North Pacific) and the basin code is relabeled AL -> CP. This gives a
second, physically identical vortex in a different ocean basin so the two-storm
composition can be exercised on the global mesh.
"""

from __future__ import annotations

from pathlib import Path

SHIFT_DEG_WEST = 80.0  # 75-90 W  ->  155-170 W (open N. Pacific, north of Hawaii)
SRC = Path(__file__).parent / "atlantic_katrina.dat"
DST = Path(__file__).parent / "pacific_shifted.dat"


def shift_longitude(field: str) -> str:
    """Shift an ATCF longitude token (e.g. '751W' = 75.1 W) west by SHIFT_DEG_WEST."""
    token = field.strip()
    hemi = token[-1]
    tenths = int(token[:-1])
    deg = tenths / 10.0
    signed = -deg if hemi == "W" else deg
    signed -= SHIFT_DEG_WEST  # move west (more negative)
    new_hemi = "W" if signed < 0 else "E"
    new_tenths = round(abs(signed) * 10.0)
    return f"{new_tenths}{new_hemi}".rjust(len(field))


def main() -> None:
    lines_out = []
    for line in SRC.read_text().splitlines():
        if not line.strip():
            continue
        cols = line.split(",")
        cols[0] = " CP" if cols[0].startswith(" ") else "CP"  # basin (col 0)
        cols[7] = shift_longitude(cols[7])  # longitude (col 7)
        lines_out.append(",".join(cols))
    DST.write_text("\n".join(lines_out) + "\n")
    print(f"Wrote {DST} ({len(lines_out)} lines)")


if __name__ == "__main__":
    main()
