"""Grid map models and sample map generation.

This module owns the map representation used everywhere else in the project.
The rest of the code assumes:
- coordinates are integer grid cells `(x, y)`,
- only 4-neighbour movement is allowed,
- `heightmap[y, x]` stores the terrain elevation for each cell,
- blocked cells are not traversable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

# All movement and planning is expressed in grid coordinates.
Coord = tuple[int, int]  # (x, y)


@dataclass
class GridMap:
    """2D grid map with elevation, blocked cells, and delivery metadata."""

    width: int
    height: int
    # Height values are stored in row-major order: heightmap[y, x].
    heightmap: np.ndarray  # shape: (height, width)
    # Blocked cells represent buildings or obstacles that the robot cannot enter.
    blocked: set[Coord]
    base: Coord
    deliveries: list[Coord]
    # V2X smart corridors: free cells where traffic-light coordination gives the
    # robot right-of-way priority, reducing effective travel time.
    smart_corridors: set[Coord] = field(default_factory=set)
    # Wireless charging hubs: free cells where the robot can top up its battery
    # mid-mission without returning to base.
    charging_hubs: set[Coord] = field(default_factory=set)
    # Street-type and land-use labels (used for rendering and stats; do not
    # affect planning).
    sidewalks: set[Coord] = field(default_factory=set)
    bike_lanes: set[Coord] = field(default_factory=set)
    roads: set[Coord] = field(default_factory=set)
    plazas: set[Coord] = field(default_factory=set)
    parks: set[Coord] = field(default_factory=set)
    # Per-cell building heights (shape: height×width). None means all flat.
    building_heights: np.ndarray | None = None
    # Cells the robot is allowed to enter.  When non-empty, is_free() checks
    # membership here instead of the blocked set.  Set to `sidewalks` so the
    # robot stays on the pavement.
    traversable: set[Coord] = field(default_factory=set)

    def in_bounds(self, cell: Coord) -> bool:
        """Return True when the coordinate lies inside the grid rectangle."""
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height

    def is_blocked(self, cell: Coord) -> bool:
        """Return True when the cell is occupied by an obstacle/building."""
        return cell in self.blocked

    def is_free(self, cell: Coord) -> bool:
        """A traversable cell must be in-bounds and permitted by the traversable set.

        When `traversable` is populated (city maps restrict movement to
        sidewalks), only cells in that set are reachable.  Otherwise any
        non-blocked cell is free (backwards-compatible with simple maps).
        """
        if not self.in_bounds(cell):
            return False
        if self.traversable:
            return cell in self.traversable
        return not self.is_blocked(cell)

    def neighbors4(self, cell: Coord) -> list[Coord]:
        """Return the four legal cardinal neighbours of a cell.

        This project intentionally disallows diagonal motion, so every planner
        calls this helper instead of generating its own moves.
        """
        x, y = cell
        candidates = ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
        return [n for n in candidates if self.is_free(n)]

    def elevation(self, cell: Coord) -> float:
        """Read the terrain height at a grid cell."""
        x, y = cell
        return float(self.heightmap[y, x])

    def surface_height(self, cell: Coord) -> float:
        """Terrain height at cell (alias used by the 3-D renderer)."""
        return self.elevation(cell)


def _generate_heightmap(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    """Smooth terrain in the range [0, 10] using layered sinusoids + noise."""
    ys, xs = np.mgrid[0:height, 0:width]
    terrain = (
        1.8 * np.sin(xs / 4.5)
        + 1.4 * np.cos(ys / 5.2)
        + 0.6 * np.sin((xs + ys) / 6.8)
        + 0.35 * np.cos((xs - ys) / 9.1)
        + 0.30 * rng.normal(size=(height, width))
    )
    t_min, t_max = float(terrain.min()), float(terrain.max())
    if t_max == t_min:
        return np.zeros_like(terrain)
    return (terrain - t_min) / (t_max - t_min) * 10.0


# ---------------------------------------------------------------------------
# City layout helpers
# ---------------------------------------------------------------------------

def _paint_v(cells: set, x0: int, w: int, width: int, height: int) -> None:
    for y in range(height):
        for x in range(max(0, x0), min(width, x0 + w)):
            cells.add((x, y))


def _paint_h(cells: set, y0: int, h: int, width: int, height: int) -> None:
    for y in range(max(0, y0), min(height, y0 + h)):
        for x in range(width):
            cells.add((x, y))


def _generate_city_layout(
    width: int,
    height: int,
    rng: np.random.Generator,
) -> tuple[
    set[Coord], set[Coord], set[Coord], set[Coord],
    set[Coord], set[Coord], set[Coord],
]:
    """Return (blocked, roads, sidewalks, bike_lanes, plazas, parks, smart_corridors).

    Street structure
    ----------------
    Major avenues  – 2-lane road + 1-cell sidewalk each side + 1-cell bike lane
                     each side = 6 cells wide.  Three vertical, three horizontal.
    Local streets  – 1-lane road + 1-cell sidewalk each side = 3 cells wide.
    V2X corridors  – bike lanes on the central major cross (width//2 vertical +
                     height//2 horizontal) only.
    Parks / plazas – built from the computed blocked set so boundaries are exact.
    """
    ROAD_M, SWALK_M, BIKE_M = 2, 1, 1
    ROAD_L, SWALK_L         = 1, 1

    roads:           set[Coord] = set()
    sidewalks:       set[Coord] = set()
    bike_lanes:      set[Coord] = set()
    plazas:          set[Coord] = set()
    parks:           set[Coord] = set()
    smart_corridors: set[Coord] = set()

    major_xs = sorted({width // 4, width // 2, 3 * width // 4})
    major_ys = sorted({height // 4, height // 2, 3 * height // 4})

    # Exact footprint helpers (first cell, exclusive-end cell)
    def mv_foot(cx: int) -> tuple[int, int]:
        r = cx - ROAD_M // 2
        return (r - SWALK_M - BIKE_M, r + ROAD_M + SWALK_M + BIKE_M)

    def mh_foot(cy: int) -> tuple[int, int]:
        r = cy - ROAD_M // 2
        return (r - SWALK_M - BIKE_M, r + ROAD_M + SWALK_M + BIKE_M)

    def lv_foot(cx: int) -> tuple[int, int]:
        return (cx - SWALK_L, cx + ROAD_L + SWALK_L)

    def lh_foot(cy: int) -> tuple[int, int]:
        return (cy - SWALK_L, cy + ROAD_L + SWALK_L)

    # Columns/rows occupied by major avenues (to exclude local streets)
    occ_cols: set[int] = set()
    for cx in major_xs:
        a, b = mv_foot(cx)
        occ_cols |= set(range(max(0, a), min(width, b)))
    occ_rows: set[int] = set()
    for cy in major_ys:
        a, b = mh_foot(cy)
        occ_rows |= set(range(max(0, a), min(height, b)))

    # Local streets in the gaps between major avenues
    sp_x = max(7, width // 9)
    sp_y = max(6, height // 8)
    off_x = int(rng.integers(2, max(3, sp_x - 2)))
    off_y = int(rng.integers(2, max(3, sp_y - 2)))

    local_xs = [
        x for x in range(off_x, width - 2, sp_x)
        if x not in occ_cols and x - 1 not in occ_cols and x + 1 not in occ_cols
    ]
    local_ys = [
        y for y in range(off_y, height - 2, sp_y)
        if y not in occ_rows and y - 1 not in occ_rows and y + 1 not in occ_rows
    ]

    # Paint major vertical avenues
    for cx in major_xs:
        r = cx - ROAD_M // 2
        _paint_v(roads,      r,                   ROAD_M,  width, height)
        _paint_v(sidewalks,  r - SWALK_M,          SWALK_M, width, height)
        _paint_v(sidewalks,  r + ROAD_M,           SWALK_M, width, height)
        _paint_v(bike_lanes, r - SWALK_M - BIKE_M, BIKE_M,  width, height)
        _paint_v(bike_lanes, r + ROAD_M + SWALK_M, BIKE_M,  width, height)

    # Paint major horizontal avenues
    for cy in major_ys:
        r = cy - ROAD_M // 2
        _paint_h(roads,      r,                   ROAD_M,  width, height)
        _paint_h(sidewalks,  r - SWALK_M,          SWALK_M, width, height)
        _paint_h(sidewalks,  r + ROAD_M,           SWALK_M, width, height)
        _paint_h(bike_lanes, r - SWALK_M - BIKE_M, BIKE_M,  width, height)
        _paint_h(bike_lanes, r + ROAD_M + SWALK_M, BIKE_M,  width, height)

    # Paint local vertical streets
    for cx in local_xs:
        _paint_v(roads,     cx,           ROAD_L,  width, height)
        _paint_v(sidewalks, cx - SWALK_L, SWALK_L, width, height)
        _paint_v(sidewalks, cx + ROAD_L,  SWALK_L, width, height)

    # Paint local horizontal streets
    for cy in local_ys:
        _paint_h(roads,     cy,           ROAD_L,  width, height)
        _paint_h(sidewalks, cy - SWALK_L, SWALK_L, width, height)
        _paint_h(sidewalks, cy + ROAD_L,  SWALK_L, width, height)

    # Resolve street overlaps before computing blocked
    sidewalks  -= roads
    bike_lanes -= roads | sidewalks

    # Pedestrian crossings: where a sidewalk column meets a road row (or a
    # sidewalk row meets a road column) the cell is a crossing — without these
    # the sidewalk network is disconnected across every avenue.
    sidewalk_xs = {x for x, _ in sidewalks}
    sidewalk_ys = {y for _, y in sidewalks}
    crossings = {(x, y) for x, y in roads
                 if x in sidewalk_xs or y in sidewalk_ys}
    sidewalks |= crossings
    roads -= crossings

    # V2X corridors: the SIDEWALK cells alongside the central major avenues.
    # The robot travels on sidewalks, so V2X priority must live there too.
    v2x_cx, v2x_cy = width // 2, height // 2
    rc = v2x_cx - ROAD_M // 2          # road-start col of central vertical
    rr = v2x_cy - ROAD_M // 2          # road-start row of central horizontal
    v2x_x_cols = {rc - SWALK_M, rc + ROAD_M}   # sidewalk cols of central vert
    v2x_y_rows = {rr - SWALK_M, rr + ROAD_M}   # sidewalk rows of central horiz
    smart_corridors = {
        (x, y) for x, y in sidewalks
        if x in v2x_x_cols or y in v2x_y_rows
    }

    # Build blocked set (everything not on a street)
    active  = roads | sidewalks | bike_lanes
    blocked = {(x, y) for y in range(height) for x in range(width)
               if (x, y) not in active}

    # Plazas: 3×3 open corners in building blocks at every major intersection.
    # Placed just outside the avenue footprint so they are currently in `blocked`.
    for cx in major_xs:
        a_cx, b_cx = mv_foot(cx)   # [a_cx, b_cx) = avenue footprint
        for cy in major_ys:
            a_cy, b_cy = mh_foot(cy)
            corners = [
                (b_cx,     b_cy),      # NE: right of vert, above horiz
                (a_cx - 3, b_cy),      # NW
                (b_cx,     a_cy - 3),  # SE
                (a_cx - 3, a_cy - 3),  # SW
            ]
            for px0, py0 in corners:
                for dy in range(3):
                    for dx in range(3):
                        px, py = px0 + dx, py0 + dy
                        if 0 <= px < width and 0 <= py < height:
                            plazas.add((px, py))

    # Parks: find building-block regions using exact footprint gaps, then
    # convert the widest gaps into park cells.
    v_footprints = sorted(
        [mv_foot(cx) for cx in major_xs] + [lv_foot(cx) for cx in local_xs]
    )
    h_footprints = sorted(
        [mh_foot(cy) for cy in major_ys] + [lh_foot(cy) for cy in local_ys]
    )

    def _gaps(footprints: list[tuple[int, int]], limit: int) -> list[tuple[int, int]]:
        gaps = []
        prev = 0
        for lo, hi in footprints:
            lo = max(0, lo)
            if lo > prev:
                gaps.append((prev, lo))
            prev = max(prev, min(limit, hi))
        if prev < limit:
            gaps.append((prev, limit))
        return [(a, b) for a, b in gaps if b - a >= 3]

    v_gaps = _gaps(v_footprints, width)
    h_gaps = _gaps(h_footprints, height)

    park_candidates = [
        (x0, y0, x1, y1)
        for x0, x1 in v_gaps
        for y0, y1 in h_gaps
        if x1 - x0 >= 3 and y1 - y0 >= 3
    ]
    n_parks = max(2, len(park_candidates) // 4)
    if park_candidates:
        chosen = rng.choice(
            len(park_candidates),
            size=min(n_parks, len(park_candidates)),
            replace=False,
        )
        for idx in chosen:
            x0, y0, x1, y1 = park_candidates[int(idx)]
            for py in range(y0, y1):
                for px in range(x0, x1):
                    parks.add((px, py))

    # Merge parks and plazas into the traversable set
    plazas -= active | parks      # keep only cells currently in blocked
    parks  -= active              # same
    blocked -= plazas | parks

    return blocked, roads, sidewalks, bike_lanes, plazas, parks, smart_corridors


def _generate_building_heights(
    width: int,
    height: int,
    blocked: set[Coord],
    heightmap: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Building heights driven by a downtown gradient plus random variation.

    Returns an (height × width) array where non-blocked cells are 0.
    Values for blocked cells range ~3–52; the renderer scales them by 0.45
    to get world-unit heights (≈1.4–23 floors).
    """
    ys, xs = np.mgrid[0:height, 0:width]
    cx, cy = width / 2.0, height / 2.0
    downtown = np.exp(
        -(((xs - cx) / (width * 0.22)) ** 2
          + ((ys - cy) / (height * 0.20)) ** 2)
    )
    noise = 0.22 * rng.normal(size=(height, width))
    raw = np.clip(4.0 + 48.0 * downtown + 6.0 * noise, 3.0, 52.0)

    mask = np.zeros((height, width), dtype=bool)
    for x, y in blocked:
        mask[y, x] = True
    return np.where(mask, raw, 0.0)


def generate_sample_map(
    width: int = 60,
    height: int = 42,
    obstacle_ratio: float = 0.20,
    num_deliveries: int = 8,
    seed: int = 7,
) -> GridMap:
    """Generate a city-grid map with terrain elevation, streets, and buildings.

    The layout follows a real city pattern:
    - Three major avenues in each axis (wide, with bike lanes).
    - Local streets filling the gaps.
    - Building blocks between streets, taller downtown.
    - Parks replace a few interior blocks with greenery.
    - Plazas open up major intersections.
    - V2X smart corridors on the central major cross only.
    - Four charging hubs on V2X corridors near major intersections.
    """
    if width <= 1 or height <= 1:
        raise ValueError("width and height must be > 1")
    if num_deliveries < 1:
        raise ValueError("num_deliveries must be >= 1")

    rng = np.random.default_rng(seed)
    heightmap = _generate_heightmap(width, height, rng)

    blocked, roads, sidewalks, bike_lanes, plazas, parks, smart_corridors = (
        _generate_city_layout(width, height, rng)
    )
    building_heights = _generate_building_heights(
        width, height, blocked, heightmap, rng
    )

    # Scatter additional random obstacles across open areas (roads/sidewalks/parks)
    # to match the requested obstacle density.
    if obstacle_ratio > 0:
        open_cells = [
            (x, y)
            for y in range(height)
            for x in range(width)
            if (x, y) not in blocked
        ]
        n_extra = int(len(open_cells) * obstacle_ratio)
        rng.shuffle(open_cells)
        for cell in open_cells[:n_extra]:
            blocked.add(cell)
        # Keep sets consistent after adding extra blocked cells.
        sidewalks  -= blocked
        bike_lanes -= blocked
        roads      -= blocked
        plazas     -= blocked
        parks      -= blocked
        smart_corridors -= blocked

    # All non-blocked cells are freely traversable.
    free_list = sorted(
        (x, y)
        for y in range(height)
        for x in range(width)
        if (x, y) not in blocked
    )

    # Base station: nearest free cell to the bottom-left corner.
    base = min(free_list, key=lambda c: abs(c[0] - 2) + abs(c[1] - 2))

    # Delivery points: spread across free cells, well away from the base.
    min_dist = max(8, (width + height) // 6)
    candidates = [
        c for c in free_list
        if c != base and abs(c[0] - base[0]) + abs(c[1] - base[1]) >= min_dist
    ]
    if len(candidates) < num_deliveries:
        candidates = [c for c in free_list if c != base]
    if len(candidates) < num_deliveries:
        raise ValueError("Not enough free cells for the requested deliveries.")

    deliveries: list[Coord] = []
    pool = list(candidates)
    rng.shuffle(pool)
    while len(deliveries) < num_deliveries and pool:
        cell = pool.pop()
        if all(abs(cell[0] - d[0]) + abs(cell[1] - d[1]) >= min_dist // 2
               for d in deliveries):
            deliveries.append(cell)
    remaining = [c for c in candidates if c not in deliveries]
    rng.shuffle(remaining)
    for cell in remaining:
        if len(deliveries) >= num_deliveries:
            break
        deliveries.append(cell)

    # Charging hubs: on V2X sidewalk cells near major intersections.
    major_xs = sorted({width // 4, width // 2, 3 * width // 4})
    major_ys = sorted({height // 4, height // 2, 3 * height // 4})
    reserved = {base, *deliveries}
    hub_targets = [
        (width // 2, major_ys[0]),
        (width // 2, major_ys[-1]),
        (major_xs[0], height // 2),
        (major_xs[-1], height // 2),
    ]
    charging_hubs: set[Coord] = set()
    v2x_swalk = sorted(smart_corridors)   # V2X corridors are already sidewalks
    for tx, ty in hub_targets:
        pool2 = [c for c in v2x_swalk if c not in reserved and c not in charging_hubs]
        if pool2:
            hub = min(pool2, key=lambda c: abs(c[0] - tx) + abs(c[1] - ty))
            charging_hubs.add(hub)

    return GridMap(
        width=width,
        height=height,
        heightmap=heightmap,
        blocked=blocked,
        base=base,
        deliveries=deliveries,
        smart_corridors=smart_corridors,
        charging_hubs=charging_hubs,
        sidewalks=sidewalks,
        bike_lanes=bike_lanes,
        roads=roads,
        plazas=plazas,
        parks=parks,
        building_heights=building_heights,
    )


def format_cells(cells: Iterable[Coord]) -> str:
    """Human-friendly `(x,y)` formatting for console output."""
    return ", ".join(f"({x},{y})" for x, y in cells)
