"""The contested urban sector: geometry, emitters and free-space queries.

Single source of truth for the layout, with no simulator dependency. Five
consumers read it, and they must all agree or the mission is incoherent:

  * `planner`          builds its occupancy grid from `building_boxes()`
  * `kinematic_platform` collides and ray-casts against the same boxes
  * `threat_field`     places emitters and tests RF occlusion
  * `swarm_viz`        draws the buildings in RViz
  * `coverage`         excludes cells the UGV cannot reach

Previously the layout lived in a generated Gazebo `.world` file, with the
emitter table duplicated into `mission.yaml` and the building list duplicated
again into a launch file. Three copies drifted apart, and two bugs came
directly out of that: the insertion and rally points sat inside building
footprints, and all four emitters were buried inside building volumes.
`validate()` below now makes either mistake a hard error.

Layout
======
120 m x 120 m. A main route runs east-west along y = 0 and a cross street
north-south along x = 0. The rest is built-up blocks of four buildings around
an internal courtyard, with 4 m alleys between them. Those courtyards are
where the emitters hide: concealed from the route, reachable on foot, visible
from altitude.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

SECTOR_HALF = 60.0
#: Cells, markers and tasks stay inside this; the belief grid uses SECTOR_HALF.
SEARCH_HALF = 55.0


@dataclass(frozen=True)
class Box:
    """Axis-aligned footprint. `height` matters only to the RF occlusion test."""

    name: str
    x: float
    y: float
    half_x: float
    half_y: float
    height: float

    def contains(self, px: float, py: float, margin: float = 0.0) -> bool:
        return (
            abs(px - self.x) <= self.half_x + margin
            and abs(py - self.y) <= self.half_y + margin
        )


@dataclass(frozen=True)
class Emitter:
    """Ground truth site of a hostile or civilian emitter.

    `kind` drives both the RF/CBRN signature and the class the swarm must
    arrive at independently:
      jammer       high power, denies GNSS and comms inside `jam_radius`,
                   no chemical signature === engageable
      ied_trigger  low power, intermittent, chemical precursor present
                   === engageable
      civilian     high power, no jamming, no chemical signature === NO-STRIKE

    The jammer and the civilian transmitter are deliberately indistinguishable
    on RF power alone. What separates them is that the jammer degrades the
    comms of anything near it, which is the discriminator the team has to go
    and measure.
    """

    name: str
    x: float
    y: float
    z: float
    kind: str
    power_dbm: float
    freq_mhz: float
    duty_cycle: float = 1.0
    jam_radius: float = 0.0

    @property
    def position(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


# ============================== block layout =================================
# Every number here is derived so that the routes are provably open, and
# `validate()` enforces it. The previous layout looked clear but was not: with
# block centres at |c| = 14 and buildings offset 8 m, a building face landed at
# y = -6 and its footprint ran to y = 0, straight through the "main route".
#
#   building half-extent  4.5 m  -> 9 m square
#   offset within a block 6.5 m  -> block half-extent 11 m, 4 m courtyard
#   ring centres  +/-20, +/-46   -> blocks span 9..31 and 35..57
#
# which gives an 18 m main route (|y| < 9), an 18 m cross street (|x| < 9),
# 4 m streets between the rings, 4 m alleys inside each block, and an outer
# edge at 57 m inside the 60 m sector.
BUILDING_HALF = 4.5
BUILDING_OFFSET = 6.5
BLOCK_HALF = BUILDING_OFFSET + BUILDING_HALF
RING_CENTRES = (-46.0, -20.0, 20.0, 46.0)

#: Half-width of the genuinely clear band along each route.
ROUTE_HALF_WIDTH = min(abs(c) for c in RING_CENTRES) - BLOCK_HALF

BLOCK_CENTRES = tuple(
    (cx, cy) for cx in RING_CENTRES for cy in RING_CENTRES
)

_BUILDING_OFFSETS = (
    (-BUILDING_OFFSET, -BUILDING_OFFSET), (BUILDING_OFFSET, -BUILDING_OFFSET),
    (-BUILDING_OFFSET, BUILDING_OFFSET), (BUILDING_OFFSET, BUILDING_OFFSET),
)
_HEIGHTS = (8.0, 14.0, 21.0, 11.0, 17.0, 26.0)


def _make_buildings() -> tuple[Box, ...]:
    out: list[Box] = []
    for seed, (cx, cy) in enumerate(BLOCK_CENTRES):
        for i, (dx, dy) in enumerate(_BUILDING_OFFSETS):
            out.append(
                Box(
                    name=f"blk{seed}_{i}",
                    x=cx + dx,
                    y=cy + dy,
                    half_x=BUILDING_HALF,
                    half_y=BUILDING_HALF,
                    height=_HEIGHTS[(seed + i) % len(_HEIGHTS)],
                )
            )
    return tuple(out)


BUILDINGS = _make_buildings()

#: Tallest structure in the sector. The drone's transit altitude is derived
#: from this rather than chosen: the planner gives the aircraft straight lines,
#: and the platform's collision test is height-aware, so straight-line flight
#: is only valid above everything it could hit.
MAX_STRUCTURE_HEIGHT = max(b.height for b in BUILDINGS)

#: A collapsed span across the east end of the main route, and a blocked
#: cross-street junction. Low enough that the drone overflies them and the
#: quadruped picks through, but impassable to the UGV === which is what forces
#: the allocator to hand the eastern cells to another platform.
RUBBLE = (
    Box("rubble_route_e", 33.0, 2.0, 3.5, 5.0, 2.2),
    Box("rubble_route_e2", 40.0, -3.0, 3.0, 3.5, 1.4),
    Box("rubble_cross_s", -3.0, -33.0, 4.0, 3.0, 1.8),
)

#: (x, y, radius) regions the UGV must not be routed into.
UGV_EXCLUSIONS = tuple((b.x, b.y, max(b.half_x, b.half_y) + 5.0) for b in RUBBLE)

#: Emitters, each in a block courtyard: concealed from the route at ground
#: level, with line of sight to it, reachable through the alleys, and visible
#: from altitude. Validated against the building list below.
EMITTERS = (
    Emitter("e_jammer", 46.0, 20.0, 1.2, "jammer", 30.0, 433.0,
            duty_cycle=1.0, jam_radius=38.0),
    Emitter("e_trig_sw", -20.0, -20.0, 0.6, "ied_trigger", 14.0, 433.5,
            duty_cycle=0.55),
    Emitter("e_trig_n", 20.0, 46.0, 0.6, "ied_trigger", 16.0, 432.5,
            duty_cycle=0.65),
    Emitter("e_civ", -46.0, 20.0, 2.4, "civilian", 24.0, 435.0,
            duty_cycle=1.0),
)

#: The force enters along the main route from the west and recovers up the
#: cross street to the north. Both are on open ground by construction.
INSERTION = (-52.0, 0.0)
RALLY = (0.0, 52.0)
#: Loose trigger for leaving INGRESS: a point further along the main route.
INGRESS_WAYPOINT = (-30.0, 0.0)

#: Insertion poses as (x, y, z, yaw), abreast on the route facing east.
SPAWN_POSES = {
    # The aircraft inserts at transit altitude, above every structure.
    "drone_1": (INSERTION[0], INSERTION[1] + 3.5, MAX_STRUCTURE_HEIGHT + 5.0, 0.0),
    "ugv_1": (INSERTION[0], INSERTION[1], 0.21, 0.0),
    "dog_1": (INSERTION[0], INSERTION[1] - 3.5, 0.419, 0.0),
}


# ============================ free-space queries =============================
def obstacles(include_rubble: bool = True) -> tuple[Box, ...]:
    return BUILDINGS + RUBBLE if include_rubble else BUILDINGS


def building_boxes(include_rubble: bool = True) -> list[tuple[float, float, float, float]]:
    """Flat (x, y, half_x, half_y) tuples, the form the consumers want."""
    return [(b.x, b.y, b.half_x, b.half_y) for b in obstacles(include_rubble)]


def blocked(x: float, y: float, margin: float = 0.0,
            include_rubble: bool = True) -> bool:
    """True if (x, y) lies in an obstacle, inflated by `margin`."""
    return any(b.contains(x, y, margin) for b in obstacles(include_rubble))


def blocking(x: float, y: float, margin: float = 0.0,
             include_rubble: bool = True) -> Box | None:
    for b in obstacles(include_rubble):
        if b.contains(x, y, margin):
            return b
    return None


def blocked_at_height(x: float, y: float, height: float,
                      margin: float = 0.0) -> bool:
    """True if a structure taller than `height` covers (x, y).

    What an aircraft needs in order to decide whether it may descend. A
    footprint test alone is the wrong question for something flying above the
    rooftops, and descending onto a block the platform then cannot translate
    out of is how the drone froze mid-mission.
    """
    return any(
        b.height > height and b.contains(x, y, margin) for b in obstacles()
    )


def blocked_at_height(x: float, y: float, z: float, margin: float = 0.0,
                      clearance: float = 1.0) -> bool:
    """True if a structure reaching altitude `z` covers (x, y).

    The aircraft's clearance question is three-dimensional: at 31 m nothing in
    this sector obstructs it, at 7 m almost every building does. Treating the
    footprint alone as an obstacle froze the drone the moment it descended
    over a block; ignoring height entirely let it descend into one.
    """
    return any(
        b.contains(x, y, margin) and b.height > z - clearance
        for b in obstacles()
    )


def nearest_free(x: float, y: float, margin: float = 0.8,
                 step: float = 1.0, max_radius: float = 20.0) -> tuple[float, float]:
    """Closest point to (x, y) that is not inside an obstacle.

    Task positions come from the belief grid and can land inside a building, in
    which case no platform can ever reach the acceptance radius and the task
    blocks the allocator forever. Nudging the goal to reachable ground is the
    difference between a stuck robot and a completed task.
    """
    if not blocked(x, y, margin):
        return (x, y)
    radius = step
    while radius <= max_radius:
        # 16 directions is enough to find the way out of a 12 m building.
        for k in range(16):
            theta = 2.0 * math.pi * k / 16.0
            cx = x + radius * math.cos(theta)
            cy = y + radius * math.sin(theta)
            if abs(cx) > SEARCH_HALF or abs(cy) > SEARCH_HALF:
                continue
            if not blocked(cx, cy, margin):
                return (cx, cy)
        radius += step
    return (x, y)


def segment_occlusions(a: tuple[float, float, float],
                       b: tuple[float, float, float]) -> int:
    """Number of buildings the a-b ray passes through.

    Slab test per box. Treated as planar, with a height gate: an observer above
    10 m is given clear line of sight, which is the entire point of putting a
    sensor on an aircraft.
    """
    if min(a[2], b[2]) > 10.0:
        return 0
    dx, dy = b[0] - a[0], b[1] - a[1]
    hits = 0
    for box in obstacles():
        t_lo, t_hi = 0.0, 1.0
        ok = True
        for origin, delta, centre, half in (
            (a[0], dx, box.x, box.half_x), (a[1], dy, box.y, box.half_y)
        ):
            if abs(delta) < 1e-9:
                if not (centre - half <= origin <= centre + half):
                    ok = False
                    break
                continue
            t1 = (centre - half - origin) / delta
            t2 = (centre + half - origin) / delta
            if t1 > t2:
                t1, t2 = t2, t1
            t_lo = max(t_lo, t1)
            t_hi = min(t_hi, t2)
            if t_lo > t_hi:
                ok = False
                break
        if ok:
            hits += 1
    return hits


# ================================ validation =================================
def validate(spawn_clearance: float = 1.5) -> list[str]:
    """Return a list of layout problems; empty means the layout is sound.

    Both failures this catches were real and cost significant debugging time:
    the insertion and rally points were inside the western corner buildings
    (a platform spawned intersecting a wall could rotate but not translate),
    and all four emitters were inside building volumes, where no thermal
    confirm and no CBRN sample is physically possible.
    """
    problems: list[str] = []

    for name, (x, y, _z, _yaw) in SPAWN_POSES.items():
        hit = blocking(x, y, spawn_clearance)
        if hit:
            problems.append(f"spawn {name} at ({x}, {y}) intersects {hit.name}")

    for label, (x, y) in (("insertion", INSERTION), ("rally", RALLY)):
        hit = blocking(x, y, spawn_clearance)
        if hit:
            problems.append(f"{label} point at ({x}, {y}) intersects {hit.name}")

    for e in EMITTERS:
        hit = blocking(e.x, e.y)
        if hit:
            problems.append(f"emitter {e.name} at ({e.x}, {e.y}) is inside {hit.name}")

    ix, iy = INGRESS_WAYPOINT
    if blocked(ix, iy, spawn_clearance):
        problems.append(f"ingress waypoint ({ix}, {iy}) is not on open ground")

    return problems


if __name__ == "__main__":
    import sys

    issues = validate()
    print(f"{len(BUILDINGS)} buildings, {len(RUBBLE)} rubble, {len(EMITTERS)} emitters")
    for e in EMITTERS:
        print(f"  emitter {e.name:10s} ({e.x:6.1f},{e.y:6.1f}) {e.kind}")
    for name, pose in SPAWN_POSES.items():
        print(f"  spawn   {name:10s} ({pose[0]:6.1f},{pose[1]:6.1f})")
    if issues:
        print("\nLAYOUT PROBLEMS:")
        for p in issues:
            print(f"  - {p}")
        sys.exit(1)
    print("\nlayout OK")
