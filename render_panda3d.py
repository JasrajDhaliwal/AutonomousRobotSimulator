"""Realtime Panda3D city renderer for the delivery robot simulator.

Run standalone:
    python3 render_panda3d.py [--map-seed N] [--stop-seed M]

Or it is launched automatically at the end of main.py.

Install Panda3D once:
    pip install panda3d
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass

import numpy as np

from costs import EnergyModel
from grid import Coord, GridMap, format_cells, generate_sample_map
from main import MissionResult, simulate_random_stops_with_recharge

try:
    from panda3d.core import (
        AmbientLight,
        CardMaker,
        DirectionalLight,
        Fog,
        Geom,
        GeomNode,
        GeomTriangles,
        GeomVertexData,
        GeomVertexFormat,
        GeomVertexWriter,
        LColor,
        LineSegs,
        NodePath,
        TextNode,
        TransparencyAttrib,
        Vec3,
        WindowProperties,
    )
    from direct.gui.OnscreenText import OnscreenText
    from direct.showbase.ShowBase import ShowBase
    from direct.task import Task
except ImportError as exc:
    raise SystemExit(
        "Panda3D is not installed.\n"
        "  pip install panda3d\n"
        "Then run:  python3 render_panda3d.py\n"
    ) from exc


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RenderConfig:
    """Rendering parameters — tweak freely without touching scene code."""

    # City scale: each grid cell becomes this many world units.
    # 2.0 makes buildings feel imposing and streets feel walkable.
    cell_world_size: float = 2.0

    # Terrain elevation multiplier (kept tiny so the city looks flat,
    # not like a mountain range).
    vertical_scale: float = 0.04

    # How tall buildings are in world units per raw height unit.
    # Independent of vertical_scale so the skyline looks right.
    building_height_scale: float = 0.45

    # Robot geometry / motion
    robot_height: float = 0.70
    robot_speed_cells_per_sec: float = 2.5

    # Free-roam camera
    camera_move_speed: float = 28.0
    camera_turn_speed_deg: float = 80.0

    # Follow camera  (third-person, behind and above the robot)
    follow_height: float = 9.0
    follow_distance: float = 14.0
    follow_lerp: float = 5.5      # higher = snappier
    follow_look_ahead: float = 6.0  # world-units ahead of robot that camera aims at


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grid_to_world(grid: GridMap, cell: Coord, cfg: RenderConfig) -> tuple[float, float, float]:
    x, y = cell
    return (
        x * cfg.cell_world_size,
        y * cfg.cell_world_size,
        grid.surface_height(cell) * cfg.vertical_scale,
    )


def _hex(color: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    c = color.lstrip("#")
    return int(c[0:2], 16) / 255, int(c[2:4], 16) / 255, int(c[4:6], 16) / 255, alpha


def _building_color(raw_height: float) -> str:
    """Vary building colours by height so downtown looks different from suburbs."""
    if raw_height > 36:
        return "#1c2530"   # very tall glass towers — dark steel blue
    if raw_height > 24:
        return "#2a3645"   # tall offices — dark slate
    if raw_height > 14:
        return "#3b4a52"   # mid-rise condos — cool gray
    if raw_height > 7:
        return "#584e48"   # low-rise — warm brown-gray
    return "#6e6058"       # small shops / houses — brick tone


# ---------------------------------------------------------------------------
# Geometry builders
# ---------------------------------------------------------------------------

def _build_unit_box() -> GeomNode:
    """Unit cube (1×1×1) centred at origin, with per-face normals."""
    fmt = GeomVertexFormat.getV3n3c4()
    vdata = GeomVertexData("unit_box", fmt, Geom.UHStatic)
    vw = GeomVertexWriter(vdata, "vertex")
    nw = GeomVertexWriter(vdata, "normal")
    cw = GeomVertexWriter(vdata, "color")

    faces = [
        ((0, 0, 1),  [(-0.5, -0.5, 0.5), (0.5, -0.5, 0.5),
         (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)]),
        ((0, 0, -1),  [(-0.5, 0.5, -0.5), (0.5, 0.5, -0.5),
         (0.5, -0.5, -0.5), (-0.5, -0.5, -0.5)]),
        ((0, 1, 0),  [(-0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
         (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5)]),
        ((0, -1, 0),  [(-0.5, -0.5, -0.5), (0.5, -0.5, -0.5),
         (0.5, -0.5, 0.5), (-0.5, -0.5, 0.5)]),
        ((1, 0, 0),  [(0.5, -0.5, 0.5), (0.5, -0.5, -0.5),
         (0.5, 0.5, -0.5), (0.5, 0.5, 0.5)]),
        ((-1, 0, 0),  [(-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5),
         (-0.5, 0.5, 0.5), (-0.5, 0.5, -0.5)]),
    ]
    tris = GeomTriangles(Geom.UHStatic)
    idx = 0
    for fn, verts in faces:
        for vx, vy, vz in verts:
            vw.addData3(vx, vy, vz)
            nw.addData3(*fn)
            cw.addData4(1, 1, 1, 1)
        tris.addVertices(idx, idx+1, idx+2)
        tris.addVertices(idx, idx+2, idx+3)
        idx += 4

    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode("unit_box")
    node.addGeom(geom)
    return node


def _build_surface_mesh(grid: GridMap, cfg: RenderConfig) -> GeomNode:
    """Flat coloured ground plane: roads, sidewalks, parks, plazas, etc."""
    fmt = GeomVertexFormat.getV3n3c4()
    vdata = GeomVertexData("surface", fmt, Geom.UHStatic)
    vw = GeomVertexWriter(vdata, "vertex")
    nw = GeomVertexWriter(vdata, "normal")
    cw = GeomVertexWriter(vdata, "color")
    tris = GeomTriangles(Geom.UHStatic)

    palette = {
        "road":      _hex("#4a5260"),
        "sidewalk":  _hex("#d0d4d8"),
        "bike":      _hex("#4a9e60"),
        "plaza":     _hex("#c8b89e"),
        "park":      _hex("#3d7a4a"),
        "corridor":  _hex("#48d8ee"),
        "hub":       _hex("#00ccff"),
    }

    idx = 0
    cs = cfg.cell_world_size
    for y in range(grid.height):
        for x in range(grid.width):
            cell = (x, y)
            if cell in grid.blocked:
                continue
            z = grid.surface_height(cell) * cfg.vertical_scale
            x0, x1 = x * cs - cs * 0.5, x * cs + cs * 0.5
            y0, y1 = y * cs - cs * 0.5, y * cs + cs * 0.5

            if cell in grid.charging_hubs:
                rgba = palette["hub"]
            elif cell in grid.smart_corridors:
                rgba = palette["corridor"]
            elif cell in grid.bike_lanes:
                rgba = palette["bike"]
            elif cell in grid.plazas:
                rgba = palette["plaza"]
            elif cell in grid.sidewalks:
                rgba = palette["sidewalk"]
            elif cell in grid.parks:
                rgba = palette["park"]
            else:
                rgba = palette["road"]

            for vx, vy in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]:
                vw.addData3(vx, vy, z)
                nw.addData3(0, 0, 1)
                cw.addData4(*rgba)
            tris.addVertices(idx, idx+1, idx+2)
            tris.addVertices(idx, idx+2, idx+3)
            idx += 4

    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode("surface")
    node.addGeom(geom)
    return node


def _build_road_lines(grid: GridMap, cfg: RenderConfig) -> NodePath:
    """White dashed centre lines along road cells for a city-street look."""
    ls = LineSegs("road_lines")
    ls.setThickness(1.2)
    ls.setColor(0.9, 0.9, 0.85, 0.55)
    cs = cfg.cell_world_size
    for cell in grid.roads:
        x, y = cell
        wx, wy = x * cs, y * cs
        wz = grid.surface_height(cell) * cfg.vertical_scale + 0.06
        left, right = (x-1, y) in grid.roads, (x+1, y) in grid.roads
        above, below = (x, y+1) in grid.roads, (x, y-1) in grid.roads
        h = cs * 0.45
        if left or right:
            ls.moveTo(wx - h, wy, wz)
            ls.drawTo(wx + h, wy, wz)
        if above or below:
            ls.moveTo(wx, wy - h, wz)
            ls.drawTo(wx, wy + h, wz)
    return NodePath(ls.create())


def _build_path_line(grid: GridMap, path: list[Coord], cfg: RenderConfig) -> NodePath:
    """Bright gold line tracing the full mission route."""
    ls = LineSegs("mission_path")
    ls.setThickness(4.0)
    ls.setColor(0.98, 0.82, 0.18, 1.0)
    for cell in path:
        x, y, z = _grid_to_world(grid, cell, cfg)
        ls.drawTo(x, y, z + 0.55)
    return NodePath(ls.create())


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class DeliveryRenderApp(ShowBase):
    """Interactive Panda3D city with a delivery robot you can follow."""

    def __init__(self, grid: GridMap, mission: MissionResult, cfg: RenderConfig, title: str) -> None:
        super().__init__()
        self.grid = grid
        self.mission = mission
        self.cfg = cfg
        self.camera_mode = "follow"

        props = WindowProperties()
        props.setTitle(title)
        props.setSize(1440, 900)
        self.win.requestProperties(props)

        # Wider FOV makes streets feel more immersive.
        self.camLens.setFov(80)

        self.setBackgroundColor(0.50, 0.68, 0.82, 1.0)   # sky-blue clear day
        self.disableMouse()
        self._setup_lights()
        self._setup_fog()
        self._setup_controls()
        self._build_scene()
        self._build_hud()

        self.robot_progress = 0.0
        self.total_segments = max(len(mission.full_path) - 1, 1)
        self.taskMgr.add(self._update_robot, "robot")
        self.taskMgr.add(self._update_camera, "camera")

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    def _setup_lights(self) -> None:
        # Soft fill from above.
        amb = AmbientLight("amb")
        amb.setColor(LColor(0.42, 0.44, 0.48, 1))
        self.render.setLight(self.render.attachNewNode(amb))

        # Warm sun from the south-west at a low angle.
        sun = DirectionalLight("sun")
        sun.setColor(LColor(1.0, 0.93, 0.78, 1))
        sun_np = self.render.attachNewNode(sun)
        sun_np.setHpr(-40, -38, 0)
        self.render.setLight(sun_np)

        # Cool blue bounce from the north — stops shadowed faces going pure black.
        bounce = DirectionalLight("bounce")
        bounce.setColor(LColor(0.18, 0.30, 0.52, 1))
        bounce_np = self.render.attachNewNode(bounce)
        bounce_np.setHpr(145, -10, 0)
        self.render.setLight(bounce_np)

    def _setup_fog(self) -> None:
        """Distance haze — makes the city feel deep rather than like a miniature."""
        fog = Fog("haze")
        fog.setColor(0.62, 0.74, 0.86)
        fog.setLinearRange(
            self.grid.width * self.cfg.cell_world_size * 0.45,
            self.grid.width * self.cfg.cell_world_size * 1.3,
        )
        self.render.setFog(fog)

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def _setup_controls(self) -> None:
        keys = ("w", "a", "s", "d", "q", "e", "left", "right", "up", "down")
        self.key_map: dict[str, bool] = {k: False for k in keys}
        for k in keys:
            self.accept(k,       self._set_key, [k, True])
            self.accept(f"{k}-up", self._set_key, [k, False])
        self.accept("c",      self._toggle_cam)
        self.accept("r",      self._reset_robot)
        self.accept("escape", sys.exit)

        # Start the free camera at a cinematic overview angle.
        cx = self.grid.width * self.cfg.cell_world_size * 0.5
        cy = self.grid.height * self.cfg.cell_world_size * 0.5
        self.camera.setPos(cx, -self.grid.height *
                           self.cfg.cell_world_size * 0.6, 55)
        self.camera.lookAt(cx, cy, 0)

    def _set_key(self, k: str, v: bool) -> None:
        self.key_map[k] = v

    def _toggle_cam(self) -> None:
        self.camera_mode = "free" if self.camera_mode == "follow" else "follow"
        # When switching to free, teleport camera to near the robot so the
        # player doesn't have to navigate back from the overview position.
        if self.camera_mode == "free":
            rpos = self.robot_np.getPos()
            self.camera.setPos(rpos.x, rpos.y - 12, rpos.z + 8)
            self.camera.lookAt(rpos)

    def _reset_robot(self) -> None:
        self.robot_progress = 0.0

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------

    def _build_scene(self) -> None:
        world = self.render.attachNewNode("world")

        # Ground plane
        surface = world.attachNewNode(_build_surface_mesh(self.grid, self.cfg))
        surface.setTwoSided(True)

        # Road centre dashes
        road_lines = _build_road_lines(self.grid, self.cfg)
        road_lines.reparentTo(world)

        # Buildings
        unit_box = NodePath(_build_unit_box())
        bh_arr = self.grid.building_heights
        if bh_arr is None:
            bh_arr = np.zeros_like(self.grid.heightmap)
        cs = self.cfg.cell_world_size
        for cell in sorted(self.grid.blocked):
            x, y, z = _grid_to_world(self.grid, cell, self.cfg)
            raw_h = float(bh_arr[cell[1], cell[0]])
            h = max(raw_h * self.cfg.building_height_scale, 1.0)
            b = world.attachNewNode(unit_box.node().copySubgraph())
            b.setScale(0.92 * cs, 0.92 * cs, h)
            b.setPos(x, y, z + h * 0.5)
            b.setColorScale(*_hex(_building_color(raw_h)))

        # Park trees — small green boxes, one per park cell
        for cell in self.grid.parks:
            x, y, z = _grid_to_world(self.grid, cell, self.cfg)
            tree_h = 1.4 + ((cell[0] * 7 + cell[1] * 13) % 8) * 0.18
            t = world.attachNewNode(unit_box.node().copySubgraph())
            t.setScale(0.28 * cs, 0.28 * cs, tree_h)
            t.setPos(x, y, z + tree_h * 0.5)
            t.setColorScale(*_hex("#2e7d3a"))

        # Charging hub pillars
        for cell in self.grid.charging_hubs:
            x, y, z = _grid_to_world(self.grid, cell, self.cfg)
            hub = world.attachNewNode(unit_box.node().copySubgraph())
            hub.setScale(0.55 * cs, 0.55 * cs, 0.6)
            hub.setPos(x, y, z + 0.3)
            hub.setColorScale(*_hex("#00ccff"))

        # Base station marker
        bx, by, bz = _grid_to_world(self.grid, self.grid.base, self.cfg)
        base = world.attachNewNode(unit_box.node().copySubgraph())
        base.setScale(0.7 * cs, 0.7 * cs, 0.7)
        base.setPos(bx, by, bz + 0.35)
        base.setColorScale(*_hex("#2288ff"))

        # Delivery markers (tall red pillars, turn green when done)
        self.delivery_markers: dict[Coord, NodePath] = {}
        for cell in self.grid.deliveries:
            x, y, z = _grid_to_world(self.grid, cell, self.cfg)
            m = world.attachNewNode(unit_box.node().copySubgraph())
            m.setScale(0.35 * cs, 0.35 * cs, 1.2)
            m.setPos(x, y, z + 0.6)
            m.setColorScale(*_hex("#e02050"))
            self.delivery_markers[cell] = m

        # Mission route line
        route = _build_path_line(self.grid, self.mission.full_path, self.cfg)
        route.reparentTo(world)
        route.setTransparency(TransparencyAttrib.MAlpha)

        # Robot
        self.robot_np = self._build_robot(world, unit_box)

    def _build_robot(self, world: NodePath, unit_box: NodePath) -> NodePath:
        """Compound robot model: body + cabin + sensor dome + four wheels."""
        cfg, cs = self.cfg, self.cfg.cell_world_size
        root = world.attachNewNode("robot")

        # Main body
        body = root.attachNewNode(unit_box.node().copySubgraph())
        body.setScale(0.52 * cs, 0.72 * cs, cfg.robot_height * 0.52)
        body.setZ(cfg.robot_height * 0.27)
        body.setColorScale(*_hex("#ffd43b"))

        # Cabin / top box (white)
        cab = root.attachNewNode(unit_box.node().copySubgraph())
        cab.setScale(0.36 * cs, 0.32 * cs, cfg.robot_height * 0.30)
        cab.setPos(0, 0.08 * cs, cfg.robot_height * 0.62)
        cab.setColorScale(*_hex("#f0f4f8"))

        # Sensor mast (cyan)
        mast = root.attachNewNode(unit_box.node().copySubgraph())
        mast.setScale(0.09 * cs, 0.09 * cs, cfg.robot_height * 0.20)
        mast.setPos(0, 0.30 * cs, cfg.robot_height * 0.88)
        mast.setColorScale(*_hex("#00ccff"))

        # Wheels (dark, four corners)
        for dx in (-0.24, 0.24):
            for dy in (-0.28, 0.28):
                w = root.attachNewNode(unit_box.node().copySubgraph())
                w.setScale(0.10 * cs, 0.13 * cs, cfg.robot_height * 0.17)
                w.setPos(dx * cs, dy * cs, cfg.robot_height * 0.11)
                w.setColorScale(*_hex("#111827"))

        return root

    # ------------------------------------------------------------------
    # HUD
    # ------------------------------------------------------------------

    def _build_hud(self) -> None:
        OnscreenText(
            text="WASD  move   |   QE  up/down   |   Arrows  look   |   C  follow/free   |   R  restart   |   Esc  quit",
            pos=(-1.30, 0.94),
            align=TextNode.ALeft,
            scale=0.040,
            fg=(0.95, 0.97, 1.0, 1.0),
            shadow=(0, 0, 0, 0.6),
            shadowOffset=(0.05, 0.05),
            parent=self.aspect2d,
        )
        self.cam_label = OnscreenText(
            text="",
            pos=(-1.30, 0.86),
            align=TextNode.ALeft,
            scale=0.046,
            fg=(1.0, 0.90, 0.40, 1.0),
            shadow=(0, 0, 0, 0.5),
            shadowOffset=(0.05, 0.05),
            parent=self.aspect2d,
            mayChange=True,
        )
        self.info_text = OnscreenText(
            text="",
            pos=(-1.30, 0.76),
            align=TextNode.ALeft,
            scale=0.048,
            fg=(0.95, 0.97, 1.0, 1.0),
            shadow=(0, 0, 0, 0.55),
            shadowOffset=(0.05, 0.05),
            parent=self.aspect2d,
            mayChange=True,
        )

    # ------------------------------------------------------------------
    # Per-frame updates
    # ------------------------------------------------------------------

    def _update_robot(self, task: Task) -> int:
        dt = globalClock.getDt()
        self.robot_progress = min(
            self.total_segments,
            self.robot_progress + self.cfg.robot_speed_cells_per_sec * dt,
        )

        seg = min(int(self.robot_progress), self.total_segments - 1)
        frac = self.robot_progress - seg
        p0 = self.mission.full_path[seg]
        p1 = self.mission.full_path[min(
            seg + 1, len(self.mission.full_path) - 1)]

        sx, sy, sz = _grid_to_world(self.grid, p0, self.cfg)
        ex, ey, ez = _grid_to_world(self.grid, p1, self.cfg)
        rx = sx + (ex - sx) * frac
        ry = sy + (ey - sy) * frac
        rz = sz + (ez - sz) * frac + self.cfg.robot_height
        self.robot_np.setPos(rx, ry, rz)

        if (sx, sy) != (ex, ey):
            self.robot_np.setH(math.degrees(
                math.atan2(ey - sy, ex - sx)) - 90.0)

        # Update delivery marker colours
        ti = min(round(self.robot_progress), len(
            self.mission.battery_trace) - 1)
        delivered = self.mission.visited_trace[ti]
        for cell, marker in self.delivery_markers.items():
            marker.setColorScale(
                *(_hex("#22c55e") if cell in delivered else _hex("#e02050")))

        # HUD
        late = len(self.mission.late_deliveries) if hasattr(
            self.mission, "late_deliveries") else 0
        n_done = len(delivered)
        n_total = len(self.grid.deliveries)
        pct = int(self.robot_progress / self.total_segments * 100)
        self.cam_label.setText(
            f"Camera: {self.camera_mode.upper()}  (C to toggle)")
        self.info_text.setText(
            f"Progress  : {pct}%\n"
            f"Stops     : {n_done}/{n_total}  (late: {late})\n"
            f"Battery   : {self.mission.battery_trace[ti]:.1f}\n"
            f"Energy    : {self.mission.energy_trace[ti]:.1f}\n"
            f"Recharges : {self.mission.recharge_count}   Hubs: {self.mission.hub_charge_count}"
        )
        return Task.cont

    def _update_camera(self, task: Task) -> int:
        dt = globalClock.getDt()

        if self.camera_mode == "follow":
            robot_pos = self.robot_np.getPos()
            h_rad = math.radians(self.robot_np.getH())

            # In Panda3D heading convention, forward = (-sin(H), cos(H), 0).
            # Place camera behind the robot (+sin, -cos) and above (+follow_height).
            target = robot_pos + Vec3(
                math.sin(h_rad) * self.cfg.follow_distance,
                -math.cos(h_rad) * self.cfg.follow_distance,
                self.cfg.follow_height,
            )
            lerp = min(1.0, self.cfg.follow_lerp * dt)
            self.camera.setPos(self.camera.getPos() *
                               (1.0 - lerp) + target * lerp)

            # Look at a point ahead of the robot so the view feels dynamic.
            look_pt = robot_pos + Vec3(
                -math.sin(h_rad) * self.cfg.follow_look_ahead,
                math.cos(h_rad) * self.cfg.follow_look_ahead,
                0.8,
            )
            self.camera.lookAt(look_pt)
            return Task.cont

        # Free-roam WASD
        move = self.cfg.camera_move_speed * dt
        turn = self.cfg.camera_turn_speed_deg * dt
        km = self.key_map
        if km["w"]:
            self.camera.setY(self.camera,  move)
        if km["s"]:
            self.camera.setY(self.camera, -move)
        if km["a"]:
            self.camera.setX(self.camera, -move)
        if km["d"]:
            self.camera.setX(self.camera,  move)
        if km["q"]:
            self.camera.setZ(self.camera,  move)
        if km["e"]:
            self.camera.setZ(self.camera, -move)
        if km["left"]:
            self.camera.setH(self.camera.getH() + turn)
        if km["right"]:
            self.camera.setH(self.camera.getH() - turn)
        if km["up"]:
            self.camera.setP(max(-89.0, self.camera.getP() + turn))
        if km["down"]:
            self.camera.setP(min(89.0, self.camera.getP() - turn))
        return Task.cont


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def launch(grid: GridMap, mission: MissionResult, title: str = "Delivery Robot — 3D City") -> None:
    """Launch the renderer with an already-computed mission (called from main.py)."""
    app = DeliveryRenderApp(grid=grid, mission=mission,
                            cfg=RenderConfig(), title=title)
    app.run()


def _build_demo_mission(map_seed: int | None, stop_seed: int | None) -> tuple[GridMap, MissionResult, int, int]:
    model = EnergyModel(base_move=1.0, k_up=1.25,
                        k_down=0.55, min_planning_cost=0.05)
    rng = random.SystemRandom()
    fallback: tuple[GridMap, MissionResult, int, int] | None = None
    attempts = 1 if (map_seed is not None and stop_seed is not None) else 300

    for _ in range(attempts):
        ms = map_seed if map_seed is not None else rng.randrange(1, 10**9)
        ss = stop_seed if stop_seed is not None else rng.randrange(1, 10**9)
        grid = generate_sample_map(
            width=60, height=42, obstacle_ratio=0.20, num_deliveries=8, seed=ms)
        mission = simulate_random_stops_with_recharge(
            grid=grid, battery_start=100.0, battery_capacity=200.0,
            model=model, stop_seed=ss,
        )
        if not mission.success:
            continue
        if map_seed is not None and stop_seed is not None:
            return grid, mission, ms, ss
        if mission.hub_charge_count > 0 or mission.recharge_count > 0:
            return grid, mission, ms, ss
        if fallback is None:
            fallback = (grid, mission, ms, ss)

    if fallback:
        return fallback
    raise RuntimeError("Could not find a successful mission scenario.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Panda3D delivery robot city renderer.")
    parser.add_argument("--map-seed",  type=int, default=None)
    parser.add_argument("--stop-seed", type=int, default=None)
    args = parser.parse_args()

    grid, mission, ms, ss = _build_demo_mission(args.map_seed, args.stop_seed)

    sep = "=" * 52
    print(sep)
    print("  PANDA3D INTERACTIVE CITY RENDERER")
    print(sep)
    print(f"  Map seed  : {ms}    Stop seed : {ss}")
    print(f"  Base      : {grid.base}")
    print(f"  Stops     : {format_cells(grid.deliveries)}")
    print(
        f"  Completed : {len(mission.completed_stops)}/{len(grid.deliveries)}")
    print(
        f"  Time      : {mission.total_time:.0f}   Energy : {mission.total_energy:.1f}")
    print()
    print("  Controls:")
    print("    WASD        — move camera (free mode)")
    print("    Q / E       — fly up / down")
    print("    Arrow keys  — look around")
    print("    C           — toggle FOLLOW / FREE camera")
    print("    R           — restart robot playback")
    print("    Esc         — quit")
    print(sep)

    launch(grid, mission, title="Delivery Robot — 3D City")


if __name__ == "__main__":
    main()
