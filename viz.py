"""Visualization helpers for the delivery robot simulator.

This module is presentation-only: it does not change planning decisions. It
simply turns map/path/battery state into 2D plots and animations.
"""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from grid import Coord, GridMap


def plot_map_2d(
    grid: GridMap,
    path: Iterable[Coord] | None = None,
    show: bool = True,
    title: str = "Delivery Robot Map",
) -> None:
    """Plot terrain, blocked cells, base, and deliveries in 2D."""
    fig, ax = plt.subplots(figsize=(8, 6))
    # The terrain image is the background layer of the plot.
    ax.imshow(grid.heightmap, origin="lower", cmap="terrain")

    if grid.blocked:
        # Obstacles are drawn as black squares on top of the terrain.
        bx = [x for x, _ in grid.blocked]
        by = [y for _, y in grid.blocked]
        ax.scatter(bx, by, marker="s", s=55, c="black", label="Obstacles")

    if grid.smart_corridors:
        vx = [x for x, _ in grid.smart_corridors]
        vy = [y for _, y in grid.smart_corridors]
        ax.scatter(vx, vy, marker="s", s=18, c="#00ccff", alpha=0.30, label="V2X Corridors")

    bx, by = grid.base
    # The base is the robot's home/recharge location.
    ax.scatter([bx], [by], marker="*", s=220, c="dodgerblue", edgecolors="white", label="Base")

    if grid.deliveries:
        # Delivery markers stay visible even if a path is plotted through them.
        dx = [x for x, _ in grid.deliveries]
        dy = [y for _, y in grid.deliveries]
        ax.scatter(dx, dy, marker="o", s=90, c="crimson", edgecolors="white", label="Deliveries")
        for i, (x, y) in enumerate(grid.deliveries, start=1):
            ax.text(x + 0.15, y + 0.15, str(i), fontsize=9, color="white", weight="bold")

    if path:
        # Overlay the chosen mission path as a polyline.
        px = [x for x, _ in path]
        py = [y for _, y in path]
        ax.plot(px, py, color="white", linewidth=2, label="Path")

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(-0.5, grid.width - 0.5)
    ax.set_ylim(-0.5, grid.height - 0.5)
    ax.grid(color="white", alpha=0.15)
    ax.legend(loc="upper right")
    fig.tight_layout()

    if show:
        plt.show()


def _urgency_color(cell: Coord, base: Coord, deadline: float) -> str:
    """Colour for an undelivered stop based on how tight its deadline is."""
    dist = max(abs(cell[0] - base[0]) + abs(cell[1] - base[1]), 1)
    tightness = deadline / dist
    if tightness < 2.5:
        return "#e84040"   # red   — tight
    if tightness < 4.0:
        return "#f5a020"   # orange — moderate
    return "#a8d460"       # yellow-green — comfortable


def _downsample(
    path: list[Coord],
    visited_trace: list[set[Coord]],
    battery_trace: list[float],
    energy_trace: list[float],
    max_frames: int,
) -> tuple[list[Coord], list[set[Coord]], list[float], list[float]]:
    n = len(path)
    if n <= max_frames:
        return path, visited_trace, battery_trace, energy_trace
    step = max(1, int(np.ceil((n - 1) / (max_frames - 1))))
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return (
        [path[i] for i in idx],
        [visited_trace[i] for i in idx],
        [battery_trace[i] for i in idx],
        [energy_trace[i] for i in idx],
    )


def animate_robot_2d(
    grid: GridMap,
    path: list[Coord],
    visited_trace: list[set[Coord]],
    battery_trace: list[float],
    energy_trace: list[float],
    delivery_deadlines: dict | None = None,
    show: bool = True,
    title: str = "Delivery Robot Simulation",
    interval_ms: int = 220,
    max_frames: int = 400,
) -> FuncAnimation:
    """Animate robot movement over the map with battery and energy overlays.

    When `delivery_deadlines` is provided, unvisited stops are colour-coded by
    urgency: red = tight deadline, orange = moderate, yellow-green = comfortable.
    """
    if not path:
        raise ValueError("path must contain at least one cell")

    n = len(path)
    if not (len(visited_trace) == len(battery_trace) == len(energy_trace) == n):
        raise ValueError("Trace arrays must all have the same length as path.")

    path, visited_trace, battery_trace, energy_trace = _downsample(
        path, visited_trace, battery_trace, energy_trace, max_frames
    )
    n = len(path)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(grid.heightmap, origin="lower", cmap="terrain")

    if grid.blocked:
        bx = [x for x, _ in grid.blocked]
        by = [y for _, y in grid.blocked]
        ax.scatter(bx, by, marker="s", s=55, c="black", label="Obstacles")

    if grid.smart_corridors:
        vx = [x for x, _ in grid.smart_corridors]
        vy = [y for _, y in grid.smart_corridors]
        ax.scatter(vx, vy, marker="s", s=18, c="#00ccff", alpha=0.30, label="V2X Corridors")

    bx, by = grid.base
    ax.scatter([bx], [by], marker="*", s=240, c="dodgerblue", edgecolors="white", label="Base")

    delivery_points = list(grid.deliveries)
    if delivery_deadlines:
        idle_colors = [
            _urgency_color(d, grid.base, delivery_deadlines.get(d, 1e9))
            for d in delivery_points
        ]
        legend_label = "Deliveries (red=tight/orange=mod/green=ok)"
    else:
        idle_colors = ["crimson"] * len(delivery_points)
        legend_label = "Deliveries"

    delivery_scatter = ax.scatter(
        [x for x, _ in delivery_points],
        [y for _, y in delivery_points],
        marker="o",
        s=95,
        c=idle_colors,
        edgecolors="white",
        label=legend_label,
    )
    for i, (x, y) in enumerate(delivery_points, start=1):
        ax.text(x + 0.12, y + 0.12, str(i), fontsize=9, color="white", weight="bold")

    ax.plot(
        [x for x, _ in path], [y for _, y in path],
        color="white", linewidth=1.5, alpha=0.35, label="Mission Path",
    )
    robot = ax.scatter(
        [path[0][0]], [path[0][1]],
        marker="o", s=170, c="gold", edgecolors="black", label="Robot",
    )
    info = ax.text(
        0.02, 0.98, "",
        transform=ax.transAxes, va="top", ha="left", fontsize=10,
        color="black",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.82, "edgecolor": "#444"},
    )

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(-0.5, grid.width - 0.5)
    ax.set_ylim(-0.5, grid.height - 0.5)
    ax.grid(color="white", alpha=0.15)
    ax.legend(loc="upper right")
    fig.tight_layout()

    def _update(frame: int):
        x, y = path[frame]
        robot.set_offsets([[x, y]])
        visited_now = visited_trace[frame]
        colors = ["limegreen" if p in visited_now else idle_colors[i]
                  for i, p in enumerate(delivery_points)]
        delivery_scatter.set_facecolors(colors)
        info.set_text(
            f"Step: {frame}/{n - 1}\n"
            f"Battery: {battery_trace[frame]:.2f}\n"
            f"Total Energy (net): {energy_trace[frame]:.2f}\n"
            f"Delivered: {len(visited_now)}/{len(delivery_points)}"
        )
        return robot, delivery_scatter, info

    anim = FuncAnimation(fig, _update, frames=n, interval=interval_ms, blit=False, repeat=False)
    if show:
        plt.show()
    return anim


# ---------------------------------------------------------------------------
# Helpers shared by both 3-D functions
# ---------------------------------------------------------------------------

# Vertical exaggeration applied to the heightmap in 3-D views.  The raw
# heightmap spans [0, 10]; scaling by 5 gives [0, 50] which makes hills
# clearly visible against a typical building footprint.
_Z_SCALE: float = 5.0


def _build_3d_surface(grid: GridMap) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_mesh, y_mesh, z_mesh, rgba) ready for plot_surface.

    Elevation drives colour via matplotlib's 'terrain' colormap so hills are
    immediately obvious: dark-green = valley, yellow-brown = mid-slope,
    white = peak.  Blocked cells are overridden to dark grey so buildings
    stand out from the ground.
    """
    xs = np.arange(grid.width)
    ys = np.arange(grid.height)
    x_mesh, y_mesh = np.meshgrid(xs, ys)
    z_mesh = grid.heightmap * _Z_SCALE

    h_min = float(grid.heightmap.min())
    h_span = max(float(grid.heightmap.max()) - h_min, 1e-6)
    h_norm = (grid.heightmap - h_min) / h_span
    rgba = plt.cm.terrain(h_norm).copy()  # shape (H, W, 4)

    # Tint blocked cells to a flat dark grey so buildings read clearly.
    for x, y in grid.blocked:
        rgba[y, x] = (0.22, 0.22, 0.24, 1.0)

    return x_mesh, y_mesh, z_mesh, rgba


def plot_map_3d(
    grid: GridMap,
    path: Iterable[Coord] | None = None,
    show: bool = True,
    title: str = "Delivery Robot Map (3D)",
) -> None:
    """Static 3-D terrain view.

    The surface colour maps directly to elevation (terrain colormap) so every
    hill and valley is visually distinct even without moving the camera.
    Buildings appear as dark-grey patches.  V2X corridors, the base, and
    delivery points float slightly above the surface so they are never hidden.
    """
    x_mesh, y_mesh, z_mesh, rgba = _build_3d_surface(grid)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        x_mesh, y_mesh, z_mesh,
        facecolors=rgba,
        shade=True,
        linewidth=0,
        antialiased=True,
    )

    # V2X corridors hover just above the terrain as thin cyan markers.
    if grid.smart_corridors:
        vx = [x for x, _ in grid.smart_corridors]
        vy = [y for _, y in grid.smart_corridors]
        vz = [grid.heightmap[y, x] * _Z_SCALE + 0.4 for x, y in grid.smart_corridors]
        ax.scatter(vx, vy, vz, c="#00ccff", s=12, alpha=0.50,
                   marker="s", depthshade=False, label="V2X Corridors")

    bx, by = grid.base
    ax.scatter([bx], [by], [grid.heightmap[by, bx] * _Z_SCALE + 1.2],
               c="dodgerblue", s=180, marker="*", depthshade=False, label="Base")

    if grid.deliveries:
        ax.scatter(
            [x for x, _ in grid.deliveries],
            [y for _, y in grid.deliveries],
            [grid.heightmap[y, x] * _Z_SCALE + 1.2 for x, y in grid.deliveries],
            c="crimson", s=60, marker="o", depthshade=False, label="Deliveries",
        )

    if path:
        pts = list(path)
        ax.plot(
            [x for x, _ in pts],
            [y for _, y in pts],
            [grid.heightmap[y, x] * _Z_SCALE + 0.8 for x, y in pts],
            color="white", linewidth=2.5, label="Path",
        )

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Elevation")
    ax.view_init(elev=28, azim=-55)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    if show:
        plt.show()


def animate_robot_3d(
    grid: GridMap,
    path: list[Coord],
    visited_trace: list[set[Coord]],
    battery_trace: list[float],
    energy_trace: list[float],
    show: bool = True,
    title: str = "Delivery Robot 3D Simulation",
    interval_ms: int = 160,
) -> FuncAnimation:
    """Animate robot movement over the 3-D terrain."""
    n = len(path)
    if not n:
        raise ValueError("path must contain at least one cell")
    if not (len(visited_trace) == len(battery_trace) == len(energy_trace) == n):
        raise ValueError("Trace arrays must all have the same length as path.")

    x_mesh, y_mesh, z_mesh, rgba = _build_3d_surface(grid)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        x_mesh, y_mesh, z_mesh,
        facecolors=rgba,
        shade=True,
        linewidth=0,
        antialiased=True,
    )

    if grid.smart_corridors:
        vx = [x for x, _ in grid.smart_corridors]
        vy = [y for _, y in grid.smart_corridors]
        vz = [grid.heightmap[y, x] * _Z_SCALE + 0.4 for x, y in grid.smart_corridors]
        ax.scatter(vx, vy, vz, c="#00ccff", s=12, alpha=0.50,
                   marker="s", depthshade=False, label="V2X Corridors")

    bx, by = grid.base
    ax.scatter([bx], [by], [grid.heightmap[by, bx] * _Z_SCALE + 1.2],
               c="dodgerblue", s=180, marker="*", depthshade=False, label="Base")

    delivery_points = list(grid.deliveries)
    delivery_zs = [grid.heightmap[y, x] * _Z_SCALE + 1.2 for x, y in delivery_points]
    delivery_scatter = ax.scatter(
        [x for x, _ in delivery_points],
        [y for _, y in delivery_points],
        delivery_zs,
        c=["crimson"] * len(delivery_points),
        s=60, marker="o", depthshade=False, label="Deliveries",
    )

    # Ghost path drawn faintly behind the robot.
    ax.plot(
        [x for x, _ in path],
        [y for _, y in path],
        [grid.heightmap[y, x] * _Z_SCALE + 0.6 for x, y in path],
        color="white", linewidth=1.2, alpha=0.25,
    )

    start = path[0]
    robot = ax.scatter(
        [start[0]], [start[1]], [grid.heightmap[start[1], start[0]] * _Z_SCALE + 1.4],
        c="gold", s=200, marker="o", edgecolors="black", depthshade=False, label="Robot",
    )
    trail, = ax.plot(
        [start[0]], [start[1]], [grid.heightmap[start[1], start[0]] * _Z_SCALE + 0.8],
        color="#ffd24d", linewidth=2.0, alpha=0.85,
    )
    info = ax.text2D(
        0.02, 0.96, "",
        transform=ax.transAxes, va="top", ha="left", fontsize=9,
        color="black",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.82, "edgecolor": "#444"},
    )

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Elevation")
    ax.view_init(elev=28, azim=-55)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    def _update(frame: int):
        x, y = path[frame]
        z = grid.heightmap[y, x] * _Z_SCALE + 1.4
        robot._offsets3d = ([x], [y], [z])

        trail.set_data([c[0] for c in path[: frame + 1]],
                       [c[1] for c in path[: frame + 1]])
        trail.set_3d_properties(
            [grid.heightmap[c[1], c[0]] * _Z_SCALE + 0.8 for c in path[: frame + 1]]
        )

        visited_now = visited_trace[frame]
        d_colors = ["limegreen" if p in visited_now else "crimson" for p in delivery_points]
        delivery_scatter._facecolor3d = delivery_scatter._edgecolor3d = np.array(
            [(*plt.matplotlib.colors.to_rgb(c), 1.0) for c in d_colors]
        )

        info.set_text(
            f"Step: {frame}/{n - 1}\n"
            f"Battery: {battery_trace[frame]:.2f}\n"
            f"Energy (net): {energy_trace[frame]:.2f}\n"
            f"Delivered: {len(visited_now)}/{len(delivery_points)}"
        )
        return robot, trail, info, delivery_scatter

    anim = FuncAnimation(fig, _update, frames=n, interval=interval_ms, blit=False, repeat=False)
    if show:
        plt.show()
    return anim
