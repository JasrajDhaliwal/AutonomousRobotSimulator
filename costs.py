"""Time/energy cost models and battery helpers.

This module is where the "robotics" part of the project lives mathematically:
- FAST mode treats every move as cost 1.
- ECO mode penalizes uphill moves more heavily.
- Downhill moves can recover some energy through regeneration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from grid import Coord, GridMap


@dataclass(frozen=True)
class EnergyModel:
    """Parameters for energy-aware movement.

    battery_delta = base_move + uphill_penalty - downhill_regen
    - Positive delta: battery consumed
    - Negative delta: battery restored (regen)
    """

    base_move: float = 1.0
    # How expensive uphill motion is per unit of height gain.
    k_up: float = 1.2
    # How much battery is recovered per unit of downhill height loss.
    k_down: float = 0.4
    # Planners need non-negative step costs, so regenerated steps are clipped
    # to this minimum value during search.
    min_planning_cost: float = 0.05


def time_move_cost(_: GridMap, __: Coord, ___: Coord) -> float:
    """Uniform time cost for FAST mode."""
    return 1.0


def v2x_time_move_cost(grid: GridMap, curr: Coord, nxt: Coord) -> float:
    """Time cost for smart-city V2X priority lanes.

    If the robot moves along a smart corridor, traffic-light coordination and
    right-of-way priority reduce the effective travel time.
    """
    if curr in grid.smart_corridors and nxt in grid.smart_corridors:
        return 0.58
    if nxt in grid.smart_corridors:
        return 0.72
    return 1.0


def height_delta(grid: GridMap, curr: Coord, nxt: Coord) -> float:
    """dh = h(next) - h(curr)."""
    return grid.elevation(nxt) - grid.elevation(curr)


def energy_move_delta(grid: GridMap, curr: Coord, nxt: Coord, model: EnergyModel) -> float:
    """Signed battery delta for one move (can be negative due to regen).

    Positive values mean the robot used battery.
    Negative values mean the robot gained some battery back on a downhill move.
    """
    dh = height_delta(grid, curr, nxt)
    uphill_penalty = model.k_up * max(0.0, dh)
    downhill_regen = model.k_down * max(0.0, -dh)
    return model.base_move + uphill_penalty - downhill_regen


def energy_move_cost_for_planning(
    grid: GridMap, curr: Coord, nxt: Coord, model: EnergyModel
) -> float:
    """Planner-safe energy cost (kept positive for Dijkstra/A* assumptions).

    Real battery change is allowed to go negative, but graph-search algorithms
    such as Dijkstra and A* assume non-negative edge weights.
    """
    return max(model.min_planning_cost, energy_move_delta(grid, curr, nxt, model))


def make_energy_move_cost(model: EnergyModel) -> Callable[[GridMap, Coord, Coord], float]:
    """Create a move-cost function compatible with planner APIs.

    This adapter lets us pass ECO movement costs into Dijkstra/A* without
    hard-coding a particular energy model inside the planner module.
    """

    def _cost(grid: GridMap, curr: Coord, nxt: Coord) -> float:
        return energy_move_cost_for_planning(grid, curr, nxt, model)

    return _cost


def path_time_cost(path: list[Coord]) -> float:
    """Total travel time under the unit-time movement model."""
    if len(path) <= 1:
        return 0.0
    return float(len(path) - 1)


def path_energy_delta(grid: GridMap, path: list[Coord], model: EnergyModel) -> float:
    """Signed net battery change across a path."""
    total = 0.0
    for i in range(len(path) - 1):
        total += energy_move_delta(grid, path[i], path[i + 1], model)
    return total


def path_energy_cost_for_planning(
    grid: GridMap, path: list[Coord], model: EnergyModel
) -> float:
    """Net planner objective value across a path (non-negative per step)."""
    total = 0.0
    for i in range(len(path) - 1):
        total += energy_move_cost_for_planning(grid,
                                               path[i], path[i + 1], model)
    return total


def path_battery_trace(
    grid: GridMap,
    path: list[Coord],
    start_battery: float,
    model: EnergyModel,
    capacity: float | None = None,
) -> list[float]:
    """Battery level after each node in the path (index 0 is initial battery).

    This is useful for debugging because it shows the battery profile, not just
    the start and end values.
    """
    battery = start_battery
    levels = [battery]

    for i in range(len(path) - 1):
        # Subtract usage because larger positive deltas represent energy spent.
        battery -= energy_move_delta(grid, path[i], path[i + 1], model)
        if capacity is not None:
            # Regeneration should not overfill the battery above its capacity.
            battery = min(capacity, battery)
        levels.append(battery)

    return levels


def is_path_battery_feasible(
    grid: GridMap,
    path: list[Coord],
    start_battery: float,
    model: EnergyModel,
    capacity: float | None = None,
) -> bool:
    """True if battery never drops below zero while traversing path."""
    for level in path_battery_trace(
        grid=grid,
        path=path,
        start_battery=start_battery,
        model=model,
        capacity=capacity,
    ):
        if level < 0.0:
            return False
    return True


def end_battery_after_path(
    grid: GridMap,
    path: list[Coord],
    start_battery: float,
    model: EnergyModel,
    capacity: float | None = None,
) -> float:
    """Battery level after fully traversing the path."""
    levels = path_battery_trace(
        grid=grid,
        path=path,
        start_battery=start_battery,
        model=model,
        capacity=capacity,
    )
    return levels[-1]
