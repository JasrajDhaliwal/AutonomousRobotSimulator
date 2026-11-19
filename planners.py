"""Path planning algorithms for 4-neighbor grid navigation.

These planners work over the `GridMap` abstraction and return both the chosen
path and basic evaluation metrics such as total cost and nodes expanded.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
from itertools import count
from math import inf
from typing import Callable

from grid import Coord, GridMap

# A move cost function scores a single step `curr -> next`.
MoveCostFn = Callable[[GridMap, Coord, Coord], float]
# A heuristic estimates distance-to-go for A*.
HeuristicFn = Callable[[Coord, Coord], float]


@dataclass
class PlanResult:
    """Common result object returned by all planners."""

    algorithm: str
    found: bool
    path: list[Coord]
    total_cost: float
    nodes_expanded: int


def unit_time_cost(_: GridMap, __: Coord, ___: Coord) -> float:
    """Default unit-cost step model used for time-minimizing search."""
    return 1.0


def manhattan_heuristic(a: Coord, b: Coord) -> float:
    """Admissible heuristic for 4-neighbour grids without diagonal motion."""
    return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))


def _reconstruct_path(came_from: dict[Coord, Coord | None], goal: Coord) -> list[Coord]:
    """Walk backward through parent pointers and return a start->goal path."""
    path = [goal]
    cur = goal
    while came_from[cur] is not None:
        cur = came_from[cur]  # type: ignore[assignment]
        path.append(cur)
    path.reverse()
    return path


def bfs(grid: GridMap, start: Coord, goal: Coord) -> PlanResult:
    """Breadth-first search for shortest path by number of moves."""
    # BFS is only optimal when every move costs the same.
    if not grid.is_free(start) or not grid.is_free(goal):
        return PlanResult("BFS", False, [], inf, 0)
    if start == goal:
        return PlanResult("BFS", True, [start], 0.0, 0)

    frontier = deque([start])
    # `came_from` doubles as our visited set for BFS.
    came_from: dict[Coord, Coord | None] = {start: None}
    nodes_expanded = 0

    while frontier:
        current = frontier.popleft()
        nodes_expanded += 1

        if current == goal:
            break

        for nxt in grid.neighbors4(current):
            if nxt in came_from:
                continue
            came_from[nxt] = current
            frontier.append(nxt)

    if goal not in came_from:
        return PlanResult("BFS", False, [], inf, nodes_expanded)

    path = _reconstruct_path(came_from, goal)
    total_cost = float(len(path) - 1)
    return PlanResult("BFS", True, path, total_cost, nodes_expanded)


def dijkstra(
    grid: GridMap,
    start: Coord,
    goal: Coord,
    move_cost: MoveCostFn = unit_time_cost,
) -> PlanResult:
    """Dijkstra shortest-path search for non-negative move costs."""
    # Dijkstra generalizes BFS to weighted graphs as long as all edge costs are
    # non-negative, which is why ECO planning clamps per-step costs positive.
    if not grid.is_free(start) or not grid.is_free(goal):
        return PlanResult("Dijkstra", False, [], inf, 0)
    if start == goal:
        return PlanResult("Dijkstra", True, [start], 0.0, 0)

    # The tie-break counter keeps heap ordering deterministic when costs match.
    tie = count()
    frontier: list[tuple[float, int, Coord]] = [(0.0, next(tie), start)]
    came_from: dict[Coord, Coord | None] = {start: None}
    best_g: dict[Coord, float] = {start: 0.0}
    # Once a node is permanently expanded in Dijkstra, its best cost is known.
    expanded: set[Coord] = set()
    nodes_expanded = 0

    while frontier:
        current_g, _, current = heapq.heappop(frontier)
        if current in expanded:
            continue

        expanded.add(current)
        nodes_expanded += 1

        if current == goal:
            break

        for nxt in grid.neighbors4(current):
            new_g = current_g + float(move_cost(grid, current, nxt))
            if new_g < best_g.get(nxt, inf):
                best_g[nxt] = new_g
                came_from[nxt] = current
                heapq.heappush(frontier, (new_g, next(tie), nxt))

    if goal not in came_from:
        return PlanResult("Dijkstra", False, [], inf, nodes_expanded)

    path = _reconstruct_path(came_from, goal)
    return PlanResult("Dijkstra", True, path, best_g[goal], nodes_expanded)


def astar(
    grid: GridMap,
    start: Coord,
    goal: Coord,
    move_cost: MoveCostFn = unit_time_cost,
    heuristic: HeuristicFn = manhattan_heuristic,
    avoided: frozenset[Coord] = frozenset(),
) -> PlanResult:
    """A* search on a 4-neighbor grid using a supplied heuristic.

    The optional `avoided` set lets callers exclude specific cells from the
    search (e.g. active road closures) without modifying the underlying grid.
    """
    # A* combines the actual path cost so far (`g`) with a heuristic estimate
    # to the goal (`h`). With a good heuristic it explores much less than
    # Dijkstra while still finding an optimal path.
    if not grid.is_free(start) or start in avoided or not grid.is_free(goal) or goal in avoided:
        return PlanResult("A*", False, [], inf, 0)
    if start == goal:
        return PlanResult("A*", True, [start], 0.0, 0)

    tie = count()
    frontier: list[tuple[float, int, Coord]] = [(0.0, next(tie), start)]
    came_from: dict[Coord, Coord | None] = {start: None}
    best_g: dict[Coord, float] = {start: 0.0}
    # This acts like a closed set: once a node is expanded, we skip it.
    expanded: set[Coord] = set()
    nodes_expanded = 0

    while frontier:
        _, _, current = heapq.heappop(frontier)
        if current in expanded:
            continue

        expanded.add(current)
        nodes_expanded += 1

        if current == goal:
            break

        current_g = best_g[current]
        for nxt in grid.neighbors4(current):
            if nxt in avoided:
                continue
            tentative_g = current_g + float(move_cost(grid, current, nxt))
            if tentative_g < best_g.get(nxt, inf):
                best_g[nxt] = tentative_g
                came_from[nxt] = current
                # Push `f = g + h` to prioritize nodes that look promising.
                f = tentative_g + float(heuristic(nxt, goal))
                heapq.heappush(frontier, (f, next(tie), nxt))

    if goal not in came_from:
        return PlanResult("A*", False, [], inf, nodes_expanded)

    path = _reconstruct_path(came_from, goal)
    return PlanResult("A*", True, path, best_g[goal], nodes_expanded)
