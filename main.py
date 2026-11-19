"""Step 8 demo: moving robot simulation with random stops and recharge behavior.

This file is the top-level "glue" for the simulator:
- it creates a map,
- chooses a randomized delivery order,
- plans each leg using the FAST/ECO policy,
- returns to base when a recharge is needed,
- logs metrics,
- and shows the result in both static and animated views.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import random
from time import perf_counter

import matplotlib

from costs import (
    EnergyModel,
    end_battery_after_path,
    energy_move_delta,
    is_path_battery_feasible,
    make_energy_move_cost,
    path_energy_delta,
    path_time_cost,
    time_move_cost,
    v2x_time_move_cost,
)
from grid import Coord, GridMap, generate_sample_map
from metrics import append_run_metrics_csv, build_run_metrics
from planners import PlanResult, astar
from viz import animate_robot_2d, plot_map_2d

# Matplotlib animations can get garbage-collected early if no reference is
# kept alive. This module-level handle prevents that during interactive runs.
_LAST_ANIMATION = None


@dataclass
class ModeChoice:
    """Planner decision for one target under the battery-safety policy."""

    mode: str | None  # V2X, FAST, ECO, or None when not feasible
    path: list[Coord]
    reason: str
    nodes_expanded: int


@dataclass
class TargetChoice:
    """Chosen next stop together with the planning result used to reach it."""

    target: Coord | None
    choice: ModeChoice
    urgency_reorder: bool = False


@dataclass
class HubChoice:
    """Plan for reaching a smart charging hub before continuing the mission."""

    hub: Coord | None
    path: list[Coord]
    reason: str
    nodes_expanded: int


@dataclass
class DeliveryDeadline:
    """Time-window deadline assigned to each delivery stop.

    The robot re-evaluates remaining delivery urgency at every planning step,
    serving the stop with the least remaining slack first. This implements a
    greedy approximation of VRPTW (Vehicle Routing Problem with Time Windows),
    which no current production delivery robot performs in real time.
    """

    cell: Coord
    deadline_steps: float   # Sim-time by which delivery must be completed
    tightness: float        # deadline / manhattan_distance — lower = tighter


@dataclass
class ExecutedLeg:
    """One mission segment that actually gets traversed by the robot."""

    leg_type: str  # DELIVERY, HUB_CHARGE, RETURN_RECHARGE, FINAL_RETURN
    mode: str
    start: Coord
    goal: Coord
    path: list[Coord]
    time_cost: float
    energy_cost: float
    battery_start: float
    battery_end: float


@dataclass
class MissionResult:
    """Full summary of one simulated mission run."""

    success: bool
    reason: str
    stop_order: list[Coord]
    completed_stops: list[Coord]
    recharge_count: int
    hub_charge_count: int
    legs: list[ExecutedLeg]
    total_time: float
    total_energy: float
    nodes_expanded: int
    runtime_ms: float
    battery_start: float
    battery_end: float
    hub_charge_events: list[Coord]
    full_path: list[Coord]
    battery_trace: list[float]
    energy_trace: list[float]
    visited_trace: list[set[Coord]]
    # Deadline / urgency fields (novel feature)
    delivery_deadlines: dict[Coord, float]   # deadline_steps per stop
    late_deliveries: list[Coord]             # stops completed after deadline
    urgency_reorder_count: int               # times urgency changed the planned order


def _append_idle_frame(
    path_trace: list[Coord],
    battery_trace: list[float],
    energy_trace: list[float],
    visited_trace: list[set[Coord]],
    pos: Coord,
    battery_now: float,
    total_energy: float,
    visited: set[Coord],
) -> None:
    """Append one stationary frame to the animation traces.

    This is used both at the beginning of the mission and during recharge
    pauses, where the robot stays in the same cell but its state changes.
    """
    path_trace.append(pos)
    battery_trace.append(battery_now)
    energy_trace.append(total_energy)
    visited_trace.append(set(visited))


def _append_segment_trace(
    grid: GridMap,
    model: EnergyModel,
    segment: list[Coord],
    battery_capacity: float,
    visited: set[Coord],
    path_trace: list[Coord],
    battery_trace: list[float],
    energy_trace: list[float],
    visited_trace: list[set[Coord]],
    battery_now: float,
    total_energy: float,
) -> tuple[float, float, list[Coord]]:
    """Apply a path segment to the simulation traces.

    The mission code already knows which path to take. This helper is just
    responsible for "playing it forward" one cell at a time so we can produce
    animation frames, battery traces, and accumulated energy.
    """
    newly_delivered: list[Coord] = []

    if not segment:
        return battery_now, total_energy, newly_delivered

    if not path_trace:
        # Seed the traces with the starting cell the first time we move.
        _append_idle_frame(
            path_trace=path_trace,
            battery_trace=battery_trace,
            energy_trace=energy_trace,
            visited_trace=visited_trace,
            pos=segment[0],
            battery_now=battery_now,
            total_energy=total_energy,
            visited=visited,
        )

    if path_trace[-1] != segment[0]:
        # If a new segment starts at a different cell, record that boundary.
        _append_idle_frame(
            path_trace=path_trace,
            battery_trace=battery_trace,
            energy_trace=energy_trace,
            visited_trace=visited_trace,
            pos=segment[0],
            battery_now=battery_now,
            total_energy=total_energy,
            visited=visited,
        )

    for i in range(len(segment) - 1):
        curr = segment[i]
        nxt = segment[i + 1]
        # Energy accounting is done per move so the animation can show the
        # battery and cumulative energy changing over time.
        delta = energy_move_delta(grid, curr, nxt, model)
        battery_now -= delta
        battery_now = min(battery_capacity, battery_now)
        total_energy += delta

        # Deliveries count the moment the robot actually reaches that cell,
        # even if that stop was not the original target of this path.
        if nxt in grid.deliveries and nxt not in visited:
            visited.add(nxt)
            newly_delivered.append(nxt)

        _append_idle_frame(
            path_trace=path_trace,
            battery_trace=battery_trace,
            energy_trace=energy_trace,
            visited_trace=visited_trace,
            pos=nxt,
            battery_now=battery_now,
            total_energy=total_energy,
            visited=visited,
        )

    return battery_now, total_energy, newly_delivered


def _urgency_slack(
    stop: Coord,
    current: Coord,
    sim_time: float,
    deadlines: dict[Coord, float],
) -> float:
    """Estimated remaining slack before this stop's deadline expires.

    Lower slack = higher urgency = serve this stop sooner.
    Uses Manhattan distance as a fast travel-time estimate.
    """
    dist_estimate = abs(stop[0] - current[0]) + abs(stop[1] - current[1])
    return deadlines.get(stop, float("inf")) - sim_time - dist_estimate


def _choose_target_from_remaining(
    grid: GridMap,
    start: Coord,
    remaining: list[Coord],
    battery_now: float,
    battery_capacity: float,
    model: EnergyModel,
    sim_time: float = 0.0,
    deadlines: dict[Coord, float] | None = None,
) -> TargetChoice:
    """Choose the most urgency-critical delivery stop that is battery-safe.

    Remaining stops are sorted by their time-window slack (deadline minus
    estimated arrival time). The most urgent safe stop is served first.
    When this differs from the original randomised order, an urgency reorder
    is recorded — this is the novel VRPTW-inspired behaviour.
    """
    if deadlines:
        urgency_order = sorted(
            remaining,
            key=lambda s: _urgency_slack(s, start, sim_time, deadlines),
        )
    else:
        urgency_order = list(remaining)

    total_nodes = 0
    for i, target in enumerate(urgency_order):
        choice = _choose_mode_with_return_safety(
            grid=grid,
            start=start,
            goal=target,
            base=grid.base,
            battery_now=battery_now,
            battery_capacity=battery_capacity,
            model=model,
        )
        total_nodes += choice.nodes_expanded
        if choice.mode is not None:
            reordered = (i > 0 or (remaining and target != remaining[0]))
            return TargetChoice(
                target=target,
                choice=ModeChoice(
                    mode=choice.mode,
                    path=choice.path,
                    reason=choice.reason
                    if not reordered
                    else f"Urgency reorder: {target} is more critical than {remaining[0]}.",
                    nodes_expanded=total_nodes,
                ),
                urgency_reorder=reordered,
            )

    return TargetChoice(
        target=None,
        choice=ModeChoice(
            mode=None,
            path=[],
            reason="No remaining stop is safely reachable with the current battery.",
            nodes_expanded=total_nodes,
        ),
        urgency_reorder=False,
    )


def _choose_mode_with_return_safety(
    grid: GridMap,
    start: Coord,
    goal: Coord,
    base: Coord,
    battery_now: float,
    battery_capacity: float,
    model: EnergyModel,
) -> ModeChoice:
    """Pick FAST or ECO for one target, but only if returning to base stays safe."""
    energy_cost_fn = make_energy_move_cost(model)

    # The smart-city extension adds a V2X priority-corridor mode that can beat
    # plain FAST by using coordinated green-wave roads.
    v2x = astar(grid, start, goal, move_cost=v2x_time_move_cost)
    fast = astar(grid, start, goal, move_cost=time_move_cost)
    eco = astar(grid, start, goal, move_cost=energy_cost_fn)
    nodes = v2x.nodes_expanded + fast.nodes_expanded + eco.nodes_expanded

    if v2x.found and is_path_battery_feasible(
        grid=grid,
        path=v2x.path,
        start_battery=battery_now,
        model=model,
        capacity=battery_capacity,
    ):
        battery_after_v2x = end_battery_after_path(
            grid=grid,
            path=v2x.path,
            start_battery=battery_now,
            model=model,
            capacity=battery_capacity,
        )
        v2x_ret = astar(grid, goal, base, move_cost=energy_cost_fn)
        nodes += v2x_ret.nodes_expanded
        if v2x_ret.found and is_path_battery_feasible(
            grid=grid,
            path=v2x_ret.path,
            start_battery=battery_after_v2x,
            model=model,
            capacity=battery_capacity,
        ):
            fast_time = fast.total_cost if fast.found else float("inf")
            v2x_uses_corridor = any(
                cell in grid.smart_corridors for cell in v2x.path)
            if v2x.total_cost + 0.05 < fast_time and v2x_uses_corridor:
                return ModeChoice(
                    mode="V2X",
                    path=v2x.path,
                    reason="V2X selected (smart corridor provides a faster safe route).",
                    nodes_expanded=nodes,
                )

    # Always evaluate FAST next because the baseline project policy prefers it
    # whenever it is safe enough and V2X is not clearly better.
    if fast.found and is_path_battery_feasible(
        grid=grid,
        path=fast.path,
        start_battery=battery_now,
        model=model,
        capacity=battery_capacity,
    ):
        battery_after_fast = end_battery_after_path(
            grid=grid,
            path=fast.path,
            start_battery=battery_now,
            model=model,
            capacity=battery_capacity,
        )
        # Even when FAST is used outbound, the safety check back to base is
        # done with ECO costs because it is the more battery-conscious option.
        fast_ret = astar(grid, goal, base, move_cost=energy_cost_fn)
        nodes += fast_ret.nodes_expanded
        if fast_ret.found and is_path_battery_feasible(
            grid=grid,
            path=fast_ret.path,
            start_battery=battery_after_fast,
            model=model,
            capacity=battery_capacity,
        ):
            return ModeChoice(
                mode="FAST",
                path=fast.path,
                reason="FAST selected (can still safely return to base).",
                nodes_expanded=nodes,
            )

    if eco.found and is_path_battery_feasible(
        grid=grid,
        path=eco.path,
        start_battery=battery_now,
        model=model,
        capacity=battery_capacity,
    ):
        battery_after_eco = end_battery_after_path(
            grid=grid,
            path=eco.path,
            start_battery=battery_now,
            model=model,
            capacity=battery_capacity,
        )
        eco_ret = astar(grid, goal, base, move_cost=energy_cost_fn)
        nodes += eco_ret.nodes_expanded
        if eco_ret.found and is_path_battery_feasible(
            grid=grid,
            path=eco_ret.path,
            start_battery=battery_after_eco,
            model=model,
            capacity=battery_capacity,
        ):
            return ModeChoice(
                mode="ECO",
                path=eco.path,
                reason="ECO selected (FAST not safe/feasible).",
                nodes_expanded=nodes,
            )

    return ModeChoice(
        mode=None,
        path=[],
        reason="Neither FAST nor ECO can safely reach target and keep return-to-base feasible.",
        nodes_expanded=nodes,
    )


def _failure_result(
    reason: str,
    stop_order: list[Coord],
    completed: list[Coord],
    recharge_count: int,
    legs: list[ExecutedLeg],
    total_time: float,
    total_energy: float,
    nodes_expanded: int,
    runtime_start: float,
    battery_start: float,
    battery_now: float,
    hub_charge_count: int,
    hub_charge_events: list[Coord],
    full_path: list[Coord],
    battery_trace: list[float],
    energy_trace: list[float],
    visited_trace: list[set[Coord]],
    delivery_deadlines: dict[Coord, float] | None = None,
    late_deliveries: list[Coord] | None = None,
    urgency_reorder_count: int = 0,
) -> MissionResult:
    """Create a failure result without duplicating the same constructor block."""
    return MissionResult(
        success=False,
        reason=reason,
        stop_order=stop_order,
        completed_stops=completed,
        recharge_count=recharge_count,
        hub_charge_count=hub_charge_count,
        legs=legs,
        total_time=total_time,
        total_energy=total_energy,
        nodes_expanded=nodes_expanded,
        runtime_ms=(perf_counter() - runtime_start) * 1000.0,
        battery_start=battery_start,
        battery_end=battery_now,
        hub_charge_events=list(hub_charge_events),
        full_path=full_path,
        battery_trace=battery_trace,
        energy_trace=energy_trace,
        visited_trace=visited_trace,
        delivery_deadlines=delivery_deadlines or {},
        late_deliveries=late_deliveries or [],
        urgency_reorder_count=urgency_reorder_count,
    )


def _choose_charge_hub(
    grid: GridMap,
    start: Coord,
    remaining: list[Coord],
    battery_now: float,
    battery_capacity: float,
    model: EnergyModel,
) -> HubChoice:
    """Choose a reachable smart charging hub that unlocks future deliveries.

    This models a futuristic curbside wireless charging network: the robot can
    detour to a city charging hub, top up, and continue the mission without
    having to retreat all the way back to base.
    """
    if not grid.charging_hubs:
        return HubChoice(
            hub=None,
            path=[],
            reason="No charging hubs are available on this map.",
            nodes_expanded=0,
        )

    energy_cost_fn = make_energy_move_cost(model)
    total_nodes = 0
    best_hub: Coord | None = None
    best_path: list[Coord] = []
    best_key: tuple[int, int, int, int] | None = None

    for hub in sorted(grid.charging_hubs):
        if hub == start:
            continue
        to_hub = astar(grid, start, hub, move_cost=energy_cost_fn)
        total_nodes += to_hub.nodes_expanded
        if not to_hub.found or not is_path_battery_feasible(
            grid=grid,
            path=to_hub.path,
            start_battery=battery_now,
            model=model,
            capacity=battery_capacity,
        ):
            continue

        reachable_targets = 0
        best_after_hub_steps: int | None = None
        best_after_hub_mode_rank = 1

        for target in remaining:
            after_hub = _choose_mode_with_return_safety(
                grid=grid,
                start=hub,
                goal=target,
                base=grid.base,
                battery_now=battery_capacity,
                battery_capacity=battery_capacity,
                model=model,
            )
            total_nodes += after_hub.nodes_expanded
            if after_hub.mode is None:
                continue
            reachable_targets += 1
            steps = len(after_hub.path) - 1
            mode_rank = 0 if after_hub.mode == "FAST" else 1
            if best_after_hub_steps is None or (steps, mode_rank) < (
                best_after_hub_steps,
                best_after_hub_mode_rank,
            ):
                best_after_hub_steps = steps
                best_after_hub_mode_rank = mode_rank

        if reachable_targets == 0 or best_after_hub_steps is None:
            continue

        candidate_key = (
            len(to_hub.path) - 1,
            -reachable_targets,
            best_after_hub_steps,
            best_after_hub_mode_rank,
        )
        if best_key is None or candidate_key < best_key:
            best_key = candidate_key
            best_hub = hub
            best_path = to_hub.path

    if best_hub is None:
        return HubChoice(
            hub=None,
            path=[],
            reason="No charging hub is reachable or useful right now.",
            nodes_expanded=total_nodes,
        )

    return HubChoice(
        hub=best_hub,
        path=best_path,
        reason=f"Detouring to smart charging hub {best_hub} unlocks more deliveries.",
        nodes_expanded=total_nodes,
    )


def _assign_deadlines(
    grid: GridMap,
    stops: list[Coord],
    rng: random.Random,
) -> dict[Coord, float]:
    """Assign a completion deadline (in sim-time steps) to each delivery stop.

    Deadlines are proportional to the Manhattan distance from base, scaled by
    a random tightness factor. Tighter deadlines force the robot to prioritise
    urgency over its natural route preference — the VRPTW novel behaviour.
    """
    # Deadlines are set relative to a rough total-mission time estimate so that
    # some stops are comfortably achievable, some require urgency reordering to
    # make, and some may still be missed — a realistic VRPTW scenario.
    n = max(1, len(stops))
    mission_budget = 55.0 * n  # rough estimate: ~55 sim-steps per stop

    deadlines: dict[Coord, float] = {}
    for stop in stops:
        dist = abs(stop[0] - grid.base[0]) + abs(stop[1] - grid.base[1])
        # Allocate a random fraction of the mission budget to this stop.
        # Lower fraction = tighter deadline = more urgency pressure.
        fraction = rng.uniform(0.12, 0.55)
        deadlines[stop] = max(dist * 1.4, mission_budget * fraction + dist)
    return deadlines


def simulate_random_stops_with_recharge(
    grid: GridMap,
    battery_start: float,
    battery_capacity: float,
    model: EnergyModel,
    stop_seed: int,
) -> MissionResult:
    """Run the full mission with a randomized stop order and base recharging.

    The high-level behavior is:
    1. Shuffle the delivery list and assign time-window deadlines (VRPTW novel feature).
    2. At each decision point, re-sort remaining stops by urgency (least slack first).
    3. Try the most urgent stop using the FAST/ECO safety policy.
    4. If it is not safe, return to base and recharge.
    5. Continue until all stops are complete, then return home.

    Urgency-driven deadline resequencing is the novel feature: the robot dynamically
    re-sorts remaining stops by slack (deadline - sim_time - travel_estimate) at every
    planning step, implementing a greedy VRPTW approximation in real time.
    """
    runtime_start = perf_counter()
    rng = random.Random(stop_seed)

    # The randomized order is stored so the user can see the exact sequence
    # chosen for this run.
    stop_order = list(grid.deliveries)
    rng.shuffle(stop_order)
    remaining = list(stop_order)

    # Assign time-window deadlines (novel VRPTW feature)
    delivery_deadlines = _assign_deadlines(grid, stop_order, rng)
    late_deliveries: list[Coord] = []
    urgency_reorder_count = 0

    current = grid.base
    battery_now = battery_start
    total_energy = 0.0
    total_time = 0.0
    nodes_expanded = 0
    recharge_count = 0
    hub_charge_count = 0
    completed: list[Coord] = []
    visited: set[Coord] = set()
    legs: list[ExecutedLeg] = []
    hub_charge_events: list[Coord] = []

    # These traces drive the animation and live status panel.
    full_path: list[Coord] = [current]
    battery_trace: list[float] = [battery_now]
    energy_trace: list[float] = [total_energy]
    visited_trace: list[set[Coord]] = [set(visited)]

    energy_cost_fn = make_energy_move_cost(model)

    while remaining:
        # Re-sort remaining stops by urgency at each decision point (VRPTW feature).
        target_choice = _choose_target_from_remaining(
            grid=grid,
            start=current,
            remaining=remaining,
            battery_now=battery_now,
            battery_capacity=battery_capacity,
            model=model,
            sim_time=total_time,
            deadlines=delivery_deadlines,
        )
        target = target_choice.target
        choice = target_choice.choice
        nodes_expanded += choice.nodes_expanded

        if target_choice.urgency_reorder:
            urgency_reorder_count += 1

        if choice.mode is None:
            if current == grid.base:
                if battery_now < battery_capacity:
                    # At base, recharge immediately and try the same target again.
                    battery_now = battery_capacity
                    recharge_count += 1
                    _append_idle_frame(
                        path_trace=full_path,
                        battery_trace=battery_trace,
                        energy_trace=energy_trace,
                        visited_trace=visited_trace,
                        pos=current,
                        battery_now=battery_now,
                        total_energy=total_energy,
                        visited=visited,
                    )
                    continue
                # Even after a full recharge, this target is still not safe.
                return _failure_result(
                    reason="No remaining stop is safely reachable even from full charge at base.",
                    stop_order=stop_order,
                    completed=completed,
                    recharge_count=recharge_count,
                    legs=legs,
                    total_time=total_time,
                    total_energy=total_energy,
                    nodes_expanded=nodes_expanded,
                    runtime_start=runtime_start,
                    battery_start=battery_start,
                    battery_now=battery_now,
                    hub_charge_count=hub_charge_count,
                    hub_charge_events=hub_charge_events,
                    full_path=full_path,
                    battery_trace=battery_trace,
                    energy_trace=energy_trace,
                    visited_trace=visited_trace,
                    delivery_deadlines=delivery_deadlines,
                    late_deliveries=late_deliveries,
                    urgency_reorder_count=urgency_reorder_count,
                )

            if current in grid.charging_hubs and battery_now < battery_capacity:
                battery_now = battery_capacity
                hub_charge_count += 1
                hub_charge_events.append(current)
                _append_idle_frame(
                    path_trace=full_path,
                    battery_trace=battery_trace,
                    energy_trace=energy_trace,
                    visited_trace=visited_trace,
                    pos=current,
                    battery_now=battery_now,
                    total_energy=total_energy,
                    visited=visited,
                )
                continue

            hub_choice = _choose_charge_hub(
                grid=grid,
                start=current,
                remaining=remaining,
                battery_now=battery_now,
                battery_capacity=battery_capacity,
                model=model,
            )
            nodes_expanded += hub_choice.nodes_expanded
            if hub_choice.hub is not None:
                leg_battery_start = battery_now
                battery_now, total_energy, newly_delivered = _append_segment_trace(
                    grid=grid,
                    model=model,
                    segment=hub_choice.path,
                    battery_capacity=battery_capacity,
                    visited=visited,
                    path_trace=full_path,
                    battery_trace=battery_trace,
                    energy_trace=energy_trace,
                    visited_trace=visited_trace,
                    battery_now=battery_now,
                    total_energy=total_energy,
                )
                legs.append(
                    ExecutedLeg(
                        leg_type="HUB_CHARGE",
                        mode="ECO",
                        start=current,
                        goal=hub_choice.hub,
                        path=hub_choice.path,
                        time_cost=path_time_cost(hub_choice.path),
                        energy_cost=path_energy_delta(
                            grid, hub_choice.path, model),
                        battery_start=leg_battery_start,
                        battery_end=battery_now,
                    )
                )
                total_time += path_time_cost(hub_choice.path)
                current = hub_choice.hub
                for delivered in newly_delivered:
                    if delivered not in completed:
                        completed.append(delivered)
                    if delivered in remaining:
                        remaining.remove(delivered)
                visited_trace[-1] = set(visited)
                battery_now = battery_capacity
                hub_charge_count += 1
                hub_charge_events.append(current)
                _append_idle_frame(
                    path_trace=full_path,
                    battery_trace=battery_trace,
                    energy_trace=energy_trace,
                    visited_trace=visited_trace,
                    pos=current,
                    battery_now=battery_now,
                    total_energy=total_energy,
                    visited=visited,
                )
                continue

            # Not enough safe margin for the next target, so retreat home first.
            to_base = astar(grid, current, grid.base, move_cost=energy_cost_fn)
            nodes_expanded += to_base.nodes_expanded
            if not to_base.found or not is_path_battery_feasible(
                grid=grid,
                path=to_base.path,
                start_battery=battery_now,
                model=model,
                capacity=battery_capacity,
            ):
                return _failure_result(
                    reason="Battery too low to reach base for recharge (robot stranded).",
                    stop_order=stop_order,
                    completed=completed,
                    recharge_count=recharge_count,
                    legs=legs,
                    total_time=total_time,
                    total_energy=total_energy,
                    nodes_expanded=nodes_expanded,
                    runtime_start=runtime_start,
                    battery_start=battery_start,
                    battery_now=battery_now,
                    hub_charge_count=hub_charge_count,
                    hub_charge_events=hub_charge_events,
                    full_path=full_path,
                    battery_trace=battery_trace,
                    energy_trace=energy_trace,
                    visited_trace=visited_trace,
                    delivery_deadlines=delivery_deadlines,
                    late_deliveries=late_deliveries,
                    urgency_reorder_count=urgency_reorder_count,
                )

            leg_battery_start = battery_now
            battery_now, total_energy, _ = _append_segment_trace(
                grid=grid,
                model=model,
                segment=to_base.path,
                battery_capacity=battery_capacity,
                visited=visited,
                path_trace=full_path,
                battery_trace=battery_trace,
                energy_trace=energy_trace,
                visited_trace=visited_trace,
                battery_now=battery_now,
                total_energy=total_energy,
            )
            legs.append(
                ExecutedLeg(
                    leg_type="RETURN_RECHARGE",
                    mode="ECO",
                    start=current,
                    goal=grid.base,
                    path=to_base.path,
                    time_cost=path_time_cost(to_base.path),
                    energy_cost=path_energy_delta(grid, to_base.path, model),
                    battery_start=leg_battery_start,
                    battery_end=battery_now,
                )
            )
            total_time += path_time_cost(to_base.path)
            current = grid.base

            # Recharging is modeled as an instant battery reset at base.
            battery_now = battery_capacity
            recharge_count += 1
            _append_idle_frame(
                path_trace=full_path,
                battery_trace=battery_trace,
                energy_trace=energy_trace,
                visited_trace=visited_trace,
                pos=current,
                battery_now=battery_now,
                total_energy=total_energy,
                visited=visited,
            )
            continue

        # Execute the chosen delivery leg and update mission state.
        leg_battery_start = battery_now
        battery_now, total_energy, newly_delivered = _append_segment_trace(
            grid=grid,
            model=model,
            segment=choice.path,
            battery_capacity=battery_capacity,
            visited=visited,
            path_trace=full_path,
            battery_trace=battery_trace,
            energy_trace=energy_trace,
            visited_trace=visited_trace,
            battery_now=battery_now,
            total_energy=total_energy,
        )
        legs.append(
            ExecutedLeg(
                leg_type="DELIVERY",
                mode=choice.mode,
                start=current,
                goal=choice.path[-1],
                path=choice.path,
                time_cost=path_time_cost(choice.path),
                energy_cost=path_energy_delta(grid, choice.path, model),
                battery_start=leg_battery_start,
                battery_end=battery_now,
            )
        )
        total_time += path_time_cost(choice.path)
        current = choice.path[-1]
        for delivered in newly_delivered:
            if delivered not in completed:
                completed.append(delivered)
                deadline = delivery_deadlines.get(delivered, float("inf"))
                if total_time > deadline:
                    late_deliveries.append(delivered)
            if delivered in remaining:
                remaining.remove(delivered)
        # Update the final frame of the just-finished segment so the animation
        # reflects any deliveries that were crossed during this path.
        visited_trace[-1] = set(visited)

    if current != grid.base:
        # Mission success requires ending at base, so add one final return leg.
        final_return = astar(grid, current, grid.base,
                             move_cost=energy_cost_fn)
        nodes_expanded += final_return.nodes_expanded
        if not final_return.found or not is_path_battery_feasible(
            grid=grid,
            path=final_return.path,
            start_battery=battery_now,
            model=model,
            capacity=battery_capacity,
        ):
            return _failure_result(
                reason="Completed stops but cannot return to base safely.",
                stop_order=stop_order,
                completed=completed,
                recharge_count=recharge_count,
                legs=legs,
                total_time=total_time,
                total_energy=total_energy,
                nodes_expanded=nodes_expanded,
                runtime_start=runtime_start,
                battery_start=battery_start,
                battery_now=battery_now,
                hub_charge_count=hub_charge_count,
                hub_charge_events=hub_charge_events,
                full_path=full_path,
                battery_trace=battery_trace,
                energy_trace=energy_trace,
                visited_trace=visited_trace,
                delivery_deadlines=delivery_deadlines,
                late_deliveries=late_deliveries,
                urgency_reorder_count=urgency_reorder_count,
            )

        leg_battery_start = battery_now
        battery_now, total_energy, _ = _append_segment_trace(
            grid=grid,
            model=model,
            segment=final_return.path,
            battery_capacity=battery_capacity,
            visited=visited,
            path_trace=full_path,
            battery_trace=battery_trace,
            energy_trace=energy_trace,
            visited_trace=visited_trace,
            battery_now=battery_now,
            total_energy=total_energy,
        )
        legs.append(
            ExecutedLeg(
                leg_type="FINAL_RETURN",
                mode="ECO",
                start=current,
                goal=grid.base,
                path=final_return.path,
                time_cost=path_time_cost(final_return.path),
                energy_cost=path_energy_delta(grid, final_return.path, model),
                battery_start=leg_battery_start,
                battery_end=battery_now,
            )
        )
        total_time += path_time_cost(final_return.path)

    return MissionResult(
        success=True,
        reason="Completed all randomized stops and returned to base.",
        stop_order=stop_order,
        completed_stops=completed,
        recharge_count=recharge_count,
        hub_charge_count=hub_charge_count,
        legs=legs,
        total_time=total_time,
        total_energy=total_energy,
        nodes_expanded=nodes_expanded,
        runtime_ms=(perf_counter() - runtime_start) * 1000.0,
        battery_start=battery_start,
        battery_end=battery_now,
        hub_charge_events=list(hub_charge_events),
        full_path=full_path,
        battery_trace=battery_trace,
        energy_trace=energy_trace,
        visited_trace=visited_trace,
        delivery_deadlines=delivery_deadlines,
        late_deliveries=late_deliveries,
        urgency_reorder_count=urgency_reorder_count,
    )


def _find_demo_simulation(
    map_width: int,
    map_height: int,
    obstacle_ratio: float,
    num_deliveries: int,
    battery_start: float,
    battery_capacity: float,
    model: EnergyModel,
    map_seed: int | None = None,
    stop_seed: int | None = None,
    max_attempts: int = 420,
) -> tuple[GridMap, int, int, MissionResult]:
    """Search for a demo scenario that actually finishes successfully.

    We prefer a run that visibly uses the new charging-hub technology or at
    least includes a recharge, because that is more interesting to present.
    """
    fallback_success: tuple[GridMap, int, int, MissionResult] | None = None
    seed_rng = random.SystemRandom()
    attempts = 1 if map_seed is not None and stop_seed is not None else max_attempts

    for attempt in range(attempts):
        chosen_map_seed = map_seed if map_seed is not None else seed_rng.randrange(
            1, 1_000_000_000)
        chosen_stop_seed = stop_seed if stop_seed is not None else seed_rng.randrange(
            1, 1_000_000_000)
        grid = generate_sample_map(
            width=map_width,
            height=map_height,
            obstacle_ratio=obstacle_ratio,
            num_deliveries=num_deliveries,
            seed=chosen_map_seed,
        )
        result = simulate_random_stops_with_recharge(
            grid=grid,
            battery_start=battery_start,
            battery_capacity=battery_capacity,
            model=model,
            stop_seed=chosen_stop_seed,
        )

        if not result.success:
            continue
        if map_seed is not None and stop_seed is not None:
            return grid, chosen_map_seed, chosen_stop_seed, result
        if (result.hub_charge_count > 0 or result.recharge_count > 0) and result.urgency_reorder_count > 0:
            return grid, chosen_map_seed, chosen_stop_seed, result
        if fallback_success is None:
            fallback_success = (grid, chosen_map_seed,
                                chosen_stop_seed, result)

    if fallback_success is not None:
        return fallback_success
    raise RuntimeError("Unable to find a successful simulation scenario.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the delivery robot city simulator.")
    parser.add_argument("--map-seed", type=int, default=None,
                        help="Optional fixed city seed.")
    parser.add_argument("--stop-seed", type=int, default=None,
                        help="Optional fixed delivery-order seed.")
    return parser.parse_args()


def _run_random_city(
    model: EnergyModel,
    map_seed: int | None,
    stop_seed: int | None,
) -> tuple[GridMap, int, int, MissionResult, float, float, float]:
    map_width = 60
    map_height = 42
    obstacle_ratio = 0.18
    num_deliveries = 8
    battery_capacity = 200.0
    battery_start = 100.0
    grid, map_seed, stop_seed, result = _find_demo_simulation(
        map_width=map_width,
        map_height=map_height,
        obstacle_ratio=obstacle_ratio,
        num_deliveries=num_deliveries,
        battery_start=battery_start,
        battery_capacity=battery_capacity,
        model=model,
        map_seed=map_seed,
        stop_seed=stop_seed,
    )
    return grid, map_seed, stop_seed, result, battery_start, battery_capacity, obstacle_ratio


def run_step8_demo(args: argparse.Namespace | None = None) -> None:
    """Top-level demo entry point for the current simulator stage."""
    args = args or _parse_args()
    model = EnergyModel(base_move=1.0, k_up=1.25,
                        k_down=0.55, min_planning_cost=0.05)
    grid, map_seed, stop_seed, result, battery_start, battery_capacity, obstacle_ratio = _run_random_city(
        model,
        args.map_seed,
        args.stop_seed,
    )
    scenario = "random_sidewalk_bikelane_city_sim"

    # Persist the run so repeated demos build a dataset for the report.
    run_metrics = build_run_metrics(
        scenario=scenario,
        seed=map_seed,
        map_width=grid.width,
        map_height=grid.height,
        obstacle_ratio=obstacle_ratio,
        num_deliveries=len(result.stop_order),
        completed_deliveries=len(result.completed_stops),
        success=result.success,
        total_time_cost=result.total_time,
        total_energy_cost=result.total_energy,
        nodes_expanded=result.nodes_expanded,
        runtime_ms=result.runtime_ms,
        battery_start=battery_start,
        battery_end=result.battery_end,
        battery_capacity=battery_capacity,
        reason=f"{result.reason} Recharge={result.recharge_count} Late={len(result.late_deliveries)}",
    )
    csv_path = Path(__file__).resolve().parent / "results" / "mission_runs.csv"
    saved_csv = append_run_metrics_csv(run_metrics, csv_path)

    sep = "=" * 64
    thin = "-" * 64
    print(sep)
    print("  AUTONOMOUS DELIVERY ROBOT SIMULATOR")
    print("  Battery-Constrained Urban Navigation with VRPTW Deadline Scheduling")
    print(sep)

    print("\n  MAP CONFIGURATION")
    print(thin)
    print(
        f"  Grid size        : {grid.width} x {grid.height}  ({grid.width * grid.height:,} cells)")
    print(f"  Map seed         : {map_seed}")
    print(f"  Stop-order seed  : {stop_seed}")
    print(f"  Obstacle ratio   : {obstacle_ratio:.0%}")
    print(f"  Sidewalks        : {len(grid.sidewalks):,} cells")
    print(f"  Bike lanes       : {len(grid.bike_lanes):,} cells")
    print(f"  V2X corridors    : {len(grid.smart_corridors):,} cells")
    print(
        f"  Charging hubs    : {len(grid.charging_hubs)}  at {sorted(grid.charging_hubs)}")
    print(f"  Base station     : {grid.base}")
    print(f"  Deliveries       : {len(result.stop_order)} stops")
    print(
        f"  Battery          : {battery_start:.0f} / {battery_capacity:.0f}  (start / capacity)")
    print(
        f"  Energy model     : base={model.base_move}  k_up={model.k_up}  k_down={model.k_down} (regen)")

    print(
        "\n  DELIVERY DEADLINES  [novel feature: urgency-driven VRPTW resequencing]")
    print(thin)
    sorted_stops = sorted(
        result.delivery_deadlines.items(), key=lambda x: x[1])
    for stop, deadline in sorted_stops:
        dist = abs(stop[0] - grid.base[0]) + abs(stop[1] - grid.base[1])
        tightness = deadline / max(dist, 1)
        status = "LATE" if stop in result.late_deliveries else "on time"
        urgency = "TIGHT" if tightness < 2.5 else (
            "moderate" if tightness < 4.0 else "comfortable")
        print(
            f"  Stop {str(stop):>10}  deadline={deadline:>6.1f}  tightness={tightness:.2f}x  [{urgency}]  [{status}]")
    print(
        f"  On-time: {len(result.stop_order) - len(result.late_deliveries)} / {len(result.stop_order)}")
    print(
        f"  Urgency reorders: {result.urgency_reorder_count}  (times urgency overrode default route preference)")

    print("\n  MISSION RESULTS")
    print(thin)
    status = "SUCCESS" if result.success else "FAILURE"
    print(f"  Status           : {status}")
    print(f"  Outcome          : {result.reason}")
    print(
        f"  Stops completed  : {len(result.completed_stops)} / {len(result.stop_order)}")
    print(f"  Recharge trips   : {result.recharge_count}")
    print(f"  Smart hub charges: {result.hub_charge_count}")
    if result.hub_charge_events:
        print(f"  Hub charge cells : {result.hub_charge_events}")
    print(f"  Nodes expanded   : {result.nodes_expanded:,}")
    print(f"  Total time cost  : {result.total_time:.2f}")
    print(f"  Total energy     : {result.total_energy:.2f}")
    print(f"  Battery end      : {result.battery_end:.2f}")
    print(f"  Planning runtime : {result.runtime_ms:.1f} ms")
    print(f"  Metrics CSV      : {saved_csv}")

    print("\n  DELIVERY ORDER")
    print(thin)
    print(f"  Planned  : {result.stop_order}")
    print(f"  Actual   : {result.completed_stops}")

    print("\n  LEG LOG")
    print(thin)
    print(f"  {'#':>3}  {'Type':<15}  {'Mode':<4}  {'From':>8}  {'To':>8}  {'Steps':>5}  {'Time':>6}  {'Energy':>7}  {'Batt':>13}")
    for i, leg in enumerate(result.legs, start=1):
        print(
            f"  {i:>3}  {leg.leg_type:<15}  {leg.mode:<4}  "
            f"{str(leg.start):>8}  {str(leg.goal):>8}  {len(leg.path)-1:>5}  "
            f"{leg.time_cost:>6.1f}  {leg.energy_cost:>7.2f}  "
            f"{leg.battery_start:>5.1f}->{leg.battery_end:>5.1f}"
        )
    print()

    if result.full_path:
        plot_map_2d(
            grid,
            path=result.full_path,
            show=True,
            title="Delivery Robot — Mission Overview (2D)",
        )
        if "agg" not in matplotlib.get_backend().lower():
            global _LAST_ANIMATION
            _LAST_ANIMATION = animate_robot_2d(
                grid=grid,
                path=result.full_path,
                visited_trace=result.visited_trace,
                battery_trace=result.battery_trace,
                energy_trace=result.energy_trace,
                delivery_deadlines=result.delivery_deadlines,
                show=True,
                title="Delivery Robot — 2D Simulation  (red=tight deadline / orange=moderate / green=ok)",
                interval_ms=150,
                max_frames=220,
            )

        try:
            from render_panda3d import launch as _panda_launch
            print("\n" + "─" * 62)
            print("  INTERACTIVE 3D CITY  (Panda3D)")
            print("─" * 62)
            print("  WASD        move camera         C  follow / free cam")
            print("  Q / E       rise / descend       R  restart robot")
            print("  Arrow keys  look around          Esc  quit")
            print("─" * 62)
            _panda_launch(
                grid, result, title="Delivery Robot — Interactive 3D City")
        except ImportError:
            print("\n  Panda3D not installed — skipping 3D city view.")
            print("  Install with: pip install panda3d")


if __name__ == "__main__":
    run_step8_demo()
