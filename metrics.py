"""Run metrics and CSV logging utilities.

The simulation and robot runs both produce summary statistics. This module
packages those statistics into a single record and appends them to CSV.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class RunMetrics:
    """Single-row schema for the results CSV."""

    run_id: str
    timestamp_utc: str
    scenario: str
    seed: int
    map_width: int
    map_height: int
    obstacle_ratio: float
    num_deliveries: int
    completed_deliveries: int
    success: bool
    total_time_cost: float
    total_energy_cost: float
    nodes_expanded: int
    runtime_ms: float
    battery_start: float
    battery_end: float
    battery_capacity: float
    reason: str


def build_run_metrics(
    scenario: str,
    seed: int,
    map_width: int,
    map_height: int,
    obstacle_ratio: float,
    num_deliveries: int,
    completed_deliveries: int,
    success: bool,
    total_time_cost: float,
    total_energy_cost: float,
    nodes_expanded: int,
    runtime_ms: float,
    battery_start: float,
    battery_end: float,
    battery_capacity: float,
    reason: str,
) -> RunMetrics:
    """Build a metrics row with a timestamped run identifier."""
    now = datetime.now(timezone.utc)
    run_id = f"{now.strftime('%Y%m%dT%H%M%S')}-seed{seed}"
    return RunMetrics(
        run_id=run_id,
        timestamp_utc=now.isoformat(),
        scenario=scenario,
        seed=seed,
        map_width=map_width,
        map_height=map_height,
        obstacle_ratio=obstacle_ratio,
        num_deliveries=num_deliveries,
        completed_deliveries=completed_deliveries,
        success=success,
        total_time_cost=total_time_cost,
        total_energy_cost=total_energy_cost,
        nodes_expanded=nodes_expanded,
        runtime_ms=runtime_ms,
        battery_start=battery_start,
        battery_end=battery_end,
        battery_capacity=battery_capacity,
        reason=reason,
    )


def append_run_metrics_csv(metrics: RunMetrics, csv_path: str | Path) -> Path:
    """Append one metrics row to CSV, creating the file and header if needed."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = asdict(metrics)
    # Write the header only the first time the file is created.
    write_header = (not path.exists()) or (path.stat().st_size == 0)

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return path
