# Autonomous Delivery Robot Simulator

A Python simulator for an energy-aware delivery robot in a procedurally generated city. The robot plans routes across hilly terrain, picks between speed and battery-saving modes, recharges before it runs out, and orders its stops by delivery deadline. Each run is animated in 2D with matplotlib and can be explored in an interactive 3D city built with Panda3D.

<!-- Add a GIF of the 2D animation and a screenshot of the 3D city here, e.g. ![2D run](docs/2d.gif) -->

## Features

- **Procedural city.** Each run generates a 60×42 grid with a smooth terrain heightmap, roads, sidewalks, bike lanes, parks, plazas, buildings (taller downtown), and V2X smart-corridor priority lanes. Seeds make any run reproducible.
- **Path planning from scratch.** BFS, Dijkstra and A* (Manhattan heuristic) on a 4-neighbour grid, all sharing one `PlanResult` interface that reports the path, cost and nodes expanded.
- **Terrain-aware energy model.** Every move costs a base amount plus an uphill penalty, minus regeneration on downhill moves. The signed battery change is tracked exactly, while planners get a clipped non-negative cost so Dijkstra and A* stay correct.
- **Three driving modes per leg:**
  - **V2X:** time-optimal route using smart-city priority lanes
  - **FAST:** time-optimal route
  - **ECO:** energy-optimal route
  
  A mode is used only if the robot can still get back to base safely afterwards; otherwise it falls back to a safer option.
- **Recharging.** The robot returns to base or detours to a smart charging hub when the battery can't safely cover the next leg.
- **Deadlines.** Each stop gets a delivery time window. The planner picks the most urgent stop that's battery-safe, reordering when a deadline gets tight.
- **Metrics.** Each run appends a row to `results/mission_runs.csv`: seed, deliveries completed, success, total time and energy cost, nodes expanded, runtime, and start/end battery.

## Visualization

- **2D (matplotlib):** a static map plus an animation of the robot's route, with stops coloured by deadline pressure (red = tight, orange = moderate, green = OK).
- **3D (Panda3D, optional):** a real-time city with buildings, road markings, the full mission route and a robot model you can follow.

| Key | 3D controls |
|---|---|
| `W A S D` | Move camera |
| `Q` / `E` | Rise / descend |
| Arrow keys | Look around |
| `C` | Follow robot / free camera |
| `R` | Restart robot |
| `Esc` | Quit |

## Project structure

```
main.py             Mission simulation: target selection, mode choice, recharging, deadlines, CLI
planners.py         BFS, Dijkstra, A*
costs.py            Time and energy cost models, battery helpers
grid.py             GridMap model and procedural city / terrain generation
metrics.py          Run metrics and CSV logging
viz.py              matplotlib 2D map and animation
render_panda3d.py   Interactive Panda3D 3D city renderer
```

## Getting started

**Requirements:** Python 3.10+

```bash
git clone https://github.com/JasrajDhaliwal/AutonomousRobotSimulator.git
cd AutonomousRobotSimulator
python -m venv .venv && source .venv/bin/activate
pip install numpy matplotlib
pip install panda3d   # optional, for the 3D city view
python main.py
```

Reproduce a specific run with fixed seeds:

```bash
python main.py --map-seed 42 --stop-seed 7
```

If Panda3D isn't installed, the simulator skips the 3D view and runs everything else.

## Notes

Built as a solo project for COMP 361 at the University of the Fraser Valley.
