# PyBullet Pick-and-Place (Franka Panda) — Industry Edition

Autonomous perception + control pipeline for a Franka Panda arm in PyBullet.
Upgraded from a single-script prototype to a production-ready package with:

- **Finite-state machine** — explicit INIT → SURVEY → DETECT → APPROACH → PICK → PLACE → RETREAT → DONE cycle
- **Structured logging** — timestamped file + console output via Python `logging`
- **CSV metrics** — per-pick result, duration, position, and success rate written to `logs/metrics.csv`
- **YAML config** — all parameters live in `config.yaml`; zero code changes needed to tune behaviour
- **Graceful shutdown** — SIGINT / SIGTERM handled cleanly; partial results always saved
- **Automatic ESTOP** — halts after N consecutive failures (configurable)
- **Modular package** — `sim/` sub-package; each concern in its own file

---

## Project Structure

```
pybullet-pick-place/
│
├── pick_place.py        # Entry point (thin — just wires modules together)
├── config.yaml          # All runtime parameters
├── requirements.txt
├── README.md
│
└── sim/                 # Core package
    ├── __init__.py
    ├── config.py        # Dataclasses + YAML loader
    ├── logger.py        # Structured logging + CSV metrics
    ├── scene.py         # Scene build, robot setup, cube spawning
    ├── vision.py        # Camera rendering, depth, detection
    ├── motion.py        # IK, trajectory, gripper, survey pose
    ├── grasp.py         # Pick and place sequences
    └── pipeline.py      # Finite-state machine orchestrator
```

---

## What Changed vs the Prototype

| Area | Before | After |
|---|---|---|
| Architecture | Single 1 000-line file | 8-module package |
| Control flow | `input()` blocking loop | Autonomous FSM |
| Logging | `print()` | `logging` to file + console |
| Config | Hard-coded constants | `config.yaml` + CLI flags |
| Metrics | None | Per-pick CSV (`logs/metrics.csv`) |
| Shutdown | Ctrl-C only | SIGINT/SIGTERM → graceful exit |
| Safety | None | ESTOP after N consecutive failures |
| Retries | Fixed retry count | Configurable; re-detects between retries |

---

## Dependencies

- Python 3.10+ recommended
- `pybullet >= 3.2.5`
- `numpy >= 1.23`
- `PyYAML >= 6.0`

---

## Setup

### Option A — Conda (recommended)

```bash
conda create -n pb_pickplace python=3.10 -y
conda activate pb_pickplace
pip install -r requirements.txt
```

### Option B — venv

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

---

## Running

### GUI mode (default)
```bash
python pick_place.py
```

### Headless / CI mode
```bash
python pick_place.py --direct
```

### CLI overrides (no config.yaml edit needed)
```bash
python pick_place.py --n_cubes 6 --seed 42 --renderer opengl
```

### Custom config file
```bash
python pick_place.py --config my_config.yaml
```

---

## What You Should See

1. PyBullet GUI opens with the Panda robot, table, and coloured cubes.
2. Arm moves to survey pose (raised observation position).
3. Overhead camera detects all visible cubes.
4. FSM selects the best-visible cube, executes approach → pick → place.
5. After placing, arm retreats and re-surveys for the next cube.
6. Terminal state `DONE` is reached when all cubes are placed.

### Console output example
```
2026-08-14 10:22:01 | INFO     | pick_place.pipeline | [SURVEY] → [DETECT]
2026-08-14 10:22:01 | INFO     | pick_place.pipeline | Overhead detection: 3(cls=red,...); 4(cls=green,...)
2026-08-14 10:22:01 | INFO     | pick_place.pipeline | Target: cube_id=3  xy=(0.512, 0.023)  class=red
2026-08-14 10:22:03 | INFO     | pick_place.grasp    | Grasp SUCCESS at dz=0.025
2026-08-14 10:22:05 | INFO     | pick_place.grasp    | Place at (0.48, 0.18) complete.
```

### Metrics CSV (`logs/metrics.csv`)
```
timestamp,cycle,cube_id,color_class,pick_xy_x,pick_xy_y,place_xy_x,place_xy_y,attempts,result,duration_s,note
2026-08-14 10:22:05,1,3,red,0.512,0.023,0.48,0.18,1,success,4.32,
```

---

## Configuration

Edit `config.yaml` to change behaviour without touching code:

```yaml
pipeline:
  autonomous:               true   # false = interactive terminal prompts
  max_consecutive_failures: 3      # triggers ESTOP

spawn:
  n_cubes: 4

grasp:
  max_pick_retries: 2
```

All parameters and their defaults are documented in `sim/config.py`.

---

## License

MIT — free to use, modify and deploy.
