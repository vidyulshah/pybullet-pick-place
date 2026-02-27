# PyBullet Pick-and-Place (Franka Panda)

Autonomous perception + control pipeline in PyBullet:

- Random colored cube spawning on a table (reachable region).
- Overhead + wrist-mounted cameras using PyBullet `getCameraImage`.
- Segmentation-based detection + simple color heuristic classification.
- Depth-buffer to metric depth conversion.
- Pixel-to-world back-projection using intrinsics/extrinsics.
- IK-driven top-down grasp: approach → descend → close → lift → place → retreat/survey.
- Retry behavior: if grasp fails, move up to survey pose, re-acquire, retry.

---

## 1. Dependencies

- Python 3.10 (recommended)
- `pybullet`
- `numpy`

---

## 2. Setup (Windows / Linux / macOS)

### Option A: Conda (recommended)

1. Create the environment:

```bash
conda env create -f environment.yml
```

2. Activate:

```bash
conda activate pb_pickplace
```

---

### Option B: Pip + venv

1. Create a virtual environment:

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 3. Run the Simulation

**GUI mode** (interactive; recommended for review):

```bash
python pick_place.py --gui --render --renderer tiny
```
