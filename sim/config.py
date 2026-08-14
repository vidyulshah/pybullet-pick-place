"""
sim/config.py
All runtime configuration. Loaded from config.yaml at startup.
Defaults are safe and match the original prototype behaviour.
"""
from __future__ import annotations

import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Tuple


# ── Simulation ────────────────────────────────────────────────────────────────
@dataclass
class SimConfig:
    gui:            bool  = True
    timestep:       float = 1.0 / 240.0
    gravity_z:      float = -9.81
    seed:           int   = 0
    renderer:       str   = "tiny"        # "tiny" | "opengl" | "auto"
    sleep_if_gui:   bool  = True


# ── Object spawning ───────────────────────────────────────────────────────────
@dataclass
class SpawnConfig:
    n_cubes:             int   = 4
    cube_size:           float = 0.04
    cube_mass:           float = 0.05
    margin_xy:           float = 0.16
    reach_min:           float = 0.20
    reach_max:           float = 0.60
    min_dist_xy:         float = 0.10
    max_tries_per_cube:  int   = 900
    spawn_clearance:     float = 0.003


# ── Cameras ───────────────────────────────────────────────────────────────────
@dataclass
class CameraConfig:
    width:          int   = 320
    height:         int   = 220
    fov_y_deg:      float = 75.0
    near:           float = 0.02
    far:            float = 5.0
    ee_cam_offset:  Tuple[float, float, float] = (0.0, 0.0, 0.10)


# ── Grasp motion ──────────────────────────────────────────────────────────────
@dataclass
class GraspConfig:
    approach_height:          float = 0.14
    lift_height:              float = 0.18
    ik_max_iters:             int   = 180
    ik_residual:              float = 1e-4
    traj_steps:               int   = 180
    gripper_open:             float = 0.04
    gripper_closed:           float = 0.0
    gripper_settle_seconds:   float = 0.30
    success_lift_threshold:   float = 0.05
    collision_check_dist:     float = 0.010
    descend_top_offsets:      Tuple[float, ...] = (0.035, 0.025, 0.018, 0.012, 0.008, 0.006)
    place_top_offset:         float = 0.030
    place_settle_seconds:     float = 0.20
    max_pick_retries:         int   = 2
    survey_height_above_table:float = 0.65
    survey_forward_from_base: float = 0.20


# ── Pipeline behaviour ────────────────────────────────────────────────────────
@dataclass
class PipelineConfig:
    autonomous:                 bool  = True   # False = interactive terminal mode
    max_consecutive_failures:   int   = 3      # triggers ESTOP
    inter_pick_pause_s:         float = 0.5    # settle time between picks


# ── Top-level bundle ──────────────────────────────────────────────────────────
@dataclass
class AppConfig:
    sim:      SimConfig      = field(default_factory=SimConfig)
    spawn:    SpawnConfig    = field(default_factory=SpawnConfig)
    camera:   CameraConfig   = field(default_factory=CameraConfig)
    grasp:    GraspConfig    = field(default_factory=GraspConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    log_dir:      str = "logs"
    metrics_csv:  str = "logs/metrics.csv"


# ── Loader ────────────────────────────────────────────────────────────────────
def load_config(path: Optional[str] = "config.yaml") -> AppConfig:
    """Load AppConfig from a YAML file.  Falls back to defaults if file absent."""
    cfg = AppConfig()
    if path is None or not Path(path).exists():
        return cfg
    with open(path) as f:
        data = yaml.safe_load(f) or {}

    def _merge(dc, key):
        if key in data:
            return dc.__class__(**{**asdict(dc), **data[key]})
        return dc

    cfg.sim      = _merge(cfg.sim,      "sim")
    cfg.spawn    = _merge(cfg.spawn,    "spawn")
    cfg.camera   = _merge(cfg.camera,   "camera")
    cfg.grasp    = _merge(cfg.grasp,    "grasp")
    cfg.pipeline = _merge(cfg.pipeline, "pipeline")
    cfg.log_dir     = data.get("log_dir",     cfg.log_dir)
    cfg.metrics_csv = data.get("metrics_csv", cfg.metrics_csv)
    return cfg
