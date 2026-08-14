"""
sim/scene.py
Build the PyBullet scene: plane, table, Franka Panda robot, coloured cubes.
"""
from __future__ import annotations

import logging
import math
import random
from typing import Dict, List, Tuple

import numpy as np
import pybullet as p
import pybullet_data

from .config import AppConfig, SpawnConfig

log = logging.getLogger("pick_place.scene")


# ── PyBullet connection ───────────────────────────────────────────────────────
def connect(cfg: AppConfig) -> int:
    if cfg.sim.gui:
        cid = p.connect(p.GUI, options="--renderer=TinyRenderer")
    else:
        cid = p.connect(p.DIRECT)
    if cid < 0:
        raise RuntimeError("Failed to connect to PyBullet.")
    log.info(f"PyBullet connected  (cid={cid}, gui={cfg.sim.gui})")
    return cid


def choose_renderer(cfg: AppConfig) -> int:
    r = cfg.sim.renderer
    if r == "tiny":
        return p.ER_TINY_RENDERER
    if r == "opengl":
        return p.ER_BULLET_HARDWARE_OPENGL
    return p.ER_BULLET_HARDWARE_OPENGL if cfg.sim.gui else p.ER_TINY_RENDERER


# ── Robot helpers ─────────────────────────────────────────────────────────────
def set_neutral_panda(robot_id: int) -> None:
    neutral = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
    for j in range(7):
        p.resetJointState(robot_id, j, neutral[j])
    for j in [9, 10]:
        if j < p.getNumJoints(robot_id):
            p.resetJointState(robot_id, j, 0.04)


def find_end_effector_link(robot_id: int) -> int:
    preferred  = ["panda_hand", "panda_grasptarget", "panda_link8"]
    name_to_idx: Dict[str, int] = {}
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        name_to_idx[info[12].decode()] = j
    for nm in preferred:
        if nm in name_to_idx:
            log.debug(f"EE link '{nm}' → index {name_to_idx[nm]}")
            return name_to_idx[nm]
    fallback = min(11, p.getNumJoints(robot_id) - 1)
    log.warning(f"Preferred EE link not found; using index {fallback}")
    return fallback


def get_arm_ik_params(robot_id: int, arm_joints: List[int], rest_poses: List[float]):
    lowers, uppers, ranges = [], [], []
    for j in arm_joints:
        info = p.getJointInfo(robot_id, j)
        lo, hi = float(info[8]), float(info[9])
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            lo, hi = -2.967, 2.967
        lowers.append(lo)
        uppers.append(hi)
        ranges.append(hi - lo)
    return lowers, uppers, ranges, rest_poses


# ── Scene construction ────────────────────────────────────────────────────────
def build_scene(cfg: AppConfig):
    """
    Returns
    -------
    table_id, robot_id, table_center, z_top, robot_base_xy, base_yaw
    """
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setRealTimeSimulation(0)
    p.setGravity(0, 0, cfg.sim.gravity_z)
    p.setTimeStep(cfg.sim.timestep)

    p.loadURDF("plane.urdf")

    table_pos = [0.75, 0.0, 0.0]
    table_id  = p.loadURDF("table/table.urdf", table_pos, useFixedBase=True)
    aabb_min, aabb_max = p.getAABB(table_id)
    z_top = float(aabb_max[2])
    table_center = [
        (aabb_min[0] + aabb_max[0]) / 2.0,
        (aabb_min[1] + aabb_max[1]) / 2.0,
        z_top,
    ]

    x_base = aabb_min[0] + 0.22
    y_base = (aabb_min[1] + aabb_max[1]) / 2.0
    dx     = table_center[0] - x_base
    dy     = table_center[1] - y_base
    yaw    = float(math.atan2(dy, dx))
    orn    = p.getQuaternionFromEuler([0.0, 0.0, yaw])

    robot_id = p.loadURDF(
        "franka_panda/panda.urdf",
        [x_base, y_base, z_top + 0.002],
        orn,
        useFixedBase=True,
    )
    set_neutral_panda(robot_id)
    log.info(f"Scene built — table_id={table_id} robot_id={robot_id} z_top={z_top:.4f}")
    return table_id, robot_id, table_center, z_top, (x_base, y_base), yaw


# ── Cube spawning ─────────────────────────────────────────────────────────────
_PALETTE = [
    (1.0, 0.0, 0.0, 1.0),   # red
    (0.0, 1.0, 0.0, 1.0),   # green
    (0.0, 0.0, 1.0, 1.0),   # blue
    (1.0, 1.0, 0.0, 1.0),   # yellow
]


def spawn_box(pos, size: float = 0.04, mass: float = 0.05, rgba=(1, 0, 0, 1)) -> int:
    half = size / 2.0
    col  = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half, half, half])
    vis  = p.createVisualShape(p.GEOM_BOX,    halfExtents=[half, half, half], rgbaColor=rgba)
    bid  = p.createMultiBody(mass, col, vis, pos)
    p.changeDynamics(bid, -1, lateralFriction=0.9, rollingFriction=0.001, spinningFriction=0.001)
    return bid


def spawn_cubes(
    table_id: int,
    spawn_cfg: SpawnConfig,
    robot_base_xy: Tuple[float, float],
) -> List[int]:
    aabb_min, aabb_max = p.getAABB(table_id)
    z_top = float(aabb_max[2])

    x_min = aabb_min[0] + spawn_cfg.margin_xy
    x_max = aabb_max[0] - spawn_cfg.margin_xy
    y_min = aabb_min[1] + spawn_cfg.margin_xy
    y_max = aabb_max[1] - spawn_cfg.margin_xy

    if x_min >= x_max or y_min >= y_max:
        raise RuntimeError("Spawn region invalid — reduce margin_xy.")

    z       = z_top + spawn_cfg.cube_size / 2.0 + spawn_cfg.spawn_clearance
    rx, ry  = robot_base_xy
    placed_xy: List[Tuple[float, float]] = []
    ids: List[int] = []

    for i in range(spawn_cfg.n_cubes):
        color = _PALETTE[i % len(_PALETTE)]
        ok    = False
        for _ in range(spawn_cfg.max_tries_per_cube):
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            if not (spawn_cfg.reach_min <= math.hypot(x - rx, y - ry) <= spawn_cfg.reach_max):
                continue
            if any((x - px) ** 2 + (y - py) ** 2 < spawn_cfg.min_dist_xy ** 2
                   for px, py in placed_xy):
                continue
            placed_xy.append((x, y))
            ids.append(spawn_box([x, y, z], size=spawn_cfg.cube_size,
                                 mass=spawn_cfg.cube_mass, rgba=color))
            log.debug(f"  cube[{i}] body_id={ids[-1]}  pos=({x:.3f},{y:.3f},{z:.3f})")
            ok = True
            break
        if not ok:
            raise RuntimeError(f"Could not place cube {i} in reachable region. "
                               "Relax reach / min_dist / margins.")

    log.info(f"Spawned {len(ids)} cubes: {ids}")
    return ids


# ── Place slots ───────────────────────────────────────────────────────────────
def compute_place_slots(
    table_id: int,
    spawn_cfg: SpawnConfig,
    robot_base_xy: Tuple[float, float],
    base_yaw: float,
    n: int,
) -> List[Tuple[float, float]]:
    aabb_min, aabb_max = p.getAABB(table_id)
    x_min = aabb_min[0] + spawn_cfg.margin_xy
    x_max = aabb_max[0] - spawn_cfg.margin_xy
    y_min = aabb_min[1] + spawn_cfg.margin_xy
    y_max = aabb_max[1] - spawn_cfg.margin_xy

    rx, ry   = robot_base_xy
    dir_xy   = np.array([math.cos(base_yaw), math.sin(base_yaw)], dtype=np.float64)
    dir_xy  /= max(1e-9, float(np.linalg.norm(dir_xy)))
    perp_xy  = np.array([-dir_xy[1], dir_xy[0]], dtype=np.float64)

    anchor  = np.array([rx, ry]) + dir_xy * 0.50 + perp_xy * 0.18
    spacing = max(0.10, spawn_cfg.min_dist_xy)

    slots: List[Tuple[float, float]] = []
    for i in range(n):
        off = (i - (n - 1) / 2.0) * spacing
        pt  = anchor + perp_xy * off
        pt[0] = float(np.clip(pt[0], x_min, x_max))
        pt[1] = float(np.clip(pt[1], y_min, y_max))
        d = float(np.linalg.norm(pt - np.array([rx, ry])))
        if d > spawn_cfg.reach_max:
            pt = np.array([rx, ry]) + (pt - np.array([rx, ry])) * (spawn_cfg.reach_max / d)
        slots.append((float(pt[0]), float(pt[1])))

    log.debug(f"Place slots: {[tuple(round(v,3) for v in s) for s in slots]}")
    return slots
