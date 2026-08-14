"""
sim/motion.py
Low-level motion control: IK solving, joint trajectory interpolation,
gripper open/close, and collision-checked EE pose moves.
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple

import numpy as np
import pybullet as p

from .config import AppConfig, GraspConfig, SimConfig

log = logging.getLogger("pick_place.motion")


# ── Simulation stepping ───────────────────────────────────────────────────────
def step_sim(cfg: AppConfig, n: int) -> None:
    for _ in range(n):
        if not p.isConnected():
            return
        try:
            p.stepSimulation()
        except p.error:
            return
        if cfg.sim.gui and cfg.sim.sleep_if_gui:
            time.sleep(cfg.sim.timestep)


# ── Joint queries / control ───────────────────────────────────────────────────
def get_joint_positions(robot_id: int, joints: List[int]) -> List[float]:
    return [float(p.getJointState(robot_id, j)[0]) for j in joints]


def set_arm_position_control(robot_id: int, joints: List[int],
                              targets: List[float]) -> None:
    p.setJointMotorControlArray(
        robot_id, joints,
        p.POSITION_CONTROL,
        targetPositions=targets,
        forces=[87.0] * len(joints),
    )


# ── Gripper ───────────────────────────────────────────────────────────────────
def open_gripper(robot_id: int, grasp_cfg: GraspConfig) -> None:
    for j in [9, 10]:
        if j < p.getNumJoints(robot_id):
            p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL,
                                    targetPosition=grasp_cfg.gripper_open,
                                    force=20.0)


def close_gripper(robot_id: int, grasp_cfg: GraspConfig) -> None:
    for j in [9, 10]:
        if j < p.getNumJoints(robot_id):
            p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL,
                                    targetPosition=grasp_cfg.gripper_closed,
                                    force=60.0)


# ── IK ────────────────────────────────────────────────────────────────────────
def compute_ik(
    robot_id: int,
    ee_link: int,
    arm_joints: List[int],
    target_pos: Tuple[float, float, float],
    target_orn: Tuple[float, float, float, float],
    ik_lowers: List[float],
    ik_uppers: List[float],
    ik_ranges: List[float],
    rest_poses: List[float],
    grasp_cfg: GraspConfig,
) -> List[float]:
    sol = p.calculateInverseKinematics(
        robot_id, ee_link,
        targetPosition    = target_pos,
        targetOrientation = target_orn,
        lowerLimits       = ik_lowers,
        upperLimits       = ik_uppers,
        jointRanges       = ik_ranges,
        restPoses         = rest_poses,
        maxNumIterations  = grasp_cfg.ik_max_iters,
        residualThreshold = grasp_cfg.ik_residual,
    )
    return [sol[j] for j in arm_joints]


# ── Trajectory ────────────────────────────────────────────────────────────────
def interpolate(q0: List[float], q1: List[float], steps: int) -> List[List[float]]:
    return [
        [(1 - t) * a + t * b for a, b in zip(q0, q1)]
        for t in [(i + 1) / steps for i in range(steps)]
    ]


# ── Cartesian EE move ─────────────────────────────────────────────────────────
def move_ee(
    cfg: AppConfig,
    robot_id: int,
    ee_link: int,
    arm_joints: List[int],
    ik_params,
    target_pos: Tuple[float, float, float],
    target_orn: Tuple[float, float, float, float],
    table_id: Optional[int] = None,
) -> bool:
    """
    Move end-effector to (target_pos, target_orn) via joint-space interpolation.
    Returns True on success, False if disconnected or IK fails.
    """
    if not p.isConnected():
        return False

    ik_lowers, ik_uppers, ik_ranges, rest_poses = ik_params
    q_tgt = compute_ik(robot_id, ee_link, arm_joints,
                       target_pos, target_orn,
                       ik_lowers, ik_uppers, ik_ranges, rest_poses,
                       cfg.grasp)
    q0    = get_joint_positions(robot_id, arm_joints)
    traj  = interpolate(q0, q_tgt, cfg.grasp.traj_steps)

    for q in traj:
        if not p.isConnected():
            return False
        set_arm_position_control(robot_id, arm_joints, q)
        try:
            p.stepSimulation()
        except p.error:
            return False

        if table_id is not None:
            try:
                p.getClosestPoints(bodyA=robot_id, bodyB=table_id,
                                   distance=cfg.grasp.collision_check_dist)
            except p.error:
                return False

        if cfg.sim.gui and cfg.sim.sleep_if_gui:
            time.sleep(cfg.sim.timestep)

    return True


# ── Survey (observation) pose ─────────────────────────────────────────────────
def move_to_survey(
    cfg: AppConfig,
    robot_id: int,
    ee_link: int,
    arm_joints: List[int],
    ik_params,
    table_id: int,
    z_top: float,
    robot_base_xy: Tuple[float, float],
    base_yaw: float,
    grasp_quat_down: Tuple,
) -> bool:
    import math
    rx, ry = robot_base_xy
    sx = rx + math.cos(base_yaw) * cfg.grasp.survey_forward_from_base
    sy = ry + math.sin(base_yaw) * cfg.grasp.survey_forward_from_base
    sz = z_top + cfg.grasp.survey_height_above_table

    open_gripper(robot_id, cfg.grasp)
    step_sim(cfg, int(0.10 / cfg.sim.timestep))
    ok = move_ee(cfg, robot_id, ee_link, arm_joints, ik_params,
                 (sx, sy, sz), grasp_quat_down, table_id=table_id)
    if not ok:
        log.warning("Survey pose IK failed.")
    return ok
