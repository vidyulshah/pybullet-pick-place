"""
sim/grasp.py
Pick and place sequences built on top of sim.motion primitives.
"""
from __future__ import annotations

import logging
from typing import Callable, List, Optional, Tuple

import pybullet as p

from .config import AppConfig
from .motion import close_gripper, move_ee, open_gripper, step_sim

log = logging.getLogger("pick_place.grasp")


def pick_once(
    cfg:             AppConfig,
    robot_id:        int,
    ee_link:         int,
    arm_joints:      List[int],
    ik_params,
    cube_id:         int,
    cube_size:       float,
    z_top:           float,
    target_xy:       Tuple[float, float],
    refine_xy_fn:    Callable[[], Optional[Tuple[float, float]]],
    grasp_quat_down: Tuple,
    table_id:        int,
) -> bool:
    """
    One pick attempt.  Sweeps through descend_top_offsets until grasp succeeds.
    Returns True if the cube is lifted above the success threshold.
    """
    if not p.isConnected():
        return False

    open_gripper(robot_id, cfg.grasp)
    step_sim(cfg, int(0.10 / cfg.sim.timestep))

    x, y     = target_xy
    refined  = refine_xy_fn()
    if refined is not None:
        x, y = refined
        log.debug(f"  Wrist refinement → ({x:.3f}, {y:.3f})")

    z_cube_top = z_top + cube_size
    pre        = (x, y, z_cube_top + cfg.grasp.approach_height)

    if not move_ee(cfg, robot_id, ee_link, arm_joints, ik_params,
                   pre, grasp_quat_down, table_id=table_id):
        log.warning("  Approach move failed (IK).")
        return False

    for dz in cfg.grasp.descend_top_offsets:
        grasp_pos = (x, y, z_cube_top + dz)
        if not move_ee(cfg, robot_id, ee_link, arm_joints, ik_params,
                       grasp_pos, grasp_quat_down, table_id=table_id):
            continue

        close_gripper(robot_id, cfg.grasp)
        step_sim(cfg, int(cfg.grasp.gripper_settle_seconds / cfg.sim.timestep))

        lift_pos = (x, y, z_cube_top + cfg.grasp.lift_height)
        if not move_ee(cfg, robot_id, ee_link, arm_joints, ik_params,
                       lift_pos, grasp_quat_down, table_id=table_id):
            open_gripper(robot_id, cfg.grasp)
            continue

        # ── Success check ──────────────────────────────────────────────────
        try:
            pos_after, _ = p.getBasePositionAndOrientation(cube_id)
        except p.error:
            return False

        if pos_after[2] >= (z_cube_top + cfg.grasp.success_lift_threshold):
            log.info(f"  Grasp SUCCESS at dz={dz}  cube_z={pos_after[2]:.4f}")
            return True

        try:
            cps = p.getContactPoints(bodyA=robot_id, bodyB=cube_id)
            if cps and len(cps) > 0:
                log.info(f"  Grasp SUCCESS via contact check at dz={dz}")
                return True
        except p.error:
            pass

        # Failed at this dz — reopen and return to pre-grasp
        open_gripper(robot_id, cfg.grasp)
        step_sim(cfg, int(0.10 / cfg.sim.timestep))
        move_ee(cfg, robot_id, ee_link, arm_joints, ik_params,
                pre, grasp_quat_down, table_id=table_id)

    log.warning("  All descend offsets exhausted — pick failed.")
    return False


def place(
    cfg:             AppConfig,
    robot_id:        int,
    ee_link:         int,
    arm_joints:      List[int],
    ik_params,
    cube_size:       float,
    z_top:           float,
    place_xy:        Tuple[float, float],
    grasp_quat_down: Tuple,
    table_id:        int,
) -> bool:
    """Move to place slot and release object."""
    if not p.isConnected():
        return False

    px, py     = place_xy
    z_cube_top = z_top + cube_size
    pre        = (px, py, z_cube_top + cfg.grasp.approach_height)
    down       = (px, py, z_cube_top + cfg.grasp.place_top_offset)

    if not move_ee(cfg, robot_id, ee_link, arm_joints, ik_params,
                   pre, grasp_quat_down, table_id=table_id):
        return False
    if not move_ee(cfg, robot_id, ee_link, arm_joints, ik_params,
                   down, grasp_quat_down, table_id=table_id):
        return False

    open_gripper(robot_id, cfg.grasp)
    step_sim(cfg, int(cfg.grasp.place_settle_seconds / cfg.sim.timestep))

    move_ee(cfg, robot_id, ee_link, arm_joints, ik_params,
            pre, grasp_quat_down, table_id=table_id)
    log.info(f"  Place at ({px:.3f}, {py:.3f}) complete.")
    return True
