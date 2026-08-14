"""
pick_place.py  —  entry point
Usage
-----
python pick_place.py                          # autonomous, GUI, config.yaml
python pick_place.py --config config.yaml     # explicit config path
python pick_place.py --direct                 # headless (no GUI window)
python pick_place.py --n_cubes 6 --seed 42    # CLI overrides
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time

import pybullet as p

from sim.config import load_config
from sim.logger import MetricsLogger, setup_logger
from sim.motion import step_sim
from sim.pipeline import Pipeline
from sim.scene import (
    build_scene,
    compute_place_slots,
    find_end_effector_link,
    get_arm_ik_params,
    spawn_cubes,
    connect,
    choose_renderer,
)


def parse_args():
    ap = argparse.ArgumentParser(
        description="PyBullet Franka Panda — autonomous pick-and-place"
    )
    ap.add_argument("--config",   default="config.yaml", help="Path to config YAML")
    ap.add_argument("--direct",   action="store_true",   help="Headless mode (no GUI)")
    ap.add_argument("--seed",     type=int,              help="Random seed override")
    ap.add_argument("--n_cubes",  type=int,              help="Number of cubes override")
    ap.add_argument("--renderer", choices=["tiny","opengl","auto"],
                                                          help="Renderer override")
    return ap.parse_args()


def main():
    args = parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    if args.direct:
        cfg.sim.gui = False
    if args.seed is not None:
        cfg.sim.seed = args.seed
    if args.n_cubes is not None:
        cfg.spawn.n_cubes = args.n_cubes
    if args.renderer is not None:
        cfg.sim.renderer = args.renderer

    # ── Logging ───────────────────────────────────────────────────────────────
    log = setup_logger(cfg.log_dir)
    log.info(f"Config: gui={cfg.sim.gui}  n_cubes={cfg.spawn.n_cubes}  "
             f"seed={cfg.sim.seed}  autonomous={cfg.pipeline.autonomous}")

    metrics = MetricsLogger(cfg.metrics_csv)

    # ── Seed ──────────────────────────────────────────────────────────────────
    random.seed(cfg.sim.seed)

    # ── Build scene ───────────────────────────────────────────────────────────
    connect(cfg)
    table_id, robot_id, table_center, z_top, robot_base_xy, base_yaw = build_scene(cfg)

    if cfg.sim.gui:
        cx = (robot_base_xy[0] + table_center[0]) / 2.0
        p.resetDebugVisualizerCamera(
            cameraDistance        = 1.20,
            cameraYaw             = 45,
            cameraPitch           = -35,
            cameraTargetPosition  = [cx, robot_base_xy[1], z_top + 0.1],
        )

    # ── Spawn objects ─────────────────────────────────────────────────────────
    obj_ids     = spawn_cubes(table_id, cfg.spawn, robot_base_xy)
    place_slots = compute_place_slots(table_id, cfg.spawn, robot_base_xy,
                                      base_yaw, len(obj_ids))
    cube_to_slot = {bid: place_slots[i] for i, bid in enumerate(obj_ids)}

    # ── Robot setup ───────────────────────────────────────────────────────────
    ee_link    = find_end_effector_link(robot_id)
    arm_joints = list(range(7))
    rest_poses = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
    ik_params  = get_arm_ik_params(robot_id, arm_joints, rest_poses)
    grasp_quat = p.getQuaternionFromEuler([math.pi, 0.0, 0.0])
    renderer   = choose_renderer(cfg)

    log.info(f"Robot: id={robot_id}  ee_link={ee_link}")
    log.info(f"Cubes: {cube_to_slot}")

    # ── Run ───────────────────────────────────────────────────────────────────
    pipeline = Pipeline(
        cfg             = cfg,
        robot_id        = robot_id,
        table_id        = table_id,
        ee_link         = ee_link,
        arm_joints      = arm_joints,
        ik_params       = ik_params,
        obj_ids         = obj_ids,
        cube_to_slot    = cube_to_slot,
        z_top           = z_top,
        robot_base_xy   = robot_base_xy,
        base_yaw        = base_yaw,
        grasp_quat_down = grasp_quat,
        renderer_id     = renderer,
        metrics         = metrics,
    )
    pipeline.run()

    # ── Hold GUI open after completion ────────────────────────────────────────
    if cfg.sim.gui and p.isConnected():
        log.info("Task complete. Close the PyBullet window or press Ctrl-C to exit.")
        try:
            while p.isConnected():
                p.stepSimulation()
                time.sleep(cfg.sim.timestep)
        except (KeyboardInterrupt, p.error):
            pass

    if p.isConnected():
        p.disconnect()

    log.info("Session ended.  " + metrics.summary())
    sys.exit(0)


if __name__ == "__main__":
    main()
