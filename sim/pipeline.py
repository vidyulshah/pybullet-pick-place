"""
sim/pipeline.py
Finite-state machine that orchestrates the full pick-and-place pipeline.

States
------
INIT      → SURVEY
SURVEY    → DETECT  (arm at observation pose, camera fired)
DETECT    → APPROACH  (target selected)
           → DONE     (no more pending cubes)
           → ESTOP    (too many consecutive failures)
APPROACH  → PICK
           → SURVEY   (IK / move failed)
PICK      → PLACE     (pick succeeded)
           → SURVEY   (all retries exhausted)
PLACE     → RETREAT
RETREAT   → SURVEY
DONE      → (terminal)
ESTOP     → (terminal)
"""
from __future__ import annotations

import logging
import signal
import time
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

import math
import pybullet as p

from .config import AppConfig
from .grasp import pick_once, place
from .logger import MetricsLogger
from .motion import move_to_survey, step_sim
from .vision import detect, fmt_det, overhead_view, projection_matrix, compute_K, view_to_4x4, render, wrist_view

log = logging.getLogger("pick_place.pipeline")


class State(Enum):
    INIT     = auto()
    SURVEY   = auto()
    DETECT   = auto()
    APPROACH = auto()
    PICK     = auto()
    PLACE    = auto()
    RETREAT  = auto()
    DONE     = auto()
    ESTOP    = auto()


class Pipeline:
    def __init__(
        self,
        cfg:             AppConfig,
        robot_id:        int,
        table_id:        int,
        ee_link:         int,
        arm_joints:      List[int],
        ik_params,
        obj_ids:         List[int],
        cube_to_slot:    Dict[int, Tuple[float, float]],
        z_top:           float,
        robot_base_xy:   Tuple[float, float],
        base_yaw:        float,
        grasp_quat_down,
        renderer_id:     int,
        metrics:         MetricsLogger,
    ) -> None:
        self.cfg             = cfg
        self.robot_id        = robot_id
        self.table_id        = table_id
        self.ee_link         = ee_link
        self.arm_joints      = arm_joints
        self.ik_params       = ik_params
        self.obj_ids         = obj_ids
        self.cube_to_slot    = cube_to_slot
        self.z_top           = z_top
        self.robot_base_xy   = robot_base_xy
        self.base_yaw        = base_yaw
        self.grasp_quat_down = grasp_quat_down
        self.renderer_id     = renderer_id
        self.metrics         = metrics

        # Vision constants
        self.proj = projection_matrix(cfg.camera)
        self.K    = compute_K(cfg.camera)
        rx, ry    = robot_base_xy
        rc_xy     = ((rx + 0.75) / 2.0, ry)     # rough region centre
        self._oh_view = overhead_view(rc_xy, cfg.spawn.reach_max + 0.12,
                                      cfg.camera, z_top)

        # State
        self.state              = State.INIT
        self.pending:    Set[int] = set(obj_ids)
        self.target_id:  Optional[int] = None
        self.target_xy:  Optional[Tuple[float, float]] = None
        self.cycle              = 0
        self.consecutive_fails  = 0
        self._pick_start:  float = 0.0
        self._estop        = False

        # Graceful shutdown on Ctrl-C / SIGTERM
        signal.signal(signal.SIGINT,  self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    # ── Signal handler ────────────────────────────────────────────────────────
    def _handle_signal(self, signum, frame):
        log.warning(f"Signal {signum} received — requesting ESTOP.")
        self._estop = True

    # ── Vision helpers ────────────────────────────────────────────────────────
    def _overhead_det(self) -> Dict[int, Dict]:
        V = view_to_4x4(self._oh_view)
        rgb, dep, seg = render(self._oh_view, self.proj, self.cfg.camera, self.renderer_id)
        return detect(rgb, dep, seg, list(self.pending), self.cfg.camera, self.K, V)

    def _wrist_det(self, look_at: Tuple[float, float, float]) -> Dict[int, Dict]:
        view = wrist_view(self.robot_id, self.ee_link, self.cfg.camera, look_at)
        V    = view_to_4x4(view)
        rgb, dep, seg = render(view, self.proj, self.cfg.camera, self.renderer_id)
        return detect(rgb, dep, seg, list(self.pending), self.cfg.camera, self.K, V)

    def _refine_xy(self, tid: int, xy: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        try:
            det = self._wrist_det((xy[0], xy[1], self.z_top + 0.02))
        except p.error:
            return None
        if tid in det:
            wx, wy, _ = det[tid]["world_xyz_surface"]
            return float(wx), float(wy)
        return None

    # ── State transitions ─────────────────────────────────────────────────────
    def _to(self, s: State, msg: str = "") -> None:
        log.info(f"  [{self.state.name}] → [{s.name}]  {msg}")
        self.state = s

    # ── One FSM tick ──────────────────────────────────────────────────────────
    def tick(self) -> bool:
        """
        Advance the FSM by one state.
        Returns False when the machine has reached a terminal state.
        """
        if self._estop:
            self._to(State.ESTOP, "external signal")

        s = self.state

        # ── INIT ──────────────────────────────────────────────────────────────
        if s == State.INIT:
            log.info(f"Pipeline starting — {len(self.pending)} cubes to place.")
            step_sim(self.cfg, int(0.4 / self.cfg.sim.timestep))
            self._to(State.SURVEY)

        # ── SURVEY ────────────────────────────────────────────────────────────
        elif s == State.SURVEY:
            move_to_survey(
                self.cfg, self.robot_id, self.ee_link, self.arm_joints,
                self.ik_params, self.table_id, self.z_top,
                self.robot_base_xy, self.base_yaw, self.grasp_quat_down,
            )
            self._to(State.DETECT)

        # ── DETECT ────────────────────────────────────────────────────────────
        elif s == State.DETECT:
            if not self.pending:
                self._to(State.DONE, "all cubes placed")
                return True

            if self.consecutive_fails >= self.cfg.pipeline.max_consecutive_failures:
                self._to(State.ESTOP,
                         f"consecutive failures={self.consecutive_fails}")
                return True

            det = self._overhead_det()
            log.info(f"Overhead detection: {fmt_det(det)}")

            # Pick cube with most pixels visible (most reliably seen)
            visible = {bid: det[bid] for bid in self.pending if bid in det}
            if not visible:
                log.warning("No pending cubes visible — re-surveying.")
                self._to(State.SURVEY)
                return True

            self.target_id  = max(visible, key=lambda b: visible[b]["pixel_count"])
            wx, wy, _       = visible[self.target_id]["world_xyz_surface"]
            self.target_xy  = (float(wx), float(wy))
            self.cycle      += 1
            self._pick_start = time.perf_counter()
            log.info(f"Target: cube_id={self.target_id}  xy={self.target_xy}  "
                     f"class={visible[self.target_id]['class']}")
            self._target_class = visible[self.target_id]["class"]
            self._to(State.APPROACH)

        # ── APPROACH (= pick loop with retries) ───────────────────────────────
        elif s == State.APPROACH:
            tid   = self.target_id
            xy    = self.target_xy
            total = 1 + self.cfg.grasp.max_pick_retries
            picked = False

            for attempt in range(total):
                log.info(f"  Pick attempt {attempt+1}/{total} for cube {tid}")
                picked = pick_once(
                    cfg             = self.cfg,
                    robot_id        = self.robot_id,
                    ee_link         = self.ee_link,
                    arm_joints      = self.arm_joints,
                    ik_params       = self.ik_params,
                    cube_id         = tid,
                    cube_size       = self.cfg.spawn.cube_size,
                    z_top           = self.z_top,
                    target_xy       = xy,
                    refine_xy_fn    = lambda: self._refine_xy(tid, xy),
                    grasp_quat_down = self.grasp_quat_down,
                    table_id        = self.table_id,
                )
                if picked:
                    break
                # Re-acquire between retries
                move_to_survey(
                    self.cfg, self.robot_id, self.ee_link, self.arm_joints,
                    self.ik_params, self.table_id, self.z_top,
                    self.robot_base_xy, self.base_yaw, self.grasp_quat_down,
                )
                det2 = self._overhead_det()
                if tid in det2:
                    wx2, wy2, _ = det2[tid]["world_xyz_surface"]
                    xy = (float(wx2), float(wy2))

            duration = time.perf_counter() - self._pick_start
            place_xy = self.cube_to_slot.get(tid, (0.0, 0.0))

            if picked:
                self.consecutive_fails = 0
                self.metrics.record(
                    cycle       = self.cycle,
                    cube_id     = tid,
                    color_class = self._target_class,
                    pick_xy_x   = round(xy[0], 4),
                    pick_xy_y   = round(xy[1], 4),
                    place_xy_x  = round(place_xy[0], 4),
                    place_xy_y  = round(place_xy[1], 4),
                    attempts    = total,
                    result      = "success",
                    duration_s  = round(duration, 2),
                )
                self._to(State.PLACE)
            else:
                self.consecutive_fails += 1
                self.metrics.record(
                    cycle       = self.cycle,
                    cube_id     = tid,
                    color_class = self._target_class,
                    pick_xy_x   = round(xy[0], 4),
                    pick_xy_y   = round(xy[1], 4),
                    attempts    = total,
                    result      = "failure",
                    duration_s  = round(duration, 2),
                    note        = "all retries exhausted",
                )
                log.warning(f"  Cube {tid} pick FAILED after {total} attempts.")
                # Remove from pending so we don't loop forever on an unreachable cube
                self.pending.discard(tid)
                self._to(State.SURVEY)

        # ── PLACE ─────────────────────────────────────────────────────────────
        elif s == State.PLACE:
            tid      = self.target_id
            place_xy = self.cube_to_slot.get(tid, (0.0, 0.0))
            ok       = place(
                cfg             = self.cfg,
                robot_id        = self.robot_id,
                ee_link         = self.ee_link,
                arm_joints      = self.arm_joints,
                ik_params       = self.ik_params,
                cube_size       = self.cfg.spawn.cube_size,
                z_top           = self.z_top,
                place_xy        = place_xy,
                grasp_quat_down = self.grasp_quat_down,
                table_id        = self.table_id,
            )
            if ok:
                log.info(f"  Cube {tid} placed at {place_xy}.")
                self.pending.discard(tid)
            else:
                log.warning(f"  Place FAILED for cube {tid}.")
            self._to(State.RETREAT)

        # ── RETREAT ───────────────────────────────────────────────────────────
        elif s == State.RETREAT:
            step_sim(self.cfg, int(self.cfg.pipeline.inter_pick_pause_s / self.cfg.sim.timestep))
            self._to(State.SURVEY)

        # ── Terminal states ───────────────────────────────────────────────────
        elif s == State.DONE:
            log.info("=" * 50)
            log.info("ALL CUBES PLACED — pipeline complete.")
            log.info(self.metrics.summary())
            log.info("=" * 50)
            return False

        elif s == State.ESTOP:
            log.error("EMERGENCY STOP — pipeline halted.")
            log.info(self.metrics.summary())
            return False

        return True

    # ── Run loop ──────────────────────────────────────────────────────────────
    def run(self) -> None:
        """Drive the FSM until a terminal state or PyBullet disconnects."""
        while True:
            if not p.isConnected():
                log.error("PyBullet disconnected unexpectedly.")
                break
            try:
                keep_going = self.tick()
            except p.error as exc:
                log.error(f"PyBullet error during tick: {exc}")
                break
            except Exception as exc:
                log.exception(f"Unexpected error during tick: {exc}")
                break
            if not keep_going:
                break
