import time
import math
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable

import pybullet as p
import pybullet_data
import numpy as np


# -----------------------------
# Config
# -----------------------------
@dataclass
class SimConfig:
    gui: bool = True
    render: bool = True
    timestep: float = 1.0 / 240.0
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)
    seed: int = 0

    run_until_window_closed: bool = True
    steps: int = 20000
    sleep_if_gui: bool = True

    renderer: str = "tiny"  # tiny | opengl | auto


@dataclass
class SpawnConfig:
    n_cubes: int = 4
    cube_size: float = 0.04
    cube_mass: float = 0.05

    margin_xy: float = 0.16

    # Reach constraints from robot base (meters)
    reach_min: float = 0.20
    reach_max: float = 0.60

    min_dist_xy: float = 0.10
    max_tries_per_cube: int = 900

    spawn_clearance: float = 0.003


@dataclass
class CameraConfig:
    width: int = 320
    height: int = 220
    fov_y_deg: float = 75.0
    near: float = 0.02
    far: float = 5.0

    # Print + prompt every N sim steps
    print_every_n_steps: int = 240

    # Wrist camera: eye offset in EE frame
    ee_cam_offset: Tuple[float, float, float] = (0.0, 0.0, 0.10)


@dataclass
class GraspConfig:
    approach_height: float = 0.14
    lift_height: float = 0.18

    ik_max_iters: int = 180
    ik_residual: float = 1e-4
    traj_steps: int = 180

    gripper_open: float = 0.04
    gripper_closed: float = 0.0
    gripper_settle_seconds: float = 0.30

    success_lift_threshold: float = 0.05
    collision_check_dist: float = 0.010

    # Robustness: sweep descend height near cube top
    descend_top_offsets: Tuple[float, ...] = (0.035, 0.025, 0.018, 0.012, 0.008, 0.006)

    # Place
    place_top_offset: float = 0.030
    place_settle_seconds: float = 0.20

    # NEW: retry behavior
    max_pick_retries: int = 2  # additional attempts after the first
    # NEW: survey pose height above tabletop for better view / less occlusion
    survey_height_above_table: float = 0.65
    # NEW: park distance forward from base (toward table) for survey pose
    survey_forward_from_base: float = 0.20


# -----------------------------
# PyBullet connect/setup
# -----------------------------
def connect(gui: bool) -> int:
    if gui:
        cid = p.connect(p.GUI, options="--renderer=TinyRenderer")
    else:
        cid = p.connect(p.DIRECT)
    if cid < 0:
        raise RuntimeError("Failed to connect to PyBullet.")
    return cid


def choose_renderer(sim_cfg: SimConfig) -> int:
    # TinyRenderer is most stable on Windows.
    if sim_cfg.renderer == "tiny":
        return p.ER_TINY_RENDERER
    if sim_cfg.renderer == "opengl":
        return p.ER_BULLET_HARDWARE_OPENGL
    return p.ER_BULLET_HARDWARE_OPENGL if sim_cfg.gui else p.ER_TINY_RENDERER


def set_neutral_panda(robot_id: int) -> None:
    neutral = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
    for j in range(7):
        p.resetJointState(robot_id, j, neutral[j])
    for j in [9, 10]:
        if j < p.getNumJoints(robot_id):
            p.resetJointState(robot_id, j, 0.04)


def get_table_aabb(table_id: int):
    return p.getAABB(table_id)


# -----------------------------
# Scene
# -----------------------------
def build_scene(sim_cfg: SimConfig):
    connect(sim_cfg.gui)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setRealTimeSimulation(0)
    p.setGravity(*sim_cfg.gravity)
    p.setTimeStep(sim_cfg.timestep)

    p.loadURDF("plane.urdf")

    table_pos = [0.75, 0.0, 0.0]
    table_id = p.loadURDF("table/table.urdf", table_pos, useFixedBase=True)

    aabb_min, aabb_max = get_table_aabb(table_id)
    z_top = aabb_max[2]
    table_center = [(aabb_min[0] + aabb_max[0]) / 2.0, (aabb_min[1] + aabb_max[1]) / 2.0, z_top]

    # Robot base near near-side table edge, centered in Y
    x_base = aabb_min[0] + 0.22
    y_base = (aabb_min[1] + aabb_max[1]) / 2.0
    base_clearance = 0.002

    dx = table_center[0] - x_base
    dy = table_center[1] - y_base
    yaw = float(math.atan2(dy, dx))
    robot_orn = p.getQuaternionFromEuler([0.0, 0.0, yaw])

    robot_id = p.loadURDF(
        "franka_panda/panda.urdf",
        [x_base, y_base, z_top + base_clearance],
        robot_orn,
        useFixedBase=True,
    )
    set_neutral_panda(robot_id)

    robot_base_xy = (x_base, y_base)
    return table_id, robot_id, table_center, z_top, robot_base_xy, yaw


# -----------------------------
# Panda indices + IK parameters
# -----------------------------
def find_end_effector_link(robot_id: int) -> int:
    preferred = ["panda_hand", "panda_grasptarget", "panda_link8"]
    name_to_index = {}
    n = p.getNumJoints(robot_id)
    for j in range(n):
        info = p.getJointInfo(robot_id, j)
        link_name = info[12].decode("utf-8")
        name_to_index[link_name] = j
    for nm in preferred:
        if nm in name_to_index:
            return name_to_index[nm]
    return min(11, n - 1)


def get_arm_ik_params(robot_id: int, arm_joints: List[int], rest_poses: List[float]):
    lowers, uppers, ranges = [], [], []
    for j in arm_joints:
        info = p.getJointInfo(robot_id, j)
        lo = float(info[8])
        hi = float(info[9])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = -2.967, 2.967
        lowers.append(lo)
        uppers.append(hi)
        ranges.append(hi - lo)
    return lowers, uppers, ranges, rest_poses


# -----------------------------
# Spawning
# -----------------------------
def spawn_box(pos, size=0.04, mass=0.05, rgba=(1, 0, 0, 1)) -> int:
    half = size / 2.0
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half, half, half])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[half, half, half], rgbaColor=rgba)
    bid = p.createMultiBody(mass, col, vis, pos)
    p.changeDynamics(bid, -1, lateralFriction=0.9, rollingFriction=0.001, spinningFriction=0.001)
    return bid


def random_palette_rgba() -> Tuple[float, float, float, float]:
    palette = [
        (1.0, 0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0, 1.0),
        (0.0, 0.0, 1.0, 1.0),
        (1.0, 1.0, 0.0, 1.0),  # "other"
    ]
    return random.choice(palette)


def spawn_objects_reachable_on_table(
    table_id: int,
    spawn_cfg: SpawnConfig,
    robot_base_xy: Tuple[float, float],
) -> List[int]:
    aabb_min, aabb_max = get_table_aabb(table_id)
    z_top = aabb_max[2]

    x_min = aabb_min[0] + spawn_cfg.margin_xy
    x_max = aabb_max[0] - spawn_cfg.margin_xy
    y_min = aabb_min[1] + spawn_cfg.margin_xy
    y_max = aabb_max[1] - spawn_cfg.margin_xy

    if x_min >= x_max or y_min >= y_max:
        raise RuntimeError("Spawn region invalid. Reduce margin_xy.")

    z = z_top + spawn_cfg.cube_size / 2.0 + spawn_cfg.spawn_clearance
    rx, ry = robot_base_xy

    placed_xy: List[Tuple[float, float]] = []
    ids: List[int] = []

    for _ in range(spawn_cfg.n_cubes):
        ok = False
        for _try in range(spawn_cfg.max_tries_per_cube):
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)

            d = math.hypot(x - rx, y - ry)
            if not (spawn_cfg.reach_min <= d <= spawn_cfg.reach_max):
                continue

            if any((x - px) ** 2 + (y - py) ** 2 < spawn_cfg.min_dist_xy ** 2 for px, py in placed_xy):
                continue

            placed_xy.append((x, y))
            ids.append(
                spawn_box(
                    [x, y, z],
                    size=spawn_cfg.cube_size,
                    mass=spawn_cfg.cube_mass,
                    rgba=random_palette_rgba(),
                )
            )
            ok = True
            break
        if not ok:
            raise RuntimeError("Failed to place all cubes in reachable region. Relax reach/min_dist/margins.")
    return ids


# -----------------------------
# Vision math
# -----------------------------
def compute_K(cam_cfg: CameraConfig) -> np.ndarray:
    W, H = cam_cfg.width, cam_cfg.height
    fov_y = math.radians(cam_cfg.fov_y_deg)
    fy = (H / 2.0) / math.tan(fov_y / 2.0)
    fx = fy * (W / H)
    cx = W / 2.0
    cy = H / 2.0
    return np.array([[fx, 0.0, cx],
                     [0.0, fy, cy],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def projection_matrix(cam_cfg: CameraConfig) -> List[float]:
    return p.computeProjectionMatrixFOV(
        fov=cam_cfg.fov_y_deg,
        aspect=float(cam_cfg.width) / float(cam_cfg.height),
        nearVal=cam_cfg.near,
        farVal=cam_cfg.far,
    )


def view_to_4x4_colmajor(view_list: List[float]) -> np.ndarray:
    return np.array(view_list, dtype=np.float64).reshape((4, 4), order="F")


def depth_buffer_to_meters(depth_buf: np.ndarray, near: float, far: float) -> np.ndarray:
    z = depth_buf.astype(np.float64)
    return (far * near) / (far - (far - near) * z)


def backproject_pixel_to_world(
    u: float,
    v: float,
    depth_m: float,
    K: np.ndarray,
    V_world_to_cam: np.ndarray,
) -> Tuple[float, float, float]:
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # OpenGL camera: +X right, +Y up, looks down -Z
    X = (u - cx) * depth_m / fx
    Y = -(v - cy) * depth_m / fy
    Z = -depth_m

    Pc = np.array([X, Y, Z, 1.0], dtype=np.float64)
    T_cam_to_world = np.linalg.inv(V_world_to_cam)
    Pw = T_cam_to_world @ Pc
    return float(Pw[0]), float(Pw[1]), float(Pw[2])


# -----------------------------
# Cameras
# -----------------------------
def compute_overhead_view_for_region(
    region_center_xy: Tuple[float, float],
    region_radius: float,
    cam_cfg: CameraConfig,
    z_top: float,
) -> List[float]:
    alpha = math.radians(cam_cfg.fov_y_deg) / 2.0
    h = (region_radius / max(1e-6, math.tan(alpha))) * 1.25
    h = max(h, 0.65)
    eye = [region_center_xy[0], region_center_xy[1], z_top + h]
    target = [region_center_xy[0], region_center_xy[1], z_top + 0.02]
    up = [1.0, 0.0, 0.0]
    return p.computeViewMatrix(eye, target, up)


def quat_to_R(q) -> np.ndarray:
    m = p.getMatrixFromQuaternion(q)
    return np.array(m, dtype=np.float64).reshape((3, 3))


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n < 1e-9 else (v / n)


def wrist_view_attached_look_at_point(
    robot_id: int,
    ee_link: int,
    cam_cfg: CameraConfig,
    look_at_world: Tuple[float, float, float],
) -> List[float]:
    ee_pos, ee_orn = p.getLinkState(robot_id, ee_link, computeForwardKinematics=True)[:2]
    R = quat_to_R(ee_orn)

    offset = np.array(cam_cfg.ee_cam_offset, dtype=np.float64)
    eye = np.array(ee_pos, dtype=np.float64) + R @ offset
    target = np.array(look_at_world, dtype=np.float64)

    fwd = normalize(target - eye)
    up_guess = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(fwd, up_guess))) > 0.95:
        up_guess = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    right = normalize(np.cross(fwd, up_guess))
    up = normalize(np.cross(right, fwd))
    return p.computeViewMatrix(eye.tolist(), target.tolist(), up.tolist())


# -----------------------------
# Segmentation detection + classification
# -----------------------------
def decode_object_ids(seg: np.ndarray) -> np.ndarray:
    seg = seg.astype(np.int64)
    seg[seg < 0] = 0
    return seg & ((1 << 24) - 1)


def color_bucket(rgb: Tuple[int, int, int]) -> str:
    r, g, b = rgb
    vals = np.array([r, g, b], dtype=np.float64)
    order = vals.argsort()
    best = int(order[-1])
    second = float(vals[order[-2]])
    if float(vals[best]) - second < 15.0:
        return "other"
    return ["red", "green", "blue"][best]


def robust_depth_at_pixel(depth_m: np.ndarray, u: int, v: int, patch: int = 5) -> float:
    H, W = depth_m.shape
    r = patch // 2
    x0 = max(0, u - r)
    x1 = min(W, u + r + 1)
    y0 = max(0, v - r)
    y1 = min(H, v + r + 1)
    vals = depth_m[y0:y1, x0:x1].reshape(-1)
    return float(np.median(vals)) if vals.size else float(depth_m[v, u])


def render_camera(view: List[float], proj: List[float], cam_cfg: CameraConfig, renderer_id: int):
    flags = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
    img = p.getCameraImage(
        cam_cfg.width,
        cam_cfg.height,
        viewMatrix=view,
        projectionMatrix=proj,
        renderer=renderer_id,
        flags=flags,
        shadow=1,
        lightDirection=[1, 1, 1],
    )
    w, h = img[0], img[1]
    rgb = np.array(img[2], dtype=np.uint8).reshape((h, w, 4))
    depth = np.array(img[3], dtype=np.float64).reshape((h, w))
    seg = np.array(img[4], dtype=np.int64).reshape((h, w))
    return rgb, depth, seg


def detect_from_buffers(
    rgb: np.ndarray,
    depth_buf: np.ndarray,
    seg: np.ndarray,
    obj_ids: List[int],
    cam_cfg: CameraConfig,
    K: np.ndarray,
    V_world_to_cam: np.ndarray,
) -> Dict[int, Dict]:
    obj_uid = decode_object_ids(seg)
    depth_m = depth_buffer_to_meters(depth_buf, cam_cfg.near, cam_cfg.far)

    det: Dict[int, Dict] = {}
    H, W = cam_cfg.height, cam_cfg.width

    for bid in obj_ids:
        mask = (obj_uid == bid)
        count = int(mask.sum())
        if count <= 0:
            continue

        ys, xs = np.nonzero(mask)
        cx = float(xs.mean())
        cy = float(ys.mean())
        u = int(round(cx))
        v = int(round(cy))
        u = max(0, min(W - 1, u))
        v = max(0, min(H - 1, v))

        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())

        d = robust_depth_at_pixel(depth_m, u, v, patch=5)
        wx, wy, wz = backproject_pixel_to_world(u, v, d, K, V_world_to_cam)

        px = rgb[v, u, :3].astype(np.int32)
        rgb_tuple = (int(px[0]), int(px[1]), int(px[2]))
        cls = color_bucket(rgb_tuple)

        det[bid] = {
            "class": cls,
            "pixel_count": count,
            "bbox_xyxy": (x0, y0, x1, y1),
            "centroid_uv_int": (u, v),
            "world_xyz_surface": (wx, wy, wz),  # surface point (often top face)
            "rgb": rgb_tuple,
        }
    return det


def format_det(det: Dict[int, Dict]) -> str:
    if not det:
        return "none"
    items = sorted(det.items(), key=lambda kv: kv[1]["pixel_count"], reverse=True)
    parts = []
    for bid, info in items:
        xyz = tuple(round(x, 3) for x in info["world_xyz_surface"])
        parts.append(f"{bid}(cls={info['class']},px={info['pixel_count']},xyz={xyz})")
    return "; ".join(parts)


# -----------------------------
# Motion control helpers
# -----------------------------
def get_joint_positions(robot_id: int, joint_indices: List[int]) -> List[float]:
    return [p.getJointState(robot_id, j)[0] for j in joint_indices]


def set_arm_position_control(robot_id: int, arm_joints: List[int], target_positions: List[float]):
    p.setJointMotorControlArray(
        robot_id,
        arm_joints,
        p.POSITION_CONTROL,
        targetPositions=target_positions,
        forces=[87.0] * len(arm_joints),
    )


def step_sim(sim_cfg: SimConfig, n: int):
    for _ in range(n):
        if not p.isConnected():
            return
        try:
            p.stepSimulation()
        except p.error:
            return
        if sim_cfg.gui and sim_cfg.sleep_if_gui:
            time.sleep(sim_cfg.timestep)


def open_gripper(robot_id: int, grasp_cfg: GraspConfig):
    for j in [9, 10]:
        if j < p.getNumJoints(robot_id):
            p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL,
                                    targetPosition=grasp_cfg.gripper_open, force=20.0)


def close_gripper(robot_id: int, grasp_cfg: GraspConfig):
    for j in [9, 10]:
        if j < p.getNumJoints(robot_id):
            p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL,
                                    targetPosition=grasp_cfg.gripper_closed, force=60.0)


def interpolate_joint_space(q0: List[float], q1: List[float], steps: int) -> List[List[float]]:
    traj = []
    for i in range(steps):
        t = (i + 1) / float(steps)
        traj.append([(1 - t) * a + t * b for a, b in zip(q0, q1)])
    return traj


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
        robot_id,
        ee_link,
        targetPosition=target_pos,
        targetOrientation=target_orn,
        lowerLimits=ik_lowers,
        upperLimits=ik_uppers,
        jointRanges=ik_ranges,
        restPoses=rest_poses,
        maxNumIterations=grasp_cfg.ik_max_iters,
        residualThreshold=grasp_cfg.ik_residual,
    )
    return [sol[j] for j in arm_joints]


def move_ee_pose(
    sim_cfg: SimConfig,
    robot_id: int,
    ee_link: int,
    arm_joints: List[int],
    ik_params,
    target_pos: Tuple[float, float, float],
    target_orn: Tuple[float, float, float, float],
    grasp_cfg: GraspConfig,
    table_id: Optional[int] = None,
) -> bool:
    if not p.isConnected():
        return False

    ik_lowers, ik_uppers, ik_ranges, rest_poses = ik_params
    q_target = compute_ik(
        robot_id, ee_link, arm_joints,
        target_pos, target_orn,
        ik_lowers, ik_uppers, ik_ranges, rest_poses,
        grasp_cfg
    )

    q0 = get_joint_positions(robot_id, arm_joints)
    traj = interpolate_joint_space(q0, q_target, grasp_cfg.traj_steps)

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
                _ = p.getClosestPoints(bodyA=robot_id, bodyB=table_id, distance=grasp_cfg.collision_check_dist)
            except p.error:
                return False

        if sim_cfg.gui and sim_cfg.sleep_if_gui:
            time.sleep(sim_cfg.timestep)

    return True


# -----------------------------
# Survey pose (NEW)
# -----------------------------
def move_to_survey_pose(
    sim_cfg: SimConfig,
    grasp_cfg: GraspConfig,
    table_id: int,
    robot_id: int,
    ee_link: int,
    arm_joints: List[int],
    ik_params,
    z_top: float,
    robot_base_xy: Tuple[float, float],
    base_yaw: float,
    grasp_quat_down: Tuple[float, float, float, float],
) -> bool:
    """
    Move the arm high and near the base to reduce occlusion and increase wrist-camera FOV.
    """
    rx, ry = robot_base_xy
    dir_xy = (math.cos(base_yaw), math.sin(base_yaw))
    sx = rx + dir_xy[0] * grasp_cfg.survey_forward_from_base
    sy = ry + dir_xy[1] * grasp_cfg.survey_forward_from_base
    sz = z_top + grasp_cfg.survey_height_above_table

    open_gripper(robot_id, grasp_cfg)
    step_sim(sim_cfg, int(0.10 / sim_cfg.timestep))
    return move_ee_pose(
        sim_cfg, robot_id, ee_link, arm_joints, ik_params,
        (sx, sy, sz), grasp_quat_down, grasp_cfg, table_id=table_id
    )


# -----------------------------
# Place slots
# -----------------------------
def compute_place_slots(
    table_id: int,
    spawn_cfg: SpawnConfig,
    robot_base_xy: Tuple[float, float],
    base_yaw: float,
    n: int,
) -> List[Tuple[float, float]]:
    aabb_min, aabb_max = get_table_aabb(table_id)
    x_min = aabb_min[0] + spawn_cfg.margin_xy
    x_max = aabb_max[0] - spawn_cfg.margin_xy
    y_min = aabb_min[1] + spawn_cfg.margin_xy
    y_max = aabb_max[1] - spawn_cfg.margin_xy

    rx, ry = robot_base_xy
    dir_xy = np.array([math.cos(base_yaw), math.sin(base_yaw)], dtype=np.float64)
    dir_xy = dir_xy / max(1e-9, np.linalg.norm(dir_xy))
    perp_xy = np.array([-dir_xy[1], dir_xy[0]], dtype=np.float64)

    anchor = np.array([rx, ry], dtype=np.float64) + dir_xy * 0.50 + perp_xy * 0.18
    spacing = max(0.10, spawn_cfg.min_dist_xy)

    slots = []
    offsets = [(i - (n - 1) / 2.0) * spacing for i in range(n)]
    for off in offsets:
        pt = anchor + perp_xy * off
        pt[0] = float(min(max(pt[0], x_min), x_max))
        pt[1] = float(min(max(pt[1], y_min), y_max))

        d = float(np.linalg.norm(pt - np.array([rx, ry])))
        if d > spawn_cfg.reach_max:
            pt = np.array([rx, ry], dtype=np.float64) + (pt - np.array([rx, ry])) * (spawn_cfg.reach_max / d)

        slots.append((float(pt[0]), float(pt[1])))

    # Ensure minimum separation (simple)
    cleaned = []
    for s in slots:
        ok = True
        for t in cleaned:
            if (s[0] - t[0]) ** 2 + (s[1] - t[1]) ** 2 < (spawn_cfg.min_dist_xy ** 2):
                ok = False
                break
        if ok:
            cleaned.append(s)

    while len(cleaned) < n:
        cleaned.append(cleaned[-1])
    return cleaned[:n]


# -----------------------------
# Pick and place
# -----------------------------
def grasp_pick_once(
    sim_cfg: SimConfig,
    grasp_cfg: GraspConfig,
    table_id: int,
    robot_id: int,
    ee_link: int,
    arm_joints: List[int],
    ik_params,
    cube_id: int,
    cube_size: float,
    z_top: float,
    target_xy: Tuple[float, float],
    refine_xy_fn: Callable[[], Optional[Tuple[float, float]]],
    grasp_quat_down: Tuple[float, float, float, float],
) -> bool:
    """
    One pick attempt (descend sweep).
    """
    if not p.isConnected():
        return False

    open_gripper(robot_id, grasp_cfg)
    step_sim(sim_cfg, int(0.10 / sim_cfg.timestep))

    x, y = target_xy
    refined = refine_xy_fn()
    if refined is not None:
        x, y = refined

    z_cube_top = z_top + cube_size

    pre = (x, y, z_cube_top + grasp_cfg.approach_height)
    if not move_ee_pose(sim_cfg, robot_id, ee_link, arm_joints, ik_params, pre, grasp_quat_down, grasp_cfg, table_id):
        return False

    for dz in grasp_cfg.descend_top_offsets:
        grasp_pose = (x, y, z_cube_top + dz)
        if not move_ee_pose(sim_cfg, robot_id, ee_link, arm_joints, ik_params, grasp_pose, grasp_quat_down, grasp_cfg, table_id):
            return False

        close_gripper(robot_id, grasp_cfg)
        step_sim(sim_cfg, int(grasp_cfg.gripper_settle_seconds / sim_cfg.timestep))

        lift_pose = (x, y, z_cube_top + grasp_cfg.lift_height)
        if not move_ee_pose(sim_cfg, robot_id, ee_link, arm_joints, ik_params, lift_pose, grasp_quat_down, grasp_cfg, table_id):
            return False

        # success check
        try:
            pos_after, _ = p.getBasePositionAndOrientation(cube_id)
        except p.error:
            return False

        if pos_after[2] >= (z_cube_top + grasp_cfg.success_lift_threshold):
            return True

        try:
            cps = p.getContactPoints(bodyA=robot_id, bodyB=cube_id)
            if cps and len(cps) > 0:
                return True
        except p.error:
            pass

        # failed dz: reopen and go back up to pre-grasp before trying next dz
        open_gripper(robot_id, grasp_cfg)
        step_sim(sim_cfg, int(0.10 / sim_cfg.timestep))
        if not move_ee_pose(sim_cfg, robot_id, ee_link, arm_joints, ik_params, pre, grasp_quat_down, grasp_cfg, table_id):
            return False

    return False


def place_object(
    sim_cfg: SimConfig,
    grasp_cfg: GraspConfig,
    table_id: int,
    robot_id: int,
    ee_link: int,
    arm_joints: List[int],
    ik_params,
    cube_size: float,
    z_top: float,
    place_xy: Tuple[float, float],
    grasp_quat_down: Tuple[float, float, float, float],
) -> bool:
    if not p.isConnected():
        return False

    px, py = place_xy
    z_cube_top = z_top + cube_size

    pre = (px, py, z_cube_top + grasp_cfg.approach_height)
    if not move_ee_pose(sim_cfg, robot_id, ee_link, arm_joints, ik_params, pre, grasp_quat_down, grasp_cfg, table_id):
        return False

    down = (px, py, z_cube_top + grasp_cfg.place_top_offset)
    if not move_ee_pose(sim_cfg, robot_id, ee_link, arm_joints, ik_params, down, grasp_quat_down, grasp_cfg, table_id):
        return False

    open_gripper(robot_id, grasp_cfg)
    step_sim(sim_cfg, int(grasp_cfg.place_settle_seconds / sim_cfg.timestep))

    if not move_ee_pose(sim_cfg, robot_id, ee_link, arm_joints, ik_params, pre, grasp_quat_down, grasp_cfg, table_id):
        return False

    return True


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--renderer", choices=["tiny", "opengl", "auto"], default="tiny")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--n_cubes", type=int, default=4)
    parser.add_argument("--run-forever", action="store_true")
    args = parser.parse_args()

    if args.gui and args.direct:
        raise ValueError("Choose only one of --gui or --direct.")

    gui = not args.direct
    render = not args.no_render
    if args.render:
        render = True

    sim_cfg = SimConfig(
        gui=gui,
        render=render,
        seed=args.seed,
        steps=args.steps,
        run_until_window_closed=(args.run_forever or True),
        renderer=args.renderer,
    )
    spawn_cfg = SpawnConfig(n_cubes=args.n_cubes)
    cam_cfg = CameraConfig()
    grasp_cfg = GraspConfig()

    random.seed(sim_cfg.seed)

    table_id, robot_id, table_center, z_top, robot_base_xy, base_yaw = build_scene(sim_cfg)
    ee_link = find_end_effector_link(robot_id)

    obj_ids = spawn_objects_reachable_on_table(table_id, spawn_cfg, robot_base_xy)

    place_slots = compute_place_slots(table_id, spawn_cfg, robot_base_xy, base_yaw, len(obj_ids))
    cube_to_slot = {bid: place_slots[i] for i, bid in enumerate(obj_ids)}

    arm_joints = list(range(7))
    rest_poses = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
    ik_params = get_arm_ik_params(robot_id, arm_joints, rest_poses)

    grasp_quat_down = p.getQuaternionFromEuler([math.pi, 0.0, 0.0])

    print(f"Table: {table_id} | Robot: {robot_id} | EE_link: {ee_link} | Cubes: {obj_ids}")
    print("Enter cube BODY ID to pick+place. Enter 'q' to quit.\n")

    print("Spawned cube mapping:")
    for i, bid in enumerate(obj_ids):
        pos, _ = p.getBasePositionAndOrientation(bid)
        print(f"  cube[{i}] -> body_id={bid}, pos={tuple(round(x, 3) for x in pos)}, place_slot={tuple(round(x,3) for x in cube_to_slot[bid])}")
    print()

    renderer_id = choose_renderer(sim_cfg)
    proj = projection_matrix(cam_cfg)
    K = compute_K(cam_cfg)

    # Overhead camera view covers full reachable region (+ margin)
    region_center_xy = ((robot_base_xy[0] + table_center[0]) / 2.0, (robot_base_xy[1] + table_center[1]) / 2.0)
    region_radius = spawn_cfg.reach_max + 0.12
    overhead_view = compute_overhead_view_for_region(region_center_xy, region_radius, cam_cfg, z_top)

    if sim_cfg.gui:
        p.resetDebugVisualizerCamera(
            cameraDistance=1.20,
            cameraYaw=45,
            cameraPitch=-35,
            cameraTargetPosition=[region_center_xy[0], region_center_xy[1], z_top + 0.1],
        )

    def compute_overhead_detection() -> Dict[int, Dict]:
        V_oh = view_to_4x4_colmajor(overhead_view)
        rgb_oh, depth_oh, seg_oh = render_camera(overhead_view, proj, cam_cfg, renderer_id)
        return detect_from_buffers(rgb_oh, depth_oh, seg_oh, obj_ids, cam_cfg, K, V_oh)

    def compute_wrist_detection_look_at(point_xyz: Tuple[float, float, float]) -> Dict[int, Dict]:
        view_wr = wrist_view_attached_look_at_point(robot_id, ee_link, cam_cfg, point_xyz)
        V_wr = view_to_4x4_colmajor(view_wr)
        rgb_wr, depth_wr, seg_wr = render_camera(view_wr, proj, cam_cfg, renderer_id)
        return detect_from_buffers(rgb_wr, depth_wr, seg_wr, obj_ids, cam_cfg, K, V_wr)

    # Settle, then go to survey pose immediately (reduces occlusion from the start)
    step_sim(sim_cfg, int(0.4 / sim_cfg.timestep))
    move_to_survey_pose(sim_cfg, grasp_cfg, table_id, robot_id, ee_link, arm_joints, ik_params,
                        z_top, robot_base_xy, base_yaw, grasp_quat_down)

    latest_det_overhead: Dict[int, Dict] = {}
    latest_det_wrist: Dict[int, Dict] = {}

    if sim_cfg.render:
        latest_det_overhead = compute_overhead_detection()
        print("Initial overhead detections:", format_det(latest_det_overhead), "\n")

    step = 0

    def should_continue() -> bool:
        if not p.isConnected():
            return False
        if sim_cfg.gui and sim_cfg.run_until_window_closed:
            return True
        return step < sim_cfg.steps

    while should_continue():
        if not p.isConnected():
            print("Disconnected from physics server. Exiting.")
            break

        try:
            p.stepSimulation()
        except p.error:
            print("Physics server disconnected. Exiting.")
            break

        if sim_cfg.gui and sim_cfg.sleep_if_gui:
            time.sleep(sim_cfg.timestep)

        if sim_cfg.render and (step % cam_cfg.print_every_n_steps == 0):
            try:
                latest_det_overhead = compute_overhead_detection()
                latest_det_wrist = compute_wrist_detection_look_at((region_center_xy[0], region_center_xy[1], z_top + 0.02))
            except p.error:
                print("Disconnected during camera render. Exiting.")
                break

            print(f"[step {step:05d}] Overhead: {format_det(latest_det_overhead)}")
            print(f"[step {step:05d}] Wrist:    {format_det(latest_det_wrist)}")

            available = sorted(latest_det_overhead.keys())
            if available:
                prompt = f"\nPick a cube BODY ID from {available} (Enter=skip, q=quit): "
            else:
                prompt = "\nNo cubes visible overhead. Enter=continue, q=quit: "

            choice = input(prompt).strip().lower()
            if choice == "q":
                break
            if choice == "":
                step += 1
                continue

            try:
                target_id = int(choice)
            except ValueError:
                print("Invalid input. Enter a numeric body ID.\n")
                step += 1
                continue

            if target_id not in obj_ids:
                print(f"Cube id {target_id} is not one of the spawned cubes.\n")
                step += 1
                continue

            # If currently not visible: first go to survey pose, re-detect once, then decide.
            if target_id not in latest_det_overhead:
                print("Target not visible now. Moving to survey pose to re-acquire...")
                move_to_survey_pose(sim_cfg, grasp_cfg, table_id, robot_id, ee_link, arm_joints, ik_params,
                                    z_top, robot_base_xy, base_yaw, grasp_quat_down)
                try:
                    latest_det_overhead = compute_overhead_detection()
                except p.error:
                    print("Disconnected during re-acquire.")
                    break
                if target_id not in latest_det_overhead:
                    print(f"Cube id {target_id} still not visible overhead. Pick again.\n")
                    step += 1
                    continue

            # Initial XY from overhead surface point
            x_s, y_s, _ = latest_det_overhead[target_id]["world_xyz_surface"]
            target_xy = (float(x_s), float(y_s))

            # Wrist refinement closure
            def refine_xy() -> Optional[Tuple[float, float]]:
                try:
                    det_wr = compute_wrist_detection_look_at((target_xy[0], target_xy[1], z_top + 0.02))
                except p.error:
                    return None
                if target_id in det_wr:
                    wx, wy, _ = det_wr[target_id]["world_xyz_surface"]
                    return (float(wx), float(wy))
                return None

            # NEW: automatic retries with re-observe
            max_total_attempts = 1 + grasp_cfg.max_pick_retries
            picked = False
            for attempt in range(max_total_attempts):
                print(f"\nPick attempt {attempt+1}/{max_total_attempts} for cube {target_id} (XY={tuple(round(v,3) for v in target_xy)})")

                picked = grasp_pick_once(
                    sim_cfg=sim_cfg,
                    grasp_cfg=grasp_cfg,
                    table_id=table_id,
                    robot_id=robot_id,
                    ee_link=ee_link,
                    arm_joints=arm_joints,
                    ik_params=ik_params,
                    cube_id=target_id,
                    cube_size=spawn_cfg.cube_size,
                    z_top=z_top,
                    target_xy=target_xy,
                    refine_xy_fn=refine_xy,
                    grasp_quat_down=grasp_quat_down,
                )

                if picked:
                    break

                # If failed: go UP to survey pose, re-acquire position, retry
                print("Pick failed. Moving up to survey pose to re-acquire and retry...")
                move_to_survey_pose(sim_cfg, grasp_cfg, table_id, robot_id, ee_link, arm_joints, ik_params,
                                    z_top, robot_base_xy, base_yaw, grasp_quat_down)

                try:
                    latest_det_overhead = compute_overhead_detection()
                except p.error:
                    picked = False
                    break

                if target_id in latest_det_overhead:
                    x_s, y_s, _ = latest_det_overhead[target_id]["world_xyz_surface"]
                    target_xy = (float(x_s), float(y_s))
                else:
                    print("Re-acquire failed: cube not visible overhead. Will still retry with last XY.")

            if not picked:
                print("Pick FAILED after retries. You can try again or pick another cube.\n")
                # Always return to survey pose so vision improves for next selection
                move_to_survey_pose(sim_cfg, grasp_cfg, table_id, robot_id, ee_link, arm_joints, ik_params,
                                    z_top, robot_base_xy, base_yaw, grasp_quat_down)
                step += 1
                continue

            # Place
            place_xy = cube_to_slot[target_id]
            print(f"Placing cube {target_id} at {tuple(round(v,3) for v in place_xy)} ...")
            placed = place_object(
                sim_cfg=sim_cfg,
                grasp_cfg=grasp_cfg,
                table_id=table_id,
                robot_id=robot_id,
                ee_link=ee_link,
                arm_joints=arm_joints,
                ik_params=ik_params,
                cube_size=spawn_cfg.cube_size,
                z_top=z_top,
                place_xy=place_xy,
                grasp_quat_down=grasp_quat_down,
            )
            print(("Place SUCCESS\n" if placed else "Place FAILED (object may have dropped)\n"))

            # NEW: after placing, go to survey pose to improve visibility for next selection
            move_to_survey_pose(sim_cfg, grasp_cfg, table_id, robot_id, ee_link, arm_joints, ik_params,
                                z_top, robot_base_xy, base_yaw, grasp_quat_down)

        step += 1

    if p.isConnected():
        p.disconnect()


if __name__ == "__main__":
    main()