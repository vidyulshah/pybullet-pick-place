"""
sim/vision.py
All vision logic: camera setup, rendering, segmentation-based detection,
depth-buffer conversion, and pixel-to-world back-projection.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Tuple

import numpy as np
import pybullet as p

from .config import CameraConfig

log = logging.getLogger("pick_place.vision")


# ── Camera math ───────────────────────────────────────────────────────────────
def compute_K(cam_cfg: CameraConfig) -> np.ndarray:
    W, H  = cam_cfg.width, cam_cfg.height
    fov_y = math.radians(cam_cfg.fov_y_deg)
    fy    = (H / 2.0) / math.tan(fov_y / 2.0)
    fx    = fy * (W / H)
    return np.array([[fx, 0.0, W / 2.0],
                     [0.0, fy, H / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def projection_matrix(cam_cfg: CameraConfig) -> List[float]:
    return p.computeProjectionMatrixFOV(
        fov    = cam_cfg.fov_y_deg,
        aspect = cam_cfg.width / cam_cfg.height,
        nearVal= cam_cfg.near,
        farVal = cam_cfg.far,
    )


def view_to_4x4(view: List[float]) -> np.ndarray:
    return np.array(view, dtype=np.float64).reshape((4, 4), order="F")


def depth_buf_to_meters(buf: np.ndarray, near: float, far: float) -> np.ndarray:
    z = buf.astype(np.float64)
    return (far * near) / (far - (far - near) * z)


def backproject(u: float, v: float, depth_m: float,
                K: np.ndarray, V_world_to_cam: np.ndarray
                ) -> Tuple[float, float, float]:
    """OpenGL camera convention: +X right, +Y up, looks down -Z."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X, Y, Z = (u - cx) * depth_m / fx, -(v - cy) * depth_m / fy, -depth_m
    Pw = np.linalg.inv(V_world_to_cam) @ np.array([X, Y, Z, 1.0])
    return float(Pw[0]), float(Pw[1]), float(Pw[2])


# ── View matrices ─────────────────────────────────────────────────────────────
def overhead_view(region_center_xy: Tuple[float, float],
                  region_radius: float,
                  cam_cfg: CameraConfig,
                  z_top: float) -> List[float]:
    alpha = math.radians(cam_cfg.fov_y_deg) / 2.0
    h     = max((region_radius / max(1e-6, math.tan(alpha))) * 1.25, 0.65)
    eye   = [region_center_xy[0], region_center_xy[1], z_top + h]
    tgt   = [region_center_xy[0], region_center_xy[1], z_top + 0.02]
    return p.computeViewMatrix(eye, tgt, [1.0, 0.0, 0.0])


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n < 1e-9 else v / n


def wrist_view(robot_id: int, ee_link: int,
               cam_cfg: CameraConfig,
               look_at: Tuple[float, float, float]) -> List[float]:
    state   = p.getLinkState(robot_id, ee_link, computeForwardKinematics=True)
    ee_pos, ee_orn = state[0], state[1]
    R       = np.array(p.getMatrixFromQuaternion(ee_orn)).reshape(3, 3)
    eye     = np.array(ee_pos) + R @ np.array(cam_cfg.ee_cam_offset)
    target  = np.array(look_at)
    fwd     = _normalize(target - eye)
    up_g    = np.array([0.0, 0.0, 1.0]) if abs(float(np.dot(fwd, [0, 0, 1]))) < 0.95 \
              else np.array([0.0, 1.0, 0.0])
    up      = _normalize(np.cross(_normalize(np.cross(fwd, up_g)), fwd))
    return p.computeViewMatrix(eye.tolist(), target.tolist(), up.tolist())


# ── Rendering + detection ─────────────────────────────────────────────────────
def render(view: List[float], proj: List[float],
           cam_cfg: CameraConfig, renderer_id: int):
    img = p.getCameraImage(
        cam_cfg.width, cam_cfg.height,
        viewMatrix       = view,
        projectionMatrix = proj,
        renderer         = renderer_id,
        flags            = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
        shadow           = 1,
        lightDirection   = [1, 1, 1],
    )
    w, h = img[0], img[1]
    rgb  = np.array(img[2], dtype=np.uint8).reshape((h, w, 4))
    dep  = np.array(img[3], dtype=np.float64).reshape((h, w))
    seg  = np.array(img[4], dtype=np.int64).reshape((h, w))
    return rgb, dep, seg


def _color_class(rgb_px: Tuple[int, int, int]) -> str:
    r, g, b = float(rgb_px[0]), float(rgb_px[1]), float(rgb_px[2])
    vals   = [r, g, b]
    idx    = int(np.argmax(vals))
    second = sorted(vals)[-2]
    if vals[idx] - second < 15.0:
        return "other"
    return ["red", "green", "blue"][idx]


def _robust_depth(depth_m: np.ndarray, u: int, v: int, patch: int = 5) -> float:
    H, W = depth_m.shape
    r    = patch // 2
    patch_vals = depth_m[max(0, v-r):min(H, v+r+1),
                         max(0, u-r):min(W, u+r+1)].reshape(-1)
    return float(np.median(patch_vals)) if patch_vals.size else float(depth_m[v, u])


def detect(rgb: np.ndarray, depth_buf: np.ndarray, seg: np.ndarray,
           obj_ids: List[int], cam_cfg: CameraConfig,
           K: np.ndarray, V_world_to_cam: np.ndarray) -> Dict[int, Dict]:
    uid_map  = (seg.astype(np.int64) & ((1 << 24) - 1))
    uid_map[seg < 0] = 0
    depth_m  = depth_buf_to_meters(depth_buf, cam_cfg.near, cam_cfg.far)
    W, H     = cam_cfg.width, cam_cfg.height
    det: Dict[int, Dict] = {}

    for bid in obj_ids:
        mask  = (uid_map == bid)
        count = int(mask.sum())
        if count <= 0:
            continue
        ys, xs = np.nonzero(mask)
        u = int(np.clip(round(float(xs.mean())), 0, W - 1))
        v = int(np.clip(round(float(ys.mean())), 0, H - 1))
        d = _robust_depth(depth_m, u, v)
        wx, wy, wz = backproject(u, v, d, K, V_world_to_cam)
        px = rgb[v, u, :3].astype(np.int32)
        det[bid] = {
            "class":             _color_class((int(px[0]), int(px[1]), int(px[2]))),
            "pixel_count":       count,
            "bbox_xyxy":         (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
            "centroid_uv":       (u, v),
            "world_xyz_surface": (wx, wy, wz),
        }
    return det


def fmt_det(det: Dict[int, Dict]) -> str:
    if not det:
        return "none"
    return "; ".join(
        f"{bid}(cls={info['class']},px={info['pixel_count']},xyz={tuple(round(v,3) for v in info['world_xyz_surface'])})"
        for bid, info in sorted(det.items(), key=lambda kv: kv[1]["pixel_count"], reverse=True)
    )
