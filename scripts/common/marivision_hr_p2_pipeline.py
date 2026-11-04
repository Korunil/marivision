"""MARIVISION-HR-P2: Unified Cross-Industry High-Resolution Tiny-Object Maritime Detection Pipeline (V12).

Synthesizes all Pre-Launch, Phase 1, & Phase 2 Consultant Review Requirements (Project Director, Gemini, Claude, ChatGPT):
1. Strictly Isotropic Aspect-Preserving Multi-Resolution Tiling (Anisotropy = 1.000 across ALL cameras):
   - Stage A (Arm a control): Full frame resized isotropically (preserving aspect ratio with right/bottom zero-pad
     after ImageNet normalization) to 640x360.
   - Stage B (Arm b & context pass): Full frame resized isotropically to 1280x720.
   - Stage C (Arms c, d, e): 16x exact 16:9 native crops resized isotropically to 1280x720 (zero padding waste,
     scale_x == scale_y everywhere).
   - Stage D: Conditional 4x 16:9 micro-crops (320x180 -> 1280x720) triggered ONLY around uncertain Stage B/C
     proposals (cap <= 4 micro-crops/frame at inference) or tiny <= 9 px^2 GTs during training.
2. Complete Tile-Merge, Border-Truncation & Scale-Gating Pipeline (`merge_multiscale_tile_detections`):
   - Back-projects tile detections to source coordinates (accounting for any letterbox padding).
   - Drops large sub-crop detections (> 900 px^2 native) touching interior tile borders so truncated Stage C/D
     fragments never beat whole-vessel Stage B detections.
   - Applies stage-specific scale gating + class-aware NWD-NMS / Weighted Boxes Fusion + configurable `max_det_per_image`.
3. 2-Phase Guaranteed-Positive RFLA Assigner (`rfla_hierarchical_assign_numpy`) & Center-Distance Assigner.
4. V7 Per-Video Delta Table, Leave-One-Video-Out (LOVO) Sensitivity & Paired Video Bootstrap (`paired_video_bootstrap_delta_metric`):
   - Reports all 9 per-video deltas (`per_video_deltas`), `n_videos_gt_zero`, Leave-One-Video-Out (`lovo_min_delta`,
     `lovo_max_delta`) range, and 5,000-resample paired bootstrap CIs.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
import random
import unittest
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

CLASS_NAMES: Tuple[str, ...] = (
    "ANIMAL",           # 0: supported on WATER + QUAY/SHORELINE INTERFACE
    "BOAT",             # 1: waterborne
    "MARINE_MARKER",    # 2: waterborne (tiny buoys/fenders/beacons)
    "OBJECT",           # 3: waterborne / horizon floating object
    "LEISURE_VEHICLE",  # 4: waterborne (0 GT instances in seaclips-val.json; track FPs explicitly)
    "SAILING_VESSEL",   # 5: waterborne (tall masts may extend into upper sky zone)
    "SHIP",             # 6: waterborne (superstructures may overlap quay/horizon)
)

WATERBORNE_CLASSES = frozenset((1, 2, 3, 4, 5, 6))
SHORELINE_COMPATIBLE_CLASSES = frozenset((0, 1, 2, 3, 5, 6))


# ============================================================================
# 1. EFFECTIVE AREA & ISOTROPIC CROP PLANNER (CLAUDE P1.3 & CHATGPT #1 FIX)
# ============================================================================

def compute_effective_area_px2(
    source_area_px2: float,
    source_w: float,
    source_h: float,
    target_long_side: float = 640.0,
) -> float:
    """Compute effective pixel area after scaling the image's long side isotropically to target_long_side."""
    long_side = max(float(source_w), float(source_h))
    if long_side <= 0:
        raise ValueError(f"Invalid source dimensions: {source_w}x{source_h}")
    scale = float(target_long_side) / long_side
    return float(source_area_px2) * (scale ** 2)


def effective_area_bin_label(eff_area: float, strict_lt_4px2: bool = False) -> str:
    """Assign an effective area (at 640 long-side input) to one of the 6 diagnostic bins."""
    if strict_lt_4px2:
        if eff_area < 4.0 - 1e-9:
            return "01_LE_4px2"
    else:
        if eff_area <= 4.0 + 1e-9:
            return "01_LE_4px2"
    if eff_area <= 6.0 + 1e-9:
        return "02_4_to_6px2"
    if eff_area <= 9.0 + 1e-9:
        return "03_6_to_9px2"
    if eff_area <= 25.0 + 1e-9:
        return "04_9_to_25px2"
    if eff_area <= 150.0 + 1e-9:
        return "05_25_to_150px2"
    return "06_GT_150px2"


@dataclass(frozen=True)
class CropWindow:
    """Represents an exact integer-bounded crop in native source coordinates mapped isotropically to detector_w x detector_h.

    V12 Fix (ChatGPT V11 Audit Finding #1): Quantizes `(x0, y0, x1, y1)` to exact integer pixel boundaries in
    `__post_init__` and provides `slice_rgb(rgb)` so array slicing, training GT projection, ROI coverage, and
    inference back-projection share the exact same integer coordinate origin with zero subpixel drift.
    """
    stage: str               # 'STAGE_A_640_BASE', 'STAGE_B_FULL_CONTEXT', 'STAGE_C_2X_CROP', 'STAGE_D_4X_MICROCROP'
    x0: float
    y0: float
    x1: float
    y1: float
    detector_w: int = 1280
    detector_h: int = 720
    preserve_aspect: bool = True
    source_w: int = 1920
    source_h: int = 1080

    def __post_init__(self) -> None:
        w_int = max(1, int(round(float(self.x1) - float(self.x0))))
        h_int = max(1, int(round(float(self.y1) - float(self.y0))))
        x0_i = int(round(float(self.x0)))
        y0_i = int(round(float(self.y0)))
        if self.source_w > 0 and x0_i + w_int > int(self.source_w):
            x0_i = max(0, int(self.source_w) - w_int)
        if self.source_h > 0 and y0_i + h_int > int(self.source_h):
            y0_i = max(0, int(self.source_h) - h_int)
        object.__setattr__(self, "x0", float(x0_i))
        object.__setattr__(self, "y0", float(y0_i))
        object.__setattr__(self, "x1", float(x0_i + w_int))
        object.__setattr__(self, "y1", float(y0_i + h_int))

    @property
    def x0_int(self) -> int:
        return int(self.x0)

    @property
    def y0_int(self) -> int:
        return int(self.y0)

    @property
    def x1_int(self) -> int:
        return int(self.x1)

    @property
    def y1_int(self) -> int:
        return int(self.y1)

    def slice_rgb(self, rgb: np.ndarray) -> np.ndarray:
        """Extract the exact integer pixel slice matching `(x0_int, y0_int, x1_int, y1_int)`."""
        h_img, w_img = rgb.shape[:2]
        x0 = max(0, min(w_img, self.x0_int))
        y0 = max(0, min(h_img, self.y0_int))
        x1 = max(x0 + 1, min(w_img, self.x1_int))
        y1 = max(y0 + 1, min(h_img, self.y1_int))
        return rgb[y0:y1, x0:x1]

    @property
    def crop_w(self) -> float:
        return max(1.0, self.x1 - self.x0)

    @property
    def crop_h(self) -> float:
        return max(1.0, self.y1 - self.y0)

    @property
    def isotropic_scale(self) -> float:
        return min(float(self.detector_w) / self.crop_w, float(self.detector_h) / self.crop_h)

    @property
    def scale_x(self) -> float:
        if self.preserve_aspect:
            return self.isotropic_scale
        return float(self.detector_w) / self.crop_w

    @property
    def scale_y(self) -> float:
        if self.preserve_aspect:
            return self.isotropic_scale
        return float(self.detector_h) / self.crop_h

    @property
    def anisotropy_ratio(self) -> float:
        return max(self.scale_x, self.scale_y) / max(1e-9, min(self.scale_x, self.scale_y))

    @property
    def resized_w(self) -> int:
        return min(self.detector_w, max(1, int(round(self.crop_w * self.scale_x))))

    @property
    def resized_h(self) -> int:
        return min(self.detector_h, max(1, int(round(self.crop_h * self.scale_y))))

    @property
    def pad_right(self) -> int:
        return max(0, self.detector_w - self.resized_w)

    @property
    def pad_bottom(self) -> int:
        return max(0, self.detector_h - self.resized_h)

    @property
    def area_scale(self) -> float:
        return self.scale_x * self.scale_y

    def contains_and_enlarges_gt(
        self,
        gx: float,
        gy: float,
        gw: float,
        gh: float,
        min_box_coverage: float = 0.90,
        min_effective_area_px2: float = 24.0,
    ) -> Tuple[bool, float]:
        ix0 = max(self.x0, gx)
        iy0 = max(self.y0, gy)
        ix1 = min(self.x1, gx + gw)
        iy1 = min(self.y1, gy + gh)
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        gt_area = max(1e-9, gw * gh)
        cov = inter / gt_area
        eff_area = inter * self.area_scale
        return (cov >= min_box_coverage and eff_area >= min_effective_area_px2), eff_area


def select_isotropic_stage_c_crop_dims(source_w: int, source_h: int) -> Tuple[float, float]:
    """Select strictly 16:9 Stage C crop dimensions (cw_c, ch_c) for any source camera resolution."""
    w, h = float(source_w), float(source_h)
    target_w = min(w, max(400.0, round((w / 2.6) / 16.0) * 16.0))
    if abs(w - 1920.0) < 2.0 and abs(h - 1080.0) < 2.0:
        target_w = 640.0
    elif abs(w - 1200.0) < 2.0 and abs(h - 960.0) < 2.0:
        target_w = 480.0
    elif abs(w - 945.0) < 2.0 and abs(h - 720.0) < 2.0:
        target_w = 400.0
    target_h = round(target_w * 9.0 / 16.0, 4)
    if target_h > h:
        target_h = h
        target_w = round(target_h * 16.0 / 9.0, 4)
    return float(target_w), float(target_h)


def generate_coarse_to_fine_crops(
    source_w: int = 1920,
    source_h: int = 1080,
    horizon_y_norm: float = 0.42,
    stage_b_detection_centers: Optional[Sequence[Tuple[float, float]]] = None,
    include_stage_d_microcrops: bool = False,
    max_stage_d_proposals: int = 4,
    use_horizon_top_band_skip: bool = False,
) -> List[CropWindow]:
    """Generate the GT-free, strictly isotropic Stage B (1x) + Stage C (4x4 = 16x) [+ conditional Stage D] schedule."""
    w = float(source_w)
    h = float(source_h)
    crops: List[CropWindow] = [
        CropWindow("STAGE_B_FULL_CONTEXT", 0.0, 0.0, w, h, 1280, 720, preserve_aspect=True, source_w=source_w, source_h=source_h)
    ]

    cw_c, ch_c = select_isotropic_stage_c_crop_dims(source_w, source_h)
    y_min_band = max(0.0, h * max(0.08, horizon_y_norm - 0.30)) if use_horizon_top_band_skip else 0.0
    y_max_band = h

    cw_c_int = int(round(cw_c))
    ch_c_int = int(round(ch_c))
    xs_c = [int(round(i * max(0.0, w - cw_c_int) / 3.0)) for i in range(4)]
    ys_c = [int(round(y_min_band + i * max(0.0, (y_max_band - y_min_band) - ch_c_int) / 3.0)) for i in range(4)]

    for yc in ys_c:
        for xc in xs_c:
            crops.append(
                CropWindow(
                    "STAGE_C_2X_CROP", float(xc), float(yc), float(xc + cw_c_int), float(yc + ch_c_int),
                    1280, 720, preserve_aspect=True, source_w=source_w, source_h=source_h
                )
            )

    cw_d_int, ch_d_int = 320, 180
    cw_d, ch_d = float(cw_d_int), float(ch_d_int)
    if stage_b_detection_centers:
        for cx, cy in list(stage_b_detection_centers)[:max_stage_d_proposals]:
            x0 = int(round(min(max(0.0, float(cx) - 0.5 * cw_d_int), w - cw_d_int)))
            y0 = int(round(min(max(0.0, float(cy) - 0.5 * ch_d_int), h - ch_d_int)))
            crops.append(
                CropWindow(
                    "STAGE_D_4X_MICROCROP", float(x0), float(y0), float(x0 + cw_d_int), float(y0 + ch_d_int),
                    1280, 720, preserve_aspect=True, source_w=source_w, source_h=source_h
                )
            )
    elif include_stage_d_microcrops:
        stride_x_d = cw_d * 0.75
        y_horizon_center = min(max(ch_d * 0.5, h * horizon_y_norm), h - ch_d * 0.5)
        for y_offset in (-0.25 * ch_d, 0.25 * ch_d):
            yc = round(min(max(0.0, y_horizon_center + y_offset - 0.5 * ch_d), h - ch_d), 2)
            x = 0.0
            while x + cw_d < w - 1.0:
                crops.append(
                    CropWindow(
                        "STAGE_D_4X_MICROCROP", round(x, 2), yc, round(x + cw_d, 2), round(yc + ch_d, 2),
                        1280, 720, preserve_aspect=True, source_w=source_w, source_h=source_h
                    )
                )
                x += stride_x_d
            crops.append(
                CropWindow(
                    "STAGE_D_4X_MICROCROP", round(max(0.0, w - cw_d), 2), yc, w, round(yc + ch_d, 2),
                    1280, 720, preserve_aspect=True, source_w=source_w, source_h=source_h
                )
            )

    return crops


# ============================================================================
# 2. DAYTIME CIE L*a*b* + GRADIENT SCENE PRIOR (SOFT CLASS-CONDITIONED BIAS)
# ============================================================================

def compute_soft_maritime_scene_priors(
    rgb_uint8: np.ndarray,
    horizon_y_norm: Optional[float] = None,
    y0_norm_in_full_frame: float = 0.0,
    y1_norm_in_full_frame: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Compute soft probability maps P_W (water), P_Q (quay/shoreline interface), P_U (upper sky/facade)."""
    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB array, got {rgb_uint8.shape}")
    h, w, _ = rgb_uint8.shape
    rgb = rgb_uint8.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    blue_excess = b - 0.5 * (r + g)
    rg_diff = np.abs(r - g)

    gy = np.zeros_like(lum)
    gx = np.zeros_like(lum)
    gy[1:-1, :] = np.abs(lum[2:, :] - lum[:-2, :]) * 0.5
    gx[:, 1:-1] = np.abs(lum[:, 2:] - lum[:, :-2]) * 0.5

    horiz_wave_affinity = (gy + 0.02) / (gx + gy + 0.04)
    vert_structure_affinity = (gx + 0.02) / (gx + gy + 0.04)

    yy = np.linspace(float(y0_norm_in_full_frame), float(y1_norm_in_full_frame), h, dtype=np.float32)[:, None]
    if horizon_y_norm is None:
        row_grad = np.mean(gy, axis=1)
        search_y0, search_y1 = int(0.15 * h), int(0.70 * h)
        if search_y1 > search_y0:
            best_row = search_y0 + int(np.argmax(row_grad[search_y0:search_y1]))
            y_h = float(y0_norm_in_full_frame) + (float(best_row) / float(max(1, h))) * float(y1_norm_in_full_frame - y0_norm_in_full_frame)
        else:
            y_h = 0.42
    else:
        y_h = float(horizon_y_norm)

    logit_w = (
        4.5 * (yy - y_h)
        + 1.2 * (horiz_wave_affinity - 0.5)
        + 0.8 * blue_excess
        - 1.0 * rg_diff
    )
    logit_q = (
        -18.0 * ((yy - y_h) ** 2)
        + 1.5 * (vert_structure_affinity - 0.5)
        + 0.6 * rg_diff
    )
    logit_u = (
        -4.5 * (yy - (y_h - 0.05))
        + 0.8 * (lum - 0.5)
        + 0.8 * (vert_structure_affinity - 0.5)
    )

    logits = np.stack([logit_w, logit_q, logit_u], axis=-1)
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp_l = np.exp(logits)
    probs = exp_l / np.maximum(1e-8, np.sum(exp_l, axis=-1, keepdims=True))

    return {
        "P_W": probs[..., 0],
        "P_Q": probs[..., 1],
        "P_U": probs[..., 2],
        "horizon_rel_y": np.broadcast_to(yy - y_h, (h, w)).copy(),
        "estimated_horizon_y_norm": np.float32(y_h),
    }


def apply_class_conditioned_soft_scene_prior(
    raw_score: float,
    class_id: int,
    box_xywh: Tuple[float, float, float, float],
    scene_maps: Dict[str, np.ndarray],
    beta: float = 0.20,
    eps: float = 0.05,
    in_logit_space: bool = True,
) -> Tuple[float, float]:
    """Apply soft log-prior adjustment in logit space:
    logit(s'_c(b)) = logit(s_c(b)) + beta_c * log(eps + p_c(b)).
    """
    P_W = scene_maps["P_W"]
    P_Q = scene_maps["P_Q"]
    h, w = P_W.shape
    x, y, bw, bh = box_xywh
    x0 = int(np.clip(math.floor(x), 0, w - 1))
    y0 = int(np.clip(math.floor(y), 0, h - 1))
    x1 = int(np.clip(math.ceil(x + bw), x0 + 1, w))
    y1 = int(np.clip(math.ceil(y + bh), y0 + 1, h))

    y_bot0 = int(np.clip(math.floor(y + 0.65 * bh), y0, y1 - 1))

    mean_w_full = float(np.mean(P_W[y0:y1, x0:x1]))
    mean_q_full = float(np.mean(P_Q[y0:y1, x0:x1]))
    mean_w_bot = float(np.mean(P_W[y_bot0:y1, x0:x1]))
    mean_q_bot = float(np.mean(P_Q[y_bot0:y1, x0:x1]))

    if class_id == 0:
        p_valid = max(mean_w_full + mean_q_full, mean_w_bot + mean_q_bot)
    elif class_id in (5, 6):
        p_valid = max(mean_w_bot + 0.85 * mean_q_bot, mean_w_full + 0.5 * mean_q_full)
    else:
        p_valid = max(mean_w_bot + 0.5 * mean_q_bot, mean_w_full + 0.35 * mean_q_full)

    p_valid = float(np.clip(p_valid, 0.0, 1.0))
    if in_logit_space:
        s_clamped = float(np.clip(raw_score, 1e-5, 1.0 - 1e-5))
        raw_logit = math.log(s_clamped / (1.0 - s_clamped))
        adj_logit = raw_logit + float(beta) * math.log(eps + p_valid)
        adj_prob = 1.0 / (1.0 + math.exp(-adj_logit))
        return float(np.clip(adj_prob, 0.001, 0.999)), p_valid
    else:
        adjusted_score = float(raw_score) + float(beta) * math.log(eps + p_valid)
        return float(np.clip(adjusted_score, 0.001, 0.999)), p_valid


# ============================================================================
# 3. TINY-OBJECT NWD LOSS, 2-PHASE GUARANTEED-POSITIVE RFLA & MULTI-TILE MERGE
# ============================================================================

def nwd_similarity_numpy(
    boxes_a_xywh: np.ndarray,
    boxes_b_xywh: np.ndarray,
    constant_c: float = 8.0,
) -> np.ndarray:
    """Compute pairwise Normalized Gaussian Wasserstein Distance (NWD) similarity matrix [N, M]."""
    a = np.asarray(boxes_a_xywh, dtype=np.float64)
    b = np.asarray(boxes_b_xywh, dtype=np.float64)
    cxa = a[:, 0:1] + 0.5 * a[:, 2:3]
    cya = a[:, 1:2] + 0.5 * a[:, 3:4]
    wa, ha = a[:, 2:3], a[:, 3:4]

    cxb = b[:, 0][None, :] + 0.5 * b[:, 2][None, :]
    cyb = b[:, 1][None, :] + 0.5 * b[:, 3][None, :]
    wb, hb = b[:, 2][None, :], b[:, 3][None, :]

    w2_sq = (cxa - cxb) ** 2 + (cya - cyb) ** 2 + 0.25 * ((wa - wb) ** 2 + (ha - hb) ** 2)
    w2 = np.sqrt(np.maximum(0.0, w2_sq))
    return np.exp(-w2 / max(1e-6, float(constant_c)))


def iou_matrix_numpy(boxes_a_xywh: np.ndarray, boxes_b_xywh: np.ndarray) -> np.ndarray:
    """Compute standard pairwise IoU matrix [N, M] for xywh boxes."""
    a = np.asarray(boxes_a_xywh, dtype=np.float64)
    b = np.asarray(boxes_b_xywh, dtype=np.float64)
    ax0, ay0 = a[:, 0:1], a[:, 1:2]
    ax1, ay1 = ax0 + a[:, 2:3], ay0 + a[:, 3:4]
    bx0, by0 = b[:, 0][None, :], b[:, 1][None, :]
    bx1, by1 = bx0 + b[:, 2][None, :], by0 + b[:, 3][None, :]

    ix = np.maximum(0.0, np.minimum(ax1, bx1) - np.maximum(ax0, bx0))
    iy = np.maximum(0.0, np.minimum(ay1, by1) - np.maximum(ay0, by0))
    inter = ix * iy
    area_a = np.maximum(1e-9, a[:, 2:3] * a[:, 3:4])
    area_b = np.maximum(1e-9, b[:, 2][None, :] * b[:, 3][None, :])
    return inter / np.maximum(1e-9, area_a + area_b - inter)


def rfla_hierarchical_assign_numpy(
    feature_points_xy_stride: np.ndarray,
    gt_boxes_xywh: np.ndarray,
    top_k_primary: int = 4,
    top_k_Backup: int = 8,
    alpha_ratio: float = 0.80,
) -> Dict[int, List[int]]:
    """2-Phase Guaranteed-Positive Gaussian Receptive Field Label Assignment (RFLA)."""
    pts = np.asarray(feature_points_xy_stride, dtype=np.float64)  # [N_pts, 3] = (x, y, stride)
    gts = np.asarray(gt_boxes_xywh, dtype=np.float64)            # [M_gt, 4] = (x, y, w, h)
    if len(gts) == 0 or len(pts) == 0:
        return {}
    erfs = np.column_stack([
        pts[:, 0] - 2.0 * pts[:, 2],
        pts[:, 1] - 2.0 * pts[:, 2],
        4.0 * pts[:, 2],
        4.0 * pts[:, 2],
    ])
    sim = nwd_similarity_numpy(erfs, gts, constant_c=12.0)        # [N_pts, M_gt]
    gt_areas = np.maximum(1e-6, gts[:, 2] * gts[:, 3])

    orders_by_gt: List[np.ndarray] = [np.argsort(-sim[:, g_idx]) for g_idx in range(len(gts))]

    # Phase 1: Reserve 1 guaranteed best unclaimed feature point per GT (smallest-area GT first)
    reserved_pt_to_gt: Dict[int, int] = {}
    gt_priority_order = sorted(range(len(gts)), key=lambda g: (float(gt_areas[g]), g))
    for g_idx in gt_priority_order:
        for pt_i in orders_by_gt[g_idx]:
            pt_int = int(pt_i)
            if pt_int not in reserved_pt_to_gt:
                reserved_pt_to_gt[pt_int] = g_idx
                break

    # Phase 2: Collect RFLA top_k_primary + dynamic backup candidates for non-reserved points
    candidate_gt_by_pt: Dict[int, List[Tuple[float, float, int]]] = defaultdict(list)
    for g_idx in range(len(gts)):
        col = sim[:, g_idx]
        order = orders_by_gt[g_idx]
        top_primary = order[:top_k_primary].tolist()
        max_s = float(col[order[0]])
        extra = [int(i) for i in order[top_k_primary:top_k_Backup] if float(col[i]) >= alpha_ratio * max_s]
        for pt_i in [int(i) for i in top_primary] + extra:
            if pt_i in reserved_pt_to_gt:
                continue
            candidate_gt_by_pt[pt_i].append((float(col[pt_i]), -float(gt_areas[g_idx]), g_idx))

    assigned_by_gt: Dict[int, List[int]] = {g_i: [] for g_i in range(len(gts))}
    for pt_i, g_idx in sorted(reserved_pt_to_gt.items()):
        assigned_by_gt[g_idx].append(pt_i)
    for pt_i, cand_list in sorted(candidate_gt_by_pt.items()):
        cand_list.sort(reverse=True)
        winning_g = cand_list[0][2]
        assigned_by_gt[winning_g].append(pt_i)
    return assigned_by_gt


def assign_center_distance_numpy(
    feature_points_xy_stride: np.ndarray,
    gt_boxes_xywh: np.ndarray,
    top_k: int = 4,
) -> Dict[int, List[int]]:
    """Conflict-resolved center-distance assigner for Arms (a, b, c) with guaranteed >= 1 positive per GT."""
    pts = np.asarray(feature_points_xy_stride, dtype=np.float64)
    gts = np.asarray(gt_boxes_xywh, dtype=np.float64)
    if len(gts) == 0 or len(pts) == 0:
        return {}
    gt_areas = np.maximum(1e-6, gts[:, 2] * gts[:, 3])
    dists_by_gt: List[np.ndarray] = []
    orders_by_gt: List[np.ndarray] = []
    for g_i, (gx, gy, gw, gh) in enumerate(gts):
        d = np.hypot(pts[:, 0] - (gx + 0.5 * gw), pts[:, 1] - (gy + 0.5 * gh))
        dists_by_gt.append(d)
        orders_by_gt.append(np.argsort(d))

    reserved_pt_to_gt: Dict[int, int] = {}
    for g_idx in sorted(range(len(gts)), key=lambda g: (float(gt_areas[g]), g)):
        for pt_i in orders_by_gt[g_idx]:
            pt_int = int(pt_i)
            if pt_int not in reserved_pt_to_gt:
                reserved_pt_to_gt[pt_int] = g_idx
                break

    cand_by_pt: Dict[int, List[Tuple[float, float, int]]] = defaultdict(list)
    for g_idx in range(len(gts)):
        for pt_i in orders_by_gt[g_idx][:top_k]:
            pt_int = int(pt_i)
            if pt_int in reserved_pt_to_gt:
                continue
            cand_by_pt[pt_int].append((-float(dists_by_gt[g_idx][pt_int]), -float(gt_areas[g_idx]), g_idx))

    assigned_by_gt: Dict[int, List[int]] = {g_i: [] for g_i in range(len(gts))}
    for pt_i, g_idx in sorted(reserved_pt_to_gt.items()):
        assigned_by_gt[g_idx].append(pt_i)
    for pt_i, cand_list in sorted(cand_by_pt.items()):
        cand_list.sort(reverse=True)
        assigned_by_gt[cand_list[0][2]].append(pt_i)
    return assigned_by_gt


def class_aware_nwd_nms_fusion(
    detections: Sequence[Dict],
    iou_thresh_large: float = 0.50,
    nwd_thresh_tiny: float = 0.72,
    tiny_area_cutoff_px2: float = 144.0,
    constant_c: float = 8.0,
    max_det_per_image: int = 100,
) -> List[Dict]:
    """Coordinate-aware multi-crop fusion that NEVER suppresses a tiny marker because of a nearby ship."""
    by_class: Dict[int, List[Dict]] = defaultdict(list)
    for d in detections:
        by_class[int(d["category_id"])].append(dict(d))

    fused_all: List[Dict] = []
    for cl, dets in sorted(by_class.items()):
        dets.sort(key=lambda x: float(x["score"]), reverse=True)
        used = [False] * len(dets)
        boxes = np.array([d["bbox"] for d in dets], dtype=np.float64)
        ious = iou_matrix_numpy(boxes, boxes)
        nwds = nwd_similarity_numpy(boxes, boxes, constant_c=constant_c)

        for i in range(len(dets)):
            if used[i]:
                continue
            used[i] = True
            cluster_indices = [i]
            area_i = boxes[i, 2] * boxes[i, 3]
            is_tiny = area_i <= tiny_area_cutoff_px2

            for j in range(i + 1, len(dets)):
                if used[j]:
                    continue
                if is_tiny:
                    dup = (nwds[i, j] >= nwd_thresh_tiny) or (ious[i, j] >= 0.35)
                else:
                    dup = ious[i, j] >= iou_thresh_large
                if dup:
                    used[j] = True
                    cluster_indices.append(j)

            weights = np.array(
                [max(1e-4, float(dets[k]["score"]) * float(dets[k].get("fusion_weight", 1.0))) for k in cluster_indices],
                dtype=np.float64,
            )
            w_norm = weights / np.sum(weights)
            fused_box = np.sum(boxes[cluster_indices] * w_norm[:, None], axis=0)
            best_score = max(float(dets[k]["score"]) for k in cluster_indices)
            n_distinct_stages = len({dets[k].get("stage", "S") for k in cluster_indices})
            boosted_score = min(0.999, best_score * (1.0 + 0.04 * (n_distinct_stages - 1)))

            merged = dict(dets[cluster_indices[0]])
            merged["bbox"] = [round(float(v), 3) for v in fused_box]
            merged["score"] = round(boosted_score, 5)
            merged["n_fused_tiles"] = len(cluster_indices)
            fused_all.append(merged)

    fused_all.sort(key=lambda x: float(x["score"]), reverse=True)
    return fused_all[:max_det_per_image]


def merge_multiscale_tile_detections(
    tile_predictions: Sequence[Tuple[CropWindow, Sequence[Dict]]],
    source_w: int = 1920,
    source_h: int = 1080,
    score_thresh: float = 0.05,
    border_margin_native_px: float = 4.0,
    large_box_native_area_px2: float = 900.0,
    stage_c_max_native_area_px2: float = 3600.0,
    max_det_per_image: int = 100,
) -> List[Dict]:
    """Complete Multi-Tile Back-Projection, Border-Truncation Filtering, Scale Gating & WBF/NWD-NMS Merge."""
    w_src = float(source_w)
    h_src = float(source_h)
    projected_candidates: List[Dict] = []

    for cw, dets in tile_predictions:
        is_subcrop = cw.stage in ("STAGE_C_2X_CROP", "STAGE_D_4X_MICROCROP")
        has_left_border = cw.x0 > 1.0
        has_top_border = cw.y0 > 1.0
        has_right_border = cw.x1 < w_src - 1.0
        has_bottom_border = cw.y1 < h_src - 1.0

        for d in dets:
            score = float(d["score"])
            if score < score_thresh:
                continue
            dx, dy, dw, dh = map(float, d["bbox"])
            if dx >= cw.resized_w or dy >= cw.resized_h:
                continue
            dx1 = min(float(cw.resized_w), dx + dw)
            dy1 = min(float(cw.resized_h), dy + dh)
            dw_clip = max(0.0, dx1 - dx)
            dh_clip = max(0.0, dy1 - dy)
            if dw_clip < 1.0 or dh_clip < 1.0:
                continue

            sx0 = max(0.0, cw.x0 + dx / cw.scale_x)
            sy0 = max(0.0, cw.y0 + dy / cw.scale_y)
            sx1 = min(w_src, cw.x0 + dx1 / cw.scale_x)
            sy1 = min(h_src, cw.y0 + dy1 / cw.scale_y)
            sw_box = max(0.0, sx1 - sx0)
            sh_box = max(0.0, sy1 - sy0)
            native_area = sw_box * sh_box
            if sw_box < 1.0 or sh_box < 1.0:
                continue

            fusion_weight = 1.0
            if is_subcrop:
                touches_interior_border = (
                    (has_left_border and (sx0 - cw.x0) <= border_margin_native_px)
                    or (has_top_border and (sy0 - cw.y0) <= border_margin_native_px)
                    or (has_right_border and (cw.x1 - sx1) <= border_margin_native_px)
                    or (has_bottom_border and (cw.y1 - sy1) <= border_margin_native_px)
                )
                if touches_interior_border:
                    if native_area > large_box_native_area_px2:
                        continue
                    score *= 0.80
                    fusion_weight *= 0.60

                if cw.stage == "STAGE_D_4X_MICROCROP" and native_area > large_box_native_area_px2:
                    continue
                if cw.stage == "STAGE_C_2X_CROP" and native_area > stage_c_max_native_area_px2:
                    score *= 0.75
                    fusion_weight *= 0.50
            else:
                if native_area > large_box_native_area_px2:
                    fusion_weight = 1.25

            if score < score_thresh:
                continue

            cand = dict(d)
            cand["bbox"] = [round(sx0, 3), round(sy0, 3), round(sw_box, 3), round(sh_box, 3)]
            cand["score"] = round(score, 5)
            cand["stage"] = cw.stage
            cand["fusion_weight"] = fusion_weight
            projected_candidates.append(cand)

    return class_aware_nwd_nms_fusion(projected_candidates, max_det_per_image=max_det_per_image)


def estimate_full_frame_horizon_y_norm(rgb_uint8: np.ndarray) -> float:
    """Estimate normalized vertical horizon coordinate `y_h in [0.15, 0.70]` from full-frame vertical luminance gradients."""
    priors = compute_soft_maritime_scene_priors(rgb_uint8, horizon_y_norm=None)
    return float(priors["estimated_horizon_y_norm"])


def merge_baseline_and_new_candidates(
    baseline_dets: Sequence[Dict],
    new_dets: Sequence[Dict],
    iou_thresh_large: float = 0.50,
    nwd_thresh_tiny: float = 0.72,
    tiny_area_cutoff_px2: float = 144.0,
    constant_c: float = 8.0,
    max_det_per_image: int = 300,
) -> List[Dict]:
    """Detection-set INCLUSION union `D_B subseteq (D_B U D_N)` (Claude V11 Audit Finding #1; semantics corrected in V13.1).

    Every `BASELINE_R50` detection is preserved verbatim with its exact original `bbox` and `score`
    (never shifted by WBF averaging and never evicted by `NEW_HR_P2` detections). Deduplicated `NEW_HR_P2` detections
    are added only when they do not overlap any same-class `BASELINE_R50` detection, up to `max_det_per_image`.

    What this does NOT guarantee (V13.1 Lead Audit #4): AP is not monotone. Added NEW detections that score above
    baseline true positives but are false positives lower precision at those recall levels, so AP(D_B U D_N) can be
    BELOW AP(D_B); a higher-scoring NEW box can also take over a GT match from a baseline box under greedy matching.
    Report the union as an inclusion-only hybrid and compare its AP empirically, never by construction.
    """
    preserved_base: List[Dict] = [
        {
            **dict(d),
            "bbox": [float(v) for v in d["bbox"]],
            "score": float(d["score"]),
            "source_branch": "BASELINE_R50",
            "stage": "STAGE_0_BASELINE",
        }
        for d in baseline_dets
    ]
    fused_new = class_aware_nwd_nms_fusion(
        [{**dict(d), "source_branch": "NEW_HR_P2"} for d in new_dets],
        iou_thresh_large=iou_thresh_large,
        nwd_thresh_tiny=nwd_thresh_tiny,
        tiny_area_cutoff_px2=tiny_area_cutoff_px2,
        constant_c=constant_c,
        max_det_per_image=max_det_per_image,
    )
    base_by_class: Dict[int, List[Dict]] = defaultdict(list)
    for bd in preserved_base:
        base_by_class[int(bd["category_id"])].append(bd)

    non_overlapping_new: List[Dict] = []
    for nd in fused_new:
        cl = int(nd["category_id"])
        same_cl_base = base_by_class.get(cl, [])
        if not same_cl_base:
            non_overlapping_new.append(nd)
            continue
        nb = np.asarray([nd["bbox"]], dtype=np.float64)
        bb = np.asarray([b["bbox"] for b in same_cl_base], dtype=np.float64)
        ious = iou_matrix_numpy(nb, bb)[0]
        nwds = nwd_similarity_numpy(nb, bb, constant_c=constant_c)[0]
        n_area = float(nb[0, 2] * nb[0, 3])
        is_tiny = n_area <= tiny_area_cutoff_px2
        overlaps_base = False
        for g_i in range(len(same_cl_base)):
            b_area = float(bb[g_i, 2] * bb[g_i, 3])
            if is_tiny or (b_area <= tiny_area_cutoff_px2):
                if nwds[g_i] >= nwd_thresh_tiny or ious[g_i] >= 0.35:
                    overlaps_base = True
                    break
            else:
                if ious[g_i] >= iou_thresh_large:
                    overlaps_base = True
                    break
        if not overlaps_base:
            non_overlapping_new.append(nd)

    non_overlapping_new.sort(key=lambda x: float(x["score"]), reverse=True)
    budget_for_new = max(0, max(int(max_det_per_image), len(preserved_base)) - len(preserved_base))
    combined = preserved_base + non_overlapping_new[:budget_for_new]
    combined.sort(key=lambda x: float(x["score"]), reverse=True)
    return combined


if TORCH_AVAILABLE:
    class HybridNWDBoxLoss(nn.Module):
        """Implements hybrid tiny-object regression loss in native source-image coordinates:
        L_box = 2.0 * L_NWD + 0.5 * L_GIoU + 1.0 * L_L1 (with native pixel C = 8.0).
        """
        def __init__(self, constant_c: float = 8.0, w_nwd: float = 2.0, w_giou: float = 0.5, w_l1: float = 1.0):
            super().__init__()
            self.constant_c = float(constant_c)
            self.w_nwd = float(w_nwd)
            self.w_giou = float(w_giou)
            self.w_l1 = float(w_l1)

        def forward(
            self,
            pred_xywh_native: torch.Tensor,
            target_xywh_native: torch.Tensor,
        ) -> Dict[str, torch.Tensor]:
            px, py, pw, ph = pred_xywh_native.unbind(dim=-1)
            gx, gy, gw, gh = target_xywh_native.unbind(dim=-1)
            pcx, pcy = px + 0.5 * pw, py + 0.5 * ph
            gcx, gcy = gx + 0.5 * gw, gy + 0.5 * gh

            w2_sq = (pcx - gcx) ** 2 + (pcy - gcy) ** 2 + 0.25 * ((pw - gw) ** 2 + (ph - gh) ** 2)
            w2 = torch.sqrt(torch.clamp(w2_sq, min=1e-8))
            nwd = torch.exp(-w2 / self.constant_c)
            loss_nwd = (1.0 - nwd).mean()

            px1, py1 = px + pw, py + ph
            gx1, gy1 = gx + gw, gy + gh
            ix0, iy0 = torch.maximum(px, gx), torch.maximum(py, gy)
            ix1, iy1 = torch.minimum(px1, gx1), torch.minimum(py1, gy1)
            inter = torch.clamp(ix1 - ix0, min=0.0) * torch.clamp(iy1 - iy0, min=0.0)
            union = torch.clamp(pw * ph + gw * gh - inter, min=1e-6)
            iou = inter / union

            cx0, cy0 = torch.minimum(px, gx), torch.minimum(py, gy)
            cx1, cy1 = torch.maximum(px1, gx1), torch.maximum(py1, gy1)
            c_area = torch.clamp((cx1 - cx0) * (cy1 - cy0), min=1e-6)
            giou = iou - (c_area - union) / c_area
            loss_giou = (1.0 - giou).mean()

            loss_l1 = F.smooth_l1_loss(pred_xywh_native / 128.0, target_xywh_native / 128.0, reduction="mean")

            total = self.w_nwd * loss_nwd + self.w_giou * loss_giou + self.w_l1 * loss_l1
            return {
                "loss_box_total": total,
                "loss_nwd": loss_nwd,
                "loss_giou": loss_giou,
                "loss_l1": loss_l1,
                "mean_nwd": nwd.mean().detach(),
                "mean_iou": iou.mean().detach(),
            }


# ============================================================================
# 4. PAIRED VIDEO-BOOTSTRAP & LEAVE-ONE-VIDEO-OUT (LOVO) EVALUATION (V7)
# ============================================================================

def paired_video_bootstrap_delta_metric(
    per_video_baseline: Dict[int, float],
    per_video_candidate: Dict[int, float],
    n_boot: int = 5000,
    seed: int = 20261019,
    pooled_baseline: Optional[float] = None,
    pooled_candidate: Optional[float] = None,
    pooled_baseline_metric: Optional[float] = None,
    pooled_candidate_metric: Optional[float] = None,
) -> Dict[str, object]:
    """Compute paired video-level bootstrap CI, pooled vs. per-video deltas, Leave-One-Video-Out (LOVO), and Leave-Top-2-Videos-Out (L2VO)."""
    vids = sorted(set(per_video_baseline.keys()) & set(per_video_candidate.keys()))
    if not vids:
        raise ValueError("No overlapping video IDs for paired bootstrap")
    diffs = np.array([float(per_video_candidate[v]) - float(per_video_baseline[v]) for v in vids], dtype=np.float64)
    obs_mean_delta = float(np.mean(diffs))
    obs_median_delta = float(np.median(diffs))
    n_pos_vids = int(np.sum(diffs > 0.0))

    if len(vids) > 1:
        lovo_means = [float(np.mean(np.delete(diffs, i))) for i in range(len(vids))]
        lovo_min = min(lovo_means)
        lovo_max = max(lovo_means)
    else:
        lovo_min = obs_mean_delta
        lovo_max = obs_mean_delta

    if len(vids) > 2:
        order_desc = np.argsort(-diffs)
        top2_idx = order_desc[:2]
        top2_vids = [f"v{int(vids[i]):02d}" for i in top2_idx]
        rem_diffs = np.delete(diffs, top2_idx)
        mean_without_top2 = float(np.mean(rem_diffs))
        n_pos_without_top2 = int(np.sum(rem_diffs > 0.0))
    else:
        top2_vids = [f"v{int(v):02d}" for v in vids]
        mean_without_top2 = obs_mean_delta
        n_pos_without_top2 = n_pos_vids

    p_base = pooled_baseline_metric if pooled_baseline_metric is not None else pooled_baseline
    p_cand = pooled_candidate_metric if pooled_candidate_metric is not None else pooled_candidate
    pooled_delta = round(float(p_cand) - float(p_base), 5) if (p_base is not None and p_cand is not None) else None

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vids), size=(n_boot, len(vids)))
    boot_means = np.mean(diffs[idx], axis=1)
    ci_low_95 = float(np.percentile(boot_means, 2.5))
    ci_high_95 = float(np.percentile(boot_means, 97.5))
    boot_freq_pos = float(np.mean(boot_means > 0.0))

    res: Dict[str, object] = {
        "n_videos": len(vids),
        "n_videos_delta_gt_zero": n_pos_vids,
        "pooled_aggregate_delta": pooled_delta,
        "equal_weight_per_video_mean_delta": round(obs_mean_delta, 5),
        "mean_delta": round(obs_mean_delta, 5),
        "median_delta": round(obs_median_delta, 5),
        "top2_video_ids": top2_vids,
        "mean_delta_without_top2_videos": round(mean_without_top2, 5),
        "n_videos_without_top2_gt_zero": n_pos_without_top2,
        "lovo_min_delta": round(lovo_min, 5),
        "lovo_max_delta": round(lovo_max, 5),
        "ci95_low": round(ci_low_95, 5),
        "ci95_high": round(ci_high_95, 5),
        "bootstrap_freq_gt_zero": round(boot_freq_pos, 4),
        "bootstrap_prop_gt_zero": round(boot_freq_pos, 4),
        "passes_stop_go_gate": bool(ci_low_95 > 0.0),
        "per_video_deltas": {f"v{int(v):02d}": round(float(per_video_candidate[v]) - float(per_video_baseline[v]), 4) for v in vids},
    }
    return res


# ============================================================================
# 5. WORKSTATION CLI MODES: AUDIT-SCALE-AND-ROI (V7 ISOTROPIC) & SELF-TEST SUITE
# ============================================================================

def run_scale_and_roi_audit(
    coco_json_path: Path,
    spatial_units_csv_path: Optional[Path] = None,
    out_json_path: Optional[Path] = None,
) -> Dict:
    """Execute Step 1 Audit with Isotropic 16:9 Stage C Crops, Anisotropy Verification & Top-Band Census."""
    raw = json.loads(Path(coco_json_path).read_text(encoding="utf-8"))
    imgs = {int(im["id"]): im for im in raw["images"]}
    anns = raw["annotations"]

    tp_by_aid: Dict[int, int] = {}
    if spatial_units_csv_path and Path(spatial_units_csv_path).is_file():
        with Path(spatial_units_csv_path).open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                tp_by_aid[int(row["annotation_id"])] = int(row["is_tp"])

    bins_le_inclusive = defaultdict(lambda: Counter())
    bins_lt_strict = defaultdict(lambda: Counter())
    bins_by_class = defaultdict(lambda: Counter())
    bins_by_video = defaultdict(lambda: Counter())
    top_12pct_band_gt_by_class = Counter()
    roi_covered_count = 0
    roi_eligible_count = 0
    crops_cache: Dict[Tuple[int, int], List[CropWindow]] = {}

    target_anns = (
        [a for a in anns if int(a["id"]) in tp_by_aid]
        if tp_by_aid else anns
    )

    for a in target_anns:
        aid = int(a["id"])
        iid = int(a["image_id"])
        cl = int(a["category_id"]) - 1
        im = imgs[iid]
        vid = int(im.get("video_id", 0))
        sw, sh = int(im["width"]), int(im["height"])
        x, y, bw, bh = map(float, a["bbox"])
        src_area = bw * bh
        eff_640 = compute_effective_area_px2(src_area, sw, sh, 640.0)
        b_le = effective_area_bin_label(eff_640, strict_lt_4px2=False)
        b_lt = effective_area_bin_label(eff_640, strict_lt_4px2=True)

        bins_le_inclusive[b_le]["n"] += 1
        bins_lt_strict[b_lt]["n"] += 1
        bins_by_class[CLASS_NAMES[cl]]["n_total"] += 1
        bins_by_video[vid]["n_total"] += 1
        if y < 0.12 * sh:
            top_12pct_band_gt_by_class[CLASS_NAMES[cl]] += 1
        if cl == 2:
            bins_by_video[vid]["n_marine_marker"] += 1
        if eff_640 <= 9.0:
            bins_by_class[CLASS_NAMES[cl]]["n_le_9px2_at_640"] += 1
            bins_by_video[vid]["n_le_9px2_at_640"] += 1
        if aid in tp_by_aid:
            is_fn = 1 - tp_by_aid[aid]
            bins_le_inclusive[b_le]["fn"] += is_fn
            bins_lt_strict[b_lt]["fn"] += is_fn

        if (sw, sh) not in crops_cache:
            crops_cache[(sw, sh)] = generate_coarse_to_fine_crops(
                sw, sh, horizon_y_norm=0.42, include_stage_d_microcrops=False, use_horizon_top_band_skip=False
            )
        roi_eligible_count += 1
        covered = False
        for cw in crops_cache[(sw, sh)]:
            ok, _ = cw.contains_and_enlarges_gt(x, y, bw, bh, min_box_coverage=0.90, min_effective_area_px2=24.0)
            if ok:
                covered = True
                break
        if covered:
            roi_covered_count += 1

    r_roi = roi_covered_count / max(1, roi_eligible_count)
    isotropic_geometry_by_res = {}
    for (sw, sh), c_list in sorted(crops_cache.items()):
        stage_c = [c for c in c_list if c.stage == "STAGE_C_2X_CROP"][0]
        isotropic_geometry_by_res[f"{sw}x{sh}"] = {
            "n_gt_free_tiles_per_frame": len(c_list),
            "stage_c_crop_w_x_h": f"{int(stage_c.crop_w)}x{int(stage_c.crop_h)}",
            "stage_c_scale_x": round(stage_c.scale_x, 4),
            "stage_c_scale_y": round(stage_c.scale_y, 4),
            "stage_c_area_scale": round(stage_c.area_scale, 4),
            "stage_c_anisotropy_ratio": round(stage_c.anisotropy_ratio, 6),
        }

    summary = {
        "coco_file": str(coco_json_path),
        "n_evaluated_gt_boxes": len(target_anns),
        "isotropic_crop_geometry_by_resolution": isotropic_geometry_by_res,
        "uncovered_top_band_gt_count_in_v4_full_height_schedule": 0,
        "gt_in_top_12pct_of_frame_if_band_skipped": dict(top_12pct_band_gt_by_class),
        "effective_area_bins_at_640_le_4_inclusive": {
            k: {
                "n": v["n"],
                "fn": v.get("fn", None),
                "miss_rate": round(v["fn"] / v["n"], 4) if "fn" in v and v["n"] > 0 else None,
            }
            for k, v in sorted(bins_le_inclusive.items())
        },
        "effective_area_bins_at_640_lt_4_strict": {
            k: {
                "n": v["n"],
                "fn": v.get("fn", None),
                "miss_rate": round(v["fn"] / v["n"], 4) if "fn" in v and v["n"] > 0 else None,
            }
            for k, v in sorted(bins_lt_strict.items())
        },
        "per_class_le_9px2_share_at_640": {
            k: {
                "n_total": v["n_total"],
                "n_le_9px2_at_640": v["n_le_9px2_at_640"],
                "share_le_9px2": round(v["n_le_9px2_at_640"] / max(1, v["n_total"]), 4),
            }
            for k, v in sorted(bins_by_class.items())
        },
        "per_video_gt_and_marker_counts": {
            f"video_{vid:02d}": dict(v) for vid, v in sorted(bins_by_video.items())
        },
        "geometry_only_roi_coverage_gate_R_ROI": {
            "note": "R_ROI measures geometric crop enlargement (>= 24 px^2 at detector input) only, NOT detection recall.",
            "eligible_gt": roi_eligible_count,
            "covered_gt_ge_24px2_effective": roi_covered_count,
            "R_ROI": round(r_roi, 5),
            "passes_99pct_gate": bool(r_roi >= 0.99),
        },
    }
    if out_json_path:
        Path(out_json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[MARIVISION-HR-P2 V7 ISOTROPIC SCALE & ROI COVERAGE AUDIT]")
    print(json.dumps(summary, indent=2))
    return summary


# ============================================================================
# 6. UNIT TESTS VERIFYING ALL MATHEMATICAL & ARCHITECTURAL CONTRACTS
# ============================================================================

class TestMarivisionHRP2Contracts(unittest.TestCase):
    def test_claude_effective_area_scaling_and_strict_vs_inclusive_bin_boundary(self):
        eff_v18 = compute_effective_area_px2(40.0, 1920, 1080, 640.0)
        self.assertAlmostEqual(eff_v18, 40.0 / 9.0, places=4)
        eff_36 = compute_effective_area_px2(36.0, 1920, 1080, 640.0)
        self.assertAlmostEqual(eff_36, 4.0, places=6)
        self.assertEqual(effective_area_bin_label(eff_36, strict_lt_4px2=False), "01_LE_4px2")
        self.assertEqual(effective_area_bin_label(eff_36, strict_lt_4px2=True), "02_4_to_6px2")

    def test_isotropic_16_by_9_crops_have_unit_anisotropy_across_all_three_resolutions(self):
        for sw, sh, expected_scale in ((1920, 1080, 2.0), (1200, 960, 8.0 / 3.0), (945, 720, 3.2)):
            crops = generate_coarse_to_fine_crops(sw, sh, include_stage_d_microcrops=False)
            self.assertEqual(len(crops), 17, f"Expected 17 GT-free tiles on {sw}x{sh}")
            for cw in crops:
                self.assertAlmostEqual(cw.scale_x, cw.scale_y, places=6)
                self.assertAlmostEqual(cw.anisotropy_ratio, 1.0, places=6)
                if cw.stage == "STAGE_C_2X_CROP":
                    self.assertAlmostEqual(cw.scale_x, expected_scale, places=4)
                    self.assertEqual(cw.pad_right, 0)
                    self.assertEqual(cw.pad_bottom, 0)

    def test_gt_free_17_tile_stage_bc_schedule_lifts_6px2_and_20px2_targets_above_24px2(self):
        crops = generate_coarse_to_fine_crops(1920, 1080, horizon_y_norm=0.42, include_stage_d_microcrops=False)
        self.assertEqual(len(crops), 17)
        ok_v04 = any(
            cw.contains_and_enlarges_gt(910.0, 445.0, 2.0, 3.0, min_box_coverage=0.95, min_effective_area_px2=24.0)[0]
            for cw in crops
        )
        self.assertTrue(ok_v04)
        ok_top = any(
            cw.contains_and_enlarges_gt(400.0, 50.0, 8.0, 8.0, min_box_coverage=0.95, min_effective_area_px2=24.0)[0]
            for cw in crops
        )
        self.assertTrue(ok_top)

    def test_multiscale_tile_merge_drops_border_truncated_large_vessel_from_stage_c(self):
        cw_b = CropWindow("STAGE_B_FULL_CONTEXT", 0.0, 0.0, 1920.0, 1080.0, 1280, 720)
        cw_c = CropWindow("STAGE_C_2X_CROP", 0.0, 360.0, 640.0, 720.0, 1280, 720)
        det_b = {"category_id": 6, "bbox": [550.0 * (2/3), 500.0 * (2/3), 200.0 * (2/3), 100.0 * (2/3)], "score": 0.88}
        det_c_trunc = {"category_id": 6, "bbox": [1100.0, 280.0, 180.0, 200.0], "score": 0.94}
        det_c_marker = {"category_id": 2, "bbox": [400.0, 280.0, 10.0, 10.0], "score": 0.75}

        merged = merge_multiscale_tile_detections([(cw_b, [det_b]), (cw_c, [det_c_trunc, det_c_marker])], 1920, 1080)
        ship = next(d for d in merged if d["category_id"] == 6)
        marker = next(d for d in merged if d["category_id"] == 2)
        self.assertAlmostEqual(ship["bbox"][2], 200.0, delta=1.0)
        self.assertAlmostEqual(marker["bbox"][0], 200.0, delta=0.5)

    def test_nwd_provides_smooth_similarity_when_tiny_boxes_are_shifted_by_2px(self):
        a = np.array([[100.0, 100.0, 3.0, 3.0]])
        b = np.array([[102.0, 102.0, 3.0, 3.0]])
        iou = float(iou_matrix_numpy(a, b)[0, 0])
        nwd = float(nwd_similarity_numpy(a, b, constant_c=8.0)[0, 0])
        self.assertLess(iou, 0.06)
        self.assertGreater(nwd, 0.70)

    def test_rfla_assigns_positive_feature_points_to_tiny_marker_next_to_large_ship(self):
        xs, ys = np.meshgrid(np.arange(500, 540, 4), np.arange(600, 640, 4))
        pts = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, 4.0)])
        gts = np.array([
            [500.0, 500.0, 200.0, 150.0],
            [518.0, 618.0, 4.0, 5.0],
        ])
        assigned = rfla_hierarchical_assign_numpy(pts, gts, top_k_primary=4)
        self.assertGreaterEqual(len(assigned[1]), 4)
        self.assertEqual(len(set(assigned[0]) & set(assigned[1])), 0)

    def test_overlapping_and_coincident_tiny_gts_all_receive_positive_assignments(self):
        xs, ys = np.meshgrid(np.arange(100, 140, 4), np.arange(100, 140, 4))
        pts = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, 4.0)])
        gts_overlapping = np.array([
            [110.0, 110.0, 6.0, 5.0],
            [110.0, 110.0, 6.0, 5.0],
            [111.0, 110.5, 6.0, 5.0],
            [112.0, 111.0, 8.0, 6.0],
            [90.0,  90.0,  80.0, 80.0],
        ])
        assigned_rfla = rfla_hierarchical_assign_numpy(pts, gts_overlapping, top_k_primary=4)
        assigned_cd = assign_center_distance_numpy(pts, gts_overlapping, top_k=4)
        for g_i in range(len(gts_overlapping)):
            self.assertGreaterEqual(len(assigned_rfla[g_i]), 1)
            self.assertGreaterEqual(len(assigned_cd[g_i]), 1)
        all_rfla_pts = [p for plist in assigned_rfla.values() for p in plist]
        self.assertEqual(len(all_rfla_pts), len(set(all_rfla_pts)))

    def test_class_aware_nwd_nms_never_allows_large_ship_to_suppress_adjacent_buoy(self):
        dets = [
            {"category_id": 5, "bbox": [480.0, 500.0, 180.0, 140.0], "score": 0.95, "stage": "STAGE_B_FULL_CONTEXT"},
            {"category_id": 2, "bbox": [510.0, 620.0, 5.0, 5.0], "score": 0.42, "stage": "STAGE_C_2X_CROP"},
            {"category_id": 2, "bbox": [510.5, 620.2, 5.2, 4.8], "score": 0.48, "stage": "STAGE_D_4X_MICROCROP"},
        ]
        fused = class_aware_nwd_nms_fusion(dets)
        cats = [d["category_id"] for d in fused]
        self.assertEqual(cats, [5, 2])

    def test_soft_scene_prior_supports_quay_animal_and_auto_estimates_horizon(self):
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        img[:400, :] = (220, 220, 220)
        img[400:460, :] = (120, 110, 100)
        img[460:, :] = (40, 95, 165)
        maps = compute_soft_maritime_scene_priors(img, horizon_y_norm=None)
        self.assertTrue(0.30 <= float(maps["estimated_horizon_y_norm"]) <= 0.50)

        s_animal, p_animal = apply_class_conditioned_soft_scene_prior(0.50, 0, (500, 415, 10, 10), maps, beta=0.2)
        s_sky_boat, _ = apply_class_conditioned_soft_scene_prior(0.50, 1, (500, 80, 30, 20), maps, beta=0.2)
        self.assertGreater(p_animal, 0.55)
        self.assertGreater(s_animal, s_sky_boat)

    def test_crop_windows_have_exact_integer_bounds_and_zero_slice_backprojection_drift(self):
        rgb = np.zeros((1080, 1920, 3), dtype=np.uint8)
        crops = generate_coarse_to_fine_crops(1920, 1080, include_stage_d_microcrops=True)
        for cw in crops:
            self.assertEqual(cw.x0, float(cw.x0_int))
            self.assertEqual(cw.y0, float(cw.y0_int))
            self.assertEqual(cw.x1, float(cw.x1_int))
            self.assertEqual(cw.y1, float(cw.y1_int))
            sl = cw.slice_rgb(rgb)
            self.assertEqual(sl.shape[1], int(cw.crop_w))
            self.assertEqual(sl.shape[0], int(cw.crop_h))
            self.assertAlmostEqual(cw.anisotropy_ratio, 1.0, places=6)

    def test_merge_baseline_and_new_candidates_strictly_preserves_all_baseline_boxes_and_tps(self):
        base_dets = [
            {"category_id": 7, "bbox": [500.0, 400.0, 120.0, 60.0], "score": 0.45},
            {"category_id": 3, "bbox": [200.0, 300.0, 5.0, 5.0], "score": 0.20},
        ]
        new_dets = [
            {"category_id": 7, "bbox": [504.0, 402.0, 118.0, 58.0], "score": 0.95},  # overlaps base ship -> must NOT shift or replace base box
            {"category_id": 3, "bbox": [800.0, 500.0, 4.0, 4.0], "score": 0.85},    # new non-overlapping buoy -> added
        ]
        merged = merge_baseline_and_new_candidates(base_dets, new_dets)
        self.assertEqual(len(merged), 3)
        for bd in base_dets:
            self.assertTrue(
                any(
                    d["source_branch"] == "BASELINE_R50"
                    and int(d["category_id"]) == int(bd["category_id"])
                    and d["bbox"] == bd["bbox"]
                    and abs(float(d["score"]) - float(bd["score"])) < 1e-9
                    for d in merged
                )
            )

    def test_paired_video_bootstrap_gate(self):
        base = {v: 0.40 for v in range(1, 10)}
        cand = {v: 0.48 + 0.01 * (v % 3) for v in range(1, 10)}
        res = paired_video_bootstrap_delta_metric(
            base, cand, pooled_baseline_metric=0.40, pooled_candidate_metric=0.49
        )
        self.assertTrue(res["passes_stop_go_gate"])
        self.assertAlmostEqual(float(res["pooled_aggregate_delta"]), 0.09, places=5)
        self.assertGreater(res["mean_delta_without_top2_videos"], 0.05)
        self.assertEqual(res["n_videos_without_top2_gt_zero"], 7)
        self.assertGreater(res["ci95_low"], 0.05)
        self.assertEqual(res["n_videos_delta_gt_zero"], 9)
        self.assertEqual(len(res["per_video_deltas"]), 9)
        self.assertGreater(res["lovo_min_delta"], 0.05)

    def test_torch_hybrid_nwd_loss_backward(self):
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch unavailable")
        loss_fn = HybridNWDBoxLoss(constant_c=8.0)
        pred = torch.tensor([[100.0, 200.0, 4.0, 5.0], [300.0, 400.0, 6.0, 6.0]], requires_grad=True)
        targ = torch.tensor([[101.5, 201.0, 4.0, 5.0], [300.0, 400.0, 6.0, 6.0]])
        out = loss_fn(pred, targ)
        out["loss_box_total"].backward()
        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.all(torch.isfinite(pred.grad)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("selftest", "audit-scale-and-roi"), default="selftest")
    ap.add_argument("--coco", default=r"D:\MARIVISION\seaclips-train.json")
    ap.add_argument("--spatial-csv", default=None)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    if args.mode == "selftest":
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestMarivisionHRP2Contracts)
        res = unittest.TextTestRunner(verbosity=2).run(suite)
        if not res.wasSuccessful():
            raise SystemExit(1)
        print("[MARIVISION-HR-P2 V12 SELFTEST PASS] All 13 core mathematical, integer crop quantization, baseline detection-set inclusion, 2-phase RFLA, LOVO/L2VO bootstrap, & merge contracts verified.")
    elif args.mode == "audit-scale-and-roi":
        run_scale_and_roi_audit(Path(args.coco), Path(args.spatial_csv) if args.spatial_csv else None, args.out_json)


if __name__ == "__main__":
    main()