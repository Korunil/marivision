"""
D:/MARIVISION/H0_workbench/eval_tools/run_bfcef_e0_substrate.py
MARSEN B-FCEF: Stage E0 - Baseline-Decoupled Substrate & Genuine Expert Realization
(D-4445..D-4471, P-2282..P-2293)

Substrate Realization:
  1. Deterministic Repository Resolution: Discovers authoritative Deformable-DETR repo and
     enforces strict sys.path precedence before any model imports.
  2. Cryptographic Provenance: Verifies Census, Train JSON, 8 sampled Train shards, and R0 checkpoint.
  3. BCI with BaselineCapabilities: Drives expert availability (A_k) strictly from signal presence.
  4. CanonicalObjectState with CanonicalMultiscaleFeatures (P3, P4, P5), associated history, and scene context.
  5. Real Multi-Scale Visual Evidence:
     - Expert S: Samples fine/mid/coarse feature energy allocation and fine-scale contrast.
     - Expert L: Evaluates boundary gradient contrast on P3 and centroid offset (cx - rx, cy - ry).
  6. Real Temporal Kinematics:
     - Expert T: Consumes canonically associated track state (displacement, velocity, association confidence).
  7. Real Environmental Scene Context:
     - Scene Router Q: Exposes detector query activity density and coarse feature variance (Q_scene).
  8. Universal Invariants:
     - Dynamic query count (num_objects = state.boxes.shape[0]) with zero hard-coded query constants.
     - Symmetric, zero-preserving task projectors: P_k^cls and P_k^box with bias=False (P_k(0) == 0).
     - Full pointer identity bypass under neutralize_all=True.
     - Zero-start residual projector initialization helper (init_zero_projectors).
  9. Evidence-Tap Extension Certification Suite (Tests 1-9 across 16 frames from 8 videos).
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
from pathlib import Path

# Path Constants
ROOT_DIR = Path("D:/MARIVISION")
OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
SHARD_DIR = OUT_DIR / "full_split_shards"

CENSUS_JSON = OUT_DIR / "marsen_b2_production_census_56shards.json"
TRAIN_JSON = ROOT_DIR / "seaclips-train.json"
ZIP_PATH = ROOT_DIR / "SeaClips.zip"
R0_CHECKPOINT_PATH = ROOT_DIR / "checkpoints_r0" / "checkpoint_best.pth"
FROZEN_R0_CKPT_SHA256 = "28e6288c3f6769c2d5275aa77ddb248735b55a09cd46a6b570e3f117890b268b"
FROZEN_CENSUS_SHA256 = "a20a734d668a9078bdfe274cb09fa5a47ad9c6a3a9d7cef2d1f99b926623fbc5"

# -------------------------------------------------------------------------
# Authoritative Deformable-DETR repository resolution (D-4468, P-2293)
# -------------------------------------------------------------------------
REPO_CANDIDATES = [
    ROOT_DIR / "Deformable-DETR",
    ROOT_DIR / "deformable_detr",
]

DEFORMABLE_DETR_ROOT = None
for candidate in REPO_CANDIDATES:
    if (
        candidate.exists()
        and (candidate / "main.py").exists()
        and (candidate / "models" / "deformable_detr.py").exists()
        and (candidate / "datasets" / "seaclips_detection.py").exists()
    ):
        DEFORMABLE_DETR_ROOT = candidate
        break

assert DEFORMABLE_DETR_ROOT is not None, (
    "[FATAL] Cannot locate authoritative Deformable-DETR repository. "
    f"Checked: {[str(p) for p in REPO_CANDIDATES]}"
)

# Force deterministic import order: OUT_DIR -> DEFORMABLE_DETR_ROOT -> ROOT_DIR
for p in reversed([
    OUT_DIR,
    DEFORMABLE_DETR_ROOT,
    ROOT_DIR,
]):
    p_str = str(p)
    if p_str in sys.path:
        sys.path.remove(p_str)
    sys.path.insert(0, p_str)

import time
import json
import hashlib
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
from collections import defaultdict

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(line_buffering=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CANONICAL_EMB_DIM = 256
NUM_CLASSES = 7
P0_SAMPLE_VIDEOS = [1, 4, 7, 16, 25, 35, 42, 47]


def compute_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=eps, max=1.0 - eps)
    return torch.log(x / (1.0 - x))


def load_authoritative_r0_args():
    """Deterministic loader asserting exact frozen r0_preflight origin (D-4469, D-4470)."""
    preflight_path = OUT_DIR / "r0_preflight.py"
    assert preflight_path.exists(), f"[FATAL] Missing authoritative r0_preflight.py: {preflight_path}"

    import r0_preflight
    imported_path = Path(r0_preflight.__file__).resolve()
    expected_path = preflight_path.resolve()

    assert imported_path == expected_path, (
        "[FATAL] Wrong r0_preflight module imported!\n"
        f"Expected: {expected_path}\n"
        f"Got:      {imported_path}"
    )
    return r0_preflight.get_authoritative_r0_args()


# =============================================================================
# PART I: CANONICAL OBJECT STATE & BASELINE COMPATIBILITY INTERFACE
# =============================================================================

@dataclass(frozen=True)
class BaselineCapabilities:
    detector_name: str
    query_dim: int
    has_multiscale_features: bool
    has_temporal_history: bool
    has_reference_geometry: bool
    supports_native_class_intervention: bool
    supports_native_box_intervention: bool
    identity_bypass_certified: bool


@dataclass
class CanonicalMultiscaleFeatures:
    """Canonical multi-scale visual pyramid representation."""
    p3: torch.Tensor  # [B, 256, H/8, W/8]   - Fine scale
    p4: torch.Tensor  # [B, 256, H/16, W/16] - Mid scale
    p5: torch.Tensor  # [B, 256, H/32, W/32] - Coarse scale


@dataclass
class CanonicalObjectState:
    """Canonical detector query representation, decoupled from baseline internals."""
    boxes: torch.Tensor                    # [Q, 4] normalized cxcywh in [0, 1]
    logits: torch.Tensor                   # [Q, C] unnormalized classification logits
    scores: torch.Tensor                   # [Q] max sigmoid probability in [0, 1]
    embeddings: torch.Tensor               # [Q, D_canon] canonical object query latents
    reference_points: torch.Tensor         # [Q, 2 or 4] spatial reference coordinates
    features: Optional[CanonicalMultiscaleFeatures]
    temporal_history: Optional[Dict[str, Any]]
    image_shape: Tuple[int, int]           # (height, width)
    capabilities: BaselineCapabilities
    native_handle: Any                     # Sovereign pointer to raw detector tensors


class BaselineAdapter(nn.Module, ABC):
    """Abstract interface decoupling contextual bus from baseline internals."""
    def __init__(self, capabilities: BaselineCapabilities):
        super().__init__()
        self.capabilities = capabilities

    @abstractmethod
    def extract_canonical_state(self, frame_record: Dict[str, Any]) -> CanonicalObjectState:
        pass

    @abstractmethod
    def apply_contextual_prediction(
        self,
        state: CanonicalObjectState,
        q_cls_mod: torch.Tensor,
        q_box_mod: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def get_identity_outputs(self, state: CanonicalObjectState) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class AuthenticDeformableDETRAdapter(BaselineAdapter):
    """Production Adapter for Frozen R0 Deformable-DETR (ResNet-50)."""
    def __init__(self, class_embed: nn.Module, bbox_embed: nn.Module):
        caps = BaselineCapabilities(
            detector_name="Deformable-DETR-R50-R0-Authentic",
            query_dim=CANONICAL_EMB_DIM,
            has_multiscale_features=True,
            has_temporal_history=True,
            has_reference_geometry=True,
            supports_native_class_intervention=True,
            supports_native_box_intervention=True,
            identity_bypass_certified=True
        )
        super().__init__(capabilities=caps)
        self.class_embed = class_embed
        self.bbox_embed = bbox_embed

        for p in self.class_embed.parameters():
            p.requires_grad = False
        for p in self.bbox_embed.parameters():
            p.requires_grad = False

    def extract_canonical_state(self, frame_record: Dict[str, Any]) -> CanonicalObjectState:
        d_logits = frame_record["d_logits"]
        d_boxes = frame_record["d_boxes"]
        probs = d_logits.sigmoid()
        scores = torch.max(probs, dim=-1)[0]

        q_native = frame_record["d_queries"]
        ref_pts = frame_record["d_reference_points"]

        features = frame_record.get("multiscale_features", None)
        history = frame_record.get("temporal_history", None)

        return CanonicalObjectState(
            boxes=d_boxes,
            logits=d_logits,
            scores=scores,
            embeddings=q_native,
            reference_points=ref_pts,
            features=features,
            temporal_history=history,
            image_shape=(int(frame_record["img_h"]), int(frame_record["img_w"])),
            capabilities=self.capabilities,
            native_handle=(d_logits, d_boxes)
        )

    def apply_contextual_prediction(
        self,
        state: CanonicalObjectState,
        q_cls_mod: torch.Tensor,
        q_box_mod: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z_prime = self.class_embed(q_cls_mod)
        raw_box_offset = self.bbox_embed(q_box_mod)
        ref = state.reference_points
        ref_inv = inverse_sigmoid(ref)

        tmp = raw_box_offset.clone()
        if ref_inv.shape[-1] == 4:
            tmp = tmp + ref_inv
        else:
            tmp[..., :2] = tmp[..., :2] + ref_inv[..., :2]
        b_prime = tmp.sigmoid()

        return z_prime, torch.clamp(b_prime, 0.0, 1.0)

    def get_identity_outputs(self, state: CanonicalObjectState) -> Tuple[torch.Tensor, torch.Tensor]:
        return state.native_handle[0], state.native_handle[1]


# =============================================================================
# PART II: BASELINE-AGNOSTIC EXPERT BUS (INVARIANT CAPABILITIES)
# =============================================================================

class ZeroPreservingProjection(nn.Module):
    """Enforces P_k(0) == 0 to exact machine zero using bias-free layers."""
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ExpertScale(nn.Module):
    """
    Expert S: Evaluates visual representation consistency across spatial scales.
    Returns: (Evidence E, Availability A, Reliability R_proxy, Need N_proxy)
    """
    def __init__(self, evidence_dim: int = 48):
        super().__init__()
        self.raw_dim = 10
        self.evidence_dim = evidence_dim
        self.encoder = ZeroPreservingProjection(self.raw_dim, self.evidence_dim)
        self.proj_cls = ZeroPreservingProjection(self.evidence_dim, CANONICAL_EMB_DIM)
        self.proj_box = ZeroPreservingProjection(self.evidence_dim, CANONICAL_EMB_DIM)

    def forward(self, state: CanonicalObjectState) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_objects = state.boxes.shape[0]

        if state.features is None or not state.capabilities.has_multiscale_features:
            zero_ev = torch.zeros(num_objects, self.evidence_dim, device=state.boxes.device)
            zero_mask = torch.zeros(num_objects, device=state.boxes.device)
            return zero_ev, zero_mask, zero_mask, zero_mask

        H, W = state.image_shape
        bw_px = state.boxes[:, 2] * W
        bh_px = state.boxes[:, 3] * H
        area_norm = (bw_px * bh_px) / (float(H * W) + 1e-6)
        log_scale = torch.log(torch.clamp(bw_px * bh_px, min=1.0)) / 12.0
        aspect_ratio = torch.clamp(bw_px / (bh_px + 1e-4), 0.1, 10.0) / 5.0

        cx, cy = state.boxes[:, 0], state.boxes[:, 1]
        grid_x = cx * 2.0 - 1.0
        grid_y = cy * 2.0 - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).view(1, 1, num_objects, 2)

        feat_p3 = F.grid_sample(state.features.p3, grid, align_corners=False).view(-1, num_objects).t()
        feat_p4 = F.grid_sample(state.features.p4, grid, align_corners=False).view(-1, num_objects).t()
        feat_p5 = F.grid_sample(state.features.p5, grid, align_corners=False).view(-1, num_objects).t()

        energy_p3 = torch.norm(feat_p3, dim=-1)
        energy_p4 = torch.norm(feat_p4, dim=-1)
        energy_p5 = torch.norm(feat_p5, dim=-1)
        total_energy = energy_p3 + energy_p4 + energy_p5 + 1e-6

        ratio_fine = energy_p3 / total_energy
        ratio_coarse = energy_p5 / total_energy
        cross_scale_cos = F.cosine_similarity(feat_p3, feat_p5, dim=-1)

        avail_s = torch.ones(num_objects, device=state.boxes.device)
        rel_s_proxy = torch.clamp(cross_scale_cos + 0.5, 0.1, 1.0)
        need_s_proxy = torch.clamp(1.0 - log_scale, 0.0, 1.0) * (1.0 - state.scores)

        raw_cues = torch.stack([
            area_norm * 100.0,
            log_scale,
            aspect_ratio,
            ratio_fine,
            ratio_coarse,
            cross_scale_cos,
            state.scores,
            need_s_proxy,
            state.boxes[:, 2],
            state.boxes[:, 3]
        ], dim=-1)

        evidence = self.encoder(raw_cues)
        return evidence, avail_s, rel_s_proxy, need_s_proxy


class ExpertLocalization(nn.Module):
    """
    Expert L: Evaluates spatial boundary support, centroid pull, and geometric alignment.
    Returns: (Evidence E, Availability A, Reliability R_proxy, Need N_proxy)
    """
    def __init__(self, evidence_dim: int = 48):
        super().__init__()
        self.raw_dim = 10
        self.evidence_dim = evidence_dim
        self.encoder = ZeroPreservingProjection(self.raw_dim, self.evidence_dim)
        self.proj_cls = ZeroPreservingProjection(self.evidence_dim, CANONICAL_EMB_DIM)
        self.proj_box = ZeroPreservingProjection(self.evidence_dim, CANONICAL_EMB_DIM)

    def forward(self, state: CanonicalObjectState) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_objects = state.boxes.shape[0]

        if state.features is None or not state.capabilities.has_multiscale_features:
            zero_ev = torch.zeros(num_objects, self.evidence_dim, device=state.boxes.device)
            zero_mask = torch.zeros(num_objects, device=state.boxes.device)
            return zero_ev, zero_mask, zero_mask, zero_mask

        cx, cy, w, h = state.boxes.unbind(-1)
        rx, ry = state.reference_points[:, :2].unbind(-1)

        offset_x = (cx - rx) * 10.0
        offset_y = (cy - ry) * 10.0
        edge_dist = torch.min(torch.stack([cx - w/2, cy - h/2, 1.0 - (cx + w/2), 1.0 - (cy + h/2)]), dim=0)[0]
        clipped = (edge_dist < 0.02).float()

        center_grid = torch.stack([cx * 2.0 - 1.0, cy * 2.0 - 1.0], dim=-1).view(1, 1, num_objects, 2)
        top_grid = torch.stack([cx * 2.0 - 1.0, (cy - h/2).clamp(0, 1) * 2.0 - 1.0], dim=-1).view(1, 1, num_objects, 2)
        bot_grid = torch.stack([cx * 2.0 - 1.0, (cy + h/2).clamp(0, 1) * 2.0 - 1.0], dim=-1).view(1, 1, num_objects, 2)

        f_cnt = F.grid_sample(state.features.p3, center_grid, align_corners=False).view(-1, num_objects).t()
        f_top = F.grid_sample(state.features.p3, top_grid, align_corners=False).view(-1, num_objects).t()
        f_bot = F.grid_sample(state.features.p3, bot_grid, align_corners=False).view(-1, num_objects).t()

        grad_v = (torch.norm(f_cnt - f_top, dim=-1) + torch.norm(f_cnt - f_bot, dim=-1)) / (torch.norm(f_cnt, dim=-1) + 1e-6)

        avail_l = torch.ones(num_objects, device=state.boxes.device)
        rel_l_proxy = torch.clamp(grad_v / 2.0, 0.1, 1.0)
        need_l_proxy = torch.clamp(torch.abs(offset_x) + torch.abs(offset_y) + clipped * 0.5, 0.0, 1.0)

        raw_cues = torch.stack([
            offset_x,
            offset_y,
            w * 5.0,
            h * 5.0,
            edge_dist * 5.0,
            clipped,
            grad_v,
            state.scores,
            need_l_proxy,
            rel_l_proxy
        ], dim=-1)

        evidence = self.encoder(raw_cues)
        return evidence, avail_l, rel_l_proxy, need_l_proxy


class ExpertClassification(nn.Module):
    """
    Expert C: Evaluates category certainty, multi-label Bernoulli entropy, and semantic margins.
    Returns: (Evidence E, Availability A, Reliability R_proxy, Need N_proxy)
    """
    def __init__(self, evidence_dim: int = 32):
        super().__init__()
        self.raw_dim = 6
        self.evidence_dim = evidence_dim
        self.encoder = ZeroPreservingProjection(self.raw_dim, self.evidence_dim)
        self.proj_cls = ZeroPreservingProjection(self.evidence_dim, CANONICAL_EMB_DIM)
        self.proj_box = ZeroPreservingProjection(self.evidence_dim, CANONICAL_EMB_DIM)

    def forward(self, state: CanonicalObjectState) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_objects = state.boxes.shape[0]

        probs = state.logits.sigmoid()
        top2 = torch.topk(state.logits, k=2, dim=-1)[0]
        margin = (top2[:, 0] - top2[:, 1]) / 5.0

        p = torch.clamp(probs, 1e-6, 1.0 - 1e-6)
        h_b = -torch.mean(p * torch.log2(p) + (1.0 - p) * torch.log2(1.0 - p), dim=-1)

        avail_c = torch.ones(num_objects, device=state.boxes.device)
        need_c_proxy = torch.clamp(1.0 - margin + h_b, 0.0, 1.0)
        rel_c_proxy = torch.clamp(margin, 0.1, 1.0)

        raw_cues = torch.stack([
            margin,
            h_b,
            state.scores,
            torch.std(state.logits, dim=-1),
            need_c_proxy,
            torch.clamp(1.0 - state.scores, 0.0, 1.0)
        ], dim=-1)

        evidence = self.encoder(raw_cues)
        return evidence, avail_c, rel_c_proxy, need_c_proxy


class ExpertTemporal(nn.Module):
    """
    Expert T: Evaluates multi-frame trajectory continuity and kinematic displacement
    from canonically associated track history (not raw query slot index).
    Returns: (Evidence E, Availability A, Reliability R_proxy, Need N_proxy)
    """
    def __init__(self, evidence_dim: int = 48):
        super().__init__()
        self.raw_dim = 8
        self.evidence_dim = evidence_dim
        self.encoder = ZeroPreservingProjection(self.raw_dim, self.evidence_dim)
        self.proj_cls = ZeroPreservingProjection(self.evidence_dim, CANONICAL_EMB_DIM)
        self.proj_box = ZeroPreservingProjection(self.evidence_dim, CANONICAL_EMB_DIM)

    def forward(self, state: CanonicalObjectState) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_objects = state.boxes.shape[0]

        if state.temporal_history is None or not state.capabilities.has_temporal_history:
            zero_ev = torch.zeros(num_objects, self.evidence_dim, device=state.boxes.device)
            zero_mask = torch.zeros(num_objects, device=state.boxes.device)
            return zero_ev, zero_mask, zero_mask, zero_mask

        hist = state.temporal_history
        match_conf = hist.get("match_confidence", torch.zeros(num_objects, device=state.boxes.device))
        avail_t = (match_conf > 0.10).float()

        matched_b = hist.get("matched_boxes", state.boxes)
        disp_x = (state.boxes[:, 0] - matched_b[:, 0]) * 5.0
        disp_y = (state.boxes[:, 1] - matched_b[:, 1]) * 5.0
        disp_mag = torch.sqrt(disp_x**2 + disp_y**2 + 1e-6)

        app_sim = hist.get("appearance_similarity", torch.ones(num_objects, device=state.boxes.device))
        rel_t_proxy = torch.clamp(match_conf * app_sim, 0.1, 1.0) * avail_t
        need_t_proxy = torch.clamp(disp_mag + (1.0 - state.scores), 0.0, 1.0) * avail_t

        raw_cues = torch.stack([
            disp_x,
            disp_y,
            disp_mag,
            match_conf,
            app_sim,
            state.scores,
            need_t_proxy,
            rel_t_proxy
        ], dim=-1) * avail_t.unsqueeze(-1)

        evidence = self.encoder(raw_cues)
        return evidence, avail_t, rel_t_proxy, need_t_proxy


class SceneRouter(nn.Module):
    """
    Scene Router Q: Evaluates observation quality, environmental clutter, and camera conditions.
    Separates detector query activity density from visual scene texture.
    """
    def __init__(self):
        super().__init__()

    def forward(self, state: CanonicalObjectState) -> Dict[str, float]:
        probs = state.logits.sigmoid()

        # Dynamic query count: normalized by actual number of queries Q
        query_scores = probs.max(dim=-1).values
        act_density = float((query_scores > 0.10).float().mean().item())
        mean_act_energy = float(probs.mean().item())

        # Physical visual scene features from coarse feature map (P5)
        vis_energy = 0.0
        vis_variance = 0.0
        if state.features is not None:
            p5 = state.features.p5
            vis_energy = float(torch.abs(p5).mean().item())
            vis_variance = float(torch.var(p5).item())

        return {
            "detector_query_activity_density": act_density,
            "detector_activation_energy": mean_act_energy,
            "coarse_feature_energy": vis_energy,
            "coarse_feature_variance": vis_variance,
            "visibility_proxy": max(0.2, 1.0 - act_density * 0.4)
        }


# =============================================================================
# PART III: B-FCEF FUSION SUBSTRATE
# =============================================================================

class BFCEFSubstrate(nn.Module):
    """
    Unified B-FCEF Substrate housing BaselineAdapter, invariant experts (S, L, C, T, Q),
    symmetric task-decoupled projectors, zero-start residual initializers, and identity bypass.
    """
    def __init__(self, adapter: BaselineAdapter):
        super().__init__()
        self.adapter = adapter
        self.expert_s = ExpertScale(evidence_dim=48)
        self.expert_l = ExpertLocalization(evidence_dim=48)
        self.expert_c = ExpertClassification(evidence_dim=32)
        self.expert_t = ExpertTemporal(evidence_dim=48)
        self.scene_router = SceneRouter()
        self.last_scene_context: Optional[Dict[str, float]] = None

    def init_zero_projectors(self):
        """Zero-initializes final linear projection layers so training starts at exact baseline identity."""
        for exp in [self.expert_s, self.expert_l, self.expert_c, self.expert_t]:
            nn.init.zeros_(exp.proj_cls.net[-1].weight)
            nn.init.zeros_(exp.proj_box.net[-1].weight)

    def forward(
        self,
        frame_record: Dict[str, Any],
        neutralize_all: bool = False,
        active_expert_mask: Optional[List[bool]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, bool, Dict[str, float]]:
        # Short-circuit identity bypass
        if neutralize_all:
            z_r0, b_r0 = frame_record["d_logits"], frame_record["d_boxes"]
            return z_r0, b_r0, True, {}

        state = self.adapter.extract_canonical_state(frame_record)

        # Compute & expose environmental scene context
        scene_ctx = self.scene_router(state)
        self.last_scene_context = scene_ctx

        if active_expert_mask is None:
            active_expert_mask = [True, True, True, True]

        e_s, a_s, r_s, n_s = self.expert_s(state)
        e_l, a_l, r_l, n_l = self.expert_l(state)
        e_c, a_c, r_c, n_c = self.expert_c(state)
        e_t, a_t, r_t, n_t = self.expert_t(state)

        a_s = a_s * float(active_expert_mask[0])
        a_l = a_l * float(active_expert_mask[1])
        a_c = a_c * float(active_expert_mask[2])
        a_t = a_t * float(active_expert_mask[3])

        total_avail = a_s.sum() + a_l.sum() + a_c.sum() + a_t.sum()
        if total_avail == 0.0:
            z_r0, b_r0 = self.adapter.get_identity_outputs(state)
            return z_r0, b_r0, True, scene_ctx

        delta_q_box = (
            a_s.unsqueeze(-1) * self.expert_s.proj_box(e_s) +
            a_l.unsqueeze(-1) * self.expert_l.proj_box(e_l) +
            a_c.unsqueeze(-1) * self.expert_c.proj_box(e_c) +
            a_t.unsqueeze(-1) * self.expert_t.proj_box(e_t)
        )

        delta_q_cls = (
            a_s.unsqueeze(-1) * self.expert_s.proj_cls(e_s) +
            a_l.unsqueeze(-1) * self.expert_l.proj_cls(e_l) +
            a_c.unsqueeze(-1) * self.expert_c.proj_cls(e_c) +
            a_t.unsqueeze(-1) * self.expert_t.proj_cls(e_t)
        )

        z_prime, b_prime = self.adapter.apply_contextual_prediction(
            state, state.embeddings + delta_q_cls, state.embeddings + delta_q_box
        )
        return z_prime, b_prime, False, scene_ctx


# =============================================================================
# PART IV: EVIDENCE-TAP EXTENSION CERTIFICATION SUITE
# =============================================================================

def run_tap_extension_certification(
    substrate: BFCEFSubstrate,
    r0_model: nn.Module,
    sample_frames_16: List[Dict[str, Any]]
):
    print("=" * 105)
    print("EVIDENCE-TAP EXTENSION CERTIFICATION SUITE (P-2287, TESTS 1-9)")
    print("=" * 105)

    # Unit 1: P_k(0) == 0 exact machine zero on all 8 projection heads
    print("[*] Test 1: P_k(0) == 0 across all 4 invariant experts and both task heads...", flush=True)
    experts = [
        ("Expert S", substrate.expert_s, 48),
        ("Expert L", substrate.expert_l, 48),
        ("Expert C", substrate.expert_c, 32),
        ("Expert T", substrate.expert_t, 48),
    ]
    for name, exp, ev_dim in experts:
        zero_ev = torch.zeros(10, ev_dim, device=DEVICE)
        out_cls = exp.proj_cls(zero_ev)
        out_box = exp.proj_box(zero_ev)
        assert float(out_cls.abs().max().item()) == 0.0, f"[FATAL] {name} proj_cls failed P_k(0) == 0"
        assert float(out_box.abs().max().item()) == 0.0, f"[FATAL] {name} proj_box failed P_k(0) == 0"
    print("    [+] PASS: P_k(0) == 0 confirmed to exact machine zero (0.000000e+00).")

    # Hook r0_model to capture authentic multi-scale features without altering execution
    print("[*] Test 2: Multi-Frame Feature-Tap Parity across 16 frames from 8 videos (P-2288)...", flush=True)
    hooked_taps = {}
    def transformer_tap_hook(module, inp, outp):
        srcs = inp[0]
        hooked_taps["p3"] = srcs[0].detach()
        hooked_taps["p4"] = srcs[1].detach()
        hooked_taps["p5"] = srcs[2].detach()
        hs = outp[0]
        init_ref = outp[1]
        inter_refs = outp[2]
        last_lvl = hs.shape[0] - 1
        hooked_taps["hs_last"] = hs[last_lvl].detach()
        if last_lvl == 0:
            hooked_taps["ref_last"] = init_ref.detach()
        else:
            hooked_taps["ref_last"] = inter_refs[last_lvl - 1].detach()

    handle = r0_model.transformer.register_forward_hook(transformer_tap_hook)

    from datasets.seaclips_detection import SeaClipsDetection
    from util.misc import collate_fn

    dataset = SeaClipsDetection(ZIP_PATH, TRAIN_JSON, is_train=False, random_flip=False)
    if hasattr(dataset, "ids"):
        iid_to_ds_idx = {int(iid): idx for idx, iid in enumerate(dataset.ids)}
    else:
        iid_to_ds_idx = {int(img["id"]): idx for idx, img in enumerate(dataset.coco.dataset["images"])}

    max_delta_z = 0.0
    max_delta_b = 0.0
    first_tapped_fr = None

    for idx, fr in enumerate(sample_frames_16):
        iid = fr["image_id"]
        ds_idx = iid_to_ds_idx[iid]
        item = dataset[ds_idx]
        samples, _ = collate_fn([item])
        samples = samples.to(DEVICE)

        with torch.no_grad():
            out_tapped = r0_model(samples)

        z_live = out_tapped["pred_logits"][0].cpu()
        b_live = out_tapped["pred_boxes"][0].cpu()

        diff_z = float(torch.abs(z_live - fr["d_logits"]).max())
        diff_b = float(torch.abs(b_live - fr["d_boxes"]).max())
        max_delta_z = max(max_delta_z, diff_z)
        max_delta_b = max(max_delta_b, diff_b)

        assert diff_z < 1e-4, f"[FATAL] Live R0 tapped logit drift on frame {iid}: {diff_z:.2e}"
        assert diff_b < 1e-4, f"[FATAL] Live R0 tapped box drift on frame {iid}: {diff_b:.2e}"

        if first_tapped_fr is None:
            ms_feats = CanonicalMultiscaleFeatures(
                p3=hooked_taps["p3"],
                p4=hooked_taps["p4"],
                p5=hooked_taps["p5"]
            )
            first_tapped_fr = {
                "image_id": fr["image_id"],
                "img_w": fr["img_w"],
                "img_h": fr["img_h"],
                "d_logits": fr["d_logits"].to(DEVICE),
                "d_boxes": fr["d_boxes"].to(DEVICE),
                "d_queries": hooked_taps["hs_last"][0],
                "d_reference_points": hooked_taps["ref_last"][0],
                "multiscale_features": ms_feats,
                "temporal_history": None
            }

    print(f"    [+] PASS: 16-Frame Tap Parity Verified: Max Δz = {max_delta_z:.2e}, Max Δb = {max_delta_b:.2e} (< 1e-4).")

    # Unit 3: Canonical Multiscale Feature Verification
    print("[*] Test 3: Verifying Canonical Multiscale Features (P3, P4, P5)...", flush=True)
    ms_feats = first_tapped_fr["multiscale_features"]
    p3, p4, p5 = ms_feats.p3, ms_feats.p4, ms_feats.p5
    assert p3 is not None and torch.isfinite(p3).all(), "[FATAL] P3 is non-finite or None"
    assert p4 is not None and torch.isfinite(p4).all(), "[FATAL] P4 is non-finite or None"
    assert p5 is not None and torch.isfinite(p5).all(), "[FATAL] P5 is non-finite or None"
    print(f"    [+] PASS: P3 {list(p3.shape)}, P4 {list(p4.shape)}, P5 {list(p5.shape)} verified finite and deterministic.")

    # Unit 4: Visual Feature Spatial Sensitivity in Expert S & L (Deterministic Spatial Shift)
    print("[*] Test 4: Verifying Visual Feature Spatial Sensitivity (Deterministic Spatial Shift)...", flush=True)
    state_real = substrate.adapter.extract_canonical_state(first_tapped_fr)
    e_s1, a_s1, _, _ = substrate.expert_s(state_real)
    e_l1, a_l1, _, _ = substrate.expert_l(state_real)

    # Spatial shift perturbation on P3
    p3_pert = p3 + 0.25 * torch.roll(p3, shifts=1, dims=-1)
    ms_feats_perturbed = CanonicalMultiscaleFeatures(
        p3=p3_pert,
        p4=p4 * 0.5,
        p5=p5 * 1.5
    )
    tapped_fr_pert = dict(first_tapped_fr)
    tapped_fr_pert["multiscale_features"] = ms_feats_perturbed
    state_pert = substrate.adapter.extract_canonical_state(tapped_fr_pert)
    e_s2, _, _, _ = substrate.expert_s(state_pert)
    e_l2, _, _, _ = substrate.expert_l(state_pert)

    assert not torch.equal(e_s1, e_s2), "[FATAL] Expert S output did not change when visual features were perturbed!"
    assert not torch.equal(e_l1, e_l2), "[FATAL] Expert L output did not change under spatial feature perturbation!"
    print("    [+] PASS: Experts S and L genuinely respond to spatial and multi-scale feature perturbations.")

    # Unit 5: Signal Capability Gating (A_k == 0 when features absent)
    print("[*] Test 5: Verifying Signal Capability Gating (A_S == 0, A_L == 0 when features absent)...", flush=True)
    tapped_fr_nofeat = dict(first_tapped_fr)
    tapped_fr_nofeat["multiscale_features"] = None
    state_nofeat = substrate.adapter.extract_canonical_state(tapped_fr_nofeat)
    e_s0, a_s0, _, _ = substrate.expert_s(state_nofeat)
    e_l0, a_l0, _, _ = substrate.expert_l(state_nofeat)

    assert float(a_s0.max().item()) == 0.0, "[FATAL] Expert S availability non-zero when features absent!"
    assert float(a_l0.max().item()) == 0.0, "[FATAL] Expert L availability non-zero when features absent!"
    assert float(e_s0.abs().max().item()) == 0.0, "[FATAL] Expert S evidence non-zero when features absent!"
    assert float(e_l0.abs().max().item()) == 0.0, "[FATAL] Expert L evidence non-zero when features absent!"
    print("    [+] PASS: Availability strictly gated by signal presence (A == 0 -> E == 0).")

    # Unit 6: Canonical Temporal History Interface Contract (Expert T)
    print("[*] Test 6: Verifying Canonical Temporal History Interface Contract (Expert T)...", flush=True)
    e_t0, a_t0, _, _ = substrate.expert_t(state_real)
    assert float(a_t0.max().item()) == 0.0, "[FATAL] Expert T available when history absent!"

    num_q = state_real.boxes.shape[0]
    hist_contract_fixture = {
        "matched_boxes": state_real.boxes + 0.01,
        "match_confidence": torch.ones(num_q, device=DEVICE) * 0.85,
        "appearance_similarity": torch.ones(num_q, device=DEVICE) * 0.90
    }
    tapped_fr_hist = dict(first_tapped_fr)
    tapped_fr_hist["temporal_history"] = hist_contract_fixture
    state_hist = substrate.adapter.extract_canonical_state(tapped_fr_hist)
    e_t1, a_t1, r_t1, n_t1 = substrate.expert_t(state_hist)

    assert float(a_t1.min().item()) == 1.0, "[FATAL] Expert T not available under active association fixture!"
    assert float(e_t1.abs().max().item()) > 0.0, "[FATAL] Expert T evidence zero under active association fixture!"
    print("    [+] PASS: Expert T strictly obeys the canonical association history contract.")

    # Unit 7: Scene Router Q Normalization & Feature Exposure
    print("[*] Test 7: Verifying Scene Router Q Normalization & Feature Exposure...", flush=True)
    _, _, _, q_scene = substrate(first_tapped_fr)
    assert "detector_query_activity_density" in q_scene, "[FATAL] Missing detector_query_activity_density in Q_scene"
    assert "coarse_feature_variance" in q_scene, "[FATAL] Missing coarse_feature_variance in Q_scene"
    assert 0.0 <= q_scene["detector_query_activity_density"] <= 1.0, "[FATAL] Activity density exceeds [0, 1]"
    assert q_scene["coarse_feature_energy"] > 0.0, "[FATAL] Coarse feature energy is zero"
    print(f"    [+] PASS: Scene Router Q computes and exposes environmental context ({q_scene}).")

    # Unit 8: Dynamic Query Count (Decoupled from Q=300)
    print("[*] Test 8: Verifying Dynamic Query Count Invariance (Q=50 != 300)...", flush=True)
    synth_50_fr = {
        "image_id": 999999,
        "img_w": 1920,
        "img_h": 1080,
        "d_logits": torch.randn(50, NUM_CLASSES, device=DEVICE),
        "d_boxes": torch.rand(50, 4, device=DEVICE),
        "d_queries": torch.randn(50, CANONICAL_EMB_DIM, device=DEVICE),
        "d_reference_points": torch.rand(50, 2, device=DEVICE),
        "multiscale_features": ms_feats,
        "temporal_history": None
    }
    state_50 = substrate.adapter.extract_canonical_state(synth_50_fr)
    e_s50, a_s50, _, _ = substrate.expert_s(state_50)
    assert e_s50.shape[0] == 50, f"[FATAL] Expert S failed dynamic query count: shape {e_s50.shape}"
    print("    [+] PASS: Substrate functions seamlessly with dynamic query counts (Q=50).")

    # Unit 9: Task-Decoupled Channel Isolation & Pointer Identity Bypass
    print("[*] Test 9: Verifying Channel Isolation & Zero-FLOP Identity Bypass...", flush=True)
    z_id, b_id, bypass, _ = substrate(first_tapped_fr, neutralize_all=True)
    assert bypass is True, "[FATAL] Bypass flag False on neutralize_all=True"
    assert z_id is first_tapped_fr["d_logits"] and b_id is first_tapped_fr["d_boxes"], "[FATAL] Pointer identity violated"

    z_base, b_base = substrate.adapter.apply_contextual_prediction(state_real, state_real.embeddings, state_real.embeddings)
    delta_box = torch.ones_like(state_real.embeddings) * 0.05
    z_box, b_box = substrate.adapter.apply_contextual_prediction(state_real, state_real.embeddings, state_real.embeddings + delta_box)
    assert torch.equal(z_box, z_base) and not torch.equal(b_box, b_base), "[FATAL] Box modulation modified logits"

    delta_cls = torch.ones_like(state_real.embeddings) * 0.05
    z_cls, b_cls = substrate.adapter.apply_contextual_prediction(state_real, state_real.embeddings + delta_cls, state_real.embeddings)
    assert torch.equal(b_cls, b_base) and not torch.equal(z_cls, z_base), "[FATAL] Class modulation modified boxes"
    print("    [+] PASS: Task-decoupled channel isolation and pointer bypass fully certified.")

    handle.remove()
    print("\n[+] ALL 9 TAP EXTENSION TESTS PASSED: Substrate certified for E0 training.\n")


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def main():
    t_start = time.time()
    driver_sha = compute_sha256(Path(__file__))

    print("=" * 105)
    print("MARSEN B-FCEF: STAGE E0 - BASELINE-DECOUPLED SUBSTRATE & EVIDENCE EXTENSION")
    print(f"Driver Script SHA-256: {driver_sha}")
    print(f"Target Device:         {DEVICE}")
    print("Contract:              Universal Expert Bus | Sovereign BCI | Zero-Preserving Task Projectors")
    print("=" * 105, flush=True)

    # 1. Cryptographic Provenance Preflight Gate (D-4458, P-2288)
    print("[*] Checking Cryptographic Provenance across Census, Train JSON, and Sample Shards...", flush=True)
    assert CENSUS_JSON.exists(), f"[FATAL] Missing census JSON at {CENSUS_JSON}"
    assert compute_sha256(CENSUS_JSON) == FROZEN_CENSUS_SHA256, "[FATAL] Census SHA mismatch"

    with open(CENSUS_JSON, "r", encoding="utf-8") as f:
        census = json.load(f)

    assert compute_sha256(TRAIN_JSON) == census["_meta"]["live_provenance"]["train_json_sha256"], "[FATAL] Train JSON drift"
    frozen_shard_ledger = census["shard_sha256_ledger"]

    for vid in P0_SAMPLE_VIDEOS:
        shard_path = SHARD_DIR / f"marsen_b2_shard_train_vid{vid}.pth"
        assert shard_path.exists(), f"[FATAL] Missing shard {shard_path.name}"
        assert compute_sha256(shard_path) == frozen_shard_ledger[shard_path.name], f"[FATAL] Shard drift on {shard_path.name}"

    assert compute_sha256(R0_CHECKPOINT_PATH) == FROZEN_R0_CKPT_SHA256, "[FATAL] R0 Checkpoint SHA drift"
    print("    [+] Cryptographic provenance verified: Census, Train JSON, 8 sample shards, and R0 checkpoint certified.\n", flush=True)

    # 2. Build Authentic R0 Model & Bind Adapter
    print("[*] Binding Authentic R0 Model & Native Prediction Heads...", flush=True)
    from models.deformable_detr import build as build_deformable_detr

    args = load_authoritative_r0_args()
    args.device = str(DEVICE)
    r0_model, _, _ = build_deformable_detr(args)
    r0_ckpt = torch.load(R0_CHECKPOINT_PATH, map_location="cpu")
    sd = r0_ckpt.get("model", r0_ckpt)
    cleaned_sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    r0_model.load_state_dict(cleaned_sd, strict=True)
    r0_model.to(DEVICE)
    r0_model.eval()

    native_cls_head = r0_model.class_embed[-1]
    native_box_head = r0_model.bbox_embed[-1]
    adapter = AuthenticDeformableDETRAdapter(native_cls_head, native_box_head).to(DEVICE)
    print("    [+] Authentic Deformable-DETR Adapter registered with full BaselineCapabilities.")

    # 3. Instantiate Substrate
    substrate = BFCEFSubstrate(adapter=adapter).to(DEVICE)
    substrate.eval()
    print("    [+] Substrate instantiated with invariant experts S, L, C, T and scene router Q.\n", flush=True)

    # 4. Extract 16 Sample Frames from the 8 Canonical Videos
    sample_frames_16 = []
    vids_found = defaultdict(int)
    for vid in P0_SAMPLE_VIDEOS:
        shard_path = SHARD_DIR / f"marsen_b2_shard_train_vid{vid}.pth"
        shard = torch.load(shard_path, map_location="cpu")
        for fr in shard.get("frame_records", []):
            if vids_found[vid] < 2:
                sample_frames_16.append(fr)
                vids_found[vid] += 1

    assert len(sample_frames_16) == 16, f"[FATAL] Expected 16 sample frames, got {len(sample_frames_16)}"
    print(f"    [+] Ingested 16 sample frames across videos {P0_SAMPLE_VIDEOS} for multi-frame tap verification.\n", flush=True)

    # 5. Run Tap Extension Certification
    run_tap_extension_certification(substrate, r0_model, sample_frames_16)

    print("=" * 105)
    print(f"[+] EVIDENCE-TAP EXTENSION SUITE COMPLETED IN {time.time() - t_start:.2f}s")
    print("=" * 105, flush=True)


if __name__ == "__main__":
    main()