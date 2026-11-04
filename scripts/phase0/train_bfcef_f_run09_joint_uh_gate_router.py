from __future__ import annotations

"""
B-FCEF-F RUN_09B
================
Joint U/H -> task Gate -> Coordinator -> Simplex Router meta-fold pilot.

This is the first training run of the COMPLETE decision mechanism downstream
of the already-trained Final Experts.

Meta-crossfit rule
------------------
For meta-fold f:
  * Final Expert source for every video remains its honest RUN_07 OOF fold model.
  * Self-assessment/UH/Gates/Coordination are trained on videos NOT in f.
  * Integrated hard-gate evaluation is performed ONLY on videos in f.

Therefore gate/router evaluation never sees the local consequence labels of
its held-out meta-fold.

RUN_09B runs META-FOLD 0 only as a scientific pilot. Its checkpoint format is
the same one intended for the later full five-fold RUN_10.

Frozen:
  R0 native heads
  Final S/L/C/T expert parameters
  fold-specific Final Context Bus parameters

Trainable together:
  four OutcomeAwareSelfAssessmentHeads (including U/H aggregators)
  four cls sparse TSK gates
  four box sparse TSK gates
  cls/box CoordinationHeads

Router has no free parameters; it receives learned gate amplitudes and learned
coordination logits.

Training:
  Phase W: one consequence/UH warm-up epoch
  Phase J: two joint epochs including Gate, Router-target and integrated DETR loss

Evaluation:
  exact hard masks, exact baseline preservation outside active queries.
"""

import argparse
import copy
import gc
import hashlib
import io
import json
import math
import os
import random
import sys
import time
import contextlib
from PIL import Image
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("R0_MSDA_BACKEND", "pytorch")

import numpy as np
import torch
import torch.nn.functional as F


# =============================================================================
# PATHS / ROOTS
# =============================================================================

ROOT = Path("D:/MARIVISION")
EVAL = ROOT / "H0_workbench" / "eval_tools"
E0 = EVAL / "e0_cache"
SPATIAL_DIR = EVAL / "bfcef_f_derived_cache" / "spatial_v2"
TEMP_DIR = EVAL / "bfcef_f_derived_cache" / "temporal_kmax7_v1"
FOLD_DIR = EVAL / "e0_adapters"
S0A = EVAL / "bfcef_f_s0" / "final_expert_crossfit_v1"
RUN09A = EVAL / "bfcef_f_s0" / "run09_final_local_consequences_v1"
OUT = EVAL / "bfcef_f_s0" / "run09_joint_uh_gate_router_pilot_v1"

TRAIN_JSON = ROOT / "seaclips-train.json"
CONSEQUENCE_MANIFEST = RUN09A / "run09_final_consequence_manifest.json"
OOF_MANIFEST = S0A / "final_expert_oof_manifest.json"

EXPECTED_OOF_SHA = "1c2569f1871b475e4aa71d20e505c8bb738d9b3ed7d4df185787c1c5546c6001"
NUM_CLASSES = 7
NUM_QUERIES = 300
NUM_FOLDS = 5
EXPERTS = ("S", "L", "C", "T")
META_FOLD = 0
K_S0 = 3

WARMUP_EPOCHS = 1
JOINT_EPOCHS = 2

LR_SA = 2.0e-4
LR_GATE = 1.0e-4
WEIGHT_DECAY = 1.0e-4
GRAD_CLIP = 0.1
SEED = 20261009

LAMBDA_UH = 0.50
LAMBDA_GATE = 0.50
LAMBDA_ROUTE = 0.25
LAMBDA_DET = 0.25

SCORE_THRESHOLD = 0.20
IOU_THRESHOLD = 0.30
TOPK = 100

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# =============================================================================
# GENERIC
# =============================================================================

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# IMPORT AUTHORITATIVE RUN_07 HELPERS
# =============================================================================

if str(EVAL) not in sys.path:
    sys.path.insert(0, str(EVAL))

import train_bfcef_f_s0_final_experts as s0a  # noqa: E402


def initialize_runtime():
    # This hard-checks the exact corrected B-FCEF-F source SHAs that RUN_07
    # used, plus frozen cache/config roots.
    s0a.verify_source_and_roots()

    (
        BFCEFFScreeningConfig,
        CanonicalObjectState,
        BFCEFFModel,
        frame_inputs_from_records,
        get_authoritative_r0_args,
        build_deformable_detr,
        box_ops,
    ) = s0a.prepare_imports()

    config = BFCEFFScreeningConfig.from_json(s0a.SCREENING_CONFIG)

    native_cls, native_box, criterion = s0a.build_native_templates_and_criterion(
        get_authoritative_r0_args,
        build_deformable_detr,
    )

    model = s0a.construct_model(
        native_cls,
        native_box,
        config,
        BFCEFFModel,
    )

    # s0a.construct_model intentionally enables experts/context for S0-A.
    # RUN_09B freezes EVERYTHING, then re-enables only the downstream mechanism.
    for p in model.parameters():
        p.requires_grad_(False)

    for module in (
        model.self_assessment,
        model.gate_cls,
        model.gate_box,
        model.coord_cls,
        model.coord_box,
    ):
        for p in module.parameters():
            p.requires_grad_(True)

    if any(p.requires_grad for p in model.adapter.parameters()):
        raise AssertionError("[FATAL] Native R0 heads are trainable in RUN_09B.")

    for module in (
        model.expert_s,
        model.expert_l,
        model.expert_c,
        model.expert_t,
        model.context_bus,
    ):
        if any(p.requires_grad for p in module.parameters()):
            raise AssertionError("[FATAL] Final Expert/Context parameter is trainable in RUN_09B.")

    return (
        model,
        criterion,
        CanonicalObjectState,
        frame_inputs_from_records,
    )


# =============================================================================
# FOLD / DATA
# =============================================================================

def parse_folds() -> Dict[int, List[int]]:
    payload = load_json(s0a.FOLD_MANIFEST)
    return {int(k): [int(v) for v in vals] for k, vals in payload["folds"].items()}


def video_fold_map(folds: Dict[int, List[int]]) -> Dict[int, int]:
    return {vid: f for f, vids in folds.items() for vid in vids}


def consequence_path(fold: int, vid: int) -> Path:
    return RUN09A / f"fold{fold}" / f"bfcef_f_run09_consequence_vid{vid}.pth"


def load_video_all(vid: int, fold: int):
    canonical, spatial, temporal = s0a.load_video_triplet(vid)
    labels_payload = torch.load(consequence_path(fold, vid), map_location="cpu")
    labels = labels_payload["records"]
    if not (len(canonical) == len(spatial) == len(temporal) == len(labels)):
        raise AssertionError(f"[FATAL] RUN_09B video alignment length drift vid={vid}")
    return canonical, spatial, temporal, labels


def load_source_fold(model, source_fold: int) -> None:
    payload = torch.load(s0a.fold_checkpoint_path(source_fold), map_location="cpu")
    model.expert_s.load_state_dict(payload["expert_s"], strict=True)
    model.expert_l.load_state_dict(payload["expert_l"], strict=True)
    model.expert_c.load_state_dict(payload["expert_c"], strict=True)
    model.expert_t.load_state_dict(payload["expert_t"], strict=True)
    model.context_bus.load_state_dict(payload["context_bus"], strict=True)

    # Keep frozen Final Experts deterministic even while downstream modules train.
    model.expert_s.eval()
    model.expert_l.eval()
    model.expert_c.eval()
    model.expert_t.eval()
    model.context_bus.eval()


# =============================================================================
# EXACT FINAL-EXPERT FEATURES + RESIDUALS
# =============================================================================

def expert_bundle(model, O, X):
    with torch.no_grad():
        C_t = model.context_bus(
            m_ext=None,
            m_hat_vis=None,
            scene_q=X.scene_q,
            m_avail=None,
            mode="auto",
        )

        dS_cls, dS_box, fS = model.expert_s(
            O.h, X.cues_s, O.b, X.scale_spatial, C_t
        )
        dL_cls, dL_box, fL, L_expected_iou = model.expert_l(
            O.h, X.cues_l, O.b, O.r, X.loc_spatial, C_t
        )
        dC_cls, dC_box, fC = model.expert_c(
            O.h, X.cues_c, O.z, O.b, C_t
        )
        dT_cls, dT_box, fT = model.expert_t(
            O.h, X.temporal, C_t
        )

        delta_cls = torch.stack([dS_cls, dL_cls, dC_cls, dT_cls], dim=1)
        delta_box = torch.stack([dS_box, dL_box, dC_box, dT_box], dim=1)

        candidate_set = model.bci.extract(O.z)
        candidate_mask = torch.zeros(
            NUM_QUERIES, 1, device=O.h.device, dtype=O.h.dtype
        )
        if candidate_set.queries:
            candidate_mask[candidate_set.queries] = 1.0

        delta_cls = delta_cls * candidate_mask.unsqueeze(1)
        delta_box = delta_box * candidate_mask.unsqueeze(1)

    return {
        "C_t": C_t,
        "features": {"S": fS, "L": fL, "C": fC, "T": fT},
        "delta_cls": delta_cls,
        "delta_box": delta_box,
        "L_expected_iou": L_expected_iou,
        "candidate_set": candidate_set,
        "candidate_mask": candidate_mask,
    }


# =============================================================================
# SELF-ASSESSMENT + GATES + ROUTER
# =============================================================================

def mechanism_forward(model, O, X, bundle, hard: bool):
    C_t = bundle["C_t"]
    delta_cls = bundle["delta_cls"]
    delta_box = bundle["delta_box"]
    features = bundle["features"]
    L_expected_iou = bundle["L_expected_iou"]
    candidate_set = bundle["candidate_set"]
    candidate_mask = bundle["candidate_mask"]

    q_count = O.h.shape[0]
    dtype = O.h.dtype
    device = O.h.device

    # Availability is candidate membership for all permanent task-capable experts.
    A_cls = candidate_mask.expand(-1, 4)
    A_box = candidate_mask.expand(-1, 4)

    cls_raw = {}
    box_raw = {}
    cls_channels = {}
    box_channels = {}

    for k, expert in enumerate(EXPERTS):
        N = X.raw_proxies[expert]["N"].reshape(q_count, 1)
        R = X.raw_proxies[expert]["R"].reshape(q_count, 1)
        ncls = delta_cls[:, k].norm(dim=-1, keepdim=True)
        nbox = delta_box[:, k].norm(dim=-1, keepdim=True)

        out = model.self_assessment[expert](
            features[expert],
            O.h,
            N,
            R,
            ncls,
            nbox,
            C_t,
        )

        if not isinstance(out, tuple) or len(out) != 4:
            raise AssertionError(
                "[FATAL] Corrected self-assessment four-output contract drift."
            )

        cp, br, cc, bc = out
        cls_raw[expert] = cp
        box_raw[expert] = br
        cls_channels[expert] = cc
        box_channels[expert] = bc

    probability = O.z.sigmoid().clamp(1e-6, 1 - 1e-6)
    sorted_probability, _ = torch.sort(probability, dim=-1, descending=True)
    score = sorted_probability[:, 0:1]
    margin = sorted_probability[:, 0:1] - sorted_probability[:, 1:2]
    bern = -(
        probability * probability.log()
        + (1 - probability) * (1 - probability).log()
    ).mean(dim=-1, keepdim=True) / math.log(2.0)
    area = (O.b[:, 2:3] * O.b[:, 3:4]).clamp_min(1e-8)
    log_area = area.log()
    aspect = O.b[:, 2:3] / O.b[:, 3:4].clamp_min(1e-4)
    candidate_rank = candidate_set.query_rank_norm
    Cq = C_t.expand(q_count, -1)

    gate_cls_features = []
    gate_box_features = []
    cls_gate_out = []
    box_gate_out = []

    for k, expert in enumerate(EXPERTS):
        N = X.raw_proxies[expert]["N"].reshape(q_count, 1)
        R = X.raw_proxies[expert]["R"].reshape(q_count, 1)
        ncls = delta_cls[:, k].norm(dim=-1, keepdim=True)
        nbox = delta_box[:, k].norm(dim=-1, keepdim=True)
        cc = cls_channels[expert]
        bc = box_channels[expert]

        cls_quality = torch.zeros(q_count, 1, device=device, dtype=dtype)
        box_quality = (
            L_expected_iou
            if expert == "L"
            else torch.zeros(q_count, 1, device=device, dtype=dtype)
        )

        gf_cls = torch.cat([
            score, margin, bern, log_area, aspect, candidate_rank,
            N, R,
            cc["p_rescue"], cc["p_loss"], cc["p_prune"], cc["p_add"], cc["p_preserve"],
            cc["U"], cc["H"], cc["sigma"],
            ncls, cls_quality, Cq,
        ], dim=-1)

        gf_box = torch.cat([
            score, margin, bern, log_area, aspect, candidate_rank,
            N, R,
            bc["p_rescue"], bc["p_loss"], bc["p_retain"], bc["exp_delta_iou"],
            bc["U"], bc["H"], bc["sigma"],
            nbox, box_quality,
            torch.zeros(q_count, 1, device=device, dtype=dtype),
            Cq,
        ], dim=-1)

        if gf_cls.shape[-1] != model.gate_dim or gf_box.shape[-1] != model.gate_dim:
            raise AssertionError("[FATAL] Gate feature dimension drift.")

        gate_cls_features.append(gf_cls)
        gate_box_features.append(gf_box)
        cls_gate_out.append(model.gate_cls[expert](gf_cls))
        box_gate_out.append(model.gate_box[expert](gf_box))

    hard_cls = torch.cat([x["mask_hard"] for x in cls_gate_out], dim=-1) * A_cls
    hard_box = torch.cat([x["mask_hard"] for x in box_gate_out], dim=-1) * A_box

    if hard:
        g_cls = torch.cat([x["g_hard"] for x in cls_gate_out], dim=-1) * A_cls
        g_box = torch.cat([x["g_hard"] for x in box_gate_out], dim=-1) * A_box
    else:
        g_cls = torch.cat([x["g_soft"] for x in cls_gate_out], dim=-1) * A_cls
        g_box = torch.cat([x["g_soft"] for x in box_gate_out], dim=-1) * A_box

    coord_in_cls = torch.cat(gate_cls_features + [g_cls, hard_cls, A_cls], dim=-1)
    coord_in_box = torch.cat(gate_box_features + [g_box, hard_box, A_box], dim=-1)

    coord_cls = model.coord_cls(coord_in_cls, delta_cls)
    coord_box = model.coord_box(coord_in_box, delta_box)

    pi_cls = model.router(g_cls, A_cls, coord_cls)
    pi_box = model.router(g_box, A_box, coord_box)

    h_cls = O.h + torch.sum(pi_cls[:, 1:].unsqueeze(-1) * delta_cls, dim=1)
    h_box = O.h + torch.sum(pi_box[:, 1:].unsqueeze(-1) * delta_box, dim=1)

    z = O.z.clone()
    b = O.b.clone()

    if hard:
        active_cls = hard_cls.sum(dim=-1) > 0
        active_box = hard_box.sum(dim=-1) > 0
    else:
        active_cls = A_cls.sum(dim=-1) > 0
        active_box = A_box.sum(dim=-1) > 0

    if bool(active_cls.any()):
        z[active_cls] = model.adapter.decode_cls(h_cls[active_cls])
    if bool(active_box.any()):
        b[active_box] = model.adapter.decode_box(h_box[active_box], O.r[active_box])

    return {
        "z": z,
        "b": b,
        "cls_raw": cls_raw,
        "box_raw": box_raw,
        "cls_channels": cls_channels,
        "box_channels": box_channels,
        "g_cls": g_cls,
        "g_box": g_box,
        "hard_cls": hard_cls,
        "hard_box": hard_box,
        "pi_cls": pi_cls,
        "pi_box": pi_box,
        "A_cls": A_cls,
        "A_box": A_box,
    }


# =============================================================================
# LABEL WEIGHTS / TARGETS
# =============================================================================

def aggregate_training_counts(manifest: dict, train_videos: List[int]) -> dict:
    counts = {
        e: {
            "cls": torch.zeros(9, dtype=torch.float64),
            "box_pos": torch.zeros(3, dtype=torch.float64),
            "box_total": 0.0,
        }
        for e in EXPERTS
    }

    for vid in train_videos:
        summ = manifest["per_video"][str(vid)]["summary"]
        for e in EXPERTS:
            for s, n in summ[e]["cls_transition_counts"].items():
                counts[e]["cls"][int(s)] += int(n)
            bc = summ[e]["box_counts"]
            counts[e]["box_pos"][0] += int(bc.get("rescue", 0))
            counts[e]["box_pos"][1] += int(bc.get("loss", 0))
            counts[e]["box_pos"][2] += int(bc.get("retain", 0))
            # every candidate query contributes one label per box channel
            counts[e]["box_total"] += (
                sum(int(n) for n in summ[e]["cls_transition_counts"].values())
                / NUM_CLASSES
            )

    weights = {}
    for e in EXPERTS:
        cls_count = counts[e]["cls"].clone()
        nonzero = cls_count > 0
        w = torch.zeros_like(cls_count)
        if bool(nonzero.any()):
            w[nonzero] = torch.sqrt(cls_count[nonzero].sum() / cls_count[nonzero])
            w[nonzero] /= w[nonzero].mean()
        pos = counts[e]["box_pos"]
        total = max(1.0, counts[e]["box_total"])
        neg = torch.clamp(torch.tensor(total, dtype=torch.float64) - pos, min=1.0)
        pos_weight = torch.sqrt(neg / torch.clamp(pos, min=1.0))
        weights[e] = {
            "cls": w.float().to(DEVICE),
            "box_pos": pos_weight.float().to(DEVICE),
        }
    return weights


def targets_from_label(label_fr: dict, expert: str, q_idx: torch.Tensor):
    ex = label_fr["experts"][expert]
    tr = ex["cls_transition"].long().to(DEVICE)
    u_cls = ex["u_cls_target"].float().to(DEVICE)
    h_cls = ex["h_cls_target"].float().to(DEVICE)

    box_y = torch.cat([
        ex["box_rescue_target"],
        ex["box_loss_target"],
        ex["box_retain_target"],
    ], dim=-1).float().to(DEVICE)
    d_iou = ex["box_delta_iou_target"].float().to(DEVICE)
    u_box = ex["u_box_target"].float().to(DEVICE)
    h_box = ex["h_box_target"].float().to(DEVICE)

    return tr, u_cls, h_cls, box_y, d_iou, u_box, h_box


# =============================================================================
# LOSS
# =============================================================================

def consequence_loss(model_out, label_fr, q_idx, weights):
    total_cls = torch.zeros((), device=DEVICE)
    total_box = torch.zeros((), device=DEVICE)
    total_uh = torch.zeros((), device=DEVICE)

    true_u_cls = []
    true_h_cls = []
    true_u_box = []
    true_h_box = []

    for e in EXPERTS:
        tr, ucls, hcls, box_y, d_iou, ubox, hbox = targets_from_label(
            label_fr, e, q_idx
        )

        pred_post = model_out["cls_raw"][e][q_idx]
        logp = torch.log(pred_post.clamp_min(1e-8))
        total_cls = total_cls + F.nll_loss(
            logp.reshape(-1, 9),
            tr.reshape(-1),
            weight=weights[e]["cls"],
        )

        raw = model_out["box_raw"][e][q_idx]
        total_box = total_box + F.binary_cross_entropy_with_logits(
            raw[:, 0:3],
            box_y,
            pos_weight=weights[e]["box_pos"],
        )
        total_box = total_box + F.smooth_l1_loss(
            torch.tanh(raw[:, 3:4]),
            d_iou,
        )

        cc = model_out["cls_channels"][e]
        bc = model_out["box_channels"][e]

        total_uh = total_uh + F.binary_cross_entropy(
            cc["U"][q_idx].clamp(1e-6, 1 - 1e-6), ucls
        )
        total_uh = total_uh + F.binary_cross_entropy(
            cc["H"][q_idx].clamp(1e-6, 1 - 1e-6), hcls
        )
        total_uh = total_uh + F.binary_cross_entropy(
            bc["U"][q_idx].clamp(1e-6, 1 - 1e-6), ubox
        )
        total_uh = total_uh + F.binary_cross_entropy(
            bc["H"][q_idx].clamp(1e-6, 1 - 1e-6), hbox
        )

        true_u_cls.append(ucls)
        true_h_cls.append(hcls)
        true_u_box.append(ubox)
        true_h_box.append(hbox)

    total_cls = total_cls / len(EXPERTS)
    total_box = total_box / len(EXPERTS)
    total_uh = total_uh / (4 * len(EXPERTS))

    # [Nq,4]
    Uc = torch.cat(true_u_cls, dim=-1)
    Hc = torch.cat(true_h_cls, dim=-1)
    Ub = torch.cat(true_u_box, dim=-1)
    Hb = torch.cat(true_h_box, dim=-1)

    amp_cls = torch.relu(Uc - Hc)
    amp_box = torch.relu(Ub - Hb)

    return total_cls, total_box, total_uh, amp_cls, amp_box


def gate_router_loss(model_out, q_idx, amp_cls, amp_box):
    pred_g_cls = model_out["g_cls"][q_idx]
    pred_g_box = model_out["g_box"][q_idx]

    L_gate = 0.5 * (
        F.mse_loss(pred_g_cls, amp_cls)
        + F.mse_loss(pred_g_box, amp_box)
    )

    def target_pi(amp):
        denom = 1.0 + amp.sum(dim=-1, keepdim=True)
        return torch.cat([1.0 / denom, amp / denom], dim=-1)

    tgt_cls = target_pi(amp_cls)
    tgt_box = target_pi(amp_box)

    p_cls = model_out["pi_cls"][q_idx].clamp_min(1e-8)
    p_box = model_out["pi_box"][q_idx].clamp_min(1e-8)

    L_route = 0.5 * (
        F.kl_div(p_cls.log(), tgt_cls, reduction="batchmean")
        + F.kl_div(p_box.log(), tgt_box, reduction="batchmean")
    )
    return L_gate, L_route


def detection_loss(criterion, z, b, can):
    targets = [{
        "boxes": can["gt_boxes_norm"].to(DEVICE).float(),
        "labels": can["gt_labels"].to(DEVICE).long(),
    }]
    loss_dict = criterion(
        {"pred_logits": z.unsqueeze(0), "pred_boxes": b.unsqueeze(0)},
        targets,
    )
    return sum(
        loss_dict[k] * criterion.weight_dict[k]
        for k in loss_dict
        if k in criterion.weight_dict
    )


# =============================================================================
# SMOKE
# =============================================================================

def exact_mechanism_smoke(
    model,
    criterion,
    CanonicalObjectState,
    frame_inputs_from_records,
    folds,
    v2f,
    manifest,
):
    train_videos = sorted(set(range(1, 48)) - set(folds[META_FOLD]))
    vid = train_videos[0]
    source_fold = v2f[vid]
    load_source_fold(model, source_fold)

    canonical, spatial, temporal, labels = load_video_all(vid, source_fold)
    idx = len(canonical) // 2

    O, X = s0a.make_O_X(
        canonical[idx], spatial[idx], temporal[idx],
        CanonicalObjectState, frame_inputs_from_records
    )
    bundle = expert_bundle(model, O, X)

    # -------------------------------------------------------------
    # A. Final-Expert checkpoint -> RUN_07 OOF reproduction
    # -------------------------------------------------------------
    oof_records = torch.load(s0a.fold_oof_path(source_fold, vid), map_location="cpu")
    oof = oof_records[idx]

    max_dz = 0.0
    max_db = 0.0
    for k in range(4):
        z_k = model.adapter.decode_cls(
            O.h + bundle["delta_cls"][:, k]
        )
        b_k = model.adapter.decode_box(
            O.h + bundle["delta_box"][:, k],
            O.r,
        )
        max_dz = max(
            max_dz,
            float((z_k.detach().cpu() - oof["expert_cls_logits"][k].float()).abs().max().item()),
        )
        max_db = max(
            max_db,
            float((b_k.detach().cpu() - oof["expert_box_boxes"][k].float()).abs().max().item()),
        )

    if max_dz >= 1.0e-5 or max_db >= 1.0e-5:
        raise AssertionError(
            f"[FATAL] Final-Expert checkpoint/OOF reproduction drift: "
            f"dz={max_dz:.3e}, db={max_db:.3e}"
        )

    print(
        f"[+] Final-Expert OOF reproduction PASS: "
        f"max dz={max_dz:.3e}, max db={max_db:.3e}"
    )

    q_idx = labels[idx]["candidate_queries"].long().to(DEVICE)

    # candidate provenance
    q_re = torch.nonzero(
        bundle["candidate_mask"].reshape(-1) > 0.5, as_tuple=False
    ).reshape(-1).cpu()
    if not torch.equal(q_re, q_idx.cpu()):
        raise AssertionError("[FATAL] RUN_09B candidate provenance drift.")

    weights = aggregate_training_counts(manifest, train_videos)

    model.train()
    for m in (
        model.expert_s, model.expert_l, model.expert_c, model.expert_t, model.context_bus
    ):
        m.eval()

    out = mechanism_forward(model, O, X, bundle, hard=False)

    # -------------------------------------------------------------
    # B. Direct RUN_09 training path == corrected production forward
    # -------------------------------------------------------------
    z_prod, b_prod, _ = model(
        O,
        X,
        m_ext=None,
        m_hat_vis=None,
        m_avail=None,
        context_mode="auto",
    )
    dz_prod = float((out["z"] - z_prod).abs().max().item())
    db_prod = float((out["b"] - b_prod).abs().max().item())
    if dz_prod >= 1.0e-6 or db_prod >= 1.0e-6:
        raise AssertionError(
            f"[FATAL] RUN_09 direct/production forward drift: "
            f"dz={dz_prod:.3e}, db={db_prod:.3e}"
        )
    print(
        f"[+] Direct mechanism / production forward parity PASS: "
        f"dz={dz_prod:.3e}, db={db_prod:.3e}"
    )

    Lc, Lb, Luh, ac, ab = consequence_loss(out, labels[idx], q_idx, weights)
    Lg, Lr = gate_router_loss(out, q_idx, ac, ab)
    Ld = detection_loss(criterion, out["z"], out["b"], canonical[idx])
    total = Lc + Lb + LAMBDA_UH * Luh + LAMBDA_GATE * Lg + LAMBDA_ROUTE * Lr + LAMBDA_DET * Ld

    model.zero_grad(set_to_none=True)
    total.backward()

    required = {
        "self_assessment": model.self_assessment,
        "gate_cls": model.gate_cls,
        "gate_box": model.gate_box,
        "coord_cls": model.coord_cls,
        "coord_box": model.coord_box,
    }
    for name, module in required.items():
        gsum = 0.0
        for p in module.parameters():
            if p.grad is not None:
                gsum += float(p.grad.abs().sum().item())
        if gsum <= 0:
            raise AssertionError(f"[FATAL] No gradient reached {name}.")

    for name, module in {
        "expert_s": model.expert_s,
        "expert_l": model.expert_l,
        "expert_c": model.expert_c,
        "expert_t": model.expert_t,
        "context_bus": model.context_bus,
        "adapter": model.adapter,
    }.items():
        if any(p.grad is not None and float(p.grad.abs().sum()) != 0.0 for p in module.parameters()):
            raise AssertionError(f"[FATAL] Frozen module received gradient: {name}")

    # Fresh neutral model eval must exactly bypass because gates start closed.
    model.eval()
    hard_out = mechanism_forward(model, O, X, bundle, hard=True)
    if hard_out["hard_cls"].sum().item() != 0 or hard_out["hard_box"].sum().item() != 0:
        raise AssertionError("[FATAL] Neutral hard-gate startup is not closed.")
    if not torch.equal(hard_out["z"], O.z) or not torch.equal(hard_out["b"], O.b):
        raise AssertionError("[FATAL] Neutral hard-gate startup did not return bitwise baseline.")

    print(
        "[PASS] RUN_09B exact mechanism smoke: gradients reached "
        "self-assessment/UH/Gates/Coordination; Final Experts/R0 frozen; "
        "neutral hard fallback bitwise exact"
    )

    del canonical, spatial, temporal, labels, O, X, bundle, out, hard_out


# =============================================================================
# TRAIN META-FOLD 0
# =============================================================================

def set_training_modes(model, joint: bool):
    model.train()
    for m in (
        model.expert_s, model.expert_l, model.expert_c, model.expert_t, model.context_bus, model.adapter
    ):
        m.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    for p in model.self_assessment.parameters():
        p.requires_grad_(True)

    if joint:
        for module in (model.gate_cls, model.gate_box, model.coord_cls, model.coord_box):
            for p in module.parameters():
                p.requires_grad_(True)


def make_optimizer(model, joint: bool):
    groups = [{"params": [p for p in model.self_assessment.parameters() if p.requires_grad], "lr": LR_SA}]
    if joint:
        gate_params = []
        for module in (model.gate_cls, model.gate_box, model.coord_cls, model.coord_box):
            gate_params.extend([p for p in module.parameters() if p.requires_grad])
        groups.append({"params": gate_params, "lr": LR_GATE})
    return torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)


def run_epoch(
    epoch_name: str,
    model,
    criterion,
    CanonicalObjectState,
    frame_inputs_from_records,
    train_videos,
    v2f,
    weights,
    joint: bool,
    seed: int,
    optimizer,
):
    set_training_modes(model, joint=joint)

    rng = random.Random(seed)
    source_folds = sorted({v2f[v] for v in train_videos})
    rng.shuffle(source_folds)

    sums = defaultdict(float)
    n = 0
    t0 = time.time()

    for source_fold in source_folds:
        load_source_fold(model, source_fold)
        vids = [v for v in train_videos if v2f[v] == source_fold]
        rng.shuffle(vids)

        for vid in vids:
            canonical, spatial, temporal, labels = load_video_all(vid, source_fold)
            order = list(range(len(canonical)))
            rng.shuffle(order)

            for idx in order:
                O, X = s0a.make_O_X(
                    canonical[idx], spatial[idx], temporal[idx],
                    CanonicalObjectState, frame_inputs_from_records
                )
                bundle = expert_bundle(model, O, X)
                q_idx = labels[idx]["candidate_queries"].long().to(DEVICE)

                optimizer.zero_grad(set_to_none=True)
                out = mechanism_forward(model, O, X, bundle, hard=False)

                Lc, Lb, Luh, ac, ab = consequence_loss(
                    out, labels[idx], q_idx, weights
                )

                if joint:
                    Lg, Lr = gate_router_loss(out, q_idx, ac, ab)
                    Ld = detection_loss(
                        criterion, out["z"], out["b"], canonical[idx]
                    )
                    total = (
                        Lc + Lb
                        + LAMBDA_UH * Luh
                        + LAMBDA_GATE * Lg
                        + LAMBDA_ROUTE * Lr
                        + LAMBDA_DET * Ld
                    )
                else:
                    Lg = torch.zeros((), device=DEVICE)
                    Lr = torch.zeros((), device=DEVICE)
                    Ld = torch.zeros((), device=DEVICE)
                    total = Lc + Lb + LAMBDA_UH * Luh

                if not torch.isfinite(total):
                    raise AssertionError(
                        f"[FATAL] Non-finite RUN_09B loss vid={vid} idx={idx}"
                    )

                total.backward()
                trainable = [p for p in model.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
                optimizer.step()

                n += 1
                sums["total"] += float(total.detach())
                sums["cls"] += float(Lc.detach())
                sums["box"] += float(Lb.detach())
                sums["uh"] += float(Luh.detach())
                sums["gate"] += float(Lg.detach())
                sums["route"] += float(Lr.detach())
                sums["det"] += float(Ld.detach())

                if n % 500 == 0:
                    print(
                        f"    {epoch_name} frames={n:,} "
                        f"total={sums['total']/n:.4f} "
                        f"SAcls={sums['cls']/n:.4f} "
                        f"SAbox={sums['box']/n:.4f} "
                        f"UH={sums['uh']/n:.4f} "
                        f"gate={sums['gate']/n:.4f} "
                        f"route={sums['route']/n:.4f} "
                        f"det={sums['det']/n:.4f}"
                    )

                del O, X, bundle, out, total

            print(f"    [+] {epoch_name} video {vid:02d} complete")
            del canonical, spatial, temporal, labels
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    return {
        "name": epoch_name,
        "frames": n,
        "mean_total": sums["total"] / max(1, n),
        "mean_cls": sums["cls"] / max(1, n),
        "mean_box": sums["box"] / max(1, n),
        "mean_uh": sums["uh"] / max(1, n),
        "mean_gate": sums["gate"] / max(1, n),
        "mean_route": sums["route"] / max(1, n),
        "mean_det": sums["det"] / max(1, n),
        "runtime_minutes": (time.time() - t0) / 60.0,
    }


# =============================================================================
# FOLD-0 HARD EVALUATION
# =============================================================================

def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1, bx2, by2 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    ix = max(0.0, min(ax2,bx2)-max(ax1,bx1))
    iy = max(0.0, min(ay2,by2)-max(ay1,by1))
    inter = ix*iy
    union = a[2]*a[3]+b[2]*b[3]-inter
    return inter/union if union > 0 else 0.0


def operational_counts(z, b, gt_boxes, gt_labels):
    p = z.sigmoid().flatten()
    scores, flat = torch.topk(p, TOPK)
    q = flat // NUM_CLASSES
    c = flat % NUM_CLASSES
    boxes = b[q]
    valid = scores >= SCORE_THRESHOLD

    scores = scores[valid].tolist()
    q = q[valid].tolist()
    c = c[valid].tolist()
    boxes = boxes[valid].tolist()
    gt = gt_boxes.tolist()
    gl = gt_labels.tolist()

    matched = set()
    tp = 0
    fp = 0
    for i in range(len(scores)):
        best = None
        best_iou = -1.0
        for g in range(len(gt)):
            if g in matched or int(gl[g]) != int(c[i]):
                continue
            iou = box_iou(boxes[i], gt[g])
            if iou >= IOU_THRESHOLD and iou > best_iou:
                best_iou = iou
                best = g
        if best is None:
            fp += 1
        else:
            matched.add(best)
            tp += 1
    return tp, fp, len(gt) - tp


def hard_eval_fold0(
    model,
    CanonicalObjectState,
    frame_inputs_from_records,
    folds,
    v2f,
):
    heldout = sorted(folds[META_FOLD])
    load_source_fold(model, META_FOLD)
    model.eval()

    base = defaultdict(int)
    integ = defaultdict(int)
    frame_count = 0
    bypass_frames = 0
    active_cls = torch.zeros(4, dtype=torch.long)
    active_box = torch.zeros(4, dtype=torch.long)
    mean_pi_cls = torch.zeros(5)
    mean_pi_box = torch.zeros(5)
    pi_n = 0

    with torch.no_grad():
        for vid in heldout:
            canonical, spatial, temporal, labels = load_video_all(vid, META_FOLD)
            for idx in range(len(canonical)):
                O, X = s0a.make_O_X(
                    canonical[idx], spatial[idx], temporal[idx],
                    CanonicalObjectState, frame_inputs_from_records
                )
                bundle = expert_bundle(model, O, X)
                out = mechanism_forward(model, O, X, bundle, hard=True)

                btp, bfp, bfn = operational_counts(
                    O.z.cpu(), O.b.cpu(),
                    canonical[idx]["gt_boxes_norm"],
                    canonical[idx]["gt_labels"],
                )
                itp, ifp, ifn = operational_counts(
                    out["z"].cpu(), out["b"].cpu(),
                    canonical[idx]["gt_boxes_norm"],
                    canonical[idx]["gt_labels"],
                )

                base["tp"] += btp; base["fp"] += bfp; base["fn"] += bfn
                integ["tp"] += itp; integ["fp"] += ifp; integ["fn"] += ifn
                frame_count += 1

                hc = out["hard_cls"].cpu()
                hb = out["hard_box"].cpu()
                active_cls += (hc > 0).sum(dim=0)
                active_box += (hb > 0).sum(dim=0)
                mean_pi_cls += out["pi_cls"].cpu().mean(dim=0)
                mean_pi_box += out["pi_box"].cpu().mean(dim=0)
                pi_n += 1

                if hc.sum().item() == 0 and hb.sum().item() == 0:
                    bypass_frames += 1
                    if not torch.equal(out["z"], O.z) or not torch.equal(out["b"], O.b):
                        raise AssertionError("[FATAL] Hard bypass frame not bitwise baseline.")

            print(f"    [+] hard eval fold0 video {vid:02d} complete")

    def metrics(x):
        tp, fp, fn = x["tp"], x["fp"], x["fn"]
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}

    return {
        "frames": frame_count,
        "heldout_videos": heldout,
        "base": metrics(base),
        "integrated": metrics(integ),
        "bypass_frames": bypass_frames,
        "bypass_fraction": bypass_frames / max(1, frame_count),
        "active_cls_by_expert": {
            e: int(active_cls[i].item()) for i, e in enumerate(EXPERTS)
        },
        "active_box_by_expert": {
            e: int(active_box[i].item()) for i, e in enumerate(EXPERTS)
        },
        "mean_pi_cls": (mean_pi_cls / max(1, pi_n)).tolist(),
        "mean_pi_box": (mean_pi_box / max(1, pi_n)).tolist(),
    }


# =============================================================================
# CHECKPOINT
# =============================================================================

def save_mechanism_checkpoint(model, history, eval_result, consequence_sha):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "meta_fold0_uh_gate_router.pth"

    payload = {
        "_meta": {
            "schema_version": "bfcef_f_run09b_joint_uh_gate_router_fold0_v1",
            "meta_fold": META_FOLD,
            "run07_oof_manifest_sha256": EXPECTED_OOF_SHA,
            "run09_consequence_manifest_sha256": consequence_sha,
            "warmup_epochs": WARMUP_EPOCHS,
            "joint_epochs": JOINT_EPOCHS,
            "lr_sa": LR_SA,
            "lr_gate": LR_GATE,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "loss_weights": {
                "uh": LAMBDA_UH,
                "gate": LAMBDA_GATE,
                "route": LAMBDA_ROUTE,
                "det": LAMBDA_DET,
            },
            "history": history,
            "heldout_eval": eval_result,
        },
        "self_assessment": {
            k: {n: v.detach().cpu() for n, v in m.state_dict().items()}
            for k, m in model.self_assessment.items()
        },
        "gate_cls": {
            k: {n: v.detach().cpu() for n, v in m.state_dict().items()}
            for k, m in model.gate_cls.items()
        },
        "gate_box": {
            k: {n: v.detach().cpu() for n, v in m.state_dict().items()}
            for k, m in model.gate_box.items()
        },
        "coord_cls": {n: v.detach().cpu() for n, v in model.coord_cls.state_dict().items()},
        "coord_box": {n: v.detach().cpu() for n, v in model.coord_box.state_dict().items()},
    }

    torch.save(payload, path)
    return path


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    print("=" * 118)
    print("B-FCEF-F RUN_09B -- JOINT U/H + TASK GATES + COORDINATOR + SIMPLEX ROUTER")
    print("META-FOLD 0 PILOT | FINAL EXPERTS + R0 FROZEN | NO VAL/TEST")
    print("=" * 118)

    if sha256(OOF_MANIFEST) != EXPECTED_OOF_SHA:
        raise AssertionError("[FATAL] RUN_07 OOF root drift.")
    if not CONSEQUENCE_MANIFEST.exists():
        raise AssertionError("[FATAL] RUN_09A consequence manifest missing.")

    consequence_manifest = load_json(CONSEQUENCE_MANIFEST)
    consequence_sha = sha256(CONSEQUENCE_MANIFEST)

    folds = parse_folds()
    v2f = video_fold_map(folds)
    train_videos = sorted(set(range(1, 48)) - set(folds[META_FOLD]))

    (
        model,
        criterion,
        CanonicalObjectState,
        frame_inputs_from_records,
    ) = initialize_runtime()

    set_seed(SEED)

    exact_mechanism_smoke(
        model,
        criterion,
        CanonicalObjectState,
        frame_inputs_from_records,
        folds,
        v2f,
        consequence_manifest,
    )

    if args.smoke_only:
        return

    # Reinitialize after smoke so pilot starts from pristine neutral gates.
    del model
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    (
        model,
        criterion,
        CanonicalObjectState,
        frame_inputs_from_records,
    ) = initialize_runtime()
    set_seed(SEED)

    weights = aggregate_training_counts(consequence_manifest, train_videos)

    history = []

    set_training_modes(model, joint=False)
    warm_optimizer = make_optimizer(model, joint=False)
    for epoch in range(WARMUP_EPOCHS):
        h = run_epoch(
            epoch_name=f"WARMUP {epoch+1}/{WARMUP_EPOCHS}",
            model=model,
            criterion=criterion,
            CanonicalObjectState=CanonicalObjectState,
            frame_inputs_from_records=frame_inputs_from_records,
            train_videos=train_videos,
            v2f=v2f,
            weights=weights,
            joint=False,
            seed=SEED + epoch,
            optimizer=warm_optimizer,
        )
        history.append(h)
        print(f"[+] {h['name']} COMPLETE: total={h['mean_total']:.6f}")

    set_training_modes(model, joint=True)
    joint_optimizer = make_optimizer(model, joint=True)
    for epoch in range(JOINT_EPOCHS):
        h = run_epoch(
            epoch_name=f"JOINT {epoch+1}/{JOINT_EPOCHS}",
            model=model,
            criterion=criterion,
            CanonicalObjectState=CanonicalObjectState,
            frame_inputs_from_records=frame_inputs_from_records,
            train_videos=train_videos,
            v2f=v2f,
            weights=weights,
            joint=True,
            seed=SEED + 100 + epoch,
            optimizer=joint_optimizer,
        )
        history.append(h)
        print(
            f"[+] {h['name']} COMPLETE: total={h['mean_total']:.6f}, "
            f"gate={h['mean_gate']:.6f}, route={h['mean_route']:.6f}, det={h['mean_det']:.6f}"
        )

    eval_result = hard_eval_fold0(
        model,
        CanonicalObjectState,
        frame_inputs_from_records,
        folds,
        v2f,
    )

    path = save_mechanism_checkpoint(
        model, history, eval_result, consequence_sha
    )

    print("=" * 118)
    print("[PASS] RUN_09B META-FOLD-0 JOINT MECHANISM PILOT COMPLETE")
    print(f"[+] Checkpoint: {path}")
    print(f"[+] SHA:        {sha256(path)}")
    print(
        "[+] Fold0 Base:       "
        f"TP={eval_result['base']['tp']:,} "
        f"FP={eval_result['base']['fp']:,} "
        f"FN={eval_result['base']['fn']:,} "
        f"F1={eval_result['base']['f1']:.6f}"
    )
    print(
        "[+] Fold0 Integrated: "
        f"TP={eval_result['integrated']['tp']:,} "
        f"FP={eval_result['integrated']['fp']:,} "
        f"FN={eval_result['integrated']['fn']:,} "
        f"F1={eval_result['integrated']['f1']:.6f}"
    )
    print(f"[+] Hard bypass fraction: {eval_result['bypass_fraction']:.4%}")
    print(f"[+] Active cls by expert: {eval_result['active_cls_by_expert']}")
    print(f"[+] Active box by expert: {eval_result['active_box_by_expert']}")
    print("[+] No Val/Test accessed. Remaining meta-folds have NOT been trained.")
    print("=" * 118)


if __name__ == "__main__":
    main()
