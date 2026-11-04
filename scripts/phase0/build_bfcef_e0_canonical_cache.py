"""
D:/MARIVISION/H0_workbench/eval_tools/build_bfcef_e0_canonical_cache.py
MARSEN B-FCEF: Stage E0 - Authoritative Canonical Feature & Cue Cache Generator
(D-4495..D-4500, P-2306..P-2307)

Locked Implementation Contract:
  1. Authoritative Universe: Exactly 20,075 frames driven strictly by the 47 frozen production
     shards (full_split_shards/marsen_b2_shard_train_vid{1..47}.pth frame_records).
  2. Dataset Indexing: Maps source image_id to SeaClipsDetection solely for image tensors and transformed GT.
  3. All-Frame Parity Oracle: Asserts |delta_z| < 1e-4 and |delta_b| < 1e-4 against production shards for
     every single one of the 20,075 frames.
  4. 16-Frame Preflight Parity Gate: Asserts cached S/L/C raw cues reproduce live expert outputs (atol=1e-6).
  5. Compact Per-Video Shards: Pre-extracts raw S/L/C/Q cues, explicit R/N proxies, float16 appearance
     descriptors, and frees full P3/P4/P5 tensors immediately.
  6. Cryptographic Provenance Manifest: Freezes bfcef_e0_cache_manifest.json with all 47 input and
     output SHA-256 ledgers, substrate hash, and parity drift maxima.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
from pathlib import Path

ROOT_DIR = Path("D:/MARIVISION")
OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
SHARD_DIR = OUT_DIR / "full_split_shards"
CACHE_DIR = OUT_DIR / "e0_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CENSUS_JSON = OUT_DIR / "marsen_b2_production_census_56shards.json"
TRAIN_JSON = ROOT_DIR / "seaclips-train.json"
ZIP_PATH = ROOT_DIR / "SeaClips.zip"
R0_CHECKPOINT_PATH = ROOT_DIR / "checkpoints_r0" / "checkpoint_best.pth"
SUBSTRATE_SCRIPT_PATH = OUT_DIR / "run_bfcef_e0_substrate.py"

FROZEN_R0_CKPT_SHA256 = "28e6288c3f6769c2d5275aa77ddb248735b55a09cd46a6b570e3f117890b268b"
FROZEN_CENSUS_SHA256 = "a20a734d668a9078bdfe274cb09fa5a47ad9c6a3a9d7cef2d1f99b926623fbc5"

REPO_CANDIDATES = [ROOT_DIR / "Deformable-DETR", ROOT_DIR / "deformable_detr"]
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

assert DEFORMABLE_DETR_ROOT is not None, "[FATAL] Cannot locate authoritative Deformable-DETR repository."

for p in reversed([OUT_DIR, DEFORMABLE_DETR_ROOT, ROOT_DIR]):
    p_str = str(p)
    if p_str in sys.path:
        sys.path.remove(p_str)
    sys.path.insert(0, p_str)

import time
import json
import hashlib
from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from run_bfcef_e0_substrate import (
    BFCEFSubstrate,
    AuthenticDeformableDETRAdapter,
    CanonicalObjectState,
    CanonicalMultiscaleFeatures,
    load_authoritative_r0_args,
    compute_sha256,
    NUM_CLASSES,
    P0_SAMPLE_VIDEOS
)

sys.stdout.reconfigure(line_buffering=True)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_VIDEOS = 47
EXPECTED_TOTAL_FRAMES = 20075


def box_iou_norm_cxcywh(b1, b2):
    b1_x1 = b1[0] - b1[2] / 2.0
    b1_y1 = b1[1] - b1[3] / 2.0
    b1_x2 = b1[0] + b1[2] / 2.0
    b1_y2 = b1[1] + b1[3] / 2.0

    b2_x1 = b2[0] - b2[2] / 2.0
    b2_y1 = b2[1] - b2[3] / 2.0
    b2_x2 = b2[0] + b2[2] / 2.0
    b2_y2 = b2[1] + b2[3] / 2.0

    ix = max(0.0, min(b1_x2, b2_x2) - max(b1_x1, b2_x1))
    iy = max(0.0, min(b1_y2, b2_y2) - max(b1_y1, b2_y1))
    inter = ix * iy
    union = b1[2] * b1[3] + b2[2] * b2[3] - inter
    return inter / union if union > 0.0 else 0.0


def compute_canonical_baseline_tp_mask(
    eval_logits: torch.Tensor,
    eval_boxes: torch.Tensor,
    gt_boxes_norm: torch.Tensor,
    gt_labels: torch.Tensor
) -> torch.Tensor:
    num_q = eval_boxes.shape[0]
    probs = eval_logits.sigmoid()
    flat_probs = probs.flatten()
    topk_scores, topk_indices = torch.topk(flat_probs, min(100, flat_probs.numel()))

    q_indices = (topk_indices // NUM_CLASSES).tolist()
    c_indices = (topk_indices % NUM_CLASSES).tolist()
    scores = topk_scores.tolist()

    matched_gt_indices = set()
    canonical_tp_mask = torch.zeros(num_q, dtype=torch.bool, device=eval_boxes.device)

    num_gt = gt_boxes_norm.shape[0]
    if num_gt == 0:
        return canonical_tp_mask

    gt_boxes_list = gt_boxes_norm.tolist()
    gt_labels_list = gt_labels.tolist()

    for sc, q_idx, c_idx in zip(scores, q_indices, c_indices):
        if sc < 0.20:
            continue

        det_box = eval_boxes[q_idx].tolist()
        best_iou = -1.0
        best_gt_idx = None

        for g_idx in range(num_gt):
            if g_idx in matched_gt_indices:
                continue
            if gt_labels_list[g_idx] != c_idx:
                continue
            iou = box_iou_norm_cxcywh(det_box, gt_boxes_list[g_idx])
            if iou >= 0.30 and iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx

        if best_gt_idx is not None:
            matched_gt_indices.add(best_gt_idx)
            canonical_tp_mask[q_idx] = True

    return canonical_tp_mask


def extract_compact_raw_cues(
    boxes: torch.Tensor,
    logits: torch.Tensor,
    ref_pts: torch.Tensor,
    p3: torch.Tensor,
    p4: torch.Tensor,
    p5: torch.Tensor,
    img_w: float = 640.0,
    img_h: float = 640.0
) -> Dict[str, Any]:
    """
    Computes compact raw input cues and explicit R/N proxies.
    Semantics are mathematically identical to ExpertScale, ExpertLocalization, ExpertClassification, SceneRouter.
    """
    num_q = boxes.shape[0]
    probs = logits.sigmoid()
    scores, _ = torch.max(probs, dim=-1)

    # 1. Geometry (Scale & Localization)
    bw_px = boxes[:, 2] * img_w
    bh_px = boxes[:, 3] * img_h
    area_norm = (bw_px * bh_px) / (float(img_w * img_h) + 1e-6)
    log_scale = torch.log(torch.clamp(bw_px * bh_px, min=1.0)) / 12.0
    aspect_ratio = torch.clamp(bw_px / (bh_px + 1e-4), 0.1, 10.0) / 5.0

    cx, cy, w, h = boxes.unbind(-1)
    rx, ry = ref_pts[:, :2].unbind(-1)
    offset_x = (cx - rx) * 10.0
    offset_y = (cy - ry) * 10.0
    edge_dist = torch.min(torch.stack([cx - w/2, cy - h/2, 1.0 - (cx + w/2), 1.0 - (cy + h/2)]), dim=0)[0]
    clipped = (edge_dist < 0.02).float()

    # 2. Multiscale Visual Sampling
    grid = torch.stack([cx * 2.0 - 1.0, cy * 2.0 - 1.0], dim=-1).view(1, 1, num_q, 2)
    feat_p3 = F.grid_sample(p3, grid, align_corners=False).view(-1, num_q).t()
    feat_p4 = F.grid_sample(p4, grid, align_corners=False).view(-1, num_q).t()
    feat_p5 = F.grid_sample(p5, grid, align_corners=False).view(-1, num_q).t()

    energy_p3 = torch.norm(feat_p3, dim=-1)
    energy_p4 = torch.norm(feat_p4, dim=-1)
    energy_p5 = torch.norm(feat_p5, dim=-1)
    total_energy = energy_p3 + energy_p4 + energy_p5 + 1e-6

    ratio_fine = energy_p3 / total_energy
    ratio_coarse = energy_p5 / total_energy
    cross_scale_cos = F.cosine_similarity(feat_p3, feat_p5, dim=-1)

    # 3. Boundary Gradient Sampling (P3)
    top_grid = torch.stack([cx * 2.0 - 1.0, (cy - h/2).clamp(0, 1) * 2.0 - 1.0], dim=-1).view(1, 1, num_q, 2)
    bot_grid = torch.stack([cx * 2.0 - 1.0, (cy + h/2).clamp(0, 1) * 2.0 - 1.0], dim=-1).view(1, 1, num_q, 2)
    f_top = F.grid_sample(p3, top_grid, align_corners=False).view(-1, num_q).t()
    f_bot = F.grid_sample(p3, bot_grid, align_corners=False).view(-1, num_q).t()
    grad_v = (torch.norm(feat_p3 - f_top, dim=-1) + torch.norm(feat_p3 - f_bot, dim=-1)) / (energy_p3 + 1e-6)

    # 4. Explicit Reliability & Need Proxies
    rel_s_proxy = torch.clamp(cross_scale_cos + 0.5, 0.1, 1.0)
    need_s_proxy = torch.clamp(1.0 - log_scale, 0.0, 1.0) * (1.0 - scores)

    rel_l_proxy = torch.clamp(grad_v / 2.0, 0.1, 1.0)
    need_l_proxy = torch.clamp(torch.abs(offset_x) + torch.abs(offset_y) + clipped * 0.5, 0.0, 1.0)

    # 5. Semantic Margins & Entropy
    top2 = torch.topk(logits, k=2, dim=-1)[0]
    margin = (top2[:, 0] - top2[:, 1]) / 5.0
    p = torch.clamp(probs, 1e-6, 1.0 - 1e-6)
    h_b = -torch.mean(p * torch.log2(p) + (1.0 - p) * torch.log2(1.0 - p), dim=-1)

    rel_c_proxy = torch.clamp(margin, 0.1, 1.0)
    need_c_proxy = torch.clamp(1.0 - margin + h_b, 0.0, 1.0)

    # 6. Assemble Compact Input Cue Tensors
    cues_s = torch.stack([
        area_norm * 100.0, log_scale, aspect_ratio, ratio_fine, ratio_coarse,
        cross_scale_cos, scores, need_s_proxy, boxes[:, 2], boxes[:, 3]
    ], dim=-1)

    cues_l = torch.stack([
        offset_x, offset_y, w * 5.0, h * 5.0, edge_dist * 5.0,
        clipped, grad_v, scores, need_l_proxy, rel_l_proxy
    ], dim=-1)

    cues_c = torch.stack([
        margin, h_b, scores, torch.std(logits, dim=-1),
        need_c_proxy, torch.clamp(1.0 - scores, 0.0, 1.0)
    ], dim=-1)

    # Scene Q Statistics
    query_scores = probs.max(dim=-1).values
    act_density = float((query_scores > 0.10).float().mean().item())
    act_energy = float(probs.mean().item())
    coarse_energy = float(torch.abs(p5).mean().item())
    coarse_var = float(torch.var(p5).item())

    scene_q = {
        "detector_query_activity_density": act_density,
        "detector_activation_energy": act_energy,
        "coarse_feature_energy": coarse_energy,
        "coarse_feature_variance": coarse_var,
        "visibility_proxy": max(0.2, 1.0 - act_density * 0.4)
    }

    # Normalized mid-scale appearance descriptor for temporal association
    appearance_desc = F.normalize(feat_p4, p=2, dim=-1)

    return {
        "cues_s": cues_s.cpu(),
        "r_s_proxy": rel_s_proxy.cpu(),
        "n_s_proxy": need_s_proxy.cpu(),
        "cues_l": cues_l.cpu(),
        "r_l_proxy": rel_l_proxy.cpu(),
        "n_l_proxy": need_l_proxy.cpu(),
        "cues_c": cues_c.cpu(),
        "r_c_proxy": rel_c_proxy.cpu(),
        "n_c_proxy": need_c_proxy.cpu(),
        "appearance_desc": appearance_desc.half().cpu(),  # float16 saves ~3GB storage
        "scene_q": scene_q
    }


def run_16frame_cache_cue_parity_gate(
    r0_model: nn.Module,
    substrate: BFCEFSubstrate,
    dataset,
    iid_to_ds_idx: Dict[int, int]
):
    """
    Evaluates 16 canonical frames (from videos 1, 4, 7, 16, 25, 35, 42, 47, two frames each)
    proving that extract_compact_raw_cues() reproduces live expert encoder outputs within atol=1e-6.
    """
    print("\n" + "=" * 105)
    print("STAGE PREFLIGHT: 16-FRAME CACHE-CUE PARITY GATE (P-2307)")
    print("=" * 105)

    from util.misc import collate_fn

    sample_frames = []
    for vid in P0_SAMPLE_VIDEOS:
        shard_path = SHARD_DIR / f"marsen_b2_shard_train_vid{vid}.pth"
        shard = torch.load(shard_path, map_location="cpu")
        for fr in shard["frame_records"][:2]:
            sample_frames.append(fr)

    assert len(sample_frames) == 16, f"[FATAL] Expected 16 sample frames, got {len(sample_frames)}"

    hooked_taps = {}
    def tap_hook(module, inp, outp):
        srcs = inp[0]
        hooked_taps["p3"] = srcs[0].detach()
        hooked_taps["p4"] = srcs[1].detach()
        hooked_taps["p5"] = srcs[2].detach()
        hs = outp[0]
        init_ref = outp[1]
        inter_refs = outp[2]
        last_lvl = hs.shape[0] - 1
        hooked_taps["q_native"] = hs[last_lvl].detach()
        if last_lvl == 0:
            hooked_taps["ref_pts"] = init_ref.detach()
        else:
            hooked_taps["ref_pts"] = inter_refs[last_lvl - 1].detach()

    handle = r0_model.transformer.register_forward_hook(tap_hook)

    max_delta_e_s = 0.0
    max_delta_e_l = 0.0
    max_delta_e_c = 0.0

    for idx, fr in enumerate(sample_frames):
        iid = int(fr["image_id"])
        ds_idx = iid_to_ds_idx[iid]
        item = dataset[ds_idx]
        samples, _ = collate_fn([item])
        samples = samples.to(DEVICE)

        with torch.no_grad():
            out = r0_model(samples)

        d_logits = out["pred_logits"][0].detach()
        d_boxes = out["pred_boxes"][0].detach()
        q_native = hooked_taps["q_native"][0].detach()
        ref_pts = hooked_taps["ref_pts"][0].detach()
        p3, p4, p5 = hooked_taps["p3"], hooked_taps["p4"], hooked_taps["p5"]

        ms_feats = CanonicalMultiscaleFeatures(p3=p3, p4=p4, p5=p5)
        fr_tapped = {
            "image_id": iid,
            "img_w": 640.0,
            "img_h": 640.0,
            "d_logits": d_logits,
            "d_boxes": d_boxes,
            "d_queries": q_native,
            "d_reference_points": ref_pts,
            "multiscale_features": ms_feats,
            "temporal_history": None
        }

        # 1. Live Substrate Forward
        state_live = substrate.adapter.extract_canonical_state(fr_tapped)
        e_s_live, _, r_s_live, n_s_live = substrate.expert_s(state_live)
        e_l_live, _, r_l_live, n_l_live = substrate.expert_l(state_live)
        e_c_live, _, r_c_live, n_c_live = substrate.expert_c(state_live)

        # 2. Cache Function Extraction
        cues = extract_compact_raw_cues(d_boxes, d_logits, ref_pts, p3, p4, p5)
        e_s_cache = substrate.expert_s.encoder(cues["cues_s"].to(DEVICE))
        e_l_cache = substrate.expert_l.encoder(cues["cues_l"].to(DEVICE))
        e_c_cache = substrate.expert_c.encoder(cues["cues_c"].to(DEVICE))

        d_s = float(torch.abs(e_s_live - e_s_cache).max().item())
        d_l = float(torch.abs(e_l_live - e_l_cache).max().item())
        d_c = float(torch.abs(e_c_live - e_c_cache).max().item())

        max_delta_e_s = max(max_delta_e_s, d_s)
        max_delta_e_l = max(max_delta_e_l, d_l)
        max_delta_e_c = max(max_delta_e_c, d_c)

        assert torch.allclose(e_s_live, e_s_cache, atol=1e-6, rtol=0), f"[FATAL] Expert S cue mismatch on frame {iid}: {d_s:.2e}"
        assert torch.allclose(e_l_live, e_l_cache, atol=1e-6, rtol=0), f"[FATAL] Expert L cue mismatch on frame {iid}: {d_l:.2e}"
        assert torch.allclose(e_c_live, e_c_cache, atol=1e-6, rtol=0), f"[FATAL] Expert C cue mismatch on frame {iid}: {d_c:.2e}"

        assert torch.allclose(r_s_live, cues["r_s_proxy"].to(DEVICE), atol=1e-6, rtol=0), f"[FATAL] R_s proxy mismatch on frame {iid}"
        assert torch.allclose(n_s_live, cues["n_s_proxy"].to(DEVICE), atol=1e-6, rtol=0), f"[FATAL] N_s proxy mismatch on frame {iid}"
        assert torch.allclose(r_l_live, cues["r_l_proxy"].to(DEVICE), atol=1e-6, rtol=0), f"[FATAL] R_l proxy mismatch on frame {iid}"
        assert torch.allclose(n_l_live, cues["n_l_proxy"].to(DEVICE), atol=1e-6, rtol=0), f"[FATAL] N_l proxy mismatch on frame {iid}"
        assert torch.allclose(r_c_live, cues["r_c_proxy"].to(DEVICE), atol=1e-6, rtol=0), f"[FATAL] R_c proxy mismatch on frame {iid}"
        assert torch.allclose(n_c_live, cues["n_c_proxy"].to(DEVICE), atol=1e-6, rtol=0), f"[FATAL] N_c proxy mismatch on frame {iid}"

    handle.remove()
    print(f"    [+] PASS: 16-Frame Cache-Cue Parity Certified:")
    print(f"        Max Delta Expert S: {max_delta_e_s:.2e} (< 1e-6)")
    print(f"        Max Delta Expert L: {max_delta_e_l:.2e} (< 1e-6)")
    print(f"        Max Delta Expert C: {max_delta_e_c:.2e} (< 1e-6)")
    print("=" * 105 + "\n", flush=True)


def main():
    t_start = time.time()
    driver_sha = compute_sha256(Path(__file__))

    print("=" * 105)
    print("MARSEN B-FCEF: STAGE E0 - ONE-TIME CANONICAL FEATURE & CUE CACHE BUILDER")
    print(f"Driver Script SHA-256: {driver_sha}")
    print(f"Target Device:         {DEVICE}")
    print(f"Authoritative Universe: Exactly {EXPECTED_TOTAL_FRAMES:,} Frames across 47 Production Shards")
    print("=" * 105, flush=True)

    # 1. Cryptographic Provenance Preflight Gate
    print("[*] Checking Cryptographic Provenance across Census, Train JSON, Shards, Substrate...")
    assert CENSUS_JSON.exists(), f"[FATAL] Missing census JSON at {CENSUS_JSON}"
    assert compute_sha256(CENSUS_JSON) == FROZEN_CENSUS_SHA256, "[FATAL] Census SHA mismatch"
    assert compute_sha256(R0_CHECKPOINT_PATH) == FROZEN_R0_CKPT_SHA256, "[FATAL] R0 Checkpoint SHA mismatch"
    assert SUBSTRATE_SCRIPT_PATH.exists(), f"[FATAL] Missing substrate script at {SUBSTRATE_SCRIPT_PATH}"
    substrate_sha = compute_sha256(SUBSTRATE_SCRIPT_PATH)

    with open(CENSUS_JSON, "r", encoding="utf-8") as f:
        census = json.load(f)

    train_json_sha = census["_meta"]["live_provenance"]["train_json_sha256"]
    assert compute_sha256(TRAIN_JSON) == train_json_sha, "[FATAL] Train JSON drift"
    source_shard_ledger = census["shard_sha256_ledger"]

    # Verify all 47 input production shards exist and match census
    source_shard_hashes = {}
    for vid in range(1, NUM_VIDEOS + 1):
        s_name = f"marsen_b2_shard_train_vid{vid}.pth"
        s_path = SHARD_DIR / s_name
        assert s_path.exists(), f"[FATAL] Missing production shard {s_name}"
        actual_sha = compute_sha256(s_path)
        assert actual_sha == source_shard_ledger[s_name], f"[FATAL] Production shard drift on {s_name}"
        source_shard_hashes[s_name] = actual_sha

    print(f"    [+] Verified 47 production shards, Census, Train JSON, Checkpoint, and Substrate (SHA: {substrate_sha[:16]}...).")

    # 2. Build R0 Model & Instantiate Substrate
    args = load_authoritative_r0_args()
    args.device = str(DEVICE)
    from models.deformable_detr import build as build_deformable_detr

    r0_model, _, _ = build_deformable_detr(args)
    r0_ckpt = torch.load(R0_CHECKPOINT_PATH, map_location="cpu")
    sd = r0_ckpt.get("model", r0_ckpt)
    cleaned_sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    r0_model.load_state_dict(cleaned_sd, strict=True)
    r0_model.to(DEVICE)
    r0_model.eval()
    for p in r0_model.parameters():
        p.requires_grad_(False)

    adapter = AuthenticDeformableDETRAdapter(r0_model.class_embed[-1], r0_model.bbox_embed[-1]).to(DEVICE)
    substrate = BFCEFSubstrate(adapter=adapter).to(DEVICE)
    substrate.eval()
    print("[*] Frozen R0 model and substrate initialized on CUDA.")

    # 3. Ingest SeaClips Detection Dataset & Build image_id Mapping
    from datasets.seaclips_detection import SeaClipsDetection
    from util.misc import collate_fn

    dataset = SeaClipsDetection(ZIP_PATH, TRAIN_JSON, is_train=False, random_flip=False)
    if hasattr(dataset, "ids"):
        iid_to_ds_idx = {int(iid): idx for idx, iid in enumerate(dataset.ids)}
    elif hasattr(dataset, "img_ids"):
        iid_to_ds_idx = {int(iid): idx for idx, iid in enumerate(dataset.img_ids)}
    else:
        iid_to_ds_idx = {int(img["id"]): idx for idx, img in enumerate(dataset.coco.dataset["images"])}

    print(f"[*] Ingested SeaClipsDetection ({len(dataset):,} total images indexed).")

    # 4. Execute 16-Frame Cache-Cue Parity Gate (Pre-Flight Barrier)
    run_16frame_cache_cue_parity_gate(r0_model, substrate, dataset, iid_to_ds_idx)

    # 5. Full 20,075-Frame Canonical Extraction Loop across 47 Production Shards
    print("[*] Beginning Canonical Cache Extraction across all 47 Production Shards...", flush=True)

    hooked_taps = {}
    def tap_hook(module, inp, outp):
        srcs = inp[0]
        hooked_taps["p3"] = srcs[0].detach()
        hooked_taps["p4"] = srcs[1].detach()
        hooked_taps["p5"] = srcs[2].detach()
        hs = outp[0]
        init_ref = outp[1]
        inter_refs = outp[2]
        last_lvl = hs.shape[0] - 1
        hooked_taps["q_native"] = hs[last_lvl].detach()
        if last_lvl == 0:
            hooked_taps["ref_pts"] = init_ref.detach()
        else:
            hooked_taps["ref_pts"] = inter_refs[last_lvl - 1].detach()

    handle = r0_model.transformer.register_forward_hook(tap_hook)

    total_cached_frames = 0
    max_delta_logits = 0.0
    max_delta_boxes = 0.0
    cache_ledger = {}

    for vid in range(1, NUM_VIDEOS + 1):
        t_v0 = time.time()
        s_name = f"marsen_b2_shard_train_vid{vid}.pth"
        s_path = SHARD_DIR / s_name
        source_shard = torch.load(s_path, map_location="cpu")
        source_frames = source_shard["frame_records"]

        video_cache_records = []

        with torch.no_grad():
            for seq_idx, source_fr in enumerate(source_frames):
                image_id = int(source_fr["image_id"])
                assert image_id in iid_to_ds_idx, f"[FATAL] Source image_id {image_id} not found in SeaClips dataset!"
                ds_idx = iid_to_ds_idx[image_id]

                item = dataset[ds_idx]
                image_tensor, target = item
                samples, _ = collate_fn([item])
                samples = samples.to(DEVICE)

                out_live = r0_model(samples)

                d_logits = out_live["pred_logits"][0].detach()
                d_boxes = out_live["pred_boxes"][0].detach()
                q_native = hooked_taps["q_native"][0].detach()
                ref_pts = hooked_taps["ref_pts"][0].detach()
                p3 = hooked_taps["p3"]
                p4 = hooked_taps["p4"]
                p5 = hooked_taps["p5"]

                # All-Frame Parity Oracle: Live R0 vs Source Production Shard
                dz = float(torch.abs(d_logits.cpu() - source_fr["d_logits"]).max().item())
                db = float(torch.abs(d_boxes.cpu() - source_fr["d_boxes"]).max().item())
                assert dz < 1e-4, f"[FATAL] Live R0 vs production shard logit drift on frame {image_id}: {dz:.2e}"
                assert db < 1e-4, f"[FATAL] Live R0 vs production shard box drift on frame {image_id}: {db:.2e}"

                if dz > max_delta_logits:
                    max_delta_logits = dz
                if db > max_delta_boxes:
                    max_delta_boxes = db

                gt_boxes_norm = target["boxes"].to(DEVICE)
                gt_labels = target["labels"].to(DEVICE)

                # Canonical Baseline TP Query Mask (Flattened Top-100, s >= 0.20, IoU >= 0.30)
                tp_mask = compute_canonical_baseline_tp_mask(
                    d_logits, d_boxes, gt_boxes_norm, gt_labels
                )

                # Compact Raw Cues and Explicit Proxies
                cues = extract_compact_raw_cues(
                    d_boxes, d_logits, ref_pts, p3, p4, p5, img_w=640.0, img_h=640.0
                )

                video_cache_records.append({
                    "image_id": image_id,
                    "video_id": vid,
                    "seq_idx": seq_idx,
                    "img_w": 640.0,
                    "img_h": 640.0,
                    "d_logits": d_logits.cpu(),
                    "d_boxes": d_boxes.cpu(),
                    "q_native": q_native.cpu(),
                    "ref_pts": ref_pts.cpu(),
                    "cues_s": cues["cues_s"],
                    "r_s_proxy": cues["r_s_proxy"],
                    "n_s_proxy": cues["n_s_proxy"],
                    "cues_l": cues["cues_l"],
                    "r_l_proxy": cues["r_l_proxy"],
                    "n_l_proxy": cues["n_l_proxy"],
                    "cues_c": cues["cues_c"],
                    "r_c_proxy": cues["r_c_proxy"],
                    "n_c_proxy": cues["n_c_proxy"],
                    "appearance_desc": cues["appearance_desc"],  # float16
                    "scene_q": cues["scene_q"],
                    "gt_boxes_norm": gt_boxes_norm.cpu(),
                    "gt_labels": gt_labels.cpu(),
                    "canonical_tp_mask": tp_mask.cpu()
                })

        cache_file = CACHE_DIR / f"bfcef_e0_cache_vid{vid}.pth"
        torch.save(video_cache_records, cache_file)
        file_sha = compute_sha256(cache_file)
        cache_ledger[cache_file.name] = file_sha
        total_cached_frames += len(video_cache_records)

        print(f"[+] Video {vid:02d}/47 cached: {len(video_cache_records):>4} frames in {time.time() - t_v0:.1f}s -> {cache_file.name} (SHA: {file_sha[:16]}...)")

    handle.remove()

    # Exact Cardinality Assertions
    assert total_cached_frames == EXPECTED_TOTAL_FRAMES, f"[FATAL] Cached frame count: {total_cached_frames} != {EXPECTED_TOTAL_FRAMES}"
    assert len(cache_ledger) == NUM_VIDEOS, f"[FATAL] Cache file count: {len(cache_ledger)} != {NUM_VIDEOS}"

    # Freeze Cache Ledger & Provenance Certificate Manifest
    manifest_payload = {
        "_meta": {
            "cache_driver_sha256": driver_sha,
            "substrate_sha256": substrate_sha,
            "r0_checkpoint_sha256": FROZEN_R0_CKPT_SHA256,
            "census_sha256": FROZEN_CENSUS_SHA256,
            "train_json_sha256": train_json_sha,
            "expected_total_frames": EXPECTED_TOTAL_FRAMES,
            "actual_total_frames": total_cached_frames,
            "num_videos": NUM_VIDEOS,
            "max_live_vs_shard_delta_logits": max_delta_logits,
            "max_live_vs_shard_delta_boxes": max_delta_boxes,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "source_shard_sha256_ledger": source_shard_hashes,
        "cache_shard_sha256_ledger": cache_ledger
    }

    manifest_path = CACHE_DIR / "bfcef_e0_cache_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)

    print("\n" + "=" * 105)
    print(f"[+] CANONICAL CACHE GENERATION COMPLETED IN {(time.time() - t_start) / 60.0:.2f} MINUTES")
    print(f"[+] Total Cached Frames: {total_cached_frames:,} across {NUM_VIDEOS} video shards (EXACT MATCH)")
    print(f"[+] Full-Cache R0 Parity: Max Δlogits = {max_delta_logits:.2e}, Max Δboxes = {max_delta_boxes:.2e} (< 1e-4)")
    print(f"[+] Manifest Certificate Frozen: {manifest_path.resolve()}")
    print("=" * 105 + "\n", flush=True)


if __name__ == "__main__":
    main()