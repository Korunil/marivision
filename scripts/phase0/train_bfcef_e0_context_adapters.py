"""
D:/MARIVISION/H0_workbench/eval_tools/train_bfcef_e0_context_adapters.py
MARSEN B-FCEF: Stage E0 - Cross-Fitted Training of Decoupled Contextual Adapters
(D-4478..D-4528, P-2294..P-2319)

Production Cross-Fitting Pipeline:
  1. Cryptographic Provenance Chain: Validates Census, Train JSON, Checkpoint, Substrate,
     Canonical Cache Manifest, Temporal Cache Manifest, and Support-Aware Fold Manifest.
  2. Aligned Video Cache Loader: Asserts length, image_id, video_id, and seq_idx identity.
  3. Precomputed Temporal Association: Zero runtime Hungarian overhead (~5 mins T training / fold).
  4. True Need Vectors: Returns and records actual N_T alongside N_S, N_L, N_C.
  5. Pristine Fold Isolation: Bitwise resets substrate to pristine_zero_state at every fold boundary.
  6. Deterministic Video Shuffling: Shuffles training video order per epoch (preserves frame chronology).
  7. Exact Dual-Regime Loss: Official SetCriterion detection loss + canonical TP protection + delta penalty.
  8. Streamed OOF Ledger (Per-Video): Persists individual video chunks, eliminating multi-GB RAM bloat.
  9. Persistent Training Diagnostics: Records loss curves, projector norms, step counts, and timing.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
from pathlib import Path

ROOT_DIR = Path("D:/MARIVISION")
OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
CACHE_DIR = OUT_DIR / "e0_cache"
E0_WORK_DIR = OUT_DIR / "e0_adapters"
E0_WORK_DIR.mkdir(parents=True, exist_ok=True)

CENSUS_JSON = OUT_DIR / "marsen_b2_production_census_56shards.json"
TRAIN_JSON = ROOT_DIR / "seaclips-train.json"
R0_CHECKPOINT_PATH = ROOT_DIR / "checkpoints_r0" / "checkpoint_best.pth"
SUBSTRATE_SCRIPT_PATH = OUT_DIR / "run_bfcef_e0_substrate.py"
CACHE_MANIFEST_PATH = CACHE_DIR / "bfcef_e0_cache_manifest.json"
TEMP_CACHE_MANIFEST_PATH = CACHE_DIR / "bfcef_e0_temporal_cache_manifest.json"
MANIFEST_JSON = E0_WORK_DIR / "bfcef_e0_5fold_manifest.json"

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
import copy
import json
import gc
import hashlib
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from run_bfcef_e0_substrate import (
    BFCEFSubstrate,
    AuthenticDeformableDETRAdapter,
    CanonicalObjectState,
    load_authoritative_r0_args,
    compute_sha256,
    NUM_CLASSES
)

sys.stdout.reconfigure(line_buffering=True)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_VIDEOS = 47
NUM_FOLDS = 5
EXPECTED_TOTAL_FRAMES = 20075
EPOCHS_PER_EXPERT = 2
LR_EXPERT = 1e-4
WEIGHT_DECAY = 1e-4
E0_INIT_SEED = 20261004
OBJECTIVE_VERSION = "fold_balance_v3_native_camera_support_aware"
ASSOCIATION_VERSION = "t1_hungarian_v1_geom07iou03center_fusion05g03a02c_iou015_cost065"


def load_stratified_folds(canonical_manifest_sha: str) -> Tuple[Dict[int, List[int]], str]:
    """Loads and validates the support- and domain-aware 5-fold manifest bound to canonical cache."""
    assert MANIFEST_JSON.exists(), f"[FATAL] Missing fold manifest: {MANIFEST_JSON}. Run optimize_bfcef_e0_5fold_manifest.py first!"
    fold_manifest_sha = compute_sha256(MANIFEST_JSON)

    with open(MANIFEST_JSON, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    meta = manifest_data.get("_meta", {})
    assert meta.get("objective_version") == OBJECTIVE_VERSION, (
        f"[FATAL] Incompatible manifest objective version: {meta.get('objective_version')} != {OBJECTIVE_VERSION}"
    )
    assert meta.get("source_canonical_cache_manifest_sha256") == canonical_manifest_sha, (
        f"[FATAL] Fold manifest source hash mismatch! "
        f"{meta.get('source_canonical_cache_manifest_sha256')} != {canonical_manifest_sha}"
    )

    raw_folds = manifest_data["folds"]
    folds = {int(k): [int(v) for v in vids] for k, vids in raw_folds.items()}
    print(f"[+] Loaded verified support-aware 5-fold manifest: {MANIFEST_JSON.resolve()} (SHA: {fold_manifest_sha[:16]}...)")
    return folds, fold_manifest_sha


def load_aligned_video_caches(vid: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Centralized loader asserting strict length, image_id, video_id, and seq_idx equality (P-2315).
    """
    canonical_path = CACHE_DIR / f"bfcef_e0_cache_vid{vid}.pth"
    temporal_path = CACHE_DIR / f"bfcef_e0_temporal_cache_vid{vid}.pth"

    assert canonical_path.exists(), f"[FATAL] Missing canonical cache: {canonical_path.name}"
    assert temporal_path.exists(), f"[FATAL] Missing temporal cache: {temporal_path.name}"

    canonical = torch.load(canonical_path, map_location="cpu")
    temporal = torch.load(temporal_path, map_location="cpu")

    assert len(canonical) == len(temporal), (
        f"[FATAL] Cache length mismatch for video {vid}: {len(canonical)} != {len(temporal)}"
    )

    for idx, (c, t) in enumerate(zip(canonical, temporal)):
        assert int(c["image_id"]) == int(t["image_id"]), (
            f"[FATAL] image_id mismatch vid={vid}, idx={idx}: {c['image_id']} != {t['image_id']}"
        )
        assert int(c["seq_idx"]) == int(t["seq_idx"]) == idx, (
            f"[FATAL] seq_idx mismatch vid={vid}, idx={idx}: c={c['seq_idx']}, t={t['seq_idx']}"
        )
        assert int(c["video_id"]) == int(t["video_id"]) == vid, (
            f"[FATAL] video_id mismatch vid={vid}, idx={idx}: c={c['video_id']}, t={t['video_id']}"
        )

    return canonical, temporal


def execute_cached_forward(
    substrate: BFCEFSubstrate,
    exp_key: str,
    cached_fr: Dict[str, Any],
    cached_temp_fr: Optional[Dict[str, Any]] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Executes isolated contextual forward pass using cached raw cues and cached temporal history.
    Returns: (z_mod, b_mod, delta_q_box, delta_q_cls, evidence, avail, r_proxy, n_proxy).
    """
    d_boxes = cached_fr["d_boxes"].to(DEVICE)
    d_logits = cached_fr["d_logits"].to(DEVICE)
    q_native = cached_fr["q_native"].to(DEVICE)
    ref_pts = cached_fr["ref_pts"].to(DEVICE)
    num_q = d_boxes.shape[0]

    probs = d_logits.sigmoid()
    scores, _ = torch.max(probs, dim=-1)

    exp_module = getattr(substrate, f"expert_{exp_key.lower()}")

    if exp_key == "S":
        cues = cached_fr["cues_s"].to(DEVICE)
        evidence = exp_module.encoder(cues)
        avail = torch.ones(num_q, device=DEVICE)
        r_proxy = cached_fr["r_s_proxy"].to(DEVICE)
        n_proxy = cached_fr["n_s_proxy"].to(DEVICE)
        t_hist = None
    elif exp_key == "L":
        cues = cached_fr["cues_l"].to(DEVICE)
        evidence = exp_module.encoder(cues)
        avail = torch.ones(num_q, device=DEVICE)
        r_proxy = cached_fr["r_l_proxy"].to(DEVICE)
        n_proxy = cached_fr["n_l_proxy"].to(DEVICE)
        t_hist = None
    elif exp_key == "C":
        cues = cached_fr["cues_c"].to(DEVICE)
        evidence = exp_module.encoder(cues)
        avail = torch.ones(num_q, device=DEVICE)
        r_proxy = cached_fr["r_c_proxy"].to(DEVICE)
        n_proxy = cached_fr["n_c_proxy"].to(DEVICE)
        t_hist = None
    elif exp_key == "T":
        assert cached_temp_fr is not None, "[FATAL] Expert T requires active temporal history record!"
        match_conf = cached_temp_fr["match_confidence"].to(DEVICE)
        avail = (match_conf > 0.10).float()
        matched_b = cached_temp_fr["matched_boxes"].to(DEVICE)
        disp_x = (d_boxes[:, 0] - matched_b[:, 0]) * 5.0
        disp_y = (d_boxes[:, 1] - matched_b[:, 1]) * 5.0
        disp_mag = torch.sqrt(disp_x**2 + disp_y**2 + 1e-6)
        app_sim = cached_temp_fr["appearance_similarity"].to(DEVICE)

        r_proxy = torch.clamp(match_conf * app_sim, 0.1, 1.0) * avail
        n_proxy = torch.clamp(disp_mag + (1.0 - scores), 0.0, 1.0) * avail

        raw_cues_t = torch.stack([
            disp_x, disp_y, disp_mag, match_conf, app_sim, scores, n_proxy, r_proxy
        ], dim=-1) * avail.unsqueeze(-1)

        evidence = exp_module.encoder(raw_cues_t)
        t_hist = cached_temp_fr
    else:
        raise ValueError(f"Unknown expert key: {exp_key}")

    delta_q_box = avail.unsqueeze(-1) * exp_module.proj_box(evidence)
    delta_q_cls = avail.unsqueeze(-1) * exp_module.proj_cls(evidence)

    state_dummy = CanonicalObjectState(
        boxes=d_boxes,
        logits=d_logits,
        scores=scores,
        embeddings=q_native,
        reference_points=ref_pts,
        features=None,
        temporal_history=t_hist,
        image_shape=(640, 640),
        capabilities=substrate.adapter.capabilities,
        native_handle=(d_logits, d_boxes)
    )

    z_mod, b_mod = substrate.adapter.apply_contextual_prediction(
        state_dummy, q_native + delta_q_cls, q_native + delta_q_box
    )

    return z_mod, b_mod, delta_q_box, delta_q_cls, evidence, avail, r_proxy, n_proxy


def train_single_expert_fold(
    substrate: BFCEFSubstrate,
    official_criterion: nn.Module,
    exp_key: str,
    fold_idx: int,
    train_vids: List[int],
    epochs: int = EPOCHS_PER_EXPERT
) -> Dict[str, Any]:
    """
    Trains single expert in isolation on Train_f using cached shards and deterministic video-order shuffling.
    """
    t_exp_0 = time.time()
    exp_module = getattr(substrate, f"expert_{exp_key.lower()}")

    optimizer = torch.optim.AdamW(
        exp_module.parameters(),
        lr=LR_EXPERT,
        weight_decay=WEIGHT_DECAY
    )

    loss_history = []
    tot_avail_sum = 0.0
    tot_step_count = 0

    expert_seed_offset = {"S": 100, "L": 200, "C": 300, "T": 400}[exp_key]

    for epoch in range(1, epochs + 1):
        tot_loss = 0.0
        tot_det = 0.0
        tot_prot = 0.0
        tot_reg = 0.0
        n_steps = 0

        # Deterministic per-epoch video order shuffle (P-2319)
        rng = np.random.default_rng(
            E0_INIT_SEED + 10000 * fold_idx + expert_seed_offset + epoch
        )
        epoch_vids = list(train_vids)
        rng.shuffle(epoch_vids)

        for vid in epoch_vids:
            if exp_key == "T":
                video_records, temp_records = load_aligned_video_caches(vid)
            else:
                cache_path = CACHE_DIR / f"bfcef_e0_cache_vid{vid}.pth"
                video_records = torch.load(cache_path, map_location="cpu")
                temp_records = None

            for idx, cached_fr in enumerate(video_records):
                d_boxes = cached_fr["d_boxes"].to(DEVICE)
                d_logits = cached_fr["d_logits"].to(DEVICE)
                gt_boxes_norm = cached_fr["gt_boxes_norm"].to(DEVICE)
                gt_labels = cached_fr["gt_labels"].to(DEVICE)
                canonical_tp_mask = cached_fr["canonical_tp_mask"].to(DEVICE)

                cached_temp_fr = temp_records[idx] if temp_records is not None else None

                optimizer.zero_grad()

                z_mod, b_mod, dq_box, dq_cls, _, avail, _, _ = execute_cached_forward(
                    substrate, exp_key, cached_fr, cached_temp_fr
                )

                if exp_key == "T":
                    tot_avail_sum += float(avail.mean().item())

                # 1. Detection Loss
                targets = [{
                    "boxes": gt_boxes_norm,
                    "labels": gt_labels
                }]
                out_dict = {"pred_logits": z_mod.unsqueeze(0), "pred_boxes": b_mod.unsqueeze(0)}
                loss_dict = official_criterion(out_dict, targets)
                loss_det = sum(loss_dict[k] * official_criterion.weight_dict[k] for k in loss_dict if k in official_criterion.weight_dict)

                # 2. Canonical Baseline Protection Loss
                loss_prot = torch.tensor(0.0, device=DEVICE)
                if canonical_tp_mask.any():
                    loss_prot = F.l1_loss(b_mod[canonical_tp_mask], d_boxes[canonical_tp_mask]) + \
                                F.mse_loss(z_mod[canonical_tp_mask], d_logits[canonical_tp_mask])

                # 3. Active Delta Regularization
                loss_reg = (dq_box ** 2).mean() + (dq_cls ** 2).mean()

                loss_total = loss_det + 2.0 * loss_prot + 0.1 * loss_reg

                if loss_total.requires_grad and float(loss_total.item()) > 0.0:
                    loss_total.backward()
                    torch.nn.utils.clip_grad_norm_(exp_module.parameters(), max_norm=1.0)
                    optimizer.step()

                tot_loss += float(loss_total.item())
                tot_det += float(loss_det.item())
                tot_prot += float(loss_prot.item())
                tot_reg += float(loss_reg.item())
                n_steps += 1
                tot_step_count += 1

        mean_l = tot_loss / max(1, n_steps)
        loss_history.append(mean_l)
        print(f"      Epoch [{epoch}/{epochs}] | Loss: {mean_l:.4f} (Det: {tot_det/n_steps:.4f}, Prot: {tot_prot/n_steps:.4f}, Reg: {tot_reg/n_steps:.4f}) | Steps: {n_steps}")

    with torch.no_grad():
        final_box_norm = float(torch.norm(exp_module.proj_box.net[-1].weight).item())
        final_cls_norm = float(torch.norm(exp_module.proj_cls.net[-1].weight).item())

    elapsed = time.time() - t_exp_0
    print(f"    [+] Expert {exp_key} trained in {elapsed:.1f}s | ||W_box|| = {final_box_norm:.4f}, ||W_cls|| = {final_cls_norm:.4f} | Loss: {loss_history[0]:.4f} -> {loss_history[-1]:.4f}")

    return {
        "loss_history": [round(l, 4) for l in loss_history],
        "final_box_projector_norm": round(final_box_norm, 4),
        "final_cls_projector_norm": round(final_cls_norm, 4),
        "elapsed_seconds": round(elapsed, 2),
        "num_steps_per_epoch": n_steps,
        "mean_avail_fraction": round(float(tot_avail_sum / max(1, tot_step_count)), 4) if exp_key == "T" else 1.0
    }


def evaluate_and_stream_oof_interventions(
    substrate: BFCEFSubstrate,
    fold_idx: int,
    oof_vids: List[int],
    driver_sha: str
) -> Tuple[int, Dict[str, Any]]:
    """
    Evaluates OOF interventions and streams records PER VIDEO to disk (P-2318).
    Eliminates multi-GB RAM bloat; peak memory overhead <= 100 MB.
    """
    print(f"\n[*] Evaluating and Streaming OOF Interventions across {len(oof_vids)} Held-Out Videos for Fold {fold_idx}...")
    t_oof_0 = time.time()
    fold_record_count = 0
    oof_video_ledger_index = {}

    with torch.no_grad():
        for vid in oof_vids:
            video_records, temp_records = load_aligned_video_caches(vid)
            video_oof_records = []

            for idx, cached_fr in enumerate(video_records):
                cached_temp_fr = temp_records[idx]
                d_boxes = cached_fr["d_boxes"]
                d_logits = cached_fr["d_logits"]

                # 4 Single-Expert Counterfactual Interventions
                z_s, b_s, dq_b_s, dq_c_s, e_s, a_s, r_s, n_s = execute_cached_forward(substrate, "S", cached_fr, None)
                z_l, b_l, dq_b_l, dq_c_l, e_l, a_l, r_l, n_l = execute_cached_forward(substrate, "L", cached_fr, None)
                z_c, b_c, dq_b_c, dq_c_c, e_c, a_c, r_c, n_c = execute_cached_forward(substrate, "C", cached_fr, None)
                z_t, b_t, dq_b_t, dq_c_t, e_t, a_t, r_t, n_t = execute_cached_forward(substrate, "T", cached_fr, cached_temp_fr)

                video_oof_records.append({
                    "video_id": vid,
                    "image_id": cached_fr["image_id"],
                    "seq_idx": cached_fr["seq_idx"],
                    "fold_idx": fold_idx,
                    # Baseline R0
                    "z_base": d_logits.cpu(),
                    "b_base": d_boxes.cpu(),
                    # Single-Expert Outputs
                    "z_s": z_s.cpu(),
                    "b_s": b_s.cpu(),
                    "z_l": z_l.cpu(),
                    "b_l": b_l.cpu(),
                    "z_c": z_c.cpu(),
                    "b_c": b_c.cpu(),
                    "z_t": z_t.cpu(),
                    "b_t": b_t.cpu(),
                    # Expert Evidence & True Proxies (D-4507)
                    "evidence_s": e_s.cpu(),
                    "avail_s": a_s.cpu(),
                    "r_s_proxy": r_s.cpu(),
                    "n_s_proxy": n_s.cpu(),
                    "norm_dq_b_s": torch.norm(dq_b_s, dim=-1).cpu(),
                    "norm_dq_c_s": torch.norm(dq_c_s, dim=-1).cpu(),
                    "evidence_l": e_l.cpu(),
                    "avail_l": a_l.cpu(),
                    "r_l_proxy": r_l.cpu(),
                    "n_l_proxy": n_l.cpu(),
                    "norm_dq_b_l": torch.norm(dq_b_l, dim=-1).cpu(),
                    "norm_dq_c_l": torch.norm(dq_c_l, dim=-1).cpu(),
                    "evidence_c": e_c.cpu(),
                    "avail_c": a_c.cpu(),
                    "r_c_proxy": r_c.cpu(),
                    "n_c_proxy": n_c.cpu(),
                    "norm_dq_b_c": torch.norm(dq_b_c, dim=-1).cpu(),
                    "norm_dq_c_c": torch.norm(dq_c_c, dim=-1).cpu(),
                    "evidence_t": e_t.cpu(),
                    "avail_t": a_t.cpu(),
                    "r_t_proxy": r_t.cpu(),
                    "n_t_proxy": n_t.cpu(),
                    "norm_dq_b_t": torch.norm(dq_b_t, dim=-1).cpu(),
                    "norm_dq_c_t": torch.norm(dq_c_t, dim=-1).cpu(),
                    # Diagnostics & Ground Truth
                    "t_match_confidence": cached_temp_fr["match_confidence"].cpu(),
                    "t_appearance_similarity": cached_temp_fr["appearance_similarity"].cpu(),
                    "scene_q": cached_fr["scene_q"],
                    "gt_boxes_norm": cached_fr["gt_boxes_norm"],
                    "gt_labels": cached_fr["gt_labels"],
                    "canonical_tp_mask": cached_fr["canonical_tp_mask"]
                })

            # Stream save immediately per video
            vid_oof_file = E0_WORK_DIR / f"bfcef_e0_oof_fold{fold_idx}_vid{vid}.pth"
            torch.save(video_oof_records, vid_oof_file)
            vid_sha = compute_sha256(vid_oof_file)

            oof_video_ledger_index[str(vid)] = {
                "file": vid_oof_file.name,
                "sha256": vid_sha,
                "num_records": len(video_oof_records)
            }
            fold_record_count += len(video_oof_records)

            # Unload from RAM immediately
            del video_oof_records
            del video_records
            del temp_records
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"    [+] Evaluated & streamed {fold_record_count:,} OOF frames across {len(oof_vids)} videos in {time.time() - t_oof_0:.1f}s")
    return fold_record_count, oof_video_ledger_index


def main():
    t_global_0 = time.time()
    driver_sha = compute_sha256(Path(__file__))

    print("=" * 115)
    print("MARSEN B-FCEF: STAGE E0 - CROSS-FITTED CONTEXT ADAPTER TRAINING & OOF LEDGER")
    print(f"Driver Script SHA-256: {driver_sha}")
    print(f"Target Device:         {DEVICE}")
    print(f"Deterministic Seed:    {E0_INIT_SEED}")
    print("=" * 115, flush=True)

    # 1. Cryptographic Provenance Preflight Gate (P-2314)
    print("[*] Checking Cryptographic Provenance across Census, Train JSON, Checkpoint, Caches...")
    assert CENSUS_JSON.exists(), f"[FATAL] Missing census JSON at {CENSUS_JSON}"
    assert compute_sha256(CENSUS_JSON) == FROZEN_CENSUS_SHA256, "[FATAL] Census SHA mismatch"
    assert compute_sha256(R0_CHECKPOINT_PATH) == FROZEN_R0_CKPT_SHA256, "[FATAL] Checkpoint SHA mismatch"
    assert CACHE_MANIFEST_PATH.exists(), f"[FATAL] Missing cache manifest: {CACHE_MANIFEST_PATH}"
    assert TEMP_CACHE_MANIFEST_PATH.exists(), f"[FATAL] Missing temporal cache manifest: {TEMP_CACHE_MANIFEST_PATH}"

    canonical_manifest_sha = compute_sha256(CACHE_MANIFEST_PATH)
    temporal_manifest_sha = compute_sha256(TEMP_CACHE_MANIFEST_PATH)

    with open(CENSUS_JSON, "r", encoding="utf-8") as f:
        census = json.load(f)
    train_json_sha = census["_meta"]["live_provenance"]["train_json_sha256"]
    assert compute_sha256(TRAIN_JSON) == train_json_sha, "[FATAL] Train JSON drift"

    current_substrate_sha = compute_sha256(SUBSTRATE_SCRIPT_PATH)
    with open(CACHE_MANIFEST_PATH, "r", encoding="utf-8") as f:
        cache_manifest = json.load(f)

    assert cache_manifest["_meta"]["substrate_sha256"] == current_substrate_sha
    assert cache_manifest["_meta"]["r0_checkpoint_sha256"] == FROZEN_R0_CKPT_SHA256
    assert cache_manifest["_meta"]["census_sha256"] == FROZEN_CENSUS_SHA256
    assert cache_manifest["_meta"]["train_json_sha256"] == train_json_sha
    assert cache_manifest["_meta"]["actual_total_frames"] == EXPECTED_TOTAL_FRAMES
    assert cache_manifest["_meta"]["max_live_vs_shard_delta_logits"] < 1e-4
    assert cache_manifest["_meta"]["max_live_vs_shard_delta_boxes"] < 1e-4

    with open(TEMP_CACHE_MANIFEST_PATH, "r", encoding="utf-8") as f:
        temp_cache_manifest = json.load(f)

    assert temp_cache_manifest["_meta"]["input_cache_manifest_sha256"] == canonical_manifest_sha, (
        "[FATAL] Temporal cache input hash mismatch against canonical cache manifest!"
    )
    assert temp_cache_manifest["_meta"]["num_videos"] == NUM_VIDEOS
    assert temp_cache_manifest["_meta"]["actual_total_frames"] == EXPECTED_TOTAL_FRAMES
    assert temp_cache_manifest["_meta"]["association_version"] == ASSOCIATION_VERSION

    # Verify all 47 canonical and 47 temporal cache shards
    for filename, expected_sha in cache_manifest["cache_shard_sha256_ledger"].items():
        p = CACHE_DIR / filename
        assert p.exists() and compute_sha256(p) == expected_sha, f"[FATAL] Drift on canonical cache {filename}"

    for filename, expected_sha in temp_cache_manifest["temporal_cache_shard_sha256_ledger"].items():
        p = CACHE_DIR / filename
        assert p.exists() and compute_sha256(p) == expected_sha, f"[FATAL] Drift on temporal cache {filename}"

    print(f"    [+] Provenance certified: 47 canonical + 47 temporal cache shards verified.")
    print(f"        Canonical Manifest: {canonical_manifest_sha[:16]}...")
    print(f"        Temporal Manifest:  {temporal_manifest_sha[:16]}...\n")

    # 2. Load Support-Aware Stratified 5-Fold Manifest (P-2316)
    folds, fold_manifest_sha = load_stratified_folds(canonical_manifest_sha)

    # 3. Set Deterministic Initialization Seed
    torch.manual_seed(E0_INIT_SEED)
    np.random.seed(E0_INIT_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(E0_INIT_SEED)

    # 4. Build R0 Native Heads & Official Detection Criterion
    print("[*] Building Authoritative R0 Model & Official SetCriterion...")
    args = load_authoritative_r0_args()
    args.device = str(DEVICE)
    from models.deformable_detr import build as build_deformable_detr

    r0_model, official_criterion, _ = build_deformable_detr(args)
    r0_ckpt = torch.load(R0_CHECKPOINT_PATH, map_location="cpu")
    sd = r0_ckpt.get("model", r0_ckpt)
    cleaned_sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    r0_model.load_state_dict(cleaned_sd, strict=True)
    r0_model.to(DEVICE)
    r0_model.eval()
    for p in r0_model.parameters():
        p.requires_grad_(False)
    official_criterion.to(DEVICE)

    adapter = AuthenticDeformableDETRAdapter(r0_model.class_embed[-1], r0_model.bbox_embed[-1]).to(DEVICE)
    substrate = BFCEFSubstrate(adapter=adapter).to(DEVICE)

    # 5. Snapshot Pristine Zero-Start State
    substrate.init_zero_projectors()
    pristine_zero_state = copy.deepcopy({
        k: v.detach().clone() for k, v in substrate.state_dict().items()
        if not k.startswith("adapter.")
    })
    print(f"[*] Snapshotted pristine zero-start parameter state (Seed: {E0_INIT_SEED}).\n")

    # 6. Streamed 5-Fold Cross-Fitting Loop
    total_oof_records = 0
    fold_checkpoint_hashes = {}
    master_fold_ledger_index = {}
    fold_training_metrics = {}

    for fold_idx in range(NUM_FOLDS):
        oof_vids = folds[fold_idx]
        train_vids = [v for f, vids in folds.items() if f != fold_idx for v in vids]

        print("=" * 115)
        print(f"STAGE E0 CROSS-FITTING: FOLD [{fold_idx + 1}/{NUM_FOLDS}]")
        print(f"Train Videos ({len(train_vids)}): {train_vids}")
        print(f"Held-Out OOF ({len(oof_vids)}): {oof_vids}")
        print("=" * 115)

        # Enforce Pristine Fold Reset
        substrate.load_state_dict(pristine_zero_state, strict=False)
        substrate.init_zero_projectors()

        current_state = substrate.state_dict()
        for k, v_ref in pristine_zero_state.items():
            assert torch.equal(current_state[k], v_ref), f"[FATAL] Reset failed on {k} before Fold {fold_idx}!"
        print(f"    [+] Fold {fold_idx} reset verified: 100% parameter equality to pristine snapshot.")

        # Train each of the 4 invariant experts in isolation
        fold_metrics = {}
        for exp_key in ["S", "L", "C", "T"]:
            print(f"    [*] Training Expert {exp_key} (Fold {fold_idx})...")
            exp_res = train_single_expert_fold(
                substrate, official_criterion, exp_key, fold_idx, train_vids, epochs=EPOCHS_PER_EXPERT
            )
            fold_metrics[exp_key] = exp_res

        fold_training_metrics[f"fold_{fold_idx}"] = fold_metrics

        # Evaluate and Stream OOF Interventions Per Video
        fold_record_count, oof_video_index = evaluate_and_stream_oof_interventions(
            substrate, fold_idx, oof_vids, driver_sha
        )
        total_oof_records += fold_record_count
        master_fold_ledger_index[f"fold_{fold_idx}"] = {
            "oof_vids": oof_vids,
            "total_records": fold_record_count,
            "videos": oof_video_index
        }

        # Save Fold Adapter Checkpoint with Training Metrics
        fold_ckpt_file = E0_WORK_DIR / f"bfcef_e0_adapters_fold{fold_idx}.pth"
        torch.save({
            "_meta": {
                "fold_idx": fold_idx,
                "train_vids": train_vids,
                "oof_vids": oof_vids,
                "driver_sha256": driver_sha,
                "init_seed": E0_INIT_SEED,
                "fold_manifest_sha256": fold_manifest_sha,
                "canonical_cache_manifest_sha256": canonical_manifest_sha,
                "temporal_cache_manifest_sha256": temporal_manifest_sha,
                "training_metrics": fold_metrics,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "expert_state_dict": {
                k: v.cpu() for k, v in substrate.state_dict().items()
                if not k.startswith("adapter.")
            }
        }, fold_ckpt_file)

        fold_sha = compute_sha256(fold_ckpt_file)
        fold_checkpoint_hashes[fold_ckpt_file.name] = fold_sha
        print(f"    [+] Fold {fold_idx} Checkpoint Saved: {fold_ckpt_file.name} (SHA: {fold_sha[:16]}...)\n")

    # 7. Final Master Index
    assert total_oof_records == EXPECTED_TOTAL_FRAMES, f"[FATAL] Count mismatch: {total_oof_records} != {EXPECTED_TOTAL_FRAMES}!"

    master_ledger_path = E0_WORK_DIR / "bfcef_e0_oof_ledger_index.json"
    master_index = {
        "_meta": {
            "driver_sha256": driver_sha,
            "init_seed": E0_INIT_SEED,
            "fold_manifest_sha256": fold_manifest_sha,
            "canonical_cache_manifest_sha256": canonical_manifest_sha,
            "temporal_cache_manifest_sha256": temporal_manifest_sha,
            "total_frames": total_oof_records,
            "num_folds": NUM_FOLDS,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "folds": master_fold_ledger_index
    }
    with open(master_ledger_path, "w", encoding="utf-8") as f:
        json.dump(master_index, f, indent=2)

    master_ledger_sha = compute_sha256(master_ledger_path)

    # 8. Training Summary Report
    summary_path = E0_WORK_DIR / "bfcef_e0_training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "_meta": {
                "driver_sha256": driver_sha,
                "init_seed": E0_INIT_SEED,
                "master_ledger_index_sha256": master_ledger_sha,
                "fold_manifest_sha256": fold_manifest_sha,
                "canonical_cache_manifest_sha256": canonical_manifest_sha,
                "temporal_cache_manifest_sha256": temporal_manifest_sha,
                "total_oof_records": total_oof_records,
                "num_folds": NUM_FOLDS,
                "execution_time_minutes": round((time.time() - t_global_0) / 60.0, 2),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "fold_checkpoints": fold_checkpoint_hashes,
            "training_metrics_by_fold": fold_training_metrics,
            "fold_manifest": {str(k): v for k, v in folds.items()}
        }, f, indent=2)

    print("=" * 115)
    print(f"[+] STAGE E0 5-FOLD CROSS-FITTING COMPLETED IN {(time.time() - t_global_0) / 60.0:.2f} MINUTES")
    print(f"[+] Total OOF Intervention Records: {total_oof_records:,} / {EXPECTED_TOTAL_FRAMES:,} (100.0% EXACT COVERAGE)")
    print(f"[+] Master OOF Ledger Index:        {master_ledger_path.resolve()} (SHA: {master_ledger_sha})")
    print(f"[+] Training Summary Report:        {summary_path.resolve()}")
    print("=" * 115 + "\n", flush=True)


if __name__ == "__main__":
    main()