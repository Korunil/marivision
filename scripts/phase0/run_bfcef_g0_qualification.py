"""
D:/MARIVISION/H0_workbench/eval_tools/run_bfcef_g0_qualification.py
MARSEN B-FCEF: Stage G0 - Out-of-Fold Counterfactual Qualification Engine
(D-4559..D-4585, P-2328..P-2334)

Locked Implementation Contract:
  1. Base Oracle Preflight: Evaluates Base alone across all 20,075 frames.
     Hard-asserts GT=64,670, TP=50,741, FP=52,801, FN=13,929, F1=0.603298,
     mAP@0.30=0.731700 (within 1e-4) before processing any expert.
  2. Sequential Evaluator Management: Evaluates S, L, C, T sequentially, capping
     memory at two active CocoEvaluator instances (class-aware and class-agnostic).
  3. Evaluator Synchronization: Calls ev.synchronize_between_processes() before accumulate().
  4. Top-100 Policy: Global Top-100 across flattened [300x7] query-class pairs.
     COCO evaluator receives all Top-100; operational evaluation applies score >= 0.20.
  5. Category ID & Native Coordinate Scaling: Vectorized category ID mapping;
     scales boxes using native image (width, height) from seaclips-train.json.
  6. SeaClips Metric Protocol: Evaluates mAP@0.30, mAPca@0.30, AP_S/M/L@0.30, per-class AP@0.30.
  7. Strict Camera Metadata: Uses videos[].camera_sensor ('e-CAM', 'FLIR', 'Axis').
  8. Canonical Outcome Transitions: Ground-truth Rescue, Loss, Retained transitions
     evaluated via the canonical operational matcher with exact conservation assertions.
  9. Exact Operational FP Transitions: Categorizes FPs via (q_idx, c_idx) set operations:
     fp_removed = Base_FP - Exp_FP, fp_added = Exp_FP - Base_FP.
 10. Image-Keyed G1 Event Ledger: Streams per-video event files (bfcef_g0_events_fold{f}_vid{vid}.pth)
     and master index (bfcef_g0_events_index.json) containing unrounded raw event identities.
 11. Complete Provenance Chain: Validates master index, training summary, fold manifest,
     canonical cache, temporal cache, and Train JSON SHA-256 hashes.
 12. Full Metric Persistence: Persists mAP, mAPca, AP_S, AP_M, AP_L, per-class AP@0.30,
     retained-TP delta distributions (mean, p10, p50, p90), and report SHA-256.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
from pathlib import Path

ROOT_DIR = Path("D:/MARIVISION")
OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
E0_WORK_DIR = OUT_DIR / "e0_adapters"
G0_WORK_DIR = OUT_DIR / "g0_qualification"
EVENTS_DIR = G0_WORK_DIR / "events"
EVENTS_DIR.mkdir(parents=True, exist_ok=True)

MASTER_INDEX_PATH = E0_WORK_DIR / "bfcef_e0_oof_ledger_index.json"
TRAINING_SUMMARY_PATH = E0_WORK_DIR / "bfcef_e0_training_summary.json"
MANIFEST_JSON = E0_WORK_DIR / "bfcef_e0_5fold_manifest.json"
CACHE_MANIFEST_PATH = OUT_DIR / "e0_cache" / "bfcef_e0_cache_manifest.json"
TEMP_CACHE_MANIFEST_PATH = OUT_DIR / "e0_cache" / "bfcef_e0_temporal_cache_manifest.json"
TRAIN_JSON = ROOT_DIR / "seaclips-train.json"

REPO_CANDIDATES = [ROOT_DIR / "Deformable-DETR", ROOT_DIR / "deformable_detr"]
DEFORMABLE_DETR_ROOT = None
for candidate in REPO_CANDIDATES:
    if (
        candidate.exists()
        and (candidate / "main.py").exists()
        and (candidate / "models" / "deformable_detr.py").exists()
        and (candidate / "datasets" / "coco_eval.py").exists()
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
from typing import Dict, Any, List, Tuple, Set

import numpy as np
import torch
import torch.nn.functional as F

from run_bfcef_e0_substrate import compute_sha256, NUM_CLASSES
from datasets.coco_eval import CocoEvaluator
from pycocotools.coco import COCO

sys.stdout.reconfigure(line_buffering=True)
DEVICE = torch.device("cpu")  # CPU guarantees stable memory ceilings and zero VRAM pressure

EXPERTS = ["S", "L", "C", "T"]
CAMERAS = ["e-CAM", "FLIR", "Axis"]
NUM_FOLDS = 5
EXPECTED_TOTAL_FRAMES = 20075


def box_cxcywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def box_iou_norm_cxcywh(b1, b2) -> float:
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


def extract_top100_predictions(logits: torch.Tensor, boxes: torch.Tensor):
    """
    Certified canonical ranking:
      prob = logits.sigmoid() -> flatten [300x7] -> global Top-100 query-class entries.
    """
    probs = logits.sigmoid()
    flat_probs = probs.flatten()
    topk_scores, topk_idx = torch.topk(flat_probs, min(100, flat_probs.numel()))

    q_idx = topk_idx // NUM_CLASSES
    c_idx = topk_idx % NUM_CLASSES
    topk_boxes = boxes[q_idx]

    valid_op = topk_scores >= 0.20

    return {
        "all_scores": topk_scores,
        "all_q_idx": q_idx,
        "all_c_idx": c_idx,
        "all_boxes": topk_boxes,
        "op_scores": topk_scores[valid_op],
        "op_q_idx": q_idx[valid_op],
        "op_c_idx": c_idx[valid_op],
        "op_boxes": topk_boxes[valid_op],
    }


def match_operational_detections_to_gt(
    op_scores: torch.Tensor,
    op_c_idx: torch.Tensor,
    op_boxes_norm: torch.Tensor,
    op_q_idx: torch.Tensor,
    gt_boxes_norm: torch.Tensor,
    gt_labels: torch.Tensor
) -> Dict[str, Any]:
    """
    Canonical operational matcher:
      Top-100 entries with score >= 0.20 matched greedily to same-class GT with IoU >= 0.30.
    """
    num_gt = gt_boxes_norm.shape[0]
    num_det = op_scores.shape[0]

    matched_gt_indices: Set[int] = set()
    matched_det_indices: Set[int] = set()
    gt_to_det: Dict[int, Tuple[int, float, float, int, int]] = {}

    if num_gt > 0 and num_det > 0:
        gt_boxes_list = gt_boxes_norm.tolist()
        gt_labels_list = gt_labels.tolist()
        det_boxes_list = op_boxes_norm.tolist()
        det_classes_list = op_c_idx.tolist()
        det_scores_list = op_scores.tolist()
        det_q_list = op_q_idx.tolist()

        for d_idx in range(num_det):
            c_idx = det_classes_list[d_idx]
            det_box = det_boxes_list[d_idx]
            score = det_scores_list[d_idx]
            q_idx = det_q_list[d_idx]

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
                matched_det_indices.add(d_idx)
                gt_to_det[best_gt_idx] = (d_idx, best_iou, score, q_idx, c_idx)

    # False Positives with exact (q_idx, c_idx) proposal identity
    fp_dict = {}
    for d_idx in range(num_det):
        if d_idx not in matched_det_indices:
            key = (int(op_q_idx[d_idx].item()), int(op_c_idx[d_idx].item()))
            fp_dict[key] = {
                "score": float(op_scores[d_idx].item()),
                "box": [float(x) for x in op_boxes_norm[d_idx].tolist()]
            }

    tp_count = len(matched_gt_indices)
    fp_count = len(fp_dict)
    fn_count = num_gt - tp_count

    return {
        "matched_gt_indices": matched_gt_indices,
        "matched_det_indices": matched_det_indices,
        "gt_to_det": gt_to_det,
        "fp_dict": fp_dict,
        "tp": tp_count,
        "fp": fp_count,
        "fn": fn_count,
        "num_det": num_det,
        "num_gt": num_gt
    }


def main():
    t_start = time.time()
    driver_sha = compute_sha256(Path(__file__))

    print("=" * 115)
    print("MARSEN B-FCEF: STAGE G0 - OUT-OF-FOLD COUNTERFACTUAL QUALIFICATION ENGINE")
    print(f"Driver Script SHA-256: {driver_sha}")
    print(f"Master Index:          {MASTER_INDEX_PATH}")
    print("=" * 115)

    # 1. Complete Cryptographic Provenance Chain Verification (Contract Item 11)
    print("[*] Verifying complete cryptographic provenance chain across all manifests...")
    assert MASTER_INDEX_PATH.exists(), f"[FATAL] Missing master index: {MASTER_INDEX_PATH}"
    assert TRAINING_SUMMARY_PATH.exists(), f"[FATAL] Missing training summary: {TRAINING_SUMMARY_PATH}"
    assert MANIFEST_JSON.exists(), f"[FATAL] Missing fold manifest: {MANIFEST_JSON}"
    assert CACHE_MANIFEST_PATH.exists(), f"[FATAL] Missing canonical cache manifest: {CACHE_MANIFEST_PATH}"
    assert TEMP_CACHE_MANIFEST_PATH.exists(), f"[FATAL] Missing temporal cache manifest: {TEMP_CACHE_MANIFEST_PATH}"
    assert TRAIN_JSON.exists(), f"[FATAL] Missing seaclips-train.json: {TRAIN_JSON}"

    computed_master_index_sha = compute_sha256(MASTER_INDEX_PATH)
    with open(TRAINING_SUMMARY_PATH, "r", encoding="utf-8") as f:
        training_summary = json.load(f)

    expected_master_index_sha = training_summary["_meta"]["master_ledger_index_sha256"]
    assert computed_master_index_sha == expected_master_index_sha, (
        f"[FATAL] Master index SHA mismatch: {computed_master_index_sha} != {expected_master_index_sha}"
    )

    fold_manifest_sha = compute_sha256(MANIFEST_JSON)
    canonical_manifest_sha = compute_sha256(CACHE_MANIFEST_PATH)
    temporal_manifest_sha = compute_sha256(TEMP_CACHE_MANIFEST_PATH)
    train_json_sha = compute_sha256(TRAIN_JSON)

    assert training_summary["_meta"]["fold_manifest_sha256"] == fold_manifest_sha
    assert training_summary["_meta"]["canonical_cache_manifest_sha256"] == canonical_manifest_sha
    assert training_summary["_meta"]["temporal_cache_manifest_sha256"] == temporal_manifest_sha

    with open(CACHE_MANIFEST_PATH, "r", encoding="utf-8") as f:
        canonical_manifest = json.load(f)
    assert canonical_manifest["_meta"]["train_json_sha256"] == train_json_sha

    with open(MASTER_INDEX_PATH, "r", encoding="utf-8") as f:
        master_index = json.load(f)
    assert master_index["_meta"]["total_frames"] == EXPECTED_TOTAL_FRAMES

    print(f"    [+] Master Ledger Index SHA:   {computed_master_index_sha[:16]}...")
    print(f"    [+] Fold Manifest SHA:         {fold_manifest_sha[:16]}...")
    print(f"    [+] Canonical Cache SHA:       {canonical_manifest_sha[:16]}...")
    print(f"    [+] Temporal Cache SHA:        {temporal_manifest_sha[:16]}...")
    print(f"    [+] Train JSON SHA:            {train_json_sha[:16]}...\n")

    # 2. Ingest Metadata & Build Vectorized Label Mappings (Contract Items 2, 5, 7)
    print("[*] Ingesting COCO metadata from seaclips-train.json...")
    coco_gt = COCO(str(TRAIN_JSON))

    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        train_coco = json.load(f)

    img_native_size = {
        int(im["id"]): (float(im["width"]), float(im["height"]))
        for im in train_coco["images"]
    }
    img_to_video = {
        int(im["id"]): int(im["video_id"])
        for im in train_coco["images"]
    }

    video_to_camera = {}
    for v in train_coco["videos"]:
        vid = int(v.get("id", v.get("video_id")))
        sensor = str(v["camera_sensor"]).strip()
        assert sensor in CAMERAS, f"[FATAL] Unknown camera sensor: {sensor}"
        video_to_camera[vid] = sensor

    cat_ids = sorted(coco_gt.getCatIds())
    assert len(cat_ids) == NUM_CLASSES
    cat_id_to_name = {c["id"]: c["name"] for c in train_coco["categories"]}
    class_names = [cat_id_to_name[cid] for cid in cat_ids]

    # Vectorized category tensor for non-synchronizing label conversion (Contract Item 9)
    cat_id_tensor = torch.tensor(cat_ids, dtype=torch.long, device=DEVICE)

    # Class-Agnostic COCO ground truth
    coco_gt_ca = copy.deepcopy(coco_gt)
    coco_gt_ca.dataset["categories"] = [{"id": 1, "name": "object"}]
    for ann in coco_gt_ca.dataset["annotations"]:
        ann["category_id"] = 1
    coco_gt_ca.createIndex()

    # Pre-read video shard metadata and derive authoritative video_ids
    video_shard_list = []
    seen_image_ids = set()
    total_shard_records = 0

    for fold_name, fold_info in master_index["folds"].items():
        fold_idx = int(fold_name.replace("fold_", ""))
        for vid_str, v_meta in fold_info["videos"].items():
            vid = int(vid_str)
            v_file = E0_WORK_DIR / v_meta["file"]
            assert v_file.exists(), f"[FATAL] Missing shard: {v_file.name}"
            assert compute_sha256(v_file) == v_meta["sha256"], f"[FATAL] Shard drift: {v_file.name}"
            video_shard_list.append((fold_idx, vid, v_file, v_meta["num_records"]))
            total_shard_records += v_meta["num_records"]

    assert len(video_shard_list) == 47
    assert total_shard_records == EXPECTED_TOTAL_FRAMES

    # Authoritative derived video_ids inventory (Contract Item 1)
    video_ids = sorted({vid for _, vid, _, _ in video_shard_list})
    assert len(video_ids) == 47, f"[FATAL] Expected 47 videos, found {len(video_ids)}"
    assert video_ids == list(range(1, 48)), f"[FATAL] video_ids not exactly 1..47: {video_ids}"
    assert set(video_to_camera.keys()) == set(video_ids), "[FATAL] Camera metadata video ID mismatch"

    expected_records_per_vid = {
        vid: num_records for _, vid, _, num_records in video_shard_list
    }

    # =========================================================================
    # PASS A: STANDALONE BASELINE ORACLE PREFLIGHT (Contract Items 1, 3)
    # =========================================================================
    print("=" * 115)
    print("PASS A: STANDALONE BASELINE ORACLE PREFLIGHT")
    print("=" * 115)

    base_evaluator = CocoEvaluator(coco_gt, ["bbox"])
    base_evaluator_ca = CocoEvaluator(coco_gt_ca, ["bbox"])
    base_evaluator.coco_eval["bbox"].params.iouThrs = np.array([0.30])
    base_evaluator_ca.coco_eval["bbox"].params.iouThrs = np.array([0.30])

    base_frame_match_cache: Dict[int, Dict[str, Any]] = {}
    events_by_video: Dict[int, Dict[int, Dict[str, Any]]] = {vid: {} for vid in video_ids}

    base_tp_total = 0
    base_fp_total = 0
    base_fn_total = 0
    base_gt_total = 0

    gt_stratum_totals = {
        "total_gt": 0,
        "by_fold": {f: {"gt": 0, "base_tp": 0, "base_fn": 0} for f in range(NUM_FOLDS)},
        "by_cam": {c: {"gt": 0, "base_tp": 0, "base_fn": 0} for c in CAMERAS},
        "by_class": {cls: {"gt": 0, "base_tp": 0, "base_fn": 0} for cls in range(NUM_CLASSES)}
    }

    t_base_0 = time.time()

    for fold_idx, vid, v_file, num_records in video_shard_list:
        records = torch.load(v_file, map_location="cpu")
        assert len(records) == num_records, f"[FATAL] Record count mismatch in {v_file.name}"
        cam_name = video_to_camera[vid]

        v_preds = {}
        v_preds_ca = {}

        for fr in records:
            img_id = int(fr["image_id"])
            assert int(fr["video_id"]) == vid
            assert int(fr["fold_idx"]) == fold_idx
            assert img_to_video[img_id] == vid
            assert img_id not in seen_image_ids, f"[FATAL] Duplicate image_id {img_id}"
            seen_image_ids.add(img_id)

            # Initialize event ledger metadata during Base pass (Contract Item 2)
            events_by_video[vid][img_id] = {
                "image_id": img_id,
                "seq_idx": int(fr["seq_idx"]),
                "fold_idx": fold_idx,
                "camera": cam_name,
                "events": {}
            }

            img_w, img_h = img_native_size[img_id]
            gt_boxes_norm = fr["gt_boxes_norm"]
            gt_labels = fr["gt_labels"]
            num_gt = gt_boxes_norm.shape[0]

            # GT Stratum Accounting (Fixed: increments strictly by 1 per GT instance)
            gt_stratum_totals["total_gt"] += num_gt
            gt_stratum_totals["by_fold"][fold_idx]["gt"] += num_gt
            gt_stratum_totals["by_cam"][cam_name]["gt"] += num_gt
            for g_idx in range(num_gt):
                c_gt = int(gt_labels[g_idx].item())
                gt_stratum_totals["by_class"][c_gt]["gt"] += 1

            base_preds = extract_top100_predictions(fr["z_base"], fr["b_base"])

            # Format COCO predictions with native scaling
            boxes_xyxy_norm = box_cxcywh_to_xyxy(base_preds["all_boxes"])
            scale = torch.tensor([img_w, img_h, img_w, img_h])
            scaled_boxes = boxes_xyxy_norm * scale
            cat_labels = cat_id_tensor[base_preds["all_c_idx"]]

            v_preds[img_id] = {
                "boxes": scaled_boxes,
                "scores": base_preds["all_scores"],
                "labels": cat_labels
            }
            v_preds_ca[img_id] = {
                "boxes": scaled_boxes,
                "scores": base_preds["all_scores"],
                "labels": torch.ones_like(cat_labels)
            }

            # Canonical operational matching
            b_match = match_operational_detections_to_gt(
                base_preds["op_scores"],
                base_preds["op_c_idx"],
                base_preds["op_boxes"],
                base_preds["op_q_idx"],
                gt_boxes_norm,
                gt_labels
            )

            base_tp_total += b_match["tp"]
            base_fp_total += b_match["fp"]
            base_fn_total += b_match["fn"]
            base_gt_total += num_gt

            base_matched_set = b_match["matched_gt_indices"]
            b_tp_cnt = len(base_matched_set)
            gt_stratum_totals["by_fold"][fold_idx]["base_tp"] += b_tp_cnt
            gt_stratum_totals["by_fold"][fold_idx]["base_fn"] += (num_gt - b_tp_cnt)
            gt_stratum_totals["by_cam"][cam_name]["base_tp"] += b_tp_cnt
            gt_stratum_totals["by_cam"][cam_name]["base_fn"] += (num_gt - b_tp_cnt)

            for g_idx in range(num_gt):
                c_gt = int(gt_labels[g_idx].item())
                if g_idx in base_matched_set:
                    gt_stratum_totals["by_class"][c_gt]["base_tp"] += 1
                else:
                    gt_stratum_totals["by_class"][c_gt]["base_fn"] += 1

            # Cache lightweight operational match results for expert pass
            base_frame_match_cache[img_id] = {
                "matched_gt_indices": b_match["matched_gt_indices"],
                "gt_to_det": b_match["gt_to_det"],
                "fp_dict": b_match["fp_dict"],
                "tp": b_match["tp"],
                "fp": b_match["fp"],
                "fn": b_match["fn"],
                "num_gt": num_gt
            }

        # Update evaluators once per video
        base_evaluator.update(v_preds)
        base_evaluator_ca.update(v_preds_ca)
        del records

    assert len(seen_image_ids) == EXPECTED_TOTAL_FRAMES, f"[FATAL] Processed {len(seen_image_ids)} != {EXPECTED_TOTAL_FRAMES}"
    assert gt_stratum_totals["total_gt"] == 64670
    assert sum(gt_stratum_totals["by_fold"][f]["gt"] for f in range(NUM_FOLDS)) == 64670
    assert sum(gt_stratum_totals["by_cam"][c]["gt"] for c in CAMERAS) == 64670
    assert sum(gt_stratum_totals["by_class"][c]["gt"] for c in range(NUM_CLASSES)) == 64670

    # Synchronize before accumulate (Contract Item 3)
    base_evaluator.synchronize_between_processes()
    base_evaluator.accumulate()
    base_evaluator.summarize()

    base_evaluator_ca.synchronize_between_processes()
    base_evaluator_ca.accumulate()
    base_evaluator_ca.summarize()

    base_stats = base_evaluator.coco_eval["bbox"].stats
    base_stats_ca = base_evaluator_ca.coco_eval["bbox"].stats
    base_map03 = float(base_stats[0])
    base_ca03 = float(base_stats_ca[0])
    base_f1 = 2.0 * base_tp_total / max(1, (2.0 * base_tp_total + base_fp_total + base_fn_total))

    # Per-class AP@0.30 extraction
    prec_base = base_evaluator.coco_eval["bbox"].eval["precision"]
    base_per_class_ap = []
    for k in range(NUM_CLASSES):
        p_k = prec_base[0, :, k, 0, 2]
        valid_p = p_k[p_k > -1]
        base_per_class_ap.append(float(np.mean(valid_p)) if len(valid_p) > 0 else 0.0)

    # -------------------------------------------------------------------------
    # HARD BASELINE IDENTITY GATE (Contract Item 1)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("CANONICAL BASELINE ORACLE IDENTITY GATE (FROZEN 20,075-FRAME R0 GROUNDING)")
    print("=" * 115)
    print(f"  Metric              | Actual Observed   | Frozen Certified Oracle | Status")
    print("-" * 115)
    print(f"  Ground-Truth (GT)   | {base_gt_total:>17,} | {64670:>23,} | {'PASS' if base_gt_total == 64670 else 'FAIL'}")
    print(f"  True Positives (TP) | {base_tp_total:>17,} | {50741:>23,} | {'PASS' if base_tp_total == 50741 else 'FAIL'}")
    print(f"  False Positives(FP) | {base_fp_total:>17,} | {52801:>23,} | {'PASS' if base_fp_total == 52801 else 'FAIL'}")
    print(f"  False Negatives(FN) | {base_fn_total:>17,} | {13929:>23,} | {'PASS' if base_fn_total == 13929 else 'FAIL'}")
    print(f"  F1 Score            | {base_f1:>17.6f} | {0.603298:>23.6f} | {'PASS' if abs(base_f1 - 0.603298) < 1e-5 else 'FAIL'}")
    print(f"  mAP @ 0.30          | {base_map03:>17.6f} | {0.731700:>23.6f} | {'PASS' if abs(base_map03 - 0.731700) <= 1e-4 else 'FAIL'}")
    print("=" * 115)

    assert base_gt_total == 64670, f"[FATAL] Base GT count mismatch: {base_gt_total} != 64670"
    assert base_tp_total == 50741, f"[FATAL] Base TP count mismatch: {base_tp_total} != 50741"
    assert base_fp_total == 52801, f"[FATAL] Base FP count mismatch: {base_fp_total} != 52801"
    assert base_fn_total == 13929, f"[FATAL] Base FN count mismatch: {base_fn_total} != 13929"
    assert abs(base_f1 - 0.603298) < 1e-5, f"[FATAL] Base F1 drift: {base_f1} vs 0.603298"
    assert abs(base_map03 - 0.731700) <= 1e-4, f"[FATAL] Base mAP@0.30 drift: {base_map03} vs 0.731700"

    print(f"[+] BASE ORACLE: PASS (Evaluated in {time.time() - t_base_0:.1f}s)")
    print("[*] Baseline identity certified. Releasing Base evaluator RAM and entering Sequential Expert Evaluation...\n")

    # Free Base evaluators immediately to cap resident memory (Contract Item 2)
    del base_evaluator
    del base_evaluator_ca
    gc.collect()

    # =========================================================================
    # PASS B: SEQUENTIAL EXPERT EVALUATION & G1 EVENT LEDGER GENERATION
    # =========================================================================
    expert_coco_stats = {}
    expert_coco_stats_ca = {}
    expert_per_class_ap = {}
    expert_op_counts = {}

    transition_records = {
        exp: {
            "rescued": 0,
            "lost": 0,
            "retained": 0,
            "fp_added": 0,
            "fp_removed": 0,
            "delta_iou_retained": [],
            "delta_score_retained": [],
            "by_fold": {f: {"rescued": 0, "lost": 0, "retained": 0, "fp_added": 0, "fp_removed": 0} for f in range(NUM_FOLDS)},
            "by_cam": {c: {"rescued": 0, "lost": 0, "retained": 0, "fp_added": 0, "fp_removed": 0} for c in CAMERAS},
            "by_class": {cls: {"rescued": 0, "lost": 0, "retained": 0, "fp_added": 0, "fp_removed": 0} for cls in range(NUM_CLASSES)}
        }
        for exp in EXPERTS
    }

    for exp in EXPERTS:
        exp_name = f"B+{exp}"
        t_exp_0 = time.time()
        print("=" * 115)
        print(f"EVALUATING EXPERT: {exp_name}")
        print("=" * 115)

        # Capped memory: exactly 2 evaluators resident per expert (Contract Item 2)
        exp_evaluator = CocoEvaluator(coco_gt, ["bbox"])
        exp_evaluator_ca = CocoEvaluator(coco_gt_ca, ["bbox"])
        exp_evaluator.coco_eval["bbox"].params.iouThrs = np.array([0.30])
        exp_evaluator_ca.coco_eval["bbox"].params.iouThrs = np.array([0.30])

        exp_tp = 0
        exp_fp = 0
        exp_fn = 0
        tr = transition_records[exp]

        for fold_idx, vid, v_file, num_records in video_shard_list:
            records = torch.load(v_file, map_location="cpu")
            cam_name = video_to_camera[vid]

            v_preds = {}
            v_preds_ca = {}

            for fr in records:
                img_id = int(fr["image_id"])
                img_w, img_h = img_native_size[img_id]
                gt_boxes_norm = fr["gt_boxes_norm"]
                gt_labels = fr["gt_labels"]
                num_gt = gt_boxes_norm.shape[0]

                exp_preds = extract_top100_predictions(fr[f"z_{exp.lower()}"], fr[f"b_{exp.lower()}"])

                # Format COCO predictions
                boxes_xyxy_norm = box_cxcywh_to_xyxy(exp_preds["all_boxes"])
                scale = torch.tensor([img_w, img_h, img_w, img_h])
                scaled_boxes = boxes_xyxy_norm * scale
                cat_labels = cat_id_tensor[exp_preds["all_c_idx"]]

                v_preds[img_id] = {
                    "boxes": scaled_boxes,
                    "scores": exp_preds["all_scores"],
                    "labels": cat_labels
                }
                v_preds_ca[img_id] = {
                    "boxes": scaled_boxes,
                    "scores": exp_preds["all_scores"],
                    "labels": torch.ones_like(cat_labels)
                }

                # Operational matching for Expert
                exp_match = match_operational_detections_to_gt(
                    exp_preds["op_scores"],
                    exp_preds["op_c_idx"],
                    exp_preds["op_boxes"],
                    exp_preds["op_q_idx"],
                    gt_boxes_norm,
                    gt_labels
                )

                exp_tp += exp_match["tp"]
                exp_fp += exp_match["fp"]
                exp_fn += exp_match["fn"]

                # Retrieve cached Base matching for this frame
                base_match = base_frame_match_cache[img_id]
                base_matched_set = base_match["matched_gt_indices"]
                exp_matched_set = exp_match["matched_gt_indices"]
                base_gt_to_det = base_match["gt_to_det"]
                exp_gt_to_det = exp_match["gt_to_det"]

                frame_rescues = []
                frame_losses = []
                frame_retained = []

                # Ground-Truth Outcome Transitions (Unrounded precision: Contract Item 4)
                for g_idx in range(num_gt):
                    c_gt = int(gt_labels[g_idx].item())
                    is_base_tp = g_idx in base_matched_set
                    is_exp_tp = g_idx in exp_matched_set

                    if not is_base_tp and is_exp_tp:
                        # RESCUE
                        tr["rescued"] += 1
                        tr["by_fold"][fold_idx]["rescued"] += 1
                        tr["by_cam"][cam_name]["rescued"] += 1
                        tr["by_class"][c_gt]["rescued"] += 1
                        d_info = exp_gt_to_det[g_idx]
                        frame_rescues.append({
                            "gt_idx": int(g_idx),
                            "class_idx": int(c_gt),
                            "q_idx_exp": int(d_info[3]),
                            "score_exp": float(d_info[2]),
                            "iou_exp": float(d_info[1])
                        })
                    elif is_base_tp and not is_exp_tp:
                        # LOSS
                        tr["lost"] += 1
                        tr["by_fold"][fold_idx]["lost"] += 1
                        tr["by_cam"][cam_name]["lost"] += 1
                        tr["by_class"][c_gt]["lost"] += 1
                        d_info = base_gt_to_det[g_idx]
                        frame_losses.append({
                            "gt_idx": int(g_idx),
                            "class_idx": int(c_gt),
                            "q_idx_base": int(d_info[3]),
                            "score_base": float(d_info[2]),
                            "iou_base": float(d_info[1])
                        })
                    elif is_base_tp and is_exp_tp:
                        # RETAINED
                        tr["retained"] += 1
                        tr["by_fold"][fold_idx]["retained"] += 1
                        tr["by_cam"][cam_name]["retained"] += 1
                        tr["by_class"][c_gt]["retained"] += 1

                        b_d = base_gt_to_det[g_idx]
                        e_d = exp_gt_to_det[g_idx]
                        d_iou = e_d[1] - b_d[1]
                        d_score = e_d[2] - b_d[2]
                        tr["delta_iou_retained"].append(d_iou)
                        tr["delta_score_retained"].append(d_score)
                        frame_retained.append({
                            "gt_idx": int(g_idx),
                            "class_idx": int(c_gt),
                            "q_idx_base": int(b_d[3]),
                            "q_idx_exp": int(e_d[3]),
                            "iou_base": float(b_d[1]),
                            "iou_exp": float(e_d[1]),
                            "score_base": float(b_d[2]),
                            "score_exp": float(e_d[2])
                        })

                # Operational False Positive Transitions
                base_fp_keys = set(base_match["fp_dict"].keys())
                exp_fp_keys = set(exp_match["fp_dict"].keys())

                fp_removed_keys = base_fp_keys - exp_fp_keys
                fp_added_keys = exp_fp_keys - base_fp_keys

                tr["fp_removed"] += len(fp_removed_keys)
                tr["fp_added"] += len(fp_added_keys)
                tr["by_fold"][fold_idx]["fp_removed"] += len(fp_removed_keys)
                tr["by_fold"][fold_idx]["fp_added"] += len(fp_added_keys)
                tr["by_cam"][cam_name]["fp_removed"] += len(fp_removed_keys)
                tr["by_cam"][cam_name]["fp_added"] += len(fp_added_keys)

                frame_fp_added = []
                for k in fp_added_keys:
                    tr["by_class"][k[1]]["fp_added"] += 1
                    frame_fp_added.append({
                        "q_idx": int(k[0]),
                        "class_idx": int(k[1]),
                        "score": float(exp_match["fp_dict"][k]["score"]),
                        "box": [float(x) for x in exp_match["fp_dict"][k]["box"]]
                    })

                frame_fp_removed = []
                for k in fp_removed_keys:
                    tr["by_class"][k[1]]["fp_removed"] += 1
                    frame_fp_removed.append({
                        "q_idx": int(k[0]),
                        "class_idx": int(k[1]),
                        "score": float(base_match["fp_dict"][k]["score"]),
                        "box": [float(x) for x in base_match["fp_dict"][k]["box"]]
                    })

                # Direct O(1) assignment into pre-initialized event ledger (Contract Item 2)
                events_by_video[vid][img_id]["events"][exp] = {
                    "rescues": frame_rescues,
                    "losses": frame_losses,
                    "retained": frame_retained,
                    "fp_added": frame_fp_added,
                    "fp_removed": frame_fp_removed
                }

            # Update evaluators once per video
            exp_evaluator.update(v_preds)
            exp_evaluator_ca.update(v_preds_ca)
            del records

        # Synchronize before accumulate (Contract Item 3)
        exp_evaluator.synchronize_between_processes()
        exp_evaluator.accumulate()
        exp_evaluator.summarize()

        exp_evaluator_ca.synchronize_between_processes()
        exp_evaluator_ca.accumulate()
        exp_evaluator_ca.summarize()

        stats_exp = exp_evaluator.coco_eval["bbox"].stats
        stats_exp_ca = exp_evaluator_ca.coco_eval["bbox"].stats
        expert_coco_stats[exp] = stats_exp
        expert_coco_stats_ca[exp] = stats_exp_ca
        expert_op_counts[exp] = {"tp": exp_tp, "fp": exp_fp, "fn": exp_fn}

        prec_exp = exp_evaluator.coco_eval["bbox"].eval["precision"]
        exp_per_c = []
        for k in range(NUM_CLASSES):
            p_k = prec_exp[0, :, k, 0, 2]
            valid_p = p_k[p_k > -1]
            exp_per_c.append(float(np.mean(valid_p)) if len(valid_p) > 0 else 0.0)
        expert_per_class_ap[exp] = exp_per_c

        # Hard Mathematical Transition Conservation Assertions (Contract Item 8)
        print("\n[*] Verifying exact mathematical transition conservation assertions...")
        assert tr["retained"] + tr["lost"] == base_tp_total, (
            f"[FATAL] Retention+Loss != Base TP for Expert {exp}: {tr['retained']} + {tr['lost']} != {base_tp_total}"
        )
        assert tr["retained"] + tr["rescued"] == exp_tp, (
            f"[FATAL] Retention+Rescue != Exp TP for Expert {exp}: {tr['retained']} + {tr['rescued']} != {exp_tp}"
        )
        assert tr["rescued"] - tr["lost"] == (exp_tp - base_tp_total), (
            f"[FATAL] Rescue - Loss != ΔTP for Expert {exp}: {tr['rescued']} - {tr['lost']} != {exp_tp - base_tp_total}"
        )
        assert tr["fp_added"] - tr["fp_removed"] == (exp_fp - base_fp_total), (
            f"[FATAL] FP Added - FP Removed != ΔFP for Expert {exp}: {tr['fp_added']} - {tr['fp_removed']} != {exp_fp - base_fp_total}"
        )
        print(f"    [+] CONSERVATION CERTIFIED for Expert {exp}: All 4 transition invariants verified.")
        print(f"    [+] Expert {exp} evaluation completed in {time.time() - t_exp_0:.1f}s.")

        # Free memory immediately
        del exp_evaluator
        del exp_evaluator_ca
        gc.collect()

    # =========================================================================
    # EVENT LEDGER COMPLETENESS GATE & SERIALIZATION (Contract Items 2, 3)
    # =========================================================================
    print("\n[*] Running event ledger completeness gate and serializing 47 per-video ledgers...")
    assert sum(len(v) for v in events_by_video.values()) == EXPECTED_TOTAL_FRAMES, (
        f"[FATAL] Total events count: {sum(len(v) for v in events_by_video.values())} != {EXPECTED_TOTAL_FRAMES}"
    )

    event_shard_hashes = {}
    for vid in video_ids:
        # Sort values strictly by seq_idx before saving
        v_events = sorted(
            events_by_video[vid].values(),
            key=lambda x: x["seq_idx"]
        )
        assert len(v_events) == expected_records_per_vid[vid], (
            f"[FATAL] Video {vid} event count mismatch: {len(v_events)} != {expected_records_per_vid[vid]}"
        )
        for ev in v_events:
            assert set(ev["events"].keys()) == set(EXPERTS), (
                f"[FATAL] Incomplete G0 events for image {ev['image_id']} in video {vid}: {ev['events'].keys()}"
            )

        f_idx = v_events[0]["fold_idx"]
        cam = v_events[0]["camera"]

        ev_file = EVENTS_DIR / f"bfcef_g0_events_fold{f_idx}_vid{vid}.pth"
        torch.save({
            "_meta": {
                "video_id": vid,
                "fold_idx": f_idx,
                "camera": cam,
                "num_frames": len(v_events),
                "driver_sha256": driver_sha,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "frames": v_events
        }, ev_file)

        ev_sha = compute_sha256(ev_file)
        event_shard_hashes[ev_file.name] = ev_sha

    events_index_path = EVENTS_DIR / "bfcef_g0_events_index.json"
    with open(events_index_path, "w", encoding="utf-8") as f:
        json.dump({
            "_meta": {
                "driver_sha256": driver_sha,
                "master_ledger_index_sha256": computed_master_index_sha,
                "total_videos": len(video_ids),
                "total_frames": EXPECTED_TOTAL_FRAMES,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "event_shards": event_shard_hashes
        }, f, indent=2)

    events_index_sha = compute_sha256(events_index_path)
    print(f"[+] 47 G1 event ledgers frozen. Master events index: {events_index_path.resolve()} (SHA: {events_index_sha[:16]}...)")

    # =========================================================================
    # PRIMARY REPORT TABLES
    # =========================================================================
    print("\n" + "=" * 115)
    print("TABLE 1: OUT-OF-FOLD COCO DETECTION PERFORMANCE AUDIT (IoU = 0.30)")
    print("=" * 115)
    print(f"{'Model':<8} | {'mAP@.3':<8} | {'ΔmAP':<8} | {'mAPca@.3':<9} | {'ΔmAPca':<8} | {'AP_S@.3':<8} | {'AP_M@.3':<8} | {'AP_L@.3':<8}")
    print("-" * 115)
    for name in ["Base", "B+S", "B+L", "B+C", "B+T"]:
        if name == "Base":
            m03 = base_map03
            ca03 = base_ca03
            s_s = float(base_stats[3])
            s_m = float(base_stats[4])
            s_l = float(base_stats[5])
            d_str = " 0.0000"
            d_ca_str = " 0.0000"
        else:
            exp_k = name.replace("B+", "")
            m03 = float(expert_coco_stats[exp_k][0])
            ca03 = float(expert_coco_stats_ca[exp_k][0])
            s_s = float(expert_coco_stats[exp_k][3])
            s_m = float(expert_coco_stats[exp_k][4])
            s_l = float(expert_coco_stats[exp_k][5])
            d_str = f"{m03 - base_map03:+.4f}"
            d_ca_str = f"{ca03 - base_ca03:+.4f}"

        print(f"{name:<8} | {m03:.4f}   | {d_str:<8} | {ca03:.4f}    | {d_ca_str:<8} | {s_s:.4f}   | {s_m:.4f}   | {s_l:.4f}")
    print("=" * 115)

    print("\n" + "=" * 115)
    print("TABLE 2: CANONICAL OPERATIONAL DETECTOR METRICS (Top-100, Score >= 0.20, IoU >= 0.30)")
    print("=" * 115)
    print(f"{'Model':<8} | {'TP':<8} | {'FP':<8} | {'FN':<8} | {'Precision':<11} | {'Recall':<10} | {'F1 Score':<10} | {'ΔF1':<8}")
    print("-" * 115)
    for name in ["Base", "B+S", "B+L", "B+C", "B+T"]:
        if name == "Base":
            tp = base_tp_total
            fp = base_fp_total
            fn = base_fn_total
            d_f1_str = " 0.0000"
        else:
            exp_k = name.replace("B+", "")
            tp = expert_op_counts[exp_k]["tp"]
            fp = expert_op_counts[exp_k]["fp"]
            fn = expert_op_counts[exp_k]["fn"]
            f1_val = 2.0 * tp / max(1, (2.0 * tp + fp + fn))
            d_f1_str = f"{f1_val - base_f1:+.4f}"

        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2.0 * tp / max(1, 2.0 * tp + fp + fn)
        print(f"{name:<8} | {tp:>6,} | {fp:>6,} | {fn:>6,} | {prec:>9.2%}   | {rec:>8.2%} | {f1:>8.4f}   | {d_f1_str:<8}")
    print("=" * 115)

    print("\n" + "=" * 115)
    print("TABLE 3: COUNTERFACTUAL GT OUTCOME TRANSITIONS & BASELINE PROTECTION AUDIT")
    print("=" * 115)
    print(f"{'Expert':<8} | {'Rescued GT':<12} | {'Lost GT':<10} | {'Net ΔTP':<10} | {'Retained GT':<12} | {'Protection Loss Rate':<22} | {'FP Added':<10} | {'FP Removed':<11} | {'Net ΔFP':<8}")
    print("-" * 115)
    for exp in EXPERTS:
        tr = transition_records[exp]
        r = tr["rescued"]
        l = tr["lost"]
        net_tp = r - l
        ret = tr["retained"]
        prot_loss_rate = l / 50741.0
        fa = tr["fp_added"]
        fr = tr["fp_removed"]
        net_fp = fa - fr
        print(f"Expert {exp:<2} | {r:>10,}   | {l:>8,}   | {net_tp:>+8,}   | {ret:>10,}   | {prot_loss_rate:>10.2%} ({l:>5,}/50,741)  | {fa:>8,}   | {fr:>9,}   | {net_fp:>+7,}")
    print("=" * 115)

    print("\n" + "-" * 115)
    print("TABLE 4: STRATIFICATION BY CAMERA DOMAIN (STRICT SENSOR METADATA & EXPLICIT DENOMINATORS)")
    print("-" * 115)
    print(f"{'Domain':<8} | {'Expert':<8} | {'GT Count':<9} | {'Base TP':<8} | {'Rescued (Rate)':<16} | {'Lost (Loss Rate)':<18} | {'Net ΔTP':<9} | {'FP Added':<9} | {'FP Removed':<10} | {'Net ΔFP':<8}")
    print("-" * 115)
    for cam in CAMERAS:
        c_gt = gt_stratum_totals["by_cam"][cam]["gt"]
        c_btp = gt_stratum_totals["by_cam"][cam]["base_tp"]
        c_bfn = gt_stratum_totals["by_cam"][cam]["base_fn"]
        for exp in EXPERTS:
            tr_c = transition_records[exp]["by_cam"][cam]
            r = tr_c["rescued"]
            l = tr_c["lost"]
            net = r - l
            fa = tr_c["fp_added"]
            fr = tr_c["fp_removed"]
            net_fp = fa - fr
            r_rate = r / max(1, c_bfn)
            l_rate = l / max(1, c_btp)
            print(f"{cam:<8} | Expert {exp:<2} | {c_gt:>7,}   | {c_btp:>6,}   | {r:>5,} ({r_rate:>5.2%})    | {l:>5,} ({l_rate:>5.2%})      | {net:>+7,}   | {fa:>7,}   | {fr:>8,}   | {net_fp:>+7,}")
        print("-" * 115)

    print("\n" + "-" * 115)
    print("TABLE 5: STRATIFICATION BY OBJECT CLASS (CLASSES 0 TO 6)")
    print("-" * 115)
    print(f"{'Class Name':<18} | {'Expert':<8} | {'GT Count':<9} | {'Base TP':<8} | {'Rescued':<8} | {'Lost':<8} | {'Net ΔTP':<9} | {'Base AP@.3':<11} | {'Exp AP@.3':<10} | {'ΔAP@.3':<8}")
    print("-" * 115)
    for c in range(NUM_CLASSES):
        c_name = class_names[c]
        c_gt = gt_stratum_totals["by_class"][c]["gt"]
        c_btp = gt_stratum_totals["by_class"][c]["base_tp"]
        b_ap = base_per_class_ap[c]
        print(f"[{c}] {c_name:<14} | Base     | {c_gt:>7,}   | {c_btp:>6,}   | {'--':<8} | {'--':<8} | {'--':<9} | {b_ap:.4f}      | {'--':<10} | {'--':<8}")
        for exp in EXPERTS:
            tr_cls = transition_records[exp]["by_class"][c]
            r = tr_cls["rescued"]
            l = tr_cls["lost"]
            net = r - l
            e_ap = expert_per_class_ap[exp][c]
            d_ap = e_ap - b_ap
            print(f"{'':<18} | Expert {exp:<2} | {c_gt:>7,}   | {c_btp:>6,}   | {r:>6,}   | {l:>6,}   | {net:>+7,}   | {b_ap:.4f}      | {e_ap:.4f}     | {d_ap:>+7.4f}")
        print("-" * 115)

    print("\n" + "-" * 115)
    print("TABLE 6: STRATIFICATION BY FOLD (CROSS-FITTING REPRODUCIBILITY)")
    print("-" * 115)
    print(f"{'Fold':<6} | {'Expert':<8} | {'GT Count':<9} | {'Base TP':<8} | {'Rescued (Rate)':<16} | {'Lost (Loss Rate)':<18} | {'Net ΔTP':<9} | {'Net ΔFP':<8}")
    print("-" * 115)
    for f in range(NUM_FOLDS):
        f_gt = gt_stratum_totals["by_fold"][f]["gt"]
        f_btp = gt_stratum_totals["by_fold"][f]["base_tp"]
        f_bfn = gt_stratum_totals["by_fold"][f]["base_fn"]
        for exp in EXPERTS:
            tr_f = transition_records[exp]["by_fold"][f]
            r = tr_f["rescued"]
            l = tr_f["lost"]
            net = r - l
            fa = tr_f["fp_added"]
            fr = tr_f["fp_removed"]
            net_fp = fa - fr
            r_rate = r / max(1, f_bfn)
            l_rate = l / max(1, f_btp)
            print(f"Fold {f:<1} | Expert {exp:<2} | {f_gt:>7,}   | {f_btp:>6,}   | {r:>5,} ({r_rate:>5.2%})    | {l:>5,} ({l_rate:>5.2%})      | {net:>+7,}   | {net_fp:>+7,}")
        print("-" * 115)

    print("\n" + "=" * 115)
    print("TABLE 7: SECONDARY CONTINUOUS DIAGNOSTICS FOR RETAINED TPs (ΔIoU & ΔScore)")
    print("=" * 115)
    print(f"{'Expert':<8} | {'Retained TPs':<14} | {'Mean ΔIoU':<11} | {'P10 ΔIoU':<10} | {'P50 ΔIoU':<10} | {'P90 ΔIoU':<10} | {'Mean ΔScore':<12}")
    print("-" * 115)
    for exp in EXPERTS:
        d_iou_arr = np.array(transition_records[exp]["delta_iou_retained"])
        d_sc_arr = np.array(transition_records[exp]["delta_score_retained"])
        ret_n = len(d_iou_arr)
        if ret_n > 0:
            print(f"Expert {exp:<2} | {ret_n:>12,}   | {np.mean(d_iou_arr):>+9.4f}   | {np.percentile(d_iou_arr, 10):>+8.4f}   | {np.percentile(d_iou_arr, 50):>+8.4f}   | {np.percentile(d_iou_arr, 90):>+8.4f}   | {np.mean(d_sc_arr):>+10.4f}")
        else:
            print(f"Expert {exp:<2} | {ret_n:>12,}   | {'--':<11} | {'--':<10} | {'--':<10} | {'--':<10} | {'--':<12}")
    print("=" * 115 + "\n")

    # =========================================================================
    # SERIALIZE STAGE G0 QUALIFICATION REPORT (Contract Item 12)
    # =========================================================================
    report_path = G0_WORK_DIR / "bfcef_g0_qualification_report.json"
    serializable_transitions = {
        exp: {
            "rescued": transition_records[exp]["rescued"],
            "lost": transition_records[exp]["lost"],
            "net_tp": transition_records[exp]["rescued"] - transition_records[exp]["lost"],
            "retained": transition_records[exp]["retained"],
            "fp_added": transition_records[exp]["fp_added"],
            "fp_removed": transition_records[exp]["fp_removed"],
            "net_fp": transition_records[exp]["fp_added"] - transition_records[exp]["fp_removed"],
            "protection_loss_rate": round(transition_records[exp]["lost"] / 50741.0, 6),
            "retained_tp_continuous": {
                "n_retained": len(transition_records[exp]["delta_iou_retained"]),
                "delta_iou": {
                    "mean": round(float(np.mean(transition_records[exp]["delta_iou_retained"])), 6) if len(transition_records[exp]["delta_iou_retained"]) > 0 else 0.0,
                    "p10": round(float(np.percentile(transition_records[exp]["delta_iou_retained"], 10)), 6) if len(transition_records[exp]["delta_iou_retained"]) > 0 else 0.0,
                    "p50": round(float(np.percentile(transition_records[exp]["delta_iou_retained"], 50)), 6) if len(transition_records[exp]["delta_iou_retained"]) > 0 else 0.0,
                    "p90": round(float(np.percentile(transition_records[exp]["delta_iou_retained"], 90)), 6) if len(transition_records[exp]["delta_iou_retained"]) > 0 else 0.0
                },
                "delta_score": {
                    "mean": round(float(np.mean(transition_records[exp]["delta_score_retained"])), 6) if len(transition_records[exp]["delta_score_retained"]) > 0 else 0.0,
                    "p10": round(float(np.percentile(transition_records[exp]["delta_score_retained"], 10)), 6) if len(transition_records[exp]["delta_score_retained"]) > 0 else 0.0,
                    "p50": round(float(np.percentile(transition_records[exp]["delta_score_retained"], 50)), 6) if len(transition_records[exp]["delta_score_retained"]) > 0 else 0.0,
                    "p90": round(float(np.percentile(transition_records[exp]["delta_score_retained"], 90)), 6) if len(transition_records[exp]["delta_score_retained"]) > 0 else 0.0
                }
            },
            "by_fold": transition_records[exp]["by_fold"],
            "by_cam": transition_records[exp]["by_cam"],
            "by_class": transition_records[exp]["by_class"]
        }
        for exp in EXPERTS
    }

    report_payload = {
        "_meta": {
            "driver_sha256": driver_sha,
            "master_ledger_index_sha256": computed_master_index_sha,
            "events_index_sha256": events_index_sha,
            "fold_manifest_sha256": fold_manifest_sha,
            "canonical_cache_manifest_sha256": canonical_manifest_sha,
            "temporal_cache_manifest_sha256": temporal_manifest_sha,
            "train_json_sha256": train_json_sha,
            "total_videos": len(video_ids),
            "total_frames": EXPECTED_TOTAL_FRAMES,
            "execution_time_minutes": round((time.time() - t_start) / 60.0, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "baseline_oracle": {
            "gt_count": base_gt_total,
            "tp_count": base_tp_total,
            "fp_count": base_fp_total,
            "fn_count": base_fn_total,
            "f1_score": round(base_f1, 6),
            "map_03": round(base_map03, 6),
            "mapca_03": round(base_ca03, 6),
            "ap_s_03": round(float(base_stats[3]), 6),
            "ap_m_03": round(float(base_stats[4]), 6),
            "ap_l_03": round(float(base_stats[5]), 6)
        },
        "detection_metrics_map03": {
            "Base": {
                "mAP": round(base_map03, 6),
                "mAPca": round(base_ca03, 6),
                "AP_S": round(float(base_stats[3]), 6),
                "AP_M": round(float(base_stats[4]), 6),
                "AP_L": round(float(base_stats[5]), 6),
                "per_class_ap": [round(x, 6) for x in base_per_class_ap]
            },
            **{
                f"B+{e}": {
                    "mAP": round(float(expert_coco_stats[e][0]), 6),
                    "mAPca": round(float(expert_coco_stats_ca[e][0]), 6),
                    "AP_S": round(float(expert_coco_stats[e][3]), 6),
                    "AP_M": round(float(expert_coco_stats[e][4]), 6),
                    "AP_L": round(float(expert_coco_stats[e][5]), 6),
                    "per_class_ap": [round(x, 6) for x in expert_per_class_ap[e]]
                }
                for e in EXPERTS
            }
        },
        "operational_metrics": {
            "Base": {
                "tp": base_tp_total,
                "fp": base_fp_total,
                "fn": base_fn_total,
                "f1": round(base_f1, 6)
            },
            **{
                f"B+{e}": {
                    "tp": expert_op_counts[e]["tp"],
                    "fp": expert_op_counts[e]["fp"],
                    "fn": expert_op_counts[e]["fn"],
                    "f1": round(2.0 * expert_op_counts[e]["tp"] / max(1, 2.0 * expert_op_counts[e]["tp"] + expert_op_counts[e]["fp"] + expert_op_counts[e]["fn"]), 6)
                }
                for e in EXPERTS
            }
        },
        "gt_stratum_totals": gt_stratum_totals,
        "counterfactual_transitions": serializable_transitions
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    report_sha = compute_sha256(report_path)
    print(f"[+] Stage G0 qualification audit completed in {(time.time() - t_start) / 60.0:.2f} minutes.")
    print(f"[+] Certified Qualification Report Frozen: {report_path.resolve()}")
    print(f"[+] Report SHA-256: {report_sha}\n")


if __name__ == "__main__":
    main()