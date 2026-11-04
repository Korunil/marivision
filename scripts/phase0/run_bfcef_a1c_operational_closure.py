"""
D:/MARIVISION/H0_workbench/eval_tools/run_bfcef_a1c_operational_closure.py
MARSEN B-FCEF: Stage A1C - Ground-Truth Relative Operational Closure
(D-4411..D-4429, P-2267..P-2273)

Diagnostic pipeline (read-only, artifact-grounded):
  1. Cryptographic Provenance Preflight Gate:
     - Verifies Census JSON SHA-256.
     - Verifies seaclips-train.json SHA-256 against live provenance ledger.
     - Verifies all 47 Train shard SHAs against frozen census ledger.
     - Computes, verifies, and records A1R itemized records SHA-256.
  2. Constants defined: NUM_CLASSES = 7, OPERATIONAL_SCORE_THRESHOLD = 0.20, EVAL_IOU_THRESHOLD = 0.30.
  3. Recomputes Cat-1A operational status from all geometrically valid native support candidates (IoU >= 0.30):
     - Evaluates target-class candidate score p_{j, c_t} across all support queries j (regardless of argmax).
     - Checks candidate conjunction: (j * NUM_CLASSES + c_t in Top100) AND (p_{j, c_t} >= 0.20).
     - Partitions into Cat 1A-Active (operational collision) vs Cat 1A-Subthreshold (confidence deficit).
     - Records max_support_target_score, max_top100_support_target_score, and candidate counts.
  4. Track A: Historical Baseline Attribution Proxy:
     - Replays query-level max-class Top-100 baseline matching to determine historical GT attribution.
     - Resolves ambiguous unrescued cases strictly within the unrescued candidate subset.
  5. Track B: Canonical R0 Evaluator Closure:
     - Replays canonical flattened Top-100 + (s >= 0.20) evaluator matching on ground truth.
     - Maps temporal proposal to same-class GT annotations at IoU >= 0.30.
     - Classifies attribution into Unique-Unrescued, Unique-AlreadyCaptured, Ambiguous, and Unattributed.
     - Resolves ambiguous unrescued cases strictly within the unrescued candidate subset.
  6. Evaluates GT-relative metrics strictly on Canonical-Unique GT-evaluable subset:
     - Native target operational conjunction: (IoU >= 0.30) AND (in Top-100) AND (s >= 0.20) on same candidate.
     - GT-relative native absence with corrected denominator: N_absent / N_GT_evaluable.
     - Cat-3 localization gap relative to ground truth (same-class and best-any IoU).
  7. Freezes summary and itemized closure records to JSON.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import time
import json
import hashlib
from pathlib import Path
from collections import defaultdict
import torch
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

# Path Constants
ROOT_DIR = Path("D:/MARIVISION")
OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
SHARD_DIR = OUT_DIR / "full_split_shards"

CENSUS_JSON = OUT_DIR / "marsen_b2_production_census_56shards.json"
TRAIN_JSON = ROOT_DIR / "seaclips-train.json"
A1R_RECORDS_JSON = OUT_DIR / "bfcef_a1r_records_1576.json"
A1C_SUMMARY_JSON = OUT_DIR / "bfcef_a1c_operational_closure_summary.json"
A1C_RECORDS_JSON = OUT_DIR / "bfcef_a1c_records_1576.json"

NUM_VIDEOS = 47
EXPECTED_TOTAL_RESCUES = 1576
NUM_CLASSES = 7
OPERATIONAL_SCORE_THRESHOLD = 0.20
EVAL_IOU_THRESHOLD = 0.30

FROZEN_CENSUS_SHA256 = "a20a734d668a9078bdfe274cb09fa5a47ad9c6a3a9d7cef2d1f99b926623fbc5"


def compute_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def box_iou_xywh_pixels(box1_px, box2_px):
    """Pairwise IoU between two pixel boxes [x, y, w, h]."""
    x1_min, y1_min = box1_px[0], box1_px[1]
    x1_max, y1_max = x1_min + box1_px[2], y1_min + box1_px[3]

    x2_min, y2_min = box2_px[0], box2_px[1]
    x2_max, y2_max = x2_min + box2_px[2], y2_min + box2_px[3]

    ix_min = max(x1_min, x2_min)
    iy_min = max(y1_min, y2_min)
    x_overlap = min(x1_max, x2_max)
    y_overlap = min(y1_max, y2_max)

    iw = max(0.0, x_overlap - ix_min)
    ih = max(0.0, y_overlap - iy_min)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    area1 = box1_px[2] * box1_px[3]
    area2 = box2_px[2] * box2_px[3]
    union = area1 + area2 - inter
    return inter / union if union > 0.0 else 0.0


def norm_cxcywh_to_pixel_xywh(box_norm, img_w, img_h):
    """Converts normalized [cx, cy, w, h] to pixel [x, y, w, h]."""
    cx = box_norm[0] * img_w
    cy = box_norm[1] * img_h
    bw = box_norm[2] * img_w
    bh = box_norm[3] * img_h
    return [cx - bw / 2.0, cy - bh / 2.0, bw, bh]


def box_iou_matrix_xywh_tensor(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Vectorized pairwise IoU between boxes1 (N, 4) and boxes2 (M, 4) in normalized cxcywh."""
    b1_x1 = boxes1[:, 0] - boxes1[:, 2] / 2.0
    b1_y1 = boxes1[:, 1] - boxes1[:, 3] / 2.0
    b1_x2 = boxes1[:, 0] + boxes1[:, 2] / 2.0
    b1_y2 = boxes1[:, 1] + boxes1[:, 3] / 2.0

    b2_x1 = boxes2[:, 0] - boxes2[:, 2] / 2.0
    b2_y1 = boxes2[:, 1] - boxes2[:, 3] / 2.0
    b2_x2 = boxes2[:, 0] + boxes2[:, 2] / 2.0
    b2_y2 = boxes2[:, 1] + boxes2[:, 3] / 2.0

    ix1 = torch.max(b1_x1.unsqueeze(1), b2_x1.unsqueeze(0))
    iy1 = torch.max(b1_y1.unsqueeze(1), b2_y1.unsqueeze(0))
    ix2 = torch.min(b1_x2.unsqueeze(1), b2_x2.unsqueeze(0))
    iy2 = torch.min(b1_y2.unsqueeze(1), b2_y2.unsqueeze(0))

    iw = torch.clamp(ix2 - ix1, min=0.0)
    ih = torch.clamp(iy2 - iy1, min=0.0)
    inter = iw * ih

    area1 = (boxes1[:, 2] * boxes1[:, 3]).unsqueeze(1)
    area2 = (boxes2[:, 2] * boxes2[:, 3]).unsqueeze(0)
    union = area1 + area2 - inter

    return torch.where(union > 0, inter / union, torch.zeros_like(inter))


def run_historical_policy_matching(eval_logits, eval_boxes, gt_annotations, img_w, img_h, cat_id_to_idx):
    """
    Replays the historical greedy utility-label base policy:
      - Query-level max score across classes
      - Top 100 queries selected by max score
      - Matched greedily against GT annotations (IoU >= 0.30, matching argmax class)
    Returns set of matched GT annotation IDs.
    """
    probs = eval_logits.sigmoid()
    max_scores, max_classes = torch.max(probs, dim=-1)
    top100_queries = torch.argsort(-max_scores)[:100].tolist()

    detections = []
    for q_idx in top100_queries:
        sc = float(max_scores[q_idx].item())
        c_idx = int(max_classes[q_idx].item())
        b_px = norm_cxcywh_to_pixel_xywh(eval_boxes[q_idx].tolist(), img_w, img_h)
        detections.append((sc, c_idx, b_px))

    detections.sort(key=lambda x: -x[0])
    matched_gt_ids = set()

    for sc, c_idx, det_box_px in detections:
        best_iou = -1.0
        best_gt_id = None
        for ann in gt_annotations:
            ann_id = ann["id"]
            if ann_id in matched_gt_ids:
                continue
            if cat_id_to_idx[ann["category_id"]] != c_idx:
                continue
            iou = box_iou_xywh_pixels(det_box_px, ann["bbox"])
            if iou >= EVAL_IOU_THRESHOLD and iou > best_iou:
                best_iou = iou
                best_gt_id = ann_id
        if best_gt_id is not None:
            matched_gt_ids.add(best_gt_id)

    return matched_gt_ids


def run_canonical_r0_matching(eval_logits, eval_boxes, gt_annotations, img_w, img_h, cat_id_to_idx):
    """
    Replays the certified canonical R0 evaluator policy:
      - Flattened 300 x 7 = 2100 detection candidates
      - Top 100 selected across all flattened pairs
      - Filtered strictly by score >= 0.20
      - Matched greedily against GT annotations (IoU >= 0.30, matching class)
    Returns set of matched GT annotation IDs.
    """
    probs = eval_logits.sigmoid()
    flat_probs = probs.flatten()
    topk_scores, topk_indices = torch.topk(flat_probs, min(100, flat_probs.numel()))

    q_indices = (topk_indices // NUM_CLASSES).tolist()
    c_indices = (topk_indices % NUM_CLASSES).tolist()
    scores = topk_scores.tolist()

    detections = []
    for sc, q_idx, c_idx in zip(scores, q_indices, c_indices):
        if sc < OPERATIONAL_SCORE_THRESHOLD:
            continue
        det_box_px = norm_cxcywh_to_pixel_xywh(eval_boxes[q_idx].tolist(), img_w, img_h)
        detections.append((sc, c_idx, det_box_px))

    detections.sort(key=lambda x: -x[0])
    matched_gt_ids = set()

    for sc, c_idx, det_box_px in detections:
        best_iou = -1.0
        best_gt_id = None
        for ann in gt_annotations:
            ann_id = ann["id"]
            if ann_id in matched_gt_ids:
                continue
            if cat_id_to_idx[ann["category_id"]] != c_idx:
                continue
            iou = box_iou_xywh_pixels(det_box_px, ann["bbox"])
            if iou >= EVAL_IOU_THRESHOLD and iou > best_iou:
                best_iou = iou
                best_gt_id = ann_id
        if best_gt_id is not None:
            matched_gt_ids.add(best_gt_id)

    return matched_gt_ids


def main():
    t_start = time.time()
    driver_sha = compute_sha256(Path(__file__))

    print("=" * 115)
    print("MARSEN B-FCEF: STAGE A1C - GROUND-TRUTH RELATIVE OPERATIONAL CLOSURE")
    print(f"Driver Script SHA-256:       {driver_sha}")
    print(f"Evaluator Baseline Contract: score >= {OPERATIONAL_SCORE_THRESHOLD:.2f} | IoU >= {EVAL_IOU_THRESHOLD:.2f} | NUM_CLASSES = {NUM_CLASSES}")
    print("=" * 115, flush=True)

    # 1. Cryptographic Provenance Preflight Gate (D-4427, P-2273)
    print("[*] Running Cryptographic Provenance Preflight Gate...", flush=True)
    assert CENSUS_JSON.exists(), f"[FATAL] Missing census JSON at {CENSUS_JSON}"
    assert compute_sha256(CENSUS_JSON) == FROZEN_CENSUS_SHA256, "[FATAL] Census SHA mismatch"

    with open(CENSUS_JSON, "r", encoding="utf-8") as f:
        census = json.load(f)

    assert compute_sha256(TRAIN_JSON) == census["_meta"]["live_provenance"]["train_json_sha256"], "[FATAL] Train JSON drift"

    frozen_shard_ledger = census["shard_sha256_ledger"]
    for vid in range(1, NUM_VIDEOS + 1):
        shard_path = SHARD_DIR / f"marsen_b2_shard_train_vid{vid}.pth"
        assert shard_path.exists(), f"[FATAL] Missing {shard_path.name}"
        assert compute_sha256(shard_path) == frozen_shard_ledger[shard_path.name], f"[FATAL] Shard drift: {shard_path.name}"

    assert A1R_RECORDS_JSON.exists(), f"[FATAL] Missing A1R records at {A1R_RECORDS_JSON}"
    a1r_records_sha = compute_sha256(A1R_RECORDS_JSON)

    with open(A1R_RECORDS_JSON, "r", encoding="utf-8") as f:
        a1r_records = json.load(f)
    assert len(a1r_records) == EXPECTED_TOTAL_RESCUES, f"[FATAL] Expected 1,576 records, got {len(a1r_records)}"

    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        train_coco = json.load(f)

    sorted_cats = sorted(train_coco["categories"], key=lambda x: x["id"])
    cat_id_to_idx = {c["id"]: idx for idx, c in enumerate(sorted_cats)}
    cat_idx_to_id = {idx: c["id"] for idx, c in enumerate(sorted_cats)}

    anns_by_img = defaultdict(list)
    for ann in train_coco["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    print(f"    [+] Provenance verified: Census, Train JSON, 47 Train shards certified.")
    print(f"    [+] Ingested A1R Records (1,576 rows) SHA-256: {a1r_records_sha}\n", flush=True)

    # 2. Index Shards for Evaluator Matching
    print("[*] Running Dual-Policy Matching (Historical Proxy vs Canonical R0)...", flush=True)
    frames_by_iid = {}
    hist_matched_gts_by_iid = {}
    canon_matched_gts_by_iid = {}

    for vid in range(1, NUM_VIDEOS + 1):
        shard_path = SHARD_DIR / f"marsen_b2_shard_train_vid{vid}.pth"
        shard = torch.load(shard_path, map_location="cpu")
        for fr in shard.get("frame_records", []):
            iid = fr["image_id"]
            frames_by_iid[iid] = fr
            anns = anns_by_img.get(iid, [])

            hist_matched = run_historical_policy_matching(
                fr["d_logits"], fr["d_boxes"], anns, fr["img_w"], fr["img_h"], cat_id_to_idx
            )
            canon_matched = run_canonical_r0_matching(
                fr["d_logits"], fr["d_boxes"], anns, fr["img_w"], fr["img_h"], cat_id_to_idx
            )
            hist_matched_gts_by_iid[iid] = hist_matched
            canon_matched_gts_by_iid[iid] = canon_matched

    print(f"    [+] Evaluated dual-policy matching across {len(frames_by_iid):,} target frames.\n", flush=True)

    # 3. Process All 1,576 Events
    print("[*] Recomputing Cat-1A Support Candidates & Resolving Dual-Policy GT Attribution...", flush=True)

    cat1a_total = 0
    cat1a_subthreshold_count = 0
    cat1a_active_count = 0

    cat1a_max_support_scores = []
    cat1a_max_top100_support_scores = []
    cat1a_temporal_scores = []
    cat1a_score_deltas = []

    track_a_attribution_counts = {
        "Unique-Unrescued-Hist": 0,
        "Ambiguous-Hist": 0,
        "AlreadyCaptured-Hist": 0,
        "Unattributed-Hist": 0
    }

    track_b_attribution_counts = {
        "Unique-Unrescued-Canon": 0,
        "Unique-AlreadyCaptured-Canon": 0,
        "Ambiguous-Canon": 0,
        "Unattributed-Canon": 0
    }

    canonical_unique_evaluable_records = []
    cat3_loc_gap_records = []
    canonical_unique_absent_count = 0

    closure_records_1576 = []

    for rec in a1r_records:
        iid = rec["image_id"]
        q_id = rec["query_id"]
        c_name = rec["category_name"]
        cat_category = rec["recoverability_category"]
        c_idx = [i for i, c in enumerate(sorted_cats) if c["name"] == c_name][0]
        c_id = cat_idx_to_id[c_idx]

        fr = frames_by_iid[iid]
        img_w = float(fr["img_w"])
        img_h = float(fr["img_h"])
        d_logits = fr["d_logits"]
        d_boxes = fr["d_boxes"]
        d_probs = d_logits.sigmoid()
        d_pred_cats = torch.argmax(d_probs, dim=-1)

        temp_box_t = fr["temp_boxes"][q_id]
        temp_box_px = norm_cxcywh_to_pixel_xywh(temp_box_t.tolist(), img_w, img_h)
        s_temp = float(rec["temporal_score"])

        anns = anns_by_img.get(iid, [])
        hist_matched_ids = hist_matched_gts_by_iid[iid]
        canon_matched_ids = canon_matched_gts_by_iid[iid]

        # ---------------------------------------------------------------------
        # RECOMPUTING CAT-1A FROM ALL NATIVE CANDIDATES (D-4417, P-2270)
        # ---------------------------------------------------------------------
        flat_probs = d_probs.flatten()
        top100_flat_indices = torch.topk(flat_probs, min(100, flat_probs.numel())).indices

        # Vectorized IoU of temporal box against all 300 native boxes
        ious_native_temp = box_iou_matrix_xywh_tensor(d_boxes, temp_box_t.unsqueeze(0)).squeeze(1)
        support_mask = (ious_native_temp >= EVAL_IOU_THRESHOLD)
        support_ids = torch.nonzero(support_mask, as_tuple=False).flatten()

        is_cat1_event = (rec.get("recoverability_category", "").startswith("Cat 1:"))
        cat1a_status = None
        max_supp_score = 0.0
        max_top100_supp_score = 0.0
        n_supp_cands = int(support_ids.numel())
        n_top100_supp_cands = 0
        n_oper_supp_cands = 0

        if n_supp_cands > 0:
            supp_target_scores = d_probs[support_ids, c_idx]
            max_supp_score = float(supp_target_scores.max().item())

            supp_flat_ids = support_ids * NUM_CLASSES + c_idx
            is_top100_each = (supp_flat_ids[:, None] == top100_flat_indices[None, :]).any(dim=1)
            n_top100_supp_cands = int(is_top100_each.sum().item())

            if n_top100_supp_cands > 0:
                top100_supp_scores = supp_target_scores[is_top100_each]
                max_top100_supp_score = float(top100_supp_scores.max().item())

                oper_mask = is_top100_each & (supp_target_scores >= OPERATIONAL_SCORE_THRESHOLD)
                n_oper_supp_cands = int(oper_mask.sum().item())

        if is_cat1_event:
            cat1a_total += 1
            cat1a_max_support_scores.append(max_supp_score)
            cat1a_max_top100_support_scores.append(max_top100_supp_score)
            cat1a_temporal_scores.append(s_temp)
            delta_s_top100 = s_temp - max_top100_supp_score
            cat1a_score_deltas.append(delta_s_top100)

            # Strict conjunction: (in Top-100) AND (s >= 0.20) on same candidate (D-4417, D-4419)
            if n_oper_supp_cands >= 1:
                cat1a_active_count += 1
                cat1a_status = "Cat 1A-Active (in Top100 AND s >= 0.20)"
            else:
                cat1a_subthreshold_count += 1
                cat1a_status = "Cat 1A-Subthreshold (in Top100 BUT s < 0.20)"

        # ---------------------------------------------------------------------
        # TRACK A: HISTORICAL BASELINE ATTRIBUTION PROXY (D-4422, D-4425)
        # ---------------------------------------------------------------------
        same_class_anns = [a for a in anns if a["category_id"] == c_id]
        same_class_overlapping_gts = []
        for a in same_class_anns:
            iou = box_iou_xywh_pixels(temp_box_px, a["bbox"])
            if iou >= EVAL_IOU_THRESHOLD:
                same_class_overlapping_gts.append((a, iou))

        unrescued_hist = [(g, iou) for g, iou in same_class_overlapping_gts if g["id"] not in hist_matched_ids]

        if len(unrescued_hist) == 1:
            track_a_type = "Unique-Unrescued-Hist"
            track_a_attribution_counts["Unique-Unrescued-Hist"] += 1
            hist_assigned_gt = unrescued_hist[0][0]
        elif len(unrescued_hist) > 1:
            track_a_type = "Ambiguous-Hist"
            track_a_attribution_counts["Ambiguous-Hist"] += 1
            hist_assigned_gt = max(unrescued_hist, key=lambda x: x[1])[0]
        elif len(same_class_overlapping_gts) > 0:
            track_a_type = "AlreadyCaptured-Hist"
            track_a_attribution_counts["AlreadyCaptured-Hist"] += 1
            hist_assigned_gt = max(same_class_overlapping_gts, key=lambda x: x[1])[0]
        else:
            track_a_type = "Unattributed-Hist"
            track_a_attribution_counts["Unattributed-Hist"] += 1
            hist_assigned_gt = None

        # ---------------------------------------------------------------------
        # TRACK B: CANONICAL R0 EVALUATOR CLOSURE (D-4422, D-4423)
        # ---------------------------------------------------------------------
        unrescued_canon = [(g, iou) for g, iou in same_class_overlapping_gts if g["id"] not in canon_matched_ids]

        if len(unrescued_canon) == 1:
            track_b_type = "Unique-Unrescued-Canon"
            track_b_attribution_counts["Unique-Unrescued-Canon"] += 1
            canon_assigned_gt = unrescued_canon[0][0]
        elif len(unrescued_canon) > 1:
            track_b_type = "Ambiguous-Canon"
            track_b_attribution_counts["Ambiguous-Canon"] += 1
            canon_assigned_gt = max(unrescued_canon, key=lambda x: x[1])[0]
        elif len(same_class_overlapping_gts) == 1:
            track_b_type = "Unique-AlreadyCaptured-Canon"
            track_b_attribution_counts["Unique-AlreadyCaptured-Canon"] += 1
            canon_assigned_gt = same_class_overlapping_gts[0][0]
        elif len(same_class_overlapping_gts) > 1:
            track_b_type = "Ambiguous-Canon"
            track_b_attribution_counts["Ambiguous-Canon"] += 1
            canon_assigned_gt = max(same_class_overlapping_gts, key=lambda x: x[1])[0]
        else:
            track_b_type = "Unattributed-Canon"
            track_b_attribution_counts["Unattributed-Canon"] += 1
            canon_assigned_gt = None

        # ---------------------------------------------------------------------
        # GT-RELATIVE EVALUATION STRICTLY ON CANONICAL-UNIQUE SET (D-4423, P-2271)
        # ---------------------------------------------------------------------
        is_canonical_unique = track_b_type in {"Unique-Unrescued-Canon", "Unique-AlreadyCaptured-Canon"}
        assigned_canonical_gt = canon_assigned_gt if is_canonical_unique else None

        best_native_gt_iou = 0.0
        best_sameclass_native_gt_iou = 0.0
        best_target_score_at_gt = 0.0
        native_target_in_top100 = False
        native_target_ge_020 = False
        native_target_operational = False
        temp_gt_iou = 0.0

        if is_canonical_unique and assigned_canonical_gt is not None:
            gt_px = assigned_canonical_gt["bbox"]
            temp_gt_iou = box_iou_xywh_pixels(temp_box_px, gt_px)

            for n_idx in range(300):
                n_box_px = norm_cxcywh_to_pixel_xywh(d_boxes[n_idx].tolist(), img_w, img_h)
                iou_n = box_iou_xywh_pixels(n_box_px, gt_px)
                if iou_n > best_native_gt_iou:
                    best_native_gt_iou = iou_n

                # Same-class native check (D-4424, P-2272)
                if int(d_pred_cats[n_idx].item()) == c_idx and iou_n > best_sameclass_native_gt_iou:
                    best_sameclass_native_gt_iou = iou_n

                p_tgt = float(d_probs[n_idx, c_idx].item())
                if iou_n >= EVAL_IOU_THRESHOLD:
                    if p_tgt > best_target_score_at_gt:
                        best_target_score_at_gt = p_tgt

                    cand_flat_idx = n_idx * NUM_CLASSES + c_idx
                    cand_in_top100 = bool((top100_flat_indices == cand_flat_idx).any().item())
                    cand_ge_020 = (p_tgt >= OPERATIONAL_SCORE_THRESHOLD)

                    if cand_in_top100:
                        native_target_in_top100 = True
                    if cand_ge_020:
                        native_target_ge_020 = True

                    # Conjoint operational candidate on the exact same slot (D-4419)
                    if cand_in_top100 and cand_ge_020:
                        native_target_operational = True

            if best_native_gt_iou < 0.10:
                canonical_unique_absent_count += 1

            canonical_unique_evaluable_records.append({
                "temp_gt_iou": temp_gt_iou,
                "native_gt_iou": best_native_gt_iou,
                "sameclass_native_gt_iou": best_sameclass_native_gt_iou,
                "best_target_score_at_gt": best_target_score_at_gt,
                "native_target_operational": native_target_operational,
                "is_unrescued": (track_b_type == "Unique-Unrescued-Canon")
            })

            # Cat 3 Localization Gap Analysis relative to GT
            if "Cat 3" in cat_category:
                cat3_loc_gap_records.append({
                    "event_id": rec["event_id"],
                    "temp_gt_iou": temp_gt_iou,
                    "best_native_gt_iou": best_native_gt_iou,
                    "sameclass_native_gt_iou": best_sameclass_native_gt_iou,
                    "iou_gap_sameclass": temp_gt_iou - best_sameclass_native_gt_iou,
                    "iou_gap_any": temp_gt_iou - best_native_gt_iou,
                    "sameclass_reaches_030": (best_sameclass_native_gt_iou >= EVAL_IOU_THRESHOLD)
                })

        closure_records_1576.append({
            "event_id": rec["event_id"],
            "video_id": rec["video_id"],
            "image_id": iid,
            "query_id": q_id,
            "category_name": c_name,
            "scale_bin": rec["scale_bin"],
            "cat1a_operational_status": cat1a_status,
            "n_support_candidates": n_supp_cands,
            "n_top100_support_candidates": n_top100_supp_cands,
            "n_operational_support_candidates": n_oper_supp_cands,
            "max_support_target_score": round(max_supp_score, 4),
            "max_top100_support_target_score": round(max_top100_supp_score, 4),
            "track_a_historical_attribution": track_a_type,
            "track_b_canonical_attribution": track_b_type,
            "assigned_canonical_gt_id": assigned_canonical_gt["id"] if assigned_canonical_gt else None,
            "temporal_gt_iou": round(temp_gt_iou, 4) if is_canonical_unique else None,
            "best_native_gt_iou": round(best_native_gt_iou, 4) if is_canonical_unique else None,
            "best_sameclass_native_gt_iou": round(best_sameclass_native_gt_iou, 4) if is_canonical_unique else None,
            "best_target_score_at_gt": round(best_target_score_at_gt, 4) if is_canonical_unique else None,
            "native_target_operational": native_target_operational if is_canonical_unique else False
        })

    # Hard-Assert Invariants
    assert len(closure_records_1576) == EXPECTED_TOTAL_RESCUES, f"[FATAL] Expected 1576 records, got {len(closure_records_1576)}"
    assert cat1a_total == 1332, f"[FATAL] Expected 1,332 Cat-1A events, got {cat1a_total}"
    assert (cat1a_subthreshold_count + cat1a_active_count) == 1332, f"[FATAL] Cat-1A split mismatch: {cat1a_subthreshold_count + cat1a_active_count} != 1332"
    assert sum(track_a_attribution_counts.values()) == EXPECTED_TOTAL_RESCUES, "[FATAL] Track A sum != 1576"
    assert sum(track_b_attribution_counts.values()) == EXPECTED_TOTAL_RESCUES, "[FATAL] Track B sum != 1576"

    # =========================================================================
    # PRIMARY REPORT SECTION
    # =========================================================================
    print("=" * 115)
    print("BFCEF-A1C PRIMARY DIAGNOSTIC FINDINGS")
    print("=" * 115)

    # 1. Cat-1A Recomputed Operational Partition
    pct_sub = (cat1a_subthreshold_count / 1332.0) * 100.0
    pct_act = (cat1a_active_count / 1332.0) * 100.0
    print("[1] Cat-1A Recomputed Operational Breakdown (Total = 1,332 Events):")
    print(f"    - Cat 1A-Subthreshold (all Top100 support candidates s < 0.20): {cat1a_subthreshold_count:>5} / 1,332 ({pct_sub:>6.2f}%)")
    print(f"    - Cat 1A-Active       (>=1 Top100 support candidate s >= 0.20):   {cat1a_active_count:>5} / 1,332 ({pct_act:>6.2f}%)")

    # 2. Score Distributions
    s_supp_arr = np.array(cat1a_max_support_scores)
    s_t100_arr = np.array(cat1a_max_top100_support_scores)
    s_tmp_arr = np.array(cat1a_temporal_scores)
    s_dlt_arr = np.array(cat1a_score_deltas)

    print("\n[2] Cat-1A Score Percentiles (p10 / p25 / p50 / p75 / p90):")
    print(f"    - Max Support Native Score:        {np.percentile(s_supp_arr, 10):.4f} / {np.percentile(s_supp_arr, 25):.4f} / {np.percentile(s_supp_arr, 50):.4f} / {np.percentile(s_supp_arr, 75):.4f} / {np.percentile(s_supp_arr, 90):.4f}")
    print(f"    - Max Top-100 Support Score:       {np.percentile(s_t100_arr, 10):.4f} / {np.percentile(s_t100_arr, 25):.4f} / {np.percentile(s_t100_arr, 50):.4f} / {np.percentile(s_t100_arr, 75):.4f} / {np.percentile(s_t100_arr, 90):.4f}")
    print(f"    - Temporal Proposal Score:         {np.percentile(s_tmp_arr, 10):.4f} / {np.percentile(s_tmp_arr, 25):.4f} / {np.percentile(s_tmp_arr, 50):.4f} / {np.percentile(s_tmp_arr, 75):.4f} / {np.percentile(s_tmp_arr, 90):.4f}")
    print(f"    - Score Delta (Temporal - Top100): {np.percentile(s_dlt_arr, 10):+.4f} / {np.percentile(s_dlt_arr, 25):+.4f} / {np.percentile(s_dlt_arr, 50):+.4f} / {np.percentile(s_dlt_arr, 75):+.4f} / {np.percentile(s_dlt_arr, 90):+.4f}")

    # 3. Dual-Policy Attribution
    print("\n[3] Dual-Policy Ground-Truth Attribution Breakdown (All 1,576 Events):")
    print(f"    [Track A: Historical Baseline Attribution Proxy]")
    for k, v in track_a_attribution_counts.items():
        print(f"      - {k:<30}: {v:>5} / 1,576 ({(v/1576.0)*100:>5.2f}%)")
    print(f"    [Track B: Canonical R0 Evaluator Closure]")
    for k, v in track_b_attribution_counts.items():
        print(f"      - {k:<30}: {v:>5} / 1,576 ({(v/1576.0)*100:>5.2f}%)")

    # 4. Canonical-Unique GT-Evaluable Population Analysis
    n_canon_unique = len(canonical_unique_evaluable_records)
    pct_canon_unique = (n_canon_unique / EXPECTED_TOTAL_RESCUES) * 100.0

    u_tmp_iou = np.array([r["temp_gt_iou"] for r in canonical_unique_evaluable_records])
    u_nat_iou = np.array([r["native_gt_iou"] for r in canonical_unique_evaluable_records])
    u_same_iou = np.array([r["sameclass_native_gt_iou"] for r in canonical_unique_evaluable_records])
    u_nat_sc = np.array([r["best_target_score_at_gt"] for r in canonical_unique_evaluable_records])
    oper_rate = np.mean([1.0 if r["native_target_operational"] else 0.0 for r in canonical_unique_evaluable_records]) * 100.0

    print(f"\n[4] Canonical-Unique GT-Evaluable Analysis (N = {n_canon_unique} / 1,576 [{pct_canon_unique:.2f}%]):")
    print(f"    - Temporal -> GT IoU (p10 / p50 / p90):          {np.percentile(u_tmp_iou, 10):.4f} / {np.percentile(u_tmp_iou, 50):.4f} / {np.percentile(u_tmp_iou, 90):.4f}")
    print(f"    - Best-Any Native -> GT IoU (p10 / p50 / p90):   {np.percentile(u_nat_iou, 10):.4f} / {np.percentile(u_nat_iou, 50):.4f} / {np.percentile(u_nat_iou, 90):.4f}")
    print(f"    - Same-Class Native -> GT IoU (p10 / p50 / p90): {np.percentile(u_same_iou, 10):.4f} / {np.percentile(u_same_iou, 50):.4f} / {np.percentile(u_same_iou, 90):.4f}")
    print(f"    - Native Score at GT (p10 / p50 / p90):          {np.percentile(u_nat_sc, 10):.4f} / {np.percentile(u_nat_sc, 50):.4f} / {np.percentile(u_nat_sc, 90):.4f}")
    print(f"    - Conjoint Operational Candidate Rate:           {oper_rate:.2f}% (in Top100 AND s >= 0.20 on same candidate)")

    # 5. Cat-3 Dual Localization Gap
    if len(cat3_loc_gap_records) > 0:
        c3_gaps_same = [r["iou_gap_sameclass"] for r in cat3_loc_gap_records]
        c3_gaps_any = [r["iou_gap_any"] for r in cat3_loc_gap_records]
        c3_reaches = sum(1 for r in cat3_loc_gap_records if r["sameclass_reaches_030"])
        pct_c3_reaches = (c3_reaches / len(cat3_loc_gap_records)) * 100.0

        print(f"\n[5] Cat-3 Localization Gap Relative to Ground Truth (N = {len(cat3_loc_gap_records)}):")
        print(f"    - Same-Class IoU Gap (Temp - SameClass) p10/p50/p90: {np.percentile(c3_gaps_same, 10):+.4f} / {np.percentile(c3_gaps_same, 50):+.4f} / {np.percentile(c3_gaps_same, 90):+.4f}")
        print(f"    - Best-Any IoU Gap   (Temp - BestAny)   p10/p50/p90: {np.percentile(c3_gaps_any, 10):+.4f} / {np.percentile(c3_gaps_any, 50):+.4f} / {np.percentile(c3_gaps_any, 90):+.4f}")
        print(f"    - Same-Class Native Already Reaches IoU >= 0.30:     {c3_reaches} / {len(cat3_loc_gap_records)} ({pct_c3_reaches:.2f}%)")

    # 6. GT-Relative Geometric Absence (Correct Denominator)
    pct_abs_evaluable = (canonical_unique_absent_count / float(n_canon_unique)) * 100.0 if n_canon_unique > 0 else 0.0
    print(f"\n[6] GT-Relative Geometric Absence (Canonical-Unique Denominator, D-4420):")
    print(f"    - Canonical-Unique Coverage:            {n_canon_unique} / 1,576 ({pct_canon_unique:.2f}%)")
    print(f"    - No Close Native Query to GT (< 0.10): {canonical_unique_absent_count} / {n_canon_unique} ({pct_abs_evaluable:.2f}% of Canonical-Unique)")

    # Freeze Summary
    summary_payload = {
        "_meta": {
            "driver_sha256": driver_sha,
            "a1r_records_sha256": a1r_records_sha,
            "total_events": EXPECTED_TOTAL_RESCUES,
            "canonical_score_threshold": OPERATIONAL_SCORE_THRESHOLD,
            "canonical_iou_threshold": EVAL_IOU_THRESHOLD,
            "num_classes": NUM_CLASSES,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "cat1a_operational_split": {
            "total": cat1a_total,
            "subthreshold_count": cat1a_subthreshold_count,
            "subthreshold_pct": round(pct_sub, 2),
            "active_count": cat1a_active_count,
            "active_pct": round(pct_act, 2),
            "max_support_score_p10_p25_p50_p75_p90": [float(np.percentile(s_supp_arr, p)) for p in [10, 25, 50, 75, 90]],
            "max_top100_support_score_p10_p25_p50_p75_p90": [float(np.percentile(s_t100_arr, p)) for p in [10, 25, 50, 75, 90]],
            "temporal_score_p10_p25_p50_p75_p90": [float(np.percentile(s_tmp_arr, p)) for p in [10, 25, 50, 75, 90]],
            "score_delta_p10_p25_p50_p75_p90": [float(np.percentile(s_dlt_arr, p)) for p in [10, 25, 50, 75, 90]]
        },
        "track_a_historical_attribution": {
            k: {"count": v, "pct": round((v / EXPECTED_TOTAL_RESCUES) * 100.0, 2)}
            for k, v in track_a_attribution_counts.items()
        },
        "track_b_canonical_attribution": {
            k: {"count": v, "pct": round((v / EXPECTED_TOTAL_RESCUES) * 100.0, 2)}
            for k, v in track_b_attribution_counts.items()
        },
        "canonical_unique_analysis": {
            "evaluable_count": n_canon_unique,
            "evaluable_coverage_pct": round(pct_canon_unique, 2),
            "temp_gt_iou_p10_p50_p90": [float(np.percentile(u_tmp_iou, p)) for p in [10, 50, 90]],
            "native_gt_iou_p10_p50_p90": [float(np.percentile(u_nat_iou, p)) for p in [10, 50, 90]],
            "sameclass_native_gt_iou_p10_p50_p90": [float(np.percentile(u_same_iou, p)) for p in [10, 50, 90]],
            "native_score_p10_p50_p90": [float(np.percentile(u_nat_sc, p)) for p in [10, 50, 90]],
            "conjoint_operational_candidate_pct": round(oper_rate, 2),
            "gt_relative_absent_count": canonical_unique_absent_count,
            "gt_relative_absent_pct": round(pct_abs_evaluable, 2)
        },
        "cat3_localization_gap": {
            "count": len(cat3_loc_gap_records),
            "iou_gap_sameclass_p10_p50_p90": [float(np.percentile(c3_gaps_same, p)) for p in [10, 50, 90]] if len(cat3_loc_gap_records) > 0 else [],
            "iou_gap_any_p10_p50_p90": [float(np.percentile(c3_gaps_any, p)) for p in [10, 50, 90]] if len(cat3_loc_gap_records) > 0 else [],
            "sameclass_reaches_030_count": c3_reaches if len(cat3_loc_gap_records) > 0 else 0
        }
    }

    with open(A1C_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    with open(A1C_RECORDS_JSON, "w", encoding="utf-8") as f:
        json.dump(closure_records_1576, f, indent=2)

    print("\n" + "=" * 115)
    print(f"[+] BFCEF-A1C EXECUTION COMPLETED IN {time.time() - t_start:.2f}s")
    print(f"[+] Summary JSON: {A1C_SUMMARY_JSON.resolve()}")
    print(f"[+] Records JSON: {A1C_RECORDS_JSON.resolve()}")
    print("=" * 115 + "\n", flush=True)


if __name__ == "__main__":
    main()