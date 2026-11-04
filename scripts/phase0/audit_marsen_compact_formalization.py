"""
D:/MARIVISION/H0_workbench/eval_tools/audit_marsen_compact_formalization.py
MARSEN-Compact Formalization Engine: Paired Capacity Control, Exact OOF Sweep, Retrained Ablation (v3)

Strict Compliance Checklist:
  - D-3777..D-3789: MARSEN-Compact architecture; 10 design videos ONLY; calibration/test UNTOUCHED.
  - P-1840: Corrected Seed-42 reproduction retrieval eliminating the rec_95_42 KeyError.
  - P-1841: Globally exact unique OOF score sweep over all threshold-induced cut-points.
  - P-1842: Paired multi-seed capacity control (Seeds 41, 42, 43) comparing 143->188->1 vs 848->32->1.
  - P-1843: Strict OOF score integrity: asserts key membership in score_map_ref; no silent zeros.
  - P-1832: Full retrained compact branch ablation for all 8 maritime evidence families.
  - P-1833: Compact fold-score calibration audit (medians, percentiles, max scores).
  - P-1837: Operational delete-one-video F1 sensitivity: re-evaluates Delta F1_(-v) vs remaining-frame M0.
  - P-1839 / D-3785: Multi-seed summary (mean, std, min, max) using Seed 41 as REFERENCE_SEED.
"""

import os
import sys
import copy
import json
import hashlib
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier

OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
TRAIN_JSON = Path("D:/MARIVISION/seaclips-train.json")
SPLIT_MANIFEST_JSON = OUT_DIR / "pol2_candidate2m_micro_split.json"
QUERY_DATASET_PTH = OUT_DIR / "pol3v_design_queries_dataset_v7.pth"
FRAME_TENSORS_PTH = OUT_DIR / "pol3v_design_frames_tensors_v7.pth"
CACHED_BRANCHES_PTH = OUT_DIR / "marsen_micro_authentic_branches_1915.pth"
RESULTS_JSON = OUT_DIR / "marsen_compact_formalization_results.json"

SCORE_THRESHOLD = 0.20
EVAL_IOU_THRESHOLD = 0.30
BASE_M0_F1 = 0.431718
ORIGINAL_4M_F1 = 0.439957
ORIGINAL_4M_AP = 0.0880
DIAG_COMPACT_AP = 0.1094
DIAG_COMPACT_F1_95 = 0.441176

SEEDS = [41, 42, 43]
REFERENCE_SEED = 41


def compute_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def box_iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[0] + b1[2], b2[0] + b2[2]), min(b1[1] + b1[3], b2[1] + b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = b1[2] * b1[3] + b2[2] * b2[3] - inter
    return inter / union if union > 0.0 else 0.0


def evaluate_operational_top100(logits, boxes, gt_boxes_px, gt_cats, label_to_cat_id, img_w, img_h):
    prob = logits.sigmoid()
    num_classes = prob.shape[-1]
    topk_scores, topk_idx = torch.topk(prob.flatten(), min(100, prob.numel()))
    q_idx = torch.div(topk_idx, num_classes, rounding_mode="floor")
    c_idx = topk_idx % num_classes
    boxes_ranked = boxes[q_idx]

    pred_cat_ids = np.array([label_to_cat_id[int(lbl.item())] for lbl in c_idx])
    pred_boxes_px = []
    for b in boxes_ranked:
        cx, cy, w, h = b.tolist()
        pred_boxes_px.append([(cx - w / 2.0) * img_w, (cy - h / 2.0) * img_h, w * img_w, h * img_h])
    pred_scores = topk_scores.cpu().numpy()

    n_gt = len(gt_cats)
    gt_matched = [False] * n_gt
    tp = 0
    fp = 0

    for p_idx in range(len(pred_scores)):
        if pred_scores[p_idx] >= SCORE_THRESHOLD:
            p_box = pred_boxes_px[p_idx]
            p_cat = pred_cat_ids[p_idx]
            best_iou = -1.0
            best_g = -1
            for g_idx in range(n_gt):
                if not gt_matched[g_idx] and gt_cats[g_idx] == p_cat:
                    iou_g = box_iou(gt_boxes_px[g_idx], p_box)
                    if iou_g >= EVAL_IOU_THRESHOLD and iou_g > best_iou:
                        best_iou = iou_g
                        best_g = g_idx
            if best_g >= 0:
                gt_matched[best_g] = True
                tp += 1
            else:
                fp += 1

    fn = n_gt - sum(gt_matched)
    return tp, fp, fn


# ------------------------------------------------------------------------------
# P-1841: GLOBALLY EXACT UNIQUE OOF SCORE SWEEP ENGINE (WITH FRAME ACCELERATION)
# ------------------------------------------------------------------------------

def get_globally_exact_thresholds(oof_scores):
    scores = np.asarray(oof_scores, dtype=np.float64)
    assert np.isfinite(scores).all(), "[FATAL] Non-finite scores in threshold extraction!"
    unique_scores = np.unique(scores)
    thresholds = unique_scores.tolist()
    # Add sentinel cutoff above maximum score to represent the reject-all baseline state
    thresholds.append(float(np.nextafter(unique_scores.max(), np.inf)))
    return sorted(list(set(thresholds)))


def run_detector_sweep_exact(score_map, frame_records, active_key_set, positive_key_set, label_to_cat_id, threshold_list, base_frame_cache=None):
    # Precompute baseline M0 frame metrics if not supplied
    if base_frame_cache is None:
        base_frame_cache = {}
        for fr in frame_records:
            tpb, fpb, fnb = evaluate_operational_top100(
                fr["d_logits"], fr["d_boxes"], fr["gt_boxes_px"], fr["gt_cats"], label_to_cat_id, fr["img_w"], fr["img_h"]
            )
            base_frame_cache[fr["image_id"]] = (tpb, fpb, fnb)

    best_tau = None
    best_f1 = -1.0
    best_rec = None
    sweep_records = {}

    for tau in threshold_list:
        tot_tp, tot_fp, tot_fn = 0, 0, 0
        admitted = 0
        retained_pos = 0

        for fr in frame_records:
            n_t = fr["n_temp"]
            iid = fr["image_id"]

            if n_t == 0:
                tpd, fpd, fnd = base_frame_cache[iid]
                tot_tp += tpd
                tot_fp += fpd
                tot_fn += fnd
                continue

            # Check if any temporal query in this frame is admitted at tau
            frame_has_admitted = False
            for q_t in range(n_t):
                key = (iid, q_t)
                if key in active_key_set:
                    assert key in score_map, f"[FATAL] Missing OOF score for active query {key}"
                    if score_map[key] >= tau:
                        frame_has_admitted = True
                        break

            # Fast-path: if no queries admitted in this frame, reuse baseline M0 counts
            if not frame_has_admitted:
                tpd, fpd, fnd = base_frame_cache[iid]
                tot_tp += tpd
                tot_fp += fpd
                tot_fn += fnd
                continue

            # Evaluate frame with admitted queries
            joint_logits = torch.cat([fr["d_logits"], fr["temp_logits"]], dim=0).clone()
            joint_boxes = torch.cat([fr["d_boxes"], fr["temp_boxes"]], dim=0)

            for q_t in range(n_t):
                key = (iid, q_t)
                if key in active_key_set and score_map[key] >= tau:
                    admitted += 1
                    if key in positive_key_set:
                        retained_pos += 1
                else:
                    joint_logits[300 + q_t, :] = -1e9

            tpf, fpf, fnf = evaluate_operational_top100(
                joint_logits, joint_boxes, fr["gt_boxes_px"], fr["gt_cats"], label_to_cat_id, fr["img_w"], fr["img_h"]
            )
            tot_tp += tpf
            tot_fp += fpf
            tot_fn += fnf

        prec = tot_tp / max(1, tot_tp + tot_fp)
        rec = tot_tp / max(1, tot_tp + tot_fn)
        f1 = 2 * prec * rec / max(1e-7, prec + rec)

        sweep_records[tau] = {
            "tau": float(tau), "f1": float(f1), "tp": tot_tp, "fp": tot_fp, "fn": tot_fn,
            "prec": float(prec), "rec": float(rec), "admitted": admitted, "retained_pos": retained_pos
        }

        # Deterministic tie-break favoring higher tau (stronger suppression)
        if f1 > best_f1 + 1e-12 or (abs(f1 - best_f1) <= 1e-12 and (best_tau is None or tau > best_tau)):
            best_f1 = f1
            best_tau = tau
            best_rec = sweep_records[tau]

    return best_tau, best_f1, best_rec, sweep_records


def train_lovo_mlp(X_feat, y_act, v_act, design_vids, hidden_layer_sizes, seed):
    n_samples = len(y_act)
    oof_scores = np.full(n_samples, np.nan)

    for vid in design_vids:
        test_mask = (v_act == vid)
        train_mask = ~test_mask

        X_tr, y_tr = X_feat[train_mask], y_act[train_mask]
        X_te = X_feat[test_mask]

        if len(X_te) == 0 or len(np.unique(y_tr)) < 2:
            continue

        m_tr = X_tr.mean(axis=0)
        s_tr = X_tr.std(axis=0)
        s_tr[s_tr < 1e-6] = 1.0

        X_tr_std = (X_tr - m_tr) / s_tr
        X_te_std = (X_te - m_tr) / s_tr

        pos_idx = np.where(y_tr == 1)[0]
        neg_idx = np.where(y_tr == 0)[0]
        rng = np.random.default_rng(seed * 1000 + vid)
        oversampled_pos = rng.choice(pos_idx, size=len(neg_idx), replace=True)
        bal_idx = np.concatenate([neg_idx, oversampled_pos])
        rng.shuffle(bal_idx)

        clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation="relu", alpha=0.05, max_iter=350, random_state=seed)
        clf.fit(X_tr_std[bal_idx], y_tr[bal_idx])
        oof_scores[test_mask] = clf.predict_proba(X_te_std)[:, 1]

    assert np.isfinite(oof_scores).all(), f"[FATAL] Non-finite scores generated for seed {seed}!"
    return oof_scores


def main():
    driver_sha256 = compute_sha256(Path(__file__))
    print("=" * 115)
    print("MARSEN-COMPACT (MARSEN-C) FORMALIZATION ENGINE (v3, P-1840..P-1843)")
    print(f"Driver Script SHA-256: {driver_sha256}")
    print("=" * 115)

    with open(SPLIT_MANIFEST_JSON, "r", encoding="utf-8") as f:
        design_vids = sorted(json.load(f)["_meta"]["train_videos"])

    cached_queries = torch.load(QUERY_DATASET_PTH, map_location="cpu")["extracted_queries"]
    frame_bundle = torch.load(FRAME_TENSORS_PTH, map_location="cpu")
    frame_records = frame_bundle["frame_records"]

    active_indices = [idx for idx, q in enumerate(cached_queries) if q["is_active"]]
    active_queries = [cached_queries[idx] for idx in active_indices]
    n_active = len(active_queries)
    y_act = np.array([q["label_op_util"] for q in active_queries], dtype=np.int32)
    v_act = np.array([q["video_id"] for q in active_queries], dtype=np.int32)
    base_prev = float(np.mean(y_act))

    branch_bundle = torch.load(CACHED_BRANCHES_PTH, map_location="cpu")
    branches = branch_bundle["branches"]
    print(f"[+] Loaded Authentic Precomputed Evidence Branches for {n_active:,} Active Queries.")

    branch_order = ["global", "hr", "memory", "horizon", "scene", "motion", "reflection", "metadata"]
    X_full_848 = np.concatenate([branches[k] for k in branch_order], axis=-1)
    assert X_full_848.shape == (n_active, 848)

    # Reconstruct 143-D Native HR + Metadata representation for Candidate 4M
    X_4m_143 = np.concatenate([branches["hr"], branches["metadata"][:, :15]], axis=-1)
    assert X_4m_143.shape == (n_active, 143)

    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        coco_full = json.load(f)
    cat_ids = sorted(c["id"] for c in coco_full["categories"])
    label_to_cat_id = {i: cat_id for i, cat_id in enumerate(cat_ids)}

    active_key_set = set((q["image_id"], q["query_id"]) for q in active_queries)
    positive_key_set = set((q["image_id"], q["query_id"]) for q in active_queries if q["label_op_util"] == 1)

    # Precompute baseline M0 frame metrics
    base_frame_cache = {}
    m0_tp, m0_fp, m0_fn = 0, 0, 0
    for fr in frame_records:
        tpb, fpb, fnb = evaluate_operational_top100(
            fr["d_logits"], fr["d_boxes"], fr["gt_boxes_px"], fr["gt_cats"], label_to_cat_id, fr["img_w"], fr["img_h"]
        )
        base_frame_cache[fr["image_id"]] = (tpb, fpb, fnb)
        m0_tp += tpb
        m0_fp += fpb
        m0_fn += fnb

    calculated_base_f1 = 2 * m0_tp / max(1e-7, 2 * m0_tp + m0_fp + m0_fn)
    assert abs(calculated_base_f1 - BASE_M0_F1) < 1e-5

    # --------------------------------------------------------------------------
    # P-1838 & P-1840: COMPACT REPRODUCTION GUARD (SEED 42)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("P-1838 / P-1840: COMPACT REPRODUCTION GUARD (SEED 42)")
    print("=" * 115)

    oof_42 = train_lovo_mlp(X_full_848, y_act, v_act, design_vids, hidden_layer_sizes=(32,), seed=42)
    ap_42 = float(average_precision_score(y_act, oof_42))
    auc_42 = float(roc_auc_score(y_act, oof_42))

    score_map_42 = { (q["image_id"], q["query_id"]): float(oof_42[i]) for i, q in enumerate(active_queries) }

    # P-1840: Correctly bind single-threshold record
    _, _, rec_95, _ = run_detector_sweep_exact(
        score_map_42, frame_records, active_key_set, positive_key_set, label_to_cat_id, [0.95], base_frame_cache
    )

    print(f"[*] Diagnostic Seed 42 AP:        {ap_42:.4f} (Expected: {DIAG_COMPACT_AP:.4f})")
    print(f"[*] Diagnostic Seed 42 F1 @ 0.95: {rec_95['f1']:.6f} (Expected: {DIAG_COMPACT_F1_95:.6f})")
    print(f"[*] Diagnostic Seed 42 Counts:    TP={rec_95['tp']}, FP={rec_95['fp']}, FN={rec_95['fn']}")

    assert abs(ap_42 - DIAG_COMPACT_AP) < 1e-3, f"[FATAL] AP reproduction mismatch: {ap_42} != {DIAG_COMPACT_AP}"
    assert abs(rec_95["f1"] - DIAG_COMPACT_F1_95) < 1e-4, f"[FATAL] F1@0.95 mismatch: {rec_95['f1']} != {DIAG_COMPACT_F1_95}"
    assert rec_95["tp"] == 210 and rec_95["fp"] == 329 and rec_95["fn"] == 203, "[FATAL] Diagnostic count mismatch!"
    print("[+] P-1838 / P-1840: Seed 42 Diagnostic Benchmark Reconstructed with 100% Exact Parity.")

    # --------------------------------------------------------------------------
    # P-1842 & P-1841: PAIRED MULTI-SEED CAPACITY CONTROL (SEEDS 41, 42, 43)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("P-1842: PAIRED MULTI-SEED CAPACITY CONTROL (4M-CAPACITY [27,261] vs. MARSEN-C [27,201])")
    print("=" * 115)
    print(f"{'Seed':<6} | {'4M-Cap AP':<12} | {'MARSEN-C AP':<14} | {'Delta AP':<12} | {'4M-Cap F1*':<14} | {'MARSEN-C F1*':<14} | {'Delta F1*':<12} | {'Exact tau* (MARSEN-C)'}")
    print("-" * 115)

    marsen_seed_results = {}
    cap_seed_results = {}
    marsen_oof_map = {}

    for s in SEEDS:
        # 1. Evaluate 4M-Capacity Control (143 -> 188 -> 1, 27,261 params)
        oof_cap_s = train_lovo_mlp(X_4m_143, y_act, v_act, design_vids, hidden_layer_sizes=(188,), seed=s)
        ap_cap_s = float(average_precision_score(y_act, oof_cap_s))
        auc_cap_s = float(roc_auc_score(y_act, oof_cap_s))

        taus_cap_s = get_globally_exact_thresholds(oof_cap_s)
        score_map_cap_s = { (q["image_id"], q["query_id"]): float(oof_cap_s[i]) for i, q in enumerate(active_queries) }
        best_tau_cap_s, best_f1_cap_s, best_rec_cap_s, _ = run_detector_sweep_exact(
            score_map_cap_s, frame_records, active_key_set, positive_key_set, label_to_cat_id, taus_cap_s, base_frame_cache
        )

        cap_seed_results[s] = {
            "ap": ap_cap_s, "auc": auc_cap_s, "tau_star": best_tau_cap_s, "f1_star": best_f1_cap_s,
            "delta_f1": best_f1_cap_s - BASE_M0_F1, "best_rec": best_rec_cap_s
        }

        # 2. Evaluate MARSEN-Compact (848 -> 32 -> 1, 27,201 params)
        oof_mar_s = train_lovo_mlp(X_full_848, y_act, v_act, design_vids, hidden_layer_sizes=(32,), seed=s) if s != 42 else oof_42
        ap_mar_s = float(average_precision_score(y_act, oof_mar_s))
        auc_mar_s = float(roc_auc_score(y_act, oof_mar_s))

        taus_mar_s = get_globally_exact_thresholds(oof_mar_s)
        score_map_mar_s = { (q["image_id"], q["query_id"]): float(oof_mar_s[i]) for i, q in enumerate(active_queries) }
        best_tau_mar_s, best_f1_mar_s, best_rec_mar_s, _ = run_detector_sweep_exact(
            score_map_mar_s, frame_records, active_key_set, positive_key_set, label_to_cat_id, taus_mar_s, base_frame_cache
        )

        marsen_seed_results[s] = {
            "ap": ap_mar_s, "auc": auc_mar_s, "tau_star": best_tau_mar_s, "f1_star": best_f1_mar_s,
            "delta_f1": best_f1_mar_s - BASE_M0_F1, "best_rec": best_rec_mar_s, "n_unique_taus": len(taus_mar_s)
        }
        marsen_oof_map[s] = oof_mar_s

        d_ap_s = ap_mar_s - ap_cap_s
        d_f1_s = best_f1_mar_s - best_f1_cap_s
        role = " (Ref)" if s == REFERENCE_SEED else ""
        print(f"Seed {s}{role:<4} | {ap_cap_s:10.4f}   | {ap_mar_s:12.4f}   | {d_ap_s:+10.4f}   | {best_f1_cap_s:12.6f}  | {best_f1_mar_s:12.6f}  | {d_f1_s:+10.4f}  | {best_tau_mar_s:12.6f}")

    # Summary statistics across paired seeds
    paired_d_aps = [marsen_seed_results[s]["ap"] - cap_seed_results[s]["ap"] for s in SEEDS]
    paired_d_f1s = [marsen_seed_results[s]["f1_star"] - cap_seed_results[s]["f1_star"] for s in SEEDS]

    print(f"\n[*] Paired Capacity-Control Summary (MARSEN-C vs. 4M-Capacity):")
    print(f"  - Paired Delta AP:  mean = {np.mean(paired_d_aps):+.4f} +/- {np.std(paired_d_aps):.4f} (min = {np.min(paired_d_aps):+.4f}, max = {np.max(paired_d_aps):+.4f})")
    print(f"  - Paired Delta F1*: mean = {np.mean(paired_d_f1s):+.6f} +/- {np.std(paired_d_f1s):.6f} (min = {np.min(paired_d_f1s):+.6f}, max = {np.max(paired_d_f1s):+.6f})")

    # P-1839: MARSEN-C Multi-seed summary
    mar_aps = [marsen_seed_results[s]["ap"] for s in SEEDS]
    mar_f1s = [marsen_seed_results[s]["f1_star"] for s in SEEDS]
    mar_df1s = [marsen_seed_results[s]["delta_f1"] for s in SEEDS]

    print(f"\n[*] MARSEN-C Standalone Performance (Seeds 41, 42, 43):")
    print(f"  - Pooled AP:        mean = {np.mean(mar_aps):.4f} +/- {np.std(mar_aps):.4f} (min = {np.min(mar_aps):.4f}, max = {np.max(mar_aps):.4f})")
    print(f"  - Optimal F1*:      mean = {np.mean(mar_f1s):.6f} +/- {np.std(mar_f1s):.6f} (min = {np.min(mar_f1s):.6f}, max = {np.max(mar_f1s):.6f})")
    print(f"  - Net Delta F1 vs M0: mean = {np.mean(mar_df1s):+.4f} +/- {np.std(mar_df1s):.4f} (min = {np.min(mar_df1s):+.4f}, max = {np.max(mar_df1s):+.4f})")

    # --------------------------------------------------------------------------
    # P-1833: MARSEN-COMPACT FOLD CALIBRATION AUDIT (REFERENCE SEED 41)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print(f"P-1833: COMPACT FOLD-SCORE CALIBRATION AUDIT (REFERENCE SEED {REFERENCE_SEED} AT tau* = {marsen_seed_results[REFERENCE_SEED]['tau_star']:.6f})")
    print("=" * 115)
    print(f"{'Video ID':<12} | {'Pos / Total':<14} | {'Pos Med':<10} | {'Neg Med':<10} | {'P90 Score':<12} | {'P95 Score':<12} | {'Max Score':<12} | {'Adm @ tau*':<14} | {'Video AP'}")
    print("-" * 120)

    compact_oof_ref = marsen_oof_map[REFERENCE_SEED]
    opt_tau_ref = marsen_seed_results[REFERENCE_SEED]["tau_star"]
    compact_calib_audit = {}

    for vid in design_vids:
        mask_v = (v_act == vid)
        y_v = y_act[mask_v]
        sc_v = compact_oof_ref[mask_v]
        n_tot = len(y_v)
        n_p = int(y_v.sum())

        pos_sc = sc_v[y_v == 1]
        neg_sc = sc_v[y_v == 0]
        pos_med = float(np.median(pos_sc)) if len(pos_sc) > 0 else np.nan
        neg_med = float(np.median(neg_sc)) if len(neg_sc) > 0 else np.nan
        p90 = float(np.percentile(sc_v, 90))
        p95 = float(np.percentile(sc_v, 95))
        max_s = float(np.max(sc_v))
        adm_opt = int(np.sum(sc_v >= opt_tau_ref))
        adm_p_opt = int(np.sum((sc_v >= opt_tau_ref) & (y_v == 1)))
        ap_v = float(average_precision_score(y_v, sc_v)) if (0 < n_p < n_tot) else np.nan

        compact_calib_audit[str(vid)] = {
            "n_total": n_tot, "n_pos": n_p, "pos_median": pos_med, "neg_median": neg_med,
            "p90": p90, "p95": p95, "max": max_s, "admitted_at_tau": adm_opt, "admitted_pos_at_tau": adm_p_opt, "ap": ap_v
        }

        pos_med_str = f"{pos_med:.4f}" if not np.isnan(pos_med) else "  -   "
        neg_med_str = f"{neg_med:.4f}" if not np.isnan(neg_med) else "  -   "
        ap_str = f"{ap_v:.4f}" if not np.isnan(ap_v) else "  -   "
        adm_str = f"{adm_opt} ({adm_p_opt})"
        print(f"video_{vid:<6d} | {n_p:2d} / {n_tot:<8d} | {pos_med_str:<10} | {neg_med_str:<10} | {p90:<12.4f} | {p95:<12.4f} | {max_s:<12.4f} | {adm_str:<14} | {ap_str}")

    # --------------------------------------------------------------------------
    # P-1832: RETRAINED COMPACT BRANCH ABLATION AUDIT (REFERENCE SEED 41)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print(f"P-1832: RETRAINED COMPACT BRANCH ABLATION MATRIX (REFERENCE SEED {REFERENCE_SEED})")
    print("=" * 115)
    print(f"{'Ablated Branch':<24} | {'Retrained Dim':<14} | {'Ablated AP':<12} | {'Delta AP':<10} | {'Exact tau*':<14} | {'Best F1*':<10} | {'Delta vs Full':<14} | {'Delta F1 vs M0'}")
    print("-" * 130)

    full_ap_ref = marsen_seed_results[REFERENCE_SEED]["ap"]
    full_f1_ref = marsen_seed_results[REFERENCE_SEED]["f1_star"]
    print(f"{'None (Full MARSEN-C)':<24} | {848:<14d} | {full_ap_ref:10.4f}   | {'REF':<10} | {opt_tau_ref:<14.6f} | {full_f1_ref:8.6f}   | {'REF':<14} | {full_f1_ref - BASE_M0_F1:+12.4f}")

    retrained_ablations = {}

    for b_drop in branch_order:
        sub_branches = [k for k in branch_order if k != b_drop]
        X_sub = np.concatenate([branches[k] for k in sub_branches], axis=-1)
        sub_dim = X_sub.shape[-1]

        oof_sub = train_lovo_mlp(X_sub, y_act, v_act, design_vids, hidden_layer_sizes=(32,), seed=REFERENCE_SEED)
        ap_sub = float(average_precision_score(y_act, oof_sub))
        auc_sub = float(roc_auc_score(y_act, oof_sub))
        d_ap = ap_sub - full_ap_ref

        taus_sub = get_globally_exact_thresholds(oof_sub)
        score_map_sub = { (q["image_id"], q["query_id"]): float(oof_sub[i]) for i, q in enumerate(active_queries) }
        best_tau_sub, best_f1_sub, best_rec_sub, _ = run_detector_sweep_exact(
            score_map_sub, frame_records, active_key_set, positive_key_set, label_to_cat_id, taus_sub, base_frame_cache
        )

        d_f1_full = best_f1_sub - full_f1_ref
        d_f1_m0 = best_f1_sub - BASE_M0_F1

        retrained_ablations[b_drop] = {
            "dim": sub_dim, "ap": ap_sub, "delta_ap": d_ap, "auc": auc_sub,
            "tau_star": best_tau_sub, "f1_star": best_f1_sub, "delta_vs_full": d_f1_full,
            "delta_vs_m0": d_f1_m0
        }

        print(f"(- {b_drop:<21}) | {sub_dim:<14d} | {ap_sub:10.4f}   | {d_ap:+10.4f} | {best_tau_sub:<14.6f} | {best_f1_sub:8.6f}   | {d_f1_full:+14.4f} | {d_f1_m0:+12.4f}")

    # --------------------------------------------------------------------------
    # P-1837 & P-1843: OPERATIONAL DELETE-ONE-VIDEO SENSITIVITY AUDIT
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("P-1837: OPERATIONAL DELETE-ONE-VIDEO SENSITIVITY AUDIT (RE-EVALUATING Delta F1_(-v) vs M0_(-v))")
    print("=" * 115)
    print(f"{'Omitted Video Sequence':<24} | {'Sub-Cohort Queries (Pos)':<28} | {'Remaining M0 F1':<16} | {'Remaining MARSEN F1':<20} | {'Delta F1_(-v)':<14} | {'Sub-Cohort AP':<14} | {'Delta AP'}")
    print("-" * 132)

    pos_vids = [vid for vid in design_vids if sum(1 for it in active_queries if it["video_id"] == vid and it["label_op_util"] == 1) > 0]
    delete_video_sensitivities = {}
    score_map_ref = { (q["image_id"], q["query_id"]): float(compact_oof_ref[i]) for i, q in enumerate(active_queries) }

    for v_omit in pos_vids:
        mask_sub = (v_act != v_omit)
        y_sub = y_act[mask_sub]
        sc_sub = compact_oof_ref[mask_sub]
        ap_sub = float(average_precision_score(y_sub, sc_sub))
        d_ap = ap_sub - full_ap_ref

        sub_frames = [fr for fr in frame_records if fr["video_id"] != v_omit]

        # 1. Re-evaluate baseline M0 on remaining frames
        b_tp_sub, b_fp_sub, b_fn_sub = 0, 0, 0
        for fr in sub_frames:
            tpb, fpb, fnb = base_frame_cache[fr["image_id"]]
            b_tp_sub += tpb
            b_fp_sub += fpb
            b_fn_sub += fnb
        m0_f1_sub = 2 * b_tp_sub / max(1e-7, 2 * b_tp_sub + b_fp_sub + b_fn_sub)

        # 2. P-1843: Evaluate MARSEN-C on remaining frames with strict membership assertions
        m_tp_sub, m_fp_sub, m_fn_sub = 0, 0, 0
        for fr in sub_frames:
            n_t = fr["n_temp"]
            iid = fr["image_id"]

            if n_t == 0:
                tpd, fpd, fnd = base_frame_cache[iid]
                m_tp_sub += tpd
                m_fp_sub += fpd
                m_fn_sub += fnd
                continue

            frame_has_admitted = False
            for q_t in range(n_t):
                key = (iid, q_t)
                if key in active_key_set:
                    assert key in score_map_ref, f"[FATAL] Missing reference score for {key}"
                    if score_map_ref[key] >= opt_tau_ref:
                        frame_has_admitted = True
                        break

            if not frame_has_admitted:
                tpd, fpd, fnd = base_frame_cache[iid]
                m_tp_sub += tpd
                m_fp_sub += fpd
                m_fn_sub += fnd
                continue

            joint_logits = torch.cat([fr["d_logits"], fr["temp_logits"]], dim=0).clone()
            joint_boxes = torch.cat([fr["d_boxes"], fr["temp_boxes"]], dim=0)

            for q_t in range(n_t):
                key = (iid, q_t)
                if key in active_key_set and score_map_ref[key] >= opt_tau_ref:
                    pass
                else:
                    joint_logits[300 + q_t, :] = -1e9

            tpm, fpm, fnm = evaluate_operational_top100(
                joint_logits, joint_boxes, fr["gt_boxes_px"], fr["gt_cats"], label_to_cat_id, fr["img_w"], fr["img_h"]
            )
            m_tp_sub += tpm
            m_fp_sub += fpm
            m_fn_sub += fnm

        marsen_f1_sub = 2 * m_tp_sub / max(1e-7, 2 * m_tp_sub + m_fp_sub + m_fn_sub)
        d_f1_sub = marsen_f1_sub - m0_f1_sub

        delete_video_sensitivities[str(v_omit)] = {
            "n_remaining_queries": int(mask_sub.sum()),
            "n_remaining_positives": int(y_sub.sum()),
            "m0_f1_remaining": float(m0_f1_sub),
            "marsen_f1_remaining": float(marsen_f1_sub),
            "delta_f1_remaining": float(d_f1_sub),
            "ap_remaining": float(ap_sub),
            "delta_ap_remaining": float(d_ap)
        }

        q_str = f"{int(mask_sub.sum())} ({int(y_sub.sum())} pos)"
        print(f"Omitting video_{v_omit:<13d} | {q_str:<28} | {m0_f1_sub:14.6f}  | {marsen_f1_sub:18.6f}  | {d_f1_sub:+12.4f}  | {ap_sub:12.4f}  | {d_ap:+10.4f}")

    # --------------------------------------------------------------------------
    # SERIALIZATION
    # --------------------------------------------------------------------------
    results_payload = {
        "_meta": {
            "artifact_name": "marsen_compact_formalization_results_v3",
            "driver_sha256": driver_sha256,
            "baseline_m0_f1": float(BASE_M0_F1),
            "original_4m_f1": float(ORIGINAL_4M_F1),
            "original_4m_ap": float(ORIGINAL_4M_AP),
            "reference_seed": REFERENCE_SEED,
            "marsen_c_reference_f1": float(marsen_seed_results[REFERENCE_SEED]["f1_star"]),
            "marsen_c_reference_ap": float(marsen_seed_results[REFERENCE_SEED]["ap"]),
            "marsen_c_reference_delta_f1": float(marsen_seed_results[REFERENCE_SEED]["delta_f1"]),
            "trainable_parameters_marsen_c": 27201,
            "trainable_parameters_4m_cap": 27261
        },
        "paired_capacity_control": {
            str(s): {
                "cap_ap": cap_seed_results[s]["ap"], "mar_ap": marsen_seed_results[s]["ap"],
                "delta_ap": marsen_seed_results[s]["ap"] - cap_seed_results[s]["ap"],
                "cap_f1": cap_seed_results[s]["f1_star"], "mar_f1": marsen_seed_results[s]["f1_star"],
                "delta_f1": marsen_seed_results[s]["f1_star"] - cap_seed_results[s]["f1_star"]
            } for s in SEEDS
        },
        "multi_seed_audit_marsen_c": {str(s): marsen_seed_results[s] for s in SEEDS},
        "calibration_audit_ref_seed": compact_calib_audit,
        "retrained_ablations": retrained_ablations,
        "delete_video_sensitivities": delete_video_sensitivities
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    res_hash = compute_sha256(RESULTS_JSON)
    print("\n" + "=" * 115)
    print(f"[+] MARSEN-Compact Results Frozen: {RESULTS_JSON.resolve()}")
    print(f"    Manifest SHA-256:              {res_hash}")
    print("=" * 115)


if __name__ == "__main__":
    main()