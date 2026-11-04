"""
D:/MARIVISION/H0_workbench/eval_tools/audit_marsen_forward_addition.py
MARSEN-C Forward-Addition Audit: Isolating Incremental Maritime Signal over Base B (P-1846)

Strict Compliance Checklist:
  - D-3793..D-3796: Design cohort ONLY (10 videos, 206 frames); calibration/test UNTOUCHED.
  - P-1840: Seed-42 reproduction guard certifying diagnostic baseline (AP=0.1094, F1@0.95=0.441176).
  - P-1841: Globally exact unique OOF score cut-points for every evaluated configuration.
  - P-1846: Forward-addition audit testing B + {Motion, Scene, Horizon, Reflection, Global, Memory}.
  - Capacity Invariant: All 7 configurations parameter-matched to ~27.2k parameters.
  - Paired Multi-Seed Audit: Evaluates Seeds 41, 42, 43 reporting paired Delta AP and Delta F1 vs. Base B.
  - Fast-Path Execution: Accelerated frame evaluation runs complete audit in under 60 seconds on RTX 3060.
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
RESULTS_JSON = OUT_DIR / "marsen_forward_addition_results.json"

SCORE_THRESHOLD = 0.20
EVAL_IOU_THRESHOLD = 0.30
BASE_M0_F1 = 0.431718
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


def get_globally_exact_thresholds(oof_scores):
    scores = np.asarray(oof_scores, dtype=np.float64)
    assert np.isfinite(scores).all(), "[FATAL] Non-finite scores in threshold extraction!"
    unique_scores = np.unique(scores)
    thresholds = unique_scores.tolist()
    thresholds.append(float(np.nextafter(unique_scores.max(), np.inf)))
    return sorted(list(set(thresholds)))


def run_detector_sweep_fast(score_map, frame_records, active_key_set, positive_key_set, label_to_cat_id, threshold_list, base_frame_cache):
    best_tau = None
    best_f1 = -1.0
    best_rec = None
    sweep_records = {}

    # Pre-filter frames with temporal queries
    temp_frames = [fr for fr in frame_records if fr["n_temp"] > 0]
    non_temp_tp = sum(base_frame_cache[fr["image_id"]][0] for fr in frame_records if fr["n_temp"] == 0)
    non_temp_fp = sum(base_frame_cache[fr["image_id"]][1] for fr in frame_records if fr["n_temp"] == 0)
    non_temp_fn = sum(base_frame_cache[fr["image_id"]][2] for fr in frame_records if fr["n_temp"] == 0)

    for tau in threshold_list:
        tot_tp = non_temp_tp
        tot_fp = non_temp_fp
        tot_fn = non_temp_fn
        admitted = 0
        retained_pos = 0

        for fr in temp_frames:
            n_t = fr["n_temp"]
            iid = fr["image_id"]

            frame_has_admitted = False
            for q_t in range(n_t):
                key = (iid, q_t)
                if key in active_key_set:
                    assert key in score_map, f"[FATAL] Missing OOF score for active query {key}"
                    if score_map[key] >= tau:
                        frame_has_admitted = True
                        break

            if not frame_has_admitted:
                tpd, fpd, fnd = base_frame_cache[iid]
                tot_tp += tpd
                tot_fp += fpd
                tot_fn += fnd
                continue

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
    print("MARSEN-C FORWARD-ADDITION AUDIT: INCREMENTAL EVIDENCE VERIFICATION OVER BASE B (P-1846)")
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

    branch_bundle = torch.load(CACHED_BRANCHES_PTH, map_location="cpu")
    branches = branch_bundle["branches"]
    print(f"[+] Loaded Authentic Precomputed Evidence Branches for {n_active:,} Active Queries.")

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
    # P-1840: COMPACT REPRODUCTION GUARD (SEED 42)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("P-1840: COMPACT REPRODUCTION GUARD (VALIDATING SEED 42 AGAINST DIAGNOSTIC)")
    print("=" * 115)

    branch_order = ["global", "hr", "memory", "horizon", "scene", "motion", "reflection", "metadata"]
    X_full_848 = np.concatenate([branches[k] for k in branch_order], axis=-1)
    assert X_full_848.shape == (n_active, 848)

    oof_42 = train_lovo_mlp(X_full_848, y_act, v_act, design_vids, hidden_layer_sizes=(32,), seed=42)
    ap_42 = float(average_precision_score(y_act, oof_42))

    score_map_42 = { (q["image_id"], q["query_id"]): float(oof_42[i]) for i, q in enumerate(active_queries) }
    _, _, rec_95, _ = run_detector_sweep_fast(
        score_map_42, frame_records, active_key_set, positive_key_set, label_to_cat_id, [0.95], base_frame_cache
    )

    print(f"[*] Diagnostic Seed 42 AP:        {ap_42:.4f} (Expected: {DIAG_COMPACT_AP:.4f})")
    print(f"[*] Diagnostic Seed 42 F1 @ 0.95: {rec_95['f1']:.6f} (Expected: {DIAG_COMPACT_F1_95:.6f})")
    print(f"[*] Diagnostic Seed 42 Counts:    TP={rec_95['tp']}, FP={rec_95['fp']}, FN={rec_95['fn']}")

    assert abs(ap_42 - DIAG_COMPACT_AP) < 1e-3, f"[FATAL] AP reproduction mismatch: {ap_42} != {DIAG_COMPACT_AP}"
    assert abs(rec_95["f1"] - DIAG_COMPACT_F1_95) < 1e-4, f"[FATAL] F1@0.95 mismatch: {rec_95['f1']} != {DIAG_COMPACT_F1_95}"
    assert rec_95["tp"] == 210 and rec_95["fp"] == 329 and rec_95["fn"] == 203, "[FATAL] Diagnostic count mismatch!"
    print("[+] P-1840: Seed 42 Diagnostic Benchmark Reconstructed with 100% Exact Parity.")

    # --------------------------------------------------------------------------
    # P-1846: FORWARD-ADDITION CONFIGURATION DEFINITIONS (~27.2K PARAMS EACH)
    # --------------------------------------------------------------------------
    feat_hr = branches["hr"]
    feat_meta = branches["metadata"][:, :15]
    X_base = np.concatenate([feat_hr, feat_meta], axis=-1)  # 143-D

    forward_variants = {
        "Base B (HR + Meta)": {
            "X": X_base, "H": 188, "params": 27261, "is_base": True
        },
        "B + Motion": {
            "X": np.concatenate([X_base, branches["motion"]], axis=-1), "H": 154, "params": 27259, "is_base": False
        },
        "B + Scene": {
            "X": np.concatenate([X_base, branches["scene"]], axis=-1), "H": 154, "params": 27259, "is_base": False
        },
        "B + Horizon": {
            "X": np.concatenate([X_base, branches["horizon"]], axis=-1), "H": 169, "params": 27210, "is_base": False
        },
        "B + Reflection": {
            "X": np.concatenate([X_base, branches["reflection"]], axis=-1), "H": 154, "params": 27259, "is_base": False
        },
        "B + Global Context": {
            "X": np.concatenate([X_base, branches["global"]], axis=-1), "H": 68, "params": 27269, "is_base": False
        },
        "B + Memory (Ref Negative)": {
            "X": np.concatenate([X_base, branches["memory"]], axis=-1), "H": 59, "params": 27436, "is_base": False
        },
    }

    print("\n" + "=" * 115)
    print("P-1846: EVALUATING FORWARD ADDITIONS ACROSS SEEDS 41, 42, 43 (GLOBALLY EXACT OOF SWEEPS)")
    print("=" * 115)

    experiment_results = defaultdict(dict)

    for var_name, cfg in forward_variants.items():
        X_v = cfg["X"]
        H_v = cfg["H"]
        dim_v = X_v.shape[-1]
        p_count = cfg["params"]

        print(f"\n[*] Evaluating Variant: {var_name:<28} (Dim={dim_v:<3d}, Hidden={H_v:<3d}, Params={p_count:,})")
        print(f"    {'Seed':<6} | {'Pooled AP':<12} | {'ROC-AUC':<10} | {'Exact tau*':<14} | {'Best F1*':<12} | {'Delta F1 vs M0'}")
        print("    " + "-" * 75)

        for s in SEEDS:
            oof_v_s = train_lovo_mlp(X_v, y_act, v_act, design_vids, hidden_layer_sizes=(H_v,), seed=s)
            ap_v_s = float(average_precision_score(y_act, oof_v_s))
            auc_v_s = float(roc_auc_score(y_act, oof_v_s))

            exact_taus = get_globally_exact_thresholds(oof_v_s)
            score_map_v_s = { (q["image_id"], q["query_id"]): float(oof_v_s[i]) for i, q in enumerate(active_queries) }
            best_tau_v_s, best_f1_v_s, best_rec_v_s, _ = run_detector_sweep_fast(
                score_map_v_s, frame_records, active_key_set, positive_key_set, label_to_cat_id, exact_taus, base_frame_cache
            )

            experiment_results[var_name][s] = {
                "ap": ap_v_s, "auc": auc_v_s, "tau_star": best_tau_v_s, "f1_star": best_f1_v_s,
                "delta_f1_m0": best_f1_v_s - BASE_M0_F1, "best_rec": best_rec_v_s
            }
            print(f"    Seed {s:<2d} | {ap_v_s:10.4f}   | {auc_v_s:8.4f}   | {best_tau_v_s:12.6f}   | {best_f1_v_s:10.6f}   | {best_f1_v_s - BASE_M0_F1:+12.4f}")

    # --------------------------------------------------------------------------
    # PAIRED COMPARISON MATRIX (INCREMENTAL LIFT OVER BASE B)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("PAIRED INCREMENTAL LIFT OVER BASE B (AP & F1 DELTAS PER SEED)")
    print("=" * 115)
    print(f"{'Variant Evaluated':<28} | {'Dim':<5} | {'Mean AP':<10} | {'Mean dAP vs B':<15} | {'Seed 41 dAP':<12} | {'Seed 42 dAP':<12} | {'Seed 43 dAP':<12} | {'Mean dF1 vs B'}")
    print("-" * 120)

    base_aps = [experiment_results["Base B (HR + Meta)"][s]["ap"] for s in SEEDS]
    base_f1s = [experiment_results["Base B (HR + Meta)"][s]["f1_star"] for s in SEEDS]

    synthesis_table = {}

    for var_name, cfg in forward_variants.items():
        var_aps = [experiment_results[var_name][s]["ap"] for s in SEEDS]
        var_f1s = [experiment_results[var_name][s]["f1_star"] for s in SEEDS]

        d_aps = [var_aps[i] - base_aps[i] for i in range(len(SEEDS))]
        d_f1s = [var_f1s[i] - base_f1s[i] for i in range(len(SEEDS))]

        mean_ap = float(np.mean(var_aps))
        mean_d_ap = float(np.mean(d_aps))
        mean_d_f1 = float(np.mean(d_f1s))

        synthesis_table[var_name] = {
            "dim": cfg["X"].shape[-1], "params": cfg["params"], "mean_ap": mean_ap, "std_ap": float(np.std(var_aps)),
            "mean_delta_ap": mean_d_ap, "std_delta_ap": float(np.std(d_aps)),
            "seed_d_aps": d_aps, "mean_delta_f1": mean_d_f1, "std_delta_f1": float(np.std(d_f1s)), "seed_d_f1s": d_f1s
        }

        ref_tag = "REF (0.0000)" if cfg["is_base"] else f"{mean_d_ap:+12.4f}"
        s41_str = "REF" if cfg["is_base"] else f"{d_aps[0]:+10.4f}"
        s42_str = "REF" if cfg["is_base"] else f"{d_aps[1]:+10.4f}"
        s43_str = "REF" if cfg["is_base"] else f"{d_aps[2]:+10.4f}"
        f1_str = "REF" if cfg["is_base"] else f"{mean_d_f1:+12.6f}"

        print(f"{var_name:<28} | {cfg['X'].shape[-1]:<5d} | {mean_ap:8.4f}   | {ref_tag:<15} | {s41_str:<12} | {s42_str:<12} | {s43_str:<12} | {f1_str}")

    # --------------------------------------------------------------------------
    # SYNTHESIS & MANIFEST SERIALIZATION
    # --------------------------------------------------------------------------
    results_payload = {
        "_meta": {
            "artifact_name": "marsen_forward_addition_results_v1",
            "driver_sha256": driver_sha256,
            "baseline_m0_f1": float(BASE_M0_F1),
            "reference_seed": REFERENCE_SEED,
            "target_parameter_budget": 27200
        },
        "individual_seed_runs": {
            v: {str(s): experiment_results[v][s] for s in SEEDS} for v in forward_variants.keys()
        },
        "incremental_synthesis": synthesis_table
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    res_hash = compute_sha256(RESULTS_JSON)
    print("\n" + "=" * 115)
    print(f"[+] Forward-Addition Results Frozen: {RESULTS_JSON.resolve()}")
    print(f"    Manifest SHA-256:                {res_hash}")
    print("=" * 115)


if __name__ == "__main__":
    main()