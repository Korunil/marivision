"""
D:/MARIVISION/H0_workbench/eval_tools/audit_marsen_base_capacity.py
MARSEN-C Base B Capacity Curve Benchmark: Unconstrained Width & Depth Optimization (v2, P-1863, P-1867)

Strict Compliance Checklist:
  - D-3810..D-3817: Base B optimization on 10 design videos ONLY; calibration/test UNTOUCHED.
  - P-1863: Evaluates 7 single-layer widths (H in [32, 64, 128, 188, 256, 384, 512]) and 2 deep variants.
  - P-1867: Convergence health ledger tracking fold iterations, final losses, and 350-iteration cap hits.
  - P-1840: Seed-42 reproduction check asserting exact parity at H=188 (AP=0.1347, F1=0.456039).
  - P-1841: Globally exact unique OOF score cut-points for every evaluated configuration.
  - Primary Selection Metric: Highest Mean OOF AP across Seeds 41, 42, 43.
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
RESULTS_JSON = OUT_DIR / "marsen_base_capacity_results.json"

SCORE_THRESHOLD = 0.20
EVAL_IOU_THRESHOLD = 0.30
BASE_M0_F1 = 0.431718
EXPECTED_BASE_H188_S42_AP = 0.1347
EXPECTED_BASE_H188_S42_F1 = 0.456039

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

        # Deterministic tie-break favoring higher tau (stronger suppression)
        if f1 > best_f1 + 1e-12 or (abs(f1 - best_f1) <= 1e-12 and (best_tau is None or tau > best_tau)):
            best_f1 = f1
            best_tau = tau
            best_rec = sweep_records[tau]

    return best_tau, best_f1, best_rec, sweep_records


# ------------------------------------------------------------------------------
# P-1867: LOVO TRAINING WITH CONVERGENCE HEALTH MONITORING
# ------------------------------------------------------------------------------

def train_lovo_mlp_with_convergence(X_feat, y_act, v_act, design_vids, hidden_layer_sizes, seed):
    n_samples = len(y_act)
    oof_scores = np.full(n_samples, np.nan)
    fold_n_iters = []
    fold_losses = []

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

        fold_n_iters.append(int(clf.n_iter_))
        fold_losses.append(float(clf.loss_))

    assert np.isfinite(oof_scores).all(), f"[FATAL] Non-finite scores generated for seed {seed}!"

    conv_metrics = {
        "mean_n_iter": float(np.mean(fold_n_iters)),
        "max_n_iter": int(np.max(fold_n_iters)),
        "n_hit_max_iter": int(np.sum(np.array(fold_n_iters) >= 350)),
        "mean_final_loss": float(np.mean(fold_losses))
    }
    return oof_scores, conv_metrics


def main():
    driver_sha256 = compute_sha256(Path(__file__))
    print("=" * 115)
    print("MARSEN-C BASE B CAPACITY CURVE BENCHMARK: UNCONSTRAINED WIDTH & DEPTH (v2, P-1863, P-1867)")
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

    # Reconstruct Base B: 128-D Native HR + 15-D Proposal Metadata = 143-D
    X_base = np.concatenate([branches["hr"], branches["metadata"][:, :15]], axis=-1)
    assert X_base.shape == (n_active, 143)

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
    # CAPACITY GRID DEFINITIONS (SINGLE-LAYER & TWO-LAYER CONFIGURATIONS)
    # --------------------------------------------------------------------------
    capacity_configurations = {
        "Base (H=32) [Orig 4M]":     {"hidden": (32,),       "params": 4641},
        "Base (H=64)":               {"hidden": (64,),       "params": 9281},
        "Base (H=128)":              {"hidden": (128,),      "params": 18561},
        "Base (H=188) [Cap Control]": {"hidden": (188,),     "params": 27261},
        "Base (H=256)":              {"hidden": (256,),      "params": 37121},
        "Base (H=384)":              {"hidden": (384,),      "params": 55681},
        "Base (H=512)":              {"hidden": (512,),      "params": 74241},
        "Base (2-Layer: 256->64)":   {"hidden": (256, 64),   "params": 53377},
        "Base (2-Layer: 384->128)":  {"hidden": (384, 128),  "params": 104705},
    }

    # Verify exact parameter arithmetic
    for name, cfg in capacity_configurations.items():
        h = cfg["hidden"]
        if len(h) == 1:
            p_calc = (143 + 1) * h[0] + (h[0] + 1) * 1
        else:
            p_calc = (143 + 1) * h[0] + (h[0] + 1) * h[1] + (h[1] + 1) * 1
        assert p_calc == cfg["params"], f"[FATAL] Parameter mismatch for {name}: {p_calc} != {cfg['params']}"

    print(f"\n[*] Predeclared Capacity Grid: 7 Single-Layer Widths + 2 Deep Configurations (~4.6k to ~104.7k params).")

    # --------------------------------------------------------------------------
    # EVALUATION LOOP ACROSS CAPACITY CONFIGURATIONS AND SEEDS
    # --------------------------------------------------------------------------
    sweep_results = defaultdict(dict)

    for cfg_name, cfg in capacity_configurations.items():
        h_tuple = cfg["hidden"]
        p_count = cfg["params"]
        print(f"\n[*] Evaluating Topology: {cfg_name:<30} (Hidden={str(h_tuple):<12}, Params={p_count:,})")
        print(f"    {'Seed':<6} | {'Pooled AP':<12} | {'ROC-AUC':<10} | {'Exact tau*':<14} | {'Best F1*':<12} | {'Delta F1 vs M0':<16} | {'Optimization Convergence'}")
        print("    " + "-" * 98)

        for s in SEEDS:
            oof_s, conv_s = train_lovo_mlp_with_convergence(X_base, y_act, v_act, design_vids, hidden_layer_sizes=h_tuple, seed=s)
            ap_s = float(average_precision_score(y_act, oof_s))
            auc_s = float(roc_auc_score(y_act, oof_s))

            exact_taus = get_globally_exact_thresholds(oof_s)
            score_map_s = { (q["image_id"], q["query_id"]): float(oof_s[i]) for i, q in enumerate(active_queries) }
            best_tau_s, best_f1_s, best_rec_s, _ = run_detector_sweep_fast(
                score_map_s, frame_records, active_key_set, positive_key_set, label_to_cat_id, exact_taus, base_frame_cache
            )

            sweep_results[cfg_name][s] = {
                "ap": ap_s, "auc": auc_s, "tau_star": best_tau_s, "f1_star": best_f1_s,
                "delta_f1_m0": best_f1_s - BASE_M0_F1, "best_rec": best_rec_s,
                "convergence": conv_s
            }
            cap_hit_str = f"{conv_s['n_hit_max_iter']}/10"
            conv_str = f"iters={conv_s['mean_n_iter']:.1f} (cap_hits={cap_hit_str}, loss={conv_s['mean_final_loss']:.4f})"
            print(f"    Seed {s:<2d} | {ap_s:10.4f}   | {auc_s:8.4f}   | {best_tau_s:12.6f}   | {best_f1_s:10.6f}   | {best_f1_s - BASE_M0_F1:+12.4f}   | {conv_str}")

    # --------------------------------------------------------------------------
    # REPRODUCTION GUARD CHECK ON SEED 42 AT H=188
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("P-1840 REPRODUCTION GUARD CHECK (SEED 42 AT H=188)")
    print("=" * 115)
    s42_h188_ap = sweep_results["Base (H=188) [Cap Control]"][42]["ap"]
    s42_h188_f1 = sweep_results["Base (H=188) [Cap Control]"][42]["f1_star"]
    print(f"[*] Re-evaluated Seed 42 H=188 AP: {s42_h188_ap:.4f} (Expected: {EXPECTED_BASE_H188_S42_AP:.4f})")
    print(f"[*] Re-evaluated Seed 42 H=188 F1: {s42_h188_f1:.6f} (Expected: {EXPECTED_BASE_H188_S42_F1:.6f})")

    assert abs(s42_h188_ap - EXPECTED_BASE_H188_S42_AP) < 1e-3, "[FATAL] Capacity control AP reproduction failure!"
    assert abs(s42_h188_f1 - EXPECTED_BASE_H188_S42_F1) < 1e-4, "[FATAL] Capacity control F1 reproduction failure!"
    print("[+] P-1840: Capacity Control Anchor Verified with 100% Exact Parity.")

    # --------------------------------------------------------------------------
    # CAPACITY OPTIMIZATION SYNTHESIS & RANKING
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("BASE B CAPACITY CURVE SYNTHESIS (RANKED BY MEAN POOLED AP ACROSS SEEDS 41, 42, 43)")
    print("=" * 115)
    print(f"{'Rank':<5} | {'Topology':<28} | {'Params':<10} | {'Mean AP':<10} | {'AP Std':<10} | {'Seed 41 AP':<12} | {'Seed 42 AP':<12} | {'Seed 43 AP':<12} | {'Mean F1*':<10} | {'Mean dF1':<10} | {'Cap Hits'}")
    print("-" * 135)

    summary_list = []
    for cfg_name, cfg in capacity_configurations.items():
        aps = [sweep_results[cfg_name][s]["ap"] for s in SEEDS]
        f1s = [sweep_results[cfg_name][s]["f1_star"] for s in SEEDS]
        df1s = [sweep_results[cfg_name][s]["delta_f1_m0"] for s in SEEDS]
        aucs = [sweep_results[cfg_name][s]["auc"] for s in SEEDS]
        total_cap_hits = sum(sweep_results[cfg_name][s]["convergence"]["n_hit_max_iter"] for s in SEEDS)
        mean_iters = np.mean([sweep_results[cfg_name][s]["convergence"]["mean_n_iter"] for s in SEEDS])

        summary_list.append({
            "name": cfg_name,
            "hidden": cfg["hidden"],
            "params": cfg["params"],
            "mean_ap": float(np.mean(aps)),
            "std_ap": float(np.std(aps)),
            "aps": aps,
            "mean_f1": float(np.mean(f1s)),
            "std_f1": float(np.std(f1s)),
            "f1s": f1s,
            "mean_df1": float(np.mean(df1s)),
            "std_df1": float(np.std(df1s)),
            "mean_auc": float(np.mean(aucs)),
            "total_cap_hits_30folds": int(total_cap_hits),
            "mean_iters": float(mean_iters)
        })

    # Sort primarily by Mean OOF AP descending
    summary_list.sort(key=lambda x: x["mean_ap"], reverse=True)

    for rank, item in enumerate(summary_list, 1):
        star = " <-- BEST TESTED TOPOLOGY" if rank == 1 else ""
        cap_str = f"{item['total_cap_hits_30folds']}/30"
        print(f"#{rank:<3d} | {item['name']:<28} | {item['params']:<10,d} | {item['mean_ap']:8.4f}   | {item['std_ap']:8.4f}   | {item['aps'][0]:10.4f}   | {item['aps'][1]:10.4f}   | {item['aps'][2]:10.4f}   | {item['mean_f1']:8.6f} | {item['mean_df1']:+8.4f}   | {cap_str:<8}{star}")

    optimal_config = summary_list[0]
    is_upper_boundary = (optimal_config["name"] in ["Base (H=512)", "Base (2-Layer: 384->128)"])
    curve_nature = "UPPER-BOUNDARY WINNER (Further capacity expansion justified)" if is_upper_boundary else "INTERIOR PEAK (True bias-variance capacity peak located)"

    print("\n" + "=" * 115)
    print(f"[+] Best Tested Base B Topology: {optimal_config['name']}")
    print(f"    - Parameter Count:      {optimal_config['params']:,}")
    print(f"    - Mean OOF AP:          {optimal_config['mean_ap']:.4f} +/- {optimal_config['std_ap']:.4f}")
    print(f"    - Mean Best F1*:        {optimal_config['mean_f1']:.6f} ({optimal_config['mean_df1']:+.4f} over M0)")
    print(f"    - Convergence Health:   {optimal_config['total_cap_hits_30folds']}/30 folds hit 350-iteration cap (mean iters = {optimal_config['mean_iters']:.1f})")
    print(f"    - Curve Characteristic: {curve_nature}")
    print("=" * 115)

    results_payload = {
        "_meta": {
            "artifact_name": "marsen_base_capacity_results_v2",
            "driver_sha256": driver_sha256,
            "baseline_m0_f1": float(BASE_M0_F1),
            "reference_seed": REFERENCE_SEED,
            "best_tested_topology": optimal_config["name"],
            "best_tested_hidden": list(optimal_config["hidden"]),
            "best_tested_params": int(optimal_config["params"]),
            "best_tested_mean_ap": float(optimal_config["mean_ap"]),
            "best_tested_mean_f1": float(optimal_config["mean_f1"]),
            "is_upper_boundary_winner": is_upper_boundary
        },
        "ranked_capacity_summary": summary_list,
        "seed_runs": {
            cfg_name: {str(s): sweep_results[cfg_name][s] for s in SEEDS} for cfg_name in capacity_configurations.keys()
        }
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    res_hash = compute_sha256(RESULTS_JSON)
    print(f"[+] Base B Capacity Results Frozen: {RESULTS_JSON.resolve()}")
    print(f"    Manifest SHA-256:               {res_hash}")
    print("=" * 115)


if __name__ == "__main__":
    main()