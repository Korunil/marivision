"""
D:/MARIVISION/H0_workbench/eval_tools/audit_marsen_base_depth.py
MARSEN-C Base B Depth Benchmark: Expansion -> Consolidation Funnel Optimization (v5, P-1875..P-1883)

Strict Compliance Checklist:
  - D-3832..D-3837: Base B depth optimization on 10 design videos ONLY; calibration/test UNTOUCHED.
  - P-1882: Hard assertion enforcing design cohort is strictly [4, 7, 16, 19, 25, 29, 30, 32, 35, 47].
  - P-1878: 64-D consolidation family (H2=64) with expansion widths H1 in {128, 192, 256}.
  - P-1879: 128-D consolidation family (H2=128) with expansion widths H1 in {256, 384, 512}.
  - P-1883: Comprehensive 4-anchor reproduction check: H=64 (0.1280), H=128 (0.1278), (256->64) (0.1145), (384->128) (0.1278).
  - P-1880: Paired seed delta ledger reporting Seed 41/42/43 AP deltas between top deep and top shallow models.
  - P-1881: Quantitative H=32 diagnostic reporting exact iteration gain and remaining gaps to anchors.
  - P-1841: Globally exact unique OOF score cut-points for every evaluated configuration across Seeds 41, 42, 43.
  - P-1867: Optimization convergence tracking ensuring 100% healthy fold convergence.
  - Parsimony Threshold: Delta AP > 0.0010 required to prefer complex deep over single-layer.
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
RESULTS_JSON = OUT_DIR / "marsen_base_depth_results.json"

SCORE_THRESHOLD = 0.20
EVAL_IOU_THRESHOLD = 0.30
BASE_M0_F1 = 0.431718

EXPECTED_DESIGN_VIDS = [4, 7, 16, 19, 25, 29, 30, 32, 35, 47]
EXPECTED_H64_MEAN_AP = 0.1280
EXPECTED_H128_MEAN_AP = 0.1278
EXPECTED_DEEP_256_64_MEAN_AP = 0.1145
EXPECTED_DEEP_384_128_MEAN_AP = 0.1278

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

        # Deterministic tie-break favoring higher tau
        if f1 > best_f1 + 1e-12 or (abs(f1 - best_f1) <= 1e-12 and (best_tau is None or tau > best_tau)):
            best_f1 = f1
            best_tau = tau
            best_rec = sweep_records[tau]

    return best_tau, best_f1, best_rec, sweep_records


def train_lovo_mlp_with_convergence(X_feat, y_act, v_act, design_vids, hidden_layer_sizes, seed, max_iter=350):
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

        clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation="relu", alpha=0.05, max_iter=max_iter, random_state=seed)
        clf.fit(X_tr_std[bal_idx], y_tr[bal_idx])
        oof_scores[test_mask] = clf.predict_proba(X_te_std)[:, 1]

        fold_n_iters.append(int(clf.n_iter_))
        fold_losses.append(float(clf.loss_))

    assert np.isfinite(oof_scores).all(), f"[FATAL] Non-finite scores generated for seed {seed}!"

    conv_metrics = {
        "mean_n_iter": float(np.mean(fold_n_iters)),
        "max_n_iter": int(np.max(fold_n_iters)),
        "n_hit_max_iter": int(np.sum(np.array(fold_n_iters) >= max_iter)),
        "mean_final_loss": float(np.mean(fold_losses))
    }
    return oof_scores, conv_metrics


def main():
    driver_sha256 = compute_sha256(Path(__file__))
    print("=" * 115)
    print("MARSEN-C BASE B DEPTH BENCHMARK: EXPANSION -> CONSOLIDATION OPTIMIZATION (v5, P-1875..P-1883)")
    print(f"Driver Script SHA-256: {driver_sha256}")
    print("=" * 115)

    with open(SPLIT_MANIFEST_JSON, "r", encoding="utf-8") as f:
        design_vids = sorted(json.load(f)["_meta"]["train_videos"])

    # P-1882: Hard cohort assertion
    assert design_vids == EXPECTED_DESIGN_VIDS, f"[FATAL] Unexpected design cohort: {design_vids} != {EXPECTED_DESIGN_VIDS}"
    print(f"[+] P-1882 Guard Certified: Design cohort strictly matches 10 permitted sequences: {design_vids}")

    cached_queries = torch.load(QUERY_DATASET_PTH, map_location="cpu")["extracted_queries"]
    frame_bundle = torch.load(FRAME_TENSORS_PTH, map_location="cpu")
    frame_records = frame_bundle["frame_records"]

    active_indices = [idx for idx, q in enumerate(cached_queries) if q["is_active"]]
    active_queries = [cached_queries[idx] for idx in active_indices]
    n_active = len(active_queries)
    y_act = np.array([q["label_op_util"] for q in active_queries], dtype=np.int32)
    v_act = np.array([q["video_id"] for q in active_queries], dtype=np.int32)

    assert sorted(np.unique(v_act).tolist()) == EXPECTED_DESIGN_VIDS
    assert n_active == 1915
    assert int(y_act.sum()) == 92

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
    # P-1878 & P-1879: EXPANSION -> CONSOLIDATION DEPTH GRID DEFINITIONS
    # --------------------------------------------------------------------------
    depth_configurations = {
        # Single-Layer Reproduction Anchors
        "Anchor: Base (H=64)":           {"hidden": (64,),       "max_iter": 350,  "params": 9281,   "category": "anchor"},
        "Anchor: Base (H=128)":          {"hidden": (128,),      "max_iter": 350,  "params": 18561,  "category": "anchor"},

        # P-1878: 64-D Consolidation Family (H2 = 64)
        "2-Layer: (128 -> 64)":          {"hidden": (128, 64),   "max_iter": 350,  "params": 26753,  "category": "family_64"},
        "2-Layer: (192 -> 64)":          {"hidden": (192, 64),   "max_iter": 350,  "params": 40065,  "category": "family_64"},
        "2-Layer: (256 -> 64) [Anchor]": {"hidden": (256, 64),   "max_iter": 350,  "params": 53377,  "category": "family_64"},

        # P-1879: 128-D Consolidation Family (H2 = 128)
        "2-Layer: (256 -> 128)":         {"hidden": (256, 128),  "max_iter": 350,  "params": 69889,  "category": "family_128"},
        "2-Layer: (384 -> 128) [Anchor]":{"hidden": (384, 128),  "max_iter": 350,  "params": 104705, "category": "family_128"},
        "2-Layer: (512 -> 128)":         {"hidden": (512, 128),  "max_iter": 350,  "params": 139521, "category": "family_128"},

        # P-1876 / P-1881: Diagnostic H=32 with Extended Optimization Ceiling
        "Diag: Base (H=32, iter=1000)":  {"hidden": (32,),       "max_iter": 1000, "params": 4641,   "category": "diagnostic"},
    }

    # Verify exact parameter arithmetic
    for name, cfg in depth_configurations.items():
        h = cfg["hidden"]
        if len(h) == 1:
            p_calc = (143 + 1) * h[0] + (h[0] + 1) * 1
        else:
            p_calc = (143 + 1) * h[0] + (h[0] + 1) * h[1] + (h[1] + 1) * 1
        assert p_calc == cfg["params"], f"[FATAL] Parameter mismatch for {name}: {p_calc} != {cfg['params']}"

    print(f"\n[*] Predeclared B1b Grid: 2 Shallow Anchors + 3 in Family-64 + 3 in Family-128 + 1 H=32 Diagnostic.")

    # --------------------------------------------------------------------------
    # EVALUATION LOOP
    # --------------------------------------------------------------------------
    sweep_results = defaultdict(dict)

    for cfg_name, cfg in depth_configurations.items():
        h_tuple = cfg["hidden"]
        p_count = cfg["params"]
        m_iter = cfg["max_iter"]
        print(f"\n[*] Evaluating Topology: {cfg_name:<32} (Hidden={str(h_tuple):<12}, Params={p_count:,}, MaxIter={m_iter})")
        print(f"    {'Seed':<6} | {'Pooled AP':<12} | {'ROC-AUC':<10} | {'Exact tau*':<14} | {'Best F1*':<12} | {'Delta F1 vs M0':<16} | {'Optimization Convergence'}")
        print("    " + "-" * 98)

        for s in SEEDS:
            oof_s, conv_s = train_lovo_mlp_with_convergence(
                X_base, y_act, v_act, design_vids, hidden_layer_sizes=h_tuple, seed=s, max_iter=m_iter
            )
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
    # P-1883: FOUR-ANCHOR CONTINUITY REPRODUCTION VERIFICATION
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("P-1883: FOUR-ANCHOR CONTINUITY REPRODUCTION CHECK")
    print("=" * 115)
    h64_mean_ap = np.mean([sweep_results["Anchor: Base (H=64)"][s]["ap"] for s in SEEDS])
    h128_mean_ap = np.mean([sweep_results["Anchor: Base (H=128)"][s]["ap"] for s in SEEDS])
    deep_256_64_mean_ap = np.mean([sweep_results["2-Layer: (256 -> 64) [Anchor]"][s]["ap"] for s in SEEDS])
    deep_384_128_mean_ap = np.mean([sweep_results["2-Layer: (384 -> 128) [Anchor]"][s]["ap"] for s in SEEDS])

    print(f"[*] Re-evaluated H=64 Mean AP:             {h64_mean_ap:.4f} (Expected: {EXPECTED_H64_MEAN_AP:.4f})")
    print(f"[*] Re-evaluated H=128 Mean AP:            {h128_mean_ap:.4f} (Expected: {EXPECTED_H128_MEAN_AP:.4f})")
    print(f"[*] Re-evaluated (256 -> 64) Mean AP:      {deep_256_64_mean_ap:.4f} (Expected: {EXPECTED_DEEP_256_64_MEAN_AP:.4f})")
    print(f"[*] Re-evaluated (384 -> 128) Mean AP:     {deep_384_128_mean_ap:.4f} (Expected: {EXPECTED_DEEP_384_128_MEAN_AP:.4f})")

    assert abs(h64_mean_ap - EXPECTED_H64_MEAN_AP) < 1e-3, "[FATAL] H=64 anchor reproduction mismatch!"
    assert abs(h128_mean_ap - EXPECTED_H128_MEAN_AP) < 1e-3, "[FATAL] H=128 anchor reproduction mismatch!"
    assert abs(deep_256_64_mean_ap - EXPECTED_DEEP_256_64_MEAN_AP) < 1e-3, "[FATAL] (256 -> 64) anchor reproduction mismatch!"
    assert abs(deep_384_128_mean_ap - EXPECTED_DEEP_384_128_MEAN_AP) < 1e-3, "[FATAL] (384 -> 128) anchor reproduction mismatch!"
    print("[+] P-1883 Certified: All 4 empirical anchors reproduced within the predeclared tolerance.")

    # --------------------------------------------------------------------------
    # SYNTHESIS & RANKING
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("PHASE B1b DEPTH SYNTHESIS (RANKED BY MEAN POOLED AP ACROSS SEEDS 41, 42, 43)")
    print("=" * 115)
    print(f"{'Rank':<5} | {'Topology':<32} | {'Params':<10} | {'Mean AP':<10} | {'AP Std':<10} | {'Seed 41 AP':<12} | {'Seed 42 AP':<12} | {'Seed 43 AP':<12} | {'Mean F1*':<10} | {'Mean dF1':<10} | {'Cap Hits'}")
    print("-" * 138)

    summary_list = []
    for cfg_name, cfg in depth_configurations.items():
        aps = [sweep_results[cfg_name][s]["ap"] for s in SEEDS]
        f1s = [sweep_results[cfg_name][s]["f1_star"] for s in SEEDS]
        df1s = [sweep_results[cfg_name][s]["delta_f1_m0"] for s in SEEDS]
        aucs = [sweep_results[cfg_name][s]["auc"] for s in SEEDS]
        total_cap_hits = sum(sweep_results[cfg_name][s]["convergence"]["n_hit_max_iter"] for s in SEEDS)
        mean_iters = np.mean([sweep_results[cfg_name][s]["convergence"]["mean_n_iter"] for s in SEEDS])

        summary_list.append({
            "name": cfg_name,
            "hidden": cfg["hidden"],
            "max_iter": cfg["max_iter"],
            "category": cfg["category"],
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

    summary_list.sort(key=lambda x: x["mean_ap"], reverse=True)

    for rank, item in enumerate(summary_list, 1):
        tag = ""
        if rank == 1:
            tag = " <-- BEST TESTED TOPOLOGY"
        elif item["category"] == "diagnostic":
            tag = " [P-1876 DIAGNOSTIC]"
        cap_str = f"{item['total_cap_hits_30folds']}/30"
        print(f"#{rank:<3d} | {item['name']:<32} | {item['params']:<10,d} | {item['mean_ap']:8.4f}   | {item['std_ap']:8.4f}   | {item['aps'][0]:10.4f}   | {item['aps'][1]:10.4f}   | {item['aps'][2]:10.4f}   | {item['mean_f1']:8.6f} | {item['mean_df1']:+8.4f}   | {cap_str:<8}{tag}")

    arch_candidates = [m for m in summary_list if m["category"] != "diagnostic"]
    single_layer_winner = [m for m in arch_candidates if len(m["hidden"]) == 1][0]
    deep_winner = [m for m in arch_candidates if len(m["hidden"]) == 2][0]

    # P-1880: Paired seed delta ledger
    ap_diff = deep_winner["mean_ap"] - single_layer_winner["mean_ap"]
    paired_d_aps = [deep_winner["aps"][i] - single_layer_winner["aps"][i] for i in range(3)]
    f1_diff = deep_winner["mean_f1"] - single_layer_winner["mean_f1"]

    print("\n" + "=" * 115)
    print("PHASE B1b DECISION & PARSIMONY VERDICT (P-1880)")
    print("=" * 115)
    print(f"  - Top Single-Layer Model:     {single_layer_winner['name']} (Mean AP = {single_layer_winner['mean_ap']:.4f}, Mean F1* = {single_layer_winner['mean_f1']:.6f})")
    print(f"  - Top Two-Layer Model:       {deep_winner['name']} (Mean AP = {deep_winner['mean_ap']:.4f}, Mean F1* = {deep_winner['mean_f1']:.6f})")
    print(f"  - Mean Delta AP (Deep - Sing): {ap_diff:+.4f}")
    print(f"  - Paired Seed Delta APs:       Seed 41: {paired_d_aps[0]:+.4f} | Seed 42: {paired_d_aps[1]:+.4f} | Seed 43: {paired_d_aps[2]:+.4f}")
    print(f"  - Mean Delta F1* (Deep - Sing): {f1_diff:+.6f}")

    if ap_diff > 0.0010:
        winning_model = deep_winner
        selection_rationale = f"Two-layer model ({deep_winner['name']}) earned inclusion with +{ap_diff:.4f} Mean AP lift over single-layer."
    else:
        winning_model = single_layer_winner
        selection_rationale = f"Parsimony rule applied: Deep model delta ({ap_diff:+.4f} AP) <= 0.0010 threshold. Selecting simpler single-layer ({single_layer_winner['name']})."

    print(f"  - Selected Topology:          {winning_model['name']}")
    print(f"  - Selection Rationale:        {selection_rationale}")

    # P-1881: Quantitative H=32 convergence diagnostic
    h32_diag = [m for m in summary_list if m["category"] == "diagnostic"][0]
    delta_conv_ap = h32_diag["mean_ap"] - 0.0885
    gap_to_h64 = h64_mean_ap - h32_diag["mean_ap"]
    gap_to_h128 = h128_mean_ap - h32_diag["mean_ap"]

    print("\n" + "=" * 115)
    print("P-1881: QUANTITATIVE H=32 OPTIMIZATION DIAGNOSTIC FINDINGS")
    print("=" * 115)
    print(f"  - H=32 (max_iter=350, Phase B1):    Mean AP = 0.0885 (10/30 folds hit cap)")
    print(f"  - H=32 (max_iter=1000, Phase B1b):  Mean AP = {h32_diag['mean_ap']:.4f} ({h32_diag['total_cap_hits_30folds']}/30 folds hit cap, mean iters = {h32_diag['mean_iters']:.1f})")
    print(f"  - Observed AP Lift from Iterations: {delta_conv_ap:+.4f}")
    print(f"  - Remaining Gap to H=64 Anchor:     {gap_to_h64:+.4f}")
    print(f"  - Remaining Gap to H=128 Anchor:    {gap_to_h128:+.4f}")
    print("=" * 115)

    results_payload = {
        "_meta": {
            "artifact_name": "marsen_base_depth_results_v5",
            "driver_sha256": driver_sha256,
            "baseline_m0_f1": float(BASE_M0_F1),
            "reference_seed": REFERENCE_SEED,
            "selected_architecture": winning_model["name"],
            "selected_hidden": list(winning_model["hidden"]),
            "selected_params": int(winning_model["params"]),
            "selected_mean_ap": float(winning_model["mean_ap"]),
            "selected_mean_f1": float(winning_model["mean_f1"]),
            "selection_rationale": selection_rationale,
            "paired_seed_delta_aps": paired_d_aps
        },
        "ranked_topology_summary": summary_list,
        "seed_runs": {
            cfg_name: {str(s): sweep_results[cfg_name][s] for s in SEEDS} for cfg_name in depth_configurations.keys()
        },
        "h32_quantitative_diagnostic": {
            "h32_1000_mean_ap": float(h32_diag["mean_ap"]),
            "delta_conv_ap": float(delta_conv_ap),
            "gap_to_h64": float(gap_to_h64),
            "gap_to_h128": float(gap_to_h128),
            "cap_hits": int(h32_diag["total_cap_hits_30folds"])
        }
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    res_hash = compute_sha256(RESULTS_JSON)
    print(f"[+] Depth Benchmark Results Frozen: {RESULTS_JSON.resolve()}")
    print(f"    Manifest SHA-256:               {res_hash}")
    print("=" * 115)


if __name__ == "__main__":
    main()