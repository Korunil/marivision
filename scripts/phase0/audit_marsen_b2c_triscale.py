"""
D:/MARIVISION/H0_workbench/eval_tools/audit_marsen_b2c_triscale.py
MARSEN Phase B2c: Tri-Scale Context Complementarity Resolution (v3, P-1933, P-1934, P-1938, P-1942, P-1943)

Strict Compliance Checklist:
  - D-3906..D-3917: Evaluates zero-parameter tri-scale fusion across 1.0x Base + 3.0x + 5.0x.
  - P-1921: Canonical runtime contract enforced (BLAS=1, ProcessPoolExecutor, memoized sweeps).
  - P-1938: Tie-aware rankdata (scipy.stats.rankdata), dual-fusion AP and operational F1* reported.
  - P-1942: Freezes marsen_b2c_oof_predictions_1915.pth (9 canonical OOF vectors).
  - P-1943: Dual positive correlation (global rank correlation vs. true within-positive Spearman rho)
            plus mean and median positive percentile rank metrics.
  - P-1938 Continuity Check: Asserts exact reproduction of canonical B2b standalones (< 1e-3).
"""

import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import time
import hashlib
from PIL import Image
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier

OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
TRAIN_JSON = Path("D:/MARIVISION/seaclips-train.json")
SPLIT_MANIFEST_JSON = OUT_DIR / "pol2_candidate2m_micro_split.json"
QUERY_DATASET_PTH = OUT_DIR / "pol3v_design_queries_dataset_v7.pth"
FRAME_TENSORS_PTH = OUT_DIR / "pol3v_design_frames_tensors_v7.pth"
CACHED_BRANCHES_PTH = OUT_DIR / "marsen_micro_authentic_branches_1915.pth"
CACHED_B2B_FOV_PTH = OUT_DIR / "marsen_b2b_fov_features_1915.pth"
B2B_RESULTS_JSON = OUT_DIR / "marsen_b2b_fov_extension_results.json"

RESULTS_JSON = OUT_DIR / "marsen_b2c_triscale_results.json"
OOF_VECTORS_PTH = OUT_DIR / "marsen_b2c_oof_predictions_1915.pth"

SCORE_THRESHOLD = 0.20
EVAL_IOU_THRESHOLD = 0.30
BASE_M0_F1 = 0.431718
EXPECTED_DESIGN_VIDS = [4, 7, 16, 19, 25, 29, 30, 32, 35, 47]

SEEDS = [41, 42, 43]
WINNING_HIDDEN = (64,)
TRISCALE_FOVS = [1.0, 3.0, 5.0]


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
    tp, fp = 0, 0

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

    frame_eval_cache = {}

    for tau in threshold_list:
        tot_tp = non_temp_tp
        tot_fp = non_temp_fp
        tot_fn = non_temp_fn
        admitted = 0
        retained_pos = 0

        for fr in temp_frames:
            n_t = fr["n_temp"]
            iid = fr["image_id"]

            admit_flags = []
            for q_t in range(n_t):
                key = (iid, q_t)
                is_adm = (key in active_key_set and score_map.get(key, -1e9) >= tau)
                admit_flags.append(is_adm)
                if is_adm:
                    admitted += 1
                    if key in positive_key_set:
                        retained_pos += 1

            if not any(admit_flags):
                tpd, fpd, fnd = base_frame_cache[iid]
                tot_tp += tpd
                tot_fp += fpd
                tot_fn += fnd
                continue

            cache_key = (iid, tuple(admit_flags))
            if cache_key in frame_eval_cache:
                tpf, fpf, fnf = frame_eval_cache[cache_key]
            else:
                joint_logits = torch.cat([fr["d_logits"], fr["temp_logits"]], dim=0).clone()
                joint_boxes = torch.cat([fr["d_boxes"], fr["temp_boxes"]], dim=0)

                for q_t in range(n_t):
                    if not admit_flags[q_t]:
                        joint_logits[300 + q_t, :] = -1e9

                tpf, fpf, fnf = evaluate_operational_top100(
                    joint_logits, joint_boxes, fr["gt_boxes_px"], fr["gt_cats"], label_to_cat_id, fr["img_w"], fr["img_h"]
                )
                frame_eval_cache[cache_key] = (tpf, fpf, fnf)

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


def evaluate_lovo_worker(task):
    scale_key, s, X_scale, y_act, v_act, design_vids, hidden_sizes = task
    n_samples = len(y_act)
    oof_scores = np.full(n_samples, np.nan)

    for vid in design_vids:
        test_mask = (v_act == vid)
        train_mask = ~test_mask

        X_tr, y_tr = X_scale[train_mask], y_act[train_mask]
        X_te = X_scale[test_mask]

        if len(X_te) == 0 or len(np.unique(y_tr)) < 2:
            continue

        m_tr = X_tr.mean(axis=0)
        s_tr = X_tr.std(axis=0)
        s_tr[s_tr < 1e-6] = 1.0

        X_tr_std = (X_tr - m_tr) / s_tr
        X_te_std = (X_te - m_tr) / s_tr

        pos_idx = np.where(y_tr == 1)[0]
        neg_idx = np.where(y_tr == 0)[0]
        rng = np.random.default_rng(s * 1000 + vid)
        oversampled_pos = rng.choice(pos_idx, size=len(neg_idx), replace=True)
        bal_idx = np.concatenate([neg_idx, oversampled_pos])
        rng.shuffle(bal_idx)

        clf = MLPClassifier(hidden_layer_sizes=hidden_sizes, activation="relu", alpha=0.05, max_iter=350, random_state=s)
        clf.fit(X_tr_std[bal_idx], y_tr[bal_idx])
        oof_scores[test_mask] = clf.predict_proba(X_te_std)[:, 1]

    assert np.isfinite(oof_scores).all(), f"[FATAL] Non-finite scores for {scale_key} seed {s}!"
    return scale_key, s, oof_scores


def main():
    t_start_total = time.time()
    driver_sha256 = compute_sha256(Path(__file__))
    print("=" * 115)
    print("MARSEN PHASE B2c: TRI-SCALE COMPLEMENTARITY RESOLUTION (v3, P-1933, P-1934, P-1938, P-1942, P-1943)")
    print(f"Driver Script SHA-256: {driver_sha256}")
    print("=" * 115)

    assert B2B_RESULTS_JSON.exists(), f"[FATAL] Missing B2b results at {B2B_RESULTS_JSON}!"
    with open(B2B_RESULTS_JSON, "r", encoding="utf-8") as f:
        b2b_data = json.load(f)
    b2b_standalone = {item["scale"]: item["mean_ap"] for item in b2b_data["standalone_fov_summary"]}
    print(f"[+] Loaded Canonical B2b References: 1.0x={b2b_standalone[1.0]:.4f}, 3.0x={b2b_standalone[3.0]:.4f}, 5.0x={b2b_standalone[5.0]:.4f}")

    with open(SPLIT_MANIFEST_JSON, "r", encoding="utf-8") as f:
        design_vids = sorted(json.load(f)["_meta"]["train_videos"])
    assert design_vids == EXPECTED_DESIGN_VIDS, f"[FATAL] Cohort mismatch: {design_vids}"

    cached_queries = torch.load(QUERY_DATASET_PTH, map_location="cpu")["extracted_queries"]
    frame_bundle = torch.load(FRAME_TENSORS_PTH, map_location="cpu")
    frame_records = frame_bundle["frame_records"]

    active_indices = [idx for idx, q in enumerate(cached_queries) if q["is_active"]]
    active_queries = [cached_queries[idx] for idx in active_indices]
    n_active = len(active_queries)
    y_act = np.array([q["label_op_util"] for q in active_queries], dtype=np.int32)
    v_act = np.array([q["video_id"] for q in active_queries], dtype=np.int32)

    assert n_active == 1915 and int(y_act.sum()) == 92, "[FATAL] Population mismatch!"

    branch_bundle = torch.load(CACHED_BRANCHES_PTH, map_location="cpu")
    feat_meta_15 = branch_bundle["branches"]["metadata"][:, :15].astype(np.float32)

    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        coco_full = json.load(f)
    cat_ids = sorted(c["id"] for c in coco_full["categories"])
    label_to_cat_id = {i: cat_id for i, cat_id in enumerate(cat_ids)}

    active_key_set = set((q["image_id"], q["query_id"]) for q in active_queries)
    positive_key_set = set((q["image_id"], q["query_id"]) for q in active_queries if q["label_op_util"] == 1)

    base_frame_cache = {}
    for fr in frame_records:
        tpb, fpb, fnb = evaluate_operational_top100(
            fr["d_logits"], fr["d_boxes"], fr["gt_boxes_px"], fr["gt_cats"], label_to_cat_id, fr["img_w"], fr["img_h"]
        )
        base_frame_cache[fr["image_id"]] = (tpb, fpb, fnb)

    c_bundle = torch.load(CACHED_B2B_FOV_PTH, map_location="cpu")
    fov_features = {scale: c_bundle[f"fov_{scale}"].astype(np.float32) for scale in TRISCALE_FOVS}
    print("[+] Ingested cached features for 1.0x, 3.0x, 5.0x.")

    tasks = []
    for scale in TRISCALE_FOVS:
        X_scale = np.concatenate([fov_features[scale], feat_meta_15], axis=-1)
        for s in SEEDS:
            tasks.append((scale, s, X_scale, y_act, v_act, design_vids, WINNING_HIDDEN))

    n_workers = int(os.environ.get("MARSEN_WORKERS", 6))
    print(f"[*] Dispatching {len(tasks)} LOVO tasks across {n_workers} CPU worker processes...")
    t0 = time.time()

    task_results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for res in executor.map(evaluate_lovo_worker, tasks):
            task_results.append(res)
    print(f"[+] All 9 Models Fitted in {time.time() - t0:.2f}s")

    oof_preds = defaultdict(dict)
    for scale, s, oof in task_results:
        oof_preds[scale][s] = oof

    # P-1942: Freeze canonical OOF predictions to disk
    torch.save({
        "driver_sha256": driver_sha256,
        "oof_preds": {scale: {s: oof_preds[scale][s] for s in SEEDS} for scale in TRISCALE_FOVS}
    }, OOF_VECTORS_PTH)
    print(f"[+] P-1942 Certified: 9 Canonical OOF Vectors Frozen: {OOF_VECTORS_PTH.resolve()}")

    # Continuity check
    print("\n[*] P-1938 Continuity Checks Against Canonical B2b Standalones:")
    for scale in TRISCALE_FOVS:
        actual_ap = np.mean([average_precision_score(y_act, oof_preds[scale][s]) for s in SEEDS])
        expected_ap = b2b_standalone[scale]
        diff = abs(actual_ap - expected_ap)
        print(f"  - FOV {scale:.1f}x: Mean AP = {actual_ap:.4f} (Expected: {expected_ap:.4f}, diff = {diff:.2e})")
        assert diff < 1e-3, f"[FATAL] FOV {scale}x failed continuity: {actual_ap} != {expected_ap}"
    print("[+] P-1938 Certified: Exact numerical continuity confirmed (< 1e-3).")

    # P-1934 & P-1943 Correlation and Positive Diagnostic
    print("\n" + "=" * 115)
    print("P-1934 & P-1943: CONTEXT CORRELATION & POSITIVE-QUERY DIAGNOSTICS")
    print("=" * 115)

    pos_mask = (y_act == 1)
    correlation_report = {}

    for pair_name, sA, sB in [("1.0x vs 3.0x", 1.0, 3.0), ("1.0x vs 5.0x", 1.0, 5.0), ("3.0x vs 5.0x", 3.0, 5.0)]:
        rhos_all, pearsons_all = [], []
        global_rank_pos_rhos, true_pos_spearmans = [], []
        disagree_A_beats_B, disagree_B_beats_A = [], []

        for s in SEEDS:
            oA, oB = oof_preds[sA][s], oof_preds[sB][s]
            rA_global = rankdata(oA, method="average") / float(n_active)
            rB_global = rankdata(oB, method="average") / float(n_active)

            rhos_all.append(float(np.corrcoef(rA_global, rB_global)[0, 1]))
            pearsons_all.append(float(np.corrcoef(oA, oB)[0, 1]))

            # Metric 1: Global percentile rank correlation among positives
            global_rank_pos_rhos.append(float(np.corrcoef(rA_global[pos_mask], rB_global[pos_mask])[0, 1]))

            # Metric 2: True within-positive Spearman rho
            rA_pos = rankdata(oA[pos_mask], method="average")
            rB_pos = rankdata(oB[pos_mask], method="average")
            true_pos_spearmans.append(float(np.corrcoef(rA_pos, rB_pos)[0, 1]))

            # Positive Discordance
            diff_pos = rA_global[pos_mask] - rB_global[pos_mask]
            disagree_A_beats_B.append(int(np.sum(diff_pos > 0.20)))
            disagree_B_beats_A.append(int(np.sum(diff_pos < -0.20)))

        pair_data = {
            "all_spearman": float(np.mean(rhos_all)),
            "all_pearson": float(np.mean(pearsons_all)),
            "global_rank_pos_rho": float(np.mean(global_rank_pos_rhos)),
            "true_within_pos_spearman": float(np.mean(true_pos_spearmans)),
            "A_beats_B_discordant": float(np.mean(disagree_A_beats_B)),
            "B_beats_A_discordant": float(np.mean(disagree_B_beats_A))
        }
        correlation_report[pair_name] = pair_data

        print(f"[*] Pair {pair_name:<14} | All Spearman: {pair_data['all_spearman']:.4f} | All Pearson: {pair_data['all_pearson']:.4f}")
        print(f"    - Global Rank Corr (Positives): {pair_data['global_rank_pos_rho']:.4f} | TRUE Within-Positive Spearman: {pair_data['true_within_pos_spearman']:.4f}")
        print(f"    - Discordant Positives (>0.20 rank diff): {sA}x higher in {pair_data['A_beats_B_discordant']:.1f}/92 | {sB}x higher in {pair_data['B_beats_A_discordant']:.1f}/92")

    # Positive Percentile Rank Statistics
    print("\n[*] Positive Target Percentile Rank Statistics (Higher = Better Ranking):")
    pos_percentile_stats = {}
    for scale in TRISCALE_FOVS:
        mean_p_ranks, med_p_ranks = [], []
        for s in SEEDS:
            r = rankdata(oof_preds[scale][s], method="average") / float(n_active)
            mean_p_ranks.append(float(np.mean(r[pos_mask])))
            med_p_ranks.append(float(np.median(r[pos_mask])))
        pos_percentile_stats[f"{scale:.1f}x"] = {
            "mean_pos_percentile": float(np.mean(mean_p_ranks)),
            "median_pos_percentile": float(np.mean(med_p_ranks))
        }
        print(f"    - FOV {scale:.1f}x: Mean Positive Percentile = {np.mean(mean_p_ranks):.4f} | Median = {np.mean(med_p_ranks):.4f}")

    # Complete Ensemble Matrix
    print("\n" + "=" * 115)
    print("P-1933 & P-1938: COMPLETE ENSEMBLE AUDIT (RANK & PROBABILITY FUSION AP + F1*)")
    print("=" * 115)
    print(f"{'Fusion Scheme':<36} | {'Rank AP':<10} | {'Rank F1*':<10} | {'Prob AP':<10} | {'Prob F1*':<10} | {'dAP (Rank)':<11} | {'dAP (Prob)'}")
    print("-" * 118)

    base_10_aps = [float(average_precision_score(y_act, oof_preds[1.0][s])) for s in SEEDS]
    mean_base_ap = float(np.mean(base_10_aps))

    def get_rank_norm(arr):
        return rankdata(arr, method="average") / float(len(arr))

    ensemble_definitions = [
        ("Standalone: FOV 1.0x (Base)",     lambda s: (get_rank_norm(oof_preds[1.0][s]), oof_preds[1.0][s])),
        ("Standalone: FOV 3.0x",            lambda s: (get_rank_norm(oof_preds[3.0][s]), oof_preds[3.0][s])),
        ("Standalone: FOV 5.0x",            lambda s: (get_rank_norm(oof_preds[5.0][s]), oof_preds[5.0][s])),
        ("Pair: 1.0x + 3.0x",               lambda s: (0.5 * get_rank_norm(oof_preds[1.0][s]) + 0.5 * get_rank_norm(oof_preds[3.0][s]), 0.5 * oof_preds[1.0][s] + 0.5 * oof_preds[3.0][s])),
        ("Pair: 1.0x + 5.0x",               lambda s: (0.5 * get_rank_norm(oof_preds[1.0][s]) + 0.5 * get_rank_norm(oof_preds[5.0][s]), 0.5 * oof_preds[1.0][s] + 0.5 * oof_preds[5.0][s])),
        ("Pair: 3.0x + 5.0x",               lambda s: (0.5 * get_rank_norm(oof_preds[3.0][s]) + 0.5 * get_rank_norm(oof_preds[5.0][s]), 0.5 * oof_preds[3.0][s] + 0.5 * oof_preds[5.0][s])),
        ("Triple: Equal (1x + 3x + 5x)/3",  lambda s: ((get_rank_norm(oof_preds[1.0][s]) + get_rank_norm(oof_preds[3.0][s]) + get_rank_norm(oof_preds[5.0][s])) / 3.0, (oof_preds[1.0][s] + oof_preds[3.0][s] + oof_preds[5.0][s]) / 3.0)),
        ("Triple: Base-Anchored (50/25/25)",lambda s: (0.50 * get_rank_norm(oof_preds[1.0][s]) + 0.25 * get_rank_norm(oof_preds[3.0][s]) + 0.25 * get_rank_norm(oof_preds[5.0][s]), 0.50 * oof_preds[1.0][s] + 0.25 * oof_preds[3.0][s] + 0.25 * oof_preds[5.0][s])),
    ]

    fusion_summary = []
    for f_name, f_builder in ensemble_definitions:
        r_aps, r_f1s, p_aps, p_f1s = [], [], [], []

        for s in SEEDS:
            r_vec, p_vec = f_builder(s)
            
            # Rank metrics
            ap_r = float(average_precision_score(y_act, r_vec))
            taus_r = get_globally_exact_thresholds(r_vec)
            s_map_r = {(q["image_id"], q["query_id"]): float(r_vec[i]) for i, q in enumerate(active_queries)}
            _, f1_r, _, _ = run_detector_sweep_fast(s_map_r, frame_records, active_key_set, positive_key_set, label_to_cat_id, taus_r, base_frame_cache)

            # Prob metrics
            ap_p = float(average_precision_score(y_act, p_vec))
            taus_p = get_globally_exact_thresholds(p_vec)
            s_map_p = {(q["image_id"], q["query_id"]): float(p_vec[i]) for i, q in enumerate(active_queries)}
            _, f1_p, _, _ = run_detector_sweep_fast(s_map_p, frame_records, active_key_set, positive_key_set, label_to_cat_id, taus_p, base_frame_cache)

            r_aps.append(ap_r)
            r_f1s.append(f1_r)
            p_aps.append(ap_p)
            p_f1s.append(f1_p)

        mean_r_ap = float(np.mean(r_aps))
        mean_r_f1 = float(np.mean(r_f1s))
        mean_p_ap = float(np.mean(p_aps))
        mean_p_f1 = float(np.mean(p_f1s))

        d_r_ap = mean_r_ap - mean_base_ap
        d_p_ap = mean_p_ap - mean_base_ap

        fusion_summary.append({
            "name": f_name,
            "mean_rank_ap": mean_r_ap, "mean_rank_f1": mean_r_f1,
            "mean_prob_ap": mean_p_ap, "mean_prob_f1": mean_p_f1,
            "d_rank_ap": d_r_ap, "d_prob_ap": d_p_ap,
            "r_aps": r_aps, "r_f1s": r_f1s,
            "p_aps": p_aps, "p_f1s": p_f1s
        })

        print(f"{f_name:<36} | {mean_r_ap:8.4f}   | {mean_r_f1:8.6f} | {mean_p_ap:8.4f}   | {mean_p_f1:8.6f} | {d_r_ap:+10.4f}  | {d_p_ap:+10.4f}")

    results_payload = {
        "_meta": {
            "artifact_name": "marsen_b2c_triscale_results_v3",
            "driver_sha256": driver_sha256,
            "base_10_ap": mean_base_ap,
            "total_seconds": float(time.time() - t_start_total)
        },
        "fusion_summary": fusion_summary,
        "correlation_report": correlation_report,
        "pos_percentile_stats": pos_percentile_stats
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    res_hash = compute_sha256(RESULTS_JSON)
    print(f"\n[+] B2c Results Frozen: {RESULTS_JSON.resolve()} (SHA-256: {res_hash})")
    print(f"    Execution Time: {time.time() - t_start_total:.2f}s")
    print("=" * 115)


if __name__ == "__main__":
    main()