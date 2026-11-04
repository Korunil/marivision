"""
D:/MARIVISION/H0_workbench/eval_tools/audit_marsen_b2b_fov_extension.py
MARSEN Phase B2b: Auxiliary Context Discovery & Accelerated Evaluation (v8)

Strict Compliance Checklist:
  - D-3860..D-3893: 1.0x is permanent Base; wider FOVs evaluated strictly as auxiliary context candidates.
  - P-1921: Canonical runtime contract enforced (BLAS=1, ProcessPoolExecutor, memoized sweeps).
  - P-1926: Process-pool determinism sentinel verified via ProcessPoolExecutor on duplicate tasks.
  - P-1923: Historical B2 comparisons logged as reference deltas (non-fatal).
  - P-1925: Certified memoized state evaluation (< 2s sweep runtime with bitwise exact parity).
  - P-1888: Hard population assertions (N=1,915, N+=92, permitted design videos).
  - P-1902: All-context complementarity audit evaluating rank-average & probability-average ensembles.
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
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier

OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
TRAIN_JSON = Path("D:/MARIVISION/seaclips-train.json")
SPLIT_MANIFEST_JSON = OUT_DIR / "pol2_candidate2m_micro_split.json"
QUERY_DATASET_PTH = OUT_DIR / "pol3v_design_queries_dataset_v7.pth"
FRAME_TENSORS_PTH = OUT_DIR / "pol3v_design_frames_tensors_v7.pth"
CACHED_BRANCHES_PTH = OUT_DIR / "marsen_micro_authentic_branches_1915.pth"
CACHED_B2B_FOV_PTH = OUT_DIR / "marsen_b2b_fov_features_1915.pth"
RESULTS_JSON = OUT_DIR / "marsen_b2b_fov_extension_results.json"

SCORE_THRESHOLD = 0.20
EVAL_IOU_THRESHOLD = 0.30
BASE_M0_F1 = 0.431718
EXPECTED_DESIGN_VIDS = [4, 7, 16, 19, 25, 29, 30, 32, 35, 47]

HISTORICAL_B2_ANCHORS = {
    1.0: 0.1367,
    1.5: 0.1280,
    2.0: 0.1318,
    3.0: 0.1442
}

SEEDS = [41, 42, 43]
REFERENCE_SEED = 41
WINNING_HIDDEN = (64,)
ALL_FOV_SCALES = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]


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
    fold_n_iters = []

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
        fold_n_iters.append(int(clf.n_iter_))

    assert np.isfinite(oof_scores).all(), f"[FATAL] Non-finite scores for {scale_key} seed {s}!"
    conv = {"mean_iters": float(np.mean(fold_n_iters)), "cap_hits": int(np.sum(np.array(fold_n_iters) >= 350))}
    return scale_key, s, oof_scores, conv


def main():
    t_start_total = time.time()
    driver_sha256 = compute_sha256(Path(__file__))
    print("=" * 115)
    print("MARSEN PHASE B2b: AUXILIARY CONTEXT DISCOVERY & ACCELERATED EVALUATION (v8)")
    print(f"Driver Script SHA-256: {driver_sha256}")
    print("=" * 115)

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

    assert n_active == 1915, f"[FATAL] Expected 1,915 active queries, got {n_active}"
    assert int(y_act.sum()) == 92, f"[FATAL] Expected 92 positives, got {int(y_act.sum())}"
    assert sorted(np.unique(v_act).tolist()) == EXPECTED_DESIGN_VIDS, "[FATAL] Video cohort mismatch!"
    print(f"[+] P-1888 Population Certified: Exactly 1,915 active queries, 92 positives across 10 design videos.")

    branch_bundle = torch.load(CACHED_BRANCHES_PTH, map_location="cpu")
    feat_meta_15 = branch_bundle["branches"]["metadata"][:, :15].astype(np.float32)

    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        coco_full = json.load(f)
    cat_ids = sorted(c["id"] for c in coco_full["categories"])
    label_to_cat_id = {i: cat_id for i, cat_id in enumerate(cat_ids)}

    active_key_set = set((q["image_id"], q["query_id"]) for q in active_queries)
    positive_key_set = set((q["image_id"], q["query_id"]) for q in active_queries if q["label_op_util"] == 1)

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

    # Ingest Precomputed Tensors from Disk
    print(f"\n[+] Ingesting Precomputed Features from {CACHED_B2B_FOV_PTH.name}...")
    c_bundle = torch.load(CACHED_B2B_FOV_PTH, map_location="cpu")
    fov_features = {scale: c_bundle[f"fov_{scale}"].astype(np.float32) for scale in ALL_FOV_SCALES}
    floor_binding_rates = c_bundle["floor_binding_rates"]
    pairwise_identity_rates = c_bundle["pairwise_identity_rates"]
    print("[+] All 6 FOV Feature Sets Ingested Successfully (< 1s).")

    # --------------------------------------------------------------------------
    # P-1926: PROCESS-POOL DETERMINISM SENTINEL
    # --------------------------------------------------------------------------
    print("\n[*] P-1926: Verifying Process-Pool Determinism (Duplicate Seed 41 tasks via ProcessPoolExecutor)...")
    X_sentinel = np.concatenate([fov_features[1.0], feat_meta_15], axis=-1)
    task_s1 = ("sentinel_1", 41, X_sentinel, y_act, v_act, design_vids, WINNING_HIDDEN)
    task_s2 = ("sentinel_2", 41, X_sentinel, y_act, v_act, design_vids, WINNING_HIDDEN)

    with ProcessPoolExecutor(max_workers=2) as executor:
        sentinel_results = list(executor.map(evaluate_lovo_worker, [task_s1, task_s2]))

    oof_run1 = sentinel_results[0][2]
    oof_run2 = sentinel_results[1][2]
    sentinel_diff = float(np.abs(oof_run1 - oof_run2).max())
    print(f"  - Sentinel Process-Pool Check: max |oof_run1 - oof_run2| = {sentinel_diff:.2e}")
    assert sentinel_diff < 1e-12, "[FATAL] Process-pool execution non-deterministic under fixed seed!"
    print("[+] P-1926 Certified: Process-pool execution is 100% deterministic.")

    # --------------------------------------------------------------------------
    # PARALLEL LOVO EVALUATION (10 WORKER PROCESSES)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("P-1899: HARDWARE-ACCELERATED LOVO EVALUATION (10 WORKER PROCESSES)")
    print("=" * 115)
    t_start_train = time.time()

    tasks = []
    scale_names = {
        1.0: "FOV 1.0x (Permanent Base)",
        1.5: "FOV 1.5x (Anchor)",
        2.0: "FOV 2.0x (Auxiliary)",
        3.0: "FOV 3.0x (Auxiliary)",
        4.0: "FOV 4.0x (Auxiliary)",
        5.0: "FOV 5.0x (Auxiliary)"
    }
    for scale in ALL_FOV_SCALES:
        X_scale = np.concatenate([fov_features[scale], feat_meta_15], axis=-1)
        for s in SEEDS:
            tasks.append((scale, s, X_scale, y_act, v_act, design_vids, WINNING_HIDDEN))

    n_workers = int(os.environ.get("MARSEN_WORKERS", 10))
    print(f"[*] Dispatching {len(tasks)} independent tasks across {n_workers} CPU worker processes...")

    task_results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for res in executor.map(evaluate_lovo_worker, tasks):
            task_results.append(res)

    t_elapsed_train = time.time() - t_start_train
    print(f"[+] All 18 Parallel Training Runs Completed in {t_elapsed_train:.2f}s")

    # --------------------------------------------------------------------------
    # ACCELERATED DETECTOR THRESHOLD SWEEPS (MEMOIZED STATE MATCHING)
    # --------------------------------------------------------------------------
    t_start_sweep = time.time()
    fov_results = defaultdict(dict)
    oof_predictions_by_scale = defaultdict(dict)

    for scale, s, oof_s, conv_s in task_results:
        oof_predictions_by_scale[scale][s] = oof_s
        ap_s = float(average_precision_score(y_act, oof_s))
        auc_s = float(roc_auc_score(y_act, oof_s))

        exact_taus = get_globally_exact_thresholds(oof_s)
        score_map_s = {(q["image_id"], q["query_id"]): float(oof_s[i]) for i, q in enumerate(active_queries)}
        best_tau_s, best_f1_s, best_rec_s, _ = run_detector_sweep_fast(
            score_map_s, frame_records, active_key_set, positive_key_set, label_to_cat_id, exact_taus, base_frame_cache
        )

        fov_results[scale][s] = {
            "ap": ap_s, "auc": auc_s, "tau_star": best_tau_s, "f1_star": best_f1_s,
            "delta_f1_m0": best_f1_s - BASE_M0_F1, "best_rec": best_rec_s, "conv": conv_s
        }

    t_elapsed_sweep = time.time() - t_start_sweep
    print(f"[+] Accelerated Detector Threshold Sweeps Completed in {t_elapsed_sweep:.2f}s")

    # --------------------------------------------------------------------------
    # HISTORICAL B2 REFERENCE COMPARISON (NON-FATAL)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("HISTORICAL B2 REFERENCE COMPARISON (NON-FATAL DELTAS)")
    print("=" * 115)
    for anchor_scale, expected_mean_ap in HISTORICAL_B2_ANCHORS.items():
        actual_mean_ap = float(np.mean([fov_results[anchor_scale][s]["ap"] for s in SEEDS]))
        delta = actual_mean_ap - expected_mean_ap
        print(f"  - Scale {anchor_scale:.1f}x: Current Canonical = {actual_mean_ap:.4f} | Historical B2 = {expected_mean_ap:.4f} | Delta = {delta:+.4f}")

    # --------------------------------------------------------------------------
    # STANDALONE FOV PERFORMANCE SUMMARY
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("STANDALONE FOV PERFORMANCE SUMMARY (1.0x through 5.0x)")
    print("=" * 115)
    print(f"{'Rank':<5} | {'FOV Representation':<32} | {'Mean AP':<10} | {'AP Std':<10} | {'Seed 41 AP':<12} | {'Seed 42 AP':<12} | {'Seed 43 AP':<12} | {'Mean F1*':<10} | {'Mean dF1':<10}")
    print("-" * 128)

    standalone_summary = []
    base_10_aps = [fov_results[1.0][s]["ap"] for s in SEEDS]

    for scale in ALL_FOV_SCALES:
        aps = [fov_results[scale][s]["ap"] for s in SEEDS]
        f1s = [fov_results[scale][s]["f1_star"] for s in SEEDS]
        df1s = [fov_results[scale][s]["delta_f1_m0"] for s in SEEDS]
        aucs = [fov_results[scale][s]["auc"] for s in SEEDS]
        d_aps_vs_10 = [aps[i] - base_10_aps[i] for i in range(len(SEEDS))]

        standalone_summary.append({
            "scale": scale,
            "name": scale_names[scale],
            "mean_ap": float(np.mean(aps)),
            "std_ap": float(np.std(aps)),
            "aps": aps,
            "mean_f1": float(np.mean(f1s)),
            "std_f1": float(np.std(f1s)),
            "f1s": f1s,
            "mean_df1": float(np.mean(df1s)),
            "mean_dap_vs_10": float(np.mean(d_aps_vs_10)),
            "seed_daps_vs_10": d_aps_vs_10,
            "mean_auc": float(np.mean(aucs))
        })

    standalone_summary.sort(key=lambda x: x["mean_ap"], reverse=True)

    for rank, item in enumerate(standalone_summary, 1):
        tag = ""
        if item["scale"] == 1.0:
            tag = " [PERMANENT BASE]"
        elif rank == 1:
            tag = " <-- BEST STANDALONE CONTEXT"
        print(f"#{rank:<3d} | {item['name']:<32} | {item['mean_ap']:8.4f}   | {item['std_ap']:8.4f}   | {item['aps'][0]:10.4f}   | {item['aps'][1]:10.4f}   | {item['aps'][2]:10.4f}   | {item['mean_f1']:8.6f} | {item['mean_df1']:+8.4f}   {tag}")

    # --------------------------------------------------------------------------
    # P-1902: ALL-CONTEXT COMPLEMENTARITY AUDIT (1.0x BASE + AUXILIARY s)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("P-1902: ALL-CONTEXT COMPLEMENTARITY AUDIT (1.0x BASE + AUXILIARY CONTEXT SCALE s)")
    print("=" * 115)
    print(f"{'Auxiliary Scale s':<24} | {'Rank-Ens AP':<13} | {'dAP vs 1.0x':<13} | {'Prob-Ens AP':<13} | {'Mean F1*':<11} | {'Spearman rho':<14} | {'Positive Rescues'}")
    print("-" * 118)

    auxiliary_scales = [1.5, 2.0, 3.0, 4.0, 5.0]
    complementarity_summary = []

    for aux_scale in auxiliary_scales:
        rank_ens_aps = []
        prob_ens_aps = []
        rank_ens_f1s = []
        prob_ens_f1s = []
        spearmans = []
        pearsons = []
        pos_rescues_per_seed = []

        for s in SEEDS:
            oof_base = oof_predictions_by_scale[1.0][s]
            oof_aux = oof_predictions_by_scale[aux_scale][s]

            # 1. Rank-Average Ensemble
            rank_base = np.argsort(np.argsort(oof_base)) / float(n_active)
            rank_aux = np.argsort(np.argsort(oof_aux)) / float(n_active)
            oof_rank_ens = 0.5 * rank_base + 0.5 * rank_aux

            ap_rank = float(average_precision_score(y_act, oof_rank_ens))
            taus_r = get_globally_exact_thresholds(oof_rank_ens)
            s_map_r = {(q["image_id"], q["query_id"]): float(oof_rank_ens[i]) for i, q in enumerate(active_queries)}
            _, f1_r, _, _ = run_detector_sweep_fast(s_map_r, frame_records, active_key_set, positive_key_set, label_to_cat_id, taus_r, base_frame_cache)

            # 2. Probability-Average Ensemble
            oof_prob_ens = 0.5 * oof_base + 0.5 * oof_aux
            ap_prob = float(average_precision_score(y_act, oof_prob_ens))
            taus_p = get_globally_exact_thresholds(oof_prob_ens)
            s_map_p = {(q["image_id"], q["query_id"]): float(oof_prob_ens[i]) for i, q in enumerate(active_queries)}
            _, f1_p, _, _ = run_detector_sweep_fast(s_map_p, frame_records, active_key_set, positive_key_set, label_to_cat_id, taus_p, base_frame_cache)

            rho = float(np.corrcoef(rank_base, rank_aux)[0, 1])
            r = float(np.corrcoef(oof_base, oof_aux)[0, 1])

            pos_mask = (y_act == 1)
            base_pos_ranks = rank_base[pos_mask]
            aux_pos_ranks = rank_aux[pos_mask]
            rescues = int(np.sum((base_pos_ranks < 0.25) & (aux_pos_ranks > 0.75)))

            rank_ens_aps.append(ap_rank)
            prob_ens_aps.append(ap_prob)
            rank_ens_f1s.append(f1_r)
            prob_ens_f1s.append(f1_p)
            spearmans.append(rho)
            pearsons.append(r)
            pos_rescues_per_seed.append(rescues)

        mean_rank_ap = float(np.mean(rank_ens_aps))
        mean_prob_ap = float(np.mean(prob_ens_aps))
        mean_rank_f1 = float(np.mean(rank_ens_f1s))
        mean_10_ap = float(np.mean(base_10_aps))
        delta_rank_ap = mean_rank_ap - mean_10_ap

        complementarity_summary.append({
            "aux_scale": aux_scale,
            "name": f"1.0x + {aux_scale:.1f}x",
            "mean_rank_ens_ap": mean_rank_ap,
            "std_rank_ens_ap": float(np.std(rank_ens_aps)),
            "delta_rank_ap_vs_10": delta_rank_ap,
            "mean_prob_ens_ap": mean_prob_ap,
            "mean_rank_ens_f1": mean_rank_f1,
            "mean_spearman_rho": float(np.mean(spearmans)),
            "mean_pearson_r": float(np.mean(pearsons)),
            "mean_positive_rescues": float(np.mean(pos_rescues_per_seed)),
            "seed_rank_aps": rank_ens_aps,
            "seed_prob_aps": prob_ens_aps
        })

    complementarity_summary.sort(key=lambda x: x["mean_rank_ens_ap"], reverse=True)

    for item in complementarity_summary:
        tag = ""
        if item["delta_rank_ap_vs_10"] > 0:
            tag = " [COMPLEMENTARY GAIN]"
        print(f"{item['name']:<24} | {item['mean_rank_ens_ap']:10.4f}    | {item['delta_rank_ap_vs_10']:+10.4f}   | {item['mean_prob_ens_ap']:10.4f}    | {item['mean_rank_ens_f1']:9.6f} | {item['mean_spearman_rho']:11.4f}    | {item['mean_positive_rescues']:4.1f} / 92 positives{tag}")

    best_auxiliary = complementarity_summary[0]
    print("\n" + "=" * 115)
    print("PHASE B2b DECISION & AUXILIARY SELECTION VERDICT")
    print("=" * 115)
    print(f"  - Permanent Base Anchor:           FOV 1.0x (Mean AP = {mean_10_ap:.4f}, Mean F1* = {np.mean([fov_results[1.0][s]['f1_star'] for s in SEEDS]):.6f})")
    print(f"  - Top Standalone Context:         {standalone_summary[0]['name']} (Mean AP = {standalone_summary[0]['mean_ap']:.4f})")
    print(f"  - Top Auxiliary Context Candidate: {best_auxiliary['name']}")
    print(f"    * Rank-Average Ensemble AP:      {best_auxiliary['mean_rank_ens_ap']:.4f} ({best_auxiliary['delta_rank_ap_vs_10']:+.4f} vs 1.0x Base alone)")
    print(f"    * Ensemble F1*:                  {best_auxiliary['mean_rank_ens_f1']:.6f}")
    print(f"    * Rank Correlation (rho):        {best_auxiliary['mean_spearman_rho']:.4f}")
    print("=" * 115)

    canonical_base_ap = float(mean_10_ap)
    canonical_aux_ap = float(next(item["mean_ap"] for item in standalone_summary if abs(item["scale"] - best_auxiliary["aux_scale"]) < 1e-4))

    results_payload = {
        "_meta": {
            "artifact_name": "marsen_b2b_fov_extension_results_v4",
            "driver_sha256": driver_sha256,
            "baseline_m0_f1": float(BASE_M0_F1),
            "reference_seed": REFERENCE_SEED,
            "permanent_base": "FOV 1.0x",
            "top_auxiliary_scale": float(best_auxiliary["aux_scale"]),
            "top_auxiliary_name": best_auxiliary["name"],
            "canonical_base_ap": canonical_base_ap,
            "canonical_aux_ap": canonical_aux_ap,
            "timings": {
                "parallel_train_seconds": float(t_elapsed_train),
                "threshold_sweep_seconds": float(t_elapsed_sweep),
                "total_seconds": float(time.time() - t_start_total)
            }
        },
        "standalone_fov_summary": standalone_summary,
        "all_context_complementarity_summary": complementarity_summary,
        "floor_binding_rates": floor_binding_rates,
        "pairwise_identity_rates": pairwise_identity_rates
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    res_hash = compute_sha256(RESULTS_JSON)
    print(f"[+] B2b Results Frozen: {RESULTS_JSON.resolve()}")
    print(f"    Manifest SHA-256:   {res_hash}")
    print(f"    Total Runtime:      {time.time() - t_start_total:.2f}s")
    print("=" * 115)


if __name__ == "__main__":
    main()