"""
D:/MARIVISION/H0_workbench/eval_tools/audit_marsen_classifier_prescreen.py
MARSEN Phase B2.6 Diagnostic: Downstream Classifier Pre-Screen on 1.0x Base Representation (P-1906)

Strict Compliance Checklist:
  - D-3866: Uses 1.0x FOV Base representation (128-D visual + 15-D metadata = 143-D), NOT legacy 1.5x.
  - Zero GPU / Zero I/O Contention: Runs CPU-only using existing cached features in parallel with Terminal 1.
  - Models Evaluated: MLP (H=64), HistGradientBoosting, RandomForest, LogisticRegression, plus CatBoost/XGBoost if installed.
  - Identical Protocol: 10-fold LOVO cross-validation across Seeds 41, 42, 43 with globally exact threshold sweeps.
"""

import os
import sys

# P-1905: Bound thread count to prevent oversubscribing CPU when Terminal 1 is active
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import json
import time
import hashlib
from PIL import Image
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Optional imports for tree packages
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
TRAIN_JSON = Path("D:/MARIVISION/seaclips-train.json")
SPLIT_MANIFEST_JSON = OUT_DIR / "pol2_candidate2m_micro_split.json"
QUERY_DATASET_PTH = OUT_DIR / "pol3v_design_queries_dataset_v7.pth"
FRAME_TENSORS_PTH = OUT_DIR / "pol3v_design_frames_tensors_v7.pth"
CACHED_BRANCHES_PTH = OUT_DIR / "marsen_micro_authentic_branches_1915.pth"
CACHED_MULTI_FOV_PTH = OUT_DIR / "marsen_multi_fov_features_1915.pth"
RESULTS_JSON = OUT_DIR / "marsen_classifier_prescreen_results.json"

SCORE_THRESHOLD = 0.20
EVAL_IOU_THRESHOLD = 0.30
BASE_M0_F1 = 0.431718
EXPECTED_DESIGN_VIDS = [4, 7, 16, 19, 25, 29, 30, 32, 35, 47]
SEEDS = [41, 42, 43]


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

        if f1 > best_f1 + 1e-12 or (abs(f1 - best_f1) <= 1e-12 and (best_tau is None or tau > best_tau)):
            best_f1 = f1
            best_tau = tau
            best_rec = sweep_records[tau]

    return best_tau, best_f1, best_rec, sweep_records


def train_classifier_lovo(model_type, X_feat, y_act, v_act, design_vids, seed):
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

        X_train_fold = X_tr_std[bal_idx]
        y_train_fold = y_tr[bal_idx]

        if model_type == "mlp_64":
            clf = MLPClassifier(hidden_layer_sizes=(64,), activation="relu", alpha=0.05, max_iter=350, random_state=seed)
            clf.fit(X_train_fold, y_train_fold)
            oof_scores[test_mask] = clf.predict_proba(X_te_std)[:, 1]

        elif model_type == "hist_gb":
            clf = HistGradientBoostingClassifier(max_iter=150, max_depth=6, l2_regularization=0.1, random_state=seed)
            clf.fit(X_train_fold, y_train_fold)
            oof_scores[test_mask] = clf.predict_proba(X_te_std)[:, 1]

        elif model_type == "random_forest":
            clf = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=5, random_state=seed, n_jobs=2)
            clf.fit(X_train_fold, y_train_fold)
            oof_scores[test_mask] = clf.predict_proba(X_te_std)[:, 1]

        elif model_type == "logreg":
            clf = LogisticRegression(C=1.0, max_iter=300, random_state=seed)
            clf.fit(X_train_fold, y_train_fold)
            oof_scores[test_mask] = clf.predict_proba(X_te_std)[:, 1]

        elif model_type == "catboost" and HAS_CATBOOST:
            clf = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.08, verbose=0, random_seed=seed, thread_count=2)
            clf.fit(X_train_fold, y_train_fold)
            oof_scores[test_mask] = clf.predict_proba(X_te_std)[:, 1]

        elif model_type == "xgboost" and HAS_XGBOOST:
            clf = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.08, eval_metric="logloss", random_state=seed, n_jobs=2)
            clf.fit(X_train_fold, y_train_fold)
            oof_scores[test_mask] = clf.predict_proba(X_te_std)[:, 1]

    assert np.isfinite(oof_scores).all(), f"[FATAL] Non-finite scores for {model_type} seed {seed}!"
    return oof_scores


def main():
    driver_sha256 = compute_sha256(Path(__file__))
    print("=" * 115)
    print("MARSEN CLASSIFIER PRE-SCREEN: BENCHMARKING HEADS ON PERMANENT 1.0x BASE (P-1906)")
    print(f"Driver Script SHA-256: {driver_sha256}")
    print("=" * 115)

    with open(SPLIT_MANIFEST_JSON, "r", encoding="utf-8") as f:
        design_vids = sorted(json.load(f)["_meta"]["train_videos"])

    assert design_vids == EXPECTED_DESIGN_VIDS, f"[FATAL] Unexpected cohort: {design_vids}"
    cached_queries = torch.load(QUERY_DATASET_PTH, map_location="cpu")["extracted_queries"]
    frame_bundle = torch.load(FRAME_TENSORS_PTH, map_location="cpu")
    frame_records = frame_bundle["frame_records"]

    active_indices = [idx for idx, q in enumerate(cached_queries) if q["is_active"]]
    active_queries = [cached_queries[idx] for idx in active_indices]
    n_active = len(active_queries)
    y_act = np.array([q["label_op_util"] for q in active_queries], dtype=np.int32)
    v_act = np.array([q["video_id"] for q in active_queries], dtype=np.int32)

    # Ingest 1.0x visual features and 15-D metadata
    print("[*] Ingesting 1.0x Native Crop Visual Embeddings & Metadata...")
    multi_fov_bundle = torch.load(CACHED_MULTI_FOV_PTH, map_location="cpu")
    feat_10x_visual = multi_fov_bundle["fov_1.0"].astype(np.float32)

    branch_bundle = torch.load(CACHED_BRANCHES_PTH, map_location="cpu")
    feat_meta_15 = branch_bundle["branches"]["metadata"][:, :15].astype(np.float32)

    X_base_10x = np.concatenate([feat_10x_visual, feat_meta_15], axis=-1)
    assert X_base_10x.shape == (n_active, 143), f"[FATAL] Shape mismatch: {X_base_10x.shape}"
    print(f"[+] Certified 1.0x Base Matrix Assembled: {X_base_10x.shape} (128-D Visual + 15-D Meta).")

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

    models_to_test = [
        ("mlp_64", "MLP (H=64) [Reference]"),
        ("hist_gb", "HistGradientBoosting (Trees)"),
        ("random_forest", "RandomForest (100 Trees)"),
        ("logreg", "LogisticRegression (Linear L2)")
    ]
    if HAS_CATBOOST:
        models_to_test.append(("catboost", "CatBoostClassifier"))
    if HAS_XGBOOST:
        models_to_test.append(("xgboost", "XGBoostClassifier"))

    print(f"[*] Pre-Screen Head Models ({len(models_to_test)} Total): {[m[1] for m in models_to_test]}")

    results_by_model = defaultdict(dict)

    for m_type, m_label in models_to_test:
        print(f"\n[*] Evaluating Classifier: {m_label:<32}")
        print(f"    {'Seed':<6} | {'Pooled AP':<12} | {'ROC-AUC':<10} | {'Exact tau*':<14} | {'Best F1*':<12} | {'Delta F1 vs M0'}")
        print("    " + "-" * 75)

        for s in SEEDS:
            oof_s = train_classifier_lovo(m_type, X_base_10x, y_act, v_act, design_vids, seed=s)
            ap_s = float(average_precision_score(y_act, oof_s))
            auc_s = float(roc_auc_score(y_act, oof_s))

            exact_taus = get_globally_exact_thresholds(oof_s)
            score_map_s = { (q["image_id"], q["query_id"]): float(oof_s[i]) for i, q in enumerate(active_queries) }
            best_tau_s, best_f1_s, best_rec_s, _ = run_detector_sweep_fast(
                score_map_s, frame_records, active_key_set, positive_key_set, label_to_cat_id, exact_taus, base_frame_cache
            )

            results_by_model[m_type][s] = {
                "ap": ap_s, "auc": auc_s, "tau_star": best_tau_s, "f1_star": best_f1_s,
                "delta_f1_m0": best_f1_s - BASE_M0_F1, "best_rec": best_rec_s
            }
            print(f"    Seed {s:<2d} | {ap_s:10.4f}   | {auc_s:8.4f}   | {best_tau_s:12.6f}   | {best_f1_s:10.6f}   | {best_f1_s - BASE_M0_F1:+12.4f}")

    # Summary synthesis
    print("\n" + "=" * 115)
    print("CLASSIFIER PRE-SCREEN SYNTHESIS ON 1.0x BASE (RANKED BY MEAN OOF AP)")
    print("=" * 115)
    print(f"{'Rank':<5} | {'Classifier Model':<32} | {'Mean AP':<10} | {'AP Std':<10} | {'Seed 41 AP':<12} | {'Seed 42 AP':<12} | {'Seed 43 AP':<12} | {'Mean F1*':<10} | {'Mean dF1':<10}")
    print("-" * 128)

    summary = []
    for m_type, m_label in models_to_test:
        aps = [results_by_model[m_type][s]["ap"] for s in SEEDS]
        f1s = [results_by_model[m_type][s]["f1_star"] for s in SEEDS]
        df1s = [results_by_model[m_type][s]["delta_f1_m0"] for s in SEEDS]
        aucs = [results_by_model[m_type][s]["auc"] for s in SEEDS]

        summary.append({
            "model_type": m_type, "label": m_label,
            "mean_ap": float(np.mean(aps)), "std_ap": float(np.std(aps)),
            "aps": aps, "mean_f1": float(np.mean(f1s)), "mean_df1": float(np.mean(df1s)),
            "mean_auc": float(np.mean(aucs))
        })

    summary.sort(key=lambda x: x["mean_ap"], reverse=True)

    for rank, item in enumerate(summary, 1):
        star = " <-- TOP CLASSIFIER" if rank == 1 else ""
        print(f"#{rank:<3d} | {item['label']:<32} | {item['mean_ap']:8.4f}   | {item['std_ap']:8.4f}   | {item['aps'][0]:10.4f}   | {item['aps'][1]:10.4f}   | {item['aps'][2]:10.4f}   | {item['mean_f1']:8.6f} | {item['mean_df1']:+8.4f}   {star}")

    top_m = summary[0]
    mlp_ref = [item for item in summary if item["model_type"] == "mlp_64"][0]
    ap_gain = top_m["mean_ap"] - mlp_ref["mean_ap"]

    print("\n" + "=" * 115)
    print("DIAGNOSTIC VERDICT")
    print("=" * 115)
    print(f"  - MLP (H=64) Reference: Mean AP = {mlp_ref['mean_ap']:.4f}, Mean F1* = {mlp_ref['mean_f1']:.6f}")
    print(f"  - Top Classifier Head:  {top_m['label']} (Mean AP = {top_m['mean_ap']:.4f}, Mean F1* = {top_m['mean_f1']:.6f})")
    print(f"  - Delta AP over MLP:    {ap_gain:+.4f}")
    if ap_gain > 0.010:
        print("  - Insight: Non-neural/Tree heads uncover substantial unexploited tabular signal over MLP!")
    else:
        print("  - Insight: Classifiers cluster tightly (<= 0.010 AP delta). Bottleneck is largely upstream representation, not MLP capacity.")
    print("=" * 115)

    save_payload = {
        "_meta": {
            "artifact_name": "marsen_classifier_prescreen_results_v1",
            "driver_sha256": driver_sha256,
            "permanent_base": "FOV 1.0x (143-D)",
            "top_classifier": top_m["label"],
            "top_mean_ap": float(top_m["mean_ap"])
        },
        "ranked_summary": summary,
        "seed_runs": {m_type: {str(s): results_by_model[m_type][s] for s in SEEDS} for m_type, _ in models_to_test}
    }
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(save_payload, f, indent=2)

    print(f"[+] Results Frozen: {RESULTS_JSON.resolve()} (SHA-256: {compute_sha256(RESULTS_JSON)})")


if __name__ == "__main__":
    main()