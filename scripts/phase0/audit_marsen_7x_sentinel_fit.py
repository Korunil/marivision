"""
D:/MARIVISION/H0_workbench/eval_tools/audit_marsen_7x_sentinel_fit.py
MARSEN Phase B2e-A: Complete 7.0x Pooling Sentinel Standalone Fitting (v2, P-1947, P-1949)

Fits all 4 pooling descriptors at 7.0x (GAP, GeM p=2, 3, 4) across Seeds 41, 42, 43.
Freezes all 12 OOF prediction vectors to marsen_7x_oof_predictions_1915.pth.
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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier

OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
TRAIN_JSON = Path("D:/MARIVISION/seaclips-train.json")
SPLIT_MANIFEST_JSON = OUT_DIR / "pol2_candidate2m_micro_split.json"
QUERY_DATASET_PTH = OUT_DIR / "pol3v_design_queries_dataset_v7.pth"
FRAME_TENSORS_PTH = OUT_DIR / "pol3v_design_frames_tensors_v7.pth"
CACHED_BRANCHES_PTH = OUT_DIR / "marsen_micro_authentic_branches_1915.pth"
TARGET_7X_DESCRIPTORS_PTH = OUT_DIR / "marsen_7x_pooled_descriptors_1915.pth"

RESULTS_JSON = OUT_DIR / "marsen_7x_sentinel_standalone_results.json"
OOF_7X_PTH = OUT_DIR / "marsen_7x_oof_predictions_1915.pth"

SCORE_THRESHOLD = 0.20
EVAL_IOU_THRESHOLD = 0.30
BASE_M0_F1 = 0.431718
EXPECTED_DESIGN_VIDS = [4, 7, 16, 19, 25, 29, 30, 32, 35, 47]
SEEDS = [41, 42, 43]
WINNING_HIDDEN = (64,)

POOLING_7X_CONFIGS = [
    ("7.0x_gap",    "7.0x + GAP"),
    ("7.0x_gem_p2", "7.0x + GeM (p=2)"),
    ("7.0x_gem_p3", "7.0x + GeM (p=3)"),
    ("7.0x_gem_p4", "7.0x + GeM (p=4)"),
]


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
    config_key, s, X_scale, y_act, v_act, design_vids, hidden_sizes = task
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

    assert np.isfinite(oof_scores).all(), f"[FATAL] Non-finite scores for {config_key} seed {s}!"
    return config_key, s, oof_scores


def main():
    t_start_total = time.time()
    driver_sha256 = compute_sha256(Path(__file__))
    print("=" * 115)
    print("MARSEN PHASE B2e-A: COMPLETE 7.0x POOLING SENTINEL STANDALONE FITTING (v2, P-1947, P-1949)")
    print(f"Driver Script SHA-256: {driver_sha256}")
    print("=" * 115)

    assert TARGET_7X_DESCRIPTORS_PTH.exists(), "[FATAL] Missing 7.0x descriptors cache!"
    desc_7x_bundle = torch.load(TARGET_7X_DESCRIPTORS_PTH, map_location="cpu")["descriptors"]

    with open(SPLIT_MANIFEST_JSON, "r", encoding="utf-8") as f:
        design_vids = sorted(json.load(f)["_meta"]["train_videos"])
    assert design_vids == EXPECTED_DESIGN_VIDS

    cached_queries = torch.load(QUERY_DATASET_PTH, map_location="cpu")["extracted_queries"]
    frame_bundle = torch.load(FRAME_TENSORS_PTH, map_location="cpu")
    frame_records = frame_bundle["frame_records"]

    active_indices = [idx for idx, q in enumerate(cached_queries) if q["is_active"]]
    active_queries = [cached_queries[idx] for idx in active_indices]
    n_active = len(active_queries)
    y_act = np.array([q["label_op_util"] for q in active_queries], dtype=np.int32)
    v_act = np.array([q["video_id"] for q in active_queries], dtype=np.int32)
    assert n_active == 1915 and int(y_act.sum()) == 92

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

    tasks = []
    for c_key, c_label in POOLING_7X_CONFIGS:
        feat_vis = desc_7x_bundle[c_key]
        X_143 = np.concatenate([feat_vis, feat_meta_15], axis=-1)
        for s in SEEDS:
            tasks.append((c_key, s, X_143, y_act, v_act, design_vids, WINNING_HIDDEN))

    n_workers = int(os.environ.get("MARSEN_WORKERS", 6))
    print(f"[*] Fitting {len(tasks)} LOVO models across {n_workers} CPU workers...")
    t0 = time.time()

    task_results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for res in executor.map(evaluate_lovo_worker, tasks):
            task_results.append(res)
    print(f"[+] All 12 7.0x Models Fitted in {time.time() - t0:.2f}s")

    oof_7x_all = defaultdict(dict)
    for c_key, s, oof in task_results:
        oof_7x_all[c_key][s] = oof

    torch.save({
        "driver_sha256": driver_sha256,
        "oof_preds": {c_key: {s: oof_7x_all[c_key][s] for s in SEEDS} for c_key, _ in POOLING_7X_CONFIGS}
    }, OOF_7X_PTH)
    print(f"[+] P-1947 Certified: All 12 7.0x OOF Vectors Frozen: {OOF_7X_PTH.resolve()}")

    print("\n" + "=" * 115)
    print("P-1947: STANDALONE 7.0x SPATIAL POOLING EVALUATION")
    print("=" * 115)
    print(f"{'7.0x Pooling Operator':<30} | {'Mean AP':<10} | {'AP Std':<10} | {'Seed 41':<10} | {'Seed 42':<10} | {'Seed 43':<10} | {'Mean F1*'}")
    print("-" * 105)

    pooling_7x_summary = []
    for c_key, c_label in POOLING_7X_CONFIGS:
        aps, f1s = [], []
        for s in SEEDS:
            oof = oof_7x_all[c_key][s]
            ap_val = float(average_precision_score(y_act, oof))
            taus = get_globally_exact_thresholds(oof)
            s_map = {(q["image_id"], q["query_id"]): float(oof[i]) for i, q in enumerate(active_queries)}
            _, f1_val, _, _ = run_detector_sweep_fast(s_map, frame_records, active_key_set, positive_key_set, label_to_cat_id, taus, base_frame_cache)
            aps.append(ap_val)
            f1s.append(f1_val)

        mean_ap = float(np.mean(aps))
        mean_f1 = float(np.mean(f1s))
        pooling_7x_summary.append({
            "key": c_key, "label": c_label,
            "mean_ap": mean_ap, "std_ap": float(np.std(aps)),
            "mean_f1": mean_f1, "aps": aps, "f1s": f1s
        })
        print(f"{c_label:<30} | {mean_ap:8.4f}   | {np.std(aps):8.4f}   | {aps[0]:8.4f}   | {aps[1]:8.4f}   | {aps[2]:8.4f}   | {mean_f1:8.6f}")

    pooling_7x_summary.sort(key=lambda x: x["mean_ap"], reverse=True)
    best_7x_op = pooling_7x_summary[0]
    print(f"\n[+] Top Standalone 7.0x Pooling Operator: {best_7x_op['label']} (Mean AP = {best_7x_op['mean_ap']:.4f}, Mean F1* = {best_7x_op['mean_f1']:.6f})")

    save_payload = {
        "_meta": {
            "artifact_name": "marsen_7x_sentinel_standalone_results_v2",
            "driver_sha256": driver_sha256,
            "best_7x_operator": best_7x_op["label"],
            "best_7x_ap": float(best_7x_op["mean_ap"]),
            "total_seconds": float(time.time() - t_start_total)
        },
        "pooling_7x_summary": pooling_7x_summary
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(save_payload, f, indent=2)

    res_hash = compute_sha256(RESULTS_JSON)
    print(f"[+] 7.0x Standalone Results Frozen: {RESULTS_JSON.resolve()} (SHA-256: {res_hash})")
    print(f"    Total Runtime: {time.time() - t_start_total:.2f}s")
    print("=" * 115)


if __name__ == "__main__":
    main()