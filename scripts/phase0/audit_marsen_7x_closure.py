"""
D:/MARIVISION/H0_workbench/eval_tools/audit_marsen_7x_closure.py
MARSEN Phase B2e-B: Pooling-Aware 7.0x Boundary Closure Audit (v2, P-1948, P-1950..P-1953)

Pure zero-training evaluation (< 5s runtime).
Ingests B2d pooling-aware OOFs (1.0x_GeM4, 3.0x_GeM4, 5.0x_GAP) and all four 7.0x OOFs.
Tests each 7.0x pooler q in:
  - 1.0x_GeM4 + 7.0x_q
  - 5.0x_GAP + 7.0x_q
  - (3.0x_GeM4 + 5.0x_GAP) + 7.0x_q
Preserves seed-level delta vectors against (3.0x_GeM4 + 5.0x_GAP) reference.
"""

import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json
import time
import hashlib
from PIL import Image
import torch
import numpy as np
from pathlib import Path
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
TRAIN_JSON = Path("D:/MARIVISION/seaclips-train.json")
SPLIT_MANIFEST_JSON = OUT_DIR / "pol2_candidate2m_micro_split.json"
QUERY_DATASET_PTH = OUT_DIR / "pol3v_design_queries_dataset_v7.pth"
FRAME_TENSORS_PTH = OUT_DIR / "pol3v_design_frames_tensors_v7.pth"

B2D_OOF_PTH = OUT_DIR / "marsen_b2d_oof_predictions_1915.pth"
OOF_7X_PTH = OUT_DIR / "marsen_7x_oof_predictions_1915.pth"
RESULTS_JSON = OUT_DIR / "marsen_7x_final_closure_results.json"

SCORE_THRESHOLD = 0.20
EVAL_IOU_THRESHOLD = 0.30
BASE_M0_F1 = 0.431718
EXPECTED_DESIGN_VIDS = [4, 7, 16, 19, 25, 29, 30, 32, 35, 47]
SEEDS = [41, 42, 43]

POOLING_7X_KEYS = [
    ("7.0x_gap",    "GAP"),
    ("7.0x_gem_p2", "GeM(p=2)"),
    ("7.0x_gem_p3", "GeM(p=3)"),
    ("7.0x_gem_p4", "GeM(p=4)"),
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


def main():
    t0 = time.time()
    driver_sha256 = compute_sha256(Path(__file__))
    print("=" * 115)
    print("MARSEN PHASE B2e-B: POOLING-AWARE 7.0x CLOSURE & INCREMENTAL AUDIT (v2, P-1948, P-1950..P-1953)")
    print(f"Driver Script SHA-256: {driver_sha256}")
    print("=" * 115)

    assert B2D_OOF_PTH.exists(), f"[FATAL] Missing B2d OOF cache: {B2D_OOF_PTH}!"
    assert OOF_7X_PTH.exists(), f"[FATAL] Missing 7.0x OOF cache: {OOF_7X_PTH}!"

    b2d_oofs = torch.load(B2D_OOF_PTH, map_location="cpu")["oof_preds"]
    oof_10_g4 = b2d_oofs["1.0x_gem_p4"]
    oof_30_g4 = b2d_oofs["3.0x_gem_p4"]
    oof_50_gap = b2d_oofs["5.0x_gap"]

    oof_7x_all = torch.load(OOF_7X_PTH, map_location="cpu")["oof_preds"]

    cached_queries = torch.load(QUERY_DATASET_PTH, map_location="cpu")["extracted_queries"]
    frame_bundle = torch.load(FRAME_TENSORS_PTH, map_location="cpu")
    frame_records = frame_bundle["frame_records"]

    active_queries = [q for q in cached_queries if q["is_active"]]
    n_active = len(active_queries)
    y_act = np.array([q["label_op_util"] for q in active_queries], dtype=np.int32)

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

    def get_rank_norm(arr):
        return rankdata(arr, method="average") / float(len(arr))

    ref_prob_by_seed = [
        float(average_precision_score(y_act, 0.5 * oof_30_g4[s] + 0.5 * oof_50_gap[s]))
        for s in SEEDS
    ]
    mean_ref_prob_ap = float(np.mean(ref_prob_by_seed))
    print(f"[*] Pooling-Aware Reference: (3.0x_GeM4 + 5.0x_GAP) Prob-AP = {mean_ref_prob_ap:.4f} (Seeds: {ref_prob_by_seed[0]:.4f}, {ref_prob_by_seed[1]:.4f}, {ref_prob_by_seed[2]:.4f})")

    print("\n" + "=" * 115)
    print("P-1950: COMPLETE 7.0x POOLING CLOSURE MATRIX (ALL 4 POOLING OPERATORS)")
    print("=" * 115)
    print(f"{'Closure Configuration':<44} | {'Rank AP':<10} | {'Rank F1*':<10} | {'Prob AP':<10} | {'Prob F1*':<10} | {'dAP vs (3G4+5GAP)'}")
    print("-" * 122)

    closure_matrix_records = []

    for k7, label7 in POOLING_7X_KEYS:
        oof_7q = oof_7x_all[k7]

        test_schemes = [
            (f"Pair: 1.0x_G4 + 7.0x_{label7}", 
             lambda s, o7=oof_7q: (0.5 * get_rank_norm(oof_10_g4[s]) + 0.5 * get_rank_norm(o7[s]), 0.5 * oof_10_g4[s] + 0.5 * o7[s]),
             False),
            (f"Pair: 5.0x_GAP + 7.0x_{label7}", 
             lambda s, o7=oof_7q: (0.5 * get_rank_norm(oof_50_gap[s]) + 0.5 * get_rank_norm(o7[s]), 0.5 * oof_50_gap[s] + 0.5 * o7[s]),
             False),
            (f"Context Triple: (3_G4 + 5_GAP) + 7.0x_{label7}", 
             lambda s, o7=oof_7q: ((get_rank_norm(oof_30_g4[s]) + get_rank_norm(oof_50_gap[s]) + get_rank_norm(o7[s])) / 3.0, (oof_30_g4[s] + oof_50_gap[s] + o7[s]) / 3.0),
             True),
        ]

        for s_name, s_builder, is_context_triple in test_schemes:
            r_aps, r_f1s, p_aps, p_f1s = [], [], [], []
            for s in SEEDS:
                r_vec, p_vec = s_builder(s)

                ap_r = float(average_precision_score(y_act, r_vec))
                taus_r = get_globally_exact_thresholds(r_vec)
                s_map_r = {(q["image_id"], q["query_id"]): float(r_vec[i]) for i, q in enumerate(active_queries)}
                _, f1_r, _, _ = run_detector_sweep_fast(s_map_r, frame_records, active_key_set, positive_key_set, label_to_cat_id, taus_r, base_frame_cache)

                ap_p = float(average_precision_score(y_act, p_vec))
                taus_p = get_globally_exact_thresholds(p_vec)
                s_map_p = {(q["image_id"], q["query_id"]): float(p_vec[i]) for i, q in enumerate(active_queries)}
                _, f1_p, _, _ = run_detector_sweep_fast(s_map_p, frame_records, active_key_set, positive_key_set, label_to_cat_id, taus_p, base_frame_cache)

                r_aps.append(ap_r)
                r_f1s.append(f1_r)
                p_aps.append(ap_p)
                p_f1s.append(f1_p)

            m_r_ap = float(np.mean(r_aps))
            m_r_f1 = float(np.mean(r_f1s))
            m_p_ap = float(np.mean(p_aps))
            m_p_f1 = float(np.mean(p_f1s))
            delta_mean = m_p_ap - mean_ref_prob_ap
            seed_deltas = [p_aps[i] - ref_prob_by_seed[i] for i in range(3)]

            closure_matrix_records.append({
                "name": s_name,
                "pooler": label7,
                "is_context_triple": is_context_triple,
                "mean_rank_ap": m_r_ap, "mean_rank_f1": m_r_f1,
                "mean_prob_ap": m_p_ap, "mean_prob_f1": m_p_f1,
                "delta_mean_vs_ref": delta_mean,
                "r_aps": r_aps, "r_f1s": r_f1s,
                "p_aps": p_aps, "p_f1s": p_f1s,
                "prob_delta_by_seed": seed_deltas
            })
            print(f"{s_name:<44} | {m_r_ap:8.4f}   | {m_r_f1:8.6f} | {m_p_ap:8.4f}   | {m_p_f1:8.6f} | {delta_mean:+10.4f}")

    print("\n" + "=" * 115)
    print("P-1952: DUAL CLOSURE REPORTING & SEED-LEVEL LEDGER")
    print("=" * 115)

    best_overall_entry = max(closure_matrix_records, key=lambda x: x["mean_prob_ap"])
    context_triples = [item for item in closure_matrix_records if item["is_context_triple"]]
    best_context_triple = max(context_triples, key=lambda x: x["mean_prob_ap"])

    print("1. Top Overall System Containing 7.0x (Alternative Exploratory Pair):")
    print(f"   - Configuration:     {best_overall_entry['name']}")
    print(f"   - Mean Prob-AP:      {best_overall_entry['mean_prob_ap']:.4f} (Mean F1* = {best_overall_entry['mean_prob_f1']:.6f})")
    print(f"   - Paired Seed APs:   Seed 41: {best_overall_entry['p_aps'][0]:.4f} | Seed 42: {best_overall_entry['p_aps'][1]:.4f} | Seed 43: {best_overall_entry['p_aps'][2]:.4f}")

    print("\n2. Incremental Addition to Context Core: Context Triple vs. (3.0x_GeM4 + 5.0x_GAP):")
    print(f"   - Configuration:     {best_context_triple['name']}")
    print(f"   - Mean Prob-AP:      {best_context_triple['mean_prob_ap']:.4f} (Reference: {mean_ref_prob_ap:.4f}, Delta = {best_context_triple['delta_mean_vs_ref']:+.4f})")
    print(f"   - Mean F1*:          {best_context_triple['mean_prob_f1']:.6f}")
    print(f"   - Paired Seed Deltas:Seed 41: {best_context_triple['prob_delta_by_seed'][0]:+.4f} | Seed 42: {best_context_triple['prob_delta_by_seed'][1]:+.4f} | Seed 43: {best_context_triple['prob_delta_by_seed'][2]:+.4f}")

    print("\n" + "=" * 115)
    print("P-1953: DESCRIPTIVE BOUNDARY CLOSURE STATEMENT (D-3910)")
    print("=" * 115)
    print(f"• Predeclared Boundary Status:  FOV search terminates permanently at 7.0x. No further scale expansion.")
    print(f"• Context Core Comparison:       (3.0x_GeM4 + 5.0x_GAP) = {mean_ref_prob_ap:.4f} vs. Best Triple = {best_context_triple['mean_prob_ap']:.4f} ({best_context_triple['delta_mean_vs_ref']:+.4f}).")
    if best_context_triple["delta_mean_vs_ref"] <= 0.0:
        verdict = "7.0x evaluated across all 4 pooling regimes; adds no incremental signal to the (3.0x_GeM4 + 5.0x_GAP) contextual core and is NOT retained as an active context branch. Tested scale expansion closed."
    else:
        verdict = f"7.0x evaluated across all 4 pooling regimes; yields {best_context_triple['delta_mean_vs_ref']:+.4f} incremental gain over context core. Seed consistency and architecture inclusion subject to final review."
    print(f"• Descriptive Summary:          {verdict}")
    print("=" * 115)

    save_payload = {
        "_meta": {
            "artifact_name": "marsen_7x_final_closure_results_v2",
            "driver_sha256": driver_sha256,
            "ref_30_50_prob_ap": mean_ref_prob_ap,
            "best_overall_system": best_overall_entry["name"],
            "best_context_triple": best_context_triple["name"],
            "best_triple_delta": float(best_context_triple["delta_mean_vs_ref"]),
            "policy": verdict,
            "total_seconds": float(time.time() - t0)
        },
        "closure_matrix": closure_matrix_records
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(save_payload, f, indent=2)

    res_hash = compute_sha256(RESULTS_JSON)
    print(f"[+] Final Closure Results Frozen: {RESULTS_JSON.resolve()} (SHA-256: {res_hash})")
    print(f"    Evaluation Time: {time.time() - t0:.2f}s")
    print("=" * 115)


if __name__ == "__main__":
    main()