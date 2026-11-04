"""
D:/MARIVISION/H0_workbench/eval_tools/audit_marsen_runtime_parity.py
MARSEN Forensic Audit: P-1920 (Tensor Diff), P-1924 (Genuine 2x2 Subprocess Matrix), P-1925 (Complete Sweep Parity)

Self-contained: Spawns clean subprocesses with pre-configured thread pools.
"""

import os
import sys

# --------------------------------------------------------------------------
# SUBPROCESS WORKER MODE
# --------------------------------------------------------------------------
if len(sys.argv) > 1 and sys.argv[1] == "--worker":
    import json
    from PIL import Image
    import torch
    import numpy as np
    from pathlib import Path
    from sklearn.metrics import average_precision_score
    from sklearn.neural_network import MLPClassifier

    cache_path = Path(sys.argv[2])
    out_dir = Path("D:/MARIVISION/H0_workbench/eval_tools")
    
    with open(out_dir / "pol2_candidate2m_micro_split.json", "r", encoding="utf-8") as f:
        design_vids = sorted(json.load(f)["_meta"]["train_videos"])
        
    cached_queries = torch.load(out_dir / "pol3v_design_queries_dataset_v7.pth", map_location="cpu")["extracted_queries"]
    active_queries = [q for q in cached_queries if q["is_active"]]
    y_act = np.array([q["label_op_util"] for q in active_queries], dtype=np.int32)
    v_act = np.array([q["video_id"] for q in active_queries], dtype=np.int32)
    
    branch_bundle = torch.load(out_dir / "marsen_micro_authentic_branches_1915.pth", map_location="cpu")
    feat_meta_15 = branch_bundle["branches"]["metadata"][:, :15].astype(np.float32)
    
    c_bundle = torch.load(cache_path, map_location="cpu")
    feat_10 = c_bundle["fov_1.0"].astype(np.float32)
    X_10 = np.concatenate([feat_10, feat_meta_15], axis=-1)
    
    results = {}
    for s in [41, 42, 43]:
        n_samples = len(y_act)
        oof_scores = np.full(n_samples, np.nan)
        for vid in design_vids:
            test_mask = (v_act == vid)
            train_mask = ~test_mask
            X_tr, y_tr = X_10[train_mask], y_act[train_mask]
            X_te = X_10[test_mask]
            
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
            
            clf = MLPClassifier(hidden_layer_sizes=(64,), activation="relu", alpha=0.05, max_iter=350, random_state=s)
            clf.fit(X_tr_std[bal_idx], y_tr[bal_idx])
            oof_scores[test_mask] = clf.predict_proba(X_te_std)[:, 1]
            
        ap = float(average_precision_score(y_act, oof_scores))
        results[s] = {"ap": ap, "oof": oof_scores.tolist()}
        
    print("__JSON_START__" + json.dumps(results) + "__JSON_END__")
    sys.exit(0)

# --------------------------------------------------------------------------
# MAIN ORCHESTRATOR MODE
# --------------------------------------------------------------------------
import json
import time
import subprocess
import torch
import numpy as np
from pathlib import Path

OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
TRAIN_JSON = Path("D:/MARIVISION/seaclips-train.json")
FRAME_TENSORS_PTH = OUT_DIR / "pol3v_design_frames_tensors_v7.pth"
QUERY_DATASET_PTH = OUT_DIR / "pol3v_design_queries_dataset_v7.pth"
OLD_CACHE_PTH = OUT_DIR / "marsen_multi_fov_features_1915.pth"
NEW_CACHE_PTH = OUT_DIR / "marsen_b2b_fov_features_1915.pth"

SEEDS = [41, 42, 43]


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
        if pred_scores[p_idx] >= 0.20:
            p_box = pred_boxes_px[p_idx]
            p_cat = pred_cat_ids[p_idx]
            best_iou = -1.0
            best_g = -1
            for g_idx in range(n_gt):
                if not gt_matched[g_idx] and gt_cats[g_idx] == p_cat:
                    iou_g = box_iou(gt_boxes_px[g_idx], p_box)
                    if iou_g >= 0.30 and iou_g > best_iou:
                        best_iou = iou_g
                        best_g = g_idx
            if best_g >= 0:
                gt_matched[best_g] = True
                tp += 1
            else:
                fp += 1

    fn = n_gt - sum(gt_matched)
    return tp, fp, fn


def run_worker_process(cache_path, blas_threads):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(blas_threads)
    env["MKL_NUM_THREADS"] = str(blas_threads)
    env["OPENBLAS_NUM_THREADS"] = str(blas_threads)
    env["VECLIB_MAXIMUM_THREADS"] = str(blas_threads)
    env["NUMEXPR_NUM_THREADS"] = str(blas_threads)
    
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", str(cache_path.resolve())]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
    raw = proc.stdout
    start_tag = "__JSON_START__"
    end_tag = "__JSON_END__"
    json_str = raw[raw.index(start_tag) + len(start_tag):raw.index(end_tag)]
    return json.loads(json_str)


def main():
    print("=" * 115)
    print("MARSEN FORENSIC REPRODUCIBILITY AUDIT: P-1920 (DIFF), P-1924 (2x2 SUBPROCESS), P-1925 (SWEEP PARITY)")
    print("=" * 115)

    # --------------------------------------------------------------------------
    # PART 1: P-1920 DIRECT TENSOR DIFFERENCE AUDIT
    # --------------------------------------------------------------------------
    print("\n[PART 1: P-1920 DIRECT FEATURE CACHE DIFFERENCE AUDIT]")
    print(f"Comparing: {OLD_CACHE_PTH.name} vs. {NEW_CACHE_PTH.name}")
    old_b = torch.load(OLD_CACHE_PTH, map_location="cpu")
    new_b = torch.load(NEW_CACHE_PTH, map_location="cpu")

    for scale in [1.0, 1.5, 2.0, 3.0]:
        old_arr = old_b[f"fov_{scale}"].astype(np.float32)
        new_arr = new_b[f"fov_{scale}"].astype(np.float32)
        diff = np.abs(old_arr - new_arr)
        print(f"  - Scale {scale:.1f}x: max|diff| = {diff.max():.2e} | mean|diff| = {diff.mean():.2e} | RMSE = {np.sqrt(np.mean((old_arr - new_arr) ** 2)):.2e}")

    # --------------------------------------------------------------------------
    # PART 2: P-1924 GENUINE 2x2 SUBPROCESS MATRIX
    # --------------------------------------------------------------------------
    print("\n[PART 2: P-1924 GENUINE 2x2 SUBPROCESS MATRIX: CACHE × BLAS THREADS]")
    print(f"{'Condition':<30} | {'Mean AP':<10} | {'Seed 41':<10} | {'Seed 42':<10} | {'Seed 43':<10} | {'Delta vs B2 (0.1367)'}")
    print("-" * 90)

    matrix_runs = [
        ("Cond A: Old Cache + BLAS=1", OLD_CACHE_PTH, 1),
        ("Cond B: Old Cache + BLAS=4", OLD_CACHE_PTH, 4),
        ("Cond C: New Cache + BLAS=1", NEW_CACHE_PTH, 1),
        ("Cond D: New Cache + BLAS=4", NEW_CACHE_PTH, 4),
    ]

    saved_oofs = {}
    for cond_name, c_path, n_threads in matrix_runs:
        res = run_worker_process(c_path, n_threads)
        aps = [res[str(s)]["ap"] for s in SEEDS]
        mean_ap = float(np.mean(aps))
        delta = mean_ap - 0.1367
        print(f"{cond_name:<30} | {mean_ap:8.4f}   | {aps[0]:8.4f}   | {aps[1]:8.4f}   | {aps[2]:8.4f}   | {delta:+10.4f}")
        saved_oofs[cond_name] = res["41"]["oof"]

    # --------------------------------------------------------------------------
    # PART 3: P-1925 COMPLETE SWEEP PARITY GATE (ALL 1,916 THRESHOLDS)
    # --------------------------------------------------------------------------
    print("\n[PART 3: P-1925 COMPLETE SWEEP PARITY GATE (SEED 41, ALL UNIQUE THRESHOLDS)]")
    cached_queries = torch.load(QUERY_DATASET_PTH, map_location="cpu")["extracted_queries"]
    active_queries = [q for q in cached_queries if q["is_active"]]
    frame_records = torch.load(FRAME_TENSORS_PTH, map_location="cpu")["frame_records"]
    
    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        coco_full = json.load(f)
    cat_ids = sorted(c["id"] for c in coco_full["categories"])
    label_to_cat_id = {i: cat_id for i, cat_id in enumerate(cat_ids)}

    active_key_set = set((q["image_id"], q["query_id"]) for q in active_queries)
    positive_key_set = set((q["image_id"], q["query_id"]) for q in active_queries if q["label_op_util"] == 1)

    base_frame_cache = {}
    non_temp_tp, non_temp_fp, non_temp_fn = 0, 0, 0
    for fr in frame_records:
        tpb, fpb, fnb = evaluate_operational_top100(
            fr["d_logits"], fr["d_boxes"], fr["gt_boxes_px"], fr["gt_cats"], label_to_cat_id, fr["img_w"], fr["img_h"]
        )
        base_frame_cache[fr["image_id"]] = (tpb, fpb, fnb)
        if fr["n_temp"] == 0:
            non_temp_tp += tpb
            non_temp_fp += fpb
            non_temp_fn += fnb

    sample_oof = np.array(saved_oofs["Cond C: New Cache + BLAS=1"], dtype=np.float64)
    all_thresholds = sorted(list(set(sample_oof.tolist() + [float(np.nextafter(sample_oof.max(), np.inf))])))
    score_map = {(q["image_id"], q["query_id"]): float(sample_oof[i]) for i, q in enumerate(active_queries)}
    temp_frames = [fr for fr in frame_records if fr["n_temp"] > 0]

    print(f"Executing full sweep comparison across all {len(all_thresholds)} thresholds...")
    t0 = time.time()

    # Un-memoized Full Sweep
    best_tau_u, best_f1_u = None, -1.0
    best_rec_u = None
    u_f1s = []
    for tau in all_thresholds:
        tot_tp, tot_fp, tot_fn = non_temp_tp, non_temp_fp, non_temp_fn
        for fr in temp_frames:
            n_t = fr["n_temp"]
            has_adm = any(score_map.get((fr["image_id"], q_t), -1e9) >= tau for q_t in range(n_t))
            if not has_adm:
                tpd, fpd, fnd = base_frame_cache[fr["image_id"]]
                tot_tp += tpd
                tot_fp += fpd
                tot_fn += fnd
                continue
            jl = torch.cat([fr["d_logits"], fr["temp_logits"]], dim=0).clone()
            jb = torch.cat([fr["d_boxes"], fr["temp_boxes"]], dim=0)
            for q_t in range(n_t):
                if score_map.get((fr["image_id"], q_t), -1e9) < tau:
                    jl[300 + q_t, :] = -1e9
            tp, fp, fn = evaluate_operational_top100(jl, jb, fr["gt_boxes_px"], fr["gt_cats"], label_to_cat_id, fr["img_w"], fr["img_h"])
            tot_tp += tp
            tot_fp += fp
            tot_fn += fn
        p = tot_tp / max(1, tot_tp + tot_fp)
        r = tot_tp / max(1, tot_tp + tot_fn)
        f1 = 2 * p * r / max(1e-7, p + r)
        u_f1s.append(f1)
        if f1 > best_f1_u + 1e-12 or (abs(f1 - best_f1_u) <= 1e-12 and (best_tau_u is None or tau > best_tau_u)):
            best_f1_u = f1
            best_tau_u = tau
            best_rec_u = {"tp": tot_tp, "fp": tot_fp, "fn": tot_fn}
    t_unmemo = time.time() - t0

    # Memoized Full Sweep
    t1 = time.time()
    best_tau_m, best_f1_m = None, -1.0
    best_rec_m = None
    m_f1s = []
    f_cache = {}
    for tau in all_thresholds:
        tot_tp, tot_fp, tot_fn = non_temp_tp, non_temp_fp, non_temp_fn
        for fr in temp_frames:
            n_t = fr["n_temp"]
            iid = fr["image_id"]
            flags = tuple(score_map.get((iid, q_t), -1e9) >= tau for q_t in range(n_t))
            if not any(flags):
                tpd, fpd, fnd = base_frame_cache[iid]
                tot_tp += tpd
                tot_fp += fpd
                tot_fn += fnd
                continue
            ckey = (iid, flags)
            if ckey in f_cache:
                tp, fp, fn = f_cache[ckey]
            else:
                jl = torch.cat([fr["d_logits"], fr["temp_logits"]], dim=0).clone()
                jb = torch.cat([fr["d_boxes"], fr["temp_boxes"]], dim=0)
                for q_t in range(n_t):
                    if not flags[q_t]:
                        jl[300 + q_t, :] = -1e9
                tp, fp, fn = evaluate_operational_top100(jl, jb, fr["gt_boxes_px"], fr["gt_cats"], label_to_cat_id, fr["img_w"], fr["img_h"])
                f_cache[ckey] = (tp, fp, fn)
            tot_tp += tp
            tot_fp += fp
            tot_fn += fn
        p = tot_tp / max(1, tot_tp + tot_fp)
        r = tot_tp / max(1, tot_tp + tot_fn)
        f1 = 2 * p * r / max(1e-7, p + r)
        m_f1s.append(f1)
        if f1 > best_f1_m + 1e-12 or (abs(f1 - best_f1_m) <= 1e-12 and (best_tau_m is None or tau > best_tau_m)):
            best_f1_m = f1
            best_tau_m = tau
            best_rec_m = {"tp": tot_tp, "fp": tot_fp, "fn": tot_fn}
    t_memo = time.time() - t1

    max_f1_diff = float(np.max(np.abs(np.array(u_f1s) - np.array(m_f1s))))
    print(f"Timing: Un-memoized = {t_unmemo:.2f}s | Memoized = {t_memo:.2f}s (Speedup: {t_unmemo / max(0.01, t_memo):.1f}x)")
    print(f"Metrics Parity:")
    print(f"  - Max |F1_diff| across all {len(all_thresholds)} thresholds: {max_f1_diff:.2e}")
    print(f"  - Best F1*:   Unmemo = {best_f1_u:.6f} | Memo = {best_f1_m:.6f} | Diff = {abs(best_f1_u - best_f1_m):.2e}")
    print(f"  - Best tau*:  Unmemo = {best_tau_u:.6f} | Memo = {best_tau_m:.6f} | Diff = {abs(best_tau_u - best_tau_m):.2e}")
    print(f"  - Counts:     Unmemo = {best_rec_u} | Memo = {best_rec_m}")

    assert max_f1_diff == 0.0, f"[FATAL] F1 deviation detected across threshold curves: {max_f1_diff}"
    assert abs(best_f1_u - best_f1_m) == 0.0, "[FATAL] Best F1* mismatch!"
    assert abs(best_tau_u - best_tau_m) == 0.0, "[FATAL] Best tau* mismatch!"
    assert best_rec_u == best_rec_m, "[FATAL] Operational match counts mismatch!"
    print("[+] P-1925 Certified: 100% exact bitwise parity confirmed across the entire threshold search space.")
    print("=" * 115)


if __name__ == "__main__":
    main()