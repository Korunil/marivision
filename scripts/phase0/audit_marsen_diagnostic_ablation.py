"""
D:/MARIVISION/H0_workbench/eval_tools/audit_marsen_diagnostic_ablation.py
MARSEN Diagnostic Engine: Reproduction Guard, Calibration Audit, Dual Ablation & Compact Benchmark (v2)

Strict Compliance Checklist:
  - D-3767..D-3773: Diagnostic evaluation on 10 design videos ONLY; calibration/test UNTOUCHED.
  - P-1826: Exact reproduction guard at tau=0.70 asserting TP=201, FP=312, FN=212, and best_tau*=0.70.
  - P-1827: Strict OOF integrity: asserts 1,915 finite scores, key membership in score_map, no silent zeros.
  - P-1824: Fold calibration audit reporting positive/negative score distributions, percentiles, and video AP.
  - P-1823: Dual ablation readout reporting fixed tau=0.70 F1, ablated AP/AUC, and re-swept optimal F1.
  - P-1825 / P-1828: MARSEN-Compact benchmark evaluating all 8 branches via 848-D oversampled MLP (27,201 params).
  - High Speed: Operates entirely on cached evidence branches; executes in under 20 seconds on RTX 3060.
"""

import os
import sys
import json
import hashlib
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
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
RESULTS_JSON = OUT_DIR / "marsen_diagnostic_ablation_results.json"

SCORE_THRESHOLD = 0.20
EVAL_IOU_THRESHOLD = 0.30
BASE_M0_F1 = 0.431718
CANDIDATE_4M_F1 = 0.439957
MARSEN_V12_F1 = 0.434125
MARSEN_V12_AP = 0.0794
TAU_GRID = [0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


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


class MARSENFusionSpecialist(nn.Module):
    def __init__(self, d_model=64, n_heads=4, d_ff=128):
        super().__init__()
        self.proj_global     = nn.Linear(256, d_model)
        self.proj_hr         = nn.Linear(128, d_model)
        self.proj_memory     = nn.Linear(320, d_model)
        self.proj_horizon    = nn.Linear(16,  d_model)
        self.proj_scene      = nn.Linear(32,  d_model)
        self.proj_motion     = nn.Linear(32,  d_model)
        self.proj_reflection = nn.Linear(32,  d_model)
        self.proj_metadata   = nn.Linear(32,  d_model)

        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.pool_weights = nn.Linear(d_model, 1)
        self.head_adm = nn.Linear(d_model, 1)

    def forward(self, branch_dict, active_mask=None):
        tokens = [
            self.proj_global(branch_dict["global"]),
            self.proj_hr(branch_dict["hr"]),
            self.proj_memory(branch_dict["memory"]),
            self.proj_horizon(branch_dict["horizon"]),
            self.proj_scene(branch_dict["scene"]),
            self.proj_motion(branch_dict["motion"]),
            self.proj_reflection(branch_dict["reflection"]),
            self.proj_metadata(branch_dict["metadata"])
        ]
        E = torch.stack(tokens, dim=1)

        key_padding_mask = None
        if active_mask is not None:
            drop_mask = (~active_mask.to(E.device)).unsqueeze(0).expand(E.shape[0], -1)
            key_padding_mask = drop_mask

        attn_out, _ = self.attn(E, E, E, key_padding_mask=key_padding_mask)
        E = self.norm1(E + attn_out)
        ffn_out = self.ffn(E)
        E = self.norm2(E + ffn_out)

        raw_pool = self.pool_weights(E)
        if active_mask is not None:
            drop_3d = (~active_mask.to(E.device)).view(1, 8, 1)
            raw_pool = raw_pool.masked_fill(drop_3d, -1e9)

        weights = F.softmax(raw_pool, dim=1)
        e_fused = (weights * E).sum(dim=1)
        return self.head_adm(e_fused)


# ------------------------------------------------------------------------------
# P-1827: OPERATIONAL DETECTOR SWEEP WITH STRICT KEY & VALUE INTEGRITY
# ------------------------------------------------------------------------------

def run_detector_sweep(score_map, frame_records, active_key_set, positive_key_set, label_to_cat_id):
    best_tau = None
    best_f1 = -1.0
    best_rec = None
    sweep_records = {}

    for tau in TAU_GRID:
        tot_tp, tot_fp, tot_fn = 0, 0, 0
        admitted = 0
        retained_pos = 0

        for fr in frame_records:
            n_t = fr["n_temp"]
            if n_t > 0:
                joint_logits = torch.cat([fr["d_logits"], fr["temp_logits"]], dim=0).clone()
                joint_boxes = torch.cat([fr["d_boxes"], fr["temp_boxes"]], dim=0)

                for q_t in range(n_t):
                    key = (fr["image_id"], q_t)
                    # P-1827: Strict key membership assertion: no silent zero suppressions
                    if key in active_key_set:
                        assert key in score_map, f"[FATAL] Missing OOF score for active query {key}"
                        if score_map[key] >= tau:
                            admitted += 1
                            if key in positive_key_set:
                                retained_pos += 1
                        else:
                            joint_logits[300 + q_t, :] = -1e9
                    else:
                        joint_logits[300 + q_t, :] = -1e9
            else:
                joint_logits = fr["d_logits"]
                joint_boxes = fr["d_boxes"]

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
            "f1": f1, "tp": tot_tp, "fp": tot_fp, "fn": tot_fn, "prec": prec, "rec": rec,
            "admitted": admitted, "retained_pos": retained_pos
        }

        if f1 > best_f1 + 1e-12 or (abs(f1 - best_f1) <= 1e-12 and (best_tau is None or tau > best_tau)):
            best_f1 = f1
            best_tau = tau
            best_rec = sweep_records[tau]

    return best_tau, best_f1, best_rec, sweep_records


def main():
    driver_sha256 = compute_sha256(Path(__file__))
    print("=" * 115)
    print("MARSEN DIAGNOSTIC ENGINE: REPRODUCTION GUARD, CALIBRATION AUDIT, DUAL ABLATION & COMPACT BENCHMARK (v2)")
    print(f"Driver Script SHA-256: {driver_sha256}")
    print("=" * 115)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Target: {device} ({torch.cuda.get_device_name(0)})")

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

    # --------------------------------------------------------------------------
    # P-1822 & P-1826: EXACT REPRODUCTION GUARD AT tau = 0.70
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("P-1822 / P-1826: FULL-MODEL REPRODUCTION GUARD (REPLAYING MARSEN-MU v1.2)")
    print("=" * 115)

    models_by_fold = {}
    scalers_by_fold = {}
    unablated_oof_scores = np.full(n_active, np.nan)

    for vid in design_vids:
        test_mask = (v_act == vid)
        train_mask = ~test_mask
        if test_mask.sum() == 0 or len(np.unique(y_act[train_mask])) < 2:
            continue

        b_tr, b_te = {}, {}
        scalers_by_fold[vid] = {}
        for k in branches.keys():
            m_k = branches[k][train_mask].mean(axis=0)
            s_k = branches[k][train_mask].std(axis=0)
            s_k[s_k < 1e-6] = 1.0
            scalers_by_fold[vid][k] = (m_k, s_k)
            b_tr[k] = torch.tensor((branches[k][train_mask] - m_k) / s_k, dtype=torch.float32, device=device)
            b_te[k] = torch.tensor((branches[k][test_mask] - m_k) / s_k, dtype=torch.float32, device=device)

        y_tr_t = torch.tensor(y_act[train_mask], dtype=torch.float32, device=device).unsqueeze(1)
        torch.manual_seed(42 * 1000 + vid)
        fold_model = MARSENFusionSpecialist().to(device)
        optimizer = torch.optim.AdamW(fold_model.parameters(), lr=1e-3, weight_decay=1e-4)
        pos_weight = torch.tensor([(len(y_tr_t) - y_tr_t.sum()) / max(1.0, y_tr_t.sum())], device=device)

        fold_model.train()
        for epoch in range(40):
            optimizer.zero_grad()
            logit_adm = fold_model(b_tr)
            loss = F.binary_cross_entropy_with_logits(logit_adm, y_tr_t, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()

        fold_model.eval()
        with torch.no_grad():
            logit_te = fold_model(b_te)
            unablated_oof_scores[test_mask] = torch.sigmoid(logit_te).cpu().numpy().flatten()

        models_by_fold[vid] = fold_model

    # P-1827: Assert finite OOF scores
    assert np.isfinite(unablated_oof_scores).all(), "[FATAL] Non-finite unablated OOF scores generated!"

    unablated_ap = float(average_precision_score(y_act, unablated_oof_scores))
    unablated_auc = float(roc_auc_score(y_act, unablated_oof_scores))

    score_map_unablated = { (q["image_id"], q["query_id"]): float(unablated_oof_scores[i]) for i, q in enumerate(active_queries) }
    assert len(score_map_unablated) == 1915

    # P-1826: Exact reproduction unpacking and assertions
    best_tau_rep, best_f1_rep, best_rec_rep, sweep_rep = run_detector_sweep(
        score_map_unablated, frame_records, active_key_set, positive_key_set, label_to_cat_id
    )
    rec_tau70 = sweep_rep[0.70]

    print(f"[*] Reconstructed OOF AP:         {unablated_ap:.4f} (Expected: {MARSEN_V12_AP:.4f})")
    print(f"[*] Reconstructed F1 @ tau=0.70: {rec_tau70['f1']:.6f} (Expected: {MARSEN_V12_F1:.6f})")
    print(f"[*] Reconstructed best tau*:      {best_tau_rep:.2f}")

    assert abs(unablated_ap - MARSEN_V12_AP) < 1e-3, f"[FATAL] AP reproduction mismatch: {unablated_ap} != {MARSEN_V12_AP}"
    assert abs(rec_tau70["f1"] - MARSEN_V12_F1) < 1e-4, f"[FATAL] F1@0.70 reproduction mismatch: {rec_tau70['f1']} != {MARSEN_V12_F1}"
    assert rec_tau70["tp"] == 201 and rec_tau70["fp"] == 312 and rec_tau70["fn"] == 212, "[FATAL] TP/FP/FN mismatch at tau=0.70"
    assert abs(best_tau_rep - 0.70) < 1e-12, f"[FATAL] Best tau mismatch: {best_tau_rep} != 0.70"
    print("[+] P-1822 / P-1826: Full MARSEN-mu v1.2 Result Reconstructed with Exact Parity.")

    # --------------------------------------------------------------------------
    # P-1824: FOLD PROBABILITY CALIBRATION & DOMAIN-SHIFT AUDIT
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("P-1824: FOLD PROBABILITY CALIBRATION & DOMAIN-SHIFT AUDIT")
    print("=" * 115)
    print(f"{'Video ID':<12} | {'Pos / Total':<14} | {'Pos Med':<10} | {'Neg Med':<10} | {'P90 Score':<12} | {'P95 Score':<12} | {'Max Score':<12} | {'Adm @ .70':<12} | {'Video AP'}")
    print("-" * 118)

    calibration_audit = {}
    for vid in design_vids:
        mask_v = (v_act == vid)
        y_v = y_act[mask_v]
        sc_v = unablated_oof_scores[mask_v]
        n_tot = len(y_v)
        n_p = int(y_v.sum())

        pos_sc = sc_v[y_v == 1]
        neg_sc = sc_v[y_v == 0]
        pos_med = float(np.median(pos_sc)) if len(pos_sc) > 0 else np.nan
        neg_med = float(np.median(neg_sc)) if len(neg_sc) > 0 else np.nan
        p90 = float(np.percentile(sc_v, 90))
        p95 = float(np.percentile(sc_v, 95))
        max_s = float(np.max(sc_v))
        adm_70 = int(np.sum(sc_v >= 0.70))
        adm_p_70 = int(np.sum((sc_v >= 0.70) & (y_v == 1)))
        ap_v = float(average_precision_score(y_v, sc_v)) if (0 < n_p < n_tot) else np.nan

        calibration_audit[str(vid)] = {
            "n_total": n_tot, "n_pos": n_p, "pos_median": pos_med, "neg_median": neg_med,
            "p90": p90, "p95": p95, "max": max_s, "admitted_at_70": adm_70, "admitted_pos_at_70": adm_p_70, "ap": ap_v
        }

        pos_med_str = f"{pos_med:.4f}" if not np.isnan(pos_med) else "  -   "
        neg_med_str = f"{neg_med:.4f}" if not np.isnan(neg_med) else "  -   "
        ap_str = f"{ap_v:.4f}" if not np.isnan(ap_v) else "  -   "
        adm_str = f"{adm_70} ({adm_p_70})"
        print(f"video_{vid:<6d} | {n_p:2d} / {n_tot:<8d} | {pos_med_str:<10} | {neg_med_str:<10} | {p90:<12.4f} | {p95:<12.4f} | {max_s:<12.4f} | {adm_str:<12} | {ap_str}")

    # --------------------------------------------------------------------------
    # P-1823: DUAL-MODE BRANCH ABLATION READOUT (FIXED TAU=0.70 & RE-SWEPT TAU*)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("P-1823: DUAL-MODE INFERENCE BRANCH ABLATION MATRIX (RANKING vs. OPERATIONAL DECOUPLING)")
    print("=" * 115)
    print(f"{'Ablated Branch':<24} | {'Ablated AP':<12} | {'Delta AP':<10} | {'Fixed F1 (.70)':<16} | {'Delta F1 (.70)':<16} | {'Best tau*':<10} | {'Best F1*':<12} | {'Delta F1 vs M0'}")
    print("-" * 128)

    branch_names = [
        "Global Context", "Native HR Visual", "Temporal Memory", "Horizon Geometry",
        "Scene Prior", "Ego-Motion", "Reflection Reasoning", "Metadata Context"
    ]

    print(f"{'None (Full MARSEN-mu)':<24} | {unablated_ap:10.4f}   | {'REF':<10} | {MARSEN_V12_F1:14.6f}   | {'REF':<16} | {0.70:<10.2f} | {MARSEN_V12_F1:10.6f}   | {MARSEN_V12_F1 - BASE_M0_F1:+12.4f}")

    ablation_readout = {}

    for drop_idx, b_name in enumerate(branch_names):
        active_mask = torch.ones(8, dtype=torch.bool)
        active_mask[drop_idx] = False

        abl_scores = np.full(n_active, np.nan)
        for vid in design_vids:
            test_mask = (v_act == vid)
            if test_mask.sum() == 0:
                continue

            fold_model = models_by_fold[vid]
            b_te_abl = {}
            for k in branches.keys():
                m_k, s_k = scalers_by_fold[vid][k]
                b_te_abl[k] = torch.tensor((branches[k][test_mask] - m_k) / s_k, dtype=torch.float32, device=device)

            with torch.no_grad():
                logit_abl = fold_model(b_te_abl, active_mask=active_mask)
                abl_scores[test_mask] = torch.sigmoid(logit_abl).cpu().numpy().flatten()

        # P-1827: Assert finite ablated scores
        assert np.isfinite(abl_scores).all(), f"[FATAL] Non-finite ablated scores for branch {b_name}!"

        abl_ap = float(average_precision_score(y_act, abl_scores))
        abl_auc = float(roc_auc_score(y_act, abl_scores))
        d_ap = abl_ap - unablated_ap

        abl_score_map = { (q["image_id"], q["query_id"]): float(abl_scores[i]) for i, q in enumerate(active_queries) }
        assert len(abl_score_map) == 1915

        best_tau_abl, best_f1_abl, best_rec_abl, sweep_grid_abl = run_detector_sweep(
            abl_score_map, frame_records, active_key_set, positive_key_set, label_to_cat_id
        )

        f1_at_fixed70 = sweep_grid_abl[0.70]["f1"]
        d_f1_fixed70 = f1_at_fixed70 - MARSEN_V12_F1
        d_f1_m0 = best_f1_abl - BASE_M0_F1

        ablation_readout[b_name] = {
            "ablated_ap": abl_ap, "delta_ap": d_ap, "ablated_auc": abl_auc,
            "fixed_f1_70": float(f1_at_fixed70), "delta_f1_fixed70": float(d_f1_fixed70),
            "re_swept_tau": float(best_tau_abl), "re_swept_best_f1": float(best_f1_abl),
            "delta_f1_vs_m0": float(d_f1_m0)
        }

        print(f"{b_name:<24} | {abl_ap:10.4f}   | {d_ap:+10.4f} | {f1_at_fixed70:14.6f}   | {d_f1_fixed70:+16.4f} | {best_tau_abl:<10.2f} | {best_f1_abl:10.6f}   | {d_f1_m0:+12.4f}")

    # --------------------------------------------------------------------------
    # P-1825 & P-1828: MARSEN-COMPACT BENCHMARK (848-D OVERSAMPLED MLP)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("P-1825 / P-1828: MARSEN-COMPACT BENCHMARK (848-D BALANCED OVERSAMPLED MLP, EXACTLY 27,201 PARAMS)")
    print("=" * 115)

    branch_order = ["global", "hr", "memory", "horizon", "scene", "motion", "reflection", "metadata"]
    X_full_848 = np.concatenate([branches[k] for k in branch_order], axis=-1)
    assert X_full_848.shape == (n_active, 848)

    compact_oof_scores = np.full(n_active, np.nan)
    for vid in design_vids:
        test_mask = (v_act == vid)
        train_mask = ~test_mask

        X_tr, y_tr = X_full_848[train_mask], y_act[train_mask]
        X_te = X_full_848[test_mask]

        m_tr = X_tr.mean(axis=0)
        s_tr = X_tr.std(axis=0)
        s_tr[s_tr < 1e-6] = 1.0

        X_tr_std = (X_tr - m_tr) / s_tr
        X_te_std = (X_te - m_tr) / s_tr

        pos_idx = np.where(y_tr == 1)[0]
        neg_idx = np.where(y_tr == 0)[0]
        rng = np.random.default_rng(42 * 1000 + vid)
        oversampled_pos = rng.choice(pos_idx, size=len(neg_idx), replace=True)
        bal_idx = np.concatenate([neg_idx, oversampled_pos])
        rng.shuffle(bal_idx)

        clf = MLPClassifier(hidden_layer_sizes=(32,), activation="relu", alpha=0.05, max_iter=350, random_state=42)
        clf.fit(X_tr_std[bal_idx], y_tr[bal_idx])
        compact_oof_scores[test_mask] = clf.predict_proba(X_te_std)[:, 1]

    # P-1827: Assert finite compact scores
    assert np.isfinite(compact_oof_scores).all(), "[FATAL] Non-finite compact OOF scores generated!"

    compact_ap = float(average_precision_score(y_act, compact_oof_scores))
    compact_auc = float(roc_auc_score(y_act, compact_oof_scores))
    base_prev_val = float(np.mean(y_act))
    compact_lift = compact_ap / base_prev_val

    print(f"[*] MARSEN-Compact Ranking: Pooled AP = {compact_ap:.4f} (Lift: {compact_lift:.2f}x) | ROC-AUC = {compact_auc:.4f}")

    score_map_compact = { (q["image_id"], q["query_id"]): float(compact_oof_scores[i]) for i, q in enumerate(active_queries) }
    assert len(score_map_compact) == 1915

    best_c_tau, best_c_f1, best_c_rec, compact_sweep_records = run_detector_sweep(
        score_map_compact, frame_records, active_key_set, positive_key_set, label_to_cat_id
    )

    print(f"\n{'Threshold (tau)':<16} | {'Admitted (Active / 1,915)':<26} | {'Retained Op Pos':<18} | {'TP':<6} | {'FP':<6} | {'FN':<6} | {'Precision':<11} | {'Recall':<10} | {'F1':<10} | {'Delta F1 vs D'}")
    print("-" * 140)
    for tau in TAU_GRID:
        rec = compact_sweep_records[tau]
        d_f1 = rec["f1"] - BASE_M0_F1
        print(f"tau = {tau:<10.2f} | {rec['admitted']:6d} / 1,915 active (2,045 tot) | {rec['retained_pos']:2d} / 92 ({rec['retained_pos']/92*100:4.1f}%) | {rec['tp']:4d} | {rec['fp']:4d} | {rec['fn']:4d} | {rec['prec']:9.4f} | {rec['rec']:8.4f} | {rec['f1']:8.4f} | {d_f1:+8.4f}")

    # --------------------------------------------------------------------------
    # DIAGNOSTIC SYNTHESIS & CORE COMPARISON
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("DIAGNOSTIC SYNTHESIS & SYSTEM COMPARISON")
    print("=" * 115)
    print(f"{'System / Model Configuration':<44} | {'Parameters':<12} | {'Pooled AP':<12} | {'tau*':<6} | {'Best F1':<10} | {'Delta F1 vs M0'}")
    print("-" * 105)
    print(f"{'Baseline M0 Detector':<44} | {'-':<12} | {'-':<12} | {'-':<6} | {BASE_M0_F1:8.6f}   | {'+0.0000'}")
    print(f"{'Candidate 4M (Native HR + Metadata)':<44} | {'4,641':<12} | {'0.0880':<12} | {0.50:<6.2f} | {CANDIDATE_4M_F1:8.6f}   | {CANDIDATE_4M_F1 - BASE_M0_F1:+8.4f}")
    print(f"{'MARSEN-mu v1.2 (8-Branch Transformer)':<44} | {'88,386':<12} | {unablated_ap:<12.4f} | {0.70:<6.2f} | {MARSEN_V12_F1:8.6f}   | {MARSEN_V12_F1 - BASE_M0_F1:+8.4f}")
    print(f"{'MARSEN-Compact (8-Branch Oversampled MLP)':<44} | {'27,201':<12} | {compact_ap:<12.4f} | {best_c_tau:<6.2f} | {best_c_f1:8.6f}   | {best_c_f1 - BASE_M0_F1:+8.4f}")

    diag_payload = {
        "_meta": {
            "artifact_name": "marsen_diagnostic_ablation_results_v2",
            "driver_sha256": driver_sha256,
            "baseline_m0_f1": float(BASE_M0_F1),
            "candidate_4m_f1": float(CANDIDATE_4M_F1),
            "marsen_v12_f1": float(MARSEN_V12_F1),
            "marsen_v12_ap": float(unablated_ap),
            "marsen_compact_f1": float(best_c_f1),
            "marsen_compact_ap": float(compact_ap),
            "marsen_compact_delta_f1": float(best_c_f1 - BASE_M0_F1),
            "marsen_compact_params": 27201
        },
        "calibration_audit": calibration_audit,
        "dual_ablation_readout": ablation_readout,
        "marsen_compact_sweep": {str(k): v for k, v in compact_sweep_records.items()}
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(diag_payload, f, indent=2)

    res_hash = compute_sha256(RESULTS_JSON)
    print(f"\n[+] Diagnostic Results Frozen: {RESULTS_JSON.resolve()}")
    print(f"    Manifest SHA-256:         {res_hash}")
    print("=" * 115)


if __name__ == "__main__":
    main()