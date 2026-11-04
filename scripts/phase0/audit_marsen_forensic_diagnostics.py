"""
D:/MARIVISION/H0_workbench/eval_tools/audit_marsen_forensic_diagnostics.py
MARSEN Stream B2: Hardened Forensic Failure-Mode Diagnostics (P-2200..P-2218, D-4276..D-4308)

Executes a complete read-only forensic audit across frozen Train shards and Stage-A OOF caches:
  1. Full Provenance & Dual Query-Key Parity Gate (P-2200): Verifies all frozen hashes, all 141 caches/sidecars,
     and asserts canonical_keys == seed_keys and frame_query_keys == canonical_keys.
  2. Hoisted Baseline Accounting (P-2218): Verifies certified baseline metrics (50741/52801/13929) in common setup.
  3. Resumption Guard (P-2217): RESUME_FROM_DIAGNOSTIC = 4 bypasses already-certified Diagnostics 1-3.
  4. Diagnostic 4 - Opportunity Stratification (P-2204, P-2213, D-4308): Retains all 206 design frames (including
     the 2 zero-proposal frames) and 1,915 active queries, evaluating OOF scores on filtered vs unrestricted regimes.
  5. Diagnostic 5 - Multi-Branch Singleton Harm Taxonomy (P-2203, P-2214): Evaluates deduplicated top-250 proposals
     across all 7 streams x 3 seeds, reporting continuous original detector score percentiles.
  6. Diagnostic 6 - Canonical COCOeval mAP@0.3 Probe (P-2206, P-2212): Evaluates official-style mAP@0.3
     restricted strictly to the 20,075 eligible target frames across sparse admission states.

Read-only: No fitting, no threshold mutations, no Validation/Test access.
"""

import sys
import time
import json
import io
import contextlib
import hashlib
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.stdout.reconfigure(line_buffering=True)

# Resumption Control: Skip completed Diagnostics 1-3 and resume directly at Diagnostic 4
RESUME_FROM_DIAGNOSTIC = 4

# Path Constants
ROOT_DIR = Path("D:/MARIVISION")
OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
SHARD_DIR = OUT_DIR / "full_split_shards"
CACHE_DIR = SHARD_DIR / "lovo_cache"

CENSUS_JSON = OUT_DIR / "marsen_b2_production_census_56shards.json"
TRAIN_JSON = ROOT_DIR / "seaclips-train.json"
OOF_MANIFEST_JSON = OUT_DIR / "marsen_b2_lovo_oof_manifest.json"
THRESHOLDS_JSON = OUT_DIR / "marsen_b2_calibrated_thresholds.json"
CERT_FRAMES_PTH = OUT_DIR / "stream_b2_reproduced_frames.pth"
CERT_QUERIES_PTH = OUT_DIR / "stream_b2_reproduced_queries.pth"

SEEDS = [41, 42, 43]
NUM_VIDEOS = 47
EXPECTED_TOTAL_QUERIES = 211712
EXPECTED_TOTAL_WINDOWS = 20075

FROZEN_CENSUS_SHA256 = "a20a734d668a9078bdfe274cb09fa5a47ad9c6a3a9d7cef2d1f99b926623fbc5"
FROZEN_OOF_MANIFEST_SHA256 = "1919112fe4280a5c60aa42e77a653f74b1a71905bee4bcd8552fb296b6916452"
FROZEN_THRESHOLDS_SHA256 = "615d514a0b2070e0338f9d7ba0e6b73e9f25e1b4773599f83b6203a8bf396f17"

CERTIFIED_BASELINE_TP = 50741
CERTIFIED_BASELINE_FP = 52801
CERTIFIED_BASELINE_FN = 13929
CERTIFIED_TOTAL_GT = 64670
CERTIFIED_BASELINE_F1 = 2.0 * CERTIFIED_BASELINE_TP / (2 * CERTIFIED_BASELINE_TP + CERTIFIED_BASELINE_FP + CERTIFIED_BASELINE_FN)

SUPPRESSION_LOGIT_VALUE = -1e9
DESIGN_VIDEOS = {4, 7, 16, 19, 25, 29, 30, 32, 35, 47}


def compute_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def extract_top100_coco_detections(eval_logits, eval_boxes, iid, label_to_cat_id, img_w, img_h):
    probs = eval_logits.sigmoid()
    num_queries, num_classes = probs.shape
    topk_scores, topk_indices = torch.topk(probs.flatten(), min(100, probs.numel()))
    box_indices = topk_indices // num_classes
    class_indices = topk_indices % num_classes

    det_boxes = eval_boxes[box_indices]
    bw = det_boxes[:, 2] * img_w
    bh = det_boxes[:, 3] * img_h
    cx = det_boxes[:, 0] * img_w
    cy = det_boxes[:, 1] * img_h
    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0

    dets = []
    for i in range(len(topk_scores)):
        cid = label_to_cat_id[int(class_indices[i])]
        sc = float(topk_scores[i])
        bbox = [float(x1[i]), float(y1[i]), float(bw[i]), float(bh[i])]
        dets.append({
            "image_id": int(iid),
            "category_id": int(cid),
            "bbox": bbox,
            "score": sc
        })
    return dets


def main():
    t_start = time.time()
    driver_sha256 = compute_sha256(Path(__file__))

    print("=" * 115, flush=True)
    print("MARSEN STREAM B2: HARDENED FORENSIC FAILURE-MODE DIAGNOSTICS (P-2200..P-2218)", flush=True)
    print(f"Driver Script SHA-256: {driver_sha256}", flush=True)
    print(f"Resumption Setting:    RESUME_FROM_DIAGNOSTIC = {RESUME_FROM_DIAGNOSTIC}", flush=True)
    print("=" * 115, flush=True)

    # =========================================================================
    # 1. FULL PROVENANCE & DUAL QUERY-KEY PARITY GATE (P-2200)
    # =========================================================================
    print("[*] Running Cryptographic Provenance Preflight Gate (P-2200)...", flush=True)

    assert CENSUS_JSON.exists(), f"[FATAL] Missing census JSON at {CENSUS_JSON}"
    assert compute_sha256(CENSUS_JSON) == FROZEN_CENSUS_SHA256, "[FATAL] Census SHA mismatch"

    with open(CENSUS_JSON, "r", encoding="utf-8") as f:
        census = json.load(f)
    frozen_shard_ledger = census["shard_sha256_ledger"]

    for vid in range(1, NUM_VIDEOS + 1):
        p = SHARD_DIR / f"marsen_b2_shard_train_vid{vid}.pth"
        assert p.exists(), f"[FATAL] Missing shard {p.name}"
        assert compute_sha256(p) == frozen_shard_ledger[p.name], f"[FATAL] Shard drift on {p.name}"

    assert compute_sha256(TRAIN_JSON) == census["_meta"]["live_provenance"]["train_json_sha256"], "[FATAL] Train JSON drift"
    assert compute_sha256(OOF_MANIFEST_JSON) == FROZEN_OOF_MANIFEST_SHA256, "[FATAL] OOF manifest drift"
    assert compute_sha256(THRESHOLDS_JSON) == FROZEN_THRESHOLDS_SHA256, "[FATAL] Thresholds manifest drift"

    with open(OOF_MANIFEST_JSON, "r", encoding="utf-8") as f:
        oof_manifest = json.load(f)
    fold_artifacts = oof_manifest["fold_artifacts"]

    for s in SEEDS:
        for v in range(1, NUM_VIDEOS + 1):
            key = f"seed{s}_vid{v}"
            assert key in fold_artifacts, f"[FATAL] Missing fold key {key}"
            expected_cache_sha = fold_artifacts[key]["sha256"]
            c_path = CACHE_DIR / f"lovo_pred_seed{s}_vid{v}.pth"
            s_path = c_path.with_suffix(".pth.sha256")
            assert c_path.exists() and s_path.exists(), f"[FATAL] Missing cache/sidecar for {key}"
            assert compute_sha256(c_path) == expected_cache_sha, f"[FATAL] Cache SHA drift on {c_path.name}"
            assert s_path.read_text(encoding="utf-8").strip() == expected_cache_sha, f"[FATAL] Sidecar drift on {s_path.name}"

    print("    [+] Census, all 47 shards, Train JSON, manifests, and all 141 caches + sidecars verified bitwise intact.")

    sys.path.insert(0, str(OUT_DIR))
    import audit_pol3v_nonlinear_separability as nonlin
    import run_marsen_stage_b_threshold_sweep as stage_b_eng

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(TRAIN_JSON))
    cat_ids = sorted(coco_gt.getCatIds())
    label_to_cat_id = {i: c for i, c in enumerate(cat_ids)}

    anns_by_img = defaultdict(list)
    for ann in coco_gt.dataset["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    train_frames = []
    canonical_keys = []
    frame_query_keys = []
    query_to_frame_map = []
    frame_global_indices = []
    all_query_labels = []
    all_query_vids = []
    all_query_orig_scores = []
    q_global = 0

    for vid in range(1, NUM_VIDEOS + 1):
        p = SHARD_DIR / f"marsen_b2_shard_train_vid{vid}.pth"
        shard = torch.load(p, map_location="cpu")
        act_q_by_img = defaultdict(list)
        for q in shard["active_queries"]:
            act_q_by_img[q["image_id"]].append(q)
            canonical_keys.append((int(q["video_id"]), int(q["image_id"]), int(q["query_id"])))
            all_query_labels.append(int(q["label_op_util"]))
            all_query_vids.append(vid)
            all_query_orig_scores.append(float(q.get("max_score", 0.0)))

        for fr in shard.get("frame_records", []):
            iid = fr["image_id"]
            anns = anns_by_img.get(iid, [])
            gt_boxes_px = [ann["bbox"] for ann in anns]
            gt_cats = [ann["category_id"] for ann in anns]

            fr_act_q = act_q_by_img.get(iid, [])
            local_active_indices = [int(q["query_id"]) for q in fr_act_q]
            local_active_labels = [int(q["label_op_util"]) for q in fr_act_q]
            f_idx = len(train_frames)

            this_frame_global = []
            for loc_i, q_lbl in zip(local_active_indices, local_active_labels):
                frame_query_keys.append((vid, iid, loc_i))
                query_to_frame_map.append((f_idx, loc_i, q_lbl))
                this_frame_global.append(q_global)
                q_global += 1

            frame_global_indices.append(this_frame_global)

            train_frames.append({
                "video_id": vid,
                "image_id": iid,
                "img_w": fr["img_w"],
                "img_h": fr["img_h"],
                "n_temp_total": int(fr["n_temp"]),
                "local_active_indices": local_active_indices,
                "local_active_labels": local_active_labels,
                "d_logits": fr["d_logits"],
                "d_boxes": fr["d_boxes"],
                "full_temp_logits": fr["temp_logits"],
                "full_temp_boxes": fr["temp_boxes"],
                "gt_boxes_px": gt_boxes_px,
                "gt_cats": gt_cats
            })

    assert len(canonical_keys) == EXPECTED_TOTAL_QUERIES, "[FATAL] Query count mismatch"
    assert len(train_frames) == EXPECTED_TOTAL_WINDOWS, "[FATAL] Frame count mismatch"
    assert q_global == EXPECTED_TOTAL_QUERIES, "[FATAL] Global query indexing mismatch"
    assert frame_query_keys == canonical_keys, "[FATAL] Frame query keys mismatch with canonical keys (P-2200)"

    global_idx_by_key = {key: i for i, key in enumerate(canonical_keys)}
    eligible_target_img_ids = [fr["image_id"] for fr in train_frames]

    all_oof_scores = {}
    for s in SEEDS:
        all_oof_scores[s] = {"p1": [], "p3": [], "p5": [], "pD": [], "S0": [], "S35": [], "SG": []}
        seed_keys = []
        for vid in range(1, NUM_VIDEOS + 1):
            c_data = torch.load(CACHE_DIR / f"lovo_pred_seed{s}_vid{vid}.pth", map_location="cpu")
            seed_keys.extend(c_data["query_keys"])
            for k in ["p1", "p3", "p5", "pD", "S0", "S35", "SG"]:
                all_oof_scores[s][k].append(c_data[k])
        assert canonical_keys == seed_keys, f"[FATAL] Query key mismatch for Seed {s}"
        for k in all_oof_scores[s]:
            all_oof_scores[s][k] = np.concatenate(all_oof_scores[s][k]).astype(np.float64)

    y_util = np.array(all_query_labels, dtype=np.int32)
    query_vids = np.array(all_query_vids, dtype=np.int32)
    s_temp_det = np.array(all_query_orig_scores, dtype=np.float32)
    n_pos_total = int(np.sum(y_util))

    print(f"    [+] Dual query-key parity confirmed across all 211,712 proposals for Seeds 41, 42, 43.")
    print(f"    [+] Total Utility Positives: {n_pos_total:,} (Prevalence: {n_pos_total/len(y_util)*100:.3f}%)\n", flush=True)

    # Precompute Baseline Frame Cache & COCO Top-100 Detections
    print("[*] Precomputing Baseline Frame Cache & COCO Top-100 Detections...", flush=True)
    base_frame_cache = {}
    base_frame_coco_dets = {}
    for f_idx, fr in enumerate(train_frames):
        tp_b, fp_b, _, _ = nonlin.evaluate_operational_top100(
            fr["d_logits"], fr["d_boxes"], fr["gt_boxes_px"], fr["gt_cats"],
            label_to_cat_id, fr["img_w"], fr["img_h"]
        )
        base_frame_cache[f_idx] = (tp_b, fp_b)
        base_frame_coco_dets[f_idx] = extract_top100_coco_detections(
            fr["d_logits"], fr["d_boxes"], fr["image_id"], label_to_cat_id, fr["img_w"], fr["img_h"]
        )
    print("    [+] Baseline caches precomputed.\n", flush=True)

    # =========================================================================
    # HOISTED BASELINE ACCOUNTING & INVARIANT VERIFICATION (P-2218)
    # =========================================================================
    total_gt = sum(len(fr["gt_cats"]) for fr in train_frames)
    assert total_gt == CERTIFIED_TOTAL_GT, f"[FATAL] Total GT: {total_gt} != {CERTIFIED_TOTAL_GT}"

    base_tp = sum(b[0] for b in base_frame_cache.values())
    base_fp = sum(b[1] for b in base_frame_cache.values())
    base_fn = total_gt - base_tp

    assert base_tp == CERTIFIED_BASELINE_TP, f"[FATAL] Base TP: {base_tp} != {CERTIFIED_BASELINE_TP}"
    assert base_fp == CERTIFIED_BASELINE_FP, f"[FATAL] Base FP: {base_fp} != {CERTIFIED_BASELINE_FP}"
    assert base_fn == CERTIFIED_BASELINE_FN, f"[FATAL] Base FN: {base_fn} != {CERTIFIED_BASELINE_FN}"

    p_base = base_tp / (base_tp + base_fp)
    r_base = base_tp / (base_tp + base_fn)
    f1_base = 2 * p_base * r_base / (p_base + r_base)
    assert abs(f1_base - CERTIFIED_BASELINE_F1) < 1e-7, f"[FATAL] Base F1 mismatch: {f1_base} vs {CERTIFIED_BASELINE_F1}"

    print(f"[*] Certified Baseline Invariants Verified: TP={base_tp}, FP={base_fp}, FN={base_fn}, GT={total_gt}, F1={f1_base:.8f}\n", flush=True)

    k_targets = [10, 25, 50, 100, 250, 500, 1000, 2500]
    score_names = ["p1", "p3", "p5", "pD", "S0", "S35", "SG"]
    display_names = {"p1": "H1 (1x)", "p3": "H3 (3x)", "p5": "H5 (5x)", "pD": "HD (Dec)", "S0": "S0 (1x+5x)", "S35": "S35 (3x+5x)", "SG": "SG (Dual CDF)"}

    # =========================================================================
    # 2. DIAGNOSTIC 1: LABEL-ORACLE ACHIEVABLE REPLAY (D-4281)
    # =========================================================================
    if RESUME_FROM_DIAGNOSTIC <= 1:
        print("=" * 115, flush=True)
        print("DIAGNOSTIC 1: LABEL-ORACLE ACHIEVABLE REPLAY (D-4281)", flush=True)
        print("=" * 115, flush=True)

        oracle_tp, oracle_fp = 0, 0
        t_diag1_0 = time.time()

        for f_idx, fr in enumerate(train_frames):
            admit_indices = [loc_idx for loc_idx, lbl in zip(fr["local_active_indices"], fr["local_active_labels"]) if lbl == 1]
            if len(admit_indices) == 0:
                tp_b, fp_b = base_frame_cache[f_idx]
                oracle_tp += tp_b
                oracle_fp += fp_b
            else:
                adm_mask = np.zeros(fr["n_temp_total"], dtype=bool)
                adm_mask[admit_indices] = True
                eff_temp_logits = fr["full_temp_logits"].clone()
                eff_temp_logits[~adm_mask, :] = SUPPRESSION_LOGIT_VALUE
                eval_logits = torch.cat([fr["d_logits"], eff_temp_logits], dim=0)
                eval_boxes = torch.cat([fr["d_boxes"], fr["full_temp_boxes"]], dim=0)
                tp_o, fp_o, _, _ = nonlin.evaluate_operational_top100(
                    eval_logits, eval_boxes, fr["gt_boxes_px"], fr["gt_cats"],
                    label_to_cat_id, fr["img_w"], fr["img_h"]
                )
                oracle_tp += tp_o
                oracle_fp += fp_o

        oracle_fn = total_gt - oracle_tp
        p_oracle = oracle_tp / (oracle_tp + oracle_fp)
        r_oracle = oracle_tp / (oracle_tp + oracle_fn)
        f1_oracle = 2 * p_oracle * r_oracle / (p_oracle + r_oracle)

        delta_tp = oracle_tp - base_tp
        delta_fp = oracle_fp - base_fp
        delta_f1 = f1_oracle - f1_base

        print(f"[*] Certified Baseline (R0):     TP={base_tp:>5} | FP={base_fp:>5} | FN={base_fn:>5} | Prec={p_base:.6f} | Rec={r_base:.6f} | F1={f1_base:.8f}")
        print(f"[*] Label-Oracle Replay (y=1):   TP={oracle_tp:>5} | FP={oracle_fp:>5} | FN={oracle_fn:>5} | Prec={p_oracle:.6f} | Rec={r_oracle:.6f} | F1={f1_oracle:.8f}")
        print(f"[*] Achievable Headroom Gain:    Delta TP={delta_tp:>+5} | Delta FP={delta_fp:>+5} | Delta F1={delta_f1:>+10.6f}")
        print(f"[*] Replay Completed in {time.time() - t_diag1_0:.2f}s\n", flush=True)
    else:
        print("[*] Diagnostic 1 bypassed (already frozen: Delta TP=+1576, Delta FP=+0, Delta F1=+0.012964).", flush=True)

    # =========================================================================
    # 3. DIAGNOSTIC 2: THRESHOLD-CONSISTENT TOP-K EXTREME-TAIL AUDIT (P-2201)
    # =========================================================================
    if RESUME_FROM_DIAGNOSTIC <= 2:
        print("=" * 115, flush=True)
        print("DIAGNOSTIC 2: THRESHOLD-CONSISTENT TOP-K EXTREME-TAIL AUDIT (ALL 7 SCORES x 3 SEEDS, P-2201)", flush=True)
        print("=" * 115, flush=True)

        for sc_name in score_names:
            print(f"\n--- {display_names[sc_name]} Tail Precision & Utility Recall Across Seeds ---")
            header_k = " | ".join([f"K={k:<4}" for k in k_targets])
            print(f"{'Seed':<8} | {header_k}")
            print("-" * 115)

            for s in SEEDS:
                scores = all_oof_scores[s][sc_name]
                k_strs = []
                for target_k in k_targets:
                    rank_score = np.partition(scores, -target_k)[-target_k]
                    adm_mask = (scores >= rank_score)
                    n_adm = int(np.count_nonzero(adm_mask))
                    n_tp = int(np.sum(y_util[adm_mask]))
                    prec = (n_tp / n_adm) * 100.0 if n_adm > 0 else 0.0
                    rec = (n_tp / 1576.0) * 100.0
                    k_strs.append(f"{prec:>4.1f}% P / {rec:>4.1f}% R ({n_adm})")

                print(f"Seed {s:<3} | " + " | ".join(k_strs))

        print(f"\n[!] Simplified Unit-Cost Reference Hurdle: ~30.2% precision required if every FP costs +1.\n", flush=True)
    else:
        print("[*] Diagnostic 2 bypassed (already frozen across all 7 score streams x 3 seeds).", flush=True)

    # =========================================================================
    # 4. DIAGNOSTIC 3: ALL 12 PURE-HEAD OPERATIONAL SWEEPS (P-2202, P-2209)
    # =========================================================================
    if RESUME_FROM_DIAGNOSTIC <= 3:
        print("=" * 115, flush=True)
        print("DIAGNOSTIC 3: ALL 12 PURE-HEAD OPERATIONAL STAGE-B SWEEPS (TESTING FUSION DILUTION, P-2202, P-2209)", flush=True)
        print("=" * 115, flush=True)
        print(f"{'Head':<10} | {'Seed':<5} | {'tau*':<18} | {'max_oof':<18} | {'tau* == tau_inf':<16} | {'Admitted Q':<11} | {'Delta TP':<9} | {'Delta FP':<9} | {'Delta F1':<12} | {'State'}")
        print("-" * 115, flush=True)

        for h_key, h_disp in [("p1", "H1 (1x)"), ("p3", "H3 (3x)"), ("p5", "H5 (5x)"), ("pD", "HD (Dec)")]:
            for s in SEEDS:
                raw_scores = all_oof_scores[s][h_key]
                max_oof = float(np.max(raw_scores))
                tau_inf = float(np.nextafter(max_oof, np.inf))

                res = stage_b_eng.run_event_driven_sweep_literal(
                    train_frames, raw_scores, nonlin, label_to_cat_id, desc=f"Diag3-{h_disp}-S{s}"
                )

                d_tp = res["tp"] - res["baseline_tp"]
                d_fp = res["fp"] - res["baseline_fp"]
                d_f1 = res["f1"] - res["baseline_f1"]
                is_tau_inf = bool(res["tau_star"] == tau_inf)
                n_admitted = int(np.count_nonzero(raw_scores >= res["tau_star"]))
                state = "CLOSED_GATE" if (n_admitted == 0) else "ACTIVE_GATE"

                print(
                    f"{h_disp:<10} | {s:<5} | {res['tau_star']:<18.12f} | {max_oof:<18.12f} | {str(is_tau_inf):<16} | "
                    f"{n_admitted:>11} | {d_tp:>+9} | {d_fp:>+9} | {d_f1:>+12.2e} | {state}",
                    flush=True
                )
    else:
        print("[*] Diagnostic 3 bypassed (already frozen: 11 closed gates, H5-S43 active with +1 TP / +2 FP).\n", flush=True)

    # =========================================================================
    # 5. DIAGNOSTIC 4: OPPORTUNITY STRATIFICATION & SAME-10-VIDEO ANALYSIS (P-2204, P-2213, D-4308)
    # =========================================================================
    print("=" * 115, flush=True)
    print("DIAGNOSTIC 4: OPPORTUNITY STRATIFICATION & SAME-10-VIDEO REPLAY (P-2204, P-2213, D-4308)", flush=True)
    print("=" * 115, flush=True)

    assert CERT_QUERIES_PTH.exists(), f"[FATAL] Missing {CERT_QUERIES_PTH}"
    assert CERT_FRAMES_PTH.exists(), f"[FATAL] Missing {CERT_FRAMES_PTH}"

    design_frames_raw = torch.load(CERT_FRAMES_PTH, map_location="cpu")["frame_records"]
    design_iids_set = set(int(fr["image_id"]) for fr in design_frames_raw)
    assert len(design_iids_set) == 206, f"[FATAL] Expected 206 design image IDs, got {len(design_iids_set)}"

    design_queries_raw = torch.load(CERT_QUERIES_PTH, map_location="cpu")["extracted_queries"]
    design_keys_set = set((int(q["video_id"]), int(q["image_id"]), int(q["query_id"])) for q in design_queries_raw if q["is_active"])
    assert len(design_keys_set) == 1915, f"[FATAL] Design active count mismatch: {len(design_keys_set)} != 1915"

    unrestricted_10v_frames = [fr for fr in train_frames if fr["video_id"] in DESIGN_VIDEOS]

    # Retain all 206 design frames; map local active queries to design_keys_set (D-4308)
    filtered_frames_data = []
    filtered_keys = []
    for fr in unrestricted_10v_frames:
        if fr["image_id"] in design_iids_set:
            cloned_fr = dict(fr)
            design_locs = [loc_i for loc_i in fr["local_active_indices"] if (fr["video_id"], fr["image_id"], loc_i) in design_keys_set]
            cloned_fr["local_active_indices"] = design_locs
            filtered_frames_data.append(cloned_fr)
            for loc_i in design_locs:
                filtered_keys.append((fr["video_id"], fr["image_id"], loc_i))

    assert len(filtered_frames_data) == 206, f"[FATAL] Expected 206 design frames, got {len(filtered_frames_data)}"
    assert len(filtered_keys) == 1915, f"[FATAL] Expected 1915 design queries, got {len(filtered_keys)}"
    assert set(filtered_keys) == design_keys_set, "[FATAL] Design keys set mismatch!"

    unrestricted_keys = []
    for fr in unrestricted_10v_frames:
        for loc_i in fr["local_active_indices"]:
            unrestricted_keys.append((fr["video_id"], fr["image_id"], loc_i))

    subset_global_indices = [global_idx_by_key[k] for k in unrestricted_keys]
    filtered_global_indices = [global_idx_by_key[k] for k in filtered_keys]

    print("[*] Regime Comparison on the Same 10 Videos:")
    print(f"    - Unrestricted 10-Video Stream: {len(unrestricted_10v_frames):,} frames | {len(unrestricted_keys):,} active queries | 327 utility positives ({327/len(unrestricted_keys)*100:.3f}% prev)")
    print(f"    - Opportunity-Filtered Subset:  {len(filtered_frames_data):,} frames |  {len(filtered_keys):,} active queries |  92 utility positives ({92/len(filtered_keys)*100:.3f}% prev)")
    print(f"    - Filter Concentration: 7.5x enrichment captured ({1915/len(unrestricted_keys)*100:.2f}% queries hold {92/327*100:.2f}% positives).\n")

    for contender in ["S0", "SG"]:
        for s in SEEDS:
            sc_all = all_oof_scores[s][contender]
            scores_filt = sc_all[filtered_global_indices]
            scores_unr = sc_all[subset_global_indices]

            res_filt = stage_b_eng.run_event_driven_sweep_literal(filtered_frames_data, scores_filt, nonlin, label_to_cat_id, desc=f"Filt-{contender}-S{s}")
            d_tp_f = res_filt["tp"] - res_filt["baseline_tp"]
            d_fp_f = res_filt["fp"] - res_filt["baseline_fp"]
            d_f1_f = res_filt["f1"] - res_filt["baseline_f1"]
            n_adm_f = int(np.count_nonzero(scores_filt >= res_filt["tau_star"]))
            state_f = "CLOSED_GATE" if (n_adm_f == 0) else "ACTIVE_GATE"

            res_unr = stage_b_eng.run_event_driven_sweep_literal(unrestricted_10v_frames, scores_unr, nonlin, label_to_cat_id, desc=f"Unr10-{contender}-S{s}")
            d_tp_u = res_unr["tp"] - res_unr["baseline_tp"]
            d_fp_u = res_unr["fp"] - res_unr["baseline_fp"]
            d_f1_u = res_unr["f1"] - res_unr["baseline_f1"]
            n_adm_u = int(np.count_nonzero(scores_unr >= res_unr["tau_star"]))
            state_u = "CLOSED_GATE" if (n_adm_u == 0) else "ACTIVE_GATE"

            print(f"[*] {contender} Seed {s}:")
            print(f"    [Filtered 206 Frames] tau*={res_filt['tau_star']:.8f} | Adm={n_adm_f:>4} | dTP={d_tp_f:>+3} | dFP={d_fp_f:>+3} | dF1={d_f1_f:>+10.4f} | State: {state_f}")
            print(f"    [Unrestricted 10-Vid] tau*={res_unr['tau_star']:.8f} | Adm={n_adm_u:>4} | dTP={d_tp_u:>+3} | dFP={d_fp_u:>+3} | dF1={d_f1_u:>+10.4f} | State: {state_u}")

    # =========================================================================
    # 6. DIAGNOSTIC 5: MULTI-BRANCH SINGLETON HARM TAXONOMY (P-2203, P-2214)
    # =========================================================================
    print("\n" + "=" * 115, flush=True)
    print("DIAGNOSTIC 5: MULTI-BRANCH SINGLETON HARM TAXONOMY & LOGIT INTERACTION (P-2203, P-2214)", flush=True)
    print("=" * 115, flush=True)

    union_candidate_indices = set()
    for s in SEEDS:
        for sc_name in score_names:
            scores = all_oof_scores[s][sc_name]
            top250 = np.argsort(-scores)[:250]
            union_candidate_indices.update(top250)

    union_candidate_indices = sorted(list(union_candidate_indices))
    print(f"[*] Evaluated Deduplicated Union: {len(union_candidate_indices):,} unique top-ranked proposals across all 7 score streams x 3 seeds.")

    harm_class_data = defaultdict(lambda: {"det_scores": []})
    t_diag5_0 = time.time()

    for q_global_idx in union_candidate_indices:
        f_idx, loc_i, q_lbl = query_to_frame_map[q_global_idx]
        fr = train_frames[f_idx]
        tp_b, fp_b = base_frame_cache[f_idx]

        adm_mask = np.zeros(fr["n_temp_total"], dtype=bool)
        adm_mask[loc_i] = True
        eff_temp_logits = fr["full_temp_logits"].clone()
        eff_temp_logits[~adm_mask, :] = SUPPRESSION_LOGIT_VALUE
        eval_logits = torch.cat([fr["d_logits"], eff_temp_logits], dim=0)
        eval_boxes = torch.cat([fr["d_boxes"], fr["full_temp_boxes"]], dim=0)

        tp_s, fp_s, _, _ = nonlin.evaluate_operational_top100(
            eval_logits, eval_boxes, fr["gt_boxes_px"], fr["gt_cats"],
            label_to_cat_id, fr["img_w"], fr["img_h"]
        )

        dtp = tp_s - tp_b
        dfp = fp_s - fp_b

        if dtp > 0 and dfp <= 0:
            category = "Safe Rescue (dTP>0, dFP<=0)"
        elif dtp > 0 and dfp > 0:
            category = "Costly Rescue (dTP>0, dFP>0)"
        elif dtp == 0 and dfp == 0:
            category = "Inert (dTP=0, dFP=0)"
        elif dtp == 0 and dfp > 0:
            category = "Pure FP Harm (dTP=0, dFP>0)"
        elif dtp < 0:
            category = "Destructive Displacement (dTP<0)"
        else:
            category = "Other"

        harm_class_data[category]["det_scores"].append(s_temp_det[q_global_idx])

    print("\n[*] Extreme-Tail Singleton Outcome Breakdown & Original Detector Score Percentiles:")
    print(f"{'Category':<32} | {'Count':<6} | {'%':<5} | {'Original Detector Score (p25 / Med / p75)'}")
    print("-" * 115)

    for cat, data_dict in sorted(harm_class_data.items(), key=lambda x: -len(x[1]["det_scores"])):
        cnt = len(data_dict["det_scores"])
        pct = (cnt / len(union_candidate_indices)) * 100.0
        d_arr = np.array(data_dict["det_scores"])
        d_p25, d_med, d_p75 = np.percentile(d_arr, 25), np.median(d_arr), np.percentile(d_arr, 75)
        d_str = f"{d_p25:.3f} / {d_med:.3f} / {d_p75:.3f}"
        print(f"{cat:<32} | {cnt:>6} | {pct:>4.1f}% | {d_str}")

    print(f"[*] Singleton Harm Taxonomy Completed in {time.time() - t_diag5_0:.2f}s.\n", flush=True)

    # =========================================================================
    # 7. DIAGNOSTIC 6: CANONICAL COCOeval TRAIN-OOF mAP@0.3 PROBE (P-2206, P-2212)
    # =========================================================================
    print("=" * 115, flush=True)
    print("DIAGNOSTIC 6: CANONICAL COCOeval TRAIN-OOF mAP@0.3 PROBE (P-2206, P-2212)", flush=True)
    print("=" * 115, flush=True)

    sparse_n_targets = [0, 1, 10, 50, 100, 250, 500, 1000]
    s0_43 = all_oof_scores[43]["S0"]
    sorted_s0_desc = np.sort(s0_43)[::-1]
    tau_inf_val = float(np.nextafter(np.max(s0_43), np.inf))

    print(f"{'Target N_adm':<13} | {'tau (S0 Seed43)':<18} | {'Actual N_adm':<13} | {'Dirty Frames':<13} | {'Whole-Det F1':<13} | {'COCO mAP@0.3':<13} | {'Delta mAP':<11}")
    print("-" * 115)

    base_map = None

    for target_n in sparse_n_targets:
        if target_n == 0:
            tau_val = tau_inf_val
            act_n = 0
            dirty_frames = set()
        else:
            tau_val = float(sorted_s0_desc[target_n - 1])
            act_n = int(np.count_nonzero(s0_43 >= tau_val))
            admitted_global_indices = np.where(s0_43 >= tau_val)[0]
            dirty_frames = set(query_to_frame_map[idx][0] for idx in admitted_global_indices)

        all_stream_dets = []
        tp_tot, fp_tot = 0, 0

        for f_idx, fr in enumerate(train_frames):
            if f_idx not in dirty_frames:
                tp_b, fp_b = base_frame_cache[f_idx]
                tp_tot += tp_b
                fp_tot += fp_b
                all_stream_dets.extend(base_frame_coco_dets[f_idx])
            else:
                adm_locs = [loc_i for loc_i, g_idx in zip(fr["local_active_indices"], frame_global_indices[f_idx]) if s0_43[g_idx] >= tau_val]
                if len(adm_locs) == 0:
                    tp_b, fp_b = base_frame_cache[f_idx]
                    tp_tot += tp_b
                    fp_tot += fp_b
                    all_stream_dets.extend(base_frame_coco_dets[f_idx])
                else:
                    adm_mask = np.zeros(fr["n_temp_total"], dtype=bool)
                    adm_mask[adm_locs] = True
                    eff_temp_logits = fr["full_temp_logits"].clone()
                    eff_temp_logits[~adm_mask, :] = SUPPRESSION_LOGIT_VALUE
                    eval_logits = torch.cat([fr["d_logits"], eff_temp_logits], dim=0)
                    eval_boxes = torch.cat([fr["d_boxes"], fr["full_temp_boxes"]], dim=0)

                    tp_s, fp_s, _, _ = nonlin.evaluate_operational_top100(
                        eval_logits, eval_boxes, fr["gt_boxes_px"], fr["gt_cats"],
                        label_to_cat_id, fr["img_w"], fr["img_h"]
                    )
                    tp_tot += tp_s
                    fp_tot += fp_s
                    all_stream_dets.extend(extract_top100_coco_detections(
                        eval_logits, eval_boxes, fr["image_id"], label_to_cat_id, fr["img_w"], fr["img_h"]
                    ))

        fn_tot = total_gt - tp_tot
        p_val = tp_tot / (tp_tot + fp_tot) if (tp_tot + fp_tot) > 0 else 0.0
        r_val = tp_tot / (tp_tot + fn_tot) if (tp_tot + fn_tot) > 0 else 0.0
        f1_curr = 2 * p_val * r_val / (p_val + r_val) if (p_val + r_val) > 0 else 0.0

        dets_for_coco = []
        for i, d in enumerate(all_stream_dets):
            dets_for_coco.append({
                "image_id": d["image_id"],
                "category_id": d["category_id"],
                "bbox": d["bbox"],
                "score": d["score"],
                "id": i + 1
            })

        with contextlib.redirect_stdout(io.StringIO()):
            coco_dt = coco_gt.loadRes(dets_for_coco)
            coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
            coco_eval.params.imgIds = eligible_target_img_ids
            coco_eval.params.iouThrs = np.array([0.30])
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            curr_map = float(coco_eval.stats[0])

        if base_map is None:
            base_map = curr_map
        d_map = curr_map - base_map

        print(f"{target_n:<13} | {tau_val:<18.12f} | {act_n:<13} | {len(dirty_frames):<13} | {f1_curr:<13.6f} | {curr_map:<13.6f} | {d_map:>+11.6f}", flush=True)

    print("\n" + "=" * 115, flush=True)
    print(f"[+] Forensic Diagnostics Suite Finished in {time.time() - t_start:.2f}s.")
    print("=" * 115, flush=True)


if __name__ == "__main__":
    main()