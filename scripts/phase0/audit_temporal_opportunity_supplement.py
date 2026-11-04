"""
D:/MARIVISION/H0_workbench/eval_tools/audit_temporal_opportunity_supplement.py
Phase-II Proof-of-Launch Stage 0 (PoL-0): Part 2 - Stage B Supplementary Analysis

Strict Compliance Checklist:
  - D-3241 / P-1515: Full paired R_CA(1..5) vs R_AGN(1..5) curves and per-depth gaps for each category.
  - D-3242: Dynamic recomputation of overall R_CA(k), R_AGN(k), and gaps from data.
  - D-3243: Accurate score distribution terminology (State-level vs. Target-level max).
  - D-3244: Provenance gates checking link train JSON hash and exact image-ID set equality.
  - D-3245 / P-1516: Opportunity lost vs. raw: Delta_lost(tau) = R5(0) - R5(tau).
  - P-1517: Renamed to audit_temporal_opportunity_supplement.py (pure read-only).
"""

import os
import sys
import json
import hashlib
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np
from pycocotools.coco import COCO

TRAIN_JSON = Path("D:/MARIVISION/seaclips-train.json")
OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
CACHE_FILE = OUT_DIR / "cache_m0_train_raw_top100.json"
LINKS_JSON = OUT_DIR / "pol0_pseudo_links_v1.json"

EXPECTED_LINKS_SHA256 = "e132f0a5782e0a53da858db80e64db0b879652057eeea9c149af2735cb25a2fb"
EXPECTED_M0_CHECKPOINT_SHA256 = "28e6288c3f6769c2d5275aa77ddb248735b55a09cd46a6b570e3f117890b268b"

EVAL_IOU_THRESHOLD = 0.30
MAX_HISTORY_FRAMES = 5
SENSITIVITY_THRESHOLDS = [0.00, 0.05, 0.10, 0.20, 0.50]


def compute_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def box_iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[0] + b1[2], b2[0] + b2[2])
    y2 = min(b1[1] + b1[3], b2[1] + b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = b1[2] * b1[3] + b2[2] * b2[3] - inter
    return inter / union if union > 0.0 else 0.0


def main():
    print("=" * 115)
    print("PROOF-OF-LAUNCH STAGE 0 (PoL-0): PART 2 [STAGE B SUPPLEMENT] - CATEGORY & SCORE SENSITIVITY AUDIT")
    print(f"Target Link Map:  {LINKS_JSON.resolve()}")
    print(f"Target Cache:     {CACHE_FILE.resolve()}")
    print("=" * 115)

    assert TRAIN_JSON.exists(), f"[FATAL] Missing {TRAIN_JSON}"
    assert CACHE_FILE.exists(), f"[FATAL] Missing {CACHE_FILE}"
    assert LINKS_JSON.exists(), f"[FATAL] Missing {LINKS_JSON}"

    train_json_hash = compute_sha256(TRAIN_JSON)
    links_json_hash = compute_sha256(LINKS_JSON)

    # --------------------------------------------------------------------------
    # STEP 1: PROVENANCE GATES (D-3244)
    # --------------------------------------------------------------------------
    assert links_json_hash == EXPECTED_LINKS_SHA256, (
        f"[FATAL] Links JSON SHA-256 mismatch!\n"
        f"  Expected: {EXPECTED_LINKS_SHA256}\n"
        f"  Found:    {links_json_hash}"
    )

    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache_data = json.load(f)
    cache_meta = cache_data.get("_meta", {})
    assert cache_meta.get("train_json_sha256") == train_json_hash, "[FATAL] Prediction cache train JSON mismatch!"
    assert cache_meta.get("m0_checkpoint_sha256") == EXPECTED_M0_CHECKPOINT_SHA256, "[FATAL] M0 Checkpoint SHA-256 mismatch!"

    raw_detections = {int(k): v for k, v in cache_data["detections"].items()}
    assert len(raw_detections) == 20310, f"[FATAL] Expected 20,310 cached images, found {len(raw_detections)}"

    with open(LINKS_JSON, "r", encoding="utf-8") as f:
        links_data = json.load(f)
    links_meta = links_data.get("_meta", {})
    # D-3244: Cross-check link artifact train hash
    assert links_meta.get("train_json_sha256") == train_json_hash, "[FATAL] Link artifact train JSON hash mismatch!"

    frozen_links = links_data["links"]
    assert len(frozen_links) == 65542, f"[FATAL] Expected 65,542 links, found {len(frozen_links)}"

    coco = COCO(str(TRAIN_JSON))
    img_ids = sorted(coco.getImgIds())
    ann_ids = sorted(coco.getAnnIds())
    cats = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    ann_lookup = {a["id"]: a for a in coco.loadAnns(ann_ids)}

    # D-3244: Exact image-id set equality
    assert set(raw_detections.keys()) == set(img_ids), "[FATAL] Prediction cache image ID set does not match dataset!"

    img_meta = {}
    for img_id in img_ids:
        info = coco.loadImgs([img_id])[0]
        img_meta[img_id] = {
            "W": float(info["width"]), "H": float(info["height"]),
            "video_id": int(info["video_id"]), "frame_id": int(info["frame_id"])
        }

    print("[+] All Provenance and Dataset Invariants Certified.")

    # --------------------------------------------------------------------------
    # STEP 2: TRACE BACKWARD CHAINS FOR ALL <4px M0 MISSES
    # --------------------------------------------------------------------------
    print("\n[*] Interrogating M0 prediction cache across backward chains for <4px misses...")

    misses_4px_records = []
    total_4px_gt = 0
    total_4px_miss = 0

    for a in coco.loadAnns(ann_ids):
        aid = a["id"]
        iid_t = a["image_id"]
        meta_t = img_meta[iid_t]
        s_x = 640.0 / meta_t["W"]
        s_y = 640.0 / meta_t["H"]

        bw, bh = a["bbox"][2], a["bbox"][3]
        min_dim_warp = min(bw * s_x, bh * s_y)
        if min_dim_warp >= 4.0:
            continue

        total_4px_gt += 1
        cid = a["category_id"]

        preds_t = raw_detections.get(iid_t, [])
        covered_t_ca = any(p["category_id"] == cid and box_iou(p["bbox"], a["bbox"]) >= EVAL_IOU_THRESHOLD for p in preds_t)
        if covered_t_ca:
            continue

        total_4px_miss += 1

        # Trace backward chain
        chain_ids = []
        curr_aid_str = str(aid)

        for _ in range(MAX_HISTORY_FRAMES):
            link_entry = frozen_links.get(curr_aid_str)
            if link_entry and link_entry["status"] == "LINKED" and link_entry["parent_id"] is not None:
                parent_id = link_entry["parent_id"]
                chain_ids.append(parent_id)
                curr_aid_str = str(parent_id)
            else:
                break

        L = len(chain_ids)

        ca_k = [False] * MAX_HISTORY_FRAMES
        agn_k = [False] * MAX_HISTORY_FRAMES
        best_scores_ca_k = [0.0] * MAX_HISTORY_FRAMES
        best_scores_agn_k = [0.0] * MAX_HISTORY_FRAMES

        for k_idx, parent_id in enumerate(chain_ids):
            parent_ann = ann_lookup[parent_id]
            parent_iid = parent_ann["image_id"]
            p_preds = raw_detections.get(parent_iid, [])
            p_box = parent_ann["bbox"]

            ca_matches = [d for d in p_preds if d["category_id"] == cid and box_iou(d["bbox"], p_box) >= EVAL_IOU_THRESHOLD]
            agn_matches = [d for d in p_preds if box_iou(d["bbox"], p_box) >= EVAL_IOU_THRESHOLD]

            if ca_matches:
                ca_k[k_idx] = True
                best_scores_ca_k[k_idx] = max(d["score"] for d in ca_matches)

            if agn_matches:
                agn_k[k_idx] = True
                best_scores_agn_k[k_idx] = max(d["score"] for d in agn_matches)

        misses_4px_records.append({
            "ann_id": aid, "image_id": iid_t, "category": cats[cid],
            "chain_len": L, "ca_k": ca_k, "agn_k": agn_k,
            "scores_ca_k": best_scores_ca_k, "scores_agn_k": best_scores_agn_k
        })

    assert total_4px_gt == 33751, f"[FATAL] <4px GT mismatch: {total_4px_gt} != 33,751"
    assert total_4px_miss == 12372, f"[FATAL] <4px Miss mismatch: {total_4px_miss} != 12,372"
    sub_all_c5 = [r for r in misses_4px_records if r["chain_len"] == 5]
    n5_tot = len(sub_all_c5)
    assert n5_tot == 5664, f"[FATAL] L=5 cohort size mismatch: {n5_tot} != 5,664"
    print(f"[+] Evaluated all {total_4px_miss:,} <4px misses (L=5 cohort: {n5_tot:,} targets).")

    # --------------------------------------------------------------------------
    # TASK 1: COMPLETE CATEGORY-WISE PAIRED SATURATION CURVES ON L=5 COHORT (D-3241, P-1515)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("TASK 1: COMPLETE PAIRED CA/AGN SATURATION CURVES BY CATEGORY (L = 5 Unbroken Cohort, <4px Misses)")
    print("D-3241: Evaluates R_CA(1..5), R_AGN(1..5), and per-depth gaps for each individual category")
    print("=" * 115)

    cat_sat_export = []

    for cname in sorted(cats.values()):
        sub_all = [r for r in misses_4px_records if r["category"] == cname]
        n_miss = len(sub_all)
        sub_c5 = [r for r in sub_all if r["chain_len"] == 5]
        n5 = len(sub_c5)

        if n5 == 0:
            continue

        rca_k = [sum(1 for r in sub_c5 if any(r["ca_k"][:k])) / n5 * 100.0 for k in range(1, 6)]
        ragn_k = [sum(1 for r in sub_c5 if any(r["agn_k"][:k])) / n5 * 100.0 for k in range(1, 6)]
        gaps_k = [agn - ca for agn, ca in zip(ragn_k, rca_k)]
        d_mem = rca_k[4] - rca_k[0]

        print(f"--- Category: {cname} (Total Misses = {n_miss:,} | L=5 Cohort = {n5:,} [{n5/n_miss*100:4.1f}%]) ---")
        print(f"  {'Metric':<10} | {'k = 1':<9} | {'k = 2':<9} | {'k = 3':<9} | {'k = 4':<9} | {'k = 5':<9} | {'Delta_mem (k5-k1)':<18}")
        print("  " + "-" * 75)
        print(f"  {'R_CA':<10} | {rca_k[0]:5.2f}%   | {rca_k[1]:5.2f}%   | {rca_k[2]:5.2f}%   | {rca_k[3]:5.2f}%   | {rca_k[4]:5.2f}%   | {d_mem:+5.2f} pp")
        print(f"  {'R_AGN':<10} | {ragn_k[0]:5.2f}%   | {ragn_k[1]:5.2f}%   | {ragn_k[2]:5.2f}%   | {ragn_k[3]:5.2f}%   | {ragn_k[4]:5.2f}%   | {ragn_k[4]-ragn_k[0]:+5.2f} pp")
        print(f"  {'Gap (A-C)':<10} | {gaps_k[0]:+5.2f} pp | {gaps_k[1]:+5.2f} pp | {gaps_k[2]:+5.2f} pp | {gaps_k[3]:+5.2f} pp | {gaps_k[4]:+5.2f} pp |")
        print()

        cat_sat_export.append({
            "category": cname, "total_misses": n_miss, "n_cohort5": n5, "retention_pct": n5 / n_miss * 100.0,
            "rca_1": rca_k[0], "rca_2": rca_k[1], "rca_3": rca_k[2], "rca_4": rca_k[3], "rca_5": rca_k[4], "delta_mem_ca": d_mem,
            "ragn_1": ragn_k[0], "ragn_2": ragn_k[1], "ragn_3": ragn_k[2], "ragn_4": ragn_k[3], "ragn_5": ragn_k[4],
            "gap_1": gaps_k[0], "gap_2": gaps_k[1], "gap_3": gaps_k[2], "gap_4": gaps_k[3], "gap_5": gaps_k[4]
        })

    # D-3242: Dynamic recomputation of overall <4px cohort curves
    ov_rca_k = [sum(1 for r in sub_all_c5 if any(r["ca_k"][:k])) / n5_tot * 100.0 for k in range(1, 6)]
    ov_ragn_k = [sum(1 for r in sub_all_c5 if any(r["agn_k"][:k])) / n5_tot * 100.0 for k in range(1, 6)]
    ov_gaps_k = [agn - ca for agn, ca in zip(ov_ragn_k, ov_rca_k)]
    ov_d_mem = ov_rca_k[4] - ov_rca_k[0]

    print("=" * 115)
    print(f"OVERALL <4px COHORT (Total Misses = {total_4px_miss:,} | L=5 Cohort = {n5_tot:,} [{n5_tot/total_4px_miss*100:4.1f}%])")
    print("-" * 115)
    print(f"  {'R_CA':<10} | {ov_rca_k[0]:5.2f}%   | {ov_rca_k[1]:5.2f}%   | {ov_rca_k[2]:5.2f}%   | {ov_rca_k[3]:5.2f}%   | {ov_rca_k[4]:5.2f}%   | {ov_d_mem:+5.2f} pp")
    print(f"  {'R_AGN':<10} | {ov_ragn_k[0]:5.2f}%   | {ov_ragn_k[1]:5.2f}%   | {ov_ragn_k[2]:5.2f}%   | {ov_ragn_k[3]:5.2f}%   | {ov_ragn_k[4]:5.2f}%   | {ov_ragn_k[4]-ov_ragn_k[0]:+5.2f} pp")
    print(f"  {'Gap (A-C)':<10} | {ov_gaps_k[0]:+5.2f} pp | {ov_gaps_k[1]:+5.2f} pp | {ov_gaps_k[2]:+5.2f} pp | {ov_gaps_k[3]:+5.2f} pp | {ov_gaps_k[4]:+5.2f} pp |")

    # --------------------------------------------------------------------------
    # TASK 2: HISTORICAL PROPOSAL SCORE DISTRIBUTION & SENSITIVITY (D-3243, D-3245)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("TASK 2: HISTORICAL PROPOSAL SCORE DISTRIBUTION & THRESHOLD SENSITIVITY")
    print("D-3243: Accurate score statistics; D-3245: Opportunity lost Delta_lost(tau) = R5(0) - R5(tau)")
    print("=" * 115)

    covered_cases = [r for r in sub_all_c5 if any(r["ca_k"])]
    state_covering_scores = []
    target_max_scores = []

    for r in covered_cases:
        scores_for_inst = [s for s, has_cov in zip(r["scores_ca_k"], r["ca_k"]) if has_cov]
        state_covering_scores.extend(scores_for_inst)
        target_max_scores.append(max(scores_for_inst))

    arr_state_scores = np.array(state_covering_scores)
    arr_target_scores = np.array(target_max_scores)

    print(f"Analyzed {len(arr_state_scores):,} covered historical states across {len(arr_target_scores):,} covered targets:")
    print(f"  1. Best Covering Score per Covered Historical State:")
    print(f"     Median = {np.median(arr_state_scores):.4f} | IQR = [{np.percentile(arr_state_scores, 25):.4f}, {np.percentile(arr_state_scores, 75):.4f}] | P90 = {np.percentile(arr_state_scores, 90):.4f}")
    print(f"  2. Maximum Historical Covering Score per Target:")
    print(f"     Median = {np.median(arr_target_scores):.4f} | IQR = [{np.percentile(arr_target_scores, 25):.4f}, {np.percentile(arr_target_scores, 75):.4f}] | P90 = {np.percentile(arr_target_scores, 90):.4f}")

    print("\n--- Cumulative Historical Coverage Opportunity across Descriptive Thresholds (N = 5,664) ---")
    print(f"{'Threshold (tau)':<16} | {'k = 1':<9} | {'k = 2':<9} | {'k = 3':<9} | {'k = 4':<9} | {'k = 5':<9} | {'Delta_mem':<11} | {'Lost vs Raw':<13} | {'Persistent >=2':<15}")
    print("-" * 115)

    raw_r5 = ov_rca_k[4]
    threshold_sensitivity_export = []

    for tau in SENSITIVITY_THRESHOLDS:
        r_tau_k = [
            sum(1 for r in sub_all_c5 if any(r["scores_ca_k"][i] >= tau and r["ca_k"][i] for i in range(k))) / n5_tot * 100.0
            for k in range(1, 6)
        ]
        d_mem_tau = r_tau_k[4] - r_tau_k[0]
        lost_vs_raw = raw_r5 - r_tau_k[4]

        p_ge2_tau = sum(
            1 for r in sub_all_c5
            if sum(1 for i in range(5) if r["scores_ca_k"][i] >= tau and r["ca_k"][i]) >= 2
        ) / n5_tot * 100.0

        label_tau = "Raw (s > 0.00)" if tau == 0.0 else f"score >= {tau:.2f}"
        print(f"{label_tau:<16} | {r_tau_k[0]:5.2f}%   | {r_tau_k[1]:5.2f}%   | {r_tau_k[2]:5.2f}%   | {r_tau_k[3]:5.2f}%   | {r_tau_k[4]:5.2f}%   | {d_mem_tau:+5.2f} pp   | {lost_vs_raw:5.2f} pp{'':<3} | {p_ge2_tau:5.2f}%")

        threshold_sensitivity_export.append({
            "threshold": tau, "r1": r_tau_k[0], "r2": r_tau_k[1], "r3": r_tau_k[2], "r4": r_tau_k[3], "r5": r_tau_k[4],
            "delta_mem": d_mem_tau, "lost_vs_raw": lost_vs_raw, "persistent_ge2": p_ge2_tau
        })

    # --------------------------------------------------------------------------
    # EXPORT SUMMARY RECORDS
    # --------------------------------------------------------------------------
    json_path = OUT_DIR / "pol0_temporal_supplement_summary.json"
    csv_cat_path = OUT_DIR / "pol0_temporal_category_paired_saturation_c5.csv"
    csv_sens_path = OUT_DIR / "pol0_temporal_score_sensitivity_c5.csv"

    with open(csv_cat_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cat_sat_export[0].keys()))
        w.writeheader()
        w.writerows(cat_sat_export)

    with open(csv_sens_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(threshold_sensitivity_export[0].keys()))
        w.writeheader()
        w.writerows(threshold_sensitivity_export)

    payload = {
        "_meta": {
            "audit_version": "PoL-0_Part-2_Causal_Temporal_Supplement_Final",
            "train_json_sha256": train_json_hash,
            "m0_checkpoint_sha256": EXPECTED_M0_CHECKPOINT_SHA256,
            "links_json_sha256": links_json_hash,
            "cohort_size": n5_tot
        },
        "overall_curves": {
            "rca": ov_rca_k, "ragn": ov_ragn_k, "gaps": ov_gaps_k, "delta_mem": ov_d_mem
        },
        "category_paired_saturation_c5": cat_sat_export,
        "score_distribution": {
            "state_covering_scores": {
                "count": len(arr_state_scores),
                "median": float(np.median(arr_state_scores)),
                "iqr_25": float(np.percentile(arr_state_scores, 25)),
                "iqr_75": float(np.percentile(arr_state_scores, 75)),
                "p90": float(np.percentile(arr_state_scores, 90))
            },
            "target_max_scores": {
                "count": len(arr_target_scores),
                "median": float(np.median(arr_target_scores)),
                "iqr_25": float(np.percentile(arr_target_scores, 25)),
                "iqr_75": float(np.percentile(arr_target_scores, 75)),
                "p90": float(np.percentile(arr_target_scores, 90))
            }
        },
        "threshold_sensitivity_c5": threshold_sensitivity_export
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 115)
    print(f"[+] All Supplementary Records successfully exported:")
    print(f"    1. {json_path}")
    print(f"    2. {csv_cat_path}")
    print(f"    3. {csv_sens_path}")
    print("=" * 115)


if __name__ == "__main__":
    main()