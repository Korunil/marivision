"""
D:/MARIVISION/H0_workbench/eval_tools/audit_temporal_structure.py
Phase-II Proof-of-Launch Stage 0 (PoL-0): Part 2 - Stage B: Causal Temporal Opportunity Evaluation

Strict Compliance Checklist:
  - D-3195 / D-3230: Hard-asserts approved frozen SHA-256 for pol0_pseudo_links_v1.json.
  - D-3208: Hard-asserts certified frozen SHA-256 for M0 checkpoint.
  - P-1507: Asserts train JSON hash match between link artifact and dataset.
  - P-1509: Defensive runtime assertions on same-category, same-video, and frame-adjacency continuity.
  - D-3188: Complete image-id set equality matching seaclips-train.json.
  - D-3177: Hard assertion gates on certified M0 error topology (12,843 misses; 12,372 in <4px).
  - D-3174 / D-3209: Full retention matrix (L>=1..5) by size bin and by category for <4px misses (L>=4 included).
  - D-3175: Paired Class-Aware (R_CA) vs. Class-Agnostic (R_AGN) saturation curves on L=5 cohort.
  - D-3168: Explicit L>=1 and L>=2 denominators for population vs. conditional incremental opportunity.
  - D-3176 / D-3190: Continuous apparent-size evolution (delta_d_min median/IQR and 4px boundary crossing).
  - D-3180: Causal "Historical Coverage Opportunity" nomenclature.
  - D-3232: Fixed OUT_DIR path literal using forward slashes.
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

# Certified Frozen Artifact Hashes (D-3195, D-3208, D-3230)
EXPECTED_LINKS_SHA256 = "e132f0a5782e0a53da858db80e64db0b879652057eeea9c149af2735cb25a2fb"
EXPECTED_M0_CHECKPOINT_SHA256 = "28e6288c3f6769c2d5275aa77ddb248735b55a09cd46a6b570e3f117890b268b"

SCORE_OPERATING_THRESHOLD = 0.20
EVAL_IOU_THRESHOLD = 0.30
MAX_HISTORY_FRAMES = 5


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
    print("PROOF-OF-LAUNCH STAGE 0 (PoL-0): PART 2 [STAGE B] - CAUSAL TEMPORAL OPPORTUNITY EVALUATION")
    print(f"Dataset Target:   {TRAIN_JSON.resolve()}")
    print(f"Prediction Cache: {CACHE_FILE.resolve()}")
    print(f"Frozen Links Map: {LINKS_JSON.resolve()}")
    print("=" * 115)

    assert TRAIN_JSON.exists(), f"[FATAL] Missing {TRAIN_JSON}"
    assert CACHE_FILE.exists(), f"[FATAL] Missing {CACHE_FILE}"
    assert LINKS_JSON.exists(), f"[FATAL] Missing {LINKS_JSON}. Execute Stage A first!"

    train_json_hash = compute_sha256(TRAIN_JSON)
    links_json_hash = compute_sha256(LINKS_JSON)

    # --------------------------------------------------------------------------
    # STEP 1: PROVENANCE VERIFICATION & HARD INTEGRITY GATES (D-3195, D-3208)
    # --------------------------------------------------------------------------
    print("\n--- STEP 1: PROVENANCE VERIFICATION & HARD INTEGRITY GATES ---")

    # Assert approved frozen link artifact hash (D-3195, D-3230)
    assert links_json_hash == EXPECTED_LINKS_SHA256, (
        f"[FATAL] Links JSON SHA-256 mismatch!\n"
        f"  Expected: {EXPECTED_LINKS_SHA256}\n"
        f"  Found:    {links_json_hash}"
    )

    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache_data = json.load(f)
    cache_meta = cache_data.get("_meta", {})
    assert cache_meta.get("train_json_sha256") == train_json_hash, "[FATAL] Prediction cache train JSON mismatch!"

    # Assert certified M0 checkpoint hash (D-3208)
    m0_ckpt_hash = cache_meta.get("m0_checkpoint_sha256")
    assert m0_ckpt_hash == EXPECTED_M0_CHECKPOINT_SHA256, (
        f"[FATAL] M0 Checkpoint SHA-256 mismatch!\n"
        f"  Expected: {EXPECTED_M0_CHECKPOINT_SHA256}\n"
        f"  Found:    {m0_ckpt_hash}"
    )

    raw_detections = {int(k): v for k, v in cache_data["detections"].items()}
    assert len(raw_detections) == 20310, f"[FATAL] Expected 20,310 cached images, found {len(raw_detections)}"

    with open(LINKS_JSON, "r", encoding="utf-8") as f:
        links_data = json.load(f)
    links_meta = links_data.get("_meta", {})

    # Cross-check train JSON hash stored in link artifact (P-1507)
    assert links_meta.get("train_json_sha256") == train_json_hash, "[FATAL] Link artifact train JSON hash mismatch!"

    frozen_links = links_data["links"]
    assert len(frozen_links) == 65542, f"[FATAL] Expected 65,542 links, found {len(frozen_links)}"

    coco = COCO(str(TRAIN_JSON))
    img_ids = sorted(coco.getImgIds())
    ann_ids = sorted(coco.getAnnIds())
    cats = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}

    # Strict image ID set equality (D-3188)
    assert set(raw_detections.keys()) == set(img_ids), "[FATAL] Prediction cache image ID set does not match dataset!"
    print(f"[+] All Provenance Invariants Certified:")
    print(f"    - M0 Checkpoint SHA-256: {m0_ckpt_hash} (VERIFIED)")
    print(f"    - Approved Links SHA-256: {links_json_hash} (VERIFIED)")
    print(f"    - Train JSON SHA-256:    {train_json_hash} (VERIFIED)")
    print(f"    - Image ID Set Equality: Exactly {len(img_ids):,} frames verified.")

    # --------------------------------------------------------------------------
    # STEP 2: BUILD BACKWARD CHAINS & QUERY FROZEN M0 CACHE (D-3172, D-3177, P-1509)
    # --------------------------------------------------------------------------
    print("\n--- STEP 2: EVALUATING M0 COVERAGE ALONG FROZEN HISTORICAL CHAINS ---")

    img_meta = {}
    for img_id in img_ids:
        info = coco.loadImgs([img_id])[0]
        img_meta[img_id] = {
            "video_id": int(info["video_id"]),
            "frame_id": int(info["frame_id"]),
            "W": float(info["width"]),
            "H": float(info["height"]),
            "sensor": info.get("camera_sensor", "unknown"),
            "reflections": info.get("water_reflections", "unknown")
        }

    ann_lookup = {a["id"]: a for a in coco.loadAnns(ann_ids)}

    size_bins = ["<4px", "[4,8)px", "[8,16)px", ">=16px"]
    eval_records = []
    size_evolution_by_bin = defaultdict(list)

    size_counts_check = defaultdict(int)
    ca_misses_check = defaultdict(int)

    for a in coco.loadAnns(ann_ids):
        aid = a["id"]
        iid_t = a["image_id"]
        meta_t = img_meta[iid_t]
        s_x = 640.0 / meta_t["W"]
        s_y = 640.0 / meta_t["H"]

        bw, bh = a["bbox"][2], a["bbox"][3]
        min_dim_warp = min(bw * s_x, bh * s_y)
        if min_dim_warp < 4.0: sb = "<4px"
        elif min_dim_warp < 8.0: sb = "[4,8)px"
        elif min_dim_warp < 16.0: sb = "[8,16)px"
        else: sb = ">=16px"

        size_counts_check[sb] += 1
        cid = a["category_id"]

        preds_t = raw_detections.get(iid_t, [])
        covered_t_ca = any(p["category_id"] == cid and box_iou(p["bbox"], a["bbox"]) >= EVAL_IOU_THRESHOLD for p in preds_t)
        covered_t_agn = any(box_iou(p["bbox"], a["bbox"]) >= EVAL_IOU_THRESHOLD for p in preds_t)
        is_miss = not covered_t_ca
        if is_miss:
            ca_misses_check[sb] += 1

        # Trace backward using frozen link dictionary with defensive assertions (P-1509)
        chain_ids = []
        curr_aid_str = str(aid)
        expected_prior_frame = meta_t["frame_id"]

        for _ in range(MAX_HISTORY_FRAMES):
            link_entry = frozen_links.get(curr_aid_str)
            if link_entry and link_entry["status"] == "LINKED" and link_entry["parent_id"] is not None:
                parent_id = link_entry["parent_id"]
                parent_ann = ann_lookup[parent_id]
                parent_meta = img_meta[parent_ann["image_id"]]

                # Defensive assertions (P-1509)
                assert parent_ann["category_id"] == cid, f"[FATAL] Category mismatch in linked parent: {parent_ann['category_id']} != {cid}"
                assert parent_meta["video_id"] == meta_t["video_id"], f"[FATAL] Video mismatch in linked parent: {parent_meta['video_id']} != {meta_t['video_id']}"
                assert parent_meta["frame_id"] == expected_prior_frame - 1, f"[FATAL] Non-consecutive frame link: {parent_meta['frame_id']} != {expected_prior_frame - 1}"

                chain_ids.append(parent_id)
                curr_aid_str = str(parent_id)
                expected_prior_frame = parent_meta["frame_id"]
            else:
                break

        L = len(chain_ids)

        # Interrogate M0 predictions for each linked historical state
        ca_k = [False] * MAX_HISTORY_FRAMES
        agn_k = [False] * MAX_HISTORY_FRAMES
        scores_k = [0.0] * MAX_HISTORY_FRAMES

        for k_idx, parent_id in enumerate(chain_ids):
            parent_ann = ann_lookup[parent_id]
            parent_iid = parent_ann["image_id"]
            p_preds = raw_detections.get(parent_iid, [])
            p_box = parent_ann["bbox"]

            ca_matches = [d for d in p_preds if d["category_id"] == cid and box_iou(d["bbox"], p_box) >= EVAL_IOU_THRESHOLD]
            agn_matches = [d for d in p_preds if box_iou(d["bbox"], p_box) >= EVAL_IOU_THRESHOLD]

            if ca_matches:
                ca_k[k_idx] = True
                scores_k[k_idx] = max(d["score"] for d in ca_matches)
            if agn_matches:
                agn_k[k_idx] = True

        eval_records.append({
            "ann_id": aid, "image_id": iid_t, "category": cats[cid],
            "size_bin": sb, "min_dim_warp": min_dim_warp,
            "is_m0_miss": is_miss, "chain_len": L,
            "ca_k": ca_k, "agn_k": agn_k, "score_t1": scores_k[0] if L >= 1 else 0.0,
            "rescued_t1_ca": ca_k[0] if L >= 1 else False,
            "rescued_t1_agn": agn_k[0] if L >= 1 else False,
            "rescued_incr_ca": any(ca_k[1:L]) if L >= 2 and not ca_k[0] else False,
            "rescued_incr_agn": any(agn_k[1:L]) if L >= 2 and not agn_k[0] else False,
            "persistence_count_ca": sum(1 for c in ca_k[:L] if c),
            "persistence_count_agn": sum(1 for c in agn_k[:L] if c),
            "sensor": meta_t["sensor"], "reflections": meta_t["reflections"]
        })

        # Apparent size evolution for misses with L >= 1 (D-3176, D-3190)
        if is_miss and L >= 1:
            p_ann = ann_lookup[chain_ids[0]]
            p_meta = img_meta[p_ann["image_id"]]
            p_dim = min(p_ann["bbox"][2] * (640.0 / p_meta["W"]), p_ann["bbox"][3] * (640.0 / p_meta["H"]))
            delta_d = p_dim - min_dim_warp

            size_evolution_by_bin[sb].append({
                "dim_t": min_dim_warp, "dim_t1": p_dim,
                "delta_d_min": delta_d,
                "crossed_4px": (min_dim_warp < 4.0 and p_dim >= 4.0),
                "covered_t1": ca_k[0]
            })

    # Hard-assert exact M0 baseline inventory reproduction (D-3177)
    assert size_counts_check["<4px"] == 33751, f"Mismatch <4px GT: {size_counts_check['<4px']}"
    assert size_counts_check["[4,8)px"] == 8130, f"Mismatch [4,8)px GT: {size_counts_check['[4,8)px']}"
    assert size_counts_check["[8,16)px"] == 11982, f"Mismatch [8,16)px GT: {size_counts_check['[8,16)px']}"
    assert size_counts_check[">=16px"] == 11679, f"Mismatch >=16px GT: {size_counts_check['>=16px']}"

    tot_miss = sum(ca_misses_check.values())
    assert tot_miss == 12843, f"Mismatch total M0 misses: {tot_miss} != 12,843"
    assert ca_misses_check["<4px"] == 12372, f"Mismatch <4px misses: {ca_misses_check['<4px']} != 12,372"
    print(f"[+] Consistency Gate Certified: Exactly 12,843 M0 misses verified (12,372 in <4px regime).")

    # --------------------------------------------------------------------------
    # TASK 1: PSEUDO-LINK RETENTION MATRIX (D-3174, D-3209)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("TASK 1: PSEUDO-LINK RETENTION & COHORT SELECTION MATRIX (M0 Misses)")
    print("Quantifies the exact survival base across chain depths L >= 1 to L = 5")
    print("=" * 115)
    print(f"{'Size Bin':<15} | {'Total Misses':<14} | {'L >= 1 (t-1)':<16} | {'L >= 2':<14} | {'L >= 3':<14} | {'L >= 4':<14} | {'L = 5 (5-Frame)':<16}")
    print("-" * 115)

    retention_by_size = {}

    for sb in size_bins:
        sub = [r for r in eval_records if r["size_bin"] == sb and r["is_m0_miss"]]
        n_m = len(sub)
        l1 = sum(1 for r in sub if r["chain_len"] >= 1)
        l2 = sum(1 for r in sub if r["chain_len"] >= 2)
        l3 = sum(1 for r in sub if r["chain_len"] >= 3)
        l4 = sum(1 for r in sub if r["chain_len"] >= 4)
        l5 = sum(1 for r in sub if r["chain_len"] >= 5)

        retention_by_size[sb] = {"misses": n_m, "l1": l1, "l2": l2, "l3": l3, "l4": l4, "l5": l5}
        print(f"{sb:<15} | {n_m:<14,d} | {l1:6,d} ({l1/n_m*100:5.1f}%)   | {l2:6,d} ({l2/n_m*100:5.1f}%) | {l3:6,d} ({l3/n_m*100:5.1f}%) | {l4:6,d} ({l4/n_m*100:5.1f}%) | {l5:6,d} ({l5/n_m*100:5.1f}%)")

    # D-3209: Complete L>=1..5 retention columns for <4px misses by category
    print("\n[+] Pseudo-Link Retention on <4px M0 Misses by Category (Complete L>=1 to L=5):")
    print(f"  {'Category':<18} | {'<4px Misses':<12} | {'L >= 1 (t-1)':<15} | {'L >= 2':<13} | {'L >= 3':<13} | {'L >= 4':<13} | {'L = 5':<13}")
    print("  " + "-" * 110)
    retention_by_cat_4px = {}
    for cname in sorted(cats.values()):
        sub_c = [r for r in eval_records if r["size_bin"] == "<4px" and r["category"] == cname and r["is_m0_miss"]]
        nc_m = len(sub_c)
        if nc_m == 0: continue
        cl1 = sum(1 for r in sub_c if r["chain_len"] >= 1)
        cl2 = sum(1 for r in sub_c if r["chain_len"] >= 2)
        cl3 = sum(1 for r in sub_c if r["chain_len"] >= 3)
        cl4 = sum(1 for r in sub_c if r["chain_len"] >= 4)
        cl5 = sum(1 for r in sub_c if r["chain_len"] >= 5)
        retention_by_cat_4px[cname] = {"misses": nc_m, "l1": cl1, "l2": cl2, "l3": cl3, "l4": cl4, "l5": cl5}
        print(f"  {cname:<18} | {nc_m:<12,d} | {cl1:5,d} ({cl1/nc_m*100:4.1f}%)   | {cl2:5,d} ({cl2/nc_m*100:4.1f}%) | {cl3:5,d} ({cl3/nc_m*100:4.1f}%) | {cl4:5,d} ({cl4/nc_m*100:4.1f}%) | {cl5:5,d} ({cl5/nc_m*100:4.1f}%)")

    # --------------------------------------------------------------------------
    # TASK 2: CAUSAL HISTORICAL OPPORTUNITY (D-3168, D-3180)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("TASK 2: CAUSAL HISTORICAL OPPORTUNITY (Fast Path F_{t-1} vs. Memory M_{t-1})")
    print("D-3168: Explicit L>=1 and L>=2 Denominators separating Population vs. Conditional Incremental Rate")
    print("=" * 115)
    print(f"{'Size Bin':<12} | {'Misses':<8} | {'L >= 1':<8} | {'t-1 Opp (R_CA)':<16} | {'L >= 2':<8} | {'Incr. Pop Rate':<18} | {'Incr. Cond Rate':<18} | {'Total Causal Opp (L>=2)':<24}")
    print("-" * 115)

    opportunity_by_size = {}

    for sb in size_bins:
        sub = [r for r in eval_records if r["size_bin"] == sb and r["is_m0_miss"]]
        n_m = len(sub)
        sub_l1 = [r for r in sub if r["chain_len"] >= 1]
        n_l1 = len(sub_l1)
        n_t1_ca = sum(1 for r in sub_l1 if r["rescued_t1_ca"])
        pct_t1_ca = (n_t1_ca / n_l1 * 100.0) if n_l1 > 0 else 0.0

        sub_l2 = [r for r in sub if r["chain_len"] >= 2]
        n_l2 = len(sub_l2)
        sub_no_t1_l2 = [r for r in sub_l2 if not r["rescued_t1_ca"]]
        n_no_t1_l2 = len(sub_no_t1_l2)

        n_incr_ca = sum(1 for r in sub_no_t1_l2 if r["rescued_incr_ca"])
        pct_incr_pop = (n_incr_ca / n_l2 * 100.0) if n_l2 > 0 else 0.0
        pct_incr_cond = (n_incr_ca / n_no_t1_l2 * 100.0) if n_no_t1_l2 > 0 else 0.0

        n_total_l2 = sum(1 for r in sub_l2 if r["rescued_t1_ca"] or r["rescued_incr_ca"])
        pct_total_l2 = (n_total_l2 / n_l2 * 100.0) if n_l2 > 0 else 0.0

        opportunity_by_size[sb] = {
            "misses": n_m, "l1": n_l1, "n_t1_ca": n_t1_ca, "pct_t1_ca": pct_t1_ca,
            "l2": n_l2, "n_incr_ca": n_incr_ca, "pct_incr_pop": pct_incr_pop, "pct_incr_cond": pct_incr_cond,
            "n_total_l2": n_total_l2, "pct_total_l2": pct_total_l2
        }
        print(f"{sb:<12} | {n_m:<8,d} | {n_l1:<8,d} | {n_t1_ca:5d} ({pct_t1_ca:5.2f}%)   | {n_l2:<8,d} | {n_incr_ca:5d} ({pct_incr_pop:5.2f}%)   | {n_incr_ca:5d} ({pct_incr_cond:5.2f}%)   | {n_total_l2:5d} ({pct_total_l2:5.2f}%)")

    # --------------------------------------------------------------------------
    # TASK 3: PAIRED SATURATION CURVES R_CA(k) & R_AGN(k) ON L=5 COHORT (D-3175)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("TASK 3: PAIRED SATURATION CURVES R_CA(k) vs. R_AGN(k) ON UNBROKEN 5-FRAME COHORT (L = 5)")
    print("Testing whether historical opportunity is primarily spatial localization or semantic classification")
    print("=" * 115)
    print(f"{'Size Bin':<10} | {'Metric':<8} | {'N (L=5)':<8} | {'k = 1':<10} | {'k = 2':<10} | {'k = 3':<10} | {'k = 4':<10} | {'k = 5':<10} | {'Delta_mem (k5-k1)':<18}")
    print("-" * 115)

    saturation_records = []

    for sb in size_bins:
        sub_c5 = [r for r in eval_records if r["size_bin"] == sb and r["is_m0_miss"] and r["chain_len"] == 5]
        n5 = len(sub_c5)
        if n5 == 0: continue

        rca1 = sum(1 for r in sub_c5 if r["ca_k"][0]) / n5 * 100.0
        rca2 = sum(1 for r in sub_c5 if any(r["ca_k"][:2])) / n5 * 100.0
        rca3 = sum(1 for r in sub_c5 if any(r["ca_k"][:3])) / n5 * 100.0
        rca4 = sum(1 for r in sub_c5 if any(r["ca_k"][:4])) / n5 * 100.0
        rca5 = sum(1 for r in sub_c5 if any(r["ca_k"][:5])) / n5 * 100.0
        d_mem_ca = rca5 - rca1

        ragn1 = sum(1 for r in sub_c5 if r["agn_k"][0]) / n5 * 100.0
        ragn2 = sum(1 for r in sub_c5 if any(r["agn_k"][:2])) / n5 * 100.0
        ragn3 = sum(1 for r in sub_c5 if any(r["agn_k"][:3])) / n5 * 100.0
        ragn4 = sum(1 for r in sub_c5 if any(r["agn_k"][:4])) / n5 * 100.0
        ragn5 = sum(1 for r in sub_c5 if any(r["agn_k"][:5])) / n5 * 100.0
        d_mem_agn = ragn5 - ragn1

        print(f"{sb:<10} | {'R_CA':<8} | {n5:<8,d} | {rca1:5.2f}%{'':<4} | {rca2:5.2f}%{'':<4} | {rca3:5.2f}%{'':<4} | {rca4:5.2f}%{'':<4} | {rca5:5.2f}%{'':<4} | {d_mem_ca:+5.2f} pp")
        print(f"{'':<10} | {'R_AGN':<8} | {n5:<8,d} | {ragn1:5.2f}%{'':<4} | {ragn2:5.2f}%{'':<4} | {ragn3:5.2f}%{'':<4} | {ragn4:5.2f}%{'':<4} | {ragn5:5.2f}%{'':<4} | {d_mem_agn:+5.2f} pp")
        print("-" * 115)

        saturation_records.append({
            "size_bin": sb, "n_cohort5": n5,
            "rca1": rca1, "rca2": rca2, "rca3": rca3, "rca4": rca4, "rca5": rca5, "delta_mem_ca": d_mem_ca,
            "ragn1": ragn1, "ragn2": ragn2, "ragn3": ragn3, "ragn4": ragn4, "ragn5": ragn5, "delta_mem_agn": d_mem_agn
        })

    # Persistence metrics on <4px L=5 cohort
    sub_c5_4px = [r for r in eval_records if r["size_bin"] == "<4px" and r["is_m0_miss"] and r["chain_len"] == 5]
    if sub_c5_4px:
        n5_4 = len(sub_c5_4px)
        p_ge2_ca = sum(1 for r in sub_c5_4px if r["persistence_count_ca"] >= 2) / n5_4 * 100.0
        p_ge3_ca = sum(1 for r in sub_c5_4px if r["persistence_count_ca"] >= 3) / n5_4 * 100.0
        print(f"[*] Multi-Frame Persistence on <4px L=5 Cohort (N={n5_4:,}):")
        print(f"    - Covered Class-Aware in >= 2 prior frames: {p_ge2_ca:5.2f}%")
        print(f"    - Covered Class-Aware in >= 3 prior frames: {p_ge3_ca:5.2f}%")

    # --------------------------------------------------------------------------
    # TASK 4: APPARENT SIZE EVOLUTION AT t-1 ACROSS SIZE BINS (D-3176, D-3190)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("TASK 4: TARGET APPARENT SIZE EVOLUTION AT t-1 (Continuous Delta_d_min & 4px Crossing)")
    print("D-3190: Continuous distribution of Delta_d_min = d_{min,t-1} - d_{min,t}")
    print("=" * 115)
    print(f"{'Size Bin':<15} | {'Linked Misses (L>=1)':<24} | {'Median Delta_d_min':<22} | {'IQR Delta_d_min':<22} | {'Crossed >=4px at t-1':<24} | {'Covered at t-1':<16}")
    print("-" * 115)

    size_evol_summary = {}

    for sb in size_bins:
        records = size_evolution_by_bin[sb]
        n_rec = len(records)
        if n_rec == 0: continue

        deltas = np.array([r["delta_d_min"] for r in records])
        med_delta = float(np.median(deltas))
        p25 = float(np.percentile(deltas, 25))
        p75 = float(np.percentile(deltas, 75))
        iqr_str = f"[{p25:+.2f}, {p75:+.2f}] px"

        n_crossed = sum(1 for r in records if r["crossed_4px"])
        pct_crossed = (n_crossed / n_rec * 100.0) if n_rec > 0 else 0.0
        n_cov = sum(1 for r in records if r["covered_t1"])
        pct_cov = (n_cov / n_rec * 100.0) if n_rec > 0 else 0.0

        size_evol_summary[sb] = {
            "n_linked": n_rec, "median_delta_d": med_delta,
            "iqr_25": p25, "iqr_75": p75,
            "n_crossed_4px": n_crossed, "pct_crossed_4px": pct_crossed,
            "n_covered_t1": n_cov, "pct_covered_t1": pct_cov
        }
        crossed_str = f"{n_crossed:5d} ({pct_crossed:5.2f}%)" if sb == "<4px" else "N/A"
        print(f"{sb:<15} | {n_rec:<24,d} | {med_delta:+6.3f} px{'':<11} | {iqr_str:<22} | {crossed_str:<24} | {n_cov:5d} ({pct_cov:5.2f}%)")

    # --------------------------------------------------------------------------
    # STEP 5: EXPORT MACHINE-READABLE RESEARCH RECORDS
    # --------------------------------------------------------------------------
    csv_ret_size_path = OUT_DIR / "pol0_temporal_link_retention_by_size.csv"
    csv_ret_cat_path = OUT_DIR / "pol0_temporal_link_retention_4px_by_cat.csv"
    csv_opp_path = OUT_DIR / "pol0_temporal_opportunity_by_size.csv"
    csv_sat_path = OUT_DIR / "pol0_temporal_cohort5_saturation.csv"
    json_path = OUT_DIR / "pol0_temporal_opportunity_summary.json"

    ret_rows = [{"size_bin": k, **v} for k, v in retention_by_size.items()]
    with open(csv_ret_size_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ret_rows[0].keys()))
        w.writeheader()
        w.writerows(ret_rows)

    cat_rows = [{"category": k, **v} for k, v in retention_by_cat_4px.items()]
    with open(csv_ret_cat_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cat_rows[0].keys()))
        w.writeheader()
        w.writerows(cat_rows)

    opp_rows = [{"size_bin": k, **v} for k, v in opportunity_by_size.items()]
    with open(csv_opp_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(opp_rows[0].keys()))
        w.writeheader()
        w.writerows(opp_rows)

    with open(csv_sat_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(saturation_records[0].keys()))
        w.writeheader()
        w.writerows(saturation_records)

    master_summary = {
        "_meta": {
            "audit_version": "PoL-0_Part-2_Causal_Temporal_Certified_StageB",
            "train_json_sha256": train_json_hash,
            "m0_checkpoint_sha256": m0_ckpt_hash,
            "links_json_sha256": links_json_hash,
            "total_m0_misses": tot_miss,
            "misses_4px": ca_misses_check["<4px"]
        },
        "retention_by_size": retention_by_size,
        "retention_by_category_4px": retention_by_cat_4px,
        "opportunity_by_size": opportunity_by_size,
        "saturation_cohort5": saturation_records,
        "size_evolution": size_evol_summary
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2)

    print("\n" + "=" * 115)
    print(f"[+] All Stage B causal temporal research records successfully exported:")
    print(f"    1. {json_path}")
    print(f"    2. {csv_ret_size_path}")
    print(f"    3. {csv_ret_cat_path}")
    print(f"    4. {csv_opp_path}")
    print(f"    5. {csv_sat_path}")
    print("=" * 115)


if __name__ == "__main__":
    main()