"""
D:/MARIVISION/H0_workbench/eval_tools/audit_pol2_candidate2m_preflight.py
PoL-2M Stage 0 Pre-Flight: Holdout-First Split Freeze & Design-Only MI Population Audit

Strict Compliance Checklist:
  - D-3552 / D-3555: Read-only pre-flight evidence gathering. Zero architecture coding, zero training.
  - D-3553 / P-1653: Excludes video_7 and video_30 from fresh test eligibility; freezes fresh test pair
                     using admissible metadata ONLY (target count & categories) before auditing MI geometry.
  - D-3556 / D-3557: Restricts Stage B audit strictly to the 10 training/design videos (v_train).
                     Calibration videos (2) and test videos (2) are completely excluded from Stage B.
  - D-3560 / P-1660: Explicit temporal-chain invariant: derives k = frame_t - frame_hist, validates alignment,
                     and asserts k in {1..5}.
  - D-3561 / P-1661: Asserts 100% reconstruction of qualifying historical evidence for all design MI targets.
  - D-3562: Hard invariant check asserting monotonically non-increasing score order across all 20,310 cached frame predictions.
  - D-3565 / P-1665 / P-1667: Idempotent manifest write contract with JSON key-type normalization.
  - P-1657: Three-way geometry decomposition (GT motion, M0 localization error, effective memory displacement).
  - P-1658 / D-3558 / P-1669: Dual retention curves reporting per-historical-frame Top-K (K90_per_frame_topk).
  - P-1659: Token multiplicity computed once per unique historical frame; reports saturation P(N = 100).
  - P-1664: Evaluates nearest duplicate-like (IoU >= 0.30) and low-overlap (IoU < 0.30) competitor distance per event.
  - P-1666 / P-1668: Complete chain alignment assertions verifying target, historical annotation, image, frame, and video IDs.
  - P-1662: Per-video design robustness table auditing target counts, age support, retention, and displacements.
  - Complete Provenance: Verifies live hashes for candidate windows, train JSON, prediction cache, and dev pool.
"""

import os
import sys
import json
import hashlib
import itertools
import numpy as np
from pathlib import Path
from collections import defaultdict
from pycocotools.coco import COCO

OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
CANDIDATES_JSON = OUT_DIR / "pol2_candidate_windows_v1.json"
TRAIN_JSON = Path("D:/MARIVISION/seaclips-train.json")
CACHE_FILE = OUT_DIR / "cache_m0_train_raw_top100.json"
DEV_POOL_JSON = OUT_DIR / "pol2_dev_pool_manifest.json"
SPLIT_MANIFEST_JSON = OUT_DIR / "pol2_candidate2m_micro_split.json"
AUDIT_RESULTS_JSON = OUT_DIR / "pol2_candidate2m_dev_audit.json"

EXPECTED_CANDIDATE_SHA256 = "f29e0b8a5cbfc46641222d84a6d370c3e9bc51ca4ca1124b7753078ca7c1e270"
EXPECTED_TRAIN_JSON_SHA256 = "edc32bdbd6b792eb83b6daf11e14708232f1f4a9e06153ebebd5bad1caef9b35"
EXPECTED_CACHE_SHA256 = "d9816083760d698935b44e9165522e709b490c3af29e710d75b8b0d2d90210e0"
EXPECTED_DEV_POOL_SHA256 = "533567e1d74544db7087a95253e640ef0959b998e5d503f57b0eb8074d8fe970"

EVAL_IOU_THRESHOLD = 0.30
OPERATING_SCORE_THRESHOLD = 0.20
K_VALUES = [1, 2, 4, 8, 12, 16, 20, 24, 32, 50, 100]


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


def save_or_verify_json(file_path: Path, payload: dict, artifact_name: str) -> str:
    """
    Enforces D-3565 / P-1665 / P-1667: Idempotent frozen-artifact contract.
    Normalizes payload via JSON round-trip so integer keys are converted to strings
    before content comparison or serialization.
    """
    normalized_payload = json.loads(json.dumps(payload))
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
        assert existing_data == normalized_payload, (
            f"[FATAL] Existing {artifact_name} content mismatch with newly generated payload! Aborting to prevent mutation."
        )
        sha = compute_sha256(file_path)
        print(f"[+] Reusing and verified existing frozen {artifact_name}: {file_path.resolve()}")
        print(f"    Existing Manifest SHA-256: {sha}")
        return sha
    else:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(normalized_payload, f, indent=2)
        sha = compute_sha256(file_path)
        print(f"[+] Created and frozen new {artifact_name}: {file_path.resolve()}")
        print(f"    Immutable Manifest SHA-256: {sha}")
        return sha


def main():
    print("=" * 115)
    print("PROOF-OF-LAUNCH STAGE 2 (PoL-2M): HOLDOUT-FIRST PARTITION FREEZE & DESIGN-ONLY AUDIT")
    print("=" * 115)

    # 1. Full Live Provenance Verification
    assert CANDIDATES_JSON.exists(), f"[FATAL] Missing {CANDIDATES_JSON}"
    assert TRAIN_JSON.exists(), f"[FATAL] Missing {TRAIN_JSON}"
    assert CACHE_FILE.exists(), f"[FATAL] Missing {CACHE_FILE}"
    assert DEV_POOL_JSON.exists(), f"[FATAL] Missing {DEV_POOL_JSON}"

    cand_hash = compute_sha256(CANDIDATES_JSON)
    train_hash = compute_sha256(TRAIN_JSON)
    cache_hash = compute_sha256(CACHE_FILE)
    dev_pool_hash = compute_sha256(DEV_POOL_JSON)

    assert cand_hash == EXPECTED_CANDIDATE_SHA256, "[FATAL] Candidate JSON SHA mismatch!"
    assert train_hash == EXPECTED_TRAIN_JSON_SHA256, "[FATAL] Train JSON SHA mismatch!"
    assert cache_hash == EXPECTED_CACHE_SHA256, "[FATAL] Prediction cache SHA mismatch!"
    assert dev_pool_hash == EXPECTED_DEV_POOL_SHA256, "[FATAL] Dev pool manifest SHA mismatch!"

    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache_data = json.load(f)
    raw_detections = {int(k): v for k, v in cache_data["detections"].items()}
    assert len(raw_detections) == 20310

    # D-3562: Hard invariant asserting monotonically non-increasing score order (allowing ties)
    for iid, p_list in raw_detections.items():
        scores = [d["score"] for d in p_list]
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1)), (
            f"[FATAL] Prediction cache for image {iid} is NOT sorted in monotonically non-increasing score order!"
        )

    print("[+] All Upstream Source Artifacts Certified by Immutable SHA-256.")
    print("[+] Hard Invariant Passed: All 20,310 cached frame predictions are monotonically non-increasing by score (D-3562).")

    # --------------------------------------------------------------------------
    # STAGE A: HOLDOUT-FIRST SPLIT FREEZE (D-3553, D-3557, P-1653)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("STAGE A: CANDIDATE-2M 10/2/2 PARTITION FREEZE (HOLDOUT-FIRST GOVERNANCE)")
    print("Rule: video_7 and video_30 are retired from test eligibility and routed strictly to train.")
    print("=" * 115)

    with open(DEV_POOL_JSON, "r", encoding="utf-8") as f:
        dev_manifest = json.load(f)

    ecam_mi_vids = [
        r["video_id"] for r in dev_manifest["development_videos"]
        if r["sensor"] == "e-CAM" and r["supports_2m"]
    ]
    ecam_mi_vids.sort()
    assert len(ecam_mi_vids) == 14, f"[FATAL] Expected 14 e-CAM MI videos, found {len(ecam_mi_vids)}"
    print(f"[*] Total Candidate-Eligible e-CAM MI Population (14 videos): {ecam_mi_vids}")

    coco = COCO(str(TRAIN_JSON))
    ann_lookup = {a["id"]: a for a in coco.loadAnns(coco.getAnnIds())}
    cats = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    img_lookup = {info["id"]: info for info in coco.loadImgs(coco.getImgIds())}

    with open(CANDIDATES_JSON, "r", encoding="utf-8") as f:
        cand_windows = json.load(f)["windows"]

    video_mi_targets = defaultdict(dict)
    video_categories = defaultdict(set)
    video_image_ids = defaultdict(set)

    for w in cand_windows:
        v_id = w["video_id"]
        # Schema fix: check n_mem_incr_s20 safely
        n_mi = w.get("n_mem_incr_s20", w.get("n_memincr_s20", 0))
        if v_id in ecam_mi_vids and n_mi >= 1:
            for t in w["targets"]:
                aid = t["ann_id"]
                is_mi = t.get("mem_incr_s20", t.get("memincr_s20", False))
                if is_mi and aid not in video_mi_targets[v_id]:
                    ann = ann_lookup[aid]
                    c_name = cats[ann["category_id"]]

                    # Authoritative chain derivation: look up parent image and frame IDs via COCO index
                    chain_ann_ids = t["chain_ann_ids"]
                    chain_image_ids = []
                    chain_frame_ids = []
                    for p_aid in chain_ann_ids:
                        assert p_aid in ann_lookup, f"[FATAL] Historical annotation {p_aid} missing from COCO index for target {aid}!"
                        p_iid = ann_lookup[p_aid]["image_id"]
                        assert p_iid in img_lookup, f"[FATAL] Historical image {p_iid} missing from COCO index for target {aid}!"
                        chain_image_ids.append(p_iid)
                        chain_frame_ids.append(img_lookup[p_iid]["frame_id"])

                    target_img_id = ann["image_id"]
                    target_frame_id = img_lookup[target_img_id]["frame_id"]

                    video_mi_targets[v_id][aid] = {
                        "ann_id": aid,
                        "category_id": ann["category_id"],
                        "category": c_name,
                        "bbox": ann["bbox"],
                        "frame_id": target_frame_id,
                        "image_id": target_img_id,
                        "chain_ann_ids": chain_ann_ids,
                        "chain_image_ids": chain_image_ids,
                        "chain_frame_ids": chain_frame_ids
                    }
                    video_categories[v_id].add(c_name)
                    video_image_ids[v_id].add(target_img_id)

    total_unique_mi = sum(len(tgts) for tgts in video_mi_targets.values())
    assert total_unique_mi == 311, f"[FATAL] Expected exactly 311 unique MI targets, found {total_unique_mi}"
    print(f"[+] Verified 311 unique Memory-Incremental targets across the 14 e-CAM videos.")

    print("\n--- Per-Video Memory-Incremental Opportunity Profile ---")
    print(f"{'Video ID':<10} | {'MI Targets':<12} | {'Unique Frames':<14} | {'Target Categories'}")
    print("-" * 75)
    for v_id in ecam_mi_vids:
        cat_str = ", ".join(sorted(video_categories[v_id]))
        n_t = len(video_mi_targets[v_id])
        n_img = len(video_image_ids[v_id])
        status = " (Spent in 2F -> Forced Train)" if v_id in [7, 30] else ""
        print(f"video_{v_id:<4d} | {n_t:<12d} | {n_img:<14d} | {cat_str}{status}")

    forced_train_vids = {7, 30}
    fresh_candidate_vids = [v for v in ecam_mi_vids if v not in forced_train_vids]
    assert len(fresh_candidate_vids) == 12

    IDEAL_TRAIN = 311.0 * (10.0 / 14.0)  # 222.14
    IDEAL_CALIB = 311.0 * (2.0 / 14.0)   # 44.43
    IDEAL_TEST = 311.0 * (2.0 / 14.0)    # 44.43

    print("\n[*] Predeclared Lexicographic Partition Selection Hierarchy:")
    print("    1. Primary:   Minimize total target L1 deviation (|N_train - 222.14| + |N_calib - 44.43| + |N_test - 44.43|)")
    print("    2. Secondary: Maximize minimum category count between calibration and test (min(|Cats_calib|, |Cats_test|))")
    print("    3. Tertiary:  Maximize total evaluation category count (|Cats_calib| + |Cats_test|)")
    print("    4. Tie-Break: Deterministic video ID order")

    best_partition = None
    all_combos_test = list(itertools.combinations(fresh_candidate_vids, 2))

    for test_vids in all_combos_test:
        test_set = set(test_vids)
        rem_fresh = [v for v in fresh_candidate_vids if v not in test_set]

        n_test = sum(len(video_mi_targets[v]) for v in test_vids)
        cats_test = set.union(*[video_categories[v] for v in test_vids])

        for calib_vids in itertools.combinations(rem_fresh, 2):
            calib_set = set(calib_vids)
            train_vids = tuple(sorted(list(set(rem_fresh) - calib_set | forced_train_vids)))

            n_calib = sum(len(video_mi_targets[v]) for v in calib_vids)
            cats_calib = set.union(*[video_categories[v] for v in calib_vids])

            n_train = sum(len(video_mi_targets[v]) for v in train_vids)
            cats_train = set.union(*[video_categories[v] for v in train_vids])

            target_dev = (
                abs(n_train - IDEAL_TRAIN)
                + abs(n_calib - IDEAL_CALIB)
                + abs(n_test - IDEAL_TEST)
            )

            min_eval_cats = min(len(cats_test), len(cats_calib))
            total_eval_cats = len(cats_test) + len(cats_calib)

            score = (
                -target_dev,
                min_eval_cats,
                total_eval_cats,
                tuple(-v for v in test_vids),
                tuple(-v for v in calib_vids)
            )

            if best_partition is None or score > best_partition["score"]:
                best_partition = {
                    "score": score,
                    "target_dev": target_dev,
                    "train_vids": train_vids,
                    "calib_vids": calib_vids,
                    "test_vids": test_vids,
                    "n_train": n_train,
                    "n_calib": n_calib,
                    "n_test": n_test,
                    "cats_train": cats_train,
                    "cats_calib": cats_calib,
                    "cats_test": cats_test
                }

    v_train = sorted(list(best_partition["train_vids"]))
    v_calib = sorted(list(best_partition["calib_vids"]))
    v_test = sorted(list(best_partition["test_vids"]))

    assert len(v_train) == 10
    assert len(v_calib) == 2
    assert len(v_test) == 2
    assert set(v_train).isdisjoint(v_calib)
    assert set(v_train).isdisjoint(v_test)
    assert set(v_calib).isdisjoint(v_test)
    assert set(v_train) | set(v_calib) | set(v_test) == set(ecam_mi_vids)
    assert best_partition["n_train"] + best_partition["n_calib"] + best_partition["n_test"] == 311
    assert 7 in v_train and 30 in v_train, "[FATAL] Spent videos 7/30 not in train pool!"
    assert 7 not in v_test and 30 not in v_test, "[FATAL] Spent videos 7/30 in fresh test set!"

    print(f"\n[+] Optimal Clean 10/2/2 Partition Selected Deterministically:")
    print(f"    - Training Pool (10 Videos):       {v_train} ({best_partition['n_train']} targets, nominal: {IDEAL_TRAIN:.1f})")
    print(f"    - Calibration Set (2 Videos):      {v_calib} ({best_partition['n_calib']} targets, nominal: {IDEAL_CALIB:.1f})")
    print(f"    - Fresh Untouched Test (2 Videos): {v_test} ({best_partition['n_test']} targets, nominal: {IDEAL_TEST:.1f})")
    print(f"    - Total Three-Subset L1 Deviation: {best_partition['target_dev']:.2f} targets")

    calib_targets_meta = [
        {"ann_id": t["ann_id"], "category_id": t["category_id"], "video_id": v, "image_id": t["image_id"], "frame_id": t["frame_id"]}
        for v in v_calib for t in video_mi_targets[v].values()
    ]
    test_targets_meta = [
        {"ann_id": t["ann_id"], "category_id": t["category_id"], "video_id": v, "image_id": t["image_id"], "frame_id": t["frame_id"]}
        for v in v_test for t in video_mi_targets[v].values()
    ]

    split_manifest_payload = {
        "_meta": {
            "artifact_name": "pol2_candidate2m_micro_split_v1",
            "role": "POL2_CANDIDATE_2M_HOLDOUT_FIRST_PARTITION",
            "candidate_windows_sha256": cand_hash,
            "train_json_sha256": train_hash,
            "m0_prediction_cache_sha256": cache_hash,
            "dev_pool_manifest_sha256": dev_pool_hash,
            "selection_policy": "lexicographic_10_2_2_proportional_excluding_spent_7_30",
            "total_candidate_videos": 14,
            "total_mi_targets": 311,
            "train_videos_count": len(v_train),
            "calib_videos_count": len(v_calib),
            "test_videos_count": len(v_test),
            "n_train_targets": best_partition["n_train"],
            "n_calib_targets": best_partition["n_calib"],
            "n_test_targets": best_partition["n_test"],
            "train_videos": v_train,
            "calib_videos": v_calib,
            "test_videos": v_test
        },
        "calibration_targets_meta": calib_targets_meta,
        "test_targets_meta": test_targets_meta
    }

    # D-3565 / P-1665 / P-1667: Idempotent JSON-normalized save or verification
    split_hash = save_or_verify_json(SPLIT_MANIFEST_JSON, split_manifest_payload, "Candidate 2M Clean Partition Manifest")

    # --------------------------------------------------------------------------
    # STAGE B: DESIGN-ONLY MI POPULATION AUDIT (D-3556, P-1654, P-1657..P-1664, P-1666, P-1668, P-1669)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("STAGE B: DESIGN-ONLY MI POPULATION AUDIT (10 DESIGN/TRAIN VIDEOS ONLY)")
    print(f"Auditing 10 Design Videos: {v_train} (Calibration {v_calib} and Test {v_test} Strictly Excluded)")
    print("=" * 115)

    design_vids = v_train
    design_targets = []
    for v in design_vids:
        design_targets.extend(video_mi_targets[v].values())

    n_design_tgts = len(design_targets)
    print(f"[+] Total Design MI Targets Audited: N = {n_design_tgts}")

    has_evidence_by_depth = {k: 0 for k in range(1, 6)}
    earliest_ages = []
    cumulative_evidence = {k: 0 for k in range(1, 6)}

    # Decomposed Geometry Collections (CLM-2786, P-1657)
    # Component A: GT Annotation-Linked Motion (GT_{t-k} -> GT_t): c(GT_t) - c(GT_{t-k})
    gt_motion_dx = defaultdict(list)
    gt_motion_dy = defaultdict(list)
    gt_motion_r = defaultdict(list)
    gt_motion_r_rel = defaultdict(list)

    # Component B: Historical Localization Error (M0_{t-k} -> GT_{t-k}): c(GT_{t-k}) - c(M0_{t-k})
    loc_err_dx = defaultdict(list)
    loc_err_dy = defaultdict(list)
    loc_err_r = defaultdict(list)
    loc_err_r_rel = defaultdict(list)

    # Component C: Effective Memory Displacement (M0_{t-k} -> GT_t): c(GT_t) - c(M0_{t-k})
    eff_disp_dx = defaultdict(list)
    eff_disp_dy = defaultdict(list)
    eff_disp_r = defaultdict(list)
    eff_disp_r_rel = defaultdict(list)

    scale_changes_w = defaultdict(list)
    scale_changes_h = defaultdict(list)
    scale_changes_area = defaultdict(list)
    scores_by_depth = defaultdict(list)

    overlapping_proposals_total = 0
    overlapping_proposals_correct = 0
    overlapping_proposals_wrong_cat = 0
    target_age_reliability_states = {"correct_only": 0, "wrong_only": 0, "both": 0, "none": 0}

    # P-1664: Nearest competitor proximity per historical evidence event (Primary)
    nearest_dist_duplicate_competitor = []
    nearest_dist_low_overlap_competitor = []

    target_min_ranks = {}
    event_ranks_all = []

    per_video_stats = defaultdict(lambda: {
        "n_targets": 0,
        "earliest_ages": defaultdict(int),
        "cumulative_ages": defaultdict(int),
        "effective_displacements": [],
        "target_min_ranks": {}
    })

    reconstructed_targets_count = 0

    for t in design_targets:
        aid = t["ann_id"]
        vid = next(v for v in design_vids if aid in video_mi_targets[v])
        per_video_stats[vid]["n_targets"] += 1

        curr_img = img_lookup[t["image_id"]]
        sx_curr = 640.0 / float(curr_img["width"])
        sy_curr = 640.0 / float(curr_img["height"])
        bx_t, by_t, bw_t, bh_t = t["bbox"]
        cx_gt_t = (bx_t + bw_t / 2.0) * sx_curr
        cy_gt_t = (by_t + bh_t / 2.0) * sy_curr
        w_gt_t = bw_t * sx_curr
        h_gt_t = bh_t * sy_curr
        scale_gt_t = max(w_gt_t, h_gt_t)

        # P-1666: Current target metadata consistency assertions
        assert ann_lookup[aid]["image_id"] == t["image_id"], f"[FATAL] Target {aid} image_id mismatch!"
        assert curr_img["frame_id"] == t["frame_id"], f"[FATAL] Target {aid} frame_id mismatch!"
        assert curr_img["video_id"] == vid, f"[FATAL] Target {aid} video_id mismatch in curr_img!"

        # D-3560 / P-1660: Chain alignment and temporal difference invariant
        n_chain = len(t["chain_ann_ids"])
        assert n_chain == len(t["chain_image_ids"]) == len(t["chain_frame_ids"]), (
            f"[FATAL] Chain array length mismatch for target {aid}!"
        )

        target_useful_k = []
        target_ranks = []
        seen_ages_in_chain = set()

        for j in range(n_chain):
            p_aid = t["chain_ann_ids"][j]
            p_iid = t["chain_image_ids"][j]
            p_fid = t["chain_frame_ids"][j]

            # D-3560: Derive k strictly from frame differences
            k = t["frame_id"] - p_fid
            assert 1 <= k <= 5, f"[FATAL] Derived historical age k={k} out of range [1..5] for target {aid}!"
            assert k not in seen_ages_in_chain, f"[FATAL] Duplicate age k={k} in chain for target {aid}!"
            seen_ages_in_chain.add(k)

            # P-1668: Hard assertions for missing historical annotations/images
            assert p_aid in ann_lookup, f"[FATAL] Historical annotation {p_aid} missing from COCO index for target {aid}!"
            assert p_iid in img_lookup, f"[FATAL] Historical image {p_iid} missing from COCO index for target {aid}!"

            p_ann = ann_lookup[p_aid]
            p_img = img_lookup[p_iid]

            # P-1666: Historical chain metadata consistency assertions
            assert p_ann["image_id"] == p_iid, f"[FATAL] Chain image_id mismatch for ann {p_aid}!"
            assert p_img["frame_id"] == p_fid, f"[FATAL] Frame ID mismatch in COCO metadata for image {p_iid}!"
            assert p_img["video_id"] == vid, f"[FATAL] Historical image {p_iid} video_id mismatch with {vid}!"

            sx_p = 640.0 / float(p_img["width"])
            sy_p = 640.0 / float(p_img["height"])
            bx_p, by_p, bw_p, bh_p = p_ann["bbox"]
            cx_gt_p = (bx_p + bw_p / 2.0) * sx_p
            cy_gt_p = (by_p + bh_p / 2.0) * sy_p
            w_gt_p = bw_p * sx_p
            h_gt_p = bh_p * sy_p
            scale_gt_p = max(w_gt_p, h_gt_p)

            p_preds = raw_detections.get(p_iid, [])

            # Target x age category reliability state
            has_corr_prop = False
            has_wrong_prop = False
            for d in p_preds:
                if d["score"] >= 0.10 and box_iou(d["bbox"], p_ann["bbox"]) >= EVAL_IOU_THRESHOLD:
                    overlapping_proposals_total += 1
                    if d["category_id"] == p_ann["category_id"]:
                        overlapping_proposals_correct += 1
                        has_corr_prop = True
                    else:
                        overlapping_proposals_wrong_cat += 1
                        has_wrong_prop = True

            if has_corr_prop and has_wrong_prop:
                target_age_reliability_states["both"] += 1
            elif has_corr_prop:
                target_age_reliability_states["correct_only"] += 1
            elif has_wrong_prop:
                target_age_reliability_states["wrong_only"] += 1
            else:
                target_age_reliability_states["none"] += 1

            # Matching class-aware detections with s >= 0.20
            matching_s20 = [
                (rank, d) for rank, d in enumerate(p_preds)
                if d["category_id"] == p_ann["category_id"]
                and box_iou(d["bbox"], p_ann["bbox"]) >= EVAL_IOU_THRESHOLD
                and d["score"] >= OPERATING_SCORE_THRESHOLD
            ]

            if matching_s20:
                best_rank, best_d = max(matching_s20, key=lambda item: item[1]["score"])
                target_useful_k.append(k)
                target_ranks.append(best_rank)
                event_ranks_all.append(best_rank)
                has_evidence_by_depth[k] += 1

                px, py, pw, ph = best_d["bbox"]
                cx_m0_p = (px + pw / 2.0) * sx_p
                cy_m0_p = (py + ph / 2.0) * sy_p
                w_m0_p = pw * sx_p
                h_m0_p = ph * sy_p

                # CLM-2786 / P-1657: Corrected vector directions (x_dest - x_src)
                # Component A: GT Annotation-Linked Motion (GT_{t-k} -> GT_t)
                dx_a = cx_gt_t - cx_gt_p
                dy_a = cy_gt_t - cy_gt_p
                r_a = np.sqrt(dx_a**2 + dy_a**2)
                gt_motion_dx[k].append(dx_a)
                gt_motion_dy[k].append(dy_a)
                gt_motion_r[k].append(r_a)
                gt_motion_r_rel[k].append(r_a / max(1.0, scale_gt_t))

                # Component B: Historical Localization Error (M0_{t-k} -> GT_{t-k})
                dx_b = cx_gt_p - cx_m0_p
                dy_b = cy_gt_p - cy_m0_p
                r_b = np.sqrt(dx_b**2 + dy_b**2)
                loc_err_dx[k].append(dx_b)
                loc_err_dy[k].append(dy_b)
                loc_err_r[k].append(r_b)
                loc_err_r_rel[k].append(r_b / max(1.0, scale_gt_p))

                # Component C: Effective Memory Displacement (M0_{t-k} -> GT_t)
                dx_c = cx_gt_t - cx_m0_p
                dy_c = cy_gt_t - cy_m0_p
                r_c = np.sqrt(dx_c**2 + dy_c**2)
                eff_disp_dx[k].append(dx_c)
                eff_disp_dy[k].append(dy_c)
                eff_disp_r[k].append(r_c)
                eff_disp_r_rel[k].append(r_c / max(1.0, scale_gt_t))
                per_video_stats[vid]["effective_displacements"].append(r_c)

                scale_changes_w[k].append(w_m0_p / max(1.0, w_gt_t))
                scale_changes_h[k].append(h_m0_p / max(1.0, h_gt_t))
                scale_changes_area[k].append((w_m0_p * h_m0_p) / max(1.0, w_gt_t * h_gt_t))
                scores_by_depth[k].append(best_d["score"])

                # P-1664: Nearest competitor tracking per evidence event
                dup_dists_event = []
                low_overlap_dists_event = []
                for r_other, d_other in enumerate(p_preds):
                    if r_other != best_rank and d_other["score"] >= 0.10:
                        ox, oy, ow, oh = d_other["bbox"]
                        cx_o = (ox + ow / 2.0) * sx_p
                        cy_o = (oy + oh / 2.0) * sy_p
                        d_dist = np.sqrt((cx_o - cx_m0_p)**2 + (cy_o - cy_m0_p)**2)
                        iou_other = box_iou(best_d["bbox"], d_other["bbox"])

                        if iou_other >= 0.30:
                            dup_dists_event.append(d_dist)
                        else:
                            low_overlap_dists_event.append(d_dist)

                if dup_dists_event:
                    nearest_dist_duplicate_competitor.append(min(dup_dists_event))
                if low_overlap_dists_event:
                    nearest_dist_low_overlap_competitor.append(min(low_overlap_dists_event))

        # D-3561 / P-1661: Invariant asserting successful historical evidence reconstruction
        assert len(target_useful_k) >= 1, (
            f"[FATAL] Design target {aid} in video {vid} marked mem_incr_s20 failed to reconstruct historical evidence!"
        )
        reconstructed_targets_count += 1

        k_min = min(target_useful_k)
        earliest_ages.append(k_min)
        per_video_stats[vid]["earliest_ages"][k_min] += 1
        for k_c in range(k_min, 6):
            cumulative_evidence[k_c] += 1
            per_video_stats[vid]["cumulative_ages"][k_c] += 1

        target_min_ranks[aid] = min(target_ranks)
        per_video_stats[vid]["target_min_ranks"][aid] = min(target_ranks)

    assert reconstructed_targets_count == n_design_tgts, (
        f"[FATAL] Reconstructed {reconstructed_targets_count} targets != design count {n_design_tgts} (D-3561)!"
    )
    print(f"[+] MI Reconstruction Contract Certified: 100% of design targets ({n_design_tgts}/{n_design_tgts}) verified with qualifying historical evidence (D-3561).")

    # 1. Evidence Age Structure Table
    print("\n--- 1. MI EVIDENCE AGE STRUCTURE & OPPORTUNITY ACCUMULATION (10 DESIGN VIDEOS) (P-1654) ---")
    print(f"{'Age Depth k':<14} | {'Active Evidence N_k':<22} | {'Earliest Useful k_min':<22} | {'Cumulative Opportunity N_<=k':<28}")
    print("-" * 92)
    for k in range(1, 6):
        n_act = has_evidence_by_depth[k]
        pct_act = (n_act / n_design_tgts) * 100
        n_earl = sum(1 for a in earliest_ages if a == k)
        pct_earl = (n_earl / n_design_tgts) * 100
        n_cum = cumulative_evidence[k]
        pct_cum = (n_cum / n_design_tgts) * 100
        print(f"t-{k:<11d} | {n_act:4d} ({pct_act:5.1f}%)            | {n_earl:4d} ({pct_earl:5.1f}%)            | {n_cum:4d} ({pct_cum:5.1f}%)")

    # 2. Decomposed Geometry Table (P-1657)
    print("\n--- 2. THREE-WAY GEOMETRY DECOMPOSITION (10 DESIGN VIDEOS) (P-1657) ---")
    print(f"{'Depth':<6} | {'N':<4} | {'(A) GT Motion r [IQR]':<26} | {'(B) Loc Error r [IQR]':<26} | {'(C) Eff Disp r [IQR] (P90)':<28}")
    print("-" * 98)
    for k in range(1, 6):
        if not eff_disp_r[k]:
            print(f"t-{k:<4d} |    0 | N/A                        | N/A                        | N/A")
            continue
        n_k = len(eff_disp_r[k])
        ra_m, ra_25, ra_75 = np.median(gt_motion_r[k]), np.percentile(gt_motion_r[k], 25), np.percentile(gt_motion_r[k], 75)
        rb_m, rb_25, rb_75 = np.median(loc_err_r[k]), np.percentile(loc_err_r[k], 25), np.percentile(loc_err_r[k], 75)
        rc_m, rc_25, rc_75, rc_p90 = np.median(eff_disp_r[k]), np.percentile(eff_disp_r[k], 25), np.percentile(eff_disp_r[k], 75), np.percentile(eff_disp_r[k], 90)

        print(f"t-{k:<4d} | {n_k:4d} | {ra_m:4.1f} px [{ra_25:4.1f}, {ra_75:4.1f}]         | {rb_m:4.1f} px [{rb_25:4.1f}, {rb_75:4.1f}]         | {rc_m:4.1f} px [{rc_25:4.1f}, {rc_75:4.1f}] (p90:{rc_p90:4.1f})")

    print("\n--- Scale & Area Ratios of Historical Evidence relative to Current Target ---")
    for k in range(1, 6):
        if scale_changes_area[k]:
            print(f"  t-{k}: Width Ratio = {np.median(scale_changes_w[k]):.2f}x | Height Ratio = {np.median(scale_changes_h[k]):.2f}x | Area Ratio = {np.median(scale_changes_area[k]):.2f}x")

    # 3. Class-Aware Evidence Scores & Category Reliability
    print("\n--- 3. DETECTOR EVIDENCE SCORES & CATEGORY RELIABILITY ---")
    print(f"{'Depth k':<8} | {'N':<4} | {'Mean Score':<12} | {'Median Score':<14} | {'IQR [Q25, Q75]':<18} | {'Min Score':<10} | {'Max Score'}")
    print("-" * 88)
    for k in range(1, 6):
        if not scores_by_depth[k]:
            continue
        sc = scores_by_depth[k]
        print(f"t-{k:<6d} | {len(sc):4d} | {np.mean(sc):<12.3f} | {np.median(sc):<14.3f} | [{np.percentile(sc, 25):.3f}, {np.percentile(sc, 75):.3f}]    | {np.min(sc):<10.3f} | {np.max(sc):.3f}")

    tot_ta_states = sum(target_age_reliability_states.values())
    print(f"\n[*] Target x Age Category Reliability States (N = {tot_ta_states} historical instances):")
    print(f"    - Correct Category Only: {target_age_reliability_states['correct_only']:4d} ({target_age_reliability_states['correct_only']/max(1, tot_ta_states)*100:5.1f}%)")
    print(f"    - Wrong Category Only:   {target_age_reliability_states['wrong_only']:4d} ({target_age_reliability_states['wrong_only']/max(1, tot_ta_states)*100:5.1f}%)")
    print(f"    - Both Present:          {target_age_reliability_states['both']:4d} ({target_age_reliability_states['both']/max(1, tot_ta_states)*100:5.1f}%)")
    print(f"    - None (No overlap):     {target_age_reliability_states['none']:4d} ({target_age_reliability_states['none']/max(1, tot_ta_states)*100:5.1f}%)")

    # 4. Frame-Unique Multiplicity & Top-100 Cache Saturation (P-1659)
    unique_hist_iids = set()
    for t in design_targets:
        unique_hist_iids.update(t["chain_image_ids"])

    frame_mult = {0.10: [], 0.20: [], 0.30: [], 0.50: []}
    for iid in unique_hist_iids:
        p_list = raw_detections.get(iid, [])
        for th in frame_mult:
            cnt = sum(1 for d in p_list if d["score"] >= th)
            frame_mult[th].append(cnt)

    print(f"\n--- 4. FRAME-UNIQUE CANDIDATE MULTIPLICITY & CACHE SATURATION (N = {len(unique_hist_iids)} UNIQUE FRAMES) (P-1659) ---")
    print(f"{'Score Floor':<14} | {'Mean Tokens/Frame':<18} | {'Median':<10} | {'P90':<10} | {'P95':<10} | {'Saturation P(N=100)':<22}")
    print("-" * 88)
    for th in [0.10, 0.20, 0.30, 0.50]:
        counts = frame_mult[th]
        sat_rate = (sum(1 for c in counts if c == 100) / len(counts)) * 100
        print(f"s >= {th:<10.2f} | {np.mean(counts):<18.2f} | {np.median(counts):<10.1f} | {np.percentile(counts, 90):<10.1f} | {np.percentile(counts, 95):<10.1f} | {sat_rate:5.1f}%")

    # P-1664: Nearest competitor proximity reporting
    print(f"\n[*] Nearest Competitor Proximity per Historical Evidence Event (P-1664):")
    if nearest_dist_duplicate_competitor:
        d_p50 = np.median(nearest_dist_duplicate_competitor)
        d_p25 = np.percentile(nearest_dist_duplicate_competitor, 25)
        d_p75 = np.percentile(nearest_dist_duplicate_competitor, 75)
        d_p10 = np.percentile(nearest_dist_duplicate_competitor, 10)
        print(f"    - Nearest Duplicate-Like Proposal (IoU >= 0.30): Median = {d_p50:4.1f} px | IQR = [{d_p25:4.1f}, {d_p75:4.1f}] px | P10 = {d_p10:4.1f} px (Events with duplicates = {len(nearest_dist_duplicate_competitor)})")
    else:
        print(f"    - Nearest Duplicate-Like Proposal (IoU >= 0.30): None detected")

    if nearest_dist_low_overlap_competitor:
        lo_p50 = np.median(nearest_dist_low_overlap_competitor)
        lo_p25 = np.percentile(nearest_dist_low_overlap_competitor, 25)
        lo_p75 = np.percentile(nearest_dist_low_overlap_competitor, 75)
        lo_p10 = np.percentile(nearest_dist_low_overlap_competitor, 10)
        print(f"    - Nearest Low-Overlap Proposal    (IoU <  0.30): Median = {lo_p50:4.1f} px | IQR = [{lo_p25:4.1f}, {lo_p75:4.1f}] px | P10 = {lo_p10:4.1f} px (Events with competitors = {len(nearest_dist_low_overlap_competitor)})")
    else:
        print(f"    - Nearest Low-Overlap Proposal    (IoU <  0.30): None detected")

    # 5. Dual Opportunity Retention Curves (P-1658, D-3558, P-1669)
    print("\n--- 5. DUAL OPPORTUNITY RETENTION CURVES: UNIQUE-TARGET (PRIMARY) VS EVENT (P-1658, D-3558, P-1669) ---")
    print(f"{'Per-Frame Top-K':<18} | {'Unique-Target Retention (PRIMARY)':<36} | {'Evidence-Event Retention':<28}")
    print("-" * 86)

    total_recoverable_targets = n_design_tgts
    total_events = len(event_ranks_all)

    target_retention_curves = {}
    event_retention_curves = {}

    for K in K_VALUES:
        retained_targets = sum(1 for r in target_min_ranks.values() if r < K)
        pct_target = (retained_targets / total_recoverable_targets) * 100
        target_retention_curves[K] = pct_target

        retained_events = sum(1 for r in event_ranks_all if r < K)
        pct_event = (retained_events / total_events) * 100 if total_events > 0 else 0.0
        event_retention_curves[K] = pct_event

    K90 = next((K for K in K_VALUES if target_retention_curves[K] >= 90.0), None)

    for K in K_VALUES:
        pct_t = target_retention_curves[K]
        pct_e = event_retention_curves[K]
        ret_t = int(round((pct_t / 100.0) * total_recoverable_targets))
        ret_e = int(round((pct_e / 100.0) * total_events))
        marker = " <--- MINIMUM PER-FRAME K >= 90% UNIQUE-TARGET RETENTION (Q-1802)" if K == K90 else ""
        print(f"Top-{K:<14d} | {ret_t:4d} / {total_recoverable_targets:4d} ({pct_t:5.1f}%)                      | {ret_e:4d} / {total_events:4d} ({pct_e:5.1f}%){marker}")

    # 6. Per-Video Design Robustness Table (P-1662)
    print("\n--- 6. PER-VIDEO DESIGN ROBUSTNESS SUMMARY (10 DESIGN VIDEOS) (P-1662) ---")
    print(f"{'Video ID':<10} | {'Targets':<8} | {'k_min=1':<8} | {'k_min<=2':<9} | {'k_min<=5':<9} | {'Top-8 Ret':<10} | {'Top-16 Ret':<11} | {'Top-32 Ret':<11} | {'Eff Disp med (p90)'}")
    print("-" * 108)

    per_video_summary = {}
    for vid in design_vids:
        v_data = per_video_stats[vid]
        nv = v_data["n_targets"]
        k1 = v_data["earliest_ages"][1]
        k2 = sum(v_data["earliest_ages"][k] for k in [1, 2])
        k5 = sum(v_data["earliest_ages"][k] for k in range(1, 6))

        ret8 = (sum(1 for r in v_data["target_min_ranks"].values() if r < 8) / nv) * 100 if nv > 0 else 0.0
        ret16 = (sum(1 for r in v_data["target_min_ranks"].values() if r < 16) / nv) * 100 if nv > 0 else 0.0
        ret32 = (sum(1 for r in v_data["target_min_ranks"].values() if r < 32) / nv) * 100 if nv > 0 else 0.0

        r_list = v_data["effective_displacements"]
        r_med = float(np.median(r_list)) if r_list else 0.0
        r_p90 = float(np.percentile(r_list, 90)) if r_list else 0.0

        per_video_summary[str(vid)] = {
            "n_targets": nv,
            "k_min_1": k1,
            "k_min_le_2": k2,
            "k_min_le_5": k5,
            "retention_top8": ret8,
            "retention_top16": ret16,
            "retention_top32": ret32,
            "eff_disp_med": r_med,
            "eff_disp_p90": r_p90
        }

        print(f"video_{vid:<4d} | {nv:<8d} | {k1:<8d} | {k2:<9d} | {k5:<9d} | {ret8:5.1f}%    | {ret16:5.1f}%     | {ret32:5.1f}%     | {r_med:4.1f} px ({r_p90:4.1f} px)")

    # Serialize Stage B Audit Manifest (P-1663, P-1665, P-1667)
    audit_payload = {
        "_meta": {
            "artifact_name": "pol2_candidate2m_dev_audit_v1",
            "role": "POL2_CANDIDATE_2M_DESIGN_POPULATION_AUDIT",
            "split_manifest_sha256": split_hash,
            "design_videos": design_vids,
            "calibration_videos_excluded": v_calib,
            "fresh_test_videos_excluded": v_test,
            "total_design_mi_targets": n_design_tgts,
            "reconstructed_targets_count": reconstructed_targets_count,
            "K90_per_frame_topk": K90
        },
        "age_distribution": {
            "has_evidence_by_depth": has_evidence_by_depth,
            "earliest_age_distribution": {k: sum(1 for a in earliest_ages if a == k) for k in range(1, 6)},
            "cumulative_evidence": cumulative_evidence
        },
        "decomposed_geometry": {
            str(k): {
                "n_samples": len(eff_disp_r[k]),
                "gt_motion_r_med": float(np.median(gt_motion_r[k])) if gt_motion_r[k] else None,
                "gt_motion_r_iqr": [float(np.percentile(gt_motion_r[k], 25)), float(np.percentile(gt_motion_r[k], 75))] if gt_motion_r[k] else None,
                "loc_error_r_med": float(np.median(loc_err_r[k])) if loc_err_r[k] else None,
                "loc_error_r_iqr": [float(np.percentile(loc_err_r[k], 25)), float(np.percentile(loc_err_r[k], 75))] if loc_err_r[k] else None,
                "eff_disp_r_med": float(np.median(eff_disp_r[k])) if eff_disp_r[k] else None,
                "eff_disp_r_iqr": [float(np.percentile(eff_disp_r[k], 25)), float(np.percentile(eff_disp_r[k], 75))] if eff_disp_r[k] else None,
                "eff_disp_r_p90": float(np.percentile(eff_disp_r[k], 90)) if eff_disp_r[k] else None,
                "eff_disp_r_rel_med": float(np.median(eff_disp_r_rel[k])) if eff_disp_r_rel[k] else None,
                "scale_w_med": float(np.median(scale_changes_w[k])) if scale_changes_w[k] else None,
                "scale_h_med": float(np.median(scale_changes_h[k])) if scale_changes_h[k] else None,
                "area_ratio_med": float(np.median(scale_changes_area[k])) if scale_changes_area[k] else None
            }
            for k in range(1, 6)
        },
        "scores_by_depth": {
            str(k): {
                "mean": float(np.mean(scores_by_depth[k])) if scores_by_depth[k] else None,
                "median": float(np.median(scores_by_depth[k])) if scores_by_depth[k] else None,
                "iqr": [float(np.percentile(scores_by_depth[k], 25)), float(np.percentile(scores_by_depth[k], 75))] if scores_by_depth[k] else None,
                "min": float(np.min(scores_by_depth[k])) if scores_by_depth[k] else None,
                "max": float(np.max(scores_by_depth[k])) if scores_by_depth[k] else None
            }
            for k in range(1, 6)
        },
        "target_age_reliability_states": target_age_reliability_states,
        "proposal_class_reliability": {
            "overlapping_proposals_total": overlapping_proposals_total,
            "overlapping_proposals_correct": overlapping_proposals_correct,
            "overlapping_proposals_wrong_cat": overlapping_proposals_wrong_cat,
            "pct_correct": float((overlapping_proposals_correct / max(1, overlapping_proposals_total)) * 100)
        },
        "nearest_competitor_proximity": {
            "duplicate_like": {
                "events_count": len(nearest_dist_duplicate_competitor),
                "p50": float(np.median(nearest_dist_duplicate_competitor)) if nearest_dist_duplicate_competitor else None,
                "iqr": [float(np.percentile(nearest_dist_duplicate_competitor, 25)), float(np.percentile(nearest_dist_duplicate_competitor, 75))] if nearest_dist_duplicate_competitor else None,
                "p10": float(np.percentile(nearest_dist_duplicate_competitor, 10)) if nearest_dist_duplicate_competitor else None
            },
            "low_overlap": {
                "events_count": len(nearest_dist_low_overlap_competitor),
                "p50": float(np.median(nearest_dist_low_overlap_competitor)) if nearest_dist_low_overlap_competitor else None,
                "iqr": [float(np.percentile(nearest_dist_low_overlap_competitor, 25)), float(np.percentile(nearest_dist_low_overlap_competitor, 75))] if nearest_dist_low_overlap_competitor else None,
                "p10": float(np.percentile(nearest_dist_low_overlap_competitor, 10)) if nearest_dist_low_overlap_competitor else None
            }
        },
        "frame_unique_multiplicity": {
            str(th): {
                "mean": float(np.mean(frame_mult[th])),
                "median": float(np.median(frame_mult[th])),
                "p90": float(np.percentile(frame_mult[th], 90)),
                "p95": float(np.percentile(frame_mult[th], 95)),
                "saturation_rate_p100": float((sum(1 for c in frame_mult[th] if c == 100) / len(frame_mult[th])) * 100)
            }
            for th in [0.10, 0.20, 0.30, 0.50]
        },
        "retention_curves": {
            "unique_target_retention": target_retention_curves,
            "evidence_event_retention": event_retention_curves
        },
        "per_video_design_robustness": per_video_summary
    }

    # D-3565 / P-1665 / P-1667: Idempotent JSON-normalized save or verification
    audit_hash = save_or_verify_json(AUDIT_RESULTS_JSON, audit_payload, "Candidate 2M Development Audit Manifest")
    print("=" * 115)


if __name__ == "__main__":
    main()