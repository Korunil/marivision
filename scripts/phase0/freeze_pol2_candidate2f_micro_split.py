"""
D:/MARIVISION/H0_workbench/eval_tools/freeze_pol2_candidate2f_micro_split.py
PoL-2F Pre-Flight: Deterministic 8/2/2 Development Video Partition & Manifest Freeze

Strict Compliance Checklist:
  - D-3481 / P-1612: Programmatically partitions the 12 candidate-eligible e-CAM FP videos
                     into 8 Train, 2 Calibration, and 2 Optimization-Untouched Development Test videos.
  - D-3486: Assigns all hash variables before assertions and serialization.
  - D-3487: Removes arbitrary N >= 20 target filters.
  - D-3494 / D-3495 / P-1622: Complete 3-subset L1 target proportionality:
      target_dev = |N_train - 156.67| + |N_calib - 39.17| + |N_test - 39.17|
  - D-3496 / P-1623: Strict pre-serialization partition invariants: 8/2/2 counts, mutual disjointness,
                     exhaustive union == 12 videos, total targets == 235.
  - P-1624: Component-wise proportionality report (actual vs. nominal L1 targets).
  - P-1618: Category overlap diagnostics (C_calib - C_train, C_test - C_train).
  - Complete Provenance: Verifies live hashes for train JSON, M0 checkpoint, links, candidate windows,
                         dev pool manifest, and exact 20,310 image ID set equality in M0 cache.
"""

import json
import hashlib
import itertools
from pathlib import Path
from collections import defaultdict
import numpy as np
from pycocotools.coco import COCO

OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
CANDIDATES_JSON = OUT_DIR / "pol2_candidate_windows_v1.json"
TRAIN_JSON = Path("D:/MARIVISION/seaclips-train.json")
CACHE_FILE = OUT_DIR / "cache_m0_train_raw_top100.json"
LINKS_JSON = OUT_DIR / "pol0_pseudo_links_v1.json"
COHORT_S_JSON = OUT_DIR / "pol2_primary_same_sensor_clips.json"
COHORT_X_JSON = OUT_DIR / "pol2_secondary_cross_sensor_clips.json"
DEV_POOL_JSON = OUT_DIR / "pol2_dev_pool_manifest.json"
M0_CKPT_PATH = Path("D:/MARIVISION/checkpoints_r0/checkpoint_best.pth")
SPLIT_MANIFEST_JSON = OUT_DIR / "pol2_candidate2f_micro_split.json"

EXPECTED_CANDIDATE_SHA256 = "f29e0b8a5cbfc46641222d84a6d370c3e9bc51ca4ca1124b7753078ca7c1e270"
EXPECTED_TRAIN_JSON_SHA256 = "edc32bdbd6b792eb83b6daf11e14708232f1f4a9e06153ebebd5bad1caef9b35"
EXPECTED_LINKS_SHA256 = "e132f0a5782e0a53da858db80e64db0b879652057eeea9c149af2735cb25a2fb"
EXPECTED_COHORT_S_SHA256 = "b157b11e0b1bb8f9a77729a4999d48693f5d1e0cb052d8423b5a8dc903f029dc"
EXPECTED_COHORT_X_SHA256 = "91afc4a0a82e0fc87d9d523a902023e19fe5ac9b8c88b186e359bbb88303dac9"
EXPECTED_DEV_POOL_SHA256 = "533567e1d74544db7087a95253e640ef0959b998e5d503f57b0eb8074d8fe970"
EXPECTED_M0_HASH = "28e6288c3f6769c2d5275aa77ddb248735b55a09cd46a6b570e3f117890b268b"


def compute_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 115)
    print("PROOF-OF-LAUNCH STAGE 2 (PoL-2F): DETERMINISTIC 8/2/2 DEVELOPMENT VIDEO PARTITION")
    print("=" * 115)

    # 1. Full Live Provenance Verification (D-3486)
    assert CANDIDATES_JSON.exists(), f"[FATAL] Missing {CANDIDATES_JSON}"
    assert TRAIN_JSON.exists(), f"[FATAL] Missing {TRAIN_JSON}"
    assert CACHE_FILE.exists(), f"[FATAL] Missing {CACHE_FILE}"
    assert LINKS_JSON.exists(), f"[FATAL] Missing {LINKS_JSON}"
    assert COHORT_S_JSON.exists(), f"[FATAL] Missing {COHORT_S_JSON}"
    assert COHORT_X_JSON.exists(), f"[FATAL] Missing {COHORT_X_JSON}"
    assert DEV_POOL_JSON.exists(), f"[FATAL] Missing {DEV_POOL_JSON}"
    assert M0_CKPT_PATH.exists(), f"[FATAL] Missing {M0_CKPT_PATH}"

    cand_hash = compute_sha256(CANDIDATES_JSON)
    train_hash = compute_sha256(TRAIN_JSON)
    links_hash = compute_sha256(LINKS_JSON)
    cohort_s_hash = compute_sha256(COHORT_S_JSON)
    cohort_x_hash = compute_sha256(COHORT_X_JSON)
    dev_pool_hash = compute_sha256(DEV_POOL_JSON)
    m0_hash = compute_sha256(M0_CKPT_PATH)

    assert cand_hash == EXPECTED_CANDIDATE_SHA256, "[FATAL] Candidate JSON SHA mismatch!"
    assert train_hash == EXPECTED_TRAIN_JSON_SHA256, "[FATAL] Train JSON SHA mismatch!"
    assert links_hash == EXPECTED_LINKS_SHA256, "[FATAL] Links JSON SHA mismatch!"
    assert cohort_s_hash == EXPECTED_COHORT_S_SHA256, "[FATAL] Cohort S SHA mismatch!"
    assert cohort_x_hash == EXPECTED_COHORT_X_SHA256, "[FATAL] Cohort X SHA mismatch!"
    assert dev_pool_hash == EXPECTED_DEV_POOL_SHA256, "[FATAL] Dev pool manifest SHA mismatch!"
    assert m0_hash == EXPECTED_M0_HASH, "[FATAL] M0 checkpoint SHA mismatch!"

    # Cache provenance certification & exact 20,310 set equality check
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache_data = json.load(f)
    assert cache_data["_meta"]["m0_checkpoint_sha256"] == EXPECTED_M0_HASH, "[FATAL] M0 checkpoint SHA mismatch in cache!"

    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        train_raw = json.load(f)
    train_image_ids = {int(x["id"]) for x in train_raw["images"]}
    raw_detections = {int(k): v for k, v in cache_data["detections"].items()}
    assert len(train_image_ids) == 20310
    assert set(raw_detections.keys()) == train_image_ids, "[FATAL] Cache image ID set does not match TRAIN JSON!"
    live_cache_hash = compute_sha256(CACHE_FILE)

    print("[+] All Upstream Manifests & Checkpoints Certified by Immutable SHA-256.")
    print(f"[+] Prediction Cache Verified: 20,310 Set Equality Certified | Live SHA: {live_cache_hash}")

    # 2. Extract Eligible e-CAM 2F Population
    with open(DEV_POOL_JSON, "r", encoding="utf-8") as f:
        dev_manifest = json.load(f)

    ecam_2f_vids = [
        r["video_id"] for r in dev_manifest["development_videos"]
        if r["sensor"] == "e-CAM" and r["supports_2f"]
    ]
    ecam_2f_vids.sort()
    assert len(ecam_2f_vids) == 12, f"[FATAL] Expected exactly 12 e-CAM 2F videos, found {len(ecam_2f_vids)}"
    print(f"\n[*] Eligible e-CAM 2F Population ({len(ecam_2f_vids)} videos): {ecam_2f_vids}")

    with open(CANDIDATES_JSON, "r", encoding="utf-8") as f:
        cand_windows = json.load(f)["windows"]

    coco = COCO(str(TRAIN_JSON))
    ann_lookup = {a["id"]: a for a in coco.loadAnns(coco.getAnnIds())}
    cats = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}

    # Aggregate unique FP targets, categories, and frame image IDs per video
    video_fp_targets = defaultdict(dict)
    video_categories = defaultdict(set)
    video_image_ids = defaultdict(set)

    for w in cand_windows:
        v_id = w["video_id"]
        if v_id in ecam_2f_vids and w["n_fastpath_s20"] >= 1:
            for t in w["targets"]:
                aid = t["ann_id"]
                if t["fastpath_s20"] and aid not in video_fp_targets[v_id]:
                    ann = ann_lookup[aid]
                    c_name = cats[ann["category_id"]]
                    video_fp_targets[v_id][aid] = {
                        "ann_id": aid,
                        "category_id": ann["category_id"],
                        "category": c_name,
                        "bbox": ann["bbox"],
                        "frame_id": w["frame_ids"][-1],
                        "image_id": w["image_ids"][-1],
                        "chain_ann_ids": t["chain_ann_ids"]
                    }
                    video_categories[v_id].add(c_name)
                    video_image_ids[v_id].add(w["image_ids"][-1])

    total_unique_fp = sum(len(tgts) for tgts in video_fp_targets.values())
    assert total_unique_fp == 235, f"[FATAL] Expected 235 unique FP targets, got {total_unique_fp}"
    print(f"[+] Verified 235 unique Fast-Path targets across the 12 e-CAM videos.")

    print("\n--- Per-Video Fast-Path Opportunity Profile ---")
    print(f"{'Video ID':<10} | {'FP Targets':<12} | {'Unique Frames':<14} | {'Target Categories'}")
    print("-" * 75)
    for v_id in ecam_2f_vids:
        cat_str = ", ".join(sorted(video_categories[v_id]))
        n_t = len(video_fp_targets[v_id])
        n_img = len(video_image_ids[v_id])
        print(f"video_{v_id:<4d} | {n_t:<12d} | {n_img:<14d} | {cat_str}")

    # 3. Exhaustive Combinatorial Search for Complete 8/2/2 L1 Proportional Partition (D-3494, D-3495, P-1622)
    print("\n[*] Evaluating all 2,970 partitions of 12 videos into (8 Train, 2 Calib, 2 Test)...")
    # Complete 3-subset nominal target targets (D-3495)
    IDEAL_TRAIN = 235.0 * (8.0 / 12.0)  # 156.6667
    IDEAL_CALIB = 235.0 * (2.0 / 12.0)  # 39.1667
    IDEAL_TEST = 235.0 * (2.0 / 12.0)   # 39.1667

    best_partition = None
    all_combos_test = list(itertools.combinations(ecam_2f_vids, 2))

    for test_vids in all_combos_test:
        test_set = set(test_vids)
        rem_vids = [v for v in ecam_2f_vids if v not in test_set]

        n_test = sum(len(video_fp_targets[v]) for v in test_vids)
        cats_test = set.union(*[video_categories[v] for v in test_vids])

        for calib_vids in itertools.combinations(rem_vids, 2):
            calib_set = set(calib_vids)
            train_vids = tuple(v for v in rem_vids if v not in calib_set)

            n_calib = sum(len(video_fp_targets[v]) for v in calib_vids)
            cats_calib = set.union(*[video_categories[v] for v in calib_vids])

            n_train = sum(len(video_fp_targets[v]) for v in train_vids)
            cats_train = set.union(*[video_categories[v] for v in train_vids])

            # P-1622: Complete three-subset L1 target proportionality
            target_dev = (
                abs(n_train - IDEAL_TRAIN)
                + abs(n_calib - IDEAL_CALIB)
                + abs(n_test - IDEAL_TEST)
            )

            # Secondary is category breadth in evaluation subsets
            min_eval_cats = min(len(cats_test), len(cats_calib))
            total_eval_cats = len(cats_test) + len(cats_calib)

            # Lexicographic score: complete L1 proportionality first, category diversity second, deterministic ID tie-breaker
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

    assert best_partition is not None, "[FATAL] Failed to identify valid 8/2/2 partition!"

    v_train = sorted(list(best_partition["train_vids"]))
    v_calib = sorted(list(best_partition["calib_vids"]))
    v_test = sorted(list(best_partition["test_vids"]))

    # 4. Strict Pre-Serialization Partition Invariants (D-3496, P-1623)
    assert len(v_train) == 8, f"[FATAL] Train video count: {len(v_train)} != 8"
    assert len(v_calib) == 2, f"[FATAL] Calib video count: {len(v_calib)} != 2"
    assert len(v_test) == 2, f"[FATAL] Test video count: {len(v_test)} != 2"
    assert set(v_train).isdisjoint(v_calib), "[FATAL] Train and Calib sets are not disjoint!"
    assert set(v_train).isdisjoint(v_test), "[FATAL] Train and Test sets are not disjoint!"
    assert set(v_calib).isdisjoint(v_test), "[FATAL] Calib and Test sets are not disjoint!"
    assert set(v_train) | set(v_calib) | set(v_test) == set(ecam_2f_vids), "[FATAL] Partition union != all 12 e-CAM videos!"
    assert (
        best_partition["n_train"]
        + best_partition["n_calib"]
        + best_partition["n_test"]
        == 235
    ), f"[FATAL] Sum of subset targets != 235: {best_partition['n_train']} + {best_partition['n_calib']} + {best_partition['n_test']}"

    print(f"\n[+] Strict Partition Invariants Certified (D-3496, P-1623): 8/2/2 Video Sets Disjoint, Exhaustive, Total Targets = 235.")

    # 5. Component-Wise Proportionality Report (P-1624)
    dev_train = abs(best_partition["n_train"] - IDEAL_TRAIN)
    dev_calib = abs(best_partition["n_calib"] - IDEAL_CALIB)
    dev_test = abs(best_partition["n_test"] - IDEAL_TEST)

    print(f"\n--- Component-Wise Proportionality Report (P-1624) ---")
    print(f"  Training:   {best_partition['n_train']:3d} targets  (Nominal: {IDEAL_TRAIN:6.2f} | Absolute Dev: {dev_train:5.2f})")
    print(f"  Calib:      {best_partition['n_calib']:3d} targets  (Nominal: {IDEAL_CALIB:6.2f} | Absolute Dev: {dev_calib:5.2f})")
    print(f"  Test:       {best_partition['n_test']:3d} targets  (Nominal: {IDEAL_TEST:6.2f} | Absolute Dev: {dev_test:5.2f})")
    print(f"  Total Three-Subset L1 Deviation: {best_partition['target_dev']:.2f} targets")

    print(f"\n[+] Optimal 8/2/2 Partition Selected Deterministically:")
    print(f"    - Training Set (8 Videos):        {v_train} ({best_partition['n_train']} targets, {len(best_partition['cats_train'])} categories)")
    print(f"    - Calibration Set (2 Videos):     {v_calib} ({best_partition['n_calib']} targets, {len(best_partition['cats_calib'])} categories)")
    print(f"    - Held-Out Test Set (2 Videos):   {v_test} ({best_partition['n_test']} targets, {len(best_partition['cats_test'])} categories)")

    # 6. Category Representation Diagnostic (P-1618)
    missing_in_train_from_calib = best_partition["cats_calib"] - best_partition["cats_train"]
    missing_in_train_from_test = best_partition["cats_test"] - best_partition["cats_train"]

    print(f"\n--- Category Representation Diagnostic (P-1618) ---")
    print(f"Training Categories ({len(best_partition['cats_train'])}):   {sorted(list(best_partition['cats_train']))}")
    print(f"Calibration Categories ({len(best_partition['cats_calib'])}):{sorted(list(best_partition['cats_calib']))}")
    print(f"Test Categories ({len(best_partition['cats_test'])}):       {sorted(list(best_partition['cats_test']))}")
    print(f"Categories in Calibration missing from Train: {sorted(list(missing_in_train_from_calib)) or 'None (100% covered)'}")
    print(f"Categories in Test missing from Train:        {sorted(list(missing_in_train_from_test)) or 'None (100% covered)'}")

    # 7. Display Partition Tables
    def format_subset_table(subset_name, vids):
        print(f"\n--- {subset_name} ---")
        print(f"{'Video ID':<10} | {'FP Targets':<12} | {'Unique Frames':<14} | {'Target Categories'}")
        print("-" * 75)
        for v_id in vids:
            cat_str = ", ".join(sorted(video_categories[v_id]))
            n_t = len(video_fp_targets[v_id])
            n_img = len(video_image_ids[v_id])
            print(f"video_{v_id:<4d} | {n_t:<12d} | {n_img:<14d} | {cat_str}")

    format_subset_table("TRAINING POOL (8 Videos)", v_train)
    format_subset_table("CALIBRATION SET (2 Videos - Frozen E* Budget Selection)", v_calib)
    format_subset_table("OPTIMIZATION-UNTOUCHED TEST SET (2 Videos - Final Causal Falsification)", v_test)

    # 8. Compile Target Manifest with Full Frame Lineage
    calib_targets = []
    calib_image_ids = set()
    for v in v_calib:
        calib_targets.extend(video_fp_targets[v].values())
        calib_image_ids.update(video_image_ids[v])

    test_targets = []
    test_image_ids = set()
    for v in v_test:
        test_targets.extend(video_fp_targets[v].values())
        test_image_ids.update(video_image_ids[v])

    manifest_payload = {
        "_meta": {
            "artifact_name": "pol2_candidate2f_micro_split_v1",
            "role": "POL2_CANDIDATE_2F_LEARNED_MICRO_FALSIFICATION_SPLIT",
            "candidate_windows_sha256": cand_hash,
            "train_json_sha256": train_hash,
            "m0_checkpoint_sha256": EXPECTED_M0_HASH,
            "links_json_sha256": EXPECTED_LINKS_SHA256,
            "dev_pool_manifest_sha256": EXPECTED_DEV_POOL_SHA256,
            "m0_prediction_cache_sha256": live_cache_hash,
            "total_candidate_videos": len(ecam_2f_vids),
            "total_fastpath_targets": total_unique_fp,
            "selection_policy": "complete_three_subset_l1_proportionality_exhaustive_2970_partition",
            "ideal_train_targets": IDEAL_TRAIN,
            "ideal_calib_targets": IDEAL_CALIB,
            "ideal_test_targets": IDEAL_TEST,
            "target_ratio_l1_deviation": best_partition["target_dev"],
            "train_videos_count": len(v_train),
            "calib_videos_count": len(v_calib),
            "test_videos_count": len(v_test),
            "n_train_targets": best_partition["n_train"],
            "n_calib_targets": best_partition["n_calib"],
            "n_test_targets": best_partition["n_test"],
            "n_calib_unique_frames": len(calib_image_ids),
            "n_test_unique_frames": len(test_image_ids),
            "train_videos": v_train,
            "calib_videos": v_calib,
            "test_videos": v_test
        },
        "calibration_targets": calib_targets,
        "test_targets": test_targets
    }

    with open(SPLIT_MANIFEST_JSON, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)

    split_hash = compute_sha256(SPLIT_MANIFEST_JSON)
    print("\n" + "=" * 115)
    print(f"[+] Candidate 2F Micro-Split Manifest Frozen: {SPLIT_MANIFEST_JSON.resolve()}")
    print(f"    Immutable Manifest SHA-256:               {split_hash}")
    print("=" * 115)


if __name__ == "__main__":
    main()