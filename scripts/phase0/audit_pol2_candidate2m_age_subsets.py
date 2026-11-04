"""
D:/MARIVISION/H0_workbench/eval_tools/audit_pol2_candidate2m_age_subsets.py
PoL-2M Stage 0B: Temporal Age-Subset Compression Audit on 10 Design Videos

Strict Compliance Checklist:
  - D-3556 / D-3575: Evaluates all 15 non-empty subsets of {2, 3, 4, 5} on the 10 design videos ONLY.
  - D-3580 / P-1676: Certifies exact reproduction of the frozen Stage-0 evidence universe:
                     N_design == 206, marginals {2: 86, 3: 111, 4: 129, 5: 143}, total events == 469.
  - D-3581 / P-1677: Reconstructs age sets A_k^(8) requiring best_rank < 8, guaranteeing Top-8 bank compatibility.
  - D-3560 / P-1660: Derives k strictly from frame differences (k = frame_t - frame_hist).
  - P-1674: Ranks subsets by: Maximize Coverage -> Minimize |S| -> Maximize Min Video Cov -> Maximize Redundancy.
  - P-1665 / P-1667: Enforces idempotent normalized JSON round-trip save via save_or_verify_json().
  - Complete Provenance: Verifies live hashes for candidate windows, train JSON, prediction cache,
                         frozen split manifest (SHA 7da471d2...), and Stage-0 audit manifest (SHA 8cbc81a5...).
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
STAGE0_AUDIT_JSON = OUT_DIR / "pol2_candidate2m_dev_audit.json"
AGE_AUDIT_JSON = OUT_DIR / "pol2_candidate2m_age_subsets.json"

EXPECTED_CANDIDATE_SHA256 = "f29e0b8a5cbfc46641222d84a6d370c3e9bc51ca4ca1124b7753078ca7c1e270"
EXPECTED_TRAIN_JSON_SHA256 = "edc32bdbd6b792eb83b6daf11e14708232f1f4a9e06153ebebd5bad1caef9b35"
EXPECTED_CACHE_SHA256 = "d9816083760d698935b44e9165522e709b490c3af29e710d75b8b0d2d90210e0"
EXPECTED_DEV_POOL_SHA256 = "533567e1d74544db7087a95253e640ef0959b998e5d503f57b0eb8074d8fe970"
EXPECTED_SPLIT_SHA256 = "7da471d2793fe3369508200dec50422bb76492860c67a1a61871645471856772"
EXPECTED_STAGE0_AUDIT_SHA256 = "8cbc81a54491f9d5baac771d2bdabc8d2b1aee81f6a69c233af20370b543435d"

EVAL_IOU_THRESHOLD = 0.30
OPERATING_SCORE_THRESHOLD = 0.20


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
    normalized_payload = json.loads(json.dumps(payload))
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
        assert existing_data == normalized_payload, (
            f"[FATAL] Existing {artifact_name} content mismatch with newly generated payload! Aborting to prevent mutation."
        )
        sha = compute_sha256(file_path)
        print(f"[+] Reusing verified existing frozen {artifact_name}: {file_path.resolve()}")
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
    print("STAGE 0B: CANDIDATE-2M TEMPORAL AGE-SUBSET COMPRESSION AUDIT (10 DESIGN VIDEOS)")
    print("=" * 115)

    # 1. Full Live Provenance Verification (P-1676)
    assert CANDIDATES_JSON.exists(), f"[FATAL] Missing {CANDIDATES_JSON}"
    assert TRAIN_JSON.exists(), f"[FATAL] Missing {TRAIN_JSON}"
    assert CACHE_FILE.exists(), f"[FATAL] Missing {CACHE_FILE}"
    assert DEV_POOL_JSON.exists(), f"[FATAL] Missing {DEV_POOL_JSON}"
    assert SPLIT_MANIFEST_JSON.exists(), f"[FATAL] Missing {SPLIT_MANIFEST_JSON}"
    assert STAGE0_AUDIT_JSON.exists(), f"[FATAL] Missing {STAGE0_AUDIT_JSON}"

    cand_hash = compute_sha256(CANDIDATES_JSON)
    train_hash = compute_sha256(TRAIN_JSON)
    cache_hash = compute_sha256(CACHE_FILE)
    dev_pool_hash = compute_sha256(DEV_POOL_JSON)
    split_hash = compute_sha256(SPLIT_MANIFEST_JSON)
    stage0_hash = compute_sha256(STAGE0_AUDIT_JSON)

    assert cand_hash == EXPECTED_CANDIDATE_SHA256, "[FATAL] Candidate JSON SHA mismatch!"
    assert train_hash == EXPECTED_TRAIN_JSON_SHA256, "[FATAL] Train JSON SHA mismatch!"
    assert cache_hash == EXPECTED_CACHE_SHA256, "[FATAL] Prediction cache SHA mismatch!"
    assert dev_pool_hash == EXPECTED_DEV_POOL_SHA256, "[FATAL] Dev pool manifest SHA mismatch!"
    assert split_hash == EXPECTED_SPLIT_SHA256, "[FATAL] Split manifest SHA mismatch!"
    assert stage0_hash == EXPECTED_STAGE0_AUDIT_SHA256, "[FATAL] Stage-0 audit manifest SHA mismatch!"

    with open(SPLIT_MANIFEST_JSON, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    design_vids = split_data["_meta"]["train_videos"]
    calib_vids = split_data["_meta"]["calib_videos"]
    test_vids = split_data["_meta"]["test_videos"]

    print("[+] All Upstream Artifacts Certified by Immutable SHA-256 (including Stage-0 Audit Manifest).")
    print(f"[+] Frozen Clean Split Loaded (SHA: {EXPECTED_SPLIT_SHA256}):")
    print(f"    - Design Videos (10):      {design_vids} ({split_data['_meta']['n_train_targets']} targets)")
    print(f"    - Calibration Videos (2): {calib_vids} (UNTOUCHED)")
    print(f"    - Fresh Test Videos (2):  {test_vids} (UNTOUCHED)")

    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache_data = json.load(f)
    raw_detections = {int(k): v for k, v in cache_data["detections"].items()}

    coco = COCO(str(TRAIN_JSON))
    ann_lookup = {a["id"]: a for a in coco.loadAnns(coco.getAnnIds())}
    img_lookup = {info["id"]: info for info in coco.loadImgs(coco.getImgIds())}

    with open(CANDIDATES_JSON, "r", encoding="utf-8") as f:
        cand_windows = json.load(f)["windows"]

    # 2. Extract Design MI Targets & Reconstruct Top-8-Compatible Active Ages (P-1677)
    target_active_ages = {}      # aid -> set of active k in {2, 3, 4, 5} satisfying rank < 8
    target_to_video = {}         # aid -> video_id
    video_targets = defaultdict(list)
    reconstructed_events_by_age = {2: 0, 3: 0, 4: 0, 5: 0}

    for w in cand_windows:
        v_id = w["video_id"]
        n_mi = w.get("n_mem_incr_s20", w.get("n_memincr_s20", 0))
        if v_id in design_vids and n_mi >= 1:
            for t in w["targets"]:
                aid = t["ann_id"]
                is_mi = t.get("mem_incr_s20", t.get("memincr_s20", False))
                if is_mi and aid not in target_active_ages:
                    ann = ann_lookup[aid]
                    target_img_id = ann["image_id"]
                    target_frame_id = img_lookup[target_img_id]["frame_id"]

                    active_k = set()
                    chain_ann_ids = t["chain_ann_ids"]

                    for p_aid in chain_ann_ids:
                        assert p_aid in ann_lookup, f"[FATAL] Missing historical annotation {p_aid}"
                        p_ann = ann_lookup[p_aid]
                        p_iid = p_ann["image_id"]
                        assert p_iid in img_lookup, f"[FATAL] Missing historical image {p_iid}"
                        p_fid = img_lookup[p_iid]["frame_id"]

                        k = target_frame_id - p_fid

                        # assert 1 <= k <= 5, f"[FATAL] Invalid age {k} for target {aid}"

                        p_preds = raw_detections.get(p_iid, [])
                        
                        # Find matching detections and verify Top-8 rank compatibility (P-1677)
                        matching_s20 = [
                            (rank, d) for rank, d in enumerate(p_preds)
                            if d["category_id"] == p_ann["category_id"]
                            and box_iou(d["bbox"], p_ann["bbox"]) >= EVAL_IOU_THRESHOLD
                            and d["score"] >= OPERATING_SCORE_THRESHOLD
                        ]
                        

                        if matching_s20:
                            assert k in {2, 3, 4, 5}, (
                                f"[FATAL] Unexpected qualifying t-{k} evidence "
                                f"for MI target {aid}; frozen Stage-0 requires N_t-1 = 0."
                            )
                        if matching_s20:
                            best_rank, best_d = max(matching_s20, key=lambda item: item[1]["score"])
                            assert best_rank < 8, (
                                f"[FATAL] Qualifying evidence event for target {aid} at age {k} fell outside Top-8 (rank {best_rank})!"
                            )
                            active_k.add(k)
                            reconstructed_events_by_age[k] += 1

                    assert len(active_k) >= 1, f"[FATAL] Target {aid} has no qualifying historical evidence!"
                    target_active_ages[aid] = active_k
                    target_to_video[aid] = v_id
                    video_targets[v_id].append(aid)

    N_design = len(target_active_ages)

    # 3. Assert Frozen Stage-0 Reconstruction Invariants (D-3580, CLM-2810, P-1676)
    assert N_design == 206, f"[FATAL] Target count mismatch: {N_design} != 206!"
    assert reconstructed_events_by_age[2] == 86,  f"[FATAL] |A_2| mismatch: {reconstructed_events_by_age[2]} != 86!"
    assert reconstructed_events_by_age[3] == 111, f"[FATAL] |A_3| mismatch: {reconstructed_events_by_age[3]} != 111!"
    assert reconstructed_events_by_age[4] == 129, f"[FATAL] |A_4| mismatch: {reconstructed_events_by_age[4]} != 129!"
    assert reconstructed_events_by_age[5] == 143, f"[FATAL] |A_5| mismatch: {reconstructed_events_by_age[5]} != 143!"
    
    total_reconstructed_events = sum(reconstructed_events_by_age.values())
    assert total_reconstructed_events == 469, f"[FATAL] Total events mismatch: {total_reconstructed_events} != 469!"

    print(f"\n[+] Frozen Stage-0 Aggregate Evidence Invariants Reproduced Exactly (D-3580, P-1676):")
    print(f"    - N_design:       {N_design} / 206 targets (100.0%)")
    print(f"    - Marginal Sets:  |A_2| = {reconstructed_events_by_age[2]}, |A_3| = {reconstructed_events_by_age[3]}, |A_4| = {reconstructed_events_by_age[4]}, |A_5| = {reconstructed_events_by_age[5]}")
    print(f"    - Total Events:   {total_reconstructed_events} / 469 qualifying events (100% verified rank < 8)")

    # 4. Evaluate All 15 Non-Empty Subsets of {2, 3, 4, 5}
    all_ages = [2, 3, 4, 5]
    all_subsets = []
    for r in range(1, 5):
        all_subsets.extend(list(itertools.combinations(all_ages, r)))

    print(f"\n[*] Evaluating all {len(all_subsets)} non-empty subsets of {{2, 3, 4, 5}} under Top-8 bank capacity...")

    subset_results = []

    for S_tuple in all_subsets:
        S = set(S_tuple)
        n_ages = len(S)
        max_tokens = n_ages * 8

        # Unique target coverage
        covered_aids = {aid for aid, ages in target_active_ages.items() if len(ages.intersection(S)) > 0}
        n_covered = len(covered_aids)
        coverage_pct = (n_covered / N_design) * 100.0

        # Multi-age redundancy (targets with >= 2 supporting ages in S)
        n_single_age = sum(1 for aid in covered_aids if len(target_active_ages[aid].intersection(S)) == 1)
        n_multi_age = sum(1 for aid in covered_aids if len(target_active_ages[aid].intersection(S)) >= 2)

        # Evidence events retained
        total_events = sum(len(target_active_ages[aid].intersection(S)) for aid in target_active_ages)

        # Per-video coverage
        video_cov = {}
        for vid in design_vids:
            v_aids = video_targets[vid]
            v_cov = sum(1 for aid in v_aids if aid in covered_aids)
            video_cov[str(vid)] = (v_cov / len(v_aids)) * 100.0 if v_aids else 0.0

        min_vid_cov = min(video_cov.values())

        subset_results.append({
            "subset": list(S_tuple),
            "subset_str": str(list(S_tuple)),
            "num_ages": n_ages,
            "max_tokens": max_tokens,
            "n_covered": n_covered,
            "coverage_pct": coverage_pct,
            "min_video_coverage": min_vid_cov,
            "n_single_age": n_single_age,
            "n_multi_age": n_multi_age,
            "total_events": total_events,
            "per_video_coverage": video_cov
        })

    # Lexicographic Ranking Rule (P-1674):
    # 1. Maximize unique target coverage (n_covered)
    # 2. Minimize number of ages (-num_ages)
    # 3. Maximize minimum video coverage (min_video_coverage)
    # 4. Maximize multi-age redundancy (n_multi_age)
    # 5. Maximize total events (total_events)
    def sort_key(res):
        return (
            res["n_covered"],
            -res["num_ages"],
            res["min_video_coverage"],
            res["n_multi_age"],
            res["total_events"]
        )

    subset_results.sort(key=sort_key, reverse=True)
    optimal_subset = subset_results[0]

    # 5. Print Comprehensive Audit Table
    print("\n" + "=" * 115)
    print("STAGE 0B: TEMPORAL AGE SUBSET COMPRESSION AUDIT TABLE")
    print("Ranked by Lexicographic Selection Rule (P-1674)")
    print("=" * 115)
    print(f"{'Rank':<5} | {'Subset S':<14} | {'Ages':<5} | {'Tokens':<7} | {'Coverage N / 206':<18} | {'Cov %':<8} | {'Min Vid Cov':<12} | {'Multi-Age':<10} | {'Events':<7} | {'Status'}")
    print("-" * 115)

    for rank_idx, res in enumerate(subset_results, 1):
        s_str = res["subset_str"]
        n_a = res["num_ages"]
        m_tok = res["max_tokens"]
        n_c = res["n_covered"]
        c_pct = res["coverage_pct"]
        min_v = res["min_video_coverage"]
        m_age = res["n_multi_age"]
        ev = res["total_events"]

        status = ""
        if res["subset"] == optimal_subset["subset"]:
            status = "<--- OPPORTUNITY-OPTIMAL (P-1674)"
        elif res["subset"] == [3, 5]:
            status = "<--- USER PROPOSAL"

        print(f"{rank_idx:<5d} | {s_str:<14} | {n_a:<5d} | {m_tok:<7d} | {n_c:3d} / 206           | {c_pct:5.1f}%  | {min_v:5.1f}%       | {m_age:<10d} | {ev:<7d} | {status}")

    print("\n" + "=" * 115)
    print(f"[+] Optimal Subset Selected: S* = {optimal_subset['subset']} ({optimal_subset['num_ages']} ages, {optimal_subset['max_tokens']} max tokens)")
    print(f"    Unique Target Coverage:   {optimal_subset['n_covered']} / 206 ({optimal_subset['coverage_pct']:.1f}%)")
    print(f"    Worst-Video Coverage:     {optimal_subset['min_video_coverage']:.1f}%")
    print(f"    Multi-Age Redundancy:     {optimal_subset['n_multi_age']} targets supported by >= 2 ages")
    print("=" * 115)

    # 6. Detailed Inspection of User Proposal {3, 5} vs. Optimal Subset
    res_35 = next(res for res in subset_results if res["subset"] == [3, 5])
    print("\n--- Comparative Breakdown: User Proposal {3, 5} vs. Optimal Subset ---")
    print(f"  • User Proposal {{3, 5}}:")
    print(f"    - Unique Target Coverage: {res_35['n_covered']} / 206 ({res_35['coverage_pct']:.1f}%)")
    print(f"    - Dropped Opportunities:  {206 - res_35['n_covered']} design targets unrecoverable")
    print(f"    - Worst-Video Coverage:   {res_35['min_video_coverage']:.1f}%")
    print(f"    - Total Events Retained:  {res_35['total_events']} / 469")
    print(f"  • Optimal Subset {optimal_subset['subset']}:")
    print(f"    - Unique Target Coverage: {optimal_subset['n_covered']} / 206 ({optimal_subset['coverage_pct']:.1f}%)")
    print(f"    - Dropped Opportunities:  {206 - optimal_subset['n_covered']} design targets unrecoverable")
    print(f"    - Worst-Video Coverage:   {optimal_subset['min_video_coverage']:.1f}%")
    print(f"    - Total Events Retained:  {optimal_subset['total_events']} / 469")

    # Serialize Stage 0B Manifest (P-1665, P-1667)
    manifest_payload = {
        "_meta": {
            "artifact_name": "pol2_candidate2m_age_subsets_v1",
            "role": "POL2_CANDIDATE_2M_TEMPORAL_AGE_COMPRESSION_AUDIT",
            "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
            "stage0_audit_manifest_sha256": EXPECTED_STAGE0_AUDIT_SHA256,
            "design_videos": design_vids,
            "total_design_mi_targets": N_design,
            "optimal_subset": optimal_subset["subset"],
            "optimal_max_tokens": optimal_subset["max_tokens"],
            "optimal_coverage_pct": optimal_subset["coverage_pct"]
        },
        "ranked_subsets": subset_results
    }

    audit_hash = save_or_verify_json(AGE_AUDIT_JSON, manifest_payload, "Candidate 2M Age Subsets Manifest")
    print("=" * 115)


if __name__ == "__main__":
    main()