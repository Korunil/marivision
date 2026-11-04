"""
D:/MARIVISION/H0_workbench/eval_tools/freeze_pol2_selected_clips.py
Phase-II Proof-of-Launch Stage 2 (PoL-2): Programmatic Cohort Optimization & Manifest Freeze

Strict Compliance Checklist:
  - D-3353 / P-1559: Evaluates best 5-video cohort for every sensor with >= 6 eligible videos;
                     selects sensor + cohort by global balanced opportunity score.
  - D-3348: Per-video window selection maximizes (min(FP, MI), FP+MI, Persist, -frame_start).
  - D-3349: Dedicated disjoint calibration clip from remaining same-sensor videos.
  - D-3355 / P-1560: Strict target enrichment assertion; fails hard if annotation ID is missing.
  - D-3356 / P-1561: Self-certifying chronology assertion: asserts range(f0, f0+6) for all clips.
  - P-1558: Asserts candidate windows SHA, live train JSON SHA, M0 SHA, and approved links SHA.
"""

import json
import hashlib
import itertools
from pathlib import Path
from collections import defaultdict
from pycocotools.coco import COCO

OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
CANDIDATES_JSON = OUT_DIR / "pol2_candidate_windows_v1.json"
FROZEN_CLIPS_JSON = OUT_DIR / "pol2_selected_clips.json"
TRAIN_JSON = Path("D:/MARIVISION/seaclips-train.json")

EXPECTED_CANDIDATE_SHA256 = "f29e0b8a5cbfc46641222d84a6d370c3e9bc51ca4ca1124b7753078ca7c1e270"
EXPECTED_TRAIN_JSON_SHA256 = "edc32bdbd6b792eb83b6daf11e14708232f1f4a9e06153ebebd5bad1caef9b35"
EXPECTED_LINKS_SHA256 = "e132f0a5782e0a53da858db80e64db0b879652057eeea9c149af2735cb25a2fb"
EXPECTED_M0_HASH = "28e6288c3f6769c2d5275aa77ddb248735b55a09cd46a6b570e3f117890b268b"


def compute_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def window_rank_key(w):
    balance = min(w["n_fastpath_s20"], w["n_mem_incr_s20"])
    total_s20 = w["n_fastpath_s20"] + w["n_mem_incr_s20"]
    persist = w["n_persist_s20"]
    return (balance, total_s20, persist, -w["frame_start"])


def main():
    print("=" * 115)
    print("PROOF-OF-LAUNCH STAGE 2 (PoL-2): GLOBAL SENSOR+COHORT OPTIMIZER & MANIFEST FREEZE")
    print("=" * 115)

    assert CANDIDATES_JSON.exists(), f"[FATAL] Missing {CANDIDATES_JSON}"
    assert TRAIN_JSON.exists(), f"[FATAL] Missing {TRAIN_JSON}"

    cand_hash = compute_sha256(CANDIDATES_JSON)
    train_hash = compute_sha256(TRAIN_JSON)

    assert cand_hash == EXPECTED_CANDIDATE_SHA256, (
        f"[FATAL] Candidate JSON SHA-256 mismatch!\n"
        f"  Expected: {EXPECTED_CANDIDATE_SHA256}\n"
        f"  Found:    {cand_hash}"
    )
    assert train_hash == EXPECTED_TRAIN_JSON_SHA256, (
        f"[FATAL] Live Train JSON SHA-256 mismatch!\n"
        f"  Expected: {EXPECTED_TRAIN_JSON_SHA256}\n"
        f"  Found:    {train_hash}"
    )

    with open(CANDIDATES_JSON, "r", encoding="utf-8") as f:
        cand_data = json.load(f)

    # Cross-check embedded artifact provenance (P-1558)
    cand_meta = cand_data.get("_meta", {})
    assert cand_meta.get("train_json_sha256") == train_hash, "[FATAL] Embedded train JSON hash mismatch!"
    assert cand_meta.get("links_json_sha256") == EXPECTED_LINKS_SHA256, "[FATAL] Embedded links hash mismatch!"
    assert cand_meta.get("m0_checkpoint_sha256") == EXPECTED_M0_HASH, "[FATAL] Embedded M0 hash mismatch!"

    print(f"[+] All Source Artifact Provenance Gates Passed:")
    print(f"    - Candidate Windows SHA-256: {cand_hash}")
    print(f"    - Live Train JSON SHA-256:   {train_hash}")
    print(f"    - Certified M0 SHA-256:       {EXPECTED_M0_HASH}")
    print(f"    - Approved Links SHA-256:    {EXPECTED_LINKS_SHA256}")

    all_windows = cand_data["windows"]
    coco = COCO(str(TRAIN_JSON))

    # --------------------------------------------------------------------------
    # STEP 1: FILTER TO STRICT JOINT OPPORTUNITY (n_fp >= 1 AND n_mi >= 1)
    # --------------------------------------------------------------------------
    joint_opp_windows = [
        w for w in all_windows
        if w["n_fastpath_s20"] >= 1 and w["n_mem_incr_s20"] >= 1
    ]
    print(f"\n[*] Total Windows with Joint Opportunity (FP_s20 >= 1 AND MI_s20 >= 1): {len(joint_opp_windows):,}")

    sensor_video_map = defaultdict(lambda: defaultdict(list))
    for w in joint_opp_windows:
        sensor_video_map[w["sensor"]][w["video_id"]].append(w)

    print("\n--- Sensor Family Availability with Joint Opportunity ---")
    eligible_sensor_families = []
    for s_name, v_dict in sorted(sensor_video_map.items()):
        n_vids = len(v_dict)
        n_wins = sum(len(wl) for wl in v_dict.values())
        print(f"  Sensor: {s_name:<10} | Distinct Eligible Videos: {n_vids:2d} | Total Joint Windows: {n_wins:4d}")
        if n_vids >= 6:
            eligible_sensor_families.append(s_name)

    assert len(eligible_sensor_families) > 0, (
        "[FATAL] No sensor family has >= 6 distinct videos meeting the joint opportunity criterion!"
    )

    # --------------------------------------------------------------------------
    # STEP 2: GLOBAL SENSOR + COHORT OPTIMIZATION (D-3353, P-1559)
    # --------------------------------------------------------------------------
    print(f"\n[*] Qualifying Sensor Families with >= 6 Eligible Videos: {eligible_sensor_families}")
    print("    Evaluating best achievable 5-video cohort for every qualifying sensor family...")

    best_sensor_result = None

    for sensor_name in sorted(eligible_sensor_families):
        # 1. Best window per video for this sensor (D-3348)
        best_window_per_video_s = {}
        for v_id, win_list in sensor_video_map[sensor_name].items():
            win_list.sort(key=window_rank_key, reverse=True)
            best_window_per_video_s[v_id] = win_list[0]

        vids = sorted(best_window_per_video_s.keys())
        all_combos = list(itertools.combinations(vids, 5))

        # 2. Exhaustive combinatorial search over binom(V, 5)
        for combo in all_combos:
            sum_fp = sum(best_window_per_video_s[v]["n_fastpath_s20"] for v in combo)
            sum_mi = sum(best_window_per_video_s[v]["n_mem_incr_s20"] for v in combo)
            sum_persist = sum(best_window_per_video_s[v]["n_persist_s20"] for v in combo)
            balance = min(sum_fp, sum_mi)
            total_s20 = sum_fp + sum_mi
            
            score = (balance, total_s20, sum_persist, tuple(-v for v in sorted(combo)))
            record = (score, sensor_name, combo, best_window_per_video_s)

            if (
                best_sensor_result is None
                or score > best_sensor_result[0]
                or (score == best_sensor_result[0] and sensor_name < best_sensor_result[1])
            ):
                best_sensor_result = record

    best_score, selected_sensor, best_combo_vids, best_window_per_video = best_sensor_result
    eval_clips = [best_window_per_video[v] for v in sorted(best_combo_vids)]

    print(f"\n[+] Global Optimization Complete:")
    print(f"    - Selected Sensor Family: {selected_sensor}")
    print(f"    - Optimal 5-Video Cohort: {sorted(best_combo_vids)}")
    print(f"    - Cohort Balanced Score:  min(sum_FP, sum_MI)={best_score[0]} | Total S20={best_score[1]} | Persist={best_score[2]}")

    # --------------------------------------------------------------------------
    # STEP 3: SELECT DISJOINT CALIBRATION CLIP (D-3349)
    # --------------------------------------------------------------------------
    remaining_vids = [v for v in sorted(best_window_per_video.keys()) if v not in best_combo_vids]
    assert len(remaining_vids) >= 1, "[FATAL] No remaining video for calibration!"

    remaining_windows = [best_window_per_video[v] for v in remaining_vids]
    remaining_windows.sort(key=lambda w: (window_rank_key(w), -w["video_id"]), reverse=True)
    calib_clip = remaining_windows[0]

    # --------------------------------------------------------------------------
    # STEP 4: STRICT TARGET ENRICHMENT ASSERTION (D-3355, P-1560)
    # --------------------------------------------------------------------------
    def enrich_clip_targets(clip_record):
        final_iid = clip_record["image_ids"][-1]
        anns_t = {a["id"]: a for a in coco.loadAnns(coco.getAnnIds(imgIds=[final_iid]))}
        for t in clip_record["targets"]:
            aid = t["ann_id"]
            assert aid in anns_t, f"[FATAL] Target annotation {aid} is not present in final image {final_iid}"
            t["bbox"] = anns_t[aid]["bbox"]
            t["category_id"] = anns_t[aid]["category_id"]

    for c in eval_clips:
        enrich_clip_targets(c)
    enrich_clip_targets(calib_clip)

    # --------------------------------------------------------------------------
    # STEP 5: PRE-SERIALIZATION INVARIANT & CHRONOLOGY ASSERTIONS (D-3356, P-1561)
    # --------------------------------------------------------------------------
    print("\n--- STEP 5: VERIFYING INVARIANTS BEFORE MANIFEST FREEZE ---")
    eval_vids = {c["video_id"] for c in eval_clips}
    assert len(eval_vids) == 5, f"[FATAL] Expected 5 distinct eval videos, got {len(eval_vids)}"
    assert calib_clip["video_id"] not in eval_vids, f"[FATAL] Calibration video {calib_clip['video_id']} overlaps with eval clips!"

    for idx, c in enumerate(eval_clips):
        assert c["sensor"] == selected_sensor, f"[FATAL] Eval clip {idx+1} sensor mismatch: {c['sensor']} != {selected_sensor}"
        assert c["n_fastpath_s20"] >= 1, f"[FATAL] Eval clip {idx+1} has 0 fastpath targets!"
        assert c["n_mem_incr_s20"] >= 1, f"[FATAL] Eval clip {idx+1} has 0 memory-incremental targets!"

    assert calib_clip["sensor"] == selected_sensor, "[FATAL] Calibration clip sensor mismatch!"

    # Self-certifying 6-frame consecutive chronology check (D-3356, P-1561)
    for c in eval_clips + [calib_clip]:
        assert len(c["frame_ids"]) == 6, f"[FATAL] Clip {c['video_id']} does not have 6 frame IDs"
        assert len(c["image_ids"]) == 6, f"[FATAL] Clip {c['video_id']} does not have 6 image IDs"
        f0 = c["frame_ids"][0]
        assert c["frame_ids"] == list(range(f0, f0 + 6)), (
            f"[FATAL] Non-consecutive frozen window for video {c['video_id']}: {c['frame_ids']}"
        )

    print("[+] All Invariants Certified: 5 Unique Eval Videos, 1 Disjoint Calib Video, Same Sensor, Dual Opp >= 1, Strict 6-Frame Chronology.")

    # --------------------------------------------------------------------------
    # STEP 6: DISPLAY ACTUAL EXTRACTED TARGET TOTALS (D-3340)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print(f"FROZEN 5-CLIP EVALUATION TESTBED ({selected_sensor} Sensor Domain)")
    print("=" * 115)
    print(f"{'Clip #':<8} | {'Video ID':<10} | {'Frames (t-5 -> t)':<20} | {'Sensor':<8} | {'FP (s20)':<10} | {'MI (s20)':<10} | {'P>=2 (s20)':<12} | {'Target Categories'}")
    print("-" * 115)

    tot_fp = 0
    tot_mi = 0
    tot_p2 = 0
    cat_distribution = defaultdict(int)

    for idx, c in enumerate(eval_clips):
        tot_fp += c["n_fastpath_s20"]
        tot_mi += c["n_mem_incr_s20"]
        tot_p2 += c["n_persist_s20"]

        f_range = f"{c['frame_start']} -> {c['frame_end']}"
        cats_in_c = set()
        for t in c["targets"]:
            if t["fastpath_s20"] or t["mem_incr_s20"]:
                cats_in_c.add(t["category"])
                cat_distribution[t["category"]] += 1

        cat_str = ", ".join(sorted(cats_in_c))
        print(f"Clip {idx+1:<3} | video_{c['video_id']:<4d} | {f_range:<20} | {c['sensor']:<8} | {c['n_fastpath_s20']:<10} | {c['n_mem_incr_s20']:<10} | {c['n_persist_s20']:<12} | {cat_str}")

    print("-" * 115)
    print(f"{'TOTALS':<8} | {'5 Videos':<10} | {'30 Frames Total':<20} | {selected_sensor:<8} | {tot_fp:<10} | {tot_mi:<10} | {tot_p2:<12} | {tot_fp + tot_mi} Predeclared Targets")

    print("\n--- Dedicated Step-Budget Calibration Window (Excluded from Evaluation) ---")
    calib_f_range = f"{calib_clip['frame_start']} -> {calib_clip['frame_end']}"
    print(f"  Calib Clip | video_{calib_clip['video_id']:<4d} | {calib_f_range:<20} | {calib_clip['sensor']:<8} | FP_s20 = {calib_clip['n_fastpath_s20']} | MI_s20 = {calib_clip['n_mem_incr_s20']} | P>=2_s20 = {calib_clip['n_persist_s20']}")

    # --------------------------------------------------------------------------
    # STEP 7: SERIALIZE FROZEN MANIFEST & COMPUTE HASH (D-3330)
    # --------------------------------------------------------------------------
    master_frozen_payload = {
        "_meta": {
            "artifact_name": "pol2_selected_clips_frozen_v1",
            "candidate_windows_sha256": cand_hash,
            "train_json_sha256": train_hash,
            "m0_checkpoint_sha256": EXPECTED_M0_HASH,
            "links_json_sha256": EXPECTED_LINKS_SHA256,
            "selection_policy": "global_sensor_and_cohort_optimization",
            "sensor_family": selected_sensor,
            "cohort_optimization_objective": "lexicographic(min(sum_FP, sum_MI), sum_FP + sum_MI, sum_Persist, -vids)",
            "total_eval_clips": len(eval_clips),
            "total_eval_frames": len(eval_clips) * 6,
            "total_predeclared_eval_targets": tot_fp + tot_mi,
            "actual_fastpath_targets_s20": tot_fp,
            "actual_memory_incremental_targets_s20": tot_mi,
            "actual_persistent_ge2_targets_s20": tot_p2,
            "target_category_breakdown": dict(cat_distribution)
        },
        "evaluation_clips": eval_clips,
        "calibration_clip": calib_clip
    }

    with open(FROZEN_CLIPS_JSON, "w", encoding="utf-8") as f:
        json.dump(master_frozen_payload, f, indent=2)

    frozen_hash = compute_sha256(FROZEN_CLIPS_JSON)
    print("\n" + "=" * 115)
    print(f"[+] PoL-2 Manifest Frozen: {FROZEN_CLIPS_JSON.resolve()}")
    print(f"    Immutable Manifest SHA-256: {frozen_hash}")
    print("=" * 115)


if __name__ == "__main__":
    main()