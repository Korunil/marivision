"""
D:/MARIVISION/H0_workbench/eval_tools/freeze_pol2_dual_cohorts.py
Phase-II Proof-of-Launch Stage 2 (PoL-2): Deterministic Dual-Cohort Manifest Freeze

Strict Compliance Checklist:
  - Cohort S (Primary Causal Falsification):
      * 100% same-sensor (e-CAM) to control for camera/ISP mismatch in Condition B.
      * Exactly 4 strict-joint videos + 1 cohort-conditional balancing video + 1 disjoint calib video.
      * Asserts (FP_s20 + MI_s20) >= 1 for both the 5th eval clip and calib clip.
  - Cohort X (Heterogeneous Stress Replication):
      * 5 of 6 global strict dual-opportunity videos across all sensors (4 e-CAM, 1 Axis, 1 FLIR).
      * Exhaustively optimized over binom(6, 5) = 6 combinations for aggregate balance.
      * 6th strict-joint video reserved as a secondary holdout.
  - Strict Target Enrichment: Asserts that every target annotation exists in the final frame.
  - Chronology: Asserts that every clip consists of exactly 6 strictly consecutive frames.
  - Provenance: Asserts candidate artifact, live train JSON, M0 checkpoint, and links hashes.
"""

import json
import hashlib
import copy
import itertools
from pathlib import Path
from collections import defaultdict
from pycocotools.coco import COCO

OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
CANDIDATES_JSON = OUT_DIR / "pol2_candidate_windows_v1.json"
TRAIN_JSON = Path("D:/MARIVISION/seaclips-train.json")

COHORT_S_JSON = OUT_DIR / "pol2_primary_same_sensor_clips.json"
COHORT_X_JSON = OUT_DIR / "pol2_secondary_cross_sensor_clips.json"

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


def window_rank_key_joint(w):
    balance = min(w["n_fastpath_s20"], w["n_mem_incr_s20"])
    total_s20 = w["n_fastpath_s20"] + w["n_mem_incr_s20"]
    persist = w["n_persist_s20"]
    return (balance, total_s20, persist, -w["frame_start"])


def enrich_targets_strictly(clip_record, coco):
    final_iid = clip_record["image_ids"][-1]
    anns_t = {a["id"]: a for a in coco.loadAnns(coco.getAnnIds(imgIds=[final_iid]))}
    for t in clip_record["targets"]:
        aid = t["ann_id"]
        assert aid in anns_t, f"[FATAL] Target annotation {aid} missing from image {final_iid}"
        t["bbox"] = anns_t[aid]["bbox"]
        t["category_id"] = anns_t[aid]["category_id"]


def assert_clip_chronology(clip_list):
    for c in clip_list:
        assert len(c["frame_ids"]) == 6, f"[FATAL] Clip {c['video_id']} frame count != 6"
        assert len(c["image_ids"]) == 6, f"[FATAL] Clip {c['video_id']} image count != 6"
        f0 = c["frame_ids"][0]
        assert c["frame_ids"] == list(range(f0, f0 + 6)), f"[FATAL] Non-consecutive frames in video {c['video_id']}"


def main():
    print("=" * 115)
    print("PROOF-OF-LAUNCH STAGE 2 (PoL-2): DUAL-COHORT MANIFEST FREEZE (COHORT S & COHORT X)")
    print("=" * 115)

    assert CANDIDATES_JSON.exists(), f"[FATAL] Missing {CANDIDATES_JSON}"
    assert TRAIN_JSON.exists(), f"[FATAL] Missing {TRAIN_JSON}"

    cand_hash = compute_sha256(CANDIDATES_JSON)
    train_hash = compute_sha256(TRAIN_JSON)

    assert cand_hash == EXPECTED_CANDIDATE_SHA256, "[FATAL] Candidate JSON SHA-256 mismatch!"
    assert train_hash == EXPECTED_TRAIN_JSON_SHA256, "[FATAL] Live Train JSON SHA-256 mismatch!"

    with open(CANDIDATES_JSON, "r", encoding="utf-8") as f:
        cand_data = json.load(f)

    cand_meta = cand_data.get("_meta", {})
    assert cand_meta.get("train_json_sha256") == train_hash
    assert cand_meta.get("links_json_sha256") == EXPECTED_LINKS_SHA256
    assert cand_meta.get("m0_checkpoint_sha256") == EXPECTED_M0_HASH

    print(f"[+] All Source Artifact Provenance Gates Certified.")

    all_windows = cand_data["windows"]
    coco = COCO(str(TRAIN_JSON))

    # Identify all strict dual-opportunity windows across dataset
    joint_windows_all = [w for w in all_windows if w["n_fastpath_s20"] >= 1 and w["n_mem_incr_s20"] >= 1]
    sensor_joint_map = defaultdict(lambda: defaultdict(list))
    for w in joint_windows_all:
        sensor_joint_map[w["sensor"]][w["video_id"]].append(w)

    print(f"\n[*] Global Strict Dual-Opportunity Population: {len(joint_windows_all)} windows across {sum(len(v) for v in sensor_joint_map.values())} videos.")
    for s, v_dict in sorted(sensor_joint_map.items()):
        print(f"    - Sensor '{s}': {len(v_dict)} distinct videos ({sum(len(w) for w in v_dict.values())} windows)")

    # ==========================================================================
    # PART 1: COHORT S (SAME-SENSOR PRIMARY FALSIFICATION COHORT)
    # ==========================================================================
    print("\n" + "=" * 115)
    print("GENERATING COHORT S: SAME-SENSOR PRIMARY FALSIFICATION TESTBED")
    print("=" * 115)

    # 1. Programmatically select sensor family with >= 4 joint videos
    qualifying_sensors = [s for s, v_dict in sensor_joint_map.items() if len(v_dict) >= 4]
    assert len(qualifying_sensors) == 1, f"Expected exactly 1 qualifying sensor family, got {qualifying_sensors}"
    sensor_s = qualifying_sensors[0]

    # Select best window for each of the 4 joint videos in sensor_s
    s_joint_vids = sorted(list(sensor_joint_map[sensor_s].keys()))
    assert len(s_joint_vids) == 4, (
        f"[FATAL] Cohort S design requires exactly 4 strict-joint {sensor_s} videos under frozen candidate artifact; "
        f"found {len(s_joint_vids)}: {s_joint_vids}"
    )

    s_joint_clips = []
    for v_id in s_joint_vids:
        win_list = sensor_joint_map[sensor_s][v_id]
        win_list.sort(key=window_rank_key_joint, reverse=True)
        s_joint_clips.append(copy.deepcopy(win_list[0]))

    sum_fp_4 = sum(c["n_fastpath_s20"] for c in s_joint_clips)
    sum_mi_4 = sum(c["n_mem_incr_s20"] for c in s_joint_clips)
    sum_p2_4 = sum(c["n_persist_s20"] for c in s_joint_clips)

    print(f"[*] Base 4 Joint Clips ({sensor_s}): FP_s20={sum_fp_4} | MI_s20={sum_mi_4} | P>=2_s20={sum_p2_4}")

    # 2. Select 5th evaluation clip to optimize aggregate cohort balance (P-1563)
    remaining_s_windows = [
        w for w in all_windows
        if w["sensor"] == sensor_s and w["video_id"] not in s_joint_vids
        and (w["n_fastpath_s20"] + w["n_mem_incr_s20"] >= 1)
    ]
    assert len(remaining_s_windows) > 0, f"No remaining {sensor_s} windows with s20 opportunity!"

    def cohort_balance_score(candidate_w):
        tot_fp = sum_fp_4 + candidate_w["n_fastpath_s20"]
        tot_mi = sum_mi_4 + candidate_w["n_mem_incr_s20"]
        tot_p2 = sum_p2_4 + candidate_w["n_persist_s20"]
        balance = min(tot_fp, tot_mi)
        total = tot_fp + tot_mi
        return (balance, total, tot_p2, -candidate_w["video_id"], -candidate_w["frame_start"])

    # Group candidate windows per remaining video and pick best window for cohort balance
    best_rem_window_per_video = {}
    for w in remaining_s_windows:
        v_id = w["video_id"]
        if v_id not in best_rem_window_per_video or cohort_balance_score(w) > cohort_balance_score(best_rem_window_per_video[v_id]):
            best_rem_window_per_video[v_id] = w

    sorted_rem_vids = sorted(
        best_rem_window_per_video.keys(),
        key=lambda v: cohort_balance_score(best_rem_window_per_video[v]),
        reverse=True
    )
    assert len(sorted_rem_vids) >= 2, f"Need >= 2 remaining {sensor_s} videos with s20 opportunity, found {len(sorted_rem_vids)}"

    v_5 = sorted_rem_vids[0]
    fifth_eval_clip = copy.deepcopy(best_rem_window_per_video[v_5])
    assert (fifth_eval_clip["n_fastpath_s20"] + fifth_eval_clip["n_mem_incr_s20"]) >= 1, (
        "[FATAL] Cohort-S fifth evaluation clip has no s>=0.20 temporal opportunity."
    )

    cohort_s_eval = sorted(s_joint_clips + [fifth_eval_clip], key=lambda w: w["video_id"])

    # 3. Select 6th disjoint calibration clip from remaining videos (D-3370)
    calib_candidates = [best_rem_window_per_video[v] for v in sorted_rem_vids[1:]]
    calib_candidates.sort(
        key=lambda w: (w["n_fastpath_s20"] + w["n_mem_incr_s20"], w["n_persist_s20"], -w["video_id"], -w["frame_start"]),
        reverse=True
    )
    cohort_s_calib = copy.deepcopy(calib_candidates[0])
    assert (cohort_s_calib["n_fastpath_s20"] + cohort_s_calib["n_mem_incr_s20"]) >= 1, (
        "[FATAL] Calibration clip has no s>=0.20 temporal opportunity."
    )

    # Enrich targets and assert invariants
    for c in cohort_s_eval:
        enrich_targets_strictly(c, coco)
    enrich_targets_strictly(cohort_s_calib, coco)

    eval_vids_s = {c["video_id"] for c in cohort_s_eval}
    assert len(eval_vids_s) == 5, f"[FATAL] Cohort S eval video count != 5"
    assert cohort_s_calib["video_id"] not in eval_vids_s, f"[FATAL] Calibration video overlaps with eval videos"
    assert_clip_chronology(cohort_s_eval + [cohort_s_calib])
    for c in cohort_s_eval + [cohort_s_calib]:
        assert c["sensor"] == sensor_s, f"[FATAL] Sensor mismatch in Cohort S"

    print(f"[+] Cohort S Certified: 5 Eval Clips + 1 Calib Clip (100% {sensor_s}).")

    # ==========================================================================
    # PART 2: COHORT X (CROSS-SENSOR HETEROGENEOUS STRESS REPLICATION)
    # ==========================================================================
    print("\n" + "=" * 115)
    print("GENERATING COHORT X: CROSS-SENSOR DUAL-OPPORTUNITY STRESS REPLICATION TESTBED")
    print("=" * 115)

    # 6 global videos with joint opportunity across all sensors
    joint_vids_6 = []
    best_joint_window_per_vid = {}
    for s_name, v_dict in sensor_joint_map.items():
        for v_id, w_list in v_dict.items():
            joint_vids_6.append(v_id)
            w_list.sort(key=window_rank_key_joint, reverse=True)
            best_joint_window_per_vid[v_id] = copy.deepcopy(w_list[0])

    assert len(joint_vids_6) == 6, f"[FATAL] Expected exactly 6 strict joint videos, got {len(joint_vids_6)}"

    # Evaluate all binom(6, 5) = 6 combinations for optimal cohort balance
    all_x_combos = list(itertools.combinations(sorted(joint_vids_6), 5))

    def cohort_x_score(combo_vids):
        s_fp = sum(best_joint_window_per_vid[v]["n_fastpath_s20"] for v in combo_vids)
        s_mi = sum(best_joint_window_per_vid[v]["n_mem_incr_s20"] for v in combo_vids)
        s_p2 = sum(best_joint_window_per_vid[v]["n_persist_s20"] for v in combo_vids)
        return (min(s_fp, s_mi), s_fp + s_mi, s_p2, tuple(-v for v in sorted(combo_vids)))

    best_combo_x = max(all_x_combos, key=cohort_x_score)
    holdout_vid_x = [v for v in joint_vids_6 if v not in best_combo_x][0]

    cohort_x_eval = sorted([best_joint_window_per_vid[v] for v in best_combo_x], key=lambda w: w["video_id"])
    cohort_x_holdout = best_joint_window_per_vid[holdout_vid_x]

    # Enrich targets and assert invariants
    for c in cohort_x_eval:
        enrich_targets_strictly(c, coco)
    enrich_targets_strictly(cohort_x_holdout, coco)

    eval_vids_x = {c["video_id"] for c in cohort_x_eval}
    assert len(eval_vids_x) == 5, f"[FATAL] Cohort X eval video count != 5"
    assert cohort_x_holdout["video_id"] not in eval_vids_x, f"[FATAL] Holdout video overlaps with eval videos"
    assert_clip_chronology(cohort_x_eval + [cohort_x_holdout])
    for c in cohort_x_eval + [cohort_x_holdout]:
        assert c["n_fastpath_s20"] >= 1 and c["n_mem_incr_s20"] >= 1, f"[FATAL] Cohort X clip missing dual opportunity"

    print(f"[+] Cohort X Certified: 5 Eval Clips + 1 Holdout Clip (Strict Dual Opportunity).")

    # ==========================================================================
    # DISPLAY TABLES & SERIALIZE BOTH MANIFESTS
    # ==========================================================================
    def print_cohort_table(title, eval_clips, aux_clip, aux_label):
        print("\n" + "=" * 115)
        print(title)
        print("=" * 115)
        print(f"{'Clip #':<8} | {'Video ID':<10} | {'Frames (t-5 -> t)':<20} | {'Sensor':<8} | {'FP (s20)':<10} | {'MI (s20)':<10} | {'P>=2 (s20)':<12} | {'Target Categories'}")
        print("-" * 115)
        tot_fp, tot_mi, tot_p2 = 0, 0, 0
        cat_dist = defaultdict(int)

        for idx, c in enumerate(eval_clips):
            tot_fp += c["n_fastpath_s20"]
            tot_mi += c["n_mem_incr_s20"]
            tot_p2 += c["n_persist_s20"]
            f_range = f"{c['frame_start']} -> {c['frame_end']}"
            cats_in_c = set()
            for t in c["targets"]:
                if t["fastpath_s20"] or t["mem_incr_s20"]:
                    cats_in_c.add(t["category"])
                    cat_dist[t["category"]] += 1
            print(f"Clip {idx+1:<3} | video_{c['video_id']:<4d} | {f_range:<20} | {c['sensor']:<8} | {c['n_fastpath_s20']:<10} | {c['n_mem_incr_s20']:<10} | {c['n_persist_s20']:<12} | {', '.join(sorted(cats_in_c))}")

        print("-" * 115)
        print(f"{'TOTALS':<8} | {'5 Videos':<10} | {'30 Frames Total':<20} | {'':<8} | {tot_fp:<10} | {tot_mi:<10} | {tot_p2:<12} | {tot_fp + tot_mi} Predeclared Targets")

        aux_f_range = f"{aux_clip['frame_start']} -> {aux_clip['frame_end']}"
        print(f"  {aux_label} | video_{aux_clip['video_id']:<4d} | {aux_f_range:<20} | {aux_clip['sensor']:<8} | FP_s20 = {aux_clip['n_fastpath_s20']} | MI_s20 = {aux_clip['n_mem_incr_s20']} | P>=2_s20 = {aux_clip['n_persist_s20']}")
        return tot_fp, tot_mi, tot_p2, dict(cat_dist)

    fp_s, mi_s, p2_s, cat_s = print_cohort_table(
        f"FROZEN COHORT S (PRIMARY SAME-SENSOR TESTBED - {sensor_s})",
        cohort_s_eval, cohort_s_calib, "Calib Clip "
    )
    fp_x, mi_x, p2_x, cat_x = print_cohort_table(
        "FROZEN COHORT X (SECONDARY HETEROGENEOUS STRESS TESTBED)",
        cohort_x_eval, cohort_x_holdout, "Holdout Vid"
    )

    # Serialize Cohort S
    payload_s = {
        "_meta": {
            "artifact_name": "pol2_primary_same_sensor_clips_v1",
            "role": "PRIMARY_CORRESPONDENCE_FALSIFICATION",
            "sensor_family": sensor_s,
            "candidate_windows_sha256": cand_hash,
            "train_json_sha256": train_hash,
            "m0_checkpoint_sha256": EXPECTED_M0_HASH,
            "links_json_sha256": EXPECTED_LINKS_SHA256,
            "total_eval_clips": len(cohort_s_eval),
            "actual_fastpath_targets_s20": fp_s,
            "actual_memory_incremental_targets_s20": mi_s,
            "actual_persistent_ge2_targets_s20": p2_s,
            "target_category_breakdown": cat_s
        },
        "evaluation_clips": cohort_s_eval,
        "calibration_clip": cohort_s_calib
    }
    with open(COHORT_S_JSON, "w", encoding="utf-8") as f:
        json.dump(payload_s, f, indent=2)
    hash_s = compute_sha256(COHORT_S_JSON)

    # Serialize Cohort X
    payload_x = {
        "_meta": {
            "artifact_name": "pol2_secondary_cross_sensor_clips_v1",
            "role": "SECONDARY_HETEROGENEOUS_STRESS_REPLICATION",
            "candidate_windows_sha256": cand_hash,
            "train_json_sha256": train_hash,
            "m0_checkpoint_sha256": EXPECTED_M0_HASH,
            "links_json_sha256": EXPECTED_LINKS_SHA256,
            "total_eval_clips": len(cohort_x_eval),
            "actual_fastpath_targets_s20": fp_x,
            "actual_memory_incremental_targets_s20": mi_x,
            "actual_persistent_ge2_targets_s20": p2_x,
            "target_category_breakdown": cat_x
        },
        "evaluation_clips": cohort_x_eval,
        "holdout_clip": cohort_x_holdout
    }
    with open(COHORT_X_JSON, "w", encoding="utf-8") as f:
        json.dump(payload_x, f, indent=2)
    hash_x = compute_sha256(COHORT_X_JSON)

    print("\n" + "=" * 115)
    print("POL-2 MANIFEST FREEZE COMPLETE: BOTH COHORTS IMMUTABLY SEALED")
    print("=" * 115)
    print(f"1. Cohort S (Primary Same-Sensor):    {COHORT_S_JSON.resolve()}")
    print(f"   SHA-256: {hash_s}")
    print(f"2. Cohort X (Heterogeneous Stress):  {COHORT_X_JSON.resolve()}")
    print(f"   SHA-256: {hash_x}")
    print("=" * 115)


if __name__ == "__main__":
    main()