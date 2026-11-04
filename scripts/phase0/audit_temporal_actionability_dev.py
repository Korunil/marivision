"""
D:/MARIVISION/H0_workbench/eval_tools/audit_temporal_actionability_dev.py
PoL-2 Pre-Flight: Temporal Actionability & Candidate-Eligible Development Pool Audit

Strict Compliance Checklist:
  - D-3426 / D-3444: PoL-0 reference clearly scoped (5,664 L5-linkable <4px misses, 19.17% at t-1 -> 41.90% by t-5);
                     audit population defined strictly as candidate-eligible historical-opportunity targets.
  - D-3427: Video is the grouping unit to prevent frame-level leakage/pseudo-replication.
  - D-3428 / P-1587: Excluded videos derived programmatically from hash-verified Cohort S/X manifests + {1}.
  - D-3435..D-3437 / P-1591: Three-way geometric decomposition:
      * Linked-GT Apparent Trajectory:       Delta_k^GT  = c_t^GT - c_{t-k}^GT
      * Historical Detector Localization:    e_k^loc     = c_{t-k}^M0 - c_{t-k}^GT
      * Runtime Evidence Offset:             Delta_k^M0  = c_t^GT - c_{t-k}^M0
  - D-3438 / P-1592: Strict assertion that chain_ann_ids[k-1] maps to exact window frame t-k.
  - D-3439: Reports median AND IQR for signed dx, dy to distinguish isotropic jitter from directional motion.
  - D-3430 / P-1588: Measures scale-relative displacement r_rel = r / min(w_t, h_t).
  - D-3431 / D-3446 / P-1595: Video-macro reporting splits V_GT and V_M0 support counts.
  - D-3440 / P-1593: Produces dedicated e-CAM-specific actionability report alongside all-sensor report.
  - D-3445 / P-1594: Part D fast-path per-video table sourced strictly from ["fp"] subset.
  - D-3432 / D-3441: Computes and stores live M0 prediction-cache SHA-256 in manifest without altering cache file.
"""

import json
import hashlib
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

EXPECTED_CANDIDATE_SHA256 = "f29e0b8a5cbfc46641222d84a6d370c3e9bc51ca4ca1124b7753078ca7c1e270"
EXPECTED_TRAIN_JSON_SHA256 = "edc32bdbd6b792eb83b6daf11e14708232f1f4a9e06153ebebd5bad1caef9b35"
EXPECTED_LINKS_SHA256 = "e132f0a5782e0a53da858db80e64db0b879652057eeea9c149af2735cb25a2fb"
EXPECTED_M0_HASH = "28e6288c3f6769c2d5275aa77ddb248735b55a09cd46a6b570e3f117890b268b"
EXPECTED_COHORT_S_SHA256 = "b157b11e0b1bb8f9a77729a4999d48693f5d1e0cb052d8423b5a8dc903f029dc"
EXPECTED_COHORT_X_SHA256 = "91afc4a0a82e0fc87d9d523a902023e19fe5ac9b8c88b186e359bbb88303dac9"

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


def compute_stats(arr):
    if len(arr) == 0:
        return {"med": np.nan, "p25": np.nan, "p75": np.nan, "p90": np.nan, "iqr": np.nan}
    a = np.array(arr)
    p25 = float(np.percentile(a, 25))
    p75 = float(np.percentile(a, 75))
    return {
        "med": float(np.median(a)),
        "p25": p25,
        "p75": p75,
        "p90": float(np.percentile(a, 90)),
        "iqr": float(p75 - p25)
    }


def format_signed_stats(st):
    if np.isnan(st["med"]):
        return "N/A"
    return f"{st['med']:+5.1f} [{st['p25']:+5.1f}, {st['p75']:+5.1f}]"


def format_unsigned_stats(st):
    if np.isnan(st["med"]):
        return "N/A"
    return f"Med: {st['med']:5.2f} | IQR: [{st['p25']:5.2f}, {st['p75']:5.2f}] | P90: {st['p90']:5.2f}"


def main():
    print("=" * 115)
    print("PROOF-OF-LAUNCH STAGE 2 (PoL-2): TEMPORAL ACTIONABILITY & DEVELOPMENT POOL AUDIT")
    print("PoL-0 Reference: 5,664 L5-linkable <4px misses (CA coverage: 19.17% at t-1 -> 41.90% by t-5)")
    print("Audit Population: Candidate-eligible historical-opportunity targets after Cohort-S/X & video_1 exclusions")
    print("=" * 115)

    assert CANDIDATES_JSON.exists(), f"[FATAL] Missing {CANDIDATES_JSON}"
    assert TRAIN_JSON.exists(), f"[FATAL] Missing {TRAIN_JSON}"
    assert CACHE_FILE.exists(), f"[FATAL] Missing {CACHE_FILE}"
    assert LINKS_JSON.exists(), f"[FATAL] Missing {LINKS_JSON}"
    assert COHORT_S_JSON.exists(), f"[FATAL] Missing {COHORT_S_JSON}"
    assert COHORT_X_JSON.exists(), f"[FATAL] Missing {COHORT_X_JSON}"

    cand_hash = compute_sha256(CANDIDATES_JSON)
    train_hash = compute_sha256(TRAIN_JSON)
    cache_hash = compute_sha256(CACHE_FILE)
    links_hash = compute_sha256(LINKS_JSON)
    cohort_s_hash = compute_sha256(COHORT_S_JSON)
    cohort_x_hash = compute_sha256(COHORT_X_JSON)

    assert cand_hash == EXPECTED_CANDIDATE_SHA256, "[FATAL] Candidate JSON SHA mismatch!"
    assert train_hash == EXPECTED_TRAIN_JSON_SHA256, "[FATAL] Train JSON SHA mismatch!"
    assert links_hash == EXPECTED_LINKS_SHA256, "[FATAL] Links JSON SHA mismatch!"
    assert cohort_s_hash == EXPECTED_COHORT_S_SHA256, "[FATAL] Cohort S SHA mismatch!"
    assert cohort_x_hash == EXPECTED_COHORT_X_SHA256, "[FATAL] Cohort X SHA mismatch!"

    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache_data = json.load(f)
    cache_meta = cache_data.get("_meta", {})
    assert cache_meta.get("m0_checkpoint_sha256") == EXPECTED_M0_HASH, "[FATAL] M0 checkpoint SHA mismatch in cache!"

    # D-3441: Defensively verify train_json_sha256 if present, without editing cache if absent
    if "train_json_sha256" in cache_meta:
        assert cache_meta["train_json_sha256"] == train_hash, "[FATAL] M0 cache train JSON mismatch!"
        cache_train_json_status = "VERIFIED_IN_META"
    else:
        cache_train_json_status = "ABSENT_IN_META_LIVE_HASH_FROZEN"

    raw_detections = {int(k): v for k, v in cache_data["detections"].items()}
    assert len(raw_detections) == 20310, f"[FATAL] Expected 20,310 frames in cache, found {len(raw_detections)}"

    # --------------------------------------------------------------------------
    # STEP 1: PROGRAMMATIC EXCLUSION DERIVATION (D-3428, P-1587)
    # --------------------------------------------------------------------------
    with open(COHORT_S_JSON, "r", encoding="utf-8") as f:
        data_s = json.load(f)
    with open(COHORT_X_JSON, "r", encoding="utf-8") as f:
        data_x = json.load(f)

    vids_s = {c["video_id"] for c in data_s["evaluation_clips"]}
    vids_s.add(data_s["calibration_clip"]["video_id"])

    vids_x = {c["video_id"] for c in data_x["evaluation_clips"]}
    vids_x.add(data_x["holdout_clip"]["video_id"])

    derived_excluded = vids_s | vids_x | {1}
    expected_excluded = {1, 18, 20, 21, 24, 34, 42, 45}
    assert derived_excluded == expected_excluded, (
        f"[FATAL] Derived exclusion set {derived_excluded} does not match expected {expected_excluded}"
    )

    print(f"[+] All Source Artifact Provenance Certified:")
    print(f"    - Cohort S Manifest SHA-256:     {cohort_s_hash}")
    print(f"    - Cohort X Manifest SHA-256:     {cohort_x_hash}")
    print(f"    - Live M0 Raw Cache SHA-256:     {cache_hash} ({cache_train_json_status})")
    print(f"    - Derived Protected Videos ({len(derived_excluded)}): {sorted(list(derived_excluded))} (P-1587)")

    with open(CANDIDATES_JSON, "r", encoding="utf-8") as f:
        cand_data = json.load(f)
    all_windows = cand_data["windows"]

    coco = COCO(str(TRAIN_JSON))
    ann_lookup = {a["id"]: a for a in coco.loadAnns(coco.getAnnIds())}
    img_lookup = {info["id"]: info for info in coco.loadImgs(coco.getImgIds())}

    # Filter candidate windows to candidate-eligible development pool (D-3433)
    dev_windows = [w for w in all_windows if w["video_id"] not in derived_excluded]
    dev_video_ids = sorted(list(set(w["video_id"] for w in dev_windows)))

    print(f"[+] Identified {len(dev_windows):,} candidate-eligible windows across {len(dev_video_ids)} development videos.")

    # --------------------------------------------------------------------------
    # STEP 2: THREE-WAY GEOMETRIC MEASUREMENT ENGINE (P-1591)
    # --------------------------------------------------------------------------
    def init_metric_store():
        return {
            "dx_gt": [], "dy_gt": [], "r_gt": [], "r_rel_gt": [], "iou_gt": [],
            "dx_m0": [], "dy_m0": [], "r_m0": [], "r_rel_m0": [], "iou_m0": [],
            "ex_loc": [], "ey_loc": [], "e_loc": [], "e_rel_loc": []
        }

    pooled_all = {k: {s: init_metric_store() for s in ("all", "fp", "mi")} for k in range(1, 6)}
    pooled_ecam = {k: {s: init_metric_store() for s in ("all", "fp", "mi")} for k in range(1, 6)}

    # Video-macro metric stores: video_id -> k -> subset -> metric_store
    video_all = defaultdict(lambda: {k: {s: init_metric_store() for s in ("all", "fp", "mi")} for k in range(1, 6)})

    seen_target_keys = set()
    total_audited_targets = 0

    for w in dev_windows:
        v_id = w["video_id"]
        sensor = w["sensor"]
        final_iid = w["image_ids"][-1]
        img_info_t = img_lookup[final_iid]
        w_t = float(img_info_t["width"])
        h_t = float(img_info_t["height"])

        sx_t = 640.0 / w_t
        sy_t = 640.0 / h_t

        for t in w["targets"]:
            t_key = (v_id, t["ann_id"])
            if t_key in seen_target_keys:
                continue
            seen_target_keys.add(t_key)
            total_audited_targets += 1

            t_type = "fp" if t["fastpath_s20"] else ("mi" if t["mem_incr_s20"] else "other")
            ann_t = ann_lookup[t["ann_id"]]
            bx_t, by_t, bw_t, bh_t = ann_t["bbox"]

            cx_t_px = (bx_t + bw_t / 2.0) * sx_t
            cy_t_px = (by_t + bh_t / 2.0) * sy_t
            min_dim_t = min(bw_t * sx_t, bh_t * sy_t)
            eps_dim = max(min_dim_t, 1e-4)
            b_t_norm = [bx_t / w_t, by_t / h_t, bw_t / w_t, bh_t / h_t]

            chain_aids = t["chain_ann_ids"]
            assert len(chain_aids) == 5, f"[FATAL] Target {t['ann_id']} chain length != 5"

            for step_idx, p_aid in enumerate(chain_aids):
                k = step_idx + 1  # 1 to 5
                p_ann = ann_lookup[p_aid]
                p_img = img_lookup[p_ann["image_id"]]
                w_k = float(p_img["width"])
                h_k = float(p_img["height"])
                sx_k = 640.0 / w_k
                sy_k = 640.0 / h_k

                # D-3438 / P-1592: Assert exact chain-depth temporal alignment
                expected_iid = w["image_ids"][-(k + 1)]
                expected_fid = w["frame_ids"][-(k + 1)]
                assert p_ann["image_id"] == expected_iid, (
                    f"[FATAL] Target {t['ann_id']} k={k} image mismatch: {p_ann['image_id']} != {expected_iid}"
                )
                assert p_img["frame_id"] == expected_fid, (
                    f"[FATAL] Target {t['ann_id']} k={k} frame mismatch: {p_img['frame_id']} != {expected_fid}"
                )

                # 1. Ground-Truth Trajectory Geometry (P-1591)
                bx_k, by_k, bw_k, bh_k = p_ann["bbox"]
                cx_k_gt_px = (bx_k + bw_k / 2.0) * sx_k
                cy_k_gt_px = (by_k + bh_k / 2.0) * sy_k
                b_k_norm = [bx_k / w_k, by_k / h_k, bw_k / w_k, bh_k / h_k]

                dx_gt = cx_t_px - cx_k_gt_px
                dy_gt = cy_t_px - cy_k_gt_px
                r_gt = np.sqrt(dx_gt ** 2 + dy_gt ** 2)
                r_rel_gt = r_gt / eps_dim
                iou_gt = box_iou(b_t_norm, b_k_norm)

                subsets = ["all"]
                if t_type in ("fp", "mi"):
                    subsets.append(t_type)

                for sub in subsets:
                    m_p = pooled_all[k][sub]
                    m_p["dx_gt"].append(dx_gt)
                    m_p["dy_gt"].append(dy_gt)
                    m_p["r_gt"].append(r_gt)
                    m_p["r_rel_gt"].append(r_rel_gt)
                    m_p["iou_gt"].append(iou_gt)

                    m_v = video_all[v_id][k][sub]
                    m_v["dx_gt"].append(dx_gt)
                    m_v["dy_gt"].append(dy_gt)
                    m_v["r_gt"].append(r_gt)
                    m_v["r_rel_gt"].append(r_rel_gt)
                    m_v["iou_gt"].append(iou_gt)

                    if sensor == "e-CAM":
                        m_pe = pooled_ecam[k][sub]
                        m_pe["dx_gt"].append(dx_gt)
                        m_pe["dy_gt"].append(dy_gt)
                        m_pe["r_gt"].append(r_gt)
                        m_pe["r_rel_gt"].append(r_rel_gt)
                        m_pe["iou_gt"].append(iou_gt)

                # 2. Historical M0 Detector Evidence Geometry & Localization Error (P-1591)
                p_preds = raw_detections.get(p_ann["image_id"], [])
                matching_preds = [
                    d for d in p_preds
                    if d["category_id"] == ann_t["category_id"] and box_iou(d["bbox"], p_ann["bbox"]) >= EVAL_IOU_THRESHOLD
                    and d["score"] >= OPERATING_SCORE_THRESHOLD
                ]

                if matching_preds:
                    best_pred = max(matching_preds, key=lambda d: d["score"])
                    px, py, pw, ph = best_pred["bbox"]
                    pred_cx_px = (px + pw / 2.0) * sx_k
                    pred_cy_px = (py + ph / 2.0) * sy_k
                    b_pred_norm = [px / w_k, py / h_k, pw / w_k, ph / h_k]

                    # Displacement of M0 evidence to current GT
                    dx_m0 = cx_t_px - pred_cx_px
                    dy_m0 = cy_t_px - pred_cy_px
                    r_m0 = np.sqrt(dx_m0 ** 2 + dy_m0 ** 2)
                    r_rel_m0 = r_m0 / eps_dim
                    iou_m0 = box_iou(b_t_norm, b_pred_norm)

                    # Detector localization error relative to historical GT at t-k
                    ex_loc = pred_cx_px - cx_k_gt_px
                    ey_loc = pred_cy_px - cy_k_gt_px
                    e_loc = np.sqrt(ex_loc ** 2 + ey_loc ** 2)
                    e_rel_loc = e_loc / eps_dim

                    for sub in subsets:
                        m_p = pooled_all[k][sub]
                        m_p["dx_m0"].append(dx_m0)
                        m_p["dy_m0"].append(dy_m0)
                        m_p["r_m0"].append(r_m0)
                        m_p["r_rel_m0"].append(r_rel_m0)
                        m_p["iou_m0"].append(iou_m0)
                        m_p["ex_loc"].append(ex_loc)
                        m_p["ey_loc"].append(ey_loc)
                        m_p["e_loc"].append(e_loc)
                        m_p["e_rel_loc"].append(e_rel_loc)

                        m_v = video_all[v_id][k][sub]
                        m_v["dx_m0"].append(dx_m0)
                        m_v["dy_m0"].append(dy_m0)
                        m_v["r_m0"].append(r_m0)
                        m_v["r_rel_m0"].append(r_rel_m0)
                        m_v["iou_m0"].append(iou_m0)
                        m_v["ex_loc"].append(ex_loc)
                        m_v["ey_loc"].append(ey_loc)
                        m_v["e_loc"].append(e_loc)
                        m_v["e_rel_loc"].append(e_rel_loc)

                        if sensor == "e-CAM":
                            m_pe = pooled_ecam[k][sub]
                            m_pe["dx_m0"].append(dx_m0)
                            m_pe["dy_m0"].append(dy_m0)
                            m_pe["r_m0"].append(r_m0)
                            m_pe["r_rel_m0"].append(r_rel_m0)
                            m_pe["iou_m0"].append(iou_m0)
                            m_pe["ex_loc"].append(ex_loc)
                            m_pe["ey_loc"].append(ey_loc)
                            m_pe["e_loc"].append(e_loc)
                            m_pe["e_rel_loc"].append(e_rel_loc)

    print(f"[+] Audited {total_audited_targets} unique <4px target instances across {len(dev_video_ids)} development videos.")

    # --------------------------------------------------------------------------
    # STEP 3: DISPLAY TARGET-POOLED REPORTS (ALL-SENSOR & e-CAM)
    # --------------------------------------------------------------------------
    def print_actionability_table(title, data_store):
        print("\n" + "=" * 115)
        print(title)
        print("Answers: How far does historical evidence sit from current <4px misses? (dx, dy med [IQR] | r, r_rel med [IQR] p90)")
        print("=" * 115)
        print(f"{'k':<4} | {'Subset':<10} | {'N GT':<5} | {'GT dx [IQR]':<18} | {'GT dy [IQR]':<18} | {'GT r (px)':<20} | {'GT r_rel':<18} | {'N M0':<5} | {'M0 r (px)':<20} | {'Loc Err e_loc':<18}")
        print("-" * 115)

        for k in range(1, 6):
            for sub, lbl in [("all", "All"), ("fp", "Fast-Path"), ("mi", "Mem-Incr")]:
                d = data_store[k][sub]
                n_gt = len(d["r_gt"])
                n_m0 = len(d["r_m0"])
                if n_gt == 0:
                    continue

                st_dx_gt = compute_stats(d["dx_gt"])
                st_dy_gt = compute_stats(d["dy_gt"])
                st_r_gt = compute_stats(d["r_gt"])
                st_rel_gt = compute_stats(d["r_rel_gt"])

                dx_gt_str = format_signed_stats(st_dx_gt)
                dy_gt_str = format_signed_stats(st_dy_gt)
                r_gt_str = f"{st_r_gt['med']:4.1f} [{st_r_gt['p25']:4.1f},{st_r_gt['p75']:4.1f}]"
                rel_gt_str = f"{st_rel_gt['med']:4.1f}x [{st_rel_gt['p25']:4.1f},{st_rel_gt['p75']:4.1f}]"

                if n_m0 > 0:
                    st_r_m0 = compute_stats(d["r_m0"])
                    st_eloc = compute_stats(d["e_loc"])
                    r_m0_str = f"{st_r_m0['med']:4.1f} [{st_r_m0['p25']:4.1f},{st_r_m0['p75']:4.1f}]"
                    eloc_str = f"{st_eloc['med']:4.1f} [{st_eloc['p25']:4.1f},{st_eloc['p75']:4.1f}]"
                else:
                    r_m0_str = "N/A"
                    eloc_str = "N/A"

                print(f"t-{k:<2} | {lbl:<10} | {n_gt:<5} | {dx_gt_str:<18} | {dy_gt_str:<18} | {r_gt_str:<20} | {rel_gt_str:<18} | {n_m0:<5} | {r_m0_str:<20} | {eloc_str:<18}")
            if k < 5:
                print("  " + "-" * 111)

    # Print Table 1: All Sensors Pooled
    print_actionability_table("PART A: TARGET-POOLED ACTIONABILITY (ALL SENSORS)", pooled_all)

    # Print Table 2: e-CAM Only Pooled (D-3440, P-1593)
    print_actionability_table("PART B: TARGET-POOLED ACTIONABILITY (e-CAM DOMAIN ONLY - PRIMARY PoL-2 TARGET)", pooled_ecam)

    # --------------------------------------------------------------------------
    # STEP 4: VIDEO-MACRO ACTIONABILITY REPORT (D-3431, D-3446, P-1589, P-1595)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("PART C: VIDEO-MACRO ACTIONABILITY SUMMARY (Distribution of Per-Video Medians Across Scenes)")
    print("Reveals whether spatial locality is broadly preserved across scenes or driven by isolated dense videos")
    print("=" * 115)
    print(f"{'k':<4} | {'Domain':<10} | {'Subset':<10} | {'V_GT':<5} | {'V_M0':<5} | {'Macro GT r (px)':<24} | {'Macro GT r_rel':<20} | {'Macro M0 r (px)':<24} | {'Macro Loc Err e_loc':<20}")
    print("-" * 115)

    for domain_lbl, target_sensor in [("All", None), ("e-CAM", "e-CAM")]:
        for k in range(1, 6):
            for sub, lbl in [("all", "All"), ("fp", "Fast-Path"), ("mi", "Mem-Incr")]:
                v_meds_r_gt = []
                v_meds_rel_gt = []
                v_meds_r_m0 = []
                v_meds_eloc = []

                for v_id, k_dict in video_all.items():
                    v_wins = [w for w in dev_windows if w["video_id"] == v_id]
                    v_sensor = v_wins[0]["sensor"]
                    if target_sensor is not None and v_sensor != target_sensor:
                        continue

                    d_sub = k_dict[k][sub]
                    if len(d_sub["r_gt"]) > 0:
                        v_meds_r_gt.append(float(np.median(d_sub["r_gt"])))
                        v_meds_rel_gt.append(float(np.median(d_sub["r_rel_gt"])))
                    if len(d_sub["r_m0"]) > 0:
                        v_meds_r_m0.append(float(np.median(d_sub["r_m0"])))
                        v_meds_eloc.append(float(np.median(d_sub["e_loc"])))

                n_v_gt = len(v_meds_r_gt)
                n_v_m0 = len(v_meds_r_m0)

                if n_v_gt > 0:
                    st_r_gt_mac = compute_stats(v_meds_r_gt)
                    st_rel_gt_mac = compute_stats(v_meds_rel_gt)
                    r_gt_str = f"Med: {st_r_gt_mac['med']:4.1f} [{st_r_gt_mac['p25']:4.1f},{st_r_gt_mac['p75']:4.1f}]"
                    rel_gt_str = f"Med: {st_rel_gt_mac['med']:4.1f}x [{st_rel_gt_mac['p25']:4.1f},{st_rel_gt_mac['p75']:4.1f}]"

                    if n_v_m0 > 0:
                        st_r_m0_mac = compute_stats(v_meds_r_m0)
                        st_eloc_mac = compute_stats(v_meds_eloc)
                        r_m0_str = f"Med: {st_r_m0_mac['med']:4.1f} [{st_r_m0_mac['p25']:4.1f},{st_r_m0_mac['p75']:4.1f}]"
                        eloc_str = f"Med: {st_eloc_mac['med']:4.1f} [{st_eloc_mac['p25']:4.1f},{st_eloc_mac['p75']:4.1f}]"
                    else:
                        r_m0_str = "N/A"
                        eloc_str = "N/A"

                    print(f"t-{k:<2} | {domain_lbl:<10} | {lbl:<10} | {n_v_gt:<5} | {n_v_m0:<5} | {r_gt_str:<24} | {rel_gt_str:<20} | {r_m0_str:<24} | {eloc_str:<20}")
            if k < 5:
                print("  " + "-" * 111)

    # --------------------------------------------------------------------------
    # STEP 5: PER-VIDEO FAST PATH TABLE (k = 1) (D-3445, P-1594)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("PART D: PER-VIDEO FAST-PATH ACTIONABILITY (k = 1, Frame t-1 -> Frame t, ['fp'] Subset Only)")
    print("Direct empirical grounding for Candidate 2F (Spatially Anchored F-Only Probe)")
    print("=" * 115)
    print(f"{'Video ID':<10} | {'Sensor':<8} | {'N Targets':<10} | {'GT dx [IQR]':<18} | {'GT dy [IQR]':<18} | {'GT r (px)':<12} | {'GT r_rel':<12} | {'M0 r (px)':<12} | {'Loc Err':<12}")
    print("-" * 115)

    vid_sensor_map = {w["video_id"]: w["sensor"] for w in dev_windows}

    for v_id in dev_video_ids:
        # Sourced strictly from ["fp"] subset
        d_k1 = video_all[v_id][1]["fp"]
        n_t = len(d_k1["r_gt"])
        if n_t > 0:
            sensor = vid_sensor_map[v_id]
            st_dx = compute_stats(d_k1["dx_gt"])
            st_dy = compute_stats(d_k1["dy_gt"])
            med_r_gt = float(np.median(d_k1["r_gt"]))
            med_rel_gt = float(np.median(d_k1["r_rel_gt"]))
            med_r_m0 = float(np.median(d_k1["r_m0"])) if len(d_k1["r_m0"]) > 0 else np.nan
            med_eloc = float(np.median(d_k1["e_loc"])) if len(d_k1["e_loc"]) > 0 else np.nan

            dx_str = format_signed_stats(st_dx)
            dy_str = format_signed_stats(st_dy)
            m0_str = f"{med_r_m0:5.2f} px" if not np.isnan(med_r_m0) else "N/A"
            eloc_str = f"{med_eloc:5.2f} px" if not np.isnan(med_eloc) else "N/A"

            print(f"video_{v_id:<4d} | {sensor:<8} | {n_t:<10d} | {dx_str:<18} | {dy_str:<18} | {med_r_gt:5.2f} px   | {med_rel_gt:5.2f}x       | {m0_str:<12} | {eloc_str:<12}")

    # --------------------------------------------------------------------------
    # STEP 6: CANDIDATE-ELIGIBLE DEVELOPMENT POOL INVENTORY & SERIALIZATION
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("PART E: CANDIDATE-ELIGIBLE DEVELOPMENT POOL INVENTORY (D-3433, P-1583)")
    print("Evaluates video capacity for Leave-One-Video-Out (LOVO) cross-validation")
    print("=" * 115)
    print(f"{'Video ID':<10} | {'Sensor':<8} | {'Total Windows':<14} | {'FP Targets (s20)':<18} | {'MI Targets (s20)':<18} | {'Persist>=2 (s20)':<18} | {'Supported Experiments'}")
    print("-" * 115)

    dev_pool_records = []
    e_cam_dev_vids = []

    for v_id in dev_video_ids:
        v_wins = [w for w in dev_windows if w["video_id"] == v_id]
        sensor = vid_sensor_map[v_id]
        tot_wins = len(v_wins)

        unique_fp = set()
        unique_mi = set()
        unique_p2 = set()

        for w in v_wins:
            for t in w["targets"]:
                if t["fastpath_s20"]: unique_fp.add(t["ann_id"])
                if t["mem_incr_s20"]: unique_mi.add(t["ann_id"])
                if t["persist_ge2_s20"]: unique_p2.add(t["ann_id"])

        n_fp = len(unique_fp)
        n_mi = len(unique_mi)
        n_p2 = len(unique_p2)

        exp_support = []
        if n_fp >= 1: exp_support.append("2F (F-only)")
        if n_mi >= 1: exp_support.append("2M (M-only)")
        if n_fp >= 1 and n_mi >= 1: exp_support.append("Joint 2FM")
        exp_str = ", ".join(exp_support) if exp_support else "Zero s20 Opp"

        if sensor == "e-CAM" and (n_fp >= 1 or n_mi >= 1):
            e_cam_dev_vids.append(v_id)

        print(f"video_{v_id:<4d} | {sensor:<8} | {tot_wins:<14d} | {n_fp:<18d} | {n_mi:<18d} | {n_p2:<18d} | {exp_str}")

        dev_pool_records.append({
            "video_id": v_id,
            "sensor": sensor,
            "total_windows": tot_wins,
            "unique_fastpath_s20": n_fp,
            "unique_mem_incr_s20": n_mi,
            "unique_persist_s20": n_p2,
            "supports_2f": (n_fp >= 1),
            "supports_2m": (n_mi >= 1),
            "supports_joint": (n_fp >= 1 and n_mi >= 1)
        })

    print("-" * 115)
    print(f"Total Candidate-Eligible Development Videos: {len(dev_pool_records)} (e-CAM: {len(e_cam_dev_vids)} videos)")
    print(f"  - Supporting 2F (Fast-Path Probe):         {sum(1 for r in dev_pool_records if r['supports_2f'])} videos")
    print(f"  - Supporting 2M (Older-Memory Probe):       {sum(1 for r in dev_pool_records if r['supports_2m'])} videos")
    print(f"  - Supporting Joint 2FM:                     {sum(1 for r in dev_pool_records if r['supports_joint'])} videos")

    # Serialize Development Pool Manifest with complete provenance
    dev_pool_manifest = {
        "_meta": {
            "artifact_name": "pol2_development_pool_manifest_v2",
            "candidate_windows_sha256": cand_hash,
            "train_json_sha256": train_hash,
            "m0_checkpoint_sha256": EXPECTED_M0_HASH,
            "links_json_sha256": EXPECTED_LINKS_SHA256,
            "cohort_s_sha256": cohort_s_hash,
            "cohort_x_sha256": cohort_x_hash,
            "m0_prediction_cache_sha256": cache_hash,
            "m0_prediction_cache_train_json_status": cache_train_json_status,
            "derived_excluded_videos": sorted(list(derived_excluded)),
            "total_dev_videos": len(dev_pool_records),
            "ecam_dev_videos_count": len(e_cam_dev_vids),
            "ecam_dev_video_ids": e_cam_dev_vids
        },
        "development_videos": dev_pool_records
    }

    with open(DEV_POOL_JSON, "w", encoding="utf-8") as f:
        json.dump(dev_pool_manifest, f, indent=2)

    dev_manifest_hash = compute_sha256(DEV_POOL_JSON)
    print("\n" + "=" * 115)
    print(f"[+] Development Pool Manifest Frozen: {DEV_POOL_JSON.resolve()}")
    print(f"    Manifest SHA-256:                 {dev_manifest_hash}")
    print("=" * 115)


if __name__ == "__main__":
    main()