"""
D:/MARIVISION/H0_workbench/eval_tools/audit_dataset_geometry_resolution.py
Phase-II Proof-of-Launch Stage 0 (PoL-0): Part 1 - Hardened Dataset Geometry, Resolution & Metadata Audit

Strict Compliance Checklist:
  - D-3070: Hard failure on missing training data (No validation fallback)
  - D-3071: Strict Semantic-V2 metadata assertion parsing data[video]['frames']
  - D-3072 / D-3081: Full training inventory & deep category/scale cross-tab certification gate
  - D-3061: Exact non-aspect 640x640 warp: s_x = 640/W, s_y = 640/H
  - D-3073 / D-3082: Ideal band area R_ideal vs. Crop distribution (mean, median, P95, max)
  - D-3074 / D-3080 / D-3086: Multi-threshold endangered tracking (<4, <8, <16 px) with Active and Global coverage
  - D-3078 / D-3079 / D-3088: Full 24-measure matrix with safe correlation wrapper (dense and 1 FPS subsampled)
  - D-3083 / D-3087: Robust metadata coalescing and complete 4-file artifact export
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
from scipy.stats import pearsonr, spearmanr

TRAIN_JSON = Path("D:/MARIVISION/seaclips-train.json")
TRAIN_TRACKS = Path("D:/MARIVISION/H0_workbench/horizon_tracks/seaclips_horizon_tracks_train_semantic_v2.json")
OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")


def compute_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def get_video_id_from_path(file_name):
    clean = str(file_name).replace("\\", "/")
    for part in clean.split("/"):
        if part.startswith("video_"):
            return part
    return "unknown"


def get_frame_num_from_path(file_name):
    return int(Path(file_name).stem)


def compute_canonical_rho(cy, h_at_x, eps=1e-6):
    if cy < h_at_x:
        return (cy - h_at_x) / (h_at_x + eps)
    else:
        return (cy - h_at_x) / (1.0 - h_at_x + eps)


def native_scale_bin(area):
    if area < 32**2:
        return "S"
    elif area <= 96**2:
        return "M"
    else:
        return "L"


def first_valid(*values):
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if s and s.lower() not in {"none", "null", "nan", "unknown", "undefined"}:
            return s
    return "unknown"


def safe_corr(x, y, method="pearson"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or len(y) < 2:
        return None
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return None
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        return None
    try:
        if method == "pearson":
            val, _ = pearsonr(x, y)
        else:
            val, _ = spearmanr(x, y)
        return float(val) if np.isfinite(val) else None
    except Exception:
        return None


def check_box_band_intersection(x1, y1, x2, y2, eta_L, eta_R, delta):
    h_x1 = (1.0 - x1) * eta_L + x1 * eta_R
    h_x2 = (1.0 - x2) * eta_L + x2 * eta_R

    y_min_x1 = h_x1 * (1.0 - delta)
    y_max_x1 = h_x1 + delta * (1.0 - h_x1)

    y_min_x2 = h_x2 * (1.0 - delta)
    y_max_x2 = h_x2 + delta * (1.0 - h_x2)

    if y1 > max(y_max_x1, y_max_x2):
        return False
    if y2 < min(y_min_x1, y_min_x2):
        return False
    return True


def load_and_flatten_horizon_tracks(tracks_path):
    with open(tracks_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    flat_map = {}
    for vid_key, vid_dict in raw_data.items():
        if isinstance(vid_dict, dict) and "frames" in vid_dict:
            frames_dict = vid_dict["frames"]
            if isinstance(frames_dict, dict):
                for frame_name, h_vec in frames_dict.items():
                    clean_k = frame_name.replace("\\", "/").strip().lstrip("/")
                    vec = [float(h_vec[0]), float(h_vec[1]), float(h_vec[2])]
                    flat_map[clean_k] = vec
                    if not clean_k.startswith("train/"):
                        flat_map[f"train/{clean_k}"] = vec
                    else:
                        flat_map[clean_k[len("train/"): ]] = vec
    return flat_map


def main():
    print("=" * 115)
    print("PROOF-OF-LAUNCH STAGE 0 (PoL-0): SEACLIPS TRAINING DATASET & GEOMETRY AUDIT")
    print(f"Annotation Target: {TRAIN_JSON.resolve()}")
    print(f"Horizon Tracks:    {TRAIN_TRACKS.resolve()}")
    print("=" * 115)

    assert TRAIN_JSON.exists(), f"[FATAL] Missing training annotations: {TRAIN_JSON}"
    assert TRAIN_TRACKS.exists(), f"[FATAL] Missing training horizon tracks: {TRAIN_TRACKS}"

    train_json_hash = compute_sha256(TRAIN_JSON)
    train_tracks_hash = compute_sha256(TRAIN_TRACKS)

    coco = COCO(str(TRAIN_JSON))
    flat_horizon_map = load_and_flatten_horizon_tracks(TRAIN_TRACKS)
    print(f"[*] Flattened active horizon mapping contains {len(flat_horizon_map):,} entry keys.")

    img_ids = sorted(coco.getImgIds())
    ann_ids = sorted(coco.getAnnIds())
    cats = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}

    # --------------------------------------------------------------------------
    # TASK 0: SCHEMA INSPECTION & DEEP TRAINING INVENTORY CERTIFICATION
    # --------------------------------------------------------------------------
    print("\n--- TASK 0: SCHEMA INSPECTION & DEEP INVENTORY CERTIFICATION ---")

    all_ann_keys = set().union(*(ann.keys() for ann in coco.loadAnns(ann_ids)))
    all_img_keys = set().union(*(coco.loadImgs([i])[0].keys() for i in img_ids))
    all_vid_keys = set().union(*(v.keys() for v in coco.dataset.get("videos", [])))

    print(f"Unioned Annotation Keys ({len(all_ann_keys)}): {sorted(list(all_ann_keys))}")
    print(f"Unioned Image Keys      ({len(all_img_keys)}): {sorted(list(all_img_keys))}")
    print(f"Unioned Video Keys      ({len(all_vid_keys)}): {sorted(list(all_vid_keys))}")

    has_track_id = any(k in all_ann_keys for k in ["track_id", "instance_id", "obj_id", "target_id"])
    print(f"[*] Persistent instance/track identifier present in annotations: {has_track_id}")

    video_lookup = {}
    if "videos" in coco.dataset:
        for v in coco.dataset["videos"]:
            if "id" in v:
                video_lookup[v["id"]] = v
            if "video_name" in v:
                video_lookup[v["video_name"]] = v

    g1_img_ids = set()
    g0_img_ids = set()
    img_meta = {}

    for img_id in img_ids:
        info = coco.loadImgs([img_id])[0]
        fname = info["file_name"]
        clean_fname = fname.replace("\\", "/").strip().lstrip("/")

        assert clean_fname in flat_horizon_map, f"[FATAL] Missing horizon record for image: {clean_fname}"
        h_vec = flat_horizon_map[clean_fname]

        g_val = float(h_vec[2])
        assert g_val in (0.0, 1.0), f"[FATAL] Invalid gate value {g_val} in {clean_fname}"

        if g_val == 1.0:
            g1_img_ids.add(img_id)
        else:
            g0_img_ids.add(img_id)

        v_info = video_lookup.get(info.get("video_id"), {})
        vid_name = get_video_id_from_path(fname)
        if not v_info and vid_name in video_lookup:
            v_info = video_lookup[vid_name]

        sensor = first_valid(info.get("camera_sensor"), v_info.get("camera_sensor"))
        refl = first_valid(info.get("water_reflections"), v_info.get("water_reflections"))
        sea = first_valid(info.get("sea_state"), v_info.get("sea_state"))
        weather = first_valid(info.get("weather"), v_info.get("weather"))

        img_meta[img_id] = {
            "W": float(info["width"]),
            "H": float(info["height"]),
            "eta_L": float(h_vec[0]),
            "eta_R": float(h_vec[1]),
            "g": g_val,
            "video": vid_name,
            "frame_num": get_frame_num_from_path(fname),
            "sensor": sensor,
            "reflections": refl,
            "sea_state": sea,
            "weather": weather
        }

    # Inventory gate assertions
    assert len(img_ids) == 20310, f"Expected 20,310 frames, found {len(img_ids)}"
    assert len(g1_img_ids) == 8999, f"Expected 8,999 g1 frames, found {len(g1_img_ids)}"
    assert len(g0_img_ids) == 11311, f"Expected 11,311 g0 frames, found {len(g0_img_ids)}"
    assert len(ann_ids) == 65542, f"Expected 65,542 annotations, found {len(ann_ids)}"

    anns_g1 = [a for a in coco.loadAnns(ann_ids) if a["image_id"] in g1_img_ids]
    anns_g0 = [a for a in coco.loadAnns(ann_ids) if a["image_id"] in g0_img_ids]
    assert len(anns_g1) == 18494, f"Expected 18,494 g1 boxes, found {len(anns_g1)}"
    assert len(anns_g0) == 47048, f"Expected 47,048 g0 boxes, found {len(anns_g0)}"

    expected_cat_crosstab = {
        "ANIMAL": {"g1": 707, "g0": 1649},
        "BOAT": {"g1": 1525, "g0": 4736},
        "MARINE_MARKER": {"g1": 1224, "g0": 11379},
        "OBJECT": {"g1": 8996, "g0": 6368},
        "LEISURE_VEHICLE": {"g1": 0, "g0": 600},
        "SAILING_VESSEL": {"g1": 4928, "g0": 21623},
        "SHIP": {"g1": 1114, "g0": 693}
    }

    actual_cat_crosstab = defaultdict(lambda: {"g1": 0, "g0": 0})
    actual_scale_crosstab = defaultdict(lambda: {"g1": 0, "g0": 0})

    for a in coco.loadAnns(ann_ids):
        cname = cats[a["category_id"]]
        g_key = "g1" if a["image_id"] in g1_img_ids else "g0"
        actual_cat_crosstab[cname][g_key] += 1
        s_bin = native_scale_bin(a["area"])
        actual_scale_crosstab[s_bin][g_key] += 1

    for cname, exp in expected_cat_crosstab.items():
        act = actual_cat_crosstab[cname]
        assert act["g1"] == exp["g1"] and act["g0"] == exp["g0"], f"Category mismatch {cname}: act={act} vs exp={exp}"

    expected_scale_crosstab = {
        "S": {"g1": 14388, "g0": 34529},
        "M": {"g1": 3330, "g0": 9715},
        "L": {"g1": 776, "g0": 2804}
    }

    for s_bin, exp in expected_scale_crosstab.items():
        act = actual_scale_crosstab[s_bin]
        assert act["g1"] == exp["g1"] and act["g0"] == exp["g0"], f"Scale mismatch {s_bin}: act={act} vs exp={exp}"

    print(f"[+] DEEP INVENTORY CERTIFIED: 20,310 Frames (g1=8,999, g0=11,311) | 65,542 Boxes")
    print(f"    Category & Scale Availability Cross-Tabs match frozen ground truth exactly.")
    print("=" * 115)

    # --------------------------------------------------------------------------
    # TASK 1: EXACT NON-ASPECT 640x640 WARP RESOLUTION AUDIT
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("TASK 1: EXACT NON-ASPECT 640x640 WARP RESOLUTION AUDIT (s_x = 640/W, s_y = 640/H)")
    print("=" * 115)

    native_counts = {"w<4": 0, "h<4": 0, "min<4": 0, "min<8": 0, "min<16": 0}
    warp_counts   = {"w<4": 0, "h<4": 0, "min<4": 0, "min<8": 0, "min<16": 0}
    cat_warp_stats = defaultdict(lambda: {"total": 0, "min<4": 0, "min<8": 0, "min<16": 0})

    endangered_ids = {4: set(), 8: set(), 16: set()}

    for a in coco.loadAnns(ann_ids):
        meta = img_meta[a["image_id"]]
        W, H = meta["W"], meta["H"]
        bw, bh = a["bbox"][2], a["bbox"][3]
        cname = cats[a["category_id"]]

        s_x = 640.0 / W
        s_y = 640.0 / H

        bw_warp = bw * s_x
        bh_warp = bh * s_y

        min_native = min(bw, bh)
        min_warp = min(bw_warp, bh_warp)

        if bw < 4.0: native_counts["w<4"] += 1
        if bh < 4.0: native_counts["h<4"] += 1
        if min_native < 4.0: native_counts["min<4"] += 1
        if min_native < 8.0: native_counts["min<8"] += 1
        if min_native < 16.0: native_counts["min<16"] += 1

        if bw_warp < 4.0: warp_counts["w<4"] += 1
        if bh_warp < 4.0: warp_counts["h<4"] += 1
        if min_warp < 4.0:
            warp_counts["min<4"] += 1
            cat_warp_stats[cname]["min<4"] += 1
            endangered_ids[4].add(a["id"])
        if min_warp < 8.0:
            warp_counts["min<8"] += 1
            cat_warp_stats[cname]["min<8"] += 1
            endangered_ids[8].add(a["id"])
        if min_warp < 16.0:
            warp_counts["min<16"] += 1
            cat_warp_stats[cname]["min<16"] += 1
            endangered_ids[16].add(a["id"])

        cat_warp_stats[cname]["total"] += 1

    print(f"Total Targets: {len(ann_ids):,}")
    print(f"Native Resolution  | Width < 4px: {native_counts['w<4']:5d} ({native_counts['w<4']/len(ann_ids)*100:5.2f}%) | Height < 4px: {native_counts['h<4']:5d} ({native_counts['h<4']/len(ann_ids)*100:5.2f}%) | Min Dim < 8px: {native_counts['min<8']:5d} ({native_counts['min<8']/len(ann_ids)*100:5.2f}%)")
    print(f"Warped (640x640)   | Width < 4px: {warp_counts['w<4']:5d} ({warp_counts['w<4']/len(ann_ids)*100:5.2f}%) | Height < 4px: {warp_counts['h<4']:5d} ({warp_counts['h<4']/len(ann_ids)*100:5.2f}%) | Min Dim < 8px: {warp_counts['min<8']:5d} ({warp_counts['min<8']/len(ann_ids)*100:5.2f}%)")
    print("-" * 115)
    print(f"{'Category':<20} | {'Total Boxes':<12} | {'Min Dim < 4px (Warp)':<22} | {'Min Dim < 8px (Warp)':<22} | {'Min Dim < 16px (Warp)':<22}")
    print("-" * 115)
    for cname, s in sorted(cat_warp_stats.items()):
        tot = s["total"]
        print(f"{cname:<20} | {tot:<12d} | {s['min<4']:<7d} ({s['min<4']/tot*100:5.1f}%)       | {s['min<8']:<7d} ({s['min<8']/tot*100:5.1f}%)       | {s['min<16']:<7d} ({s['min<16']/tot*100:5.1f}%)")

    # --------------------------------------------------------------------------
    # TASK 2: HORIZON-BAND COVERAGE VS. CROP COST DISTRIBUTION
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("TASK 2: HORIZON-BAND COVERAGE (ACTIVE & GLOBAL) VS. FULL-WIDTH CROP FRACTION")
    print("=" * 115)

    delta_candidates = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 1.00]
    crop_cost_distributions = defaultdict(list)

    active_coco_small = []
    active_endangered = {4: [], 8: [], 16: []}

    for img_id in g1_img_ids:
        meta = img_meta[img_id]
        eta_L, eta_R = meta["eta_L"], meta["eta_R"]
        W, H = meta["W"], meta["H"]

        for delta in delta_candidates:
            y_min_0 = eta_L * (1.0 - delta)
            y_max_0 = eta_L + delta * (1.0 - eta_L)
            y_min_1 = eta_R * (1.0 - delta)
            y_max_1 = eta_R + delta * (1.0 - eta_R)

            Y_min = max(0.0, min(y_min_0, y_min_1))
            Y_max = min(1.0, max(y_max_0, y_max_1))
            crop_cost_distributions[delta].append(Y_max - Y_min)

        for a in coco.loadAnns(coco.getAnnIds(imgIds=[img_id])):
            bx, by, bw, bh = a["bbox"]
            cx = (bx + bw / 2.0) / W
            cy = (by + bh / 2.0) / H
            x1, y1 = bx / W, by / H
            x2, y2 = (bx + bw) / W, (by + bh) / H

            h_at_x = (1.0 - cx) * eta_L + cx * eta_R
            rho_c = compute_canonical_rho(cy, h_at_x)

            rec = {
                "id": a["id"],
                "rho_center": abs(rho_c),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "eta_L": eta_L, "eta_R": eta_R
            }

            if a["area"] < 32**2:
                active_coco_small.append(rec)
            for thr in [4, 8, 16]:
                if a["id"] in endangered_ids[thr]:
                    active_endangered[thr].append(rec)

    n_act_small = len(active_coco_small)
    n_glob_small = len([a for a in coco.loadAnns(ann_ids) if a["area"] < 32**2])

    print(f"Population Totals (Active g=1 vs. Global Training Set):")
    print(f"  COCO-Small:             Active = {n_act_small:,} / Global = {n_glob_small:,} ({n_act_small/n_glob_small*100:5.2f}%)")
    print(f"  Warped Min Dim < 4px:   Active = {len(active_endangered[4]):,} / Global = {len(endangered_ids[4]):,} ({len(active_endangered[4])/len(endangered_ids[4])*100:5.2f}%)")
    print(f"  Warped Min Dim < 8px:   Active = {len(active_endangered[8]):,} / Global = {len(endangered_ids[8]):,} ({len(active_endangered[8])/len(endangered_ids[8])*100:5.2f}%)")
    print(f"  Warped Min Dim < 16px:  Active = {len(active_endangered[16]):,} / Global = {len(endangered_ids[16]):,} ({len(active_endangered[16])/len(endangered_ids[16])*100:5.2f}%)")
    print("-" * 125)
    print(f"{'Delta':<6} | {'Ideal':<6} | {'Crop Mean (P95, Max)':<25} | {'Small Box Act (Glob)':<22} | {'<4px Box Act (Glob)':<22} | {'<8px Box Act (Glob)':<22} | {'<16px Box Act (Glob)':<22}")
    print("-" * 125)

    band_coverage_csv = []

    for delta in delta_candidates:
        costs = np.array(crop_cost_distributions[delta])
        mean_c = np.mean(costs)
        p95_c = np.percentile(costs, 95)
        max_c = np.max(costs)
        med_c = np.median(costs)

        cov_s_c = sum(1 for t in active_coco_small if t["rho_center"] <= delta)
        cov_s_b = sum(1 for t in active_coco_small if check_box_band_intersection(t["x1"], t["y1"], t["x2"], t["y2"], t["eta_L"], t["eta_R"], delta))
        pct_s_c = (cov_s_c / n_act_small) * 100.0 if n_act_small > 0 else 0.0
        pct_s_b_act = (cov_s_b / n_act_small) * 100.0 if n_act_small > 0 else 0.0
        pct_s_b_glob = (cov_s_b / n_glob_small) * 100.0 if n_glob_small > 0 else 0.0

        cov_e4 = sum(1 for t in active_endangered[4] if check_box_band_intersection(t["x1"], t["y1"], t["x2"], t["y2"], t["eta_L"], t["eta_R"], delta))
        pct_e4_act = (cov_e4 / len(active_endangered[4])) * 100.0 if len(active_endangered[4]) > 0 else 0.0
        pct_e4_glob = (cov_e4 / len(endangered_ids[4])) * 100.0 if len(endangered_ids[4]) > 0 else 0.0

        cov_e8 = sum(1 for t in active_endangered[8] if check_box_band_intersection(t["x1"], t["y1"], t["x2"], t["y2"], t["eta_L"], t["eta_R"], delta))
        pct_e8_act = (cov_e8 / len(active_endangered[8])) * 100.0 if len(active_endangered[8]) > 0 else 0.0
        pct_e8_glob = (cov_e8 / len(endangered_ids[8])) * 100.0 if len(endangered_ids[8]) > 0 else 0.0

        cov_e16 = sum(1 for t in active_endangered[16] if check_box_band_intersection(t["x1"], t["y1"], t["x2"], t["y2"], t["eta_L"], t["eta_R"], delta))
        pct_e16_act = (cov_e16 / len(active_endangered[16])) * 100.0 if len(active_endangered[16]) > 0 else 0.0
        pct_e16_glob = (cov_e16 / len(endangered_ids[16])) * 100.0 if len(endangered_ids[16]) > 0 else 0.0

        crop_str = f"{mean_c*100:4.1f}% ({p95_c*100:4.1f}%, {max_c*100:4.1f}%)"
        small_str = f"{pct_s_b_act:4.1f}% ({pct_s_b_glob:4.1f}%)"
        e4_str = f"{pct_e4_act:4.1f}% ({pct_e4_glob:4.1f}%)"
        e8_str = f"{pct_e8_act:4.1f}% ({pct_e8_glob:4.1f}%)"
        e16_str = f"{pct_e16_act:4.1f}% ({pct_e16_glob:4.1f}%)"

        print(f"{delta:<6.2f} | {delta*100:4.1f}% | {crop_str:<25} | {small_str:<22} | {e4_str:<22} | {e8_str:<22} | {e16_str:<22}")

        band_coverage_csv.append({
            "delta": delta,
            "r_ideal": delta,
            "crop_mean": float(mean_c),
            "crop_median": float(med_c),
            "crop_p95": float(p95_c),
            "crop_max": float(max_c),
            "coco_small_center_cov": pct_s_c,
            "coco_small_box_cov_active": pct_s_b_act,
            "coco_small_box_cov_global": pct_s_b_glob,
            "endangered_4px_box_cov_active": pct_e4_act,
            "endangered_4px_box_cov_global": pct_e4_glob,
            "endangered_8px_box_cov_active": pct_e8_act,
            "endangered_8px_box_cov_global": pct_e8_glob,
            "endangered_16px_box_cov_active": pct_e16_act,
            "endangered_16px_box_cov_global": pct_e16_glob
        })

    # --------------------------------------------------------------------------
    # TASK 3: FULL 24-MEASURE HORIZON-SCALE MATRIX
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("TASK 3: STRATIFIED HORIZON-SCALE CORRELATION MATRIX (4 Coordinates x 3 Scales x 2 Metrics = 24)")
    print("=" * 115)

    coords_to_test = ["dc", "db", "rhoc", "rhob"]
    scales_to_test = ["log_w", "log_h", "log_sqrt_a"]

    strata_data = {
        "dense": defaultdict(lambda: {k: [] for k in coords_to_test + scales_to_test}),
        "sub_1fps": defaultdict(lambda: {k: [] for k in coords_to_test + scales_to_test})
    }
    camera_strata = {
        "dense": defaultdict(lambda: {k: [] for k in coords_to_test + scales_to_test}),
        "sub_1fps": defaultdict(lambda: {k: [] for k in coords_to_test + scales_to_test})
    }

    for img_id in g1_img_ids:
        meta = img_meta[img_id]
        W, H = meta["W"], meta["H"]
        eta_L, eta_R = meta["eta_L"], meta["eta_R"]
        vid = meta["video"]
        sensor = meta["sensor"]
        is_1fps = (meta["frame_num"] % 30 == 0)

        for a in coco.loadAnns(coco.getAnnIds(imgIds=[img_id])):
            cname = cats[a["category_id"]]
            bx, by, bw, bh = a["bbox"]
            cx = (bx + bw / 2.0) / W
            cy = (by + bh / 2.0) / H
            y_bot = (by + bh) / H

            h_at_x = (1.0 - cx) * eta_L + cx * eta_R
            dc = cy - h_at_x
            db = y_bot - h_at_x
            rhoc = compute_canonical_rho(cy, h_at_x)
            rhob = compute_canonical_rho(y_bot, h_at_x)

            w_n = bw / W
            h_n = bh / H
            sqrt_a_n = np.sqrt(w_n * h_n)

            vals_dict = {
                "dc": dc, "db": db, "rhoc": rhoc, "rhob": rhob,
                "log_w": np.log(w_n + 1e-6),
                "log_h": np.log(h_n + 1e-6),
                "log_sqrt_a": np.log(sqrt_a_n + 1e-6)
            }

            for k, v in vals_dict.items():
                strata_data["dense"][(vid, cname)][k].append(v)
                camera_strata["dense"][(sensor, cname)][k].append(v)
                if is_1fps:
                    strata_data["sub_1fps"][(vid, cname)][k].append(v)
                    camera_strata["sub_1fps"][(sensor, cname)][k].append(v)

    csv_strata_records = []
    summary_collector = defaultdict(list)

    def evaluate_strata_dict(data_dict, strata_type):
        for s_key, vals in sorted(data_dict.items()):
            N = len(vals["dc"])
            if N >= 20:
                s_id = f"{s_key[0]} x {s_key[1]}"
                row = {"type": strata_type, "stratum": s_id, "N": N}

                for c_var in coords_to_test:
                    for s_var in scales_to_test:
                        x = vals[c_var]
                        y = vals[s_var]
                        r_p = safe_corr(x, y, "pearson")
                        rho_s = safe_corr(x, y, "spearman")

                        row[f"pearson_{c_var}_{s_var}"] = r_p
                        row[f"spearman_{c_var}_{s_var}"] = rho_s

                        if strata_type == "video_x_class_dense":
                            if rho_s is not None:
                                summary_collector[(c_var, s_var, "spearman")].append(rho_s)
                            if r_p is not None:
                                summary_collector[(c_var, s_var, "pearson")].append(r_p)

                csv_strata_records.append(row)

    evaluate_strata_dict(strata_data["dense"], "video_x_class_dense")
    evaluate_strata_dict(strata_data["sub_1fps"], "video_x_class_sub_1fps")
    evaluate_strata_dict(camera_strata["dense"], "camera_x_class_dense")
    evaluate_strata_dict(camera_strata["sub_1fps"], "camera_x_class_sub_1fps")

    print(f"Evaluated complete 24-measure matrix across {len(csv_strata_records)} eligible strata records.")
    print("\n--- Summary: Spearman Rank Correlations across Video x Class Strata (Dense 30 FPS) ---")
    print(f"{'Coordinate':<12} | {'Scale Target':<14} | {'Valid N':<8} | {'Median rho':<12} | {'IQR':<22} | {'Positive Fraction':<18}")
    print("-" * 105)
    for c_var in coords_to_test:
        for s_var in scales_to_test:
            arr = np.array(summary_collector[(c_var, s_var, "spearman")])
            if len(arr) > 0:
                med = np.median(arr)
                p25 = np.percentile(arr, 25)
                p75 = np.percentile(arr, 75)
                pos = np.sum(arr > 0)
                tot = len(arr)
                iqr_str = f"[{p25:+.3f}, {p75:+.3f}]"
                print(f"{c_var:<12} | {s_var:<14} | {tot:<8d} | {med:+.3f}{'':<6} | {iqr_str:<22} | {pos:2d} / {tot:2d} ({pos/tot*100:5.1f}%)")

    # --------------------------------------------------------------------------
    # TASK 4: METADATA COMPLETENESS & COMPOSITION AUDIT
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("TASK 4: METADATA JOIN COMPLETENESS & ENVIRONMENTAL COMPOSITION AUDIT")
    print("=" * 115)

    dimensions = ["sensor", "reflections", "sea_state", "weather"]
    csv_meta_records = []

    for dim in dimensions:
        unique_vals = sorted(list(set(img_meta[i][dim] for i in img_ids)))
        known_frames = sum(1 for i in img_ids if img_meta[i][dim].lower() != "unknown")
        unknown_frames = len(img_ids) - known_frames
        print(f"Dimension: {dim.upper():<18} | Known Frames: {known_frames:6d} ({known_frames/len(img_ids)*100:5.1f}%) | Unknown: {unknown_frames:6d} ({unknown_frames/len(img_ids)*100:5.1f}%)")

        for val in unique_vals:
            frames_all = [i for i in img_ids if img_meta[i][dim] == val]
            frames_g1  = [i for i in frames_all if img_meta[i]["g"] == 1.0]
            frames_g0  = [i for i in frames_all if img_meta[i]["g"] == 0.0]

            anns_all = [a for a in coco.loadAnns(ann_ids) if a["image_id"] in set(frames_all)]
            anns_g1  = [a for a in anns_all if a["image_id"] in set(frames_g1)]
            anns_g0  = [a for a in anns_all if a["image_id"] in set(frames_g0)]

            s_all = sum(1 for a in anns_all if a["area"] < 32**2)
            s_g1  = sum(1 for a in anns_g1 if a["area"] < 32**2)
            s_g0  = sum(1 for a in anns_g0 if a["area"] < 32**2)

            e8_all = sum(1 for a in anns_all if a["id"] in endangered_ids[8])
            e8_g1  = sum(1 for a in anns_g1 if a["id"] in endangered_ids[8])
            e8_g0  = sum(1 for a in anns_g0 if a["id"] in endangered_ids[8])

            csv_meta_records.append({
                "dimension": dim,
                "value": val,
                "frames_all": len(frames_all),
                "frames_g1": len(frames_g1),
                "frames_g0": len(frames_g0),
                "boxes_all": len(anns_all),
                "boxes_g1": len(anns_g1),
                "boxes_g0": len(anns_g0),
                "small_all": s_all,
                "small_g1": s_g1,
                "small_g0": s_g0,
                "endangered_8px_all": e8_all,
                "endangered_8px_g1": e8_g1,
                "endangered_8px_g0": e8_g0
            })

    # --------------------------------------------------------------------------
    # WRITE ALL 4 RESEARCH RECORDS TO DISK
    # --------------------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "pol0_geometry_resolution.json"
    band_csv_path = OUT_DIR / "pol0_band_coverage.csv"
    strata_csv_path = OUT_DIR / "pol0_horizon_scale_strata.csv"
    meta_csv_path = OUT_DIR / "pol0_metadata_composition.csv"

    with open(band_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(band_coverage_csv[0].keys()))
        w.writeheader()
        w.writerows(band_coverage_csv)

    with open(strata_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_strata_records[0].keys()))
        w.writeheader()
        w.writerows(csv_strata_records)

    with open(meta_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_meta_records[0].keys()))
        w.writeheader()
        w.writerows(csv_meta_records)

    strata_summary_export = {}
    for (c_var, s_var, m_type), arr in summary_collector.items():
        if len(arr) > 0:
            strata_summary_export[f"{m_type}_{c_var}_{s_var}"] = {
                "median": float(np.median(arr)),
                "iqr_25": float(np.percentile(arr, 25)),
                "iqr_75": float(np.percentile(arr, 75)),
                "valid_strata_count": len(arr),
                "fraction_positive": float(np.sum(np.array(arr) > 0) / len(arr))
            }

    master_summary = {
        "_meta": {
            "audit_version": "PoL-0_Part-1_Final_Certified",
            "train_json": str(TRAIN_JSON.resolve()),
            "train_json_sha256": train_json_hash,
            "train_tracks": str(TRAIN_TRACKS.resolve()),
            "train_tracks_sha256": train_tracks_hash,
            "total_frames": len(img_ids),
            "g1_frames": len(g1_img_ids),
            "g0_frames": len(g0_img_ids),
            "total_annotations": len(ann_ids),
            "schema_has_track_id": has_track_id
        },
        "inventory_certification": "PASSED_ALL_DEEP_INVENTORY_CHECKS",
        "warp_640x640_survival": {
            "native": native_counts,
            "warped": warp_counts,
            "categories": {k: dict(v) for k, v in cat_warp_stats.items()}
        },
        "band_coverage": band_coverage_csv,
        "strata_summary": strata_summary_export,
        "strata_matrix_records_count": len(csv_strata_records)
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2)

    print("\n" + "=" * 115)
    print(f"[+] All 4 machine-readable audit records written successfully:")
    print(f"    1. {json_path}")
    print(f"    2. {band_csv_path}")
    print(f"    3. {strata_csv_path}")
    print(f"    4. {meta_csv_path}")
    print("=" * 115)


if __name__ == "__main__":
    main()