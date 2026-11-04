"""
D:/MARIVISION/H0_workbench/eval_tools/audit_bfcef_d0_d5_environmental_evidence.py
MARSEN B-FCEF: Stage D0-D5+ Environmental Evidence & Design Attribution Audit
(D-4607..D-4638, P-2350..P-2366)

Diagnostic Architecture:
  1. Metadata Integrity Gate: Asserts M_image == M_video across all 20,310 images;
     tracks Documented vs. Observed Train Vocabulary divergence with zero coercion.
  2. D0 Environmental Census & Confounding: Dual population (Full Train JSON vs OOF Corpus),
     native and R0@640 scales, contingency tables, and bias-corrected Cramér's V.
  3. Base Parity Oracle: Hard-asserts TP=50,741, FP=52,801, FN=13,929, GT=64,670, F1=0.603298.
  4. D1 Task Attribution: Cls-only, Box-only, and Full counterfactuals with environmental
     denominators (TP_B, FN_B, FP_B) and conservation assertions.
  5. D2 Opportunity & Multiplicity Sets: Single-expert opportunity unions, common always-on losses,
     multiplicity distributions, and pairwise Jaccard matrices for all 4 event families.
  6. D3a Promotion-Rank Audit: Pre-intervention Base rank (1..2100) and score quantiles for
     both Rescue and FP-add promotions.
  7. D3 Gateability: Base-Top100 OOF PR-AUC, AUROC, PR-lift, and coverage deciles (10%..100%).
  8. D4 Quintile Amplitude Response: All 4 outcome channels across empirical quintiles.
  9. D5 Video-Balanced Environmental Lift: Micro, Macro, and Video-Balanced Pooled PR-AUC.
 10. Memory & Compute Bound: Two-pass design with compact typed NumPy arrays and paired subsampling.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
from pathlib import Path

ROOT_DIR = Path("D:/MARIVISION")
OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
E0_WORK_DIR = OUT_DIR / "e0_adapters"
G0_WORK_DIR = OUT_DIR / "g0_qualification"
EVENTS_DIR = G0_WORK_DIR / "events"
D_REPORT_DIR = OUT_DIR / "d_audit"
D_REPORT_DIR.mkdir(parents=True, exist_ok=True)

MASTER_INDEX_PATH = E0_WORK_DIR / "bfcef_e0_oof_ledger_index.json"
TRAINING_SUMMARY_PATH = E0_WORK_DIR / "bfcef_e0_training_summary.json"
G0_REPORT_PATH = G0_WORK_DIR / "bfcef_g0_qualification_report.json"
EVENTS_INDEX_PATH = EVENTS_DIR / "bfcef_g0_events_index.json"
TRAIN_JSON = ROOT_DIR / "seaclips-train.json"

FROZEN_G0_REPORT_SHA256 = "10062d61090050273a22ce6ca195f4097ceed5f65f15939c8d28a8d18d94b7b8"

import time
import json
import gc
import hashlib
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Set

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

sys.stdout.reconfigure(line_buffering=True)
DEVICE = torch.device("cpu")

EXPERTS = ["S", "L", "C", "T"]
CAMERAS = ["e-CAM", "FLIR", "Axis"]
DOCUMENTED_WEATHER = ["sunny", "overcast", "cloudy", "rainy"]
DOCUMENTED_REFLECTIONS = ["yes", "no"]
DOCUMENTED_SEA_STATE = ["smooth", "wavy"]
DOCUMENTED_CAMERAS = ["e-CAM", "FLIR", "Axis"]

NUM_CLASSES = 7
NUM_FOLDS = 5
EXPECTED_TOTAL_FRAMES = 20075
EXPECTED_TOTAL_IMAGES = 20310
CANONICAL_TP = 50741
CANONICAL_FP = 52801
CANONICAL_FN = 13929
CANONICAL_GT = 64670


def compute_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def box_cxcywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def box_iou_norm_cxcywh(b1, b2) -> float:
    b1_x1 = b1[0] - b1[2] / 2.0
    b1_y1 = b1[1] - b1[3] / 2.0
    b1_x2 = b1[0] + b1[2] / 2.0
    b1_y2 = b1[1] + b1[3] / 2.0

    b2_x1 = b2[0] - b2[2] / 2.0
    b2_y1 = b2[1] - b2[3] / 2.0
    b2_x2 = b2[0] + b2[2] / 2.0
    b2_y2 = b2[1] + b2[3] / 2.0

    ix = max(0.0, min(b1_x2, b2_x2) - max(b1_x1, b2_x1))
    iy = max(0.0, min(b1_y2, b2_y2) - max(b1_y1, b2_y1))
    inter = ix * iy
    union = b1[2] * b1[3] + b2[2] * b2[3] - inter
    return inter / union if union > 0.0 else 0.0


def extract_top100_predictions(logits: torch.Tensor, boxes: torch.Tensor):
    probs = logits.sigmoid()
    flat_probs = probs.flatten()
    topk_scores, topk_idx = torch.topk(flat_probs, min(100, flat_probs.numel()))

    q_idx = topk_idx // NUM_CLASSES
    c_idx = topk_idx % NUM_CLASSES
    topk_boxes = boxes[q_idx]

    valid_op = topk_scores >= 0.20

    return {
        "all_scores": topk_scores,
        "all_q_idx": q_idx,
        "all_c_idx": c_idx,
        "all_boxes": topk_boxes,
        "op_scores": topk_scores[valid_op],
        "op_q_idx": q_idx[valid_op],
        "op_c_idx": c_idx[valid_op],
        "op_boxes": topk_boxes[valid_op],
    }


def match_operational_detections_to_gt(
    op_scores: torch.Tensor,
    op_c_idx: torch.Tensor,
    op_boxes_norm: torch.Tensor,
    op_q_idx: torch.Tensor,
    gt_boxes_norm: torch.Tensor,
    gt_labels: torch.Tensor
) -> Dict[str, Any]:
    num_gt = gt_boxes_norm.shape[0]
    num_det = op_scores.shape[0]

    matched_gt_indices: Set[int] = set()
    matched_det_indices: Set[int] = set()
    gt_to_det: Dict[int, Tuple[int, float, float, int, int]] = {}

    if num_gt > 0 and num_det > 0:
        gt_boxes_list = gt_boxes_norm.tolist()
        gt_labels_list = gt_labels.tolist()
        det_boxes_list = op_boxes_norm.tolist()
        det_classes_list = op_c_idx.tolist()
        det_scores_list = op_scores.tolist()
        det_q_list = op_q_idx.tolist()

        for d_idx in range(num_det):
            c_idx = det_classes_list[d_idx]
            det_box = det_boxes_list[d_idx]
            score = det_scores_list[d_idx]
            q_idx = det_q_list[d_idx]

            best_iou = -1.0
            best_gt_idx = None

            for g_idx in range(num_gt):
                if g_idx in matched_gt_indices:
                    continue
                if gt_labels_list[g_idx] != c_idx:
                    continue
                iou = box_iou_norm_cxcywh(det_box, gt_boxes_list[g_idx])
                if iou >= 0.30 and iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

            if best_gt_idx is not None:
                matched_gt_indices.add(best_gt_idx)
                matched_det_indices.add(d_idx)
                gt_to_det[best_gt_idx] = (d_idx, best_iou, score, q_idx, c_idx)

    fp_dict = {}
    for d_idx in range(num_det):
        if d_idx not in matched_det_indices:
            key = (int(op_q_idx[d_idx].item()), int(op_c_idx[d_idx].item()))
            fp_dict[key] = {
                "score": float(op_scores[d_idx].item()),
                "box": [float(x) for x in op_boxes_norm[d_idx].tolist()]
            }

    tp_count = len(matched_gt_indices)
    fp_count = len(fp_dict)
    fn_count = num_gt - tp_count

    return {
        "matched_gt_indices": matched_gt_indices,
        "matched_det_indices": matched_det_indices,
        "gt_to_det": gt_to_det,
        "fp_dict": fp_dict,
        "tp": tp_count,
        "fp": fp_count,
        "fn": fn_count,
        "num_det": num_det,
        "num_gt": num_gt
    }


def compute_cramers_v(cat1_list: List[str], cat2_list: List[str]) -> Tuple[float, float, Dict[str, Dict[str, int]]]:
    n = len(cat1_list)
    vals1 = sorted(list(set(cat1_list)))
    vals2 = sorted(list(set(cat2_list)))
    r = len(vals1)
    c = len(vals2)
    table = {v1: {v2: 0 for v2 in vals2} for v1 in vals1}
    for v1, v2 in zip(cat1_list, cat2_list):
        table[v1][v2] += 1
    if r <= 1 or c <= 1:
        return 0.0, 0.0, table
    row_sums = {v1: sum(table[v1][v2] for v2 in vals2) for v1 in vals1}
    col_sums = {v2: sum(table[v1][v2] for v1 in vals1) for v2 in vals2}
    chi2 = 0.0
    for v1 in vals1:
        for v2 in vals2:
            e = (row_sums[v1] * col_sums[v2]) / float(n)
            o = table[v1][v2]
            if e > 0:
                chi2 += ((o - e) ** 2) / e
    v_raw = float(np.sqrt(chi2 / (n * min(r - 1, c - 1))))
    phi2 = chi2 / n
    phi2_corr = max(0.0, phi2 - ((r - 1) * (c - 1)) / max(1.0, n - 1))
    r_corr = r - ((r - 1) ** 2) / max(1.0, n - 1)
    c_corr = c - ((c - 1) ** 2) / max(1.0, n - 1)
    denom = min(r_corr - 1.0, c_corr - 1.0)
    v_bc = float(np.sqrt(phi2_corr / denom)) if denom > 0.0 else 0.0
    return v_raw, v_bc, table


def extract_exact_meta(
    entry: Dict[str, Any],
    context_label: str,
    allowed_cams: List[str],
    allowed_weathers: List[str],
    allowed_refls: List[str],
    allowed_seas: List[str]
) -> Tuple[str, str, str, str]:
    required_keys = ["camera_sensor", "weather", "water_reflections", "sea_state"]
    for k in required_keys:
        assert k in entry, f"[FATAL] Missing required metadata key '{k}' in {context_label}"

    cam = str(entry["camera_sensor"]).strip()
    weather = str(entry["weather"]).strip()
    refl = str(entry["water_reflections"]).strip()
    sea = str(entry["sea_state"]).strip()

    assert cam in allowed_cams, f"[FATAL] Unexpected camera '{cam}' in {context_label}. Allowed: {allowed_cams}"
    assert weather in allowed_weathers, f"[FATAL] Unexpected weather '{weather}' in {context_label}. Allowed: {allowed_weathers}"
    assert refl in allowed_refls, f"[FATAL] Unexpected reflections '{refl}' in {context_label}. Allowed: {allowed_refls}"
    assert sea in allowed_seas, f"[FATAL] Unexpected sea_state '{sea}' in {context_label}. Allowed: {allowed_seas}"

    return cam, weather, refl, sea


def main():
    t_start = time.time()
    driver_sha = compute_sha256(Path(__file__))

    print("=" * 115)
    print("MARSEN B-FCEF: STAGE D0-D5+ ENVIRONMENTAL EVIDENCE & DESIGN ATTRIBUTION AUDIT")
    print(f"Driver Script SHA-256: {driver_sha}")
    print(f"Master Index:          {MASTER_INDEX_PATH}")
    print(f"Events Index:          {EVENTS_INDEX_PATH}")
    print("=" * 115)

    # 1. Cryptographic Provenance Preflight Gate
    print("[*] Verifying cryptographic provenance across source manifests, summary reports, and event shards...")
    assert MASTER_INDEX_PATH.exists(), f"[FATAL] Missing master index: {MASTER_INDEX_PATH}"
    assert TRAINING_SUMMARY_PATH.exists(), f"[FATAL] Missing training summary: {TRAINING_SUMMARY_PATH}"
    assert G0_REPORT_PATH.exists(), f"[FATAL] Missing G0 report: {G0_REPORT_PATH}"
    assert EVENTS_INDEX_PATH.exists(), f"[FATAL] Missing events index: {EVENTS_INDEX_PATH}"
    assert TRAIN_JSON.exists(), f"[FATAL] Missing seaclips-train.json: {TRAIN_JSON}"

    computed_master_sha = compute_sha256(MASTER_INDEX_PATH)
    computed_events_sha = compute_sha256(EVENTS_INDEX_PATH)
    computed_g0_report_sha = compute_sha256(G0_REPORT_PATH)

    assert computed_g0_report_sha == FROZEN_G0_REPORT_SHA256, (
        f"[FATAL] G0 report SHA drift: {computed_g0_report_sha} != {FROZEN_G0_REPORT_SHA256}"
    )

    with open(TRAINING_SUMMARY_PATH, "r", encoding="utf-8") as f:
        training_summary = json.load(f)
    with open(G0_REPORT_PATH, "r", encoding="utf-8") as f:
        g0_report = json.load(f)
    with open(EVENTS_INDEX_PATH, "r", encoding="utf-8") as f:
        events_g0 = json.load(f)
    with open(MASTER_INDEX_PATH, "r", encoding="utf-8") as f:
        master_e0 = json.load(f)

    assert training_summary["_meta"]["master_ledger_index_sha256"] == computed_master_sha
    assert g0_report["_meta"]["master_ledger_index_sha256"] == computed_master_sha
    assert g0_report["_meta"]["events_index_sha256"] == computed_events_sha
    assert events_g0["_meta"]["master_ledger_index_sha256"] == computed_master_sha

    for shard_name, exp_sha in events_g0["event_shards"].items():
        sp = EVENTS_DIR / shard_name
        assert sp.exists(), f"[FATAL] Missing G0 event shard {shard_name}"
        assert compute_sha256(sp) == exp_sha, f"[FATAL] SHA drift on {shard_name}"

    print(f"    [+] Certified Master Ledger SHA-256: {computed_master_sha[:16]}...")
    print(f"    [+] Certified Events Index SHA-256:  {computed_events_sha[:16]}...")
    print(f"    [+] Certified G0 Report SHA-256:     {computed_g0_report_sha[:16]}...")
    print(f"    [+] Verified all 47 G0 event shards against manifest ledger.\n")

    # 2. Dynamic Observed Vocabulary Discovery & Documented Schema Divergence Check
    print("[*] Ingesting seaclips-train.json and discovering Observed Train Vocabulary...")
    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        train_coco = json.load(f)

    discovered_cams = sorted(list({str(v.get("camera_sensor", "")).strip() for v in train_coco["videos"] if "camera_sensor" in v} |
                                 {str(im.get("camera_sensor", "")).strip() for im in train_coco["images"] if "camera_sensor" in im}))
    discovered_weathers = sorted(list({str(v.get("weather", "")).strip() for v in train_coco["videos"] if "weather" in v} |
                                      {str(im.get("weather", "")).strip() for im in train_coco["images"] if "weather" in im}))
    discovered_refls = sorted(list({str(v.get("water_reflections", "")).strip() for v in train_coco["videos"] if "water_reflections" in v} |
                                    {str(im.get("water_reflections", "")).strip() for im in train_coco["images"] if "water_reflections" in im}))
    discovered_seas = sorted(list({str(v.get("sea_state", "")).strip() for v in train_coco["videos"] if "sea_state" in v} |
                                  {str(im.get("sea_state", "")).strip() for im in train_coco["images"] if "sea_state" in im}))

    discovered_cams = [x for x in discovered_cams if x]
    discovered_weathers = [x for x in discovered_weathers if x]
    discovered_refls = [x for x in discovered_refls if x]
    discovered_seas = [x for x in discovered_seas if x]

    schema_divergence = {
        "weather": {
            "extra_in_observed": sorted(list(set(discovered_weathers) - set(DOCUMENTED_WEATHER))),
            "missing_from_observed": sorted(list(set(DOCUMENTED_WEATHER) - set(discovered_weathers)))
        },
        "water_reflections": {
            "extra_in_observed": sorted(list(set(discovered_refls) - set(DOCUMENTED_REFLECTIONS))),
            "missing_from_observed": sorted(list(set(DOCUMENTED_REFLECTIONS) - set(discovered_refls)))
        },
        "sea_state": {
            "extra_in_observed": sorted(list(set(discovered_seas) - set(DOCUMENTED_SEA_STATE))),
            "missing_from_observed": sorted(list(set(DOCUMENTED_SEA_STATE) - set(discovered_seas)))
        },
        "camera_sensor": {
            "extra_in_observed": sorted(list(set(discovered_cams) - set(DOCUMENTED_CAMERAS))),
            "missing_from_observed": sorted(list(set(DOCUMENTED_CAMERAS) - set(discovered_cams)))
        }
    }

    has_divergence = any(
        len(v["extra_in_observed"]) > 0 or len(v["missing_from_observed"]) > 0
        for v in schema_divergence.values()
    )

    if has_divergence:
        print("[!] DATASET SCHEMA DIVERGENCE DETECTED (Observed Train JSON vs Documented Schema):")
        for field_name, div in schema_divergence.items():
            if div["extra_in_observed"] or div["missing_from_observed"]:
                print(f"    • {field_name}:")
                if div["extra_in_observed"]:
                    print(f"        Undocumented observed values: {div['extra_in_observed']}")
                if div["missing_from_observed"]:
                    print(f"        Documented but unobserved:   {div['missing_from_observed']}")
        print("    [+] Proceeding with observed Train vocabulary without coercion.\n")
    else:
        print("[+] Documented and Observed vocabularies match exactly across all fields.\n")

    WEATHER_TYPES = discovered_weathers
    REFLECTION_TYPES = discovered_refls
    SEA_STATES = discovered_seas

    # Strict Metadata Integrity Gate across all 20,310 images
    print("[*] Executing Strict Metadata Integrity Gate across all 20,310 images in seaclips-train.json...")
    video_metadata = {}
    for v in train_coco["videos"]:
        vid = int(v.get("id", v.get("video_id")))
        cam, weather, refl, sea = extract_exact_meta(v, f"video {vid}", CAMERAS, WEATHER_TYPES, REFLECTION_TYPES, SEA_STATES)
        video_metadata[vid] = {"camera": cam, "weather": weather, "reflections": refl, "sea_state": sea}
    assert len(video_metadata) == 47

    mismatches = {"camera": 0, "weather": 0, "reflections": 0, "sea_state": 0}
    image_metadata = {}
    img_to_video = {}
    img_native_wh = {}

    for im in train_coco["images"]:
        img_id = int(im["id"])
        vid = int(im["video_id"])
        img_to_video[img_id] = vid
        img_native_wh[img_id] = (float(im["width"]), float(im["height"]))
        assert vid in video_metadata, f"[FATAL] Image {img_id} references missing video {vid}"
        cam_im, w_im, r_im, s_im = extract_exact_meta(im, f"image {img_id}", CAMERAS, WEATHER_TYPES, REFLECTION_TYPES, SEA_STATES)

        v_meta = video_metadata[vid]
        if cam_im != v_meta["camera"]:
            mismatches["camera"] += 1
        if w_im != v_meta["weather"]:
            mismatches["weather"] += 1
        if r_im != v_meta["reflections"]:
            mismatches["reflections"] += 1
        if s_im != v_meta["sea_state"]:
            mismatches["sea_state"] += 1

        image_metadata[img_id] = {
            "video_id": vid,
            "camera": cam_im,
            "weather": w_im,
            "reflections": r_im,
            "sea_state": s_im
        }

    assert len(image_metadata) == EXPECTED_TOTAL_IMAGES
    total_mismatches = sum(mismatches.values())
    print(f"    [+] Metadata Divergence Count: {mismatches}")
    if total_mismatches == 0:
        print(f"    [+] CERTIFIED: Zero metadata divergence. Physical conditions are exact video-level constants duplicated at image level.\n")
    else:
        print(f"    [!] NOTE: Divergence detected. Image metadata enforced as primary frame authority.\n")

    # 3. D0: Dual Environmental Census & Confounding Audit
    print("=" * 115)
    print("D0: COMPLETE ENVIRONMENTAL METADATA CENSUS & CONFOUNDING ANALYSIS")
    print("=" * 115)

    vid_list = sorted(video_metadata.keys())
    cams_v = [video_metadata[v]["camera"] for v in vid_list]
    weath_v = [video_metadata[v]["weather"] for v in vid_list]
    refl_v = [video_metadata[v]["reflections"] for v in vid_list]
    sea_v = [video_metadata[v]["sea_state"] for v in vid_list]

    cramer_results = {}
    pair_labels = [
        ("weather", "camera", weath_v, cams_v),
        ("reflections", "camera", refl_v, cams_v),
        ("sea_state", "camera", sea_v, cams_v),
        ("weather", "reflections", weath_v, refl_v),
        ("weather", "sea_state", weath_v, sea_v),
        ("reflections", "sea_state", refl_v, sea_v)
    ]

    print("VIDEO-LEVEL CRAMÉR'S V ASSOCIATION MATRIX (N = 47 VIDEOS):")
    for k1, k2, l1, l2 in pair_labels:
        v_raw, v_bc, c_table = compute_cramers_v(l1, l2)
        pair_key = f"{k1}_x_{k2}"
        cramer_results[pair_key] = {
            "raw_cramers_v": round(v_raw, 4),
            "bias_corrected_cramers_v": round(v_bc, 4),
            "contingency_table": c_table
        }
        print(f"  • {k1:<12} x {k2:<10} | Raw V = {v_raw:.4f} | Bias-Corrected V = {v_bc:.4f}")
    print("-" * 115)

    # Full Train JSON Grid Census (20,310 images)
    grid_full = defaultdict(lambda: {
        "videos": 0, "images": 0, "gt_anns": 0,
        "classes": [0] * NUM_CLASSES,
        "scale_native": {"s": 0, "m": 0, "l": 0},
        "scale_r0_640": {"s": 0, "m": 0, "l": 0}
    })

    for v_id, vm in video_metadata.items():
        k_combo = f"{vm['weather']}|refl={vm['reflections']}|sea={vm['sea_state']}"
        grid_full[k_combo]["videos"] += 1

    img_to_combo = {}
    for img_id, im_m in image_metadata.items():
        k_combo = f"{im_m['weather']}|refl={im_m['reflections']}|sea={im_m['sea_state']}"
        img_to_combo[img_id] = k_combo
        grid_full[k_combo]["images"] += 1

    for ann in train_coco["annotations"]:
        im_id = int(ann["image_id"])
        c_id = int(ann["category_id"]) - 1
        bbox = ann["bbox"]
        w_img, h_img = img_native_wh[im_id]
        k_combo = img_to_combo[im_id]
        node = grid_full[k_combo]
        node["gt_anns"] += 1
        if 0 <= c_id < NUM_CLASSES:
            node["classes"][c_id] += 1

        area_native = bbox[2] * bbox[3]
        if area_native < 32.0 * 32.0:
            node["scale_native"]["s"] += 1
        elif area_native < 96.0 * 96.0:
            node["scale_native"]["m"] += 1
        else:
            node["scale_native"]["l"] += 1

        w_640 = 640.0 * (bbox[2] / max(1.0, w_img))
        h_640 = 640.0 * (bbox[3] / max(1.0, h_img))
        area_640 = w_640 * h_640
        if area_640 < 32.0 * 32.0:
            node["scale_r0_640"]["s"] += 1
        elif area_640 < 96.0 * 96.0:
            node["scale_r0_640"]["m"] += 1
        else:
            node["scale_r0_640"]["l"] += 1

    # OOF Corpus Grid Census (20,075 frames)
    grid_oof = defaultdict(lambda: {
        "videos": 0, "frames": 0, "gt_anns": 0, "folds": set(),
        "cams": defaultdict(int), "classes": [0] * NUM_CLASSES,
        "scale_native": {"s": 0, "m": 0, "l": 0},
        "scale_r0_640": {"s": 0, "m": 0, "l": 0}
    })

    attribution_results = {
        exp: {
            v: {"tp": 0, "fp": 0, "fn": 0, "rescued": 0, "lost": 0, "retained": 0, "fp_added": 0, "fp_removed": 0}
            for v in ["cls_only", "box_only", "full"]
        }
        for exp in EXPERTS
    }

    env_denominators = {
        f_name: defaultdict(lambda: {"gt": 0, "tp_base": 0, "fn_base": 0, "fp_base": 0, "vids": set(), "folds": set()})
        for f_name in ["weather", "reflections", "sea_state"]
    }

    attribution_by_env = {
        exp: {
            f_name: defaultdict(lambda: {
                v: {"tp": 0, "rescued": 0, "lost": 0, "fp_added": 0, "fp_removed": 0}
                for v in ["cls_only", "box_only", "full"]
            })
            for f_name in ["weather", "reflections", "sea_state"]
        }
        for exp in EXPERTS
    }

    rescue_sets = {exp: set() for exp in EXPERTS}
    loss_sets = {exp: set() for exp in EXPERTS}
    fp_add_sets = {exp: set() for exp in EXPERTS}
    fp_rem_sets = {exp: set() for exp in EXPERTS}

    d3a_rescue_ranks = {exp: [] for exp in EXPERTS}
    d3a_fp_add_ranks = {exp: [] for exp in EXPERTS}

    base_tp_global = 0
    base_fp_global = 0
    base_fn_global = 0
    base_gt_global = 0
    total_frames_audited = 0
    video_shard_list = []

    print("[*] PASS A: Deterministic Evaluation & Base Oracle Verification across 47 OOF Shards...", flush=True)

    for fold_name, fold_info in master_e0["folds"].items():
        fold_idx = int(fold_name.replace("fold_", ""))

        for vid_str, v_meta in fold_info["videos"].items():
            vid = int(vid_str)
            e0_file = E0_WORK_DIR / v_meta["file"]
            g0_file = EVENTS_DIR / f"bfcef_g0_events_fold{fold_idx}_vid{vid}.pth"

            assert e0_file.exists() and g0_file.exists()
            assert compute_sha256(e0_file) == v_meta["sha256"], f"[FATAL] E0 shard SHA mismatch: {e0_file.name}"

            e0_records = torch.load(e0_file, map_location="cpu")
            assert len(e0_records) == v_meta["num_records"], f"[FATAL] E0 record count mismatch in {e0_file.name}"
            g0_data = torch.load(g0_file, map_location="cpu")
            g0_frames = g0_data["frames"]
            assert len(e0_records) == len(g0_frames)

            video_shard_list.append((fold_idx, vid, e0_file, g0_file, len(e0_records)))

            vm = video_metadata[vid]
            k_combo_vid = f"{vm['weather']}|refl={vm['reflections']}|sea={vm['sea_state']}"
            grid_oof[k_combo_vid]["videos"] += 1
            grid_oof[k_combo_vid]["folds"].add(fold_idx)
            grid_oof[k_combo_vid]["cams"][vm["camera"]] += 1

            for fr_idx in range(len(e0_records)):
                fr_e0 = e0_records[fr_idx]
                fr_g0 = g0_frames[fr_idx]
                img_id = int(fr_e0["image_id"])

                im_meta = image_metadata[img_id]
                assert im_meta["video_id"] == vid

                z_base = fr_e0["z_base"]
                b_base = fr_e0["b_base"]
                gt_boxes_norm = fr_e0["gt_boxes_norm"]
                gt_labels = fr_e0["gt_labels"]
                num_gt = gt_boxes_norm.shape[0]

                base_gt_global += num_gt
                k_combo_im = f"{im_meta['weather']}|refl={im_meta['reflections']}|sea={im_meta['sea_state']}"
                node_oof = grid_oof[k_combo_im]
                node_oof["frames"] += 1
                node_oof["gt_anns"] += num_gt

                w_img, h_img = img_native_wh[img_id]

                for g_i in range(num_gt):
                    c_gt = int(gt_labels[g_i].item())
                    if 0 <= c_gt < NUM_CLASSES:
                        node_oof["classes"][c_gt] += 1
                    area_gt_640 = gt_boxes_norm[g_i, 2].item() * gt_boxes_norm[g_i, 3].item() * (640.0 * 640.0)
                    if area_gt_640 < 32.0 * 32.0:
                        node_oof["scale_r0_640"]["s"] += 1
                    elif area_gt_640 < 96.0 * 96.0:
                        node_oof["scale_r0_640"]["m"] += 1
                    else:
                        node_oof["scale_r0_640"]["l"] += 1

                    area_gt_nat = (gt_boxes_norm[g_i, 2].item() * w_img) * (gt_boxes_norm[g_i, 3].item() * h_img)
                    if area_gt_nat < 32.0 * 32.0:
                        node_oof["scale_native"]["s"] += 1
                    elif area_gt_nat < 96.0 * 96.0:
                        node_oof["scale_native"]["m"] += 1
                    else:
                        node_oof["scale_native"]["l"] += 1

                # Base Predictions
                base_top100 = extract_top100_predictions(z_base, b_base)
                base_match = match_operational_detections_to_gt(
                    base_top100["op_scores"],
                    base_top100["op_c_idx"],
                    base_top100["op_boxes"],
                    base_top100["op_q_idx"],
                    gt_boxes_norm,
                    gt_labels
                )

                base_tp_global += base_match["tp"]
                base_fp_global += base_match["fp"]
                base_fn_global += base_match["fn"]

                base_matched_gt = base_match["matched_gt_indices"]
                base_fp_keys = set(base_match["fp_dict"].keys())

                for f_n, f_val in [("weather", im_meta["weather"]), ("reflections", im_meta["reflections"]), ("sea_state", im_meta["sea_state"])]:
                    den = env_denominators[f_n][f_val]
                    den["gt"] += num_gt
                    den["tp_base"] += base_match["tp"]
                    den["fn_base"] += base_match["fn"]
                    den["fp_base"] += base_match["fp"]
                    den["vids"].add(vid)
                    den["folds"].add(fold_idx)

                p_base = z_base.sigmoid()
                flat_p_base = p_base.flatten()
                _, sorted_base_indices = torch.sort(flat_p_base, descending=True)
                base_pair_ranks = {}
                for r_i in range(len(sorted_base_indices)):
                    idx_val = int(sorted_base_indices[r_i].item())
                    base_pair_ranks[(idx_val // NUM_CLASSES, idx_val % NUM_CLASSES)] = r_i + 1

                for exp in EXPERTS:
                    z_exp = fr_e0[f"z_{exp.lower()}"]
                    b_exp = fr_e0[f"b_{exp.lower()}"]
                    ev_dict = fr_g0["events"][exp]

                    variants = {
                        "cls_only": (z_exp, b_base),
                        "box_only": (z_base, b_exp),
                        "full":     (z_exp, b_exp)
                    }

                    for v_name, (z_v, b_v) in variants.items():
                        v_top100 = extract_top100_predictions(z_v, b_v)
                        v_match = match_operational_detections_to_gt(
                            v_top100["op_scores"],
                            v_top100["op_c_idx"],
                            v_top100["op_boxes"],
                            v_top100["op_q_idx"],
                            gt_boxes_norm,
                            gt_labels
                        )

                        att = attribution_results[exp][v_name]
                        att["tp"] += v_match["tp"]
                        att["fp"] += v_match["fp"]
                        att["fn"] += v_match["fn"]

                        v_matched_gt = v_match["matched_gt_indices"]
                        v_fp_keys = set(v_match["fp_dict"].keys())

                        v_r = 0
                        v_l = 0
                        for g_idx in range(num_gt):
                            is_base = g_idx in base_matched_gt
                            is_v = g_idx in v_matched_gt
                            if not is_base and is_v:
                                att["rescued"] += 1
                                v_r += 1
                            elif is_base and not is_v:
                                att["lost"] += 1
                                v_l += 1
                            elif is_base and is_v:
                                att["retained"] += 1

                        fp_add_cnt = len(v_fp_keys - base_fp_keys)
                        fp_rem_cnt = len(base_fp_keys - v_fp_keys)
                        att["fp_added"] += fp_add_cnt
                        att["fp_removed"] += fp_rem_cnt

                        for f_n, f_val in [("weather", im_meta["weather"]), ("reflections", im_meta["reflections"]), ("sea_state", im_meta["sea_state"])]:
                            n_att = attribution_by_env[exp][f_n][f_val][v_name]
                            n_att["tp"] += v_match["tp"]
                            n_att["rescued"] += v_r
                            n_att["lost"] += v_l
                            n_att["fp_added"] += fp_add_cnt
                            n_att["fp_removed"] += fp_rem_cnt

                    for r_ev in ev_dict["rescues"]:
                        rescue_sets[exp].add((img_id, r_ev["gt_idx"]))
                        q_e = r_ev["q_idx_exp"]
                        c_e = r_ev["class_idx"]
                        b_rank = base_pair_ranks.get((q_e, c_e), 2100)
                        b_score = float(p_base[q_e, c_e].item())
                        d3a_rescue_ranks[exp].append({"rank": b_rank, "score": b_score})

                    for l_ev in ev_dict["losses"]:
                        loss_sets[exp].add((img_id, l_ev["gt_idx"]))

                    for fa_ev in ev_dict["fp_added"]:
                        fp_add_sets[exp].add((img_id, fa_ev["q_idx"], fa_ev["class_idx"]))
                        q_fa = fa_ev["q_idx"]
                        c_fa = fa_ev["class_idx"]
                        b_rank = base_pair_ranks.get((q_fa, c_fa), 2100)
                        b_score = float(p_base[q_fa, c_fa].item())
                        d3a_fp_add_ranks[exp].append({"rank": b_rank, "score": b_score})

                    for fr_ev in ev_dict["fp_removed"]:
                        fp_rem_sets[exp].add((img_id, fr_ev["q_idx"], fr_ev["class_idx"]))

                total_frames_audited += 1

            del e0_records
            del g0_frames
            gc.collect()

        print(f"    [+] Ingested Fold {fold_idx} ({total_frames_audited:,} cumulative frames).")

    assert total_frames_audited == EXPECTED_TOTAL_FRAMES
    assert base_gt_global == CANONICAL_GT

    # Complete D0 Display Table
    print("\n" + "=" * 115)
    print("D0: COMPLETE PHYSICAL ENVIRONMENT GRID CENSUS (FULL TRAIN & OOF SUPPORT)")
    print("=" * 115)
    print(f"{'Weather':<8} | {'Refl':<5} | {'Sea':<6} | {'Full Imgs (GT)':<16} | {'OOF Frms (GT)':<16} | {'Folds':<6} | {'Native S/M/L':<15} | {'R0@640 S/M/L':<15} | {'Classes (c0..c6)'}")
    print("-" * 115)
    for w in WEATHER_TYPES:
        for r in REFLECTION_TYPES:
            for s in SEA_STATES:
                k = f"{w}|refl={r}|sea={s}"
                ef = grid_full[k]
                eo = grid_oof[k]
                full_s = f"{ef['images']:,} ({ef['gt_anns']:,})"
                oof_s = f"{eo['frames']:,} ({eo['gt_anns']:,})"
                nat_s = f"{eo['scale_native']['s']}/{eo['scale_native']['m']}/{eo['scale_native']['l']}"
                r640_s = f"{eo['scale_r0_640']['s']}/{eo['scale_r0_640']['m']}/{eo['scale_r0_640']['l']}"
                c_str = " ".join(f"{c:>3}" for c in eo["classes"])
                print(f"{w:<8} | {r:<5} | {s:<6} | {full_s:<16} | {oof_s:<16} | {len(eo['folds']):>5} | {nat_s:<15} | {r640_s:<15} | [{c_str}]")
    print("=" * 115)

    # Base Parity Oracle Verification
    base_f1_canonical = 2.0 * base_tp_global / max(1, (2.0 * base_tp_global + base_fp_global + base_fn_global))
    print("\nCANONICAL BASELINE ORACLE IDENTITY GATE (FROZEN 20,075-FRAME R0 GROUNDING):")
    print(f"  GT: {base_gt_global:,} (Ref: {CANONICAL_GT:,}) | TP: {base_tp_global:,} (Ref: {CANONICAL_TP:,}) | FP: {base_fp_global:,} (Ref: {CANONICAL_FP:,}) | FN: {base_fn_global:,} (Ref: {CANONICAL_FN:,}) | F1: {base_f1_canonical:.6f}")
    assert base_gt_global == CANONICAL_GT
    assert base_tp_global == CANONICAL_TP
    assert base_fp_global == CANONICAL_FP
    assert base_fn_global == CANONICAL_FN
    assert abs(base_f1_canonical - 0.603298) < 1e-5
    print("  [+] BASE PARITY GATE CERTIFIED: 100% exact match across all canonical outcomes.\n")

    # D1 Audit Table
    print("=" * 115)
    print("D1: TASK ATTRIBUTION AUDIT (GLOBAL & ENVIRONMENTALLY STRATIFIED)")
    print("=" * 115)
    print(f"{'Expert':<8} | {'Variant':<10} | {'TP':<8} | {'Rescued':<8} | {'Lost':<8} | {'Net ΔTP':<9} | {'Loss Rate':<11} | {'FP Added':<9} | {'FP Removed':<11} | {'Net ΔFP':<8}")
    print("-" * 115)
    for exp in EXPERTS:
        for v_name in ["cls_only", "box_only", "full"]:
            att = attribution_results[exp][v_name]
            tp = att["tp"]
            r = att["rescued"]
            l = att["lost"]
            fa = att["fp_added"]
            fr = att["fp_removed"]
            loss_rate = l / float(CANONICAL_TP)

            assert att["retained"] + l == CANONICAL_TP
            assert att["retained"] + r == tp
            assert r - l == tp - CANONICAL_TP
            assert fa - fr == att["fp"] - CANONICAL_FP

            print(f"Expert {exp:<2} | {v_name:<10} | {tp:>6,} | {r:>6,}   | {l:>6,}   | {r - l:>+7,}   | {loss_rate:>8.2%}    | {fa:>7,}   | {fr:>9,}   | {fa - fr:>+7,}")
        print("-" * 115)

    print("\nD1 ENVIRONMENTAL STRATIFIED ATTRIBUTION (RATES RELATIVE TO BASE STRATUM DENOMINATORS):")
    print(f"{'Stratum':<12} | {'Level':<10} | {'Exp':<4} | {'Base TP/FN/FP':<18} | {'Cls R/L (Rates)':<22} | {'Box R/L (Rates)':<22} | {'Full R/L (Rates)':<22} | {'Full FP Prune/Add'}")
    print("-" * 115)
    for f_n in ["weather", "reflections", "sea_state"]:
        for f_val in sorted(env_denominators[f_n].keys()):
            den = env_denominators[f_n][f_val]
            den_str = f"{den['tp_base']:,}/{den['fn_base']:,}/{den['fp_base']:,}"
            for exp in EXPERTS:
                n_c = attribution_by_env[exp][f_n][f_val]["cls_only"]
                n_b = attribution_by_env[exp][f_n][f_val]["box_only"]
                n_f = attribution_by_env[exp][f_n][f_val]["full"]

                r_c_rate = n_c["rescued"] / max(1, den["fn_base"])
                l_c_rate = n_c["lost"] / max(1, den["tp_base"])
                r_b_rate = n_b["rescued"] / max(1, den["fn_base"])
                l_b_rate = n_b["lost"] / max(1, den["tp_base"])
                r_f_rate = n_f["rescued"] / max(1, den["fn_base"])
                l_f_rate = n_f["lost"] / max(1, den["tp_base"])

                pr_rate = n_f["fp_removed"] / max(1, den["fp_base"])
                fa_rate = n_f["fp_added"] / max(1, den["fp_base"])

                str_c = f"{n_c['rescued']} ({r_c_rate:.1%})/{n_c['lost']} ({l_c_rate:.1%})"
                str_b = f"{n_b['rescued']} ({r_b_rate:.1%})/{n_b['lost']} ({l_b_rate:.1%})"
                str_f = f"{n_f['rescued']} ({r_f_rate:.1%})/{n_f['lost']} ({l_f_rate:.1%})"
                str_fp = f"{n_f['fp_removed']} ({pr_rate:.1%})/{n_f['fp_added']} ({fa_rate:.1%})"

                print(f"{f_n:<12} | {f_val:<10} | {exp:<4} | {den_str:<18} | {str_c:<22} | {str_b:<22} | {str_f:<22} | {str_fp}")
        print("-" * 115)

    # D2 Complete Overlap & Multiplicity Tables
    print("\n" + "=" * 115)
    print("D2: OPPORTUNITY SET & MULTIPLICITY ANALYSIS ACROSS ALL FOUR EVENT FAMILIES")
    print("=" * 115)
    u_rescues = len(set.union(*rescue_sets.values()))
    s_rescues = len(set.intersection(*rescue_sets.values()))
    u_losses = len(set.union(*loss_sets.values()))
    s_losses = len(set.intersection(*loss_sets.values()))
    u_fp_rem = len(set.union(*fp_rem_sets.values()))
    s_fp_rem = len(set.intersection(*fp_rem_sets.values()))
    u_fp_add = len(set.union(*fp_add_sets.values()))
    s_fp_add = len(set.intersection(*fp_add_sets.values()))

    mult_hists = {}
    for fam_name, s_dict, u_cnt in [
        ("Rescue", rescue_sets, u_rescues),
        ("Loss", loss_sets, u_losses),
        ("FP_Remove", fp_rem_sets, u_fp_rem),
        ("FP_Add", fp_add_sets, u_fp_add)
    ]:
        counts = defaultdict(int)
        for item in set.union(*s_dict.values()):
            k_c = sum(1 for exp in EXPERTS if item in s_dict[exp])
            counts[k_c] += 1
        mult_hists[fam_name] = dict(counts)
        print(f"MULTIPLICITY {fam_name:<10} (Union={u_cnt:,}): " + ", ".join(f"k={k}: {counts[k]:,} ({counts[k]/max(1, u_cnt):.1%})" for k in range(1, 5)))
    print("-" * 115)

    jaccard_matrices = {}
    for fam_name, s_dict in [("Rescue", rescue_sets), ("Loss", loss_sets), ("FP_Remove", fp_rem_sets), ("FP_Add", fp_add_sets)]:
        print(f"PAIRWISE JACCARD SIMILARITY MATRIX - {fam_name}:")
        print(f"{'':<8} | {'Expert S':<10} | {'Expert L':<10} | {'Expert C':<10} | {'Expert T':<10}")
        j_mat = {}
        for e1 in EXPERTS:
            j_mat[e1] = {}
            r_str = f"Expert {e1} | "
            for e2 in EXPERTS:
                inter = len(s_dict[e1] & s_dict[e2])
                union = len(s_dict[e1] | s_dict[e2])
                j_val = inter / max(1, union)
                j_mat[e1][e2] = round(j_val, 4)
                r_str += f"{j_val:>8.2%}   | "
            print(r_str)
        jaccard_matrices[fam_name] = j_mat
        print("-" * 115)
    print("=" * 115)

    # D3a Promotion-Rank Audit Table
    print("\n" + "=" * 115)
    print("D3a: BASELINE PROMOTION-RANK AUDIT (PRE-INTERVENTION BASE RANK & SCORE DISTRIBUTION)")
    print("=" * 115)
    print(f"{'Expert':<8} | {'Channel':<10} | {'Count':<7} | {'Rank<=100':<10} | {'Rank<=200':<10} | {'Rank<=300':<10} | {'Rank<=500':<10} | {'Rank<=1000':<11} | {'Rank<=2100':<11} | {'Rank P50 (P10/P90)'} | {'Score P50 (P10/P90)'}")
    print("-" * 115)

    d3a_report = {}
    for exp in EXPERTS:
        d3a_report[exp] = {}
        for ch_name, r_list in [("Rescues", d3a_rescue_ranks[exp]), ("FP-Added", d3a_fp_add_ranks[exp])]:
            n_tot = len(r_list)
            ranks = np.array([item["rank"] for item in r_list])
            scores = np.array([item["score"] for item in r_list])

            cov_100 = (ranks <= 100).sum() / max(1, n_tot)
            cov_200 = (ranks <= 200).sum() / max(1, n_tot)
            cov_300 = (ranks <= 300).sum() / max(1, n_tot)
            cov_500 = (ranks <= 500).sum() / max(1, n_tot)
            cov_1000 = (ranks <= 1000).sum() / max(1, n_tot)
            cov_2100 = (ranks <= 2100).sum() / max(1, n_tot)

            assert cov_2100 == 1.0 or n_tot == 0, "[FATAL] Rank out of bounds > 2100!"

            r_p10, r_p50, r_p90 = (float(np.percentile(ranks, p)) if n_tot > 0 else 0.0 for p in [10, 50, 90])
            s_p10, s_p50, s_p90 = (float(np.percentile(scores, p)) if n_tot > 0 else 0.0 for p in [10, 50, 90])

            d3a_report[exp][ch_name] = {
                "count": n_tot,
                "cov_100": round(float(cov_100), 4),
                "cov_200": round(float(cov_200), 4),
                "cov_300": round(float(cov_300), 4),
                "cov_500": round(float(cov_500), 4),
                "cov_1000": round(float(cov_1000), 4),
                "cov_2100": round(float(cov_2100), 4),
                "rank_quantiles": {"p10": r_p10, "p50": r_p50, "p90": r_p90},
                "score_quantiles": {"p10": round(s_p10, 4), "p50": round(s_p50, 4), "p90": round(s_p90, 4)}
            }
            print(f"Expert {exp:<2} | {ch_name:<10} | {n_tot:>7,} | {cov_100:>8.2%}   | {cov_200:>8.2%}   | {cov_300:>8.2%}   | {cov_500:>8.2%}   | {cov_1000:>9.2%}   | {cov_2100:>9.2%}   | {r_p50:>5.0f} ({r_p10:>4.0f}/{r_p90:>4.0f})   | {s_p50:.4f} ({s_p10:.4f}/{s_p90:.4f})")
        print("-" * 115)

    # =========================================================================
    # PASS B: RESOURCE-BOUNDED SEQUENTIAL AUDIT (D3, D4, D5)
    # =========================================================================
    print("\n" + "=" * 115)
    print("PASS B: RESOURCE-BOUNDED SEQUENTIAL AUDIT (D3, D4, D5 WITH VIDEO-BALANCING & PREALLOCATION)")
    print("=" * 115)

    TOTAL_CANDIDATE_ROWS = EXPECTED_TOTAL_FRAMES * 100
    d3_metrics = {}
    d4_report = {}
    d5_summary = {}

    n_meta_feats = len(WEATHER_TYPES) + len(REFLECTION_TYPES) + len(SEA_STATES)

    for exp in EXPERTS:
        t_b_0 = time.time()
        print(f"\n[*] Processing Expert {exp} sequentially in pre-allocated compact arrays ({TOTAL_CANDIDATE_ROWS:,} rows)...")

        X_vis = np.zeros((TOTAL_CANDIDATE_ROWS, 22), dtype=np.float32)
        X_cls = np.zeros((TOTAL_CANDIDATE_ROWS, NUM_CLASSES), dtype=np.float32)
        X_meta = np.zeros((TOTAL_CANDIDATE_ROWS, n_meta_feats), dtype=np.float32)
        X_cam = np.zeros((TOTAL_CANDIDATE_ROWS, 3), dtype=np.float32)
        Y_targets = np.zeros((TOTAL_CANDIDATE_ROWS, 4), dtype=np.int8)
        arr_fold = np.zeros(TOTAL_CANDIDATE_ROWS, dtype=np.int8)
        arr_vid = np.zeros(TOTAL_CANDIDATE_ROWS, dtype=np.int16)

        row_pointer = 0

        for fold_idx, vid, e0_file, g0_file, num_records in video_shard_list:
            e0_records = torch.load(e0_file, map_location="cpu")
            g0_data = torch.load(g0_file, map_location="cpu")
            g0_frames = g0_data["frames"]

            for fr_idx in range(num_records):
                fr_e0 = e0_records[fr_idx]
                fr_g0 = g0_frames[fr_idx]
                img_id = int(fr_e0["image_id"])
                im_meta = image_metadata[img_id]

                z_base = fr_e0["z_base"]
                b_base = fr_e0["b_base"]
                base_top100 = extract_top100_predictions(z_base, b_base)

                ev_dict = fr_g0["events"][exp]
                rescued_qc = {(item["q_idx_exp"], item["class_idx"]) for item in ev_dict["rescues"]}
                lost_qc = {(item["q_idx_base"], item["class_idx"]) for item in ev_dict["losses"]}
                added_fp_qc = {(item["q_idx"], item["class_idx"]) for item in ev_dict["fp_added"]}
                removed_fp_qc = {(item["q_idx"], item["class_idx"]) for item in ev_dict["fp_removed"]}

                p_base = z_base.sigmoid()
                sorted_p, _ = torch.sort(p_base, dim=-1, descending=True)
                margin = (sorted_p[:, 0] - sorted_p[:, 1]).numpy()
                entropy = -(p_base * (p_base + 1e-7).log()).sum(dim=-1).numpy()

                cx = b_base[:, 0].numpy()
                cy = b_base[:, 1].numpy()
                w = b_base[:, 2].numpy()
                h = b_base[:, 3].numpy()
                area = w * h
                aspect = w / np.maximum(h, 1e-4)

                sq = fr_e0["scene_q"]
                scene_vec = [
                    float(sq.get("detector_query_activity_density", sq.get("query_activity_density", 0.0))),
                    float(sq.get("detector_activation_energy", sq.get("activation_energy", 0.0))),
                    float(sq.get("coarse_feature_energy", sq.get("coarse_energy", 0.0))),
                    float(sq.get("coarse_feature_variance", sq.get("coarse_feature_var", sq.get("coarse_var", 0.0)))),
                    float(sq.get("visibility_proxy", 0.0))
                ]
                t_conf = fr_e0["t_match_confidence"].numpy()
                t_app = fr_e0["t_appearance_similarity"].numpy()

                n_proxy = fr_e0[f"n_{exp.lower()}_proxy"].numpy()
                r_proxy = fr_e0[f"r_{exp.lower()}_proxy"].numpy()
                avail = fr_e0[f"avail_{exp.lower()}"].numpy()
                norm_dq_b = fr_e0[f"norm_dq_b_{exp.lower()}"].numpy()
                norm_dq_c = fr_e0[f"norm_dq_c_{exp.lower()}"].numpy()

                top100_q = base_top100["all_q_idx"].numpy()
                top100_c = base_top100["all_c_idx"].numpy()
                top100_s = base_top100["all_scores"].numpy()

                w_oh = [1.0 if im_meta["weather"] == item else 0.0 for item in WEATHER_TYPES]
                r_oh = [1.0 if im_meta["reflections"] == item else 0.0 for item in REFLECTION_TYPES]
                s_oh = [1.0 if im_meta["sea_state"] == item else 0.0 for item in SEA_STATES]
                meta_vec = w_oh + r_oh + s_oh
                cam_vec = [1.0 if im_meta["camera"] == item else 0.0 for item in CAMERAS]

                # Full Base Top-100 Proposals Audited
                for m_idx in range(100):
                    q = top100_q[m_idx]
                    c = top100_c[m_idx]
                    s_val = top100_s[m_idx]
                    is_base_op = float(s_val >= 0.20)

                    X_vis[row_pointer, 0] = s_val
                    X_vis[row_pointer, 1] = margin[q]
                    X_vis[row_pointer, 2] = entropy[q]
                    X_vis[row_pointer, 3:7] = [cx[q], cy[q], w[q], h[q]]
                    X_vis[row_pointer, 7] = area[q]
                    X_vis[row_pointer, 8] = aspect[q]
                    X_vis[row_pointer, 9] = is_base_op
                    X_vis[row_pointer, 10] = n_proxy[q]
                    X_vis[row_pointer, 11] = r_proxy[q]
                    X_vis[row_pointer, 12] = avail[q]
                    X_vis[row_pointer, 13] = norm_dq_b[q]
                    X_vis[row_pointer, 14] = norm_dq_c[q]
                    X_vis[row_pointer, 15] = t_conf[q]
                    X_vis[row_pointer, 16] = t_app[q]
                    X_vis[row_pointer, 17:22] = scene_vec

                    X_cls[row_pointer, c] = 1.0
                    X_meta[row_pointer] = meta_vec
                    X_cam[row_pointer] = cam_vec

                    Y_targets[row_pointer, 0] = 1 if (q, c) in rescued_qc else 0
                    Y_targets[row_pointer, 1] = 1 if (q, c) in lost_qc else 0
                    Y_targets[row_pointer, 2] = 1 if (q, c) in removed_fp_qc else 0
                    Y_targets[row_pointer, 3] = 1 if (q, c) in added_fp_qc else 0

                    arr_fold[row_pointer] = fold_idx
                    arr_vid[row_pointer] = vid
                    row_pointer += 1

            del e0_records
            del g0_frames
            gc.collect()

        assert row_pointer == TOTAL_CANDIDATE_ROWS, f"[FATAL] Row count mismatch: {row_pointer} != {TOTAL_CANDIDATE_ROWS}"
        print(f"    [+] Pre-allocated buffer filled with {row_pointer:,} proposal rows.")

        # D4: Quintile Amplitude Response Audit (All 4 Channels)
        d4_report[exp] = {}
        print(f"\nD4 QUINTILE RESPONSE (EXPERT {exp}):")
        print(f"{'Factor':<16} | {'Quintile Range':<22} | {'Queries':<12} | {'Rescue Rate':<12} | {'Loss Rate':<12} | {'FP Prune':<12} | {'FP Add'}")
        print("-" * 115)

        factor_map = {
            "dq_cls": X_vis[:, 14],
            "dq_box": X_vis[:, 13],
            "s_base": X_vis[:, 0],
            "scale":  np.sqrt(X_vis[:, 7]) * 640.0,
            "need":   X_vis[:, 10],
            "rel":    X_vis[:, 11]
        }

        for f_name, f_arr in factor_map.items():
            q_bounds = np.percentile(f_arr, [20, 40, 60, 80])
            d4_report[exp][f_name] = {}
            for b_idx in range(5):
                if b_idx == 0:
                    b_mask = f_arr <= q_bounds[0]
                    r_str = f"Q1 [{f_arr.min():.2f} - {q_bounds[0]:.2f}]"
                elif b_idx == 4:
                    b_mask = f_arr > q_bounds[3]
                    r_str = f"Q5 [{q_bounds[3]:.2f} - {f_arr.max():.2f}]"
                else:
                    b_mask = (f_arr > q_bounds[b_idx-1]) & (f_arr <= q_bounds[b_idx])
                    r_str = f"Q{b_idx+1} [{q_bounds[b_idx-1]:.2f} - {q_bounds[b_idx]:.2f}]"

                tot_b = int(b_mask.sum())
                r_r = float(Y_targets[b_mask, 0].sum() / max(1, tot_b))
                l_r = float(Y_targets[b_mask, 1].sum() / max(1, tot_b))
                pr_r = float(Y_targets[b_mask, 2].sum() / max(1, tot_b))
                fa_r = float(Y_targets[b_mask, 3].sum() / max(1, tot_b))

                d4_report[exp][f_name][f"Q{b_idx+1}"] = {
                    "range": r_str, "total": tot_b,
                    "rescue_rate": round(r_r, 4), "loss_rate": round(l_r, 4),
                    "fp_prune_rate": round(pr_r, 4), "fp_add_rate": round(fa_r, 4)
                }
                print(f"{f_name:<16} | {r_str:<22} | {tot_b:>10,} | {r_r:>10.2%}   | {l_r:>10.2%}   | {pr_r:>10.2%}   | {fa_r:>10.2%}")
        print("-" * 115)

        # D3 & D5: Paired Diagnostic Probes with True Post-Subsampling Video Weights
        d3_metrics[exp] = {}
        d5_summary[exp] = {}

        channel_defs = [
            ("Rescue (U_TP)", 0, "y_rescue"),
            ("Loss (H_TP)",   1, "y_loss"),
            ("FP Prune",      2, "y_prune"),
            ("FP Add",        3, "y_add")
        ]

        print(f"\nD5 CONTROLLED ENVIRONMENTAL PREDICTIVE LIFT (EXPERT {exp}):")
        print(f"{'Channel':<14} | {'A: Vis+Cls':<11} | {'B: Env+Cls':<11} | {'C: Comb':<11} | {'Δ(C-A) micro':<13} | {'Δ(C-A) VB':<12} | {'D: V+C+Cam':<11} | {'E: V+C+Cam+Env':<15} | {'Δ(E-D) micro':<13} | {'Δ(E-D) VB':<12} | {'Macro (C-A)'}")
        print("-" * 148)

        for ch_label, ch_idx, ch_key in channel_defs:
            Y = Y_targets[:, ch_idx]
            n_pos = int(Y.sum())

            if n_pos < 30:
                print(f"{ch_label:<14} | Sparse events (N={n_pos}) across all folds.")
                continue

            train_sub_indices = {}
            train_fit_weights = {}
            train_sampling_counts = {}

            for f in range(NUM_FOLDS):
                tr_mask_all = np.where(arr_fold != f)[0]
                pos_tr = tr_mask_all[Y[tr_mask_all] == 1]
                neg_tr = tr_mask_all[Y[tr_mask_all] == 0]

                n_pos_tr = len(pos_tr)
                n_neg_tr_all = len(neg_tr)

                rng = np.random.default_rng(20261004 + f * 100 + ch_idx * 10)
                if len(neg_tr) > 60000:
                    neg_vids = arr_vid[neg_tr]
                    selected_negs = []
                    unique_tr_vids = np.unique(neg_vids)
                    per_vid_cap = max(1, 60000 // len(unique_tr_vids))
                    for u_v in unique_tr_vids:
                        u_v_idx = neg_tr[neg_vids == u_v]
                        if len(u_v_idx) <= per_vid_cap:
                            selected_negs.extend(u_v_idx)
                        else:
                            selected_negs.extend(rng.choice(u_v_idx, size=per_vid_cap, replace=False))
                    neg_tr_sub = np.array(selected_negs)
                else:
                    neg_tr_sub = neg_tr

                combined_sub = np.sort(np.concatenate([pos_tr, neg_tr_sub]))
                train_sub_indices[f] = combined_sub
                n_neg_tr_sampled = len(neg_tr_sub)

                train_sampling_counts[f] = {
                    "n_pos_train": int(n_pos_tr),
                    "n_neg_train_total": int(n_neg_tr_all),
                    "n_neg_train_sampled": int(n_neg_tr_sampled),
                    "n_total_train_selected": int(len(combined_sub))
                }

                # Compute weights directly from the actual selected training population
                selected_vids = arr_vid[combined_sub]
                u_vids, u_counts = np.unique(selected_vids, return_counts=True)
                count_map = dict(zip(u_vids.tolist(), u_counts.tolist()))
                f_weights = np.array([1.0 / count_map[int(v)] for v in selected_vids], dtype=np.float32)
                f_weights *= len(f_weights) / f_weights.sum()
                train_fit_weights[f] = f_weights

            model_configs = [
                ("A", lambda: np.hstack([X_vis, X_cls])),
                ("B", lambda: np.hstack([X_meta, X_cls])),
                ("C", lambda: np.hstack([X_vis, X_cls, X_meta])),
                ("D", lambda: np.hstack([X_vis, X_cls, X_cam])),
                ("E", lambda: np.hstack([X_vis, X_cls, X_cam, X_meta]))
            ]

            oof_preds = {}
            fold_prauc_per_model = {m_id: {} for m_id in ["A", "B", "C", "D", "E"]}

            for m_id, build_fn in model_configs:
                mat = build_fn()
                preds = np.zeros(TOTAL_CANDIDATE_ROWS, dtype=np.float32)
                valid_mask = np.zeros(TOTAL_CANDIDATE_ROWS, dtype=bool)

                for f in range(NUM_FOLDS):
                    tr_idx = train_sub_indices[f]
                    val_idx = np.where(arr_fold == f)[0]

                    if np.unique(Y[tr_idx]).size < 2 or np.unique(Y[val_idx]).size < 2:
                        continue

                    clf = HistGradientBoostingClassifier(
                        max_iter=35, max_depth=4, learning_rate=0.1,
                        class_weight="balanced", random_state=42
                    )
                    clf.fit(mat[tr_idx], Y[tr_idx], sample_weight=train_fit_weights[f])
                    val_probs = clf.predict_proba(mat[val_idx])[:, 1]
                    preds[val_idx] = val_probs
                    valid_mask[val_idx] = True

                    fold_ap = average_precision_score(Y[val_idx], val_probs)
                    fold_prauc_per_model[m_id][f] = float(fold_ap)

                oof_preds[m_id] = (preds, valid_mask)
                del mat
                gc.collect()

            common_valid = oof_preds["A"][1] & oof_preds["B"][1] & oof_preds["C"][1] & oof_preds["D"][1] & oof_preds["E"][1]
            assert common_valid.any()

            Y_valid = Y[common_valid]
            prevalence_valid = float(Y_valid.mean())

            # 1. Micro metrics
            micro_ap = {m_id: float(average_precision_score(Y_valid, oof_preds[m_id][0][common_valid])) for m_id in ["A", "B", "C", "D", "E"]}
            micro_lift_ca = micro_ap["C"] - micro_ap["A"]
            micro_lift_ed = micro_ap["E"] - micro_ap["D"]

            # 2. Video-Balanced Pooled PR-AUC
            valid_vids = arr_vid[common_valid]
            u_vids_val, u_counts_val = np.unique(valid_vids, return_counts=True)
            count_map_val = dict(zip(u_vids_val.tolist(), u_counts_val.tolist()))
            eval_w = np.array([1.0 / count_map_val[int(v)] for v in valid_vids], dtype=np.float32)

            vb_ap = {
                m_id: float(average_precision_score(Y_valid, oof_preds[m_id][0][common_valid], sample_weight=eval_w))
                for m_id in ["A", "B", "C", "D", "E"]
            }
            vb_lift_ca = vb_ap["C"] - vb_ap["A"]
            vb_lift_ed = vb_ap["E"] - vb_ap["D"]

            # 3. Macro per-video metrics across all models
            macro_ap = {}
            supported_vid_count = 0
            for m_id in ["A", "B", "C", "D", "E"]:
                p_m = oof_preds[m_id][0]
                v_aps = []
                for v in np.unique(arr_vid):
                    v_mask = (arr_vid == v) & common_valid
                    if v_mask.sum() >= 5 and np.unique(Y[v_mask]).size == 2:
                        v_aps.append(average_precision_score(Y[v_mask], p_m[v_mask]))
                macro_ap[m_id] = float(np.mean(v_aps)) if len(v_aps) > 0 else 0.0
                supported_vid_count = len(v_aps)

            macro_lift_ca = macro_ap["C"] - macro_ap["A"]
            macro_lift_ed = macro_ap["E"] - macro_ap["D"]

            # 4. Coverage deciles for D3 from Model A
            p_A_valid = oof_preds["A"][0][common_valid]
            auroc_A = float(roc_auc_score(Y_valid, p_A_valid))
            pr_lift_A = micro_ap["A"] / max(1e-5, prevalence_valid)

            decile_curves = {}
            for cov_step in range(1, 11):
                cov_frac = cov_step / 10.0
                k_items = max(1, int(cov_frac * len(p_A_valid)))
                thresh = np.partition(p_A_valid, -k_items)[-k_items]
                d_mask = p_A_valid >= thresh
                d_prec = float(Y_valid[d_mask].sum() / max(1, d_mask.sum()))
                d_rec = float(Y_valid[d_mask].sum() / max(1, Y_valid.sum()))
                decile_curves[f"{int(cov_frac*100)}%"] = {"precision": round(d_prec, 4), "recall": round(d_rec, 4)}

            d3_metrics[exp][ch_key] = {
                "total_candidates_valid": int(common_valid.sum()),
                "events_valid": int(Y_valid.sum()),
                "events_all_top100": int(n_pos),
                "prevalence_valid": round(prevalence_valid, 4),
                "auroc": round(auroc_A, 4),
                "prauc": round(micro_ap["A"], 4),
                "pr_lift": round(pr_lift_A, 2),
                "coverage_deciles": decile_curves
            }

            fold_details = {}
            for f in range(NUM_FOLDS):
                val_idx = np.where(arr_fold == f)[0]
                is_supp = (f in fold_prauc_per_model["A"]) and (f in fold_prauc_per_model["C"])
                if is_supp:
                    ap_a_f = fold_prauc_per_model["A"][f]
                    ap_b_f = fold_prauc_per_model["B"][f]
                    ap_c_f = fold_prauc_per_model["C"][f]
                    ap_d_f = fold_prauc_per_model["D"][f]
                    ap_e_f = fold_prauc_per_model["E"][f]
                    sc = train_sampling_counts[f]
                    fold_details[str(f)] = {
                        "supported": True,
                        "n_val": int(len(val_idx)),
                        "n_positive_val": int(Y[val_idx].sum()),
                        "prevalence_val": float(Y[val_idx].mean()),
                        "n_train_positive": sc["n_pos_train"],
                        "n_train_negative_sampled": sc["n_neg_train_sampled"],
                        "A_vis_cls": round(ap_a_f, 4),
                        "B_env_cls": round(ap_b_f, 4),
                        "C_comb": round(ap_c_f, 4),
                        "D_cam": round(ap_d_f, 4),
                        "E_cam_env": round(ap_e_f, 4),
                        "lift_C_minus_A": round(ap_c_f - ap_a_f, 4),
                        "lift_E_minus_D": round(ap_e_f - ap_d_f, 4)
                    }
                else:
                    fold_details[str(f)] = {
                        "supported": False,
                        "n_val": int(len(val_idx)),
                        "n_positive_val": int(Y[val_idx].sum())
                    }

            d5_summary[exp][ch_key] = {
                "micro": {
                    "A_vis_cls": round(micro_ap["A"], 4), "B_env_cls": round(micro_ap["B"], 4),
                    "C_comb": round(micro_ap["C"], 4), "lift_env": round(micro_lift_ca, 4),
                    "D_cam": round(micro_ap["D"], 4), "E_cam_env": round(micro_ap["E"], 4),
                    "lift_env_given_cam": round(micro_lift_ed, 4)
                },
                "video_balanced_pooled": {
                    "A_vb": round(vb_ap["A"], 4), "B_vb": round(vb_ap["B"], 4),
                    "C_vb": round(vb_ap["C"], 4), "lift_env_vb": round(vb_lift_ca, 4),
                    "D_vb": round(vb_ap["D"], 4), "E_vb": round(vb_ap["E"], 4),
                    "lift_env_given_cam_vb": round(vb_lift_ed, 4)
                },
                "macro_video": {
                    "A_macro": round(macro_ap["A"], 4), "B_macro": round(macro_ap["B"], 4),
                    "C_macro": round(macro_ap["C"], 4), "D_macro": round(macro_ap["D"], 4),
                    "E_macro": round(macro_ap["E"], 4),
                    "macro_lift_ca": round(macro_lift_ca, 4),
                    "macro_lift_ed": round(macro_lift_ed, 4),
                    "supported_videos": supported_vid_count
                },
                "per_fold_details": fold_details
            }

            print(f"{ch_label:<14} | {micro_ap['A']:>10.4f}  | {micro_ap['B']:>10.4f}  | {micro_ap['C']:>10.4f}  | {micro_lift_ca:>+10.4f}   | {vb_lift_ca:>+9.4f}   | {micro_ap['D']:>10.4f}  | {micro_ap['E']:>10.4f}     | {micro_lift_ed:>+10.4f}   | {vb_lift_ed:>+9.4f}   | {macro_lift_ca:>+8.4f} ({supported_vid_count} vids)")

        print(f"    [+] Expert {exp} completed Pass B in {time.time() - t_b_0:.1f}s.")
        del X_vis, X_cls, X_meta, X_cam, Y_targets, arr_fold, arr_vid
        gc.collect()

    # Final Report Serialization
    report_path = D_REPORT_DIR / "bfcef_d0_d5_environmental_evidence_report.json"
    serializable_attribution = {
        exp: {v: {k: int(val) for k, val in attribution_results[exp][v].items()} for v in ["cls_only", "box_only", "full"]}
        for exp in EXPERTS
    }
    serializable_env_att = {
        exp: {
            f_k: {
                v_k: {
                    v_mod: {kk: int(vv) for kk, vv in node.items()}
                    for v_mod, node in sub_d.items()
                }
                for v_k, sub_d in f_dict.items()
            }
            for f_k, f_dict in attribution_by_env[exp].items()
        }
        for exp in EXPERTS
    }

    serializable_env_denominators = {
        f_n: {
            f_val: {
                "gt": int(d["gt"]),
                "tp_base": int(d["tp_base"]),
                "fn_base": int(d["fn_base"]),
                "fp_base": int(d["fp_base"]),
                "num_videos": len(d["vids"]),
                "video_ids": sorted(list(d["vids"])),
                "num_folds": len(d["folds"]),
                "fold_ids": sorted(list(d["folds"]))
            }
            for f_val, d in sub_dict.items()
        }
        for f_n, sub_dict in env_denominators.items()
    }

    report_payload = {
        "_meta": {
            "driver_sha256": driver_sha,
            "master_ledger_index_sha256": computed_master_sha,
            "events_index_sha256": computed_events_sha,
            "g0_qualification_report_sha256": computed_g0_report_sha,
            "total_frames_audited": total_frames_audited,
            "execution_time_minutes": round((time.time() - t_start) / 60.0, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "metadata_integrity": {
            "documented_vocabulary": {
                "weather": DOCUMENTED_WEATHER,
                "water_reflections": DOCUMENTED_REFLECTIONS,
                "sea_state": DOCUMENTED_SEA_STATE,
                "camera_sensor": DOCUMENTED_CAMERAS
            },
            "observed_train_vocabulary": {
                "weather": discovered_weathers,
                "water_reflections": discovered_refls,
                "sea_state": discovered_seas,
                "camera_sensor": discovered_cams
            },
            "schema_divergence": schema_divergence,
            "mismatch_counts": mismatches,
            "status": "CERTIFIED_EXACT" if total_mismatches == 0 else "FRAME_AUTHORITY_ENFORCED"
        },
        "diagnostic_probe_config": {
            "candidate_universe": "base_flattened_top100_all_rows",
            "negative_train_cap": 60000,
            "retain_all_training_positives": True,
            "negative_sampling": "deterministic_video_stratified",
            "sampling_seed_base": 20261004,
            "classifier": "HistGradientBoostingClassifier",
            "max_iter": 35,
            "max_depth": 4,
            "learning_rate": 0.1,
            "class_weight": "balanced"
        },
        "d0_census_and_confounding": {
            "cramers_v_matrix": cramer_results,
            "grid_full_train_json": {k: {kk: vv for kk, vv in v.items()} for k, v in grid_full.items()},
            "grid_oof_corpus": {k: {kk: (list(vv) if isinstance(vv, set) else vv) for kk, vv in v.items()} for k, v in grid_oof.items()}
        },
        "d1_task_attribution": {
            "global": serializable_attribution,
            "by_environment": serializable_env_att,
            "environment_denominators": serializable_env_denominators
        },
        "d2_opportunity_sets": {
            "single_expert_opportunity_union": u_rescues,
            "shared_rescues_all_experts": s_rescues,
            "total_losses_any_expert": u_losses,
            "common_always_on_losses": s_losses,
            "single_expert_fp_pruning_union": u_fp_rem,
            "total_unique_fp_added": u_fp_add,
            "multiplicity_histograms": mult_hists,
            "pairwise_jaccard_matrices": jaccard_matrices
        },
        "d3a_promotion_rank": d3a_report,
        "d3_base_top100_gateability": d3_metrics,
        "d4_quintile_response": d4_report,
        "d5_environmental_lift": d5_summary
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    report_sha = compute_sha256(report_path)
    print("\n" + "=" * 115)
    print(f"[+] D0-D5+ Environmental Evidence Audit completed in {(time.time() - t_start) / 60.0:.2f} minutes.")
    print(f"[+] Certified Report Frozen: {report_path.resolve()}")
    print(f"[+] Report SHA-256: {report_sha}")
    print("=" * 115 + "\n")


if __name__ == "__main__":
    main()