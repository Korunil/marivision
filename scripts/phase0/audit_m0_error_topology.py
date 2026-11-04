"""
D:/MARIVISION/H0_workbench/eval_tools/audit_m0_error_topology.py
Phase-II Proof-of-Launch Stage 0 (PoL-0): Part 3 - Hardened Frozen M0 Baseline Error Topology Audit

Strict Compliance Checklist:
  - Provenance: Locked M0 checkpoint path with SHA-256 verification and cache envelope invalidation.
  - Precision: Pure FP32 inference (NO autocast/AMP).
  - Raw Predictions: Preserves top-100 raw predictions per image without score cutoff.
  - Schema Check: Hard-assert sum(iscrowd) == 0.
  - Layer 1: Top-100 GT Coverage (Class-Aware vs. Class-Agnostic) by size bin.
  - Layer 2: Mutually exclusive failure taxonomy (Complete Miss <0.10, Localization [0.10, 0.30), Confusion >=0.30).
  - Spatial Neutrality: Categorized strictly as 'above' vs 'below' curated geometric horizon.
  - Environmental Stratification: Pooled instance rates AND per-video macro medians.
  - Research Artifacts: 4 machine-readable files (JSON summary, FN CSV, FP CSV, Environment Macro CSV).
"""

import os
import sys
import json
import hashlib
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from scipy.optimize import linear_sum_assignment

DETR_PATH = Path("D:/MARIVISION/Deformable-DETR")
if str(DETR_PATH) not in sys.path:
    sys.path.insert(0, str(DETR_PATH))

if "R0_MSDA_BACKEND" not in os.environ:
    os.environ["R0_MSDA_BACKEND"] = "pytorch"

from models.deformable_detr import build
from util.misc import collate_fn
from datasets.seaclips_detection import SeaClipsDetection
from r0_preflight import get_authoritative_r0_args

ZIP_PATH = Path("D:/MARIVISION/SeaClips.zip")
TRAIN_JSON = Path("D:/MARIVISION/seaclips-train.json")
TRAIN_TRACKS = Path("D:/MARIVISION/H0_workbench/horizon_tracks/seaclips_horizon_tracks_train_semantic_v2.json")
OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
CACHE_FILE = OUT_DIR / "cache_m0_train_raw_top100.json"

# Authoritative frozen M0 baseline candidate paths
M0_EXPLICIT_PATHS = [
    Path("D:/MARIVISION/checkpoints_m0_baseline/checkpoint_best.pth"),
    Path("D:/MARIVISION/checkpoints_m0/checkpoint_best.pth"),
    Path("D:/MARIVISION/r0_baseline/checkpoint_best.pth"),
    Path("D:/MARIVISION/checkpoints_r0/checkpoint_best.pth"),
    Path("D:/MARIVISION/checkpoints/checkpoint_best.pth"),
    Path("D:/MARIVISION/Deformable-DETR/checkpoints/checkpoint_best.pth")
]

OPERATING_SCORE_THRESHOLD = 0.20
EVAL_IOU_THRESHOLD = 0.30


def compute_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def resolve_m0_checkpoint():
    env_override = os.environ.get("M0_CKPT")
    if env_override and Path(env_override).exists():
        return Path(env_override)
    for p in M0_EXPLICIT_PATHS:
        if p.exists():
            return p
    for p in Path("D:/MARIVISION").glob("**/checkpoint_best.pth"):
        p_str = str(p).lower().replace("\\", "/")
        if "m1" not in p_str and "m2" not in p_str and "hrpc" not in p_str:
            return p
    raise FileNotFoundError("[FATAL] Could not find M0 baseline checkpoint_best.pth")


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


def compute_canonical_rho(cy, h_at_x, eps=1e-6):
    if cy < h_at_x:
        return (cy - h_at_x) / (h_at_x + eps)
    else:
        return (cy - h_at_x) / (1.0 - h_at_x + eps)


def first_valid(*values):
    for v in values:
        if v is None: continue
        s = str(v).strip()
        if s and s.lower() not in {"none", "null", "nan", "unknown", "undefined"}:
            return s
    return "unknown"


def box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = box1[2] * box1[3] + box2[2] * box2[3] - inter
    return inter / union if union > 0.0 else 0.0


def extract_m0_raw_predictions_fp32(m0_ckpt_path, m0_hash, train_json_hash, dataset, loader, device):
    print(f"[*] Extracting raw top-100 predictions across {len(dataset):,} frames in pure FP32 (NO AMP)...")
    args = get_authoritative_r0_args()
    if hasattr(args, "conditioning_mode"):
        args.conditioning_mode = "none"

    model, _, postprocessors = build(args)
    ckpt = torch.load(m0_ckpt_path, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    detections_by_image = {}
    total_batches = len(loader)

    with torch.inference_mode():
        for batch_idx, (samples, targets) in enumerate(loader):
            samples = samples.to(device)
            # Pure FP32 - No autocast
            outputs = model(samples)

            orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)
            results = postprocessors["bbox"](outputs, orig_sizes)

            for target, result in zip(targets, results):
                img_id = target["image_id"].item()
                scores = result["scores"].cpu()
                labels = result["labels"].cpu()
                boxes = result["boxes"].cpu()

                # Extract top 100 detections per image without score thresholding
                k = min(100, len(scores))
                top_scores, top_indices = torch.topk(scores, k=k)

                img_dets = []
                for s, idx in zip(top_scores, top_indices):
                    l = labels[idx].item()
                    box = boxes[idx].tolist()
                    cat_id = dataset.label_to_cat_id[l]
                    img_dets.append({
                        "category_id": cat_id,
                        "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                        "score": float(s.item())
                    })
                detections_by_image[img_id] = img_dets

            if (batch_idx + 1) % 500 == 0 or (batch_idx + 1) == total_batches:
                print(f"    Inference Progress: [{batch_idx+1:5d}/{total_batches:5d}] batches ({(batch_idx+1)/total_batches*100:5.1f}%)")

    # Serialize cache with metadata envelope
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_payload = {
        "_meta": {
            "m0_checkpoint": str(m0_ckpt_path.resolve()),
            "m0_checkpoint_sha256": m0_hash,
            "train_json_sha256": train_json_hash,
            "precision": "FP32",
            "image_count": len(detections_by_image)
        },
        "detections": {str(k): v for k, v in detections_by_image.items()}
    }
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache_payload, f)
    print(f"[+] Raw predictions cached successfully with provenance envelope to: {CACHE_FILE}")
    return detections_by_image


def load_validated_cache(cache_path, current_m0_hash, current_json_hash):
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "_meta" not in data or "detections" not in data:
            print("[!] Cache missing _meta envelope. Invalidating...")
            return None
        meta = data["_meta"]
        if meta.get("m0_checkpoint_sha256") != current_m0_hash:
            print("[!] Checkpoint SHA-256 mismatch in cache. Invalidating...")
            return None
        if meta.get("train_json_sha256") != current_json_hash:
            print("[!] Train JSON SHA-256 mismatch in cache. Invalidating...")
            return None
        print(f"[+] Cache validated against checkpoint & dataset hashes ({meta.get('image_count')} images).")
        return {int(k): v for k, v in data["detections"].items()}
    except Exception as e:
        print(f"[!] Error reading cache: {e}. Invalidating...")
        return None


def main():
    print("=" * 115)
    print("PROOF-OF-LAUNCH STAGE 0 (PoL-0): PART 3 - FROZEN M0 BASELINE ERROR TOPOLOGY AUDIT")
    print(f"Dataset Target: {TRAIN_JSON.resolve()}")
    print("=" * 115)

    assert TRAIN_JSON.exists(), f"[FATAL] Missing {TRAIN_JSON}"
    assert TRAIN_TRACKS.exists(), f"[FATAL] Missing {TRAIN_TRACKS}"

    train_json_hash = compute_sha256(TRAIN_JSON)
    train_tracks_hash = compute_sha256(TRAIN_TRACKS)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Target Hardware Device: {device}")

    dataset = SeaClipsDetection(ZIP_PATH, TRAIN_JSON, is_train=False, random_flip=False, horizon_tracks_path=TRAIN_TRACKS)
    coco = dataset.coco
    flat_horizon_map = load_and_flatten_horizon_tracks(TRAIN_TRACKS)

    img_ids = sorted(coco.getImgIds())
    ann_ids = sorted(coco.getAnnIds())
    cats = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}

    # Verify iscrowd constraint
    crowd_sum = sum(int(a.get("iscrowd", 0)) for a in coco.loadAnns(ann_ids))
    assert crowd_sum == 0, f"[FATAL] Expected 0 iscrowd annotations, found {crowd_sum}"
    print(f"[*] Schema Check Certified: sum(iscrowd) == 0 across all {len(ann_ids):,} annotations.")

    # Locate and verify M0 Checkpoint
    m0_ckpt_path = resolve_m0_checkpoint()
    m0_ckpt_hash = compute_sha256(m0_ckpt_path)
    print(f"[*] M0 Baseline Checkpoint: {m0_ckpt_path.resolve()}")
    print(f"[*] Checkpoint SHA-256:     {m0_ckpt_hash}")

    # Load or run raw inference
    detections_by_image = load_validated_cache(CACHE_FILE, m0_ckpt_hash, train_json_hash)
    if detections_by_image is None:
        loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)
        detections_by_image = extract_m0_raw_predictions_fp32(m0_ckpt_path, m0_ckpt_hash, train_json_hash, dataset, loader, device)

    # Build image and video metadata lookup
    video_lookup = {}
    if "videos" in coco.dataset:
        for v in coco.dataset["videos"]:
            if "id" in v: video_lookup[v["id"]] = v
            if "video_name" in v: video_lookup[v["video_name"]] = v

    img_meta = {}
    for img_id in img_ids:
        info = coco.loadImgs([img_id])[0]
        fname = info["file_name"]
        clean_fname = fname.replace("\\", "/").strip().lstrip("/")
        h_vec = flat_horizon_map[clean_fname]

        v_info = video_lookup.get(info.get("video_id"), {})
        vid_name = v_info.get("video_name", clean_fname.split("/")[0])

        sensor = first_valid(info.get("camera_sensor"), v_info.get("camera_sensor"))
        refl = first_valid(info.get("water_reflections"), v_info.get("water_reflections"))
        sea = first_valid(info.get("sea_state"), v_info.get("sea_state"))
        weather = first_valid(info.get("weather"), v_info.get("weather"))

        img_meta[img_id] = {
            "W": float(info["width"]),
            "H": float(info["height"]),
            "eta_L": float(h_vec[0]),
            "eta_R": float(h_vec[1]),
            "g": float(h_vec[2]),
            "video_id": vid_name,
            "sensor": sensor,
            "reflections": refl,
            "sea_state": sea,
            "weather": weather
        }

    # --------------------------------------------------------------------------
    # LAYER 1: THRESHOLD-INDEPENDENT TOP-100 GT COVERAGE
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("LAYER 1: THRESHOLD-INDEPENDENT TOP-100 GT COVERAGE (IoU >= 0.30)")
    print("Comparing Class-Aware vs. Class-Agnostic Top-100 Coverage to Isolate Representation from Classification")
    print("=" * 115)

    size_bins = ["<4px", "[4,8)px", "[8,16)px", ">=16px"]
    layer1_records = []

    for img_id in img_ids:
        meta = img_meta[img_id]
        W, H = meta["W"], meta["H"]
        s_x = 640.0 / W
        s_y = 640.0 / H

        g_val = meta["g"]
        eta_L, eta_R = meta["eta_L"], meta["eta_R"]

        raw_anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
        preds = detections_by_image.get(img_id, [])

        for gt in raw_anns:
            bw, bh = gt["bbox"][2], gt["bbox"][3]
            min_dim_warp = min(bw * s_x, bh * s_y)

            if min_dim_warp < 4.0: sb = "<4px"
            elif min_dim_warp < 8.0: sb = "[4,8)px"
            elif min_dim_warp < 16.0: sb = "[8,16)px"
            else: sb = ">=16px"

            cx = (gt["bbox"][0] + bw / 2.0) / W
            cy = (gt["bbox"][1] + bh / 2.0) / H
            h_at_x = (1.0 - cx) * eta_L + cx * eta_R
            rho_c = abs(compute_canonical_rho(cy, h_at_x))

            ca_matches = [d for d in preds if d["category_id"] == gt["category_id"] and box_iou(d["bbox"], gt["bbox"]) >= EVAL_IOU_THRESHOLD]
            has_ca = len(ca_matches) > 0
            best_ca_score = max((d["score"] for d in ca_matches), default=0.0)

            agn_matches = [d for d in preds if box_iou(d["bbox"], gt["bbox"]) >= EVAL_IOU_THRESHOLD]
            has_agn = len(agn_matches) > 0
            best_agn_score = max((d["score"] for d in agn_matches), default=0.0)

            layer1_records.append({
                "ann_id": gt["id"],
                "image_id": img_id,
                "video_id": meta["video_id"],
                "category": cats[gt["category_id"]],
                "size_bin": sb,
                "min_dim_warp": min_dim_warp,
                "has_ca_coverage": has_ca,
                "has_agn_coverage": has_agn,
                "best_ca_score": best_ca_score,
                "best_agn_score": best_agn_score,
                "g": g_val,
                "rho": rho_c,
                "sensor": meta["sensor"],
                "reflections": meta["reflections"],
                "sea_state": meta["sea_state"],
                "weather": meta["weather"]
            })

    total_eval_gt = len(layer1_records)
    print(f"Total Evaluated Ground Truth Targets: {total_eval_gt:,}")
    print("-" * 115)
    print(f"{'Size Bin (Warped)':<20} | {'Total GT':<10} | {'Class-Aware Top100':<22} | {'Class-Agnostic Top100':<24} | {'Classification Gap':<20} | {'Mean CA Score':<14}")
    print("-" * 115)

    for sb in size_bins:
        sub = [r for r in layer1_records if r["size_bin"] == sb]
        n_tot = len(sub)
        n_ca = sum(1 for r in sub if r["has_ca_coverage"])
        n_agn = sum(1 for r in sub if r["has_agn_coverage"])

        ca_pct = (n_ca / n_tot * 100.0) if n_tot > 0 else 0.0
        agn_pct = (n_agn / n_tot * 100.0) if n_tot > 0 else 0.0
        gap = agn_pct - ca_pct

        scores_ca = [r["best_ca_score"] for r in sub if r["has_ca_coverage"]]
        mean_s = np.mean(scores_ca) if scores_ca else 0.0

        print(f"{sb:<20} | {n_tot:<10,d} | {n_ca:6,d} ({ca_pct:5.2f}%)       | {n_agn:6,d} ({agn_pct:5.2f}%)         | {gap:+5.2f} pp{'':<12} | {mean_s:6.3f}")

    # --------------------------------------------------------------------------
    # LAYER 2: MUTUALLY EXCLUSIVE BIPARTITE ASSIGNMENT (Operating Score >= 0.20)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print(f"LAYER 2: DIAGNOSTIC FAILURE TAXONOMY (Operating Score >= {OPERATING_SCORE_THRESHOLD:.2f}, IoU >= {EVAL_IOU_THRESHOLD:.2f})")
    print("Strict Mutually Exclusive Hungarian Bipartite Assignment for Failure Mode Decomposition")
    print("=" * 115)

    gt_failure_records = []
    fp_diagnostic_records = []

    for img_id in img_ids:
        meta = img_meta[img_id]
        W, H = meta["W"], meta["H"]
        s_x = 640.0 / W
        s_y = 640.0 / H

        g_val = meta["g"]
        eta_L, eta_R = meta["eta_L"], meta["eta_R"]

        raw_anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
        preds = [d for d in detections_by_image.get(img_id, []) if d["score"] >= OPERATING_SCORE_THRESHOLD]

        M = len(raw_anns)
        K = len(preds)

        gt_assigned = [False] * M
        dt_assigned = [False] * K
        gt_mode = [None] * M
        dt_mode = ["FP_BACKGROUND"] * K

        if M > 0 and K > 0:
            cost_matrix = np.zeros((M, K), dtype=float)
            same_cat_mask = np.zeros((M, K), dtype=bool)
            for m in range(M):
                for k in range(K):
                    cost_matrix[m, k] = box_iou(raw_anns[m]["bbox"], preds[k]["bbox"])
                    same_cat_mask[m, k] = (raw_anns[m]["category_id"] == preds[k]["category_id"])

            # Pass 1: True Positives (Same category, IoU >= 0.30)
            tp_cost = np.where((cost_matrix >= EVAL_IOU_THRESHOLD) & same_cat_mask, 1.0 - cost_matrix, 1e6)
            row_ind, col_ind = linear_sum_assignment(tp_cost)

            for r, c in zip(row_ind, col_ind):
                if tp_cost[r, c] < 1e5:
                    gt_assigned[r] = True
                    dt_assigned[c] = True
                    gt_mode[r] = "TP"
                    dt_mode[c] = "TP"

            # Pass 2: Classification Confusion (Different category, IoU >= 0.30)
            conf_cost = np.where((cost_matrix >= EVAL_IOU_THRESHOLD) & (~same_cat_mask), 1.0 - cost_matrix, 1e6)
            for r in range(M):
                if gt_assigned[r]: conf_cost[r, :] = 1e6
            for c in range(K):
                if dt_assigned[c]: conf_cost[:, c] = 1e6

            row_ind, col_ind = linear_sum_assignment(conf_cost)
            for r, c in zip(row_ind, col_ind):
                if conf_cost[r, c] < 1e5:
                    gt_assigned[r] = True
                    dt_assigned[c] = True
                    gt_mode[r] = "CLASS_CONFUSION"
                    dt_mode[c] = "FP_CONFUSION"

            # Pass 3: Localization Failure (Same category, 0.10 <= IoU < 0.30)
            loc_cost = np.where((cost_matrix >= 0.10) & (cost_matrix < EVAL_IOU_THRESHOLD) & same_cat_mask, 1.0 - cost_matrix, 1e6)
            for r in range(M):
                if gt_assigned[r]: loc_cost[r, :] = 1e6
            for c in range(K):
                if dt_assigned[c]: loc_cost[:, c] = 1e6

            row_ind, col_ind = linear_sum_assignment(loc_cost)
            for r, c in zip(row_ind, col_ind):
                if loc_cost[r, c] < 1e5:
                    gt_assigned[r] = True
                    dt_assigned[c] = True
                    gt_mode[r] = "LOCALIZATION_FAILURE"
                    dt_mode[c] = "FP_LOCALIZATION"

            # Pass 4: Mathematical Complete Miss Verification
            for m in range(M):
                if not gt_assigned[m]:
                    max_iou_all = np.max(cost_matrix[m, :]) if K > 0 else 0.0
                    if max_iou_all < 0.10:
                        gt_mode[m] = "COMPLETE_MISS"
                    else:
                        gt_mode[m] = "SUB_THRESHOLD_OVERLAP"
        else:
            for m in range(M):
                gt_mode[m] = "COMPLETE_MISS"

        for m in range(M):
            gt = raw_anns[m]
            bw, bh = gt["bbox"][2], gt["bbox"][3]
            min_dim_warp = min(bw * s_x, bh * s_y)
            if min_dim_warp < 4.0: sb = "<4px"
            elif min_dim_warp < 8.0: sb = "[4,8)px"
            elif min_dim_warp < 16.0: sb = "[8,16)px"
            else: sb = ">=16px"

            cx = (gt["bbox"][0] + bw / 2.0) / W
            cy = (gt["bbox"][1] + bh / 2.0) / H
            h_at_x = (1.0 - cx) * eta_L + cx * eta_R
            rho_c = abs(compute_canonical_rho(cy, h_at_x))

            gt_failure_records.append({
                "ann_id": gt["id"],
                "image_id": img_id,
                "video_id": meta["video_id"],
                "category": cats[gt["category_id"]],
                "size_bin": sb,
                "min_dim_warp": min_dim_warp,
                "mode": gt_mode[m],
                "is_tp": (gt_mode[m] == "TP"),
                "g": g_val,
                "rho": rho_c,
                "sensor": meta["sensor"],
                "reflections": meta["reflections"],
                "sea_state": meta["sea_state"],
                "weather": meta["weather"]
            })

        for k in range(K):
            if dt_mode[k] == "TP":
                continue
            dt = preds[k]
            cx = (dt["bbox"][0] + dt["bbox"][2] / 2.0) / W
            cy = (dt["bbox"][1] + dt["bbox"][3] / 2.0) / H
            h_at_x = (1.0 - cx) * eta_L + cx * eta_R

            spatial_pos = "below_curated_horizon" if cy >= h_at_x else "above_curated_horizon"

            fp_diagnostic_records.append({
                "image_id": img_id,
                "video_id": meta["video_id"],
                "category": cats[dt["category_id"]],
                "score": dt["score"],
                "fp_mode": dt_mode[k],
                "g": g_val,
                "spatial_pos": spatial_pos if g_val == 1.0 else "no_curated_horizon",
                "sensor": meta["sensor"],
                "reflections": meta["reflections"],
                "sea_state": meta["sea_state"],
                "weather": meta["weather"]
            })

    print(f"{'Size Bin':<15} | {'Total GT':<10} | {'True Positives':<16} | {'Complete Miss (<0.10)':<25} | {'Localization [0.10, 0.30)':<28} | {'Class Confusion':<18}")
    print("-" * 115)
    for sb in size_bins:
        sub = [r for r in gt_failure_records if r["size_bin"] == sb]
        n_tot = len(sub)
        n_tp = sum(1 for r in sub if r["mode"] == "TP")
        n_miss = sum(1 for r in sub if r["mode"] == "COMPLETE_MISS")
        n_loc = sum(1 for r in sub if r["mode"] == "LOCALIZATION_FAILURE")
        n_conf = sum(1 for r in sub if r["mode"] == "CLASS_CONFUSION")

        p_tp = (n_tp / n_tot * 100.0) if n_tot > 0 else 0.0
        p_miss = (n_miss / n_tot * 100.0) if n_tot > 0 else 0.0
        p_loc = (n_loc / n_tot * 100.0) if n_tot > 0 else 0.0
        p_conf = (n_conf / n_tot * 100.0) if n_tot > 0 else 0.0

        print(f"{sb:<15} | {n_tot:<10,d} | {n_tp:6,d} ({p_tp:5.1f}%)   | {n_miss:6,d} ({p_miss:5.1f}%)            | {n_loc:6,d} ({p_loc:5.1f}%)               | {n_conf:5,d} ({p_conf:5.1f}%)")

    # --------------------------------------------------------------------------
    # TASK 3: ENVIRONMENTAL STRATIFICATION: POOLED AND PER-VIDEO MACRO (D-3112)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("TASK 3: ENVIRONMENTAL ERROR STRATIFICATION (POOLED INSTANCE vs. PER-VIDEO MACRO MEDIAN)")
    print("=" * 115)

    macro_csv_records = []

    def evaluate_env_dimension(dim_name):
        dim_values = sorted(list(set(img_meta[i][dim_name] for i in img_ids)))
        print(f"--- Dimension: {dim_name.upper()} ---")
        print(f"  {'Condition':<18} | {'Videos':<8} | {'Frames':<8} | {'Pooled FN Rate':<16} | {'Macro Median FN Rate':<22} | {'Pooled FP/Frame':<17} | {'Macro Median FP/Frame':<22}")
        print("  " + "-" * 115)

        for val in dim_values:
            val_imgs = [i for i in img_ids if img_meta[i][dim_name] == val]
            val_vids = sorted(list(set(img_meta[i]["video_id"] for i in val_imgs)))

            val_gts = [r for r in gt_failure_records if r[dim_name] == val]
            val_fps = [r for r in fp_diagnostic_records if r[dim_name] == val]

            pooled_fn_rate = (sum(1 for r in val_gts if not r["is_tp"]) / len(val_gts) * 100.0) if val_gts else 0.0
            pooled_fp_rate = (len(val_fps) / len(val_imgs)) if val_imgs else 0.0

            vid_fn_rates = []
            vid_fp_rates = []

            for vid in val_vids:
                v_imgs = [i for i in val_imgs if img_meta[i]["video_id"] == vid]
                v_gts = [r for r in val_gts if r["video_id"] == vid]
                v_fps = [r for r in val_fps if r["video_id"] == vid]

                if v_gts:
                    v_fn = sum(1 for r in v_gts if not r["is_tp"]) / len(v_gts) * 100.0
                    vid_fn_rates.append(v_fn)
                if v_imgs:
                    v_fp = len(v_fps) / len(v_imgs)
                    vid_fp_rates.append(v_fp)

            macro_fn_med = float(np.median(vid_fn_rates)) if vid_fn_rates else 0.0
            macro_fp_med = float(np.median(vid_fp_rates)) if vid_fp_rates else 0.0

            print(f"  {val:<18} | {len(val_vids):<8d} | {len(val_imgs):<8d} | {pooled_fn_rate:6.2f}%{'':<9} | {macro_fn_med:6.2f}%{'':<15} | {pooled_fp_rate:6.3f} FP/frame{'':<4} | {macro_fp_med:6.3f} FP/frame")

            macro_csv_records.append({
                "dimension": dim_name,
                "value": val,
                "num_videos": len(val_vids),
                "num_frames": len(val_imgs),
                "total_gt": len(val_gts),
                "pooled_fn_rate": pooled_fn_rate,
                "macro_median_fn_rate": macro_fn_med,
                "total_fp": len(val_fps),
                "pooled_fp_rate": pooled_fp_rate,
                "macro_median_fp_rate": macro_fp_med
            })
        print()

    evaluate_env_dimension("reflections")
    evaluate_env_dimension("sea_state")
    evaluate_env_dimension("weather")
    evaluate_env_dimension("sensor")

    # --------------------------------------------------------------------------
    # TASK 4: SPATIAL FALSE POSITIVE BREAKDOWN ON ACTIVE FRAMES
    # --------------------------------------------------------------------------
    print("=" * 115)
    print("TASK 4: SPATIAL FALSE POSITIVE DISTRIBUTION ON ACTIVE (g=1) FRAMES")
    print("=" * 115)

    act_fps = [r for r in fp_diagnostic_records if r["g"] == 1.0]
    n_act = len(act_fps)
    n_below = sum(1 for r in act_fps if r["spatial_pos"] == "below_curated_horizon")
    n_above = sum(1 for r in act_fps if r["spatial_pos"] == "above_curated_horizon")

    print(f"Total Confident False Positives on Active Frames: {n_act:,}")
    print(f"  Below Curated Geometric Horizon: {n_below:6,d} ({n_below/n_act*100:5.2f}%)")
    print(f"  Above Curated Geometric Horizon: {n_above:6,d} ({n_above/n_act*100:5.2f}%)")

    # --------------------------------------------------------------------------
    # EXPORT MASTER RECORDS (4 Artifacts)
    # --------------------------------------------------------------------------
    json_path = OUT_DIR / "pol0_m0_error_topology.json"
    fn_csv_path = OUT_DIR / "pol0_m0_fn_breakdown.csv"
    fp_csv_path = OUT_DIR / "pol0_m0_fp_breakdown.csv"
    macro_csv_path = OUT_DIR / "pol0_m0_env_macro.csv"

    with open(fn_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(gt_failure_records[0].keys()))
        w.writeheader()
        w.writerows(gt_failure_records)

    with open(fp_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fp_diagnostic_records[0].keys()))
        w.writeheader()
        w.writerows(fp_diagnostic_records)

    with open(macro_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(macro_csv_records[0].keys()))
        w.writeheader()
        w.writerows(macro_csv_records)

    summary_export = {
        "_meta": {
            "audit_version": "PoL-0_Part-3_Certified_M0_Error_Topology",
            "m0_checkpoint": str(m0_ckpt_path.resolve()),
            "m0_checkpoint_sha256": m0_ckpt_hash,
            "operating_score_threshold": OPERATING_SCORE_THRESHOLD,
            "eval_iou_threshold": EVAL_IOU_THRESHOLD,
            "total_gt_evaluated": total_eval_gt
        },
        "layer1_top100_coverage_by_size": {
            sb: {
                "total": len([r for r in layer1_records if r["size_bin"] == sb]),
                "class_aware_coverage": sum(1 for r in layer1_records if r["size_bin"] == sb and r["has_ca_coverage"]) / len([r for r in layer1_records if r["size_bin"] == sb]),
                "class_agnostic_coverage": sum(1 for r in layer1_records if r["size_bin"] == sb and r["has_agn_coverage"]) / len([r for r in layer1_records if r["size_bin"] == sb])
            } for sb in size_bins
        },
        "layer2_taxonomy_totals": {
            "total_tp": sum(1 for r in gt_failure_records if r["is_tp"]),
            "complete_miss": sum(1 for r in gt_failure_records if r["mode"] == "COMPLETE_MISS"),
            "localization_failure": sum(1 for r in gt_failure_records if r["mode"] == "LOCALIZATION_FAILURE"),
            "class_confusion": sum(1 for r in gt_failure_records if r["mode"] == "CLASS_CONFUSION"),
            "total_fp": len(fp_diagnostic_records)
        },
        "spatial_fp_active": {
            "total_active_fp": n_act,
            "below_horizon": n_below,
            "above_horizon": n_above
        }
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_export, f, indent=2)

    print("\n" + "=" * 115)
    print(f"[+] All 4 error topology research records written successfully:")
    print(f"    1. {json_path}")
    print(f"    2. {fn_csv_path}")
    print(f"    3. {fp_csv_path}")
    print(f"    4. {macro_csv_path}")
    print("=" * 115)


if __name__ == "__main__":
    main()