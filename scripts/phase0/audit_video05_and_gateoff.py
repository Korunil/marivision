"""
D:/MARIVISION/H0_workbench/eval_tools/audit_video05_and_gateoff.py
Certified Diagnostic & De-Gating Sensitivity Evaluator
- Full active-stratum parity gate for M0, M1, and M2
- Post-hoc Common-4 LOO sensitivity analysis (BOAT, MARKER, OBJECT, SAIL)
- Video_05 triplet co-occurrence and frame-to-frame dynamics (c_t, s_t, m_t) audit
- Isolated Video_05 benchmark across M0, M1, and M2
- Test-time de-gating sensitivity (g <- 0) on frozen M2 (Full g=1 and Video_05)
"""

import os
import sys
import copy
import json
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

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
VAL_JSON = Path("D:/MARIVISION/seaclips-val.json")
VAL_TRACKS = Path("D:/MARIVISION/H0_workbench/horizon_tracks/seaclips_horizon_tracks_val_semantic_v2.json")
M0_CKPT = Path("D:/MARIVISION/R0_frozen_baseline/checkpoint_best_r0_epoch04.pth")
M1_CKPT = Path("D:/MARIVISION/checkpoints_m1_abs_y/checkpoint_best.pth")
M2_CKPT = Path("D:/MARIVISION/checkpoints_m2_hrpc/checkpoint_best.pth")

M0_SIDECAR = Path("D:/MARIVISION/R0_frozen_baseline/checkpoint_best_r0_epoch04.strata_metrics.json")
M1_SIDECAR = Path("D:/MARIVISION/checkpoints_m1_abs_y/checkpoint_best.strata_metrics.json")
M2_SIDECAR = Path("D:/MARIVISION/checkpoints_m2_hrpc/checkpoint_best.strata_metrics.json")

CACHE_M0 = Path("D:/MARIVISION/H0_workbench/eval_tools/cache_dt_m0_g1.json")
CACHE_M1 = Path("D:/MARIVISION/H0_workbench/eval_tools/cache_dt_m1_g1.json")
CACHE_M2 = Path("D:/MARIVISION/H0_workbench/eval_tools/cache_dt_m2_g1.json")

COMMON_4_IDS = [2, 3, 4, 6]  # BOAT, MARINE_MARKER, OBJECT, SAILING_VESSEL
EXPECTED_ACTIVE_VIDEOS = ["video_03", "video_04", "video_05", "video_06", "video_08", "video_09"]


def get_video_id_from_path(file_name):
    clean = str(file_name).replace("\\", "/")
    for part in clean.split("/"):
        if part.startswith("video_"):
            return part
    return "unknown"


def extract_predictions(ckpt_path, conditioning_mode, loader, dataset, device, g1_img_ids, force_gate_zero=False):
    args = get_authoritative_r0_args()
    args.conditioning_mode = conditioning_mode
    model, _, postprocessors = build(args)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    ckpt_epoch = ckpt.get("epoch", "N/A")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    detections = []
    mode_str = f"{conditioning_mode} (degated g=0)" if force_gate_zero else str(conditioning_mode)
    print(f"[*] Extracting predictions for: {ckpt_path.name} (Epoch {ckpt_epoch}, Mode: {mode_str})...")

    with torch.inference_mode():
        for samples, targets in loader:
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            batch_img_ids = [t["image_id"].item() for t in targets]
            if not any(i in g1_img_ids for i in batch_img_ids):
                continue

            if conditioning_mode is not None:
                horizons = torch.stack([t["horizon"] for t in targets], dim=0).to(device)
                if force_gate_zero:
                    horizons[:, 2] = 0.0
                outputs = model(samples, horizons=horizons)
            else:
                outputs = model(samples)

            orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)
            results = postprocessors["bbox"](outputs, orig_sizes)

            for target, result in zip(targets, results):
                img_id = target["image_id"].item()
                if img_id not in g1_img_ids:
                    continue

                scores = result["scores"].cpu()
                labels = result["labels"].cpu()
                boxes = result["boxes"].cpu()

                for s, l, box in zip(scores, labels, boxes):
                    x1, y1, x2, y2 = box.tolist()
                    cat_id = dataset.label_to_cat_id[l.item()]
                    detections.append({
                        "image_id": img_id,
                        "category_id": cat_id,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": s.item()
                    })

    hash_id = hashlib.sha256(str(sorted(list(g1_img_ids))).encode()).hexdigest()
    payload = {
        "_meta": {
            "checkpoint_path": str(ckpt_path.resolve()),
            "checkpoint_epoch": ckpt_epoch,
            "conditioning_mode": conditioning_mode,
            "force_gate_zero": force_gate_zero,
            "checkpoint_bytes": os.path.getsize(ckpt_path),
            "num_g1_images": len(g1_img_ids),
            "g1_image_ids_hash": hash_id,
            "val_json": str(VAL_JSON.resolve())
        },
        "detections": detections
    }
    return payload


def load_or_generate_cache(cache_path, ckpt_path, mode, loader, dataset, device, g1_img_ids):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    target_epoch = ckpt.get("epoch", "N/A")
    target_bytes = os.path.getsize(ckpt_path)
    hash_id = hashlib.sha256(str(sorted(list(g1_img_ids))).encode()).hexdigest()

    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            meta = data.get("_meta", {})
            if (meta.get("checkpoint_epoch") == target_epoch and
                meta.get("conditioning_mode") == mode and
                meta.get("force_gate_zero", False) is False and
                meta.get("checkpoint_bytes") == target_bytes and
                meta.get("num_g1_images") == len(g1_img_ids) and
                meta.get("g1_image_ids_hash") == hash_id):
                print(f"[+] Reusing certified prediction cache: {cache_path.name} (Epoch {target_epoch}, {mode})")
                return data["detections"]
            else:
                print(f"[!] Stale cache detected for {cache_path.name}. Regenerating...")
        except Exception as e:
            print(f"[!] Failed to read {cache_path.name} ({e}). Regenerating...")

    payload = extract_predictions(ckpt_path, mode, loader, dataset, device, g1_img_ids)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"[+] Saved certified cache to: {cache_path.name}")
    return payload["detections"]


def run_coco_eval_subset(coco_gt, detections, cat_ids):
    if not detections:
        return {"map_03": 0.0, "map_ca_03": 0.0, "map_4_03": None, "ar_03": 0.0, "per_class": {}}

    coco_dt = coco_gt.loadRes(detections)
    eval_03 = COCOeval(coco_gt, coco_dt, "bbox")
    eval_03.params.iouThrs = np.array([0.30])
    eval_03.evaluate()
    eval_03.accumulate()
    eval_03.summarize()

    # Class-agnostic evaluation
    coco_gt_ca = copy.deepcopy(coco_gt)
    for ann in coco_gt_ca.dataset["annotations"]:
        ann["category_id"] = 1
    coco_gt_ca.dataset["categories"] = [{"id": 1, "name": "foreground"}]
    coco_gt_ca.createIndex()

    dt_ca = copy.deepcopy(detections)
    for dt in dt_ca:
        dt["category_id"] = 1
    coco_dt_ca = coco_gt_ca.loadRes(dt_ca)
    eval_ca = COCOeval(coco_gt_ca, coco_dt_ca, "bbox")
    eval_ca.params.iouThrs = np.array([0.30])
    eval_ca.evaluate()
    eval_ca.accumulate()
    eval_ca.summarize()

    per_class = {}
    for cid in cat_ids:
        c_name = coco_gt.loadCats([cid])[0]["name"]
        cat_ann_count = len(coco_gt.getAnnIds(catIds=[cid]))
        if cat_ann_count == 0:
            per_class[c_name] = {"ap": None, "support": 0}
            continue
        cat_idx = eval_03.params.catIds.index(cid)
        p = eval_03.eval["precision"][0, :, cat_idx, 0, 2]
        valid_p = p[p > -1]
        c_ap = float(valid_p.mean() * 100.0) if valid_p.size > 0 else 0.0
        per_class[c_name] = {"ap": c_ap, "support": cat_ann_count}

    c4_aps = [
        per_class[coco_gt.loadCats([cid])[0]["name"]]["ap"]
        for cid in COMMON_4_IDS
        if coco_gt.loadCats([cid])[0]["name"] in per_class and per_class[coco_gt.loadCats([cid])[0]["name"]]["ap"] is not None
    ]
    map_4 = float(np.mean(c4_aps)) if len(c4_aps) == 4 else None

    return {
        "map_03": float(eval_03.stats[0] * 100.0),
        "map_ca_03": float(eval_ca.stats[0] * 100.0),
        "map_4_03": map_4,
        "ar_03": float(eval_03.stats[8] * 100.0),
        "per_class": per_class
    }


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_val = SeaClipsDetection(ZIP_PATH, VAL_JSON, is_train=False, random_flip=False, horizon_tracks_path=VAL_TRACKS)
    loader_val = DataLoader(dataset_val, batch_size=2, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)

    coco_full = dataset_val.coco
    cat_ids = sorted(coco_full.getCatIds())

    # Build active image set directly from loader and capture tensors
    g1_img_ids = set()
    img_to_video = {}
    img_to_horizon = {}
    for samples, targets in loader_val:
        for t in targets:
            img_id = t["image_id"].item()
            h_tensor = t["horizon"].cpu().numpy()
            assert h_tensor[2] in (0.0, 1.0), f"Invalid gate value: {h_tensor[2]}"
            if h_tensor[2] == 1.0:
                g1_img_ids.add(img_id)
                img_to_horizon[img_id] = h_tensor.copy()
            img_info = coco_full.loadImgs([img_id])[0]
            img_to_video[img_id] = get_video_id_from_path(img_info["file_name"])

    assert len(g1_img_ids) == 2403, f"Expected 2403 active frames, found {len(g1_img_ids)}"

    # Load and certify caches for M0, M1, and M2
    dt_m0 = load_or_generate_cache(CACHE_M0, M0_CKPT, None, loader_val, dataset_val, device, g1_img_ids)
    dt_m1 = load_or_generate_cache(CACHE_M1, M1_CKPT, "abs_y", loader_val, dataset_val, device, g1_img_ids)
    dt_m2 = load_or_generate_cache(CACHE_M2, M2_CKPT, "hrpc", loader_val, dataset_val, device, g1_img_ids)

    # Parity gate against authoritative sidecars on full active stratum
    coco_g1 = copy.deepcopy(coco_full)
    coco_g1.dataset["images"] = [img for img in coco_g1.dataset["images"] if img["id"] in g1_img_ids]
    coco_g1.dataset["annotations"] = [ann for ann in coco_g1.dataset["annotations"] if ann["image_id"] in g1_img_ids]
    coco_g1.createIndex()

    print("\n[*] Running mandatory parity check for M0, M1, and M2 against certified sidecar JSONs...")
    with open(M0_SIDECAR, "r", encoding="utf-8") as f:
        m0_exp = json.load(f)["metrics"]["g1"]
    with open(M1_SIDECAR, "r", encoding="utf-8") as f:
        m1_exp = json.load(f)["metrics"]["g1"]
    with open(M2_SIDECAR, "r", encoding="utf-8") as f:
        m2_exp = json.load(f)["metrics"]["g1"]

    m0_p = run_coco_eval_subset(coco_g1, dt_m0, cat_ids)
    m1_p = run_coco_eval_subset(coco_g1, dt_m1, cat_ids)
    m2_p = run_coco_eval_subset(coco_g1, dt_m2, cat_ids)

    assert np.isclose(m0_p["map_03"], m0_exp["map_03"] * 100.0, atol=1e-2), f"M0 Parity Failed: {m0_p['map_03']} vs {m0_exp['map_03']*100}"
    assert np.isclose(m0_p["per_class"]["SHIP"]["ap"], m0_exp["per_class"]["SHIP"]["ap"], atol=1e-2), "M0 SHIP Parity Failed"

    assert np.isclose(m1_p["map_03"], m1_exp["map_03"] * 100.0, atol=1e-2), f"M1 Parity Failed: {m1_p['map_03']} vs {m1_exp['map_03']*100}"
    assert np.isclose(m1_p["per_class"]["SHIP"]["ap"], m1_exp["per_class"]["SHIP"]["ap"], atol=1e-2), "M1 SHIP Parity Failed"

    assert np.isclose(m2_p["map_03"], m2_exp["map_03"] * 100.0, atol=1e-2), f"M2 Parity Failed: {m2_p['map_03']} vs {m2_exp['map_03']*100}"
    assert np.isclose(m2_p["per_class"]["SHIP"]["ap"], m2_exp["per_class"]["SHIP"]["ap"], atol=1e-2), "M2 SHIP Parity Failed"

    print("[+] PARITY CERTIFIED (Full g=1 Set, 2,403 Frames):")
    print(f"    M0 (Baseline): mAP={m0_p['map_03']:.2f}% | SHIP={m0_p['per_class']['SHIP']['ap']:.2f}%")
    print(f"    M1 (Abs-Y):    mAP={m1_p['map_03']:.2f}% | SHIP={m1_p['per_class']['SHIP']['ap']:.2f}%")
    print(f"    M2 (HRPC):     mAP={m2_p['map_03']:.2f}% | SHIP={m2_p['per_class']['SHIP']['ap']:.2f}%")

    # --------------------------------------------------------------------------
    # TASK 1: Fixed Common-4 LOO Sensitivity Table
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("TASK 1: FIXED COMMON-FOUR LOO SENSITIVITY TABLE (BOAT, MARKER, OBJECT, SAIL)")
    print("=" * 115)
    print(f"{'Omitted Video':<14} | {'Frames':<7} | {'M1 mAP_4':<10} | {'M2 mAP_4':<10} | {'Delta_mAP4':<12} | {'M1 mAP':<8} | {'M2 mAP':<8} | {'Delta_mAP':<10}")
    print("-" * 115)

    for vid in EXPECTED_ACTIVE_VIDEOS:
        sub_ids = set(i for i in g1_img_ids if img_to_video[i] != vid)
        sub_coco = copy.deepcopy(coco_full)
        sub_coco.dataset["images"] = [img for img in sub_coco.dataset["images"] if img["id"] in sub_ids]
        sub_coco.dataset["annotations"] = [ann for ann in sub_coco.dataset["annotations"] if ann["image_id"] in sub_ids]
        sub_coco.createIndex()

        s_dt_m1 = [d for d in dt_m1 if d["image_id"] in sub_ids]
        s_dt_m2 = [d for d in dt_m2 if d["image_id"] in sub_ids]

        r1 = run_coco_eval_subset(sub_coco, s_dt_m1, cat_ids)
        r2 = run_coco_eval_subset(sub_coco, s_dt_m2, cat_ids)

        d_m4 = r2["map_4_03"] - r1["map_4_03"]
        d_map = r2["map_03"] - r1["map_03"]
        print(f"{vid:<14} | {len(sub_ids):<7} | {r1['map_4_03']:.2f}%{'':<4} | {r2['map_4_03']:.2f}%{'':<4} | {d_m4:+.2f} pp{'':<5} | {r1['map_03']:.2f}%{'':<2} | {r2['map_03']:.2f}%{'':<2} | {d_map:+.2f} pp")
    print("=" * 115)

    # --------------------------------------------------------------------------
    # TASK 2: Video_05 Annotation Co-Occurrence & Horizon Dynamics Audit
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("TASK 2: VIDEO_05 SCENE CONFIGURATION & HORIZON TRAJECTORY DYNAMICS AUDIT")
    print("=" * 115)
    v05_img_ids = sorted([i for i in g1_img_ids if img_to_video[i] == "video_05"])
    assert len(v05_img_ids) == 481, f"Expected 481 frames in video_05, found {len(v05_img_ids)}"

    co_occur_triplets = 0
    c_vals = []
    s_vals = []

    for img_id in v05_img_ids:
        anns = coco_full.loadAnns(coco_full.getAnnIds(imgIds=[img_id]))
        c_set = set(a["category_id"] for a in anns)
        # Category IDs: 4=OBJECT, 6=SAILING_VESSEL, 7=SHIP
        if c_set == {4, 6, 7} and len(anns) == 3:
            co_occur_triplets += 1

        h = img_to_horizon[img_id]
        c_t = (h[0] + h[1]) / 2.0
        s_t = h[1] - h[0]
        c_vals.append(c_t)
        s_vals.append(s_t)

    c_vals = np.array(c_vals)
    s_vals = np.array(s_vals)
    delta_c = np.abs(np.diff(c_vals))
    delta_s = np.abs(np.diff(s_vals))
    m_t = np.sqrt(delta_c**2 + delta_s**2)

    print(f"[*] Triplet Co-occurrence: {co_occur_triplets} / 481 frames ({co_occur_triplets/481*100:.1f}%) contain exactly 1 OBJECT + 1 SAIL + 1 SHIP.")
    print(f"[*] Midpoint coordinate proxy c_t: mean={np.mean(c_vals):.4f}, std={np.std(c_vals):.4f}, range=[{np.min(c_vals):.4f}, {np.max(c_vals):.4f}] (span={np.ptp(c_vals):.4f})")
    print(f"[*] Slope proxy s_t:               mean={np.mean(s_vals):.4f}, std={np.std(s_vals):.4f}, range=[{np.min(s_vals):.4f}, {np.max(s_vals):.4f}] (span={np.ptp(s_vals):.4f})")
    print(f"[*] Frame dynamics index m_t:      mean={np.mean(m_t):.6f}, median={np.median(m_t):.6f}, max={np.max(m_t):.6f}")
    print("=" * 115)

    # --------------------------------------------------------------------------
    # TASK 3: Isolated Video_05 Performance Benchmark (M0, M1, M2)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("TASK 3: ISOLATED VIDEO_05 BENCHMARK (Derived from Parity-Certified Full Caches)")
    print("=" * 115)

    coco_v05 = copy.deepcopy(coco_full)
    coco_v05.dataset["images"] = [img for img in coco_v05.dataset["images"] if img["id"] in v05_img_ids]
    coco_v05.dataset["annotations"] = [ann for ann in coco_v05.dataset["annotations"] if ann["image_id"] in v05_img_ids]
    coco_v05.createIndex()

    dt_m0_v05 = [d for d in dt_m0 if d["image_id"] in v05_img_ids]
    dt_m1_v05 = [d for d in dt_m1 if d["image_id"] in v05_img_ids]
    dt_m2_v05 = [d for d in dt_m2 if d["image_id"] in v05_img_ids]

    res_m0_v05 = run_coco_eval_subset(coco_v05, dt_m0_v05, cat_ids)
    res_m1_v05 = run_coco_eval_subset(coco_v05, dt_m1_v05, cat_ids)
    res_m2_v05 = run_coco_eval_subset(coco_v05, dt_m2_v05, cat_ids)

    print(f"{'Metric / Category':<25} | {'M0 (R0 Baseline)':<18} | {'M1 (Abs-Y Control)':<18} | {'M2 (HRPC Proposed)':<18} | {'Delta_coord (M2-M1)':<18}")
    print("-" * 105)
    print(f"{'Mean AP@0.3 (v05)':<25} | {res_m0_v05['map_03']:.2f}%{'':<12} | {res_m1_v05['map_03']:.2f}%{'':<12} | {res_m2_v05['map_03']:.2f}%{'':<12} | {res_m2_v05['map_03'] - res_m1_v05['map_03']:+.2f} pp")
    print(f"{'Class-Agnostic AP (v05)':<25} | {res_m0_v05['map_ca_03']:.2f}%{'':<12} | {res_m1_v05['map_ca_03']:.2f}%{'':<12} | {res_m2_v05['map_ca_03']:.2f}%{'':<12} | {res_m2_v05['map_ca_03'] - res_m1_v05['map_ca_03']:+.2f} pp")
    print(f"{'Average Recall AR (v05)':<25} | {res_m0_v05['ar_03']:.2f}%{'':<12} | {res_m1_v05['ar_03']:.2f}%{'':<12} | {res_m2_v05['ar_03']:.2f}%{'':<12} | {res_m2_v05['ar_03'] - res_m1_v05['ar_03']:+.2f} pp")
    print("-" * 105)
    for c_id, c_name in [(4, "OBJECT"), (6, "SAILING_VESSEL"), (7, "SHIP")]:
        ap0 = res_m0_v05["per_class"][c_name]["ap"]
        ap1 = res_m1_v05["per_class"][c_name]["ap"]
        ap2 = res_m2_v05["per_class"][c_name]["ap"]
        delta = ap2 - ap1
        print(f"{c_name:<25} | {ap0:.2f}%{'':<12} | {ap1:.2f}%{'':<12} | {ap2:.2f}%{'':<12} | {delta:+.2f} pp")
    print("=" * 115)

    # --------------------------------------------------------------------------
    # TASK 4: Frozen M2 De-Gating Sensitivity (g <- 0) on Full g=1 and Video_05
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("TASK 4: FROZEN M2 TEST-TIME DE-GATING SENSITIVITY TEST (g <- 0)")
    print("=" * 115)
    dt_m2_degated_payload = extract_predictions(M2_CKPT, "hrpc", loader_val, dataset_val, device, g1_img_ids, force_gate_zero=True)
    dt_m2_degated_g1 = dt_m2_degated_payload["detections"]

    res_m2_active = m2_p
    res_m2_degated = run_coco_eval_subset(coco_g1, dt_m2_degated_g1, cat_ids)

    dt_m2_degated_v05 = [d for d in dt_m2_degated_g1 if d["image_id"] in v05_img_ids]
    res_m2_v05_active = res_m2_v05
    res_m2_v05_degated = run_coco_eval_subset(coco_v05, dt_m2_degated_v05, cat_ids)

    print("\n--- De-Gating on Entire Active Horizon (g=1, 2,403 Frames) ---")
    print(f"  mAP@0.3:           Active = {res_m2_active['map_03']:.2f}% | De-gated (g=0) = {res_m2_degated['map_03']:.2f}% | Delta = {res_m2_degated['map_03'] - res_m2_active['map_03']:+.2f} pp")
    print(f"  Class-Agnostic AP: Active = {res_m2_active['map_ca_03']:.2f}% | De-gated (g=0) = {res_m2_degated['map_ca_03']:.2f}% | Delta = {res_m2_degated['map_ca_03'] - res_m2_active['map_ca_03']:+.2f} pp")
    print(f"  AR@0.3:            Active = {res_m2_active['ar_03']:.2f}% | De-gated (g=0) = {res_m2_degated['ar_03']:.2f}% | Delta = {res_m2_degated['ar_03'] - res_m2_active['ar_03']:+.2f} pp")
    print(f"  SHIP AP:           Active = {res_m2_active['per_class']['SHIP']['ap']:.2f}% | De-gated (g=0) = {res_m2_degated['per_class']['SHIP']['ap']:.2f}% | Delta = {res_m2_degated['per_class']['SHIP']['ap'] - res_m2_active['per_class']['SHIP']['ap']:+.2f} pp")

    print("\n--- De-Gating on Isolated Video_05 (481 Frames) ---")
    print(f"  mAP@0.3:           Active = {res_m2_v05_active['map_03']:.2f}% | De-gated (g=0) = {res_m2_v05_degated['map_03']:.2f}% | Delta = {res_m2_v05_degated['map_03'] - res_m2_v05_active['map_03']:+.2f} pp")
    print(f"  Class-Agnostic AP: Active = {res_m2_v05_active['map_ca_03']:.2f}% | De-gated (g=0) = {res_m2_v05_degated['map_ca_03']:.2f}% | Delta = {res_m2_v05_degated['map_ca_03'] - res_m2_v05_active['map_ca_03']:+.2f} pp")
    print(f"  OBJECT AP:         Active = {res_m2_v05_active['per_class']['OBJECT']['ap']:.2f}% | De-gated (g=0) = {res_m2_v05_degated['per_class']['OBJECT']['ap']:.2f}% | Delta = {res_m2_v05_degated['per_class']['OBJECT']['ap'] - res_m2_v05_active['per_class']['OBJECT']['ap']:+.2f} pp")
    print(f"  SAIL AP:           Active = {res_m2_v05_active['per_class']['SAILING_VESSEL']['ap']:.2f}% | De-gated (g=0) = {res_m2_v05_degated['per_class']['SAILING_VESSEL']['ap']:.2f}% | Delta = {res_m2_v05_degated['per_class']['SAILING_VESSEL']['ap'] - res_m2_v05_active['per_class']['SAILING_VESSEL']['ap']:+.2f} pp")
    print(f"  SHIP AP:           Active = {res_m2_v05_active['per_class']['SHIP']['ap']:.2f}% | De-gated (g=0) = {res_m2_v05_degated['per_class']['SHIP']['ap']:.2f}% | Delta = {res_m2_v05_degated['per_class']['SHIP']['ap'] - res_m2_v05_active['per_class']['SHIP']['ap']:+.2f} pp")
    print("=" * 115)


if __name__ == "__main__":
    main()