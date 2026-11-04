"""
D:/MARIVISION/H0_workbench/eval_tools/audit_active_video_robustness.py
Certified Prediction-Level Robustness and Composition Evaluator for Active-Horizon (g=1) Regime
- Enforces strict composition assertions on 2,403 active frames (9,811 boxes)
- Enforces cached-prediction parity against frozen sidecar evaluation metrics
- Executes 6-way Leave-One-Video-Out (LOO) sensitivity analysis with SHIP-support tracking
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
M1_CKPT = Path("D:/MARIVISION/checkpoints_m1_abs_y/checkpoint_best.pth")
M2_CKPT = Path("D:/MARIVISION/checkpoints_m2_hrpc/checkpoint_best.pth")
M1_SIDECAR = Path("D:/MARIVISION/checkpoints_m1_abs_y/checkpoint_best.strata_metrics.json")
M2_SIDECAR = Path("D:/MARIVISION/checkpoints_m2_hrpc/checkpoint_best.strata_metrics.json")

CACHE_M1 = Path("D:/MARIVISION/H0_workbench/eval_tools/cache_dt_m1_g1.json")
CACHE_M2 = Path("D:/MARIVISION/H0_workbench/eval_tools/cache_dt_m2_g1.json")

COMMON_5_IDS = [2, 3, 4, 6, 7]  # BOAT, MARINE_MARKER, OBJECT, SAILING_VESSEL, SHIP
EXPECTED_ACTIVE_VIDEOS = ["video_03", "video_04", "video_05", "video_06", "video_08", "video_09"]


def get_video_id_from_path(file_name):
    clean = str(file_name).replace("\\", "/")
    for part in clean.split("/"):
        if part.startswith("video_"):
            return part
    return "unknown"


def extract_predictions(ckpt_path, conditioning_mode, loader, dataset, device, g1_img_ids):
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
    print(f"[*] Running inference to cache detections for: {ckpt_path.name} (Epoch {ckpt_epoch}, Mode: {conditioning_mode})...")
    with torch.inference_mode():
        for samples, targets in loader:
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            horizons = torch.stack([t["horizon"] for t in targets], dim=0).to(device)
            outputs = model(samples, horizons=horizons)
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


def run_sub_eval(sub_coco, detections, cat_ids):
    if not detections:
        return {"map_03": 0.0, "map_5_03": 0.0, "ship_ap": 0.0, "per_class": {}}

    coco_dt = sub_coco.loadRes(detections)
    eval_03 = COCOeval(sub_coco, coco_dt, "bbox")
    eval_03.params.iouThrs = np.array([0.30])
    eval_03.evaluate()
    eval_03.accumulate()
    eval_03.summarize()

    per_class = {}
    for cid in cat_ids:
        c_name = sub_coco.loadCats([cid])[0]["name"]
        cat_ann_count = len(sub_coco.getAnnIds(catIds=[cid]))
        if cat_ann_count == 0:
            per_class[c_name] = {"ap": None, "support": 0}
            continue

        cat_idx = eval_03.params.catIds.index(cid)
        p = eval_03.eval["precision"][0, :, cat_idx, 0, 2]
        valid_p = p[p > -1]
        c_ap = float(valid_p.mean() * 100.0) if valid_p.size > 0 else 0.0
        per_class[c_name] = {"ap": c_ap, "support": cat_ann_count}

    common_aps = [
        per_class[sub_coco.loadCats([cid])[0]["name"]]["ap"]
        for cid in COMMON_5_IDS
        if sub_coco.loadCats([cid])[0]["name"] in per_class and per_class[sub_coco.loadCats([cid])[0]["name"]]["ap"] is not None
    ]
    map_5 = float(np.mean(common_aps)) if len(common_aps) == 5 else None
    ship_ap = per_class.get("SHIP", {}).get("ap", None)

    return {
        "map_03": float(eval_03.stats[0] * 100.0),
        "map_5_03": map_5,
        "ship_ap": ship_ap,
        "per_class": per_class
    }


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_val = SeaClipsDetection(ZIP_PATH, VAL_JSON, is_train=False, random_flip=False, horizon_tracks_path=VAL_TRACKS)
    loader_val = DataLoader(dataset_val, batch_size=2, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)

    coco_full = dataset_val.coco
    cat_ids = sorted(coco_full.getCatIds())

    # Direct target-level collection of active-horizon (g=1) frames
    g1_img_ids = set()
    img_to_video = {}
    for samples, targets in loader_val:
        for t in targets:
            img_id = t["image_id"].item()
            g = t["horizon"][2].item()
            assert g in (0.0, 1.0), f"Invalid gate value: {g}"
            if g == 1.0:
                g1_img_ids.add(img_id)
            img_info = coco_full.loadImgs([img_id])[0]
            img_to_video[img_id] = get_video_id_from_path(img_info["file_name"])

    # Hard active composition gate
    assert len(g1_img_ids) == 2403, f"Composition Error: expected 2,403 g=1 frames, found {len(g1_img_ids)}"
    active_anns = coco_full.loadAnns(coco_full.getAnnIds(imgIds=list(g1_img_ids)))
    assert len(active_anns) == 9811, f"Composition Error: expected 9,811 active annotations, found {len(active_anns)}"

    cat_counts = {cid: 0 for cid in cat_ids}
    scale_counts = {"s": 0, "m": 0, "l": 0}
    for a in active_anns:
        cat_counts[a["category_id"]] += 1
        area = a["area"]
        if area < 32**2:
            scale_counts["s"] += 1
        elif area < 96**2:
            scale_counts["m"] += 1
        else:
            scale_counts["l"] += 1

    expected_cats = {1: 253, 2: 1139, 3: 4991, 4: 1835, 5: 0, 6: 1112, 7: 481}
    assert cat_counts == expected_cats, f"Category mismatch: {cat_counts} vs {expected_cats}"
    expected_scales = {"s": 9158, "m": 322, "l": 331}
    assert scale_counts == expected_scales, f"Scale mismatch: {scale_counts} vs {expected_scales}"

    active_videos = sorted(list(set(img_to_video[i] for i in g1_img_ids)))
    assert active_videos == EXPECTED_ACTIVE_VIDEOS, f"Active videos mismatch: {active_videos} vs {EXPECTED_ACTIVE_VIDEOS}"

    # Print composition audit table
    print("=" * 105)
    print("ACTIVE-HORIZON VALIDATION COMPOSITION AUDIT (6 Active Videos, 2,403 Frames, 9,811 Annotations)")
    print("=" * 105)
    print(f"{'Video ID':<10} | {'Frames':<7} | {'ANIMAL':<7} | {'BOAT':<7} | {'MARKER':<7} | {'OBJECT':<7} | {'SAIL':<7} | {'SHIP':<7} | {'Small':<7} | {'Med':<7} | {'Large':<7}")
    print("-" * 105)
    for vid in active_videos:
        v_imgs = [i for i in g1_img_ids if img_to_video[i] == vid]
        v_anns = coco_full.loadAnns(coco_full.getAnnIds(imgIds=v_imgs))
        c_cnt = {cid: 0 for cid in cat_ids}
        s_cnt = {"s": 0, "m": 0, "l": 0}
        for a in v_anns:
            c_cnt[a["category_id"]] += 1
            area = a["area"]
            if area < 32**2:
                s_cnt["s"] += 1
            elif area < 96**2:
                s_cnt["m"] += 1
            else:
                s_cnt["l"] += 1
        print(f"{vid:<10} | {len(v_imgs):<7} | {c_cnt.get(1,0):<7} | {c_cnt.get(2,0):<7} | {c_cnt.get(3,0):<7} | {c_cnt.get(4,0):<7} | {c_cnt.get(6,0):<7} | {c_cnt.get(7,0):<7} | {s_cnt['s']:<7} | {s_cnt['m']:<7} | {s_cnt['l']:<7}")
    print("=" * 105)

    # Load and verify cached predictions
    dt_m1 = load_or_generate_cache(CACHE_M1, M1_CKPT, "abs_y", loader_val, dataset_val, device, g1_img_ids)
    dt_m2 = load_or_generate_cache(CACHE_M2, M2_CKPT, "hrpc", loader_val, dataset_val, device, g1_img_ids)

    # Build active sub-coco
    coco_g1 = copy.deepcopy(coco_full)
    coco_g1.dataset["images"] = [img for img in coco_g1.dataset["images"] if img["id"] in g1_img_ids]
    coco_g1.dataset["annotations"] = [ann for ann in coco_g1.dataset["annotations"] if ann["image_id"] in g1_img_ids]
    coco_g1.createIndex()

    print("\n[*] Running mandatory full active-stratum parity gate against certified sidecar JSONs...")
    with open(M1_SIDECAR, "r", encoding="utf-8") as f:
        m1_expected = json.load(f)["metrics"]["g1"]
    with open(M2_SIDECAR, "r", encoding="utf-8") as f:
        m2_expected = json.load(f)["metrics"]["g1"]

    m1_parity = run_sub_eval(coco_g1, dt_m1, cat_ids)
    m2_parity = run_sub_eval(coco_g1, dt_m2, cat_ids)

    assert np.isclose(m1_parity["map_03"], m1_expected["map_03"] * 100.0, atol=1e-2), f"M1 mAP Parity Failed: {m1_parity['map_03']} vs {m1_expected['map_03']*100}"
    assert np.isclose(m1_parity["map_5_03"], m1_expected["map_5_03"] * 100.0, atol=1e-2), f"M1 mAP5 Parity Failed: {m1_parity['map_5_03']} vs {m1_expected['map_5_03']*100}"
    assert np.isclose(m1_parity["ship_ap"], m1_expected["per_class"]["SHIP"]["ap"], atol=1e-2), f"M1 SHIP AP Parity Failed: {m1_parity['ship_ap']} vs {m1_expected['per_class']['SHIP']['ap']}"

    assert np.isclose(m2_parity["map_03"], m2_expected["map_03"] * 100.0, atol=1e-2), f"M2 mAP Parity Failed: {m2_parity['map_03']} vs {m2_expected['map_03']*100}"
    assert np.isclose(m2_parity["map_5_03"], m2_expected["map_5_03"] * 100.0, atol=1e-2), f"M2 mAP5 Parity Failed: {m2_parity['map_5_03']} vs {m2_expected['map_5_03']*100}"
    assert np.isclose(m2_parity["ship_ap"], m2_expected["per_class"]["SHIP"]["ap"], atol=1e-2), f"M2 SHIP AP Parity Failed: {m2_parity['ship_ap']} vs {m2_expected['per_class']['SHIP']['ap']}"

    print("[+] PARITY CERTIFIED: Cached predictions reproduce exact active-stratum metrics:")
    print(f"    M1 (Abs-Y): mAP={m1_parity['map_03']:.2f}% | mAP5={m1_parity['map_5_03']:.2f}% | SHIP={m1_parity['ship_ap']:.2f}%")
    print(f"    M2 (HRPC):  mAP={m2_parity['map_03']:.2f}% | mAP5={m2_parity['map_5_03']:.2f}% | SHIP={m2_parity['ship_ap']:.2f}%")

    # Six-Way Leave-One-Video-Out (LOO) Sensitivity Analysis
    print("\n" + "=" * 125)
    print("LEAVE-ONE-VIDEO-OUT (LOO) SENSITIVITY ANALYSIS ON ACTIVE HORIZON (g=1)")
    print("=" * 125)
    print(f"{'Omitted Video':<14} | {'Frames':<7} | {'N_GT':<6} | {'N_SHIP':<7} | {'Cats':<5} | {'M1 mAP':<8} | {'M2 mAP':<8} | {'Delta_mAP':<10} | {'Delta_mAP5':<11} | {'M1 SHIP':<8} | {'M2 SHIP':<8} | {'Delta_SHIP':<10}")
    print("-" * 125)

    for vid in active_videos:
        sub_img_ids = set(i for i in g1_img_ids if img_to_video[i] != vid)
        sub_coco = copy.deepcopy(coco_full)
        sub_coco.dataset["images"] = [img for img in sub_coco.dataset["images"] if img["id"] in sub_img_ids]
        sub_coco.dataset["annotations"] = [ann for ann in sub_coco.dataset["annotations"] if ann["image_id"] in sub_img_ids]
        sub_coco.createIndex()

        n_gt = len(sub_coco.dataset["annotations"])
        n_ship = len(sub_coco.getAnnIds(catIds=[7]))
        active_cids = [cid for cid in cat_ids if len(sub_coco.getAnnIds(catIds=[cid])) > 0]
        n_cats = len(active_cids)

        sub_dt_m1 = [d for d in dt_m1 if d["image_id"] in sub_img_ids]
        sub_dt_m2 = [d for d in dt_m2 if d["image_id"] in sub_img_ids]

        res_m1 = run_sub_eval(sub_coco, sub_dt_m1, cat_ids)
        res_m2 = run_sub_eval(sub_coco, sub_dt_m2, cat_ids)

        d_map = res_m2["map_03"] - res_m1["map_03"]
        d_m5 = (res_m2["map_5_03"] - res_m1["map_5_03"]) if (res_m2["map_5_03"] is not None and res_m1["map_5_03"] is not None) else None

        s1 = res_m1["ship_ap"]
        s2 = res_m2["ship_ap"]
        d_ship = (s2 - s1) if (s1 is not None and s2 is not None) else None

        d_m5_str = f"{d_m5:+.2f} pp" if d_m5 is not None else "N/A"
        s1_str = f"{s1:.2f}%" if s1 is not None else "N/A"
        s2_str = f"{s2:.2f}%" if s2 is not None else "N/A"
        d_ship_str = f"{d_ship:+.2f} pp" if d_ship is not None else "N/A"

        print(f"{vid:<14} | {len(sub_img_ids):<7} | {n_gt:<6} | {n_ship:<7} | {n_cats:<5} | {res_m1['map_03']:.2f}%{'':<2} | {res_m2['map_03']:.2f}%{'':<2} | {d_map:+.2f} pp{'':<3} | {d_m5_str:<11} | {s1_str:<8} | {s2_str:<8} | {d_ship_str:<10}")

    print("=" * 125)


if __name__ == "__main__":
    main()