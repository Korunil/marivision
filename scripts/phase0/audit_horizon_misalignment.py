"""
D:/MARIVISION/H0_workbench/eval_tools/audit_horizon_misalignment.py
Final Certified Deterministic Horizon Misalignment Falsification Test for Frozen M2 (HRPC)
- Enforces strict chronological ordering and frame-gap-aware segmentation (g_t == 1.0 and f_t == f_{t-1} + 1)
- Circularly shifts horizon coordinates strictly within each contiguous active segment: k_s = floor(N_s / 2)
- Asserts parity within 0.01 pp of cached correct M2 predictions against M2_SIDECAR
- Reports segment manifest (first/last frame, N_s, k_s) and pre-inference mismatch D_t
- Evaluates 3-way matrix: Correct HRPC (A) vs Misaligned HRPC (B) vs De-gated HRPC (C)
"""

import os
import sys
import copy
import json
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
M2_CKPT = Path("D:/MARIVISION/checkpoints_m2_hrpc/checkpoint_best.pth")
M2_SIDECAR = Path("D:/MARIVISION/checkpoints_m2_hrpc/checkpoint_best.strata_metrics.json")
CACHE_M2 = Path("D:/MARIVISION/H0_workbench/eval_tools/cache_dt_m2_g1.json")

EXPECTED_ACTIVE_VIDEOS = ["video_03", "video_04", "video_05", "video_06", "video_08", "video_09"]


def get_video_id_from_path(file_name):
    clean = str(file_name).replace("\\", "/")
    for part in clean.split("/"):
        if part.startswith("video_"):
            return part
    return "unknown"


def get_frame_num_from_path(file_name):
    return int(Path(file_name).stem)


def run_coco_eval_subset(coco_gt, detections, cat_ids):
    if not detections:
        return {"map_03": 0.0, "map_ca_03": 0.0, "ar_03": 0.0, "per_class": {}}

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

    return {
        "map_03": float(eval_03.stats[0] * 100.0),
        "map_ca_03": float(eval_ca.stats[0] * 100.0),
        "ar_03": float(eval_03.stats[8] * 100.0),
        "per_class": per_class
    }


def extract_predictions_custom_horizon(ckpt_path, loader, dataset, device, g1_img_ids, custom_horizon_map=None, force_gate_zero=False):
    args = get_authoritative_r0_args()
    args.conditioning_mode = "hrpc"
    model, _, postprocessors = build(args)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    detections = []
    with torch.inference_mode():
        for samples, targets in loader:
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            batch_img_ids = [t["image_id"].item() for t in targets]
            if not any(i in g1_img_ids for i in batch_img_ids):
                continue

            horizons = []
            for t in targets:
                img_id = t["image_id"].item()
                if force_gate_zero:
                    h = t["horizon"].clone()
                    h[2] = 0.0
                elif custom_horizon_map is not None and img_id in custom_horizon_map:
                    h = torch.tensor(custom_horizon_map[img_id], dtype=torch.float32, device=device)
                else:
                    h = t["horizon"]
                horizons.append(h)
            horizons = torch.stack(horizons, dim=0).to(device)

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
    return detections


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_val = SeaClipsDetection(ZIP_PATH, VAL_JSON, is_train=False, random_flip=False, horizon_tracks_path=VAL_TRACKS)
    loader_val = DataLoader(dataset_val, batch_size=2, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)

    coco_full = dataset_val.coco
    cat_ids = sorted(coco_full.getCatIds())

    # Ingest ALL validation frames to map complete video streams
    video_to_all_frames = {}
    img_to_horizon = {}
    g1_img_ids = set()

    for samples, targets in loader_val:
        for t in targets:
            img_id = t["image_id"].item()
            h_tensor = t["horizon"].cpu().numpy()
            img_to_horizon[img_id] = h_tensor.copy()

            img_info = coco_full.loadImgs([img_id])[0]
            f_name = img_info["file_name"]
            vid = get_video_id_from_path(f_name)
            f_num = get_frame_num_from_path(f_name)

            if vid not in video_to_all_frames:
                video_to_all_frames[vid] = []
            video_to_all_frames[vid].append((f_num, img_id, float(h_tensor[2])))

            if h_tensor[2] == 1.0:
                g1_img_ids.add(img_id)

    assert len(g1_img_ids) == 2403, f"Expected 2403 active frames, found {len(g1_img_ids)}"

    # Parity check on cached correct M2 predictions against M2_SIDECAR (within 0.01 pp)
    print("=" * 115)
    print("TASK 0: MANDATORY M2 CORRECT-PREDICTION PARITY GATE AGAINST AUTHORITATIVE SIDECAR")
    print("=" * 115)
    assert CACHE_M2.exists(), "Cache cache_dt_m2_g1.json missing. Run audit_active_video_robustness.py first."
    assert M2_SIDECAR.exists(), f"Sidecar {M2_SIDECAR} missing."

    with open(CACHE_M2, "r", encoding="utf-8") as f:
        dt_m2_correct = json.load(f)["detections"]
    with open(M2_SIDECAR, "r", encoding="utf-8") as f:
        m2_sidecar_exp = json.load(f)["metrics"]["g1"]

    coco_g1 = copy.deepcopy(coco_full)
    coco_g1.dataset["images"] = [img for img in coco_g1.dataset["images"] if img["id"] in g1_img_ids]
    coco_g1.dataset["annotations"] = [ann for ann in coco_g1.dataset["annotations"] if ann["image_id"] in g1_img_ids]
    coco_g1.createIndex()

    res_correct_g1 = run_coco_eval_subset(coco_g1, dt_m2_correct, cat_ids)

    assert np.isclose(res_correct_g1["map_03"], m2_sidecar_exp["map_03"] * 100.0, atol=1e-2), "M2 mAP Parity Failed"
    assert np.isclose(res_correct_g1["map_ca_03"], m2_sidecar_exp["map_ca_03"] * 100.0, atol=1e-2), "M2 mAP_ca Parity Failed"
    ar_target = m2_sidecar_exp["ar"] if "ar" in m2_sidecar_exp else m2_sidecar_exp["ar_03"]
    assert np.isclose(res_correct_g1["ar_03"], ar_target * 100.0, atol=1e-2), "M2 AR Parity Failed"
    assert np.isclose(res_correct_g1["per_class"]["SHIP"]["ap"], m2_sidecar_exp["per_class"]["SHIP"]["ap"], atol=1e-2), "M2 SHIP AP Parity Failed"

    print(f"[+] PARITY CERTIFIED (Within 0.01 pp of Authoritative Sidecar):")
    print(f"    mAP@0.3:    {res_correct_g1['map_03']:.2f}% (Expected: {m2_sidecar_exp['map_03']*100:.2f}%)")
    print(f"    mAP_ca@0.3: {res_correct_g1['map_ca_03']:.2f}% (Expected: {m2_sidecar_exp['map_ca_03']*100:.2f}%)")
    print(f"    AR@0.3:     {res_correct_g1['ar_03']:.2f}% (Expected: {ar_target*100:.2f}%)")
    print(f"    SHIP AP:    {res_correct_g1['per_class']['SHIP']['ap']:.2f}% (Expected: {m2_sidecar_exp['per_class']['SHIP']['ap']:.2f}%)")
    print("=" * 115)

    # Contiguous segment discovery (splitting on g == 0.0 AND numeric frame index gaps)
    print("\n" + "=" * 115)
    print("TASK 1: FRAME-GAP-AWARE CONTIGUOUS SEGMENT DISCOVERY & MISALIGNMENT CONSTRUCTION")
    print("=" * 115)

    misaligned_horizon_map = {}
    all_shifted_img_ids = set()
    total_segments_discovered = 0
    segment_manifest = []

    for vid in EXPECTED_ACTIVE_VIDEOS:
        all_frames = video_to_all_frames[vid]
        all_frames_sorted = sorted(all_frames, key=lambda x: x[0])

        # Verify strict monotonic ordering of frame indices
        f_nums = [x[0] for x in all_frames_sorted]
        assert all(x < y for x, y in zip(f_nums, f_nums[1:])), f"Non-monotonic frame indices in {vid}"

        # Segment splitting: requires both g == 1.0 AND consecutive frame numbers (f_t == f_{t-1} + 1)
        segments = []
        current_seg = []
        for f_num, img_id, g_val in all_frames_sorted:
            if g_val == 1.0:
                if current_seg and f_num != current_seg[-1][0] + 1:
                    segments.append(current_seg)
                    current_seg = []
                current_seg.append((f_num, img_id))
            else:
                if len(current_seg) > 0:
                    segments.append(current_seg)
                    current_seg = []
        if len(current_seg) > 0:
            segments.append(current_seg)

        total_segments_discovered += len(segments)
        seg_lengths = [len(s) for s in segments]
        print(f"[*] {vid:<10} | Total Frames: {len(all_frames_sorted):<5} | Active Frames: {sum(seg_lengths):<5} | Discovered Segments: {len(segments)} {seg_lengths}")

        for s_idx, seg in enumerate(segments):
            N_s = len(seg)
            k_s = N_s // 2
            d_mismatches = []
            seg_img_ids = [item[1] for item in seg]

            for idx, img_id in enumerate(seg_img_ids):
                src_img_id = seg_img_ids[(idx + k_s) % N_s]
                src_h = img_to_horizon[src_img_id]
                curr_h = img_to_horizon[img_id]

                # Map shifted coordinates while keeping g == 1.0
                misaligned_horizon_map[img_id] = [src_h[0], src_h[1], 1.0]
                all_shifted_img_ids.add(img_id)

                curr_c = (curr_h[0] + curr_h[1]) / 2.0
                curr_s = curr_h[1] - curr_h[0]
                src_c = (src_h[0] + src_h[1]) / 2.0
                src_s = src_h[1] - src_h[0]

                d_dist = np.sqrt((curr_c - src_c)**2 + (curr_s - src_s)**2)
                d_mismatches.append(d_dist)

            segment_manifest.append({
                "video": vid,
                "seg_idx": s_idx + 1,
                "first_frame": seg[0][0],
                "last_frame": seg[-1][0],
                "N_s": N_s,
                "k_s": k_s,
                "mean_D": np.mean(d_mismatches),
                "median_D": np.median(d_mismatches),
                "max_D": np.max(d_mismatches)
            })

    # Assert complete active mapping
    assert all_shifted_img_ids == g1_img_ids, f"Shift mapping incomplete: {len(all_shifted_img_ids)} vs {len(g1_img_ids)}"
    assert len(misaligned_horizon_map) == 2403, "Misaligned map size error"

    print("\n--- Segment Manifest & Induced Geometric Displacement (Shift k_s = floor(N_s / 2)) ---")
    print(f"{'Video ID':<10} | {'Seg':<4} | {'Frame Range':<19} | {'N_s':<6} | {'Shift k_s':<10} | {'Mean D_t':<10} | {'Median D_t':<11} | {'Max D_t':<10}")
    print("-" * 95)
    for stat in segment_manifest:
        f_range = f"[{stat['first_frame']:06d} -> {stat['last_frame']:06d}]"
        print(f"{stat['video']:<10} | {stat['seg_idx']:<4} | {f_range:<19} | {stat['N_s']:<6} | {stat['k_s']:<10} | {stat['mean_D']:.4f}{'':<4} | {stat['median_D']:.4f}{'':<5} | {stat['max_D']:.4f}")
    print("=" * 115)

    # Run inference for Misaligned and De-gated conditions
    print("\n[*] Running inference for Misaligned HRPC (Circularly shifted within segments, g=1)...")
    dt_m2_misaligned = extract_predictions_custom_horizon(M2_CKPT, loader_val, dataset_val, device, g1_img_ids, custom_horizon_map=misaligned_horizon_map)

    print("[*] Running inference for De-gated HRPC (Residual disabled, g=0)...")
    dt_m2_degated = extract_predictions_custom_horizon(M2_CKPT, loader_val, dataset_val, device, g1_img_ids, force_gate_zero=True)

    # Evaluate on Full Active Stratum (g=1, 2,403 frames)
    res_misalign_g1 = run_coco_eval_subset(coco_g1, dt_m2_misaligned, cat_ids)
    res_degated_g1 = run_coco_eval_subset(coco_g1, dt_m2_degated, cat_ids)

    # Evaluate on Isolated Video_05 (481 frames)
    v05_img_ids = set(item[1] for item in video_to_all_frames["video_05"] if item[2] == 1.0)
    assert len(v05_img_ids) == 481

    coco_v05 = copy.deepcopy(coco_full)
    coco_v05.dataset["images"] = [img for img in coco_v05.dataset["images"] if img["id"] in v05_img_ids]
    coco_v05.dataset["annotations"] = [ann for ann in coco_v05.dataset["annotations"] if ann["image_id"] in v05_img_ids]
    coco_v05.createIndex()

    dt_correct_v05 = [d for d in dt_m2_correct if d["image_id"] in v05_img_ids]
    dt_misalign_v05 = [d for d in dt_m2_misaligned if d["image_id"] in v05_img_ids]
    dt_degated_v05 = [d for d in dt_m2_degated if d["image_id"] in v05_img_ids]

    res_correct_v05 = run_coco_eval_subset(coco_v05, dt_correct_v05, cat_ids)
    res_misalign_v05 = run_coco_eval_subset(coco_v05, dt_misalign_v05, cat_ids)
    res_degated_v05 = run_coco_eval_subset(coco_v05, dt_degated_v05, cat_ids)

    # --------------------------------------------------------------------------
    # Output 3-Way Falsification Matrices
    # --------------------------------------------------------------------------
    print("\n" + "=" * 125)
    print("TASK 2: THREE-WAY GEOMETRY FALSIFICATION MATRIX: FULL ACTIVE HORIZON (g=1, 2,403 Frames)")
    print("=" * 125)
    print(f"{'Metric':<22} | {'(A) Correct HRPC':<18} | {'(B) Misaligned (g=1)':<20} | {'(C) De-gated (g=0)':<18} | {'Delta (B - A)':<15} | {'Delta (C - A)':<15}")
    print("-" * 125)
    for m_key, m_name in [("map_03", "mAP@0.3"), ("map_ca_03", "Class-Agnostic AP"), ("ar_03", "Recall AR@100")]:
        va = res_correct_g1[m_key]
        vb = res_misalign_g1[m_key]
        vc = res_degated_g1[m_key]
        print(f"{m_name:<22} | {va:.2f}%{'':<12} | {vb:.2f}%{'':<14} | {vc:.2f}%{'':<12} | {vb - va:+.2f} pp{'':<7} | {vc - va:+.2f} pp")

    va_s = res_correct_g1["per_class"]["SHIP"]["ap"]
    vb_s = res_misalign_g1["per_class"]["SHIP"]["ap"]
    vc_s = res_degated_g1["per_class"]["SHIP"]["ap"]
    print(f"{'SHIP AP@0.3':<22} | {va_s:.2f}%{'':<12} | {vb_s:.2f}%{'':<14} | {vc_s:.2f}%{'':<12} | {vb_s - va_s:+.2f} pp{'':<7} | {vc_s - va_s:+.2f} pp")
    print("=" * 125)

    print("\n" + "=" * 125)
    print("TASK 3: THREE-WAY GEOMETRY FALSIFICATION MATRIX: ISOLATED VIDEO_05 (481 Frames)")
    print("=" * 125)
    print(f"{'Metric / Category':<22} | {'(A) Correct HRPC':<18} | {'(B) Misaligned (g=1)':<20} | {'(C) De-gated (g=0)':<18} | {'Delta (B - A)':<15} | {'Delta (C - A)':<15}")
    print("-" * 125)
    for m_key, m_name in [("map_03", "mAP@0.3"), ("map_ca_03", "Class-Agnostic AP"), ("ar_03", "Recall AR@100")]:
        va = res_correct_v05[m_key]
        vb = res_misalign_v05[m_key]
        vc = res_degated_v05[m_key]
        print(f"{m_name:<22} | {va:.2f}%{'':<12} | {vb:.2f}%{'':<14} | {vc:.2f}%{'':<12} | {vb - va:+.2f} pp{'':<7} | {vc - va:+.2f} pp")

    for cname in ["OBJECT", "SAILING_VESSEL", "SHIP"]:
        va = res_correct_v05["per_class"][cname]["ap"]
        vb = res_misalign_v05["per_class"][cname]["ap"]
        vc = res_degated_v05["per_class"][cname]["ap"]
        print(f"{cname + ' AP':<22} | {va:.2f}%{'':<12} | {vb:.2f}%{'':<14} | {vc:.2f}%{'':<12} | {vb - va:+.2f} pp{'':<7} | {vc - va:+.2f} pp")
    print("=" * 125)


if __name__ == "__main__":
    main()