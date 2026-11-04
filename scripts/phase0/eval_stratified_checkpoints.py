"""
D:/MARIVISION/H0_workbench/eval_tools/eval_stratified_checkpoints.py
Authoritative Offline Stratified Evaluator for SeaClips Experiments (M0, M1, M2)
- Conditioning-mode provenance verification against saved checkpoint args
- In-loop tensor derivation of g=1 and g=0 subsets (zero path-matching dependencies)
- Computes pre-specified 5-class common-support mAP_5@0.3 on native [0, 1] scale
- Strict assertions across all 6 strata with enriched _meta payload export
"""

import os
import sys
import copy
import json
import argparse
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
DYNAMICS_BINS_JSON = Path("D:/MARIVISION/H0_workbench/horizon_tracks/val_horizon_dynamics_bins.json")


def canonical_val_key(path_value):
    s = str(path_value).replace("\\", "/").strip("/")
    parts = [p for p in s.split("/") if p]
    lower = [p.lower() for p in parts]
    if "val" in lower:
        idx = len(lower) - 1 - lower[::-1].index("val")
        return "/".join(parts[idx:])
    if len(parts) >= 2:
        return "val/" + "/".join(parts[-2:])
    return "val/" + s


def run_coco_eval(coco_gt, detections, cat_ids):
    if not detections:
        return {
            "map_03": 0.0,
            "map_ca_03": 0.0,
            "map_5_03": None,
            "ap_s": 0.0,
            "ap_m": 0.0,
            "ap_l": 0.0,
            "ar": 0.0,
            "per_class": {}
        }

    coco_dt = coco_gt.loadRes(detections)

    # 1. Headline & Scale Evaluation at IoU=0.30
    eval_03 = COCOeval(coco_gt, coco_dt, "bbox")
    eval_03.params.iouThrs = np.array([0.30])
    eval_03.evaluate()
    eval_03.accumulate()
    eval_03.summarize()

    # 2. Class-Agnostic Headline Evaluation: mAPca@0.3
    coco_gt_ca = copy.deepcopy(coco_gt)
    for ann in coco_gt_ca.dataset["annotations"]:
        ann["category_id"] = 1
    coco_gt_ca.dataset["categories"] = [{"id": 1, "name": "foreground"}]
    coco_gt_ca.createIndex()

    dt_ca_list = copy.deepcopy(detections)
    for dt in dt_ca_list:
        dt["category_id"] = 1
    coco_dt_ca = coco_gt_ca.loadRes(dt_ca_list)

    eval_ca = COCOeval(coco_gt_ca, coco_dt_ca, "bbox")
    eval_ca.params.iouThrs = np.array([0.30])
    eval_ca.evaluate()
    eval_ca.accumulate()
    eval_ca.summarize()

    # 3. Support-aware per-class AP (percentage scale: 0.0 - 100.0)
    per_class = {}
    for cid in cat_ids:
        c_name = coco_gt.loadCats([cid])[0]["name"]
        cat_ann_count = len(coco_gt.getAnnIds(catIds=[cid]))
        if cat_ann_count == 0:
            per_class[c_name] = {"ap": None, "recall": None, "support": 0}
            continue

        cat_idx = eval_03.params.catIds.index(cid)
        p = eval_03.eval["precision"][0, :, cat_idx, 0, 2]
        valid_p = p[p > -1]
        c_ap = float(valid_p.mean() * 100.0) if valid_p.size > 0 else 0.0
        c_rec = float(eval_03.eval["recall"][0, cat_idx, 0, 2] * 100.0)
        per_class[c_name] = {"ap": c_ap, "recall": c_rec, "support": cat_ann_count}

    # 4. Pre-specified 5-class common-support metric (native [0, 1] scale)
    COMMON_5_IDS = [2, 3, 4, 6, 7]  # BOAT, MARINE_MARKER, OBJECT, SAILING_VESSEL, SHIP
    common_aps_pct = []
    has_full_common_support = True

    for cid in COMMON_5_IDS:
        c_name = coco_gt.loadCats([cid])[0]["name"]
        rec = per_class.get(c_name)
        if rec is None or rec.get("support", 0) <= 0 or rec.get("ap") is None:
            has_full_common_support = False
            break
        common_aps_pct.append(float(rec["ap"]))

    map_5_03 = float(np.mean(common_aps_pct) / 100.0) if (has_full_common_support and len(common_aps_pct) == 5) else None

    return {
        "map_03": float(eval_03.stats[0]),
        "map_ca_03": float(eval_ca.stats[0]),
        "map_5_03": map_5_03,
        "ap_s": float(eval_03.stats[3]),
        "ap_m": float(eval_03.stats[4]),
        "ap_l": float(eval_03.stats[5]),
        "ar": float(eval_03.stats[8]),
        "per_class": per_class
    }


def evaluate_checkpoint_strata(ckpt_path, conditioning_mode=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = get_authoritative_r0_args()
    args.conditioning_mode = conditioning_mode

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    ckpt_epoch = ckpt.get("epoch", "N/A")

    # Provenance verification of checkpoint conditioning mode
    saved_args = ckpt.get("args", None)
    saved_mode = None
    if saved_args is not None:
        if isinstance(saved_args, dict):
            saved_mode = saved_args.get("conditioning_mode", None)
        else:
            saved_mode = getattr(saved_args, "conditioning_mode", None)

    if conditioning_mode in ("abs_y", "hrpc"):
        assert saved_mode == conditioning_mode, (
            f"Checkpoint conditioning mismatch: checkpoint={saved_mode!r}, requested={conditioning_mode!r}"
        )
    elif conditioning_mode is None and saved_args is not None:
        assert saved_mode in (None, "None"), (
            f"Expected unconditioned R0 checkpoint, found mode={saved_mode!r}"
        )

    print(f"[Checkpoint] File: {Path(ckpt_path).name}")
    print(f"[Checkpoint] Epoch: {ckpt_epoch}")
    print(f"[Checkpoint] Saved mode: {saved_mode}")
    print(f"[Evaluator]  Requested mode: {conditioning_mode}")

    dataset_val = SeaClipsDetection(ZIP_PATH, VAL_JSON, is_train=False, random_flip=False, horizon_tracks_path=VAL_TRACKS)
    loader_val = DataLoader(dataset_val, batch_size=2, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)

    model, criterion, postprocessors = build(args)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    assert DYNAMICS_BINS_JSON.exists(), f"Missing: {DYNAMICS_BINS_JSON}"
    with open(DYNAMICS_BINS_JSON, "r", encoding="utf-8") as f:
        bins_data = json.load(f)

    q33 = float(bins_data["tertiles"]["composite"][0])
    q67 = float(bins_data["tertiles"]["composite"][1])
    frame_records = bins_data["frames"]

    coco_full = dataset_val.coco
    cat_ids = sorted(coco_full.getCatIds())

    id_to_key = {}
    for img_id in coco_full.getImgIds():
        img_info = coco_full.loadImgs([img_id])[0]
        id_to_key[img_id] = canonical_val_key(img_info["file_name"])

    dyn_membership = {}
    for full_key, rec in frame_records.items():
        if rec.get("valid_delta", False):
            m_t = (rec["d_c"]**2 + rec["d_s"]**2)**0.5
            c_key = canonical_val_key(full_key)
            if m_t <= q33:
                dyn_membership[c_key] = "Low"
            elif m_t <= q67:
                dyn_membership[c_key] = "Medium"
            else:
                dyn_membership[c_key] = "High"

    all_detections = []
    g1_detections = []
    g0_detections = []
    dyn_detections = {"Low": [], "Medium": [], "High": []}

    all_img_ids = set()
    g1_img_ids = set()
    g0_img_ids = set()

    print(f"\n[*] Running inference on full validation set for: {Path(ckpt_path).name} (Mode: {conditioning_mode})...")
    with torch.inference_mode():
        for samples, targets in loader_val:
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            if conditioning_mode is not None:
                horizons = torch.stack([t["horizon"] for t in targets], dim=0).to(device)
                outputs = model(samples, horizons=horizons)
            else:
                outputs = model(samples)

            orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)
            results = postprocessors["bbox"](outputs, orig_sizes)

            for target, result in zip(targets, results):
                img_id = target["image_id"].item()
                g = target["horizon"][2].item()
                c_key = id_to_key[img_id]

                all_img_ids.add(img_id)
                if g == 1.0:
                    g1_img_ids.add(img_id)
                else:
                    g0_img_ids.add(img_id)

                scores = result["scores"].cpu()
                labels = result["labels"].cpu()
                boxes = result["boxes"].cpu()

                for s, l, box in zip(scores, labels, boxes):
                    x1, y1, x2, y2 = box.tolist()
                    cat_id = dataset_val.label_to_cat_id[l.item()]
                    det = {
                        "image_id": img_id,
                        "category_id": cat_id,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": s.item()
                    }
                    all_detections.append(det)
                    if g == 1.0:
                        g1_detections.append(det)
                    else:
                        g0_detections.append(det)

                    if c_key in dyn_membership:
                        s_name = dyn_membership[c_key]
                        dyn_detections[s_name].append(det)

    # Invariant assertions
    assert len(all_img_ids) == 4266, f"Expected 4266 images, found {len(all_img_ids)}"
    assert len(g1_img_ids) == 2403, f"Expected 2403 g=1 images, found {len(g1_img_ids)}"
    assert len(g0_img_ids) == 1863, f"Expected 1863 g=0 images, found {len(g0_img_ids)}"
    assert g1_img_ids.isdisjoint(g0_img_ids), "g=1 and g=0 subsets overlap"
    assert g1_img_ids | g0_img_ids == all_img_ids, "Subset union does not cover all images"

    dyn_img_ids = {
        s: set(img_id for img_id, key in id_to_key.items() if dyn_membership.get(key) == s)
        for s in ["Low", "Medium", "High"]
    }

    assert len(dyn_membership) == 2396, f"Expected 2396 dynamics records, found {len(dyn_membership)}"
    assert len(dyn_img_ids["Low"]) == 799, f"Expected 799 Low frames, found {len(dyn_img_ids['Low'])}"
    assert len(dyn_img_ids["Medium"]) == 799, f"Expected 799 Medium frames, found {len(dyn_img_ids['Medium'])}"
    assert len(dyn_img_ids["High"]) == 798, f"Expected 798 High frames, found {len(dyn_img_ids['High'])}"
    assert (dyn_img_ids["Low"] | dyn_img_ids["Medium"] | dyn_img_ids["High"]).issubset(g1_img_ids), "Dynamics images not subset of g=1"
    assert dyn_img_ids["Low"].isdisjoint(dyn_img_ids["Medium"]), "Low and Medium overlap"
    assert dyn_img_ids["Low"].isdisjoint(dyn_img_ids["High"]), "Low and High overlap"
    assert dyn_img_ids["Medium"].isdisjoint(dyn_img_ids["High"]), "Medium and High overlap"

    def make_sub_coco(img_id_set):
        sub_coco = copy.deepcopy(coco_full)
        sub_coco.dataset["images"] = [img for img in sub_coco.dataset["images"] if img["id"] in img_id_set]
        sub_coco.dataset["annotations"] = [ann for ann in sub_coco.dataset["annotations"] if ann["image_id"] in img_id_set]
        sub_coco.createIndex()
        return sub_coco

    results_payload = {}
    strata_configs = [
        ("full", coco_full, all_detections, "Full Validation Set (4,266 frames)"),
        ("g1", make_sub_coco(g1_img_ids), g1_detections, "Active-Horizon Stratum (g=1, 2,403 frames)"),
        ("g0", make_sub_coco(g0_img_ids), g0_detections, "Unavailable-Horizon Stratum (g=0, 1,863 frames)"),
        ("dynamics_low", make_sub_coco(dyn_img_ids["Low"]), dyn_detections["Low"], "Low Dynamics Stratum (799 frames)"),
        ("dynamics_medium", make_sub_coco(dyn_img_ids["Medium"]), dyn_detections["Medium"], "Medium Dynamics Stratum (799 frames)"),
        ("dynamics_high", make_sub_coco(dyn_img_ids["High"]), dyn_detections["High"], "High Dynamics Stratum (798 frames)"),
    ]

    for s_tag, sub_gt, sub_dt, desc in strata_configs:
        print("\n" + "=" * 90)
        print(f"EVALUATING STRATUM: {desc}")
        print("=" * 90)
        res = run_coco_eval(sub_gt, sub_dt, cat_ids)

        if "dynamics" in s_tag:
            assert res["map_5_03"] is not None, (
                f"Common-support mAP unavailable for dynamics stratum {s_tag}; "
                "this contradicts the frozen pre-M2 composition audit."
            )

        results_payload[s_tag] = res

        print(f"  mAP@0.3:    {res['map_03']*100:.2f}%")
        print(f"  mAPca@0.3:  {res['map_ca_03']*100:.2f}%")
        if res["map_5_03"] is not None:
            print(f"  mAP_5@0.3:  {res['map_5_03']*100:.2f}% (Common 5-class)")
        print(f"  AP_S@0.3:   {res['ap_s']*100:.2f}%")
        print(f"  AP_M@0.3:   {res['ap_m']*100:.2f}%")
        print(f"  AP_L@0.3:   {res['ap_l']*100:.2f}%")
        print(f"  AR@0.3:     {res['ar']*100:.2f}%")

    export_payload = {
        "_meta": {
            "checkpoint": str(Path(ckpt_path).resolve()),
            "checkpoint_epoch": ckpt_epoch,
            "conditioning_mode": conditioning_mode,
            "saved_mode": saved_mode,
            "dynamics_thresholds": {
                "q33": q33,
                "q67": q67
            },
            "expected_frame_counts": {
                "full": 4266,
                "g1": 2403,
                "g0": 1863,
                "dynamics_low": 799,
                "dynamics_medium": 799,
                "dynamics_high": 798
            }
        },
        "metrics": results_payload
    }

    return export_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pth")
    parser.add_argument("--mode", type=str, default="None", choices=["None", "abs_y", "hrpc"])
    cli_args = parser.parse_args()

    mode = None if cli_args.mode == "None" else cli_args.mode
    eval_output = evaluate_checkpoint_strata(cli_args.ckpt, mode)

    out_json = Path(cli_args.ckpt).with_suffix(".strata_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(eval_output, f, indent=2)
    print(f"\n[+] Stratified evaluation complete. Saved with _meta envelope to: {out_json.name}")