r"""export_r0_val_predictions.py  (MARIVISION-HR-P2, Claude lead, 2026-10-25)

INFERENCE-ONLY export of the frozen R0 baseline (Deformable-DETR R50, 300 queries, 4 levels, 640x640) on the SeaClips
VALIDATION split, using R0's OWN code path, unchanged:
  get_authoritative_r0_args() -> models.deformable_detr.build(args) -> checkpoint["model"] (strict load)
  -> SeaClipsDetection(VAL, is_train=False) -> postprocessors["bbox"](outputs, orig_sizes)
  -> xyxy -> xywh exactly as train_seaclips_r0_baseline.evaluate_seaclips_val() does.

PROVENANCE PROOF: after export it recomputes pycocotools mAP@0.30 on the full VAL split exactly as R0's training script
did, and compares it with the `metrics.map_03` stored inside the checkpoint. A match proves the exported boxes are the
predictions that produced R0's recorded number (same weights, same preprocessing, same post-processing).

Writes ONLY into --out-dir (new folder):
  r0_val_predictions_<tag>.json                 COCO results, native-pixel xywh, top-100 per frame, no score cut
  r0_val_predictions_<tag>.json.manifest.json   MARIVISION_FROZEN_R50_VAL_PREDS_MANIFEST_V1 (+ reproduction check)
Never trains, never writes into the checkpoint folder, never reads the TEST split.

Run from D:\MARIVISION\R0_frozen_baseline (so `main`, `models`, `datasets`, `util`, `r0_preflight` import):
  python export_r0_val_predictions.py --checkpoint checkpoint_best_r0_epoch04.pth --tag epoch04 ^
      --out-dir D:\MARIVISION\R0_frozen_baseline\val_exports
"""
from PIL import Image  # noqa: F401  (R0 convention: must precede torchvision imports)
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

MANIFEST_SCHEMA = "MARIVISION_FROZEN_R50_VAL_PREDS_MANIFEST_V1"
BOX_FORMAT = "xywh_abs_pixels_native_frame"
DEFAULT_ZIP = Path("D:/MARIVISION/SeaClips.zip")
DEFAULT_VAL_JSON = Path("D:/MARIVISION/seaclips-val.json")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checkpoint_trusted(p: Path):
    # R0 checkpoints embed an argparse.Namespace ("args"), which torch>=2.6 weights_only loading rejects.
    # The file is the Project Director's own local checkpoint, so full unpickling is acceptable here.
    try:
        return torch.load(p, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 1.13 has no weights_only argument
        return torch.load(p, map_location="cpu")


def coco_map_at_030(coco_gt, detections):
    """Replicates train_seaclips_r0_baseline.evaluate_seaclips_val() headline metrics (pycocotools, IoU=0.30)."""
    from pycocotools.cocoeval import COCOeval
    coco_dt = coco_gt.loadRes(copy.deepcopy(detections))
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.params.iouThrs = np.array([0.30])
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    gt_ca = copy.deepcopy(coco_gt)
    for ann in gt_ca.dataset["annotations"]:
        ann["category_id"] = 1
    gt_ca.dataset["categories"] = [{"id": 1, "name": "foreground"}]
    gt_ca.createIndex()
    dt_ca = copy.deepcopy(detections)
    for d in dt_ca:
        d["category_id"] = 1
    ev_ca = COCOeval(gt_ca, gt_ca.loadRes(dt_ca), "bbox")
    ev_ca.params.iouThrs = np.array([0.30])
    ev_ca.evaluate()
    ev_ca.accumulate()
    ev_ca.summarize()
    return {
        "map_03": float(ev.stats[0]), "ap_s_03": float(ev.stats[3]), "ap_m_03": float(ev.stats[4]),
        "ap_l_03": float(ev.stats[5]), "ar_03": float(ev.stats[8]), "map_ca_03": float(ev_ca.stats[0]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="R0 checkpoint (.pth) with keys 'model', 'epoch', 'metrics'")
    ap.add_argument("--tag", required=True, help="Short label for output names, e.g. epoch04 or epoch15")
    ap.add_argument("--out-dir", required=True, help="NEW output folder (must not be the checkpoint folder)")
    ap.add_argument("--zip", default=str(DEFAULT_ZIP))
    ap.add_argument("--coco-val", default=str(DEFAULT_VAL_JSON))
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--reproduction-tolerance", type=float, default=2e-3,
                    help="Max |reproduced map_03 - stored map_03| to declare the export faithful (GPU non-determinism)")
    ap.add_argument("--max-images", type=int, default=0, help="Debug only: stop after N images (disables the parity check)")
    args_cli = ap.parse_args()

    ckpt_path = Path(args_cli.checkpoint).resolve()
    val_json = Path(args_cli.coco_val).resolve()
    out_dir = Path(args_cli.out_dir).resolve()
    if "test" in val_json.name.lower() or "test" in [p.lower() for p in val_json.parts[-2:-1]]:
        raise SystemExit("[SPLIT BLOCK] TEST split is locked; refusing to run.")
    if "val" not in val_json.name.lower():
        raise SystemExit(f"[SPLIT BLOCK] --coco-val must be the VALIDATION json, got {val_json.name}")
    if out_dir == ckpt_path.parent:
        raise SystemExit("[REFUSED] --out-dir must be a new folder, not the checkpoint folder.")
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / f"r0_val_predictions_{args_cli.tag}.json"
    manifest_path = Path(str(preds_path) + ".manifest.json")
    if preds_path.exists() or manifest_path.exists():
        raise SystemExit(f"[REFUSED] {preds_path.name} (or its manifest) already exists; choose a new --tag or --out-dir.")

    # R0's own code (must be run from the R0 folder; the current folder is put on the import path explicitly)
    sys.path.insert(0, str(Path.cwd()))
    from models.deformable_detr import build
    from util.misc import collate_fn
    from datasets.seaclips_detection import SeaClipsDetection
    from r0_preflight import get_authoritative_r0_args

    print(f"[R0 EXPORT] checkpoint={ckpt_path}")
    ckpt_sha = sha256_file(ckpt_path)
    print(f"[R0 EXPORT] checkpoint sha256={ckpt_sha}")
    ckpt = load_checkpoint_trusted(ckpt_path)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise SystemExit("[REFUSED] checkpoint has no 'model' state dict")
    stored_metrics = dict(ckpt.get("metrics", {}) or {})
    stored_epoch = ckpt.get("epoch")
    print(f"[R0 EXPORT] stored epoch={stored_epoch} stored metrics={stored_metrics}")

    r0_args = get_authoritative_r0_args()
    device = torch.device(r0_args.device)
    model, _criterion, postprocessors = build(r0_args)
    load_msg = model.load_state_dict(ckpt["model"], strict=True)
    print(f"[R0 EXPORT] strict state_dict load OK ({load_msg})")
    model.to(device)
    model.eval()

    dataset_val = SeaClipsDetection(Path(args_cli.zip), val_json, is_train=False, random_flip=False)
    n_val = len(dataset_val)
    label_to_cat = dict(dataset_val.label_to_cat_id)
    if not set(int(v) for v in label_to_cat.values()) <= set(range(1, 8)):
        raise SystemExit(f"[REFUSED] label_to_cat_id values outside 1..7: {label_to_cat}")
    loader = DataLoader(dataset_val, batch_size=args_cli.batch_size, shuffle=False, collate_fn=collate_fn,
                        num_workers=args_cli.num_workers, pin_memory=(device.type == "cuda"))
    print(f"[R0 EXPORT] VAL frames={n_val} device={device} batch={args_cli.batch_size}")

    detections = []
    seen_images = set()
    t0 = time.time()
    with torch.inference_mode():
        for b_idx, (samples, targets) in enumerate(loader):
            samples = samples.to(device)
            outputs = model(samples)
            orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)
            results = postprocessors["bbox"](outputs, orig_sizes)
            for target, result in zip(targets, results):
                img_id = int(target["image_id"].item())
                seen_images.add(img_id)
                for s, lbl, box in zip(result["scores"].cpu(), result["labels"].cpu(), result["boxes"].cpu()):
                    x1, y1, x2, y2 = box.tolist()
                    detections.append({
                        "image_id": img_id,
                        "category_id": int(label_to_cat[int(lbl.item())]),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(s.item()),
                    })
            if (b_idx + 1) % 200 == 0:
                print(f"  {len(seen_images)}/{n_val} frames  ({time.time() - t0:.0f}s)")
            if args_cli.max_images and len(seen_images) >= args_cli.max_images:
                break
    elapsed = time.time() - t0
    per_img = {}
    for d in detections:
        per_img[d["image_id"]] = per_img.get(d["image_id"], 0) + 1
    print(f"[R0 EXPORT] frames={len(seen_images)} detections={len(detections)} "
          f"max/frame={max(per_img.values()) if per_img else 0} elapsed={elapsed:.0f}s")

    complete = len(seen_images) == n_val and not args_cli.max_images
    reproduction = {"status": "NOT_RUN_PARTIAL_EXPORT"}
    if complete:
        reproduced = coco_map_at_030(dataset_val.coco, detections)
        stored = stored_metrics.get("map_03")
        diff = None if stored is None else abs(float(reproduced["map_03"]) - float(stored))
        ok = diff is not None and diff <= args_cli.reproduction_tolerance
        reproduction = {
            "status": "REPRODUCED" if ok else ("NO_STORED_METRIC" if stored is None else "MISMATCH"),
            "reproduced": reproduced,
            "stored_in_checkpoint": stored_metrics,
            "abs_diff_map_03": diff,
            "tolerance": args_cli.reproduction_tolerance,
        }
        print(f"[R0 EXPORT] reproduction check: reproduced map_03={reproduced['map_03']:.6f} "
              f"stored={stored} diff={diff} -> {reproduction['status']}")

    preds_path.write_text(json.dumps(detections), encoding="utf-8")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "split": "val",
        "model_description": (
            "R0 frozen baseline: Deformable-DETR ResNet-50, 300 queries, 4 feature levels, no box refine, no two-stage, "
            "640x640 non-aspect resize, COCO-pretrained, 15-epoch SeaClips fine-tune; checkpoint selected by VAL mAP@0.30"
        ),
        "source_checkpoint_path": str(ckpt_path),
        "source_checkpoint_sha256": ckpt_sha,
        "source_checkpoint_epoch": stored_epoch,
        "export_command": " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
        "export_script_sha256": sha256_file(Path(__file__).resolve()),
        "box_format": BOX_FORMAT,
        "category_id_base": 1,
        "label_to_cat_id": {str(k): int(v) for k, v in label_to_cat.items()},
        "score_threshold_applied": 0.0,
        "max_dets_per_image": int(max(per_img.values()) if per_img else 0),
        "n_val_frames_exported": len(seen_images),
        "n_val_frames_in_split": n_val,
        "coco_val_sha256": sha256_file(val_json),
        "predictions_file_sha256": sha256_file(preds_path),
        "reproduction_check": reproduction,
        "selection_caveat": "Checkpoint was chosen by mAP@0.30 on the full VAL split; VAL results for it are optimistically biased.",
        "torch_version": torch.__version__,
        "device": str(device),
    }
    if reproduction["status"] != "REPRODUCED":
        # Never place an unverified manifest where the runner's loader looks for it (<preds>.manifest.json).
        manifest_path = Path(str(preds_path) + ".UNVERIFIED_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[R0 EXPORT] wrote {preds_path}")
    print(f"[R0 EXPORT] wrote {manifest_path}")
    print(f"[R0 EXPORT] predictions sha256={manifest['predictions_file_sha256']}")
    if reproduction["status"] != "REPRODUCED":
        print("[R0 EXPORT] WARNING: reproduction check did not pass (or was not run); manifest saved as UNVERIFIED and "
              "the runner will refuse this file until the discrepancy is explained.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
