r"""m1_r0_tiled_inference.py - M1: resolution-preserving tiled INFERENCE for the frozen R0 (Deformable-DETR R50).
MARIVISION-HR-P2, Claude lead, 2026-10-27. Inference only, VALIDATION split only. Never trains, never reads TEST.

Question it answers: does the HR-P2 tiling, bolted onto an UNCHANGED established detector at inference time only,
raise its VAL accuracy? No weights change and no constant is tuned.

Method (all constants are the HR-P2 pipeline's pre-existing defaults, declared before any result is seen):
  1. Evaluated subset = VAL frames at stride 10, offset 0 per video (432 frames), the HR-P2 evaluator's rule.
  2. Tiles per frame from marivision_hr_p2_pipeline.generate_coarse_to_fine_crops:
       Stage B = the full frame; Stage C = a 4x4 grid of overlapping 16:9 crops (640x360 on a 1920x1080 frame).
     Each tile goes through R0's own eval transform (resize to 640x640, to_tensor, ImageNet normalisation).
     Tiles are 16:9 like the full frame, so R0 sees the same aspect distortion it was trained with, objects 3x larger
     than in plain R0 (which shrinks 1920 px to 640 px).
  3. R0's own post-processor returns its top-100 boxes per tile. They are mapped to native pixels and fused with
     marivision_hr_p2_pipeline.merge_multiscale_tile_detections (border-truncation filter, Stage C large-box
     down-weighting, class-aware NWD/IoU fusion), score_thresh 0.0, max 100 per frame (R0's native protocol).
  4. Writes a COCO-results JSON + MARIVISION_FROZEN_R50_VAL_PREDS_MANIFEST_V1 manifest for the HR-P2 runner.

Provenance gate (export mode): the Stage B pass of THIS script must reproduce the previously exported plain R0
predictions of the same checkpoint (--reference-plain-json) on every evaluated frame: every detection with
score >= 0.05 must have a same-class partner within --parity-score-tol and --parity-box-tol px, in both directions.
If it fails, nothing is written.

Modes:
  --mode selftest   CPU, no model, no images: checks subset rule, tile geometry, coordinate round trip through the
                    pipeline merge, and the parity matcher. Needs only --pipeline-dir.
  --mode export     GPU inference. Run from D:\MARIVISION\R0_frozen_baseline with R0_MSDA_BACKEND=pytorch set.
"""
from PIL import Image  # noqa: F401  (R0 convention: PIL before torchvision)
import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_VERSION = "M1.0"
MANIFEST_SCHEMA = "MARIVISION_FROZEN_R50_VAL_PREDS_MANIFEST_V1"
BOX_FORMAT = "xywh_abs_pixels_native_frame"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TILE_STAGES = ("STAGE_B_FULL_CONTEXT", "STAGE_C_2X_CROP")
PARITY_STRICT_SCORE = 0.05


# ----------------------------------------------------------------------------------------------- pure helpers
def sha256_file(p) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def eval_subset_ids(images, stride: int, offset: int):
    """HR-P2 evaluator rule: per video, sort by (frame_id, id), keep index % stride == offset."""
    by_video = defaultdict(list)
    for im in images:
        by_video[int(im.get("video_id", 0))].append(im)
    out = []
    for _, lst in sorted(by_video.items()):
        lst = sorted(lst, key=lambda x: (int(x.get("frame_id", x["id"])), int(x["id"])))
        for i, im in enumerate(lst):
            if stride <= 1 or i % stride == offset:
                out.append(int(im["id"]))
    return out


def build_tiles(hr_p2, w: int, h: int):
    crops = [c for c in hr_p2.generate_coarse_to_fine_crops(w, h, horizon_y_norm=0.42, include_stage_d_microcrops=False)
             if c.stage in TILE_STAGES]
    n_b = sum(1 for c in crops if c.stage == TILE_STAGES[0])
    n_c = sum(1 for c in crops if c.stage == TILE_STAGES[1])
    if n_b != 1 or n_c != 16:
        raise AssertionError(f"[TILES] expected 1 Stage B + 16 Stage C crops, got {n_b} + {n_c}")
    for c in crops:
        if not (0 <= c.x0_int < c.x1_int <= w and 0 <= c.y0_int < c.y1_int <= h):
            raise AssertionError(f"[TILES] crop outside frame: {c}")
    return crops


def tile_dets_for_merge(cw, scores, labels, boxes_xyxy_tile, label_to_cat):
    """R0 boxes in TILE pixels (xyxy) -> dicts in the CropWindow's detector coordinates (xywh), which is what
    merge_multiscale_tile_detections back-projects (native = x0 + det / scale)."""
    sx, sy = float(cw.scale_x), float(cw.scale_y)
    out = []
    for s, lb, (x1, y1, x2, y2) in zip(scores, labels, boxes_xyxy_tile):
        out.append({"category_id": int(label_to_cat[int(lb)]), "score": float(s),
                    "bbox": [float(x1) * sx, float(y1) * sy, (float(x2) - float(x1)) * sx, (float(y2) - float(y1)) * sy]})
    return out


def plain_dets_native(scores, labels, boxes_xyxy, label_to_cat):
    """Exactly the plain exporter's conversion (export_r0_val_predictions.py): xyxy -> xywh, no clipping."""
    return [{"category_id": int(label_to_cat[int(lb)]), "score": float(s),
             "bbox": [float(x1), float(y1), float(x2) - float(x1), float(y2) - float(y1)]}
            for s, lb, (x1, y1, x2, y2) in zip(scores, labels, boxes_xyxy)]


def _one_way_match(src, dst, score_tol, strict_score):
    """For every src det with score >= strict_score find a same-class dst det within score_tol; return
    (n_unmatched, max box deviation in px over matched pairs)."""
    by_cls = defaultdict(list)
    for d in dst:
        by_cls[int(d["category_id"])].append(d)
    n_unmatched, max_dev = 0, 0.0
    for d in src:
        if float(d["score"]) < strict_score:
            continue
        best = None
        for t in by_cls.get(int(d["category_id"]), []):
            if abs(float(t["score"]) - float(d["score"])) > score_tol:
                continue
            dev = max(abs(float(p) - float(q)) for p, q in zip(d["bbox"], t["bbox"]))
            best = dev if best is None else min(best, dev)
        if best is None:
            n_unmatched += 1
        else:
            max_dev = max(max_dev, best)
    return n_unmatched, max_dev


def parity_check(mine, ref, score_tol, box_tol, strict_score=PARITY_STRICT_SCORE):
    u1, d1 = _one_way_match(mine, ref, score_tol, strict_score)
    u2, d2 = _one_way_match(ref, mine, score_tol, strict_score)
    ok = (u1 == 0 and u2 == 0 and max(d1, d2) <= box_tol)
    return {"ok": bool(ok), "unmatched_mine": u1, "unmatched_ref": u2, "max_box_dev_px": max(d1, d2),
            "n_strict_mine": sum(1 for d in mine if float(d["score"]) >= strict_score),
            "n_strict_ref": sum(1 for d in ref if float(d["score"]) >= strict_score)}


def import_pipeline(pipeline_dir):
    sys.path.insert(0, str(Path(pipeline_dir).resolve()))
    import marivision_hr_p2_pipeline as hr_p2
    return hr_p2


# ----------------------------------------------------------------------------------------------- selftest
def run_selftest(a) -> None:
    hr_p2 = import_pipeline(a.pipeline_dir)
    n_ok = 0

    def ok(msg):
        nonlocal n_ok
        n_ok += 1
        print(f"  [PASS] {msg}")

    # 1 subset rule
    imgs = [{"id": 1000 * v + i, "video_id": v, "frame_id": i} for v in (2, 1) for i in range(25)]
    ids = eval_subset_ids(imgs, 10, 0)
    assert ids == [1000, 1010, 1020, 2000, 2010, 2020], ids
    assert len(eval_subset_ids(imgs, 1, 0)) == 50
    ok("evaluated-subset rule (stride 10 -> 3 frames per 25-frame video, ordered by video; stride 1 -> all)")

    # 2 tile geometry on SeaClips frame size
    crops = build_tiles(hr_p2, 1920, 1080)
    c_sizes = {(c.x1_int - c.x0_int, c.y1_int - c.y0_int) for c in crops if c.stage == TILE_STAGES[1]}
    assert c_sizes == {(640, 360)}, c_sizes
    b = [c for c in crops if c.stage == TILE_STAGES[0]][0]
    assert (b.x0_int, b.y0_int, b.x1_int, b.y1_int) == (0, 0, 1920, 1080)
    cover = np.zeros((1080, 1920), bool)
    for c in crops:
        if c.stage == TILE_STAGES[1]:
            cover[c.y0_int:c.y1_int, c.x0_int:c.x1_int] = True
    assert cover.all()
    ok("tiles: 1 full frame + 16 integer 640x360 crops covering the whole 1920x1080 frame")

    # 3 coordinate round trip: an object seen in Stage B and in the Stage C crops containing it
    l2c = {i: i + 1 for i in range(7)}
    obj = (1000.0, 300.0, 1012.0, 306.0)  # native xyxy, a 12x6 px marker
    tile_preds = []
    for c in crops:
        if c.x0_int <= obj[0] and obj[2] <= c.x1_int and c.y0_int <= obj[1] and obj[3] <= c.y1_int:
            tb = (obj[0] - c.x0_int, obj[1] - c.y0_int, obj[2] - c.x0_int, obj[3] - c.y0_int)
            sc = 0.6 if c.stage == TILE_STAGES[0] else 0.8
            tile_preds.append((c, tile_dets_for_merge(c, [sc], [2], [tb], l2c)))
        else:
            tile_preds.append((c, []))
    merged = hr_p2.merge_multiscale_tile_detections(tile_preds, source_w=1920, source_h=1080, score_thresh=0.0,
                                                    max_det_per_image=100)
    assert len(merged) == 1, merged
    bx = merged[0]["bbox"]
    assert max(abs(bx[0] - 1000), abs(bx[1] - 300), abs(bx[2] - 12), abs(bx[3] - 6)) < 0.01, bx
    assert merged[0]["category_id"] == 3 and merged[0]["score"] > 0.8 - 1e-9
    ok(f"native box survives tile->merge round trip exactly, fused once from {merged[0]['n_fused_tiles']} tiles "
       f"(score {merged[0]['score']}, class id 3 = MARINE_MARKER)")

    # 4 a large box cut at an interior Stage C border is dropped; overlapping boxes of different classes stay separate
    c0 = [c for c in crops if c.stage == TILE_STAGES[1] and c.x0_int == 0 and c.y0_int == 0][0]
    tw = c0.x1_int - c0.x0_int
    cut = (tw - 100, 50, tw, 150)  # 100x100 box touching the right (interior) border
    tp = [(c0, tile_dets_for_merge(c0, [0.9, 0.7, 0.5], [6, 2, 0], [cut, (10, 10, 20, 16), (12, 11, 22, 17)], l2c))]
    m2 = hr_p2.merge_multiscale_tile_detections(tp, source_w=1920, source_h=1080, score_thresh=0.0)
    assert sorted(d["category_id"] for d in m2) == [1, 3], m2
    ok("interior-border-truncated large Stage C box dropped; overlapping boxes of different classes not merged")

    # 5 parity matcher
    ref = [{"category_id": 3, "score": 0.9, "bbox": [1, 2, 3, 4]}, {"category_id": 7, "score": 0.5, "bbox": [10, 10, 50, 20]},
           {"category_id": 1, "score": 0.01, "bbox": [0, 0, 5, 5]}]
    mine = [{"category_id": 3, "score": 0.90001, "bbox": [1.1, 2, 3, 4]}, {"category_id": 7, "score": 0.5, "bbox": [10, 10, 50, 20]},
            {"category_id": 2, "score": 0.02, "bbox": [0, 0, 5, 5]}]
    r = parity_check(mine, ref, 1e-3, 0.5)
    assert r["ok"] and abs(r["max_box_dev_px"] - 0.1) < 1e-9, r
    assert not parity_check([dict(mine[0], category_id=4), mine[1]], ref, 1e-3, 0.5)["ok"]
    assert not parity_check([dict(mine[0], bbox=[3, 2, 3, 4]), mine[1]], ref, 1e-3, 0.5)["ok"]
    assert not parity_check([mine[1]], ref, 1e-3, 0.5)["ok"]
    assert not parity_check([dict(mine[0], score=0.95), mine[1]], ref, 1e-3, 0.5)["ok"]
    ok("parity matcher: tolerates <0.05-score churn and float noise; catches class swap, box shift, score shift, missing det")

    # 6 plain conversion identical to the plain exporter
    p = plain_dets_native([0.5], [6], [(10.0, 20.0, 30.0, 25.0)], l2c)
    assert p == [{"category_id": 7, "score": 0.5, "bbox": [10.0, 20.0, 20.0, 5.0]}]
    ok("plain xyxy->xywh conversion matches export_r0_val_predictions.py")

    print(f"[M1 SELFTEST] {n_ok}/6 passed  (script {SCRIPT_VERSION}, sha256 {sha256_file(Path(__file__).resolve())},\n"
          f"              pipeline sha256 {sha256_file(Path(hr_p2.__file__).resolve())})")


# ----------------------------------------------------------------------------------------------- export
def preprocess_r0(pil_img, torch, TF, size=640):
    """R0 eval transform, verified bit-exact against the dataset in r0_preflight.py."""
    t = TF.to_tensor(TF.resize(pil_img, [size, size]))
    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std = torch.tensor(IMAGENET_STD)[:, None, None]
    return (t - mean) / std


def run_export(a) -> None:
    import torch
    import torchvision.transforms.functional as TF

    if os.environ.get("R0_MSDA_BACKEND", "").lower() != "pytorch":
        raise SystemExit("[REFUSED] Run 'set R0_MSDA_BACKEND=pytorch' first (required for R0 code).")
    for name in ("checkpoint", "tag", "out_dir", "reference_plain_json"):
        if not getattr(a, name):
            raise SystemExit(f"[REFUSED] --{name.replace('_', '-')} is required in export mode.")
    val_json = Path(a.coco_val).resolve()
    if "test" in val_json.name.lower() or "val" not in val_json.name.lower():
        raise SystemExit("[SPLIT BLOCK] --coco-val must be the VALIDATION json; TEST is locked.")
    ckpt_path = Path(a.checkpoint).resolve()
    out_dir = Path(a.out_dir).resolve()
    if out_dir == ckpt_path.parent:
        raise SystemExit("[REFUSED] --out-dir must be its own folder, not the checkpoint folder.")
    preds_path = out_dir / f"r0_val_predictions_{a.tag}.json"
    if preds_path.exists() or Path(str(preds_path) + ".manifest.json").exists():
        raise SystemExit(f"[REFUSED] {preds_path} (or its manifest) already exists; choose a new --tag.")
    ref_path = Path(a.reference_plain_json).resolve()
    ref_manifest_path = Path(str(ref_path) + ".manifest.json")
    if not ref_manifest_path.is_file():
        raise SystemExit(f"[REFUSED] reference manifest not found: {ref_manifest_path}")
    ref_manifest = json.loads(ref_manifest_path.read_text(encoding="utf-8"))
    ckpt_sha = sha256_file(ckpt_path)
    if str(ref_manifest.get("source_checkpoint_sha256", "")).lower() != ckpt_sha:
        raise SystemExit("[REFUSED] --reference-plain-json was exported from a different checkpoint "
                         f"({ref_manifest.get('source_checkpoint_sha256')} vs {ckpt_sha}).")
    if str(ref_manifest.get("predictions_file_sha256", "")).lower() != sha256_file(ref_path):
        raise SystemExit("[REFUSED] reference predictions file does not match its manifest sha256.")
    if str(ref_manifest.get("coco_val_sha256", "")).lower() != sha256_file(val_json):
        raise SystemExit("[REFUSED] reference export was made against a different VAL json.")

    sys.path.insert(0, str(Path.cwd()))
    hr_p2 = import_pipeline(a.pipeline_dir)
    from models.deformable_detr import build
    from datasets.seaclips_detection import SeaClipsDetection
    from r0_preflight import get_authoritative_r0_args
    from util.misc import NestedTensor

    val_coco = json.loads(val_json.read_text(encoding="utf-8"))
    eval_ids = eval_subset_ids(val_coco["images"], a.val_frame_stride, a.val_frame_offset)
    debug = bool(a.max_images)
    if debug:
        eval_ids = eval_ids[: a.max_images]
    print(f"[M1 {SCRIPT_VERSION}] evaluated VAL frames: {len(eval_ids)} (stride {a.val_frame_stride}, offset {a.val_frame_offset})"
          + ("  [DEBUG: nothing will be written]" if debug else ""))

    ref_by_iid = defaultdict(list)
    for d in json.loads(ref_path.read_text(encoding="utf-8")):
        ref_by_iid[int(d["image_id"])].append(d)

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    r0_args = get_authoritative_r0_args()
    device = torch.device(r0_args.device)
    model, _crit, postprocessors = build(r0_args)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    dataset = SeaClipsDetection(Path(a.zip), val_json, is_train=False, random_flip=False)
    label_to_cat = {int(k): int(v) for k, v in dict(dataset.label_to_cat_id).items()}

    tiled, worst = [], {"max_box_dev_px": 0.0, "unmatched": 0, "frames_failed": 0}
    n_tiles, t0 = 0, time.time()
    with torch.inference_mode():
        for n_done, iid in enumerate(eval_ids, 1):
            pil = dataset._load_image(dataset.images[iid]).convert("RGB")
            W, H = pil.size
            crops = build_tiles(hr_p2, W, H)
            tensors = [preprocess_r0(pil.crop((c.x0_int, c.y0_int, c.x1_int, c.y1_int)), torch, TF) for c in crops]
            sizes = [[c.y1_int - c.y0_int, c.x1_int - c.x0_int] for c in crops]  # post-processor takes (h, w)
            results = []
            for i in range(0, len(tensors), a.batch_size):
                x = torch.stack(tensors[i:i + a.batch_size]).to(device)
                mask = torch.zeros(x.shape[0], x.shape[2], x.shape[3], dtype=torch.bool, device=device)
                out = model(NestedTensor(x, mask))
                res = postprocessors["bbox"](out, torch.tensor(sizes[i:i + a.batch_size], dtype=torch.float32, device=device))
                results.extend(res)
            n_tiles += len(crops)

            tile_preds = []
            for c, r in zip(crops, results):
                sc, lb, bx = r["scores"].cpu().tolist(), r["labels"].cpu().tolist(), r["boxes"].cpu().tolist()
                if c.stage == TILE_STAGES[0]:
                    pr = parity_check(plain_dets_native(sc, lb, bx, label_to_cat), ref_by_iid.get(iid, []),
                                      a.parity_score_tol, a.parity_box_tol)
                    worst["max_box_dev_px"] = max(worst["max_box_dev_px"], pr["max_box_dev_px"])
                    worst["unmatched"] += pr["unmatched_mine"] + pr["unmatched_ref"]
                    if not pr["ok"]:
                        worst["frames_failed"] += 1
                        print(f"  [PARITY MISS] image {iid}: {pr}")
                tile_preds.append((c, tile_dets_for_merge(c, sc, lb, bx, label_to_cat)))
            merged = hr_p2.merge_multiscale_tile_detections(tile_preds, source_w=W, source_h=H, score_thresh=0.0,
                                                            max_det_per_image=100)
            for d in merged:
                tiled.append({"image_id": int(iid), "category_id": int(d["category_id"]),
                              "bbox": [float(v) for v in d["bbox"]], "score": float(d["score"])})
            if n_done % 25 == 0 or n_done == len(eval_ids):
                el = time.time() - t0
                print(f"  {n_done}/{len(eval_ids)} frames, {el:.0f}s (~{el / n_done * (len(eval_ids) - n_done) / 60:.1f} min left)"
                      f" | parity: failed frames {worst['frames_failed']}, max box dev {worst['max_box_dev_px']:.3f}px")
            if worst["frames_failed"] > 0 and not debug:
                raise SystemExit("[M1] PROVENANCE FAIL: the full-frame pass does not reproduce the plain R0 export. "
                                 "Nothing written. Send this output to Claude.")

    parity_ok = worst["frames_failed"] == 0
    print(f"[M1] tiles={n_tiles} frames={len(eval_ids)} fused detections={len(tiled)} elapsed={time.time() - t0:.0f}s")
    print(f"[M1] PROVENANCE (full-frame pass vs plain export, dets >= {PARITY_STRICT_SCORE}): "
          f"{'PASS' if parity_ok else 'FAIL'} | max box dev {worst['max_box_dev_px']:.4f}px, unmatched {worst['unmatched']}")
    if debug:
        print("[M1] debug run finished: nothing written.")
        return
    if not parity_ok:
        raise SystemExit("[M1] Provenance check failed; nothing written.")

    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path.write_text(json.dumps(tiled), encoding="utf-8")
    per_img = defaultdict(int)
    for d in tiled:
        per_img[d["image_id"]] += 1
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "split": "val",
        "model_description": (
            "M1: frozen R0 (Deformable-DETR R50, weights unchanged) + resolution-preserving tiled INFERENCE: "
            "HR-P2 Stage B full frame + Stage C 4x4 16:9 crops (640x360), each through R0's 640x640 eval transform, "
            "fused with marivision_hr_p2_pipeline.merge_multiscale_tile_detections (defaults; score_thresh 0, max 100)"
        ),
        "m1_script_version": SCRIPT_VERSION,
        "source_checkpoint_path": str(ckpt_path),
        "source_checkpoint_sha256": ckpt_sha,
        "source_checkpoint_epoch": ckpt.get("epoch"),
        "export_command": " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
        "export_script_sha256": sha256_file(Path(__file__).resolve()),
        "pipeline_py_sha256": sha256_file(Path(hr_p2.__file__).resolve()),
        "box_format": BOX_FORMAT,
        "category_id_base": 1,
        "label_to_cat_id": {str(k): v for k, v in sorted(label_to_cat.items())},
        "score_threshold_applied": 0.0,
        "max_dets_per_image": 100,
        "max_dets_observed_per_image": int(max(per_img.values()) if per_img else 0),
        "n_val_frames_exported": len(eval_ids),
        "n_val_frames_with_detections": len(per_img),
        "val_frame_stride": a.val_frame_stride,
        "val_frame_offset": a.val_frame_offset,
        "coverage_note": "Only the evaluated VAL subset was run; other VAL frames have no predictions by design.",
        "coco_val_sha256": sha256_file(val_json),
        "predictions_file_sha256": sha256_file(preds_path),
        "reference_plain_json": str(ref_path),
        "reference_plain_json_sha256": sha256_file(ref_path),
        "provenance_full_frame_parity": {"status": "PASS", "strict_score_min": PARITY_STRICT_SCORE,
                                         "max_box_dev_px": worst["max_box_dev_px"], "unmatched": worst["unmatched"],
                                         "score_tol": a.parity_score_tol, "box_tol_px": a.parity_box_tol},
        "n_tiles_run": n_tiles,
        "torch_version": torch.__version__,
        "device": str(device),
    }
    Path(str(preds_path) + ".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[M1] wrote {preds_path}\n     sha256 {manifest['predictions_file_sha256']} (+ .manifest.json)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("selftest", "export"), required=True)
    ap.add_argument("--pipeline-dir", default="D:/MARIVISION/H0_workbench/eval_tools",
                    help="Folder holding marivision_hr_p2_pipeline.py")
    ap.add_argument("--checkpoint")
    ap.add_argument("--tag")
    ap.add_argument("--out-dir")
    ap.add_argument("--reference-plain-json", help="Plain R0 VAL export of the same checkpoint (with .manifest.json)")
    ap.add_argument("--zip", default="D:/MARIVISION/SeaClips.zip")
    ap.add_argument("--coco-val", default="D:/MARIVISION/seaclips-val.json")
    ap.add_argument("--val-frame-stride", type=int, default=10)
    ap.add_argument("--val-frame-offset", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4, help="Tiles per forward pass; lower to 2 if CUDA runs out of memory")
    ap.add_argument("--parity-box-tol", type=float, default=0.5, help="px, native frame")
    ap.add_argument("--parity-score-tol", type=float, default=2e-3)
    ap.add_argument("--max-images", type=int, default=0, help="Debug: first N frames only; writes nothing")
    a = ap.parse_args()
    if a.mode == "selftest":
        run_selftest(a)
    else:
        run_export(a)


if __name__ == "__main__":
    main()
