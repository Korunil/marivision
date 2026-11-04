r"""m2_fp_gallery.py - READ-ONLY, CPU-only look at the false positives (MARIVISION-HR-P2, 2026-10-28). VAL only.

Question: what ARE arm T's confident "background" false positives? Two readings predict different fixes:
  (a) texture: waves, glare, foam, wakes  -> the detector is wrong; hard-negative / data fix
  (b) real but UNLABELLED objects (distant debris, buoys, birds) that only become visible at tile resolution
      -> annotation-resolution bias: the metric penalises seeing more than the annotators labelled
Numbers cannot separate (a) from (b); looking can. This writes contact sheets (PNG) of the highest-scoring
BACKGROUND false positives (score >= 0.05; m1_1_fp_anatomy's BACKGROUND bucket: not a duplicate, not a wrong class on a
GT, not inside a larger GT, and no same-class GT within IoU 0.10) for
OBJECT and MARINE_MARKER, for arm T tiled and, as reference, arm F plain. Each crop shows 4x the box (min 48 px) around
it, upscaled to 160 px: red = the prediction, green = every GT box in the crop. Caption: video, image id, score.

The images stay on your machine (m2_reports\). Nothing is sent anywhere.

Run from D:\MARIVISION\H0_workbench\eval_tools:
  python -W error m2_fp_gallery.py --mode selftest
  python -W error m2_fp_gallery.py --mode run --tag seed42
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CLASSES = ("ANIMAL", "BOAT", "MARINE_MARKER", "OBJECT", "LEISURE_VEHICLE", "SAILING_VESSEL", "SHIP")
CELL = 160


def background_fps(FA, preds_by_img, gts_by_img, cat_id):
    """All BACKGROUND false positives of one predicted class, highest score first."""
    out = []
    for iid, ps in preds_by_img.items():
        for p, o in FA.classify_frame(ps, gts_by_img.get(iid, [])):
            if o == "BACKGROUND" and p["category_id"] == cat_id:
                out.append((p["score"], iid, p))
    out.sort(key=lambda t: -t[0])
    return out


def crop_cell(rgb, box, gts, caption):
    H, W = rgb.shape[:2]
    x, y, w, h = box
    side = max(48.0, 4.0 * max(w, h))
    cx, cy = x + w / 2, y + h / 2
    x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x1, y1 = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
    im = Image.fromarray(rgb[y0:y1, x0:x1]).resize((CELL, CELL), Image.BILINEAR)
    sx, sy = CELL / max(1, x1 - x0), CELL / max(1, y1 - y0)
    d = ImageDraw.Draw(im)
    for g in gts:
        gx, gy, gw, gh = g["bbox"]
        if gx + gw >= x0 and gx <= x1 and gy + gh >= y0 and gy <= y1:
            d.rectangle([(gx - x0) * sx, (gy - y0) * sy, (gx + gw - x0) * sx, (gy + gh - y0) * sy], outline=(0, 255, 0))
    d.rectangle([(x - x0) * sx, (y - y0) * sy, (x + w - x0) * sx, (y + h - y0) * sy], outline=(255, 0, 0))
    canvas = Image.new("RGB", (CELL, CELL + 14), (0, 0, 0))
    canvas.paste(im, (0, 0))
    ImageDraw.Draw(canvas).text((2, CELL + 1), caption, fill=(255, 255, 255))
    return canvas


def sheet(cells, cols=8):
    rows = max(1, (len(cells) + cols - 1) // cols)
    out = Image.new("RGB", (cols * CELL, rows * (CELL + 14)), (40, 40, 40))
    for i, c in enumerate(cells):
        out.paste(c, ((i % cols) * CELL, (i // cols) * (CELL + 14)))
    return out


def run_selftest() -> None:
    rgb = np.zeros((1080, 1920, 3), dtype=np.uint8)
    rgb[500:510, 900:920] = 255
    c = crop_cell(rgb, [900, 500, 20, 10], [{"category_id": 4, "bbox": [905, 502, 6, 4]}], "v01 #1 0.90")
    a = np.asarray(c)
    assert c.size == (CELL, CELL + 14)
    assert (a[:CELL, :, 0] == 255).any() and (a[:CELL, :, 1] == 255).any()
    s = sheet([c] * 10, cols=8)
    assert s.size == (8 * CELL, 2 * (CELL + 14))
    print("  [PASS] crop around the prediction with red prediction and green GT boxes; 8-wide contact sheet")
    print("[FP-GALLERY SELFTEST] 1/1 passed")


def run(a) -> None:
    if "test" in Path(a.coco_val).name.lower():
        raise SystemExit("[SPLIT BLOCK] TEST is locked.")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import m1_1_fp_anatomy as FA
    import train_marivision_hr_p2_arms as R
    idx = R.SeaClipsZipNativeIndex(Path(a.zip), Path(a.coco_val), split_hint="val")
    val = json.loads(Path(a.coco_val).read_text(encoding="utf-8"))
    vid = {int(im["id"]): int(im.get("video_id", 0)) for im in val["images"]}
    gts = defaultdict(list)
    for g in val["annotations"]:
        if int(g.get("iscrowd", 0)) == 0:
            gts[int(g["image_id"])].append({"category_id": int(g["category_id"]), "bbox": [float(v) for v in g["bbox"]]})
    out_dir = Path(a.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for variant in ("T_tiled", "F_plain"):
        preds = defaultdict(list)
        for d in json.loads((Path(a.preds_dir) / f"m2_{a.tag}_{variant}.json").read_text(encoding="utf-8")):
            preds[int(d["image_id"])].append(d)
        for cname in ("OBJECT", "MARINE_MARKER"):
            cid = CLASSES.index(cname) + 1
            fps = background_fps(FA, preds, gts, cid)
            pick = fps[: a.n]
            cells = []
            cache = {}
            for score, iid, p in pick:
                if iid not in cache:
                    cache = {iid: idx.load_rgb_numpy(iid)}  # keep one frame in memory
                cells.append(crop_cell(cache[iid], p["bbox"], gts.get(iid, []), f"v{vid[iid]:02d} #{iid} {score:.2f}"))
            path = out_dir / f"fp_gallery_{a.tag}_{variant}_{cname}.png"
            if cells:
                sheet(cells).save(path)
            print(f"  {variant} {cname}: {len(fps)} background FPs >= 0.05; top {len(pick)} saved to {path if cells else '(none)'}"
                  + (f" | score range {pick[-1][0]:.2f}-{pick[0][0]:.2f} | videos "
                     + ", ".join(f'v{v:02d}:{n}' for v, n in sorted(Counter(vid[i] for _, i, _ in pick).items()))
                     if pick else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("selftest", "run"), required=True)
    ap.add_argument("--tag", default="seed42")
    ap.add_argument("--n", type=int, default=48, help="crops per sheet")
    ap.add_argument("--preds-dir", default="D:/MARIVISION/R0_frozen_baseline/m2_preds")
    ap.add_argument("--zip", default="D:/MARIVISION/SeaClips.zip")
    ap.add_argument("--coco-val", default="D:/MARIVISION/seaclips-val.json")
    ap.add_argument("--report-dir", default="m2_reports")
    a = ap.parse_args()
    run_selftest() if a.mode == "selftest" else run(a)


if __name__ == "__main__":
    main()
