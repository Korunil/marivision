r"""m2_crowd_audit.py - READ-ONLY, CPU-only (MARIVISION-HR-P2, 2026-10-28). Writes one small report.

Why: the FP gallery shows arm T's most confident "background" OBJECTs are persistent bright blobs on the horizon and
shoreline, repeated frame after frame in the same videos, and arm F (= R0) fires on the same spots. Two facts make
COCO "crowd" annotations (iscrowd=1) a prime suspect:
  * R0's training adapter (datasets/seaclips_detection.py) keeps EVERY annotation, crowd included, as an ordinary
    positive target - so R0 and both M2 arms were trained to detect crowd regions as objects of their category;
  * the HR-P2 evaluator (and its pycocotools parity check) DROPS crowd annotations entirely, so a detection on a crowd
    region counts as a false positive - whereas standard COCOeval, and R0's own training-time evaluation, treat crowd
    regions as "ignore" (detections matched to them are neither TP nor FP).
This audit measures how much that matters. It does not change any file.

  1. iscrowd counts per class, TRAIN and VAL, with median box area; VAL crowd boxes per video on the evaluated frames.
  2. For each M2 prediction file: share of the score >= 0.05 BACKGROUND false positives (m1_1_fp_anatomy rules) that lie
     on a VAL crowd box (>= 50% of the prediction inside it, or IoU >= 0.10).
  3. pycocotools macro mAP@0.3 for every variant under BOTH protocols on the same 432 frames:
       A  non-crowd GT only (current HR-P2 protocol)
       B  standard COCOeval: crowd boxes included as iscrowd=1 (ignore regions)
     B is reported, not adopted: which protocol the SeaClips authors use must be checked against their code/paper.

Run from D:\MARIVISION\H0_workbench\eval_tools:
  python -W error m2_crowd_audit.py --mode selftest
  python -W error m2_crowd_audit.py --mode run --tag seed42
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CLASSES = ("ANIMAL", "BOAT", "MARINE_MARKER", "OBJECT", "LEISURE_VEHICLE", "SAILING_VESSEL", "SHIP")
VARIANTS = ("R0e04_plain", "R0e04_tiled", "R0e15_plain", "R0e15_tiled", "F_plain", "F_tiled", "T_plain", "T_tiled")


def inside_or_overlap(p, g):
    ix = max(0.0, min(p[0] + p[2], g[0] + g[2]) - max(p[0], g[0]))
    iy = max(0.0, min(p[1] + p[3], g[1] + g[3]) - max(p[1], g[1]))
    inter = ix * iy
    u = p[2] * p[3] + g[2] * g[3] - inter
    return inter >= 0.5 * p[2] * p[3] or (u > 0 and inter / u >= 0.10)


def coco_map30(images, anns, cats, preds, include_crowd):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt = COCO()
    gt.dataset = {"images": images, "categories": cats,
                  "annotations": [a for a in anns if include_crowd or int(a.get("iscrowd", 0)) == 0]}
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
        dt = gt.loadRes(copy.deepcopy(preds)) if preds else COCO()
        ev = COCOeval(gt, dt, "bbox")
        ev.params.iouThrs = np.array([0.30])
        ev.params.imgIds = [im["id"] for im in images]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    per = {}
    for k, cid in enumerate(ev.params.catIds):
        p = ev.eval["precision"][0, :, k, 0, 2]
        p = p[p > -1]
        per[CLASSES[cid - 1]] = round(float(p.mean()), 4) if p.size else None
    return round(float(ev.stats[0]), 4), per


def selftest() -> None:
    images = [{"id": 1, "width": 1920, "height": 1080}]
    cats = [{"id": i, "name": n} for i, n in enumerate(CLASSES, 1)]
    anns = [{"id": 1, "image_id": 1, "category_id": 4, "bbox": [100, 100, 20, 10], "area": 200, "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 4, "bbox": [900, 500, 300, 40], "area": 12000, "iscrowd": 1}]
    preds = [{"image_id": 1, "category_id": 4, "bbox": [100, 100, 20, 10], "score": 0.5},
             {"image_id": 1, "category_id": 4, "bbox": [1000, 505, 30, 20], "score": 0.9}]  # sits on the crowd region
    a, _ = coco_map30(images, anns, cats, preds, include_crowd=False)
    b, _ = coco_map30(images, anns, cats, preds, include_crowd=True)
    assert b > a and abs(b - 1.0) < 1e-6, (a, b)
    assert inside_or_overlap([1000, 505, 30, 20], [900, 500, 300, 40]) and not inside_or_overlap([10, 10, 5, 5], [900, 500, 300, 40])
    print(f"  [PASS] a confident detection on a crowd region: AP {a} when crowd is dropped vs {b} when crowd = ignore")
    print("[CROWD-AUDIT SELFTEST] 1/1 passed")


def run(a) -> None:
    for p in (a.coco_val, a.coco_train):
        if "test" in Path(p).name.lower():
            raise SystemExit("[SPLIT BLOCK] TEST is locked.")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import m1_1_fp_anatomy as FA
    L = [f"# Crowd-annotation audit - {a.tag}", ""]
    for name, path in (("TRAIN", a.coco_train), ("VAL", a.coco_val)):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        c0, c1, ar1 = Counter(), Counter(), defaultdict(list)
        for x in d["annotations"]:
            k = CLASSES[int(x["category_id"]) - 1]
            if int(x.get("iscrowd", 0)):
                c1[k] += 1
                ar1[k].append(float(x["bbox"][2]) * float(x["bbox"][3]))
            else:
                c0[k] += 1
        L += [f"### {name}: annotations by class", "", "| class | non-crowd | crowd (iscrowd=1) | median crowd box area px^2 |", "|---|---|---|---|"]
        for k in CLASSES:
            L.append(f"| {k} | {c0[k]} | {c1[k]} | {int(np.median(ar1[k])) if ar1[k] else '-'} |")
        L.append(f"| total | {sum(c0.values())} | {sum(c1.values())} | |")
        L.append("")
    val = json.loads(Path(a.coco_val).read_text(encoding="utf-8"))
    preds = {}
    for v in VARIANTS:
        preds[v] = json.loads((Path(a.preds_dir) / f"m2_{a.tag}_{v}.json").read_text(encoding="utf-8"))
    frames = sorted({int(d["image_id"]) for d in preds["T_tiled"]})
    fset = set(frames)
    images = [im for im in val["images"] if int(im["id"]) in fset]
    vid = {int(im["id"]): int(im.get("video_id", 0)) for im in images}
    anns = [x for x in val["annotations"] if int(x["image_id"]) in fset]
    crowd_by = defaultdict(list)
    gts_nc = defaultdict(list)
    for x in anns:
        b = [float(v) for v in x["bbox"]]
        if int(x.get("iscrowd", 0)):
            crowd_by[int(x["image_id"])].append(b)
        else:
            gts_nc[int(x["image_id"])].append({"category_id": int(x["category_id"]), "bbox": b})
    per_vid = Counter(vid[i] for i, bs in crowd_by.items() for _ in bs)
    L += [f"### VAL evaluated frames ({len(frames)}): crowd boxes per video", "",
          "| video | crowd boxes | frames with crowd |", "|---|---|---|"]
    for v in sorted(set(vid.values())):
        L.append(f"| v{v:02d} | {per_vid[v]} | {sum(1 for i in crowd_by if vid[i] == v)} |")
    L += ["", "### Background FPs (score >= 0.05) lying on a VAL crowd box", "",
          "| variant | background FPs | on a crowd box | share | of these: OBJECT | MARKER |", "|---|---|---|---|---|---|"]
    for v in VARIANTS:
        by = defaultdict(list)
        for d in preds[v]:
            by[int(d["image_id"])].append(d)
        n_bg, n_on, cls_on = 0, 0, Counter()
        for i, ps in by.items():
            for p, o in FA.classify_frame(ps, gts_nc.get(i, [])):
                if o != "BACKGROUND":
                    continue
                n_bg += 1
                if any(inside_or_overlap(p["bbox"], c) for c in crowd_by.get(i, [])):
                    n_on += 1
                    cls_on[CLASSES[p["category_id"] - 1]] += 1
        L.append(f"| {v} | {n_bg} | {n_on} | {n_on / max(1, n_bg):.1%} | {cls_on['OBJECT']} | {cls_on['MARINE_MARKER']} |")
    cats = [{"id": i, "name": n} for i, n in enumerate(CLASSES, 1)]
    L += ["", "### pycocotools macro mAP@0.3 on the same frames: A = crowd dropped (HR-P2 protocol), B = crowd as ignore (standard COCOeval)", "",
          "| variant | A macro | B macro | B - A | " + " | ".join(f"{c} A/B" for c in CLASSES if c != "LEISURE_VEHICLE") + " |",
          "|---|---|---|---|" + "---|" * 6]
    res = {}
    for v in VARIANTS:
        clean = [{k: d[k] for k in ("image_id", "category_id", "bbox", "score")} for d in preds[v]]
        ma, pa = coco_map30(images, anns, cats, clean, include_crowd=False)
        mb, pb = coco_map30(images, anns, cats, clean, include_crowd=True)
        res[v] = (ma, mb)
        L.append(f"| {v} | {ma} | {mb} | {mb - ma:+.4f} | " + " | ".join(f"{pa.get(c)}/{pb.get(c)}" for c in CLASSES if c != "LEISURE_VEHICLE") + " |")
    L += ["", f"Primary pair under B: T_tiled {res['T_tiled'][1]} vs F_plain {res['F_plain'][1]} "
          f"(delta {res['T_tiled'][1] - res['F_plain'][1]:+.4f}); under A: {res['T_tiled'][0]} vs {res['F_plain'][0]}."]
    out = Path(a.report_dir) / f"m2_crowd_audit_{a.tag}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L) + f"\n\n[CROWD-AUDIT] wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("selftest", "run"), required=True)
    ap.add_argument("--tag", default="seed42")
    ap.add_argument("--preds-dir", default="D:/MARIVISION/R0_frozen_baseline/m2_preds")
    ap.add_argument("--coco-val", default="D:/MARIVISION/seaclips-val.json")
    ap.add_argument("--coco-train", default="D:/MARIVISION/seaclips-train.json")
    ap.add_argument("--report-dir", default="m2_reports")
    a = ap.parse_args()
    selftest() if a.mode == "selftest" else run(a)


if __name__ == "__main__":
    main()
