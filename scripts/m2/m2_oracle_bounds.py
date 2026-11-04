r"""m2_oracle_bounds.py - READ-ONLY, CPU-only HEADROOM analysis (MARIVISION-HR-P2, 2026-10-28). Writes one small report.

!! ORACLE: this uses the VAL ground truth to delete or relabel predictions. Its numbers are UPPER BOUNDS that tell us how
!! much a perfect second-stage verifier could ever recover. They are NEVER results, never used to pick a model, and must
!! never appear in the paper as performance.

Question: arm T (tiled training + tiled inference) finds 944 of 1,386 VAL objects with the right class (arm F plain: 430),
yet macro mAP@0.3 barely moves because of confident false positives. Is it worth building a lightweight verifier
(the "classification/verification expert") that re-scores the proposals? For each variant, pycocotools macro mAP@0.3
after an oracle removes, using m1_1_fp_anatomy's buckets at score >= 0.05:
  O1  BACKGROUND false positives
  O2  BACKGROUND + PART_OF_OBJECT
  O3  all false positives except LOCALIZATION/DUPLICATE (i.e. O2 + WRONG_CLASS removed)
  O4  O3, and WRONG_CLASS predictions relabelled to the GT class they overlap instead of removed (perfect classifier)
Predictions below 0.05 are left untouched in every row.

Run from D:\MARIVISION\H0_workbench\eval_tools:
  python -W error m2_oracle_bounds.py --mode selftest
  python -W error m2_oracle_bounds.py --mode run --tag seed42
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

VARIANTS = ("F_plain", "F_tiled", "T_plain", "T_tiled", "R0e04_plain", "R0e04_tiled")


def best_other_class_gt(p, gts, FA):
    best, cls = 0.30, None
    for g in gts:
        if g["category_id"] != p["category_id"]:
            v = FA.iou(p["bbox"], g["bbox"])
            if v >= best:
                best, cls = v, g["category_id"]
    return cls


def apply_oracle(FA, preds_by_img, gts_by_img, level):
    out = []
    for iid, ps in preds_by_img.items():
        gts = gts_by_img.get(iid, [])
        low = [p for p in ps if p["score"] < FA.SCORE_MIN]
        out.extend(low)
        for p, o in FA.classify_frame(ps, gts):
            drop = (o == "BACKGROUND" and level >= 1) or (o == "PART_OF_OBJECT" and level >= 2) or \
                   (o == "WRONG_CLASS" and level == 3)
            if o == "WRONG_CLASS" and level == 4:
                c = best_other_class_gt(p, gts, FA)
                out.append(dict(p, category_id=c) if c else p)
                continue
            if o in ("BACKGROUND", "PART_OF_OBJECT") and level == 4:
                continue
            if not drop:
                out.append(p)
    return out


def selftest() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import m1_1_fp_anatomy as FA
    gts = {1: [{"category_id": 2, "bbox": [0, 0, 10, 10]}, {"category_id": 6, "bbox": [100, 100, 200, 200]}]}
    preds = {1: [{"image_id": 1, "category_id": 2, "score": 0.9, "bbox": [0, 0, 10, 10]},
                 {"image_id": 1, "category_id": 4, "score": 0.8, "bbox": [500, 500, 10, 10]},      # background
                 {"image_id": 1, "category_id": 3, "score": 0.7, "bbox": [150, 150, 20, 20]},      # part of object
                 {"image_id": 1, "category_id": 4, "score": 0.6, "bbox": [100, 100, 200, 200]},    # wrong class
                 {"image_id": 1, "category_id": 4, "score": 0.01, "bbox": [900, 900, 5, 5]}]}     # low score: untouched
    n = [len(apply_oracle(FA, preds, gts, k)) for k in (1, 2, 3, 4)]
    assert n == [4, 3, 2, 3], n
    o4 = apply_oracle(FA, preds, gts, 4)
    assert any(p["category_id"] == 6 and p["score"] == 0.6 for p in o4)
    print("  [PASS] O1-O4 remove background / +parts / +wrong class, O4 relabels wrong class; < 0.05 untouched")
    print("[ORACLE SELFTEST] 1/1 passed")


def run(a) -> None:
    if "test" in Path(a.coco_val).name.lower():
        raise SystemExit("[SPLIT BLOCK] TEST is locked.")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import m1_1_fp_anatomy as FA
    import m2_crowd_audit as CA
    val = json.loads(Path(a.coco_val).read_text(encoding="utf-8"))
    L = ["# ORACLE headroom (upper bounds, uses VAL GT - NOT results) - " + a.tag, "",
         "| variant | as is | O1 no background FP | O2 + no part FP | O3 + no wrong-class FP | O4 O3 with perfect relabel |",
         "|---|---|---|---|---|---|"]
    for v in VARIANTS:
        preds = json.loads((Path(a.preds_dir) / f"m2_{a.tag}_{v}.json").read_text(encoding="utf-8"))
        frames = {int(d["image_id"]) for d in preds}
        images = [im for im in val["images"] if int(im["id"]) in frames]
        anns = [x for x in val["annotations"] if int(x["image_id"]) in frames and int(x.get("iscrowd", 0)) == 0]
        cats = [{"id": i, "name": n} for i, n in enumerate(CA.CLASSES, 1)]
        gts = defaultdict(list)
        for x in anns:
            gts[int(x["image_id"])].append({"category_id": int(x["category_id"]), "bbox": [float(t) for t in x["bbox"]]})
        by = defaultdict(list)
        for d in preds:
            by[int(d["image_id"])].append({k: d[k] for k in ("image_id", "category_id", "bbox", "score")})
        row = [CA.coco_map30(images, anns, cats, [p for ps in by.values() for p in ps], include_crowd=False)[0]]
        for k in (1, 2, 3, 4):
            row.append(CA.coco_map30(images, anns, cats, apply_oracle(FA, by, gts, k), include_crowd=False)[0])
        L.append(f"| {v} | " + " | ".join(str(x) for x in row) + " |")
    L += ["", "Upper bounds only: a real verifier recovers some fraction of the O1-O4 headroom, never more."]
    out = Path(a.report_dir) / f"m2_oracle_bounds_{a.tag}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L) + f"\n\n[ORACLE] wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("selftest", "run"), required=True)
    ap.add_argument("--tag", default="seed42")
    ap.add_argument("--preds-dir", default="D:/MARIVISION/R0_frozen_baseline/m2_preds")
    ap.add_argument("--coco-val", default="D:/MARIVISION/seaclips-val.json")
    ap.add_argument("--report-dir", default="m2_reports")
    a = ap.parse_args()
    selftest() if a.mode == "selftest" else run(a)


if __name__ == "__main__":
    main()
