r"""m1_1_fp_anatomy.py - READ-ONLY, CPU-only diagnostic (MARIVISION-HR-P2, 2026-10-27). Writes one small report pair.

Question: WHY does tiled inference of the frozen R0 lose AP on large classes (SAILING_VESSEL, SHIP, ANIMAL) while it
finds far more objects? It does NOT choose or tune anything; it explains the M1/M1.1 result and informs the M2
(tiled fine-tune) design.

For plain R0 (the cache's Stage B tile) and for tiled R0 (V0 = M1 merge), every prediction with score >= 0.05 is matched
1-to-1 (greedy by score, per image and class, IoU >= 0.30, as the AP evaluator does). Each unmatched prediction is put
in exactly one bucket, in this order:
  DUPLICATE       IoU >= 0.30 with a same-class GT that is already matched
  WRONG_CLASS     IoU >= 0.30 with a GT of another class
  PART_OF_OBJECT  >= 70% of the prediction lies inside one GT box (any class) that is >= 2x its area (an object fragment)
  LOCALIZATION    0.10 <= best same-class IoU < 0.30
  BACKGROUND      everything else
Reported per predicted class, per source stage (tiled: cluster led by a Stage B or a Stage C box), and as false
positives scoring above each class's median true-positive score (the ones that actually cost AP).

Run from D:\MARIVISION\H0_workbench\eval_tools:
  python -W error m1_1_fp_anatomy.py --mode selftest
  python -W error m1_1_fp_anatomy.py --mode run --cache D:\MARIVISION\R0_frozen_baseline\m1_1_tile_cache\r0_tile_cache_epoch15_stride10_offset5.json.gz
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = ("ANIMAL", "BOAT", "MARINE_MARKER", "OBJECT", "LEISURE_VEHICLE", "SAILING_VESSEL", "SHIP")
BUCKETS = ("DUPLICATE", "WRONG_CLASS", "PART_OF_OBJECT", "LOCALIZATION", "BACKGROUND")
SCORE_MIN = 0.05


def iou(a, b):
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0


def inside_frac(p, g):
    ix = max(0.0, min(p[0] + p[2], g[0] + g[2]) - max(p[0], g[0]))
    iy = max(0.0, min(p[1] + p[3], g[1] + g[3]) - max(p[1], g[1]))
    return ix * iy / max(1e-9, p[2] * p[3])


def classify_frame(preds, gts):
    """preds: dicts with category_id, score, bbox (+ optional stage); gts: dicts with category_id, bbox.
    Returns list of (pred, outcome) where outcome is 'TP' or a bucket."""
    out = []
    matched = set()
    for p in sorted([p for p in preds if p["score"] >= SCORE_MIN], key=lambda d: -d["score"]):
        best_j, best = -1, 0.30
        for j, g in enumerate(gts):
            if j in matched or g["category_id"] != p["category_id"]:
                continue
            v = iou(p["bbox"], g["bbox"])
            if v >= best:
                best_j, best = j, v
        if best_j >= 0:
            matched.add(best_j)
            out.append((p, "TP"))
            continue
        same = [iou(p["bbox"], g["bbox"]) for g in gts if g["category_id"] == p["category_id"]]
        if any(v >= 0.30 for v in same):
            b = "DUPLICATE"
        elif any(iou(p["bbox"], g["bbox"]) >= 0.30 for g in gts if g["category_id"] != p["category_id"]):
            b = "WRONG_CLASS"
        elif any(inside_frac(p["bbox"], g["bbox"]) >= 0.70 and g["bbox"][2] * g["bbox"][3] >= 2.0 * p["bbox"][2] * p["bbox"][3]
                 for g in gts):
            b = "PART_OF_OBJECT"
        elif any(v >= 0.10 for v in same):
            b = "LOCALIZATION"
        else:
            b = "BACKGROUND"
        out.append((p, b))
    return out


def summarize(rows, by_stage=False):
    tp_scores = defaultdict(list)
    for p, o in rows:
        if o == "TP":
            tp_scores[p["category_id"]].append(p["score"])
    med = {c: (sorted(v)[len(v) // 2] if v else 1.0) for c, v in tp_scores.items()}
    tab = defaultdict(Counter)
    hi = defaultdict(Counter)
    for p, o in rows:
        key = CLASSES[p["category_id"] - 1] + ((" @" + p.get("stage", "?").split("_")[1]) if by_stage else "")
        tab[key][o] += 1
        if o != "TP" and p["score"] >= med.get(p["category_id"], 1.0):
            hi[key][o] += 1
    return tab, hi, med


def table(title, tab, hi):
    L = [f"### {title}", "", "| predicted class | TP | " + " | ".join(BUCKETS) + " | FP total | FP >= median TP score (by bucket) |",
         "|---|---|" + "---|" * len(BUCKETS) + "---|---|"]
    for k in sorted(tab):
        c = tab[k]
        fp = sum(c[b] for b in BUCKETS)
        hs = ", ".join(f"{b[:4]} {hi[k][b]}" for b in BUCKETS if hi[k][b])
        L.append(f"| {k} | {c['TP']} | " + " | ".join(str(c[b]) for b in BUCKETS) + f" | {fp} | {hs or '-'} |")
    return L


def run_selftest(a) -> None:
    gts = [{"category_id": 6, "bbox": [100, 100, 200, 300]}, {"category_id": 3, "bbox": [500, 500, 10, 6]}]
    preds = [{"category_id": 6, "score": 0.9, "bbox": [102, 101, 198, 298]},   # TP
             {"category_id": 6, "score": 0.8, "bbox": [104, 104, 190, 290]},   # duplicate
             {"category_id": 6, "score": 0.7, "bbox": [150, 120, 40, 60]},     # part of the sailing vessel
             {"category_id": 1, "score": 0.6, "bbox": [501, 500, 10, 6]},      # wrong class on the marker
             {"category_id": 3, "score": 0.5, "bbox": [497, 500, 10, 6]},      # TP marker (IoU ~0.54)
             {"category_id": 7, "score": 0.4, "bbox": [900, 900, 50, 50]},     # background
             {"category_id": 3, "score": 0.02, "bbox": [0, 0, 5, 5]}]          # below 0.05: ignored
    got = [o for _, o in classify_frame(preds, gts)]
    assert got == ["TP", "DUPLICATE", "PART_OF_OBJECT", "WRONG_CLASS", "TP", "BACKGROUND"], got
    print("  [PASS] bucket rules: TP, DUPLICATE, PART_OF_OBJECT, WRONG_CLASS, BACKGROUND, score floor 0.05")
    loc = classify_frame([{"category_id": 3, "score": 0.5, "bbox": [506, 500, 10, 6]}], gts)  # IoU 0.25
    assert loc[0][1] == "LOCALIZATION", loc
    print("  [PASS] LOCALIZATION bucket (0.10 <= IoU < 0.30)")
    print("[FP-ANATOMY SELFTEST] 2/2 passed")


def run(a) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import m1_1_merge_variants as MV
    hr_p2, M1 = MV.import_modules(a)
    if "test" in Path(a.coco_val).name.lower():
        raise SystemExit("[SPLIT BLOCK] TEST is locked.")
    with gzip.open(a.cache, "rt", encoding="utf-8") as f:
        cache = json.load(f)
    meta = cache["meta"]
    if meta["coco_val_sha256"] != MV.sha256_file(a.coco_val):
        raise SystemExit("[REFUSED] cache was built against a different VAL json.")
    l2c = {int(k): int(v) for k, v in meta["label_to_cat_id"].items()}
    val = json.loads(Path(a.coco_val).read_text(encoding="utf-8"))
    gts = defaultdict(list)
    for g in val["annotations"]:
        if int(g.get("iscrowd", 0)) == 0:
            gts[int(g["image_id"])].append({"category_id": int(g["category_id"]), "bbox": [float(v) for v in g["bbox"]]})
    rows_plain, rows_tiled = [], []
    for iid, fr in cache["frames"].items():
        tiles = MV.frame_tiles(M1, hr_p2, fr)
        b = [t for t in tiles if t[0].stage == MV.STAGE_B][0][1]
        plain = M1.plain_dets_native([d[0] for d in b], [d[1] for d in b], [d[2:6] for d in b], l2c)
        tiled = MV.build_variant("V0", M1, hr_p2, l2c, fr)
        rows_plain += classify_frame(plain, gts[int(iid)])
        rows_tiled += classify_frame(tiled, gts[int(iid)])
    tp, hp, _ = summarize(rows_plain)
    tt, ht, _ = summarize(rows_tiled)
    ts, hs, _ = summarize(rows_tiled, by_stage=True)
    tag = Path(a.cache).name.replace("r0_tile_cache_", "").replace(".json.gz", "")
    L = [f"# FP anatomy (score >= {SCORE_MIN}, 1-to-1 @ IoU 0.30) - {tag}", "",
         f"Frames {len(cache['frames'])}; checkpoint epoch {meta.get('source_checkpoint_epoch')}. Tiled = V0 (M1 merge).", ""]
    L += table("Plain R0", tp, hp) + [""] + table("Tiled R0 (V0)", tt, ht) + [""]
    L += table("Tiled R0 (V0) by leading source stage (B = full frame, C = 640x360 tile)", ts, hs)
    out = Path(a.report_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"m1_1_fp_anatomy_{tag}.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L) + f"\n\n[FP-ANATOMY] wrote {p}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("selftest", "run"), required=True)
    ap.add_argument("--cache")
    ap.add_argument("--m1-dir", default="D:/MARIVISION/R0_frozen_baseline")
    ap.add_argument("--coco-val", default="D:/MARIVISION/seaclips-val.json")
    ap.add_argument("--report-dir", default="m1_reports")
    a = ap.parse_args()
    run_selftest(a) if a.mode == "selftest" else run(a)


if __name__ == "__main__":
    main()
