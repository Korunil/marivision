r"""m2_class_diagnostic.py - READ-ONLY, CPU-only (MARIVISION-HR-P2, 2026-10-27). Writes one small report.

Question: why did ANIMAL and OBJECT collapse in M2 arm T (plain AND tiled), while BOAT, MARKER and SAILING_VESSEL improved?
It reads the prediction files m2_evaluate.py already wrote (no GPU) and, for every ground-truth class, reports what
happened to each GT box in each variant:
  FOUND_RIGHT_CLASS   a same-class prediction (score >= 0.05) matched it 1-to-1 at IoU >= 0.30
  FOUND_AS_<X>        not matched, but the best-scoring other-class prediction (score >= 0.05) overlaps it at IoU >= 0.30
  SEEN_LOW_SCORE      only a same-class prediction below 0.05 (inside the top-100) overlaps it at IoU >= 0.30
  MISSED              nothing overlaps it at IoU >= 0.30
It also prints, per predicted class, how many predictions score >= 0.05 and their score quartiles, and per video the
number of ANIMAL / OBJECT GT boxes (how much evidence those two AP values rest on).

Run from D:\MARIVISION\H0_workbench\eval_tools:
  python -W error m2_class_diagnostic.py --mode selftest
  python -W error m2_class_diagnostic.py --mode run --tag seed42
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = ("ANIMAL", "BOAT", "MARINE_MARKER", "OBJECT", "LEISURE_VEHICLE", "SAILING_VESSEL", "SHIP")
VARIANTS = ("R0e15_plain", "F_plain", "F_tiled", "T_plain", "T_tiled")
S_MIN = 0.05


def iou(a, b):
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0


def gt_outcomes(preds, gts):
    """Returns list of (gt_class_id, outcome) for one frame."""
    strong = sorted([p for p in preds if p["score"] >= S_MIN], key=lambda d: -d["score"])
    matched_gt, used = {}, set()
    for pi, p in enumerate(strong):  # 1-to-1 greedy, same class, as the AP evaluator
        best, bj = 0.30, -1
        for j, g in enumerate(gts):
            if j in matched_gt or g["category_id"] != p["category_id"]:
                continue
            v = iou(p["bbox"], g["bbox"])
            if v >= best:
                best, bj = v, j
        if bj >= 0:
            matched_gt[bj] = pi
            used.add(pi)
    out = []
    for j, g in enumerate(gts):
        c = g["category_id"]
        if j in matched_gt:
            out.append((c, "FOUND_RIGHT_CLASS"))
            continue
        others = [p for p in strong if p["category_id"] != c and iou(p["bbox"], g["bbox"]) >= 0.30]
        if others:
            out.append((c, "FOUND_AS_" + CLASSES[max(others, key=lambda d: d["score"])["category_id"] - 1]))
        elif any(p["category_id"] == c and p["score"] < S_MIN and iou(p["bbox"], g["bbox"]) >= 0.30 for p in preds):
            out.append((c, "SEEN_LOW_SCORE"))
        else:
            out.append((c, "MISSED"))
    return out


def quartiles(v):
    if not v:
        return "-"
    v = sorted(v)
    q = lambda f: v[min(len(v) - 1, int(f * (len(v) - 1)))]
    return f"{q(0.25):.3f} / {q(0.5):.3f} / {q(0.75):.3f} / max {v[-1]:.3f}"


def selftest() -> None:
    gts = [{"category_id": 1, "bbox": [0, 0, 10, 10]}, {"category_id": 4, "bbox": [100, 100, 10, 10]},
           {"category_id": 2, "bbox": [200, 200, 10, 10]}, {"category_id": 6, "bbox": [300, 300, 10, 10]}]
    preds = [{"category_id": 1, "score": 0.5, "bbox": [0, 0, 10, 10]},
             {"category_id": 3, "score": 0.4, "bbox": [100, 100, 10, 10]},
             {"category_id": 2, "score": 0.02, "bbox": [200, 200, 10, 10]}]
    got = gt_outcomes(preds, gts)
    assert got == [(1, "FOUND_RIGHT_CLASS"), (4, "FOUND_AS_MARINE_MARKER"), (2, "SEEN_LOW_SCORE"), (6, "MISSED")], got
    print("  [PASS] GT outcomes: right class / found as other class / seen only below 0.05 / missed")
    print("[M2-CLASS-DIAG SELFTEST] 1/1 passed")


def run(a) -> None:
    if "test" in Path(a.coco_val).name.lower():
        raise SystemExit("[SPLIT BLOCK] TEST is locked.")
    val = json.loads(Path(a.coco_val).read_text(encoding="utf-8"))
    vid = {int(im["id"]): int(im.get("video_id", 0)) for im in val["images"]}
    preds = {}
    for v in VARIANTS:
        p = Path(a.preds_dir) / f"m2_{a.tag}_{v}.json"
        preds[v] = defaultdict(list)
        for d in json.loads(p.read_text(encoding="utf-8")):
            preds[v][int(d["image_id"])].append(d)
    frames = sorted(preds["T_tiled"].keys() | preds["F_plain"].keys())
    gts = defaultdict(list)
    for g in val["annotations"]:
        if int(g.get("iscrowd", 0)) == 0 and int(g["image_id"]) in set(frames):
            gts[int(g["image_id"])].append({"category_id": int(g["category_id"]), "bbox": [float(x) for x in g["bbox"]]})
    L = [f"# M2 class diagnostic - {a.tag}", "", f"{len(frames)} VAL frames; GT outcome counts (score >= {S_MIN}, IoU >= 0.30)", ""]
    for c in (1, 4, 2, 3, 6, 7):
        name = CLASSES[c - 1]
        n_gt = sum(1 for i in frames for g in gts[i] if g["category_id"] == c)
        L += [f"### GT class {name} (n = {n_gt})", "", "| outcome | " + " | ".join(VARIANTS) + " |", "|---|" + "---|" * len(VARIANTS)]
        tab = {}
        for v in VARIANTS:
            cnt = Counter()
            for i in frames:
                for gc, o in gt_outcomes(preds[v][i], gts[i]):
                    if gc == c:
                        cnt[o] += 1
            tab[v] = cnt
        keys = sorted(set().union(*[set(t) for t in tab.values()]), key=lambda k: (k != "FOUND_RIGHT_CLASS", k))
        for k in keys:
            L.append(f"| {k} | " + " | ".join(str(tab[v][k]) for v in VARIANTS) + " |")
        L.append("")
    L += ["### Predictions with score >= 0.05, per predicted class: count and score quartiles (25/50/75/max)", "",
          "| predicted class | " + " | ".join(VARIANTS) + " |", "|---|" + "---|" * len(VARIANTS)]
    for c in range(1, 8):
        cells = []
        for v in VARIANTS:
            s = [d["score"] for i in frames for d in preds[v][i] if d["category_id"] == c and d["score"] >= S_MIN]
            cells.append(f"{len(s)}: {quartiles(s)}")
        L.append(f"| {CLASSES[c - 1]} | " + " | ".join(cells) + " |")
    L += ["", "### ANIMAL and OBJECT GT boxes per video (evidence behind those two AP values)", "",
          "| video | frames | ANIMAL GT | OBJECT GT |", "|---|---|---|---|"]
    per = defaultdict(Counter)
    for i in frames:
        per[vid[i]]["frames"] += 1
        for g in gts[i]:
            per[vid[i]][g["category_id"]] += 1
    for v in sorted(per):
        L.append(f"| v{v:02d} | {per[v]['frames']} | {per[v][1]} | {per[v][4]} |")
    out = Path(a.report_dir) / f"m2_class_diagnostic_{a.tag}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L) + f"\n\n[M2-CLASS-DIAG] wrote {out}")


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
