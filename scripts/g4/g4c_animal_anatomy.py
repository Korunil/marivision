r"""
G4c: why does ANIMAL fail in the ConvNeXt-B + tiles models, and can class routing recover it?  (CPU only, read-only)

Location: D:/MARIVISION/H0_workbench/eval_tools/g4c_animal_anatomy.py  (run FROM that folder, next to
g4f_followup.py, which is SHA-checked and reused: cache -> plain / tiled predictions, pycocotools AP@0.3, GT fates).

Reads only existing files: TRAIN / VAL annotation json and the 10 VAL tile caches below (offsets 0 and 5).
No images, no model, no GPU, no TEST labels (TEST appears only as the public size bins of the paper supplement).

Q1  ANIMAL prediction anatomy (score >= 0.05, offsets 0 + 5 pooled): TP / duplicate / wrong class (which class) /
    background; in v09 (the only VAL video with ANIMAL) or elsewhere; tiled predictions backed by the full-frame tile
    (same-class plain prediction at IoU >= 0.5) or tile-only; how many score >= 0.3; size.
Q2  Fate of every pooled ANIMAL GT (g4f_followup rule): found as ANIMAL / found as another class / low score / missed.
Q3  ANIMAL size census: TRAIN per video vs VAL, native sqrt(area) and size at the 640 network input (full frame and
    Stage C tile); TEST public bins (supplement Table 1).
Q4  Class routing (pre-declared fallback): X tiled for the 5 other classes + ANIMAL from
      SELF  = X's own plain (full-frame) output,   R0 = R0 e15 plain output.
    AP is per class, so the other 5 classes are unchanged by construction (asserted).
    Frozen rule for the G5 inference configuration (X = G4F_e15): SELF if its pooled ANIMAL AP >= 0.05, else R0 if
    its pooled ANIMAL AP >= 0.05, else none.

Modes:  --mode selftest  |  --mode run
"""
from PIL import Image  # noqa: F401  -- PIL before torch (Director's Windows env)
import argparse
import gzip
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

VERSION = "G4C.0"
FU_SHA = "07dbf9ce8d58213721c892a0129d0cd0dd5d74d1d935f1a9dc386ca5c36f881d"
ANIMAL = 1
S_MIN = 0.05
RULE = {"route_animal_ap_floor": 0.05, "order": ["SELF", "R0"], "base": "G4F_e15"}
TEST_ANIMAL_PUBLIC = {"small": 928, "medium": 4, "large": 0}      # SeaClips supplement Table 1
SMALL, MEDIUM = 32 ** 2, 96 ** 2

D = "D:/MARIVISION"
M1D = f"{D}/R0_frozen_baseline/m1_1_tile_cache"
G4F = f"{D}/G4_full/run/tile_cache"
G4B = f"{D}/G4b/run/tile_cache"
CACHES = {
    "R0_e15": {0: f"{M1D}/r0_tile_cache_epoch15_stride10_offset0.json.gz",
               5: f"{M1D}/r0_tile_cache_epoch15_stride10_offset5.json.gz"},
    "G4F_e15": {0: f"{G4F}/r0_tile_cache_g4f_e15_stride10_offset0.json.gz",
                5: f"{G4F}/r0_tile_cache_g4f_e15_stride10_offset5.json.gz"},
    **{arm: {o: f"{G4B}/r0_tile_cache_g4b_{arm}_stride10_offset{o}.json.gz" for o in (0, 5)}
       for arm in ("CTRL", "NOTILE", "RFS")},
}
PINNED = {   # full SHA-256 printed by the G4b export runs (2026-10-31); a mismatch refuses
    f"{G4B}/r0_tile_cache_g4b_CTRL_stride10_offset0.json.gz": "c30990601902de9bb91e754542c9199a79131a5f977ff8513a09ee9113367897",
    f"{G4B}/r0_tile_cache_g4b_CTRL_stride10_offset5.json.gz": "032c1049b1d86d91938ed406551932d32fadda6633726e20d7d9277d7333bb1e",
    f"{G4B}/r0_tile_cache_g4b_NOTILE_stride10_offset0.json.gz": "bd5c6dc5f649f63b0c49c5c1c11730aacc6e416af4bd9ae17e67cb2903b1e51f",
    f"{G4B}/r0_tile_cache_g4b_NOTILE_stride10_offset5.json.gz": "1ee781b54f2e51698d7c26f9f941673b4dd62473ff1c509967a35163dbed20c9",
    f"{G4B}/r0_tile_cache_g4b_RFS_stride10_offset0.json.gz": "f83e9aa663836a3461253c36a578987bb9e23962fe50fddca702771d188119be",
    f"{G4B}/r0_tile_cache_g4b_RFS_stride10_offset5.json.gz": "71c9eeb2608ef27a8d1d27afea67f2c1156701d296c6aee382b77191c398baff",
}
OUT_DIR = Path("g4c_reports")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_fu(check=True):
    here = Path(__file__).resolve().parent
    p = here / "g4f_followup.py"
    if not p.exists():
        raise SystemExit(f"[REFUSED] g4f_followup.py must sit next to this script ({here})")
    if check and sha256_file(p) != FU_SHA:
        raise SystemExit(f"[REFUSED] g4f_followup.py sha {sha256_file(p)} != {FU_SHA}")
    sys.path.insert(0, str(here))
    import g4f_followup as FU
    return FU


# =============================================================================
# Q1 prediction anatomy
# =============================================================================
def anatomy(FU, preds, plain, gts, ids, vid_of, animal_videos):
    """ANIMAL predictions >= S_MIN, greedy by score.  plain=None for plain mode (no backing test)."""
    rows = Counter()
    wrong_as = Counter()
    by_video = defaultdict(Counter)
    sizes, tp_scores = defaultdict(list), []
    names = FU.CLASSES
    for i in ids:
        g_an = [g for g in gts.get(i, []) if g["category_id"] == ANIMAL and not g["iscrowd"]]
        g_ot = [g for g in gts.get(i, []) if not (g["category_id"] == ANIMAL and not g["iscrowd"])]
        pa = sorted([d for d in preds.get(i, []) if d["category_id"] == ANIMAL and d["score"] >= S_MIN],
                    key=lambda d: -d["score"])
        pl = [d for d in (plain or {}).get(i, []) if d["category_id"] == ANIMAL]
        used = set()
        for d in pa:
            best, bj = 0.30, -1
            for j, g in enumerate(g_an):
                if j in used:
                    continue
                v = FU.iou(d["bbox"], g["bbox"])
                if v >= best:
                    best, bj = v, j
            if bj >= 0:
                used.add(bj)
                kind = "TP"
                tp_scores.append(d["score"])
            elif any(FU.iou(d["bbox"], g["bbox"]) >= 0.30 for g in g_an):
                kind = "DUPLICATE"
            else:
                ov = [(FU.iou(d["bbox"], g["bbox"]), g) for g in g_ot]
                ov = [x for x in ov if x[0] >= 0.30]
                if ov:
                    g = max(ov, key=lambda x: x[0])[1]
                    kind = "ON_CROWD" if g["iscrowd"] else "WRONG_CLASS"
                    if kind == "WRONG_CLASS":
                        wrong_as[names[g["category_id"] - 1]] += 1
                else:
                    kind = "BACKGROUND"
            where = "in_animal_videos" if vid_of[i] in animal_videos else "other_videos"
            rows[kind] += 1
            rows[f"{kind}|{where}"] += 1
            if d["score"] >= 0.3:
                rows[f"{kind}|score>=0.3"] += 1
            if plain is not None and kind != "TP":
                backed = any(FU.iou(d["bbox"], q["bbox"]) >= 0.5 for q in pl)
                rows[f"FP|{'full_frame_backed' if backed else 'tile_only'}"] += 1
            by_video[f"v{vid_of[i]:02d}"][kind] += 1
            sizes[kind].append(math.sqrt(max(0.0, d["bbox"][2] * d["bbox"][3])))
    fp = sum(v for k, v in rows.items() if k in ("DUPLICATE", "WRONG_CLASS", "ON_CROWD", "BACKGROUND"))
    return {"counts": {k: rows[k] for k in sorted(rows)}, "fp_total": fp, "wrong_class_hits_gt_of": dict(wrong_as),
            "by_video": {v: dict(c) for v, c in sorted(by_video.items())},
            "median_sqrt_area_px": {k: float(np.median(v)) for k, v in sizes.items()},
            "tp_scores": sorted(tp_scores, reverse=True)[:20],
            "fp_scoring_above_best_tp": None}


def fp_above_best_tp(preds, gts, ids, FU):
    """FP ANIMAL predictions (any score) ranking above the best-scoring true ANIMAL match -- what drives AP to ~0."""
    best_tp, fps = 0.0, []
    for i in ids:
        g_an = [g for g in gts.get(i, []) if g["category_id"] == ANIMAL and not g["iscrowd"]]
        for d in preds.get(i, []):
            if d["category_id"] != ANIMAL:
                continue
            if any(FU.iou(d["bbox"], g["bbox"]) >= 0.30 for g in g_an):
                best_tp = max(best_tp, d["score"])
            else:
                fps.append(d["score"])
    return {"best_tp_score": best_tp, "fp_above_best_tp": sum(1 for s in fps if s > best_tp) if best_tp > 0 else None}


# =============================================================================
# Q3 census
# =============================================================================
def census(FU, M1, hr_p2, train, val, ids0, ids5):
    def rows(ds, ids=None):
        im = {i["id"]: i for i in ds["images"]}
        per = defaultdict(lambda: {"boxes": 0, "frames": set(), "native": [], "net_full": [], "net_tile": [],
                                   "small": 0, "medium": 0, "large": 0})
        wc = {}
        for a in ds["annotations"]:
            if a["category_id"] != ANIMAL or (ids is not None and a["image_id"] not in ids):
                continue
            m = im[a["image_id"]]
            W, H = int(m["width"]), int(m["height"])
            x, y, w, h = [float(v) for v in a["bbox"]]
            if w <= 0 or h <= 0:
                continue
            if (W, H) not in wc:
                c = [t for t in M1.build_tiles(hr_p2, W, H) if t.stage == FU.STAGE_C][0]
                wc[(W, H)] = (c.x1_int - c.x0_int, c.y1_int - c.y0_int)
            tw, th = wc[(W, H)]
            r = per[FU.video_key(m)]
            r["boxes"] += 1
            r["frames"].add(a["image_id"])
            ar = w * h
            r["native"].append(math.sqrt(ar))
            r["net_full"].append(math.sqrt(w * 640 / W * h * 640 / H))
            r["net_tile"].append(math.sqrt(w * 640 / tw * h * 640 / th))
            r["small" if ar < SMALL else ("medium" if ar < MEDIUM else "large")] += 1
        out, allr = {}, {"native": [], "net_full": [], "net_tile": [], "boxes": 0, "small": 0, "medium": 0, "large": 0}
        for v, r in sorted(per.items()):
            out[f"v{v:02d}"] = summarize(r)
            for k in ("native", "net_full", "net_tile"):
                allr[k] += r[k]
            for k in ("boxes", "small", "medium", "large"):
                allr[k] += r[k]
        allr["frames"] = set().union(*[r["frames"] for r in per.values()]) if per else set()
        out["ALL"] = summarize(allr)
        return out
    return {"train": rows(train), "val_full": rows(val), "val_off0_off5": rows(val, set(ids0) | set(ids5)),
            "test_public_bins": TEST_ANIMAL_PUBLIC}


def summarize(r):
    q = lambda a, p: float(np.percentile(a, p)) if a else None
    return {"boxes": r["boxes"], "frames": len(r["frames"]), "small": r["small"], "medium": r["medium"], "large": r["large"],
            "native_sqrt_area_q10_50_90": [q(r["native"], p) for p in (10, 50, 90)],
            "net_full_px_q10_50_90": [q(r["net_full"], p) for p in (10, 50, 90)],
            "net_tile_px_q10_50_90": [q(r["net_tile"], p) for p in (10, 50, 90)],
            "share_native_below_8px": (sum(1 for s in r["native"] if s < 8) / len(r["native"])) if r["native"] else None}


# =============================================================================
# Q4 routing
# =============================================================================
def route(tiled_x, animal_src, ids):
    return {i: [d for d in tiled_x.get(i, []) if d["category_id"] != ANIMAL] +
               [d for d in animal_src.get(i, []) if d["category_id"] == ANIMAL] for i in ids}


def route_rule(r, rule=RULE):
    b = r[rule["base"]]
    for k in rule["order"]:
        a = b["routes"][k]["animal_pooled"]
        if a is not None and a >= rule["route_animal_ap_floor"]:
            return k
    return None


# =============================================================================
def analyse(FU, MV, M1, hr_p2, val, caches, log=print, verify=True):
    imgs = {im["id"]: im for im in val["images"]}
    ids = {o: M1.eval_subset_ids(val["images"], 10, o) for o in (0, 5)}
    pooled = ids[0] + ids[5]
    assert not set(ids[0]) & set(ids[5])
    vid_of = {i: FU.video_key(imgs[i]) for i in pooled}
    gts = defaultdict(list)
    for g in val["annotations"]:
        if g["image_id"] in vid_of:
            gts[g["image_id"]].append({"category_id": int(g["category_id"]), "bbox": [float(x) for x in g["bbox"]],
                                       "iscrowd": int(g.get("iscrowd", 0))})
    animal_videos = {vid_of[i] for i in pooled for g in gts[i] if g["category_id"] == ANIMAL}
    P = {}
    shas = {}
    for name, offs in caches.items():
        P[name] = {"plain": {}, "tiled": {}}
        for o, path in offs.items():
            s = sha256_file(path)
            shas[str(path)] = s
            if verify and str(path) in PINNED and PINNED[str(path)] != s:
                raise SystemExit(f"[REFUSED] {path} sha {s} != pinned {PINNED[str(path)]}")
            c = FU.load_cache(path)
            m = c["meta"]
            if (int(m["val_frame_stride"]), int(m["val_frame_offset"])) != (10, o):
                raise SystemExit(f"[REFUSED] {path} is not stride 10 / offset {o}")
            if set(int(k) for k in c["frames"]) != set(ids[o]):
                raise SystemExit(f"[REFUSED] {path} does not cover the offset-{o} frames")
            pl, ti = FU.preds_from_cache(MV, M1, hr_p2, c)
            del c
            P[name]["plain"].update(pl)
            P[name]["tiled"].update(ti)
        log(f"  loaded {name}")
    res = {}
    for name in caches:
        r = {}
        for mode in ("plain", "tiled"):
            X = P[name][mode]
            e0, e5 = FU.coco_eval(val, ids[0], X), FU.coco_eval(val, ids[5], X)
            ep = FU.coco_eval(val, pooled, X)
            fates = Counter()
            for i in pooled:
                fates.update(o for c, o in FU.gt_outcomes(X.get(i, []),
                                                          [g for g in gts[i] if not g["iscrowd"]]) if c == ANIMAL)
            an = anatomy(FU, X, P[name]["plain"] if mode == "tiled" else None, gts, pooled, vid_of, animal_videos)
            an.update(fp_above_best_tp(X, gts, pooled, FU))
            r[mode] = {"macro_off0": e0["map"], "macro_off5": e5["map"], "per_class_off0": e0["per_class"],
                       "animal_pooled": ep["per_class"].get("ANIMAL"), "animal_gt_fates_pooled": dict(fates),
                       "anatomy": an}
        r["routes"] = {}
        base0, base5 = (FU.coco_eval(val, ids[o], P[name]["tiled"]) for o in (0, 5))
        for k, src in (("SELF", P[name]["plain"]), ("R0", P["R0_e15"]["plain"])):
            R = route(P[name]["tiled"], src, pooled)
            e0, e5, ep = FU.coco_eval(val, ids[0], R), FU.coco_eval(val, ids[5], R), FU.coco_eval(val, pooled, R)
            for e, b in ((e0, base0), (e5, base5)):
                for c, v in b["per_class"].items():
                    if c != "ANIMAL":
                        assert abs(e["per_class"][c] - v) < 1e-12, (name, k, c)
            r["routes"][k] = {"macro_off0": e0["map"], "macro_off5": e5["map"], "animal_off0": e0["per_class"].get("ANIMAL"),
                              "animal_pooled": ep["per_class"].get("ANIMAL")}
        res[name] = r
        log(f"  {name}: tiled macro off0 {r['tiled']['macro_off0']:.4f} | ANIMAL pooled plain "
            f"{(r['plain']['animal_pooled'] or 0):.4f} tiled {(r['tiled']['animal_pooled'] or 0):.4f} | route SELF "
            f"{r['routes']['SELF']['macro_off0']:.4f} R0 {r['routes']['R0']['macro_off0']:.4f}")
    return res, shas, sorted(animal_videos), ids


# =============================================================================
def report(res, cen, decision, animal_videos):
    f = lambda x: "–" if x is None else f"{x:.4f}"
    L = [f"# G4c ANIMAL anatomy and class routing ({VERSION})", "",
         f"VAL videos with ANIMAL GT in offsets 0 + 5: {animal_videos}", "",
         "## Q4 routing (frozen rule) and plain vs tiled", "",
         "| model | plain macro off0 | tiled macro off0 | tiled macro off5 | ANIMAL pooled plain | ANIMAL pooled tiled | "
         "SELF route macro off0 / off5 / ANIMAL pooled | R0 route macro off0 / off5 / ANIMAL pooled |",
         "|---|---|---|---|---|---|---|---|"]
    for n, r in res.items():
        s, z = r["routes"]["SELF"], r["routes"]["R0"]
        L.append(f"| {n} | {f(r['plain']['macro_off0'])} | {f(r['tiled']['macro_off0'])} | {f(r['tiled']['macro_off5'])} | "
                 f"{f(r['plain']['animal_pooled'])} | {f(r['tiled']['animal_pooled'])} | "
                 f"{f(s['macro_off0'])} / {f(s['macro_off5'])} / {f(s['animal_pooled'])} | "
                 f"{f(z['macro_off0'])} / {f(z['macro_off5'])} / {f(z['animal_pooled'])} |")
    L += ["", f"Rule: for {RULE['base']}, SELF if its pooled ANIMAL AP >= {RULE['route_animal_ap_floor']}, else R0 if >= "
          f"{RULE['route_animal_ap_floor']}, else none.  The other 5 classes are unchanged by construction (asserted).", "",
          f"**G5 INFERENCE ROUTING: {decision or 'none'}**", "",
          "## Q2 fate of the pooled ANIMAL GT (score >= 0.05, IoU >= 0.30)", ""]
    keys = sorted({k for r in res.values() for m in ("plain", "tiled") for k in r[m]["animal_gt_fates_pooled"]})
    L += ["| model | mode | " + " | ".join(keys) + " |", "|---|---|" + "---|" * len(keys)]
    for n, r in res.items():
        for m in ("plain", "tiled"):
            L.append(f"| {n} | {m} | " + " | ".join(str(r[m]["animal_gt_fates_pooled"].get(k, 0)) for k in keys) + " |")
    L += ["", "## Q1 anatomy of ANIMAL predictions >= 0.05 (offsets 0 + 5)", "",
          "| model | mode | TP | duplicate | wrong class | on crowd | background | FP in ANIMAL videos | FP in other videos | "
          "FP >= 0.3 | tile-only FP | full-frame-backed FP | best TP score | FP above best TP | wrong class hits GT of |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in res.items():
        for m in ("plain", "tiled"):
            a = r[m]["anatomy"]
            c = a["counts"]
            fpk = ("DUPLICATE", "WRONG_CLASS", "ON_CROWD", "BACKGROUND")
            inv = sum(c.get(f"{k}|in_animal_videos", 0) for k in fpk)
            oth = sum(c.get(f"{k}|other_videos", 0) for k in fpk)
            hi = sum(c.get(f"{k}|score>=0.3", 0) for k in fpk)
            L.append(f"| {n} | {m} | {c.get('TP', 0)} | {c.get('DUPLICATE', 0)} | {c.get('WRONG_CLASS', 0)} | "
                     f"{c.get('ON_CROWD', 0)} | {c.get('BACKGROUND', 0)} | {inv} | {oth} | {hi} | "
                     f"{c.get('FP|tile_only', 0) if m == 'tiled' else '–'} | "
                     f"{c.get('FP|full_frame_backed', 0) if m == 'tiled' else '–'} | {a['best_tp_score']:.3f} | "
                     f"{a['fp_above_best_tp'] if a['fp_above_best_tp'] is not None else '–'} | "
                     f"{json.dumps(a['wrong_class_hits_gt_of'])} |")
    L += ["", "FP by video (tiled):", ""]
    for n, r in res.items():
        L.append(f"- {n}: " + json.dumps(r["tiled"]["anatomy"]["by_video"]))
    L += ["", "## Q3 ANIMAL size census (native px; network-input px at 640, full frame and Stage C tile)", "",
          "| split | video | boxes | frames | small / medium / large | native sqrt(area) q10/50/90 | net full q10/50/90 | "
          "net tile q10/50/90 | share < 8 px native |", "|---|---|---|---|---|---|---|---|---|"]
    g = lambda v: " / ".join("–" if x is None else f"{x:.1f}" for x in v)
    for sp in ("train", "val_full", "val_off0_off5"):
        for v, s in cen[sp].items():
            L.append(f"| {sp} | {v} | {s['boxes']} | {s['frames']} | {s['small']} / {s['medium']} / {s['large']} | "
                     f"{g(s['native_sqrt_area_q10_50_90'])} | {g(s['net_full_px_q10_50_90'])} | "
                     f"{g(s['net_tile_px_q10_50_90'])} | {f(s['share_native_below_8px'])} |")
    t = cen["test_public_bins"]
    L.append(f"| TEST (public bins only) | – | {sum(t.values())} | – | {t['small']} / {t['medium']} / {t['large']} | – | – | – | – |")
    return "\n".join(L) + "\n"


# =============================================================================
def selftest(a):
    import tempfile
    print(f"[selftest] {VERSION}")
    FU = load_fu()
    hr_p2, M1, MV = FU.import_modules(a.m1_dir)
    # 1 anatomy by hand
    gts = {1: [{"category_id": 1, "bbox": [0, 0, 10, 10], "iscrowd": 0},
               {"category_id": 3, "bbox": [100, 100, 10, 10], "iscrowd": 0},
               {"category_id": 4, "bbox": [300, 300, 50, 50], "iscrowd": 1}], 2: []}
    preds = {1: [{"category_id": 1, "score": 0.9, "bbox": [0, 0, 10, 10]},
                 {"category_id": 1, "score": 0.8, "bbox": [1, 0, 10, 10]},
                 {"category_id": 1, "score": 0.5, "bbox": [100, 100, 10, 10]},
                 {"category_id": 1, "score": 0.2, "bbox": [300, 300, 50, 50]},
                 {"category_id": 1, "score": 0.03, "bbox": [500, 500, 5, 5]}],
             2: [{"category_id": 1, "score": 0.95, "bbox": [7, 7, 4, 4]}, {"category_id": 2, "score": 0.9, "bbox": [0, 0, 9, 9]}]}
    plain = {1: [{"category_id": 1, "score": 0.4, "bbox": [100, 100, 10, 10]}], 2: []}
    an = anatomy(FU, preds, plain, gts, [1, 2], {1: 9, 2: 3}, {9})
    c = an["counts"]
    assert (c["TP"], c["DUPLICATE"], c["WRONG_CLASS"], c["ON_CROWD"], c["BACKGROUND"]) == (1, 1, 1, 1, 1), c
    assert c["BACKGROUND|other_videos"] == 1 and c["FP|full_frame_backed"] == 1 and c["FP|tile_only"] == 3, c
    assert an["wrong_class_hits_gt_of"] == {"MARINE_MARKER": 1} and an["fp_total"] == 4
    assert c["BACKGROUND|score>=0.3"] == 1 and c["WRONG_CLASS|score>=0.3"] == 1
    fb = fp_above_best_tp(preds, gts, [1, 2], FU)
    assert fb == {"best_tp_score": 0.9, "fp_above_best_tp": 1}, fb
    print("  [ok] Q1 anatomy by hand: TP / duplicate / wrong class (-> GT class) / crowd / background, video, "
          "tile-only vs full-frame-backed, FP above best TP")
    # 2 routing + rule
    R = route({1: [{"category_id": 2, "score": 0.5, "bbox": [0, 0, 1, 1]}, {"category_id": 1, "score": 0.1, "bbox": [0, 0, 1, 1]}]},
              {1: [{"category_id": 1, "score": 0.7, "bbox": [2, 2, 1, 1]}, {"category_id": 2, "score": 0.9, "bbox": [2, 2, 1, 1]}]}, [1])
    assert [(d["category_id"], d["score"]) for d in R[1]] == [(2, 0.5), (1, 0.7)]
    mk = lambda s, r: {"G4F_e15": {"routes": {"SELF": {"animal_pooled": s}, "R0": {"animal_pooled": r}}}}
    assert route_rule(mk(0.06, 0.2)) == "SELF" and route_rule(mk(0.01, 0.2)) == "R0" and route_rule(mk(0.0, 0.049)) is None
    assert route_rule(mk(None, 0.05)) == "R0"
    print("  [ok] Q4 routing keeps the 5 other classes from X tiled and ANIMAL from the source; rule SELF > R0 > none")
    # 3 end to end on synthetic caches: X labels ANIMAL as BOAT outside video 1; "R0" labels every ANIMAL correctly
    val, c0 = FU._synthetic(M1, hr_p2, 0)
    _v5, c5 = FU._synthetic(M1, hr_p2, 5)
    def as_r0(cache):
        cc = json.loads(json.dumps(cache))
        for fr in cc["frames"].values():
            for t in fr["tiles"]:
                if t[0] == FU.STAGE_B and t[5]:
                    t[5][0][1] = 0
        return cc
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        caches = {}
        for name, mod in (("R0_e15", as_r0), ("G4F_e15", lambda x: x)):
            caches[name] = {}
            for o, cc in ((0, c0), (5, c5)):
                p = Path(td) / f"{name}_{o}.json.gz"
                with gzip.open(p, "wt", encoding="utf-8") as fh:
                    json.dump(mod(cc), fh)
                caches[name][o] = str(p)
        res, shas, avids, ids = analyse(FU, MV, M1, hr_p2, val, caches, log=lambda *x: None, verify=False)
        n = len(ids[0]) + len(ids[5])
        n1 = sum(1 for i in ids[0] + ids[5] if 1 <= i <= 40)
        g = res["G4F_e15"]
        assert g["plain"]["animal_gt_fates_pooled"] == {"FOUND_RIGHT_CLASS": n1, "FOUND_AS_BOAT": n - n1}, g["plain"]
        assert res["R0_e15"]["plain"]["animal_gt_fates_pooled"] == {"FOUND_RIGHT_CLASS": n}
        assert abs(g["routes"]["R0"]["animal_pooled"] - 1.0) < 1e-9 and g["routes"]["R0"]["animal_pooled"] > g["tiled"]["animal_pooled"]
        dec = route_rule(res)
        train = {"images": val["images"], "annotations": val["annotations"], "categories": val["categories"]}
        cen = census(FU, M1, hr_p2, train, val, ids[0], ids[5])
        assert cen["val_off0_off5"]["ALL"]["boxes"] == n and cen["train"]["ALL"]["small"] == len(val["images"])
        assert abs(cen["train"]["ALL"]["net_full_px_q10_50_90"][1] - math.sqrt(12 * 640 / 1920 * 8 * 640 / 1080)) < 1e-9
        md = report(res, cen, dec, avids)
        out = Path(td) / "rep"
        write_outputs(out, md, {"x": 1})
        try:
            write_outputs(out, md, {"x": 1})
            raise AssertionError("overwrite not refused")
        except SystemExit:
            pass
        print(f"  [ok] end to end (2 synthetic models, offsets 0 + 5, {n} frames): fates, routing to ANIMAL AP 1.0 with "
              f"the other classes asserted unchanged, census sizes, report, refusal to overwrite; decision {dec}")
    print("SELFTEST PASS")


def write_outputs(out, md, js):
    out = Path(out)
    if (out / "g4c_report.md").exists():
        raise SystemExit(f"[REFUSED] {out / 'g4c_report.md'} exists (results are kept; rename the folder to rerun)")
    out.mkdir(parents=True, exist_ok=True)
    (out / "g4c_report.json").write_text(json.dumps(js, indent=1, default=str), encoding="utf-8")
    (out / "g4c_report.md").write_text(md, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["selftest", "run"], required=True)
    ap.add_argument("--m1-dir", default=f"{D}/R0_frozen_baseline")
    a = ap.parse_args()
    if a.mode == "selftest":
        return selftest(a)
    FU = load_fu()
    hr_p2, M1, MV = FU.import_modules(a.m1_dir)
    if (OUT_DIR / "g4c_report.md").exists():
        raise SystemExit(f"[REFUSED] {OUT_DIR / 'g4c_report.md'} exists")
    for offs in CACHES.values():
        for p in offs.values():
            if not Path(p).exists():
                raise SystemExit(f"[REFUSED] missing {p}")
    print(f"[{VERSION}] loading VAL / TRAIN annotations ...")
    val = json.loads(Path(FU.VAL_JSON).read_text(encoding="utf-8"))
    train = json.loads(Path(FU.TRAIN_JSON).read_text(encoding="utf-8"))
    res, shas, avids, ids = analyse(FU, MV, M1, hr_p2, val, CACHES)
    cen = census(FU, M1, hr_p2, train, val, ids[0], ids[5])
    dec = route_rule(res)
    md = report(res, cen, dec, avids)
    write_outputs(OUT_DIR, md, {"version": VERSION, "rule": RULE, "decision": dec, "input_sha256": shas,
                                "animal_videos": avids, "results": res, "census": cen})
    print(md)
    print(f"  wrote {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
