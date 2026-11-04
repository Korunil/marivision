r"""
G4F follow-up (CPU only, VAL + TRAIN annotations, existing tile caches; no GPU, no TEST).

Location: D:/MARIVISION/H0_workbench/eval_tools/g4f_followup.py   (run from that folder, like m2_evaluate.py)

  --mode selftest   synthetic data; needs m1_r0_tiled_inference.py (M1.0) and m1_1_merge_variants.py (M1.1)
                    in --m1-dir and the HR-P2 pipeline in this folder, exactly as m2_evaluate.py does
  --mode animal     A) why ANIMAL collapsed:
                       1. GT census: ANIMAL boxes per video in TRAIN, full VAL, offset-0 and offset-5 frames
                       2. tile-training exposure per class on TRAIN (M2 rule: a box is a target in a Stage C window
                          only if >= 50 % of it is visible; 0-50 % visible = dropped, i.e. trained as background),
                          object size at the network input for full frames vs tiles
                       3. per model and inference mode, the fate of every offset-0 ANIMAL GT box (the M2 class-
                          diagnostic rule): FOUND_RIGHT_CLASS / FOUND_AS_<X> / SEEN_LOW_SCORE / MISSED, the
                          ANIMAL-class scores on boxes that overlap ANIMAL GT, ANIMAL predictions >= 0.05, ANIMAL AP30
  --mode off5       B) the pre-declared offset-5 replication (reported only): G4F epoch 15 tiled vs R0 epoch 15 plain
                       on the 432 offset-5 frames, pooled and per video, pycocotools macro AP30.
                       Parity gates: G4F plain / tiled must equal curve.json epoch 15 (same scorer, 1e-6);
                       R0e15 plain / tiled must match the M1.1 offset-5 report (0.3578 / 0.3459, HR-P2 evaluator, 2e-3).
Outputs: g4f_reports/*.md + *.json (dataset-derived; stay on the workstation).
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import gzip
import hashlib
import io
import json
import math
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

VERSION = "G4FU.0"
M1_SHA = "73044625fb5b477f857018ba1ace3c0dcfe32f08d2b27d0beccd8d333153a388"
MV_SHA = "018f70d4c45b140a9e44db16e925b3d9479f32bff27a8ccebdcf940b93b6834e"
CLASSES = ("ANIMAL", "BOAT", "MARINE_MARKER", "OBJECT", "LEISURE_VEHICLE", "SAILING_VESSEL", "SHIP")
ANIMAL = 1
S_MIN = 0.05
MIN_VISIBLE = 0.5
STAGE_B, STAGE_C = "STAGE_B_FULL_CONTEXT", "STAGE_C_2X_CROP"

D = "D:/MARIVISION"
VAL_JSON = f"{D}/seaclips-val.json"
TRAIN_JSON = f"{D}/seaclips-train.json"
M1D = f"{D}/R0_frozen_baseline/m1_1_tile_cache"
PLT = f"{D}/G4_pilot/tile_cache"
G4F = f"{D}/G4_full/run"
CACHES_OFF0 = {   # name -> (path, what it is)
    "G4F_e15": (f"{G4F}/tile_cache/r0_tile_cache_g4f_e15_stride10_offset0.json.gz", "ConvNeXt-B + tiles, 15 ep, fp32"),
    "R0_e15": (f"{M1D}/r0_tile_cache_epoch15_stride10_offset0.json.gz", "R50, 15 ep (R0)"),
    "R0_e04": (f"{M1D}/r0_tile_cache_epoch04_stride10_offset0.json.gz", "R50, 4 ep (R0, VAL-selected)"),
    "M2_armF": (f"{M1D}/r0_tile_cache_armF_seed42_stride10_offset0.json.gz", "R0e15 + 1 ep full frames"),
    "M2_armT": (f"{M1D}/r0_tile_cache_armT_seed42_stride10_offset0.json.gz", "R0e15 + 1 ep with tiles"),
    "P_R50_fp32": (f"{PLT}/r0_tile_cache_g4_R50_fp32_at100_stride10_offset0.json.gz", "pilot R50 fp32, no tiles"),
    "P_R50_bf16": (f"{PLT}/r0_tile_cache_g4_R50_bf16_at100_stride10_offset0.json.gz", "pilot R50 bf16, no tiles"),
    "P_CNB": (f"{PLT}/r0_tile_cache_g4_CNB_bf16_at100_stride10_offset0.json.gz", "pilot ConvNeXt-B, no tiles"),
    "P_CNBT": (f"{PLT}/r0_tile_cache_g4_CNBT_bf16_at100_stride10_offset0.json.gz", "pilot ConvNeXt-B + tiles"),
}
OFF5 = {"G4F_e15": f"{G4F}/tile_cache/r0_tile_cache_g4f_e15_stride10_offset5.json.gz",
        "R0_e15": f"{M1D}/r0_tile_cache_epoch15_stride10_offset5.json.gz"}
CURVE = f"{G4F}/curve.json"
R0E15_OFF5_REF = {"plain": 0.3578, "tiled": 0.3459}    # m1_1_variants_epoch15_stride10_offset5.md (V0)


# =============================================================================
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def import_modules(m1_dir, check=True):
    here = Path(__file__).resolve().parent
    for f, s in (("m1_r0_tiled_inference.py", M1_SHA), ("m1_1_merge_variants.py", MV_SHA)):
        p = next((d / f for d in (Path(m1_dir), here) if (d / f).exists()), None)
        if p is None:
            raise SystemExit(f"[REFUSED] {f} not found in {m1_dir} or {here}")
        if check and sha256_file(p) != s:
            raise SystemExit(f"[REFUSED] {p} sha {sha256_file(p)} != expected {s}")
    sys.path.insert(0, str(here))
    sys.path.insert(1, str(Path(m1_dir).resolve()))
    import marivision_hr_p2_pipeline as hr_p2
    import m1_r0_tiled_inference as M1
    import m1_1_merge_variants as MV
    return hr_p2, M1, MV


def iou(a, b):
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0


def gt_outcomes(preds, gts):
    """m2_class_diagnostic.py rule (copied unchanged): per GT box, its fate at score >= 0.05 / IoU >= 0.30."""
    strong = sorted([p for p in preds if p["score"] >= S_MIN], key=lambda d: -d["score"])
    matched_gt = {}
    for pi, p in enumerate(strong):
        best, bj = 0.30, -1
        for j, g in enumerate(gts):
            if j in matched_gt or g["category_id"] != p["category_id"]:
                continue
            v = iou(p["bbox"], g["bbox"])
            if v >= best:
                best, bj = v, j
        if bj >= 0:
            matched_gt[bj] = pi
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


def quart(v):
    if not v:
        return "–"
    v = sorted(v)
    q = lambda f: v[min(len(v) - 1, int(f * (len(v) - 1)))]
    return f"{q(0.25):.3f} / {q(0.5):.3f} / {q(0.75):.3f} / max {v[-1]:.3f}"


def video_key(im):
    return int(im.get("video_id", 0))


# =============================================================================
# predictions from a tile cache (exactly as m2_evaluate.preds_from_cache)
# =============================================================================
def load_cache(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def preds_from_cache(MV, M1, hr_p2, cache):
    l2c = {int(k): int(v) for k, v in cache["meta"]["label_to_cat_id"].items()}
    plain, tiled = defaultdict(list), defaultdict(list)
    for iid, fr in cache["frames"].items():
        tiles = MV.frame_tiles(M1, hr_p2, fr)
        b = [t for t in tiles if t[0].stage == MV.STAGE_B][0][1]
        for d in M1.plain_dets_native([x[0] for x in b], [x[1] for x in b], [x[2:6] for x in b], l2c):
            plain[int(iid)].append({"image_id": int(iid), "category_id": int(d["category_id"]),
                                    "bbox": [float(v) for v in d["bbox"]], "score": float(d["score"])})
        for d in MV.build_variant("V0", M1, hr_p2, l2c, fr):
            tiled[int(iid)].append({"image_id": int(iid), "category_id": int(d["category_id"]),
                                    "bbox": [float(v) for v in d["bbox"]], "score": float(d["score"])})
    return plain, tiled


def coco_eval(val, img_ids, dets_by_img, groups=None):
    """pycocotools AP@0.3, maxDets 100: macro (= stats[0]), per class, and per group via masked re-accumulation."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    ids = set(img_ids)
    ds = {"images": [im for im in val["images"] if im["id"] in ids],
          "annotations": [a for a in val["annotations"] if a["image_id"] in ids], "categories": val["categories"]}
    dets = [d for i in ids for d in dets_by_img.get(i, [])]
    names = {c["id"]: c["name"] for c in val["categories"]}
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        gt = COCO()
        gt.dataset = copy.deepcopy(ds)
        gt.createIndex()
        if not dets:
            return {"map": 0.0, "per_class": {}, "per_group": {}}
        E = COCOeval(gt, gt.loadRes(copy.deepcopy(dets)), "bbox")
        E.params.iouThrs = np.array([0.30])
        E.evaluate()
        E.accumulate()
        E.summarize()
        out = {"map": float(E.stats[0]), "per_class": {}, "per_group": {}, "per_group_class": {}}
        for k, cid in enumerate(E.params.catIds):
            p = E.eval["precision"][0, :, k, 0, 2]
            p = p[p > -1]
            if p.size:
                out["per_class"][names[cid]] = float(p.mean())
        full = E.evalImgs
        I0 = len(E._paramsEval.imgIds)
        for g, gids in (groups or {}).items():
            keep = np.array([i in gids for i in E._paramsEval.imgIds])
            E.evalImgs = [e if keep[n % I0] else None for n, e in enumerate(full)]
            E.accumulate()
            pr = E.eval["precision"][0, :, :, 0, 2]
            v = pr[pr > -1]
            out["per_group"][g] = float(v.mean()) if v.size else None
        E.evalImgs = full
    return out


# =============================================================================
# TRAIN tile exposure (M2 rule)
# =============================================================================
def window_visibility(box, win):
    x, y, w, h = box
    x0, y0, x1, y1 = win
    iw = max(0.0, min(x + w, x1) - max(x, x0))
    ih = max(0.0, min(y + h, y1) - max(y, y0))
    return (iw * ih) / (w * h) if w > 0 and h > 0 else 0.0


def train_exposure(train, M1, hr_p2):
    imgs = {im["id"]: im for im in train["images"]}
    wcache = {}
    st = defaultdict(lambda: {"boxes": 0, "kept_windows": 0, "dropped_windows": 0, "never_kept": 0,
                              "net_full": [], "net_tile": []})
    for a in train["annotations"]:
        im = imgs[a["image_id"]]
        W, H = int(im["width"]), int(im["height"])
        if (W, H) not in wcache:
            wcache[(W, H)] = [(c.x0_int, c.y0_int, c.x1_int, c.y1_int) for c in M1.build_tiles(hr_p2, W, H)
                              if c.stage == STAGE_C]
        x, y, w, h = [float(v) for v in a["bbox"]]
        if w <= 0 or h <= 0:
            continue
        s = st[int(a["category_id"])]
        s["boxes"] += 1
        s["net_full"].append(math.sqrt(w * 640 / W * h * 640 / H))
        kept = 0
        for win in wcache[(W, H)]:
            v = window_visibility([x, y, w, h], win)
            if v >= MIN_VISIBLE:
                kept += 1
                s["net_tile"].append(math.sqrt(min(w, win[2] - win[0]) * 640 / (win[2] - win[0])
                                               * min(h, win[3] - win[1]) * 640 / (win[3] - win[1])))
            elif v > 0:
                s["dropped_windows"] += 1
        s["kept_windows"] += kept
        s["never_kept"] += int(kept == 0)
    rows = {}
    for c, s in sorted(st.items()):
        n = s["boxes"]
        rows[CLASSES[c - 1]] = {
            "train_boxes": n, "share_full_frame": None,
            "kept_windows_per_box": s["kept_windows"] / n, "never_kept_frac": s["never_kept"] / n,
            "dropped_as_background_per_box": s["dropped_windows"] / n,
            "net_px_full_median": float(np.median(s["net_full"])),
            "net_px_tile_median": float(np.median(s["net_tile"])) if s["net_tile"] else None}
    tot_full = sum(r["train_boxes"] for r in rows.values())
    tot_tile = sum(r["train_boxes"] * r["kept_windows_per_box"] for r in rows.values())
    for r in rows.values():
        r["share_full_frame"] = r["train_boxes"] / tot_full
        r["share_tile_targets"] = r["train_boxes"] * r["kept_windows_per_box"] / tot_tile if tot_tile else None
    return rows


def census(train, val, M1, off0_ids, off5_ids):
    def per_video(ds, ids=None):
        im = {i["id"]: i for i in ds["images"]}
        c = Counter()
        for a in ds["annotations"]:
            if a["category_id"] == ANIMAL and (ids is None or a["image_id"] in ids):
                c[video_key(im[a["image_id"]])] += 1
        return dict(sorted(c.items()))
    return {"train": per_video(train), "val_full": per_video(val),
            "val_off0": per_video(val, set(off0_ids)), "val_off5": per_video(val, set(off5_ids))}


# =============================================================================
# mode animal
# =============================================================================
def animal_diag(val, MV, M1, hr_p2, caches, log=print):
    ids0 = M1.eval_subset_ids(val["images"], 10, 0)
    gts = defaultdict(list)
    for g in val["annotations"]:
        if int(g.get("iscrowd", 0)) == 0 and g["image_id"] in set(ids0):
            gts[g["image_id"]].append({"category_id": int(g["category_id"]), "bbox": [float(x) for x in g["bbox"]]})
    res = {}
    for name, (path, what) in caches.items():
        if not Path(path).exists():
            res[name] = {"missing": path, "what": what}
            log(f"  [skip] {name}: {path} not found")
            continue
        cache = load_cache(path)
        m = cache["meta"]
        if (int(m["val_frame_stride"]), int(m["val_frame_offset"])) != (10, 0):
            raise SystemExit(f"[REFUSED] {name} is not stride 10 / offset 0")
        plain, tiled = preds_from_cache(MV, M1, hr_p2, cache)
        del cache
        r = {"what": what, "cache_sha256": sha256_file(path)}
        for mode, P in (("plain", plain), ("tiled", tiled)):
            fates, ascore, overlap_scores = Counter(), [], []
            for i in ids0:
                fates.update(o for c, o in gt_outcomes(P.get(i, []), gts[i]) if c == ANIMAL)
                ascore += [d["score"] for d in P.get(i, []) if d["category_id"] == ANIMAL and d["score"] >= S_MIN]
                for g in gts[i]:
                    if g["category_id"] == ANIMAL:
                        s = [d["score"] for d in P.get(i, []) if d["category_id"] == ANIMAL and iou(d["bbox"], g["bbox"]) >= 0.3]
                        overlap_scores.append(max(s) if s else 0.0)
            ev = coco_eval(val, ids0, P)
            r[mode] = {"animal_fates": dict(fates), "animal_preds_ge_005": len(ascore),
                       "animal_pred_score_quartiles": quart(ascore),
                       "best_animal_score_on_animal_gt": quart(overlap_scores),
                       "animal_gt_with_any_animal_pred_in_top100": sum(1 for s in overlap_scores if s > 0),
                       "animal_ap30": ev["per_class"].get("ANIMAL"), "macro_ap30": ev["map"]}
        res[name] = r
        log(f"  {name}: ANIMAL AP plain {r['plain']['animal_ap30']:.4f} tiled {r['tiled']['animal_ap30']:.4f} | "
            f"fates tiled {r['tiled']['animal_fates']}")
    return res, ids0


def animal_md(cen, expo, res):
    L = [f"# ANIMAL diagnosis ({VERSION})", "", "## 1. ANIMAL GT per video", "",
         "| split | " + " | ".join(f"v{v:02d}" for v in sorted(set().union(*[set(d) for d in cen.values()]))) + " |"]
    vids = sorted(set().union(*[set(d) for d in cen.values()]))
    L.append("|---|" + "---|" * len(vids))
    for k, d in cen.items():
        L.append(f"| {k} (total {sum(d.values())}) | " + " | ".join(str(d.get(v, 0)) for v in vids) + " |")
    L += ["", "## 2. TRAIN tile-training exposure per class (M2 rule, 16 Stage C windows)", "",
          "| class | TRAIN boxes | share full frames | share tile targets | kept windows / box | never kept in a window | "
          "dropped-as-background / box | median size at net input, full frame px | median size, tile px |",
          "|---|---|---|---|---|---|---|---|---|"]
    for c, r in expo.items():
        L.append(f"| {c} | {r['train_boxes']} | {r['share_full_frame']:.3f} | {r['share_tile_targets']:.3f} | "
                 f"{r['kept_windows_per_box']:.2f} | {r['never_kept_frac']:.3f} | {r['dropped_as_background_per_box']:.2f} | "
                 f"{r['net_px_full_median']:.1f} | {(r['net_px_tile_median'] or 0):.1f} |")
    L += ["", "## 3. Fate of the offset-0 ANIMAL GT boxes per model (score >= 0.05, IoU >= 0.30)", ""]
    keys = sorted({k for r in res.values() if "plain" in r for m in ("plain", "tiled") for k in r[m]["animal_fates"]})
    L += ["| model | mode | ANIMAL AP30 | " + " | ".join(keys) + " | ANIMAL preds >= 0.05 (score q25/50/75/max) | "
          "best ANIMAL score on ANIMAL GT (q25/50/75/max) |", "|---|---|---|" + "---|" * len(keys) + "---|---|"]
    for n, r in res.items():
        if "plain" not in r:
            L.append(f"| {n} | missing | | " + " | " * len(keys) + " | |")
            continue
        for m in ("plain", "tiled"):
            x = r[m]
            L.append(f"| {n} ({r['what']}) | {m} | {x['animal_ap30']:.4f} | " +
                     " | ".join(str(x["animal_fates"].get(k, 0)) for k in keys) +
                     f" | {x['animal_preds_ge_005']}: {x['animal_pred_score_quartiles']} | {x['best_animal_score_on_animal_gt']} |")
    return "\n".join(L) + "\n"


# =============================================================================
# mode off5
# =============================================================================
def off5_replication(val, MV, M1, hr_p2, paths, curve_e15, r0_ref, log=print):
    ids5 = M1.eval_subset_ids(val["images"], 10, 5)
    imgs = {im["id"]: im for im in val["images"]}
    groups = defaultdict(set)
    for i in ids5:
        groups[f"v{video_key(imgs[i]):02d}"].add(i)
    out = {}
    for name, path in paths.items():
        cache = load_cache(path)
        m = cache["meta"]
        if (int(m["val_frame_stride"]), int(m["val_frame_offset"])) != (10, 5):
            raise SystemExit(f"[REFUSED] {name} is not stride 10 / offset 5")
        if set(int(k) for k in cache["frames"]) != set(ids5):
            raise SystemExit(f"[REFUSED] {name} does not cover the 432 offset-5 frames")
        plain, tiled = preds_from_cache(MV, M1, hr_p2, cache)
        del cache
        out[name] = {k: coco_eval(val, ids5, P, groups) for k, P in (("plain", plain), ("tiled", tiled))}
        out[name]["cache_sha256"] = sha256_file(path)
        log(f"  {name}: plain {out[name]['plain']['map']:.4f} tiled {out[name]['tiled']['map']:.4f}")
    gate = {}
    if curve_e15 is not None:
        gate["G4F_plain_equals_curve"] = abs(out["G4F_e15"]["plain"]["map"] - curve_e15["plain"]) < 1e-6
        gate["G4F_tiled_equals_curve"] = abs(out["G4F_e15"]["tiled"]["map"] - curve_e15["tiled"]) < 1e-6
    gate["R0e15_plain_matches_M1.1"] = abs(out["R0_e15"]["plain"]["map"] - r0_ref["plain"]) < 2e-3
    gate["R0e15_tiled_matches_M1.1"] = abs(out["R0_e15"]["tiled"]["map"] - r0_ref["tiled"]) < 2e-3
    a, b = out["G4F_e15"]["tiled"], out["R0_e15"]["plain"]
    pv = {v: (a["per_group"][v] - b["per_group"][v]) if a["per_group"][v] is not None and b["per_group"][v] is not None else None
          for v in sorted(groups)}
    summ = {"delta_pooled": a["map"] - b["map"], "per_video_delta": pv,
            "videos_positive": sum(1 for x in pv.values() if x is not None and x > 0),
            "videos": sum(1 for x in pv.values() if x is not None)}
    return out, gate, summ, sorted(groups)


def off5_md(out, gate, summ, vids):
    a, b = out["G4F_e15"], out["R0_e15"]
    L = [f"# Offset-5 replication ({VERSION}) — reported only, not a decision input", "",
         f"Parity gates: {json.dumps(gate)}", "",
         f"**G4F e15 tiled {a['tiled']['map']:.4f} vs R0 e15 plain {b['plain']['map']:.4f}: Δ {summ['delta_pooled']:+.4f}, "
         f"{summ['videos_positive']}/{summ['videos']} videos positive**", "",
         "| video | G4F e15 tiled | R0 e15 plain | Δ |", "|---|---|---|---|"]
    for v in vids:
        x, y, dv = a["tiled"]["per_group"][v], b["plain"]["per_group"][v], summ["per_video_delta"][v]
        f = lambda z, sign=False: "–" if z is None else (f"{z:+.4f}" if sign else f"{z:.4f}")
        L.append(f"| {v} | {f(x)} | {f(y)} | {f(dv, True)} |")
    L += ["", "| class | G4F e15 tiled | G4F e15 plain | R0 e15 plain | R0 e15 tiled |", "|---|---|---|---|---|"]
    for c in CLASSES:
        if c in a["tiled"]["per_class"]:
            L.append(f"| {c} | {a['tiled']['per_class'][c]:.4f} | {a['plain']['per_class'].get(c, 0):.4f} | "
                     f"{b['plain']['per_class'].get(c, 0):.4f} | {b['tiled']['per_class'].get(c, 0):.4f} |")
    L.append(f"| macro | {a['tiled']['map']:.4f} | {a['plain']['map']:.4f} | {b['plain']['map']:.4f} | {b['tiled']['map']:.4f} |")
    return "\n".join(L) + "\n"


# =============================================================================
# selftest
# =============================================================================
def _synthetic(M1, hr_p2, offset, n_vid=3, n_frames=40, seed=0):
    """VAL-schema json + a tile cache: ANIMAL found as ANIMAL by the full-frame tile in video 1, as BOAT elsewhere."""
    rng = np.random.default_rng(seed)
    cats = [{"id": i + 1, "name": n} for i, n in enumerate(CLASSES)]
    images, anns, frames = [], [], {}
    W, H = 1920, 1080
    crops = M1.build_tiles(hr_p2, W, H)
    l2c = {str(i): i + 1 for i in range(7)}
    aid = 1
    for v in range(1, n_vid + 1):
        for f in range(n_frames):
            iid = len(images) + 1
            images.append({"id": iid, "file_name": f"video_{v:02d}/{f + 1:06d}.png", "width": W, "height": H,
                           "video_id": v, "frame_id": f})
            animal = [100.0 + f, 200.0, 12.0, 8.0]
            boat = [700.0, 500.0, 60.0, 30.0]
            anns += [{"id": aid, "image_id": iid, "category_id": 1, "bbox": animal, "area": 96.0, "iscrowd": 0},
                     {"id": aid + 1, "image_id": iid, "category_id": 2, "bbox": boat, "area": 1800.0, "iscrowd": 0}]
            aid += 2
            tiles = []
            for c in crops:
                dets = []
                if c.stage == STAGE_B:
                    xa, ya, wa, ha = animal
                    lab = 0 if v == 1 else 1           # label 0 -> cat 1 (ANIMAL), label 1 -> cat 2 (BOAT)
                    dets.append([0.8, lab, xa, ya, xa + wa, ya + ha])
                    dets.append([0.9, 1, boat[0], boat[1], boat[0] + boat[2], boat[1] + boat[3]])
                    dets.append([float(rng.uniform(0.01, 0.04)), 0, 1500.0, 900.0, 1510.0, 910.0])
                tiles.append([c.stage, c.x0_int, c.y0_int, c.x1_int, c.y1_int, dets])
            frames[str(iid)] = {"W": W, "H": H, "tiles": tiles}
    val = {"images": images, "annotations": anns, "categories": cats}
    ids = M1.eval_subset_ids(images, 10, offset)
    cache = {"meta": {"label_to_cat_id": l2c, "val_frame_stride": 10, "val_frame_offset": offset},
             "frames": {k: v for k, v in frames.items() if int(k) in set(ids)}}
    return val, cache


def selftest(a):
    import tempfile
    print(f"[selftest] {VERSION}")
    hr_p2, M1, MV = import_modules(a.m1_dir)
    gts = [{"category_id": 1, "bbox": [0, 0, 10, 10]}, {"category_id": 4, "bbox": [100, 100, 10, 10]},
           {"category_id": 2, "bbox": [200, 200, 10, 10]}, {"category_id": 6, "bbox": [300, 300, 10, 10]}]
    preds = [{"category_id": 1, "score": 0.5, "bbox": [0, 0, 10, 10]},
             {"category_id": 3, "score": 0.4, "bbox": [100, 100, 10, 10]},
             {"category_id": 2, "score": 0.02, "bbox": [200, 200, 10, 10]}]
    assert gt_outcomes(preds, gts) == [(1, "FOUND_RIGHT_CLASS"), (4, "FOUND_AS_MARINE_MARKER"),
                                       (2, "SEEN_LOW_SCORE"), (6, "MISSED")]
    print("  [ok] GT fate rule = m2_class_diagnostic (right class / as other class / low score / missed)")
    wins = [(c.x0_int, c.y0_int, c.x1_int, c.y1_int) for c in M1.build_tiles(hr_p2, 1920, 1080) if c.stage == STAGE_C]
    assert len(wins) == 16
    w0 = wins[0]
    assert window_visibility([w0[0] + 10, w0[1] + 10, 20, 20], w0) == 1.0
    assert abs(window_visibility([w0[2] - 8, w0[1] + 10, 20, 10], w0) - 0.4) < 1e-12
    train = {"images": [{"id": 1, "width": 1920, "height": 1080}],
             "annotations": [{"image_id": 1, "category_id": 1, "bbox": [w0[0] + 10, w0[1] + 10, 20, 20]},
                             {"image_id": 1, "category_id": 2, "bbox": [w0[2] - 8, w0[1] + 10, 20, 10]}],
             "categories": []}
    ex = train_exposure(train, M1, hr_p2)
    assert ex["ANIMAL"]["never_kept_frac"] == 0.0 and ex["ANIMAL"]["kept_windows_per_box"] >= 1
    assert ex["BOAT"]["dropped_as_background_per_box"] >= 1
    print("  [ok] Stage C windows (16), visibility, kept / dropped-as-background counting")
    val, cache = _synthetic(M1, hr_p2, 0)
    with tempfile.TemporaryDirectory() as td:
        p0 = Path(td) / "c0.json.gz"
        with gzip.open(p0, "wt", encoding="utf-8") as f:
            json.dump(cache, f)
        res, ids0 = animal_diag(val, MV, M1, hr_p2, {"X": (str(p0), "synthetic"), "Y": (str(Path(td) / "none"), "absent")},
                                log=lambda *x: None)
        fates = res["X"]["plain"]["animal_fates"]
        n1 = sum(1 for i in ids0 if 1 <= i <= 40)
        assert fates == {"FOUND_RIGHT_CLASS": n1, "FOUND_AS_BOAT": len(ids0) - n1}, fates
        assert res["X"]["plain"]["animal_preds_ge_005"] == n1 and "missing" in res["Y"]
        assert res["X"]["tiled"]["animal_fates"].get("FOUND_RIGHT_CLASS") == n1
        cen = census(val, val, M1, ids0, M1.eval_subset_ids(val["images"], 10, 5))
        animal_md(cen, ex, res)
        print(f"  [ok] animal mode end-to-end: fates {fates}; missing caches are reported, not fatal")
        # off5: per-video masked values equal separate evaluations; parity gate logic
        v5, c5 = _synthetic(M1, hr_p2, 5)
        p5 = Path(td) / "c5.json.gz"
        with gzip.open(p5, "wt", encoding="utf-8") as f:
            json.dump(c5, f)
        out, gate, summ, vids = off5_replication(v5, MV, M1, hr_p2, {"G4F_e15": str(p5), "R0_e15": str(p5)}, None,
                                                 {"plain": 0.0, "tiled": 0.0}, log=lambda *x: None)
        plain, _t = preds_from_cache(MV, M1, hr_p2, c5)
        ids5 = M1.eval_subset_ids(v5["images"], 10, 5)
        for v in vids:
            sep = coco_eval(v5, [i for i in ids5 if f"v{video_key(v5['images'][i - 1]):02d}" == v], plain)
            assert abs(sep["map"] - out["G4F_e15"]["plain"]["per_group"][v]) < 1e-12, v
        assert summ["delta_pooled"] == out["G4F_e15"]["tiled"]["map"] - out["R0_e15"]["plain"]["map"]
        assert gate["R0e15_plain_matches_M1.1"] is False
        off5_md(out, gate, summ, vids)
        print("  [ok] off5 mode: per-video masked AP = separate per-video evaluation; parity gate fires on a mismatch")
    print("SELFTEST PASS")


# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["selftest", "animal", "off5"], required=True)
    ap.add_argument("--m1-dir", default=f"{D}/R0_frozen_baseline")
    ap.add_argument("--report-dir", default="g4f_reports")
    a = ap.parse_args()
    if a.mode == "selftest":
        return selftest(a)
    hr_p2, M1, MV = import_modules(a.m1_dir)
    val = json.loads(Path(VAL_JSON).read_text(encoding="utf-8"))
    rd = Path(a.report_dir)
    rd.mkdir(parents=True, exist_ok=True)
    if a.mode == "animal":
        train = json.loads(Path(TRAIN_JSON).read_text(encoding="utf-8"))
        print(f"[{VERSION}] TRAIN exposure ...")
        expo = train_exposure(train, M1, hr_p2)
        cen = census(train, val, M1, M1.eval_subset_ids(val["images"], 10, 0), M1.eval_subset_ids(val["images"], 10, 5))
        print(f"[{VERSION}] model fates on offset-0 ANIMAL GT ...")
        res, _ = animal_diag(val, MV, M1, hr_p2, CACHES_OFF0)
        md = animal_md(cen, expo, res)
        (rd / "g4f_animal_diag.md").write_text(md, encoding="utf-8")
        (rd / "g4f_animal_diag.json").write_text(json.dumps({"version": VERSION, "census": cen, "exposure": expo,
                                                             "models": res}, indent=1), encoding="utf-8")
        print(md)
    else:
        curve = json.loads(Path(CURVE).read_text(encoding="utf-8"))
        e15 = next(r for r in curve if r["epoch"] == 15)
        out, gate, summ, vids = off5_replication(val, MV, M1, hr_p2, OFF5, e15, R0E15_OFF5_REF)
        md = off5_md(out, gate, summ, vids)
        (rd / "g4f_off5_replication.md").write_text(md, encoding="utf-8")
        (rd / "g4f_off5_replication.json").write_text(json.dumps({"version": VERSION, "gates": gate, "summary": summ,
                                                                  "results": out}, indent=1, default=str), encoding="utf-8")
        print(md)
        if not all(gate.values()):
            print("PARITY GATE FAILED -- do not interpret; send this output to Claude")
    print(f"  wrote {rd.resolve()}")


if __name__ == "__main__":
    main()
