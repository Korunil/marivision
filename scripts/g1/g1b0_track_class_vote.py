"""
G1b-0: track-level class vote on R0 (epoch04) every-frame VAL predictions.

Location: D:/MARIVISION/H0_workbench/eval_tools/g1b0_track_class_vote.py
CPU only. Reads the existing prediction cache; no model, no images, no TEST.

Question: is R0's wrong-class error on unseen videos *temporal flicker*
(the right class wins in neighbouring frames of the same object) or
*persistent* (the object is misread consistently)?  If flicker, a causal
track vote fixes it for free.  If persistent, it does not, and the class
expert (G2) has to bring new information.

Modes
  --mode selftest   synthetic controls, no files needed
  --mode run        real cache + VAL json, writes g1b0_reports/

Frozen hyper-parameters are in HP below; they are NOT tuned on VAL.
"""

import argparse
import contextlib
import copy
import hashlib
import io
import json
import math
import os
import re
import sys
import warnings
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

VERSION = "G1b-0.0"

CACHE_M0 = Path("D:/MARIVISION/H0_workbench/eval_tools/cache_dt_m0_g1.json")
VAL_JSON = Path("D:/MARIVISION/seaclips-val.json")
M0_SIDECAR = Path("D:/MARIVISION/R0_frozen_baseline/checkpoint_best_r0_epoch04.strata_metrics.json")
OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools/g1b0_reports")

EXPECTED_VIDEOS = ["video_03", "video_04", "video_05", "video_06", "video_08", "video_09"]
EXPECTED_IMAGES = 2403
BASELINE_MAP30 = 0.4109          # certified H0 parity value (printed 41.09%)
BASELINE_TOL = 6e-5              # covers the 2-decimal rounding of 41.09%

HP = {
    "topk": 100,            # detections kept per frame (COCO maxDets)
    "window": 15,           # causal window, frames of track history incl. current
    "max_gap": 3,           # a track may skip up to 2 frames (frame diff <= 3)
    "gate": 1.0,            # centre distance / sqrt(w*h of track box)
    "min_scale_px": 4.0,    # floor on sqrt(w*h) in the gate
    "area_ratio": 4.0,      # area(entity)/area(track) within [1/4, 4]
    "link_min_obj": 0.05,   # entities below this are passed through unchanged
    "lambda_blend": 0.5,    # arm A2
}

PASS_RULE = {
    "primary_arm": "A1_causal",
    "min_delta_map": 0.01,
    "min_videos_positive": 4,
    "min_worst_video_delta": -0.01,
}


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def video_of(file_name):
    for part in str(file_name).replace("\\", "/").split("/"):
        if part.startswith("video_"):
            return part
    return "unknown"


def frame_of(file_name):
    base = str(file_name).replace("\\", "/").split("/")[-1]
    m = re.findall(r"\d+", base)
    if not m:
        raise ValueError(f"no frame number in {file_name}")
    return int(m[-1])


# --------------------------------------------------------------------------
# Entities: one per distinct query box in a frame, with a class-score vector
# --------------------------------------------------------------------------
def build_entities(dets, cat_ids):
    """dets: list of {image_id, category_id, bbox, score}.
    Returns {image_id: [entity]} ; entity = {box, s(np.array over cat_ids)}.
    Same bbox => same Deformable-DETR query.  A repeated class inside one
    bbox (should not happen) starts a separate entity so the rebuild stays
    an exact identity."""
    cidx = {c: i for i, c in enumerate(cat_ids)}
    per_img = defaultdict(dict)
    n_split = 0
    for d in dets:
        key = tuple(float(v) for v in d["bbox"])
        slot = per_img[d["image_id"]]
        k = (key, 0)
        while k in slot and slot[k]["s"][cidx[d["category_id"]]] > 0:
            k = (key, k[1] + 1)
            n_split += 1
        if k not in slot:
            slot[k] = {"box": list(key), "s": np.zeros(len(cat_ids))}
        slot[k]["s"][cidx[d["category_id"]]] = float(d["score"])
    out = {}
    for img, slot in per_img.items():
        ents = list(slot.values())
        for e in ents:
            e["obj"] = float(e["s"].max())
            e["p"] = e["s"] / e["obj"] if e["obj"] > 0 else e["s"]
        ents.sort(key=lambda e: -e["obj"])
        out[img] = ents
    return out, n_split


def emit(ent_by_img, cat_ids, score_key, topk):
    dets = []
    for img, ents in ent_by_img.items():
        cand = []
        for e in ents:
            sv = e[score_key]
            for i in np.nonzero(sv > 0)[0]:
                cand.append((float(sv[i]), cat_ids[i], e["box"]))
        cand.sort(key=lambda t: -t[0])
        for sc, c, b in cand[:topk]:
            dets.append({"image_id": img, "category_id": c, "bbox": list(b), "score": sc})
    return dets


# --------------------------------------------------------------------------
# Causal linking and voting
# --------------------------------------------------------------------------
def _cost(e, tr, hp):
    bx, by, bw, bh = e["box"]
    tx, ty, tw, th = tr["box"]
    scale = max(math.sqrt(max(tw, 0) * max(th, 0)), hp["min_scale_px"])
    ar = (bw * bh + 1e-9) / (tw * th + 1e-9)
    if ar > hp["area_ratio"] or ar < 1.0 / hp["area_ratio"]:
        return None
    d = math.hypot((bx + bw / 2) - (tx + tw / 2), (by + bh / 2) - (ty + th / 2)) / scale
    return d if d <= hp["gate"] else None


def link_and_vote(ent_by_img, img_order, hp):
    """img_order: {video: [(frame, image_id), ...] sorted}.
    Writes per entity: s_A1 (causal replace), s_A2 (blend), track id.
    Returns track membership for the offline arm."""
    tracks_all = {}
    next_id = 0
    for vid, seq in img_order.items():
        live = []
        for fr, img in seq:
            ents = ent_by_img.get(img, [])
            live = [t for t in live if fr - t["last_frame"] <= hp["max_gap"]]
            used = set()
            for e in ents:                       # already sorted by obj desc
                e["track"] = None
                if e["obj"] < hp["link_min_obj"]:
                    continue
                best, best_c = None, None
                for ti, t in enumerate(live):
                    if ti in used or t["last_frame"] >= fr:
                        continue
                    c = _cost(e, t, hp)
                    if c is not None and (best_c is None or c < best_c):
                        best, best_c = ti, c
                if best is None:
                    t = {"id": next_id, "hist": deque(maxlen=hp["window"]),
                         "members": [], "last_frame": fr, "box": e["box"]}
                    next_id += 1
                    live.append(t)
                    tracks_all[t["id"]] = t
                    best = len(live) - 1
                t = live[best]
                used.add(best)
                t["hist"].append((e["obj"], e["p"]))
                t["members"].append(e)
                t["new_frame"] = fr
                t["new_box"] = e["box"]
                e["track"] = t["id"]
                w = np.array([o for o, _ in t["hist"]])
                P = np.stack([p for _, p in t["hist"]])
                ph = (w[:, None] * P).sum(0) / w.sum()
                ph = ph / ph.max()
                e["ph_causal"] = ph
            for t in live:
                if "new_frame" in t:
                    t["last_frame"] = t.pop("new_frame")
                    t["box"] = t.pop("new_box")
    for ents in ent_by_img.values():
        for e in ents:
            if e.get("track") is None:
                e["ph_causal"] = e["p"]
    return tracks_all


def apply_arms(ent_by_img, tracks_all, hp):
    lam = hp["lambda_blend"]
    for t in tracks_all.values():
        w = np.array([m["obj"] for m in t["members"]])
        P = np.stack([m["p"] for m in t["members"]])
        ph = (w[:, None] * P).sum(0) / w.sum()
        ph = ph / ph.max()
        for m in t["members"]:
            m["ph_offline"] = ph
    for ents in ent_by_img.values():
        for e in ents:
            e["s_A0"] = e["s"]
            e["s_A1"] = e["obj"] * e["ph_causal"]
            b = lam * e["p"] + (1 - lam) * e["ph_causal"]
            e["s_A2"] = e["obj"] * b / b.max()
            e["s_A3"] = e["obj"] * e.get("ph_offline", e["p"])


# --------------------------------------------------------------------------
# COCO evaluation at IoU 0.3 (same protocol as the H0 parity scripts)
# --------------------------------------------------------------------------
class Evaluator:
    def __init__(self, gt_dataset):
        from pycocotools.coco import COCO
        self.COCO = COCO
        self.base = gt_dataset
        self.cat_ids = sorted(c["id"] for c in gt_dataset["categories"])
        self.cat_names = {c["id"]: c["name"] for c in gt_dataset["categories"]}
        self.warn_log = []

    def _gt(self, img_ids, agnostic=False):
        ds = {"images": [im for im in self.base["images"] if im["id"] in img_ids],
              "annotations": [copy.copy(a) for a in self.base["annotations"] if a["image_id"] in img_ids],
              "categories": copy.deepcopy(self.base["categories"])}
        if agnostic:
            for a in ds["annotations"]:
                a["category_id"] = 1
            ds["categories"] = [{"id": 1, "name": "foreground"}]
        g = self.COCO()
        g.dataset = ds
        with contextlib.redirect_stdout(io.StringIO()):
            g.createIndex()
        return g

    @staticmethod
    def _map_from(E, keep_ids=None):
        """Mean AP over categories with GT, IoU 0.3, area all, maxDets 100;
        identical to COCOeval.summarize() stats[0].  If keep_ids is given,
        the per-image matches already computed by evaluate() are masked to
        those images and re-accumulated.  (Passing a reduced params object to
        accumulate() is NOT used: pycocotools indexes evalImgs by position in
        the reduced list, which selects the wrong images.)"""
        if keep_ids is None:
            E.accumulate()
        else:
            full = E.evalImgs
            I0 = len(E._paramsEval.imgIds)
            keep = np.array([i in keep_ids for i in E._paramsEval.imgIds])
            E.evalImgs = [e if keep[n % I0] else None for n, e in enumerate(full)]
            try:
                E.accumulate()
            finally:
                E.evalImgs = full
        prec = E.eval["precision"][0, :, :, 0, 2]
        v = prec[prec > -1]
        return (float(v.mean()) if v.size else -1.0), prec

    def evaluate(self, dets, img_ids, per_class=False, groups=None):
        """groups: {name: set(img_ids)} evaluated by re-accumulation (no re-matching)."""
        from pycocotools.cocoeval import COCOeval
        img_ids = set(img_ids)
        out = {}
        for agn in (False, True):
            gt = self._gt(img_ids, agnostic=agn)
            dd = [d for d in dets if d["image_id"] in img_ids]
            if agn:
                dd = [dict(d, category_id=1) for d in dd]
            if not dd:
                out["map_ca" if agn else "map"] = 0.0
                continue
            with warnings.catch_warnings(record=True) as wl, contextlib.redirect_stdout(io.StringIO()):
                warnings.simplefilter("always")
                dt = gt.loadRes(copy.deepcopy(dd))
                E = COCOeval(gt, dt, "bbox")
                E.params.iouThrs = np.array([0.30])
                E.evaluate()
                E.accumulate()
                E.summarize()
                m_all, prec = self._map_from(E)
                assert abs(m_all - float(E.stats[0])) < 1e-12
                grp = {}
                for name, ids in (groups or {}).items():
                    grp[name] = self._map_from(E, ids)[0]
                E.accumulate()
            self.warn_log.extend(sorted({str(w.message)[:160] for w in wl}))
            key = "map_ca" if agn else "map"
            out[key] = float(E.stats[0])
            if groups:
                out["per_video_ca" if agn else "per_video"] = grp
            if not agn:
                out["ar"] = float(E.stats[8])
                if per_class:
                    pc = {}
                    for k, cid in enumerate(E.params.catIds):
                        if not gt.getAnnIds(catIds=[cid]):
                            continue
                        pk = E.eval["precision"][0, :, k, 0, 2]
                        pk = pk[pk > -1]
                        pc[self.cat_names[cid]] = float(pk.mean()) if pk.size else 0.0
                    out["per_class"] = pc
        return out


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------
def iou_xywh(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0


def class_diagnostics(ent_by_img, gt_dataset, cat_ids, arms, min_obj=0.3):
    """Per GT box: highest-obj entity with IoU>=0.3 and obj>=min_obj.
    Report whether its argmax class is right before/after each arm."""
    cidx = {c: i for i, c in enumerate(cat_ids)}
    gts = defaultdict(list)
    for a in gt_dataset["annotations"]:
        gts[a["image_id"]].append(a)
    res = {a: {"fix": 0, "break": 0, "right_before": 0, "wrong_before": 0} for a in arms}
    n_cov = 0
    for img, ents in ent_by_img.items():
        for g in gts.get(img, []):
            cand = [e for e in ents if e["obj"] >= min_obj and iou_xywh(e["box"], g["bbox"]) >= 0.3]
            if not cand:
                continue
            n_cov += 1
            e = max(cand, key=lambda x: x["obj"])
            true = cidx[g["category_id"]]
            before = int(np.argmax(e["s"])) == true
            for a in arms:
                after = int(np.argmax(e["s_" + a])) == true
                r = res[a]
                r["right_before" if before else "wrong_before"] += 1
                if before and not after:
                    r["break"] += 1
                if (not before) and after:
                    r["fix"] += 1
    return {"gt_covered": n_cov, "arms": res}


def flicker_stats(ent_by_img, tracks_all, min_obj=0.3):
    switches, pairs, lens = 0, 0, []
    for t in tracks_all.values():
        ms = [m for m in t["members"] if m["obj"] >= min_obj]
        lens.append(len(t["members"]))
        for a, b in zip(ms, ms[1:]):
            pairs += 1
            switches += int(np.argmax(a["s"]) != np.argmax(b["s"]))
    n_ent = sum(len(v) for v in ent_by_img.values())
    n_linked = sum(1 for v in ent_by_img.values() for e in v if e.get("track") is not None)
    lens = np.array(lens) if lens else np.zeros(1)
    return {"entities": n_ent, "entities_linked": n_linked, "tracks": len(tracks_all),
            "tracks_len>=10": int((lens >= 10).sum()), "tracks_len>=50": int((lens >= 50).sum()),
            "track_len_median": float(np.median(lens)), "track_len_p90": float(np.percentile(lens, 90)),
            "consecutive_pairs_obj>=0.3": pairs, "class_switch_rate": (switches / pairs) if pairs else None}


# --------------------------------------------------------------------------
# Core pipeline (shared by selftest and run)
# --------------------------------------------------------------------------
def pipeline(dets, gt_dataset, hp, videos=None):
    cat_ids = sorted(c["id"] for c in gt_dataset["categories"])
    img_meta = {im["id"]: im for im in gt_dataset["images"]}
    det_imgs = sorted({d["image_id"] for d in dets})
    order = defaultdict(list)
    for img in det_imgs:
        fn = img_meta[img]["file_name"]
        order[video_of(fn)].append((frame_of(fn), img))
    for v in order:
        order[v].sort()
        frs = [f for f, _ in order[v]]
        assert len(frs) == len(set(frs)), f"duplicate frame numbers in {v}"
    ents, n_split = build_entities(dets, cat_ids)
    tracks = link_and_vote(ents, order, hp)
    apply_arms(ents, tracks, hp)
    arms = ["A0", "A1", "A2", "A3"]
    out = {a: emit(ents, cat_ids, "s_" + a, hp["topk"]) for a in arms}
    return ents, tracks, order, out, n_split


def evaluate_all(ev, arm_dets, order, raw_dets):
    all_imgs = {i for seq in order.values() for _, i in seq}
    groups = {v: {i for _, i in seq} for v, seq in sorted(order.items())}
    res = {"raw": ev.evaluate(raw_dets, all_imgs, per_class=True)}
    for a, d in arm_dets.items():
        res[a] = ev.evaluate(d, all_imgs, per_class=True, groups=groups)
    return res


def verdict(res, n_videos_required=PASS_RULE["min_videos_positive"]):
    pr = PASS_RULE
    out = {}
    for a in ["A1", "A2", "A3"]:
        d = res[a]["map"] - res["A0"]["map"]
        dca = res[a]["map_ca"] - res["A0"]["map_ca"]
        pv = {v: res[a]["per_video"][v] - res["A0"]["per_video"][v] for v in res["A0"]["per_video"]}
        npos = sum(1 for x in pv.values() if x > 0)
        out[a] = {"delta_map": d, "delta_map_ca": dca, "per_video_delta": pv, "videos_positive": npos,
                  "worst_video_delta": min(pv.values()),
                  "meets_rule": bool(d >= pr["min_delta_map"] and npos >= n_videos_required
                                     and min(pv.values()) >= pr["min_worst_video_delta"])}
    out["PASS"] = out["A1"]["meets_rule"]
    return out


# --------------------------------------------------------------------------
# Selftest
# --------------------------------------------------------------------------
def _synth(n_videos=6, n_frames=40, flip=0.35, persistent=False, seed=0):
    """Each frame: 3 objects (classes 1, 2, 3) + low-score junk of every class.
    The detector finds every object with high obj.  When it misreads an
    object, the wrong class gets obj and the true class only obj*0.05, so the
    true class falls below the junk (as in DETR, where the losing class is
    usually far down the top-100).  Misreads are random per frame (flicker)
    or, if persistent, always on object 0 in odd videos."""
    rng = np.random.default_rng(seed)
    cats = [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}, {"id": 3, "name": "C"}]
    images, anns, dets = [], [], []
    iid, aid = 1, 1
    for v in range(n_videos):
        vid = f"video_{v + 1:02d}"
        objs = [(100.0, 100.0, 20.0, 12.0, 1), (300.0, 200.0, 8.0, 8.0, 2), (450.0, 60.0, 14.0, 10.0, 3)]
        for f in range(1, n_frames + 1):
            images.append({"id": iid, "file_name": f"{vid}/{f:06d}.png", "width": 640, "height": 480})
            for k, (x, y, w, h, c) in enumerate(objs):
                xx, yy = x + 0.5 * f + rng.normal(0, 0.3), y + rng.normal(0, 0.3)
                anns.append({"id": aid, "image_id": iid, "category_id": c, "bbox": [xx, yy, w, h],
                             "area": w * h, "iscrowd": 0})
                aid += 1
                obj = float(rng.uniform(0.5, 0.9))
                wrong = (rng.random() < flip) if not persistent else (k == 0 and v % 2 == 1)
                other = c % 3 + 1
                box = [xx + rng.normal(0, 0.5), yy + rng.normal(0, 0.5), w, h]
                if wrong:
                    dets.append({"image_id": iid, "category_id": other, "bbox": box, "score": obj})
                    dets.append({"image_id": iid, "category_id": c, "bbox": box, "score": obj * 0.05})
                else:
                    dets.append({"image_id": iid, "category_id": c, "bbox": box, "score": obj})
                    dets.append({"image_id": iid, "category_id": other, "bbox": box, "score": obj * 0.05})
            for j in range(6):
                jb = [float(rng.uniform(0, 600)), float(rng.uniform(300, 440)), 6.0, 6.0]
                dets.append({"image_id": iid, "category_id": int(rng.integers(1, 4)), "bbox": jb,
                             "score": float(rng.uniform(0.05, 0.45))})
            iid += 1
    return {"images": images, "annotations": anns, "categories": cats}, dets


def selftest():
    print(f"[selftest] {VERSION}")
    # 1. metric sanity: perfect detections -> mAP 1
    gt, dets = _synth(n_videos=2, n_frames=10, flip=0.0)
    ev = Evaluator(gt)
    perfect = [{"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"], "score": 0.9}
               for a in gt["annotations"]]
    m = ev.evaluate(perfect, {im["id"] for im in gt["images"]})
    assert abs(m["map"] - 1.0) < 1e-9 and abs(m["map_ca"] - 1.0) < 1e-9, m
    print("  [ok] metric: perfect detections give mAP 1.0")

    # 2. identity: rebuild from entities reproduces raw cache exactly
    gt, dets = _synth(flip=0.35, seed=1)
    ev = Evaluator(gt)
    ents, tracks, order, arm, n_split = pipeline(dets, gt, HP)
    res = evaluate_all(ev, arm, order, dets)
    assert n_split == 0
    assert abs(res["raw"]["map"] - res["A0"]["map"]) < 1e-12, (res["raw"]["map"], res["A0"]["map"])
    assert abs(res["raw"]["map_ca"] - res["A0"]["map_ca"]) < 1e-12
    print(f"  [ok] identity: rebuilt A0 = raw ({res['A0']['map']:.4f})")
    for vname, seq in order.items():
        sep = ev.evaluate(arm["A1"], {i for _, i in seq})
        assert abs(sep["map"] - res["A1"]["per_video"][vname]) < 1e-12, vname
        assert abs(sep["map_ca"] - res["A1"]["per_video_ca"][vname]) < 1e-12, vname
    print("  [ok] per-video re-accumulation equals separate per-video evaluation")

    # 3. positive control: flicker -> A1 passes
    v = verdict(res, n_videos_required=4)
    print(f"  flicker control: A0 {res['A0']['map']:.4f}  A1 {res['A1']['map']:.4f}  "
          f"A3 {res['A3']['map']:.4f}  videos+ {v['A1']['videos_positive']}/6")
    assert v["PASS"], v["A1"]
    fs = flicker_stats(ents, tracks)
    long_tracks = sum(1 for t in tracks.values() if len(t["members"]) >= 35)
    assert long_tracks == 18, long_tracks        # 3 objects x 6 videos, 40 frames each
    print(f"  [ok] positive control passes; {long_tracks}/18 object tracks recovered, "
          f"switch rate {fs['class_switch_rate']:.2f}")

    # 4. negative control: persistent mislabel -> no gain
    gt, dets = _synth(flip=0.0, persistent=True, seed=2)
    ev = Evaluator(gt)
    _, _, order, arm, _ = pipeline(dets, gt, HP)
    res = evaluate_all(ev, arm, order, dets)
    v = verdict(res)
    print(f"  persistent control: A0 {res['A0']['map']:.4f}  A1 {res['A1']['map']:.4f}")
    assert not v["PASS"]
    print("  [ok] negative control does not pass")

    # 5. no flicker, no error -> A1 is (almost) identity
    gt, dets = _synth(flip=0.0, seed=3)
    ev = Evaluator(gt)
    _, _, order, arm, _ = pipeline(dets, gt, HP)
    res = evaluate_all(ev, arm, order, dets)
    assert abs(res["A1"]["map"] - res["A0"]["map"]) < 0.01, (res["A0"]["map"], res["A1"]["map"])
    assert not verdict(res)["PASS"]
    print("  [ok] null control: no change when classes are already stable")

    # 6. causality: perturbing future frames does not change A1 scores at t
    gt, dets = _synth(n_videos=1, n_frames=30, flip=0.35, seed=4)
    ents1, _, _, _, _ = pipeline(dets, gt, HP)
    cut = 15
    img_by_frame = {frame_of(im["file_name"]): im["id"] for im in gt["images"]}
    future = {img_by_frame[f] for f in img_by_frame if f > cut}
    dets2 = [dict(d, category_id=3, score=0.99) if d["image_id"] in future else d for d in dets]
    ents2, _, _, _, _ = pipeline(dets2, gt, HP)
    for f in range(1, cut + 1):
        a = sorted((tuple(e["box"]), tuple(np.round(e["s_A1"], 12))) for e in ents1[img_by_frame[f]])
        b = sorted((tuple(e["box"]), tuple(np.round(e["s_A1"], 12))) for e in ents2[img_by_frame[f]])
        assert a == b, f"A1 not causal at frame {f}"
    print("  [ok] causality: A1 at frame t ignores frames > t")

    # 7. top-k respected, gating
    assert all(c <= HP["topk"] for c in np.bincount([d["image_id"] for d in arm["A1"]]))
    e = {"box": [0, 0, 10, 10]}
    assert _cost(e, {"box": [0, 0, 10, 10]}, HP) == 0.0
    assert _cost(e, {"box": [20, 0, 10, 10]}, HP) is None
    assert _cost(e, {"box": [0, 0, 30, 30]}, HP) is None
    print("  [ok] top-k and gate")
    if ev.warn_log:
        print(f"  note: pycocotools warnings recorded: {sorted(set(ev.warn_log))[:3]}")
    print("SELFTEST PASS")


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
def run():
    print(f"[run] {VERSION}")
    for p in (CACHE_M0, VAL_JSON):
        if not p.exists():
            sys.exit(f"missing {p}")
    sha_cache, sha_val = sha256_file(CACHE_M0), sha256_file(VAL_JSON)
    print(f"  cache SHA-256 {sha_cache}\n  val   SHA-256 {sha_val}")
    with open(CACHE_M0, "r", encoding="utf-8") as f:
        cache = json.load(f)
    meta, raw = cache["_meta"], cache["detections"]
    with open(VAL_JSON, "r", encoding="utf-8") as f:
        val = json.load(f)

    # ---- parity gates ----
    img_ids = sorted({d["image_id"] for d in raw})
    gate = {}
    gate["meta_epoch_4"] = meta.get("checkpoint_epoch") == 4
    gate["meta_unconditioned"] = meta.get("conditioning_mode") is None
    gate["meta_2403"] = meta.get("num_g1_images") == EXPECTED_IMAGES
    gate["images_2403"] = len(img_ids) == EXPECTED_IMAGES
    gate["image_hash"] = hashlib.sha256(str(img_ids).encode()).hexdigest() == meta.get("g1_image_ids_hash")
    per_img = np.bincount([d["image_id"] for d in raw])
    gate["max_100_per_image"] = int(per_img.max()) <= HP["topk"]
    vids = sorted({video_of(im["file_name"]) for im in val["images"] if im["id"] in set(img_ids)})
    gate["videos"] = vids == EXPECTED_VIDEOS
    print("  gates:", gate)
    if not all(gate.values()):
        sys.exit("PARITY GATE FAILED (inputs) -- stop, do not interpret")

    sub = {"images": [im for im in val["images"] if im["id"] in set(img_ids)],
           "annotations": [a for a in val["annotations"] if a["image_id"] in set(img_ids)],
           "categories": val["categories"]}
    ev = Evaluator(sub)
    ents, tracks, order, arm, n_split = pipeline(raw, sub, HP)
    res = evaluate_all(ev, arm, order, raw)

    expect = BASELINE_MAP30
    if M0_SIDECAR.exists():
        with open(M0_SIDECAR, "r", encoding="utf-8") as f:
            expect = float(json.load(f)["metrics"]["g1"]["map_03"])
    gate2 = {"raw_matches_baseline": abs(res["raw"]["map"] - expect) <= (1e-4 if M0_SIDECAR.exists() else BASELINE_TOL),
             "rebuild_identity": abs(res["raw"]["map"] - res["A0"]["map"]) < 1e-12
                                 and abs(res["raw"]["map_ca"] - res["A0"]["map_ca"]) < 1e-12,
             "no_entity_splits": n_split == 0}
    print(f"  raw mAP30 {res['raw']['map']:.6f}  (expected {expect:.6f})")
    print("  gates:", gate2)
    if not all(gate2.values()):
        sys.exit("PARITY GATE FAILED (evaluation) -- stop, do not interpret")

    diag = class_diagnostics(ents, sub, sorted(c["id"] for c in sub["categories"]), ["A1", "A2", "A3"])
    fl = flicker_stats(ents, tracks)
    v = verdict(res)

    # ---- report ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"version": VERSION, "hp": HP, "pass_rule": PASS_RULE,
               "sha": {"cache": sha_cache, "val_json": sha_val, "script": sha256_file(Path(__file__))},
               "gates": {**gate, **gate2}, "results": res, "verdict": v, "class_diag": diag, "tracking": fl,
               "pycocotools_warnings": sorted(set(ev.warn_log))}
    with open(OUT_DIR / "g1b0_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)

    L = [f"# G1b-0 results ({VERSION})", "",
         "| Arm | mAP30 | mAPca30 | AR | Δ mAP | Δ mAPca | videos + |", "|---|---|---|---|---|---|---|"]
    for a in ["A0", "A1", "A2", "A3"]:
        r = res[a]
        if a == "A0":
            L.append(f"| {a} | {r['map']:.4f} | {r['map_ca']:.4f} | {r['ar']:.4f} | – | – | – |")
        else:
            L.append(f"| {a} | {r['map']:.4f} | {r['map_ca']:.4f} | {r['ar']:.4f} | {v[a]['delta_map']:+.4f} | "
                     f"{v[a]['delta_map_ca']:+.4f} | {v[a]['videos_positive']}/6 |")
    L += ["", "| Video | A0 | A1 | A2 | A3 |", "|---|---|---|---|---|"]
    for vid in sorted(res["A0"]["per_video"]):
        L.append(f"| {vid} | " + " | ".join(f"{res[a]['per_video'][vid]:.4f}" for a in ["A0", "A1", "A2", "A3"]) + " |")
    L += ["", "| Class | A0 | A1 | A2 | A3 |", "|---|---|---|---|---|"]
    for c in res["A0"]["per_class"]:
        L.append(f"| {c} | " + " | ".join(f"{res[a]['per_class'][c]:.4f}" for a in ["A0", "A1", "A2", "A3"]) + " |")
    L += ["", f"Class diagnostics (GT covered by an entity with obj ≥ 0.3: {diag['gt_covered']}):", "",
          "| Arm | right before | wrong before | fixed | broken |", "|---|---|---|---|---|"]
    for a, r in diag["arms"].items():
        L.append(f"| {a} | {r['right_before']} | {r['wrong_before']} | {r['fix']} | {r['break']} |")
    L += ["", f"Tracking: {json.dumps(fl)}", "", f"VERDICT (primary A1): {'PASS' if v['PASS'] else 'FAIL'}"]
    (OUT_DIR / "g1b0_results_tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n  wrote {OUT_DIR / 'g1b0_results.json'}\n  wrote {OUT_DIR / 'g1b0_results_tables.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["selftest", "run"], required=True)
    a = ap.parse_args()
    selftest() if a.mode == "selftest" else run()


if __name__ == "__main__":
    main()
