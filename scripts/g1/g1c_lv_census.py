"""
G1c: LEISURE_VEHICLE census on TRAIN (CPU, annotations only).

Location: D:/MARIVISION/H0_workbench/eval_tools/g1c_lv_census.py

Why: LEISURE_VEHICLE (LV) is 1/7 of TEST mAP.  It has 0 VAL boxes,
~600 TRAIN boxes (nearly all medium) and ~12.1k TEST boxes, of which
9,912 are COCO-small (paper supplement, Table 1).  We cannot read TEST
labels, so the only question we can answer is: what does TRAIN LV look
like, and how far must it be shrunk to resemble TEST LV?

This is a census, not a hypothesis test.  Its output fixes the design of
the LV scale augmentation and of an LV held-out-video check (VAL has
no LV, so LV can only be validated on held-out TRAIN videos).

Modes:  --mode selftest   |   --mode run
"""

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

VERSION = "G1c.0"
TRAIN_JSON = Path("D:/MARIVISION/seaclips-train.json")
OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools/g1c_reports")

# public numbers from the SeaClips supplement (TEST split, LEISURE_VEHICLE)
TEST_LV = {"small": 9912, "medium": 2203}
SMALL, MEDIUM = 32 ** 2, 96 ** 2           # COCO area thresholds (native pixels)
WARP = 640                                  # R0 protocol, for reference only
LV_NAME = "LEISURE_VEHICLE"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def video_of(fn):
    for part in str(fn).replace("\\", "/").split("/"):
        if part.startswith("video_"):
            return part
    return "unknown"


def scale_bin(area):
    return "small" if area < SMALL else ("medium" if area < MEDIUM else "large")


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else None


def census(ds):
    cats = {c["id"]: c["name"] for c in ds["categories"]}
    name2id = {v: k for k, v in cats.items()}
    if LV_NAME not in name2id:
        raise SystemExit(f"{LV_NAME} not in categories {sorted(cats.values())}")
    lv = name2id[LV_NAME]
    imgs = {im["id"]: im for im in ds["images"]}
    frames_per_video = Counter(video_of(im["file_name"]) for im in ds["images"])

    rows = []
    for a in ds["annotations"]:
        im = imgs[a["image_id"]]
        x, y, w, h = a["bbox"]
        area = w * h                         # bbox area; COCO uses ann["area"], reported too
        rows.append({"cat": cats[a["category_id"]], "video": video_of(im["file_name"]),
                     "img": a["image_id"], "w": w, "h": h, "area_bbox": area,
                     "area_ann": float(a.get("area", area)), "H": im["height"], "W": im["width"],
                     "sensor": im.get("camera_sensor"), "weather": im.get("weather"),
                     "sea": im.get("sea_state"), "refl": im.get("water_reflections"),
                     "yc": (y + h / 2) / im["height"], "xc": (x + w / 2) / im["width"]})

    out = {"n_images": len(ds["images"]), "n_annotations": len(ds["annotations"])}
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["cat"]].append(r)

    # ---- all classes: scale mix (context for LV) ----
    out["scale_mix_by_class"] = {
        c: dict(Counter(scale_bin(r["area_ann"]) for r in rs)) for c, rs in sorted(by_cat.items())}

    L = by_cat[LV_NAME]
    out["lv_boxes"] = len(L)
    if not L:
        return out
    sq = np.array([math.sqrt(r["area_ann"]) for r in L])
    ms = np.array([min(r["w"], r["h"]) for r in L])
    ar = np.array([r["w"] / max(r["h"], 1e-6) for r in L])
    warp_ms = np.array([min(r["w"] * WARP / r["W"], r["h"] * WARP / r["H"]) for r in L])
    out["lv_scale"] = dict(Counter(scale_bin(r["area_ann"]) for r in L))
    out["lv_sqrt_area_px"] = {q: pct(sq, q) for q in (5, 25, 50, 75, 95)}
    out["lv_min_side_px"] = {q: pct(ms, q) for q in (5, 25, 50, 75, 95)}
    out["lv_aspect_w_over_h"] = {q: pct(ar, q) for q in (5, 25, 50, 75, 95)}
    out["lv_min_side_after_640_warp_px"] = {q: pct(warp_ms, q) for q in (5, 25, 50, 75, 95)}
    out["lv_area_bbox_vs_ann_max_rel_diff"] = float(max(abs(r["area_bbox"] - r["area_ann"]) / max(r["area_ann"], 1)
                                                      for r in L))

    # ---- where LV lives ----
    pv = defaultdict(list)
    for r in L:
        pv[r["video"]].append(r)
    out["lv_per_video"] = {
        v: {"boxes": len(rs), "frames_with_lv": len({r["img"] for r in rs}), "frames_in_video": frames_per_video[v],
            "median_sqrt_area": float(np.median([math.sqrt(r["area_ann"]) for r in rs])),
            "sensor": Counter(r["sensor"] for r in rs).most_common(1)[0][0],
            "weather": Counter(r["weather"] for r in rs).most_common(1)[0][0],
            "sea": Counter(r["sea"] for r in rs).most_common(1)[0][0],
            "scale": dict(Counter(scale_bin(r["area_ann"]) for r in rs))}
        for v, rs in sorted(pv.items(), key=lambda kv: -len(kv[1]))}
    out["lv_videos"] = len(pv)
    out["lv_sensor"] = dict(Counter(r["sensor"] for r in L))
    out["lv_weather"] = dict(Counter(r["weather"] for r in L))
    out["lv_y_centre"] = {q: pct(np.array([r["yc"] for r in L]), q) for q in (5, 50, 95)}

    # ---- co-occurrence: what else is in LV frames ----
    lv_imgs = {r["img"] for r in L}
    co = Counter(r["cat"] for r in rows if r["img"] in lv_imgs and r["cat"] != LV_NAME)
    out["lv_frame_cooccurrence"] = dict(co)

    # ---- the gap to TEST ----
    tot = TEST_LV["small"] + TEST_LV["medium"]
    out["test_lv_public"] = {**TEST_LV, "small_share": TEST_LV["small"] / tot}
    out["train_lv_small_share"] = out["lv_scale"].get("small", 0) / len(L)
    # shrink factor s (applied to both sides) that makes each TRAIN LV box COCO-small
    s_small = np.minimum(1.0, 31.99 / sq)
    out["shrink_to_small"] = {q: pct(s_small, q) for q in (5, 25, 50, 75, 95)}
    # min side each box would have once just inside 'small'
    out["min_side_when_just_small_px"] = {q: pct(ms * s_small, q) for q in (5, 25, 50, 75, 95)}
    # reference classes of similar kind for appearance at small scale
    ref = {}
    for c in ("BOAT", "SAILING_VESSEL", "OBJECT"):
        if c in by_cat:
            s = np.array([math.sqrt(r["area_ann"]) for r in by_cat[c]])
            ref[c] = {"n": len(s), "median_sqrt_area": pct(s, 50), "small_share": float((s ** 2 < SMALL).mean())}
    out["reference_classes"] = ref
    return out


def tables(o):
    L = [f"# G1c LV census ({VERSION})", "",
         f"TRAIN: {o['n_images']} images, {o['n_annotations']} boxes; LV boxes {o.get('lv_boxes', 0)} "
         f"in {o.get('lv_videos', 0)} videos", ""]
    if not o.get("lv_boxes"):
        return "\n".join(L)
    L += ["| Scale (COCO, native px) | TRAIN LV | TEST LV (public) |", "|---|---|---|"]
    for k in ("small", "medium", "large"):
        L.append(f"| {k} | {o['lv_scale'].get(k, 0)} | {o['test_lv_public'].get(k, 0)} |")
    def q(d, fmt="{:.1f}"):
        return " / ".join(fmt.format(d[k]) for k in (5, 25, 50, 75, 95))
    L += ["", "Percentiles 5 / 25 / 50 / 75 / 95:", "",
          f"- √area px: {q(o['lv_sqrt_area_px'])}",
          f"- min side px: {q(o['lv_min_side_px'])}",
          f"- min side after 640 warp px: {q(o['lv_min_side_after_640_warp_px'])}",
          "- aspect w/h: " + q(o["lv_aspect_w_over_h"], "{:.2f}"),
          "- shrink factor to reach small: " + q(o["shrink_to_small"], "{:.2f}"),
          f"- min side once just small px: {q(o['min_side_when_just_small_px'])}", "",
          "| Video | boxes | frames w/ LV | frames | median √area | sensor | weather | sea | scale |",
          "|---|---|---|---|---|---|---|---|---|"]
    for v, r in o["lv_per_video"].items():
        L.append(f"| {v} | {r['boxes']} | {r['frames_with_lv']} | {r['frames_in_video']} | {r['median_sqrt_area']:.1f} | "
                 f"{r['sensor']} | {r['weather']} | {r['sea']} | {json.dumps(r['scale'])} |")
    L += ["", f"Co-occurring classes in LV frames: {json.dumps(o['lv_frame_cooccurrence'])}",
          f"LV y-centre (5/50/95): {o['lv_y_centre']}",
          f"Reference classes: {json.dumps(o['reference_classes'])}",
          "", "Scale mix, all classes:", ""]
    for c, d in o["scale_mix_by_class"].items():
        L.append(f"- {c}: {json.dumps(d)}")
    return "\n".join(L) + "\n"


def _synth():
    cats = [{"id": 1, "name": "BOAT"}, {"id": 5, "name": LV_NAME}, {"id": 4, "name": "OBJECT"}]
    images, anns, aid = [], [], 1
    for v, n in (("video_01", 10), ("video_02", 10)):
        for f in range(n):
            iid = len(images) + 1
            images.append({"id": iid, "file_name": f"{v}/{f + 1:06d}.png", "height": 960, "width": 1200,
                           "camera_sensor": "e-CAM", "weather": "sunny", "sea_state": "calm",
                           "water_reflections": "no"})
            for cid, w, h in ((1, 20, 10), (5, 64, 40) if v == "video_01" else (5, 20, 20), (4, 4, 4)):
                anns.append({"id": aid, "image_id": iid, "category_id": cid, "bbox": [100.0, 400.0, w, h],
                             "area": float(w * h), "iscrowd": 0})
                aid += 1
    return {"images": images, "annotations": anns, "categories": cats}


def selftest():
    print(f"[selftest] {VERSION}")
    o = census(_synth())
    assert o["lv_boxes"] == 20 and o["lv_videos"] == 2
    assert o["lv_scale"] == {"medium": 10, "small": 10}, o["lv_scale"]
    assert abs(o["lv_sqrt_area_px"][50] - (math.sqrt(400) + math.sqrt(2560)) / 2) < 1e-9
    s = o["shrink_to_small"]
    assert abs(s[95] - 1.0) < 1e-12 and abs(s[5] - 31.99 / math.sqrt(2560)) < 1e-9
    assert abs(o["lv_min_side_after_640_warp_px"][95] - min(40 * 640 / 960, 64 * 640 / 1200)) < 1e-9
    assert o["lv_frame_cooccurrence"] == {"BOAT": 20, "OBJECT": 20}
    assert o["scale_mix_by_class"]["OBJECT"] == {"small": 20}
    tables(o)
    print("  [ok] counts, scale bins, percentiles, shrink factor, warp, co-occurrence, tables")
    print("SELFTEST PASS")


def run():
    print(f"[run] {VERSION}")
    if not TRAIN_JSON.exists():
        sys.exit(f"missing {TRAIN_JSON}")
    sha = sha256_file(TRAIN_JSON)
    print(f"  train SHA-256 {sha}")
    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        ds = json.load(f)
    inv = {"images_20310": len(ds["images"]) == 20310, "annotations_65542": len(ds["annotations"]) == 65542}
    print("  inventory:", inv)
    if not all(inv.values()):
        sys.exit("INVENTORY FAILED -- stop")
    o = census(ds)
    o["version"], o["train_sha"] = VERSION, sha
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "g1c_lv_census.json").write_text(json.dumps(o, indent=1), encoding="utf-8")
    t = tables(o)
    (OUT_DIR / "g1c_lv_census_tables.md").write_text(t, encoding="utf-8")
    print(t)
    print(f"  wrote {OUT_DIR / 'g1c_lv_census.json'}\n  wrote {OUT_DIR / 'g1c_lv_census_tables.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["selftest", "run"], required=True)
    a = ap.parse_args()
    selftest() if a.mode == "selftest" else run()
