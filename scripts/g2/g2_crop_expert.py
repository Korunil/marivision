"""
G2: crop + context class expert trained on TRAIN ground truth, judged on
unseen VAL videos against R0's own class decisions on the same boxes.

Location: D:/MARIVISION/H0_workbench/g2/g2_crop_expert.py

Modes (run in this order; each refuses to run if its inputs are missing):
  --mode selftest   CPU, synthetic zip + jsons, tiny CNN; no real files touched
  --mode smoke      GPU, a few real images: builds a mini cache, overfit test,
                    throughput / memory / time projections, end-to-end eval.
                    This is the proof that the full run can start.
  --mode build      full crop cache from the zip (read-only; never extracted)
  --mode train      trains the expert (fixed schedule, last epoch kept; no VAL
                    selection).  --holdout-video video_23 for the LV check (D3)
  --mode eval       primary paired readout + secondary mAP rescoring + LV guard

Nothing in the cache or the run directory may leave the workstation
(dataset-derived).  Only this code and the methodology are shareable.
"""

import argparse
import contextlib
import copy
import hashlib
import io
import json
import math
import os
import sys
import time
import warnings
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image  # noqa: F401  -- must be imported before torch on the Director's Windows env (PIL DLL)

VERSION = "G2.0.1"   # G2.0 + PIL-before-torch import only; cache built by G2.0 is unchanged and valid

ZIP_PATH = Path("D:/MARIVISION/SeaClips.zip")
TRAIN_JSON = Path("D:/MARIVISION/seaclips-train.json")
VAL_JSON = Path("D:/MARIVISION/seaclips-val.json")
R0_EXPORT = Path("D:/MARIVISION/R0_frozen_baseline/val_exports/r0_val_predictions_epoch04.json")
CACHE_DIR = Path("D:/MARIVISION/G2_cache")
SMOKE_CACHE_DIR = Path("D:/MARIVISION/G2_cache_smoke")
RUN_DIR = Path("D:/MARIVISION/G2_runs")

EXPECT = {"train_images": 20310, "train_anns": 65542, "val_images": 4266, "val_anns": 13699,
          "export_dets": 426600, "eval_frames": 432, "eval_gt": 1386}

CFG = {
    "context_mult": 4.0,      # context window side S = clip(4 * max(w, h), S_min, S_max)
    "s_min": 64.0,
    "s_max": 1024.0,
    "store_mult": 1.5,        # stored window = 1.5 S, so scale/shift jitter has room
    "store_px": 192,          # stored crop size  (S maps to 128 px)
    "out_px": 128,            # network input
    "train_frame_stride": 2,  # every 2nd TRAIN frame (adjacent frames are near-duplicates)
    "bg_patch_px": 256,       # native-resolution background windows for LV paste
    "n_bg_train": 3000,
    "n_bg_val": 500,
    "bg_y_range": (0.45, 0.62),
    "lv_patch_px": 320,       # padded native LV patch (2x box extent around centre)
    "lv_synth_p": 0.7,        # D1/D2: LV sample replaced by shrink+paste version
    "balance_power": 0.5,     # class share proportional to n_c ** 0.5
    "epochs": 8,
    "samples_per_epoch": 30000,
    "batch": 64,
    "lr": 2e-4,
    "weight_decay": 0.05,
    "warmup_steps": 200,
    "label_smoothing": 0.1,
    "seed": 0,
    "match_iou": 0.3,         # R0 entity <-> GT, as in G1b-0
    "match_min_obj": 0.3,
    "topk": 100,
    "lv_check_sizes": [32, 16, 8],  # D3: sqrt(area) of shrunk held-out LV, native px
}

PASS_RULE = {
    "min_delta_bal_acc": 0.05,          # expert - R0, paired, pooled over covered VAL GT
    "min_video_boxes": 30,              # videos with fewer covered boxes are not counted
    "min_video_share_not_worse": 2 / 3, # expert accuracy >= R0 accuracy in >= 2/3 of counted videos
}


# =============================================================================
# Generic helpers
# =============================================================================
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


def eval_subset_ids(images, stride=10, offset=0):
    """Identical to the V13/M2 protocol (432 frames, 1,386 GT on VAL)."""
    by_video = defaultdict(list)
    for im in images.values():
        by_video[int(im.get("video_id", 0))].append(im)
    out = []
    for _, lst in sorted(by_video.items()):
        lst.sort(key=lambda x: (int(x.get("frame_id", x["id"])), int(x["id"])))
        for i, im in enumerate(lst):
            if stride <= 1 or i % stride == offset:
                out.append(int(im["id"]))
    return out


def iou_xywh(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0


class ZipImages:
    """Read-only access to SeaClips/<split>/<file_name> inside the zip."""

    def __init__(self, zip_path, split):
        self.zip_path, self.split, self.z, self.map = Path(zip_path), split, None, None

    def _open(self):
        if self.z is None:
            self.z = zipfile.ZipFile(self.zip_path, "r")
            self.map = {}
            for m in self.z.namelist():
                n = m.replace("\\", "/").strip("/")
                if n in self.map:
                    raise RuntimeError(f"duplicate zip path {n}")
                self.map[n] = m

    def load(self, file_name):
        from PIL import Image
        self._open()
        key = f"SeaClips/{self.split}/{file_name.replace(chr(92), '/').strip('/')}"
        if key not in self.map:
            raise FileNotFoundError(f"missing zip member {key}")
        return np.asarray(Image.open(io.BytesIO(self.z.read(self.map[key]))).convert("RGB"))


# =============================================================================
# Crop geometry
# =============================================================================
def context_side(w, h, cfg=CFG):
    return float(min(max(cfg["context_mult"] * max(w, h), cfg["s_min"]), cfg["s_max"]))


def extract_square(img, cx, cy, side, out_px, fill=None):
    """Square window of native side `side` centred at (cx, cy), resized to out_px.
    Pixels outside the image take the image mean colour."""
    from PIL import Image
    H, W = img.shape[:2]
    if fill is None:
        fill = img.reshape(-1, 3).mean(0)
    x0, y0 = int(math.floor(cx - side / 2)), int(math.floor(cy - side / 2))
    n = int(math.ceil(side))
    canvas = np.empty((n, n, 3), dtype=np.uint8)
    canvas[:] = np.asarray(fill, dtype=np.uint8)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x0 + n), min(H, y0 + n)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    if n == out_px:
        return canvas
    return np.asarray(Image.fromarray(canvas).resize((out_px, out_px), Image.BILINEAR))


def stored_crop(img, box, cfg=CFG):
    x, y, w, h = box
    S = context_side(w, h, cfg)
    return extract_square(img, x + w / 2, y + h / 2, cfg["store_mult"] * S, cfg["store_px"])


def eval_view(stored, cfg=CFG):
    """Central S window of the stored 1.5 S crop = exactly the context window."""
    c = cfg["store_px"] // 2
    r = int(round(cfg["store_px"] / cfg["store_mult"] / 2))
    v = stored[c - r:c + r, c - r:c + r]
    if v.shape[0] != cfg["out_px"]:
        from PIL import Image
        v = np.asarray(Image.fromarray(v).resize((cfg["out_px"], cfg["out_px"]), Image.BILINEAR))
    return v


def train_view(stored, rng, cfg=CFG):
    from PIL import Image
    base = cfg["store_px"] / cfg["store_mult"]           # 128
    side = base * rng.uniform(0.85, 1.15)
    c = cfg["store_px"] / 2
    cx, cy = c + rng.uniform(-0.08, 0.08) * base, c + rng.uniform(-0.08, 0.08) * base
    return extract_square(stored, cx, cy, side, cfg["out_px"], fill=stored.reshape(-1, 3).mean(0))


# =============================================================================
# LV shrink + paste (G1c D1/D2)
# =============================================================================
def lv_patch(img, box, cfg=CFG):
    """Native patch of 2x the box extent around its centre, padded to lv_patch_px."""
    x, y, w, h = box
    pw, ph = int(math.ceil(2 * w)), int(math.ceil(2 * h))
    P = cfg["lv_patch_px"]
    if pw > P or ph > P:
        return None
    H, W = img.shape[:2]
    out = np.zeros((P, P, 3), dtype=np.uint8)
    fill = img.reshape(-1, 3).mean(0).astype(np.uint8)
    out[:] = fill
    x0, y0 = int(math.floor(x + w / 2 - pw / 2)), int(math.floor(y + h / 2 - ph / 2))
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x0 + pw), min(H, y0 + ph)
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return out, (pw, ph)


def feather_mask(pw, ph):
    """1 inside the central box (half the patch extent), linear fade to 0 at the patch edge."""
    yy = np.abs((np.arange(ph) + 0.5) / ph - 0.5) * 2   # 0 centre .. 1 edge
    xx = np.abs((np.arange(pw) + 0.5) / pw - 0.5) * 2
    my = np.clip((1 - yy) / 0.5, 0, 1)
    mx = np.clip((1 - xx) / 0.5, 0, 1)
    return np.minimum(my[:, None], mx[None, :])


def lv_synthetic(patch, psize, box_wh, bg, target_sqrt_area, rng, cfg=CFG, jitter=True):
    """Shrink the LV (never enlarge) to target sqrt(area), paste at the centre of a
    native background window, return the network view (out_px).  None if the
    required context window does not fit inside the background patch."""
    from PIL import Image
    w, h = box_wh
    f = min(1.0, target_sqrt_area / math.sqrt(w * h))
    nw, nh = w * f, h * f
    S = context_side(nw, nh, cfg) * (rng.uniform(0.85, 1.15) if jitter else 1.0)
    if S > bg.shape[0]:
        return None
    pw, ph = psize
    tw, th = max(1, int(round(pw * f))), max(1, int(round(ph * f)))
    obj = np.asarray(Image.fromarray(patch[:ph, :pw]).resize((tw, th), Image.LANCZOS)).astype(np.float32)
    m = feather_mask(tw, th)[..., None]
    c = bg.shape[0] / 2
    win = extract_square(bg, c, c, S, int(math.ceil(S)), fill=bg.reshape(-1, 3).mean(0)).astype(np.float32)
    n = win.shape[0]
    ox, oy = int(round(n / 2 - tw / 2)), int(round(n / 2 - th / 2))
    region = win[oy:oy + th, ox:ox + tw]
    region[:] = m * obj[:region.shape[0], :region.shape[1]] + (1 - m) * region
    if f < 1.0:
        win = win + rng.normal(0, rng.uniform(0, 3), win.shape)
    win = np.clip(win, 0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(win).resize((cfg["out_px"], cfg["out_px"]), Image.BILINEAR))


# =============================================================================
# R0 entities and matching
# =============================================================================
def r0_entities(dets, cat_ids):
    """Same-bbox detections = one Deformable-DETR query = one entity."""
    cidx = {c: i for i, c in enumerate(cat_ids)}
    per = defaultdict(dict)
    for d in dets:
        k = tuple(float(v) for v in d["bbox"])
        e = per[d["image_id"]].setdefault(k, {"box": list(k), "s": np.zeros(len(cat_ids))})
        e["s"][cidx[d["category_id"]]] = max(e["s"][cidx[d["category_id"]]], float(d["score"]))
    out = {}
    for img, m in per.items():
        ents = list(m.values())
        for e in ents:
            e["obj"] = float(e["s"].max())
        ents.sort(key=lambda e: -e["obj"])
        out[img] = ents
    return out


def match_gt(gt_anns, ents, cfg=CFG):
    """Per GT: highest-obj entity with IoU >= 0.3 and obj >= 0.3 (G1b-0 rule)."""
    out = []
    for g in gt_anns:
        cand = [e for e in ents if e["obj"] >= cfg["match_min_obj"] and iou_xywh(e["box"], g["bbox"]) >= cfg["match_iou"]]
        out.append(max(cand, key=lambda e: e["obj"]) if cand else None)
    return out


# =============================================================================
# Cache building
# =============================================================================
def _open_memmap(path, shape):
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=shape)


def build_cache(train, val, export, zip_path, out_dir, cfg=CFG, train_image_ids=None, val_image_ids=None,
                n_bg_train=None, n_bg_val=None, log=print):
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])
    cat_ids = sorted(c["id"] for c in train["categories"])
    names = {c["id"]: c["name"] for c in train["categories"]}
    lv_id = [c for c in cat_ids if names[c] == "LEISURE_VEHICLE"]
    lv_id = lv_id[0] if lv_id else None
    label = {c: i for i, c in enumerate(cat_ids)}

    # ---------- TRAIN ----------
    t_imgs = {im["id"]: im for im in train["images"]}
    by_vid = defaultdict(list)
    for im in t_imgs.values():
        by_vid[video_of(im["file_name"])].append(im)
    keep = set()
    for v, lst in by_vid.items():
        lst.sort(key=lambda x: (int(x.get("frame_id", x["id"])), x["id"]))
        keep.update(im["id"] for i, im in enumerate(lst) if i % cfg["train_frame_stride"] == 0)
    if train_image_ids is not None:
        keep &= set(train_image_ids)
    t_anns = defaultdict(list)
    for a in train["annotations"]:
        if a["image_id"] in keep:
            t_anns[a["image_id"]].append(a)
    t_list = sorted(keep, key=lambda i: (video_of(t_imgs[i]["file_name"]), t_imgs[i].get("frame_id", i), i))
    n_crops = sum(len(t_anns[i]) for i in t_list)
    n_lv = sum(1 for i in t_list for a in t_anns[i] if a["category_id"] == lv_id)
    S, P = cfg["store_px"], cfg["lv_patch_px"]
    tr = _open_memmap(out_dir / "train_crops.npy", (n_crops, S, S, 3))
    lvp = _open_memmap(out_dir / "lv_patches.npy", (max(n_lv, 1), P, P, 3))
    non_lv_vids = sorted(v for v in by_vid if not any(a["category_id"] == lv_id for i in keep
                                                       if video_of(t_imgs[i]["file_name"]) == v for a in t_anns[i]))
    nbg_t = cfg["n_bg_train"] if n_bg_train is None else n_bg_train
    bgt = _open_memmap(out_dir / "bg_train.npy", (nbg_t, cfg["bg_patch_px"], cfg["bg_patch_px"], 3))
    bg_frames = [i for i in t_list if video_of(t_imgs[i]["file_name"]) in set(non_lv_vids)]
    bg_pick = Counter(rng.choice(bg_frames, size=nbg_t, replace=True).tolist()) if bg_frames and nbg_t else Counter()
    zt = ZipImages(zip_path, "train")
    idx, lv_idx, bg_t_meta, k, kl, kb = [], [], [], 0, 0, 0
    for n_done, i in enumerate(t_list):
        im = t_imgs[i]
        img = zt.load(im["file_name"])
        for a in t_anns[i]:
            tr[k] = stored_crop(img, a["bbox"], cfg)
            idx.append({"k": k, "video": video_of(im["file_name"]), "image_id": i, "cat": a["category_id"],
                        "label": label[a["category_id"]], "bbox": a["bbox"]})
            k += 1
            if a["category_id"] == lv_id:
                r = lv_patch(img, a["bbox"], cfg)
                if r is not None:
                    lvp[kl] = r[0]
                    lv_idx.append({"k": kl, "video": video_of(im["file_name"]), "psize": r[1],
                                   "wh": [a["bbox"][2], a["bbox"][3]]})
                    kl += 1
        for _ in range(bg_pick.get(i, 0)):
            p = _bg_window(img, t_anns[i], rng, cfg)
            if p is not None:
                bgt[kb] = p
                bg_t_meta.append({"video": video_of(im["file_name"])})
                kb += 1
        if n_done % 500 == 0:
            log(f"  train frames {n_done}/{len(t_list)}  {time.time() - t0:.0f}s")
    assert k == n_crops
    sizes = {"BOAT", "SAILING_VESSEL"}
    vessel_sqrt = sorted(math.sqrt(a["bbox"][2] * a["bbox"][3]) for a in train["annotations"]
                         if names[a["category_id"]] in sizes)

    # ---------- VAL ----------
    v_imgs = {im["id"]: im for im in val["images"]}
    v_ids = sorted(v_imgs) if val_image_ids is None else sorted(set(val_image_ids))
    v_anns = defaultdict(list)
    for a in val["annotations"]:
        v_anns[a["image_id"]].append(a)
    ents = r0_entities(export, cat_ids)
    sub = set(eval_subset_ids(v_imgs))
    gt_rows, match_rows, ent_rows = [], [], []
    for i in v_ids:
        m = match_gt(v_anns[i], ents.get(i, []), cfg)
        for a, e in zip(v_anns[i], m):
            gt_rows.append((i, a, e))
        if i in sub:
            for j, e in enumerate(ents.get(i, [])):
                ent_rows.append((i, j, e))
    vg = _open_memmap(out_dir / "val_gt_crops.npy", (max(len(gt_rows), 1), S, S, 3))
    n_m = sum(1 for r in gt_rows if r[2] is not None)
    vm = _open_memmap(out_dir / "val_match_crops.npy", (max(n_m, 1), S, S, 3))
    ve = _open_memmap(out_dir / "val_ent_crops.npy", (max(len(ent_rows), 1), S, S, 3))
    nbg_v = cfg["n_bg_val"] if n_bg_val is None else n_bg_val
    bgv = _open_memmap(out_dir / "bg_val.npy", (max(nbg_v, 1), cfg["bg_patch_px"], cfg["bg_patch_px"], 3))
    bgv_pick = Counter(rng.choice(v_ids, size=nbg_v, replace=True).tolist()) if nbg_v else Counter()
    zv = ZipImages(zip_path, "val")
    rows_by_img = defaultdict(list)
    for r in gt_rows:
        rows_by_img[r[0]].append(r)
    ents_by_img = defaultdict(list)
    for r in ent_rows:
        ents_by_img[r[0]].append(r)
    vg_idx, vm_idx, ve_idx, kg, km, ke, kbv = [], [], [], 0, 0, 0, 0
    for n_done, i in enumerate(v_ids):
        im = v_imgs[i]
        need = rows_by_img.get(i) or ents_by_img.get(i) or bgv_pick.get(i, 0)
        if not need:
            continue
        img = zv.load(im["file_name"])
        for (_, a, e) in rows_by_img.get(i, []):
            vg[kg] = stored_crop(img, a["bbox"], cfg)
            row = {"k": kg, "video": video_of(im["file_name"]), "image_id": i, "ann_id": a["id"],
                   "cat": a["category_id"], "label": label[a["category_id"]], "bbox": a["bbox"], "match": None}
            if e is not None:
                vm[km] = stored_crop(img, e["box"], cfg)
                row["match"] = {"k": km, "box": e["box"], "s": e["s"].tolist()}
                km += 1
            vg_idx.append(row)
            kg += 1
        for (_, j, e) in ents_by_img.get(i, []):
            ve[ke] = stored_crop(img, e["box"], cfg)
            ve_idx.append({"k": ke, "video": video_of(im["file_name"]), "image_id": i, "box": e["box"],
                           "s": e["s"].tolist()})
            ke += 1
        for _ in range(bgv_pick.get(i, 0)):
            p = _bg_window(img, v_anns[i], rng, cfg)
            if p is not None:
                bgv[kbv] = p
                kbv += 1
        if n_done % 500 == 0:
            log(f"  val frames {n_done}/{len(v_ids)}  {time.time() - t0:.0f}s")
    for arr in (tr, lvp, bgt, vg, vm, ve, bgv):
        arr.flush()
    index = {"version": VERSION, "cfg": cfg, "cat_ids": cat_ids, "names": {str(c): names[c] for c in cat_ids},
             "lv_label": label[lv_id] if lv_id is not None else None,
             "train": idx, "lv": lv_idx, "n_bg_train": kb, "bg_train_videos": sorted({m["video"] for m in bg_t_meta}),
             "bg_train_meta": bg_t_meta, "vessel_sqrt_area": vessel_sqrt,
             "val_gt": vg_idx, "val_ent": ve_idx, "n_bg_val": kbv, "eval_subset": sorted(sub & set(v_ids)),
             "seconds": time.time() - t0, "n_train_frames": len(t_list)}
    (out_dir / "index.json").write_text(json.dumps(index), encoding="utf-8")
    log(f"  cache: train {k} crops ({kl} LV patches, {kb} bg), val GT {kg} ({km} R0-matched), "
        f"val entities {ke}, bg val {kbv}; {time.time() - t0:.0f}s")
    return index


def _bg_window(img, anns, rng, cfg):
    """Native background window near the horizon band, not overlapping any GT box."""
    H, W = img.shape[:2]
    P = cfg["bg_patch_px"]
    if H < P or W < P:
        return None
    for _ in range(30):
        cx = rng.uniform(P / 2, W - P / 2)
        cy = rng.uniform(*cfg["bg_y_range"]) * H
        cy = min(max(cy, P / 2), H - P / 2)
        box = [cx - P / 2, cy - P / 2, P, P]
        if all(iou_xywh(box, a["bbox"]) == 0.0 for a in anns):
            x0, y0 = int(cx - P / 2), int(cy - P / 2)
            return img[y0:y0 + P, x0:x0 + P].copy()
    return None


# =============================================================================
# Torch parts
# =============================================================================
def torch_mods():
    import torch
    import torch.nn as nn
    return torch, nn


class TrainSet:
    """Picklable (Windows workers): memmaps opened lazily in each worker."""

    def __init__(self, cache_dir, rows, lv_rows, n_bg, lv_label, vessel_sqrt, cfg, seed, holdout=None):
        self.cache_dir, self.rows, self.lv_rows = str(cache_dir), rows, lv_rows
        self.n_bg, self.lv_label, self.vessel = n_bg, lv_label, np.asarray(vessel_sqrt)
        self.cfg, self.seed, self.mm = cfg, seed, None

    def _open(self):
        if self.mm is None:
            d = Path(self.cache_dir)
            self.mm = {"tr": np.load(d / "train_crops.npy", mmap_mode="r"),
                       "lv": np.load(d / "lv_patches.npy", mmap_mode="r"),
                       "bg": np.load(d / "bg_train.npy", mmap_mode="r")}

    def __len__(self):
        return len(self.rows)

    def view(self, i, rng):
        self._open()
        r = self.rows[i]
        if r["label"] == self.lv_label and self.lv_rows and self.n_bg and rng.random() < self.cfg["lv_synth_p"]:
            lr = self.lv_rows[int(rng.integers(len(self.lv_rows)))]
            bg = np.asarray(self.mm["bg"][int(rng.integers(self.n_bg))])
            a = float(self.vessel[int(rng.integers(len(self.vessel)))])
            v = lv_synthetic(np.asarray(self.mm["lv"][lr["k"]]), lr["psize"], lr["wh"], bg, a, rng, self.cfg)
            if v is not None:
                return v, r["label"], 1
        return train_view(np.asarray(self.mm["tr"][r["k"]]), rng, self.cfg), r["label"], 0

    def __getitem__(self, i):
        torch, _ = torch_mods()
        seed = (self.seed * 1_000_003 + int(i) * 7919 + int(torch.initial_seed() % 100000)) % (2 ** 32)
        rng = np.random.default_rng(seed)
        v, y, syn = self.view(i, rng)
        if rng.random() < 0.5:
            v = v[:, ::-1]
        x = torch.from_numpy(np.array(v, dtype=np.uint8, copy=True)).permute(2, 0, 1)
        return x, y, syn


def photometric(x, torch, gen):
    """GPU batch augmentation on uint8 NCHW -> float normalised."""
    x = x.float() / 255.0
    n = x.shape[0]
    dev = x.device
    b = 1 + (torch.rand(n, 1, 1, 1, device=dev, generator=gen) - 0.5) * 0.6
    c = 1 + (torch.rand(n, 1, 1, 1, device=dev, generator=gen) - 0.5) * 0.6
    s = 1 + (torch.rand(n, 1, 1, 1, device=dev, generator=gen) - 0.5) * 0.6
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - m) * c + m
    x = x * b
    g = x.mean(dim=1, keepdim=True)
    x = (x - g) * s + g
    gray = torch.rand(n, 1, 1, 1, device=dev, generator=gen) < 0.1
    x = torch.where(gray, x.mean(dim=1, keepdim=True).expand_as(x), x)
    x = x + torch.randn(x.shape, device=dev, generator=gen) * (torch.rand(n, 1, 1, 1, device=dev, generator=gen) * 0.02)
    return normalise(x.clamp(0, 1), torch)


def normalise(x, torch):
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def build_model(n_classes, arch="convnext_tiny", pretrained=True):
    torch, nn = torch_mods()
    if arch == "tiny":
        return nn.Sequential(nn.Conv2d(3, 16, 5, 2, 2), nn.ReLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
                             nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, n_classes))
    import torchvision
    w = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    m = torchvision.models.convnext_tiny(weights=w)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, n_classes)
    return m


def sampler_weights(rows, power):
    cnt = Counter(r["label"] for r in rows)
    return np.array([cnt[r["label"]] ** (power - 1.0) for r in rows], dtype=np.float64)


def train_model(cache_dir, run_dir, cfg=CFG, arch="convnext_tiny", pretrained=True, holdout=None,
                device=None, max_steps=None, num_workers=4, log=print):
    torch, nn = torch_mods()
    from torch.utils.data import DataLoader, WeightedRandomSampler
    cache_dir, run_dir = Path(cache_dir), Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    index = json.loads((cache_dir / "index.json").read_text(encoding="utf-8"))
    rows = [r for r in index["train"] if r["video"] != holdout]
    lv_rows = [r for r in index["lv"] if r["video"] != holdout]
    if holdout:
        bg_ok = [k for k, m in enumerate(index["bg_train_meta"]) if m["video"] != holdout]
        assert len(bg_ok) == index["n_bg_train"], "holdout video must not be a background source"
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    ds = TrainSet(cache_dir, rows, lv_rows, index["n_bg_train"], index["lv_label"], index["vessel_sqrt_area"],
                  cfg, cfg["seed"])
    w = sampler_weights(rows, cfg["balance_power"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    total = cfg["epochs"] * cfg["samples_per_epoch"]
    sampler = WeightedRandomSampler(torch.as_tensor(w), num_samples=total, replacement=True, generator=gen)
    dl = DataLoader(ds, batch_size=cfg["batch"], sampler=sampler, num_workers=num_workers,
                    persistent_workers=num_workers > 0, drop_last=True, pin_memory=device == "cuda")
    n_cls = len(index["cat_ids"])
    model = build_model(n_cls, arch, pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = total // cfg["batch"] if max_steps is None else min(max_steps, total // cfg["batch"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / cfg["warmup_steps"]) *
                                              0.5 * (1 + math.cos(math.pi * min(s, steps) / max(steps, 1))))
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    lossf = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    g = torch.Generator(device=device).manual_seed(cfg["seed"])
    model.train()
    t0, hist, seen_cls, n_syn = time.time(), [], Counter(), 0
    for step, (x, y, syn) in enumerate(dl):
        if step >= steps:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda" if use_amp else "cpu", dtype=torch.float16 if use_amp else torch.bfloat16, enabled=use_amp):
            loss = lossf(model(photometric(x, torch, g)), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        seen_cls.update(y.tolist())
        n_syn += int(syn.sum())
        if step % 50 == 0 or step == steps - 1:
            hist.append({"step": step, "loss": loss.item(), "lr": sched.get_last_lr()[0], "t": time.time() - t0})
            log(f"  step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s")
    ck = {"model": model.state_dict(), "arch": arch, "n_classes": n_cls, "cfg": cfg, "holdout": holdout,
          "version": VERSION, "steps": steps, "cat_ids": index["cat_ids"]}
    torch.save(ck, run_dir / "expert_last.pth")
    info = {"steps": steps, "seconds": time.time() - t0, "history": hist, "class_counts_seen": dict(seen_cls),
            "lv_synthetic_seen": n_syn, "train_rows": len(rows), "holdout": holdout, "arch": arch}
    (run_dir / "train_log.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return model, info


@contextlib.contextmanager
def _quiet():
    with warnings.catch_warnings(record=True) as wl, contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("always")
        yield wl


def predict(model, crops_mm, ks, device, batch=256, views=None):
    """Softmax probabilities on eval views of stored crops (or precomputed views)."""
    torch, _ = torch_mods()
    model.eval()
    out = []
    with torch.inference_mode():
        for s in range(0, len(ks), batch):
            if views is not None:
                v = np.stack(views[s:s + batch])
            else:
                v = np.stack([eval_view(np.asarray(crops_mm[k])) for k in ks[s:s + batch]])
            x = torch.from_numpy(v).permute(0, 3, 1, 2).to(device).float() / 255.0
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.float16 if device == "cuda" else torch.bfloat16, enabled=device == "cuda"):
                p = torch.softmax(model(normalise(x, torch)).float(), dim=1)
            out.append(p.cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 1))


# =============================================================================
# Readouts
# =============================================================================
def bal_acc(y, p, classes):
    rec = [float(np.mean(p[y == c] == c)) for c in classes if np.any(y == c)]
    return float(np.mean(rec)) if rec else float("nan"), rec


def paired_readout(val_rows, probs_match, probs_gt, n_cls, cfg=CFG, rule=PASS_RULE, names=None):
    cov = [r for r in val_rows if r["match"] is not None]
    y = np.array([r["label"] for r in cov])
    r0 = np.array([int(np.argmax(r["match"]["s"])) for r in cov])
    ex = np.array([int(np.argmax(probs_match[r["match"]["k"]])) for r in cov]) if cov else np.array([])
    classes = sorted(set(y.tolist()))
    ba_r0, rec_r0 = bal_acc(y, r0, classes)
    ba_ex, rec_ex = bal_acc(y, ex, classes)
    per_vid = {}
    for v in sorted({r["video"] for r in cov}):
        m = np.array([r["video"] == v for r in cov])
        per_vid[v] = {"n": int(m.sum()), "acc_r0": float(np.mean(r0[m] == y[m])), "acc_expert": float(np.mean(ex[m] == y[m]))}
    counted = [v for v, d in per_vid.items() if d["n"] >= rule["min_video_boxes"]]
    not_worse = sum(1 for v in counted if per_vid[v]["acc_expert"] >= per_vid[v]["acc_r0"])
    conf = lambda pred: np.bincount(y * n_cls + pred, minlength=n_cls * n_cls).reshape(n_cls, n_cls).tolist()
    yg = np.array([r["label"] for r in val_rows])
    eg = np.argmax(probs_gt, axis=1) if len(val_rows) else np.array([])
    ba_gt, _ = bal_acc(yg, eg, sorted(set(yg.tolist())))
    out = {"gt_total": len(val_rows), "gt_covered": len(cov),
           "acc_r0": float(np.mean(r0 == y)) if cov else None, "acc_expert": float(np.mean(ex == y)) if cov else None,
           "bal_acc_r0": ba_r0, "bal_acc_expert": ba_ex, "delta_bal_acc": ba_ex - ba_r0,
           "recall_r0": dict(zip([names[c] if names else c for c in classes], rec_r0)),
           "recall_expert": dict(zip([names[c] if names else c for c in classes], rec_ex)),
           "fixed": int(np.sum((r0 != y) & (ex == y))), "broken": int(np.sum((r0 == y) & (ex != y))),
           "per_video": per_vid, "videos_counted": len(counted), "videos_not_worse": not_worse,
           "confusion_r0": conf(r0) if cov else None, "confusion_expert": conf(ex) if cov else None,
           "bal_acc_expert_on_gt_boxes_all": ba_gt,
           "lv_predicted_on_covered": None}
    out["PASS"] = bool(cov and out["delta_bal_acc"] >= rule["min_delta_bal_acc"] and counted
                       and not_worse >= math.ceil(rule["min_video_share_not_worse"] * len(counted) - 1e-9))
    return out


def rescore_dets(ent_rows, probs, cat_ids, arm, topk=100):
    """arm: A0 raw, E1 obj * normmax(p_expert), E2 obj * normmax(0.5 p_r0 + 0.5 p_expert)."""
    per = defaultdict(list)
    for r in ent_rows:
        s = np.asarray(r["s"])
        obj = float(s.max())
        if arm == "A0":
            sv = s
        else:
            pe = probs[r["k"]] / probs[r["k"]].max()
            q = pe if arm == "E1" else 0.5 * (s / obj) + 0.5 * pe
            sv = obj * q / q.max()
        for i in np.nonzero(sv > 0)[0]:
            per[r["image_id"]].append((float(sv[i]), cat_ids[i], r["box"]))
    dets = []
    for img, c in per.items():
        c.sort(key=lambda t: -t[0])
        dets += [{"image_id": img, "category_id": cid, "bbox": list(b), "score": sc} for sc, cid, b in c[:topk]]
    return dets


def coco_map30(gt_ds, dets, img_ids, groups=None):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    img_ids = set(img_ids)
    ds = {"images": [im for im in gt_ds["images"] if im["id"] in img_ids],
          "annotations": [a for a in gt_ds["annotations"] if a["image_id"] in img_ids],
          "categories": gt_ds["categories"]}
    with _quiet() as wl:
        g = COCO()
        g.dataset = copy.deepcopy(ds)
        g.createIndex()
        dd = [d for d in dets if d["image_id"] in img_ids]
        if not dd:
            return {"map": 0.0, "per_video": {}, "per_class": {}}
        E = COCOeval(g, g.loadRes(copy.deepcopy(dd)), "bbox")
        E.params.iouThrs = np.array([0.30])
        E.evaluate()
        E.accumulate()
        E.summarize()
        res = {"map": float(E.stats[0]), "per_class": {}, "per_video": {}}
        for k, cid in enumerate(E.params.catIds):
            p = E.eval["precision"][0, :, k, 0, 2]
            p = p[p > -1]
            if p.size:
                res["per_class"][str(cid)] = float(p.mean())
        full = E.evalImgs
        I0 = len(E._paramsEval.imgIds)
        for name, ids in (groups or {}).items():
            keep = np.array([i in ids for i in E._paramsEval.imgIds])
            E.evalImgs = [e if keep[n % I0] else None for n, e in enumerate(full)]
            E.accumulate()
            pr = E.eval["precision"][0, :, :, 0, 2]
            v = pr[pr > -1]
            res["per_video"][name] = float(v.mean()) if v.size else -1.0
        E.evalImgs = full
    return res


def lv_check(model, cache_dir, index, holdout, device, cfg=CFG):
    """D3: LV of the held-out video, native and shrunk onto VAL backgrounds."""
    d = Path(cache_dir)
    tr = np.load(d / "train_crops.npy", mmap_mode="r")
    lvp = np.load(d / "lv_patches.npy", mmap_mode="r")
    bgv = np.load(d / "bg_val.npy", mmap_mode="r")
    lvl = index["lv_label"]
    rows = [r for r in index["train"] if r["video"] == holdout and r["label"] == lvl]
    lrows = [r for r in index["lv"] if r["video"] == holdout]
    out = {"holdout": holdout, "n": len(rows)}
    p = predict(model, tr, [r["k"] for r in rows], device)
    out["native_lv_recall"] = float(np.mean(np.argmax(p, 1) == lvl)) if len(rows) else None
    out["native_pred"] = dict(Counter(int(c) for c in np.argmax(p, 1)))
    rng = np.random.default_rng(12345)
    for a in cfg["lv_check_sizes"]:
        views = []
        for lr in lrows:
            v = lv_synthetic(np.asarray(lvp[lr["k"]]), lr["psize"], lr["wh"],
                             np.asarray(bgv[int(rng.integers(index["n_bg_val"]))]), float(a), rng, cfg, jitter=False)
            if v is not None:
                views.append(v)
        if views:
            pp = predict(model, None, list(range(len(views))), device, views=views)
            out[f"shrunk_{a}px_lv_recall"] = float(np.mean(np.argmax(pp, 1) == lvl))
            out[f"shrunk_{a}px_pred"] = dict(Counter(int(c) for c in np.argmax(pp, 1)))
    return out


def run_eval(cache_dir, run_dir, val, device, log=print, ckpt_name="expert_last.pth", arch_override=None, export=None):
    torch, _ = torch_mods()
    cache_dir, run_dir = Path(cache_dir), Path(run_dir)
    index = json.loads((cache_dir / "index.json").read_text(encoding="utf-8"))
    ck = torch.load(run_dir / ckpt_name, map_location="cpu", weights_only=False)
    model = build_model(ck["n_classes"], arch_override or ck["arch"], pretrained=False)
    model.load_state_dict(ck["model"])
    model.to(device)
    cat_ids = index["cat_ids"]
    names = {i: index["names"][str(c)] for i, c in enumerate(cat_ids)}
    vg = np.load(cache_dir / "val_gt_crops.npy", mmap_mode="r")
    vm = np.load(cache_dir / "val_match_crops.npy", mmap_mode="r")
    ve = np.load(cache_dir / "val_ent_crops.npy", mmap_mode="r")
    rows = index["val_gt"]
    n_m = sum(1 for r in rows if r["match"] is not None)
    p_gt = predict(model, vg, [r["k"] for r in rows], device)
    p_m = predict(model, vm, list(range(n_m)), device) if n_m else np.zeros((0, len(cat_ids)))
    prim = paired_readout(rows, p_m, p_gt, len(cat_ids), names=names)
    lvl = index["lv_label"]
    if lvl is not None and n_m:
        prim["lv_predicted_on_covered"] = int(np.sum(np.argmax(p_m, 1) == lvl))
    ent = index["val_ent"]
    p_e = predict(model, ve, [r["k"] for r in ent], device) if ent else np.zeros((0, len(cat_ids)))
    groups = defaultdict(set)
    for r in ent:
        groups[r["video"]].add(r["image_id"])
    sub = index["eval_subset"]
    sec = {}
    for arm in ("A0", "E1", "E2"):
        d = rescore_dets(ent, p_e, cat_ids, arm, CFG["topk"])
        sec[arm] = coco_map30(val, d, sub, groups)
        if arm != "A0":
            sec[arm]["delta"] = sec[arm]["map"] - sec["A0"]["map"]
            sec[arm]["videos_positive"] = sum(1 for v in sec["A0"]["per_video"]
                                              if sec[arm]["per_video"][v] > sec["A0"]["per_video"][v])
    if export is not None:
        raw = coco_map30(val, export, sub)["map"]
        sec["parity_raw_export_432"] = raw
        sec["parity_ok"] = abs(raw - sec["A0"]["map"]) < 1e-9
        sec["raw_export_full_val_map30"] = coco_map30(val, export, [im["id"] for im in val["images"]])["map"]
        if not sec["parity_ok"]:
            raise SystemExit(f"PARITY FAILED: rebuilt A0 {sec['A0']['map']:.6f} != raw export {raw:.6f}")
    lv_ent = int(np.sum((np.argmax(p_e, 1) == lvl) & (np.array([max(r["s"]) for r in ent]) >= 0.3))) if ent and lvl is not None else None
    res = {"version": VERSION, "ckpt": str(run_dir / ckpt_name), "holdout": ck.get("holdout"), "primary": prim,
           "secondary_map30_432": sec, "lv_guard": {"lv_argmax_on_confident_val_entities": lv_ent},
           "n_eval_frames": len(sub)}
    if ck.get("holdout"):
        res["lv_check"] = lv_check(model, cache_dir, index, ck["holdout"], device)
    return res, names


def report_md(res, names):
    p, s = res["primary"], res["secondary_map30_432"]
    L = [f"# G2 results ({VERSION})", "", f"Checkpoint: `{res['ckpt']}`  holdout: {res['holdout']}", "",
         "## Primary: paired class decision on R0-covered VAL GT boxes", "",
         f"GT {p['gt_total']}, covered {p['gt_covered']}", "",
         "| | R0 | Expert |", "|---|---|---|",
         f"| accuracy | {p['acc_r0']:.4f} | {p['acc_expert']:.4f} |",
         f"| balanced accuracy | {p['bal_acc_r0']:.4f} | {p['bal_acc_expert']:.4f} |", "",
         f"Δ balanced accuracy {p['delta_bal_acc']:+.4f}; fixed {p['fixed']}, broken {p['broken']}; "
         f"videos not worse {p['videos_not_worse']}/{p['videos_counted']}", "",
         "| Class | recall R0 | recall expert |", "|---|---|---|"]
    for c in p["recall_r0"]:
        L.append(f"| {c} | {p['recall_r0'][c]:.4f} | {p['recall_expert'][c]:.4f} |")
    L += ["", "| Video | n | acc R0 | acc expert |", "|---|---|---|---|"]
    for v, d in p["per_video"].items():
        L.append(f"| {v} | {d['n']} | {d['acc_r0']:.4f} | {d['acc_expert']:.4f} |")
    L += ["", f"Expert balanced accuracy on all GT boxes (oracle boxes, incl. not covered): {p['bal_acc_expert_on_gt_boxes_all']:.4f}",
          f"LV predicted on covered boxes (all wrong on VAL): {p['lv_predicted_on_covered']}", "",
          f"**PRIMARY VERDICT: {'PASS' if p['PASS'] else 'FAIL'}**", "",
          f"## Secondary: mAP30 on {res['n_eval_frames']} VAL frames (stride 10, offset 0), R0 entities rescored", "",
          "| Arm | mAP30 | Δ | videos + |", "|---|---|---|---|"]
    for a in ("A0", "E1", "E2"):
        L.append(f"| {a} | {s[a]['map']:.4f} | {s[a].get('delta', 0):+.4f} | {s[a].get('videos_positive', '–')} |")
    L += ["", "| Video | A0 | E1 | E2 |", "|---|---|---|---|"]
    for v in s["A0"]["per_video"]:
        L.append(f"| {v} | " + " | ".join(f"{s[a]['per_video'].get(v, float('nan')):.4f}" for a in ("A0", "E1", "E2")) + " |")
    if "parity_ok" in s:
        L += ["", f"Parity: rebuilt A0 = raw export on these frames: {s['parity_ok']} ({s['parity_raw_export_432']:.6f}); "
              f"raw export full-VAL mAP30 {s['raw_export_full_val_map30']:.6f}"]
    L += ["", f"LV guard: {json.dumps(res['lv_guard'])}"]
    if "lv_check" in res:
        L += ["", f"LV held-out check (D3, descriptive): {json.dumps(res['lv_check'])}"]
    return "\n".join(L) + "\n"


# =============================================================================
# Real-file gates
# =============================================================================
def load_inputs(need_export=True):
    for p in [ZIP_PATH, TRAIN_JSON, VAL_JSON] + ([R0_EXPORT] if need_export else []):
        if not p.exists():
            sys.exit(f"missing {p}")
    train = json.loads(TRAIN_JSON.read_text(encoding="utf-8"))
    val = json.loads(VAL_JSON.read_text(encoding="utf-8"))
    export = json.loads(R0_EXPORT.read_text(encoding="utf-8")) if need_export else []
    g = {"train_images": len(train["images"]) == EXPECT["train_images"],
         "train_anns": len(train["annotations"]) == EXPECT["train_anns"],
         "val_images": len(val["images"]) == EXPECT["val_images"],
         "val_anns": len(val["annotations"]) == EXPECT["val_anns"]}
    if need_export:
        g["export_dets"] = len(export) == EXPECT["export_dets"]
        g["export_100_per_image"] = max(Counter(d["image_id"] for d in export).values()) <= 100
        vimgs = {im["id"]: im for im in val["images"]}
        sub = eval_subset_ids(vimgs)
        g["eval_frames_432"] = len(sub) == EXPECT["eval_frames"]
        g["eval_gt_1386"] = sum(1 for a in val["annotations"] if a["image_id"] in set(sub)) == EXPECT["eval_gt"]
    print("  gates:", g)
    if not all(g.values()):
        sys.exit("INPUT GATE FAILED -- stop")
    return train, val, export


def smoke_ids(train, val):
    """A few real frames: 4 frames from each of 8 TRAIN videos (both LV videos
    included) and 20 VAL frames (10 from the offset-0 subset)."""
    t = defaultdict(list)
    for im in train["images"]:
        t[video_of(im["file_name"])].append(im)
    vids = sorted(t)
    pick_v = [v for v in vids if v in ("video_18", "video_23")] + [v for v in vids if v not in ("video_18", "video_23")][:6]
    tids = []
    for v in pick_v:
        lst = sorted(t[v], key=lambda x: (x.get("frame_id", x["id"]), x["id"]))
        tids += [im["id"] for im in lst[:8:2]]
    vimgs = {im["id"]: im for im in val["images"]}
    sub = eval_subset_ids(vimgs)
    vids_val = sorted(sub[::max(1, len(sub) // 10)][:10] + [im["id"] for im in val["images"][:10]])
    return tids, vids_val


# =============================================================================
# Selftest (synthetic, CPU)
# =============================================================================
def _synth_world(tmp):
    """Images where class = colour of a rectangle; zip + jsons in the real schema."""
    from PIL import Image
    rng = np.random.default_rng(0)
    names = ["ANIMAL", "BOAT", "LEISURE_VEHICLE", "MARINE_MARKER", "OBJECT", "SAILING_VESSEL", "SHIP"]
    cats = [{"id": i + 1, "name": n} for i, n in enumerate(names)]
    colours = np.array([[255, 0, 0], [0, 255, 0], [255, 255, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255], [0, 0, 0]])
    zpath = Path(tmp) / "SeaClips.zip"
    jsons = {}
    with zipfile.ZipFile(zpath, "w") as z:
        for split, nv, nf in (("train", 5, 12), ("val", 3, 20)):
            images, anns, aid = [], [], 1
            for v in range(nv):
                vid = f"video_{v + (18 if split == 'train' else 1):02d}"
                for f in range(nf):
                    iid = len(images) + 1
                    H, W = 300, 400
                    img = np.full((H, W, 3), (40, 80, 120), np.uint8) + rng.integers(0, 20, (H, W, 3)).astype(np.uint8)
                    for c in range(7):
                        if split == "val" and names[c] == "LEISURE_VEHICLE":
                            continue
                        if split == "train" and names[c] == "LEISURE_VEHICLE" and v > 1:
                            continue
                        w, h = (30, 40) if names[c] == "LEISURE_VEHICLE" else (int(rng.integers(10, 30)), int(rng.integers(10, 20)))
                        x, y = float(rng.uniform(0, W - w)), float(rng.uniform(60, 150))
                        img[int(y):int(y) + h, int(x):int(x) + w] = colours[c]
                        anns.append({"id": aid, "image_id": iid, "category_id": c + 1, "bbox": [x, y, float(w), float(h)],
                                     "area": float(w * h), "iscrowd": 0})
                        aid += 1
                    fn = f"{vid}/{f + 1:06d}.png"
                    b = io.BytesIO()
                    Image.fromarray(img).save(b, format="PNG")
                    z.writestr(f"SeaClips/{split}/{fn}", b.getvalue())
                    images.append({"id": iid, "file_name": fn, "height": H, "width": W, "video_id": v + 1, "frame_id": f})
            jsons[split] = {"images": images, "annotations": anns, "categories": cats}
    # synthetic R0 export: each VAL GT found; class wrong for ~40%; plus junk
    export = []
    for a in jsons["val"]["annotations"]:
        b = [a["bbox"][0] + 1, a["bbox"][1], a["bbox"][2], a["bbox"][3]]
        wrong = rng.random() < 0.4
        c = (a["category_id"] % 7) + 1 if wrong else a["category_id"]
        export.append({"image_id": a["image_id"], "category_id": c, "bbox": b, "score": 0.8})
        export.append({"image_id": a["image_id"], "category_id": a["category_id"] if wrong else (a["category_id"] % 7) + 1,
                       "bbox": b, "score": 0.1})
    for im in jsons["val"]["images"]:
        export.append({"image_id": im["id"], "category_id": 2, "bbox": [5.0, 5.0, 8.0, 8.0], "score": 0.05})
    return zpath, jsons["train"], jsons["val"], export, colours


def selftest():
    import tempfile
    print(f"[selftest] {VERSION}")
    # geometry
    img = np.zeros((100, 120, 3), np.uint8)
    img[40:50, 50:70] = 255
    c = extract_square(img, 60, 45, 20, 20)
    assert c.shape == (20, 20, 3) and c[5:15, 0:20].min() == 255 and c[0:5].max() == 0
    c = extract_square(img, 2, 2, 20, 20)            # padding with mean colour
    assert c[0, 0, 0] == int(img.reshape(-1, 3).mean(0)[0])
    assert context_side(5, 3) == 64.0 and context_side(30, 10) == 120.0 and context_side(400, 10) == 1024.0
    st = stored_crop(img, [50, 40, 20, 10])
    ev = eval_view(st)
    assert st.shape == (192, 192, 3) and ev.shape == (128, 128, 3)
    # box (20 px wide) in an 80 px context window -> 32 px in the 128 view, centred
    col = ev[64].astype(int)
    on = np.nonzero(col > 128)[0]
    assert abs(on.min() - 48) <= 2 and abs(on.max() - 79) <= 2, (on.min(), on.max())
    print("  [ok] crop geometry: context side, padding, centring, eval view scale")
    # LV shrink/paste: object ends at the expected size
    lvimg = np.zeros((400, 400, 3), np.uint8)
    lvimg[180:240, 180:220] = 250                     # 40 x 60 LV
    patch, ps = lv_patch(lvimg, [180, 180, 40, 60])
    bg = np.zeros((256, 256, 3), np.uint8)
    rng = np.random.default_rng(0)
    v = lv_synthetic(patch, ps, (40, 60), bg, 16.0, rng, jitter=False)
    f = 16.0 / math.sqrt(2400)
    S = context_side(40 * f, 60 * f)
    exp_h = 60 * f * 128 / S
    rows_on = np.nonzero(v[:, 64, 0] > 125)[0]
    assert abs((rows_on.max() - rows_on.min() + 1) - exp_h) <= 3, (rows_on.max() - rows_on.min() + 1, exp_h)
    assert lv_synthetic(patch, ps, (40, 60), bg, 1000.0, rng, jitter=False) is not None  # never enlarged
    print(f"  [ok] LV shrink+paste: object height {rows_on.max() - rows_on.min() + 1}px vs expected {exp_h:.1f}px")
    # eval subset function
    ims = {i: {"id": i, "video_id": 1 + (i > 25), "frame_id": i} for i in range(1, 51)}
    assert eval_subset_ids(ims) == [1, 11, 21, 26, 36, 46]
    # sampler: class share ~ n ** 0.5
    rows = [{"label": 0}] * 100 + [{"label": 1}] * 400
    w = sampler_weights(rows, 0.5)
    share = w[:100].sum() / w.sum()
    assert abs(share - 10 / 30) < 1e-9
    print("  [ok] offset-0 subset, sampler shares")
    # paired readout logic by hand
    rows = [{"label": 0, "video": "v", "match": {"k": 0, "s": [0.9, 0.1]}},
            {"label": 1, "video": "v", "match": {"k": 1, "s": [0.8, 0.2]}},
            {"label": 1, "video": "v", "match": None}]
    pm = np.array([[0.9, 0.1], [0.1, 0.9]])
    pr = paired_readout(rows, pm, np.array([[1, 0], [0, 1], [0, 1.0]]), 2, rule=dict(PASS_RULE, min_video_boxes=1))
    assert pr["acc_r0"] == 0.5 and pr["acc_expert"] == 1.0 and pr["fixed"] == 1 and pr["broken"] == 0
    assert abs(pr["delta_bal_acc"] - 0.5) < 1e-12 and pr["PASS"] and pr["gt_covered"] == 2
    print("  [ok] paired readout (accuracy, balanced accuracy, fix/break, coverage)")
    # rescoring identity: expert probs = R0's normalised scores -> E1 == A0
    ent = [{"k": 0, "image_id": 1, "box": [0, 0, 5, 5], "s": [0.6, 0.3, 0.0]},
           {"k": 1, "image_id": 1, "box": [9, 9, 5, 5], "s": [0.0, 0.2, 0.1]}]
    pid = np.array([np.array(e["s"]) / max(e["s"]) for e in ent])
    a0 = sorted((d["category_id"], d["score"]) for d in rescore_dets(ent, pid, [1, 2, 3], "A0"))
    e1 = sorted((d["category_id"], round(d["score"], 12)) for d in rescore_dets(ent, pid, [1, 2, 3], "E1"))
    assert [(c, round(s, 12)) for c, s in a0] == e1
    print("  [ok] rescoring identity (E1 with R0's own scores == A0)")

    # end-to-end on a synthetic world (tiny CNN, CPU)
    with tempfile.TemporaryDirectory() as tmp:
        zpath, train, val, export, _ = _synth_world(tmp)
        cfg = dict(CFG, n_bg_train=40, n_bg_val=20, epochs=1, samples_per_epoch=4800, batch=32, warmup_steps=5, lr=3e-3,
                   bg_patch_px=96, bg_y_range=(0.8, 0.85))
        idx = build_cache(train, val, export, zpath, Path(tmp) / "cache", cfg, log=lambda *a: None)
        assert len(idx["train"]) == sum(1 for a in train["annotations"]
                                        if a["image_id"] in {im["id"] for im in train["images"] if im["frame_id"] % 2 == 0})
        assert idx["lv"] and all(r["video"] in ("video_18", "video_19") for r in idx["lv"])
        assert idx["n_bg_train"] > 0 and idx["n_bg_val"] > 0
        assert "video_18" not in idx["bg_train_videos"] and "video_19" not in idx["bg_train_videos"]
        print(f"  [ok] cache: {len(idx['train'])} train crops, {len(idx['lv'])} LV patches, "
              f"{idx['n_bg_train']} bg from non-LV videos, {len(idx['val_gt'])} val GT, {len(idx['val_ent'])} entities")
        model, info = train_model(Path(tmp) / "cache", Path(tmp) / "run", cfg, arch="tiny", pretrained=False,
                                  device="cpu", num_workers=0, log=lambda *a: None)
        assert info["lv_synthetic_seen"] > 0
        first, last = info["history"][0]["loss"], info["history"][-1]["loss"]
        assert last < first, (first, last)
        res, names = run_eval(Path(tmp) / "cache", Path(tmp) / "run", val, "cpu", export=export)
        assert res["secondary_map30_432"]["parity_ok"]
        p = res["primary"]
        assert p["gt_covered"] == p["gt_total"] and p["acc_r0"] < 0.75
        assert p["acc_expert"] > p["acc_r0"], (p["acc_expert"], p["acc_r0"])
        assert res["secondary_map30_432"]["E1"]["map"] > res["secondary_map30_432"]["A0"]["map"]
        report_md(res, names)
        print(f"  [ok] train (loss {first:.3f} -> {last:.3f}, {info['lv_synthetic_seen']} synthetic LV seen); "
              f"eval: acc R0 {p['acc_r0']:.3f} -> expert {p['acc_expert']:.3f}; "
              f"mAP A0 {res['secondary_map30_432']['A0']['map']:.3f} -> E1 {res['secondary_map30_432']['E1']['map']:.3f}")
        # holdout: video_19 LV excluded from training rows and LV patches; lv_check runs
        cfg2 = dict(cfg, samples_per_epoch=320)
        train_model(Path(tmp) / "cache", Path(tmp) / "run_ho", cfg2, arch="tiny", pretrained=False, holdout="video_19",
                    device="cpu", num_workers=0, log=lambda *a: None)
        res2, _ = run_eval(Path(tmp) / "cache", Path(tmp) / "run_ho", val, "cpu")
        lc = res2["lv_check"]
        assert lc["n"] > 0 and "shrunk_8px_lv_recall" in lc
        print(f"  [ok] LV held-out check runs (n={lc['n']}, keys {sorted(k for k in lc if 'recall' in k)})")
    print("SELFTEST PASS")


# =============================================================================
# Smoke (GPU, a few real images)
# =============================================================================
def smoke():
    torch, nn = torch_mods()
    print(f"[smoke] {VERSION}")
    assert torch.cuda.is_available(), "CUDA not available"
    train, val, export = load_inputs()
    tids, vids = smoke_ids(train, val)
    t0 = time.time()
    idx = build_cache(train, val, export, ZIP_PATH, SMOKE_CACHE_DIR, CFG, train_image_ids=tids, val_image_ids=vids,
                      n_bg_train=60, n_bg_val=20)
    t_build = time.time() - t0
    n_fr = idx["n_train_frames"] + len(vids)
    sec_per_frame = t_build / max(n_fr, 1)
    assert idx["lv"], "smoke cache has no LV patches"
    # 1) overfit test: 64 fixed crops, pretrained ConvNeXt-T, no augmentation
    dev = "cuda"
    tr = np.load(SMOKE_CACHE_DIR / "train_crops.npy", mmap_mode="r")
    rows = idx["train"][:64]
    x = torch.from_numpy(np.stack([eval_view(np.asarray(tr[r["k"]])) for r in rows])).permute(0, 3, 1, 2).to(dev)
    y = torch.tensor([r["label"] for r in rows], device=dev)
    torch.manual_seed(0)
    m = build_model(len(idx["cat_ids"]), "convnext_tiny", pretrained=True).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-4)
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.reset_peak_memory_stats()
    m.train()
    for it in range(150):
        with torch.autocast("cuda", dtype=torch.float16):
            loss = nn.functional.cross_entropy(m(normalise(x.float() / 255, torch)), y)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    m.eval()
    with torch.inference_mode():
        acc = float((m(normalise(x.float() / 255, torch)).argmax(1) == y).float().mean())
    print(f"  overfit 64 crops / 150 steps: train acc {acc:.3f}, loss {loss.item():.4f}")
    # 2) throughput with the real training loop (augmentation + workers)
    cfg = dict(CFG, epochs=1, samples_per_epoch=CFG["batch"] * 60, warmup_steps=10)
    torch.cuda.reset_peak_memory_stats()
    run = RUN_DIR / "smoke"
    _, info = train_model(SMOKE_CACHE_DIR, run, cfg, device=dev, num_workers=4)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    t_step = info["history"][-1]["t"] / max(info["history"][-1]["step"], 1)
    # 3) end-to-end eval on the smoke cache (numbers meaningless; exercises the code)
    res, names = run_eval(SMOKE_CACHE_DIR, run, val, dev, export=[d for d in export if d["image_id"] in set(vids)])
    (run / "smoke_eval_tables.md").write_text(report_md(res, names), encoding="utf-8")
    full_train_frames = math.ceil(EXPECT["train_images"] / CFG["train_frame_stride"])
    proj = {"build_sec_per_frame": sec_per_frame,
            "build_full_minutes_est": sec_per_frame * (full_train_frames + EXPECT["val_images"]) / 60,
            "train_sec_per_step": t_step,
            "train_full_minutes_est": t_step * CFG["epochs"] * CFG["samples_per_epoch"] / CFG["batch"] / 60,
            "peak_gpu_mem_gb": peak, "overfit_train_acc": acc, "lv_synthetic_seen": info["lv_synthetic_seen"],
            "class_counts_seen": info["class_counts_seen"],
            "cache_full_gb_est": (EXPECT["train_anns"] / CFG["train_frame_stride"] + EXPECT["val_anns"] * 1.7 + 20000)
                                 * CFG["store_px"] ** 2 * 3 / 2 ** 30
                                 + (CFG["n_bg_train"] + CFG["n_bg_val"]) * CFG["bg_patch_px"] ** 2 * 3 / 2 ** 30}
    ok = {"overfit_acc>=0.9": acc >= 0.9, "peak_mem<10GB": peak < 10, "lv_synth_seen": info["lv_synthetic_seen"] > 0,
          "eval_ran": "primary" in res}
    out = {"version": VERSION, "projections": proj, "checks": ok, "smoke_train_frames": idx["n_train_frames"],
           "smoke_val_frames": len(vids)}
    (run / "smoke_report.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    print("SMOKE PASS" if all(ok.values()) else "SMOKE FAIL")


# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["selftest", "smoke", "build", "train", "eval"], required=True)
    ap.add_argument("--holdout-video", default=None, help="D3 LV check, e.g. video_23")
    ap.add_argument("--run-name", default="main")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.mode == "selftest":
        return selftest()
    if a.mode == "smoke":
        return smoke()
    torch, _ = torch_mods()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.mode == "build":
        if (CACHE_DIR / "index.json").exists():
            sys.exit(f"{CACHE_DIR} already holds a cache; refusing to overwrite (delete it yourself if intended)")
        train, val, export = load_inputs()
        print(f"  shas: train {sha256_file(TRAIN_JSON)}\n        val {sha256_file(VAL_JSON)}\n"
              f"        export {sha256_file(R0_EXPORT)}")
        build_cache(train, val, export, ZIP_PATH, CACHE_DIR, CFG)
        return
    run = RUN_DIR / (a.run_name + (f"_holdout_{a.holdout_video}" if a.holdout_video else ""))
    if a.mode == "train":
        if (run / "expert_last.pth").exists():
            sys.exit(f"{run} already has a checkpoint; refusing to overwrite")
        if a.holdout_video:
            idx = json.loads((CACHE_DIR / "index.json").read_text(encoding="utf-8"))
            if a.holdout_video not in {r["video"] for r in idx["lv"]}:
                sys.exit(f"{a.holdout_video} is not an LV video")
        train_model(CACHE_DIR, run, CFG, device=dev, holdout=a.holdout_video, num_workers=a.workers)
        return
    if a.mode == "eval":
        _, val, export = load_inputs()
        res, names = run_eval(CACHE_DIR, run, val, dev, export=export)
        res["script_sha"] = sha256_file(Path(__file__))
        (run / "g2_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        md = report_md(res, names)
        (run / "g2_results_tables.md").write_text(md, encoding="utf-8")
        print(md)
        print(f"  wrote {run / 'g2_results.json'}\n  wrote {run / 'g2_results_tables.md'}")


if __name__ == "__main__":
    main()
