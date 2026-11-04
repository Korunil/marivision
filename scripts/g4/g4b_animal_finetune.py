r"""
G4b: can a one-epoch fine-tune of G4F epoch 15 restore ANIMAL without losing the tile gains?

Location: D:/MARIVISION/R0_frozen_baseline/g4b_animal_finetune.py  (run FROM that folder, next to g4_pilot.py and
g4_full.py, both SHA-checked).  R0_MSDA_BACKEND=pytorch, HF_HUB_DISABLE_SYMLINKS_WARNING=1.

Arms (identical except the named factor; all start from G4F g4f_epoch15.pth, fp32, fresh AdamW at 0.1 x the R0 LRs
= 2e-5 core / 2e-6 backbone / 2e-6 linear_proj, 10,155 steps at batch 2, seed 42, every arm the same base frame
permutation):
  CTRL     the G4F recipe unchanged (M2 tiles, p_tile 0.5)          -> separates "one more epoch" from the fixes
  NOTILE   full frames only (p_tile 0)                              -> M2 armF/armT pair: no tiles kept ANIMAL
  RFS      p_tile 0.5 + ANIMAL repeat-factor sampling of frames (LVIS rule r = max(1, sqrt(t / f)), t = 0.3,
           f = share of TRAIN frames containing ANIMAL, stochastic rounding) + an ANIMAL frame that is tiled takes a
           window in which an ANIMAL box is >= 50 % visible (uniform among them)
Inference / scoring: fp32, 17-tile V0 merge (and plain), on the 432 offset-0 and 432 offset-5 VAL frames.

Modes: selftest | smoke | train --arm A | export --arm A --offset 0|5 | score
"""
from PIL import Image  # noqa: F401  -- PIL before torch (Director's Windows env)
import sys
if "wandb" not in sys.modules:
    sys.modules["wandb"] = None
import argparse
import contextlib
import copy
import gzip
import hashlib
import io
import json
import math
import os
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

VERSION = "G4B.0"
PILOT_SHA = "3128fbf2985c7aa26fc5d38ec334e5cc67ac8281947c8b2da5932e31bdcd496b"
FULL_SHA = "c1fbed7f03bbfaac525c422a00b85a436d582116d4b1b2e72d2d69472021479d"
ANIMAL = 1
ARMS = {"CTRL": {"p_tile": 0.5, "rfs": False}, "NOTILE": {"p_tile": 0.0, "rfs": False},
        "RFS": {"p_tile": 0.5, "rfs": True}}
RECIPE = {"steps": 10155, "batch": 2, "seed": 42, "lr_factor": 0.1, "rfs_t": 0.3, "min_visible": 0.5,
          "workers": 2, "log_every": 500, "export_batch": 4}
RULE = {"animal_ap_floor": 0.05, "animal_ap_gain_vs_ctrl": 0.03, "others_guard": -0.01}

ROOT = Path("D:/MARIVISION")
E15 = ROOT / "G4_full/run/g4f_epoch15.pth"
BASE_CACHE = {0: ROOT / "G4_full/run/tile_cache/r0_tile_cache_g4f_e15_stride10_offset0.json.gz",
              5: ROOT / "G4_full/run/tile_cache/r0_tile_cache_g4f_e15_stride10_offset5.json.gz"}
R0_CACHE = {0: ROOT / "R0_frozen_baseline/m1_1_tile_cache/r0_tile_cache_epoch15_stride10_offset0.json.gz",
            5: ROOT / "R0_frozen_baseline/m1_1_tile_cache/r0_tile_cache_epoch15_stride10_offset5.json.gz"}
OUT = ROOT / "G4b"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


# ---- module-level dataset subclass (Windows spawn workers must unpickle it by name) --------------------------------
sys.path.insert(0, str(Path.cwd()))
try:
    import m2_train_r0_finetune as _M2
    _BASE = _M2.get_dataset_class() if getattr(_M2, "_R0Dataset", None) is not None else None
except Exception:  # noqa: BLE001  -- outside the R0 folder (nothing to subclass)
    _M2, _BASE = None, None

if _BASE is not None:
    class AnimalAwareTiles(_BASE):
        """M2 tile dataset; when a frame with ANIMAL is tiled, the window is drawn among those keeping an ANIMAL box."""

        def animal_windows(self, idx):
            import torch  # noqa: F401
            img_id = self.img_ids[idx]
            info = self.coco.imgs[img_id]
            anns = [a for a in self.coco.imgToAnns.get(img_id, []) if a["category_id"] == ANIMAL]
            if not anns:
                return []
            wins = self.windows(int(info["width"]), int(info["height"]))
            keep = []
            for k, (x0, y0, x1, y1) in enumerate(wins):
                for a in anns:
                    x, y, w, h = [float(v) for v in a["bbox"]]
                    iw = max(0.0, min(x + w, x1) - max(x, x0))
                    ih = max(0.0, min(y + h, y1) - max(y, y0))
                    if w > 0 and h > 0 and iw * ih >= self.min_visible * w * h:
                        keep.append(k)
                        break
            return keep

        def __getitem__(self, idx):
            import torch
            if self.p_tile > 0.0 and torch.rand(1).item() < self.p_tile:
                wk = self.animal_windows(idx)
                if wk:
                    return self.tile_item(idx, win_index=wk[int(torch.randint(len(wk), (1,)).item())])
                return self.tile_item(idx)
            img, target = _BASE.__bases__[0].__getitem__(self, idx)
            target["m2_is_tile"] = torch.tensor([0])
            target["m2_dropped"] = torch.tensor([0])
            return img, target


def load_modules(check=True):
    here = Path.cwd()
    for f, s in (("g4_pilot.py", PILOT_SHA), ("g4_full.py", FULL_SHA)):
        if not (here / f).exists():
            raise SystemExit(f"[REFUSED] {f} must sit in the run folder ({here})")
        if check and sha256_file(here / f) != s:
            raise SystemExit(f"[REFUSED] {f} sha {sha256_file(here / f)} != {s}")
    import g4_pilot as G
    import g4_full as F
    return G, F


# =============================================================================
# sampling
# =============================================================================
def animal_flags(ds):
    return [any(a["category_id"] == ANIMAL for a in ds.coco.imgToAnns.get(i, [])) for i in ds.img_ids]


def epoch_order(n_items, flags, arm, recipe=RECIPE):
    """Base permutation (identical for all arms); RFS repeats ANIMAL frames r times (stochastic rounding), then
    shuffles; every arm is cut / cycled to exactly steps x batch samples."""
    import torch
    n_samples = recipe["steps"] * recipe["batch"]
    g = torch.Generator().manual_seed(recipe["seed"] * 1000 + 99)
    base = torch.randperm(n_items, generator=g).tolist()
    info = {"n_items": n_items, "animal_frames": int(sum(flags))}
    if ARMS[arm]["rfs"]:
        f = sum(flags) / n_items
        r = max(1.0, math.sqrt(recipe["rfs_t"] / f)) if f > 0 else 1.0
        info.update({"animal_frame_share": f, "repeat_factor": r})
        g2 = torch.Generator().manual_seed(recipe["seed"] * 1000 + 98)
        pool = []
        for i in base:
            k = 1
            if flags[i]:
                k = int(math.floor(r)) + int(torch.rand(1, generator=g2).item() < r - math.floor(r))
            pool += [i] * k
        perm = torch.randperm(len(pool), generator=g2).tolist()
        base = [pool[j] for j in perm]
    order = (base * (n_samples // len(base) + 1))[:n_samples]
    info["animal_samples"] = int(sum(flags[i] for i in order))
    info["animal_sample_share"] = info["animal_samples"] / n_samples
    return order, info


class ListSampler:
    def __init__(self, order):
        self.order = order

    def __iter__(self):
        return iter(self.order)

    def __len__(self):
        return len(self.order)


def make_dataset(M2, arm, zip_path, train_json, pipeline_dir, recipe=RECIPE):
    with contextlib.redirect_stdout(io.StringIO()):
        if ARMS[arm]["rfs"]:
            if _BASE is None:
                raise SystemExit("[REFUSED] run from the R0 folder")
            return AnimalAwareTiles(Path(zip_path), Path(train_json), p_tile=ARMS[arm]["p_tile"],
                                    min_visible=recipe["min_visible"], pipeline_dir=str(pipeline_dir), random_flip=True)
        DS = M2.get_dataset_class()
        return DS(Path(zip_path), Path(train_json), p_tile=ARMS[arm]["p_tile"], min_visible=recipe["min_visible"],
                  pipeline_dir=str(pipeline_dir), random_flip=True)


# =============================================================================
# training
# =============================================================================
def train_arm(G, F, P, M2, arm, device, out_dir, init_ckpt, recipe=RECIPE, n_steps=None, zip_path=None,
              train_json=None, pipeline_dir=None, workers=None, log=print, selftest_downscale=None, timm_override=None):
    import torch
    from torch.utils.data import DataLoader
    from util.misc import collate_fn
    zip_path, train_json = zip_path or G.ZIP, train_json or G.TRAIN_JSON
    pipeline_dir = pipeline_dir or G.PIPELINE_DIR
    n_steps = n_steps or recipe["steps"]
    rec = dict(recipe, steps=n_steps)
    final = out_dir / f"g4b_{arm}.pth"
    if final.exists():
        raise SystemExit(f"[REFUSED] {final} exists")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(recipe["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(recipe["seed"])
        torch.cuda.reset_peak_memory_stats()
    args, model, criterion, _post, _ = G.build_arm(P, F.CFG, device, init="none", pretrained_backbone=False,
                                                   timm_override=timm_override)
    ck = torch.load(init_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"], strict=True)
    groups = G.param_groups(P, model, args)
    for g in groups:
        g["lr"] = g["lr"] * recipe["lr_factor"]
    opt = torch.optim.AdamW(groups, lr=args.lr * recipe["lr_factor"], weight_decay=args.weight_decay)
    ds = make_dataset(M2, arm, zip_path, train_json, pipeline_dir, recipe)
    flags = animal_flags(ds)
    order, info = epoch_order(len(ds), flags, arm, rec)
    workers = recipe["workers"] if workers is None else workers
    g = torch.Generator().manual_seed(recipe["seed"])
    dl = DataLoader(ds, batch_size=recipe["batch"], sampler=ListSampler(order), collate_fn=collate_fn,
                    num_workers=workers, drop_last=True, generator=g, pin_memory=torch.cuda.is_available(),
                    multiprocessing_context=("spawn" if workers > 0 else None))
    log(f"[G4b {VERSION}] arm {arm} {ARMS[arm]} | init {Path(init_ckpt).name} | steps {n_steps} | "
        f"LRs {[x['lr'] for x in opt.param_groups]} | sampling {json.dumps(info)}")
    model.train()
    criterion.train()
    t0, run, step, n_tile, n_anim_targets = time.time(), 0.0, 0, 0, 0
    for samples, targets in dl:
        samples = samples.to(device)
        if selftest_downscale:
            import torch.nn.functional as Fn
            from util.misc import NestedTensor
            x = Fn.interpolate(samples.tensors, size=(selftest_downscale,) * 2, mode="bilinear", align_corners=False)
            samples = NestedTensor(x, torch.zeros(x.shape[0], x.shape[2], x.shape[3], dtype=torch.bool))
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        for t in targets:
            n_tile += int(t["m2_is_tile"].item())
            n_anim_targets += int((t["labels"] == ds.cat_id_to_label[ANIMAL]).sum().item())
        loss = G.forward_loss(model, criterion, samples, targets, "fp32", device)
        if not math.isfinite(loss.item()):
            raise SystemExit(f"[G4b] non-finite loss at step {step + 1}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
        opt.step()
        step += 1
        run += loss.item()
        if step % recipe["log_every"] == 0 or step == n_steps:
            el = time.time() - t0
            mem = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0
            log(f"  step {step}/{n_steps} loss {loss.item():.4f} avg {run / step:.4f} | {el / step:.3f} s/step "
                f"(~{el / step * (n_steps - step) / 60:.0f} min left) | peak {mem:.2f} GB | tiles {n_tile / (2 * step):.2f} | "
                f"ANIMAL targets {n_anim_targets}")
        if step >= n_steps:
            break
    meta = {"version": VERSION, "script_sha256": sha256_file(Path(__file__).resolve()), "cfg": F.CFG,
            "timm_override": timm_override, "arm": arm, "arm_cfg": ARMS[arm], "recipe": rec, "sampling": info,
            "init": str(init_ckpt), "init_sha256": sha256_file(init_ckpt), "avg_loss": run / max(step, 1),
            "animal_targets_seen": n_anim_targets, "tile_fraction": n_tile / max(1, 2 * step),
            "epoch_label": f"G4b_{arm}"}
    F.save_ckpt(final, model, meta)
    log(f"  [saved] {final} ({(time.time() - t0) / 60:.1f} min)")
    return model, meta


# =============================================================================
# scoring
# =============================================================================
def iou(a, b):
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    u = a[2] * a[3] + b[2] * b[3] - ix * iy
    return ix * iy / u if u > 0 else 0.0


def animal_indicators(preds_by_img, val, img_ids):
    ids = set(img_ids)
    gts = defaultdict(list)
    for a in val["annotations"]:
        if a["image_id"] in ids and a["category_id"] == ANIMAL and int(a.get("iscrowd", 0)) == 0:
            gts[a["image_id"]].append([float(v) for v in a["bbox"]])
    best, fp = [], 0
    for i in ids:
        an = [d for d in preds_by_img.get(i, []) if d["category_id"] == ANIMAL]
        for g in gts[i]:
            s = [d["score"] for d in an if iou(d["bbox"], g) >= 0.3]
            best.append(max(s) if s else 0.0)
        fp += sum(1 for d in an if d["score"] >= 0.05 and not any(iou(d["bbox"], g) >= 0.3 for g in gts[i]))
    return {"animal_gt": len(best), "animal_gt_confident_ge_0.3": sum(1 for b in best if b >= 0.3),
            "best_animal_score_median": float(np.median(best)) if best else None,
            "best_animal_score_max": max(best) if best else None, "animal_fp_ge_0.05": fp}


def coco_ap(val, img_ids, preds_by_img):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    ids = set(img_ids)
    ds = {"images": [im for im in val["images"] if im["id"] in ids],
          "annotations": [a for a in val["annotations"] if a["image_id"] in ids], "categories": val["categories"]}
    dets = [d for i in ids for d in preds_by_img.get(i, [])]
    names = {c["id"]: c["name"] for c in val["categories"]}
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        gt = COCO()
        gt.dataset = copy.deepcopy(ds)
        gt.createIndex()
        E = COCOeval(gt, gt.loadRes(copy.deepcopy(dets)), "bbox")
        E.params.iouThrs = np.array([0.30])
        E.evaluate()
        E.accumulate()
        E.summarize()
    pc = {}
    for k, cid in enumerate(E.params.catIds):
        p = E.eval["precision"][0, :, k, 0, 2]
        p = p[p > -1]
        if p.size:
            pc[names[cid]] = float(p.mean())
    others = [v for c, v in pc.items() if c != "ANIMAL"]
    return {"macro": float(E.stats[0]), "per_class": pc, "others_macro": float(np.mean(others)) if others else None}


def preds_of(MV, M1, hr_p2, cache):
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


def score_model(MV, M1, hr_p2, val, caches):
    """caches: {0: path, 5: path}.  Tiled + plain on offset 0, offset 5, and ANIMAL pooled over both."""
    out, pooled = {}, {"plain": {}, "tiled": {}}
    for off, path in caches.items():
        with gzip.open(path, "rt", encoding="utf-8") as f:
            c = json.load(f)
        if int(c["meta"]["val_frame_offset"]) != off or int(c["meta"]["val_frame_stride"]) != 10:
            raise SystemExit(f"[REFUSED] {path} is not stride 10 / offset {off}")
        plain, tiled = preds_of(MV, M1, hr_p2, c)
        ids = [int(k) for k in c["frames"]]
        del c
        out[f"off{off}"] = {m: dict(coco_ap(val, ids, P), animal=animal_indicators(P, val, ids))
                            for m, P in (("plain", plain), ("tiled", tiled))}
        for m, P in (("plain", plain), ("tiled", tiled)):
            pooled[m].update(P)
        out.setdefault("_ids", []).extend(ids)
    ids = out.pop("_ids")
    out["pooled_0_5"] = {m: {"animal_ap": coco_ap(val, ids, P)["per_class"].get("ANIMAL"),
                             "animal": animal_indicators(P, val, ids)} for m, P in pooled.items()}
    return out


def verdict(scores, rule=RULE):
    ctrl = scores["CTRL"]
    res = {}
    for arm in ("NOTILE", "RFS"):
        s = scores[arm]
        a, ac = s["pooled_0_5"]["tiled"]["animal_ap"] or 0.0, ctrl["pooled_0_5"]["tiled"]["animal_ap"] or 0.0
        o, oc = s["off0"]["tiled"]["others_macro"], ctrl["off0"]["tiled"]["others_macro"]
        fix = a >= rule["animal_ap_floor"] and a >= ac + rule["animal_ap_gain_vs_ctrl"]
        guard = o >= oc + rule["others_guard"]
        res[arm] = {"animal_ap_pooled": a, "ctrl_animal_ap_pooled": ac, "others_off0": o, "ctrl_others_off0": oc,
                    "animal_fixed": bool(fix), "others_guard": bool(guard), "PASS": bool(fix and guard),
                    "macro_off0_tiled": s["off0"]["tiled"]["macro"]}
    passing = [a for a in ("NOTILE", "RFS") if res[a]["PASS"]]
    best = max(r["macro_off0_tiled"] for a, r in res.items() if a in passing) if passing else None
    res["winner"] = (min(passing, key=lambda a: (-res[a]["macro_off0_tiled"], a != "NOTILE")) if passing else None)
    res["winner_rule"] = "passing arm with the highest offset-0 tiled macro AP30; ties -> NOTILE (simpler)"
    res["_best"] = best
    return res


def report_md(scores, v):
    L = [f"# G4b results ({VERSION})", "",
         "| model | off0 tiled macro | off0 tiled 5 classes w/o ANIMAL | off0 ANIMAL AP | off5 tiled macro | "
         "ANIMAL AP pooled 0+5 (tiled) | ANIMAL GT confident >= 0.3 | best ANIMAL score median / max | ANIMAL FP >= 0.05 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for n, s in scores.items():
        t0, t5, pa = s["off0"]["tiled"], s["off5"]["tiled"], s["pooled_0_5"]["tiled"]
        an = pa["animal"]
        L.append(f"| {n} | {t0['macro']:.4f} | {t0['others_macro']:.4f} | {t0['per_class'].get('ANIMAL', 0):.4f} | "
                 f"{t5['macro']:.4f} | {(pa['animal_ap'] or 0):.4f} | {an['animal_gt_confident_ge_0.3']}/{an['animal_gt']} | "
                 f"{an['best_animal_score_median']:.3f} / {an['best_animal_score_max']:.3f} | {an['animal_fp_ge_0.05']} |")
    L += ["", "| class (off0 tiled) | " + " | ".join(scores) + " |", "|---|" + "---|" * len(scores)]
    for c in ("ANIMAL", "BOAT", "MARINE_MARKER", "OBJECT", "SAILING_VESSEL", "SHIP"):
        L.append(f"| {c} | " + " | ".join(f"{s['off0']['tiled']['per_class'].get(c, 0):.4f}" for s in scores.values()) + " |")
    L += ["", "Rule: ANIMAL AP (tiled, offset 0 + 5 pooled) >= 0.05 AND >= CTRL + 0.03, AND offset-0 tiled macro of the "
          "5 other classes >= CTRL - 0.01.", ""]
    for a in ("NOTILE", "RFS"):
        L.append(f"- {a}: {json.dumps(v[a])}")
    L += ["", f"**WINNER: {v['winner'] or 'none -- no recipe change; do not start G5, diagnose further'}**"]
    return "\n".join(L) + "\n"


# =============================================================================
# selftest
# =============================================================================
def selftest(a):
    import gc
    import tempfile
    import torch
    print(f"[selftest] {VERSION}")
    assert os.environ.get("R0_MSDA_BACKEND") == "pytorch", "set R0_MSDA_BACKEND=pytorch"
    G, F = load_modules()
    P, M2, M1, hr_p2 = G.r0_imports(a.pipeline_dir)
    import m1_1_merge_variants as MV
    # 1 sampling
    flags = [i % 10 == 0 for i in range(1000)]
    o_c, i_c = epoch_order(1000, flags, "CTRL", dict(RECIPE, steps=500))
    o_n, _ = epoch_order(1000, flags, "NOTILE", dict(RECIPE, steps=500))
    o_r, i_r = epoch_order(1000, flags, "RFS", dict(RECIPE, steps=500))
    assert o_c == o_n and len(o_r) == 1000 and abs(i_r["repeat_factor"] - math.sqrt(3)) < 1e-12
    assert i_r["animal_sample_share"] > 1.4 * i_c["animal_sample_share"], (i_c, i_r)
    assert epoch_order(1000, flags, "RFS", dict(RECIPE, steps=500))[0] == o_r
    print(f"  [ok] sampling: CTRL = NOTILE order; RFS r = sqrt(0.3/0.1) = 1.73, ANIMAL share "
          f"{i_c['animal_sample_share']:.3f} -> {i_r['animal_sample_share']:.3f}; deterministic")
    # 2 indicators + verdict
    val = {"images": [{"id": 1}], "categories": [{"id": 1, "name": "ANIMAL"}],
           "annotations": [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]},
                           {"image_id": 1, "category_id": 1, "bbox": [50, 50, 10, 10]}]}
    pr = {1: [{"category_id": 1, "score": 0.6, "bbox": [0, 0, 10, 10]}, {"category_id": 1, "score": 0.1, "bbox": [90, 90, 5, 5]},
              {"category_id": 2, "score": 0.9, "bbox": [50, 50, 10, 10]}]}
    ind = animal_indicators(pr, val, [1])
    assert ind == {"animal_gt": 2, "animal_gt_confident_ge_0.3": 1, "best_animal_score_median": 0.3,
                   "best_animal_score_max": 0.6, "animal_fp_ge_0.05": 1}, ind
    mk = lambda ap, oth, mac: {"pooled_0_5": {"tiled": {"animal_ap": ap}}, "off0": {"tiled": {"others_macro": oth, "macro": mac}}}
    v = verdict({"CTRL": mk(0.0, 0.60, 0.50), "NOTILE": mk(0.10, 0.58, 0.52), "RFS": mk(0.06, 0.60, 0.51)})
    assert v["NOTILE"]["PASS"] is False and v["RFS"]["PASS"] is True and v["winner"] == "RFS", v
    v = verdict({"CTRL": mk(0.0, 0.60, 0.50), "NOTILE": mk(0.10, 0.595, 0.52), "RFS": mk(0.04, 0.60, 0.53)})
    assert v["winner"] == "NOTILE" and v["RFS"]["animal_fixed"] is False
    print("  [ok] ANIMAL indicators by hand; verdict rule (floor, gain vs CTRL, 5-class guard, winner)")
    with tempfile.TemporaryDirectory() as td:
        zp, tj, vj = G._synthetic_world(td)
        js = json.loads(Path(tj).read_text(encoding="utf-8"))
        for an in js["annotations"]:
            if an["category_id"] == 4 and an["image_id"] % 3 == 0:
                an["category_id"] = ANIMAL
        Path(tj).write_text(json.dumps(js), encoding="utf-8")
        # 3 animal-aware windows
        ds = make_dataset(M2, "RFS", zp, tj, a.pipeline_dir)
        idx = next(k for k, f in enumerate(animal_flags(ds)) if f)
        lab = ds.cat_id_to_label[ANIMAL]
        torch.manual_seed(0)
        hits = sum(int((ds[idx][1]["labels"] == lab).any()) for _ in range(12))
        assert ds.animal_windows(idx) and hits == 12, hits
        print(f"  [ok] RFS dataset: a tiled or full ANIMAL frame always carries an ANIMAL target ({hits}/12)")
        # 4 init from a checkpoint + LR x0.1 + training steps per arm
        _a, m0, _c, _p, _ = G.build_arm(P, F.CFG, "cpu", init="none", pretrained_backbone=False, timm_override="convnext_tiny")
        init = Path(td) / "e15.pth"
        torch.save({"model": m0.state_dict(), "g4": {"cfg": F.CFG, "timm_override": "convnext_tiny"}}, init)
        sd0 = {k: v.clone() for k, v in m0.state_dict().items()}
        del m0
        gc.collect()
        for arm in ARMS:
            model, meta = train_arm(G, F, P, M2, arm, "cpu", Path(td) / "run", init, n_steps=2, zip_path=zp,
                                    train_json=tj, pipeline_dir=a.pipeline_dir, workers=0, log=lambda *x: None,
                                    selftest_downscale=256, timm_override="convnext_tiny")
            assert meta["recipe"]["lr_factor"] == 0.1 and math.isfinite(meta["avg_loss"])
            k = "transformer.encoder.layers.0.self_attn.sampling_offsets.weight"
            assert not torch.equal(model.state_dict()[k], sd0[k])     # trained from the init
            del model
            gc.collect()
        print("  [ok] all 3 arms: init from checkpoint (strict), LRs x0.1, finite steps, atomic save")
        try:
            train_arm(G, F, P, M2, "CTRL", "cpu", Path(td) / "run", init, n_steps=1, zip_path=zp, train_json=tj,
                      pipeline_dir=a.pipeline_dir, workers=0, log=lambda *x: None, timm_override="convnext_tiny")
            raise AssertionError("overwrite not refused")
        except SystemExit:
            pass
        # 5 export + score end-to-end (rebuilt checkpoint, both offsets)
        val = json.loads(Path(vj).read_text(encoding="utf-8"))
        for an in val["annotations"]:
            if an["category_id"] == 4:
                an["category_id"] = ANIMAL
        Path(vj).write_text(json.dumps(val), encoding="utf-8")
        scores = {}
        for arm in ARMS:
            m, post, _ck = G.rebuild_from_ckpt(P, Path(td) / "run" / f"g4b_{arm}.pth", "cpu")
            caches = {}
            for off in (0, 5):
                c, _ = F.export_with_model(G, M1, hr_p2, m, post, "cpu", vj, zp, off, 10, 4, log=lambda *x: None)
                p = Path(td) / f"{arm}_{off}.json.gz"
                with gzip.open(p, "wt", encoding="utf-8") as f:
                    json.dump(c, f)
                caches[off] = p
            del m
            gc.collect()
            scores[arm] = score_model(MV, M1, hr_p2, val, caches)
        v = verdict(scores)
        report_md(scores, v)
        assert scores["CTRL"]["pooled_0_5"]["tiled"]["animal"]["animal_gt"] > 0
        print("  [ok] export (offset 0 and 5) -> score -> verdict -> report, end to end")
    print("SELFTEST PASS")


# =============================================================================
# smoke
# =============================================================================
def smoke(a):
    import gc
    import torch
    print(f"[smoke] {VERSION}")
    if os.environ.get("R0_MSDA_BACKEND") != "pytorch":
        sys.exit("set R0_MSDA_BACKEND=pytorch first")
    G, F = load_modules()
    P, M2, M1, hr_p2 = G.r0_imports(a.pipeline_dir)
    for p in (E15, BASE_CACHE[0], BASE_CACHE[5], R0_CACHE[0], R0_CACHE[5]):
        if not Path(p).exists():
            sys.exit(f"missing {p}")
    out = OUT / "smoke"
    import shutil
    if out.exists():
        shutil.rmtree(out)
    rep = {"version": VERSION, "init_sha256": sha256_file(E15)}
    for arm in ARMS:
        logs = []
        _m, meta = train_arm(G, F, P, M2, arm, "cuda", out, E15, n_steps=40,
                             recipe=dict(RECIPE, log_every=20), log=lambda s: (logs.append(s), print("   ", s)))
        s = [float(x.split(" s/step")[0].split("| ")[-1]) for x in logs if " s/step" in x]
        rep[arm] = {"sampling": meta["sampling"], "animal_targets_seen_40_steps": meta["animal_targets_seen"],
                    "tile_fraction": meta["tile_fraction"], "s_per_step_steps21_40": 2 * s[-1] - s[0] if len(s) == 2 else None,
                    "avg_loss": meta["avg_loss"]}
        del _m
        gc.collect()
        torch.cuda.empty_cache()
    # export parity of the rebuilt e15 checkpoint against the existing e15 offset-0 cache (3 frames)
    with gzip.open(BASE_CACHE[0], "rt", encoding="utf-8") as f:
        ref = json.load(f)
    first3 = [int(k) for k in sorted(ref["frames"], key=int)[:3]]
    m, post, _ck = G.rebuild_from_ckpt(P, E15, "cuda")
    mine, sec = F.export_with_model(G, M1, hr_p2, m, post, "cuda", G.VAL_JSON, G.ZIP, 0, 10, 4, image_ids=first3,
                                    log=lambda *x: None)
    l2c = {int(k): int(v) for k, v in ref["meta"]["label_to_cat_id"].items()}
    rep["export_parity_e15"] = G.compare_caches(M1, mine, ref, l2c)
    sps = max(rep[x]["s_per_step_steps21_40"] or 0 for x in ARMS)
    rep["projection_h"] = {"train_per_arm": sps * RECIPE["steps"] / 3600, "export_per_cache": sec / 3 * 432 / 3600,
                           "total": 3 * sps * RECIPE["steps"] / 3600 + 6 * sec / 3 * 432 / 3600}
    rep["checks"] = {"rfs_more_animal": rep["RFS"]["sampling"]["animal_sample_share"] > rep["CTRL"]["sampling"]["animal_sample_share"],
                     "notile_no_tiles": rep["NOTILE"]["tile_fraction"] == 0.0,
                     "export_parity": rep["export_parity_e15"]["ok"]}
    (out / "g4b_smoke_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))
    print("SMOKE PASS" if all(rep["checks"].values()) else "SMOKE FAIL")


# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["selftest", "smoke", "train", "export", "score"], required=True)
    ap.add_argument("--arm", choices=sorted(ARMS))
    ap.add_argument("--offset", type=int, choices=[0, 5])
    ap.add_argument("--pipeline-dir", default="D:/MARIVISION/H0_workbench/eval_tools")
    a = ap.parse_args()
    if a.mode == "selftest":
        return selftest(a)
    if a.mode == "smoke":
        return smoke(a)
    if os.environ.get("R0_MSDA_BACKEND") != "pytorch":
        sys.exit("set R0_MSDA_BACKEND=pytorch first")
    G, F = load_modules()
    P, M2, M1, hr_p2 = G.r0_imports(a.pipeline_dir)
    run = OUT / "run"
    if a.mode == "train":
        train_arm(G, F, P, M2, a.arm, "cuda", run, E15)
    elif a.mode == "export":
        ck = run / f"g4b_{a.arm}.pth"
        cp = run / "tile_cache" / f"r0_tile_cache_g4b_{a.arm}_stride10_offset{a.offset}.json.gz"
        if cp.exists():
            sys.exit(f"[REFUSED] {cp} exists")
        m, post, ckd = G.rebuild_from_ckpt(P, ck, "cuda")
        c, sec = F.export_with_model(G, M1, hr_p2, m, post, "cuda", G.VAL_JSON, G.ZIP, a.offset, 10, RECIPE["export_batch"],
                                     meta_extra={"source_checkpoint_path": str(ck.resolve()),
                                                 "source_checkpoint_sha256": sha256_file(ck),
                                                 "source_checkpoint_epoch": ckd.get("epoch"), "g4": ckd.get("g4")})
        cp.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(cp, "wt", encoding="utf-8") as f:
            json.dump(c, f)
        print(f"  wrote {cp} ({sec / 60:.1f} min)\n  sha256 {sha256_file(cp)}")
    else:
        import m1_1_merge_variants as MV
        val = json.loads(Path(G.VAL_JSON).read_text(encoding="utf-8"))
        models = {"R0_e15": R0_CACHE, "G4F_e15": BASE_CACHE}
        for arm in ARMS:
            models[arm] = {o: run / "tile_cache" / f"r0_tile_cache_g4b_{arm}_stride10_offset{o}.json.gz" for o in (0, 5)}
        scores = {}
        for n, c in models.items():
            print(f"  scoring {n} ...")
            scores[n] = score_model(MV, M1, hr_p2, val, c)
        v = verdict(scores)
        md = report_md(scores, v)
        (run / "g4b_results.json").write_text(json.dumps({"version": VERSION, "scores": scores, "verdict": v,
                                                          "rule": RULE}, indent=1, default=str), encoding="utf-8")
        (run / "g4b_results_tables.md").write_text(md, encoding="utf-8")
        print(md)


if __name__ == "__main__":
    main()
