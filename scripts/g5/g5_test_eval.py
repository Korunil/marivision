r"""
G5 TEST evaluation (Part C of claude/g5_predeclaration.md): export TEST predictions once per frozen model, then score
ONCE with the paper's metric and the pre-declared Welch / Bonferroni test.

Location: D:/MARIVISION/R0_frozen_baseline/g5_test_eval.py  (run FROM that folder, next to g4_full.py G4F.0.1 and
g4_pilot.py G4P.0, both SHA-checked).  R0_MSDA_BACKEND=pytorch, HF_HUB_DISABLE_SYMLINKS_WARNING=1.

Models (frozen in Part A):
  seed42 / seed43 / seed44   G4F recipe, epoch 15, 17-tile export (Stage B full frame + 16 Stage C windows)
  r0e15 / r0e04              R0 R50 epochs 15 / 4, full-frame (Stage B) export only
System (primary): seed S tiled for BOAT, LEISURE_VEHICLE, MARINE_MARKER, OBJECT, SAILING_VESSEL, SHIP
                  + r0e15 plain for ANIMAL.
Metric (paper): pycocotools, maxDets 100, original-image boxes; mAP@0.3 = mean over the classes with GT (7 on TEST);
                per class; small / medium / large (COCO areas); class-agnostic mAPca@0.3; mAP@0.5.
Primary test:   3-seed mean routed mAP@0.3 vs TransVOD++ / Swin-B / R50 (paper mean +- SD, n = 3), one-tailed Welch,
                Bonferroni over 3 comparisons (alpha 0.05 / 3).

Modes:
  selftest                               CPU, synthetic zip with a synthetic TEST split
  proof                                  VAL only (GPU, ~10 min): this tool's export reproduces the existing VAL
                                         caches on 20 frames, and its scorer reproduces the recorded VAL numbers
  export --model M --test-unlock         TEST predictions of one model (resumable parts of 250 frames)
  score  --test-unlock                   the single TEST scoring; refuses if any model is incomplete or if it ran before
Outputs: D:/MARIVISION/G5/test_cache/<model>/ and D:/MARIVISION/G5/g5_test_results.md / .json (stay on the workstation).
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
from collections import defaultdict
from pathlib import Path

VERSION = "G5T.0"
FULL_SHA = "c1fbed7f03bbfaac525c422a00b85a436d582116d4b1b2e72d2d69472021479d"
PILOT_SHA = "3128fbf2985c7aa26fc5d38ec334e5cc67ac8281947c8b2da5932e31bdcd496b"
TEST_JSON_SHA = "524569f50e9a5820e449cfcb27c34744f43de8bc53acd1b07d7a6dc2701c195d"   # G5 inventory, never parsed before
ANIMAL = 1
PART = 250
ROOT = Path("D:/MARIVISION")
G5 = ROOT / "G5"
TEST_JSON = ROOT / "seaclips-test.json"
MODELS = {
    "seed42": {"ckpt": ROOT / "G4_full/run/g4f_epoch15.pth", "kind": "tiled", "seed": 42},
    "seed43": {"ckpt": G5 / "seed43/run/g4f_epoch15.pth", "kind": "tiled", "seed": 43},
    "seed44": {"ckpt": G5 / "seed44/run/g4f_epoch15.pth", "kind": "tiled", "seed": 44},
    "r0e15": {"ckpt": ROOT / "checkpoints_r0/checkpoint_epoch_15.pth", "kind": "plain", "seed": None},
    "r0e04": {"ckpt": ROOT / "checkpoints_r0/checkpoint_epoch_04.pth", "kind": "plain", "seed": None},
}
SEEDS = ("seed42", "seed43", "seed44")
ROUTE_SOURCE = "r0e15"
PAPER = {"TransVOD++ Swin-B": (31.95, 3.28, 3), "Def. DETR Swin-B": (31.28, 3.72, 3), "Def. DETR R50": (22.86, 0.91, 3)}
ALPHA = 0.05
# VAL numbers recorded earlier (G4b / G4c / M1 reports), offset-0 caches, for the scorer proof
VAL_REF = {"G4F_e15_tiled": 0.5011, "G4F_e15_routed": 0.5248, "R0_e15_plain": 0.3647, "R0_e04_plain": 0.4241}
VAL_CACHES = {"G4F_e15": ROOT / "G4_full/run/tile_cache/r0_tile_cache_g4f_e15_stride10_offset0.json.gz",
              "R0_e15": ROOT / "R0_frozen_baseline/m1_1_tile_cache/r0_tile_cache_epoch15_stride10_offset0.json.gz",
              "R0_e04": ROOT / "R0_frozen_baseline/m1_1_tile_cache/r0_tile_cache_epoch04_stride10_offset0.json.gz"}
STAGE_B = "STAGE_B_FULL_CONTEXT"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_full(check=True):
    here = Path.cwd()
    for f, s in (("g4_pilot.py", PILOT_SHA), ("g4_full.py", FULL_SHA)):
        if not (here / f).exists():
            raise SystemExit(f"[REFUSED] {f} must sit in the run folder ({here})")
        if check and sha256_file(here / f) != s:
            raise SystemExit(f"[REFUSED] {f} sha {sha256_file(here / f)} != {s}")
    sys.path.insert(0, str(here))
    import g4_full as F
    G = F.load_pilot()
    return G, F


# =============================================================================
# statistics: one-tailed Welch from summary statistics
# =============================================================================
def _betacf(a, b, x):
    tiny, qab, qap, qam = 1e-300, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-15:
            break
    return h


def betainc(a, b, x):
    """Regularized incomplete beta I_x(a, b) (continued fraction)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lb) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lb) * _betacf(b, a, 1 - x) / b


def t_sf(t, df):
    """P(T > t) for Student t with df degrees of freedom."""
    p = 0.5 * betainc(df / 2.0, 0.5, df / (df + t * t))
    return p if t >= 0 else 1.0 - p


def welch_greater(m1, s1, n1, m2, s2, n2):
    v1, v2 = s1 * s1 / n1, s2 * s2 / n2
    se = math.sqrt(v1 + v2)
    t = (m1 - m2) / se
    df = (v1 + v2) ** 2 / (v1 * v1 / (n1 - 1) + v2 * v2 / (n2 - 1))
    return {"t": t, "df": df, "p_one_tailed": t_sf(t, df)}


# =============================================================================
# export (same tile schedule / transform / box format as g4_full.export_with_model)
# =============================================================================
def frame_order(images):
    return [int(im["id"]) for im in sorted(images, key=lambda m: (int(m.get("video_id", 0)),
                                                                 int(m.get("frame_id", m["id"])), int(m["id"])))]


def export_frames(M1, hr_p2, model, post, device, ds, ids, kind, batch=4, log=print):
    import torch
    import torchvision.transforms.functional as TF
    from util.misc import NestedTensor
    model.eval()
    frames, t0 = {}, time.time()
    with torch.inference_mode():
        for n, iid in enumerate(ids, 1):
            pil = ds._load_image(ds.images[iid]).convert("RGB")
            W, H = pil.size
            crops = M1.build_tiles(hr_p2, W, H)
            if kind == "plain":
                crops = [c for c in crops if c.stage == STAGE_B]
            ts = [M1.preprocess_r0(pil.crop((c.x0_int, c.y0_int, c.x1_int, c.y1_int)), torch, TF) for c in crops]
            sizes = [[c.y1_int - c.y0_int, c.x1_int - c.x0_int] for c in crops]
            res = []
            for i in range(0, len(ts), batch):
                x = torch.stack(ts[i:i + batch]).to(device)
                m = torch.zeros(x.shape[0], x.shape[2], x.shape[3], dtype=torch.bool, device=device)
                res.extend(post["bbox"](model(NestedTensor(x, m)),
                                        torch.tensor(sizes[i:i + batch], dtype=torch.float32, device=device)))
            tiles = []
            for c, r in zip(crops, res):
                sc, lb, bx = r["scores"].cpu().tolist(), r["labels"].cpu().tolist(), r["boxes"].cpu().tolist()
                tiles.append([c.stage, c.x0_int, c.y0_int, c.x1_int, c.y1_int,
                              [[round(float(s), 6), int(l), round(b[0], 3), round(b[1], 3), round(b[2], 3), round(b[3], 3)]
                               for s, l, b in zip(sc, lb, bx)]])
            frames[str(iid)] = {"W": W, "H": H, "tiles": tiles}
            if n % 50 == 0 or n == len(ids):
                log(f"    {n}/{len(ids)} frames, {time.time() - t0:.0f}s")
    return frames, time.time() - t0


def open_dataset(zip_path, ann_json):
    from datasets.seaclips_detection import SeaClipsDetection
    with contextlib.redirect_stdout(io.StringIO()):
        ds = SeaClipsDetection(Path(zip_path), Path(ann_json), is_train=False, random_flip=False)
    return ds, {int(k): int(v) for k, v in dict(ds.label_to_cat_id).items()}


def check_ckpt(G, P, name, spec, device):
    m, post, ck = G.rebuild_from_ckpt(P, spec["ckpt"], device)
    g = ck.get("g4") or {}
    if spec["kind"] == "tiled":
        if int(g.get("epoch", -1)) != 15 or int(g.get("recipe", {}).get("seed", -1)) != spec["seed"]:
            raise SystemExit(f"[REFUSED] {spec['ckpt']} is not epoch 15 of seed {spec['seed']} "
                             f"(epoch {g.get('epoch')}, seed {g.get('recipe', {}).get('seed')})")
    return m, post, ck


def export_model(G, P, M1, hr_p2, name, spec, ann_json, zip_path, out_root, device, json_sha=TEST_JSON_SHA,
                 val_l2c=None, part=PART, max_parts=None, log=print):
    if json_sha is not None and sha256_file(ann_json) != json_sha:
        raise SystemExit(f"[REFUSED] {ann_json} sha differs from the locked TEST label file {json_sha}")
    out = Path(out_root) / name
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise SystemExit(f"[REFUSED] {out} is complete (manifest.json exists)")
    ds, l2c = open_dataset(zip_path, ann_json)
    if val_l2c is not None and l2c != val_l2c:
        raise SystemExit(f"[REFUSED] class mapping of {ann_json} {l2c} differs from the training/VAL mapping {val_l2c}")
    ids = frame_order(list(ds.images.values()))
    ck_sha = sha256_file(spec["ckpt"])
    m, post, ck = check_ckpt(G, P, name, spec, device)
    n_parts = (len(ids) + part - 1) // part
    secs, done_now = 0.0, 0
    for k in range(n_parts):
        pp = out / f"part_{k:03d}.json.gz"
        if pp.exists():
            continue
        if max_parts is not None and done_now >= max_parts:
            break
        sub = ids[k * part:(k + 1) * part]
        frames, sec = export_frames(M1, hr_p2, m, post, device, ds, sub, spec["kind"], log=log)
        tmp = Path(str(pp) + ".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            json.dump({"part": k, "ids": sub, "sec": sec, "frames": frames}, fh)
        os.replace(tmp, pp)
        secs += sec
        done_now += 1
        log(f"  [part {k + 1}/{n_parts}] {len(sub)} frames, {sec / len(sub):.2f} s/frame")
    if all((out / f"part_{k:03d}.json.gz").exists() for k in range(n_parts)):
        tot = 0.0
        for k in range(n_parts):
            with gzip.open(out / f"part_{k:03d}.json.gz", "rt", encoding="utf-8") as fh:
                tot += json.load(fh)["sec"]
        man = {"version": VERSION, "script_sha256": sha256_file(Path(__file__).resolve()), "model": name,
               "kind": spec["kind"], "seed": spec["seed"], "ckpt": str(spec["ckpt"]), "ckpt_sha256": ck_sha,
               "ckpt_epoch": ck.get("epoch"), "ann_json_sha256": sha256_file(ann_json), "n_frames": len(ids),
               "n_parts": n_parts, "label_to_cat_id": {str(k): v for k, v in sorted(l2c.items())},
               "export_sec_total": tot, "sec_per_frame": tot / len(ids)}
        (out / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
        log(f"  [complete] {name}: {len(ids)} frames, {tot / len(ids):.2f} s/frame")
        return man
    return None


# =============================================================================
# predictions and scoring
# =============================================================================
def iter_cache_frames(src):
    src = Path(src)
    if src.is_dir():
        man = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
        l2c = {int(k): int(v) for k, v in man["label_to_cat_id"].items()}
        for k in range(man["n_parts"]):
            with gzip.open(src / f"part_{k:03d}.json.gz", "rt", encoding="utf-8") as fh:
                d = json.load(fh)
            for iid, fr in d["frames"].items():
                yield l2c, int(iid), fr
    else:
        with gzip.open(src, "rt", encoding="utf-8") as fh:
            c = json.load(fh)
        l2c = {int(k): int(v) for k, v in c["meta"]["label_to_cat_id"].items()}
        for iid, fr in c["frames"].items():
            yield l2c, int(iid), fr


def cache_preds(MV, M1, hr_p2, src, tiled=True):
    """plain = Stage B tile (exactly M1.plain_dets_native); tiled = V0 merge of all 17 tiles."""
    plain, tl = defaultdict(list), defaultdict(list)
    for l2c, iid, fr in iter_cache_frames(src):
        b = [t for t in fr["tiles"] if t[0] == STAGE_B]
        if len(b) != 1:
            raise SystemExit(f"[REFUSED] frame {iid}: {len(b)} Stage B tiles")
        d = b[0][5]
        for x in M1.plain_dets_native([q[0] for q in d], [q[1] for q in d], [q[2:6] for q in d], l2c):
            plain[iid].append({"image_id": iid, "category_id": int(x["category_id"]),
                               "bbox": [float(v) for v in x["bbox"]], "score": float(x["score"])})
        if tiled:
            for x in MV.build_variant("V0", M1, hr_p2, l2c, fr):
                tl[iid].append({"image_id": iid, "category_id": int(x["category_id"]),
                                "bbox": [float(v) for v in x["bbox"]], "score": float(x["score"])})
    return plain, (tl if tiled else None)


def route(tiled, animal_src, ids):
    return {i: [d for d in tiled.get(i, []) if d["category_id"] != ANIMAL] +
               [d for d in animal_src.get(i, []) if d["category_id"] == ANIMAL] for i in ids}


def evaluate(gt_json, ids, preds):
    """mAP@0.3, per-class AP@0.3, small/medium/large AP@0.3, mAP@0.5, class-agnostic mAPca@0.3."""
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    ids = list(ids)
    keep = set(ids)
    ds = {"images": [im for im in gt_json["images"] if im["id"] in keep],
          "annotations": [a for a in gt_json["annotations"] if a["image_id"] in keep],
          "categories": gt_json["categories"]}
    dets = [d for i in ids for d in preds.get(i, [])]
    names = {c["id"]: c["name"] for c in gt_json["categories"]}

    def run(dsx, detsx, thr):
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            gt = COCO()
            gt.dataset = copy.deepcopy(dsx)
            gt.createIndex()
            E = COCOeval(gt, gt.loadRes(copy.deepcopy(detsx)), "bbox")
            E.params.iouThrs = np.array(thr)
            E.evaluate()
            E.accumulate()
        return E

    out = {}
    if not dets:
        return {"map30": 0.0, "map50": 0.0, "per_class30": {}, "size30": {}, "mapca30": 0.0}
    E = run(ds, dets, [0.30, 0.50])
    pr = E.eval["precision"]                      # [T, R, K, A, M]
    mean_valid = lambda a: float(a[a > -1].mean()) if (a > -1).any() else None
    out["map30"] = mean_valid(pr[0, :, :, 0, 2])
    out["map50"] = mean_valid(pr[1, :, :, 0, 2])
    out["per_class30"] = {names[c]: mean_valid(pr[0, :, k, 0, 2]) for k, c in enumerate(E.params.catIds)
                          if mean_valid(pr[0, :, k, 0, 2]) is not None}
    out["size30"] = {s: mean_valid(pr[0, :, :, a, 2]) for a, s in ((1, "small"), (2, "medium"), (3, "large"))}
    dca = {"images": ds["images"], "categories": [{"id": 1, "name": "object"}],
           "annotations": [dict(a, category_id=1) for a in ds["annotations"]]}
    Eca = run(dca, [dict(d, category_id=1) for d in dets], [0.30])
    out["mapca30"] = mean_valid(Eca.eval["precision"][0, :, :, 0, 2])
    return out


def score_all(MV, M1, hr_p2, gt_json, ids, sources, log=print):
    """sources: name -> cache (dir or .json.gz). Seeds give tiled + plain + routed, R0 models give plain.
    Models are scored one at a time (memory); the ANIMAL source is loaded first and kept."""
    src_name = ROUTE_SOURCE if ROUTE_SOURCE in sources else "R0_e15"
    order = [src_name] + [n for n in sources if n != src_name]
    animal = None
    res = {}
    for n in order:
        log(f"  predictions {n} ...")
        pl, tl = cache_preds(MV, M1, hr_p2, sources[n], tiled=n in SEEDS or n.startswith("G4F"))
        if n == src_name:
            animal = {i: [d for d in pl.get(i, []) if d["category_id"] == ANIMAL] for i in ids}
        r = {"plain": evaluate(gt_json, ids, pl)}
        if tl is not None:
            r["tiled"] = evaluate(gt_json, ids, tl)
            r["routed"] = evaluate(gt_json, ids, route(tl, animal, ids))
            for c, v in r["tiled"]["per_class30"].items():
                if c != "ANIMAL":
                    assert abs(r["routed"]["per_class30"][c] - v) < 1e-12, (n, c)
        del pl, tl
        res[n] = r
        log(f"  {n}: " + ", ".join(f"{k} mAP30 {v['map30']:.4f}" for k, v in r.items()))
    return {n: res[n] for n in sources}


def primary(res):
    vals = [100.0 * res[s]["routed"]["map30"] for s in SEEDS]
    m = sum(vals) / 3
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / 2)
    out = {"seeds_routed_map30": vals, "mean": m, "sd": sd, "alpha_bonferroni": ALPHA / len(PAPER), "vs": {}}
    for k, (pm, ps, pn) in PAPER.items():
        w = welch_greater(m, sd, 3, pm, ps, pn) if sd > 0 or ps > 0 else None
        out["vs"][k] = dict(w, paper_mean=pm, paper_sd=ps, diff=m - pm,
                            significant=bool(w and w["p_one_tailed"] < ALPHA / len(PAPER)))
    un = [100.0 * res[s]["tiled"]["map30"] for s in SEEDS]
    out["secondary_unrouted"] = {"seeds": un, "mean": sum(un) / 3}
    return out


def report(res, prim, mans, latency):
    f = lambda v: "–" if v is None else f"{100 * v:.2f}"
    L = [f"# G5 TEST results ({VERSION}) -- single evaluation, all 7,030 TEST frames", "",
         f"**Primary: routed mAP@0.3 over 3 seeds = {prim['mean']:.2f} ± {prim['sd']:.2f}** "
         f"(seeds {', '.join(f'{v:.2f}' for v in prim['seeds_routed_map30'])})", "",
         "| vs (paper, mean ± SD, n = 3) | paper | diff | Welch t | df | one-tailed p | significant at 0.05/3 |",
         "|---|---|---|---|---|---|---|"]
    for k, v in prim["vs"].items():
        L.append(f"| {k} | {v['paper_mean']:.2f} ± {v['paper_sd']:.2f} | {v['diff']:+.2f} | {v['t']:.3f} | {v['df']:.2f} | "
                 f"{v['p_one_tailed']:.4f} | {'YES' if v['significant'] else 'no'} |")
    L += ["", f"Secondary, unrouted (G4F tiled only): {prim['secondary_unrouted']['mean']:.2f} "
          f"(seeds {', '.join(f'{v:.2f}' for v in prim['secondary_unrouted']['seeds'])})", "",
          "| model | variant | mAP@0.3 | small | medium | large | mAPca@0.3 | mAP@0.5 |", "|---|---|---|---|---|---|---|---|"]
    for n, r in res.items():
        for var, e in r.items():
            L.append(f"| {n} | {var} | {f(e['map30'])} | {f(e['size30'].get('small'))} | {f(e['size30'].get('medium'))} | "
                     f"{f(e['size30'].get('large'))} | {f(e['mapca30'])} | {f(e['map50'])} |")
    classes = sorted({c for r in res.values() for e in r.values() for c in e["per_class30"]})
    L += ["", "| model | variant | " + " | ".join(classes) + " |", "|---|---|" + "---|" * len(classes)]
    for n, r in res.items():
        for var, e in r.items():
            L.append(f"| {n} | {var} | " + " | ".join(f(e["per_class30"].get(c)) for c in classes) + " |")
    L += ["", "Export throughput (s/frame, RTX 3060, includes image decode): " + json.dumps(latency), "",
          "Inputs: " + json.dumps({n: {"ckpt_sha256": m["ckpt_sha256"], "n_frames": m["n_frames"]} for n, m in mans.items()})]
    return "\n".join(L) + "\n"


# =============================================================================
def selftest(a):
    import shutil
    import tempfile
    import zipfile
    import torch
    print(f"[selftest] {VERSION}")
    assert os.environ.get("R0_MSDA_BACKEND") == "pytorch", "set R0_MSDA_BACKEND=pytorch"
    # 1 statistics
    assert abs(t_sf(2.132, 4) - 0.05) < 5e-4 and abs(t_sf(1.0, 10) - 0.17044) < 1e-4 and abs(t_sf(0.0, 7) - 0.5) < 1e-12
    assert abs(t_sf(-1.0, 10) - 0.82956) < 1e-4 and abs(t_sf(12.706, 1) - 0.025) < 1e-4
    w = welch_greater(35.0, 2.0, 3, 31.95, 3.28, 3)
    se = math.sqrt(4 / 3 + 3.28 ** 2 / 3)
    assert abs(w["t"] - 3.05 / se) < 1e-12 and 2 < w["df"] < 4
    print(f"  [ok] Student t tail vs tables (t 2.132 df 4 -> 0.050; t 12.706 df 1 -> 0.025); Welch t/df "
          f"(35.0±2.0 vs 31.95±3.28: t {w['t']:.3f}, df {w['df']:.2f}, p {w['p_one_tailed']:.4f})")
    G, F = load_full()
    P, M2, M1, hr_p2 = G.r0_imports(a.pipeline_dir)
    import m1_1_merge_variants as MV
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        td = Path(td)
        zp, tj, vj = G._synthetic_world(td)
        val = json.loads(Path(vj).read_text(encoding="utf-8"))
        for an in val["annotations"]:
            if an["category_id"] == 4 and an["image_id"] % 2 == 0:
                an["category_id"] = ANIMAL
        Path(vj).write_text(json.dumps(val), encoding="utf-8")
        with zipfile.ZipFile(zp, "a") as z:
            for n in [x for x in z.namelist() if x.startswith("SeaClips/val/")]:
                z.writestr(n.replace("SeaClips/val/", "SeaClips/test/"), z.read(n))
        tsj = td / "seaclips-test.json"
        shutil.copy(vj, tsj)
        tst = json.loads(tsj.read_text(encoding="utf-8"))           # keep 3 frames per video (CPU time)
        keep = {i for v in (1, 2) for i in frame_order([m for m in tst["images"] if m["video_id"] == v])[:3]}
        tst["images"] = [m for m in tst["images"] if m["id"] in keep]
        tst["annotations"] = [x for x in tst["annotations"] if x["image_id"] in keep]
        tsj.write_text(json.dumps(tst), encoding="utf-8")
        # models: two tiny ConvNeXt "seeds" with g4 meta, one R50 without
        specs = {}
        for s in (42, 43, 44):
            torch.manual_seed(s)
            _a, m, _c, _p, _ = G.build_arm(P, F.CFG, "cpu", init="none", pretrained_backbone=False, timm_override="convnext_tiny")
            p = td / f"s{s}.pth"
            torch.save({"model": m.state_dict(), "epoch": "x", "g4": {"cfg": F.CFG, "timm_override": "convnext_tiny",
                                                                     "epoch": 15, "recipe": {"seed": s}}}, p)
            specs[f"seed{s}"] = {"ckpt": p, "kind": "tiled", "seed": s}
            del m
        _a, m, _c, _p, _ = G.build_arm(P, {"backbone": None, "precision": "fp32", "p_tile": 0.0}, "cpu", init="none",
                                       pretrained_backbone=False)
        p = td / "r0.pth"
        torch.save({"model": m.state_dict()}, p)
        del m
        specs["r0e15"] = {"ckpt": p, "kind": "plain", "seed": None}
        specs["r0e04"] = {"ckpt": p, "kind": "plain", "seed": None}
        # 2 refusals: wrong label-file SHA, wrong epoch/seed, mapping mismatch
        for bad in ({"json_sha": "0" * 64}, {"val_l2c": {0: 9}}):
            try:
                export_model(G, P, M1, hr_p2, "seed42", specs["seed42"], tsj, zp, td / "x", "cpu",
                             **dict({"json_sha": None}, **bad), part=10, log=lambda *x: None)
                raise AssertionError(f"not refused: {bad}")
            except SystemExit:
                pass
        try:
            export_model(G, P, M1, hr_p2, "seed43", dict(specs["seed43"], seed=99), tsj, zp, td / "x", "cpu",
                         json_sha=None, part=10, log=lambda *x: None)
            raise AssertionError("wrong seed not refused")
        except SystemExit:
            pass
        print("  [ok] refuses a TEST label file with another SHA, a class mapping that differs from VAL, "
              "a checkpoint that is not epoch 15 of the declared seed")
        # 3 export with interruption + resume == uninterrupted; parity with g4_full's VAL export path
        tc = td / "test_cache"
        assert export_model(G, P, M1, hr_p2, "seed42", specs["seed42"], tsj, zp, tc, "cpu", json_sha=None,
                            part=2, max_parts=1, log=lambda *x: None) is None
        man = export_model(G, P, M1, hr_p2, "seed42", specs["seed42"], tsj, zp, tc, "cpu", json_sha=None,
                           part=2, log=lambda *x: None)
        assert man and man["n_frames"] == 6 and man["n_parts"] == 3
        try:
            export_model(G, P, M1, hr_p2, "seed42", specs["seed42"], tsj, zp, tc, "cpu", json_sha=None, part=2,
                         log=lambda *x: None)
            raise AssertionError("complete export not protected")
        except SystemExit:
            pass
        m42, post, _ = G.rebuild_from_ckpt(P, specs["seed42"]["ckpt"], "cpu")
        ref, _ = F.export_with_model(G, M1, hr_p2, m42, post, "cpu", vj, zp, 0, 1, 4, image_ids=sorted(keep),
                                     log=lambda *x: None)
        mine = {"frames": {str(i): fr for _l, i, fr in iter_cache_frames(tc / "seed42")}}
        l2c = {int(k): int(v) for k, v in ref["meta"]["label_to_cat_id"].items()}
        par = G.compare_caches(M1, mine, ref, l2c)
        assert par["ok"] and par["max_box_dev_px"] == 0.0 and par["frames"] == 6, par
        print("  [ok] export interrupted after part 1 + resumed = 6 frames in 3 parts; a complete export is protected; "
              "tile-by-tile identical to g4_full.export_with_model on the same frames")
        for n in ("seed43", "seed44", "r0e15", "r0e04"):
            export_model(G, P, M1, hr_p2, n, specs[n], tsj, zp, tc, "cpu", json_sha=None, part=2, log=lambda *x: None)
        # 4 scorer: plain/tiled equal g4_full.score_cache; routed keeps the 5 other classes; one-time lock
        gt = json.loads(tsj.read_text(encoding="utf-8"))
        ids = frame_order(gt["images"])
        sources = {n: tc / n for n in MODELS}
        res = score_all(MV, M1, hr_p2, gt, ids, sources, log=lambda *x: None)
        sc = F.score_cache(M1, hr_p2, ref, gt)
        assert abs(res["seed42"]["plain"]["map30"] - sc["plain"]) < 1e-12 and abs(res["seed42"]["tiled"]["map30"] - sc["tiled"]) < 1e-12
        rp = cache_preds(MV, M1, hr_p2, tc / "r0e15", tiled=False)[0]
        tl42 = cache_preds(MV, M1, hr_p2, tc / "seed42")[1]
        rt = route(tl42, rp, ids)
        for i in ids:
            assert [d for d in rt[i] if d["category_id"] == ANIMAL] == [d for d in rp[i] if d["category_id"] == ANIMAL]
            assert [d for d in rt[i] if d["category_id"] != ANIMAL] == [d for d in tl42[i] if d["category_id"] != ANIMAL]
        prim = primary(res)
        assert len(prim["seeds_routed_map30"]) == 3 and set(prim["vs"]) == set(PAPER)
        mans = {n: json.loads((tc / n / "manifest.json").read_text(encoding="utf-8")) for n in MODELS}
        report(res, prim, mans, {n: m["sec_per_frame"] for n, m in mans.items()})
        out = td / "results"
        write_results(out, res, prim, mans)
        try:
            write_results(out, res, prim, mans)
            raise AssertionError("second scoring not refused")
        except SystemExit:
            pass
        print(f"  [ok] scorer = g4_full.score_cache (plain {sc['plain']:.4f}, tiled {sc['tiled']:.4f}); routed swaps only "
              f"ANIMAL (asserted per class); Welch vs 3 paper models; report; a second scoring is refused")
    print("SELFTEST PASS")


def write_results(out, res, prim, mans):
    out = Path(out)
    lock = out / "g5_test_scored.lock"
    if lock.exists():
        raise SystemExit(f"[REFUSED] TEST was already scored ({lock}); G5 allows one evaluation")
    out.mkdir(parents=True, exist_ok=True)
    lat = {n: m["sec_per_frame"] for n, m in mans.items()}
    lat["system_routed_per_frame_est"] = {s: lat[s] + lat[ROUTE_SOURCE] for s in SEEDS if s in lat and ROUTE_SOURCE in lat}
    md = report(res, prim, mans, lat)
    (out / "g5_test_results.json").write_text(json.dumps({"version": VERSION, "results": res, "primary": prim,
                                                          "manifests": mans, "latency": lat}, indent=1), encoding="utf-8")
    (out / "g5_test_results.md").write_text(md, encoding="utf-8")
    lock.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    return md


# =============================================================================
def proof(a):
    import torch
    print(f"[proof] {VERSION} -- VAL only; TEST is not touched")
    if os.environ.get("R0_MSDA_BACKEND") != "pytorch":
        sys.exit("set R0_MSDA_BACKEND=pytorch first")
    G, F = load_full()
    P, M2, M1, hr_p2 = G.r0_imports(a.pipeline_dir)
    import m1_1_merge_variants as MV
    rep = {"version": VERSION}
    val = json.loads(Path(G.VAL_JSON).read_text(encoding="utf-8"))
    with gzip.open(VAL_CACHES["G4F_e15"], "rt", encoding="utf-8") as fh:
        refG = json.load(fh)
    with gzip.open(VAL_CACHES["R0_e15"], "rt", encoding="utf-8") as fh:
        refR = json.load(fh)
    ids20 = sorted(int(k) for k in refG["frames"])[:20]
    ds, l2c = open_dataset(G.ZIP, G.VAL_JSON)
    rep["val_l2c"] = l2c
    # 1 this tool's export on 20 VAL frames vs the existing caches
    m, post, _ = check_ckpt(G, P, "seed42", MODELS["seed42"], "cuda")
    fr, sec = export_frames(M1, hr_p2, m, post, "cuda", ds, ids20, "tiled", log=lambda *x: None)
    rep["seed42_export_parity"] = G.compare_caches(M1, {"frames": fr}, refG, l2c)
    rep["seed42_sec_per_frame"] = sec / 20
    del m
    torch.cuda.empty_cache()
    m, post, _ = check_ckpt(G, P, "r0e15", MODELS["r0e15"], "cuda")
    fr, sec = export_frames(M1, hr_p2, m, post, "cuda", ds, ids20, "plain", log=lambda *x: None)
    per = [M1.parity_check(cache_preds_one(M1, l2c, fr[str(i)]), cache_preds_one(M1, l2c, refR["frames"][str(i)]),
                           0.02, 0.5) for i in ids20]
    rep["r0e15_plain_parity"] = {"ok": all(x["ok"] for x in per), "frames": len(per),
                                 "max_box_dev_px": max(x["max_box_dev_px"] for x in per),
                                 "unmatched": sum(x["unmatched_mine"] + x["unmatched_ref"] for x in per)}
    rep["r0e15_sec_per_frame"] = sec / 20
    del m
    torch.cuda.empty_cache()
    # 2 this tool's scorer on the full existing VAL offset-0 caches vs the recorded numbers
    ids0 = M1.eval_subset_ids(val["images"], 10, 0)
    res = score_all(MV, M1, hr_p2, val, ids0, VAL_CACHES, log=print)
    got = {"G4F_e15_tiled": res["G4F_e15"]["tiled"]["map30"], "G4F_e15_routed": res["G4F_e15"]["routed"]["map30"],
           "R0_e15_plain": res["R0_e15"]["plain"]["map30"], "R0_e04_plain": res["R0_e04"]["plain"]["map30"]}
    rep["val_scorer"] = {k: {"tool": got[k], "recorded": VAL_REF[k], "ok": abs(got[k] - VAL_REF[k]) < 6e-4} for k in got}
    rep["val_detail"] = res
    rep["test_json_sha_matches_lock"] = sha256_file(TEST_JSON) == TEST_JSON_SHA
    rep["checks"] = {"seed42_export_parity": rep["seed42_export_parity"]["ok"],
                     "r0e15_plain_parity": rep["r0e15_plain_parity"]["ok"],
                     "val_scorer": all(v["ok"] for v in rep["val_scorer"].values()),
                     "test_json_locked": rep["test_json_sha_matches_lock"]}
    G5.mkdir(parents=True, exist_ok=True)
    (G5 / "g5_test_eval_proof.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "val_detail"}, indent=1, default=str))
    tl = 7030 * (rep["seed42_sec_per_frame"]) / 3600
    print(f"  projected TEST export: {tl:.1f} h per seed, {7030 * rep['r0e15_sec_per_frame'] / 3600:.2f} h per R0 model")
    print("PROOF PASS" if all(rep["checks"].values()) else "PROOF FAIL")


def cache_preds_one(M1, l2c, fr):
    b = [t for t in fr["tiles"] if t[0] == STAGE_B][0][5]
    return M1.plain_dets_native([q[0] for q in b], [q[1] for q in b], [q[2:6] for q in b], l2c)


# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["selftest", "proof", "export", "score"], required=True)
    ap.add_argument("--model", choices=sorted(MODELS))
    ap.add_argument("--test-unlock", action="store_true")
    ap.add_argument("--pipeline-dir", default="D:/MARIVISION/H0_workbench/eval_tools")
    a = ap.parse_args()
    if a.mode == "selftest":
        return selftest(a)
    if a.mode == "proof":
        return proof(a)
    if not a.test_unlock:
        sys.exit("[REFUSED] TEST modes need --test-unlock (Director's authorization, G5 Part C)")
    if os.environ.get("R0_MSDA_BACKEND") != "pytorch":
        sys.exit("set R0_MSDA_BACKEND=pytorch first")
    G, F = load_full()
    P, M2, M1, hr_p2 = G.r0_imports(a.pipeline_dir)
    if sha256_file(TEST_JSON) != TEST_JSON_SHA:
        sys.exit(f"[REFUSED] {TEST_JSON} sha differs from the lock {TEST_JSON_SHA}")
    _vds, val_l2c = open_dataset(G.ZIP, G.VAL_JSON)
    tc = G5 / "test_cache"
    if a.mode == "export":
        if not a.model:
            sys.exit("--model is required")
        export_model(G, P, M1, hr_p2, a.model, MODELS[a.model], TEST_JSON, G.ZIP, tc, "cuda", val_l2c=val_l2c)
        return
    import m1_1_merge_variants as MV
    missing = [n for n in MODELS if not (tc / n / "manifest.json").exists()]
    if missing:
        sys.exit(f"[REFUSED] incomplete exports: {missing}; G5 scores once, with every model")
    if (G5 / "g5_test_scored.lock").exists():
        sys.exit("[REFUSED] TEST was already scored; G5 allows one evaluation")
    mans = {n: json.loads((tc / n / "manifest.json").read_text(encoding="utf-8")) for n in MODELS}
    for n, mm in mans.items():
        if mm["ann_json_sha256"] != TEST_JSON_SHA or mm["ckpt_sha256"] != sha256_file(MODELS[n]["ckpt"]):
            sys.exit(f"[REFUSED] {n}: manifest does not match the locked label file or its checkpoint")
    gt = json.loads(TEST_JSON.read_text(encoding="utf-8"))
    ids = frame_order(gt["images"])
    res = score_all(MV, M1, hr_p2, gt, ids, {n: tc / n for n in MODELS})
    prim = primary(res)
    md = write_results(G5, res, prim, mans)
    print(md)


if __name__ == "__main__":
    main()
