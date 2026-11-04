r"""
G4 full run: ConvNeXt-B + tiles (CNBT), fp32, the R0 / paper recipe for 15 epochs on all TRAIN frames.

Location: D:/MARIVISION/R0_frozen_baseline/g4_full.py  (run FROM that folder, like g4_pilot.py)
Needs: g4_pilot.py (G4P.0, SHA-checked) in the same folder; R0_MSDA_BACKEND=pytorch;
       HF_HUB_DISABLE_SYMLINKS_WARNING=1.

Recipe (frozen; the G4 pilot winner on the R0 recipe):
  model      ConvNeXt-B in22k backbone in R0's Deformable-DETR (g4_pilot.build_arm, arm CNBT_bf16 config but fp32)
  init       COCO Deformable-DETR transformer / queries / box head (as the pilot)
  data       all 20,310 TRAIN frames, M2 tile sampling p_tile 0.5, min_visible 0.5, flip 0.5
  schedule   15 epochs x 10,155 steps, batch 2, AdamW (2e-4 / backbone 2e-5 / linear_proj x0.1), wd 1e-4,
             clip 0.1, LR x0.1 after epochs 11 and 14 (= R0's MultiStepLR([11, 14]))
  precision  fp32 (pilot rule: bf16 was 0.029 worse on R50)
  seed       42; epoch e uses the seeded permutation (42, e) -> an interrupted epoch restarts identically
Per epoch (inline, after the checkpoint is written): tiled + plain export of the 432 VAL frames at
stride 10 / OFFSET 5 (the selection frames, disjoint from the offset-0 evaluation frames) and pycocotools
macro AP30 for both -> curve.json.  The best epoch is chosen on offset 5 only.

Modes:
  selftest                         CPU, synthetic zip
  smoke                            GPU: real steps, epoch-end checkpoint, resume equality, inline eval memory,
                                   export parity vs the R0 epoch-15 cache, projections
  train [--resume]                 the 15-epoch run (resumes from the last finished epoch)
  export --epoch K --offset 0|5    re-export one epoch's cache (M1.1 schema, for m2_evaluate.py)
  curve                            print the offset-5 curve and the selected epoch
"""
from PIL import Image  # noqa: F401  -- PIL before torch (Director's Windows env)
import sys
if "wandb" not in sys.modules:
    sys.modules["wandb"] = None
import argparse
import copy
import contextlib
import gzip
import hashlib
import io
import json
import math
import os
import time
from pathlib import Path

VERSION = "G4F.0.1"   # G4F.0 + corrected GPU resume check in smoke (training code unchanged)
PILOT_SHA = "3128fbf2985c7aa26fc5d38ec334e5cc67ac8281947c8b2da5932e31bdcd496b"

OUT = Path("D:/MARIVISION/G4_full")
RECIPE = {"epochs": 15, "batch": 2, "seed": 42, "milestones": (11, 14), "gamma": 0.1, "p_tile": 0.5,
          "min_visible": 0.5, "workers": 2, "log_every": 500, "eval_offset": 5, "eval_stride": 10,
          "export_batch": 4, "backbone": "convnext_base.fb_in22k", "precision": "fp32"}
CFG = {"backbone": RECIPE["backbone"], "precision": "fp32", "p_tile": RECIPE["p_tile"]}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_pilot(check=True):
    here = Path.cwd()
    p = here / "g4_pilot.py"
    if not p.exists():
        raise SystemExit("[REFUSED] g4_pilot.py must sit in the run folder (D:\\MARIVISION\\R0_frozen_baseline)")
    if check and sha256_file(p) != PILOT_SHA:
        raise SystemExit(f"[REFUSED] g4_pilot.py sha {sha256_file(p)} != G4P.0 {PILOT_SHA}")
    sys.path.insert(0, str(here))
    import g4_pilot as G
    return G


def lr_factor(epoch, recipe=RECIPE):
    """1-based epoch -> LR multiplier, identical to MultiStepLR(milestones, gamma) stepped once per epoch."""
    return recipe["gamma"] ** sum(1 for m in recipe["milestones"] if epoch > m)


def epoch_order(n, epoch, seed):
    import torch
    g = torch.Generator().manual_seed(seed * 1000 + epoch)
    return torch.randperm(n, generator=g).tolist()


class ListSampler:
    def __init__(self, order):
        self.order = order

    def __iter__(self):
        return iter(self.order)

    def __len__(self):
        return len(self.order)


def make_epoch_loader(ds, epoch, recipe, workers, limit=None):
    import torch
    from torch.utils.data import DataLoader
    from util.misc import collate_fn
    order = epoch_order(len(ds), epoch, recipe["seed"])
    if limit:
        order = order[:limit * recipe["batch"]]
    g = torch.Generator().manual_seed(recipe["seed"] * 1000 + epoch)
    return DataLoader(ds, batch_size=recipe["batch"], sampler=ListSampler(order), collate_fn=collate_fn,
                      num_workers=workers, drop_last=True, generator=g, pin_memory=torch.cuda.is_available(),
                      multiprocessing_context=("spawn" if workers > 0 else None)), order


# =============================================================================
# Inline export + scoring (same tile schedule / transform / cache schema as g4_pilot.export_cache)
# =============================================================================
def val_ids(M1, val_json, offset, stride):
    val = json.loads(Path(val_json).read_text(encoding="utf-8"))
    return val, M1.eval_subset_ids(val["images"], stride, offset)


def export_with_model(G, M1, hr_p2, model, post, device, val_json, zip_path, offset, stride, batch,
                      image_ids=None, meta_extra=None, log=print):
    import torch
    import torchvision.transforms.functional as TF
    from datasets.seaclips_detection import SeaClipsDetection
    from util.misc import NestedTensor
    vj = Path(val_json)
    if "test" in vj.name.lower() or "val" not in vj.name.lower():
        raise SystemExit("[SPLIT BLOCK] VAL json only; TEST is locked.")
    _val, ids = val_ids(M1, vj, offset, stride)
    if image_ids is not None:
        keep = set(image_ids)
        ids = [i for i in ids if i in keep]
    with contextlib.redirect_stdout(io.StringIO()):
        ds = SeaClipsDetection(Path(zip_path), vj, is_train=False, random_flip=False)
    l2c = {int(k): int(v) for k, v in dict(ds.label_to_cat_id).items()}
    was_training = model.training
    model.eval()
    frames, t0 = {}, time.time()
    with torch.inference_mode():
        for n, iid in enumerate(ids, 1):
            pil = ds._load_image(ds.images[iid]).convert("RGB")
            W, H = pil.size
            crops = M1.build_tiles(hr_p2, W, H)
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
            if n % 100 == 0 or n == len(ids):
                log(f"    eval export {n}/{len(ids)} frames, {time.time() - t0:.0f}s")
    if was_training:
        model.train()
    meta = {"schema": G.CACHE_SCHEMA, "script_version": VERSION, "script_sha256": sha256_file(Path(__file__).resolve()),
            "m1_module_sha256": sha256_file(Path(M1.__file__).resolve()),
            "pipeline_py_sha256": sha256_file(Path(hr_p2.__file__).resolve()),
            "coco_val_sha256": sha256_file(vj), "val_frame_stride": stride, "val_frame_offset": offset,
            "n_frames": len(ids), "label_to_cat_id": {str(k): v for k, v in sorted(l2c.items())},
            "box_format": "per tile: [score, r0_label, x1, y1, x2, y2] in TILE pixels (tile = native crop, no scaling)",
            "provenance": {"status": "SKIPPED_NO_REFERENCE_EXPORT",
                           "reason": "G4 full checkpoint; export path verified against the R0 epoch-15 cache in the G4F smoke"},
            "inference_precision": "fp32", "batch_size": batch, "device": str(device)}
    meta.update(meta_extra or {})
    return {"meta": meta, "frames": frames}, time.time() - t0


def score_cache(M1, hr_p2, cache, val):
    """pycocotools macro AP30 (maxDets 100, area all) for plain (= Stage B) and tiled (= V0 merge)."""
    import numpy as np
    import warnings
    import m1_1_merge_variants as MV
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    l2c = {int(k): int(v) for k, v in cache["meta"]["label_to_cat_id"].items()}
    ids = {int(i) for i in cache["frames"]}
    plain, tiled = [], []
    for iid, fr in cache["frames"].items():
        tiles = MV.frame_tiles(M1, hr_p2, fr)
        b = [t for t in tiles if t[0].stage == MV.STAGE_B][0][1]
        plain += [dict(d, image_id=int(iid)) for d in
                  M1.plain_dets_native([x[0] for x in b], [x[1] for x in b], [x[2:6] for x in b], l2c)]
        tiled += [{"image_id": int(iid), "category_id": int(d["category_id"]), "bbox": [float(v) for v in d["bbox"]],
                   "score": float(d["score"])} for d in MV.build_variant("V0", M1, hr_p2, l2c, fr)]
    ds = {"images": [im for im in val["images"] if im["id"] in ids],
          "annotations": [a for a in val["annotations"] if a["image_id"] in ids], "categories": val["categories"]}
    out = {}
    for name, dets in (("plain", plain), ("tiled", tiled)):
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            gt = COCO()
            gt.dataset = copy.deepcopy(ds)
            gt.createIndex()
            if not dets:
                out[name] = 0.0
                continue
            E = COCOeval(gt, gt.loadRes(copy.deepcopy(dets)), "bbox")
            E.params.iouThrs = np.array([0.30])
            E.evaluate()
            E.accumulate()
            E.summarize()
        out[name] = float(E.stats[0])
        pc = {}
        for k, cid in enumerate(E.params.catIds):
            p = E.eval["precision"][0, :, k, 0, 2]
            p = p[p > -1]
            if p.size:
                pc[str(cid)] = float(p.mean())
        out[name + "_per_class"] = pc
    return out


def select_epoch(curve):
    """Highest offset-5 tiled macro AP30; ties -> the earlier epoch (pre-declared)."""
    done = [r for r in curve if r.get("tiled") is not None]
    if not done:
        return None
    best = max(r["tiled"] for r in done)
    return min(r["epoch"] for r in done if r["tiled"] == best)


# =============================================================================
# Training
# =============================================================================
def save_ckpt(path, model, meta, opt=None, extra=None):
    import torch
    obj = {"model": model.state_dict(), "epoch": meta.get("epoch_label"), "g4": meta}
    if opt is not None:
        obj["optimizer"] = opt.state_dict()
    obj.update(extra or {})
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)      # atomic: a crash never leaves a half-written checkpoint


def run_training(G, P, M2, M1, hr_p2, device, out_dir, recipe=RECIPE, resume=False, steps_per_epoch=None,
                 epochs=None, zip_path=None, train_json=None, val_json=None, pipeline_dir=None, coco_ckpt=None,
                 pretrained=True, timm_override=None, eval_image_ids=None, workers=None, log=print,
                 stop_after_epoch=None, selftest_downscale=None):
    import torch
    zip_path = zip_path or G.ZIP
    train_json = train_json or G.TRAIN_JSON
    val_json = val_json or G.VAL_JSON
    pipeline_dir = pipeline_dir or G.PIPELINE_DIR
    coco_ckpt = coco_ckpt or G.COCO_CKPT
    epochs = epochs or recipe["epochs"]
    workers = recipe["workers"] if workers is None else workers
    out_dir.mkdir(parents=True, exist_ok=True)
    resume_p = out_dir / "resume_latest.pth"
    curve_p = out_dir / "curve.json"
    torch.manual_seed(recipe["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(recipe["seed"])
    args, model, criterion, post, n_coco = G.build_arm(P, CFG, device, "coco", pretrained, coco_ckpt, timm_override)
    groups = G.param_groups(P, model, args)
    base = [g["lr"] for g in groups]
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    start, curve = 1, []
    if resume:
        if not resume_p.exists():
            raise SystemExit(f"[REFUSED] --resume but {resume_p} does not exist")
        ck = torch.load(resume_p, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=True)
        opt.load_state_dict(ck["optimizer"])
        start = int(ck["g4"]["epoch"]) + 1
        if curve_p.exists():
            curve = json.loads(curve_p.read_text(encoding="utf-8"))
        log(f"[G4F] resumed after epoch {start - 1}")
    elif resume_p.exists():
        raise SystemExit(f"[REFUSED] {resume_p} exists: use --resume, or move the folder away for a fresh run")
    with contextlib.redirect_stdout(io.StringIO()):
        DS = M2.get_dataset_class()
        ds = DS(Path(zip_path), Path(train_json), p_tile=recipe["p_tile"], min_visible=recipe["min_visible"],
                pipeline_dir=str(pipeline_dir), random_flip=True)
    n_steps = steps_per_epoch or len(ds) // recipe["batch"]
    val = json.loads(Path(val_json).read_text(encoding="utf-8"))
    log(f"[G4F {VERSION}] CNBT fp32 | frames {len(ds)} | {epochs} epochs x {n_steps} steps | COCO tensors {n_coco} | "
        f"base LRs {base} | milestones {recipe['milestones']}")
    model.train()
    criterion.train()
    for epoch in range(start, epochs + 1):
        f = lr_factor(epoch, recipe)
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * f
        torch.manual_seed(recipe["seed"] * 1000 + epoch)          # per-epoch RNG: dropout, tiles, flips (workers=0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(recipe["seed"] * 1000 + epoch)
        dl, order = make_epoch_loader(ds, epoch, recipe, workers, limit=n_steps)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0, run, step = time.time(), 0.0, 0
        for samples, targets in dl:
            samples = samples.to(device)
            if selftest_downscale:   # selftest only (CPU memory); normalised boxes stay valid
                import torch.nn.functional as F
                from util.misc import NestedTensor
                x = F.interpolate(samples.tensors, size=(selftest_downscale,) * 2, mode="bilinear", align_corners=False)
                samples = NestedTensor(x, torch.zeros(x.shape[0], x.shape[2], x.shape[3], dtype=torch.bool))
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss = G.forward_loss(model, criterion, samples, targets, "fp32", device)
            if not math.isfinite(loss.item()):
                raise SystemExit(f"[G4F] non-finite loss at epoch {epoch} step {step + 1}; stopping. "
                                 f"Resume restarts this epoch from {resume_p.name}.")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
            opt.step()
            step += 1
            run += loss.item()
            if step % recipe["log_every"] == 0 or step == n_steps:
                el = time.time() - t0
                mem = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0
                log(f"  epoch {epoch}/{epochs} step {step}/{n_steps} loss {loss.item():.4f} avg {run / step:.4f} | "
                    f"{el / step:.3f} s/step (~{el / step * (n_steps - step) / 60:.0f} min left in epoch) | "
                    f"peak {mem:.2f} GB | lr {opt.param_groups[0]['lr']:.1e}")
            if step >= n_steps:
                break
        train_s = time.time() - t0
        meta = {"version": VERSION, "script_sha256": sha256_file(Path(__file__).resolve()), "cfg": CFG,
                "timm_override": timm_override, "recipe": dict(recipe, milestones=list(recipe["milestones"])),
                "epoch": epoch, "epoch_label": f"G4F_CNBT_fp32_epoch{epoch:02d}", "steps": n_steps,
                "avg_loss": run / max(step, 1), "lr_factor": f, "first_ids_positions": order[:4]}
        ck_p = out_dir / f"g4f_epoch{epoch:02d}.pth"
        save_ckpt(ck_p, model, meta)
        save_ckpt(resume_p, model, meta, opt)
        log(f"  [saved] {ck_p.name} and {resume_p.name} ({train_s / 60:.1f} min training)")
        rec = {"epoch": epoch, "avg_loss": run / max(step, 1), "train_min": train_s / 60, "lr_factor": f,
               "plain": None, "tiled": None}
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            cache, sec = export_with_model(G, M1, hr_p2, model, post, device, val_json, zip_path,
                                           recipe["eval_offset"], recipe["eval_stride"], recipe["export_batch"],
                                           image_ids=eval_image_ids,
                                           meta_extra={"source_checkpoint_path": str(ck_p.resolve()),
                                                       "source_checkpoint_sha256": sha256_file(ck_p),
                                                       "source_checkpoint_epoch": meta["epoch_label"], "g4": meta},
                                           log=log)
            cp = out_dir / "tile_cache" / f"r0_tile_cache_g4f_e{epoch:02d}_stride{recipe['eval_stride']}_offset{recipe['eval_offset']}.json.gz"
            cp.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(cp, "wt", encoding="utf-8") as fh:
                json.dump(cache, fh)
            sc = score_cache(M1, hr_p2, cache, val)
            rec.update({"plain": sc["plain"], "tiled": sc["tiled"], "per_class_tiled": sc.get("tiled_per_class"),
                        "eval_min": sec / 60, "cache": str(cp), "cache_sha256": sha256_file(cp),
                        "eval_peak_gb": torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None})
            log(f"  [eval offset {recipe['eval_offset']}] epoch {epoch}: plain {sc['plain']:.4f}  tiled {sc['tiled']:.4f} "
                f"({sec / 60:.1f} min)")
        except Exception as e:  # noqa: BLE001 -- training must survive an evaluation failure
            rec["eval_error"] = f"{type(e).__name__}: {str(e)[:300]}"
            log(f"  [eval FAILED, training continues; re-export later with --mode export] {rec['eval_error']}")
        curve = [r for r in curve if r["epoch"] != epoch] + [rec]
        curve_p.write_text(json.dumps(curve, indent=1), encoding="utf-8")
        if stop_after_epoch and epoch >= stop_after_epoch:
            break
    return model, curve


# =============================================================================
# Selftest
# =============================================================================
def selftest(a):
    import tempfile
    import torch
    print(f"[selftest] {VERSION}")
    assert os.environ.get("R0_MSDA_BACKEND") == "pytorch", "set R0_MSDA_BACKEND=pytorch"
    G = load_pilot()
    P, M2, M1, hr_p2 = G.r0_imports(a.pipeline_dir)
    assert [lr_factor(e) for e in range(1, 16)] == [1.0] * 11 + [0.1] * 3 + [0.1 ** 2]
    assert epoch_order(50, 3, 42) == epoch_order(50, 3, 42) != epoch_order(50, 4, 42)
    assert select_epoch([{"epoch": 1, "tiled": 0.3}, {"epoch": 2, "tiled": 0.5}, {"epoch": 3, "tiled": 0.5}]) == 2
    print("  [ok] LR = MultiStepLR([11, 14]) per epoch; seeded per-epoch order; best-epoch rule (ties -> earlier)")
    with tempfile.TemporaryDirectory() as td:
        zp, tj, vj = G._synthetic_world(td)
        _a, m0, _c, _p, _ = G.build_arm(P, CFG, "cpu", init="none", pretrained_backbone=False, timm_override="convnext_tiny")
        fake = Path(td) / "coco.pth"
        torch.save({"model": m0.state_dict()}, fake)
        del m0
        kw = dict(steps_per_epoch=3, epochs=2, zip_path=zp, train_json=tj, val_json=vj, pipeline_dir=a.pipeline_dir,
                  coco_ckpt=fake, pretrained=False, timm_override="convnext_tiny", workers=0, log=lambda *x: None,
                  selftest_downscale=256)
        # uninterrupted 2 epochs
        mA, curveA = run_training(G, P, M2, M1, hr_p2, "cpu", Path(td) / "A", **kw)
        # interrupted after epoch 1, then resumed
        run_training(G, P, M2, M1, hr_p2, "cpu", Path(td) / "B", stop_after_epoch=1, **kw)
        mB, curveB = run_training(G, P, M2, M1, hr_p2, "cpu", Path(td) / "B", resume=True, **kw)
        sa, sb = mA.state_dict(), mB.state_dict()
        maxdiff = max((sa[k].float() - sb[k].float()).abs().max().item() for k in sa)
        assert maxdiff < 1e-6, maxdiff
        assert [r["epoch"] for r in curveB] == [1, 2] and all(r["tiled"] is not None for r in curveA + curveB), curveB
        print(f"  [ok] 2 epochs on CPU; stop after epoch 1 + resume reproduces the uninterrupted run (max |diff| {maxdiff:.1e}); "
              f"inline offset-5 export + pycocotools scoring per epoch")
        # checkpoint -> m2_evaluate-compatible cache via the export mode path
        ck = Path(td) / "A" / "g4f_epoch02.pth"
        model, post, _ck = G.rebuild_from_ckpt(P, ck, "cpu")
        c1, _ = export_with_model(G, M1, hr_p2, model, post, "cpu", vj, zp, 5, 10, 4)
        c0 = json.loads(gzip.open(curveA[-1]["cache"], "rt", encoding="utf-8").read())
        l2c = {int(k): int(v) for k, v in c1["meta"]["label_to_cat_id"].items()}
        par = G.compare_caches(M1, c1, c0, l2c)
        assert par["ok"] and par["max_box_dev_px"] == 0.0, par
        print("  [ok] rebuilt epoch checkpoint re-exports the inline cache exactly")
        assert (Path(td) / "A" / "resume_latest.pth").exists()
        try:
            run_training(G, P, M2, M1, hr_p2, "cpu", Path(td) / "A", **kw)
            raise AssertionError("fresh run over an existing run was not refused")
        except SystemExit:
            pass
        print("  [ok] a fresh run refuses to overwrite an existing run folder")
    print("SELFTEST PASS")


# =============================================================================
# Smoke (GPU)
# =============================================================================
def smoke(a):
    import shutil
    import torch
    print(f"[smoke] {VERSION}")
    if os.environ.get("R0_MSDA_BACKEND") != "pytorch":
        sys.exit("set R0_MSDA_BACKEND=pytorch first")
    assert torch.cuda.is_available()
    G = load_pilot()
    P, M2, M1, hr_p2 = G.r0_imports(a.pipeline_dir)
    out = OUT / "smoke"
    if out.exists():
        shutil.rmtree(out)
    rep = {"version": VERSION}
    _val, ids5 = val_ids(M1, G.VAL_JSON, RECIPE["eval_offset"], RECIPE["eval_stride"])
    three = ids5[:3]
    logs = []
    lg = lambda s: (logs.append(s), print("   ", s))
    # 1 two short "epochs" + inline eval (3 frames), then stop/resume equality on a third
    t0 = time.time()
    rec20 = dict(RECIPE, log_every=20)
    run_training(G, P, M2, M1, hr_p2, "cuda", out / "A", recipe=rec20, steps_per_epoch=40, epochs=2,
                 eval_image_ids=three, log=lg)
    import gc

    def free():
        gc.collect()
        torch.cuda.empty_cache()
    free()
    run_training(G, P, M2, M1, hr_p2, "cuda", out / "B", steps_per_epoch=40, epochs=2, eval_image_ids=three,
                 stop_after_epoch=1, log=lg)
    free()
    # what --resume will load: the optimizer must carry 40 steps of state and the model must equal epoch 1
    rz = torch.load(out / "B" / "resume_latest.pth", map_location="cpu", weights_only=False)
    e1B = torch.load(out / "B" / "g4f_epoch01.pth", map_location="cpu", weights_only=False)["model"]
    steps_in_opt = {int(v["step"]) for v in rz["optimizer"]["state"].values()}
    rep["resume_file"] = {"epoch": rz["g4"]["epoch"], "optimizer_steps": sorted(steps_in_opt),
                          "model_equals_epoch01": all(torch.equal(rz["model"][k], e1B[k]) for k in e1B)}
    del rz
    run_training(G, P, M2, M1, hr_p2, "cuda", out / "B", steps_per_epoch=40, epochs=2, eval_image_ids=three,
                 resume=True, log=lg)
    free()
    ca = json.loads((out / "A" / "curve.json").read_text(encoding="utf-8"))
    cb = json.loads((out / "B" / "curve.json").read_text(encoding="utf-8"))
    ld = lambda r, e: torch.load(out / r / f"g4f_epoch{e:02d}.pth", map_location="cpu", weights_only=False)
    a1, b1, a2, b2 = ld("A", 1), ld("B", 1), ld("A", 2), ld("B", 2)
    diff = lambda x, y: max((x["model"][k].float() - y["model"][k].float()).abs().max().item() for k in x["model"])
    # GPU kernels (grid_sample backward atomics) are not bit-deterministic, so two FRESH identical runs already
    # differ after epoch 1.  The resumed epoch 2 must look like that same noise, not like a broken resume.
    rep["gpu_nondeterminism_epoch1_fresh_vs_fresh"] = diff(a1, b1)
    rep["resume_epoch2_uninterrupted_vs_resumed"] = diff(a2, b2)
    rep["epoch2_same_frame_order"] = a2["g4"]["first_ids_positions"] == b2["g4"]["first_ids_positions"]
    rep["epoch2_same_lr_factor"] = a2["g4"]["lr_factor"] == b2["g4"]["lr_factor"]
    rep["resume_epoch2_avg_loss"] = {"uninterrupted": ca[-1]["avg_loss"], "resumed": cb[-1]["avg_loss"],
                                     "epoch1_fresh_A": ca[0]["avg_loss"], "epoch1_fresh_B": cb[0]["avg_loss"]}
    rep["resume_ok"] = (rep["resume_file"]["epoch"] == 1 and rep["resume_file"]["optimizer_steps"] == [40]
                        and rep["resume_file"]["model_equals_epoch01"] and rep["epoch2_same_frame_order"]
                        and rep["epoch2_same_lr_factor"]
                        and rep["resume_epoch2_uninterrupted_vs_resumed"] <= 5 * max(rep["gpu_nondeterminism_epoch1_fresh_vs_fresh"], 1e-6))
    del a1, b1, a2, b2
    rep["curve_A"] = ca
    rep["eval_ran"] = all(r.get("tiled") is not None for r in ca)
    rep["eval_peak_gb"] = max((r.get("eval_peak_gb") or 0) for r in ca)

    def avg_at(step):   # cumulative s/step printed at `step` in run A, epoch 1
        for s in logs:
            if s.strip().startswith(f"epoch 1/2 step {step}/40"):
                return float(s.split(" s/step")[0].split("| ")[-1])
        return None
    a20, a40 = avg_at(20), avg_at(40)
    rep["s_per_step_fp32"] = (40 * a40 - 20 * a20) / 20 if (a20 and a40) else None   # steps 21-40: no worker start-up
    rep["eval_sec_per_frame"] = ca[-1]["eval_min"] * 60 / 3 if ca[-1].get("eval_min") else None
    # 2 export parity of THIS export path: R0 epoch 15 vs the existing M1.1 cache, 3 offset-0 frames
    with gzip.open(G.R0E15_CACHE, "rt", encoding="utf-8") as f:
        ref = json.load(f)
    first3 = [int(k) for k in sorted(ref["frames"], key=int)[:3]]
    m, post, _ck = G.rebuild_from_ckpt(P, G.R0E15_CKPT, "cuda")
    mine, _ = export_with_model(G, M1, hr_p2, m, post, "cuda", G.VAL_JSON, G.ZIP, 0, 10, 4, image_ids=first3)
    l2c = {int(k): int(v) for k, v in ref["meta"]["label_to_cat_id"].items()}
    rep["export_parity_vs_R0e15_cache"] = G.compare_caches(M1, mine, ref, l2c)
    # 3 projections
    steps = 20310 // 2
    ev = (rep["eval_sec_per_frame"] or 0) * 432 / 3600
    rep["projection_h"] = {"train_per_epoch": (rep["s_per_step_fp32"] or 0) * steps / 3600, "eval_per_epoch": ev,
                           "total_15_epochs": 15 * ((rep["s_per_step_fp32"] or 0) * steps / 3600 + ev)}
    rep["disk_gb_per_epoch_ckpt"] = (out / "A" / "g4f_epoch01.pth").stat().st_size / 2 ** 30
    rep["checks"] = {"resume": rep["resume_ok"], "inline_eval": rep["eval_ran"],
                     "export_parity": rep["export_parity_vs_R0e15_cache"]["ok"],
                     "eval_peak_under_11GB": rep["eval_peak_gb"] < 11}
    (out / "g4f_smoke_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "curve_A"}, indent=1))
    print(f"  smoke wall time {(time.time() - t0) / 60:.1f} min")
    print("SMOKE PASS" if all(rep["checks"].values()) else "SMOKE FAIL")


# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["selftest", "smoke", "train", "export", "curve"], required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--epoch", type=int)
    ap.add_argument("--offset", type=int, choices=[0, 5])
    ap.add_argument("--pipeline-dir", default="D:/MARIVISION/H0_workbench/eval_tools")
    a = ap.parse_args()
    if a.mode == "selftest":
        return selftest(a)
    if a.mode == "smoke":
        return smoke(a)
    run_dir = OUT / "run"
    if a.mode == "curve":
        curve = json.loads((run_dir / "curve.json").read_text(encoding="utf-8"))
        print("| epoch | LR factor | avg loss | plain AP30 (offset 5) | tiled AP30 (offset 5) |\n|---|---|---|---|---|")
        for r in sorted(curve, key=lambda r: r["epoch"]):
            f = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
            print(f"| {r['epoch']} | {r['lr_factor']} | {r['avg_loss']:.3f} | {f(r.get('plain'))} | {f(r.get('tiled'))} |")
        print(f"\nselected epoch (offset-5 tiled, ties -> earlier): {select_epoch(curve)}")
        return
    if os.environ.get("R0_MSDA_BACKEND") != "pytorch":
        sys.exit("set R0_MSDA_BACKEND=pytorch first")
    G = load_pilot()
    P, M2, M1, hr_p2 = G.r0_imports(a.pipeline_dir)
    if a.mode == "train":
        run_training(G, P, M2, M1, hr_p2, "cuda", run_dir, resume=a.resume)
        return
    if a.mode == "export":
        if a.epoch is None or a.offset is None:
            sys.exit("--epoch and --offset are required")
        ck = run_dir / f"g4f_epoch{a.epoch:02d}.pth"
        cp = run_dir / "tile_cache" / f"r0_tile_cache_g4f_e{a.epoch:02d}_stride10_offset{a.offset}.json.gz"
        if a.offset == 0 and cp.exists():
            sys.exit(f"[REFUSED] {cp} exists")
        m, post, ckd = G.rebuild_from_ckpt(P, ck, "cuda")
        cache, sec = export_with_model(G, M1, hr_p2, m, post, "cuda", G.VAL_JSON, G.ZIP, a.offset, 10, 4,
                                       meta_extra={"source_checkpoint_path": str(ck.resolve()),
                                                   "source_checkpoint_sha256": sha256_file(ck),
                                                   "source_checkpoint_epoch": ckd.get("epoch"), "g4": ckd.get("g4")})
        cp.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(cp, "wt", encoding="utf-8") as fh:
            json.dump(cache, fh)
        print(f"  wrote {cp} ({sec / 60:.1f} min)\n  sha256 {sha256_file(cp)}")


if __name__ == "__main__":
    main()
