r"""
G4 pilot: four small-budget Deformable-DETR arms, paired, to decide the long run.

Location: D:/MARIVISION/R0_frozen_baseline/g4_pilot.py
Run FROM D:\MARIVISION\R0_frozen_baseline (R0's frozen model / dataset / r0_preflight code, as M2 did),
with R0_MSDA_BACKEND=pytorch and HF_HUB_DISABLE_SYMLINKS_WARNING=1.

Arms (identical recipe; only the named factor differs):
  R50_fp32   R0 setting                 (stage 2)
  R50_bf16   R0 setting, bf16 autocast  (stage 2)
  CNB_bf16   ConvNeXt-B in22k backbone  (stage 1)
  CNBT_bf16  ConvNeXt-B + M2 tiles      (stage 1)

Recipe (frozen): every 5th TRAIN frame of every video (4,062 frames), 10,155 steps at batch 2 (= one R0 epoch of
steps), the same seeded frame order for every arm, seed 42, R0's AdamW groups and LRs (2e-4 / backbone 2e-5 /
linear_proj x0.1), weight decay 1e-4, grad clip 0.1, LR x0.1 at 80 % of the steps, COCO Deformable-DETR init
(R50 arms: everything but the class head, exactly as R0; ConvNeXt arms: transformer + queries + box head, new
backbone / input projections / class head), checkpoints at 50 % and 100 %.  Tiles (CNBT only): M2's
SeaClipsM2Train, p_tile 0.5, min_visible 0.5, imported unchanged from m2_train_r0_finetune.py (M2.1).
Inference for every arm: fp32, R0's eval transform, the 17-tile M1 schedule, cached in the M1.1 tile-cache schema
so that the unchanged m2_evaluate.py scores it (plain = full frame, tiled = V0 merge).

Modes:
  selftest   CPU, synthetic zip (sandbox or workstation)
  smoke      GPU, real data: 20 steps of each arm with the real loader, bf16-vs-fp32 loss parity, checkpoint
             reload, export parity of THIS export path against the existing R0 epoch-15 tile cache, projections
  train      --arm <name>
  export     --arm <name> --at 50|100   -> tile cache for m2_evaluate.py
Nothing reads TEST.  Outputs stay on the workstation (dataset-derived).
"""
from PIL import Image  # noqa: F401  -- PIL before torch (Director's Windows env)
import sys
if "wandb" not in sys.modules:          # timm's optional wandb import is noisy under -W error; never used
    sys.modules["wandb"] = None
import argparse
import gzip
import hashlib
import json
import math
import os
import time
import warnings
from collections import defaultdict
from pathlib import Path

VERSION = "G4P.0"

ZIP = Path("D:/MARIVISION/SeaClips.zip")
TRAIN_JSON = Path("D:/MARIVISION/seaclips-train.json")
VAL_JSON = Path("D:/MARIVISION/seaclips-val.json")
COCO_CKPT = Path("D:/MARIVISION/r50_deformable_detr-checkpoint.pth")
R0E15_CKPT = Path("D:/MARIVISION/checkpoints_r0/checkpoint_epoch_15.pth")
R0E15_CACHE = Path("D:/MARIVISION/R0_frozen_baseline/m1_1_tile_cache/r0_tile_cache_epoch15_stride10_offset0.json.gz")
PIPELINE_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
OUT = Path("D:/MARIVISION/G4_pilot")
M1_SHA = "73044625fb5b477f857018ba1ace3c0dcfe32f08d2b27d0beccd8d333153a388"   # m1_r0_tiled_inference.py M1.0
M2_SHA = "8711c96c730bc21916591cb8f2d91d11c73fe4d17bec17716d729e050b700f1f"   # m2_train_r0_finetune.py M2.1
CACHE_SCHEMA = "MARIVISION_M1_1_R0_TILE_CACHE_V1"

CNB = "convnext_base.fb_in22k"
ARMS = {
    "R50_fp32": {"backbone": None, "precision": "fp32", "p_tile": 0.0},
    "R50_bf16": {"backbone": None, "precision": "bf16", "p_tile": 0.0},
    "CNB_bf16": {"backbone": CNB, "precision": "bf16", "p_tile": 0.0},
    "CNBT_bf16": {"backbone": CNB, "precision": "bf16", "p_tile": 0.5},
}
RECIPE = {"steps": 10155, "batch": 2, "seed": 42, "frame_stride": 5, "lr_drop_frac": 0.8, "lr_drop_factor": 0.1,
          "save_fracs": (0.5, 1.0), "min_visible": 0.5, "workers": 2, "log_every": 250,
          "val_frame_stride": 10, "val_frame_offset": 0, "export_batch": 4}

KNOWN_REPO_WARNINGS = [r"The parameter 'pretrained' is deprecated",
                       r"Arguments other than a weight enum or `None` for 'weights' are deprecated",
                       r"torch.meshgrid: in an upcoming release"]


def silence_known():
    for m in KNOWN_REPO_WARNINGS:
        warnings.filterwarnings("ignore", message=m, category=UserWarning)


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


# =============================================================================
# R0-folder imports (must run from the R0 folder)
# =============================================================================
def r0_imports(pipeline_dir=PIPELINE_DIR, check_shas=True):
    silence_known()
    cwd = Path.cwd()
    for need in ("r0_preflight.py", "m2_train_r0_finetune.py", "m1_r0_tiled_inference.py", "models", "datasets"):
        if not (cwd / need).exists():
            raise SystemExit(f"[REFUSED] run from the R0 folder (D:\\MARIVISION\\R0_frozen_baseline): {need} not found in {cwd}")
    if check_shas:
        for f, s in (("m1_r0_tiled_inference.py", M1_SHA), ("m2_train_r0_finetune.py", M2_SHA)):
            got = sha256_file(cwd / f)
            if got != s:
                raise SystemExit(f"[REFUSED] {f} sha {got} != expected {s}")
    sys.path.insert(0, str(cwd))
    sys.path.insert(1, str(Path(pipeline_dir).resolve()))
    import r0_preflight as P
    import m2_train_r0_finetune as M2
    import m1_r0_tiled_inference as M1
    import marivision_hr_p2_pipeline as hr_p2
    return P, M2, M1, hr_p2


# =============================================================================
# Model
# =============================================================================
def make_timm_backbone(timm_id, pretrained):
    import torch
    import torch.nn.functional as F
    from torch import nn
    import timm
    from util.misc import NestedTensor

    class TimmBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = timm.create_model(timm_id, pretrained=pretrained, features_only=True, out_indices=(1, 2, 3))
            self.num_channels = list(self.body.feature_info.channels())
            self.strides = list(self.body.feature_info.reduction())
            assert self.strides == [8, 16, 32], self.strides
            for n, p in self.body.named_parameters():       # as G3 / R0: stem + first stage frozen
                if n.startswith(("stem", "stages_0", "stages.0")):
                    p.requires_grad_(False)

        def forward(self, tensor_list):
            out = {}
            for i, x in enumerate(self.body(tensor_list.tensors)):
                m = F.interpolate(tensor_list.mask[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
                out[str(i)] = NestedTensor(x, m)
            return out

    return TimmBackbone()


def swap_backbone(model, timm_id, pretrained):
    from torch import nn
    from models.backbone import Joiner
    tb = make_timm_backbone(timm_id, pretrained)
    model.backbone = Joiner(tb, model.backbone[1])
    hidden, projs, in_ch = model.transformer.d_model, [], None
    for c in tb.num_channels:
        projs.append(nn.Sequential(nn.Conv2d(c, hidden, kernel_size=1), nn.GroupNorm(32, hidden)))
        in_ch = c
    for _ in range(model.num_feature_levels - len(tb.num_channels)):
        projs.append(nn.Sequential(nn.Conv2d(in_ch, hidden, kernel_size=3, stride=2, padding=1), nn.GroupNorm(32, hidden)))
        in_ch = hidden
    for p in projs:
        nn.init.xavier_uniform_(p[0].weight, gain=1)
        nn.init.constant_(p[0].bias, 0)
    model.input_proj = nn.ModuleList(projs)


def coco_transfer(model, ckpt_path, new_backbone):
    import torch
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    own = model.state_dict()
    skip = ("class_embed",) + (("backbone.", "input_proj.") if new_backbone else ())
    take = {k: v for k, v in sd.items() if not k.startswith(skip) and k in own and own[k].shape == v.shape}
    bad = [k for k, v in sd.items() if not k.startswith(skip) and k in own and own[k].shape != v.shape]
    if bad:
        raise SystemExit(f"[REFUSED] COCO transfer shape mismatch outside the skipped parts: {bad[:5]}")
    model.load_state_dict(take, strict=False)
    return len(take)


def build_arm(P, cfg, device, init="coco", pretrained_backbone=True, coco_ckpt=COCO_CKPT, timm_override=None):
    """init: 'coco' (training start) or 'none' (rebuild for a checkpoint load)."""
    silence_known()
    from models.deformable_detr import build
    args = P.get_authoritative_r0_args()
    args.device = device
    if hasattr(args, "conditioning_mode"):
        args.conditioning_mode = None
    with warnings.catch_warnings():
        silence_known()
        model, criterion, post = build(args)
    tid = timm_override or cfg["backbone"]
    if tid:
        swap_backbone(model, tid, pretrained_backbone)
    n = coco_transfer(model, coco_ckpt, bool(tid)) if init == "coco" else 0
    model.to(device)
    criterion.to(device)
    return args, model, criterion, post, n


def param_groups(P, model, args):
    m = P.match_name_keywords
    core = [p for n, p in model.named_parameters() if not m(n, args.lr_backbone_names)
            and not m(n, args.lr_linear_proj_names) and p.requires_grad]
    bb = [p for n, p in model.named_parameters() if m(n, args.lr_backbone_names) and p.requires_grad]
    lin = [p for n, p in model.named_parameters() if m(n, args.lr_linear_proj_names) and p.requires_grad]
    assert len(core) + len(bb) + len(lin) == sum(1 for p in model.parameters() if p.requires_grad)
    return [{"params": core, "lr": args.lr}, {"params": bb, "lr": args.lr_backbone},
            {"params": lin, "lr": args.lr * args.lr_linear_proj_mult}]


def lr_at(step, base, recipe=RECIPE):
    """Learning rate for the 0-based optimizer step: base until lr_drop_frac, then base x factor."""
    return base * (recipe["lr_drop_factor"] if step >= int(round(recipe["lr_drop_frac"] * recipe["steps"])) else 1.0)


# =============================================================================
# Data
# =============================================================================
def subset_indices(ds, stride):
    """Positions in ds of every `stride`-th frame of every video (sorted by frame_id, id)."""
    by_v = defaultdict(list)
    for pos, iid in enumerate(ds.img_ids):
        im = ds.images[iid]
        by_v[video_of(im["file_name"])].append((int(im.get("frame_id", iid)), int(iid), pos))
    out = []
    for v in sorted(by_v):
        lst = sorted(by_v[v])
        out += [pos for i, (_, _, pos) in enumerate(lst) if i % stride == 0]
    return out


class FixedOrderSampler:
    """Seeded permutations of the subset, concatenated to exactly n_samples.  Identical for every arm."""

    def __init__(self, indices, n_samples, seed):
        import torch
        g = torch.Generator().manual_seed(seed)
        order = []
        while len(order) < n_samples:
            order += [indices[i] for i in torch.randperm(len(indices), generator=g).tolist()]
        self.order = order[:n_samples]

    def __iter__(self):
        return iter(self.order)

    def __len__(self):
        return len(self.order)


def make_loader(M2, cfg, recipe, zip_path, train_json, pipeline_dir, n_steps, workers):
    import torch
    from torch.utils.data import DataLoader
    from util.misc import collate_fn
    DS = M2.get_dataset_class()
    ds = DS(Path(zip_path), Path(train_json), p_tile=cfg["p_tile"], min_visible=recipe["min_visible"],
            pipeline_dir=str(pipeline_dir), random_flip=True)
    idx = subset_indices(ds, recipe["frame_stride"])
    sampler = FixedOrderSampler(idx, n_steps * recipe["batch"], recipe["seed"])
    g = torch.Generator().manual_seed(recipe["seed"])
    dl = DataLoader(ds, batch_size=recipe["batch"], sampler=sampler, collate_fn=collate_fn, num_workers=workers,
                    drop_last=True, generator=g, pin_memory=torch.cuda.is_available(),
                    persistent_workers=workers > 0, multiprocessing_context=("spawn" if workers > 0 else None))
    return ds, idx, sampler, dl


# =============================================================================
# Training
# =============================================================================
def forward_loss(model, criterion, samples, targets, precision, device):
    import torch
    dt = "cuda" if str(device).startswith("cuda") else "cpu"
    with torch.autocast(device_type=dt, dtype=torch.bfloat16, enabled=(precision == "bf16")):
        out = model(samples)
    out = {k: v.float() for k, v in out.items() if k != "aux_outputs"} | \
          {"aux_outputs": [{kk: vv.float() for kk, vv in a.items()} for a in out.get("aux_outputs", [])]}
    ld = criterion(out, targets)
    return sum(ld[k] * criterion.weight_dict[k] for k in ld if k in criterion.weight_dict)


def train(arm, P, M2, device, out_dir, recipe=RECIPE, n_steps=None, save_fracs=None, workers=None,
          zip_path=ZIP, train_json=TRAIN_JSON, pipeline_dir=PIPELINE_DIR, coco_ckpt=COCO_CKPT,
          pretrained_backbone=True, timm_override=None, log=print, tag=None, selftest_downscale=None):
    import torch
    cfg = ARMS[arm]
    n_steps = n_steps or recipe["steps"]
    save_at = {max(1, int(round(f * n_steps))) for f in (save_fracs or recipe["save_fracs"])}
    torch.manual_seed(recipe["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(recipe["seed"])
        torch.cuda.reset_peak_memory_stats()
    args, model, criterion, _post, n_coco = build_arm(P, cfg, device, "coco", pretrained_backbone, coco_ckpt, timm_override)
    groups = param_groups(P, model, args)
    base_lrs = [g["lr"] for g in groups]
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    ds, idx, sampler, dl = make_loader(M2, cfg, recipe, zip_path, train_json, pipeline_dir, n_steps,
                                       recipe["workers"] if workers is None else workers)
    log(f"[G4 {VERSION}] arm {arm} {cfg} | frames {len(idx)} of {len(ds)} | steps {n_steps} | COCO tensors {n_coco} | "
        f"LRs {base_lrs} (x{recipe['lr_drop_factor']} from step {int(round(recipe['lr_drop_frac'] * recipe['steps']))})")
    model.train()
    criterion.train()
    out_dir.mkdir(parents=True, exist_ok=True)
    step, t0, hist, run, n_tile, n_samp, n_box = 0, time.time(), [], 0.0, 0, 0, 0
    first_order = []
    for samples, targets in dl:
        if step >= n_steps:
            break
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = lr_at(step, b, dict(recipe, steps=n_steps))
        samples = samples.to(device)
        if selftest_downscale:      # selftest only (CPU memory): shrink the 640 input; normalised boxes stay valid
            import torch.nn.functional as F
            from util.misc import NestedTensor
            x = F.interpolate(samples.tensors, size=(selftest_downscale, selftest_downscale), mode="bilinear",
                              align_corners=False)
            samples = NestedTensor(x, torch.zeros(x.shape[0], x.shape[2], x.shape[3], dtype=torch.bool, device=x.device))
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        if step < 3:
            first_order += [int(t["image_id"].item()) for t in targets]
        for t in targets:
            n_samp += 1
            n_tile += int(t["m2_is_tile"].item())
            n_box += int(t["labels"].numel())
        loss = forward_loss(model, criterion, samples, targets, cfg["precision"], device)
        if not math.isfinite(loss.item()):
            raise SystemExit(f"[G4] non-finite loss at step {step + 1}; stopping (nothing saved after this point).")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
        opt.step()
        step += 1
        run += loss.item()
        if step % recipe["log_every"] == 0 or step == n_steps or step <= 2:
            el = time.time() - t0
            mem = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0
            rec = {"step": step, "loss": loss.item(), "avg_loss": run / step, "s_per_step": el / step,
                   "eta_min": el / step * (n_steps - step) / 60, "peak_gb": mem, "lr0": opt.param_groups[0]["lr"],
                   "tile_frac": n_tile / max(1, n_samp), "boxes_per_sample": n_box / max(1, n_samp)}
            hist.append(rec)
            log(f"  step {step}/{n_steps} loss {rec['loss']:.4f} avg {rec['avg_loss']:.4f} | {rec['s_per_step']:.3f} s/step "
                f"(~{rec['eta_min']:.0f} min left) | peak {mem:.2f} GB | lr {rec['lr0']:.1e} | tiles {rec['tile_frac']:.2f}")
        if step in save_at:
            p = out_dir / f"g4_{arm}{'_' + tag if tag else ''}_step{step:05d}.pth"
            torch.save({"model": model.state_dict(), "epoch": f"G4_{arm}_step{step}",
                        "g4": {"version": VERSION, "script_sha256": sha256_file(Path(__file__).resolve()), "arm": arm,
                               "cfg": cfg, "timm_override": timm_override, "recipe": dict(recipe, steps=n_steps),
                               "step": step, "frames": len(idx), "first_image_ids": first_order, "history": hist}}, p)
            log(f"  [saved] {p}")
    return model, hist, first_order


def rebuild_from_ckpt(P, path, device):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    g = ck.get("g4")
    cfg = g["cfg"] if g else {"backbone": None, "precision": "fp32", "p_tile": 0.0}
    _a, model, _c, post, _ = build_arm(P, cfg, device, init="none", pretrained_backbone=False,
                                       timm_override=g.get("timm_override") if g else None)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    return model, post, ck


# =============================================================================
# Export: M1.1 tile-cache schema, fp32 inference
# =============================================================================
def export_cache(P, M1, hr_p2, ckpt_path, out_path, device, val_json=VAL_JSON, zip_path=ZIP, image_ids=None,
                 batch=RECIPE["export_batch"], log=print):
    import torch
    import torchvision.transforms.functional as TF
    from datasets.seaclips_detection import SeaClipsDetection
    from util.misc import NestedTensor
    val_json = Path(val_json)
    if "test" in val_json.name.lower() or "val" not in val_json.name.lower():
        raise SystemExit("[SPLIT BLOCK] VAL json only; TEST is locked.")
    model, post, ck = rebuild_from_ckpt(P, ckpt_path, device)
    val = json.loads(val_json.read_text(encoding="utf-8"))
    ids = M1.eval_subset_ids(val["images"], RECIPE["val_frame_stride"], RECIPE["val_frame_offset"])
    if image_ids is not None:
        ids = [i for i in ids if i in set(image_ids)] or list(image_ids)
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        ds = SeaClipsDetection(Path(zip_path), val_json, is_train=False, random_flip=False)
    l2c = {int(k): int(v) for k, v in dict(ds.label_to_cat_id).items()}
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
            if n % 50 == 0 or n == len(ids):
                el = time.time() - t0
                log(f"  export {n}/{len(ids)} frames, {el:.0f}s (~{el / n * (len(ids) - n) / 60:.1f} min left)")
    meta = {"schema": CACHE_SCHEMA, "script_version": VERSION, "script_sha256": sha256_file(Path(__file__).resolve()),
            "m1_module_sha256": sha256_file(Path(M1.__file__).resolve()),
            "pipeline_py_sha256": sha256_file(Path(hr_p2.__file__).resolve()),
            "source_checkpoint_path": str(Path(ckpt_path).resolve()), "source_checkpoint_sha256": sha256_file(ckpt_path),
            "source_checkpoint_epoch": ck.get("epoch"), "g4": {k: v for k, v in (ck.get("g4") or {}).items() if k != "history"},
            "coco_val_sha256": sha256_file(val_json), "val_frame_stride": RECIPE["val_frame_stride"],
            "val_frame_offset": RECIPE["val_frame_offset"], "n_frames": len(ids),
            "label_to_cat_id": {str(k): v for k, v in sorted(l2c.items())},
            "box_format": "per tile: [score, r0_label, x1, y1, x2, y2] in TILE pixels (tile = native crop, no scaling)",
            "provenance": {"status": "SKIPPED_NO_REFERENCE_EXPORT",
                           "reason": "G4 checkpoint; export path verified against the R0 epoch-15 cache in the G4 smoke"},
            "inference_precision": "fp32", "batch_size": batch, "torch_version": torch.__version__, "device": str(device)}
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(out_path, "wt", encoding="utf-8") as f:
            json.dump({"meta": meta, "frames": frames}, f)
    return {"meta": meta, "frames": frames}, time.time() - t0


def compare_caches(M1, mine, ref, l2c, score_tol=0.02, box_tol=0.5):
    """Tile-by-tile parity (M1.1 tolerance) of two caches on their common frames."""
    worst = {"frames": 0, "tiles": 0, "tile_window_mismatch": 0, "failed_tiles": 0, "max_box_dev_px": 0.0}
    for iid, fr in mine["frames"].items():
        if iid not in ref["frames"]:
            continue
        rf = ref["frames"][iid]
        worst["frames"] += 1
        for a, b in zip(fr["tiles"], rf["tiles"]):
            worst["tiles"] += 1
            if list(a[:5]) != list(b[:5]):
                worst["tile_window_mismatch"] += 1
                continue
            da = M1.plain_dets_native([d[0] for d in a[5]], [d[1] for d in a[5]], [d[2:6] for d in a[5]], l2c)
            db = M1.plain_dets_native([d[0] for d in b[5]], [d[1] for d in b[5]], [d[2:6] for d in b[5]], l2c)
            pr = M1.parity_check(da, db, score_tol, box_tol)
            worst["max_box_dev_px"] = max(worst["max_box_dev_px"], pr["max_box_dev_px"])
            worst["failed_tiles"] += int(not pr["ok"])
    worst["ok"] = worst["frames"] > 0 and worst["tile_window_mismatch"] == 0 and worst["failed_tiles"] == 0
    return worst


# =============================================================================
# Selftest (CPU, synthetic zip)
# =============================================================================
def _synthetic_world(td):
    import io as _io
    import zipfile
    import numpy as np
    rng = np.random.default_rng(0)
    cats = [{"id": i + 1, "name": n} for i, n in enumerate(
        ["ANIMAL", "BOAT", "LEISURE_VEHICLE", "MARINE_MARKER", "OBJECT", "SAILING_VESSEL", "SHIP"])]
    zp = Path(td) / "SeaClips.zip"
    out = {}
    with zipfile.ZipFile(zp, "w") as z:
        for split, nv, nf in (("train", 3, 12), ("val", 2, 12)):
            images, anns, aid = [], [], 1
            for v in range(nv):
                for f in range(nf):
                    iid = len(images) + 1
                    W, H = 480, 270
                    arr = np.full((H, W, 3), (30, 60, 90), np.uint8)
                    for k in range(3):
                        x, y, w, h = float(rng.uniform(10, 400)), float(rng.uniform(100, 200)), 30.0, 16.0
                        arr[int(y):int(y + h), int(x):int(x + w)] = (240, 40 * k, 20)
                        anns.append({"id": aid, "image_id": iid, "category_id": 2 + k, "bbox": [x, y, w, h],
                                     "area": w * h, "iscrowd": 0})
                        aid += 1
                    fn = f"video_{v + 1:02d}/{f + 1:06d}.png"
                    b = _io.BytesIO()
                    Image.fromarray(arr).save(b, format="PNG")
                    z.writestr(f"SeaClips/{split}/{fn}", b.getvalue())
                    images.append({"id": iid, "file_name": fn, "width": W, "height": H, "video_id": v + 1, "frame_id": f})
            js = {"images": images, "annotations": anns, "categories": cats}
            p = Path(td) / f"seaclips-{split}.json"
            p.write_text(json.dumps(js), encoding="utf-8")
            out[split] = p
    return zp, out["train"], out["val"]


def selftest(a):
    import tempfile
    import torch
    print(f"[selftest] {VERSION}")
    assert os.environ.get("R0_MSDA_BACKEND") == "pytorch", "set R0_MSDA_BACKEND=pytorch"
    P, M2, M1, hr_p2 = r0_imports(a.pipeline_dir, check_shas=True)
    # 1 recipe and arms
    assert set(ARMS) == {"R50_fp32", "R50_bf16", "CNB_bf16", "CNBT_bf16"}
    diff = lambda x, y: {k for k in ARMS[x] if ARMS[x][k] != ARMS[y][k]}
    assert diff("R50_fp32", "R50_bf16") == {"precision"} and diff("R50_bf16", "CNB_bf16") == {"backbone"} \
        and diff("CNB_bf16", "CNBT_bf16") == {"p_tile"}
    assert lr_at(0, 1.0) == 1.0 and lr_at(8123, 1.0) == 1.0 and abs(lr_at(8124, 1.0) - 0.1) < 1e-12
    print("  [ok] arms differ by exactly one factor per comparison; LR x0.1 from step 8124 of 10155")
    with tempfile.TemporaryDirectory() as td:
        zp, tj, vj = _synthetic_world(td)
        # 2 subset + sampler
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            DS = M2.get_dataset_class()
            ds = DS(zp, tj, p_tile=0.0, min_visible=0.5, pipeline_dir=str(a.pipeline_dir), random_flip=True)
        idx = subset_indices(ds, 5)
        assert len(idx) == 3 * 3, idx                    # frames 0, 5, 10 of each of 3 videos
        assert all(ds.images[ds.img_ids[p]]["frame_id"] % 5 == 0 for p in idx)
        s1, s2 = FixedOrderSampler(idx, 40, 42).order, FixedOrderSampler(idx, 40, 42).order
        assert s1 == s2 and len(s1) == 40 and set(s1) == set(idx)
        print("  [ok] every 5th frame of every video; seeded frame order identical across instantiations")
        # 3 build + groups + COCO-like transfer (fake ckpt = R50 model's own init)
        _args, m0, _c, _p, _ = build_arm(P, ARMS["R50_fp32"], "cpu", init="none", pretrained_backbone=False)
        fake = Path(td) / "coco.pth"
        torch.save({"model": {k: v + 0.001 if v.is_floating_point() else v for k, v in m0.state_dict().items()}}, fake)
        for arm in ARMS:
            args, m, c, post, n = build_arm(P, ARMS[arm], "cpu", "coco", False, fake,
                                            timm_override="convnext_tiny" if ARMS[arm]["backbone"] else None)
            param_groups(P, m, args)
            k = "transformer.encoder.layers.0.self_attn.sampling_offsets.weight"
            assert torch.allclose(m.state_dict()[k], m0.state_dict()[k] + 0.001)
            del m, c, post
        del m0
        import gc
        gc.collect()
        print("  [ok] all arms build; optimizer partition complete; COCO transfer loads the transformer")
        # 4 short training on the synthetic world, each precision and the tile arm; checkpoint rebuild
        outs = {}
        for arm in ("R50_bf16", "CNBT_bf16"):
            rec = dict(RECIPE, workers=0, log_every=1000)
            model, hist, order = train(arm, P, M2, "cpu", Path(td) / arm, rec, n_steps=4, save_fracs=(1.0,), workers=0,
                                       zip_path=zp, train_json=tj, pipeline_dir=a.pipeline_dir, coco_ckpt=fake,
                                       pretrained_backbone=False,
                                       timm_override="convnext_tiny" if ARMS[arm]["backbone"] else None,
                                       log=lambda *x: None, selftest_downscale=256)
            assert all(math.isfinite(h["loss"]) for h in hist)
            outs[arm] = order
            ck = Path(td) / arm / "g4_" f"{arm}_step00004.pth"
            m2, post, meta = rebuild_from_ckpt(P, ck, "cpu")
            sd1, sd2 = model.state_dict(), m2.state_dict()
            assert all(torch.equal(sd1[k], sd2[k]) for k in sd1)
            del model, m2, sd1, sd2
            gc.collect()
        assert outs["R50_bf16"] == outs["CNBT_bf16"], "frame order differs between arms"
        print("  [ok] bf16 training steps finite (R50, ConvNeXt + tiles); same frame order in both arms; "
              "checkpoint rebuild is exact")
        # 5 export -> M1.1 cache schema -> plain + V0 tiled predictions as m2_evaluate builds them
        sys.path.insert(0, str(Path.cwd()))
        import m1_1_merge_variants as MV
        cache, sec = export_cache(P, M1, hr_p2, ck, Path(td) / "cache.json.gz", "cpu", vj, zp, image_ids=None,
                                  log=lambda *x: None)
        with gzip.open(Path(td) / "cache.json.gz", "rt", encoding="utf-8") as f:
            back = json.load(f)
        assert back["meta"]["schema"] == CACHE_SCHEMA and back["meta"]["val_frame_stride"] == 10
        l2c = {int(k): int(v) for k, v in back["meta"]["label_to_cat_id"].items()}
        n_plain = n_tiled = 0
        for iid, fr in back["frames"].items():
            tiles = MV.frame_tiles(M1, hr_p2, fr)
            assert len(tiles) == 17
            n_plain += len([t for t in tiles if t[0].stage == MV.STAGE_B][0][1])
            n_tiled += len(MV.build_variant("V0", M1, hr_p2, l2c, fr))
        assert n_plain == 100 * len(back["frames"]) and 0 < n_tiled <= 100 * len(back["frames"])
        par = compare_caches(M1, cache, back, l2c)
        assert par["ok"] and par["max_box_dev_px"] == 0.0
        print(f"  [ok] export: {len(back['frames'])} VAL frames x 17 tiles in the M1.1 schema; m2_evaluate's plain/V0 "
              f"builders read it; cache self-parity exact")
    print("SELFTEST PASS")


# =============================================================================
# Smoke (GPU, real data)
# =============================================================================
def smoke(a):
    import torch
    print(f"[smoke] {VERSION}")
    for e in ("R0_MSDA_BACKEND",):
        if os.environ.get(e) != "pytorch":
            sys.exit(f"set {e}=pytorch first")
    assert torch.cuda.is_available()
    for p in (ZIP, TRAIN_JSON, VAL_JSON, COCO_CKPT, R0E15_CKPT, R0E15_CACHE):
        if not p.exists():
            sys.exit(f"missing {p}")
    P, M2, M1, hr_p2 = r0_imports(a.pipeline_dir)
    out = OUT / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    report = {"version": VERSION, "arms": {}}
    # 1 each arm: 20 real steps with the real loader
    for arm in ARMS:
        torch.cuda.empty_cache()
        t0 = time.time()
        _m, hist, order = train(arm, P, M2, "cuda", out / arm, RECIPE, n_steps=20, save_fracs=(1.0,),
                                log=lambda s: print("   ", s))
        sps = (hist[-1]["s_per_step"] * 20 - hist[1]["s_per_step"] * 2) / 18      # excludes the first 2 steps
        report["arms"][arm] = {"s_per_step": sps, "peak_gb": hist[-1]["peak_gb"], "loss_last": hist[-1]["loss"],
                               "first_image_ids": order, "proj_train_h": sps * RECIPE["steps"] / 3600}
        del _m
    orders = {tuple(v["first_image_ids"]) for v in report["arms"].values()}
    report["same_frame_order"] = len(orders) == 1
    # 2 bf16 vs fp32 loss on identical weights and batch
    par = {}
    for arm in ("R50_bf16", "CNB_bf16"):
        torch.manual_seed(0)
        args, m, c, _p, _ = build_arm(P, ARMS[arm], "cuda", "coco", True)
        m.eval()   # deterministic forward (no dropout) for the comparison
        _ds, _idx, _s, dl = make_loader(M2, ARMS[arm], RECIPE, ZIP, TRAIN_JSON, a.pipeline_dir, 4, 0)
        lf, lb = [], []
        with torch.no_grad():
            for k, (x, t) in enumerate(dl):
                x = x.to("cuda")
                t = [{kk: vv.to("cuda") for kk, vv in tt.items()} for tt in t]
                lf.append(forward_loss(m, c, x, t, "fp32", "cuda").item())
                lb.append(forward_loss(m, c, x, t, "bf16", "cuda").item())
                if k == 3:
                    break
        rel = max(abs(p - q) / max(abs(p), 1e-6) for p, q in zip(lf, lb))
        par[arm] = {"fp32": lf, "bf16": lb, "max_rel_diff": rel, "ok": rel <= 0.05}
        del m
        torch.cuda.empty_cache()
    report["bf16_parity"] = par
    # 3 export parity: THIS export path on R0 epoch 15 vs the existing M1.1 cache (first 3 offset-0 frames)
    with gzip.open(R0E15_CACHE, "rt", encoding="utf-8") as f:
        ref = json.load(f)
    first3 = [int(k) for k in sorted(ref["frames"], key=int)[:3]]
    mine, sec = export_cache(P, M1, hr_p2, R0E15_CKPT, None, "cuda", image_ids=first3)
    l2c = {int(k): int(v) for k, v in ref["meta"]["label_to_cat_id"].items()}
    report["export_parity_vs_R0e15_cache"] = compare_caches(M1, mine, ref, l2c)
    report["export_sec_per_frame_R50"] = sec / 3
    # 4 export of a ConvNeXt smoke checkpoint (schema + speed)
    ck = out / "CNBT_bf16" / "g4_CNBT_bf16_step00020.pth"
    c2, sec2 = export_cache(P, M1, hr_p2, ck, out / "smoke_cache_CNBT.json.gz", "cuda", image_ids=first3)
    report["export_sec_per_frame_CNB"] = sec2 / 3
    report["export_432_min_CNB"] = sec2 / 3 * 432 / 60
    report["checks"] = {"same_frame_order": report["same_frame_order"],
                        "bf16_parity": all(v["ok"] for v in par.values()),
                        "export_parity": report["export_parity_vs_R0e15_cache"]["ok"],
                        "all_peak_under_11GB": all(v["peak_gb"] < 11 for v in report["arms"].values())}
    report["total_projection_h"] = {"stage1_train": report["arms"]["CNB_bf16"]["proj_train_h"] + report["arms"]["CNBT_bf16"]["proj_train_h"],
                                    "stage2_train": report["arms"]["R50_bf16"]["proj_train_h"] + report["arms"]["R50_fp32"]["proj_train_h"],
                                    "exports_each_h": report["export_432_min_CNB"] / 60}
    (out / "g4_smoke_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "arms"}, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "first_image_ids"} for k, v in report["arms"].items()}, indent=1))
    print("SMOKE PASS" if all(report["checks"].values()) else "SMOKE FAIL")


# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["selftest", "smoke", "train", "export"], required=True)
    ap.add_argument("--arm", choices=sorted(ARMS))
    ap.add_argument("--at", type=int, choices=[50, 100], default=100)
    ap.add_argument("--pipeline-dir", default=str(PIPELINE_DIR))
    a = ap.parse_args()
    if a.mode == "selftest":
        return selftest(a)
    if a.mode == "smoke":
        return smoke(a)
    import torch
    if os.environ.get("R0_MSDA_BACKEND") != "pytorch":
        sys.exit("set R0_MSDA_BACKEND=pytorch first")
    if not a.arm:
        sys.exit("--arm is required")
    P, M2, M1, hr_p2 = r0_imports(a.pipeline_dir)
    run_dir = OUT / "runs" / a.arm
    step = int(round(a.at / 100 * RECIPE["steps"]))
    ck = run_dir / f"g4_{a.arm}_step{step:05d}.pth"
    if a.mode == "train":
        if (run_dir / f"g4_{a.arm}_step{RECIPE['steps']:05d}.pth").exists():
            sys.exit(f"[REFUSED] {run_dir} already holds a final checkpoint")
        train(a.arm, P, M2, "cuda", run_dir)
        return
    if a.mode == "export":
        if not ck.exists():
            sys.exit(f"missing {ck}")
        cp = OUT / "tile_cache" / f"r0_tile_cache_g4_{a.arm}_at{a.at}_stride10_offset0.json.gz"
        if cp.exists():
            sys.exit(f"[REFUSED] {cp} exists")
        _c, sec = export_cache(P, M1, hr_p2, ck, cp, "cuda")
        print(f"  wrote {cp}  ({sec / 60:.1f} min)\n  sha256 {sha256_file(cp)}")


if __name__ == "__main__":
    main()
