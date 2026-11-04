r"""
G3: backbone feasibility smoke for Deformable-DETR on the RTX 3060.

Location: D:/MARIVISION/H0_workbench/g3/g3_backbone_smoke.py
Run with R0_MSDA_BACKEND=pytorch.  Reads 2 TRAIN frames from the zip
(never extracted).  Writes only D:/MARIVISION/G3_runs/smoke/.  No VAL, no
TEST, no training run: this decides WHICH backbone a later, separately
pre-declared training run may use, and what it would cost.

For every candidate backbone it measures, on real frames:
  build + COCO Deformable-DETR transfer (transformer/queries/box head; the
  backbone and input projections are new, the 7-class head is new as in R0),
  parameter counts, peak GPU memory and seconds per training iteration at
  640x640 batch 2 (fp32 and AMP) and 1024x1024 batch 2 (AMP; the native-
  resolution proxy), finite loss and gradients, and a 40-step overfit on the
  2 frames (loss must fall by >= 30 %).  It then projects one 15-epoch run.

Nothing in the Deformable-DETR repository is modified: the timm backbone is
swapped in at run time (model.backbone[0] and model.input_proj).

Modes:  --mode selftest   CPU (sandbox), random weights, tiny inputs
        --mode smoke      GPU, real frames, pretrained weights (downloads)
"""
from PIL import Image  # noqa: F401  -- PIL before torch (Director's Windows env)
import sys
# timm optionally does `try: import wandb except ImportError`. On the Director's env the installed wandb
# emits pydantic warnings at import (fatal under -W error) and leaves a temp dir behind. We never use
# wandb, so block it: a None entry in sys.modules makes `import wandb` raise ImportError, which timm handles.
if "wandb" not in sys.modules:
    sys.modules["wandb"] = None
import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from pathlib import Path

VERSION = "G3.0.1"

DETR_PATH = Path(os.environ.get("G3_DETR_PATH", "D:/MARIVISION/Deformable-DETR"))
ZIP_PATH = Path("D:/MARIVISION/SeaClips.zip")
TRAIN_JSON = Path("D:/MARIVISION/seaclips-train.json")
COCO_CKPT = Path("D:/MARIVISION/r50_deformable_detr-checkpoint.pth")
OUT_DIR = Path("D:/MARIVISION/G3_runs/smoke")

CANDIDATES = [
    # name, timm id (None = R0's torchvision ResNet-50 path), extra timm kwargs
    ("R50 (R0 control)", None, {}),
    ("ConvNeXt-T in22k", "convnext_tiny.fb_in22k", {}),
    ("ConvNeXt-S in22k", "convnext_small.fb_in22k", {}),
    ("ConvNeXt-B in22k", "convnext_base.fb_in22k", {}),
    ("Swin-T in22k", "swin_tiny_patch4_window7_224.ms_in22k", {"img_size": 640, "strict_img_size": False}),
    ("Swin-B in22k", "swin_base_patch4_window7_224.ms_in22k", {"img_size": 640, "strict_img_size": False}),
]
SETTINGS = [("fp32", 640, 2), ("amp", 640, 2), ("amp", 1024, 2)]
TIMING = {"warmup": 2, "iters": 8, "overfit_steps": 40, "overfit_min_drop": 0.30}
ITERS_PER_EPOCH = 20310 // 2     # R0: batch 2, every TRAIN frame
EPOCHS = 15
MEM_LIMIT_GB = 11.0

SELECTION_RULE = (
    "Feasible = finite loss and gradients, overfit loss drop >= 30%, and peak memory <= 11.0 GB at "
    "640x640 batch 2 (fp32 or AMP). Among feasible candidates, the recommendation is the highest-capacity "
    "backbone whose projected 15-epoch AMP time at 640 is <= 4x the R50 fp32 time; the Director sets the "
    "GPU budget, so the full table is reported either way.")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


KNOWN_REPO_WARNINGS = [
    # torchvision >= 0.13 deprecation inside the original Deformable-DETR backbone.py (pretrained=...)
    r"The parameter 'pretrained' is deprecated",
    r"Arguments other than a weight enum or `None` for 'weights' are deprecated",
    # torch.meshgrid without indexing= inside the unmodified Deformable-DETR transformer
    r"torch.meshgrid: in an upcoming release",
]


def silence_known_repo_warnings():
    import warnings
    for msg in KNOWN_REPO_WARNINGS:
        warnings.filterwarnings("ignore", message=msg, category=UserWarning)


def build_model(args):
    """models.deformable_detr.build with ONLY the known, harmless torchvision
    deprecation warnings of the unmodified repository silenced (so -W error
    still catches anything new)."""
    import warnings
    from models.deformable_detr import build
    with warnings.catch_warnings():
        for msg in KNOWN_REPO_WARNINGS:
            warnings.filterwarnings("ignore", message=msg, category=UserWarning)
        return build(args)


def r0_args(device):
    """Same values as r0_preflight.get_authoritative_r0_args (copied, so this
    script does not depend on where r0_preflight.py lives)."""
    from main import get_args_parser
    args = get_args_parser().parse_args([])
    args.lr, args.lr_backbone, args.lr_linear_proj_mult = 2e-4, 2e-5, 0.1
    args.weight_decay, args.epochs, args.batch_size, args.clip_max_norm = 1e-4, 15, 2, 0.1
    args.num_queries, args.num_feature_levels, args.aux_loss = 300, 4, True
    args.two_stage, args.with_box_refine, args.dilation = False, False, False
    args.dataset_file, args.num_classes_override, args.device = "seaclips", 7, device
    if hasattr(args, "conditioning_mode"):
        args.conditioning_mode = None
    return args


# =============================================================================
# timm backbone in Deformable-DETR's backbone interface
# =============================================================================
def make_timm_backbone(timm_id, kw, pretrained):
    import torch
    import torch.nn.functional as F
    from torch import nn
    import timm
    from util.misc import NestedTensor

    class TimmBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = timm.create_model(timm_id, pretrained=pretrained, features_only=True,
                                          out_indices=(1, 2, 3), **kw)
            self.num_channels = list(self.body.feature_info.channels())
            self.strides = list(self.body.feature_info.reduction())
            assert self.strides == [8, 16, 32], self.strides
            frozen = 0
            for n, p in self.body.named_parameters():      # mirror R0: stem + first stage frozen
                if n.startswith(("stem", "stages_0", "stages.0", "patch_embed", "layers_0", "layers.0")):
                    p.requires_grad_(False)
                    frozen += p.numel()
            self.frozen_params = frozen

        def forward(self, tensor_list):
            xs = self.body(tensor_list.tensors)
            out = {}
            for i, x in enumerate(xs):
                if x.shape[1] != self.num_channels[i] and x.shape[-1] == self.num_channels[i]:
                    x = x.permute(0, 3, 1, 2).contiguous()          # Swin: NHWC -> NCHW
                m = tensor_list.mask
                mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
                out[str(i)] = NestedTensor(x, mask)
            return out

    return TimmBackbone()


def swap_backbone(model, timm_id, kw, pretrained):
    """Replace backbone[0] and rebuild input_proj exactly as DeformableDETR.__init__ does."""
    from torch import nn
    from models.backbone import Joiner
    tb = make_timm_backbone(timm_id, kw, pretrained)
    model.backbone = Joiner(tb, model.backbone[1])
    hidden = model.transformer.d_model
    projs, in_ch = [], None
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
    return tb


def load_coco_transfer(model, ckpt_path, new_backbone):
    """COCO Deformable-DETR weights for everything that keeps its shape.
    Skipped: class head (7 classes, as in R0) and, for a new backbone, the
    backbone and input projections."""
    import torch
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    own = model.state_dict()
    skip_prefix = ("class_embed",) + (("backbone.", "input_proj.") if new_backbone else ())
    take = {k: v for k, v in sd.items() if not k.startswith(skip_prefix) and k in own and own[k].shape == v.shape}
    missing, unexpected = model.load_state_dict(take, strict=False)
    shape_mismatch = [k for k, v in sd.items() if k in own and own[k].shape != v.shape and not k.startswith(skip_prefix)]
    return {"loaded": len(take), "missing": len(missing), "shape_mismatch_not_skipped": shape_mismatch}


# =============================================================================
# One candidate
# =============================================================================
def param_groups(model, args):
    from util.misc import NestedTensor  # noqa: F401
    def has(n, keys):
        return any(k in n for k in keys)
    core = [p for n, p in model.named_parameters() if not has(n, args.lr_backbone_names)
            and not has(n, args.lr_linear_proj_names) and p.requires_grad]
    bb = [p for n, p in model.named_parameters() if has(n, args.lr_backbone_names) and p.requires_grad]
    lin = [p for n, p in model.named_parameters() if has(n, args.lr_linear_proj_names) and p.requires_grad]
    n_train = sum(1 for p in model.parameters() if p.requires_grad)
    assert len(core) + len(bb) + len(lin) == n_train, "optimizer partition incomplete"
    return [{"params": core, "lr": args.lr}, {"params": bb, "lr": args.lr_backbone},
            {"params": lin, "lr": args.lr * args.lr_linear_proj_mult}]


def step(model, criterion, x, targets, opt, scaler, amp, device, clip):
    import torch
    from util.misc import NestedTensor
    mask = torch.zeros(x.shape[0], x.shape[2], x.shape[3], dtype=torch.bool, device=device)
    with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                        dtype=torch.float16 if device == "cuda" else torch.bfloat16, enabled=amp):
        out = model(NestedTensor(x, mask))
    out = {k: (v.float() if hasattr(v, "float") else v) for k, v in out.items() if k != "aux_outputs"} | \
          {"aux_outputs": [{kk: vv.float() for kk, vv in a.items()} for a in out.get("aux_outputs", [])]}
    ld = criterion(out, targets)
    loss = sum(ld[k] * criterion.weight_dict[k] for k in ld if k in criterion.weight_dict)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    scaler.step(opt)
    scaler.update()
    return float(loss.detach()), float(gn)


def run_candidate(name, timm_id, kw, batch_src, device, pretrained, coco_ckpt, settings, timing, log=print):
    import torch
    import torch.nn.functional as F
    res = {"name": name, "timm_id": timm_id, "settings": {}}
    try:
        args = r0_args(device)
        model, criterion, _ = build_model(args)
        if timm_id is not None:
            tb = swap_backbone(model, timm_id, kw, pretrained)
            res["backbone_frozen_params_M"] = tb.frozen_params / 1e6
        res["coco_transfer"] = load_coco_transfer(model, coco_ckpt, timm_id is not None) if coco_ckpt else None
        model.to(device)
        criterion.to(device)
        res["params_total_M"] = sum(p.numel() for p in model.parameters()) / 1e6
        res["params_backbone_M"] = sum(p.numel() for p in model.backbone.parameters()) / 1e6
        res["params_trainable_M"] = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        imgs, tgts = batch_src
        for prec, size, bs in settings:
            key = f"{prec}_{size}_b{bs}"
            r = {}
            try:
                if device == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                x = torch.stack([imgs[i % len(imgs)] for i in range(bs)]).to(device)
                if x.shape[-1] != size:
                    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
                t = [{k: v.to(device) for k, v in tgts[i % len(tgts)].items()} for i in range(bs)]
                # fresh optimizer per setting so timing includes the optimizer step
                opt = torch.optim.AdamW(param_groups(model, args), lr=args.lr, weight_decay=args.weight_decay)
                scaler = torch.amp.GradScaler("cuda", enabled=(prec == "amp" and device == "cuda"))
                model.train()
                times, losses = [], []
                for it in range(timing["warmup"] + timing["iters"]):
                    if device == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    l, gn = step(model, criterion, x, t, opt, scaler, prec == "amp", device, args.clip_max_norm)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    if it >= timing["warmup"]:
                        times.append(time.perf_counter() - t0)
                    losses.append(l)
                grads_ok = all(torch.isfinite(p.grad).all().item() for p in model.parameters()
                               if p.requires_grad and p.grad is not None)
                r = {"ok": True, "sec_per_iter": sorted(times)[len(times) // 2],
                     "peak_gb": (torch.cuda.max_memory_allocated() / 2 ** 30) if device == "cuda" else None,
                     "loss_finite": all(math.isfinite(v) for v in losses), "grads_finite": bool(grads_ok),
                     "loss_first": losses[0], "loss_last": losses[-1]}
            except torch.cuda.OutOfMemoryError:
                r = {"ok": False, "oom": True}
            except Exception as e:  # noqa: BLE001
                r = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
            res["settings"][key] = r
            log(f"    {name:18s} {key:12s} {json.dumps(r)}")
        # overfit on the 2 frames at 640, AMP if available (fp32 on CPU)
        try:
            if device == "cuda":
                torch.cuda.empty_cache()
            args2 = r0_args(device)
            model2, crit2, _ = build_model(args2)
            if timm_id is not None:
                swap_backbone(model2, timm_id, kw, pretrained)
            if coco_ckpt:
                load_coco_transfer(model2, coco_ckpt, timm_id is not None)
            model2.to(device)
            crit2.to(device)
            x = torch.stack(imgs[:2]).to(device)
            size = settings[0][1]
            if x.shape[-1] != size:
                x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
            t = [{k: v.to(device) for k, v in tg.items()} for tg in tgts[:2]]
            opt = torch.optim.AdamW(param_groups(model2, args2), lr=args2.lr, weight_decay=args2.weight_decay)
            amp = device == "cuda"
            scaler = torch.amp.GradScaler("cuda", enabled=amp)
            model2.train()
            ls = [step(model2, crit2, x, t, opt, scaler, amp, device, args2.clip_max_norm)[0]
                  for _ in range(timing["overfit_steps"])]
            first, last = sum(ls[:3]) / 3, sum(ls[-3:]) / 3
            res["overfit"] = {"loss_first3": first, "loss_last3": last, "drop": 1 - last / first,
                              "pass": (1 - last / first) >= timing["overfit_min_drop"]}
            del model2, crit2, opt
        except Exception as e:  # noqa: BLE001
            res["overfit"] = {"pass": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
        log(f"    {name:18s} overfit {json.dumps(res['overfit'])}")
        res["built"] = True
        del model, criterion
    except Exception as e:  # noqa: BLE001
        res["built"] = False
        res["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        res["trace"] = traceback.format_exc()[-1500:]
        log(f"    {name:18s} BUILD FAILED {res['error']}")
    if device == "cuda":
        torch.cuda.empty_cache()
    return res


def summarise(results, mem_limit=MEM_LIMIT_GB):
    base = next((r for r in results if r["timm_id"] is None), None)
    base_t = base["settings"].get("fp32_640_b2", {}).get("sec_per_iter") if base and base.get("built") else None
    rows = []
    for r in results:
        s = r.get("settings", {})
        f, a, h = s.get("fp32_640_b2", {}), s.get("amp_640_b2", {}), s.get("amp_1024_b2", {})
        fits = [k for k, v in (("fp32", f), ("amp", a)) if v.get("ok") and v.get("peak_gb") is not None
                and v["peak_gb"] <= mem_limit and v.get("loss_finite") and v.get("grads_finite")]
        feasible = bool(r.get("built") and fits and r.get("overfit", {}).get("pass"))
        t_amp = a.get("sec_per_iter") if a.get("ok") else None
        proj_h = (t_amp * ITERS_PER_EPOCH * EPOCHS / 3600) if t_amp else None
        rows.append({"name": r["name"], "built": r.get("built"), "feasible": feasible, "fits_640": fits,
                     "params_total_M": r.get("params_total_M"), "params_backbone_M": r.get("params_backbone_M"),
                     "fp32_640": f, "amp_640": a, "amp_1024": h, "overfit": r.get("overfit"),
                     "proj_15ep_amp640_h": proj_h,
                     "proj_15ep_amp1024_h": (h["sec_per_iter"] * ITERS_PER_EPOCH * EPOCHS / 3600) if h.get("ok") else None,
                     "x_r50_fp32": (t_amp / base_t) if (t_amp and base_t) else None})
    base_h = base_t * ITERS_PER_EPOCH * EPOCHS / 3600 if base_t else None
    ok = [x for x in rows if x["feasible"] and x["x_r50_fp32"] is not None and x["x_r50_fp32"] <= 4.0
          and x["name"] != "R50 (R0 control)"]
    rec = max(ok, key=lambda x: x["params_backbone_M"] or 0)["name"] if ok else None
    return {"rows": rows, "r50_fp32_15ep_h": base_h, "recommended": rec, "rule": SELECTION_RULE}


def table_md(summary, env):
    def g(d, k, fmt):
        v = d.get(k) if d else None
        return fmt.format(v) if isinstance(v, (int, float)) else ("OOM" if d and d.get("oom") else "–")
    L = [f"# G3 backbone smoke ({VERSION})", "", f"Environment: {json.dumps(env)}", "",
         "| Backbone | params M (bb) | fp32 640 s/it | fp32 640 GB | AMP 640 s/it | AMP 640 GB | AMP 1024 s/it | AMP 1024 GB "
         "| overfit drop | 15 ep AMP 640 h | 15 ep AMP 1024 h | × R50 | feasible |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in summary["rows"]:
        L.append(f"| {r['name']} | {g(r, 'params_total_M', '{:.0f}')} ({g(r, 'params_backbone_M', '{:.0f}')}) | "
                 f"{g(r['fp32_640'], 'sec_per_iter', '{:.3f}')} | {g(r['fp32_640'], 'peak_gb', '{:.1f}')} | "
                 f"{g(r['amp_640'], 'sec_per_iter', '{:.3f}')} | {g(r['amp_640'], 'peak_gb', '{:.1f}')} | "
                 f"{g(r['amp_1024'], 'sec_per_iter', '{:.3f}')} | {g(r['amp_1024'], 'peak_gb', '{:.1f}')} | "
                 f"{g(r['overfit'], 'drop', '{:.2f}')} | {g(r, 'proj_15ep_amp640_h', '{:.1f}')} | "
                 f"{g(r, 'proj_15ep_amp1024_h', '{:.1f}')} | {g(r, 'x_r50_fp32', '{:.2f}')} | {'yes' if r['feasible'] else 'no'} |")
    L += ["", f"R50 fp32 15-epoch projection: {summary['r50_fp32_15ep_h']}", "",
          f"Rule: {summary['rule']}", "", f"**Recommended: {summary['recommended']}**", "",
          "Projections are GPU compute only (data loading excluded); a 15-epoch run is ~10,155 iterations per epoch."]
    return "\n".join(L) + "\n"


# =============================================================================
def environment():
    import torch
    import torchvision
    try:
        import timm
        tv = timm.__version__
    except Exception:  # noqa: BLE001
        tv = None
    return {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
            "timm": tv, "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "msda_backend": os.environ.get("R0_MSDA_BACKEND")}


def prepare_path():
    silence_known_repo_warnings()
    if str(DETR_PATH) not in sys.path:
        sys.path.insert(0, str(DETR_PATH))
    os.chdir(DETR_PATH)


def selftest():
    import torch
    print(f"[selftest] {VERSION}")
    assert os.environ.get("R0_MSDA_BACKEND") == "pytorch", "set R0_MSDA_BACKEND=pytorch"
    prepare_path()
    from util.misc import NestedTensor
    # 1. backbone wrapper: channels, strides, NHWC handling, frozen stem
    for tid, kw in (("convnext_tiny", {}), ("swin_tiny_patch4_window7_224", {"img_size": 128, "strict_img_size": False})):
        tb = make_timm_backbone(tid, kw, pretrained=False)
        x = torch.zeros(1, 3, 128, 128)
        out = tb(NestedTensor(x, torch.zeros(1, 128, 128, dtype=torch.bool)))
        shapes = [tuple(v.tensors.shape) for v in out.values()]
        assert shapes == [(1, c, 128 // s, 128 // s) for c, s in zip(tb.num_channels, tb.strides)], shapes
        assert tb.frozen_params > 0
        print(f"  [ok] {tid}: channels {tb.num_channels}, strides {tb.strides}, NCHW out, frozen {tb.frozen_params} params")
    # 2. full model swap + COCO-like transfer + train steps (tiny inputs, CPU)
    args = r0_args("cpu")
    model, crit, _ = build_model(args)
    ref = {k: v.clone() for k, v in model.state_dict().items()}
    fake_ckpt = Path(os.environ.get("TMP", "/tmp")) / "g3_fake_coco.pth"
    sd = {k: (v + 0.001 * torch.randn_like(v)) if v.is_floating_point() else v for k, v in ref.items()}
    sd["class_embed.0.weight"] = torch.randn(91, 256)
    torch.save({"model": sd}, fake_ckpt)
    swap_backbone(model, "convnext_tiny", {}, pretrained=False)
    tr = load_coco_transfer(model, fake_ckpt, True)
    own = model.state_dict()
    k = "transformer.encoder.layers.0.self_attn.sampling_offsets.weight"
    assert torch.equal(own[k], sd[k]), "transformer weights not transferred"
    assert not any(kk.startswith(("backbone.", "input_proj.", "class_embed")) for kk in tr["shape_mismatch_not_skipped"])
    assert [m[0].in_channels for m in model.input_proj] == [192, 384, 768, 768]
    print(f"  [ok] swap + transfer: {tr['loaded']} tensors loaded; input_proj in-channels "
          f"{[m[0].in_channels for m in model.input_proj]}")
    imgs = [torch.randn(3, 128, 128) for _ in range(2)]
    tgts = [{"labels": torch.tensor([1, 3]), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.1], [0.3, 0.6, 0.05, 0.05]]),
             "image_id": torch.tensor([i]), "orig_size": torch.tensor([128, 128]), "size": torch.tensor([128, 128])}
            for i in range(2)]
    res = run_candidate("ConvNeXt-T (rand)", "convnext_tiny", {}, (imgs, tgts), "cpu", False, fake_ckpt,
                        [("fp32", 128, 2)], {"warmup": 1, "iters": 2, "overfit_steps": 40, "overfit_min_drop": 0.10},
                        log=lambda *a: None)
    s = res["settings"]["fp32_128_b2"]
    assert res["built"] and s["ok"] and s["loss_finite"] and s["grads_finite"], res
    assert res["overfit"]["pass"], res["overfit"]
    print(f"  [ok] train steps on CPU: loss finite, grads finite; overfit drop {res['overfit']['drop']:.2f}")
    res0 = run_candidate("R50 (rand)", None, {}, (imgs, tgts), "cpu", False, None,
                         [("fp32", 128, 2)], {"warmup": 1, "iters": 1, "overfit_steps": 3, "overfit_min_drop": 0.0},
                         log=lambda *a: None)
    assert res0["built"] and res0["settings"]["fp32_128_b2"]["ok"]
    fake_ckpt.unlink(missing_ok=True)
    # 3. summary / selection logic
    fake = [{"name": "R50 (R0 control)", "timm_id": None, "built": True, "params_backbone_M": 23,
             "settings": {"fp32_640_b2": {"ok": True, "sec_per_iter": 0.5, "peak_gb": 5, "loss_finite": True, "grads_finite": True},
                          "amp_640_b2": {"ok": True, "sec_per_iter": 0.4, "peak_gb": 4, "loss_finite": True, "grads_finite": True}},
             "overfit": {"pass": True, "drop": 0.5}},
            {"name": "Big", "timm_id": "x", "built": True, "params_backbone_M": 88,
             "settings": {"fp32_640_b2": {"ok": False, "oom": True},
                          "amp_640_b2": {"ok": True, "sec_per_iter": 1.8, "peak_gb": 10.5, "loss_finite": True, "grads_finite": True}},
             "overfit": {"pass": True, "drop": 0.5}},
            {"name": "Huge", "timm_id": "y", "built": True, "params_backbone_M": 200,
             "settings": {"amp_640_b2": {"ok": True, "sec_per_iter": 2.5, "peak_gb": 11.5, "loss_finite": True, "grads_finite": True}},
             "overfit": {"pass": True, "drop": 0.5}}]
    sm = summarise(fake)
    assert sm["recommended"] == "Big", sm["recommended"]
    assert abs(sm["r50_fp32_15ep_h"] - 0.5 * ITERS_PER_EPOCH * EPOCHS / 3600) < 1e-9
    table_md(sm, {"selftest": True})
    print("  [ok] selection rule (memory limit, ×R50 cap, largest feasible) and table")
    print("SELFTEST PASS")


def smoke():
    import torch
    print(f"[smoke] {VERSION}")
    assert os.environ.get("R0_MSDA_BACKEND") == "pytorch", "set R0_MSDA_BACKEND=pytorch first"
    assert torch.cuda.is_available(), "CUDA not available"
    for p in (ZIP_PATH, TRAIN_JSON, COCO_CKPT, DETR_PATH):
        if not p.exists():
            sys.exit(f"missing {p}")
    prepare_path()
    env = environment()
    print("  env:", env)
    from datasets.seaclips_detection import SeaClipsDetection
    ds = SeaClipsDetection(ZIP_PATH, TRAIN_JSON, is_train=False, random_flip=False)
    # two frames that contain boxes: the first GT frame of video_01 and of video_18 (LV).
    # The dataset's attribute names differ between versions, so pick via its COCO object and
    # verify every pick by the image_id the dataset itself returns.
    coco = ds.coco
    ids = list(getattr(ds, "img_ids", None) or getattr(ds, "ids", None) or sorted(coco.imgs))
    picks, picked_files = [], []
    for want in ("video_01", "video_18"):
        for i, iid in enumerate(ids):
            if coco.imgs[iid]["file_name"].replace("\\", "/").startswith(want) and coco.imgToAnns.get(iid):
                picks.append((i, iid))
                break
    if len(picks) != 2:
        sys.exit(f"could not find GT frames for video_01 / video_18 (found {picks})")
    imgs, tgts = [], []
    for i, iid in picks:
        x, t = ds[i]
        got = int(t["image_id"].item()) if hasattr(t["image_id"], "item") else int(t["image_id"])
        if got != iid:
            sys.exit(f"dataset index {i} returned image_id {got}, expected {iid}: index order differs -- stop")
        imgs.append(x)
        tgts.append({k: v for k, v in t.items() if k in ("labels", "boxes", "image_id", "orig_size", "size")})
        picked_files.append(coco.imgs[iid]["file_name"])
    print(f"  frames: {picked_files}, "
          f"boxes {[int(t['labels'].numel()) for t in tgts]}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for name, tid, kw in CANDIDATES:
        print(f"  [{name}]")
        results.append(run_candidate(name, tid, kw, (imgs, tgts), "cuda", True, COCO_CKPT, SETTINGS, TIMING))
        (OUT_DIR / "g3_smoke_raw.json").write_text(json.dumps({"env": env, "results": results}, indent=1), encoding="utf-8")
    sm = summarise(results)
    out = {"version": VERSION, "script_sha": sha256_file(Path(__file__).resolve()) if Path(__file__).exists() else None,
           "env": env, "summary": sm, "results": results}
    (OUT_DIR / "g3_smoke_report.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = table_md(sm, env)
    (OUT_DIR / "g3_smoke_tables.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"  wrote {OUT_DIR / 'g3_smoke_report.json'}\n  wrote {OUT_DIR / 'g3_smoke_tables.md'}")


if __name__ == "__main__":
    SCRIPT = Path(__file__).resolve()
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["selftest", "smoke"], required=True)
    a = ap.parse_args()
    __file__ = str(SCRIPT)
    selftest() if a.mode == "selftest" else smoke()
