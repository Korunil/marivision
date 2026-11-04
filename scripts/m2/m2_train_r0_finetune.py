r"""m2_train_r0_finetune.py - M2: does TRAINING on resolution-preserving tiles fix what tiled inference broke?
MARIVISION-HR-P2, Claude lead, 2026-10-27. Trains on TRAIN only; never reads VAL or TEST during training.

Two arms, identical except for the data:
  Arm F (control)  whole frames exactly as R0 was trained (SeaClipsDetection.__getitem__, untouched code path)
  Arm T (tiled)    each sample is the whole frame with prob 1 - p_tile, else ONE random Stage C window of that frame
                   (the same 4x4 grid of 640x360 windows used at inference, from marivision_hr_p2_pipeline), resized
                   to 640x640 exactly as R0 resizes whole frames. GT boxes are clipped to the window; a box is kept
                   only if >= min_visible (0.5) of its area is inside, otherwise it is dropped (background).

PRE-DECLARED RECIPE (both arms; written 2026-10-27 before any M2 run):
  init          R0 epoch15 checkpoint (end of the authors' schedule, not selected on VAL), strict load
  optimizer     fresh AdamW, R0's own param groups, at R0's post-milestone LRs (x0.1): base 2e-5, backbone 2e-6,
                linear_proj 2e-6; weight_decay 1e-4; grad clip 0.1; CONSTANT LR
  steps         10,155 optimizer steps at batch 2 (= one R0 epoch), shuffle, flip 0.5, seed 42 (same frame order in both arms)
  p_tile 0.5, min_visible 0.5 (arm T only)
  checkpoints   at 25/50/75/100% of steps; the 100% checkpoint is the only one used for the primary M2 endpoint.

Modes (run from D:\MARIVISION\R0_frozen_baseline):
  --mode selftest   CPU, no model, no SeaClips data: checks the tile target transform and that arm F == R0's dataset
  --mode smoke      GPU, a few steps: timing, peak memory, finite loss, checkpoint save + strict reload + forward
  --mode train      the real run (R0_MSDA_BACKEND=pytorch required)
"""
from PIL import Image  # noqa: F401  (R0 convention: PIL before torchvision)
import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

SCRIPT_VERSION = "M2.1"
TILE_STAGE = "STAGE_C_2X_CROP"


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_c_windows(pipeline_dir, w, h):
    sys.path.insert(0, str(Path(pipeline_dir).resolve()))
    import marivision_hr_p2_pipeline as hr_p2
    ws = [(c.x0_int, c.y0_int, c.x1_int, c.y1_int) for c in
          hr_p2.generate_coarse_to_fine_crops(w, h, horizon_y_norm=0.42, include_stage_d_microcrops=False) if c.stage == TILE_STAGE]
    if len(ws) != 16:
        raise AssertionError(f"expected 16 Stage C windows, got {len(ws)}")
    return ws


def clip_boxes_to_window(anns, win, cat_id_to_label, min_visible):
    """COCO xywh native boxes -> (normalized cxcywh in the window, labels, n_dropped). Mirrors R0's w<=0/h<=0 skip."""
    x0, y0, x1, y1 = win
    tw, th = float(x1 - x0), float(y1 - y0)
    boxes, labels, dropped = [], [], 0
    for a in anns:
        x, y, w, h = [float(v) for v in a["bbox"]]
        if w <= 0 or h <= 0:
            continue
        cx0, cy0 = max(x, x0), max(y, y0)
        cx1, cy1 = min(x + w, x1), min(y + h, y1)
        if cx1 <= cx0 or cy1 <= cy0:
            continue  # entirely outside: not part of this sample at all
        if (cx1 - cx0) * (cy1 - cy0) < min_visible * w * h:
            dropped += 1
            continue
        boxes.append([((cx0 + cx1) / 2.0 - x0) / tw, ((cy0 + cy1) / 2.0 - y0) / th, (cx1 - cx0) / tw, (cy1 - cy0) / th])
        labels.append(cat_id_to_label[a["category_id"]])
    return boxes, labels, dropped


# The dataset class must live at MODULE level: Windows DataLoader workers are started with "spawn" and must be able to
# unpickle it by name (M2.0 defined it inside a function and failed on the workstation).
sys.path.insert(0, str(Path.cwd()))
import torch  # noqa: E402
import torchvision.transforms.functional as TF  # noqa: E402
try:
    from datasets.seaclips_detection import SeaClipsDetection as _R0Dataset, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
except ImportError:  # running outside the R0 folder
    _R0Dataset = None


if _R0Dataset is not None:
    class SeaClipsM2Train(_R0Dataset):
        """Arm F: p_tile=0 -> parent __getitem__ untouched. Arm T: p_tile>0 -> sometimes a Stage C window."""

        def __init__(self, img_source_path, ann_file_path, p_tile, min_visible, pipeline_dir, random_flip=True):
            super().__init__(img_source_path, ann_file_path, is_train=True, random_flip=random_flip)
            self.p_tile, self.min_visible, self.pipeline_dir = float(p_tile), float(min_visible), str(pipeline_dir)
            self._win_cache = {}

        def windows(self, w, h):
            if (w, h) not in self._win_cache:
                self._win_cache[(w, h)] = stage_c_windows(self.pipeline_dir, w, h)
            return self._win_cache[(w, h)]

        def tile_item(self, idx, win_index=None):
            img_id = self.img_ids[idx]
            info = self.images[img_id]
            img = self._load_image(info)
            W, H = img.size
            wins = self.windows(W, H)
            k = int(torch.randint(len(wins), (1,)).item()) if win_index is None else int(win_index)
            win = wins[k]
            boxes, labels, dropped = clip_boxes_to_window(self.annotations.get(img_id, []), win, self.cat_id_to_label,
                                                          self.min_visible)
            boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
            tile = TF.resize(img.crop(win), [640, 640])
            if self.random_flip and torch.rand(1).item() > 0.5:
                tile = TF.hflip(tile)
                if len(boxes) > 0:
                    boxes[:, 0] = 1.0 - boxes[:, 0]
            t = TF.normalize(TF.to_tensor(tile), mean=IMAGENET_MEAN, std=IMAGENET_STD)
            target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([img_id]),
                      "orig_size": torch.tensor([win[3] - win[1], win[2] - win[0]]), "size": torch.tensor([640, 640]),
                      "m2_is_tile": torch.tensor([1]), "m2_dropped": torch.tensor([dropped])}
            return t, target

        def __getitem__(self, idx):
            if self.p_tile > 0.0 and torch.rand(1).item() < self.p_tile:
                return self.tile_item(idx)
            img, target = super().__getitem__(idx)
            target["m2_is_tile"] = torch.tensor([0])
            target["m2_dropped"] = torch.tensor([0])
            return img, target


def get_dataset_class():
    if _R0Dataset is None:
        raise SystemExit("[REFUSED] run from D:\\MARIVISION\\R0_frozen_baseline (datasets/seaclips_detection.py not found).")
    return SeaClipsM2Train


# ------------------------------------------------------------------------------------------------ selftest
def run_selftest(a) -> None:
    import tempfile
    import torch
    import torchvision.transforms.functional as TF
    sys.path.insert(0, str(Path.cwd()))
    from datasets.seaclips_detection import SeaClipsDetection, IMAGENET_MEAN, IMAGENET_STD
    n_ok = 0

    def ok(m):
        nonlocal n_ok
        n_ok += 1
        print(f"  [PASS] {m}")

    wins = stage_c_windows(a.pipeline_dir, 1920, 1080)
    assert wins[0] == (0, 0, 640, 360) and len(set(wins)) == 16
    ok(f"16 Stage C training windows from the pipeline (first {wins[0]}, last {wins[-1]})")

    l2 = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}
    anns = [{"bbox": [100, 100, 40, 20], "category_id": 3},    # fully inside window 0
            {"bbox": [600, 50, 100, 10], "category_id": 6},    # 40% inside -> dropped
            {"bbox": [560, 200, 100, 10], "category_id": 7},   # 80% inside -> kept, clipped
            {"bbox": [1000, 900, 30, 30], "category_id": 1},   # outside -> ignored
            {"bbox": [5, 5, 0, 7], "category_id": 2}]          # zero width -> skipped (as R0)
    b, l, d = clip_boxes_to_window(anns, wins[0], l2, 0.5)
    assert l == [2, 6] and d == 1, (l, d)
    assert all(abs(p - q) < 1e-9 for p, q in zip(b[0], [120 / 640, 110 / 360, 40 / 640, 20 / 360])), b[0]
    assert all(abs(p - q) < 1e-9 for p, q in zip(b[1], [600 / 640, 205 / 360, 80 / 640, 10 / 360])), b[1]
    ok("box clipping: inside kept exact, 80%-visible kept & clipped, 40%-visible dropped, outside ignored, w=0 skipped")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        img = Image.new("RGB", (1920, 1080), (10, 40, 80))
        for x in range(100, 140):
            for y in range(100, 120):
                img.putpixel((x, y), (250, 10, 10))
        img.save(td / "f.png")
        coco = {"images": [{"id": 1, "file_name": "f.png", "width": 1920, "height": 1080}],
                "annotations": [dict(id=i + 1, image_id=1, iscrowd=0, area=1.0, **x) for i, x in enumerate(anns)],
                "categories": [{"id": i, "name": str(i)} for i in range(1, 8)]}
        jp = td / "seaclips-train.json"
        jp.write_text(json.dumps(coco), encoding="utf-8")
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            DS = get_dataset_class()
            base = SeaClipsDetection(td, jp, is_train=True, random_flip=True)
            armF = DS(td, jp, p_tile=0.0, min_visible=0.5, pipeline_dir=a.pipeline_dir, random_flip=True)
            armT = DS(td, jp, p_tile=1.0, min_visible=0.5, pipeline_dir=a.pipeline_dir, random_flip=False)
        for seed in (1, 2, 3, 4):
            torch.manual_seed(seed)
            x0, t0 = base[0]
            torch.manual_seed(seed)
            x1, t1 = armF[0]
            assert torch.equal(x0, x1) and torch.equal(t0["boxes"], t1["boxes"]) and torch.equal(t0["labels"], t1["labels"])
        ok("arm F sample is bit-identical to R0's SeaClipsDetection sample (4 seeds, incl. random flips)")

        x, t = armT.tile_item(0, win_index=0)
        ref = TF.normalize(TF.to_tensor(TF.resize(img.crop(wins[0]), [640, 640])), mean=IMAGENET_MEAN, std=IMAGENET_STD)
        assert torch.equal(x, ref) and t["labels"].tolist() == [2, 6] and int(t["m2_dropped"]) == 1
        assert t["orig_size"].tolist() == [360, 640]
        # the red marker (native 100..140 x 100..120) must sit where the target box says, in 640x640 tile space
        cx, cy, bw, bh = t["boxes"][0].tolist()
        red = (x[0] > 1.5) & (x[1] < -1.0)
        ys, xs = torch.nonzero(red, as_tuple=True)
        assert abs(xs.float().mean().item() / 640 - cx) < 0.01 and abs(ys.float().mean().item() / 640 - cy) < 0.01
        ok("tile sample: image = R0 transform of the crop; target box lands on the object's pixels in the tile")

        armT.random_flip = True
        flips = 0
        for s in range(20):
            torch.manual_seed(s)
            x, t = armT.tile_item(0, win_index=0)
            if t["boxes"][0, 0].item() > 0.5:
                flips += 1
                red = (x[0] > 1.5) & (x[1] < -1.0)
                xs = torch.nonzero(red, as_tuple=True)[1]
                assert abs(xs.float().mean().item() / 640 - t["boxes"][0, 0].item()) < 0.01
        assert 0 < flips < 20
        ok(f"flip keeps image and box consistent on tiles ({flips}/20 flipped)")
    print(f"[M2 SELFTEST] {n_ok}/5 passed (script {SCRIPT_VERSION}, sha {sha256_file(Path(__file__).resolve())})")


# ------------------------------------------------------------------------------------------------ training
def build_everything(a, torch):
    sys.path.insert(0, str(Path.cwd()))
    import torchvision.transforms.functional as TF
    from torch.utils.data import DataLoader
    from models.deformable_detr import build
    from util.misc import collate_fn
    from datasets.seaclips_detection import SeaClipsDetection, IMAGENET_MEAN, IMAGENET_STD
    from r0_preflight import get_authoritative_r0_args, match_name_keywords

    args = get_authoritative_r0_args()
    device = torch.device(args.device)
    model, criterion, post = build(args)
    init = Path(a.init_checkpoint).resolve()
    try:
        ck = torch.load(init, map_location="cpu", weights_only=False)
    except TypeError:
        ck = torch.load(init, map_location="cpu")
    model.load_state_dict(ck["model"], strict=True)
    model.to(device)
    lr_scale = 0.1  # R0's post-milestone LRs (pre-declared)
    groups = [
        {"params": [p for n, p in model.named_parameters() if not match_name_keywords(n, args.lr_backbone_names)
                    and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad], "lr": args.lr * lr_scale},
        {"params": [p for n, p in model.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
         "lr": args.lr_backbone * lr_scale},
        {"params": [p for n, p in model.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
         "lr": args.lr * args.lr_linear_proj_mult * lr_scale},
    ]
    opt = torch.optim.AdamW(groups, lr=args.lr * lr_scale, weight_decay=args.weight_decay)
    DS = get_dataset_class()
    p_tile = a.p_tile if a.arm == "T" else 0.0
    ds = DS(Path(a.zip), Path(a.coco_train), p_tile=p_tile, min_visible=a.min_visible, pipeline_dir=a.pipeline_dir)
    g = torch.Generator()
    g.manual_seed(a.seed)
    loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate_fn, num_workers=a.num_workers,
                        pin_memory=(device.type == "cuda"), generator=g, persistent_workers=a.num_workers > 0,
                        multiprocessing_context=("spawn" if a.num_workers > 0 else None))  # same as Windows everywhere
    return args, device, model, criterion, post, opt, loader, ck, init, p_tile, build


def train_loop(a, n_steps, save_at, out_dir, tag):
    import torch
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
        torch.cuda.reset_peak_memory_stats()
    args, device, model, criterion, post, opt, loader, ck, init, p_tile, build = build_everything(a, torch)
    print(f"[M2 {SCRIPT_VERSION}] arm {a.arm} | init {init.name} (epoch {ck.get('epoch')}) | steps {n_steps} | p_tile {p_tile} | "
          f"min_visible {a.min_visible} | LRs {[g['lr'] for g in opt.param_groups]} | frames {len(loader.dataset)}")
    model.train()
    criterion.train()
    step, t0, run_loss = 0, time.time(), 0.0
    n_tile, n_samples, n_dropped, n_boxes = 0, 0, 0, 0
    log = []
    while step < n_steps:
        for samples, targets in loader:
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            for t in targets:
                n_samples += 1
                n_tile += int(t["m2_is_tile"].item())
                n_dropped += int(t["m2_dropped"].item())
                n_boxes += int(t["labels"].numel())
            out = model(samples)
            ld = criterion(out, targets)
            loss = sum(ld[k] * criterion.weight_dict[k] for k in ld if k in criterion.weight_dict)
            if not math.isfinite(loss.item()):
                raise SystemExit(f"[M2] non-finite loss at step {step + 1}; stopping.")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
            opt.step()
            step += 1
            run_loss += loss.item()
            if step % a.log_every == 0 or step == n_steps:
                el = time.time() - t0
                mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
                rec = {"step": step, "avg_loss": run_loss / step, "loss": loss.item(), "s_per_step": el / step,
                       "eta_min": el / step * (n_steps - step) / 60, "peak_mem_gib": mem,
                       "tile_fraction": n_tile / max(1, n_samples), "boxes_per_sample": n_boxes / max(1, n_samples),
                       "dropped_partial_boxes": n_dropped}
                log.append(rec)
                print(f"  step {step}/{n_steps} | loss {rec['loss']:.4f} avg {rec['avg_loss']:.4f} | {rec['s_per_step']:.3f}s/step "
                      f"(~{rec['eta_min']:.1f} min left) | peak {mem:.2f} GiB | tiles {rec['tile_fraction']:.2f} | "
                      f"boxes/sample {rec['boxes_per_sample']:.2f} | dropped {n_dropped}")
            if step in save_at:
                p = out_dir / f"m2_arm{a.arm}_{tag}_step{step:05d}.pth"
                torch.save({"epoch": f"M2_arm{a.arm}_step{step}", "model": model.state_dict(),
                            "m2": {"script_version": SCRIPT_VERSION, "script_sha256": sha256_file(Path(__file__).resolve()),
                                   "arm": a.arm, "step": step, "n_steps": n_steps, "p_tile": p_tile,
                                   "min_visible": a.min_visible, "seed": a.seed, "init_checkpoint": str(init),
                                   "init_sha256": sha256_file(init), "lrs": [g["lr"] for g in opt.param_groups],
                                   "log": log}}, p)
                print(f"  [saved] {p}")
            if step >= n_steps:
                break
    return model, build, args, device, log


def run_smoke(a) -> None:
    import torch
    if os.environ.get("R0_MSDA_BACKEND", "").lower() != "pytorch":
        raise SystemExit("[REFUSED] Run 'set R0_MSDA_BACKEND=pytorch' first.")
    out_dir = Path(a.out_dir).resolve() / f"smoke_arm{a.arm}"
    if out_dir.exists():
        raise SystemExit(f"[REFUSED] {out_dir} exists; delete it or choose another --out-dir.")
    out_dir.mkdir(parents=True)
    n = a.smoke_steps
    model, build, args, device, log = train_loop(a, n, {n}, out_dir, "smoke")
    p = out_dir / f"m2_arm{a.arm}_smoke_step{n:05d}.pth"
    m2, _c, post = build(args)
    ck = torch.load(p, map_location="cpu", weights_only=False)
    m2.load_state_dict(ck["model"], strict=True)
    m2.to(device).eval()
    sd1, sd2 = model.state_dict(), m2.state_dict()
    assert all(torch.equal(sd1[k].cpu(), sd2[k].cpu()) for k in sd1)
    from util.misc import NestedTensor
    with torch.inference_mode():
        x = torch.zeros(1, 3, 640, 640, device=device)
        r = post["bbox"](m2(NestedTensor(x, torch.zeros(1, 640, 640, dtype=torch.bool, device=device))),
                         torch.tensor([[1080.0, 1920.0]], device=device))
    assert r[0]["boxes"].shape[0] == 100
    s = log[-1]["s_per_step"]
    print(f"[M2 SMOKE] arm {a.arm}: PASS | {s:.3f} s/step -> full run of {a.steps} steps ~{s * a.steps / 60:.0f} min "
          f"(+ checkpoint saves) | peak GPU {log[-1]['peak_mem_gib']:.2f} GiB | checkpoint save + strict reload + forward OK")


def run_train(a) -> None:
    if os.environ.get("R0_MSDA_BACKEND", "").lower() != "pytorch":
        raise SystemExit("[REFUSED] Run 'set R0_MSDA_BACKEND=pytorch' first.")
    if "train" not in Path(a.coco_train).name.lower():
        raise SystemExit("[SPLIT BLOCK] --coco-train must be the TRAIN json.")
    out_dir = Path(a.out_dir).resolve()
    final = out_dir / f"m2_arm{a.arm}_seed{a.seed}_step{a.steps:05d}.pth"
    if final.exists():
        raise SystemExit(f"[REFUSED] {final} exists.")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_at = {int(round(a.steps * f)) for f in (0.25, 0.5, 0.75, 1.0)}
    t0 = time.time()
    _m, _b, _a, _d, log = train_loop(a, a.steps, save_at, out_dir, f"seed{a.seed}")
    (out_dir / f"m2_arm{a.arm}_seed{a.seed}_trainlog.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(f"[M2] arm {a.arm} done in {(time.time() - t0) / 60:.1f} min; final checkpoint {final}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("selftest", "smoke", "train"), required=True)
    ap.add_argument("--arm", choices=("F", "T"))
    ap.add_argument("--init-checkpoint", default="D:/MARIVISION/checkpoints_r0/checkpoint_epoch_15.pth")
    ap.add_argument("--out-dir", default="D:/MARIVISION/R0_frozen_baseline/m2_runs")
    ap.add_argument("--pipeline-dir", default="D:/MARIVISION/H0_workbench/eval_tools")
    ap.add_argument("--zip", default="D:/MARIVISION/SeaClips.zip")
    ap.add_argument("--coco-train", default="D:/MARIVISION/seaclips-train.json")
    ap.add_argument("--steps", type=int, default=10155)
    ap.add_argument("--smoke-steps", type=int, default=20)
    ap.add_argument("--p-tile", type=float, default=0.5)
    ap.add_argument("--min-visible", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=500)
    a = ap.parse_args()
    if a.mode != "selftest" and not a.arm:
        raise SystemExit("[REFUSED] --arm F or --arm T is required.")
    if a.mode == "smoke":
        a.log_every = 5
    {"selftest": run_selftest, "smoke": run_smoke, "train": run_train}[a.mode](a)


if __name__ == "__main__":
    main()
