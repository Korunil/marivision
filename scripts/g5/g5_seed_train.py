r"""
G5 seeds: the frozen G4F recipe (ConvNeXt-B + tiles, fp32, 15 epochs) re-run with seeds 43 and 44.
Seed 42 is the existing G4F run (D:/MARIVISION/G4_full/run), so G5 has 3 seeds.

Location: D:/MARIVISION/R0_frozen_baseline/g5_seed_train.py  (run FROM that folder, next to g4_full.py G4F.0.1 and
g4_pilot.py G4P.0, both SHA-checked).  R0_MSDA_BACKEND=pytorch, HF_HUB_DISABLE_SYMLINKS_WARNING=1.

Nothing in the training code changes: this wrapper calls g4_full.run_training with g4_full.RECIPE, in which only
"seed" is replaced, and writes to D:/MARIVISION/G5/seed{S}/run.  The seed sets: the initialisation of the
non-transferred layers, the per-epoch frame order (seed*1000+epoch), tiles, flips and dropout -- exactly as seed 42 did.

Modes:
  selftest                          CPU, synthetic zip
  smoke   --seed S                  GPU: seed-42 recipe identity vs the real G4F run, 40 real steps with seed S,
                                    inline eval, refusal to overwrite, disk space, projections
  train   --seed S [--resume]       the 15-epoch run (resume restarts the unfinished epoch)
  export  --seed S --epoch 15 --offset 0|5    VAL cache of one epoch (reported only; G5 does not select epochs)
  curve   --seed S                  the per-epoch offset-5 curve
"""
from PIL import Image  # noqa: F401  -- PIL before torch (Director's Windows env)
import sys
if "wandb" not in sys.modules:
    sys.modules["wandb"] = None
import argparse
import gzip
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

VERSION = "G5S.0"
FULL_SHA = "c1fbed7f03bbfaac525c422a00b85a436d582116d4b1b2e72d2d69472021479d"
PILOT_SHA = "3128fbf2985c7aa26fc5d38ec334e5cc67ac8281947c8b2da5932e31bdcd496b"
SEEDS = (43, 44)                 # pre-declared; seed 42 = the existing G4F run
G5 = Path("D:/MARIVISION/G5")
G4F_RUN = Path("D:/MARIVISION/G4_full/run")


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


def seed_recipe(F, seed):
    r = dict(F.RECIPE)
    r["seed"] = int(seed)
    return r


def run_dir(seed):
    return G5 / f"seed{seed}" / "run"


def write_marker(d, seed, F):
    d.mkdir(parents=True, exist_ok=True)
    m = d / "g5_seed_marker.json"
    info = {"g5_version": VERSION, "g5_script_sha256": sha256_file(Path(__file__).resolve()), "seed": seed,
            "g4_full_version": F.VERSION, "g4_full_sha256": FULL_SHA,
            "recipe": dict(seed_recipe(F, seed), milestones=list(F.RECIPE["milestones"]))}
    if m.exists():
        old = json.loads(m.read_text(encoding="utf-8"))
        if old.get("seed") != seed or old.get("recipe") != info["recipe"]:
            raise SystemExit(f"[REFUSED] {m} belongs to a different seed / recipe")
        return
    m.write_text(json.dumps(info, indent=1), encoding="utf-8")


def check_seed(seed):
    if seed not in SEEDS:
        raise SystemExit(f"[REFUSED] seed {seed} not pre-declared; G5 seeds are {SEEDS} (42 = existing G4F run)")


# =============================================================================
def selftest(a):
    import tempfile
    import torch
    print(f"[selftest] {VERSION}")
    assert os.environ.get("R0_MSDA_BACKEND") == "pytorch", "set R0_MSDA_BACKEND=pytorch"
    G, F = load_full()
    P, M2, M1, hr_p2 = G.r0_imports(a.pipeline_dir)
    r42, r43 = seed_recipe(F, 42), seed_recipe(F, 43)
    assert r42 == F.RECIPE and {k for k in r43 if r43[k] != F.RECIPE[k]} == {"seed"}
    assert F.epoch_order(1000, 1, 43) != F.epoch_order(1000, 1, 42) == F.epoch_order(1000, 1, 42)
    for bad in (42, 45):
        try:
            check_seed(bad)
            raise AssertionError("undeclared seed accepted")
        except SystemExit:
            pass
    print("  [ok] recipe = g4_full.RECIPE with only the seed replaced; seeds 43/44 only; seeded frame order differs")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        zp, tj, vj = G._synthetic_world(td)
        _a, m0, _c, _p, _ = G.build_arm(P, F.CFG, "cpu", init="none", pretrained_backbone=False, timm_override="convnext_tiny")
        fake = Path(td) / "coco.pth"
        torch.save({"model": m0.state_dict()}, fake)
        del m0
        kw = dict(steps_per_epoch=2, epochs=1, zip_path=zp, train_json=tj, val_json=vj, pipeline_dir=a.pipeline_dir,
                  coco_ckpt=fake, pretrained=False, timm_override="convnext_tiny", workers=0, log=lambda *x: None,
                  selftest_downscale=256)
        sd = {}
        for s in (43, 44):
            d = Path(td) / f"seed{s}" / "run"
            write_marker(d, s, F)
            m, curve = F.run_training(G, P, M2, M1, hr_p2, "cpu", d, recipe=seed_recipe(F, s), **kw)
            ck = torch.load(d / "g4f_epoch01.pth", map_location="cpu", weights_only=False)
            assert ck["g4"]["recipe"]["seed"] == s and curve[0]["tiled"] is not None
            n = ck["g4"]["first_ids_positions"]
            n_img = len(json.loads(Path(tj).read_text(encoding="utf-8"))["images"])
            assert n == F.epoch_order(n_img, 1, s)[:4], (n, n_img)
            sd[s] = {k: v.clone() for k, v in m.state_dict().items()}
            del m
        k = "transformer.encoder.layers.0.self_attn.sampling_offsets.weight"
        assert not torch.equal(sd[43][k], sd[44][k])
        try:
            write_marker(Path(td) / "seed43" / "run", 44, F)
            raise AssertionError("marker of another seed accepted")
        except SystemExit:
            pass
        try:
            F.run_training(G, P, M2, M1, hr_p2, "cpu", Path(td) / "seed43" / "run", recipe=seed_recipe(F, 43), **kw)
            raise AssertionError("overwrite not refused")
        except SystemExit:
            pass
        print("  [ok] seeds 43 and 44 train through g4_full.run_training into separate folders; checkpoints carry the "
              "seed; the models differ; a folder of another seed or a fresh run over an existing one is refused")
    print("SELFTEST PASS")


# =============================================================================
def smoke(a):
    import gc
    import torch
    check_seed(a.seed)
    print(f"[smoke] {VERSION} seed {a.seed}")
    if os.environ.get("R0_MSDA_BACKEND") != "pytorch":
        sys.exit("set R0_MSDA_BACKEND=pytorch first")
    assert torch.cuda.is_available()
    G, F = load_full()
    P, M2, M1, hr_p2 = G.r0_imports(a.pipeline_dir)
    rep = {"version": VERSION, "seed": a.seed}
    # 1 the wrapper reproduces the seed-42 recipe of the real G4F run
    e15 = torch.load(G4F_RUN / "g4f_epoch15.pth", map_location="cpu", weights_only=False)["g4"]
    e01 = torch.load(G4F_RUN / "g4f_epoch01.pth", map_location="cpu", weights_only=False)["g4"]
    n_train = 20310
    stored = dict(e15["recipe"])
    mine = dict(seed_recipe(F, 42), milestones=list(F.RECIPE["milestones"]))
    rep["seed42_recipe_equals_G4F_run"] = stored == mine
    rep["seed42_epoch1_order_equals_G4F_run"] = e01["first_ids_positions"] == F.epoch_order(n_train, 1, 42)[:4]
    rep["seed_order_differs_from_42"] = F.epoch_order(n_train, 1, a.seed)[:4] != F.epoch_order(n_train, 1, 42)[:4]
    # 2 real steps with this seed + inline eval on 3 offset-5 frames
    out = G5 / "smoke" / f"seed{a.seed}"
    if out.exists():
        shutil.rmtree(out)
    _val, ids5 = F.val_ids(M1, G.VAL_JSON, 5, 10)
    logs = []
    lg = lambda s: (logs.append(s), print("   ", s))
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    rec = dict(seed_recipe(F, a.seed), log_every=20)
    _m, curve = F.run_training(G, P, M2, M1, hr_p2, "cuda", out, recipe=rec, steps_per_epoch=40, epochs=1,
                               eval_image_ids=ids5[:3], log=lg)
    del _m
    gc.collect()
    torch.cuda.empty_cache()
    ck = torch.load(out / "g4f_epoch01.pth", map_location="cpu", weights_only=False)["g4"]
    rep["checkpoint_seed"] = ck["recipe"]["seed"]
    rep["checkpoint_order_equals_seed_order"] = ck["first_ids_positions"] == F.epoch_order(n_train, 1, a.seed)[:4]
    rep["inline_eval_ran"] = curve[0].get("tiled") is not None
    rep["eval_peak_gb"] = curve[0].get("eval_peak_gb")

    def avg_at(step):
        for s in logs:
            if s.strip().startswith(f"epoch 1/1 step {step}/40"):
                return float(s.split(" s/step")[0].split("| ")[-1])
        return None
    a20, a40 = avg_at(20), avg_at(40)
    rep["s_per_step"] = (40 * a40 - 20 * a20) / 20 if (a20 and a40) else None
    rep["train_peak_gb"] = max((float(s.split("peak ")[1].split(" GB")[0]) for s in logs if "peak " in s), default=None)
    # 3 refusal to overwrite
    try:
        F.run_training(G, P, M2, M1, hr_p2, "cuda", out, recipe=rec, steps_per_epoch=1, epochs=1,
                       eval_image_ids=ids5[:1], log=lambda *x: None)
        rep["fresh_run_over_existing_refused"] = False
    except SystemExit:
        rep["fresh_run_over_existing_refused"] = True
    # 4 disk and the real run folder
    ck_gb = (out / "g4f_epoch01.pth").stat().st_size / 2 ** 30
    rz_gb = (out / "resume_latest.pth").stat().st_size / 2 ** 30
    need = 15 * ck_gb + rz_gb + 15 * 0.03
    rep["disk"] = {"epoch_ckpt_gb": ck_gb, "resume_gb": rz_gb, "need_per_seed_gb": need,
                   "need_both_seeds_gb": 2 * need, "free_gb": shutil.disk_usage(str(G5.anchor)).free / 2 ** 30}
    rep["real_run_folder"] = {"path": str(run_dir(a.seed)), "exists": (run_dir(a.seed) / "resume_latest.pth").exists()}
    # 5 projection (G4F eval: ~13.8 min per 432-frame export)
    ev_min = 13.8
    tr_h = (rep["s_per_step"] or 0) * (n_train // 2) / 3600
    rep["projection_h"] = {"train_per_epoch": tr_h, "eval_per_epoch": ev_min / 60, "total_15_epochs": 15 * (tr_h + ev_min / 60)}
    rep["wall_min"] = (time.time() - t0) / 60
    rep["checks"] = {"seed42_recipe_identity": rep["seed42_recipe_equals_G4F_run"] and rep["seed42_epoch1_order_equals_G4F_run"],
                     "seed_applied": rep["checkpoint_seed"] == a.seed and rep["checkpoint_order_equals_seed_order"]
                     and rep["seed_order_differs_from_42"],
                     "inline_eval": rep["inline_eval_ran"], "refuses_overwrite": rep["fresh_run_over_existing_refused"],
                     "memory_under_11GB": max(rep["train_peak_gb"] or 99, rep["eval_peak_gb"] or 0) < 11,
                     "disk_for_this_seed": rep["disk"]["free_gb"] > 1.1 * need,
                     "real_run_not_started": not rep["real_run_folder"]["exists"]}
    (out / "g5_smoke_report.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print(json.dumps(rep, indent=1, default=str))
    print("SMOKE PASS" if all(rep["checks"].values()) else "SMOKE FAIL")


# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["selftest", "smoke", "train", "export", "curve"], required=True)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--epoch", type=int)
    ap.add_argument("--offset", type=int, choices=[0, 5])
    ap.add_argument("--pipeline-dir", default="D:/MARIVISION/H0_workbench/eval_tools")
    a = ap.parse_args()
    if a.mode == "selftest":
        return selftest(a)
    if a.seed is None:
        sys.exit("--seed is required")
    check_seed(a.seed)
    if a.mode == "smoke":
        return smoke(a)
    G, F = load_full()
    d = run_dir(a.seed)
    if a.mode == "curve":
        curve = json.loads((d / "curve.json").read_text(encoding="utf-8"))
        print(f"seed {a.seed}\n| epoch | LR factor | avg loss | plain AP30 (offset 5) | tiled AP30 (offset 5) |\n|---|---|---|---|---|")
        for r in sorted(curve, key=lambda r: r["epoch"]):
            f = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
            print(f"| {r['epoch']} | {r['lr_factor']} | {r['avg_loss']:.3f} | {f(r.get('plain'))} | {f(r.get('tiled'))} |")
        return
    if os.environ.get("R0_MSDA_BACKEND") != "pytorch":
        sys.exit("set R0_MSDA_BACKEND=pytorch first")
    P, M2, M1, hr_p2 = G.r0_imports(a.pipeline_dir)
    if a.mode == "train":
        write_marker(d, a.seed, F)
        F.run_training(G, P, M2, M1, hr_p2, "cuda", d, recipe=seed_recipe(F, a.seed), resume=a.resume)
        return
    if a.epoch is None or a.offset is None:
        sys.exit("--epoch and --offset are required")
    ck = d / f"g4f_epoch{a.epoch:02d}.pth"
    cp = d / "tile_cache" / f"r0_tile_cache_g5_seed{a.seed}_e{a.epoch:02d}_stride10_offset{a.offset}.json.gz"
    if cp.exists():
        sys.exit(f"[REFUSED] {cp} exists")
    m, post, ckd = G.rebuild_from_ckpt(P, ck, "cuda")
    if ckd["g4"]["recipe"]["seed"] != a.seed:
        sys.exit(f"[REFUSED] {ck} was trained with seed {ckd['g4']['recipe']['seed']}")
    cache, sec = F.export_with_model(G, M1, hr_p2, m, post, "cuda", G.VAL_JSON, G.ZIP, a.offset, 10, 4,
                                     meta_extra={"source_checkpoint_path": str(ck.resolve()),
                                                 "source_checkpoint_sha256": sha256_file(ck),
                                                 "source_checkpoint_epoch": ckd.get("epoch"), "g4": ckd.get("g4"),
                                                 "g5_version": VERSION, "seed": a.seed})
    cp.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cp, "wt", encoding="utf-8") as fh:
        json.dump(cache, fh)
    print(f"  wrote {cp} ({sec / 60:.1f} min)\n  sha256 {sha256_file(cp)}")


if __name__ == "__main__":
    main()
