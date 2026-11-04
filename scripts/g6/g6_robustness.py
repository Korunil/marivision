r"""
G6 robustness: the frozen G5 models on the SeaClips paper's two corrupted copies of TEST (supplement Table 4), scored
once per corruption.  Descriptive secondary of claude/g5_predeclaration.md Part C ("robustness ... declared before
running"); no model, routing or threshold changes.

Location: D:/MARIVISION/R0_frozen_baseline/g6_robustness.py  (run FROM that folder, next to g5_test_eval.py G5T.0,
g4_full.py G4F.0.1 and g4_pilot.py G4P.0, all SHA-checked).  R0_MSDA_BACKEND=pytorch, HF_HUB_DISABLE_SYMLINKS_WARNING=1.
Needs: albumentations (selftest was run with 2.0.8; the version is recorded in every output).

Corruptions (the paper's Albumentations settings, applied to the full native frame before tiling / resizing):
  rain      RandomRain(p=1, rain_type='drizzle', drop_width=10, drop_length=20, drop_color=(100, 100, 100))
  sunflare  RandomSunFlare(p=1, method='physics_based', flare_roi=(0, 0, 1, 0.3), src_radius=120)
  Every other parameter = the library default.  Randomness: one seed per frame = base + image id
  (rain 1,000,000; sunflare 2,000,000), so each corrupted frame is identical for every model and every rerun.
Models: seed42 / seed43 / seed44 (tiled) and r0e15 / r0e04 (full frame), exactly the G5 checkpoints.
Report per corruption: routed (G5 primary system) and unrouted mAP@0.3 per seed and mean +- SD, R0 references,
drop vs the clean G5 TEST result, per class, and the paper's corrupted-TEST means (no SDs published -> no test).

Modes:
  selftest                                        CPU, synthetic
  proof                                           VAL only (GPU, ~5 min): corruption 'none' reproduces the existing
                                                  VAL cache; corrupted export deterministic; sample images saved
  export --corruption C --model M --test-unlock   resumable, 250-frame parts
  score  --corruption C --test-unlock             once per corruption (lock)
"""
from PIL import Image
import sys
if "wandb" not in sys.modules:
    sys.modules["wandb"] = None
import os
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")   # no online version check (would warn under -W error)
import argparse
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path

VERSION = "G6R.0"
G5T_SHA = "ecb20f5681ce914ca25c5a9811c790d58685d748f5a1d0fd8069adc37df12adf"
CORRUPTIONS = {
    "rain": {"cls": "RandomRain", "kw": {"p": 1.0, "rain_type": "drizzle", "drop_width": 10, "drop_length": 20,
                                         "drop_color": (100, 100, 100)}, "seed_base": 1_000_000},
    "sunflare": {"cls": "RandomSunFlare", "kw": {"p": 1.0, "method": "physics_based", "flare_roi": (0, 0, 1, 0.3),
                                                 "src_radius": 120}, "seed_base": 2_000_000},
}
PAPER_CORRUPTED = {"rain": {"Def. DETR R50": 5.52, "Def. DETR Swin-B": 23.71, "TransVOD++ Swin-B": 24.47},
                   "sunflare": {"Def. DETR R50": 18.16, "Def. DETR Swin-B": 30.11, "TransVOD++ Swin-B": 32.64}}
PAPER_CLEAN = {"Def. DETR R50": 22.86, "Def. DETR Swin-B": 31.28, "TransVOD++ Swin-B": 31.95}
ROOT = Path("D:/MARIVISION")
G5 = ROOT / "G5"
ROBUST = G5 / "robust"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_e5(check=True):
    here = Path.cwd()
    p = here / "g5_test_eval.py"
    if not p.exists():
        raise SystemExit(f"[REFUSED] g5_test_eval.py must sit in the run folder ({here})")
    if check and sha256_file(p) != G5T_SHA:
        raise SystemExit(f"[REFUSED] g5_test_eval.py sha {sha256_file(p)} != {G5T_SHA}")
    sys.path.insert(0, str(here))
    import g5_test_eval as E
    return E


# =============================================================================
# corruption
# =============================================================================
class Corruptor:
    def __init__(self, name):
        import albumentations as A
        self.name, self.version = name, A.__version__
        if name == "none":
            self.t = None
            return
        c = CORRUPTIONS[name]
        self.base = c["seed_base"]
        self.t = A.Compose([getattr(A, c["cls"])(**c["kw"])], seed=0)

    def __call__(self, pil, iid):
        if self.t is None:
            return pil
        import numpy as np
        self.t.set_random_seed(self.base + int(iid))
        return Image.fromarray(self.t(image=np.asarray(pil.convert("RGB")))["image"])

    def describe(self):
        return {"corruption": self.name, "albumentations": self.version,
                "params": {k: list(v) if isinstance(v, tuple) else v for k, v in CORRUPTIONS.get(self.name, {}).get("kw", {}).items()},
                "seed": "set_random_seed(seed_base + image id)", "seed_base": CORRUPTIONS.get(self.name, {}).get("seed_base")}


class CorruptDS:
    """What g5_test_eval.export_frames needs from a dataset: .images and ._load_image(info)."""

    def __init__(self, ds, corr):
        self.ds, self.corr, self.images = ds, corr, ds.images

    def _load_image(self, info):
        return self.corr(self.ds._load_image(info), int(info["id"]))


def patched_open(E, corr):
    orig = E.open_dataset

    def op(zip_path, ann):
        ds, l2c = orig(zip_path, ann)
        return CorruptDS(ds, corr), l2c
    return orig, op


def export_corrupted(E, G, P, M1, hr_p2, corruption, name, spec, ann, zip_path, out_root, device, **kw):
    corr = Corruptor(corruption)
    orig, op = patched_open(E, corr)
    E.open_dataset = op
    try:
        man = E.export_model(G, P, M1, hr_p2, name, spec, ann, zip_path, out_root, device, **kw)
    finally:
        E.open_dataset = orig
    if man:
        info = dict(corr.describe(), g6_version=VERSION, g6_sha256=sha256_file(Path(__file__).resolve()))
        (Path(out_root) / name / "corruption.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return man


# =============================================================================
# summary
# =============================================================================
def msd(v):
    return (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0)


def summarize(res, clean, corruption, seeds):
    out = {"corruption": corruption, "rows": {}}
    for var in ("routed", "tiled"):
        vals = [100 * res[s][var]["map30"] for s in seeds]
        cl = [100 * clean[s][var]["map30"] for s in seeds] if clean else None
        m, sd = msd(vals)
        out["rows"][var] = {"seeds": vals, "mean": m, "sd": sd,
                            "clean_mean": statistics.mean(cl) if cl else None,
                            "drop": (m - statistics.mean(cl)) if cl else None}
    for r in ("r0e15", "r0e04"):
        if r in res:
            v = 100 * res[r]["plain"]["map30"]
            c = 100 * clean[r]["plain"]["map30"] if clean and r in clean else None
            out["rows"][r] = {"value": v, "clean": c, "drop": (v - c) if c is not None else None}
    out["paper"] = {k: {"corrupted": v, "clean": PAPER_CLEAN[k], "drop": v - PAPER_CLEAN[k]}
                    for k, v in PAPER_CORRUPTED[corruption].items()}
    return out


def report(summ, res, corr_info, seeds):
    f = lambda v: "–" if v is None else f"{v:.2f}"
    s = summ["rows"]
    L = [f"# G6 robustness: {summ['corruption']} ({VERSION}) -- single scoring, all TEST frames", "",
         f"Corruption: {json.dumps(corr_info)}", "",
         "| system | mAP@0.3 (corrupted) | clean (G5) | change |", "|---|---|---|---|",
         f"| ours routed (G5 primary), 3 seeds | {f(s['routed']['mean'])} ± {f(s['routed']['sd'])} "
         f"({', '.join(f(v) for v in s['routed']['seeds'])}) | {f(s['routed']['clean_mean'])} | {f(s['routed']['drop'])} |",
         f"| ours unrouted, 3 seeds | {f(s['tiled']['mean'])} ± {f(s['tiled']['sd'])} "
         f"({', '.join(f(v) for v in s['tiled']['seeds'])}) | {f(s['tiled']['clean_mean'])} | {f(s['tiled']['drop'])} |"]
    for r in ("r0e15", "r0e04"):
        if r in s:
            L.append(f"| {r} plain (ours) | {f(s[r]['value'])} | {f(s[r]['clean'])} | {f(s[r]['drop'])} |")
    for k, v in summ["paper"].items():
        L.append(f"| {k} (paper, mean of 3 seeds) | {f(v['corrupted'])} | {f(v['clean'])} | {f(v['drop'])} |")
    L += ["", "The paper publishes no SDs for the corrupted sets, so no significance test is made.", "",
          "| model | variant | mAP@0.3 | small | medium | large | mAPca@0.3 | mAP@0.5 |", "|---|---|---|---|---|---|---|---|"]
    g = lambda v: "–" if v is None else f"{100 * v:.2f}"
    for n, r in res.items():
        for var, e in r.items():
            L.append(f"| {n} | {var} | {g(e['map30'])} | {g(e['size30'].get('small'))} | {g(e['size30'].get('medium'))} | "
                     f"{g(e['size30'].get('large'))} | {g(e['mapca30'])} | {g(e['map50'])} |")
    classes = sorted({c for r in res.values() for e in r.values() for c in e["per_class30"]})
    L += ["", "| model | variant | " + " | ".join(classes) + " |", "|---|---|" + "---|" * len(classes)]
    for n, r in res.items():
        for var, e in r.items():
            L.append(f"| {n} | {var} | " + " | ".join(g(e["per_class30"].get(c)) for c in classes) + " |")
    return "\n".join(L) + "\n"


def write_results(out, corruption, res, summ, corr_info, mans, seeds):
    out = Path(out)
    lock = out / f"g6_{corruption}_scored.lock"
    if lock.exists():
        raise SystemExit(f"[REFUSED] {corruption} was already scored ({lock})")
    out.mkdir(parents=True, exist_ok=True)
    md = report(summ, res, corr_info, seeds)
    (out / f"g6_{corruption}_results.json").write_text(json.dumps({"version": VERSION, "summary": summ, "results": res,
                                                                    "corruption": corr_info, "manifests": mans},
                                                                   indent=1), encoding="utf-8")
    (out / f"g6_{corruption}_results.md").write_text(md, encoding="utf-8")
    lock.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    return md


# =============================================================================
def selftest(a):
    import gc
    import shutil
    import tempfile
    import zipfile
    import numpy as np
    import torch
    print(f"[selftest] {VERSION}")
    assert os.environ.get("R0_MSDA_BACKEND") == "pytorch", "set R0_MSDA_BACKEND=pytorch"
    # 1 corruptions
    img = Image.fromarray(np.full((270, 480, 3), 80, np.uint8))
    for c in CORRUPTIONS:
        k = Corruptor(c)
        x1, x2, x3 = (np.asarray(k(img, i)) for i in (7, 7, 8))
        assert (x1 == x2).all() and (x1 != x3).any() and (x1 != 80).any(), c
    sf = np.abs(np.asarray(Corruptor("sunflare")(img, 7)).astype(int) - 80)
    assert sf[:81].mean() > 10 * sf[189:].mean(), (sf[:81].mean(), sf[189:].mean())
    assert (np.asarray(Corruptor("none")(img, 7)) == 80).all()
    print(f"  [ok] rain / sunflare deterministic per frame seed, differ between frames, change the image; sunflare acts "
          f"in the top 30 % (mean change {sf[:81].mean():.1f} vs {sf[189:].mean():.1f} at the bottom); 'none' = identity")
    # 2 summary arithmetic
    mk = lambda v: {"map30": v}
    res = {s: {"routed": mk(0.30 + i / 100), "tiled": mk(0.33 + i / 100)} for i, s in enumerate(("seed42", "seed43", "seed44"))}
    res["r0e15"] = {"plain": mk(0.10)}
    clean = {s: {"routed": mk(0.35), "tiled": mk(0.38)} for s in ("seed42", "seed43", "seed44")}
    clean["r0e15"] = {"plain": mk(0.207)}
    sm = summarize(res, clean, "rain", ("seed42", "seed43", "seed44"))
    assert abs(sm["rows"]["routed"]["mean"] - 31.0) < 1e-9 and abs(sm["rows"]["routed"]["sd"] - 1.0) < 1e-9
    assert abs(sm["rows"]["tiled"]["drop"] - (34.0 - 38.0)) < 1e-9 and abs(sm["rows"]["r0e15"]["drop"] + 10.7) < 1e-9
    assert abs(sm["paper"]["Def. DETR R50"]["drop"] - (5.52 - 22.86)) < 1e-12
    print("  [ok] summary: mean ± SD over seeds, drop vs clean G5, paper drops (R50 rain −17.34)")
    # 3 end to end: synthetic TEST split, tiny models, corrupted export + resume + score + lock
    E = load_e5()
    G, F = E.load_full()
    P, M2, M1, hr_p2 = G.r0_imports(a.pipeline_dir)
    import m1_1_merge_variants as MV
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        td = Path(td)
        zp, tj, vj = G._synthetic_world(td)
        with zipfile.ZipFile(zp, "a") as z:
            for n in [x for x in z.namelist() if x.startswith("SeaClips/val/")]:
                z.writestr(n.replace("SeaClips/val/", "SeaClips/test/"), z.read(n))
        tsj = td / "seaclips-test.json"
        shutil.copy(vj, tsj)
        tst = json.loads(tsj.read_text(encoding="utf-8"))
        keep = {i for v in (1, 2) for i in E.frame_order([m for m in tst["images"] if m["video_id"] == v])[:2]}
        tst["images"] = [m for m in tst["images"] if m["id"] in keep]
        tst["annotations"] = [x for x in tst["annotations"] if x["image_id"] in keep]
        tsj.write_text(json.dumps(tst), encoding="utf-8")
        _a, m, _c, _p, _ = G.build_arm(P, F.CFG, "cpu", init="none", pretrained_backbone=False, timm_override="convnext_tiny")
        ps = td / "s42.pth"
        torch.save({"model": m.state_dict(), "g4": {"cfg": F.CFG, "timm_override": "convnext_tiny", "epoch": 15,
                                                     "recipe": {"seed": 42}}}, ps)
        del m
        _a, m, _c, _p, _ = G.build_arm(P, {"backbone": None, "precision": "fp32", "p_tile": 0.0}, "cpu", init="none",
                                       pretrained_backbone=False)
        pr = td / "r0.pth"
        torch.save({"model": m.state_dict()}, pr)
        del m
        gc.collect()
        specs = {"seed42": {"ckpt": ps, "kind": "tiled", "seed": 42}, "r0e15": {"ckpt": pr, "kind": "plain", "seed": None}}
        tc = td / "robust" / "rain" / "test_cache"
        assert export_corrupted(E, G, P, M1, hr_p2, "rain", "seed42", specs["seed42"], tsj, zp, tc, "cpu",
                                json_sha=None, part=2, max_parts=1, log=lambda *x: None) is None
        man = export_corrupted(E, G, P, M1, hr_p2, "rain", "seed42", specs["seed42"], tsj, zp, tc, "cpu",
                               json_sha=None, part=2, log=lambda *x: None)
        assert man and man["n_frames"] == 4 and (tc / "seed42" / "corruption.json").exists()
        export_corrupted(E, G, P, M1, hr_p2, "rain", "r0e15", specs["r0e15"], tsj, zp, tc, "cpu", json_sha=None,
                         part=2, log=lambda *x: None)
        # the corrupted export differs from a clean export of the same frames (the corruption reached the model)
        tcc = td / "clean_cache"
        E.export_model(G, P, M1, hr_p2, "seed42", specs["seed42"], tsj, zp, tcc, "cpu", json_sha=None, part=4,
                       log=lambda *x: None)
        fc = {i: fr for _l, i, fr in E.iter_cache_frames(tcc / "seed42")}
        fr_ = {i: fr for _l, i, fr in E.iter_cache_frames(tc / "seed42")}
        assert set(fc) == set(fr_) and any(fc[i]["tiles"] != fr_[i]["tiles"] for i in fc)
        gt = json.loads(tsj.read_text(encoding="utf-8"))
        ids = E.frame_order(gt["images"])
        res = E.score_all(MV, M1, hr_p2, gt, ids, {"seed42": tc / "seed42", "r0e15": tc / "r0e15"}, log=lambda *x: None)
        sm = summarize(res, None, "rain", ("seed42",))
        md = write_results(td / "robust" / "rain", "rain", res, sm, Corruptor("rain").describe(), {}, ("seed42",))
        assert "routed" in md
        try:
            write_results(td / "robust" / "rain", "rain", res, sm, Corruptor("rain").describe(), {}, ("seed42",))
            raise AssertionError("second scoring not refused")
        except SystemExit:
            pass
        print("  [ok] end to end on a synthetic TEST split: corrupted export interrupted + resumed, corruption record "
              "written, corrupted predictions differ from clean ones, routed/unrouted scoring, report, one-time lock")
    print("SELFTEST PASS")


# =============================================================================
def proof(a):
    import numpy as np
    import torch
    print(f"[proof] {VERSION} -- VAL only; TEST is not touched")
    if os.environ.get("R0_MSDA_BACKEND") != "pytorch":
        sys.exit("set R0_MSDA_BACKEND=pytorch first")
    E = load_e5()
    G, F = E.load_full()
    P, M2, M1, hr_p2 = G.r0_imports(a.pipeline_dir)
    import gzip
    with gzip.open(E.VAL_CACHES["G4F_e15"], "rt", encoding="utf-8") as fh:
        ref = json.load(fh)
    ids = sorted(int(k) for k in ref["frames"])[:10]
    ds, l2c = E.open_dataset(G.ZIP, G.VAL_JSON)
    m, post, _ = E.check_ckpt(G, P, "seed42", E.MODELS["seed42"], "cuda")
    rep = {"version": VERSION}
    fr, _s = E.export_frames(M1, hr_p2, m, post, "cuda", CorruptDS(ds, Corruptor("none")), ids, "tiled", log=lambda *x: None)
    rep["none_equals_existing_val_cache"] = G.compare_caches(M1, {"frames": fr}, ref, l2c)
    out = ROBUST / "samples"
    out.mkdir(parents=True, exist_ok=True)
    for c in CORRUPTIONS:
        cds = CorruptDS(ds, Corruptor(c))
        a1, t1 = E.export_frames(M1, hr_p2, m, post, "cuda", cds, ids[:3], "tiled", log=lambda *x: None)
        a2, _t = E.export_frames(M1, hr_p2, m, post, "cuda", cds, ids[:3], "tiled", log=lambda *x: None)
        same = all(a1[k]["tiles"] == a2[k]["tiles"] for k in a1)
        im1 = np.asarray(cds._load_image(ds.images[ids[0]]))
        im2 = np.asarray(cds._load_image(ds.images[ids[0]]))
        rep[c] = {"export_deterministic": same, "image_deterministic": bool((im1 == im2).all()),
                  "sec_per_frame": t1 / 3, "changed_vs_clean": any(a1[k]["tiles"] != fr[str(k)]["tiles"] if str(k) in fr else True
                                                                    for k in a1)}
        for i in ids[:2]:
            cds._load_image(ds.images[i]).save(out / f"val_{i}_{c}.jpg", quality=90)
    for i in ids[:2]:
        ds._load_image(ds.images[i]).save(out / f"val_{i}_clean.jpg", quality=90)
    del m
    torch.cuda.empty_cache()
    import albumentations as A
    rep["albumentations"] = A.__version__
    rep["test_json_locked"] = sha256_file(E.TEST_JSON) == E.TEST_JSON_SHA
    rep["clean_g5_results_present"] = (G5 / "g5_test_results.json").exists()
    rep["checks"] = {"none_parity": rep["none_equals_existing_val_cache"]["ok"],
                     "deterministic": all(rep[c]["export_deterministic"] and rep[c]["image_deterministic"] for c in CORRUPTIONS),
                     "corruption_reaches_model": all(rep[c]["changed_vs_clean"] for c in CORRUPTIONS),
                     "test_json_locked": rep["test_json_locked"], "clean_results": rep["clean_g5_results_present"]}
    ROBUST.mkdir(parents=True, exist_ok=True)
    (ROBUST / "g6_proof.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print(json.dumps(rep, indent=1, default=str))
    print(f"  sample images (VAL, stay on this machine): {out}")
    print("PROOF PASS" if all(rep["checks"].values()) else "PROOF FAIL")


# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["selftest", "proof", "export", "score"], required=True)
    ap.add_argument("--corruption", choices=sorted(CORRUPTIONS))
    ap.add_argument("--model")
    ap.add_argument("--test-unlock", action="store_true")
    ap.add_argument("--pipeline-dir", default="D:/MARIVISION/H0_workbench/eval_tools")
    a = ap.parse_args()
    if a.mode == "selftest":
        return selftest(a)
    if a.mode == "proof":
        return proof(a)
    if not a.test_unlock or not a.corruption:
        sys.exit("[REFUSED] TEST modes need --corruption and --test-unlock")
    if os.environ.get("R0_MSDA_BACKEND") != "pytorch":
        sys.exit("set R0_MSDA_BACKEND=pytorch first")
    E = load_e5()
    G, F = E.load_full()
    P, M2, M1, hr_p2 = G.r0_imports(a.pipeline_dir)
    if sha256_file(E.TEST_JSON) != E.TEST_JSON_SHA:
        sys.exit("[REFUSED] TEST label file differs from the lock")
    _v, val_l2c = E.open_dataset(G.ZIP, G.VAL_JSON)
    out = ROBUST / a.corruption
    tc = out / "test_cache"
    if a.mode == "export":
        if a.model not in E.MODELS:
            sys.exit(f"--model must be one of {sorted(E.MODELS)}")
        export_corrupted(E, G, P, M1, hr_p2, a.corruption, a.model, E.MODELS[a.model], E.TEST_JSON, G.ZIP, tc, "cuda",
                         val_l2c=val_l2c)
        return
    import m1_1_merge_variants as MV
    missing = [n for n in E.MODELS if not (tc / n / "manifest.json").exists()]
    if missing:
        sys.exit(f"[REFUSED] incomplete exports for {a.corruption}: {missing}")
    if (out / f"g6_{a.corruption}_scored.lock").exists():
        sys.exit(f"[REFUSED] {a.corruption} was already scored")
    clean_p = G5 / "g5_test_results.json"
    if not clean_p.exists():
        sys.exit(f"[REFUSED] {clean_p} missing (clean G5 results are the reference)")
    clean = json.loads(clean_p.read_text(encoding="utf-8"))["results"]
    mans = {}
    for n in E.MODELS:
        mm = json.loads((tc / n / "manifest.json").read_text(encoding="utf-8"))
        ci = json.loads((tc / n / "corruption.json").read_text(encoding="utf-8"))
        if mm["ann_json_sha256"] != E.TEST_JSON_SHA or mm["ckpt_sha256"] != sha256_file(E.MODELS[n]["ckpt"]) \
                or ci["corruption"] != a.corruption:
            sys.exit(f"[REFUSED] {n}: manifest / corruption record does not match")
        mans[n] = dict(mm, corruption=ci)
    infos = {json.dumps({k: v for k, v in m["corruption"].items() if k not in ("g6_sha256",)}, sort_keys=True) for m in mans.values()}
    if len(infos) != 1:
        sys.exit("[REFUSED] models were exported with different corruption settings or library versions")
    gt = json.loads(E.TEST_JSON.read_text(encoding="utf-8"))
    ids = E.frame_order(gt["images"])
    res = E.score_all(MV, M1, hr_p2, gt, ids, {n: tc / n for n in E.MODELS})
    summ = summarize(res, clean, a.corruption, E.SEEDS)
    md = write_results(out, a.corruption, res, summ, next(iter(mans.values()))["corruption"], mans, E.SEEDS)
    print(md)


if __name__ == "__main__":
    main()
