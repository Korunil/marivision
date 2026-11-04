r"""m2_evaluate.py - M2 scoring (CPU only, VAL only): plain and tiled inference of the two M2 arms and of R0, from
raw tile caches (m1_1_r0_tile_cache.py), with the HR-P2 runner's evaluator, pycocotools parity and the 9-video bootstrap.
MARIVISION-HR-P2, Claude lead, 2026-10-27.

For every cache it builds two prediction sets from the SAME forward passes:
  <name>_plain  the full-frame tile (Stage B) = ordinary R0-style inference
  <name>_tiled  V0 = M1 merge of all 17 tiles (pipeline defaults)
Each is written as a COCO-results file + MARIVISION_FROZEN_R50_VAL_PREDS_MANIFEST_V1 manifest (runner-compatible).

PRE-DECLARED M2 COMPARISONS (from m2_predeclaration.md; written before any M2 result):
  PRIMARY    T_tiled  vs F_plain   macro AP30 (native protocol: no score cut, top-100), 432 VAL frames offset 0;
                                   SUCCESS = bootstrap 95% CI low > 0 AND >= 6/9 videos positive
  SECONDARY  T_tiled vs F_tiled | T_plain vs F_plain | F_plain vs R0e15_plain (effect of the extra epoch alone) |
             T_tiled vs R0e15_plain | T_tiled vs R0e04_plain | R0e15_tiled vs R0e15_plain (M1, for reference)
  Plus per-class AP30, small-object recall bins and the FP anatomy (score >= 0.05) of every variant.

Run from D:\MARIVISION\H0_workbench\eval_tools:
  python -W error m2_evaluate.py --mode selftest
  python -W error m2_evaluate.py --mode score --tag seed42 --cache-T ... --cache-F ... --cache-R0e15 ... --cache-R0e04 ...
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_VERSION = "M2E.0"
CLASSES6 = ("ANIMAL", "BOAT", "MARINE_MARKER", "OBJECT", "SAILING_VESSEL", "SHIP")
COMPARISONS = [  # (candidate, baseline, role)
    ("T_tiled", "F_plain", "PRIMARY"),
    ("T_tiled", "F_tiled", "secondary"),
    ("T_plain", "F_plain", "secondary"),
    ("F_plain", "R0e15_plain", "secondary"),
    ("T_tiled", "R0e15_plain", "secondary"),
    ("T_tiled", "R0e04_plain", "secondary"),
    ("R0e15_tiled", "R0e15_plain", "secondary (= M1)"),
]


def import_modules(a):
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    sys.path.insert(1, str(Path(a.m1_dir).resolve()))
    import marivision_hr_p2_pipeline as hr_p2
    import m1_r0_tiled_inference as M1
    import m1_1_merge_variants as MV
    import m1_compare_plain_vs_tiled as CMP
    import m1_1_fp_anatomy as FA
    import train_marivision_hr_p2_arms as R
    return hr_p2, M1, MV, CMP, FA, R


def preds_from_cache(MV, M1, hr_p2, cache):
    l2c = {int(k): int(v) for k, v in cache["meta"]["label_to_cat_id"].items()}
    plain, tiled = [], []
    for iid, fr in cache["frames"].items():
        tiles = MV.frame_tiles(M1, hr_p2, fr)
        b = [t for t in tiles if t[0].stage == MV.STAGE_B][0][1]
        for d in M1.plain_dets_native([x[0] for x in b], [x[1] for x in b], [x[2:6] for x in b], l2c):
            plain.append({"image_id": int(iid), "category_id": d["category_id"], "bbox": d["bbox"], "score": d["score"]})
        for d in MV.build_variant("V0", M1, hr_p2, l2c, fr):
            tiled.append({"image_id": int(iid), "category_id": int(d["category_id"]),
                          "bbox": [float(v) for v in d["bbox"]], "score": float(d["score"]), "stage": d.get("stage", "?")})
    return plain, tiled


def fp_buckets(FA, preds, gts_by_img):
    by_img = defaultdict(list)
    for p in preds:
        by_img[p["image_id"]].append(p)
    c = Counter()
    for iid, ps in by_img.items():
        for _, o in FA.classify_frame(ps, gts_by_img.get(iid, [])):
            c[o] += 1
    return c


def run_selftest(a) -> None:
    hr_p2, M1, MV, CMP, FA, R = import_modules(a)
    names = {n for c in COMPARISONS for n in c[:2]}
    assert names == {"T_tiled", "T_plain", "F_tiled", "F_plain", "R0e15_plain", "R0e15_tiled", "R0e04_plain"}, names
    assert COMPARISONS[0] == ("T_tiled", "F_plain", "PRIMARY")
    print("  [PASS] comparison list = pre-declared M2 plan; primary is T_tiled vs F_plain")
    l2c = {i: i + 1 for i in range(7)}
    crops = M1.build_tiles(hr_p2, 1920, 1080)
    tiles = [[c.stage, c.x0_int, c.y0_int, c.x1_int, c.y1_int, []] for c in crops]
    tiles[0][5] = [[0.8, 5, 300.0, 50.0, 500.0, 330.0]]
    cache = {"meta": {"label_to_cat_id": {str(k): v for k, v in l2c.items()}},
             "frames": {"7": {"W": 1920, "H": 1080, "tiles": tiles}}}
    plain, tiled = preds_from_cache(MV, M1, hr_p2, cache)
    assert plain == [{"image_id": 7, "category_id": 6, "bbox": [300.0, 50.0, 200.0, 280.0], "score": 0.8}], plain
    assert len(tiled) == 1 and tiled[0]["bbox"] == [300.0, 50.0, 200.0, 280.0]
    print("  [PASS] plain = Stage B tile exactly; tiled = V0 merge (single-source object unchanged)")
    c = fp_buckets(FA, plain, {7: [{"category_id": 6, "bbox": [300, 50, 200, 280]}]})
    assert c == Counter({"TP": 1})
    print("  [PASS] FP anatomy hook")
    print(f"[M2E SELFTEST] 3/3 passed (script {SCRIPT_VERSION}, sha {MV.sha256_file(Path(__file__).resolve())})")


def run_score(a) -> None:
    hr_p2, M1, MV, CMP, FA, R = import_modules(a)
    if "test" in Path(a.coco_val).name.lower() or "val" not in Path(a.coco_val).name.lower():
        raise SystemExit("[SPLIT BLOCK] --coco-val must be the VALIDATION json; TEST is locked.")
    val_sha = MV.sha256_file(a.coco_val)
    report_dir = Path(a.report_dir)
    stem = report_dir / f"m2_results_{a.tag}"
    if Path(str(stem) + ".json").exists():
        raise SystemExit(f"[REFUSED] {stem}.json exists; choose a new --tag.")
    pred_dir = Path(a.out_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    caches, metas = {}, {}
    for name, path in (("T", a.cache_T), ("F", a.cache_F), ("R0e15", a.cache_R0e15), ("R0e04", a.cache_R0e04)):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            caches[name] = json.load(f)
        m = caches[name]["meta"]
        if m.get("schema") != "MARIVISION_M1_1_R0_TILE_CACHE_V1" or m["coco_val_sha256"] != val_sha:
            raise SystemExit(f"[REFUSED] cache {name}: wrong schema or VAL json.")
        if (int(m["val_frame_stride"]), int(m["val_frame_offset"])) != (10, 0):
            raise SystemExit(f"[REFUSED] cache {name} is not stride 10 / offset 0 (the pre-declared frames).")
        metas[name] = dict(m, cache_path=str(Path(path).resolve()), cache_sha256=MV.sha256_file(path))
    frame_sets = {n: set(c["frames"]) for n, c in caches.items()}
    if len({frozenset(s) for s in frame_sets.values()}) != 1:
        raise SystemExit("[REFUSED] caches cover different frames.")
    shas = {n: m["source_checkpoint_sha256"] for n, m in metas.items()}
    if len(set(shas.values())) != 4:
        raise SystemExit(f"[REFUSED] the four caches must come from four different checkpoints: {shas}")

    val_idx = R.SeaClipsZipNativeIndex(Path(a.zip), Path(a.coco_val), split_hint="val")
    eval_ids = [int(i) for i in sorted(frame_sets["T"], key=int)]
    eval_images = {i: val_idx.images[i] for i in eval_ids}
    eval_anns = {i: val_idx.anns_by_image.get(i, []) for i in eval_ids}
    gts_fa = {i: [{"category_id": int(g["category_id"]), "bbox": [float(v) for v in g["bbox"]]} for g in eval_anns[i]]
              for i in eval_ids}
    print(f"[M2E {SCRIPT_VERSION}] {len(eval_ids)} VAL frames, {sum(len(v) for v in eval_anns.values())} GT")

    variants, per_video, parity, fa, files = {}, {}, {}, {}, {}
    for name in ("T", "F", "R0e15", "R0e04"):
        plain, tiled = preds_from_cache(MV, M1, hr_p2, caches[name])
        for kind, preds in (("plain", plain), ("tiled", tiled)):
            k = f"{name}_{kind}"
            clean = [{x: d[x] for x in ("image_id", "category_id", "bbox", "score")} for d in preds]
            pj = pred_dir / f"m2_{a.tag}_{k}.json"
            if pj.exists():
                raise SystemExit(f"[REFUSED] {pj} exists.")
            pj.write_text(json.dumps(clean), encoding="utf-8")
            man = {"schema": "MARIVISION_FROZEN_R50_VAL_PREDS_MANIFEST_V1", "split": "val",
                   "model_description": f"M2 {k}: {'full-frame' if kind == 'plain' else 'V0 tiled'} inference from tile cache",
                   "source_checkpoint_path": metas[name]["source_checkpoint_path"],
                   "source_checkpoint_sha256": metas[name]["source_checkpoint_sha256"],
                   "source_checkpoint_epoch": metas[name].get("source_checkpoint_epoch"),
                   "export_command": " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
                   "export_script_sha256": MV.sha256_file(Path(__file__).resolve()),
                   "tile_cache": metas[name]["cache_path"], "tile_cache_sha256": metas[name]["cache_sha256"],
                   "box_format": "xywh_abs_pixels_native_frame", "category_id_base": 1, "score_threshold_applied": 0.0,
                   "max_dets_per_image": 100, "coco_val_sha256": val_sha, "val_frame_stride": 10, "val_frame_offset": 0,
                   "predictions_file_sha256": MV.sha256_file(pj)}
            Path(str(pj) + ".manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
            files[k] = {"path": str(pj), "sha256": man["predictions_file_sha256"]}
            ov, pv = R.evaluate_predictions_dict_and_per_video(eval_images, eval_anns, clean)
            variants[k], per_video[k] = ov, pv
            parity[k] = R.verify_against_pycocotools_if_available(eval_images, eval_anns, clean, ov, fail_fast=True,
                                                                  require_pycocotools=not a.allow_missing_pycocotools)
            fa[k] = fp_buckets(FA, preds, gts_fa)
            print(f"  {k}: macro AP30 {ov['AP30_mAP']} | parity {parity[k].get('parity_status')}")

    comps = {}
    for cand, base, role in COMPARISONS:
        boots = {m: CMP.paired_bootstrap(hr_p2, per_video[base][m], per_video[cand][m],
                                         CMP.pooled_value(variants[base], m), CMP.pooled_value(variants[cand], m))
                 for m in CMP.BOOT_METRICS}
        comps[f"{cand}_vs_{base}"] = {"role": role, "verdict": CMP.primary_verdict(boots["AP30_mAP"]), "bootstrap": boots}
    primary = comps["T_tiled_vs_F_plain"]["verdict"]
    parity_ok = all(str(v.get("parity_status")) == "PARITY_PASSED" for v in parity.values())
    report = {"script_version": SCRIPT_VERSION, "script_sha256": MV.sha256_file(Path(__file__).resolve()), "tag": a.tag,
              "n_eval_images": len(eval_ids), "caches": metas | {}, "prediction_files": files,
              "pycocotools_parity_all_passed": parity_ok, "primary": primary, "comparisons": comps,
              "summary": {k: CMP.summarize(v) for k, v in variants.items()},
              "fp_anatomy_s005": {k: dict(v) for k, v in fa.items()}, "metrics_full": variants}
    for n in report["caches"]:
        report["caches"][n] = {k: v for k, v in report["caches"][n].items() if k != "label_to_cat_id"}
    report_dir.mkdir(parents=True, exist_ok=True)
    Path(str(stem) + ".json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")

    order = ["R0e04_plain", "R0e04_tiled", "R0e15_plain", "R0e15_tiled", "F_plain", "F_tiled", "T_plain", "T_tiled"]
    sm = report["summary"]
    L = [f"# M2 results - {a.tag}", "",
         f"{len(eval_ids)} VAL frames (stride 10, offset 0); pycocotools parity: {'PASSED' if parity_ok else 'NOT ALL PASSED'}", "",
         f"**PRIMARY (T_tiled vs F_plain, macro AP30 native): {primary['status']}** - pooled delta {primary['pooled_delta']}, "
         f"95% CI [{primary['ci95_low']}, {primary['ci95_high']}], {primary['n_videos_positive']}/{primary['n_videos']} videos "
         f"positive (rule: {primary['rule']})", "",
         "| variant | macro AP30 | AP50 | GT-wt AP30 | CA-AP50 | recall@0.3 | FP/frame (native, IoU .5) | " + " | ".join(CLASSES6) + " |",
         "|---|---|---|---|---|---|---|" + "---|" * len(CLASSES6)]
    for k in order:
        s = sm[k]
        L.append(f"| {k} | {s['AP30_mAP']} | {s['AP50_mAP']} | {s['AP30_gt_weighted_mAP']} | {s['mAP_class_agnostic_50']} | "
                 f"{s['recall_1to1_iou30']} | {variants[k].get('fp_per_frame_1to1_iou50')} | "
                 + " | ".join(str(s['AP30_per_class'][c]) for c in CLASSES6) + " |")
    L += ["", "| comparison | role | pooled delta AP30 | CI95 | videos > 0 | verdict | pooled delta AP50 | pooled delta recall@0.3 |",
          "|---|---|---|---|---|---|---|---|"]
    for name, c in comps.items():
        v, b = c["verdict"], c["bootstrap"]
        L.append(f"| {name} | {c['role']} | {v['pooled_delta']} | [{v['ci95_low']}, {v['ci95_high']}] | "
                 f"{v['n_videos_positive']}/{v['n_videos']} | {v['status']} | {b['AP50_mAP']['pooled_aggregate_delta']} | "
                 f"{b['overall_gt_recall_30']['pooled_aggregate_delta']} |")
    L += ["", "| FP anatomy (score >= 0.05, 1-to-1 IoU 0.3) | TP | DUPLICATE | WRONG_CLASS | PART_OF_OBJECT | LOCALIZATION | BACKGROUND |",
          "|---|---|---|---|---|---|---|"]
    for k in order:
        c = fa[k]
        L.append(f"| {k} | {c['TP']} | {c['DUPLICATE']} | {c['WRONG_CLASS']} | {c['PART_OF_OBJECT']} | {c['LOCALIZATION']} | {c['BACKGROUND']} |")
    L += ["", "| recall@0.3 by area bin (640-scale) | " + " | ".join(order) + " |", "|---|" + "---|" * len(order)]
    for b in sm["T_tiled"]["recall_by_area_bin_iou30"]:
        L.append(f"| {b} | " + " | ".join(str(round(sm[k]["recall_by_area_bin_iou30"].get(b, float('nan')), 4)) for k in order) + " |")
    L += ["", "Per-video AP30 deltas, primary: " + json.dumps(comps["T_tiled_vs_F_plain"]["bootstrap"]["AP30_mAP"].get("per_video_deltas"))]
    Path(str(stem) + "_tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n" + "\n".join(L) + f"\n\n[M2E] wrote {stem}.json and {stem}_tables.md")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("selftest", "score"), required=True)
    ap.add_argument("--tag", default="seed42")
    ap.add_argument("--cache-T")
    ap.add_argument("--cache-F")
    ap.add_argument("--cache-R0e15", default="D:/MARIVISION/R0_frozen_baseline/m1_1_tile_cache/r0_tile_cache_epoch15_stride10_offset0.json.gz")
    ap.add_argument("--cache-R0e04", default="D:/MARIVISION/R0_frozen_baseline/m1_1_tile_cache/r0_tile_cache_epoch04_stride10_offset0.json.gz")
    ap.add_argument("--m1-dir", default="D:/MARIVISION/R0_frozen_baseline")
    ap.add_argument("--zip", default="D:/MARIVISION/SeaClips.zip")
    ap.add_argument("--coco-val", default="D:/MARIVISION/seaclips-val.json")
    ap.add_argument("--out-dir", default="D:/MARIVISION/R0_frozen_baseline/m2_preds")
    ap.add_argument("--report-dir", default="m2_reports")
    ap.add_argument("--allow-missing-pycocotools", action="store_true")
    a = ap.parse_args()
    if a.mode == "selftest":
        run_selftest(a)
    else:
        if not (a.cache_T and a.cache_F):
            raise SystemExit("[REFUSED] --cache-T and --cache-F are required.")
        run_score(a)


if __name__ == "__main__":
    main()
