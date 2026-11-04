r"""m1_1_merge_variants.py - M1.1 step 2 (CPU only): build pre-declared merge variants from a raw R0 tile cache and score
each one against plain R0 with m1_compare_plain_vs_tiled.py (same evaluator, pycocotools parity, 9-video bootstrap).
MARIVISION-HR-P2, Claude lead, 2026-10-27. VAL only. No GPU.

PRE-DECLARED VARIANTS (written 2026-10-27, after the M1 epoch15 offset-0 result was seen, BEFORE any offset-5 data):
  V0_M1_DEFAULT       all 17 tiles -> pipeline merge_multiscale_tile_detections, defaults (= M1.0 exactly).
  V1_SCALE_ROUTED     as V0, but Stage C tiles may only contribute boxes with native area <= 3600 px^2 (60x60; the
                      pipeline's existing stage_c_max_native_area_px2). Large objects come from the full frame only.
  V2_PLAIN_PLUS_SMALL plain R0 output (Stage B) kept untouched; Stage C small boxes (<= 3600 px^2), deduplicated by the
                      pipeline merge over Stage C tiles only, are ADDED. A tile box that overlaps a same-class plain box
                      (IoU >= 0.30, the evaluation IoU) is dropped if that plain box scores >= it, otherwise it replaces
                      the lower-scoring plain box(es). Top-100 per frame by score.
  No other variant, threshold or constant will be tried on the confirmation frames.

PRE-DECLARED PROTOCOL
  Development: epoch15 checkpoint, VAL offset 5 (432 frames disjoint from offset 0, same videos). Choose the variant with
               the highest pooled native macro AP30; ties -> the lower-numbered variant.
  Confirmation: the chosen variant only, offset 0, epoch04 (primary) and epoch15 (replication), M1 success rule
               (bootstrap 95% CI low > 0 AND >= 6/9 videos positive). Reported as a post-M1 revision.
  Consistency check: V0 on the offset-0 epoch15 cache must reproduce M1.0's macro AP30 (0.3459) within 0.002.

Run from D:\MARIVISION\H0_workbench\eval_tools:
  python -W error m1_1_merge_variants.py --mode selftest
  python -W error m1_1_merge_variants.py --mode score --cache <cache.json.gz> --variants V0,V1,V2
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_VERSION = "M1.1-merge.0"
SMALL_MAX_AREA = 3600.0
OVERLAP_IOU = 0.30
VARIANTS = {"V0": "V0_M1_DEFAULT", "V1": "V1_SCALE_ROUTED", "V2": "V2_PLAIN_PLUS_SMALL"}
STAGE_B, STAGE_C = "STAGE_B_FULL_CONTEXT", "STAGE_C_2X_CROP"


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iou_xywh(a, b) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0


def frame_tiles(M1, hr_p2, fr):
    """Rebuild CropWindows and check them against the cached tile bounds."""
    crops = M1.build_tiles(hr_p2, int(fr["W"]), int(fr["H"]))
    if len(crops) != len(fr["tiles"]):
        raise AssertionError("tile count mismatch between cache and pipeline")
    out = []
    for c, t in zip(crops, fr["tiles"]):
        if [c.stage, c.x0_int, c.y0_int, c.x1_int, c.y1_int] != list(t[:5]):
            raise AssertionError(f"tile window mismatch: {t[:5]} vs pipeline {c}")
        out.append((c, t[5]))
    return out


def _tile_preds(M1, l2c, tiles, stages, small_only_stage_c):
    tp = []
    for c, dets in tiles:
        if c.stage not in stages:
            continue
        keep = [d for d in dets if not (small_only_stage_c and c.stage == STAGE_C
                                         and (d[4] - d[2]) * (d[5] - d[3]) > SMALL_MAX_AREA)]
        tp.append((c, M1.tile_dets_for_merge(c, [d[0] for d in keep], [d[1] for d in keep],
                                             [d[2:6] for d in keep], l2c)))
    return tp


def build_variant(name, M1, hr_p2, l2c, fr):
    tiles = frame_tiles(M1, hr_p2, fr)
    W, H = int(fr["W"]), int(fr["H"])
    if name in ("V0", "V1"):
        tp = _tile_preds(M1, l2c, tiles, (STAGE_B, STAGE_C), small_only_stage_c=(name == "V1"))
        return hr_p2.merge_multiscale_tile_detections(tp, source_w=W, source_h=H, score_thresh=0.0, max_det_per_image=100)
    if name == "V2":
        b = [t for t in tiles if t[0].stage == STAGE_B][0][1]
        plain = M1.plain_dets_native([d[0] for d in b], [d[1] for d in b], [d[2:6] for d in b], l2c)
        tp = _tile_preds(M1, l2c, tiles, (STAGE_C,), small_only_stage_c=True)
        small = hr_p2.merge_multiscale_tile_detections(tp, source_w=W, source_h=H, score_thresh=0.0, max_det_per_image=10_000)
        final = [dict(p) for p in plain]
        alive = [True] * len(final)
        added = []
        for s in sorted(small, key=lambda d: -float(d["score"])):
            over = [i for i, p in enumerate(final) if alive[i] and p["category_id"] == s["category_id"]
                    and iou_xywh(p["bbox"], s["bbox"]) >= OVERLAP_IOU]
            if any(float(final[i]["score"]) >= float(s["score"]) for i in over):
                continue
            for i in over:
                alive[i] = False
            added.append({"category_id": int(s["category_id"]), "score": float(s["score"]), "bbox": list(s["bbox"])})
        out = [p for p, ok in zip(final, alive) if ok] + added
        out.sort(key=lambda d: -float(d["score"]))
        return out[:100]
    raise ValueError(name)


def import_modules(a):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(1, str(Path(a.m1_dir).resolve()))
    import marivision_hr_p2_pipeline as hr_p2
    import m1_r0_tiled_inference as M1
    return hr_p2, M1


def run_selftest(a) -> None:
    hr_p2, M1 = import_modules(a)
    l2c = {i: i + 1 for i in range(7)}
    crops = M1.build_tiles(hr_p2, 1920, 1080)

    def tile_of(x, y):
        return [i for i, c in enumerate(crops) if c.stage == STAGE_C and c.x0_int <= x and c.y0_int <= y
                and x + 1 <= c.x1_int and y + 1 <= c.y1_int]

    # synthetic frame: a big sailing vessel seen in Stage B (label 5) and as a large fragment in one Stage C tile;
    # a tiny marker (label 2) visible only in Stage C tiles; a low-score plain box near the marker of the same class.
    tiles = [[c.stage, c.x0_int, c.y0_int, c.x1_int, c.y1_int, []] for c in crops]
    tiles[0][5] = [[0.80, 5, 300.0, 50.0, 500.0, 330.0], [0.04, 2, 1001.0, 301.0, 1011.0, 307.0]]
    c_sv = tile_of(300, 60)[0]
    cx, cy = crops[c_sv].x0_int, crops[c_sv].y0_int
    assert (cx, cy) == (0, 0)
    tiles[c_sv][5] = [[0.70, 5, 300.0, 60.0, 470.0, 320.0]]  # inside the tile, not touching its borders
    for i in tile_of(1000, 300):
        x0, y0 = crops[i].x0_int, crops[i].y0_int
        tiles[i][5].append([0.60, 2, 1000.0 - x0, 300.0 - y0, 1012.0 - x0, 306.0 - y0])
    fr = {"W": 1920, "H": 1080, "tiles": tiles}

    v0 = build_variant("V0", M1, hr_p2, l2c, fr)
    sv0 = [d for d in v0 if d["category_id"] == 6]
    assert len(sv0) == 1 and abs(sv0[0]["bbox"][2] - 200.0) > 1.0, sv0  # fused with the fragment -> shrunk box
    v1 = build_variant("V1", M1, hr_p2, l2c, fr)
    sv1 = [d for d in v1 if d["category_id"] == 6]
    assert len(sv1) == 1 and max(abs(p - q) for p, q in zip(sv1[0]["bbox"], [300, 50, 200, 280])) < 1e-6, sv1
    print("  [PASS] V0 fuses the large Stage C fragment into the full-frame box (box shrinks); V1 keeps the full-frame box exact")
    v2 = build_variant("V2", M1, hr_p2, l2c, fr)
    mk = [d for d in v2 if d["category_id"] == 3]
    assert len(mk) == 1 and mk[0]["score"] >= 0.6 and abs(mk[0]["bbox"][0] - 1000) < 0.01, mk
    sv2 = [d for d in v2 if d["category_id"] == 6]
    assert sv2 == [{"category_id": 6, "score": 0.8, "bbox": [300.0, 50.0, 200.0, 280.0]}], sv2
    print("  [PASS] V2 keeps plain detections untouched, adds the tile-only marker and replaces the weaker overlapping plain box")
    # V0 must equal M1.0's in-line merge
    tp = []
    for c, t in zip(crops, tiles):
        tp.append((c, M1.tile_dets_for_merge(c, [d[0] for d in t[5]], [d[1] for d in t[5]], [d[2:6] for d in t[5]], l2c)))
    ref = hr_p2.merge_multiscale_tile_detections(tp, source_w=1920, source_h=1080, score_thresh=0.0, max_det_per_image=100)
    assert ref == v0
    print("  [PASS] V0 is identical to the M1.0 merge")
    bad = json.loads(json.dumps(fr))
    bad["tiles"][3][1] += 1
    try:
        build_variant("V0", M1, hr_p2, l2c, bad)
        raise AssertionError("tile mismatch not caught")
    except AssertionError as e:
        assert "mismatch" in str(e)
    print("  [PASS] cache/pipeline tile-window mismatch is refused")
    print(f"[M1.1-merge SELFTEST] 4/4 passed (script {SCRIPT_VERSION}, sha {sha256_file(Path(__file__).resolve())})")


def run_score(a) -> None:
    hr_p2, M1 = import_modules(a)
    import m1_compare_plain_vs_tiled as CMP
    cache_path = Path(a.cache).resolve()
    with gzip.open(cache_path, "rt", encoding="utf-8") as f:
        cache = json.load(f)
    meta = cache["meta"]
    if meta.get("schema") != "MARIVISION_M1_1_R0_TILE_CACHE_V1":
        raise SystemExit("[REFUSED] not an M1.1 tile cache.")
    if meta["coco_val_sha256"] != sha256_file(a.coco_val):
        raise SystemExit("[REFUSED] cache was built against a different VAL json.")
    l2c = {int(k): int(v) for k, v in meta["label_to_cat_id"].items()}
    plain_json = Path(a.plain_json or meta["reference_plain_json"])
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_tag = cache_path.name.replace("r0_tile_cache_", "").replace(".json.gz", "")
    summary = []
    for v in [x.strip() for x in a.variants.split(",")]:
        if v not in VARIANTS:
            raise SystemExit(f"[REFUSED] unknown variant {v}; pre-declared: {list(VARIANTS)}")
        preds = []
        for iid, fr in cache["frames"].items():
            for d in build_variant(v, M1, hr_p2, l2c, fr):
                preds.append({"image_id": int(iid), "category_id": int(d["category_id"]),
                              "bbox": [float(x) for x in d["bbox"]], "score": float(d["score"])})
        pj = out_dir / f"r0_val_predictions_{cache_tag}_{VARIANTS[v]}.json"
        if pj.exists():
            raise SystemExit(f"[REFUSED] {pj} exists.")
        pj.write_text(json.dumps(preds), encoding="utf-8")
        man = {"schema": "MARIVISION_FROZEN_R50_VAL_PREDS_MANIFEST_V1", "split": "val",
               "model_description": f"M1.1 {VARIANTS[v]}: frozen R0 tiled inference, merge variant from raw tile cache",
               "source_checkpoint_path": meta["source_checkpoint_path"],
               "source_checkpoint_sha256": meta["source_checkpoint_sha256"],
               "source_checkpoint_epoch": meta.get("source_checkpoint_epoch"),
               "export_command": " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
               "export_script_sha256": sha256_file(Path(__file__).resolve()),
               "tile_cache": str(cache_path), "tile_cache_sha256": sha256_file(cache_path),
               "box_format": "xywh_abs_pixels_native_frame", "category_id_base": 1, "score_threshold_applied": 0.0,
               "max_dets_per_image": 100, "coco_val_sha256": meta["coco_val_sha256"],
               "val_frame_stride": meta["val_frame_stride"], "val_frame_offset": meta["val_frame_offset"],
               "predictions_file_sha256": sha256_file(pj)}
        Path(str(pj) + ".manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
        tag = f"{cache_tag}_{VARIANTS[v]}"
        ns = argparse.Namespace(mode="compare", tag=tag, plain_json=str(plain_json), tiled_json=str(pj), zip=a.zip,
                                coco_val=a.coco_val, out_dir=a.report_dir, val_frame_stride=int(meta["val_frame_stride"]),
                                val_frame_offset=int(meta["val_frame_offset"]), allow_missing_pycocotools=False)
        print(f"\n######## {VARIANTS[v]} ########")
        CMP.run_compare(ns)
        rep = json.loads(Path(a.report_dir, f"m1_plain_vs_tiled_{tag}.json").read_text(encoding="utf-8"))
        s = rep["summary"]
        summary.append((VARIANTS[v], s["plain_native"], s["tiled_native"], rep["primary_endpoint"]))
    L = [f"# M1.1 merge variants - {cache_tag}", "",
         "| variant | macro AP30 plain | macro AP30 tiled | pooled delta | CI95 | videos > 0 | MARKER | SAILING_VESSEL | ANIMAL | SHIP | CA-AP50 | recall@0.3 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, p, t, pe in summary:
        pc = t["AP30_per_class"]
        L.append(f"| {name} | {p['AP30_mAP']} | {t['AP30_mAP']} | {pe['pooled_delta']} | [{pe['ci95_low']}, {pe['ci95_high']}] | "
                 f"{pe['n_videos_positive']}/{pe['n_videos']} | {pc['MARINE_MARKER']} | {pc['SAILING_VESSEL']} | {pc['ANIMAL']} | "
                 f"{pc['SHIP']} | {t['mAP_class_agnostic_50']} | {t['recall_1to1_iou30']} |")
    p0 = summary[0][1]["AP30_per_class"]
    L.append(f"| (plain per class) | | | | | | {p0['MARINE_MARKER']} | {p0['SAILING_VESSEL']} | {p0['ANIMAL']} | {p0['SHIP']} | "
             f"{summary[0][1]['mAP_class_agnostic_50']} | {summary[0][1]['recall_1to1_iou30']} |")
    best = max(summary, key=lambda r: (r[2]["AP30_mAP"], -list(VARIANTS.values()).index(r[0])))
    L += ["", f"Pre-declared development choice (highest pooled tiled macro AP30, ties -> lower number): **{best[0]}**"
          f" (only meaningful on the development cache, offset 5)."]
    out_md = Path(a.report_dir, f"m1_1_variants_{cache_tag}.md")
    out_md.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n" + "\n".join(L) + f"\n\n[M1.1-merge] wrote {out_md}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("selftest", "score"), required=True)
    ap.add_argument("--m1-dir", default="D:/MARIVISION/R0_frozen_baseline", help="Folder with m1_r0_tiled_inference.py")
    ap.add_argument("--cache")
    ap.add_argument("--variants", default="V0,V1,V2")
    ap.add_argument("--plain-json", default=None, help="Default: the reference plain export recorded in the cache")
    ap.add_argument("--zip", default="D:/MARIVISION/SeaClips.zip")
    ap.add_argument("--coco-val", default="D:/MARIVISION/seaclips-val.json")
    ap.add_argument("--out-dir", default="D:/MARIVISION/R0_frozen_baseline/m1_1_variant_preds")
    ap.add_argument("--report-dir", default="m1_reports")
    a = ap.parse_args()
    if "test" in Path(a.coco_val).name.lower():
        raise SystemExit("[SPLIT BLOCK] TEST is locked.")
    if a.mode == "selftest":
        run_selftest(a)
    else:
        if not a.cache:
            raise SystemExit("[REFUSED] --cache is required.")
        run_score(a)


if __name__ == "__main__":
    main()
