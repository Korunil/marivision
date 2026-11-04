r"""m1_compare_plain_vs_tiled.py - M1 scoring: plain R0 vs tiled R0 (same checkpoint) on the evaluated VAL subset.
MARIVISION-HR-P2, Claude lead, 2026-10-27. CPU only, VAL only, reads two prediction files, writes a report pair.

Uses the HR-P2 runner's own evaluator (train_marivision_hr_p2_arms.py, V13.6) so the numbers are directly comparable
with every earlier report: same 432-frame subset rule, same AP code, pycocotools parity (fail-fast), same forensics,
same 9-video paired bootstrap (hr_p2.paired_video_bootstrap_delta_metric, 5000 resamples, fixed seed).

PRE-DECLARED M1 ENDPOINT (written 2026-10-27, before any tiled prediction exists):
  Primary:   delta macro mAP@0.30 (AP30_mAP, the SeaClips paper's metric), tiled minus plain, at R0's native protocol
             (no score cut, top-100 per frame), same checkpoint, 432 VAL frames (stride 10, offset 0).
  Success:   paired-video bootstrap 95% CI lower bound > 0 AND delta > 0 in at least 6 of 9 videos.
  Secondary (reported, not gating): AP50_mAP, class-agnostic AP50, 1-to-1 recall@0.3/0.5, per-class AP30,
             small-object recall bins, the same comparisons at the 0.05 operating point, MARKER->ANIMAL confusion.
  The checkpoint is fixed (no re-selection), tiling constants are the pipeline defaults (no tuning).

Run from D:\MARIVISION\H0_workbench\eval_tools:
  python -W error m1_compare_plain_vs_tiled.py --mode selftest
  python -W error m1_compare_plain_vs_tiled.py --mode compare --tag epoch04 ^
      --plain-json D:\MARIVISION\R0_frozen_baseline\val_exports\r0_val_predictions_epoch04.json ^
      --tiled-json D:\MARIVISION\R0_frozen_baseline\m1_tiled_exports\r0_val_predictions_epoch04_tiled.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_VERSION = "M1C.0"
BOOT_METRICS = ("AP30_mAP", "AP50_mAP", "mAP_class_agnostic_50", "overall_gt_recall_30", "overall_gt_recall_50")
PRIMARY_MIN_POSITIVE_VIDEOS = 6
CLASSES = ("ANIMAL", "BOAT", "MARINE_MARKER", "OBJECT", "SAILING_VESSEL", "SHIP")


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pooled_value(ov, m_name):
    """Same mapping as the runner's _pooled_metric_val."""
    return {
        "AP50_mAP": lambda: float(ov.get("AP50", {}).get("mAP", 0.0)),
        "AP30_mAP": lambda: float(ov.get("AP30_mAP", 0.0)),
        "mAP_class_agnostic_50": lambda: float(ov.get("mAP_class_agnostic_50", 0.0)),
        "overall_gt_recall_50": lambda: float(ov.get("overall_1to1_gt_recall_iou50", 0.0)),
        "overall_gt_recall_30": lambda: float(ov.get("overall_1to1_gt_recall_iou30", 0.0)),
    }[m_name]()


def paired_bootstrap(hr_p2, pv_base, pv_cand, pooled_base, pooled_cand):
    """Same call and the same L2VO additions as the runner's _call_paired_bootstrap_safe."""
    res = dict(hr_p2.paired_video_bootstrap_delta_metric(pv_base, pv_cand, n_boot=5000))
    vids = sorted(set(pv_base) & set(pv_cand))
    diffs = np.array([float(pv_cand[v]) - float(pv_base[v]) for v in vids], dtype=np.float64)
    if len(vids) > 2:
        top2 = np.argsort(-diffs)[:2]
        rem = np.delete(diffs, top2)
        res["top2_video_ids"] = [f"v{int(vids[i]):02d}" for i in top2]
        res["mean_delta_without_top2_videos"] = round(float(np.mean(rem)), 5)
        res["n_videos_without_top2_gt_zero"] = int(np.sum(rem > 0.0))
    res["pooled_aggregate_delta"] = round(float(pooled_cand) - float(pooled_base), 5)
    res["equal_weight_per_video_mean_delta"] = round(float(np.mean(diffs)) if len(diffs) else 0.0, 5)
    res["n_videos_delta_gt_zero_strict"] = int(np.sum(diffs > 0.0))
    return res


def primary_verdict(boot_ap30):
    ci_low = float(boot_ap30.get("ci95_low", float("nan")))
    n_pos = int(boot_ap30.get("n_videos_delta_gt_zero_strict", 0))
    ok = (ci_low > 0.0) and (n_pos >= PRIMARY_MIN_POSITIVE_VIDEOS)
    return {"endpoint": "AP30_mAP tiled - plain, native protocol (no cut, top-100)",
            "ci95_low": ci_low, "ci95_high": float(boot_ap30.get("ci95_high", float("nan"))),
            "pooled_delta": boot_ap30.get("pooled_aggregate_delta"), "n_videos_positive": n_pos,
            "n_videos": int(boot_ap30.get("n_videos", 0)),
            "rule": f"ci95_low > 0 AND >= {PRIMARY_MIN_POSITIVE_VIDEOS} videos positive",
            "status": "SUCCESS" if ok else "NOT_MET"}


def run_selftest() -> None:
    import marivision_hr_p2_pipeline as hr_p2
    pv_b = {v: 0.30 for v in range(1, 10)}
    pv_c = {v: 0.30 + 0.05 + 0.01 * (v % 3) for v in range(1, 10)}
    b = paired_bootstrap(hr_p2, pv_b, pv_c, 0.30, 0.36)
    assert b["n_videos_delta_gt_zero_strict"] == 9 and b["ci95_low"] > 0, b
    assert primary_verdict(b)["status"] == "SUCCESS"
    pv_c2 = {v: (0.40 if v <= 5 else 0.25) for v in range(1, 10)}
    b2 = paired_bootstrap(hr_p2, pv_b, pv_c2, 0.30, 0.33)
    assert primary_verdict(b2)["status"] == "NOT_MET", b2
    print("  [PASS] paired bootstrap + pre-declared verdict: consistent gain -> SUCCESS; 5/9 mixed -> NOT_MET")
    ov = {"AP50": {"mAP": 0.2}, "AP30_mAP": 0.3, "mAP_class_agnostic_50": 0.25,
          "overall_1to1_gt_recall_iou50": 0.4, "overall_1to1_gt_recall_iou30": 0.5}
    assert [pooled_value(ov, m) for m in BOOT_METRICS] == [0.3, 0.2, 0.25, 0.5, 0.4]
    print("  [PASS] pooled-metric mapping identical to the runner")
    print(f"[M1C SELFTEST] 2/2 passed (script {SCRIPT_VERSION}, sha256 {sha256_file(Path(__file__).resolve())})")


def summarize(ov):
    return {"AP30_mAP": ov.get("AP30_mAP"), "AP50_mAP": ov.get("AP50", {}).get("mAP"),
            "AP30_gt_weighted_mAP": ov.get("AP30_gt_weighted_mAP"), "AP50_gt_weighted_mAP": ov.get("AP50_gt_weighted_mAP"),
            "mAP_class_agnostic_50": ov.get("mAP_class_agnostic_50"), "AP_S_at_50_mAP": ov.get("AP_S_at_50_mAP"),
            "recall_1to1_iou30": ov.get("overall_1to1_gt_recall_iou30"), "recall_1to1_iou50": ov.get("overall_1to1_gt_recall_iou50"),
            "fp_per_frame_1to1_iou50": ov.get("fp_per_frame_1to1_iou50"),
            "AP30_per_class": {c: ov.get("AP30_per_class", {}).get(c) for c in CLASSES},
            "recall_by_area_bin_iou30": {k: v.get("recall_iou30", v.get("n_tp_iou30", 0) / max(1, v.get("n_gt", 0)))
                                         for k, v in ov.get("recall_by_effective_area_bin_at_640", {}).items()}}


def run_compare(a) -> None:
    import marivision_hr_p2_pipeline as hr_p2
    import train_marivision_hr_p2_arms as R
    if "test" in Path(a.coco_val).name.lower() or "val" not in Path(a.coco_val).name.lower():
        raise SystemExit("[SPLIT BLOCK] --coco-val must be the VALIDATION json; TEST is locked.")
    out_dir = Path(a.out_dir)
    stem = out_dir / f"m1_plain_vs_tiled_{a.tag}"
    if Path(str(stem) + ".json").exists():
        raise SystemExit(f"[REFUSED] {stem}.json exists; choose a new --tag.")

    val_idx = R.SeaClipsZipNativeIndex(Path(a.zip), Path(a.coco_val), split_hint="val")
    by_video = defaultdict(list)
    for iid, im in val_idx.images.items():
        by_video[int(im.get("video_id", 0))].append(im)
    off = int(a.val_frame_offset) % max(1, int(a.val_frame_stride))
    eval_ids = []
    for _, lst in sorted(by_video.items()):
        lst.sort(key=lambda x: (int(x.get("frame_id", x["id"])), int(x["id"])))
        for i, im in enumerate(lst):
            if a.val_frame_stride <= 1 or i % a.val_frame_stride == off:
                eval_ids.append(int(im["id"]))
    eval_images = {i: val_idx.images[i] for i in eval_ids}
    eval_anns = {i: val_idx.anns_by_image.get(i, []) for i in eval_ids}
    n_gt = sum(len(v) for v in eval_anns.values())
    print(f"[M1C {SCRIPT_VERSION}] evaluated VAL frames {len(eval_ids)}, GT {n_gt}")

    loaded, variants, per_video, parity, forensic = {}, {}, {}, {}, {}
    for name, path in (("plain", a.plain_json), ("tiled", a.tiled_json)):
        preds_all, info = R.load_frozen_r50_val_predictions_with_manifest(Path(path), Path(a.coco_val), eval_images, score_thresh=0.0)
        if info["n_eval_images_with_predictions"] != len(eval_ids):
            raise SystemExit(f"[REFUSED] {name}: predictions cover {info['n_eval_images_with_predictions']}/{len(eval_ids)} evaluated frames.")
        loaded[name] = info
        for op, preds in (("native", preds_all), ("s005", [d for d in preds_all if float(d["score"]) >= 0.05])):
            k = f"{name}_{op}"
            ov, pv = R.evaluate_predictions_dict_and_per_video(eval_images, eval_anns, preds)
            variants[k], per_video[k] = ov, pv
            parity[k] = R.verify_against_pycocotools_if_available(eval_images, eval_anns, preds, ov, fail_fast=True,
                                                                  require_pycocotools=not a.allow_missing_pycocotools)
            if op == "s005":
                forensic[name] = R.diagnose_tiny_le_4px2_and_ship_confusion(eval_images, eval_anns, preds, preds_all)
    m_plain, m_tiled = loaded["plain"]["manifest"], loaded["tiled"]["manifest"]
    if m_plain["source_checkpoint_sha256"] != m_tiled["source_checkpoint_sha256"]:
        raise SystemExit("[REFUSED] plain and tiled files come from different checkpoints; M1 is a same-weights comparison.")

    boots = {}
    for op in ("native", "s005"):
        boots[op] = {m: paired_bootstrap(hr_p2, per_video[f"plain_{op}"][m], per_video[f"tiled_{op}"][m],
                                         pooled_value(variants[f"plain_{op}"], m), pooled_value(variants[f"tiled_{op}"], m))
                     for m in BOOT_METRICS}
    verdict = primary_verdict(boots["native"]["AP30_mAP"])
    parity_ok = all(str(v.get("parity_status")) == "PARITY_PASSED" for v in parity.values())
    report = {
        "script_version": SCRIPT_VERSION, "script_sha256": sha256_file(Path(__file__).resolve()),
        "runner_sha256": sha256_file(Path(R.__file__).resolve()), "pipeline_sha256": sha256_file(Path(hr_p2.__file__).resolve()),
        "tag": a.tag, "n_eval_images": len(eval_ids), "n_eval_gt": n_gt,
        "val_frame_stride": a.val_frame_stride, "val_frame_offset": off,
        "checkpoint_sha256": m_plain["source_checkpoint_sha256"], "checkpoint_epoch": m_plain.get("source_checkpoint_epoch"),
        "inputs": {k: {"path": v["source_path"], "sha256": v["source_sha256"]} for k, v in loaded.items()},
        "pycocotools_parity_all_passed": parity_ok,
        "primary_endpoint": verdict,
        "summary": {k: summarize(v) for k, v in variants.items()},
        "paired_video_bootstrap_tiled_minus_plain": boots,
        "forensics_s005": forensic,
        "pycocotools_parity": parity,
        "metrics_full": variants,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(str(stem) + ".json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")

    L = [f"# M1 plain vs tiled R0 - {a.tag}", "",
         f"{len(eval_ids)} VAL frames, {n_gt} GT, checkpoint epoch {report['checkpoint_epoch']} (sha {report['checkpoint_sha256'][:12]}), "
         f"pycocotools parity: {'PASSED' if parity_ok else 'NOT ALL PASSED'}", "",
         f"**Primary endpoint ({verdict['endpoint']}): {verdict['status']}** - pooled delta {verdict['pooled_delta']}, "
         f"95% CI [{verdict['ci95_low']}, {verdict['ci95_high']}], {verdict['n_videos_positive']}/{verdict['n_videos']} videos positive "
         f"(rule: {verdict['rule']})", "",
         "| metric | plain native | tiled native | plain s>=0.05 | tiled s>=0.05 |", "|---|---|---|---|---|"]
    sm = report["summary"]
    for m in ("AP30_mAP", "AP50_mAP", "AP30_gt_weighted_mAP", "AP50_gt_weighted_mAP", "mAP_class_agnostic_50", "AP_S_at_50_mAP",
              "recall_1to1_iou30", "recall_1to1_iou50", "fp_per_frame_1to1_iou50"):
        L.append(f"| {m} | " + " | ".join(str(sm[k][m]) for k in ("plain_native", "tiled_native", "plain_s005", "tiled_s005")) + " |")
    L += ["", "| AP30 per class | plain native | tiled native |", "|---|---|---|"]
    for c in CLASSES:
        L.append(f"| {c} | {sm['plain_native']['AP30_per_class'][c]} | {sm['tiled_native']['AP30_per_class'][c]} |")
    L += ["", "| recall@0.3 by area bin (640-scale) | plain native | tiled native |", "|---|---|---|"]
    for k in sm["plain_native"]["recall_by_area_bin_iou30"]:
        pv_, tv_ = sm["plain_native"]["recall_by_area_bin_iou30"][k], sm["tiled_native"]["recall_by_area_bin_iou30"].get(k)
        L.append(f"| {k} | {round(pv_, 4)} | {round(tv_, 4) if tv_ is not None else '-'} |")
    L += ["", "| bootstrap tiled - plain | op | pooled delta | CI95 | videos > 0 | L2VO mean |", "|---|---|---|---|---|---|"]
    for op in ("native", "s005"):
        for m in BOOT_METRICS:
            b = boots[op][m]
            L.append(f"| {m} | {op} | {b.get('pooled_aggregate_delta')} | [{b.get('ci95_low')}, {b.get('ci95_high')}] | "
                     f"{b.get('n_videos_delta_gt_zero_strict')}/{b.get('n_videos')} | {b.get('mean_delta_without_top2_videos')} |")
    L += ["", "Per-video AP30 deltas (native): " + json.dumps(boots["native"]["AP30_mAP"].get("per_video_deltas"))]
    Path(str(stem) + "_tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n[M1C] wrote {stem}.json and {stem}_tables.md")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("selftest", "compare"), required=True)
    ap.add_argument("--tag")
    ap.add_argument("--plain-json")
    ap.add_argument("--tiled-json")
    ap.add_argument("--zip", default="D:/MARIVISION/SeaClips.zip")
    ap.add_argument("--coco-val", default="D:/MARIVISION/seaclips-val.json")
    ap.add_argument("--out-dir", default="m1_reports")
    ap.add_argument("--val-frame-stride", type=int, default=10)
    ap.add_argument("--val-frame-offset", type=int, default=0)
    ap.add_argument("--allow-missing-pycocotools", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if a.mode == "selftest":
        run_selftest()
    else:
        for n in ("tag", "plain_json", "tiled_json"):
            if not getattr(a, n):
                raise SystemExit(f"[REFUSED] --{n.replace('_', '-')} is required in compare mode.")
        run_compare(a)


if __name__ == "__main__":
    main()
