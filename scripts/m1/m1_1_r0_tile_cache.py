r"""m1_1_r0_tile_cache.py - M1.1 step 1 (GPU): cache the RAW per-tile R0 detections once, so every merge rule can be
scored later on the CPU without touching the GPU again. MARIVISION-HR-P2, Claude lead, 2026-10-27. VAL only.

Same tiles, same preprocessing and same provenance gate as m1_r0_tiled_inference.py (M1.0), whose tested functions it
imports (that file must sit in the same folder). Instead of merging, it stores every tile's raw R0 top-100 output
(score, R0 label, xyxy in TILE pixels) plus the tile window, in one gzip JSON per (checkpoint, offset).

Provenance tolerance: --parity-score-tol defaults to 2e-3 as in M1.0. The m1_parity_probe of 2026-10-27 showed R0
scores move by up to ~0.012 with GPU batch composition alone (batch 1/2/4, identical input tensors), so
the M1.1 commands pass 0.02 explicitly; the value used is recorded in the cache meta.

M1.1-cache.1 (2026-10-27) adds --no-reference-check for NEW checkpoints that have no plain VAL export (the M2 arms).
The full-frame path of this script was already verified against the plain exporter on three R0 caches (every frame,
max box dev <= 0.142 px), and every M2 comparison uses this same path for plain AND tiled, so the check is skipped
and the skip is recorded in the cache meta. Default behaviour is unchanged.

Run from D:\MARIVISION\R0_frozen_baseline with R0_MSDA_BACKEND=pytorch:
  python m1_1_r0_tile_cache.py --mode selftest
  python m1_1_r0_tile_cache.py --mode export --checkpoint ... --reference-plain-json ... --val-frame-offset 5 --tag ...
"""
from PIL import Image  # noqa: F401
import argparse
import gzip
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m1_r0_tiled_inference as M1  # noqa: E402

SCRIPT_VERSION = "M1.1-cache.1"
CACHE_SCHEMA = "MARIVISION_M1_1_R0_TILE_CACHE_V1"
M1_EXPECTED_SHA = "73044625fb5b477f857018ba1ace3c0dcfe32f08d2b27d0beccd8d333153a388"


def selftest(a) -> None:
    got = M1.sha256_file(Path(M1.__file__).resolve())
    assert got == M1_EXPECTED_SHA, f"m1_r0_tiled_inference.py sha {got} != expected M1.0 {M1_EXPECTED_SHA}"
    print(f"  [PASS] imports the installed M1.0 file (sha {got[:12]}...)")
    hr_p2 = M1.import_pipeline(a.pipeline_dir)
    for (w, h) in ((1920, 1080), (1200, 960), (945, 720)):
        crops = M1.build_tiles(hr_p2, w, h)
        rows = [[c.stage, c.x0_int, c.y0_int, c.x1_int, c.y1_int] for c in crops]
        again = [[c.stage, c.x0_int, c.y0_int, c.x1_int, c.y1_int] for c in M1.build_tiles(hr_p2, w, h)]
        assert rows == again and len(rows) == 17
    print("  [PASS] tile schedule deterministic and 17 tiles for all three SeaClips frame sizes (1920x1080, 1200x960, 945x720)")
    tmp = Path(a.selftest_tmp) / "selftest_cache.json.gz"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    obj = {"schema": CACHE_SCHEMA, "frames": {"7": {"W": 1920, "H": 1080, "tiles": [["STAGE_B_FULL_CONTEXT", 0, 0, 1920, 1080,
           [[0.5, 2, 1.25, 2.5, 3.75, 5.0]]]]}}}
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(obj, f)
    with gzip.open(tmp, "rt", encoding="utf-8") as f:
        assert json.load(f) == obj
    tmp.unlink()
    print("  [PASS] gzip JSON cache round trip is lossless")
    print(f"[M1.1-cache SELFTEST] 3/3 passed (script {SCRIPT_VERSION}, sha {M1.sha256_file(Path(__file__).resolve())})")


def export(a) -> None:
    import torch
    import torchvision.transforms.functional as TF
    if os.environ.get("R0_MSDA_BACKEND", "").lower() != "pytorch":
        raise SystemExit("[REFUSED] Run 'set R0_MSDA_BACKEND=pytorch' first.")
    no_ref = bool(a.no_reference_check)
    if no_ref and a.reference_plain_json:
        raise SystemExit("[REFUSED] use either --reference-plain-json or --no-reference-check, not both.")
    for n in ("checkpoint", "tag") + (() if no_ref else ("reference_plain_json",)):
        if not getattr(a, n):
            raise SystemExit(f"[REFUSED] --{n.replace('_', '-')} is required.")
    val_json = Path(a.coco_val).resolve()
    if "test" in val_json.name.lower() or "val" not in val_json.name.lower():
        raise SystemExit("[SPLIT BLOCK] --coco-val must be the VALIDATION json; TEST is locked.")
    ckpt_path = Path(a.checkpoint).resolve()
    out_dir = Path(a.out_dir).resolve()
    cache_path = out_dir / f"r0_tile_cache_{a.tag}_stride{a.val_frame_stride}_offset{a.val_frame_offset}.json.gz"
    if cache_path.exists():
        raise SystemExit(f"[REFUSED] {cache_path} exists; choose a new --tag.")
    ckpt_sha = M1.sha256_file(ckpt_path)
    ref_path = None if no_ref else Path(a.reference_plain_json).resolve()
    ref_man = {"source_checkpoint_sha256": ckpt_sha} if no_ref else \
        json.loads(Path(str(ref_path) + ".manifest.json").read_text(encoding="utf-8"))
    if ref_man.get("source_checkpoint_sha256", "").lower() != ckpt_sha:
        raise SystemExit("[REFUSED] reference export comes from a different checkpoint.")
    if not no_ref and ref_man.get("predictions_file_sha256", "").lower() != M1.sha256_file(ref_path):
        raise SystemExit("[REFUSED] reference predictions do not match their manifest.")
    if not no_ref and ref_man.get("coco_val_sha256", "").lower() != M1.sha256_file(val_json):
        raise SystemExit("[REFUSED] reference export used a different VAL json.")

    sys.path.insert(0, str(Path.cwd()))
    hr_p2 = M1.import_pipeline(a.pipeline_dir)
    from models.deformable_detr import build
    from datasets.seaclips_detection import SeaClipsDetection
    from r0_preflight import get_authoritative_r0_args
    from util.misc import NestedTensor

    val = json.loads(val_json.read_text(encoding="utf-8"))
    ids = M1.eval_subset_ids(val["images"], a.val_frame_stride, a.val_frame_offset)
    debug = bool(a.max_images)
    if debug:
        ids = ids[: a.max_images]
    print(f"[M1.1-cache] frames {len(ids)} (stride {a.val_frame_stride}, offset {a.val_frame_offset})"
          + ("  [DEBUG: nothing written]" if debug else ""))
    ref_by = defaultdict(list)
    for d in ([] if no_ref else json.loads(ref_path.read_text(encoding="utf-8"))):
        ref_by[int(d["image_id"])].append(d)
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    args = get_authoritative_r0_args()
    device = torch.device(args.device)
    model, _c, pp = build(args)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    ds = SeaClipsDetection(Path(a.zip), val_json, is_train=False, random_flip=False)
    l2c = {int(k): int(v) for k, v in dict(ds.label_to_cat_id).items()}

    frames, worst = {}, {"max_box_dev_px": 0.0, "max_unmatched": 0, "frames_failed": 0}
    t0 = time.time()
    with torch.inference_mode():
        for n, iid in enumerate(ids, 1):
            pil = ds._load_image(ds.images[iid]).convert("RGB")
            W, H = pil.size
            crops = M1.build_tiles(hr_p2, W, H)
            ts = [M1.preprocess_r0(pil.crop((c.x0_int, c.y0_int, c.x1_int, c.y1_int)), torch, TF) for c in crops]
            sizes = [[c.y1_int - c.y0_int, c.x1_int - c.x0_int] for c in crops]
            res = []
            for i in range(0, len(ts), a.batch_size):
                x = torch.stack(ts[i:i + a.batch_size]).to(device)
                m = torch.zeros(x.shape[0], x.shape[2], x.shape[3], dtype=torch.bool, device=device)
                res.extend(pp["bbox"](model(NestedTensor(x, m)),
                                      torch.tensor(sizes[i:i + a.batch_size], dtype=torch.float32, device=device)))
            tiles = []
            for c, r in zip(crops, res):
                sc, lb, bx = r["scores"].cpu().tolist(), r["labels"].cpu().tolist(), r["boxes"].cpu().tolist()
                if c.stage == M1.TILE_STAGES[0] and not no_ref:
                    pr = M1.parity_check(M1.plain_dets_native(sc, lb, bx, l2c), ref_by.get(iid, []),
                                         a.parity_score_tol, a.parity_box_tol)
                    worst["max_box_dev_px"] = max(worst["max_box_dev_px"], pr["max_box_dev_px"])
                    worst["max_unmatched"] = max(worst["max_unmatched"], pr["unmatched_mine"] + pr["unmatched_ref"])
                    if not pr["ok"]:
                        worst["frames_failed"] += 1
                        print(f"  [PARITY MISS] image {iid}: {pr}")
                tiles.append([c.stage, c.x0_int, c.y0_int, c.x1_int, c.y1_int,
                              [[round(float(s), 6), int(l), round(b[0], 3), round(b[1], 3), round(b[2], 3), round(b[3], 3)]
                               for s, l, b in zip(sc, lb, bx)]])
            frames[str(iid)] = {"W": W, "H": H, "tiles": tiles}
            if n % 25 == 0 or n == len(ids):
                el = time.time() - t0
                print(f"  {n}/{len(ids)} frames, {el:.0f}s (~{el / n * (len(ids) - n) / 60:.1f} min left) | parity: failed "
                      f"{worst['frames_failed']}, max box dev {worst['max_box_dev_px']:.3f}px")
            if worst["frames_failed"] and not debug:
                raise SystemExit("[M1.1-cache] PROVENANCE FAIL; nothing written. Send this output to Claude.")
    print("[M1.1-cache] PROVENANCE: SKIPPED (--no-reference-check, recorded in meta)" if no_ref else
          f"[M1.1-cache] PROVENANCE: {'PASS' if not worst['frames_failed'] else 'FAIL'} | {worst}")
    if debug:
        print("[M1.1-cache] debug run finished: nothing written.")
        return
    meta = {"schema": CACHE_SCHEMA, "script_version": SCRIPT_VERSION,
            "script_sha256": M1.sha256_file(Path(__file__).resolve()),
            "m1_module_sha256": M1.sha256_file(Path(M1.__file__).resolve()),
            "pipeline_py_sha256": M1.sha256_file(Path(hr_p2.__file__).resolve()),
            "source_checkpoint_path": str(ckpt_path), "source_checkpoint_sha256": ckpt_sha,
            "source_checkpoint_epoch": ckpt.get("epoch"),
            "coco_val_sha256": M1.sha256_file(val_json), "val_frame_stride": a.val_frame_stride,
            "val_frame_offset": a.val_frame_offset, "n_frames": len(ids),
            "reference_plain_json": None if no_ref else str(ref_path),
            "reference_plain_json_sha256": None if no_ref else M1.sha256_file(ref_path),
            "label_to_cat_id": {str(k): v for k, v in sorted(l2c.items())},
            "box_format": "per tile: [score, r0_label, x1, y1, x2, y2] in TILE pixels (tile = native crop, no scaling)",
            "provenance": ({"status": "SKIPPED_NO_REFERENCE_EXPORT", "reason": "new checkpoint without a plain VAL export; "
                            "full-frame path verified on three R0 caches"} if no_ref else
                           dict(worst, status="PASS", score_tol=a.parity_score_tol, box_tol_px=a.parity_box_tol)),
            "batch_size": a.batch_size, "torch_version": torch.__version__, "device": str(device),
            "command": " ".join([Path(sys.argv[0]).name] + sys.argv[1:])}
    out_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache_path, "wt", encoding="utf-8") as f:
        json.dump({"meta": meta, "frames": frames}, f)
    print(f"[M1.1-cache] wrote {cache_path}\n     sha256 {M1.sha256_file(cache_path)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("selftest", "export"), required=True)
    ap.add_argument("--pipeline-dir", default="D:/MARIVISION/H0_workbench/eval_tools")
    ap.add_argument("--checkpoint")
    ap.add_argument("--tag")
    ap.add_argument("--reference-plain-json")
    ap.add_argument("--out-dir", default="D:/MARIVISION/R0_frozen_baseline/m1_1_tile_cache")
    ap.add_argument("--zip", default="D:/MARIVISION/SeaClips.zip")
    ap.add_argument("--coco-val", default="D:/MARIVISION/seaclips-val.json")
    ap.add_argument("--val-frame-stride", type=int, default=10)
    ap.add_argument("--val-frame-offset", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--parity-box-tol", type=float, default=0.5)
    ap.add_argument("--parity-score-tol", type=float, default=2e-3)
    ap.add_argument("--no-reference-check", action="store_true",
                    help="New checkpoints without a plain VAL export (M2 arms): skip the parity gate, recorded in meta")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--selftest-tmp", default="m1_1_selftest_tmp")
    a = ap.parse_args()
    selftest(a) if a.mode == "selftest" else export(a)


if __name__ == "__main__":
    main()
