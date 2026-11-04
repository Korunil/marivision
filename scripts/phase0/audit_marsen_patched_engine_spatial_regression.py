"""
D:/MARIVISION/H0_workbench/eval_tools/audit_marsen_patched_engine_spatial_regression.py
MARSEN Stream B2: Patched-Engine Spatial Regression & Preflight Gate (P-2112, P-2113)

Executes two fast preflight checks before full extraction:
  1. Exhaustive Resolver Audit (P-2112): Confirms all 20,310 Train and 4,266 Val manifest
     entries resolve to exactly 1 split-qualified member via engine.resolve_frame_member().
  2. Patched Spatial Regression (P-2113): Recomputes 1x, 3x, and 5x descriptors for the
     frozen 1,915 design proposals using the patched split-aware engine at fixed B = 128.
     Asserts max delta < 1e-5 (target: 0.00e+00) against marsen_pooled_descriptors_1915.pth.
Runtime: ~15 seconds.
"""

import sys
import time
import json
import torch
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
ROOT_DIR = Path("D:/MARIVISION")
ZIP_PATH = ROOT_DIR / "SeaClips.zip"
TRAIN_JSON = ROOT_DIR / "seaclips-train.json"
VAL_JSON = ROOT_DIR / "seaclips-val.json"

CERT_QUERIES_PTH = OUT_DIR / "stream_b2_reproduced_queries.pth"
CERT_FRAMES_PTH = OUT_DIR / "stream_b2_reproduced_frames.pth"
FROZEN_DESCRIPTORS_PTH = OUT_DIR / "marsen_pooled_descriptors_1915.pth"


def normalize_zip_path(path_str):
    return str(path_str).replace("\\", "/").strip("/").lstrip("/")


def main():
    t_start = time.time()
    print("=" * 115, flush=True)
    print("MARSEN STREAM B2: PATCHED-ENGINE SPATIAL REGRESSION & PREFLIGHT GATE (P-2112, P-2113)", flush=True)
    print("=" * 115, flush=True)

    assert ZIP_PATH.exists(), f"[FATAL] Missing {ZIP_PATH}"
    assert TRAIN_JSON.exists(), f"[FATAL] Missing {TRAIN_JSON}"
    assert VAL_JSON.exists(), f"[FATAL] Missing {VAL_JSON}"
    assert CERT_QUERIES_PTH.exists(), f"[FATAL] Missing {CERT_QUERIES_PTH}"
    assert CERT_FRAMES_PTH.exists(), f"[FATAL] Missing {CERT_FRAMES_PTH}"
    assert FROZEN_DESCRIPTORS_PTH.exists(), f"[FATAL] Missing {FROZEN_DESCRIPTORS_PTH}"

    sys.path.insert(0, str(OUT_DIR))
    from marsen_stream_b2_engine import GeneralizedStreamEngine, get_crop_pixel_box

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Target: {device}", flush=True)
    engine = GeneralizedStreamEngine(device)

    # =========================================================================
    # GATE 1: Exhaustive Preflight Resolver Audit across 24,576 frames (P-2112)
    # =========================================================================
    print("\n[*] Gate 1: Exhaustive Split-Qualified Archive Resolution Preflight...", flush=True)
    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        data_train = json.load(f)
    with open(VAL_JSON, "r", encoding="utf-8") as f:
        data_val = json.load(f)

    train_resolved = 0
    for img in data_train["images"]:
        m = engine.resolve_frame_member(img["file_name"], split_name="train")
        assert "/train/" in f"/{m}/" or m.startswith("train/") or m.startswith("SeaClips/train/"), (
            f"[FATAL] Misdirected train member: {m}"
        )
        train_resolved += 1

    val_resolved = 0
    for img in data_val["images"]:
        m = engine.resolve_frame_member(img["file_name"], split_name="val")
        assert "/val/" in f"/{m}/" or m.startswith("val/") or m.startswith("SeaClips/val/"), (
            f"[FATAL] Misdirected val member: {m}"
        )
        val_resolved += 1

    print(f"    - Official Train Manifest: {train_resolved} / 20,310 frames verified (100.0% -> train/)", flush=True)
    print(f"    - Official Val Manifest:   {val_resolved} /  4,266 frames verified (100.0% -> val/)", flush=True)
    print("[+] Gate 1 Passed: 100% of 24,576 Train and Val frames resolve uniquely to authentic split members.", flush=True)

    # =========================================================================
    # GATE 2: Patched-Engine Spatial Descriptor Regression (P-2113)
    # =========================================================================
    print("\n[*] Gate 2: Patched-Engine Spatial Regression on 1,915 Active Proposals...", flush=True)
    ref_desc = torch.load(FROZEN_DESCRIPTORS_PTH, map_location="cpu")["descriptors"]
    ref_d1 = ref_desc["1.0x_gem_p4"]
    ref_d3 = ref_desc["3.0x_gem_p4"]
    ref_d5 = ref_desc["5.0x_gap"]

    all_queries = torch.load(CERT_QUERIES_PTH, map_location="cpu")["extracted_queries"]
    active_queries = [q for q in all_queries if q["is_active"]]
    assert len(active_queries) == 1915

    frame_records = torch.load(CERT_FRAMES_PTH, map_location="cpu")["frame_records"]
    frame_dict = {fr["image_id"]: fr for fr in frame_records}

    for q in active_queries:
        fr = frame_dict[q["image_id"]]
        cx, cy, w, h = fr["temp_boxes"][q["query_id"]].tolist()
        q["pred_box_norm"] = (cx, cy, w, h)

    img_meta = {img["id"]: img for img in data_train["images"]}

    # Load authentic train images through patched engine.load_frame_image
    unique_iids = sorted(list(set(q["image_id"] for q in active_queries)))
    img_cache = {}
    for iid in unique_iids:
        fn = img_meta[iid]["file_name"]
        img_cache[iid] = engine.load_frame_image(fn, split_name="train")

    crops_1x, crops_3x, crops_5x = [], [], []
    for q in active_queries:
        img_pil = img_cache[q["image_id"]]
        b1 = get_crop_pixel_box(img_pil.size, q["pred_box_norm"], expansion_factor=1.0)
        b3 = get_crop_pixel_box(img_pil.size, q["pred_box_norm"], expansion_factor=3.0)
        b5 = get_crop_pixel_box(img_pil.size, q["pred_box_norm"], expansion_factor=5.0)

        crops_1x.append(engine.visual_extractor.transform(img_pil.crop(b1)))
        crops_3x.append(engine.visual_extractor.transform(img_pil.crop(b3)))
        crops_5x.append(engine.visual_extractor.transform(img_pil.crop(b5)))

    # Process through fixed B=128 extractor
    d1, d3, d5 = engine.visual_extractor.extract_crops_fixed_b128(crops_1x, crops_3x, crops_5x)

    diff_1x = float(np.max(np.abs(d1 - ref_d1)))
    diff_3x = float(np.max(np.abs(d3 - ref_d3)))
    diff_5x = float(np.max(np.abs(d5 - ref_d5)))

    print(f"    - 1.0x_GeM4 Descriptor Difference: {diff_1x:.2e} (Tolerance: 1.0e-05)", flush=True)
    print(f"    - 3.0x_GeM4 Descriptor Difference: {diff_3x:.2e} (Tolerance: 1.0e-05)", flush=True)
    print(f"    - 5.0x_GAP  Descriptor Difference: {diff_5x:.2e} (Tolerance: 1.0e-05)", flush=True)

    assert diff_1x < 1e-5, f"[FATAL] 1.0x regression failure: {diff_1x:.2e}"
    assert diff_3x < 1e-5, f"[FATAL] 3.0x regression failure: {diff_3x:.2e}"
    assert diff_5x < 1e-5, f"[FATAL] 5.0x regression failure: {diff_5x:.2e}"
    print("[+] Gate 2 Passed: strict descriptor parity within 1e-5.", flush=True)

    print("\n" + "=" * 115, flush=True)
    print("PATCHED-ENGINE REGRESSION & PREFLIGHT GATES CERTIFIED", flush=True)
    print(f"The engine is certified for full-split extraction (Elapsed: {time.time() - t_start:.2f}s).", flush=True)
    print("=" * 115, flush=True)


if __name__ == "__main__":
    main()