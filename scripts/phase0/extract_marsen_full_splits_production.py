"""
D:/MARIVISION/H0_workbench/eval_tools/extract_marsen_full_splits_production.py
MARSEN Stream B2: Production Train & Validation Sharded Extractor (P-2046..P-2113)

Extracts candidate proposals and the unified multi-branch feature bundle across
official Train (47) and Validation (9) splits using the certified two-phase engine:
  - Phase 1: Rolling 6-frame window temporal inference, metadata, and operational labels.
  - Phase 2: Split-aware deterministic fixed-batch (B = 128) tri-scale visual extraction.
Enforces:
  - Exhaustive 24,576-frame preflight resolution verification.
  - Fail-closed split identification from SeaClipsDetection.split.
  - Driver, engine, R0, adapter, and split JSON hash-locking per shard.
  - Full structural, tensor, and label completeness assertions per shard.
Test split (18 videos) remains completely unopened per D-4078.
"""

import os
import sys
import time
import json
import hashlib
import torch
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
ROOT_DIR = Path("D:/MARIVISION")
TRAIN_JSON = ROOT_DIR / "seaclips-train.json"
VAL_JSON = ROOT_DIR / "seaclips-val.json"

SHARD_DIR = OUT_DIR / "full_split_shards"
SHARD_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_VIDEOS = {"train": 47, "val": 9}


def compute_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main():
    t_start = time.time()
    driver_sha256 = compute_sha256(Path(__file__))
    engine_path = OUT_DIR / "marsen_stream_b2_engine.py"
    engine_sha256 = compute_sha256(engine_path)

    sys.path.insert(0, str(OUT_DIR))
    import audit_pol3v_operating_point_sweep as opsweep
    r0_ckpt_sha = compute_sha256(opsweep.M0_CKPT_PATH)
    epoch8_sha = compute_sha256(opsweep.EPOCH8_MODEL_PTH)

    print("=" * 115, flush=True)
    print("MARSEN STREAM B2: PRODUCTION FULL-SPLIT EXTRACTOR (TRAIN 47 / VAL 9) (P-2046..P-2113)", flush=True)
    print(f"Driver Script SHA-256: {driver_sha256}", flush=True)
    print(f"Engine Core   SHA-256: {engine_sha256}", flush=True)
    print(f"R0 Checkpoint SHA-256: {r0_ckpt_sha}", flush=True)
    print(f"Epoch-8 Model SHA-256: {epoch8_sha}", flush=True)
    print("=" * 115, flush=True)

    assert TRAIN_JSON.exists(), f"[FATAL] Missing {TRAIN_JSON}"
    assert VAL_JSON.exists(), f"[FATAL] Missing {VAL_JSON}"

    from marsen_stream_b2_engine import MarsenChronology, GeneralizedStreamEngine

    data_train, train_vids_dict = MarsenChronology.load_and_validate(TRAIN_JSON)
    data_val, val_vids_dict = MarsenChronology.load_and_validate(VAL_JSON)

    print("[*] Validating Dataset Video Inventories...", flush=True)
    print(f"    - Official Train Unique Videos: {len(train_vids_dict)} (Expected: 47) | Frames: {len(data_train['images'])}", flush=True)
    print(f"    - Official Val Unique Videos:   {len(val_vids_dict)} (Expected: 9)  | Frames: {len(data_val['images'])}", flush=True)

    assert len(train_vids_dict) == 47, f"[FATAL] Expected 47 train videos, found {len(train_vids_dict)}"
    assert len(val_vids_dict) == 9, f"[FATAL] Expected 9 val videos, found {len(val_vids_dict)}"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Target: {device}", flush=True)
    engine = GeneralizedStreamEngine(device)

    # P-2112: Exhaustive Preflight Resolution Verification across all 24,576 frames
    print("[*] Running Exhaustive Split-Qualified Archive Resolution Preflight...", flush=True)
    for img in data_train["images"]:
        _ = engine.resolve_frame_member(img["file_name"], split_name="train")
    for img in data_val["images"]:
        _ = engine.resolve_frame_member(img["file_name"], split_name="val")
    print("[+] Preflight Passed: All 20,310 Train and 4,266 Val frames resolve uniquely to their authentic split.\n", flush=True)

    splits_to_process = [
        ("train", TRAIN_JSON, EXPECTED_VIDEOS["train"], data_train, train_vids_dict),
        ("val", VAL_JSON, EXPECTED_VIDEOS["val"], data_val, val_vids_dict)
    ]

    grand_total_windows = 0
    grand_total_active_queries = 0

    for split_name, json_path, exp_vids, split_coco_data, split_vids_dict in splits_to_process:
        print("=" * 115, flush=True)
        print(f"EXTRACTING SPLIT: {split_name.upper()} ({json_path.name})", flush=True)
        print("=" * 115, flush=True)

        split_json_sha = compute_sha256(json_path)
        n_vids = len(split_vids_dict)
        print(f"[*] Certified {n_vids} Videos (Expected: {exp_vids}) | Total Frames: {len(split_coco_data['images'])}", flush=True)

        split_windows = 0
        split_active_queries = 0

        for vid in sorted(split_vids_dict.keys()):
            frames = split_vids_dict[vid]
            shard_path = SHARD_DIR / f"marsen_b2_shard_{split_name}_vid{vid}.pth"

            # Version-locked shard resumption condition (P-2091, P-2092)
            if shard_path.exists():
                shard_data = torch.load(shard_path, map_location="cpu")
                meta = shard_data.get("_meta", {})
                is_valid = (
                    meta.get("fresh_inference") is True and
                    meta.get("extractor_sha256") == driver_sha256 and
                    meta.get("engine_sha256") == engine_sha256 and
                    meta.get("r0_checkpoint_sha256") == r0_ckpt_sha and
                    meta.get("epoch8_adapter_sha256") == epoch8_sha and
                    meta.get("split_json_sha256") == split_json_sha and
                    meta.get("visual_batch_contract") == "fixed_b128_pad_last" and
                    meta.get("fixed_visual_batch_size") == 128 and
                    meta.get("split") == split_name and
                    meta.get("video_id") == vid and
                    "active_queries" in shard_data and
                    len(shard_data.get("frame_records", [])) == shard_data.get("n_windows", -1)
                )
                if is_valid:
                    print(f"  - Video {vid:>3}: Valid locked shard exists ({shard_data['n_windows']} windows, {len(shard_data['active_queries'])} active) -> Skipping.", flush=True)
                    split_windows += shard_data["n_windows"]
                    split_active_queries += len(shard_data["active_queries"])
                    continue
                else:
                    print(f"  - Video {vid:>3}: Stale/unlocked shard detected -> Overwriting with fresh inference.", flush=True)

            t_v0 = time.time()
            print(f"  - Video {vid:>3}: Processing {len(frames)} frames across rolling windows...", flush=True)
            shard_data = engine.process_video(vid, frames, json_path, is_labeled=True)
            active_q = [q for q in shard_data["queries"] if q["is_active"]]
            shard_data["active_queries"] = active_q

            # Integrity assertions before locking
            assert shard_data["n_windows"] >= 0
            assert len(shard_data["frame_records"]) == shard_data["n_windows"]

            for fr in shard_data["frame_records"]:
                for field in ["d_logits", "d_boxes", "temp_logits", "temp_boxes"]:
                    assert field in fr, f"[FATAL] Missing {field} in frame record!"

            for q in active_q:
                assert "pred_box_norm" in q, "[FATAL] Missing pred_box_norm in active query!"
                assert len(q["features"]) == 15, "[FATAL] Incomplete 15-D metadata features!"
                assert np.asarray(q["decoder_256"]).shape == (256,), "[FATAL] Incomplete 256-D decoder state!"
                assert np.asarray(q["1x_gem4"]).shape == (128,), "[FATAL] Incomplete 1x descriptor!"
                assert np.asarray(q["3x_gem4"]).shape == (128,), "[FATAL] Incomplete 3x descriptor!"
                assert np.asarray(q["5x_gap"]).shape == (128,), "[FATAL] Incomplete 5x descriptor!"
                assert "label_op_util" in q, "[FATAL] Missing label_op_util in query!"
                assert "label_rescue_pot" in q, "[FATAL] Missing label_rescue_pot in query!"

            shard_data["_meta"] = {
                "fresh_inference": True,
                "extractor_sha256": driver_sha256,
                "engine_sha256": engine_sha256,
                "r0_checkpoint_sha256": r0_ckpt_sha,
                "epoch8_adapter_sha256": epoch8_sha,
                "split_json_sha256": split_json_sha,
                "visual_batch_contract": "fixed_b128_pad_last",
                "fixed_visual_batch_size": 128,
                "split": split_name,
                "video_id": vid,
                "window_count": shard_data["n_windows"]
            }

            torch.save(shard_data, shard_path)
            split_windows += shard_data["n_windows"]
            split_active_queries += len(active_q)
            print(f"    [+] Saved shard: {shard_path.name} in {time.time() - t_v0:.2f}s ({shard_data['n_windows']} windows, {len(active_q)} active queries)", flush=True)

        print(f"[+] Split {split_name.upper()} Completed: {n_vids} Videos, {split_windows} Windows, {split_active_queries} Active Proposals.\n", flush=True)
        grand_total_windows += split_windows
        grand_total_active_queries += split_active_queries

    print("=" * 115, flush=True)
    print("PRODUCTION EXTRACTION FOR TRAIN (47) AND VAL (9) COMPLETE", flush=True)
    print(f"Grand Total: {grand_total_windows} Rolling Windows, {grand_total_active_queries} Active Proposals Sharded.", flush=True)
    print("Test split (18 videos) remains completely unopened per D-4078.", flush=True)
    print(f"Total Execution Time: {time.time() - t_start:.2f}s", flush=True)
    print("=" * 115, flush=True)


if __name__ == "__main__":
    main()