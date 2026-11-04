"""
D:/MARIVISION/H0_workbench/eval_tools/audit_marsen_production_shard_census.py
MARSEN Stream B2: Production 56-Shard Population & Provenance Census (P-2118..P-2131)

Audits all 56 production shards (47 Train + 9 Val):
  1. External Live Disk Provenance Lock (P-2121, P-2122): Compares shard metadata directly
     against live extractor driver, engine core, R0, adapter, and split JSON hashes.
  2. Shard Identity & Freshness (P-2128, P-2129): Enforces fresh_inference=True and
     exact filename <-> meta video_id/split equivalence.
  3. Query-Ledger Integrity & Uniqueness (P-2123, P-2130): Asserts unique query keys,
     sum(n_temp) == len(queries), and exact key-by-key active proposal identity.
  4. Authoritative Production Aggregate Anchors (P-2124): Hard-asserts 20,075/211,712 for Train
     and 4,221/48,223 for Val.
  5. Video-ID Inventory: Asserts exact {1..47} for Train and {1..9} for Val.
  6. Tensor & Discrete Field Integrity: 15-D, 256-D, 3x128-D finite assertions,
     valid label domains, and score >= 0.20.
  7. Descriptive Class Census (P-2125): Compiles 7-class prediction distributions per split.
  8. Master Census Ledger Self-Hash (P-2126, P-2131): Embeds census driver SHA-256 and
     computes SHA-256 of marsen_b2_production_census_56shards.json.
Runtime: ~15 seconds.
"""

import sys
import time
import json
import hashlib
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

sys.stdout.reconfigure(line_buffering=True)

OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
ROOT_DIR = Path("D:/MARIVISION")
TRAIN_JSON = ROOT_DIR / "seaclips-train.json"
VAL_JSON = ROOT_DIR / "seaclips-val.json"
SHARD_DIR = OUT_DIR / "full_split_shards"
CENSUS_JSON = OUT_DIR / "marsen_b2_production_census_56shards.json"

EXTRACTOR_PATH = OUT_DIR / "extract_marsen_full_splits_production.py"
ENGINE_PATH = OUT_DIR / "marsen_stream_b2_engine.py"

EXPECTED_VIDEOS = {"train": 47, "val": 9}
EXPECTED_AGGREGATES = {
    "train": {"windows": 20075, "active_queries": 211712},
    "val": {"windows": 4221, "active_queries": 48223}
}


def compute_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main():
    t_start = time.time()
    # P-2131: Self-hash of census driver script
    census_driver_sha256 = compute_sha256(Path(__file__))

    print("=" * 115, flush=True)
    print("MARSEN STREAM B2: PRODUCTION 56-SHARD POPULATION & PROVENANCE CENSUS (P-2118..P-2131)", flush=True)
    print(f"Census Driver SHA-256: {census_driver_sha256}", flush=True)
    print("=" * 115, flush=True)

    assert SHARD_DIR.exists(), f"[FATAL] Missing shard directory at {SHARD_DIR}"
    assert EXTRACTOR_PATH.exists(), f"[FATAL] Missing extractor driver at {EXTRACTOR_PATH}"
    assert ENGINE_PATH.exists(), f"[FATAL] Missing engine core at {ENGINE_PATH}"
    assert TRAIN_JSON.exists(), f"[FATAL] Missing train manifest at {TRAIN_JSON}"
    assert VAL_JSON.exists(), f"[FATAL] Missing val manifest at {VAL_JSON}"

    sys.path.insert(0, str(OUT_DIR))
    import audit_pol3v_operating_point_sweep as opsweep

    # P-2121 & P-2122: Compute authoritative external live disk hashes
    print("[*] Computing Live External Disk Provenance Hashes...", flush=True)
    live_provenance = {
        "extractor_sha256": compute_sha256(EXTRACTOR_PATH),
        "engine_sha256": compute_sha256(ENGINE_PATH),
        "r0_checkpoint_sha256": compute_sha256(opsweep.M0_CKPT_PATH),
        "epoch8_adapter_sha256": compute_sha256(opsweep.EPOCH8_MODEL_PTH),
        "train_json_sha256": compute_sha256(TRAIN_JSON),
        "val_json_sha256": compute_sha256(VAL_JSON),
        "visual_batch_contract": "fixed_b128_pad_last",
        "fixed_visual_batch_size": 128
    }
    for k, v in live_provenance.items():
        print(f"    - {k:<25}: {v}", flush=True)

    train_shard_files = sorted(list(SHARD_DIR.glob("marsen_b2_shard_train_vid*.pth")))
    val_shard_files = sorted(list(SHARD_DIR.glob("marsen_b2_shard_val_vid*.pth")))

    print(f"\n[*] Discovered Shards: {len(train_shard_files)} Train / {len(val_shard_files)} Val (Total: {len(train_shard_files) + len(val_shard_files)})", flush=True)
    assert len(train_shard_files) == EXPECTED_VIDEOS["train"], (
        f"[FATAL] Expected {EXPECTED_VIDEOS['train']} train shards, found {len(train_shard_files)}"
    )
    assert len(val_shard_files) == EXPECTED_VIDEOS["val"], (
        f"[FATAL] Expected {EXPECTED_VIDEOS['val']} val shards, found {len(val_shard_files)}"
    )

    all_shards = [("train", p) for p in train_shard_files] + [("val", p) for p in val_shard_files]

    census_data = {
        "_meta": {
            "census_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "census_driver_sha256": census_driver_sha256,
            "total_shards": len(all_shards),
            "train_shards": len(train_shard_files),
            "val_shards": len(val_shard_files),
            "live_provenance": live_provenance
        },
        "splits": {},
        "per_video": [],
        "shard_sha256_ledger": {}
    }

    split_stats = {
        "train": defaultdict(int),
        "val": defaultdict(int)
    }
    split_classes = {
        "train": Counter(),
        "val": Counter()
    }
    observed_video_ids = {
        "train": set(),
        "val": set()
    }

    print("\n" + "=" * 115, flush=True)
    print(f"{'Split':<6} | {'Vid':<4} | {'Windows':<8} | {'EmptyMem':<8} | {'TotalQ':<8} | {'ActiveQ':<8} | {'Util(y=1)':<9} | {'Rescue':<8} | {'Prevalence':<10} | {'Status'}")
    print("-" * 115, flush=True)

    for split_name, shard_path in all_shards:
        sha256_hash = compute_sha256(shard_path)
        census_data["shard_sha256_ledger"][shard_path.name] = sha256_hash

        shard = torch.load(shard_path, map_location="cpu")
        meta = shard.get("_meta", {})

        # P-2128: Hard fresh inference assertion
        assert meta.get("fresh_inference") is True, (
            f"[FATAL] Shard {shard_path.name} is not marked fresh_inference=True"
        )

        # P-2121: External disk provenance lock verification
        assert meta.get("extractor_sha256") == live_provenance["extractor_sha256"], (
            f"[FATAL] Shard {shard_path.name} extractor SHA mismatch: {meta.get('extractor_sha256')} != {live_provenance['extractor_sha256']}"
        )
        assert meta.get("engine_sha256") == live_provenance["engine_sha256"], (
            f"[FATAL] Shard {shard_path.name} engine SHA mismatch: {meta.get('engine_sha256')} != {live_provenance['engine_sha256']}"
        )
        assert meta.get("r0_checkpoint_sha256") == live_provenance["r0_checkpoint_sha256"], (
            f"[FATAL] Shard {shard_path.name} R0 checkpoint SHA mismatch: {meta.get('r0_checkpoint_sha256')} != {live_provenance['r0_checkpoint_sha256']}"
        )
        assert meta.get("epoch8_adapter_sha256") == live_provenance["epoch8_adapter_sha256"], (
            f"[FATAL] Shard {shard_path.name} adapter SHA mismatch: {meta.get('epoch8_adapter_sha256')} != {live_provenance['epoch8_adapter_sha256']}"
        )
        assert meta.get("visual_batch_contract") == live_provenance["visual_batch_contract"], (
            f"[FATAL] Shard {shard_path.name} visual contract mismatch: {meta.get('visual_batch_contract')}"
        )
        assert meta.get("fixed_visual_batch_size") == live_provenance["fixed_visual_batch_size"], (
            f"[FATAL] Shard {shard_path.name} batch size mismatch: {meta.get('fixed_visual_batch_size')}"
        )

        # P-2122: Split-specific manifest hash lock
        expected_manifest_sha = live_provenance["train_json_sha256"] if split_name == "train" else live_provenance["val_json_sha256"]
        assert meta.get("split_json_sha256") == expected_manifest_sha, (
            f"[FATAL] Shard {shard_path.name} split JSON SHA mismatch: {meta.get('split_json_sha256')} != {expected_manifest_sha}"
        )

        # Split identity check
        assert meta.get("split") == split_name, f"[FATAL] Shard split mismatch: {meta.get('split')} != {split_name}"
        vid = int(meta.get("video_id"))
        observed_video_ids[split_name].add(vid)

        # P-2129: Shard filename <-> metadata binding
        expected_name = f"marsen_b2_shard_{split_name}_vid{vid}.pth"
        assert shard_path.name == expected_name, (
            f"[FATAL] Filename/metadata identity mismatch: {shard_path.name} != {expected_name}"
        )

        n_windows = shard.get("n_windows", 0)
        frame_records = shard.get("frame_records", [])
        queries = shard.get("queries", [])
        active_queries = shard.get("active_queries", [])

        assert len(frame_records) == n_windows, f"[FATAL] Frame record count mismatch in {shard_path.name}"
        assert meta.get("window_count") == n_windows, f"[FATAL] Meta window_count mismatch in {shard_path.name}"

        # P-2130: Query key uniqueness check
        query_keys = [(int(q["image_id"]), int(q["query_id"])) for q in queries]
        assert len(query_keys) == len(set(query_keys)), (
            f"[FATAL] Duplicate query keys detected in {shard_path.name}"
        )

        # P-2123: Query-ledger conservation & exact identity
        n_temp_sum = sum(int(fr.get("n_temp", 0)) for fr in frame_records)
        assert n_temp_sum == len(queries), (
            f"[FATAL] Query conservation violated in {shard_path.name}: sum(n_temp)={n_temp_sum} != len(queries)={len(queries)}"
        )

        derived_active = [q for q in queries if q["is_active"]]
        derived_keys = [(int(q["image_id"]), int(q["query_id"])) for q in derived_active]
        active_keys = [(int(q["image_id"]), int(q["query_id"])) for q in active_queries]
        assert derived_keys == active_keys, f"[FATAL] Active query ordered identity mismatch in {shard_path.name}"

        n_empty_mem = sum(1 for fr in frame_records if fr.get("n_temp", 0) == 0)
        n_util = sum(1 for q in active_queries if q.get("label_op_util", 0) == 1)
        n_rescue = sum(1 for q in active_queries if q.get("label_rescue_pot", 0) == 1)
        prev = (n_util / len(active_queries) * 100.0) if len(active_queries) > 0 else 0.0

        # Discrete-field and tensor completeness check
        for q in active_queries:
            assert q["is_active"] is True
            cls_idx = int(q["pred_class"])
            assert cls_idx in range(7), f"[FATAL] Invalid pred_class {cls_idx} in {shard_path.name}"
            assert int(q["label_op_util"]) in {0, 1}, f"[FATAL] Invalid label_op_util in {shard_path.name}"
            assert int(q["label_rescue_pot"]) in {0, 1}, f"[FATAL] Invalid label_rescue_pot in {shard_path.name}"

            score = float(q["max_score"])
            assert np.isfinite(score) and score >= 0.20, f"[FATAL] Invalid score {score} in {shard_path.name}"

            # P-2125: Record class prediction
            split_classes[split_name][cls_idx] += 1

            feat = np.asarray(q["features"])
            dec = np.asarray(q["decoder_256"])
            d1 = np.asarray(q["1x_gem4"])
            d3 = np.asarray(q["3x_gem4"])
            d5 = np.asarray(q["5x_gap"])

            assert feat.shape == (15,), f"[FATAL] 15-D feature shape mismatch in {shard_path.name}"
            assert dec.shape == (256,), f"[FATAL] Decoder-256 shape mismatch in {shard_path.name}"
            assert d1.shape == (128,), f"[FATAL] 1x descriptor shape mismatch in {shard_path.name}"
            assert d3.shape == (128,), f"[FATAL] 3x descriptor shape mismatch in {shard_path.name}"
            assert d5.shape == (128,), f"[FATAL] 5x descriptor shape mismatch in {shard_path.name}"

            assert np.all(np.isfinite(feat)), f"[FATAL] Non-finite features in {shard_path.name}"
            assert np.all(np.isfinite(dec)), f"[FATAL] Non-finite decoder state in {shard_path.name}"
            assert np.all(np.isfinite(d1)), f"[FATAL] Non-finite 1x descriptor in {shard_path.name}"
            assert np.all(np.isfinite(d3)), f"[FATAL] Non-finite 3x descriptor in {shard_path.name}"
            assert np.all(np.isfinite(d5)), f"[FATAL] Non-finite 5x descriptor in {shard_path.name}"

        # Accumulate statistics
        split_stats[split_name]["windows"] += n_windows
        split_stats[split_name]["empty_mem_windows"] += n_empty_mem
        split_stats[split_name]["total_queries"] += len(queries)
        split_stats[split_name]["active_queries"] += len(active_queries)
        split_stats[split_name]["util_positives"] += n_util
        split_stats[split_name]["rescue_positives"] += n_rescue

        video_record = {
            "split": split_name,
            "video_id": vid,
            "shard_name": shard_path.name,
            "sha256": sha256_hash,
            "n_windows": n_windows,
            "n_empty_mem": n_empty_mem,
            "n_total_queries": len(queries),
            "n_active_queries": len(active_queries),
            "n_util_positives": n_util,
            "n_rescue_positives": n_rescue,
            "prevalence_pct": round(prev, 3)
        }
        census_data["per_video"].append(video_record)

        print(f"{split_name:<6} | {vid:>4} | {n_windows:>8} | {n_empty_mem:>8} | {len(queries):>8} | {len(active_queries):>8} | {n_util:>9} | {n_rescue:>8} | {prev:>9.2f}% | VERIFIED", flush=True)

    print("=" * 115, flush=True)

    # Video-ID Inventory Assertions
    expected_train_vids = set(range(1, 48))
    expected_val_vids = set(range(1, 10))
    assert observed_video_ids["train"] == expected_train_vids, (
        f"[FATAL] Train video ID inventory mismatch: {observed_video_ids['train']} != {expected_train_vids}"
    )
    assert observed_video_ids["val"] == expected_val_vids, (
        f"[FATAL] Val video ID inventory mismatch: {observed_video_ids['val']} != {expected_val_vids}"
    )
    print("\n[+] Video-ID Inventory Verified: Train={1..47} exactly, Val={1..9} exactly.", flush=True)

    # P-2124: Production Aggregate Anchors Assertions
    for s_name in ["train", "val"]:
        st = split_stats[s_name]
        exp = EXPECTED_AGGREGATES[s_name]
        assert st["windows"] == exp["windows"], (
            f"[FATAL] {s_name} window aggregate mismatch: {st['windows']} != {exp['windows']}"
        )
        assert st["active_queries"] == exp["active_queries"], (
            f"[FATAL] {s_name} active query aggregate mismatch: {st['active_queries']} != {exp['active_queries']}"
        )

    # Split-Level Aggregates Table
    print("\n" + "=" * 115, flush=True)
    print("SPLIT-LEVEL POPULATION AGGREGATES")
    print("=" * 115, flush=True)
    print(f"{'Split':<8} | {'Videos':<7} | {'Windows':<10} | {'EmptyMem':<10} | {'Total Temp Q':<14} | {'Active Q':<12} | {'Util (y=1)':<12} | {'Rescue (y=1)':<12} | {'Prevalence'}")
    print("-" * 115, flush=True)

    for s_name in ["train", "val"]:
        st = split_stats[s_name]
        prev_s = (st["util_positives"] / st["active_queries"] * 100.0) if st["active_queries"] > 0 else 0.0
        n_v = EXPECTED_VIDEOS[s_name]
        census_data["splits"][s_name] = {
            "n_videos": n_v,
            "n_windows": st["windows"],
            "n_empty_mem_windows": st["empty_mem_windows"],
            "n_total_queries": st["total_queries"],
            "n_active_queries": st["active_queries"],
            "n_util_positives": st["util_positives"],
            "n_rescue_positives": st["rescue_positives"],
            "utility_prevalence_pct": round(prev_s, 4),
            "class_distribution": dict(split_classes[s_name])
        }
        print(f"{s_name.upper():<8} | {n_v:>7} | {st['windows']:>10} | {st['empty_mem_windows']:>10} | {st['total_queries']:>14} | {st['active_queries']:>12} | {st['util_positives']:>12} | {st['rescue_positives']:>12} | {prev_s:>9.3f}%", flush=True)

    print("=" * 115, flush=True)

    # P-2125: Descriptive Class Distribution Table
    print("\n" + "=" * 115, flush=True)
    print("DESCRIPTIVE CLASS PREVALENCE CENSUS (ACTIVE PROPOSALS)")
    print("=" * 115, flush=True)
    print(f"{'Class Index':<12} | {'Train Count':<16} | {'Train Pct':<14} | {'Val Count':<16} | {'Val Pct'}")
    print("-" * 115, flush=True)

    total_train_act = split_stats["train"]["active_queries"]
    total_val_act = split_stats["val"]["active_queries"]

    for c in range(7):
        cnt_t = split_classes["train"][c]
        pct_t = (cnt_t / total_train_act * 100.0) if total_train_act > 0 else 0.0
        cnt_v = split_classes["val"][c]
        pct_v = (cnt_v / total_val_act * 100.0) if total_val_act > 0 else 0.0
        print(f"Class {c:<6} | {cnt_t:>16} | {pct_t:>13.2f}% | {cnt_v:>16} | {pct_v:>13.2f}%", flush=True)

    print("=" * 115, flush=True)

    # Freeze JSON Ledger
    with open(CENSUS_JSON, "w", encoding="utf-8") as f:
        json.dump(census_data, f, indent=2)

    # P-2126: Compute and print ledger self-hash
    census_sha256 = compute_sha256(CENSUS_JSON)

    print(f"\n[+] Production Census Ledger Frozen: {CENSUS_JSON.resolve()}", flush=True)
    print(f"[+] Census Driver SHA-256:          {census_driver_sha256}", flush=True)
    print(f"[+] Master Census Ledger SHA-256:    {census_sha256}", flush=True)
    print("[+] All 56 shards structurally verified, provenance-locked, finite, and SHA-256 ledger frozen.", flush=True)
    print(f"[+] Census Execution Time: {time.time() - t_start:.2f}s", flush=True)
    print("=" * 115, flush=True)


if __name__ == "__main__":
    main()