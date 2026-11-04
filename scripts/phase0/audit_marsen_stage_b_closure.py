"""
D:/MARIVISION/H0_workbench/eval_tools/audit_marsen_stage_b_closure.py
MARSEN Stream B2: Read-Only Stage-B Closure & Sentinel Audit (P-2185..P-2187)

Performs a strict, read-only audit of marsen_b2_calibrated_thresholds.json:
  1. Verifies SHA-256 of marsen_b2_calibrated_thresholds.json and OOF manifest.
  2. For all 9 contender x seed pairs, loads OOF scores and extracts:
     - exact tau_star repr
     - max_oof_score exact repr
     - tau_inf = nextafter(max_oof_score, +inf) exact repr
     - is_closed = (tau_star == tau_inf) or (tau_star > max_oof_score)
     - n_admitted (number of OOF queries >= tau_star)
     - n_frames_admitted (number of frames with >= 1 admitted query)
     - exact best_f1, baseline_f1, and delta_f1
     - exact TP, FP, FN, baseline_tp, baseline_fp, baseline_fn, total_gt
  3. Verifies baseline count invariance across all 9 calibrations.
No fitting, no threshold sweeps, no file mutations. Runtime: ~5 seconds.
"""

import sys
import time
import json
import hashlib
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)

OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
SHARD_DIR = OUT_DIR / "full_split_shards"
CACHE_DIR = SHARD_DIR / "lovo_cache"

OOF_MANIFEST_JSON = OUT_DIR / "marsen_b2_lovo_oof_manifest.json"
THRESHOLDS_JSON = OUT_DIR / "marsen_b2_calibrated_thresholds.json"

SEEDS = [41, 42, 43]
NUM_VIDEOS = 47
EXPECTED_TOTAL_QUERIES = 211712

FROZEN_OOF_MANIFEST_SHA256 = "1919112fe4280a5c60aa42e77a653f74b1a71905bee4bcd8552fb296b6916452"
FROZEN_THRESHOLDS_SHA256 = "615d514a0b2070e0338f9d7ba0e6b73e9f25e1b4773599f83b6203a8bf396f17"


def compute_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main():
    t_start = time.time()
    print("=" * 115, flush=True)
    print("MARSEN STREAM B2: READ-ONLY STAGE-B CLOSURE & SENTINEL AUDIT (P-2185..P-2187)", flush=True)
    print("=" * 115, flush=True)

    # 1. Cryptographic File Verification
    assert THRESHOLDS_JSON.exists(), f"[FATAL] Missing thresholds JSON at {THRESHOLDS_JSON}"
    actual_thresh_sha = compute_sha256(THRESHOLDS_JSON)
    assert actual_thresh_sha == FROZEN_THRESHOLDS_SHA256, (
        f"[FATAL] Threshold manifest SHA drift: expected {FROZEN_THRESHOLDS_SHA256}, got {actual_thresh_sha}"
    )

    assert OOF_MANIFEST_JSON.exists(), f"[FATAL] Missing OOF manifest at {OOF_MANIFEST_JSON}"
    actual_oof_sha = compute_sha256(OOF_MANIFEST_JSON)
    assert actual_oof_sha == FROZEN_OOF_MANIFEST_SHA256, (
        f"[FATAL] OOF manifest SHA drift: expected {FROZEN_OOF_MANIFEST_SHA256}, got {actual_oof_sha}"
    )

    print(f"[+] Threshold Manifest Verified SHA: {actual_thresh_sha}")
    print(f"[+] OOF Manifest Verified SHA:       {actual_oof_sha}\n", flush=True)

    with open(THRESHOLDS_JSON, "r", encoding="utf-8") as f:
        thresh_data = json.load(f)

    # 2. Build Query-to-Frame Index for Frame Admission Auditing
    print("[*] Indexing Frame-Level Query Association...", flush=True)
    query_to_frame_idx = []
    frame_counter = 0

    for vid in range(1, NUM_VIDEOS + 1):
        p = SHARD_DIR / f"marsen_b2_shard_train_vid{vid}.pth"
        shard = torch.load(p, map_location="cpu")
        act_q_by_img = defaultdict(list)
        for q in shard["active_queries"]:
            act_q_by_img[q["image_id"]].append(q)

        for fr in shard.get("frame_records", []):
            iid = fr["image_id"]
            fr_act_q = act_q_by_img.get(iid, [])
            for _ in fr_act_q:
                query_to_frame_idx.append(frame_counter)
            frame_counter += 1

    assert len(query_to_frame_idx) == EXPECTED_TOTAL_QUERIES, (
        f"[FATAL] Query indexing mismatch: {len(query_to_frame_idx)} != {EXPECTED_TOTAL_QUERIES}"
    )
    query_to_frame_idx = np.array(query_to_frame_idx, dtype=np.int32)
    print(f"[+] Successfully indexed {len(query_to_frame_idx):,} queries across {frame_counter:,} frames.\n", flush=True)

    # 3. Audit All 9 Calibrations
    print("=" * 115, flush=True)
    print(f"{'Key':<11} | {'tau* (Exact)':<20} | {'max_oof (Exact)':<20} | {'tau* == tau_inf':<16} | {'Admitted Q':<11} | {'Admitted Fr':<11} | {'Operating State'}")
    print("-" * 115, flush=True)

    baseline_records = []
    audit_summary = []

    for contender in ["S0", "S35", "SG"]:
        for seed in SEEDS:
            key_name = f"{contender}_seed{seed}"
            assert key_name in thresh_data["thresholds"], f"[FATAL] Missing key {key_name} in manifest"
            rec = thresh_data["thresholds"][key_name]

            # Reconstruct concatenated OOF score vector
            score_list = []
            for vid in range(1, NUM_VIDEOS + 1):
                c_data = torch.load(CACHE_DIR / f"lovo_pred_seed{seed}_vid{vid}.pth", map_location="cpu")
                score_list.append(c_data[contender])
            oof_scores = np.concatenate(score_list)
            scores64 = np.asarray(oof_scores, dtype=np.float64)

            max_oof = float(scores64.max())
            tau_inf = float(np.nextafter(max_oof, np.inf))
            tau_star = float(rec["tau_star"])

            is_exact_tau_inf = bool(tau_star == tau_inf)
            is_greater_than_max = bool(tau_star > max_oof)
            is_closed = is_exact_tau_inf or is_greater_than_max

            admitted_mask = (scores64 >= tau_star)
            n_admitted = int(np.count_nonzero(admitted_mask))
            admitted_pct = (n_admitted / len(scores64)) * 100.0

            if n_admitted > 0:
                admitted_frames = set(query_to_frame_idx[admitted_mask])
                n_frames_admitted = len(admitted_frames)
            else:
                n_frames_admitted = 0

            operating_state = "CLOSED_GATE" if is_closed else "ACTIVE_GATE"

            print(
                f"{key_name:<11} | {tau_star:<20.14f} | {max_oof:<20.14f} | {str(is_exact_tau_inf):<16} | "
                f"{n_admitted:>11} | {n_frames_admitted:>11} | {operating_state}",
                flush=True
            )

            # Record baseline for invariance check
            b_tuple = (rec["baseline_tp"], rec["baseline_fp"], rec["baseline_fn"], rec["baseline_f1"])
            baseline_records.append(b_tuple)

            audit_summary.append({
                "key": key_name,
                "contender": contender,
                "seed": seed,
                "tau_star": tau_star,
                "max_oof": max_oof,
                "tau_inf": tau_inf,
                "is_closed": is_closed,
                "n_admitted": n_admitted,
                "admitted_pct": admitted_pct,
                "n_frames_admitted": n_frames_admitted,
                "operating_state": operating_state,
                "f1": rec["f1"],
                "baseline_f1": rec["baseline_f1"],
                "delta_f1": rec["f1"] - rec["baseline_f1"],
                "tp": rec["tp"],
                "fp": rec["fp"],
                "fn": rec["fn"],
                "baseline_tp": rec["baseline_tp"],
                "baseline_fp": rec["baseline_fp"],
                "baseline_fn": rec["baseline_fn"],
                "precision": rec["precision"],
                "recall": rec["recall"],
                "baseline_precision": rec["baseline_precision"],
                "baseline_recall": rec["baseline_recall"],
                "total_gt": rec["total_gt"]
            })

    print("=" * 115, flush=True)

    # 4. Detailed Metrics and Baseline Invariance Audit
    print("\n" + "=" * 115, flush=True)
    print("DETAILED METRIC & BASELINE INVARIANCE AUDIT (FROM FROZEN JSON)")
    print("=" * 115, flush=True)

    ref_b = baseline_records[0]
    all_baseline_match = all(b == ref_b for b in baseline_records)

    print(f"[*] Baseline Invariance Across All 9 Calibrations: {'PASS (Bitwise Identical)' if all_baseline_match else 'FAIL'}")
    print(f"    - Certified Baseline TP: {ref_b[0]}")
    print(f"    - Certified Baseline FP: {ref_b[1]}")
    print(f"    - Certified Baseline FN: {ref_b[2]}")
    print(f"    - Certified Total GT:    {audit_summary[0]['total_gt']}")
    print(f"    - Certified Baseline P:  {audit_summary[0]['baseline_precision']:.8f}")
    print(f"    - Certified Baseline R:  {audit_summary[0]['baseline_recall']:.8f}")
    print(f"    - Certified Baseline F1: {ref_b[3]:.8f}\n")

    print(f"{'Key':<11} | {'TP':<7} | {'FP':<7} | {'FN':<6} | {'Delta TP':<9} | {'Delta FP':<9} | {'Best F1':<12} | {'Delta F1':<14} | {'State'}")
    print("-" * 115, flush=True)

    for item in audit_summary:
        d_tp = item["tp"] - item["baseline_tp"]
        d_fp = item["fp"] - item["baseline_fp"]
        d_f1 = item["delta_f1"]
        print(
            f"{item['key']:<11} | {item['tp']:>7} | {item['fp']:>7} | {item['fn']:>6} | "
            f"{d_tp:>+9} | {d_fp:>+9} | {item['f1']:>12.8f} | {d_f1:>+14.2e} | {item['operating_state']}"
        )

    print("=" * 115, flush=True)
    print(f"[+] Read-Only Audit Finished in {time.time() - t_start:.2f}s.\n", flush=True)


if __name__ == "__main__":
    main()