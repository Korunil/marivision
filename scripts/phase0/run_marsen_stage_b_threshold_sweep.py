"""
D:/MARIVISION/H0_workbench/eval_tools/run_marsen_stage_b_threshold_sweep.py
MARSEN Stream B2: Stage B - Exact Whole-Detector Train-OOF Threshold Calibration (P-2152..P-2183)

Calibrates the 9 exact operating thresholds (S0, S35, SG x Seeds 41, 42, 43):
  1. Full Provenance Gate: Verifies census JSON, all 47 Train shards, seaclips-train.json,
     OOF manifest, all 141 OOF caches/sidecars, and the historical prescreen script.
  2. Query-Key Alignment Gate: Hard-asserts exact chronological query-key equality
     between reconstructed frame queries and OOF score vectors for all three seeds.
  3. Literal Historical Suppression & Zero-Admission Bypass:
     - When 0 temporal queries admitted in a frame, bypasses concatenation entirely and reuses base_frame_cache.
     - When >= 1 temporal queries admitted, concatenates full temporal tensors with non-admitted rows set to -1e9.
  4. Float64 Search Space: Extracts unique thresholds in float64 with tau_inf = np.nextafter(max, np.inf).
  5. Direct Historical Gate B1 (P-2179..P-2183): Directly invokes audit_marsen_classifier_prescreen.run_detector_sweep_fast
     on 500 production frames, asserting exact tau*, TP, FP, FN equality and |delta_F1| < 1e-12.
  6. Baseline Invariance: Asserts identical baseline TP/FP/FN/F1 across all 9 sweeps.
  7. Manifest Freeze: Records calibrated thresholds to marsen_b2_calibrated_thresholds.json.
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

# Path Constants
ROOT_DIR = Path("D:/MARIVISION")
TRAIN_JSON = ROOT_DIR / "seaclips-train.json"

OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
SHARD_DIR = OUT_DIR / "full_split_shards"
CACHE_DIR = SHARD_DIR / "lovo_cache"

CENSUS_JSON = OUT_DIR / "marsen_b2_production_census_56shards.json"
OOF_MANIFEST_JSON = OUT_DIR / "marsen_b2_lovo_oof_manifest.json"
THRESHOLDS_JSON = OUT_DIR / "marsen_b2_calibrated_thresholds.json"
HIST_REF_SCRIPT = OUT_DIR / "audit_marsen_classifier_prescreen.py"

SEEDS = [41, 42, 43]
NUM_VIDEOS = 47
EXPECTED_TOTAL_QUERIES = 211712
EXPECTED_TOTAL_WINDOWS = 20075

FROZEN_CENSUS_SHA256 = "a20a734d668a9078bdfe274cb09fa5a47ad9c6a3a9d7cef2d1f99b926623fbc5"
FROZEN_OOF_MANIFEST_SHA256 = "1919112fe4280a5c60aa42e77a653f74b1a71905bee4bcd8552fb296b6916452"
FROZEN_HIST_REF_SHA256 = "d5352557efa538460f490fc45cb6c1eb2921abb9ee8a689fa2334bd5fd30286a"

TIE_TOLERANCE = 1e-12
SUPPRESSION_LOGIT_VALUE = -1e9


def compute_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def run_event_driven_sweep_literal(frames_data, scores_array, nonlin, label_to_cat_id, desc="Sweep"):
    """
    Event-driven exact sweep across unique float64 score thresholds with literal historical semantics:
      - Starts at tau = +inf (0 admissions; baseline state equals base_frame_cache).
      - Steps downward through score transition events.
      - Re-evaluates only frames whose admissions change (re-runs top-100 with temporal concat only when >= 1 admitted).
      - When 0 candidates admitted, strictly uses base_frame_cache.
      - Enforces higher-tau tie-breaking with 1e-12 tolerance.
    """
    total_gt = sum(len(fr["gt_cats"]) for fr in frames_data)

    # 1. Precompute baseline R0 cache (pure d_logits, d_boxes)
    base_frame_cache = {}
    frame_tp = np.zeros(len(frames_data), dtype=np.int32)
    frame_fp = np.zeros(len(frames_data), dtype=np.int32)

    for f_idx, fr in enumerate(frames_data):
        tp_b, fp_b, _, _ = nonlin.evaluate_operational_top100(
            fr["d_logits"], fr["d_boxes"], fr["gt_boxes_px"], fr["gt_cats"],
            label_to_cat_id, fr["img_w"], fr["img_h"]
        )
        base_frame_cache[f_idx] = (tp_b, fp_b)
        frame_tp[f_idx] = tp_b
        frame_fp[f_idx] = fp_b

    # Immutable baseline counters
    baseline_tp = int(np.sum(frame_tp))
    baseline_fp = int(np.sum(frame_fp))
    baseline_fn = int(total_gt - baseline_tp)

    prec_b = baseline_tp / (baseline_tp + baseline_fp) if (baseline_tp + baseline_fp) > 0 else 0.0
    rec_b = baseline_tp / (baseline_tp + baseline_fn) if (baseline_tp + baseline_fn) > 0 else 0.0
    f1_baseline = (2.0 * prec_b * rec_b) / (prec_b + rec_b) if (prec_b + rec_b) > 0 else 0.0

    curr_tp = baseline_tp
    curr_fp = baseline_fp

    # Float64 exact threshold space
    scores64 = np.asarray(scores_array, dtype=np.float64)
    unique_scores = np.unique(scores64)
    max_score = float(unique_scores.max()) if len(unique_scores) > 0 else 1.0
    tau_inf = float(np.nextafter(max_score, np.inf))

    best_tau = tau_inf
    best_f1 = f1_baseline
    best_tp = baseline_tp
    best_fp = baseline_fp
    best_fn = baseline_fn

    # Map each active query to (frame_idx, local_active_idx)
    events_by_score = defaultdict(list)
    q_offset = 0
    for f_idx, fr in enumerate(frames_data):
        n_act = len(fr["local_active_indices"])
        if n_act > 0:
            for l_pos, local_idx in enumerate(fr["local_active_indices"]):
                s_val = float(scores64[q_offset + l_pos])
                events_by_score[s_val].append((f_idx, local_idx))
            q_offset += n_act

    # Descending sweep: admissions activate
    unique_scores_desc = sorted(events_by_score.keys(), reverse=True)
    frame_admissions = [np.zeros(fr["n_temp_total"], dtype=bool) for fr in frames_data]

    t_sweep_0 = time.time()
    n_events = len(unique_scores_desc)

    for step_idx, tau in enumerate(unique_scores_desc):
        dirty_frames = set()
        for f_idx, local_idx in events_by_score[tau]:
            frame_admissions[f_idx][local_idx] = True
            dirty_frames.add(f_idx)

        # Re-evaluate only dirty frames
        for f_idx in dirty_frames:
            fr = frames_data[f_idx]
            adm_mask = frame_admissions[f_idx]

            if np.any(adm_mask):
                eff_temp_logits = fr["full_temp_logits"].clone()
                eff_temp_logits[~adm_mask, :] = SUPPRESSION_LOGIT_VALUE
                eval_logits = torch.cat([fr["d_logits"], eff_temp_logits], dim=0)
                eval_boxes = torch.cat([fr["d_boxes"], fr["full_temp_boxes"]], dim=0)

                new_tp, new_fp, _, _ = nonlin.evaluate_operational_top100(
                    eval_logits, eval_boxes, fr["gt_boxes_px"], fr["gt_cats"],
                    label_to_cat_id, fr["img_w"], fr["img_h"]
                )
            else:
                new_tp, new_fp = base_frame_cache[f_idx]

            curr_tp += (new_tp - frame_tp[f_idx])
            curr_fp += (new_fp - frame_fp[f_idx])
            frame_tp[f_idx] = new_tp
            frame_fp[f_idx] = new_fp

        curr_fn = total_gt - curr_tp
        prec = curr_tp / (curr_tp + curr_fp) if (curr_tp + curr_fp) > 0 else 0.0
        rec = curr_tp / (curr_tp + curr_fn) if (curr_tp + curr_fn) > 0 else 0.0
        f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        if f1 > best_f1 + TIE_TOLERANCE:
            best_f1 = f1
            best_tau = tau
            best_tp = curr_tp
            best_fp = curr_fp
            best_fn = curr_fn
        elif abs(f1 - best_f1) <= TIE_TOLERANCE and tau > best_tau:
            best_f1 = f1
            best_tau = tau
            best_tp = curr_tp
            best_fp = curr_fp
            best_fn = curr_fn

        if (step_idx + 1) % 50000 == 0 or (step_idx + 1) == n_events:
            elapsed = time.time() - t_sweep_0
            print(f"    - [{desc}] Step {step_idx + 1:>6} / {n_events} | Best F1: {best_f1:.4f} (tau*={best_tau:.4f}) | Elapsed: {elapsed:.1f}s", flush=True)

    return {
        "tau_star": float(best_tau),
        "f1": float(best_f1),
        "tp": int(best_tp),
        "fp": int(best_fp),
        "fn": int(best_fn),
        "precision": float(best_tp / (best_tp + best_fp)) if (best_tp + best_fp) > 0 else 0.0,
        "recall": float(best_tp / (best_tp + best_fn)) if (best_tp + best_fn) > 0 else 0.0,
        "baseline_f1": float(f1_baseline),
        "baseline_tp": int(baseline_tp),
        "baseline_fp": int(baseline_fp),
        "baseline_fn": int(baseline_fn),
        "baseline_precision": float(prec_b),
        "baseline_recall": float(rec_b),
        "total_gt": int(total_gt)
    }


def main():
    t_start = time.time()
    driver_sha256 = compute_sha256(Path(__file__))

    print("=" * 115, flush=True)
    print("MARSEN STREAM B2: STAGE B - EXACT WHOLE-DETECTOR THRESHOLD CALIBRATION (P-2152..P-2183)", flush=True)
    print(f"Driver Script SHA-256: {driver_sha256}", flush=True)
    print(f"Census Target SHA-256: {FROZEN_CENSUS_SHA256}", flush=True)
    print(f"OOF Manifest Target:   {FROZEN_OOF_MANIFEST_SHA256}", flush=True)
    print(f"Hist Ref Target SHA:   {FROZEN_HIST_REF_SHA256}", flush=True)
    print("=" * 115, flush=True)

    # =========================================================================
    # PROVENANCE GATE (P-2168, P-2178, P-2180)
    # =========================================================================
    print("[*] Running Cryptographic Provenance Gate (P-2168, P-2178, P-2180)...", flush=True)

    # 1. Verify Census & Train Shards
    assert CENSUS_JSON.exists(), f"[FATAL] Missing census JSON at {CENSUS_JSON}"
    actual_census_sha = compute_sha256(CENSUS_JSON)
    assert actual_census_sha == FROZEN_CENSUS_SHA256, (
        f"[FATAL] Census ledger SHA mismatch: expected {FROZEN_CENSUS_SHA256}, got {actual_census_sha}"
    )

    with open(CENSUS_JSON, "r", encoding="utf-8") as f:
        census = json.load(f)
    frozen_shard_ledger = census["shard_sha256_ledger"]

    for vid in range(1, NUM_VIDEOS + 1):
        p = SHARD_DIR / f"marsen_b2_shard_train_vid{vid}.pth"
        assert p.exists(), f"[FATAL] Missing train shard {p.name}"
        actual_shard_sha = compute_sha256(p)
        expected_shard_sha = frozen_shard_ledger[p.name]
        assert actual_shard_sha == expected_shard_sha, (
            f"[FATAL] Train shard SHA drift: {p.name}"
        )

    print("    [+] Master census ledger and all 47 Train shards verified against frozen hashes.")

    # 2. Verify Ground-Truth Train JSON Manifest
    assert TRAIN_JSON.exists(), f"[FATAL] Missing train manifest at {TRAIN_JSON}"
    actual_train_json_sha = compute_sha256(TRAIN_JSON)
    expected_train_json_sha = census["_meta"]["live_provenance"]["train_json_sha256"]
    assert actual_train_json_sha == expected_train_json_sha, (
        f"[FATAL] Train JSON manifest SHA drift:\n"
        f"  Expected: {expected_train_json_sha}\n"
        f"  Actual:   {actual_train_json_sha}"
    )
    print("    [+] Ground-truth seaclips-train.json verified against census live provenance.")

    # 3. Verify OOF Manifest & All 141 Fold Caches + Sidecars
    assert OOF_MANIFEST_JSON.exists(), f"[FATAL] Missing OOF manifest at {OOF_MANIFEST_JSON}"
    actual_manifest_sha = compute_sha256(OOF_MANIFEST_JSON)
    assert actual_manifest_sha == FROZEN_OOF_MANIFEST_SHA256, (
        f"[FATAL] OOF manifest SHA mismatch: expected {FROZEN_OOF_MANIFEST_SHA256}, got {actual_manifest_sha}"
    )

    with open(OOF_MANIFEST_JSON, "r", encoding="utf-8") as f:
        oof_manifest = json.load(f)
    fold_artifacts = oof_manifest["fold_artifacts"]

    for s in SEEDS:
        for v in range(1, NUM_VIDEOS + 1):
            key = f"seed{s}_vid{v}"
            assert key in fold_artifacts, f"[FATAL] Missing fold key {key} in manifest"
            expected_cache_sha = fold_artifacts[key]["sha256"]

            cache_file = CACHE_DIR / f"lovo_pred_seed{s}_vid{v}.pth"
            sha_file = cache_file.with_suffix(".pth.sha256")
            assert cache_file.exists(), f"[FATAL] Missing cache file: {cache_file.name}"
            assert sha_file.exists(), f"[FATAL] Missing sidecar file: {sha_file.name}"

            actual_cache_sha = compute_sha256(cache_file)
            assert actual_cache_sha == expected_cache_sha, (
                f"[FATAL] Cache SHA drift for {cache_file.name}"
            )
            sidecar_sha = sha_file.read_text(encoding="utf-8").strip()
            assert sidecar_sha == expected_cache_sha, (
                f"[FATAL] Sidecar SHA mismatch for {sha_file.name}"
            )

    print("    [+] OOF manifest and all 141 fold caches + sidecars verified against frozen hashes.")

    # 4. Verify Historical Prescreen Reference Script
    assert HIST_REF_SCRIPT.exists(), f"[FATAL] Missing historical prescreen script at {HIST_REF_SCRIPT}"
    actual_hist_ref_sha = compute_sha256(HIST_REF_SCRIPT)
    assert actual_hist_ref_sha == FROZEN_HIST_REF_SHA256, (
        f"[FATAL] Historical prescreen script SHA drift:\n"
        f"  Expected: {FROZEN_HIST_REF_SHA256}\n"
        f"  Actual:   {actual_hist_ref_sha}"
    )
    print("    [+] Historical prescreen script verified against FROZEN_HIST_REF_SHA256.\n", flush=True)

    # Ingest Category IDs and Ground Truth Directly from JSON
    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        train_coco_raw = json.load(f)

    cat_ids = sorted([c["id"] for c in train_coco_raw["categories"]])
    label_to_cat_id = {i: c for i, c in enumerate(cat_ids)}

    anns_by_img = defaultdict(list)
    for ann in train_coco_raw["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    sys.path.insert(0, str(OUT_DIR))
    import audit_pol3v_nonlinear_separability as nonlin
    import audit_marsen_classifier_prescreen as hist_ref

    # Reconstruct Literal Frame State & Global Query-Key Ledger
    print("[*] Reconstructing Literal Frame Tensors & Auditing Query Alignment (P-2169, P-2174)...", flush=True)
    train_frames = []
    frame_active_keys = []

    for vid in range(1, NUM_VIDEOS + 1):
        p = SHARD_DIR / f"marsen_b2_shard_train_vid{vid}.pth"
        shard = torch.load(p, map_location="cpu")
        assert shard["_meta"]["split"] == "train"

        act_q_by_img = defaultdict(list)
        for q in shard["active_queries"]:
            act_q_by_img[q["image_id"]].append(q)

        frame_active_keys_vid = []

        for fr in shard.get("frame_records", []):
            iid = fr["image_id"]
            anns = anns_by_img.get(iid, [])
            gt_boxes_px = [ann["bbox"] for ann in anns]
            gt_cats = [ann["category_id"] for ann in anns]

            fr_act_q = act_q_by_img.get(iid, [])
            local_active_indices = [int(q["query_id"]) for q in fr_act_q]

            for q in fr_act_q:
                frame_active_keys_vid.append((int(q["video_id"]), int(q["image_id"]), int(q["query_id"])))

            full_temp_logits = fr["temp_logits"]
            full_temp_boxes = fr["temp_boxes"]
            n_temp_total = int(fr["n_temp"])

            train_frames.append({
                "video_id": vid,
                "image_id": iid,
                "img_w": fr["img_w"],
                "img_h": fr["img_h"],
                "n_temp_total": n_temp_total,
                "local_active_indices": local_active_indices,
                "d_logits": fr["d_logits"],
                "d_boxes": fr["d_boxes"],
                "full_temp_logits": full_temp_logits,
                "full_temp_boxes": full_temp_boxes,
                "gt_boxes_px": gt_boxes_px,
                "gt_cats": gt_cats
            })

        expected_vid_keys = [(int(q["video_id"]), int(q["image_id"]), int(q["query_id"])) for q in shard["active_queries"]]
        assert frame_active_keys_vid == expected_vid_keys, f"[FATAL] Video {vid} within-shard query key misalignment!"
        frame_active_keys.extend(frame_active_keys_vid)

    assert len(frame_active_keys) == EXPECTED_TOTAL_QUERIES, (
        f"[FATAL] Total active query count mismatch: {len(frame_active_keys)} != {EXPECTED_TOTAL_QUERIES}"
    )
    assert len(train_frames) == EXPECTED_TOTAL_WINDOWS, (
        f"[FATAL] Total window count mismatch: {len(train_frames)} != {EXPECTED_TOTAL_WINDOWS}"
    )

    # Verify Alignment against all three seeds
    for s in SEEDS:
        oof_keys_seed = []
        for vid in range(1, NUM_VIDEOS + 1):
            c_data = torch.load(CACHE_DIR / f"lovo_pred_seed{s}_vid{vid}.pth", map_location="cpu")
            oof_keys_seed.extend(c_data["query_keys"])
        assert frame_active_keys == oof_keys_seed, (
            f"[FATAL] Global query key misalignment between frames and OOF cache for Seed {s}!"
        )

    print(f"    [+] Exact query-key parity confirmed across all 211,712 proposals for Seeds 41, 42, 43.\n", flush=True)

    # =========================================================================
    # PARITY GATE B1: DIRECT INVOCATION OF HISTORICAL REFERENCE EVALUATOR (P-2179..P-2183)
    # =========================================================================
    print("=" * 115, flush=True)
    print("PARITY VERIFICATION GATE B1: EVENT ENGINE VS. HISTORICAL PRESCREEN EVALUATOR (P-2179..P-2183)", flush=True)
    print("=" * 115, flush=True)

    test_slice_frames = train_frames[:500]
    n_test_q = sum(len(fr["local_active_indices"]) for fr in test_slice_frames)

    s41_s0_slice = []
    for vid in range(1, NUM_VIDEOS + 1):
        c_data = torch.load(CACHE_DIR / f"lovo_pred_seed41_vid{vid}.pth", map_location="cpu")
        s41_s0_slice.append(c_data["S0"])
    s41_s0_all = np.concatenate(s41_s0_slice)
    test_scores_slice = s41_s0_all[:n_test_q]

    scores64_slice = np.asarray(test_scores_slice, dtype=np.float64)
    unique_scores_test = np.unique(scores64_slice)
    max_test_score = float(unique_scores_test.max()) if len(unique_scores_test) > 0 else 1.0
    tau_inf_test = float(np.nextafter(max_test_score, np.inf))
    unique_thresh_test = [tau_inf_test] + sorted(list(unique_scores_test), reverse=True)

    # Construct exact adapter for frozen historical evaluator (P-2182, P-2183)
    historical_frames = []
    score_map = {}
    active_key_set = set()
    positive_key_set = set()
    base_frame_cache = {}
    qoff = 0

    for fr in test_slice_frames:
        iid = fr["image_id"]
        historical_frames.append({
            "image_id": iid,
            "n_temp": fr["n_temp_total"],
            "d_logits": fr["d_logits"],
            "d_boxes": fr["d_boxes"],
            "temp_logits": fr["full_temp_logits"],
            "temp_boxes": fr["full_temp_boxes"],
            "gt_boxes_px": fr["gt_boxes_px"],
            "gt_cats": fr["gt_cats"],
            "img_w": fr["img_w"],
            "img_h": fr["img_h"],
        })
        base_frame_cache[iid] = hist_ref.evaluate_operational_top100(
            fr["d_logits"],
            fr["d_boxes"],
            fr["gt_boxes_px"],
            fr["gt_cats"],
            label_to_cat_id,
            fr["img_w"],
            fr["img_h"],
        )
        for local_idx in fr["local_active_indices"]:
            key = (iid, local_idx)
            active_key_set.add(key)
            score_map[key] = float(test_scores_slice[qoff])
            qoff += 1

    assert qoff == len(test_scores_slice), (
        f"[FATAL B1] Historical score-map coverage mismatch: {qoff} != {len(test_scores_slice)}"
    )

    t_ex_0 = time.time()
    ex_tau, ex_f1, ex_rec, _ = hist_ref.run_detector_sweep_fast(
        score_map,
        historical_frames,
        active_key_set,
        positive_key_set,
        label_to_cat_id,
        unique_thresh_test,
        base_frame_cache,
    )
    t_ex = time.time() - t_ex_0

    ex_tp = int(ex_rec["tp"])
    ex_fp = int(ex_rec["fp"])
    ex_fn = int(ex_rec["fn"])
    ex_f1 = float(ex_f1)
    ex_tau = float(ex_tau)

    print(f"[*] Running Event-Driven Engine on identical 500 frames...", flush=True)
    t_ev_0 = time.time()
    ev_res = run_event_driven_sweep_literal(test_slice_frames, test_scores_slice, nonlin, label_to_cat_id, desc="Parity-B1")
    t_ev = time.time() - t_ev_0

    print(f"\n[*] Parity Gate B1 Results:")
    print(f"    - Historical Prescreen Script: tau*={ex_tau:.8f} | F1={ex_f1:.8f} | TP={ex_tp} | FP={ex_fp} | FN={ex_fn} | Time: {t_ex:.2f}s")
    print(f"    - Event-Driven Engine:        tau*={ev_res['tau_star']:.8f} | F1={ev_res['f1']:.8f} | TP={ev_res['tp']} | FP={ev_res['fp']} | FN={ev_res['fn']} | Time: {t_ev:.2f}s")

    assert ex_tau == ev_res["tau_star"], f"[FATAL B1] Threshold divergence: {ex_tau} != {ev_res['tau_star']}"
    assert ex_tp == ev_res["tp"], f"[FATAL B1] TP divergence: {ex_tp} != {ev_res['tp']}"
    assert ex_fp == ev_res["fp"], f"[FATAL B1] FP divergence: {ex_fp} != {ev_res['fp']}"
    assert ex_fn == ev_res["fn"], f"[FATAL B1] FN divergence: {ex_fn} != {ev_res['fn']}"
    assert abs(ex_f1 - ev_res["f1"]) < TIE_TOLERANCE, f"[FATAL B1] F1 divergence: {abs(ex_f1 - ev_res['f1']):.2e}"

    speedup = t_ex / t_ev if t_ev > 0 else 1.0
    print(f"[+] Parity Gate B1 Passed: 100% exact numerical agreement with certified historical evaluator ({speedup:.1f}x speedup).\n", flush=True)

    # =========================================================================
    # CALIBRATION: FULL TRAIN-OOF SWEEP ACROSS 9 CONTENDER-SEED PAIRS
    # =========================================================================
    print("=" * 115, flush=True)
    print("FULL TRAIN-OOF CALIBRATION: EVALUATING 9 OPERATING THRESHOLDS", flush=True)
    print("=" * 115, flush=True)

    calibrated_results = {
        "_meta": {
            "stage": "B_THRESHOLD_CALIBRATION",
            "driver_sha256": driver_sha256,
            "census_sha256": FROZEN_CENSUS_SHA256,
            "train_json_sha256": actual_train_json_sha,
            "oof_manifest_sha256": FROZEN_OOF_MANIFEST_SHA256,
            "hist_ref_script_sha256": actual_hist_ref_sha,
            "total_windows": EXPECTED_TOTAL_WINDOWS,
            "total_queries": EXPECTED_TOTAL_QUERIES,
            "contenders": ["S0", "S35", "SG"],
            "seeds": SEEDS,
            "suppression_logit_value": SUPPRESSION_LOGIT_VALUE,
            "tie_tolerance": TIE_TOLERANCE,
            "calibration_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "thresholds": {}
    }

    reference_baseline = None

    print(f"{'Contender':<10} | {'Seed':<5} | {'tau*':<10} | {'Whole-Det F1':<13} | {'Baseline F1':<13} | {'Delta F1':<10} | {'Precision':<10} | {'Recall':<10} | {'Status'}")
    print("-" * 115, flush=True)

    for contender in ["S0", "S35", "SG"]:
        for seed in SEEDS:
            score_list = []
            for vid in range(1, NUM_VIDEOS + 1):
                c_data = torch.load(CACHE_DIR / f"lovo_pred_seed{seed}_vid{vid}.pth", map_location="cpu")
                score_list.append(c_data[contender])
            oof_scores = np.concatenate(score_list)
            assert len(oof_scores) == EXPECTED_TOTAL_QUERIES

            sweep_res = run_event_driven_sweep_literal(
                train_frames, oof_scores, nonlin, label_to_cat_id, desc=f"{contender}-S{seed}"
            )

            # Invariant: Baseline TP/FP/FN/F1 must be identical across all 9 calibrations
            if reference_baseline is None:
                reference_baseline = (
                    sweep_res["baseline_tp"],
                    sweep_res["baseline_fp"],
                    sweep_res["baseline_fn"],
                    sweep_res["baseline_f1"]
                )
            else:
                assert (sweep_res["baseline_tp"], sweep_res["baseline_fp"], sweep_res["baseline_fn"]) == reference_baseline[:3], (
                    "[FATAL] Invariant violation: baseline counts differ across calibrations!"
                )
                assert abs(sweep_res["baseline_f1"] - reference_baseline[3]) < TIE_TOLERANCE, (
                    "[FATAL] Invariant violation: baseline F1 differs across calibrations!"
                )

            delta_f1 = sweep_res["f1"] - sweep_res["baseline_f1"]
            key_name = f"{contender}_seed{seed}"
            calibrated_results["thresholds"][key_name] = sweep_res

            print(
                f"{contender:<10} | {seed:<5} | {sweep_res['tau_star']:>10.4f} | {sweep_res['f1']:>13.4f} | "
                f"{sweep_res['baseline_f1']:>13.4f} | {delta_f1:>+10.4f} | {sweep_res['precision']:>10.4f} | "
                f"{sweep_res['recall']:>10.4f} | CERTIFIED",
                flush=True
            )

    print("=" * 115, flush=True)

    # Freeze threshold manifest
    with open(THRESHOLDS_JSON, "w", encoding="utf-8") as f:
        json.dump(calibrated_results, f, indent=2)

    threshold_sha256 = compute_sha256(THRESHOLDS_JSON)
    print(f"\n[+] Calibrated Threshold Manifest Frozen: {THRESHOLDS_JSON.resolve()}")
    print(f"[+] Threshold Manifest SHA-256:         {threshold_sha256}")
    print(f"[+] Total Stage B Execution Time: {(time.time() - t_start) / 60.0:.2f} minutes")
    print("=" * 115, flush=True)


if __name__ == "__main__":
    main()