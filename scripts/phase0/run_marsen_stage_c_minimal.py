"""
D:/MARIVISION/H0_workbench/eval_tools/run_marsen_stage_c_minimal.py
MARSEN Stream B2: Stage C (Minimal) - Final Full-Train Fit for S0 Seed 43 (D-4255..D-4270, P-2194..P-2196)

Executes the minimal confirmatory model fitting for the sole active configuration (S0 Seed 43):
  1. Full Provenance Gate: Verifies Census JSON, all 47 Train shards, OOF Manifest,
     and Threshold Manifest against frozen SHA-256 digests.
  2. Single Threshold Authority (P-2194): Loads tau_star directly from the verified
     thresholds manifest; no duplicated hard-coded decimal constants.
  3. Query-Key Parity Gate (P-2195): Reconstructs canonical query keys and verifies exact
     ordered equality against the frozen Seed-43 OOF query ledger before fitting.
  4. Deterministic Full-Train Bootstrap: Uses rng = np.random.default_rng(43000) shared across H1 and H5.
  5. Manual Standardization: Fits standardizers (mu, sd with 1e-6 floor) on all 211,712 rows.
  6. Estimator Fitting & Convergence Gate (P-2196): Fits canonical sklearn MLPClassifier(hidden_layer_sizes=(64,),
     alpha=0.05, max_iter=350, tol=1e-4, random_state=43) on CPU. Asserts n_iter_ < 350 for both heads.
  7. Score-Scale Diagnostic: Evaluates S0 = 0.5 * p1 + 0.5 * p5 across all 211,712 Train queries
     and counts admissions at the exact manifest tau_star (read-only diagnostic, no tuning).
  8. Atomic Artifact Persistence: Saves models, scalers, parameters, and metadata to
     marsen_final_model_s0_seed43.pth with an external .sha256 sidecar.
Runtime: ~3-5 minutes on CPU.
"""

import os

# Thread clamping prior to numerical imports
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import time
import json
import hashlib
import torch
import numpy as np
from pathlib import Path
from sklearn.neural_network import MLPClassifier

sys.stdout.reconfigure(line_buffering=True)

# Path Constants
ROOT_DIR = Path("D:/MARIVISION")
OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
SHARD_DIR = OUT_DIR / "full_split_shards"
CACHE_DIR = SHARD_DIR / "lovo_cache"
MODEL_DIR = SHARD_DIR / "final_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CENSUS_JSON = OUT_DIR / "marsen_b2_production_census_56shards.json"
OOF_MANIFEST_JSON = OUT_DIR / "marsen_b2_lovo_oof_manifest.json"
THRESHOLDS_JSON = OUT_DIR / "marsen_b2_calibrated_thresholds.json"
OUTPUT_MODEL_PTH = MODEL_DIR / "marsen_final_model_s0_seed43.pth"

NUM_VIDEOS = 47
EXPECTED_TOTAL_QUERIES = 211712

FROZEN_CENSUS_SHA256 = "a20a734d668a9078bdfe274cb09fa5a47ad9c6a3a9d7cef2d1f99b926623fbc5"
FROZEN_OOF_MANIFEST_SHA256 = "1919112fe4280a5c60aa42e77a653f74b1a71905bee4bcd8552fb296b6916452"
FROZEN_THRESHOLDS_SHA256 = "615d514a0b2070e0338f9d7ba0e6b73e9f25e1b4773599f83b6203a8bf396f17"

FULL_TRAIN_SEED = 43
FULL_BOOTSTRAP_SEED = 43000


def compute_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def compute_sha256_bytes(data_bytes):
    return hashlib.sha256(data_bytes).hexdigest()


def fit_standardizer(X):
    """Canonical manual standardization with 1e-6 variance floor."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-6] = 1.0
    return mu, sd


def main():
    t_start = time.time()
    driver_sha256 = compute_sha256(Path(__file__))

    print("=" * 115, flush=True)
    print("MARSEN STREAM B2: STAGE C (MINIMAL) - FINAL FULL-TRAIN FIT FOR S0 SEED 43", flush=True)
    print(f"Driver Script SHA-256: {driver_sha256}", flush=True)
    print(f"Census Target SHA-256: {FROZEN_CENSUS_SHA256}", flush=True)
    print(f"OOF Manifest Target:   {FROZEN_OOF_MANIFEST_SHA256}", flush=True)
    print(f"Thresholds Target SHA: {FROZEN_THRESHOLDS_SHA256}", flush=True)
    print("=" * 115, flush=True)

    # 1. Full Provenance Preflight Gate
    print("[*] Running Cryptographic Provenance Preflight...", flush=True)

    assert CENSUS_JSON.exists(), f"[FATAL] Missing census JSON at {CENSUS_JSON}"
    actual_census_sha = compute_sha256(CENSUS_JSON)
    assert actual_census_sha == FROZEN_CENSUS_SHA256, (
        f"[FATAL] Census SHA mismatch: expected {FROZEN_CENSUS_SHA256}, got {actual_census_sha}"
    )

    with open(CENSUS_JSON, "r", encoding="utf-8") as f:
        census = json.load(f)
    frozen_shard_ledger = census["shard_sha256_ledger"]

    for vid in range(1, NUM_VIDEOS + 1):
        p = SHARD_DIR / f"marsen_b2_shard_train_vid{vid}.pth"
        assert p.exists(), f"[FATAL] Missing shard {p.name}"
        actual_shard_sha = compute_sha256(p)
        expected_shard_sha = frozen_shard_ledger[p.name]
        assert actual_shard_sha == expected_shard_sha, (
            f"[FATAL] Shard SHA drift on {p.name}"
        )

    assert OOF_MANIFEST_JSON.exists(), f"[FATAL] Missing OOF manifest at {OOF_MANIFEST_JSON}"
    actual_oof_sha = compute_sha256(OOF_MANIFEST_JSON)
    assert actual_oof_sha == FROZEN_OOF_MANIFEST_SHA256, (
        f"[FATAL] OOF manifest SHA mismatch: expected {FROZEN_OOF_MANIFEST_SHA256}, got {actual_oof_sha}"
    )

    assert THRESHOLDS_JSON.exists(), f"[FATAL] Missing thresholds JSON at {THRESHOLDS_JSON}"
    actual_thresh_sha = compute_sha256(THRESHOLDS_JSON)
    assert actual_thresh_sha == FROZEN_THRESHOLDS_SHA256, (
        f"[FATAL] Threshold manifest SHA mismatch: expected {FROZEN_THRESHOLDS_SHA256}, got {actual_thresh_sha}"
    )

    # P-2194: Single Threshold Authority directly from manifest
    with open(THRESHOLDS_JSON, "r", encoding="utf-8") as f:
        thresh_manifest = json.load(f)

    assert "S0_seed43" in thresh_manifest["thresholds"], "[FATAL] Missing S0_seed43 in threshold manifest"
    tau_star = float(thresh_manifest["thresholds"]["S0_seed43"]["tau_star"])
    print(f"    [+] Authoritative tau* for S0 Seed 43 loaded from manifest: {tau_star:.14f}")
    print("    [+] Census JSON, all 47 Train shards, OOF manifest, and Thresholds manifest verified bitwise intact.\n", flush=True)

    # 2. Ingest Full Train Population & Enforce Query-Key Parity (P-2195)
    print("[*] Ingesting Full 47 Train Shards in Canonical Ascending Order {1..47}...", flush=True)
    all_train_queries = []
    canonical_keys = []

    for vid in range(1, NUM_VIDEOS + 1):
        p = SHARD_DIR / f"marsen_b2_shard_train_vid{vid}.pth"
        shard = torch.load(p, map_location="cpu")
        assert shard["_meta"]["split"] == "train"
        act_q = shard["active_queries"]
        all_train_queries.extend(act_q)
        for q in act_q:
            canonical_keys.append((int(q["video_id"]), int(q["image_id"]), int(q["query_id"])))

    assert len(all_train_queries) == EXPECTED_TOTAL_QUERIES, (
        f"[FATAL] Total query count mismatch: {len(all_train_queries)} != {EXPECTED_TOTAL_QUERIES}"
    )

    # P-2195: Hard Query-Key Parity Check against frozen Seed-43 OOF Cache
    expected_seed43_oof_keys = []
    for vid in range(1, NUM_VIDEOS + 1):
        c_path = CACHE_DIR / f"lovo_pred_seed43_vid{vid}.pth"
        assert c_path.exists(), f"[FATAL] Missing Seed-43 OOF cache for Vid {vid}"
        c_data = torch.load(c_path, map_location="cpu")
        expected_seed43_oof_keys.extend(c_data["query_keys"])

    assert canonical_keys == expected_seed43_oof_keys, (
        "[FATAL] Query-key sequence mismatch between full-Train population and frozen Seed-43 OOF ledger!"
    )
    print(f"    [+] Query-key parity confirmed: exactly {len(canonical_keys):,} ordered keys matched against Seed-43 OOF ledger.\n", flush=True)

    # 3. Deterministic Shared Balanced Bootstrap
    print(f"[*] Initializing Full-Train Balanced Bootstrap (RNG Seed = {FULL_BOOTSTRAP_SEED})...", flush=True)
    y_train = np.array([int(q["label_op_util"]) for q in all_train_queries], dtype=np.int32)
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    print(f"    - Population Class Breakdown: Negatives={n_neg:,}, Positives={n_pos:,} (Prevalence: {n_pos/len(y_train)*100:.3f}%)")

    rng = np.random.default_rng(FULL_BOOTSTRAP_SEED)
    resampled_pos_idx = rng.choice(pos_idx, size=n_neg, replace=True)
    balanced_boot_indices = np.concatenate([neg_idx, resampled_pos_idx])
    rng.shuffle(balanced_boot_indices)
    y_train_boot = y_train[balanced_boot_indices]

    boot_sha256 = compute_sha256_bytes(balanced_boot_indices.tobytes())
    print(f"    - Balanced Training Rows: {len(balanced_boot_indices):,} (50.0% Positives / 50.0% Negatives)")
    print(f"    - Shared Bootstrap Ledger SHA-256: {boot_sha256}\n", flush=True)

    # 4. Feature Extraction & Manual Standardization for H1 and H5
    print("[*] Extracting and Standardizing Features for H1 (1x GeM4) and H5 (5x GAP)...", flush=True)

    X_h1_raw = np.array([
        np.concatenate([np.asarray(q["1x_gem4"], dtype=np.float32), np.asarray(q["features"][:15], dtype=np.float32)])
        for q in all_train_queries
    ], dtype=np.float32)

    X_h5_raw = np.array([
        np.concatenate([np.asarray(q["5x_gap"], dtype=np.float32), np.asarray(q["features"][:15], dtype=np.float32)])
        for q in all_train_queries
    ], dtype=np.float32)

    mu_h1, sd_h1 = fit_standardizer(X_h1_raw)
    mu_h5, sd_h5 = fit_standardizer(X_h5_raw)

    X_h1_scaled = (X_h1_raw - mu_h1) / sd_h1
    X_h5_scaled = (X_h5_raw - mu_h5) / sd_h5

    X_h1_boot = X_h1_scaled[balanced_boot_indices]
    X_h5_boot = X_h5_scaled[balanced_boot_indices]

    print("    [+] Feature matrices standardized and indexed via shared bootstrap ledger.\n", flush=True)

    # 5. Fit Estimator H1 with Convergence Gate (P-2196)
    print("[*] Fitting Canonical MLP H1 (Seed 43, alpha=0.05, max_iter=350)...", flush=True)
    t_h1_0 = time.time()
    clf_h1 = MLPClassifier(
        hidden_layer_sizes=(64,),
        activation="relu",
        solver="adam",
        alpha=0.05,
        learning_rate_init=1e-3,
        max_iter=350,
        early_stopping=False,
        tol=1e-4,
        random_state=FULL_TRAIN_SEED
    )
    clf_h1.fit(X_h1_boot, y_train_boot)
    t_h1 = time.time() - t_h1_0

    assert clf_h1.n_iter_ < 350, f"[FATAL P-2196] H1 exhausted iteration budget without converging (n_iter_={clf_h1.n_iter_})"
    print(f"    [+] H1 Fit Completed in {t_h1:.2f}s | Iterations: {clf_h1.n_iter_} / 350 | Final Loss: {clf_h1.loss_:.5f}")

    # 6. Fit Estimator H5 with Convergence Gate (P-2196)
    print("\n[*] Fitting Canonical MLP H5 (Seed 43, alpha=0.05, max_iter=350)...", flush=True)
    t_h5_0 = time.time()
    clf_h5 = MLPClassifier(
        hidden_layer_sizes=(64,),
        activation="relu",
        solver="adam",
        alpha=0.05,
        learning_rate_init=1e-3,
        max_iter=350,
        early_stopping=False,
        tol=1e-4,
        random_state=FULL_TRAIN_SEED
    )
    clf_h5.fit(X_h5_boot, y_train_boot)
    t_h5 = time.time() - t_h5_0

    assert clf_h5.n_iter_ < 350, f"[FATAL P-2196] H5 exhausted iteration budget without converging (n_iter_={clf_h5.n_iter_})"
    print(f"    [+] H5 Fit Completed in {t_h5:.2f}s | Iterations: {clf_h5.n_iter_} / 350 | Final Loss: {clf_h5.loss_:.5f}\n")

    # 7. Compute Full-Train Predictions & Evaluate Score-Scale Diagnostic
    print("[*] Predicting Full Train Population with Refitted Models...", flush=True)
    p1_train = clf_h1.predict_proba(X_h1_scaled)[:, 1].astype(np.float64)
    p5_train = clf_h5.predict_proba(X_h5_scaled)[:, 1].astype(np.float64)

    s0_train = 0.5 * p1_train + 0.5 * p5_train

    print("=" * 115, flush=True)
    print("READ-ONLY SCORE-SCALE DIAGNOSTIC ON FULL TRAIN POPULATION", flush=True)
    print("=" * 115, flush=True)
    print(f"[*] Contender S0 Refit Distribution across {len(s0_train):,} Train Queries:")
    print(f"    - Mean:   {np.mean(s0_train):.8f}")
    print(f"    - Std:    {np.std(s0_train):.8f}")
    print(f"    - Min:    {np.min(s0_train):.8f}")
    print(f"    - Median: {np.median(s0_train):.8f}")
    print(f"    - Max:    {np.max(s0_train):.8f}")

    # Evaluate admissions using authoritative manifest tau_star
    admitted_train_mask = (s0_train >= tau_star)
    n_train_admitted = int(np.count_nonzero(admitted_train_mask))
    train_admitted_pct = (n_train_admitted / len(s0_train)) * 100.0

    print(f"\n[*] Full-Train Admissions at Manifest Threshold (tau* = {tau_star:.14f}):")
    print(f"    - Admitted Proposal Count:    {n_train_admitted:,} / {len(s0_train):,}")
    print(f"    - Admitted Proposal Fraction: {train_admitted_pct:.6f}%")
    if n_train_admitted > 0:
        admitted_labels = y_train[admitted_train_mask]
        tp_rescue = int(np.sum(admitted_labels == 1))
        fp_noise = int(np.sum(admitted_labels == 0))
        print(f"    - Admitted Proposals Label Breakdown: Positives (y=1)={tp_rescue}, Negatives (y=0)={fp_noise}")
    print("    [!] Diagnostic recorded. Threshold is immutable per Decision D-4257.\n")

    # 8. Atomic Serialization with SHA-256 Sidecar
    print("[*] Serializing Minimal Stage C Model Checkpoint...", flush=True)
    payload = {
        "_meta": {
            "stage": "C_MINIMAL_FINAL_FIT",
            "driver_sha256": driver_sha256,
            "census_sha256": FROZEN_CENSUS_SHA256,
            "oof_manifest_sha256": FROZEN_OOF_MANIFEST_SHA256,
            "thresholds_sha256": FROZEN_THRESHOLDS_SHA256,
            "full_train_seed": FULL_TRAIN_SEED,
            "bootstrap_seed": FULL_BOOTSTRAP_SEED,
            "bootstrap_sha256": boot_sha256,
            "total_train_queries": EXPECTED_TOTAL_QUERIES,
            "tau_star": tau_star,
            "operating_state": "ACTIVE_GATE",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "models": {
            "clf_h1": clf_h1,
            "clf_h5": clf_h5
        },
        "scalers": {
            "h1": {"mu": mu_h1, "sd": sd_h1},
            "h5": {"mu": mu_h5, "sd": sd_h5}
        },
        "diagnostics": {
            "n_train_admitted": n_train_admitted,
            "train_admitted_pct": train_admitted_pct,
            "s0_train_max": float(np.max(s0_train)),
            "s0_train_mean": float(np.mean(s0_train)),
            "s0_train_std": float(np.std(s0_train)),
            "h1_n_iter": int(clf_h1.n_iter_),
            "h1_loss": float(clf_h1.loss_),
            "h1_fit_time_seconds": round(t_h1, 2),
            "h5_n_iter": int(clf_h5.n_iter_),
            "h5_loss": float(clf_h5.loss_),
            "h5_fit_time_seconds": round(t_h5, 2)
        }
    }

    tmp_path = OUTPUT_MODEL_PTH.with_suffix(".pth.tmp")
    sha_path = OUTPUT_MODEL_PTH.with_suffix(".pth.sha256")
    tmp_sha_path = sha_path.with_suffix(".sha256.tmp")

    torch.save(payload, tmp_path)
    os.replace(tmp_path, OUTPUT_MODEL_PTH)

    artifact_sha = compute_sha256(OUTPUT_MODEL_PTH)
    tmp_sha_path.write_text(artifact_sha, encoding="utf-8")
    os.replace(tmp_sha_path, sha_path)

    print(f"[+] Final Model Artifact Frozen: {OUTPUT_MODEL_PTH.resolve()}")
    print(f"[+] Artifact SHA-256:            {artifact_sha}")
    print(f"[+] Total Script Elapsed Time:   {time.time() - t_start:.2f}s")
    print("=" * 115, flush=True)


if __name__ == "__main__":
    main()