"""
D:/MARIVISION/H0_workbench/eval_tools/benchmark_marsen_lovo_sentinel.py
MARSEN Stream B2: Canonical LOVO Runtime Sentinel Benchmark (P-2136..P-2143)

Executes one canonical LOVO fold (Fold 1: held-out Vid 1, training Vids 2..47)
under Seed 41 across all four verifier heads (H1, H3, H5, HD):
  1. Thread Clamping: Clamps BLAS and OpenMP to 1 thread prior to numerical imports.
  2. Train-Only Quarantine: Asserts all ingested shards have split == "train".
  3. Canonical Numeric Row Ordering: Strict ascending numeric video ID iteration {1..47}.
  4. Canonical Manual Standardization: Mean and std with 1e-6 variance floor.
  5. Canonical Bootstrap RNG: default_rng(seed * 1000 + vid) oversampling.
  6. Shared Bootstrap Ledger: Identical resampled index ledger across H1, H3, H5, HD.
  7. Canonical MLP: sklearn MLPClassifier(hidden_layer_sizes=(64,), alpha=0.05, max_iter=350).
  8. Granular Profiling: Measures t_pre, t_fit, t_pred, and end-to-end fold wall time.
  9. Fold-Clean SG: Empirical CDF fit strictly on 46 training-video scores.
"""

import os

# D-4182: Clamp BLAS and OpenMP threading prior to numerical library imports
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

OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
SHARD_DIR = OUT_DIR / "full_split_shards"

BENCHMARK_SEED = 41
BENCHMARK_VID = 1  # Held-out video ID


def compute_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def fit_fold_standardizer(X_tr):
    """D-4181: Canonical fold-local standardization with 1e-6 variance floor."""
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0)
    sd[sd < 1e-6] = 1.0
    return mu, sd


def empirical_cdf_transform(ref_scores, eval_scores):
    """P-2134: Empirical CDF: F(x) = #{z_i <= x} / N via searchsorted(side='right')."""
    sorted_ref = np.sort(ref_scores)
    n = len(sorted_ref)
    return np.searchsorted(sorted_ref, eval_scores, side="right") / float(n)


def main():
    t_start = time.time()
    driver_sha256 = compute_sha256(Path(__file__))

    print("=" * 115, flush=True)
    print("MARSEN STREAM B2: CANONICAL LOVO RUNTIME SENTINEL BENCHMARK (P-2136..P-2143)", flush=True)
    print(f"Sentinel Driver SHA-256: {driver_sha256}", flush=True)
    print(f"Configuration: Fold Video={BENCHMARK_VID}, Seed={BENCHMARK_SEED}, BLAS Threads=1", flush=True)
    print("=" * 115, flush=True)

    # 1. Ingest 47 Train shards with strict quarantine
    print("[*] Ingesting 47 Official Train Shards (Hard Quarantine Check)...", flush=True)
    train_shards = sorted(list(SHARD_DIR.glob("marsen_b2_shard_train_vid*.pth")))
    assert len(train_shards) == 47, f"[FATAL] Expected 47 train shards, found {len(train_shards)}"

    train_data_by_vid = {}
    for p in train_shards:
        shard = torch.load(p, map_location="cpu")
        meta = shard.get("_meta", {})
        assert meta.get("split") == "train", f"[FATAL] Quarantine violation: {p.name}"
        vid = int(meta["video_id"])
        train_data_by_vid[vid] = shard["active_queries"]

    # P-2143: Exact numeric video inventory verification {1..47}
    assert set(train_data_by_vid.keys()) == set(range(1, 48)), (
        f"[FATAL] Train video ID inventory mismatch: {set(train_data_by_vid.keys())} != {set(range(1, 48))}"
    )

    total_active_train = sum(len(q_list) for q_list in train_data_by_vid.values())
    print(f"[+] Loaded 47 Train Shards: {total_active_train:,} total active proposals verified.", flush=True)

    # 2. Partition into 46 Training Videos and 1 Held-Out Video in numeric ascending order
    val_queries = train_data_by_vid[BENCHMARK_VID]
    train_queries = []
    for vid in sorted(train_data_by_vid.keys()):
        if vid != BENCHMARK_VID:
            train_queries.extend(train_data_by_vid[vid])

    n_train = len(train_queries)
    n_val = len(val_queries)
    print(f"[*] Fold Partition: Training={n_train:,} queries (46 vids, ascending) | Held-Out={n_val:,} queries (Vid {BENCHMARK_VID})", flush=True)

    # 3. Target Distribution and Canonical Bootstrap RNG
    y_train = np.array([int(q["label_op_util"]) for q in train_queries], dtype=np.int32)
    y_val = np.array([int(q["label_op_util"]) for q in val_queries], dtype=np.int32)

    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    prev = (n_pos / float(n_train)) * 100.0
    print(f"[*] Training Class Distribution: Negatives={n_neg:,}, Positives={n_pos:,} (Prevalence: {prev:.3f}%)", flush=True)

    # D-4180: Canonical LOVO bootstrap generator with seed = 1000 * seed + vid
    boot_seed = BENCHMARK_SEED * 1000 + BENCHMARK_VID
    rng = np.random.default_rng(boot_seed)
    resampled_pos_idx = rng.choice(pos_idx, size=n_neg, replace=True)
    balanced_boot_indices = np.concatenate([neg_idx, resampled_pos_idx])
    rng.shuffle(balanced_boot_indices)
    y_train_boot = y_train[balanced_boot_indices]

    n_boot = len(balanced_boot_indices)
    boot_hash = hashlib.sha256(balanced_boot_indices.tobytes()).hexdigest()
    print(f"[+] Canonical Bootstrap RNG Initialized (Seed={boot_seed}): {n_boot:,} balanced training rows.")
    print(f"    - Shared Bootstrap Ledger SHA-256: {boot_hash}\n", flush=True)

    # 4. Feature Branch Specifications
    head_specs = [
        ("H1", "1x_gem4", 128),
        ("H3", "3x_gem4", 128),
        ("H5", "5x_gap", 128),
        ("HD", "decoder_256", 256)
    ]

    results_table = []
    p_train_scores = {}
    p_val_scores = {}

    t_fold_start = time.time()

    print("=" * 115, flush=True)
    print(f"{'Head':<6} | {'Dim':<5} | {'Fit Samples':<12} | {'Iters':<7} | {'Loss':<10} | {'t_pre (s)':<11} | {'t_fit (s)':<11} | {'t_pred (s)':<12} | {'Total (s)'}")
    print("-" * 115, flush=True)

    for head_name, feat_key, feat_dim in head_specs:
        t_h0 = time.time()

        # Step A: Preprocessing & Standardization
        t_pre_0 = time.time()
        X_train_raw = np.array([
            np.concatenate([np.asarray(q[feat_key], dtype=np.float32), np.asarray(q["features"][:15], dtype=np.float32)])
            for q in train_queries
        ], dtype=np.float32)

        X_val_raw = np.array([
            np.concatenate([np.asarray(q[feat_key], dtype=np.float32), np.asarray(q["features"][:15], dtype=np.float32)])
            for q in val_queries
        ], dtype=np.float32)

        mu, sd = fit_fold_standardizer(X_train_raw)
        X_train_scaled = (X_train_raw - mu) / sd
        X_val_scaled = (X_val_raw - mu) / sd

        # P-2145: Reuse exact shared bootstrap ledger
        X_train_boot = X_train_scaled[balanced_boot_indices]
        t_pre = time.time() - t_pre_0

        # Step B: Model Fitting
        t_fit_0 = time.time()
        clf = MLPClassifier(
            hidden_layer_sizes=(64,),
            activation="relu",
            solver="adam",
            alpha=0.05,
            learning_rate_init=1e-3,
            max_iter=350,
            early_stopping=False,
            tol=1e-4,
            random_state=BENCHMARK_SEED
        )
        clf.fit(X_train_boot, y_train_boot)
        t_fit = time.time() - t_fit_0

        # Step C: Dual Prediction (46-Video Training CDF Pool + Held-Out Video)
        t_pred_0 = time.time()
        p_train = clf.predict_proba(X_train_scaled)[:, 1]
        p_val = clf.predict_proba(X_val_scaled)[:, 1]
        t_pred = time.time() - t_pred_0

        t_head_total = time.time() - t_h0

        p_train_scores[head_name] = p_train
        p_val_scores[head_name] = p_val

        results_table.append({
            "head": head_name,
            "dim": X_train_raw.shape[1],
            "n_boot": n_boot,
            "n_iter": clf.n_iter_,
            "loss": clf.loss_,
            "t_pre": t_pre,
            "t_fit": t_fit,
            "t_pred": t_pred,
            "t_head_total": t_head_total
        })

        print(f"{head_name:<6} | {X_train_raw.shape[1]:>5} | {n_boot:>12,} | {clf.n_iter_:>7} | {clf.loss_:>10.4f} | {t_pre:>11.2f} | {t_fit:>11.2f} | {t_pred:>12.2f} | {t_head_total:>9.2f}", flush=True)

    print("=" * 115, flush=True)

    # 5. Fold-Clean Score & CDF Transformation
    print("\n[*] Evaluating Fold-Clean Contenders on Held-Out Video 1...", flush=True)
    t_cdf_0 = time.time()

    # S0 and S35 combinations
    s0_train = 0.5 * p_train_scores["H1"] + 0.5 * p_train_scores["H5"]
    s0_val = 0.5 * p_val_scores["H1"] + 0.5 * p_val_scores["H5"]
    s35_val = 0.5 * p_val_scores["H3"] + 0.5 * p_val_scores["H5"]

    # Fold-clean SG empirical CDF transformation
    cdf_s0_val = empirical_cdf_transform(s0_train, s0_val)
    cdf_d_val = empirical_cdf_transform(p_train_scores["HD"], p_val_scores["HD"])
    sg_val = 0.5 * cdf_s0_val + 0.5 * cdf_d_val

    t_cdf = time.time() - t_cdf_0
    full_fold_wall = time.time() - t_fold_start

    print(f"    - Reference Set Size for F_S0: {len(s0_train):,} (from 46 training videos)")
    print(f"    - Reference Set Size for F_D:  {len(p_train_scores['HD']):,} (from 46 training videos)")
    print(f"    - Held-Out S0  Summary: Mean={np.mean(s0_val):.4f}, Std={np.std(s0_val):.4f}, Min={np.min(s0_val):.4f}, Max={np.max(s0_val):.4f}")
    print(f"    - Held-Out S35 Summary: Mean={np.mean(s35_val):.4f}, Std={np.std(s35_val):.4f}, Min={np.min(s35_val):.4f}, Max={np.max(s35_val):.4f}")
    print(f"    - Held-Out SG  Summary: Mean={np.mean(sg_val):.4f}, Std={np.std(sg_val):.4f}, Min={np.min(sg_val):.4f}, Max={np.max(sg_val):.4f}")
    print(f"    - CDF Transformation Time: {t_cdf:.3f}s")
    print(f"    - All transformed scores finite: {np.all(np.isfinite(sg_val))}")

    # Workload Projection
    total_folds_to_run = 47 * 3  # 141 folds (each running 4 heads = 564 model fits)
    total_serial_hours = (full_fold_wall * total_folds_to_run) / 3600.0

    print("\n" + "=" * 115, flush=True)
    print("CANONICAL LOVO SENTINEL BENCHMARK SYNTHESIS")
    print("=" * 115, flush=True)
    print(f"[*] Complete 4-Head Fold Wall Time:         {full_fold_wall:.2f}s ({full_fold_wall / 60.0:.2f} minutes)")
    print(f"[*] Projected Serial Execution (141 Folds): {total_serial_hours:.2f} hours")
    print(f"[*] Projected 4-Worker Execution:           {total_serial_hours / 4.0:.2f} hours (~{(total_serial_hours / 4.0) * 60:.1f} minutes)")
    print(f"[*] Projected 6-Worker Execution:           {total_serial_hours / 6.0:.2f} hours (~{(total_serial_hours / 6.0) * 60:.1f} minutes)")
    print(f"[+] Total Script Elapsed Time: {time.time() - t_start:.2f}s", flush=True)
    print("=" * 115, flush=True)


if __name__ == "__main__":
    main()