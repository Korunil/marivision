"""
D:/MARIVISION/H0_workbench/eval_tools/run_marsen_lovo_stage_a.py
MARSEN Stream B2: Stage A - 141-Fold Train-Only LOVO OOF Generator (P-2149..P-2158)

Executes 141 LOVO folds (47 videos x Seeds 41, 42, 43 = 564 model fits):
  1. Provenance Gate (P-2155): Cryptographic preflight verifying census JSON and
     all 47 physical Train shard SHA-256 values against the master census ledger.
  2. Concurrency: 6 parallel worker processes with single-threaded BLAS/OpenMP clamping.
  3. Data Contract: Strictly ascending numeric video ID iteration {1..47}; Train shards only.
  4. Preprocessing: Canonical manual standardization with 1e-6 variance floor.
  5. Sampling: Deterministic balanced bootstrap with seed = 1000 * s + v, shared across all 4 heads.
  6. Architecture: sklearn MLPClassifier(hidden_layer_sizes=(64,), alpha=0.05, max_iter=350).
  7. Calibration: Fold-clean empirical CDFs F_{S0}^{(-v)} and F_D^{(-v)} fit on the 46 training
     videos to evaluate held-out S0, S35, and SG scores.
  8. Atomic Persistence & Sidecars (P-2156, P-2157): Atomic .pth commits with external .sha256 sidecars.
  9. Full Cache Validation (P-2158): Validates shape and finiteness across all 7 score vectors on resume.
  10. OOF Coverage Gate: Post-run verification of 100% (211,712 queries) coverage per seed.
Runtime: ~5 hours on 6 workers.
"""

import os

# D-4182 / D-4193: Hard thread clamping prior to numerical library imports
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.neural_network import MLPClassifier

sys.stdout.reconfigure(line_buffering=True)

OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
SHARD_DIR = OUT_DIR / "full_split_shards"
CACHE_DIR = SHARD_DIR / "lovo_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CENSUS_JSON = OUT_DIR / "marsen_b2_production_census_56shards.json"
OOF_MANIFEST_JSON = OUT_DIR / "marsen_b2_lovo_oof_manifest.json"

SEEDS = [41, 42, 43]
NUM_VIDEOS = 47
EXPECTED_TOTAL_TRAIN_QUERIES = 211712
MAX_WORKERS = 6

FROZEN_CENSUS_SHA256 = "a20a734d668a9078bdfe274cb09fa5a47ad9c6a3a9d7cef2d1f99b926623fbc5"


def compute_sha256_file(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def compute_sha256_bytes(data_bytes):
    return hashlib.sha256(data_bytes).hexdigest()


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


# Global storage for worker processes to avoid IPC pickling overhead
WORKER_TRAIN_DATA = {}
WORKER_DRIVER_SHA = None


def worker_init(shard_dir_str, driver_sha):
    """Initializes each worker process by loading the 47 Train shards once into memory."""
    global WORKER_TRAIN_DATA, WORKER_DRIVER_SHA
    WORKER_DRIVER_SHA = driver_sha
    shard_dir = Path(shard_dir_str)

    train_shards = sorted(list(shard_dir.glob("marsen_b2_shard_train_vid*.pth")))
    assert len(train_shards) == 47, f"[WORKER FATAL] Expected 47 train shards, found {len(train_shards)}"

    for p in train_shards:
        shard = torch.load(p, map_location="cpu")
        meta = shard.get("_meta", {})
        assert meta.get("split") == "train", f"[WORKER FATAL] Quarantine violation: {p.name}"
        vid = int(meta["video_id"])
        WORKER_TRAIN_DATA[vid] = shard["active_queries"]

    assert set(WORKER_TRAIN_DATA.keys()) == set(range(1, 48)), "[WORKER FATAL] Shard inventory mismatch."


def process_single_lovo_fold(seed, heldout_vid):
    """
    Executes or verifies a single (seed, heldout_vid) fold across H1, H3, H5, HD.
    Returns: (seed, heldout_vid, status_str, execution_time, summary_dict)
    """
    global WORKER_TRAIN_DATA, WORKER_DRIVER_SHA
    t_f0 = time.time()
    cache_path = CACHE_DIR / f"lovo_pred_seed{seed}_vid{heldout_vid}.pth"
    sha_path = cache_path.with_suffix(".pth.sha256")

    val_queries = WORKER_TRAIN_DATA[heldout_vid]
    expected_keys = [(int(q["video_id"]), int(q["image_id"]), int(q["query_id"])) for q in val_queries]

    # Pre-derive deterministic bootstrap seed and indices
    boot_seed = seed * 1000 + heldout_vid
    train_queries = []
    for vid in sorted(WORKER_TRAIN_DATA.keys()):
        if vid != heldout_vid:
            train_queries.extend(WORKER_TRAIN_DATA[vid])

    y_train = np.array([int(q["label_op_util"]) for q in train_queries], dtype=np.int32)
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]

    rng = np.random.default_rng(boot_seed)
    resampled_pos_idx = rng.choice(pos_idx, size=len(neg_idx), replace=True)
    balanced_boot_indices = np.concatenate([neg_idx, resampled_pos_idx])
    rng.shuffle(balanced_boot_indices)
    y_train_boot = y_train[balanced_boot_indices]

    boot_sha = compute_sha256_bytes(balanced_boot_indices.tobytes())

    # P-2157 & P-2158: Cryptographically validated resumption check
    if cache_path.exists() and sha_path.exists():
        try:
            expected_cache_sha = sha_path.read_text(encoding="utf-8").strip()
            actual_cache_sha = compute_sha256_file(cache_path)

            if actual_cache_sha == expected_cache_sha:
                cached = torch.load(cache_path, map_location="cpu")
                score_names = ["p1", "p3", "p5", "pD", "S0", "S35", "SG"]
                scores_valid = all(
                    k in cached and
                    np.asarray(cached[k]).shape == (len(val_queries),) and
                    np.all(np.isfinite(np.asarray(cached[k])))
                    for k in score_names
                )
                is_valid = (
                    scores_valid and
                    cached.get("seed") == seed and
                    cached.get("heldout_video") == heldout_vid and
                    cached.get("census_sha256") == FROZEN_CENSUS_SHA256 and
                    cached.get("driver_sha256") == WORKER_DRIVER_SHA and
                    cached.get("bootstrap_sha256") == boot_sha and
                    cached.get("query_keys") == expected_keys
                )
                if is_valid:
                    return (seed, heldout_vid, "CACHED", 0.0, {
                        "seed": seed, "heldout_vid": heldout_vid,
                        "n_queries": len(val_queries),
                        "n_iter": cached.get("n_iter"),
                        "loss": cached.get("loss")
                    })
        except Exception:
            pass  # Fall through to recompute on any error

    # Feature definitions
    head_specs = [
        ("H1", "1x_gem4"),
        ("H3", "3x_gem4"),
        ("H5", "5x_gap"),
        ("HD", "decoder_256")
    ]

    p_train_scores = {}
    p_val_scores = {}
    scaler_hashes = {}
    n_iter_dict = {}
    loss_dict = {}
    fit_time_dict = {}

    for head_name, feat_key in head_specs:
        t_h0 = time.time()
        X_train_raw = np.array([
            np.concatenate([np.asarray(q[feat_key], dtype=np.float32), np.asarray(q["features"][:15], dtype=np.float32)])
            for q in train_queries
        ], dtype=np.float32)

        X_val_raw = np.array([
            np.concatenate([np.asarray(q[feat_key], dtype=np.float32), np.asarray(q["features"][:15], dtype=np.float32)])
            for q in val_queries
        ], dtype=np.float32)

        mu, sd = fit_fold_standardizer(X_train_raw)
        scaler_hashes[head_name] = {
            "mu": compute_sha256_bytes(mu.tobytes()),
            "sd": compute_sha256_bytes(sd.tobytes())
        }

        X_train_scaled = (X_train_raw - mu) / sd
        X_val_scaled = (X_val_raw - mu) / sd

        X_train_boot = X_train_scaled[balanced_boot_indices]

        clf = MLPClassifier(
            hidden_layer_sizes=(64,),
            activation="relu",
            solver="adam",
            alpha=0.05,
            learning_rate_init=1e-3,
            max_iter=350,
            early_stopping=False,
            tol=1e-4,
            random_state=seed
        )
        clf.fit(X_train_boot, y_train_boot)
        t_fit = time.time() - t_h0

        p_train = clf.predict_proba(X_train_scaled)[:, 1]
        p_val = clf.predict_proba(X_val_scaled)[:, 1]

        p_train_scores[head_name] = p_train
        p_val_scores[head_name] = p_val
        n_iter_dict[head_name] = int(clf.n_iter_)
        loss_dict[head_name] = float(clf.loss_)
        fit_time_dict[head_name] = round(t_fit, 2)

    # Fold-clean score & CDF transformation
    s0_train = 0.5 * p_train_scores["H1"] + 0.5 * p_train_scores["H5"]
    s0_val = 0.5 * p_val_scores["H1"] + 0.5 * p_val_scores["H5"]
    s35_val = 0.5 * p_val_scores["H3"] + 0.5 * p_val_scores["H5"]

    cdf_s0_val = empirical_cdf_transform(s0_train, s0_val)
    cdf_d_val = empirical_cdf_transform(p_train_scores["HD"], p_val_scores["HD"])
    sg_val = 0.5 * cdf_s0_val + 0.5 * cdf_d_val

    cdf_hashes = {
        "S0": compute_sha256_bytes(s0_train.tobytes()),
        "D": compute_sha256_bytes(p_train_scores["HD"].tobytes())
    }

    fold_wall = time.time() - t_f0

    payload = {
        "seed": seed,
        "heldout_video": heldout_vid,
        "query_keys": expected_keys,
        "p1": p_val_scores["H1"].astype(np.float32),
        "p3": p_val_scores["H3"].astype(np.float32),
        "p5": p_val_scores["H5"].astype(np.float32),
        "pD": p_val_scores["HD"].astype(np.float32),
        "S0": s0_val.astype(np.float32),
        "S35": s35_val.astype(np.float32),
        "SG": sg_val.astype(np.float32),
        "bootstrap_sha256": boot_sha,
        "census_sha256": FROZEN_CENSUS_SHA256,
        "driver_sha256": WORKER_DRIVER_SHA,
        "scaler_sha256": scaler_hashes,
        "cdf_ref_sha256": cdf_hashes,
        "n_iter": n_iter_dict,
        "loss": loss_dict,
        "fit_time": fit_time_dict,
        "fold_wall_time": round(fold_wall, 2)
    }

    # P-2156 & P-2157: Atomic save and SHA sidecar persistence
    tmp_path = cache_path.with_suffix(".pth.tmp")
    tmp_sha_path = sha_path.with_suffix(".sha256.tmp")

    torch.save(payload, tmp_path)
    os.replace(tmp_path, cache_path)

    cache_sha = compute_sha256_file(cache_path)
    tmp_sha_path.write_text(cache_sha, encoding="utf-8")
    os.replace(tmp_sha_path, sha_path)

    summary = {
        "seed": seed,
        "heldout_vid": heldout_vid,
        "n_queries": len(val_queries),
        "n_iter": n_iter_dict,
        "loss": loss_dict,
        "fit_time": fit_time_dict,
        "fold_wall_time": round(fold_wall, 2)
    }
    return (seed, heldout_vid, "COMPLETED", fold_wall, summary)


def main():
    t_start = time.time()
    driver_sha256 = compute_sha256_file(Path(__file__))

    print("=" * 115, flush=True)
    print("MARSEN STREAM B2: STAGE A - 141-FOLD TRAIN-ONLY LOVO OOF GENERATOR (P-2149..P-2158)", flush=True)
    print(f"Driver Script SHA-256: {driver_sha256}", flush=True)
    print(f"Master Census Ledger:  {FROZEN_CENSUS_SHA256}", flush=True)
    print(f"Execution Target:      141 LOVO Folds (47 videos x Seeds 41, 42, 43 = 564 model fits)", flush=True)
    print(f"Process Concurrency:   {MAX_WORKERS} Workers (Single-Threaded BLAS/OpenMP)", flush=True)
    print("=" * 115, flush=True)

    assert SHARD_DIR.exists(), f"[FATAL] Missing shard directory at {SHARD_DIR}"

    # P-2155: Physical Shard Provenance Gate
    print("[*] Running Physical Shard Provenance Gate (P-2155)...", flush=True)
    assert CENSUS_JSON.exists(), f"[FATAL] Missing census JSON at {CENSUS_JSON}"
    actual_census_sha = compute_sha256_file(CENSUS_JSON)
    assert actual_census_sha == FROZEN_CENSUS_SHA256, (
        f"[FATAL] Census ledger SHA mismatch: expected {FROZEN_CENSUS_SHA256}, got {actual_census_sha}"
    )

    with open(CENSUS_JSON, "r", encoding="utf-8") as f:
        census = json.load(f)
    frozen_shard_ledger = census["shard_sha256_ledger"]

    for vid in range(1, 48):
        shard_file = SHARD_DIR / f"marsen_b2_shard_train_vid{vid}.pth"
        assert shard_file.exists(), f"[FATAL] Missing train shard {shard_file.name}"
        actual_shard_sha = compute_sha256_file(shard_file)
        expected_shard_sha = frozen_shard_ledger.get(shard_file.name)
        assert actual_shard_sha == expected_shard_sha, (
            f"[FATAL] Train shard SHA drift for {shard_file.name}:\n"
            f"  Expected: {expected_shard_sha}\n"
            f"  Actual:   {actual_shard_sha}"
        )

    print("    [+] Master census JSON verified against FROZEN_CENSUS_SHA256.")
    print("    [+] All 47 physical Train shard SHA-256 values verified against census ledger.\n", flush=True)

    # Build fold queue: 141 folds
    task_queue = []
    for s in SEEDS:
        for v in range(1, NUM_VIDEOS + 1):
            task_queue.append((s, v))

    print(f"[*] Dispatching {len(task_queue)} LOVO folds to {MAX_WORKERS}-process worker pool...", flush=True)
    print(f"[*] Cache Directory: {CACHE_DIR.resolve()}\n", flush=True)

    completed_count = 0
    total_tasks = len(task_queue)
    fold_summaries = []

    t_pool_start = time.time()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=worker_init, initargs=(str(SHARD_DIR), driver_sha256)) as executor:
        futures = {executor.submit(process_single_lovo_fold, s, v): (s, v) for s, v in task_queue}

        for future in as_completed(futures):
            s, v = futures[future]
            try:
                seed, vid, status, wall_t, smry = future.result()
                completed_count += 1
                fold_summaries.append(smry)

                n_q = smry.get("n_queries", 0)
                h1_it = smry.get("n_iter", {}).get("H1", -1)
                hd_it = smry.get("n_iter", {}).get("HD", -1)
                hd_t = smry.get("fit_time", {}).get("HD", -1.0)

                elapsed_pool = time.time() - t_pool_start
                rate = completed_count / (elapsed_pool / 60.0) if elapsed_pool > 0 else 0.0
                remaining = (total_tasks - completed_count) / rate if rate > 0 else 0.0

                print(
                    f"[{completed_count:>3}/{total_tasks}] Seed {seed} | Vid {vid:>2} | {n_q:>5} Qs | "
                    f"H1 it={h1_it:>3} | HD it={hd_it:>3} ({hd_t:>5.1f}s) | Fold Wall: {wall_t:>6.1f}s | "
                    f"Status: {status:<9} | Rem: ~{remaining:>5.1f}m",
                    flush=True
                )
            except Exception as e:
                print(f"[FATAL EXCEPTION] Fold Seed {s} Vid {v} failed: {e}", flush=True)
                raise e

    print("\n" + "=" * 115, flush=True)
    print("STAGE A: 141-FOLD PARALLEL EXECUTION FINISHED", flush=True)
    print(f"Elapsed Pool Time: {(time.time() - t_pool_start) / 3600.0:.2f} hours ({(time.time() - t_pool_start) / 60.0:.1f} minutes)", flush=True)
    print("=" * 115, flush=True)

    # =========================================================================
    # OOF COVERAGE GATE (P-2151)
    # =========================================================================
    print("\n" + "=" * 115, flush=True)
    print("STAGE A: OOF COVERAGE & DATASET INTEGRITY GATE (P-2151)", flush=True)
    print("=" * 115, flush=True)

    from sklearn.metrics import average_precision_score

    all_train_queries = []
    for vid in range(1, NUM_VIDEOS + 1):
        p = SHARD_DIR / f"marsen_b2_shard_train_vid{vid}.pth"
        shard = torch.load(p, map_location="cpu")
        all_train_queries.extend(shard["active_queries"])

    assert len(all_train_queries) == EXPECTED_TOTAL_TRAIN_QUERIES, (
        f"[FATAL] Train query inventory mismatch: {len(all_train_queries)} != {EXPECTED_TOTAL_TRAIN_QUERIES}"
    )

    y_train_canonical = np.array([int(q["label_op_util"]) for q in all_train_queries], dtype=np.int32)
    canonical_keys = [(int(q["video_id"]), int(q["image_id"]), int(q["query_id"])) for q in all_train_queries]

    manifest_data = {
        "_meta": {
            "stage": "A_LOVO_OOF_GENERATION",
            "driver_sha256": driver_sha256,
            "census_sha256": FROZEN_CENSUS_SHA256,
            "total_folds": 141,
            "seeds": SEEDS,
            "total_queries_per_seed": EXPECTED_TOTAL_TRAIN_QUERIES,
            "execution_time_seconds": round(time.time() - t_start, 2)
        },
        "diagnostic_ap": {},
        "fold_artifacts": {}
    }

    for seed in SEEDS:
        print(f"\n[*] Auditing OOF Coverage for Seed {seed} across 47 Videos...", flush=True)
        assembled_keys = []
        p1_list, p3_list, p5_list, pD_list = [], [], [], []
        s0_list, s35_list, sg_list = [], [], []

        for vid in range(1, NUM_VIDEOS + 1):
            cache_file = CACHE_DIR / f"lovo_pred_seed{seed}_vid{vid}.pth"
            sha_file = cache_file.with_suffix(".pth.sha256")
            assert cache_file.exists(), f"[FATAL] Missing fold cache: {cache_file.name}"
            assert sha_file.exists(), f"[FATAL] Missing fold SHA sidecar: {sha_file.name}"

            actual_sha = compute_sha256_file(cache_file)
            expected_sha = sha_file.read_text(encoding="utf-8").strip()
            assert actual_sha == expected_sha, (
                f"[FATAL] Shard cache corruption detected in {cache_file.name}!"
            )

            c_data = torch.load(cache_file, map_location="cpu")
            assert c_data["seed"] == seed
            assert c_data["heldout_video"] == vid
            assert c_data["census_sha256"] == FROZEN_CENSUS_SHA256

            assembled_keys.extend(c_data["query_keys"])
            p1_list.append(c_data["p1"])
            p3_list.append(c_data["p3"])
            p5_list.append(c_data["p5"])
            pD_list.append(c_data["pD"])
            s0_list.append(c_data["S0"])
            s35_list.append(c_data["S35"])
            sg_list.append(c_data["SG"])

            manifest_data["fold_artifacts"][f"seed{seed}_vid{vid}"] = {
                "sha256": actual_sha,
                "bootstrap_sha256": c_data["bootstrap_sha256"],
                "fold_wall_time": c_data["fold_wall_time"]
            }

        assert len(assembled_keys) == EXPECTED_TOTAL_TRAIN_QUERIES, (
            f"[FATAL] Assembled key count mismatch for Seed {seed}: {len(assembled_keys)} != {EXPECTED_TOTAL_TRAIN_QUERIES}"
        )
        assert assembled_keys == canonical_keys, (
            f"[FATAL] Ordered query key sequence mismatch for Seed {seed}!"
        )

        p1_arr = np.concatenate(p1_list)
        p3_arr = np.concatenate(p3_list)
        p5_arr = np.concatenate(p5_list)
        pD_arr = np.concatenate(pD_list)
        s0_arr = np.concatenate(s0_list)
        s35_arr = np.concatenate(s35_list)
        sg_arr = np.concatenate(sg_list)

        for arr_name, arr in [("p1", p1_arr), ("p3", p3_arr), ("p5", p5_arr), ("pD", pD_arr), ("S0", s0_arr), ("S35", s35_arr), ("SG", sg_arr)]:
            assert arr.shape == (EXPECTED_TOTAL_TRAIN_QUERIES,), f"[FATAL] Shape mismatch for {arr_name}"
            assert np.all(np.isfinite(arr)), f"[FATAL] Non-finite values in {arr_name}"

        # Diagnostic-only pooled AP (operational utility)
        ap_h1 = average_precision_score(y_train_canonical, p1_arr)
        ap_h3 = average_precision_score(y_train_canonical, p3_arr)
        ap_h5 = average_precision_score(y_train_canonical, p5_arr)
        ap_hD = average_precision_score(y_train_canonical, pD_arr)
        ap_s0 = average_precision_score(y_train_canonical, s0_arr)
        ap_s35 = average_precision_score(y_train_canonical, s35_arr)
        ap_sG = average_precision_score(y_train_canonical, sg_arr)

        manifest_data["diagnostic_ap"][str(seed)] = {
            "H1": round(float(ap_h1), 6),
            "H3": round(float(ap_h3), 6),
            "H5": round(float(ap_h5), 6),
            "HD": round(float(ap_hD), 6),
            "S0": round(float(ap_s0), 6),
            "S35": round(float(ap_s35), 6),
            "SG": round(float(ap_sG), 6)
        }

        print(f"[+] Seed {seed} Verified: 100% (211,712/211,712) complete coverage and exact key parity.", flush=True)
        print(f"    - Diagnostic OOF AP (Utility): H1={ap_h1:.4f} | H3={ap_h3:.4f} | H5={ap_h5:.4f} | HD={ap_hD:.4f}")
        print(f"    - Diagnostic OOF AP (Contenders): S0={ap_s0:.4f} | S35={ap_s35:.4f} | SG={ap_sG:.4f}")

    # Freeze Stage A Manifest
    with open(OOF_MANIFEST_JSON, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    manifest_sha256 = compute_sha256_file(OOF_MANIFEST_JSON)
    print("\n" + "=" * 115, flush=True)
    print("STAGE A: 141-FOLD LOVO OOF GENERATION COMPLETE & CERTIFIED", flush=True)
    print(f"[+] OOF Manifest Frozen: {OOF_MANIFEST_JSON.resolve()}", flush=True)
    print(f"[+] Manifest SHA-256:    {manifest_sha256}", flush=True)
    print("[+] All 141 fold predictions verified, finite, and frozen.", flush=True)
    print("Stage B (Threshold Sweep) and Stage C (Full-Train Fits) remain queued.", flush=True)
    print(f"Total Stage A Runtime: {(time.time() - t_start) / 3600.0:.2f} hours ({(time.time() - t_start) / 60.0:.1f} minutes)", flush=True)
    print("=" * 115, flush=True)


if __name__ == "__main__":
    main()