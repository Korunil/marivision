"""
D:/MARIVISION/H0_workbench/eval_tools/audit_marsen_six_cues_screen.py
MARSEN Stream A: Standardized Six-Cue Design Screening & Shuffled Controls (v3, P-1999, P-2000)

Strict Compliance Checklist:
  - P-1999: Ingests certified frozen Base OOF predictions from marsen_b2d_oof_predictions_1915.pth. Zero Base retraining.
  - P-2000: Trains each cue model once per fold/seed (18 parallel tasks); evaluates authentic and shuffled test outputs.
  - P-1988: Evaluates within-video shuffled control to distinguish query-aligned signal from capacity inflation.
  - P-1986: Strictly enforces per-seed survival: (auth > base) AND (auth > shuff) across all 3 seeds.
  - P-1994: Plain dictionary serialization preventing PicklingError during torch.save().
  - P-1997: Fatal assertions on cue tensor completeness, shapes (1915, D), and finiteness.
"""

import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import time
import hashlib
from PIL import Image
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import average_precision_score
from sklearn.neural_network import MLPClassifier

OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
SPLIT_MANIFEST_JSON = OUT_DIR / "pol2_candidate2m_micro_split.json"
QUERY_DATASET_PTH = OUT_DIR / "pol3v_design_queries_dataset_v7.pth"
CACHED_BRANCHES_PTH = OUT_DIR / "marsen_micro_authentic_branches_1915.pth"
B2D_OOF_PTH = OUT_DIR / "marsen_b2d_oof_predictions_1915.pth"

RESULTS_JSON = OUT_DIR / "marsen_six_cues_screen_results.json"
OOF_CUES_PTH = OUT_DIR / "marsen_six_cues_oof_predictions_1915.pth"

EXPECTED_DESIGN_VIDS = [4, 7, 16, 19, 25, 29, 30, 32, 35, 47]
SEEDS = [41, 42, 43]
EXPECTED_CUES = {
    "Decoder-Global",
    "Memory",
    "Motion",
    "Scene",
    "Horizon",
    "Reflection"
}


def compute_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def evaluate_cue_seed_lovo(task_args):
    cue_name, cue_feat, s, oof_base_s, feat_meta, y_act, v_act, design_vids = task_args
    n_samples = len(y_act)

    oof_cue_auth = np.full(n_samples, np.nan)
    oof_cue_shuff = np.full(n_samples, np.nan)

    for vid in design_vids:
        test_mask = (v_act == vid)
        train_mask = ~test_mask

        cue_tr = cue_feat[train_mask]
        cue_te_auth = cue_feat[test_mask]
        cue_te_shuff = cue_feat[test_mask].copy()

        # Shuffled control: permute cue features strictly within held-out video
        rng_shuff = np.random.default_rng(s * 777 + vid)
        if len(cue_te_shuff) > 1:
            cue_te_shuff = rng_shuff.permutation(cue_te_shuff)

        Xk_tr = np.concatenate([cue_tr, feat_meta[train_mask]], axis=-1)
        Xk_te_auth = np.concatenate([cue_te_auth, feat_meta[test_mask]], axis=-1)
        Xk_te_shuff = np.concatenate([cue_te_shuff, feat_meta[test_mask]], axis=-1)

        # Standardize using training fold statistics only
        mk_tr, sk_tr = Xk_tr.mean(axis=0), Xk_tr.std(axis=0)
        sk_tr[sk_tr < 1e-6] = 1.0
        Xk_tr_std = (Xk_tr - mk_tr) / sk_tr
        Xk_te_auth_std = (Xk_te_auth - mk_tr) / sk_tr
        Xk_te_shuff_std = (Xk_te_shuff - mk_tr) / sk_tr

        y_tr = y_act[train_mask]

        # Balanced bootstrap oversampling matching canonical MLP protocol
        pos_idx = np.where(y_tr == 1)[0]
        neg_idx = np.where(y_tr == 0)[0]
        rng = np.random.default_rng(s * 1000 + vid)
        bal_idx = np.concatenate([neg_idx, rng.choice(pos_idx, size=len(neg_idx), replace=True)])
        rng.shuffle(bal_idx)

        # Fit single cue model per fold
        clf_k = MLPClassifier(hidden_layer_sizes=(64,), activation="relu", alpha=0.05, max_iter=350, random_state=s)
        clf_k.fit(Xk_tr_std[bal_idx], y_tr[bal_idx])

        oof_cue_auth[test_mask] = clf_k.predict_proba(Xk_te_auth_std)[:, 1]
        oof_cue_shuff[test_mask] = clf_k.predict_proba(Xk_te_shuff_std)[:, 1]

    # Evaluate late probability fusions using certified frozen Base vector
    oof_fused_auth = 0.5 * oof_base_s + 0.5 * oof_cue_auth
    oof_fused_shuff = 0.5 * oof_base_s + 0.5 * oof_cue_shuff

    ap_base = float(average_precision_score(y_act, oof_base_s))
    ap_auth = float(average_precision_score(y_act, oof_fused_auth))
    ap_shuff = float(average_precision_score(y_act, oof_fused_shuff))

    return {
        "cue_name": cue_name,
        "seed": s,
        "ap_base": ap_base,
        "ap_auth": ap_auth,
        "ap_shuff": ap_shuff,
        "delta_ap_vs_base": ap_auth - ap_base,
        "signal_lift_vs_shuff": ap_auth - ap_shuff,
        "oof_base": oof_base_s,
        "oof_cue_auth": oof_cue_auth,
        "oof_cue_shuff": oof_cue_shuff,
        "oof_fused_auth": oof_fused_auth,
        "oof_fused_shuff": oof_fused_shuff
    }


def main():
    t0 = time.time()
    driver_sha256 = compute_sha256(Path(__file__))
    print("=" * 115)
    print("MARSEN STREAM A: STANDARDIZED SIX-CUE SCREENING & SHUFFLED CONTROLS (v3, P-1999, P-2000)")
    print(f"Driver Script SHA-256: {driver_sha256}")
    print("=" * 115)

    assert SPLIT_MANIFEST_JSON.exists(), "[FATAL] Missing split manifest!"
    assert QUERY_DATASET_PTH.exists(), "[FATAL] Missing design queries dataset!"
    assert CACHED_BRANCHES_PTH.exists(), "[FATAL] Missing cached branches bundle!"
    assert B2D_OOF_PTH.exists(), "[FATAL] Missing B2d frozen OOF predictions cache!"

    with open(SPLIT_MANIFEST_JSON, "r", encoding="utf-8") as f:
        design_vids = sorted(json.load(f)["_meta"]["train_videos"])
    assert design_vids == EXPECTED_DESIGN_VIDS

    cached_queries = torch.load(QUERY_DATASET_PTH, map_location="cpu")["extracted_queries"]
    active_queries = [q for q in cached_queries if q["is_active"]]
    n_active = len(active_queries)
    y_act = np.array([q["label_op_util"] for q in active_queries], dtype=np.int32)
    v_act = np.array([q["video_id"] for q in active_queries], dtype=np.int32)
    assert n_active == 1915 and int(y_act.sum()) == 92

    # P-1999: Load Certified Frozen Base (1.0x_GeM4) OOF Vectors
    print("[*] Ingesting Certified Frozen Base (1.0x_GeM4) OOF Vectors from B2d...")
    b2d_bundle = torch.load(B2D_OOF_PTH, map_location="cpu")
    frozen_base_oof = {
        s: np.asarray(b2d_bundle["oof_preds"]["1.0x_gem_p4"][s], dtype=np.float64)
        for s in SEEDS
    }
    for s in SEEDS:
        assert frozen_base_oof[s].shape == (1915,), f"[FATAL] Base OOF shape mismatch on seed {s}"
        assert np.isfinite(frozen_base_oof[s]).all(), f"[FATAL] Non-finite values in Base OOF seed {s}"
        base_ap_s = average_precision_score(y_act, frozen_base_oof[s])
        print(f"  - Certified Base Seed {s}: AP = {base_ap_s:.4f}")

    branch_bundle = torch.load(CACHED_BRANCHES_PTH, map_location="cpu")
    branches = branch_bundle["branches"]
    feat_meta = branches["metadata"][:, :15].astype(np.float32)
    assert feat_meta.shape == (1915, 15) and np.isfinite(feat_meta).all()

    # Load and validate all 6 cues
    cue_registry = {}
    key_mapping = [
        ("global", "Decoder-Global"),
        ("memory", "Memory"),
        ("motion", "Motion"),
        ("scene", "Scene"),
        ("horizon", "Horizon"),
        ("reflection", "Reflection")
    ]
    for key, name in key_mapping:
        assert key in branches, f"[FATAL] Required cue key '{key}' not found in cached branches bundle!"
        arr = branches[key].astype(np.float32)
        assert arr.shape[0] == n_active, f"[FATAL] Row count mismatch in cue {name}: {arr.shape[0]} != {n_active}"
        assert np.isfinite(arr).all(), f"[FATAL] Non-finite values in cue {name}!"
        cue_registry[name] = arr
        print(f"[+] Loaded & Validated Cue '{name}': Shape = {arr.shape}")

    assert set(cue_registry.keys()) == EXPECTED_CUES, f"[FATAL] Cue registry mismatch: {set(cue_registry.keys())}"

    # Build 18 parallel tasks (6 cues × 3 seeds)
    tasks = []
    for c_name, c_feat in cue_registry.items():
        for s in SEEDS:
            tasks.append((c_name, c_feat, s, frozen_base_oof[s], feat_meta, y_act, v_act, design_vids))

    n_workers = int(os.environ.get("MARSEN_WORKERS", 10))
    print(f"\n[*] Dispatching {len(tasks)} LOVO evaluation tasks across {n_workers} CPU workers...")
    t_disp = time.time()

    task_results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for res in executor.map(evaluate_cue_seed_lovo, tasks):
            task_results.append(res)
    print(f"[+] All 18 Cue Models Fitted and Evaluated in {time.time() - t_disp:.2f}s")

    # Organize results by cue and seed
    results_by_cue = defaultdict(dict)
    oof_persistence_map = defaultdict(dict)

    for r in task_results:
        c_name = r["cue_name"]
        s = r["seed"]
        results_by_cue[c_name][s] = r
        oof_persistence_map[c_name][s] = {
            "oof_base": r["oof_base"],
            "oof_cue_auth": r["oof_cue_auth"],
            "oof_cue_shuff": r["oof_cue_shuff"],
            "oof_fused_auth": r["oof_fused_auth"],
            "oof_fused_shuff": r["oof_fused_shuff"]
        }

    # P-1994: Plain dictionary serialization
    oof_serializable = {
        cue: {s: oof_persistence_map[cue][s] for s in SEEDS}
        for cue in oof_persistence_map
    }
    torch.save({
        "driver_sha256": driver_sha256,
        "oof_preds": oof_serializable
    }, OOF_CUES_PTH)
    print(f"[+] P-1991/P-1994 Certified: All Cue OOF Vectors Frozen: {OOF_CUES_PTH.resolve()}")

    print("\n" + "=" * 115)
    print("SIX-CUE SCREENING SYNTHESIS: AUTHENTIC VS. SHUFFLED CONTROL COMPARISON")
    print("=" * 115)
    print(f"{'Maritime Cue Name':<18} | {'Base AP':<8} | {'Auth AP':<8} | {'dAP (Auth)':<11} | {'Shuff AP':<9} | {'Signal Lift':<12} | {'Seed Deltas (Auth vs Base)':<28} | {'Seed Signal Lifts'}")
    print("-" * 125)

    synthesis_report = []
    survivors = []

    for c_name in sorted(cue_registry.keys()):
        base_aps = [results_by_cue[c_name][s]["ap_base"] for s in SEEDS]
        auth_aps = [results_by_cue[c_name][s]["ap_auth"] for s in SEEDS]
        shuff_aps = [results_by_cue[c_name][s]["ap_shuff"] for s in SEEDS]

        m_base = float(np.mean(base_aps))
        m_auth = float(np.mean(auth_aps))
        m_shuff = float(np.mean(shuff_aps))

        d_auth = m_auth - m_base
        signal_lift = m_auth - m_shuff

        seed_deltas = [results_by_cue[c_name][s]["delta_ap_vs_base"] for s in SEEDS]
        seed_signal_lifts = [results_by_cue[c_name][s]["signal_lift_vs_shuff"] for s in SEEDS]

        # Strict Per-Seed Criterion: All seeds must improve over Base AND beat Shuffled control
        all_seeds_positive = all(d > 0.0 for d in seed_deltas)
        all_seeds_beat_shuffle = all(l > 0.0 for l in seed_signal_lifts)
        survived = all_seeds_positive and all_seeds_beat_shuffle

        if survived:
            survivors.append(c_name)

        status = " <-- ADVANCES TO STAGE 2" if survived else ""
        deltas_str = f"[{seed_deltas[0]:+.4f}, {seed_deltas[1]:+.4f}, {seed_deltas[2]:+.4f}]"
        lifts_str = f"[{seed_signal_lifts[0]:+.4f}, {seed_signal_lifts[1]:+.4f}, {seed_signal_lifts[2]:+.4f}]"

        print(f"{c_name:<18} | {m_base:6.4f}   | {m_auth:6.4f}   | {d_auth:+9.4f}   | {m_shuff:7.4f}   | {signal_lift:+10.4f}   | {deltas_str:<28} | {lifts_str}{status}")

        synthesis_report.append({
            "cue_name": c_name,
            "mean_base_ap": m_base, "mean_auth_ap": m_auth, "mean_shuff_ap": m_shuff,
            "d_auth": d_auth, "signal_lift": signal_lift,
            "seed_deltas_auth_vs_base": seed_deltas,
            "seed_signal_lifts_auth_vs_shuff": seed_signal_lifts,
            "all_seeds_positive": all_seeds_positive,
            "all_seeds_beat_shuffle": all_seeds_beat_shuffle,
            "survived": survived
        })

    print("\n" + "=" * 115)
    print("STAGE 1 CUE SCREENING VERDICT (P-1988)")
    print("=" * 115)
    if len(survivors) > 0:
        print(f"[+] Qualifying Cues for Non-Redundancy Test against B_S: {survivors}")
    else:
        print("[-] ZERO cues met the strict per-seed survival criteria across all three seeds.")
        print("    All tested cues either failed to improve over B_0 or failed to outperform their capacity-matched controls.")
        print("    Conclusion: Auxiliary maritime cues are closed. Spatial verifier B_S is finalized as MARSEN-Verify.")
    print("=" * 115)

    save_payload = {
        "_meta": {
            "artifact_name": "marsen_six_cues_screen_v3",
            "driver_sha256": driver_sha256,
            "surviving_cues": survivors,
            "total_seconds": float(time.time() - t0)
        },
        "cues_synthesis": synthesis_report
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(save_payload, f, indent=2)

    res_hash = compute_sha256(RESULTS_JSON)
    print(f"[+] Screening Ledger Frozen: {RESULTS_JSON.resolve()} (SHA-256: {res_hash})")
    print(f"    Total Runtime: {time.time() - t0:.2f}s")
    print("=" * 115)


if __name__ == "__main__":
    main()