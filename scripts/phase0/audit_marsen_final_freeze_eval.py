"""
D:/MARIVISION/H0_workbench/eval_tools/audit_marsen_six_cues_screen.py
MARSEN Stream A: Standardized Six-Cue Design Screening & Shuffled Controls (P-1987, P-1988)

Strict Compliance Checklist:
  - Fixed Adapter Contract: Cue_k + Metadata_15 -> H64 -> p_k with 50/50 late probability fusion.
  - Compares Authentic Cue vs. Within-Video Shuffled Cue Control across Seeds 41, 42, 43.
  - Ingests cached features from marsen_micro_authentic_branches_1915.pth and marsen_pooled_descriptors_1915.pth.
  - Evaluates all 6 cues (Decoder-Global, Memory, Motion, Scene, Horizon, Reflection) in parallel via ProcessPoolExecutor.
  - Zero hyperparameter tuning: Alpha=0.05, Adam, 50/50 bootstrap oversampling.
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
CACHED_POOLED_DESCRIPTORS_PTH = OUT_DIR / "marsen_pooled_descriptors_1915.pth"
RESULTS_JSON = OUT_DIR / "marsen_six_cues_screen_results.json"

EXPECTED_DESIGN_VIDS = [4, 7, 16, 19, 25, 29, 30, 32, 35, 47]
SEEDS = [41, 42, 43]


def compute_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def evaluate_cue_seed_lovo(task_args):
    cue_name, cue_feat, is_shuffled, s, feat_1x, feat_meta, y_act, v_act, design_vids = task_args
    n_samples = len(y_act)

    oof_p1 = np.full(n_samples, np.nan)
    oof_pk = np.full(n_samples, np.nan)

    for vid in design_vids:
        test_mask = (v_act == vid)
        train_mask = ~test_mask

        # Base 1.0x feature construction
        X1_tr = np.concatenate([feat_1x[train_mask], feat_meta[train_mask]], axis=-1)
        X1_te = np.concatenate([feat_1x[test_mask], feat_meta[test_mask]], axis=-1)

        # Cue feature construction
        cue_tr = cue_feat[train_mask]
        cue_te = cue_feat[test_mask].copy()

        # Shuffled control: permute cue features strictly within held-out video
        if is_shuffled:
            rng_shuff = np.random.default_rng(s * 777 + vid)
            if len(cue_te) > 1:
                cue_te = rng_shuff.permutation(cue_te)

        Xk_tr = np.concatenate([cue_tr, feat_meta[train_mask]], axis=-1)
        Xk_te = np.concatenate([cue_te, feat_meta[test_mask]], axis=-1)

        # Standardize using training fold statistics only
        m1_tr, s1_tr = X1_tr.mean(axis=0), X1_tr.std(axis=0)
        s1_tr[s1_tr < 1e-6] = 1.0
        X1_tr_std = (X1_tr - m1_tr) / s1_tr
        X1_te_std = (X1_te - m1_tr) / s1_tr

        mk_tr, sk_tr = Xk_tr.mean(axis=0), Xk_tr.std(axis=0)
        sk_tr[sk_tr < 1e-6] = 1.0
        Xk_tr_std = (Xk_tr - mk_tr) / sk_tr
        Xk_te_std = (Xk_te - mk_tr) / sk_tr

        y_tr = y_act[train_mask]

        # Balanced bootstrap oversampling matching canonical MLP protocol
        pos_idx = np.where(y_tr == 1)[0]
        neg_idx = np.where(y_tr == 0)[0]
        rng = np.random.default_rng(s * 1000 + vid)
        bal_idx = np.concatenate([neg_idx, rng.choice(pos_idx, size=len(neg_idx), replace=True)])
        rng.shuffle(bal_idx)

        # Fit Base 1.0x model
        clf_1x = MLPClassifier(hidden_layer_sizes=(64,), activation="relu", alpha=0.05, max_iter=350, random_state=s)
        clf_1x.fit(X1_tr_std[bal_idx], y_tr[bal_idx])
        oof_p1[test_mask] = clf_1x.predict_proba(X1_te_std)[:, 1]

        # Fit Cue model
        clf_k = MLPClassifier(hidden_layer_sizes=(64,), activation="relu", alpha=0.05, max_iter=350, random_state=s)
        clf_k.fit(Xk_tr_std[bal_idx], y_tr[bal_idx])
        oof_pk[test_mask] = clf_k.predict_proba(Xk_te_std)[:, 1]

    # Evaluate late probability fusion: 0.5 * p1 + 0.5 * pk
    oof_fused = 0.5 * oof_p1 + 0.5 * oof_pk
    ap_base = float(average_precision_score(y_act, oof_p1))
    ap_fused = float(average_precision_score(y_act, oof_fused))

    return {
        "cue_name": cue_name,
        "is_shuffled": is_shuffled,
        "seed": s,
        "ap_base": ap_base,
        "ap_fused": ap_fused,
        "delta_ap": ap_fused - ap_base
    }


def main():
    t0 = time.time()
    driver_sha256 = compute_sha256(Path(__file__))
    print("=" * 115)
    print("MARSEN STREAM A: STANDARDIZED SIX-CUE SCREENING & SHUFFLED CONTROLS (P-1987, P-1988)")
    print(f"Driver Script SHA-256: {driver_sha256}")
    print("=" * 115)

    with open(SPLIT_MANIFEST_JSON, "r", encoding="utf-8") as f:
        design_vids = sorted(json.load(f)["_meta"]["train_videos"])
    assert design_vids == EXPECTED_DESIGN_VIDS

    cached_queries = torch.load(QUERY_DATASET_PTH, map_location="cpu")["extracted_queries"]
    active_queries = [q for q in cached_queries if q["is_active"]]
    n_active = len(active_queries)
    y_act = np.array([q["label_op_util"] for q in active_queries], dtype=np.int32)
    v_act = np.array([q["video_id"] for q in active_queries], dtype=np.int32)
    assert n_active == 1915 and int(y_act.sum()) == 92

    # Load visual base and metadata
    desc_bundle = torch.load(CACHED_POOLED_DESCRIPTORS_PTH, map_location="cpu")["descriptors"]
    feat_1x = desc_bundle["1.0x_gem_p4"].astype(np.float32)

    branch_bundle = torch.load(CACHED_BRANCHES_PTH, map_location="cpu")
    branches = branch_bundle["branches"]
    feat_meta = branches["metadata"][:, :15].astype(np.float32)

    # Map candidate cues from authentic branch bundle
    cue_registry = {}
    for key, name in [
        ("global", "Decoder-Global"),
        ("memory", "Memory"),
        ("motion", "Motion"),
        ("scene", "Scene"),
        ("horizon", "Horizon"),
        ("reflection", "Reflection")
    ]:
        if key in branches:
            cue_registry[name] = branches[key].astype(np.float32)
            print(f"[+] Loaded Cue '{name}': Shape = {cue_registry[name].shape}")
        else:
            print(f"[!] Warning: Cue key '{key}' not found in cached branches bundle.")

    # Build tasks: 6 cues × 2 conditions (authentic, shuffled) × 3 seeds = 36 parallel tasks
    tasks = []
    for c_name, c_feat in cue_registry.items():
        for is_shuff in [False, True]:
            for s in SEEDS:
                tasks.append((c_name, c_feat, is_shuff, s, feat_1x, feat_meta, y_act, v_act, design_vids))

    n_workers = int(os.environ.get("MARSEN_WORKERS", 10))
    print(f"\n[*] Dispatching {len(tasks)} LOVO evaluation tasks across {n_workers} CPU worker processes...")
    t_disp = time.time()

    task_results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for res in executor.map(evaluate_cue_seed_lovo, tasks):
            task_results.append(res)
    print(f"[+] All {len(tasks)} Models Fitted and Evaluated in {time.time() - t_disp:.2f}s")

    # Organize metrics
    results_map = defaultdict(lambda: {"auth": {}, "shuff": {}})
    for r in task_results:
        cond = "shuff" if r["is_shuffled"] else "auth"
        results_map[r["cue_name"]][cond][r["seed"]] = r

    print("\n" + "=" * 115)
    print("SIX-CUE SCREENING SYNTHESIS: AUTHENTIC VS. SHUFFLED CONTROL COMPARISON")
    print("=" * 115)
    print(f"{'Maritime Cue Name':<20} | {'Base AP':<10} | {'Auth AP':<10} | {'dAP (Auth)':<12} | {'Shuff AP':<10} | {'dAP (Shuff)':<12} | {'Signal Lift (Auth-Shuff)'}")
    print("-" * 105)

    synthesis_report = []
    survivors = []

    for c_name in sorted(cue_registry.keys()):
        base_aps = [results_map[c_name]["auth"][s]["ap_base"] for s in SEEDS]
        auth_aps = [results_map[c_name]["auth"][s]["ap_fused"] for s in SEEDS]
        shuff_aps = [results_map[c_name]["shuff"][s]["ap_fused"] for s in SEEDS]

        m_base = float(np.mean(base_aps))
        m_auth = float(np.mean(auth_aps))
        m_shuff = float(np.mean(shuff_aps))

        d_auth = m_auth - m_base
        d_shuff = m_shuff - m_base
        signal_lift = m_auth - m_shuff

        # Seed consistency check
        seed_deltas = [results_map[c_name]["auth"][s]["delta_ap"] for s in SEEDS]
        all_seeds_positive = all(d > 0.0 for d in seed_deltas)
        beats_shuffled = signal_lift > 0.001

        survived = all_seeds_positive and beats_shuffled
        if survived:
            survivors.append(c_name)

        status = " <-- ADVANCES TO STAGE 2" if survived else ""
        print(f"{c_name:<20} | {m_base:8.4f}   | {m_auth:8.4f}   | {d_auth:+10.4f}   | {m_shuff:8.4f}   | {d_shuff:+10.4f}   | {signal_lift:+12.4f}{status}")

        synthesis_report.append({
            "cue_name": c_name,
            "mean_base_ap": m_base, "mean_auth_ap": m_auth, "mean_shuff_ap": m_shuff,
            "d_auth": d_auth, "d_shuff": d_shuff, "signal_lift": signal_lift,
            "seed_deltas_auth": seed_deltas, "survived": survived
        })

    print("\n" + "=" * 115)
    print("STAGE 1 CUE SCREENING VERDICT")
    print("=" * 115)
    if len(survivors) > 0:
        print(f"[+] Qualifying Cues for Non-Redundancy Test against B_S: {survivors}")
    else:
        print("[-] ZERO cues survived the authentic-vs-shuffled screen across all seeds.")
        print("    All tested cues either failed to improve over B_0 or matched their capacity-shuffled controls.")
        print("    Conclusion: Auxiliary maritime cues are closed. Spatial verifier B_S is finalized as MARSEN-Verify.")
    print("=" * 115)

    save_payload = {
        "_meta": {
            "artifact_name": "marsen_six_cues_screen_v1",
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