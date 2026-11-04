"""
D:/MARIVISION/H0_workbench/eval_tools/audit_marsen_cues_capacity_sentinel.py
MARSEN Stream A: Three-Layer Capacity-Bracketing Sentinel (H32 vs H64 vs H128) (v2, P-2009, P-2012..P-2016)

Evaluates Memory, Decoder-Global, and Reflection under H in {32, 128} alongside frozen H64 baselines.
Uses certified frozen Base OOFs. Reports Standalone, Prob-Fusion, and Rank-Fusion metrics with seed-level deltas.
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
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
from sklearn.neural_network import MLPClassifier

OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
SPLIT_MANIFEST_JSON = OUT_DIR / "pol2_candidate2m_micro_split.json"
QUERY_DATASET_PTH = OUT_DIR / "pol3v_design_queries_dataset_v7.pth"
CACHED_BRANCHES_PTH = OUT_DIR / "marsen_micro_authentic_branches_1915.pth"
OOF_CUES_H64_PTH = OUT_DIR / "marsen_six_cues_oof_predictions_1915.pth"

RESULTS_JSON = OUT_DIR / "marsen_cues_capacity_sentinel_results.json"
OOF_PERSIST_PTH = OUT_DIR / "marsen_cues_capacity_sentinel_oofs.pth"

EXPECTED_DESIGN_VIDS = [4, 7, 16, 19, 25, 29, 30, 32, 35, 47]
SEEDS = [41, 42, 43]
TEST_WIDTHS = [32, 128]


def compute_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def evaluate_cue_capacity_task(task_args):
    cue_name, cue_feat, hidden_dim, s, oof_base_s, feat_meta, y_act, v_act, design_vids = task_args
    n_samples = len(y_act)

    oof_cue_auth = np.full(n_samples, np.nan)
    oof_cue_shuff = np.full(n_samples, np.nan)

    for vid in design_vids:
        test_mask = (v_act == vid)
        train_mask = ~test_mask

        cue_tr = cue_feat[train_mask]
        cue_te_auth = cue_feat[test_mask]
        cue_te_shuff = cue_feat[test_mask].copy()

        # D-4020: Enforce identical shuffle seed contract matching Stream A
        rng_shuff = np.random.default_rng(s * 777 + vid)
        if len(cue_te_shuff) > 1:
            cue_te_shuff = rng_shuff.permutation(cue_te_shuff)

        Xk_tr = np.concatenate([cue_tr, feat_meta[train_mask]], axis=-1)
        Xk_te_auth = np.concatenate([cue_te_auth, feat_meta[test_mask]], axis=-1)
        Xk_te_shuff = np.concatenate([cue_te_shuff, feat_meta[test_mask]], axis=-1)

        mk_tr, sk_tr = Xk_tr.mean(axis=0), Xk_tr.std(axis=0)
        sk_tr[sk_tr < 1e-6] = 1.0
        Xk_tr_std = (Xk_tr - mk_tr) / sk_tr
        Xk_te_auth_std = (Xk_te_auth - mk_tr) / sk_tr
        Xk_te_shuff_std = (Xk_te_shuff - mk_tr) / sk_tr

        y_tr = y_act[train_mask]

        pos_idx = np.where(y_tr == 1)[0]
        neg_idx = np.where(y_tr == 0)[0]
        rng = np.random.default_rng(s * 1000 + vid)
        bal_idx = np.concatenate([neg_idx, rng.choice(pos_idx, size=len(neg_idx), replace=True)])
        rng.shuffle(bal_idx)

        clf_k = MLPClassifier(hidden_layer_sizes=(hidden_dim,), activation="relu", alpha=0.05, max_iter=350, random_state=s)
        clf_k.fit(Xk_tr_std[bal_idx], y_tr[bal_idx])

        oof_cue_auth[test_mask] = clf_k.predict_proba(Xk_te_auth_std)[:, 1]
        oof_cue_shuff[test_mask] = clf_k.predict_proba(Xk_te_shuff_std)[:, 1]

    # Metrics computation
    ap_base = float(average_precision_score(y_act, oof_base_s))
    ap_stand_auth = float(average_precision_score(y_act, oof_cue_auth))
    ap_stand_shuff = float(average_precision_score(y_act, oof_cue_shuff))

    # Probability fusion
    p_fused_auth = 0.5 * oof_base_s + 0.5 * oof_cue_auth
    p_fused_shuff = 0.5 * oof_base_s + 0.5 * oof_cue_shuff
    ap_prob_auth = float(average_precision_score(y_act, p_fused_auth))
    ap_prob_shuff = float(average_precision_score(y_act, p_fused_shuff))

    # Rank fusion
    r_base = rankdata(oof_base_s, method="average") / float(n_samples)
    r_cue_auth = rankdata(oof_cue_auth, method="average") / float(n_samples)
    r_cue_shuff = rankdata(oof_cue_shuff, method="average") / float(n_samples)
    ap_rank_auth = float(average_precision_score(y_act, 0.5 * r_base + 0.5 * r_cue_auth))
    ap_rank_shuff = float(average_precision_score(y_act, 0.5 * r_base + 0.5 * r_cue_shuff))

    return {
        "cue_name": cue_name,
        "hidden_dim": hidden_dim,
        "seed": s,
        "ap_base": ap_base,
        "ap_stand_auth": ap_stand_auth,
        "ap_stand_shuff": ap_stand_shuff,
        "delta_stand_lift": ap_stand_auth - ap_stand_shuff,
        "ap_prob_auth": ap_prob_auth,
        "ap_prob_shuff": ap_prob_shuff,
        "delta_prob_vs_base": ap_prob_auth - ap_base,
        "ap_rank_auth": ap_rank_auth,
        "ap_rank_shuff": ap_rank_shuff,
        "delta_rank_vs_base": ap_rank_auth - ap_base,
        "delta_rank_lift": ap_rank_auth - ap_rank_shuff,
        "oof_cue_auth": oof_cue_auth,
        "oof_cue_shuff": oof_cue_shuff
    }


def main():
    t0 = time.time()
    driver_sha256 = compute_sha256(Path(__file__))
    print("=" * 115)
    print("MARSEN STREAM A: THREE-LAYER CAPACITY SENTINEL (H32 vs H64 vs H128) (v2, P-2009, P-2012..P-2016)")
    print(f"Driver Script SHA-256: {driver_sha256}")
    print("=" * 115)

    assert QUERY_DATASET_PTH.exists() and CACHED_BRANCHES_PTH.exists() and OOF_CUES_H64_PTH.exists()

    with open(SPLIT_MANIFEST_JSON, "r", encoding="utf-8") as f:
        design_vids = sorted(json.load(f)["_meta"]["train_videos"])
    assert design_vids == EXPECTED_DESIGN_VIDS

    cached_queries = torch.load(QUERY_DATASET_PTH, map_location="cpu")["extracted_queries"]
    active_queries = [q for q in cached_queries if q["is_active"]]
    n_active = len(active_queries)
    y_act = np.array([q["label_op_util"] for q in active_queries], dtype=np.int32)
    v_act = np.array([q["video_id"] for q in active_queries], dtype=np.int32)
    assert n_active == 1915 and int(y_act.sum()) == 92

    # Ingest frozen H64 arrays from Stream A
    h64_bundle = torch.load(OOF_CUES_H64_PTH, map_location="cpu")["oof_preds"]
    frozen_base_oof = {s: h64_bundle["Memory"][s]["oof_base"] for s in SEEDS}

    branch_bundle = torch.load(CACHED_BRANCHES_PTH, map_location="cpu")
    branches = branch_bundle["branches"]
    feat_meta = branches["metadata"][:, :15].astype(np.float32)

    target_cues = {
        "Memory": branches["memory"].astype(np.float32),
        "Decoder-Global": branches["global"].astype(np.float32),
        "Reflection": branches["reflection"].astype(np.float32),
    }

    tasks = []
    for c_name, c_feat in target_cues.items():
        for h_dim in TEST_WIDTHS:
            for s in SEEDS:
                tasks.append((c_name, c_feat, h_dim, s, frozen_base_oof[s], feat_meta, y_act, v_act, design_vids))

    n_workers = int(os.environ.get("MARSEN_WORKERS", 10))
    print(f"[*] Dispatching {len(tasks)} LOVO capacity tasks across {n_workers} CPU workers...")
    t_disp = time.time()

    task_results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for res in executor.map(evaluate_cue_capacity_task, tasks):
            task_results.append(res)
    print(f"[+] All 18 Models Fitted in {time.time() - t_disp:.2f}s")

    # Reconstruct exact H64 records from frozen Stream-A OOF vectors
    results_map = defaultdict(lambda: defaultdict(dict))
    for r in task_results:
        results_map[r["cue_name"]][r["hidden_dim"]][r["seed"]] = r

    for c_name in target_cues.keys():
        for s in SEEDS:
            o_base = h64_bundle[c_name][s]["oof_base"]
            o_auth = h64_bundle[c_name][s]["oof_cue_auth"]
            o_shuff = h64_bundle[c_name][s]["oof_cue_shuff"]

            ap_b = float(average_precision_score(y_act, o_base))
            ap_c_auth = float(average_precision_score(y_act, o_auth))
            ap_c_shuff = float(average_precision_score(y_act, o_shuff))

            ap_p_auth = float(average_precision_score(y_act, 0.5 * o_base + 0.5 * o_auth))
            ap_p_shuff = float(average_precision_score(y_act, 0.5 * o_base + 0.5 * o_shuff))

            r_base = rankdata(o_base, method="average") / float(n_active)
            r_auth = rankdata(o_auth, method="average") / float(n_active)
            r_shuff = rankdata(o_shuff, method="average") / float(n_active)
            ap_r_auth = float(average_precision_score(y_act, 0.5 * r_base + 0.5 * r_auth))
            ap_r_shuff = float(average_precision_score(y_act, 0.5 * r_base + 0.5 * r_shuff))

            results_map[c_name][64][s] = {
                "cue_name": c_name, "hidden_dim": 64, "seed": s, "ap_base": ap_b,
                "ap_stand_auth": ap_c_auth, "ap_stand_shuff": ap_c_shuff,
                "delta_stand_lift": ap_c_auth - ap_c_shuff,
                "ap_prob_auth": ap_p_auth, "ap_prob_shuff": ap_p_shuff,
                "delta_prob_vs_base": ap_p_auth - ap_b,
                "ap_rank_auth": ap_r_auth, "ap_rank_shuff": ap_r_shuff,
                "delta_rank_vs_base": ap_r_auth - ap_b,
                "delta_rank_lift": ap_r_auth - ap_r_shuff,
                "oof_cue_auth": o_auth, "oof_cue_shuff": o_shuff
            }

    # P-2015: Serialize all H32, H64, and H128 OOF prediction vectors safely
    oof_persistence_payload = {
        c_name: {
            h_dim: {
                s: {
                    "oof_cue_auth": results_map[c_name][h_dim][s]["oof_cue_auth"],
                    "oof_cue_shuff": results_map[c_name][h_dim][s]["oof_cue_shuff"]
                }
                for s in SEEDS
            }
            for h_dim in [32, 64, 128]
        }
        for c_name in target_cues.keys()
    }
    torch.save({
        "driver_sha256": driver_sha256,
        "oof_preds": oof_persistence_payload
    }, OOF_PERSIST_PTH)
    print(f"[+] P-2015 Certified: All H32/H64/H128 OOF Arrays Frozen: {OOF_PERSIST_PTH.resolve()}")

    # Output synthesis tables
    print("\n" + "=" * 115)
    print("CAPACITY SENTINEL SYNTHESIS: THREE-LAYER METRIC LEDGER (H32 vs. H64 vs. H128)")
    print("=" * 115)
    print(f"{'Cue Name':<16} | {'H-Dim':<6} | {'Standalone':<11} | {'Prob AP':<8} | {'dAP (Prob)':<11} | {'Rank AP':<8} | {'dAP (Rank)':<11} | {'Rank Lift vs Shuff'}")
    print("-" * 105)

    capacity_summary = []
    for c_name in sorted(target_cues.keys()):
        for h_dim in [32, 64, 128]:
            m_stand = float(np.mean([results_map[c_name][h_dim][s]["ap_stand_auth"] for s in SEEDS]))
            m_prob = float(np.mean([results_map[c_name][h_dim][s]["ap_prob_auth"] for s in SEEDS]))
            m_base = float(np.mean([results_map[c_name][h_dim][s]["ap_base"] for s in SEEDS]))
            d_prob = m_prob - m_base
            m_rank = float(np.mean([results_map[c_name][h_dim][s]["ap_rank_auth"] for s in SEEDS]))
            d_rank = m_rank - m_base
            m_rank_shuff = float(np.mean([results_map[c_name][h_dim][s]["ap_rank_shuff"] for s in SEEDS]))
            rank_lift = m_rank - m_rank_shuff

            s_prob_deltas = [results_map[c_name][h_dim][s]["delta_prob_vs_base"] for s in SEEDS]
            s_rank_deltas = [results_map[c_name][h_dim][s]["delta_rank_vs_base"] for s in SEEDS]
            s_rank_lifts = [results_map[c_name][h_dim][s]["delta_rank_lift"] for s in SEEDS]

            label_tag = " (Frozen Stream-A H64)" if h_dim == 64 else ""
            print(f"{c_name:<16} | H{h_dim:<5d} | {m_stand:8.4f}    | {m_prob:6.4f}   | {d_prob:+9.4f}   | {m_rank:6.4f}   | {d_rank:+9.4f}   | {rank_lift:+10.4f}{label_tag}")

            capacity_summary.append({
                "cue_name": c_name, "hidden_dim": h_dim,
                "mean_stand_ap": m_stand, "mean_prob_ap": m_prob, "d_prob": d_prob,
                "mean_rank_ap": m_rank, "d_rank": d_rank, "rank_lift": rank_lift,
                "seed_prob_deltas": s_prob_deltas,
                "seed_rank_deltas": s_rank_deltas,
                "seed_rank_lifts": s_rank_lifts
            })

    print("\n" + "=" * 115)
    print("P-2016 SEED-LEVEL LEDGER (SEED DELTAS VS BASE: [Seed 41, Seed 42, Seed 43])")
    print("=" * 115)
    for item in capacity_summary:
        c_name = item["cue_name"]
        h_dim = item["hidden_dim"]
        p_str = f"[{item['seed_prob_deltas'][0]:+.4f}, {item['seed_prob_deltas'][1]:+.4f}, {item['seed_prob_deltas'][2]:+.4f}]"
        r_str = f"[{item['seed_rank_deltas'][0]:+.4f}, {item['seed_rank_deltas'][1]:+.4f}, {item['seed_rank_deltas'][2]:+.4f}]"
        l_str = f"[{item['seed_rank_lifts'][0]:+.4f}, {item['seed_rank_lifts'][1]:+.4f}, {item['seed_rank_lifts'][2]:+.4f}]"
        print(f"[*] {c_name} (H{h_dim:<3d}) | Prob dAP: {p_str:<26} | Rank dAP: {r_str:<26} | Rank Lift: {l_str}")

    save_payload = {
        "_meta": {
            "artifact_name": "marsen_cues_capacity_sentinel_v2",
            "driver_sha256": driver_sha256,
            "total_seconds": float(time.time() - t0)
        },
        "capacity_summary": capacity_summary
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(save_payload, f, indent=2)

    res_hash = compute_sha256(RESULTS_JSON)
    print(f"\n[+] Capacity Sentinel Ledger Frozen: {RESULTS_JSON.resolve()} (SHA-256: {res_hash})")
    print(f"    Elapsed Time: {time.time() - t0:.2f}s")
    print("=" * 115)


if __name__ == "__main__":
    main()