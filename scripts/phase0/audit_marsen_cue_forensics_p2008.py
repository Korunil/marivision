"""
D:/MARIVISION/H0_workbench/eval_tools/audit_marsen_cue_forensics_p2008.py
MARSEN Stream A: Zero-Training Cue Forensics & Calibration-Free Rank Fusion (P-2008)

Ingests frozen OOF prediction vectors from marsen_six_cues_oof_predictions_1915.pth.
Computes in < 3 seconds:
  1. Standalone Cue AP (Authentic vs. Shuffled).
  2. Spearman rank correlation with Base B0 (overall and on positives).
  3. Calibration-free Rank Fusion: 0.5 * rank(p_base) + 0.5 * rank(p_cue).
  4. Per-seed rank-fusion deltas vs. Base and vs. Shuffled control.
"""

import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import time
import hashlib
from PIL import Image
import torch
import numpy as np
from pathlib import Path
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
QUERY_DATASET_PTH = OUT_DIR / "pol3v_design_queries_dataset_v7.pth"
OOF_CUES_PTH = OUT_DIR / "marsen_six_cues_oof_predictions_1915.pth"
RESULTS_JSON = OUT_DIR / "marsen_cue_forensics_p2008_results.json"

SEEDS = [41, 42, 43]


def compute_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main():
    t0 = time.time()
    driver_sha256 = compute_sha256(Path(__file__))
    print("=" * 115)
    print("MARSEN STREAM A: ZERO-TRAINING CUE FORENSICS & RANK FUSION (P-2008)")
    print(f"Driver Script SHA-256: {driver_sha256}")
    print("=" * 115)

    assert QUERY_DATASET_PTH.exists(), "[FATAL] Missing query dataset!"
    assert OOF_CUES_PTH.exists(), f"[FATAL] Missing OOF predictions at {OOF_CUES_PTH}!"

    cached_queries = torch.load(QUERY_DATASET_PTH, map_location="cpu")["extracted_queries"]
    active_queries = [q for q in cached_queries if q["is_active"]]
    n_active = len(active_queries)
    y_act = np.array([q["label_op_util"] for q in active_queries], dtype=np.int32)
    pos_mask = (y_act == 1)
    assert n_active == 1915 and int(y_act.sum()) == 92

    oof_bundle = torch.load(OOF_CUES_PTH, map_location="cpu")["oof_preds"]
    cues = sorted(list(oof_bundle.keys()))

    print(f"[+] Loaded frozen OOF vectors for {len(cues)} cues across Seeds {SEEDS}.")

    print("\n" + "=" * 115)
    print("CUE FORENSIC LEDGER: STANDALONE AP & BASE CORRELATIONS")
    print("=" * 115)
    print(f"{'Cue Name':<18} | {'Standalone Auth':<16} | {'Standalone Shuff':<17} | {'Auth Lift':<11} | {'Spearman rho (All)':<20} | {'rho (Positives)'}")
    print("-" * 115)

    forensic_summary = []

    for cue_name in cues:
        cue_auth_aps, cue_shuff_aps = [], []
        rhos_all, rhos_pos = [], []
        base_aps = []

        rank_fused_auth_aps, rank_fused_shuff_aps = [], []
        rank_deltas_vs_base, rank_lifts_vs_shuff = [], []

        for s in SEEDS:
            o_base = oof_bundle[cue_name][s]["oof_base"]
            o_auth = oof_bundle[cue_name][s]["oof_cue_auth"]
            o_shuff = oof_bundle[cue_name][s]["oof_cue_shuff"]

            ap_b = float(average_precision_score(y_act, o_base))
            ap_c_auth = float(average_precision_score(y_act, o_auth))
            ap_c_shuff = float(average_precision_score(y_act, o_shuff))

            base_aps.append(ap_b)
            cue_auth_aps.append(ap_c_auth)
            cue_shuff_aps.append(ap_c_shuff)

            r_base = rankdata(o_base, method="average") / float(n_active)
            r_c_auth = rankdata(o_auth, method="average") / float(n_active)
            r_c_shuff = rankdata(o_shuff, method="average") / float(n_active)

            rhos_all.append(float(np.corrcoef(r_base, r_c_auth)[0, 1]))
            rhos_pos.append(float(np.corrcoef(rankdata(o_base[pos_mask], method="average"), rankdata(o_auth[pos_mask], method="average"))[0, 1]))

            # Rank Fusion: 0.5 * rank_base + 0.5 * rank_cue
            r_fused_auth = 0.5 * r_base + 0.5 * r_c_auth
            r_fused_shuff = 0.5 * r_base + 0.5 * r_c_shuff

            ap_r_auth = float(average_precision_score(y_act, r_fused_auth))
            ap_r_shuff = float(average_precision_score(y_act, r_fused_shuff))

            rank_fused_auth_aps.append(ap_r_auth)
            rank_fused_shuff_aps.append(ap_r_shuff)
            rank_deltas_vs_base.append(ap_r_auth - ap_b)
            rank_lifts_vs_shuff.append(ap_r_auth - ap_r_shuff)

        m_auth_stand = float(np.mean(cue_auth_aps))
        m_shuff_stand = float(np.mean(cue_shuff_aps))
        m_stand_lift = m_auth_stand - m_shuff_stand
        m_rho_all = float(np.mean(rhos_all))
        m_rho_pos = float(np.mean(rhos_pos))

        print(f"{cue_name:<18} | {m_auth_stand:8.4f}         | {m_shuff_stand:8.4f}          | {m_stand_lift:+9.4f}   | {m_rho_all:12.4f}         | {m_rho_pos:12.4f}")

        forensic_summary.append({
            "cue_name": cue_name,
            "mean_base_ap": float(np.mean(base_aps)),
            "mean_auth_standalone": m_auth_stand,
            "mean_shuff_standalone": m_shuff_stand,
            "standalone_signal_lift": m_stand_lift,
            "mean_rho_all": m_rho_all,
            "mean_rho_pos": m_rho_pos,
            "mean_rank_fused_auth": float(np.mean(rank_fused_auth_aps)),
            "mean_rank_fused_shuff": float(np.mean(rank_fused_shuff_aps)),
            "mean_rank_delta_vs_base": float(np.mean(rank_deltas_vs_base)),
            "mean_rank_lift_vs_shuff": float(np.mean(rank_lifts_vs_shuff)),
            "seed_rank_deltas": rank_deltas_vs_base,
            "seed_rank_lifts": rank_lifts_vs_shuff
        })

    print("\n" + "=" * 115)
    print("CALIBRATION-FREE RANK FUSION SYNTHESIS (0.5 * rank_B0 + 0.5 * rank_Cue)")
    print("=" * 115)
    print(f"{'Cue Name':<18} | {'Base AP':<8} | {'Rank AP':<8} | {'dAP (Rank)':<11} | {'Shuff Rank':<11} | {'Rank Lift':<11} | {'Seed Deltas vs Base':<26} | {'Seed Lifts vs Shuff'}")
    print("-" * 122)

    rank_survivors = []
    for item in forensic_summary:
        c_name = item["cue_name"]
        m_base = item["mean_base_ap"]
        m_r_auth = item["mean_rank_fused_auth"]
        d_r = item["mean_rank_delta_vs_base"]
        m_r_shuff = item["mean_rank_fused_shuff"]
        r_lift = item["mean_rank_lift_vs_shuff"]
        s_deltas = item["seed_rank_deltas"]
        s_lifts = item["seed_rank_lifts"]

        all_pos = all(d > 0.0 for d in s_deltas)
        all_beat = all(l > 0.0 for l in s_lifts)
        survived = all_pos and all_beat
        if survived:
            rank_survivors.append(c_name)

        status = " <-- RANK SURVIVOR" if survived else ""
        deltas_str = f"[{s_deltas[0]:+.4f}, {s_deltas[1]:+.4f}, {s_deltas[2]:+.4f}]"
        lifts_str = f"[{s_lifts[0]:+.4f}, {s_lifts[1]:+.4f}, {s_lifts[2]:+.4f}]"

        print(f"{c_name:<18} | {m_base:6.4f}   | {m_r_auth:6.4f}   | {d_r:+9.4f}   | {m_r_shuff:8.4f}    | {r_lift:+9.4f}   | {deltas_str:<26} | {lifts_str}{status}")

    print("\n" + "=" * 115)
    print("P-2008 FORENSIC DIAGNOSTIC VERDICT")
    print("=" * 115)
    if len(rank_survivors) > 0:
        print(f"[+] The following cues survive under calibration-free rank fusion: {rank_survivors}")
        print("    Confirms probability calibration mismatch was suppressing these cues in late fusion.")
    else:
        print("[-] Rank fusion confirms: Cues remain sub-Base on the 10-video design set even when calibration is removed.")
    print("=" * 115)

    save_payload = {
        "_meta": {
            "artifact_name": "marsen_cue_forensics_p2008_v1",
            "driver_sha256": driver_sha256,
            "rank_survivors": rank_survivors,
            "total_seconds": float(time.time() - t0)
        },
        "summary": forensic_summary
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(save_payload, f, indent=2)

    res_hash = compute_sha256(RESULTS_JSON)
    print(f"[+] Forensic Results Frozen: {RESULTS_JSON.resolve()} (SHA-256: {res_hash})")
    print(f"    Elapsed Time: {time.time() - t0:.2f}s")
    print("=" * 115)


if __name__ == "__main__":
    main()