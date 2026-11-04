"""
D:/MARIVISION/H0_workbench/eval_tools/audit_bfcef_e0_temporal_cache.py
MARSEN B-FCEF: Stage E0 - Temporal Association Structure & Operational Audit
(P-2321, P-2322)

Audits precomputed temporal association cache shards across all 20,075 frames:
  1. Cryptographically re-verifies all 47 temporal cache shards against manifest ledger.
  2. Enforces strict length, image_id, video_id, and seq_idx alignment with canonical cache.
  3. Evaluates canonical Top-100 operational query selection (flatten 300x7 -> Top100 -> score >= 0.20).
  4. Stratifies match rates:
       - P(match | operational) vs P(match | non-operational)
       - P(same_slot | operational, match) vs P(same_slot | non-operational, match)
  5. Accumulates quantiles via PyTorch tensor chunks (prevents multi-million float list bloat).
  6. Asserts exact non-initial frame count = 20,075 - 47 = 20,028.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch

OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
CACHE_DIR = OUT_DIR / "e0_cache"
TEMP_MANIFEST_PATH = CACHE_DIR / "bfcef_e0_temporal_cache_manifest.json"
CANONICAL_MANIFEST_PATH = CACHE_DIR / "bfcef_e0_cache_manifest.json"

NUM_VIDEOS = 47
NUM_CLASSES = 7
EXPECTED_TOTAL_FRAMES = 20075
EXPECTED_NONINITIAL_FRAMES = EXPECTED_TOTAL_FRAMES - NUM_VIDEOS  # 20,028


def compute_sha256(file_path):
    import hashlib
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 115)
    print("MARSEN B-FCEF: STAGE E0 - TEMPORAL ASSOCIATION STRUCTURE & OPERATIONAL AUDIT")
    print(f"Target Manifest: {TEMP_MANIFEST_PATH}")
    print("=" * 115)

    assert TEMP_MANIFEST_PATH.exists(), f"[FATAL] Missing temporal cache manifest: {TEMP_MANIFEST_PATH}"
    with open(TEMP_MANIFEST_PATH, "r", encoding="utf-8") as f:
        temp_manifest = json.load(f)

    assert CANONICAL_MANIFEST_PATH.exists(), f"[FATAL] Missing canonical cache manifest: {CANONICAL_MANIFEST_PATH}"
    canonical_manifest_sha = compute_sha256(CANONICAL_MANIFEST_PATH)
    assert temp_manifest["_meta"]["input_cache_manifest_sha256"] == canonical_manifest_sha, (
        "[FATAL] Temporal cache input hash mismatch against canonical manifest!"
    )

    print("[*] Re-verifying cryptographic SHA-256 hashes of all 47 temporal cache shards...")
    temp_ledger = temp_manifest["temporal_cache_shard_sha256_ledger"]
    for filename, expected_sha in temp_ledger.items():
        p = CACHE_DIR / filename
        assert p.exists(), f"[FATAL] Missing temporal cache file: {filename}"
        assert compute_sha256(p) == expected_sha, f"[FATAL] Temporal cache SHA drift on {filename}"
    print(f"    [+] Verified 47 temporal cache shards on disk.\n")

    total_noninitial_frames = 0
    total_queries = 0
    total_matched = 0
    total_same_slot = 0
    total_reassigned = 0

    op_queries = 0
    op_matched = 0
    op_same_slot = 0
    op_reassigned = 0

    nonop_queries = 0
    nonop_matched = 0
    nonop_same_slot = 0
    nonop_reassigned = 0

    confidence_chunks = []
    appearance_chunks = []
    op_confidence_chunks = []
    op_appearance_chunks = []

    print("[*] Auditing temporal association structure across 47 video shards...", flush=True)

    for vid in range(1, NUM_VIDEOS + 1):
        temp_file = CACHE_DIR / f"bfcef_e0_temporal_cache_vid{vid}.pth"
        can_file = CACHE_DIR / f"bfcef_e0_cache_vid{vid}.pth"

        temp_records = torch.load(temp_file, map_location="cpu")
        can_records = torch.load(can_file, map_location="cpu")

        assert len(temp_records) == len(can_records), (
            f"[FATAL] Length mismatch for video {vid}: {len(temp_records)} != {len(can_records)}"
        )

        for idx, (t_r, c_r) in enumerate(zip(temp_records, can_records)):
            # Assert strict record alignment
            assert int(t_r["image_id"]) == int(c_r["image_id"]), f"[FATAL] image_id mismatch vid={vid}, idx={idx}"
            assert int(t_r["seq_idx"]) == int(c_r["seq_idx"]) == idx, f"[FATAL] seq_idx mismatch vid={vid}, idx={idx}"
            assert int(t_r["video_id"]) == int(c_r["video_id"]) == vid, f"[FATAL] video_id mismatch vid={vid}, idx={idx}"

            # Frame 0 of sequence is identity initialized (no prior history)
            if idx == 0:
                continue

            total_noninitial_frames += 1

            conf = t_r["match_confidence"]
            app = t_r["appearance_similarity"]
            prev_idx = t_r["matched_prev_index"]
            logits = c_r["d_logits"]
            num_q = conf.shape[0]

            # Canonical Top-100 Operational Query Mask (flatten 300x7 -> Top100 -> score >= 0.20)
            probs = logits.sigmoid()
            flat_probs = probs.flatten()
            topk_scores, topk_indices = torch.topk(flat_probs, min(100, flat_probs.numel()))
            q_indices = topk_indices // NUM_CLASSES
            valid = topk_scores >= 0.20

            operational_q = torch.zeros(num_q, dtype=torch.bool)
            if valid.any():
                operational_q[q_indices[valid]] = True
            non_operational_q = ~operational_q

            matched_mask = conf > 0.0
            curr_indices = torch.arange(num_q)
            same_mask = matched_mask & (prev_idx == curr_indices)
            reassigned_mask = matched_mask & (prev_idx != curr_indices)

            # Global tallies
            total_queries += num_q
            total_matched += int(matched_mask.sum().item())
            total_same_slot += int(same_mask.sum().item())
            total_reassigned += int(reassigned_mask.sum().item())

            # Operational tallies
            n_op = int(operational_q.sum().item())
            op_queries += n_op
            op_m_mask = matched_mask & operational_q
            op_s_mask = same_mask & operational_q
            op_r_mask = reassigned_mask & operational_q

            op_matched += int(op_m_mask.sum().item())
            op_same_slot += int(op_s_mask.sum().item())
            op_reassigned += int(op_r_mask.sum().item())

            # Non-operational tallies
            n_nonop = int(non_operational_q.sum().item())
            nonop_queries += n_nonop
            nonop_m_mask = matched_mask & non_operational_q
            nonop_s_mask = same_mask & non_operational_q
            nonop_r_mask = reassigned_mask & non_operational_q

            nonop_matched += int(nonop_m_mask.sum().item())
            nonop_same_slot += int(nonop_s_mask.sum().item())
            nonop_reassigned += int(nonop_r_mask.sum().item())

            # Accumulate tensor chunks
            if matched_mask.any():
                confidence_chunks.append(conf[matched_mask].float().cpu())
                appearance_chunks.append(app[matched_mask].float().cpu())
            if op_m_mask.any():
                op_confidence_chunks.append(conf[op_m_mask].float().cpu())
                op_appearance_chunks.append(app[op_m_mask].float().cpu())

    assert total_noninitial_frames == EXPECTED_NONINITIAL_FRAMES, (
        f"[FATAL] Non-initial frame count: {total_noninitial_frames} != {EXPECTED_NONINITIAL_FRAMES}"
    )

    conf_all = torch.cat(confidence_chunks).numpy() if confidence_chunks else np.array([])
    app_all = torch.cat(appearance_chunks).numpy() if appearance_chunks else np.array([])
    op_conf_all = torch.cat(op_confidence_chunks).numpy() if op_confidence_chunks else np.array([])
    op_app_all = torch.cat(op_appearance_chunks).numpy() if op_appearance_chunks else np.array([])

    # Compute Rates
    p_matched_all = total_matched / max(1, total_queries)
    p_same_all = total_same_slot / max(1, total_matched)
    p_reassigned_all = total_reassigned / max(1, total_matched)

    p_matched_op = op_matched / max(1, op_queries)
    p_same_op = op_same_slot / max(1, op_matched)
    p_reassigned_op = op_reassigned / max(1, op_matched)

    p_matched_nonop = nonop_matched / max(1, nonop_queries)
    p_same_nonop = nonop_same_slot / max(1, nonop_matched)
    p_reassigned_nonop = nonop_reassigned / max(1, nonop_matched)

    print("\n" + "=" * 115)
    print("TEMPORAL ASSOCIATION STRUCTURE & OPERATIONAL AUDIT RESULTS")
    print("=" * 115)
    print(f"Total Non-Initial Frames Audited:       {total_noninitial_frames:,} / {EXPECTED_NONINITIAL_FRAMES:,} (EXACT)")
    print(f"Total Query Opportunities:              {total_queries:,} (300 queries/frame)")
    print(f"Total Admissible Matches:               {total_matched:,} ({p_matched_all:.2%} match rate)")
    print(f"  - Same-Slot Retention P(match_i == i): {total_same_slot:,} ({p_same_all:.2%} of matched)")
    print(f"  - Cross-Slot Reassigned P(match_i != i):{total_reassigned:,} ({p_reassigned_all:.2%} of matched)")
    print("-" * 115)
    print(f"OPERATIONAL QUERIES (Canonical Flattened Top-100, s >= 0.20):")
    print(f"  Total Operational Query Slots:        {op_queries:,} ({op_queries / total_queries:.2%} of all queries)")
    print(f"  Match Rate P(match | operational):    {op_matched:,} / {op_queries:,} ({p_matched_op:.2%})")
    print(f"  Same-Slot Rate P(same | op, match):   {op_same_slot:,} / {op_matched:,} ({p_same_op:.2%})")
    print(f"  Reassigned Rate P(reass | op, match): {op_reassigned:,} / {op_matched:,} ({p_reassigned_op:.2%})")
    print("-" * 115)
    print(f"NON-OPERATIONAL / DORMANT QUERIES:")
    print(f"  Total Non-Operational Slots:          {nonop_queries:,}")
    print(f"  Match Rate P(match | non-op):         {nonop_matched:,} / {nonop_queries:,} ({p_matched_nonop:.2%})")
    print(f"  Same-Slot Rate P(same | non-op, match):{nonop_same_slot:,} / {nonop_matched:,} ({p_same_nonop:.2%})")
    print("-" * 115)
    print("ALL MATCHES QUANTILE DISTRIBUTIONS:")
    print(f"  Confidence:  Mean={np.mean(conf_all):.4f} | P10={np.percentile(conf_all, 10):.4f} | P50={np.percentile(conf_all, 50):.4f} | P90={np.percentile(conf_all, 90):.4f}")
    print(f"  Appearance:  Mean={np.mean(app_all):.4f} | P10={np.percentile(app_all, 10):.4f} | P50={np.percentile(app_all, 50):.4f} | P90={np.percentile(app_all, 90):.4f}")
    print("-" * 115)
    print("OPERATIONAL MATCHES QUANTILE DISTRIBUTIONS:")
    print(f"  Confidence:  Mean={np.mean(op_conf_all):.4f} | P10={np.percentile(op_conf_all, 10):.4f} | P50={np.percentile(op_conf_all, 50):.4f} | P90={np.percentile(op_conf_all, 90):.4f}")
    print(f"  Appearance:  Mean={np.mean(op_app_all):.4f} | P10={np.percentile(op_app_all, 10):.4f} | P50={np.percentile(op_app_all, 50):.4f} | P90={np.percentile(op_app_all, 90):.4f}")
    print("=" * 115 + "\n")


if __name__ == "__main__":
    main()