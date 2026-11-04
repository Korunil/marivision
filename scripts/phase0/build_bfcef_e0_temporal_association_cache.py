"""
D:/MARIVISION/H0_workbench/eval_tools/build_bfcef_e0_temporal_association_cache.py
MARSEN B-FCEF: Stage E0 - Precomputed Temporal Association Cache Generator
(D-4512..D-4528, P-2312..P-2314)

Precomputes first-order past-only temporal association ONCE across all 20,075 frames:
  - Re-verifies all 47 canonical cache shards on disk before processing.
  - Uses the EXACT certified CanonicalTemporalAssociationService (P-2294):
    C_ij = 0.5*C_geom + 0.3*C_app + 0.2*C_cls, IoU >= 0.15, cost <= 0.65.
  - Re-normalizes float16 appearance descriptors in float32 before cosine matching.
  - Records association diagnostics (match rate, mean conf, mean app) and matched_prev_index.
  - Persists per-video association files: e0_cache/bfcef_e0_temporal_cache_vid{vid}.pth.
  - Freezes e0_cache/bfcef_e0_temporal_cache_manifest.json with complete provenance lock.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import time
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
CACHE_DIR = OUT_DIR / "e0_cache"
CACHE_MANIFEST_PATH = CACHE_DIR / "bfcef_e0_cache_manifest.json"
TEMP_MANIFEST_PATH = CACHE_DIR / "bfcef_e0_temporal_cache_manifest.json"

NUM_VIDEOS = 47
NUM_CLASSES = 7
EXPECTED_TOTAL_FRAMES = 20075
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ASSOCIATION_VERSION = "t1_hungarian_v1_geom07iou03center_fusion05g03a02c_iou015_cost065"
IOU_THRESHOLD = 0.15
MAX_COST_THRESHOLD = 0.65
FUSION_WEIGHTS = {"geometry": 0.5, "appearance": 0.3, "semantic": 0.2}


def compute_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


class CanonicalTemporalAssociationService:
    def __init__(self, iou_thresh: float = IOU_THRESHOLD, max_cost_thresh: float = MAX_COST_THRESHOLD):
        self.iou_thresh = iou_thresh
        self.max_cost_thresh = max_cost_thresh
        self.prev_boxes: Optional[torch.Tensor] = None
        self.prev_appearance: Optional[torch.Tensor] = None
        self.prev_probs: Optional[torch.Tensor] = None
        self.current_vid: Optional[int] = None

    def reset(self, vid: int):
        self.prev_boxes = None
        self.prev_appearance = None
        self.prev_probs = None
        self.current_vid = vid

    def associate(
        self,
        curr_boxes: torch.Tensor,
        curr_appearance: torch.Tensor,
        curr_probs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        num_q = curr_boxes.shape[0]
        device = curr_boxes.device

        curr_app = F.normalize(curr_appearance.float(), p=2, dim=-1)

        if self.prev_boxes is None:
            self.prev_boxes = curr_boxes.detach().clone()
            self.prev_appearance = curr_app.detach().clone()
            self.prev_probs = curr_probs.detach().clone()
            return {
                "matched_boxes": curr_boxes.detach().clone(),
                "match_confidence": torch.zeros(num_q, device=device),
                "appearance_similarity": torch.zeros(num_q, device=device),
                "matched_prev_index": torch.full((num_q,), -1, dtype=torch.int64, device=device)
            }

        prev_b = self.prev_boxes
        prev_app = F.normalize(self.prev_appearance.float(), p=2, dim=-1)
        prev_p = self.prev_probs

        b1_x1 = curr_boxes[:, 0] - curr_boxes[:, 2] / 2.0
        b1_y1 = curr_boxes[:, 1] - curr_boxes[:, 3] / 2.0
        b1_x2 = curr_boxes[:, 0] + curr_boxes[:, 2] / 2.0
        b1_y2 = curr_boxes[:, 1] + curr_boxes[:, 3] / 2.0

        b2_x1 = prev_b[:, 0] - prev_b[:, 2] / 2.0
        b2_y1 = prev_b[:, 1] - prev_b[:, 3] / 2.0
        b2_x2 = prev_b[:, 0] + prev_b[:, 2] / 2.0
        b2_y2 = prev_b[:, 1] + prev_b[:, 3] / 2.0

        ix = (torch.min(b1_x2.unsqueeze(1), b2_x2.unsqueeze(0)) - torch.max(b1_x1.unsqueeze(1), b2_x1.unsqueeze(0))).clamp(min=0.0)
        iy = (torch.min(b1_y2.unsqueeze(1), b2_y2.unsqueeze(0)) - torch.max(b1_y1.unsqueeze(1), b2_y1.unsqueeze(0))).clamp(min=0.0)
        inter = ix * iy

        area1 = (curr_boxes[:, 2] * curr_boxes[:, 3]).unsqueeze(1)
        area2 = (prev_b[:, 2] * prev_b[:, 3]).unsqueeze(0)
        union = area1 + area2 - inter
        iou_matrix = torch.where(union > 0, inter / union, torch.zeros_like(inter))

        c_dist = torch.norm(curr_boxes[:, :2].unsqueeze(1) - prev_b[:, :2].unsqueeze(0), dim=-1).clamp(max=1.0)
        c_geom = 0.7 * (1.0 - iou_matrix) + 0.3 * c_dist

        cos_sim = torch.mm(curr_app, prev_app.t()).clamp(min=0.0, max=1.0)
        c_app = 1.0 - cos_sim

        c_cls = torch.norm(curr_probs.unsqueeze(1) - prev_p.unsqueeze(0), dim=-1) / np.sqrt(NUM_CLASSES)
        cost_matrix = 0.5 * c_geom + 0.3 * c_app + 0.2 * c_cls

        admissible_mask = (iou_matrix >= self.iou_thresh) & (cost_matrix <= self.max_cost_thresh)
        cost_cpu = cost_matrix.detach().cpu().numpy()
        admissible_cpu = admissible_mask.detach().cpu().numpy()
        cost_cpu[~admissible_cpu] = 1e6

        row_ind, col_ind = linear_sum_assignment(cost_cpu)

        matched_boxes = curr_boxes.detach().clone()
        match_confidence = torch.zeros(num_q, device=device)
        appearance_similarity = torch.zeros(num_q, device=device)
        matched_prev_index = torch.full((num_q,), -1, dtype=torch.int64, device=device)

        for r, c in zip(row_ind, col_ind):
            if admissible_cpu[r, c] and cost_cpu[r, c] < 1e5:
                matched_boxes[r] = prev_b[c]
                match_confidence[r] = float(1.0 - cost_matrix[r, c].item())
                appearance_similarity[r] = float(cos_sim[r, c].item())
                matched_prev_index[r] = int(c)

        self.prev_boxes = curr_boxes.detach().clone()
        self.prev_appearance = curr_app.detach().clone()
        self.prev_probs = curr_probs.detach().clone()

        return {
            "matched_boxes": matched_boxes,
            "match_confidence": match_confidence,
            "appearance_similarity": appearance_similarity,
            "matched_prev_index": matched_prev_index
        }


def main():
    t_start = time.time()
    driver_sha = compute_sha256(Path(__file__))

    print("=" * 105)
    print("MARSEN B-FCEF: STAGE E0 - PRECOMPUTED TEMPORAL ASSOCIATION CACHE BUILDER")
    print(f"Driver Script SHA-256: {driver_sha}")
    print(f"Target Device:         {DEVICE}")
    print(f"Association Version:   {ASSOCIATION_VERSION}")
    print("=" * 105)

    # 1. Provenance Gate: Verify Canonical Cache Manifest and all 47 Shards
    assert CACHE_MANIFEST_PATH.exists(), f"[FATAL] Missing canonical cache manifest: {CACHE_MANIFEST_PATH}"
    canonical_manifest_sha = compute_sha256(CACHE_MANIFEST_PATH)

    with open(CACHE_MANIFEST_PATH, "r", encoding="utf-8") as f:
        cache_manifest = json.load(f)

    assert cache_manifest["_meta"]["actual_total_frames"] == EXPECTED_TOTAL_FRAMES
    assert cache_manifest["_meta"]["num_videos"] == NUM_VIDEOS

    cache_ledger = cache_manifest["cache_shard_sha256_ledger"]
    assert len(cache_ledger) == NUM_VIDEOS, f"[FATAL] Ledger shard count mismatch: {len(cache_ledger)} != {NUM_VIDEOS}"

    print("[*] Re-verifying cryptographic SHA-256 hashes of all 47 canonical cache shards on disk...")
    for filename, expected_sha in cache_ledger.items():
        p = CACHE_DIR / filename
        assert p.exists(), f"[FATAL] Missing canonical cache file: {filename}"
        assert compute_sha256(p) == expected_sha, f"[FATAL] Canonical cache drift on {filename}"

    print(f"    [+] Verified 47 canonical cache shards against manifest {canonical_manifest_sha[:16]}...\n")

    # 2. Extract Temporal Association with Diagnostics
    assoc_service = CanonicalTemporalAssociationService()
    temp_cache_ledger = {}
    total_processed_frames = 0

    tot_matched_queries = 0
    tot_eval_queries = 0
    sum_confidence = 0.0
    sum_appearance = 0.0

    print("[*] Beginning past-only temporal association extraction across 20,075 frames...", flush=True)

    for vid in range(1, NUM_VIDEOS + 1):
        t_v0 = time.time()
        c_path = CACHE_DIR / f"bfcef_e0_cache_vid{vid}.pth"
        records = torch.load(c_path, map_location="cpu")

        assoc_service.reset(vid)
        video_temp_records = []

        vid_matched = 0
        vid_queries = 0
        vid_conf_sum = 0.0
        vid_app_sum = 0.0

        with torch.no_grad():
            for r in records:
                d_boxes = r["d_boxes"].to(DEVICE)
                d_logits = r["d_logits"].to(DEVICE)
                app_desc = r["appearance_desc"].to(DEVICE)
                probs = d_logits.sigmoid()

                t_hist = assoc_service.associate(d_boxes, app_desc, probs)

                # Diagnostics
                m_conf = t_hist["match_confidence"]
                matched_mask = m_conf > 0.0
                n_m = int(matched_mask.sum().item())
                vid_matched += n_m
                vid_queries += d_boxes.shape[0]

                if n_m > 0:
                    vid_conf_sum += float(m_conf[matched_mask].sum().item())
                    vid_app_sum += float(t_hist["appearance_similarity"][matched_mask].sum().item())

                video_temp_records.append({
                    "image_id": r["image_id"],
                    "video_id": vid,
                    "seq_idx": r["seq_idx"],
                    "matched_boxes": t_hist["matched_boxes"].cpu(),
                    "match_confidence": t_hist["match_confidence"].cpu(),
                    "appearance_similarity": t_hist["appearance_similarity"].cpu(),
                    "matched_prev_index": t_hist["matched_prev_index"].cpu()
                })

        temp_cache_file = CACHE_DIR / f"bfcef_e0_temporal_cache_vid{vid}.pth"
        torch.save(video_temp_records, temp_cache_file)
        file_sha = compute_sha256(temp_cache_file)
        temp_cache_ledger[temp_cache_file.name] = file_sha
        total_processed_frames += len(video_temp_records)

        tot_matched_queries += vid_matched
        tot_eval_queries += vid_queries
        sum_confidence += vid_conf_sum
        sum_appearance += vid_app_sum

        v_match_rate = vid_matched / max(1, vid_queries)
        v_mean_conf = vid_conf_sum / max(1, vid_matched)
        v_mean_app = vid_app_sum / max(1, vid_matched)

        print(f"[+] Video {vid:02d}/47 temporal cache: {len(video_temp_records):>4} frames in {time.time() - t_v0:.1f}s | Match Rate: {v_match_rate:.2%} | Conf: {v_mean_conf:.3f} | App: {v_mean_app:.3f} -> {temp_cache_file.name}")

    assert total_processed_frames == EXPECTED_TOTAL_FRAMES
    assert len(temp_cache_ledger) == NUM_VIDEOS

    overall_match_rate = tot_matched_queries / max(1, tot_eval_queries)
    overall_mean_conf = sum_confidence / max(1, tot_matched_queries)
    overall_mean_app = sum_appearance / max(1, tot_matched_queries)

    # 3. Freeze Provenance Manifest
    manifest_payload = {
        "_meta": {
            "driver_sha256": driver_sha,
            "input_cache_manifest_sha256": canonical_manifest_sha,
            "association_version": ASSOCIATION_VERSION,
            "expected_total_frames": EXPECTED_TOTAL_FRAMES,
            "actual_total_frames": total_processed_frames,
            "num_videos": NUM_VIDEOS,
            "iou_threshold": IOU_THRESHOLD,
            "max_cost_threshold": MAX_COST_THRESHOLD,
            "fusion_weights": FUSION_WEIGHTS,
            "overall_match_rate": round(overall_match_rate, 4),
            "overall_mean_confidence": round(overall_mean_conf, 4),
            "overall_mean_appearance_similarity": round(overall_mean_app, 4),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "temporal_cache_shard_sha256_ledger": temp_cache_ledger
    }

    with open(TEMP_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)

    t_elapsed = (time.time() - t_start) / 60.0
    print("\n" + "=" * 105)
    print(f"[+] TEMPORAL ASSOCIATION CACHE GENERATED IN {t_elapsed:.2f} MINUTES")
    print(f"[+] Total Frames: {total_processed_frames:,} across {NUM_VIDEOS} video shards (EXACT MATCH)")
    print(f"[*] Overall Association Diagnostics: Match Rate = {overall_match_rate:.2%}, Mean Conf = {overall_mean_conf:.4f}, Mean App = {overall_mean_app:.4f}")
    print(f"[+] Manifest Frozen: {TEMP_MANIFEST_PATH.resolve()}")
    print("=" * 105 + "\n")


if __name__ == "__main__":
    main()