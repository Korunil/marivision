"""
D:/MARIVISION/H0_workbench/eval_tools/optimize_bfcef_e0_5fold_manifest.py
MARSEN B-FCEF: Stage E0 - Capacity-Constrained 5-Fold Partition Optimizer
(D-4512..D-4554, P-2313..P-2326)

Multi-Objective Simulated Annealing Search:
  - Enforces hard capacities: [10, 10, 9, 9, 9] (disjoint union of 47 videos).
  - Cryptographically verifies all 47 canonical cache shards and seaclips-train.json.
  - Shard-to-COCO Video Identity Gate (P-2326): Asserts that cached image_ids for shard vid{k}
    map strictly to JSON video_id == k, eliminating namespace aliasing.
  - Strict Camera Metadata: Maps videos[].camera_sensor strictly to:
      * 'e-CAM' -> Index 0
      * 'FLIR'  -> Index 1
      * 'Axis'  -> Index 2
    Fails loudly on any unexpected or missing sensor string.
  - Retains released_image_sizes strictly as descriptive metadata.
  - Support-aware objective: MSE evaluates only classes/cameras with support >= 5;
    scarce classes/cameras governed strictly by achievable fold coverage.
  - Tracks best state encountered within each restart.
  - Enforces structural acceptance assertions before freezing manifest.
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Tuple
import numpy as np
import torch

OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
CACHE_DIR = OUT_DIR / "e0_cache"
E0_WORK_DIR = OUT_DIR / "e0_adapters"
E0_WORK_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_JSON = Path("D:/MARIVISION/seaclips-train.json")
CACHE_MANIFEST_PATH = CACHE_DIR / "bfcef_e0_cache_manifest.json"
MANIFEST_JSON = E0_WORK_DIR / "bfcef_e0_5fold_manifest.json"

NUM_VIDEOS = 47
NUM_FOLDS = 5
NUM_CLASSES = 7
NUM_CAMERAS = 3

CAMERA_TO_IDX = {
    "e-CAM": 0,
    "FLIR": 1,
    "Axis": 2,
}
CAMERAS = ["e-CAM", "FLIR", "Axis"]
CAPACITIES = [10, 10, 9, 9, 9]
EXPECTED_TOTAL_FRAMES = 20075
OPTIMIZER_SEED = 42
OBJECTIVE_VERSION = "fold_balance_v3_native_camera_support_aware"


def compute_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main():
    driver_sha = compute_sha256(Path(__file__))
    print("=" * 115)
    print("MARSEN B-FCEF: STAGE E0 - CAPACITY-CONSTRAINED 5-FOLD PARTITION OPTIMIZER")
    print(f"Driver Script SHA-256: {driver_sha}")
    print(f"Capacities:            {CAPACITIES} (Sum = {sum(CAPACITIES)})")
    print(f"Optimizer Seed:        {OPTIMIZER_SEED}")
    print("=" * 115)

    # 1. Cryptographic Provenance Pre-Verification (Canonical Cache & Train JSON)
    assert CACHE_MANIFEST_PATH.exists(), f"[FATAL] Missing canonical cache manifest: {CACHE_MANIFEST_PATH}"
    canonical_cache_manifest_sha = compute_sha256(CACHE_MANIFEST_PATH)

    with open(CACHE_MANIFEST_PATH, "r", encoding="utf-8") as f:
        cache_manifest = json.load(f)

    assert cache_manifest["_meta"]["actual_total_frames"] == EXPECTED_TOTAL_FRAMES
    assert cache_manifest["_meta"]["num_videos"] == NUM_VIDEOS

    assert TRAIN_JSON.exists(), f"[FATAL] Missing seaclips-train.json at {TRAIN_JSON}"
    actual_train_json_sha = compute_sha256(TRAIN_JSON)
    expected_train_json_sha = cache_manifest["_meta"]["train_json_sha256"]
    assert actual_train_json_sha == expected_train_json_sha, (
        f"[FATAL] Train JSON SHA mismatch: {actual_train_json_sha} != {expected_train_json_sha}"
    )

    print("[*] Re-verifying cryptographic SHA-256 hashes of all 47 canonical cache shards on disk...")
    for filename, expected_sha in cache_manifest["cache_shard_sha256_ledger"].items():
        p = CACHE_DIR / filename
        assert p.exists(), f"[FATAL] Missing cache shard: {filename}"
        assert compute_sha256(p) == expected_sha, f"[FATAL] Cache shard SHA drift: {filename}"
    print(f"    [+] Verified 47 canonical cache shards against manifest {canonical_cache_manifest_sha[:16]}...")
    print(f"    [+] Verified Train JSON SHA-256: {actual_train_json_sha[:16]}...\n")

    # 2. Strict Camera Metadata Extraction & Video Inventory Validation
    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        train_coco = json.load(f)

    assert "videos" in train_coco, "[FATAL] 'videos' field missing in seaclips-train.json"
    video_camera = {}
    for v in train_coco["videos"]:
        vid = int(v.get("id", v.get("video_id")))
        assert "camera_sensor" in v, f"[FATAL] 'camera_sensor' missing for video {vid}"
        sensor = str(v["camera_sensor"]).strip()
        assert sensor in CAMERA_TO_IDX, (
            f"[FATAL] Unexpected camera_sensor '{sensor}' for video {vid}. Expected one of {list(CAMERA_TO_IDX.keys())}"
        )
        video_camera[vid] = CAMERA_TO_IDX[sensor]

    assert set(video_camera.keys()) == set(range(1, NUM_VIDEOS + 1)), (
        f"[FATAL] JSON video-id inventory mismatch: {sorted(video_camera.keys())}"
    )
    print(f"[*] Ingested strict camera_sensor metadata for all {NUM_VIDEOS} videos from seaclips-train.json.")

    img_native_size = {
        int(im["id"]): (int(im["width"]), int(im["height"]))
        for im in train_coco["images"]
    }
    img_to_json_video = {
        int(im["id"]): int(im["video_id"])
        for im in train_coco["images"]
    }

    # 3. Ingest Video Statistics & Shard-to-COCO Video Identity Gate (P-2326)
    video_stats = {}
    for vid in range(1, NUM_VIDEOS + 1):
        c_path = CACHE_DIR / f"bfcef_e0_cache_vid{vid}.pth"
        records = torch.load(c_path, map_location="cpu")

        # Shard-to-COCO Video Identity Gate: Verify cached image_ids resolve strictly to vid
        json_video_ids = {
            img_to_json_video[int(r["image_id"])]
            for r in records
        }
        assert json_video_ids == {vid}, (
            f"[FATAL] Shard/JSON video identity mismatch for shard vid{vid}: "
            f"cached image_ids map to JSON video_id(s) {sorted(json_video_ids)}"
        )

        dims = {
            img_native_size[int(r["image_id"])]
            for r in records
        }
        cam_idx = video_camera[vid]

        total_anns = 0
        small_anns = 0
        class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

        for r in records:
            gt_b = r["gt_boxes_norm"]
            gt_l = r["gt_labels"]
            n_gt = gt_b.shape[0]
            total_anns += n_gt
            if n_gt > 0:
                areas = (gt_b[:, 2] * 640.0) * (gt_b[:, 3] * 640.0)
                small_anns += int((areas < 32.0 * 32.0).sum().item())
                for label in gt_l.tolist():
                    if 0 <= label < NUM_CLASSES:
                        class_counts[label] += 1

        video_stats[vid] = {
            "vid": vid,
            "frames": len(records),
            "anns": total_anns,
            "small": small_anns,
            "class_counts": class_counts,
            "camera": cam_idx,
            "released_image_sizes": sorted(list(dims))
        }

    tot_frames = sum(v["frames"] for v in video_stats.values())
    tot_anns = sum(v["anns"] for v in video_stats.values())
    tot_small = sum(v["small"] for v in video_stats.values())
    tot_classes = sum(v["class_counts"] for v in video_stats.values()).astype(np.float64)

    assert tot_frames == EXPECTED_TOTAL_FRAMES, f"[FATAL] Frame count: {tot_frames} != {EXPECTED_TOTAL_FRAMES}"

    tot_cameras = np.zeros(NUM_CAMERAS, dtype=np.float64)
    for v in video_stats.values():
        tot_cameras[v["camera"]] += 1.0

    target_anns = np.array([tot_anns * (c / NUM_VIDEOS) for c in CAPACITIES])
    target_small = np.array([tot_small * (c / NUM_VIDEOS) for c in CAPACITIES])
    target_frames = np.array([tot_frames * (c / NUM_VIDEOS) for c in CAPACITIES])
    target_classes = np.outer(np.array(CAPACITIES) / float(NUM_VIDEOS), tot_classes)
    target_cameras = np.outer(np.array(CAPACITIES) / float(NUM_VIDEOS), tot_cameras)

    vids = sorted(video_stats.keys())
    V_anns = np.array([video_stats[v]["anns"] for v in vids], dtype=np.float64)
    V_small = np.array([video_stats[v]["small"] for v in vids], dtype=np.float64)
    V_frames = np.array([video_stats[v]["frames"] for v in vids], dtype=np.float64)
    V_classes = np.array([video_stats[v]["class_counts"] for v in vids], dtype=np.float64)
    V_cameras = np.array([
        [1.0 if video_stats[v]["camera"] == c else 0.0 for c in range(NUM_CAMERAS)]
        for v in vids
    ], dtype=np.float64)

    class_video_support = np.array([
        sum(video_stats[v]["class_counts"][c] > 0 for v in vids)
        for c in range(NUM_CLASSES)
    ])
    camera_video_support = np.array([int(tot_cameras[c]) for c in range(NUM_CAMERAS)])
    fully_distributable_classes = class_video_support >= NUM_FOLDS
    fully_distributable_cameras = camera_video_support >= NUM_FOLDS

    print("[*] Authoritative Dataset Baseline Composition:")
    print(f"    Total Frames: {tot_frames:,} | Total Annotations: {tot_anns:,} | Small@640: {tot_small:,}")
    print(f"    Cameras: e-CAM={int(tot_cameras[0])} videos, FLIR={int(tot_cameras[1])} videos, Axis={int(tot_cameras[2])} videos")
    for c in range(NUM_CLASSES):
        dist_status = "Distributable" if fully_distributable_classes[c] else f"Scarce (Supp={class_video_support[c]})"
        print(f"    Class {c}: {int(tot_classes[c]):>6} instances | {class_video_support[c]:>2} supporting videos ({dist_status})")

    # 4. Objective Function (Support-Aware for Both Classes and Cameras)
    def evaluate_cost(assignment):
        f_anns = np.zeros(NUM_FOLDS)
        f_small = np.zeros(NUM_FOLDS)
        f_frames = np.zeros(NUM_FOLDS)
        f_classes = np.zeros((NUM_FOLDS, NUM_CLASSES))
        f_cameras = np.zeros((NUM_FOLDS, NUM_CAMERAS))

        for idx, f in enumerate(assignment):
            f_anns[f] += V_anns[idx]
            f_small[f] += V_small[idx]
            f_frames[f] += V_frames[idx]
            f_classes[f] += V_classes[idx]
            f_cameras[f] += V_cameras[idx]

        loss_anns = np.mean(((f_anns - target_anns) / target_anns) ** 2)
        loss_small = np.mean(((f_small - target_small) / target_small) ** 2)
        loss_frames = np.mean(((f_frames - target_frames) / target_frames) ** 2)

        # Evaluate class count MSE strictly on fully distributable classes (support >= 5)
        if fully_distributable_classes.any():
            class_err = (
                (f_classes[:, fully_distributable_classes] - target_classes[:, fully_distributable_classes])
                / np.maximum(target_classes[:, fully_distributable_classes], 1.0)
            )
            loss_classes = np.mean(class_err ** 2)
        else:
            loss_classes = 0.0

        # Evaluate camera count MSE strictly on fully distributable cameras (support >= 5)
        if fully_distributable_cameras.any():
            cam_err = (
                (f_cameras[:, fully_distributable_cameras] - target_cameras[:, fully_distributable_cameras])
                / np.maximum(target_cameras[:, fully_distributable_cameras], 1.0)
            )
            loss_cameras = np.mean(cam_err ** 2)
        else:
            loss_cameras = 0.0

        class_coverage_penalty = 0.0
        for c in range(NUM_CLASSES):
            supp = class_video_support[c]
            folds_with_class = sum(f_classes[f, c] > 0 for f in range(NUM_FOLDS))
            max_achievable = min(NUM_FOLDS, supp)
            if folds_with_class < max_achievable:
                class_coverage_penalty += 35.0 * (max_achievable - folds_with_class)

        camera_coverage_penalty = 0.0
        for cam in range(NUM_CAMERAS):
            supp = camera_video_support[cam]
            folds_with_cam = sum(f_cameras[f, cam] > 0 for f in range(NUM_FOLDS))
            max_achievable = min(NUM_FOLDS, supp)
            if folds_with_cam < max_achievable:
                camera_coverage_penalty += 35.0 * (max_achievable - folds_with_cam)

        return (
            3.0 * loss_anns
            + 2.0 * loss_small
            + 1.0 * loss_frames
            + 4.0 * loss_classes
            + 2.5 * loss_cameras
            + class_coverage_penalty
            + camera_coverage_penalty
        )

    # 5. Annealing Search (50 Restarts x 5,000 Steps, Tracking Best Visited State)
    print("\n[*] Running simulated annealing search (50 restarts x 5,000 steps)...")
    np.random.seed(OPTIMIZER_SEED)
    best_global_cost = float("inf")
    best_global_assignment = None

    base_partition = []
    for f_idx, cap in enumerate(CAPACITIES):
        base_partition.extend([f_idx] * cap)
    base_partition = np.array(base_partition)

    t_opt_0 = time.time()

    for restart in range(50):
        curr_assignment = np.random.permutation(base_partition)
        curr_cost = evaluate_cost(curr_assignment)

        best_restart_cost = curr_cost
        best_restart_assignment = curr_assignment.copy()

        temp = 1.0
        cooling = 0.995

        for step in range(5000):
            i, j = np.random.choice(NUM_VIDEOS, size=2, replace=False)
            if curr_assignment[i] == curr_assignment[j]:
                continue

            curr_assignment[i], curr_assignment[j] = curr_assignment[j], curr_assignment[i]
            new_cost = evaluate_cost(curr_assignment)

            delta = new_cost - curr_cost
            if delta < 0 or np.random.rand() < np.exp(-delta / max(temp, 1e-6)):
                curr_cost = new_cost
                if curr_cost < best_restart_cost:
                    best_restart_cost = curr_cost
                    best_restart_assignment = curr_assignment.copy()
            else:
                curr_assignment[i], curr_assignment[j] = curr_assignment[j], curr_assignment[i]

            temp *= cooling

        if best_restart_cost < best_global_cost:
            best_global_cost = best_restart_cost
            best_global_assignment = best_restart_assignment.copy()
            print(f"    Restart {restart:>2}/50 | Visited Best Cost: {best_global_cost:.4f}")

    print(f"[*] Annealing completed in {time.time() - t_opt_0:.1f}s.")

    # 6. Extract Partition & Run Acceptance Assertions
    folds = {f: [] for f in range(NUM_FOLDS)}
    for v_idx, f in enumerate(best_global_assignment):
        folds[int(f)].append(vids[v_idx])
    folds = {f: sorted(folds[f]) for f in range(NUM_FOLDS)}
    all_assigned = sorted([v for vids in folds.values() for v in vids])

    assert best_global_assignment is not None, "[FATAL] Optimization failed to yield assignment!"
    assert all_assigned == list(range(1, NUM_VIDEOS + 1)), "[FATAL] Partition is not a disjoint union!"
    assert [len(folds[f]) for f in range(NUM_FOLDS)] == CAPACITIES, "[FATAL] Fold capacities violated!"

    f_class_coverage = np.zeros((NUM_FOLDS, NUM_CLASSES))
    f_camera_counts = np.zeros((NUM_FOLDS, NUM_CAMERAS))

    for f in range(NUM_FOLDS):
        for v in folds[f]:
            f_class_coverage[f] += video_stats[v]["class_counts"]
            f_camera_counts[f, video_stats[v]["camera"]] += 1

    for c in range(NUM_CLASSES):
        covered = int(sum(f_class_coverage[f, c] > 0 for f in range(NUM_FOLDS)))
        expected_covered = min(NUM_FOLDS, int(class_video_support[c]))
        assert covered == expected_covered, (
            f"[FATAL] Class {c} coverage failure: {covered} != {expected_covered}"
        )

    for cam in range(NUM_CAMERAS):
        cam_covered = int(sum(f_camera_counts[f, cam] > 0 for f in range(NUM_FOLDS)))
        expected_cam = min(NUM_FOLDS, int(camera_video_support[cam]))
        assert cam_covered == expected_cam, (
            f"[FATAL] Camera {CAMERAS[cam]} coverage failure: {cam_covered} != {expected_cam}"
        )

    # 7. Print Certified Output Tables
    print("\n" + "=" * 115)
    print("OPTIMIZED 5-FOLD MANIFEST DISTRIBUTION")
    print("=" * 115)
    print(f"{'Fold':<6} | {'Videos':<8} | {'Frames':<8} | {'GT Anns':<9} | {'Small@640':<9} | {'Class Counts (c0..c6)':<35}")
    print("-" * 115)

    per_fold_stats = {}
    for f in range(NUM_FOLDS):
        v_list = folds[f]
        f_frames = sum(video_stats[v]["frames"] for v in v_list)
        f_anns = sum(video_stats[v]["anns"] for v in v_list)
        f_small = sum(video_stats[v]["small"] for v in v_list)
        f_cc = f_class_coverage[f]
        cc_str = " ".join(f"{int(c):>5}" for c in f_cc)

        per_fold_stats[str(f)] = {
            "videos": v_list,
            "num_videos": len(v_list),
            "frames": f_frames,
            "anns": f_anns,
            "small_at_640": f_small,
            "class_counts": [int(c) for c in f_cc],
            "camera_counts": [int(f_camera_counts[f, cam]) for cam in range(NUM_CAMERAS)]
        }
        print(f"Fold {f:<2} | {len(v_list):<8} | {f_frames:<8} | {f_anns:<9} | {f_small:<9} | [{cc_str}]")

    print("=" * 115)

    print("\n" + "-" * 115)
    print("PER-CATEGORY VIDEO SUPPORT & FOLD COVERAGE AUDIT")
    print("-" * 115)
    print(f"{'Class':<8} | {'Total Instances':<16} | {'Supporting Videos':<18} | {'Folds Covered':<16} | {'Max Achievable':<16}")
    print("-" * 115)
    for c in range(NUM_CLASSES):
        covered = int(sum(f_class_coverage[f, c] > 0 for f in range(NUM_FOLDS)))
        supp = int(class_video_support[c])
        max_ach = min(NUM_FOLDS, supp)
        status = "FULL" if covered == max_ach else f"DEFICIT ({covered}/{max_ach})"
        print(f"Class {c:<3} | {int(tot_classes[c]):<16} | {supp:<18} | {covered:>2}/5 ({status:<12}) | {max_ach:>2}/5")
    print("-" * 115)

    print("\n" + "-" * 115)
    print("CAMERA / DOMAIN FOLD DISTRIBUTION AUDIT (AUTHORITATIVE CAMERA SENSORS)")
    print("-" * 115)
    print(f"{'Fold':<6} | {'e-CAM Videos':<22} | {'FLIR Videos':<24} | {'Axis Videos':<22}")
    print("-" * 115)
    for f in range(NUM_FOLDS):
        print(f"Fold {f:<2} | {int(f_camera_counts[f, 0]):<22} | {int(f_camera_counts[f, 1]):<24} | {int(f_camera_counts[f, 2]):<22}")
    print("-" * 115 + "\n")

    # 8. Save Manifest with Provenance Lock
    manifest_payload = {
        "_meta": {
            "objective_version": OBJECTIVE_VERSION,
            "optimizer_driver_sha256": driver_sha,
            "optimizer_seed": OPTIMIZER_SEED,
            "source_canonical_cache_manifest_sha256": canonical_cache_manifest_sha,
            "source_train_json_sha256": actual_train_json_sha,
            "best_search_cost": round(best_global_cost, 4),
            "num_videos": NUM_VIDEOS,
            "num_folds": NUM_FOLDS,
            "fold_capacity": CAPACITIES,
            "class_video_support": [int(c) for c in class_video_support],
            "camera_video_support": [int(c) for c in camera_video_support],
            "per_fold_stats": per_fold_stats,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "folds": {str(k): v for k, v in folds.items()},
        "video_to_fold": {str(vid): int(best_global_assignment[idx]) for idx, vid in enumerate(vids)}
    }

    with open(MANIFEST_JSON, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)

    m_sha = compute_sha256(MANIFEST_JSON)
    print(f"[+] Verified fold manifest frozen: {MANIFEST_JSON.resolve()}")
    print(f"[+] Manifest SHA-256: {m_sha}\n")


if __name__ == "__main__":
    main()