"""
D:/MARIVISION/H0_workbench/eval_tools/audit_marsen_stream_b2_mode2a_equivalence.py
MARSEN Stream B2: Mode-2A True Integrated Equivalence Gate Driver (Strict 1e-5 Gate)
"""

import os
import sys
import json
import time
import hashlib
import torch
import numpy as np
from pathlib import Path

OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
ROOT_DIR = Path("D:/MARIVISION")
TRAIN_JSON = ROOT_DIR / "seaclips-train.json"

CERT_QUERIES_PTH = OUT_DIR / "stream_b2_reproduced_queries.pth"
CERT_FRAMES_PTH = OUT_DIR / "stream_b2_reproduced_frames.pth"
FROZEN_DESCRIPTORS_PTH = OUT_DIR / "marsen_pooled_descriptors_1915.pth"
FROZEN_BRANCHES_PTH = OUT_DIR / "marsen_micro_authentic_branches_1915.pth"
RESULTS_JSON = OUT_DIR / "marsen_stream_b2_mode2a_equivalence_results.json"

EXPECTED_DESIGN_VIDS = [4, 7, 16, 19, 25, 29, 30, 32, 35, 47]
PARITY_TOLERANCE_TENSOR = 1e-6
PARITY_TOLERANCE_DESC = 1e-5
PARITY_TOLERANCE_DEC = 1e-5


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
    print("MARSEN STREAM B2: MODE-2A TRUE INTEGRATED EQUIVALENCE GATE (STRICT PARITY)")
    print(f"Driver Script SHA-256: {driver_sha256}")
    print("=" * 115)

    assert TRAIN_JSON.exists(), f"[FATAL] Missing {TRAIN_JSON}"
    assert CERT_QUERIES_PTH.exists(), f"[FATAL] Missing {CERT_QUERIES_PTH}"
    assert CERT_FRAMES_PTH.exists(), f"[FATAL] Missing {CERT_FRAMES_PTH}"
    assert FROZEN_DESCRIPTORS_PTH.exists(), f"[FATAL] Missing {FROZEN_DESCRIPTORS_PTH}"
    assert FROZEN_BRANCHES_PTH.exists(), f"[FATAL] Missing {FROZEN_BRANCHES_PTH}"

    sys.path.insert(0, str(OUT_DIR))
    from marsen_stream_b2_engine import MarsenChronology, GeneralizedStreamEngine

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Target: {device}")
    engine = GeneralizedStreamEngine(device)

    # 1. Chronology Verification
    data, images_by_video = MarsenChronology.load_and_validate(TRAIN_JSON)
    design_images_by_video = {vid: images_by_video[vid] for vid in EXPECTED_DESIGN_VIDS}

    rolling_windows = []
    for vid in sorted(design_images_by_video.keys()):
        rolling_windows.extend(MarsenChronology.build_contiguous_windows(design_images_by_video[vid]))
    print(f"[+] Total Rolling Contiguous Windows: {len(rolling_windows)}")

    # 2. Window Identity Audit (P-2052)
    ref_frames = torch.load(CERT_FRAMES_PTH, map_location="cpu")["frame_records"]
    hist_target_iids = set(fr["image_id"] for fr in ref_frames)
    assert len(hist_target_iids) == 206

    matched_windows = [w for w in rolling_windows if w["current_frame"]["id"] in hist_target_iids]
    assert len(matched_windows) == 206, f"[FATAL] Matched {len(matched_windows)} != 206 windows!"

    for w in matched_windows:
        c_fnum = int(w["current_frame"]["frame_id"])
        h_fnums = [int(h["frame_id"]) for h in w["history_frames"]]
        expected_h_fnums = [c_fnum - age for age in [5, 4, 3, 2]]
        assert h_fnums == expected_h_fnums, f"[FATAL] Age mismatch on frame {c_fnum}: {h_fnums} != {expected_h_fnums}"
    print("[+] P-2052 Passed: All 206 historical windows match history frame ages {5, 4, 3, 2} bitwise.")

    # 3. Fresh Shared-Engine Inference Across All Design Videos
    print("\n[*] Executing Fresh Shared-Engine Inference via engine.process_video()...")
    gen_frames_by_iid = {}
    gen_queries_by_key = {}

    for vid in EXPECTED_DESIGN_VIDS:
        t_v0 = time.time()
        res = engine.process_video(vid, design_images_by_video[vid], TRAIN_JSON, is_labeled=True)
        v_matched_fr = 0
        v_matched_q = 0
        for fr in res["frame_records"]:
            if fr["image_id"] in hist_target_iids:
                gen_frames_by_iid[fr["image_id"]] = fr
                v_matched_fr += 1
        for q in res["queries"]:
            if q["image_id"] in hist_target_iids:
                gen_queries_by_key[(q["video_id"], q["image_id"], q["query_id"])] = q
                v_matched_q += 1
        print(f"  - Video {vid:>2}: {res['n_windows']} windows processed in {time.time() - t_v0:.2f}s ({v_matched_fr} target frames, {v_matched_q} target queries)")

    assert len(gen_frames_by_iid) == 206, f"[FATAL] Extracted {len(gen_frames_by_iid)} != 206 target frames!"
    assert len(gen_queries_by_key) == 2045, f"[FATAL] Extracted {len(gen_queries_by_key)} != 2,045 target queries!"

    # 4. Gate 1: Upstream Frame Tensor Parity Audit
    print("\n[*] Gate 1: Auditing Frame Record Tensor Parity...")
    max_tl_diff = 0.0
    max_tb_diff = 0.0
    max_dl_diff = 0.0
    max_db_diff = 0.0

    for fr_ref in ref_frames:
        iid = fr_ref["image_id"]
        fr_new = gen_frames_by_iid[iid]

        assert fr_new["n_temp"] == fr_ref["n_temp"]
        d_tl = float(torch.max(torch.abs(fr_new["temp_logits"].cpu() - fr_ref["temp_logits"].cpu())).item())
        d_tb = float(torch.max(torch.abs(fr_new["temp_boxes"].cpu() - fr_ref["temp_boxes"].cpu())).item())
        d_dl = float(torch.max(torch.abs(fr_new["d_logits"].cpu() - fr_ref["d_logits"].cpu())).item())
        d_db = float(torch.max(torch.abs(fr_new["d_boxes"].cpu() - fr_ref["d_boxes"].cpu())).item())

        max_tl_diff = max(max_tl_diff, d_tl)
        max_tb_diff = max(max_tb_diff, d_tb)
        max_dl_diff = max(max_dl_diff, d_dl)
        max_db_diff = max(max_db_diff, d_db)

    print(f"  - Max Temporal Logits Diff: {max_tl_diff:.2e} (Tolerance: {PARITY_TOLERANCE_TENSOR:.1e})")
    print(f"  - Max Temporal Boxes Diff:  {max_tb_diff:.2e} (Tolerance: {PARITY_TOLERANCE_TENSOR:.1e})")
    print(f"  - Max Baseline Logits Diff: {max_dl_diff:.2e} (Tolerance: {PARITY_TOLERANCE_TENSOR:.1e})")
    print(f"  - Max Baseline Boxes Diff:  {max_db_diff:.2e} (Tolerance: {PARITY_TOLERANCE_TENSOR:.1e})")

    assert max_tl_diff < PARITY_TOLERANCE_TENSOR and max_tb_diff < PARITY_TOLERANCE_TENSOR
    assert max_dl_diff < PARITY_TOLERANCE_TENSOR and max_db_diff < PARITY_TOLERANCE_TENSOR
    print("[+] Gate 1 Passed: 100% tensor parity confirmed across all 206 frames.")

    # 5. Gate 2: Query-Keyed Field and Label Parity Audit
    print("\n[*] Gate 2: Auditing Query-Keyed Field and Label Parity...")
    ref_queries = torch.load(CERT_QUERIES_PTH, map_location="cpu")["extracted_queries"]
    max_score_diff = 0.0
    max_feat_diff = 0.0

    n_active = 0
    n_op_pos = 0
    n_resc_pos = 0

    for q_ref in ref_queries:
        key = (q_ref["video_id"], q_ref["image_id"], q_ref["query_id"])
        q_new = gen_queries_by_key[key]

        assert q_new["is_active"] == q_ref["is_active"]
        assert q_new["pred_class"] == q_ref["pred_class"]
        assert q_new["label_op_util"] == q_ref["label_op_util"]
        assert q_new["label_rescue_pot"] == q_ref["label_rescue_pot"]

        d_s = abs(float(q_new["max_score"]) - float(q_ref["max_score"]))
        max_score_diff = max(max_score_diff, d_s)
        d_f = float(np.max(np.abs(np.array(q_new["features"][:15]) - np.array(q_ref["features"][:15]))))
        max_feat_diff = max(max_feat_diff, d_f)

        if q_new["is_active"]:
            n_active += 1
            if q_new["label_op_util"] == 1:
                n_op_pos += 1
            if q_new["label_rescue_pot"] == 1:
                n_resc_pos += 1

    print(f"  - Max Score Difference:   {max_score_diff:.2e} (Tolerance: {PARITY_TOLERANCE_TENSOR:.1e})")
    print(f"  - Max Feature Difference: {max_feat_diff:.2e} (Tolerance: {PARITY_TOLERANCE_TENSOR:.1e})")
    print(f"  - Census Invariants:      Active={n_active} (1,915), Utility={n_op_pos} (92), Rescue={n_resc_pos} (96)")

    assert max_score_diff < PARITY_TOLERANCE_TENSOR and max_feat_diff < PARITY_TOLERANCE_TENSOR
    assert n_active == 1915 and n_op_pos == 92 and n_resc_pos == 96
    print("[+] Gate 2 Passed: 100% field parity confirmed across all 2,045 queries.")

    # 6. Gate 3: Tri-Scale & Genuine Decoder-Global Latent Parity (Strict 1e-5 Gate)
    print("\n[*] Gate 3: Auditing Tri-Scale Descriptors and Genuine Decoder Latents (Strict 1e-5)...")
    ref_desc = torch.load(FROZEN_DESCRIPTORS_PTH, map_location="cpu")["descriptors"]
    ref_d1 = ref_desc["1.0x_gem_p4"]
    ref_d3 = ref_desc["3.0x_gem_p4"]
    ref_d5 = ref_desc["5.0x_gap"]
    ref_dec = torch.load(FROZEN_BRANCHES_PTH, map_location="cpu")["branches"]["global"]

    active_ref_queries = [q for q in ref_queries if q["is_active"]]
    assert len(active_ref_queries) == 1915

    new_d1 = np.zeros_like(ref_d1)
    new_d3 = np.zeros_like(ref_d3)
    new_d5 = np.zeros_like(ref_d5)
    new_dec = np.zeros_like(ref_dec)

    for idx, q_ref in enumerate(active_ref_queries):
        key = (q_ref["video_id"], q_ref["image_id"], q_ref["query_id"])
        q_new = gen_queries_by_key[key]

        new_d1[idx] = q_new["1x_gem4"]
        new_d3[idx] = q_new["3x_gem4"]
        new_d5[idx] = q_new["5x_gap"]
        new_dec[idx] = q_new["decoder_256"]

    diff_v1 = float(np.max(np.abs(new_d1 - ref_d1)))
    diff_v3 = float(np.max(np.abs(new_d3 - ref_d3)))
    diff_v5 = float(np.max(np.abs(new_d5 - ref_d5)))
    diff_dec = float(np.max(np.abs(new_dec - ref_dec)))

    print(f"  - 1.0x_GeM4 Descriptor Difference: {diff_v1:.2e} (Tolerance: {PARITY_TOLERANCE_DESC:.1e})")
    print(f"  - 3.0x_GeM4 Descriptor Difference: {diff_v3:.2e} (Tolerance: {PARITY_TOLERANCE_DESC:.1e})")
    print(f"  - 5.0x_GAP  Descriptor Difference: {diff_v5:.2e} (Tolerance: {PARITY_TOLERANCE_DESC:.1e})")
    print(f"  - 256-D Decoder Latent Difference:  {diff_dec:.2e} (Tolerance: {PARITY_TOLERANCE_DEC:.1e})")

    assert diff_v1 < PARITY_TOLERANCE_DESC, f"[FATAL] 1.0x parity failure: {diff_v1:.2e}"
    assert diff_v3 < PARITY_TOLERANCE_DESC, f"[FATAL] 3.0x parity failure: {diff_v3:.2e}"
    assert diff_v5 < PARITY_TOLERANCE_DESC, f"[FATAL] 5.0x parity failure: {diff_v5:.2e}"
    assert diff_dec < PARITY_TOLERANCE_DEC, f"[FATAL] Decoder parity failure: {diff_dec:.2e}"
    print("[+] Gate 3 Passed: Tri-scale descriptors and genuine decoder latents certified under strict tolerance.")

    save_payload = {
        "_meta": {
            "artifact_name": "marsen_stream_b2_mode2a_equivalence_results_final",
            "driver_sha256": driver_sha256,
            "max_tl_diff": max_tl_diff,
            "max_tb_diff": max_tb_diff,
            "max_score_diff": max_score_diff,
            "max_feat_diff": max_feat_diff,
            "diff_1x": diff_v1,
            "diff_3x": diff_v3,
            "diff_5x": diff_v5,
            "diff_dec": diff_dec,
            "elapsed_seconds": float(time.time() - t0)
        }
    }
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(save_payload, f, indent=2)

    print("\n" + "=" * 115)
    print("MODE-2A INTEGRATED EQUIVALENCE CERTIFICATION COMPLETE (STRICT ZERO-DRIFT)")
    print(f"Artifact Frozen: {RESULTS_JSON.resolve()}")
    print("=" * 115)


if __name__ == "__main__":
    main()