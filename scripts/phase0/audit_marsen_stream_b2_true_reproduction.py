"""
D:/MARIVISION/H0_workbench/eval_tools/audit_marsen_stream_b2_true_reproduction.py
MARSEN Stream B2: Mode 1 True Reproduction & Independent Parity Audit (v2, P-2036..P-2039)

Invokes the authoritative extraction functions directly with redirected target paths,
forces fresh model inference by invalidating prior outputs, and audits field/tensor parity
against canonical v7 references. Zero execution of historical main() routines.
"""

import os
import sys
import json
import inspect
import hashlib
import importlib.util
import torch
import numpy as np
from pathlib import Path

OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
CANONICAL_QUERIES_PTH = OUT_DIR / "pol3v_design_queries_dataset_v7.pth"
CANONICAL_FRAMES_PTH = OUT_DIR / "pol3v_design_frames_tensors_v7.pth"

REPRODUCED_QUERIES_PTH = OUT_DIR / "stream_b2_reproduced_queries.pth"
REPRODUCED_FRAMES_PTH = OUT_DIR / "stream_b2_reproduced_frames.pth"
RESULTS_JSON = OUT_DIR / "marsen_stream_b2_true_reproduction_results.json"

PARITY_TOLERANCE = 1e-6


def compute_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def load_module_from_file(file_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    driver_sha256 = compute_sha256(Path(__file__))
    print("=" * 115)
    print("MARSEN STREAM B2: TRUE FUNCTION-LEVEL REPRODUCTION & PARITY GATE (P-2036..P-2039)")
    print(f"Driver Script SHA-256: {driver_sha256}")
    print("=" * 115)

    assert CANONICAL_QUERIES_PTH.exists(), "[FATAL] Missing canonical v7 query reference!"
    assert CANONICAL_FRAMES_PTH.exists(), "[FATAL] Missing canonical v7 frame tensors reference!"

    # P-2039: Force deletion of prior reproduction files to ensure fresh model inference
    if REPRODUCED_QUERIES_PTH.exists():
        REPRODUCED_QUERIES_PTH.unlink()
        print(f"[*] Cleared stale reproduction artifact: {REPRODUCED_QUERIES_PTH.name}")
    if REPRODUCED_FRAMES_PTH.exists():
        REPRODUCED_FRAMES_PTH.unlink()
        print(f"[*] Cleared stale reproduction artifact: {REPRODUCED_FRAMES_PTH.name}")

    # 1. P-2038: Execute Fresh Query Extraction via Function Call (No main() side effects)
    print("\n[*] Invoking load_or_extract_query_dataset() directly from audit_pol3v_nonlinear_separability...")
    src_q = OUT_DIR / "audit_pol3v_nonlinear_separability.py"
    qmod = load_module_from_file(src_q, "b2_query_generator")
    qmod.QUERY_DATASET_PTH = REPRODUCED_QUERIES_PTH

    sig_q = inspect.signature(qmod.load_or_extract_query_dataset)
    sha_q = compute_sha256(src_q)
    if len(sig_q.parameters) >= 1:
        qmod.load_or_extract_query_dataset(sha_q)
    else:
        qmod.load_or_extract_query_dataset()

    assert REPRODUCED_QUERIES_PTH.exists(), "[FATAL] Fresh query extraction failed to write target file!"
    print(f"[+] Fresh query dataset generated via direct inference: {REPRODUCED_QUERIES_PTH.name}")

    # 2. P-2038: Execute Fresh Frame Tensors Extraction via Function Call (No main() side effects)
    print("\n[*] Invoking load_or_extract_frame_tensors() directly from audit_pol3v_operating_point_sweep...")
    src_f = OUT_DIR / "audit_pol3v_operating_point_sweep.py"
    fmod = load_module_from_file(src_f, "b2_frame_generator")
    fmod.FRAME_TENSORS_PTH = REPRODUCED_FRAMES_PTH

    sig_f = inspect.signature(fmod.load_or_extract_frame_tensors)
    sha_f = compute_sha256(src_f)
    if len(sig_f.parameters) >= 1:
        fmod.load_or_extract_frame_tensors(sha_f)
    else:
        fmod.load_or_extract_frame_tensors()

    assert REPRODUCED_FRAMES_PTH.exists(), "[FATAL] Fresh frame extraction failed to write target file!"
    print(f"[+] Fresh frame tensors generated via direct inference: {REPRODUCED_FRAMES_PTH.name}")

    # 3. Independent Query Dataset Field Parity Audit (Canonical vs. Fresh Reproduced)
    print("\n[*] Gate 1: Auditing Query Dataset Field Parity (Canonical vs. Fresh Reproduced)...")
    ref_queries = torch.load(CANONICAL_QUERIES_PTH, map_location="cpu")["extracted_queries"]
    rep_queries = torch.load(REPRODUCED_QUERIES_PTH, map_location="cpu")["extracted_queries"]

    assert len(rep_queries) == len(ref_queries) == 2045, f"[FATAL] Query count mismatch: {len(rep_queries)} != 2045"

    max_score_diff = 0.0
    max_feat_diff = 0.0

    for i in range(len(ref_queries)):
        q_ref = ref_queries[i]
        q_rep = rep_queries[i]

        assert (q_rep["video_id"], q_rep["image_id"], q_rep["query_id"]) == \
               (q_ref["video_id"], q_ref["image_id"], q_ref["query_id"]), \
               f"[FATAL] Query key mismatch at index {i}"
        assert q_rep["is_active"] == q_ref["is_active"], f"[FATAL] is_active mismatch at index {i}"
        assert q_rep["pred_class"] == q_ref["pred_class"], f"[FATAL] pred_class mismatch at index {i}"
        assert q_rep["label_op_util"] == q_ref["label_op_util"], f"[FATAL] label_op_util mismatch at index {i}"
        assert q_rep["label_rescue_pot"] == q_ref["label_rescue_pot"], f"[FATAL] label_rescue_pot mismatch at index {i}"

        d_s = abs(float(q_rep["max_score"]) - float(q_ref["max_score"]))
        max_score_diff = max(max_score_diff, d_s)

        f_ref = np.asarray(q_ref["features"][:15], dtype=np.float32)
        f_rep = np.asarray(q_rep["features"][:15], dtype=np.float32)
        d_f = float(np.max(np.abs(f_rep - f_ref)))
        max_feat_diff = max(max_feat_diff, d_f)

    print(f"  - Max Score Difference:   {max_score_diff:.2e} (Tolerance: {PARITY_TOLERANCE:.1e})")
    print(f"  - Max Feature Difference: {max_feat_diff:.2e} (Tolerance: {PARITY_TOLERANCE:.1e})")

    assert max_score_diff < PARITY_TOLERANCE, f"[FATAL] Score parity failure: {max_score_diff:.2e}"
    assert max_feat_diff < PARITY_TOLERANCE, f"[FATAL] Feature parity failure: {max_feat_diff:.2e}"
    print("[+] Query Parity Certified: 100% field match across all 2,045 queries.")

    # 4. Independent Frame Record Tensor Parity Audit (Canonical vs. Fresh Reproduced)
    print("\n[*] Gate 2: Auditing Frame Record Tensor Parity (Canonical vs. Fresh Reproduced)...")
    ref_frames = torch.load(CANONICAL_FRAMES_PTH, map_location="cpu")["frame_records"]
    rep_frames = torch.load(REPRODUCED_FRAMES_PTH, map_location="cpu")["frame_records"]

    assert len(rep_frames) == len(ref_frames) == 206, f"[FATAL] Frame count mismatch: {len(rep_frames)} != 206"

    max_tl_diff = 0.0
    max_tb_diff = 0.0
    max_dl_diff = 0.0
    max_db_diff = 0.0

    for i in range(len(ref_frames)):
        fr_ref = ref_frames[i]
        fr_rep = rep_frames[i]

        assert (fr_rep["image_id"], fr_rep["video_id"], fr_rep["n_temp"]) == \
               (fr_ref["image_id"], fr_ref["video_id"], fr_ref["n_temp"]), \
               f"[FATAL] Frame key mismatch at index {i}"

        d_tl = float(torch.max(torch.abs(fr_rep["temp_logits"] - fr_ref["temp_logits"])).item())
        d_tb = float(torch.max(torch.abs(fr_rep["temp_boxes"] - fr_ref["temp_boxes"])).item())
        d_dl = float(torch.max(torch.abs(fr_rep["d_logits"] - fr_ref["d_logits"])).item())
        d_db = float(torch.max(torch.abs(fr_rep["d_boxes"] - fr_ref["d_boxes"])).item())

        max_tl_diff = max(max_tl_diff, d_tl)
        max_tb_diff = max(max_tb_diff, d_tb)
        max_dl_diff = max(max_dl_diff, d_dl)
        max_db_diff = max(max_db_diff, d_db)

    print(f"  - Max Temporal Logits Diff: {max_tl_diff:.2e}")
    print(f"  - Max Temporal Boxes Diff:  {max_tb_diff:.2e}")
    print(f"  - Max Baseline Logits Diff: {max_dl_diff:.2e}")
    print(f"  - Max Baseline Boxes Diff:  {max_db_diff:.2e}")

    assert max_tl_diff < PARITY_TOLERANCE, f"[FATAL] temp_logits parity failure: {max_tl_diff:.2e}"
    assert max_tb_diff < PARITY_TOLERANCE, f"[FATAL] temp_boxes parity failure: {max_tb_diff:.2e}"
    assert max_dl_diff < PARITY_TOLERANCE, f"[FATAL] d_logits parity failure: {max_dl_diff:.2e}"
    assert max_db_diff < PARITY_TOLERANCE, f"[FATAL] d_boxes parity failure: {max_db_diff:.2e}"
    print("[+] Frame Parity Certified: 100% tensor match across all 206 frames.")

    save_payload = {
        "_meta": {
            "artifact_name": "marsen_stream_b2_true_reproduction_results_v2",
            "driver_sha256": driver_sha256,
            "queries_reproduced_sha256": compute_sha256(REPRODUCED_QUERIES_PTH),
            "frames_reproduced_sha256": compute_sha256(REPRODUCED_FRAMES_PTH),
            "max_score_diff": max_score_diff,
            "max_feat_diff": max_feat_diff,
            "max_temp_logits_diff": max_tl_diff,
            "max_temp_boxes_diff": max_tb_diff
        }
    }
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(save_payload, f, indent=2)

    print("\n" + "=" * 115)
    print("STREAM B2 MODE 1 REPRODUCTION OFFICIALLY CERTIFIED")
    print("The upstream generation contract reproduces without drift. Safe to run Mode 2 Full-Split Extractor.")
    print(f"Certification Record Frozen: {RESULTS_JSON.resolve()}")
    print("=" * 115)


if __name__ == "__main__":
    main()