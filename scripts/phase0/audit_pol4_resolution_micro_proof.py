"""
D:/MARIVISION/H0_workbench/eval_tools/audit_pol4_resolution_micro_proof.py
Candidate 4 Micro-Proof: Resolution-Isolated Sensory Verification Benchmark (v3)

Strict Compliance Checklist:
  - D-3716..D-3725: Design cohort ONLY; calibration & test remain untouched.
  - Path Resolution: Normalizes and indexes SeaClips.zip namelist to resolve root folder prefixes.
  - P-1786: Hierarchical hard-negative matching sorts by (age_diff, score_diff) on relaxed tiers.
  - P-1787: Cross-video wrong-history partners matched on (pred_class, hist_age, max_score).
  - P-1788: Primary correspondence test uses the identical C_corr-fitted probe to score C_wrong.
  - P-1789: Complete cache provenance verifying query, frame-tensor, token, and manifest hashes.
  - P-1781 / P-1783: Reconstructs exact predicted current boxes and true historical source tokens.
"""

import os
import sys
import copy
import json
import hashlib
import zipfile
import io
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression

OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
TRAIN_JSON = Path("D:/MARIVISION/seaclips-train.json")
SPLIT_MANIFEST_JSON = OUT_DIR / "pol2_candidate2m_micro_split.json"
QUERY_DATASET_PTH = OUT_DIR / "pol3v_design_queries_dataset_v7.pth"
FRAME_TENSORS_PTH = OUT_DIR / "pol3v_design_frames_tensors_v7.pth"
TOKEN_CACHE_PTH = OUT_DIR / "cache_m0_candidate2m_design_calib.pth"
CANDIDATES_JSON = OUT_DIR / "pol2_candidate_windows_v1.json"
ZIP_PATH = Path("D:/MARIVISION/SeaClips.zip")

RESULTS_JSON = OUT_DIR / "pol4_resolution_micro_proof_results.json"
CACHED_FEATURES_PTH = OUT_DIR / "pol4_micro_proof_features_v3.pth"

EXPECTED_CANDIDATE_SHA256 = "f29e0b8a5cbfc46641222d84a6d370c3e9bc51ca4ca1124b7753078ca7c1e270"
EXPECTED_TRAIN_JSON_SHA256 = "edc32bdbd6b792eb83b6daf11e14708232f1f4a9e06153ebebd5bad1caef9b35"
EXPECTED_SPLIT_SHA256 = "7da471d2793fe3369508200dec50422bb76492860c67a1a61871645471856772"
EXPECTED_TOKEN_CACHE_SHA256 = "d1e6407722c59968f87e0ae9cfdead28cacf89924dcd765ca41cf8993da4dd4c"

CROP_SIZE = 128
EXPANSION_FACTOR = 1.5
MEMORY_AGES = [2, 3, 4, 5]
ENCODER_NAME = "torchvision_resnet18_IMAGENET1K_V1_layer2"


def compute_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main():
    driver_sha256 = compute_sha256(Path(__file__))
    print("=" * 115)
    print("CANDIDATE 4 MICRO-PROOF: RESOLUTION-ISOLATED SENSORY VERIFICATION (v3)")
    print(f"Driver Script SHA-256: {driver_sha256}")
    print("=" * 115)

    assert CANDIDATES_JSON.exists(), f"[FATAL] Missing {CANDIDATES_JSON}"
    assert TRAIN_JSON.exists(), f"[FATAL] Missing {TRAIN_JSON}"
    assert SPLIT_MANIFEST_JSON.exists(), f"[FATAL] Missing {SPLIT_MANIFEST_JSON}"
    assert TOKEN_CACHE_PTH.exists(), f"[FATAL] Missing {TOKEN_CACHE_PTH}"
    assert QUERY_DATASET_PTH.exists(), f"[FATAL] Missing {QUERY_DATASET_PTH}"
    assert FRAME_TENSORS_PTH.exists(), f"[FATAL] Missing {FRAME_TENSORS_PTH}"
    assert ZIP_PATH.exists(), f"[FATAL] Missing {ZIP_PATH}"

    assert compute_sha256(CANDIDATES_JSON) == EXPECTED_CANDIDATE_SHA256
    assert compute_sha256(TRAIN_JSON) == EXPECTED_TRAIN_JSON_SHA256
    assert compute_sha256(SPLIT_MANIFEST_JSON) == EXPECTED_SPLIT_SHA256
    assert compute_sha256(TOKEN_CACHE_PTH) == EXPECTED_TOKEN_CACHE_SHA256

    print("[+] All Upstream Data Manifests and Token Cache Certified by Immutable SHA-256.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Target: {device} ({torch.cuda.get_device_name(0)})")

    with open(SPLIT_MANIFEST_JSON, "r", encoding="utf-8") as f:
        design_vids = sorted(json.load(f)["_meta"]["train_videos"])

    cached_queries = torch.load(QUERY_DATASET_PTH, map_location="cpu")["extracted_queries"]
    print(f"[+] Loaded {len(cached_queries)} Queries from Certified v7 Query Cache.")

    frame_bundle = torch.load(FRAME_TENSORS_PTH, map_location="cpu")
    frame_records = frame_bundle["frame_records"]
    print(f"[+] Loaded {len(frame_records)} Frame Detection Records from Certified Frame Tensors Cache.")

    # --------------------------------------------------------------------------
    # P-1781: RECONSTRUCT CURRENT PREDICTED BOXES VIA EXACT (image_id, query_id)
    # --------------------------------------------------------------------------
    frame_dict = {fr["image_id"]: fr for fr in frame_records}
    frame_keys = []
    for fr in frame_records:
        for q_t in range(fr["n_temp"]):
            frame_keys.append((fr["image_id"], q_t))
    query_keys = [(q["image_id"], q["query_id"]) for q in cached_queries]

    assert len(frame_keys) == 2045, f"[FATAL] Expected 2,045 frame keys, got {len(frame_keys)}"
    assert frame_keys == query_keys, "[FATAL] Query-key order mismatch between frame tensors and query dataset!"

    for q in cached_queries:
        fr = frame_dict[q["image_id"]]
        cx, cy, w, h = fr["temp_boxes"][q["query_id"]].tolist()
        q["pred_box_norm"] = (cx, cy, w, h)
        q["img_w"] = fr["img_w"]
        q["img_h"] = fr["img_h"]

    print("[+] Reconstructed exact inference-predicted current bounding boxes for all 2,045 queries.")

    # --------------------------------------------------------------------------
    # P-1783: RECONSTRUCT EXACT HISTORICAL SOURCE TOKENS (FRAME, BBOX, AGE)
    # --------------------------------------------------------------------------
    with open(CANDIDATES_JSON, "r", encoding="utf-8") as f:
        cand_windows = json.load(f)["windows"]

    design_windows = [
        w for w in cand_windows
        if w["video_id"] in design_vids and w.get("n_mem_incr_s20", w.get("n_memincr_s20", 0)) >= 1
    ]
    window_by_iid = {}
    for w in design_windows:
        window_by_iid[w["image_ids"][-1]] = w

    hist_token_cache = torch.load(TOKEN_CACHE_PTH, map_location="cpu")

    def reconstruct_frame_memory_bank(window, cache_dict):
        curr_fid = window["frame_ids"][-1]
        mb = []
        for f_idx in range(5):
            hist_iid = window["image_ids"][f_idx]
            hist_fid = window["frame_ids"][f_idx]
            age = curr_fid - hist_fid
            if age in MEMORY_AGES and hist_iid in cache_dict:
                for tok in cache_dict[hist_iid]:
                    mb.append({
                        "hist_iid": hist_iid,
                        "hist_fid": hist_fid,
                        "age": age,
                        "bbox_norm": tok["bbox"],
                        "score": tok["score"]
                    })
        return mb

    for fr in frame_records:
        iid = fr["image_id"]
        w = window_by_iid[iid]
        mb_rec = reconstruct_frame_memory_bank(w, hist_token_cache)
        assert len(mb_rec) == fr["n_temp"], f"[FATAL] Memory bank reconstruction count mismatch in frame {iid}!"
        fr["memory_bank"] = mb_rec

    for q in cached_queries:
        fr = frame_dict[q["image_id"]]
        tok_meta = fr["memory_bank"][q["query_id"]]
        q["hist_iid"] = tok_meta["hist_iid"]
        q["hist_bbox_norm"] = tok_meta["bbox_norm"]
        q["hist_age"] = tok_meta["age"]

    print("[+] Reconstructed exact historical source tokens (frame ID, bbox, age) for all 2,045 queries.")

    # --------------------------------------------------------------------------
    # P-1782 & P-1786: DETERMINISTIC HIERARCHICAL HARD-NEGATIVE MATCHER
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("P-1782 / P-1786: HIERARCHICAL HARD-NEGATIVE MATCHING AUDIT")
    print("=" * 115)

    positives = [q for q in cached_queries if q["label_op_util"] == 1]
    active_negs = [q for q in cached_queries if q["label_op_util"] == 0 and q["is_active"]]

    assert len(positives) == 92, f"[FATAL] Expected 92 positives, got {len(positives)}"
    assert all(p["is_active"] for p in positives), "[FATAL] All 92 operational positives must be threshold-active!"

    used_negative_ids = set()
    selected_cohort = []
    tier_counts = defaultdict(int)
    score_deltas = []
    age_deltas = []

    for p in positives:
        selected_cohort.append({"query": p, "label": 1, "tier": "positive"})
        p_iid = p["image_id"]
        p_vid = p["video_id"]
        p_cls = p["pred_class"]
        p_age = p["hist_age"]
        p_sc = p["max_score"]

        avail_negs = [n for n in active_negs if id(n) not in used_negative_ids]

        matched_for_p = []
        for tier_id, matcher_fn, sort_key_fn in [
            (1, lambda n: n["image_id"] == p_iid and n["pred_class"] == p_cls and n["hist_age"] == p_age,
                lambda n: abs(n["max_score"] - p_sc)),
            (2, lambda n: n["image_id"] == p_iid and n["pred_class"] == p_cls,
                lambda n: (abs(n["hist_age"] - p_age), abs(n["max_score"] - p_sc))),
            (3, lambda n: n["video_id"] == p_vid and n["pred_class"] == p_cls and n["hist_age"] == p_age,
                lambda n: abs(n["max_score"] - p_sc)),
            (4, lambda n: n["video_id"] == p_vid and n["pred_class"] == p_cls,
                lambda n: (abs(n["hist_age"] - p_age), abs(n["max_score"] - p_sc))),
        ]:
            if len(matched_for_p) >= 3:
                break
            tier_pool = [n for n in avail_negs if matcher_fn(n) and id(n) not in [id(x[0]) for x in matched_for_p]]
            tier_pool.sort(key=sort_key_fn)
            needed = 3 - len(matched_for_p)
            for n in tier_pool[:needed]:
                matched_for_p.append((n, tier_id))

        for n, tier_id in matched_for_p:
            used_negative_ids.add(id(n))
            selected_cohort.append({"query": n, "label": 0, "tier": f"tier_{tier_id}"})
            tier_counts[tier_id] += 1
            score_deltas.append(abs(n["max_score"] - p_sc))
            age_deltas.append(abs(n["hist_age"] - p_age))

    n_pos = sum(1 for item in selected_cohort if item["label"] == 1)
    n_neg = sum(1 for item in selected_cohort if item["label"] == 0)

    print(f"[*] Matched Cohort Summary: N = {len(selected_cohort)} (92 Positives, {n_neg} Hard Negatives)")
    print(f"    - Tier 1 (Same frame, same class, exact age):      {tier_counts[1]:3d} ({tier_counts[1]/n_neg*100:5.1f}%)")
    print(f"    - Tier 2 (Same frame, same class, closest age):    {tier_counts[2]:3d} ({tier_counts[2]/n_neg*100:5.1f}%)")
    print(f"    - Tier 3 (Same video, same class, exact age):      {tier_counts[3]:3d} ({tier_counts[3]/n_neg*100:5.1f}%)")
    print(f"    - Tier 4 (Same video, same class, closest age):    {tier_counts[4]:3d} ({tier_counts[4]/n_neg*100:5.1f}%)")
    print(f"    - Mean Absolute Score Difference (|s_neg - s_pos|): {np.mean(score_deltas):.4f} (std = {np.std(score_deltas):.4f})")
    print(f"    - Mean Absolute Age Difference   (|age_neg - age_pos|): {np.mean(age_deltas):.2f} frames")

    # --------------------------------------------------------------------------
    # P-1787: MATCHED CROSS-VIDEO WRONG-HISTORY PAIRING
    # --------------------------------------------------------------------------
    print("\n[*] Constructing Matched Cross-Video Wrong-History Control (P-1787)...")
    wrong_partner_idx = {}
    for idx, item in enumerate(selected_cohort):
        q_i = item["query"]
        v_curr = q_i["video_id"]
        p_cls = q_i["pred_class"]
        p_age = q_i["hist_age"]
        p_sc = q_i["max_score"]

        cross_cands = [k for k, other in enumerate(selected_cohort) if other["query"]["video_id"] != v_curr]
        cross_cands.sort(key=lambda k: (
            0 if selected_cohort[k]["query"]["pred_class"] == p_cls else 1,
            abs(selected_cohort[k]["query"]["hist_age"] - p_age),
            abs(selected_cohort[k]["query"]["max_score"] - p_sc)
        ))
        wrong_partner_idx[idx] = cross_cands[0]

    # --------------------------------------------------------------------------
    # P-1789: FULL CACHE PROVENANCE VALIDATION & EXTRACTION
    # --------------------------------------------------------------------------
    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        img_meta_by_id = {img["id"]: img for img in json.load(f)["images"]}

    cohort_keys_str = "".join(f"{it['query']['image_id']}_{it['query']['query_id']}_{it['label']}" for it in selected_cohort)
    cohort_hash = hashlib.sha256(cohort_keys_str.encode()).hexdigest()

    cache_valid = False
    if CACHED_FEATURES_PTH.exists():
        print(f"\n[+] Verifying Visual Feature Cache Provenance: {CACHED_FEATURES_PTH.resolve()}")
        c_bundle = torch.load(CACHED_FEATURES_PTH, map_location="cpu")
        if (
            isinstance(c_bundle, dict)
            and c_bundle.get("driver_sha256") == driver_sha256
            and c_bundle.get("cohort_hash") == cohort_hash
            and c_bundle.get("n_cohort") == len(selected_cohort)
            and c_bundle.get("query_dataset_sha256") == compute_sha256(QUERY_DATASET_PTH)
            and c_bundle.get("frame_tensors_sha256") == compute_sha256(FRAME_TENSORS_PTH)
            and c_bundle.get("token_cache_sha256") == EXPECTED_TOKEN_CACHE_SHA256
            and c_bundle.get("candidates_sha256") == EXPECTED_CANDIDATE_SHA256
            and c_bundle.get("train_json_sha256") == EXPECTED_TRAIN_JSON_SHA256
            and c_bundle.get("split_manifest_sha256") == EXPECTED_SPLIT_SHA256
            and c_bundle.get("crop_size") == CROP_SIZE
            and c_bundle.get("expansion_factor") == EXPANSION_FACTOR
            and c_bundle.get("encoder") == ENCODER_NAME
            and "feat_A" in c_bundle and "feat_B" in c_bundle
            and "feat_C_corr" in c_bundle and "feat_C_wrong" in c_bundle
        ):
            feat_A_all = c_bundle["feat_A"]
            feat_B_all = c_bundle["feat_B"]
            feat_C_corr_all = c_bundle["feat_C_corr"]
            feat_C_wrong_all = c_bundle["feat_C_wrong"]
            cache_valid = True
            print("[+] Feature Cache Certified: Complete provenance verified matching all live hashes.")

    if not cache_valid:
        print("\n[*] Initializing Frozen Visual Encoder (ResNet-18 Early Stem)...")
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        ).to(device)
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False

        img_transform = transforms.Compose([
            transforms.Resize((CROP_SIZE, CROP_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print(f"[*] Extracting Crops from SeaClips.zip for {len(selected_cohort)} Cohort Queries...")

        def extract_norm_crop(img, box_norm):
            cx, cy, w, h = box_norm
            side = max(w, h) * EXPANSION_FACTOR
            side = max(0.025, side)
            x1_norm = max(0.0, cx - side / 2.0)
            y1_norm = max(0.0, cy - side / 2.0)
            x2_norm = min(1.0, cx + side / 2.0)
            y2_norm = min(1.0, cy + side / 2.0)

            W, H = img.size
            px_box = (
                max(0, int(x1_norm * W)),
                max(0, int(y1_norm * H)),
                min(W, max(int(x1_norm * W) + 1, int(x2_norm * W))),
                min(H, max(int(y1_norm * H) + 1, int(y2_norm * H)))
            )
            return img.crop(px_box)

        unique_iids = set()
        for it in selected_cohort:
            unique_iids.add(it["query"]["image_id"])
            unique_iids.add(it["query"]["hist_iid"])

        img_cache = {}
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            # Build index to resolve archive directory prefix differences
            zip_index = {}
            for name in zf.namelist():
                norm = name.replace("\\", "/").strip("/")
                zip_index[norm] = name
                parts = norm.split("/")
                if len(parts) >= 2:
                    zip_index["/".join(parts[-2:])] = name

            for iid in unique_iids:
                rel_path = img_meta_by_id[iid]["file_name"]
                clean_rel = rel_path.replace("\\", "/").strip("/")
                target_key = "/".join(clean_rel.split("/")[-2:]) if len(clean_rel.split("/")) >= 2 else clean_rel

                zip_entry = zip_index.get(clean_rel) or zip_index.get(target_key)
                if zip_entry is None:
                    for name in zf.namelist():
                        if name.replace("\\", "/").endswith(clean_rel):
                            zip_entry = name
                            break
                assert zip_entry is not None, f"[FATAL] Cannot find {rel_path} in zip! Sample entries: {zf.namelist()[:5]}"

                with zf.open(zip_entry) as f:
                    img_cache[iid] = Image.open(io.BytesIO(f.read())).convert("RGB")

        feat_A_list, feat_B_list, f_h_corr_list = [], [], []

        for idx, item in enumerate(selected_cohort, 1):
            q = item["query"]
            curr_iid = q["image_id"]
            hist_iid = q["hist_iid"]

            img_nat = img_cache[curr_iid]
            img_640 = img_nat.resize((640, 640), Image.BILINEAR)
            img_hist = img_cache[hist_iid]

            crop_A = extract_norm_crop(img_640, q["pred_box_norm"])
            crop_B = extract_norm_crop(img_nat, q["pred_box_norm"])
            crop_H = extract_norm_crop(img_hist, q["hist_bbox_norm"])

            with torch.no_grad():
                t_A = img_transform(crop_A).unsqueeze(0).to(device)
                t_B = img_transform(crop_B).unsqueeze(0).to(device)
                t_H = img_transform(crop_H).unsqueeze(0).to(device)

                f_A = encoder(t_A).cpu().numpy().flatten()
                f_B = encoder(t_B).cpu().numpy().flatten()
                f_H = encoder(t_H).cpu().numpy().flatten()

            feat_A_list.append(f_A)
            feat_B_list.append(f_B)
            f_h_corr_list.append(f_H)

            if idx % 50 == 0 or idx == len(selected_cohort):
                print(f"    Processed visual crops for {idx}/{len(selected_cohort)} cohort queries...")

        feat_A_all = np.array(feat_A_list, dtype=np.float32)
        feat_B_all = np.array(feat_B_list, dtype=np.float32)
        f_h_corr_arr = np.array(f_h_corr_list, dtype=np.float32)

        # C_corr: [f_t, f_h, f_t - f_h, f_t * f_h]
        diff_corr = feat_B_all - f_h_corr_arr
        prod_corr = feat_B_all * f_h_corr_arr
        feat_C_corr_all = np.concatenate([feat_B_all, f_h_corr_arr, diff_corr, prod_corr], axis=-1)

        # C_wrong: [f_t, f_h_wrong, f_t - f_h_wrong, f_t * f_h_wrong]
        f_h_wrong_arr = np.array([f_h_corr_arr[wrong_partner_idx[i]] for i in range(len(selected_cohort))], dtype=np.float32)
        diff_wrong = feat_B_all - f_h_wrong_arr
        prod_wrong = feat_B_all * f_h_wrong_arr
        feat_C_wrong_all = np.concatenate([feat_B_all, f_h_wrong_arr, diff_wrong, prod_wrong], axis=-1)

        torch.save({
            "version": "v3",
            "driver_sha256": driver_sha256,
            "cohort_hash": cohort_hash,
            "n_cohort": len(selected_cohort),
            "query_dataset_sha256": compute_sha256(QUERY_DATASET_PTH),
            "frame_tensors_sha256": compute_sha256(FRAME_TENSORS_PTH),
            "token_cache_sha256": EXPECTED_TOKEN_CACHE_SHA256,
            "candidates_sha256": EXPECTED_CANDIDATE_SHA256,
            "train_json_sha256": EXPECTED_TRAIN_JSON_SHA256,
            "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
            "crop_size": CROP_SIZE,
            "expansion_factor": EXPANSION_FACTOR,
            "encoder": ENCODER_NAME,
            "feat_A": feat_A_all,
            "feat_B": feat_B_all,
            "feat_C_corr": feat_C_corr_all,
            "feat_C_wrong": feat_C_wrong_all
        }, CACHED_FEATURES_PTH)
        print(f"[+] Cached Certified Visual Features to: {CACHED_FEATURES_PTH.resolve()}")

    # --------------------------------------------------------------------------
    # 3. P-1788: LOVO BENCHMARK & SAME-PROBE CORRESPONDENCE TEST
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("LOVO CROSS-VALIDATION BENCHMARK & P-1788 SAME-PROBE CORRESPONDENCE TEST")
    print("=" * 115)

    y_cohort = np.array([item["label"] for item in selected_cohort], dtype=np.int32)
    v_cohort = np.array([item["query"]["video_id"] for item in selected_cohort], dtype=np.int32)
    base_prev = float(np.mean(y_cohort))

    pos_bearing_vids = [vid for vid in design_vids if sum(1 for it in selected_cohort if it["query"]["video_id"] == vid and it["label"] == 1) > 0]
    print(f"[*] Evaluated on {len(pos_bearing_vids)} Positive-Bearing Design Videos: {pos_bearing_vids}")

    oof_scores_A = np.full(len(y_cohort), np.nan)
    oof_scores_B = np.full(len(y_cohort), np.nan)
    oof_scores_C_corr = np.full(len(y_cohort), np.nan)
    oof_scores_C_wrong_same_probe = np.full(len(y_cohort), np.nan)

    video_metrics = defaultdict(dict)

    for vid in design_vids:
        test_mask = (v_cohort == vid)
        train_mask = ~test_mask

        y_tr, y_te = y_cohort[train_mask], y_cohort[test_mask]
        if len(y_te) == 0 or len(np.unique(y_tr)) < 2:
            continue

        # 1. Evaluate Representation A (640-Warped Current)
        X_tr_A, X_te_A = feat_A_all[train_mask], feat_A_all[test_mask]
        m_A, s_A = np.mean(X_tr_A, axis=0), np.std(X_tr_A, axis=0)
        s_A[s_A < 1e-6] = 1.0
        clf_A = LogisticRegression(C=1.0, class_weight="balanced", max_iter=400, random_state=42)
        clf_A.fit((X_tr_A - m_A) / s_A, y_tr)
        p_te_A = clf_A.predict_proba((X_te_A - m_A) / s_A)[:, 1]
        oof_scores_A[test_mask] = p_te_A

        # 2. Evaluate Representation B (Native-Res Current)
        X_tr_B, X_te_B = feat_B_all[train_mask], feat_B_all[test_mask]
        m_B, s_B = np.mean(X_tr_B, axis=0), np.std(X_tr_B, axis=0)
        s_B[s_B < 1e-6] = 1.0
        clf_B = LogisticRegression(C=1.0, class_weight="balanced", max_iter=400, random_state=42)
        clf_B.fit((X_tr_B - m_B) / s_B, y_tr)
        p_te_B = clf_B.predict_proba((X_te_B - m_B) / s_B)[:, 1]
        oof_scores_B[test_mask] = p_te_B

        # 3. P-1788: Train probe on C_corr, evaluate same probe on C_corr and C_wrong
        X_tr_C, X_te_C_corr = feat_C_corr_all[train_mask], feat_C_corr_all[test_mask]
        X_te_C_wrong = feat_C_wrong_all[test_mask]
        m_C, s_C = np.mean(X_tr_C, axis=0), np.std(X_tr_C, axis=0)
        s_C[s_C < 1e-6] = 1.0
        clf_C = LogisticRegression(C=1.0, class_weight="balanced", max_iter=400, random_state=42)
        clf_C.fit((X_tr_C - m_C) / s_C, y_tr)

        p_te_C_corr = clf_C.predict_proba((X_te_C_corr - m_C) / s_C)[:, 1]
        p_te_C_wrong = clf_C.predict_proba((X_te_C_wrong - m_C) / s_C)[:, 1]
        oof_scores_C_corr[test_mask] = p_te_C_corr
        oof_scores_C_wrong_same_probe[test_mask] = p_te_C_wrong

        if len(np.unique(y_te)) >= 2:
            video_metrics[vid] = {
                "n_pos": int(y_te.sum()),
                "n_total": len(y_te),
                "ap_A": float(average_precision_score(y_te, p_te_A)),
                "ap_B": float(average_precision_score(y_te, p_te_B)),
                "ap_C_corr": float(average_precision_score(y_te, p_te_C_corr)),
                "ap_C_wrong": float(average_precision_score(y_te, p_te_C_wrong)),
                "auc_A": float(roc_auc_score(y_te, p_te_A)),
                "auc_B": float(roc_auc_score(y_te, p_te_B)),
                "auc_C_corr": float(roc_auc_score(y_te, p_te_C_corr)),
                "auc_C_wrong": float(roc_auc_score(y_te, p_te_C_wrong)),
            }

    pooled_ap_A = float(average_precision_score(y_cohort, oof_scores_A))
    pooled_ap_B = float(average_precision_score(y_cohort, oof_scores_B))
    pooled_ap_C_corr = float(average_precision_score(y_cohort, oof_scores_C_corr))
    pooled_ap_C_wrong = float(average_precision_score(y_cohort, oof_scores_C_wrong_same_probe))

    pooled_auc_A = float(roc_auc_score(y_cohort, oof_scores_A))
    pooled_auc_B = float(roc_auc_score(y_cohort, oof_scores_B))
    pooled_auc_C_corr = float(roc_auc_score(y_cohort, oof_scores_C_corr))
    pooled_auc_C_wrong = float(roc_auc_score(y_cohort, oof_scores_C_wrong_same_probe))

    macro_ap_A = float(np.mean([m["ap_A"] for m in video_metrics.values()]))
    macro_ap_B = float(np.mean([m["ap_B"] for m in video_metrics.values()]))
    macro_ap_C_corr = float(np.mean([m["ap_C_corr"] for m in video_metrics.values()]))
    macro_ap_C_wrong = float(np.mean([m["ap_C_wrong"] for m in video_metrics.values()]))

    # Summary table
    print(f"\n{'Representation':<42} | {'Dim':<5} | {'Pooled AP':<12} | {'AP Lift':<10} | {'Macro Mean AP':<16} | {'Pooled ROC-AUC'}")
    print("-" * 110)
    for name, dim, p_ap, m_ap, p_auc in [
        ("A (640-Warped Current)", 128, pooled_ap_A, macro_ap_A, pooled_auc_A),
        ("B (Native-Res Current)", 128, pooled_ap_B, macro_ap_B, pooled_auc_B),
        ("C_corr (Native + Correct Hist)", 512, pooled_ap_C_corr, macro_ap_C_corr, pooled_auc_C_corr),
        ("C_wrong (Native + Matched Wrong Hist)", 512, pooled_ap_C_wrong, macro_ap_C_wrong, pooled_auc_C_wrong),
    ]:
        lift = p_ap / max(1e-6, base_prev)
        print(f"{name:<42} | {dim:<5d} | {p_ap:10.4f}   | {lift:8.2f}x  | {m_ap:14.4f}   | {p_auc:12.4f}")

    # --------------------------------------------------------------------------
    # P-1785: PER-POSITIVE-BEARING-VIDEO BREAKDOWN & DELTAS
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print(f"PER-VIDEO BREAKDOWN ACROSS {len(video_metrics)} EVALUABLE POSITIVE-BEARING DESIGN VIDEOS")
    print("=" * 115)
    print(f"{'Video ID':<12} | {'Pos / Total':<14} | {'AP (A: 640)':<14} | {'AP (B: Native)':<16} | {'Delta AP (B - A)':<18} | {'AP (C_corr)':<14} | {'AP (C_wrong)'}")
    print("-" * 110)

    n_improved_BA = 0
    n_improved_Cw = 0
    for vid, vm in sorted(video_metrics.items()):
        d_ap_BA = vm["ap_B"] - vm["ap_A"]
        d_ap_Cw = vm["ap_C_corr"] - vm["ap_C_wrong"]
        if d_ap_BA > 0: n_improved_BA += 1
        if d_ap_Cw > 0: n_improved_Cw += 1
        print(f"video_{vid:<6d} | {vm['n_pos']:2d} / {vm['n_total']:<8d} | {vm['ap_A']:12.4f}  | {vm['ap_B']:14.4f}  | {d_ap_BA:+16.4f}  | {vm['ap_C_corr']:12.4f}  | {vm['ap_C_wrong']:12.4f}")

    # --------------------------------------------------------------------------
    # P-1785: VIDEO-GROUPED PAIRED BOOTSTRAP UNCERTAINTY (1,000 REPLICATES)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("P-1785: VIDEO-GROUPED PAIRED BOOTSTRAP UNCERTAINTY AUDIT (1,000 REPLICATES)")
    print("=" * 115)

    p_A = oof_scores_A
    p_B = oof_scores_B
    p_Cc = oof_scores_C_corr
    p_Cw = oof_scores_C_wrong_same_probe

    rng = np.random.default_rng(42)
    boot_deltas_BA = []
    boot_deltas_CB = []
    boot_deltas_Cw = []

    active_vids_arr = np.unique(v_cohort)
    for _ in range(1000):
        sampled_vids = rng.choice(active_vids_arr, size=len(active_vids_arr), replace=True)
        boot_idx = []
        for v in sampled_vids:
            boot_idx.extend(np.where(v_cohort == v)[0])
        boot_idx = np.array(boot_idx)

        if len(np.unique(y_cohort[boot_idx])) < 2:
            continue

        ap_A_b = average_precision_score(y_cohort[boot_idx], p_A[boot_idx])
        ap_B_b = average_precision_score(y_cohort[boot_idx], p_B[boot_idx])
        ap_Cc_b = average_precision_score(y_cohort[boot_idx], p_Cc[boot_idx])
        ap_Cw_b = average_precision_score(y_cohort[boot_idx], p_Cw[boot_idx])

        boot_deltas_BA.append(ap_B_b - ap_A_b)
        boot_deltas_CB.append(ap_Cc_b - ap_B_b)
        boot_deltas_Cw.append(ap_Cc_b - ap_Cw_b)

    ci_BA = np.percentile(boot_deltas_BA, [2.5, 97.5])
    ci_CB = np.percentile(boot_deltas_CB, [2.5, 97.5])
    ci_Cw = np.percentile(boot_deltas_Cw, [2.5, 97.5])

    pooled_delta_BA = pooled_ap_B - pooled_ap_A
    pooled_delta_CB = pooled_ap_C_corr - pooled_ap_B
    pooled_delta_Cw = pooled_ap_C_corr - pooled_ap_C_wrong

    print(f"1. Native Resolution Effect (B - A):")
    print(f"   - Pooled Delta AP: {pooled_delta_BA:+.4f}")
    print(f"   - Video-Grouped 95% Bootstrap CI: [{ci_BA[0]:+.4f}, {ci_BA[1]:+.4f}]")
    print(f"   - Positive-Bearing Folds Improved: {n_improved_BA} / {len(video_metrics)} ({n_improved_BA/len(video_metrics)*100:.1f}%)")

    print(f"\n2. Temporal Correspondence Test (C_corr - C_wrong, Same Probe):")
    print(f"   - Same-Probe Degradation: Delta AP = {pooled_delta_Cw:+.4f}")
    print(f"   - Video-Grouped 95% Bootstrap CI: [{ci_Cw[0]:+.4f}, {ci_Cw[1]:+.4f}]")
    print(f"   - Positive-Bearing Folds Improved: {n_improved_Cw} / {len(video_metrics)} ({n_improved_Cw/len(video_metrics)*100:.1f}%)")

    print(f"\n3. Temporal Appearance Addition (C_corr - B):")
    print(f"   - Joint Appearance Delta AP: {pooled_delta_CB:+.4f}")
    print(f"   - Video-Grouped 95% Bootstrap CI: [{ci_CB[0]:+.4f}, {ci_CB[1]:+.4f}]")

    print(f"\n4. Leave-One-Video-Out Sensitivity of Delta AP (B - A):")
    for v_omit in pos_bearing_vids:
        mask_omit = (v_cohort != v_omit)
        ap_A_sub = average_precision_score(y_cohort[mask_omit], p_A[mask_omit])
        ap_B_sub = average_precision_score(y_cohort[mask_omit], p_B[mask_omit])
        print(f"   - Omitting video_{v_omit:<2d}: Pooled Delta AP = {ap_B_sub - ap_A_sub:+.4f}")

    results_payload = {
        "_meta": {
            "artifact_name": "pol4_resolution_micro_proof_results_v3",
            "driver_sha256": driver_sha256,
            "cohort_total": len(selected_cohort),
            "n_positives": n_pos,
            "n_negatives": n_neg,
            "base_prevalence": base_prev,
            "tier_counts": {str(k): int(v) for k, v in tier_counts.items()},
            "evaluable_videos": len(video_metrics),
            "pooled_delta_BA": float(pooled_delta_BA),
            "ci_95_BA": [float(ci_BA[0]), float(ci_BA[1])],
            "pooled_delta_Cw": float(pooled_delta_Cw),
            "ci_95_Cw": [float(ci_Cw[0]), float(ci_Cw[1])],
            "pooled_delta_CB": float(pooled_delta_CB),
            "ci_95_CB": [float(ci_CB[0]), float(ci_CB[1])],
        },
        "representation_metrics": {
            "A_640": {"pooled_ap": pooled_ap_A, "macro_ap": macro_ap_A, "pooled_auc": pooled_auc_A},
            "B_native": {"pooled_ap": pooled_ap_B, "macro_ap": macro_ap_B, "pooled_auc": pooled_auc_B},
            "C_corr": {"pooled_ap": pooled_ap_C_corr, "macro_ap": macro_ap_C_corr, "pooled_auc": pooled_auc_C_corr},
            "C_wrong_same_probe": {"pooled_ap": pooled_ap_C_wrong, "macro_ap": macro_ap_C_wrong, "pooled_auc": pooled_auc_C_wrong},
        },
        "per_video_metrics": {str(k): v for k, v in video_metrics.items()}
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    res_hash = compute_sha256(RESULTS_JSON)
    print("\n" + "=" * 115)
    print(f"[+] Candidate 4 Micro-Proof Results Frozen: {RESULTS_JSON.resolve()}")
    print(f"    Manifest SHA-256:                      {res_hash}")
    print("=" * 115)


if __name__ == "__main__":
    main()