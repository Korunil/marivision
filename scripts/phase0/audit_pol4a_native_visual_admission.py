"""
D:/MARIVISION/H0_workbench/eval_tools/audit_pol4a_native_visual_admission.py
Candidate 4A: Native Visual Operational Admission Benchmark & Verifier Freeze (v3)

Strict Compliance Checklist:
  - D-3726..D-3737: Design cohort ONLY (10 videos, 206 frames); calibration & test remain untouched.
  - P-1793: Primary Candidate-4A gate is strictly visual-only (Model 1 vs Model 2). Composite is exploratory.
  - P-1794: Gated artifact creation: freezes pol4a_candidate4a_visual_verifier.pth iff visual delta_f1 > 0.
  - P-1795: OOF integrity assertions: verifies 1,915 finite scores, exact key equality, no silent missing scores.
  - P-1796 / P-1797: Per-video operational breakdown reports admitted operational-utility positives explicitly.
  - P-1798: Full-fit score-scale provenance audit recording OOF vs full-fit admission counts and quantiles.
  - P-1767: Implements deterministic equal-F1 tie-breaking favoring higher tau, then simpler model.
  - Encoder Specification: torchvision.models.ResNet18_Weights.IMAGENET1K_V1.
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
from sklearn.neural_network import MLPClassifier

OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
TRAIN_JSON = Path("D:/MARIVISION/seaclips-train.json")
SPLIT_MANIFEST_JSON = OUT_DIR / "pol2_candidate2m_micro_split.json"
QUERY_DATASET_PTH = OUT_DIR / "pol3v_design_queries_dataset_v7.pth"
FRAME_TENSORS_PTH = OUT_DIR / "pol3v_design_frames_tensors_v7.pth"
TOKEN_CACHE_PTH = OUT_DIR / "cache_m0_candidate2m_design_calib.pth"
CANDIDATES_JSON = OUT_DIR / "pol2_candidate_windows_v1.json"
ZIP_PATH = Path("D:/MARIVISION/SeaClips.zip")

RESULTS_JSON = OUT_DIR / "pol4a_native_visual_admission_results.json"
CACHED_VISUAL_FEATURES_PTH = OUT_DIR / "pol4a_native_visual_features_1915.pth"
FINAL_VERIFIER_PTH = OUT_DIR / "pol4a_candidate4a_visual_verifier.pth"

EXPECTED_CANDIDATE_SHA256 = "f29e0b8a5cbfc46641222d84a6d370c3e9bc51ca4ca1124b7753078ca7c1e270"
EXPECTED_TRAIN_JSON_SHA256 = "edc32bdbd6b792eb83b6daf11e14708232f1f4a9e06153ebebd5bad1caef9b35"
EXPECTED_SPLIT_SHA256 = "7da471d2793fe3369508200dec50422bb76492860c67a1a61871645471856772"
EXPECTED_TOKEN_CACHE_SHA256 = "d1e6407722c59968f87e0ae9cfdead28cacf89924dcd765ca41cf8993da4dd4c"

CROP_SIZE = 128
EXPANSION_FACTOR = 1.5
SCORE_THRESHOLD = 0.20
EVAL_IOU_THRESHOLD = 0.30
ENCODER_NAME = "torchvision_resnet18_IMAGENET1K_V1_layer2"
TAU_GRID = [0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def compute_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def box_iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[0] + b1[2], b2[0] + b2[2]), min(b1[1] + b1[3], b2[1] + b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = b1[2] * b1[3] + b2[2] * b2[3] - inter
    return inter / union if union > 0.0 else 0.0


def evaluate_operational_top100(logits, boxes, gt_boxes_px, gt_cats, label_to_cat_id, img_w, img_h):
    prob = logits.sigmoid()
    num_classes = prob.shape[-1]
    topk_scores, topk_idx = torch.topk(prob.flatten(), min(100, prob.numel()))
    q_idx = torch.div(topk_idx, num_classes, rounding_mode="floor")
    c_idx = topk_idx % num_classes
    boxes_ranked = boxes[q_idx]

    pred_cat_ids = np.array([label_to_cat_id[int(lbl.item())] for lbl in c_idx])
    pred_boxes_px = []
    for b in boxes_ranked:
        cx, cy, w, h = b.tolist()
        pred_boxes_px.append([(cx - w / 2.0) * img_w, (cy - h / 2.0) * img_h, w * img_w, h * img_h])
    pred_scores = topk_scores.cpu().numpy()

    n_gt = len(gt_cats)
    gt_matched = [False] * n_gt
    tp = 0
    fp = 0

    for p_idx in range(len(pred_scores)):
        if pred_scores[p_idx] >= SCORE_THRESHOLD:
            p_box = pred_boxes_px[p_idx]
            p_cat = pred_cat_ids[p_idx]
            best_iou = -1.0
            best_g = -1
            for g_idx in range(n_gt):
                if not gt_matched[g_idx] and gt_cats[g_idx] == p_cat:
                    iou_g = box_iou(gt_boxes_px[g_idx], p_box)
                    if iou_g >= EVAL_IOU_THRESHOLD and iou_g > best_iou:
                        best_iou = iou_g
                        best_g = g_idx
            if best_g >= 0:
                gt_matched[best_g] = True
                tp += 1
            else:
                fp += 1

    fn = n_gt - sum(gt_matched)
    return tp, fp, fn


def main():
    driver_sha256 = compute_sha256(Path(__file__))
    print("=" * 115)
    print("CANDIDATE 4A: NATIVE VISUAL OPERATIONAL ADMISSION PROOF (v3)")
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

    # Reconstruct exact inference-predicted current bounding boxes
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

    # Filter to all 1,915 threshold-active queries
    active_indices = [idx for idx, q in enumerate(cached_queries) if q["is_active"]]
    active_queries = [cached_queries[idx] for idx in active_indices]

    n_active = len(active_queries)
    n_pos = sum(1 for q in active_queries if q["label_op_util"] == 1)
    base_prev = n_pos / n_active

    assert n_active == 1915, f"[FATAL] Expected 1,915 active queries, got {n_active}"
    assert n_pos == 92, f"[FATAL] Expected 92 operational positives, got {n_pos}"

    print(f"\n[*] Natural Operational Population Loaded:")
    print(f"    - Total Threshold-Active Queries: {n_active:,}")
    print(f"    - Operational Utility Positives:  {n_pos:5d}")
    print(f"    - Natural Positive Prevalence:    {base_prev * 100:.2f}%\n")

    # --------------------------------------------------------------------------
    # 1. VISUAL FEATURE EXTRACTION WITH COMPLETE CACHE PROVENANCE
    # --------------------------------------------------------------------------
    cache_valid = False
    if CACHED_VISUAL_FEATURES_PTH.exists():
        print(f"[+] Verifying Visual Feature Cache Provenance: {CACHED_VISUAL_FEATURES_PTH.resolve()}")
        c_bundle = torch.load(CACHED_VISUAL_FEATURES_PTH, map_location="cpu")
        if (
            isinstance(c_bundle, dict)
            and c_bundle.get("driver_sha256") == driver_sha256
            and c_bundle.get("n_active") == n_active
            and c_bundle.get("query_dataset_sha256") == compute_sha256(QUERY_DATASET_PTH)
            and c_bundle.get("frame_tensors_sha256") == compute_sha256(FRAME_TENSORS_PTH)
            and c_bundle.get("candidates_sha256") == EXPECTED_CANDIDATE_SHA256
            and c_bundle.get("train_json_sha256") == EXPECTED_TRAIN_JSON_SHA256
            and c_bundle.get("crop_size") == CROP_SIZE
            and c_bundle.get("expansion_factor") == EXPANSION_FACTOR
            and c_bundle.get("encoder") == ENCODER_NAME
            and "feat_visual" in c_bundle
        ):
            feat_visual_all = c_bundle["feat_visual"]
            cache_valid = True
            print("[+] Feature Cache Certified: Provenance verified against all live hashes.")

    if not cache_valid:
        print("[*] Initializing Frozen Visual Encoder (ResNet-18 Early Stem, 128-D)...")
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
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

        with open(TRAIN_JSON, "r", encoding="utf-8") as f:
            img_meta_by_id = {img["id"]: img for img in json.load(f)["images"]}

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

        unique_iids = sorted(list(set(q["image_id"] for q in active_queries)))
        img_cache = {}
        print(f"[*] Preloading {len(unique_iids)} Target Frames from SeaClips.zip...")

        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
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
                assert zip_entry is not None, f"[FATAL] Cannot find {rel_path} in zip!"

                with zf.open(zip_entry) as f:
                    img_cache[iid] = Image.open(io.BytesIO(f.read())).convert("RGB")

        print(f"[*] Extracting Native Visual Crops for {n_active:,} Active Queries...")
        crop_tensors = []
        for idx, q in enumerate(active_queries):
            img_nat = img_cache[q["image_id"]]
            crop = extract_norm_crop(img_nat, q["pred_box_norm"])
            crop_tensors.append(img_transform(crop))

        print("[*] Forwarding Crops through Frozen ResNet-18 Early Stem in Batches...")
        feat_visual_list = []
        batch_size = 64
        with torch.no_grad():
            for i in range(0, len(crop_tensors), batch_size):
                batch = torch.stack(crop_tensors[i:i + batch_size]).to(device)
                feats = encoder(batch).cpu().numpy()
                feat_visual_list.append(feats)

        feat_visual_all = np.concatenate(feat_visual_list, axis=0).astype(np.float32)
        assert len(feat_visual_all) == n_active

        torch.save({
            "version": "v1",
            "driver_sha256": driver_sha256,
            "n_active": n_active,
            "query_dataset_sha256": compute_sha256(QUERY_DATASET_PTH),
            "frame_tensors_sha256": compute_sha256(FRAME_TENSORS_PTH),
            "token_cache_sha256": EXPECTED_TOKEN_CACHE_SHA256,
            "candidates_sha256": EXPECTED_CANDIDATE_SHA256,
            "train_json_sha256": EXPECTED_TRAIN_JSON_SHA256,
            "crop_size": CROP_SIZE,
            "expansion_factor": EXPANSION_FACTOR,
            "encoder": ENCODER_NAME,
            "feat_visual": feat_visual_all
        }, CACHED_VISUAL_FEATURES_PTH)
        print(f"[+] Cached Certified Visual Features to: {CACHED_VISUAL_FEATURES_PTH.resolve()}")

    # --------------------------------------------------------------------------
    # 2. P-1793: LOVO BENCHMARK ACROSS MODELS 1, 2, AND 3
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("P-1793: LOVO CROSS-VALIDATION BENCHMARK (N = 1,915 ACTIVE QUERIES, PREVALENCE = 4.80%)")
    print("=" * 115)

    y_act = np.array([q["label_op_util"] for q in active_queries], dtype=np.int32)
    v_act = np.array([q["video_id"] for q in active_queries], dtype=np.int32)
    X_scalar_15 = np.array([q["features"] for q in active_queries], dtype=np.float32)

    X_visual_128 = feat_visual_all
    X_composite_143 = np.concatenate([X_visual_128, X_scalar_15], axis=-1)

    model_configs = {
        "Model 1: Visual Logistic Reg (128-D) [GATE]": {
            "features": X_visual_128,
            "type": "logistic",
            "category": "visual_only"
        },
        "Model 2: Visual Shallow MLP (128-D) [GATE]": {
            "features": X_visual_128,
            "type": "mlp",
            "category": "visual_only"
        },
        "Model 3: Composite MLP (143-D) [EXPLORATORY]": {
            "features": X_composite_143,
            "type": "mlp",
            "category": "composite_exploratory"
        }
    }

    model_oof_scores = {}
    model_metrics = {}

    for m_name, cfg in model_configs.items():
        X_data = cfg["features"]
        m_type = cfg["type"]

        lovo_scores = np.full(n_active, np.nan)
        video_aps = {}
        video_aucs = {}

        for vid in design_vids:
            test_mask = (v_act == vid)
            train_mask = ~test_mask

            X_tr, y_tr = X_data[train_mask], y_act[train_mask]
            X_te, y_te = X_data[test_mask], y_act[test_mask]

            if len(y_te) == 0 or len(np.unique(y_tr)) < 2:
                continue

            mean_tr = np.mean(X_tr, axis=0)
            std_tr = np.std(X_tr, axis=0)
            std_tr[std_tr < 1e-6] = 1.0

            X_tr_std = (X_tr - mean_tr) / std_tr
            X_te_std = (X_te - mean_tr) / std_tr

            if m_type == "logistic":
                clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=400, random_state=42)
                clf.fit(X_tr_std, y_tr)
                p_te = clf.predict_proba(X_te_std)[:, 1]
            else:
                pos_idx = np.where(y_tr == 1)[0]
                neg_idx = np.where(y_tr == 0)[0]
                rng = np.random.default_rng(42 * 1000 + vid)
                oversampled_pos = rng.choice(pos_idx, size=len(neg_idx), replace=True)
                bal_idx = np.concatenate([neg_idx, oversampled_pos])
                rng.shuffle(bal_idx)

                clf = MLPClassifier(hidden_layer_sizes=(32,), activation="relu", alpha=0.05, max_iter=350, random_state=42)
                clf.fit(X_tr_std[bal_idx], y_tr[bal_idx])
                p_te = clf.predict_proba(X_te_std)[:, 1]

            lovo_scores[test_mask] = p_te

            if len(np.unique(y_te)) >= 2:
                video_aps[vid] = float(average_precision_score(y_te, p_te))
                video_aucs[vid] = float(roc_auc_score(y_te, p_te))

        assert np.isfinite(lovo_scores).all(), f"[FATAL] Model {m_name} produced non-finite or NaN OOF scores!"

        pooled_ap = float(average_precision_score(y_act, lovo_scores))
        pooled_auc = float(roc_auc_score(y_act, lovo_scores))
        macro_ap = float(np.mean(list(video_aps.values()))) if video_aps else 0.0
        macro_auc = float(np.mean(list(video_aucs.values()))) if video_aucs else 0.5

        model_oof_scores[m_name] = lovo_scores
        model_metrics[m_name] = {
            "pooled_ap": pooled_ap,
            "pooled_auc": pooled_auc,
            "macro_ap": macro_ap,
            "macro_auc": macro_auc,
            "video_aps": video_aps,
            "video_aucs": video_aucs
        }

    print(f"\n{'Model Configuration':<46} | {'Dim':<5} | {'Pooled AP':<12} | {'AP Lift':<10} | {'Macro Mean AP':<16} | {'Pooled ROC-AUC'}")
    print("-" * 118)
    for m_name, res in model_metrics.items():
        dim = model_configs[m_name]["features"].shape[-1]
        lift = res["pooled_ap"] / max(1e-6, base_prev)
        print(f"{m_name:<46} | {dim:<5d} | {res['pooled_ap']:10.4f}   | {lift:8.2f}x  | {res['macro_ap']:14.4f}   | {res['pooled_auc']:12.4f}")

    # --------------------------------------------------------------------------
    # 3. P-1795: OPERATIONAL GATE SWEEP WITH STRICT OOF INTEGRITY (206 FRAMES)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("OPERATIONAL GATE SWEEP: EXACT TOP-100 RERUN (206 FRAMES, BASELINE F1 = 0.431718)")
    print("=" * 115)

    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        coco_full = json.load(f)
    cat_ids = sorted(c["id"] for c in coco_full["categories"])
    label_to_cat_id = {i: cat_id for i, cat_id in enumerate(cat_ids)}

    base_tp, base_fp, base_fn = 0, 0, 0
    for fr in frame_records:
        tpd, fpd, fnd = evaluate_operational_top100(
            fr["d_logits"], fr["d_boxes"], fr["gt_boxes_px"], fr["gt_cats"], label_to_cat_id, fr["img_w"], fr["img_h"]
        )
        base_tp += tpd
        base_fp += fpd
        base_fn += fnd

    base_prec = base_tp / max(1, base_tp + base_fp)
    base_rec = base_tp / max(1, base_tp + base_fn)
    base_f1 = 2 * base_prec * base_rec / max(1e-7, base_prec + base_rec)

    assert base_tp == 196 and base_fp == 299 and base_fn == 217
    assert abs(base_f1 - 0.431718) < 1e-5

    active_key_set = set((q["image_id"], q["query_id"]) for q in active_queries)
    positive_key_set = set((q["image_id"], q["query_id"]) for q in active_queries if q["label_op_util"] == 1)
    assert len(active_key_set) == 1915
    assert len(positive_key_set) == 92

    sweep_results = {}

    for m_name in model_configs.keys():
        print(f"\n[*] Sweeping Operational Gate for {m_name}...")
        oof_scores = model_oof_scores[m_name]

        score_map = {}
        for idx_act, act_idx in enumerate(active_indices):
            q = cached_queries[act_idx]
            key = (q["image_id"], q["query_id"])
            score_map[key] = float(oof_scores[idx_act])

        assert len(score_map) == 1915, f"[FATAL] score_map length mismatch: {len(score_map)} != 1915"
        assert set(score_map.keys()) == active_key_set, "[FATAL] score_map keys do not match active_key_set exactly!"

        print(f"{'Threshold (tau)':<16} | {'Admitted (Active / 1,915)':<26} | {'Retained Op Pos':<18} | {'TP':<6} | {'FP':<6} | {'FN':<6} | {'Precision':<11} | {'Recall':<10} | {'F1':<10} | {'Delta F1 vs D'}")
        print("-" * 140)

        best_tau = None
        best_f1 = -1.0
        best_delta_f1 = -999.0
        best_record = None
        tau_records = {}

        for tau in TAU_GRID:
            tot_tp, tot_fp, tot_fn = 0, 0, 0
            admitted_count = 0
            retained_pos = 0

            for fr in frame_records:
                n_t = fr["n_temp"]
                if n_t > 0:
                    joint_logits = torch.cat([fr["d_logits"], fr["temp_logits"]], dim=0).clone()
                    joint_boxes = torch.cat([fr["d_boxes"], fr["temp_boxes"]], dim=0)

                    for q_t in range(n_t):
                        key = (fr["image_id"], q_t)
                        if key in active_key_set:
                            assert key in score_map, f"[FATAL] Key {key} missing from score_map!"
                            sc = score_map[key]
                            if sc >= tau:
                                admitted_count += 1
                                if key in positive_key_set:
                                    retained_pos += 1
                            else:
                                joint_logits[300 + q_t, :] = -1e9
                        else:
                            joint_logits[300 + q_t, :] = -1e9
                else:
                    joint_logits = fr["d_logits"]
                    joint_boxes = fr["d_boxes"]

                tpf, fpf, fnf = evaluate_operational_top100(
                    joint_logits, joint_boxes, fr["gt_boxes_px"], fr["gt_cats"], label_to_cat_id, fr["img_w"], fr["img_h"]
                )
                tot_tp += tpf
                tot_fp += fpf
                tot_fn += fnf

            prec = tot_tp / max(1, tot_tp + tot_fp)
            rec = tot_tp / max(1, tot_tp + tot_fn)
            f1 = 2 * prec * rec / max(1e-7, prec + rec)
            delta_f1 = f1 - base_f1

            tau_records[str(tau)] = {
                "tau": tau,
                "admitted": admitted_count,
                "retained_pos": retained_pos,
                "tp": tot_tp,
                "fp": tot_fp,
                "fn": tot_fn,
                "prec": float(prec),
                "rec": float(rec),
                "f1": float(f1),
                "delta_f1": float(delta_f1)
            }

            print(f"tau = {tau:<10.2f} | {admitted_count:6d} / 1,915 active (2,045 tot) | {retained_pos:2d} / 92 ({retained_pos/92*100:4.1f}%) | {tot_tp:4d} | {tot_fp:4d} | {tot_fn:4d} | {prec:9.4f} | {rec:8.4f} | {f1:8.4f} | {delta_f1:+8.4f}")

            # P-1767: Tie-break favoring higher tau (stronger suppression)
            is_better = False
            if f1 > best_f1 + 1e-12:
                is_better = True
            elif abs(f1 - best_f1) <= 1e-12:
                if best_tau is None or tau > best_tau:
                    is_better = True

            if is_better:
                best_f1 = f1
                best_tau = tau
                best_delta_f1 = delta_f1
                best_record = tau_records[str(tau)]

        sweep_results[m_name] = {
            "optimal_tau": best_tau,
            "optimal_f1": float(best_f1),
            "optimal_delta_f1": float(best_delta_f1),
            "best_record": best_record,
            "sweep_grid": tau_records
        }

    # --------------------------------------------------------------------------
    # 4. P-1793: PRIMARY VISUAL-ONLY GATE EVALUATION & MODEL SELECTION
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("P-1793: PRIMARY VISUAL-ONLY CANDIDATE-4A GATE EVALUATION")
    print("=" * 115)

    m1_name = "Model 1: Visual Logistic Reg (128-D) [GATE]"
    m2_name = "Model 2: Visual Shallow MLP (128-D) [GATE]"
    m3_name = "Model 3: Composite MLP (143-D) [EXPLORATORY]"

    res_m1 = sweep_results[m1_name]
    res_m2 = sweep_results[m2_name]
    res_m3 = sweep_results[m3_name]

    if res_m1["optimal_f1"] > res_m2["optimal_f1"] + 1e-12:
        winning_visual_name = m1_name
    elif res_m2["optimal_f1"] > res_m1["optimal_f1"] + 1e-12:
        winning_visual_name = m2_name
    else:
        if res_m1["optimal_tau"] > res_m2["optimal_tau"]:
            winning_visual_name = m1_name
        elif res_m2["optimal_tau"] > res_m1["optimal_tau"]:
            winning_visual_name = m2_name
        else:
            winning_visual_name = m1_name

    win_res = sweep_results[winning_visual_name]
    win_tau = win_res["optimal_tau"]
    win_delta_f1 = win_res["optimal_delta_f1"]

    visual_gate_passed = win_delta_f1 > 0

    print(f"[*] Visual-Only Models Compared:")
    print(f"    - {m1_name}: Best F1 = {res_m1['optimal_f1']:.6f} (Delta F1 = {res_m1['optimal_delta_f1']:+.4f} at tau = {res_m1['optimal_tau']:.2f})")
    print(f"    - {m2_name}: Best F1 = {res_m2['optimal_f1']:.6f} (Delta F1 = {res_m2['optimal_delta_f1']:+.4f} at tau = {res_m2['optimal_tau']:.2f})")
    print(f"    - Selected Winning Visual Architecture: {winning_visual_name}")
    print(f"    - Primary Visual-Only Operational Verdict: {'PASSED' if visual_gate_passed else 'FAILED'} (Delta F1 = {win_delta_f1:+.4f})")

    print(f"\n[*] Secondary Exploratory Composite Diagnostic (D-3732):")
    print(f"    - {m3_name}: Best F1 = {res_m3['optimal_f1']:.6f} (Delta F1 = {res_m3['optimal_delta_f1']:+.4f} at tau = {res_m3['optimal_tau']:.2f})")
    if res_m3["optimal_delta_f1"] > win_delta_f1:
        print(f"    - Diagnostic Observation: Composite visual+scalar evidence achieves {res_m3['optimal_delta_f1'] - win_delta_f1:+.4f} additional F1 over visual-only.")

    # --------------------------------------------------------------------------
    # 5. P-1796 / P-1797: PER-VIDEO OPERATIONAL DECOMPOSITION (WITH adm_pos_v)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print(f"P-1796 / P-1797: PER-VIDEO OPERATIONAL DECOMPOSITION: {winning_visual_name} AT tau* = {win_tau:.2f}")
    print("=" * 115)
    print(f"{'Video ID':<12} | {'Frames':<8} | {'Active (Pos)':<16} | {'Admitted (Op Pos)':<20} | {'Baseline (TP/FP/FN/F1)':<24} | {'Model (TP/FP/FN/F1)':<24} | {'Delta F1'}")
    print("-" * 126)

    winning_oof_scores = model_oof_scores[winning_visual_name]
    win_score_map = { (cached_queries[act_idx]["image_id"], cached_queries[act_idx]["query_id"]): float(winning_oof_scores[i]) for i, act_idx in enumerate(active_indices) }

    per_video_decomp = {}
    for vid in design_vids:
        vid_frames = [fr for fr in frame_records if fr["video_id"] == vid]
        b_tp_v, b_fp_v, b_fn_v = 0, 0, 0
        m_tp_v, m_fp_v, m_fn_v = 0, 0, 0
        adm_v, adm_pos_v, act_v, pos_v = 0, 0, 0, 0

        for fr in vid_frames:
            tpb, fpb, fnb = evaluate_operational_top100(
                fr["d_logits"], fr["d_boxes"], fr["gt_boxes_px"], fr["gt_cats"], label_to_cat_id, fr["img_w"], fr["img_h"]
            )
            b_tp_v += tpb
            b_fp_v += fpb
            b_fn_v += fnb

            n_t = fr["n_temp"]
            if n_t > 0:
                j_logits = torch.cat([fr["d_logits"], fr["temp_logits"]], dim=0).clone()
                j_boxes = torch.cat([fr["d_boxes"], fr["temp_boxes"]], dim=0)

                for q_t in range(n_t):
                    key = (fr["image_id"], q_t)
                    if key in active_key_set:
                        act_v += 1
                        if key in positive_key_set:
                            pos_v += 1
                        if win_score_map[key] >= win_tau:
                            adm_v += 1
                            if key in positive_key_set:
                                adm_pos_v += 1
                        else:
                            j_logits[300 + q_t, :] = -1e9
                    else:
                        j_logits[300 + q_t, :] = -1e9
            else:
                j_logits = fr["d_logits"]
                j_boxes = fr["d_boxes"]

            tpm, fpm, fnm = evaluate_operational_top100(
                j_logits, j_boxes, fr["gt_boxes_px"], fr["gt_cats"], label_to_cat_id, fr["img_w"], fr["img_h"]
            )
            m_tp_v += tpm
            m_fp_v += fpm
            m_fn_v += fnm

        f1_b_v = 2 * b_tp_v / max(1e-7, 2 * b_tp_v + b_fp_v + b_fn_v)
        f1_m_v = 2 * m_tp_v / max(1e-7, 2 * m_tp_v + m_fp_v + m_fn_v)
        d_f1_v = f1_m_v - f1_b_v

        per_video_decomp[str(vid)] = {
            "n_frames": len(vid_frames),
            "n_active": act_v,
            "n_pos": pos_v,
            "n_admitted": adm_v,
            "n_admitted_op_pos": adm_pos_v,
            "delta_tp": m_tp_v - b_tp_v,
            "delta_fp": m_fp_v - b_fp_v,
            "delta_fn": m_fn_v - b_fn_v,
            "f1_baseline": float(f1_b_v),
            "f1_model": float(f1_m_v),
            "delta_f1": float(d_f1_v)
        }

        b_str = f"{b_tp_v:2d}/{b_fp_v:2d}/{b_fn_v:2d} ({f1_b_v:.4f})"
        m_str = f"{m_tp_v:2d}/{m_fp_v:2d}/{m_fn_v:2d} ({f1_m_v:.4f})"
        adm_pos_str = f"{adm_v:4d} ({adm_pos_v:2d})"
        print(f"video_{vid:<6d} | {len(vid_frames):<8d} | {act_v:4d} ({pos_v:2d})       | {adm_pos_str:<20} | {b_str:<24} | {m_str:<24} | {d_f1_v:+8.4f}")

    # --------------------------------------------------------------------------
    # 6. P-1794 / P-1798: CONDITIONAL VERIFIER FREEZE & SCORE-SCALE AUDIT
    # --------------------------------------------------------------------------
    verifier_hash = None
    score_scale_audit = {}

    if visual_gate_passed:
        print("\n" + "=" * 115)
        print("P-1794: FITTING & FREEZING DEPLOYMENT CANDIDATE-4A VISUAL VERIFIER ARTIFACT")
        print("=" * 115)

        selected_features = model_configs[winning_visual_name]["features"]
        model_type = model_configs[winning_visual_name]["type"]

        mean_full = np.mean(selected_features, axis=0)
        std_full = np.std(selected_features, axis=0)
        std_full[std_full < 1e-6] = 1.0

        X_deploy_std = (selected_features - mean_full) / std_full

        if model_type == "logistic":
            deploy_clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=400, random_state=42)
            deploy_clf.fit(X_deploy_std, y_act)
            n_params = 128 * 1 + 1
        else:
            pos_idx_all = np.where(y_act == 1)[0]
            neg_idx_all = np.where(y_act == 0)[0]
            rng_deploy = np.random.default_rng(42 * 1000 + 777)
            oversampled_pos_all = rng_deploy.choice(pos_idx_all, size=len(neg_idx_all), replace=True)
            bal_idx_all = np.concatenate([neg_idx_all, oversampled_pos_all])
            rng_deploy.shuffle(bal_idx_all)

            deploy_clf = MLPClassifier(hidden_layer_sizes=(32,), activation="relu", alpha=0.05, max_iter=350, random_state=42)
            deploy_clf.fit(X_deploy_std[bal_idx_all], y_act[bal_idx_all])
            n_params = 128 * 32 + 32 + 32 * 1 + 1

        # P-1798: Full-Fit Score-Scale Provenance Audit (frozen tau* unchanged)
        deploy_scores_design = deploy_clf.predict_proba(X_deploy_std)[:, 1]
        oof_admitted_at_tau = int(np.sum(winning_oof_scores >= win_tau))
        fullfit_admitted_at_tau = int(np.sum(deploy_scores_design >= win_tau))

        percentiles_grid = [0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
        oof_quantiles = { f"p{int(q*100)}": float(np.percentile(winning_oof_scores, int(q*100))) for q in percentiles_grid }
        fullfit_quantiles = { f"p{int(q*100)}": float(np.percentile(deploy_scores_design, int(q*100))) for q in percentiles_grid }

        score_scale_audit = {
            "frozen_tau_star": float(win_tau),
            "oof_admissions_at_tau": oof_admitted_at_tau,
            "fullfit_admissions_at_tau": fullfit_admitted_at_tau,
            "oof_quantiles": oof_quantiles,
            "fullfit_quantiles": fullfit_quantiles
        }

        print(f"[*] P-1798 Deployment Score-Scale Audit (frozen tau* = {win_tau:.2f}):")
        print(f"    - OOF Admissions at tau*:      {oof_admitted_at_tau:4d} / 1,915 ({oof_admitted_at_tau/1915*100:5.2f}%)")
        print(f"    - Full-Fit Admissions at tau*: {fullfit_admitted_at_tau:4d} / 1,915 ({fullfit_admitted_at_tau/1915*100:5.2f}%)")
        print(f"    - OOF Score Median / Max:      {oof_quantiles['p50']:.4f} / {oof_quantiles['p100']:.4f}")
        print(f"    - Full-Fit Score Median / Max: {fullfit_quantiles['p50']:.4f} / {fullfit_quantiles['p100']:.4f}")

        final_verifier_payload = {
            "version": "v1",
            "model_architecture": winning_visual_name,
            "model_type": model_type,
            "n_params": n_params,
            "seed": 42,
            "feature_dim": 128,
            "mean": mean_full.tolist(),
            "std": std_full.tolist(),
            "tau_star": float(win_tau),
            "optimal_delta_f1": float(win_delta_f1),
            "temporal_prefilter_score": float(SCORE_THRESHOLD),
            "crop_size": CROP_SIZE,
            "expansion_factor": EXPANSION_FACTOR,
            "encoder": ENCODER_NAME,
            "score_scale_audit": score_scale_audit,
            "clf_model": deploy_clf,
            "source_driver_sha256": driver_sha256,
            "query_dataset_sha256": compute_sha256(QUERY_DATASET_PTH),
            "frame_tensors_sha256": compute_sha256(FRAME_TENSORS_PTH),
        }

        torch.save(final_verifier_payload, FINAL_VERIFIER_PTH)
        verifier_hash = compute_sha256(FINAL_VERIFIER_PTH)
        print(f"[+] Final Candidate-4A Deployment Verifier Frozen: {FINAL_VERIFIER_PTH.resolve()}")
        print(f"    Verifier Artifact SHA-256:                    {verifier_hash}")
    else:
        print("\n" + "=" * 115)
        print("[!] Candidate 4A Primary Visual-Only Gate FAILED (Delta F1 <= 0).")
        print("    No deployment verifier artifact will be created (P-1794).")
        if FINAL_VERIFIER_PTH.exists():
            FINAL_VERIFIER_PTH.unlink()
            print("    [!] Removed stale verifier artifact from disk.")

    results_payload = {
        "_meta": {
            "artifact_name": "pol4a_native_visual_admission_results_v3",
            "driver_sha256": driver_sha256,
            "design_videos": design_vids,
            "n_active_queries": n_active,
            "n_positives": n_pos,
            "base_prevalence": base_prev,
            "baseline_m0_f1": float(base_f1),
            "winning_visual_model": winning_visual_name,
            "visual_gate_passed": visual_gate_passed,
            "final_verifier_sha256": verifier_hash,
            "score_scale_audit": score_scale_audit,
            "query_dataset_sha256": compute_sha256(QUERY_DATASET_PTH),
            "frame_tensors_sha256": compute_sha256(FRAME_TENSORS_PTH)
        },
        "model_lovo_metrics": model_metrics,
        "operational_sweep_results": sweep_results,
        "winning_model_per_video_decomposition": per_video_decomp
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    res_hash = compute_sha256(RESULTS_JSON)
    print("\n" + "=" * 115)
    print(f"[+] Candidate 4A Operational Admission Results Frozen: {RESULTS_JSON.resolve()}")
    print(f"    Manifest SHA-256:                                  {res_hash}")
    print("=" * 115)


if __name__ == "__main__":
    main()