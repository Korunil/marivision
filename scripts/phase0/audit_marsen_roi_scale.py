"""
D:/MARIVISION/H0_workbench/eval_tools/audit_marsen_roi_scale.py
MARSEN Phase B2: Native-ROI Context Scale Optimization (v4, P-1864, P-1877, P-1887, P-1888, P-1894)

Strict Compliance Checklist:
  - D-3825..D-3854: Base B context sweep on 10 design videos ONLY; calibration/test UNTOUCHED.
  - P-1894: Runtime bug fix for SeaClips.zip archive entry resolution with robust endswith fallback.
  - P-1888: Hard assertions enforcing design cohort is strictly [4, 7, 16, 19, 25, 29, 30, 32, 35, 47],
            N = 1,915 active queries, and N+ = 92 operational positives.
  - P-1877: Crop parity certification: identical crop-floor (0.025) across all scales; asserts exact 1.5x reproduction.
  - P-1887: Effective-FOV diagnostics: reports crop-floor binding rates and pairwise identical integer crop box rates.
  - P-1864: Evaluates native FOV expansion scales {1.0x, 1.5x, 2.0x, 3.0x} at fixed 128-D ResNet-18 stem embedding.
  - Anchored on Phase B1b Winner: Evaluated using winning topology: 143 -> 64 -> 1 (9,281 params).
  - P-1841: Globally exact unique OOF score cut-points for every evaluated scale across Seeds 41, 42, 43.
  - Primary Selector: Highest Mean OOF AP across Seeds 41, 42, and 43.
"""

import os
import sys
import copy
import json
import hashlib
import zipfile
import io
import cv2
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict
from torchvision import transforms, models
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier

OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
TRAIN_JSON = Path("D:/MARIVISION/seaclips-train.json")
SPLIT_MANIFEST_JSON = OUT_DIR / "pol2_candidate2m_micro_split.json"
QUERY_DATASET_PTH = OUT_DIR / "pol3v_design_queries_dataset_v7.pth"
FRAME_TENSORS_PTH = OUT_DIR / "pol3v_design_frames_tensors_v7.pth"
CACHED_BRANCHES_PTH = OUT_DIR / "marsen_micro_authentic_branches_1915.pth"
CACHED_VISUAL_FEATURES_PTH = OUT_DIR / "pol4a_native_visual_features_1915.pth"
ZIP_PATH = Path("D:/MARIVISION/SeaClips.zip")

RESULTS_JSON = OUT_DIR / "marsen_roi_scale_results.json"
CACHED_MULTI_FOV_PTH = OUT_DIR / "marsen_multi_fov_features_1915.pth"

SCORE_THRESHOLD = 0.20
EVAL_IOU_THRESHOLD = 0.30
BASE_M0_F1 = 0.431718
EXPECTED_BASE_H64_AP = 0.1280
EXPECTED_DESIGN_VIDS = [4, 7, 16, 19, 25, 29, 30, 32, 35, 47]

SEEDS = [41, 42, 43]
REFERENCE_SEED = 41
WINNING_HIDDEN = (64,)
CROP_SIZE = 128
CROP_MIN_SIDE_NORM = 0.025  # P-1877: Certified Candidate-4A crop floor
ENCODER_NAME = "torchvision_resnet18_IMAGENET1K_V1_layer2"
FOV_SCALES = [1.0, 1.5, 2.0, 3.0]


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


def get_globally_exact_thresholds(oof_scores):
    scores = np.asarray(oof_scores, dtype=np.float64)
    assert np.isfinite(scores).all(), "[FATAL] Non-finite scores in threshold extraction!"
    unique_scores = np.unique(scores)
    thresholds = unique_scores.tolist()
    thresholds.append(float(np.nextafter(unique_scores.max(), np.inf)))
    return sorted(list(set(thresholds)))


def run_detector_sweep_fast(score_map, frame_records, active_key_set, positive_key_set, label_to_cat_id, threshold_list, base_frame_cache):
    best_tau = None
    best_f1 = -1.0
    best_rec = None
    sweep_records = {}

    temp_frames = [fr for fr in frame_records if fr["n_temp"] > 0]
    non_temp_tp = sum(base_frame_cache[fr["image_id"]][0] for fr in frame_records if fr["n_temp"] == 0)
    non_temp_fp = sum(base_frame_cache[fr["image_id"]][1] for fr in frame_records if fr["n_temp"] == 0)
    non_temp_fn = sum(base_frame_cache[fr["image_id"]][2] for fr in frame_records if fr["n_temp"] == 0)

    for tau in threshold_list:
        tot_tp = non_temp_tp
        tot_fp = non_temp_fp
        tot_fn = non_temp_fn
        admitted = 0
        retained_pos = 0

        for fr in temp_frames:
            n_t = fr["n_temp"]
            iid = fr["image_id"]

            frame_has_admitted = False
            for q_t in range(n_t):
                key = (iid, q_t)
                if key in active_key_set:
                    assert key in score_map, f"[FATAL] Missing OOF score for active query {key}"
                    if score_map[key] >= tau:
                        frame_has_admitted = True
                        break

            if not frame_has_admitted:
                tpd, fpd, fnd = base_frame_cache[iid]
                tot_tp += tpd
                tot_fp += fpd
                tot_fn += fnd
                continue

            joint_logits = torch.cat([fr["d_logits"], fr["temp_logits"]], dim=0).clone()
            joint_boxes = torch.cat([fr["d_boxes"], fr["temp_boxes"]], dim=0)

            for q_t in range(n_t):
                key = (iid, q_t)
                if key in active_key_set and score_map[key] >= tau:
                    admitted += 1
                    if key in positive_key_set:
                        retained_pos += 1
                else:
                    joint_logits[300 + q_t, :] = -1e9

            tpf, fpf, fnf = evaluate_operational_top100(
                joint_logits, joint_boxes, fr["gt_boxes_px"], fr["gt_cats"], label_to_cat_id, fr["img_w"], fr["img_h"]
            )
            tot_tp += tpf
            tot_fp += fpf
            tot_fn += fnf

        prec = tot_tp / max(1, tot_tp + tot_fp)
        rec = tot_tp / max(1, tot_tp + tot_fn)
        f1 = 2 * prec * rec / max(1e-7, prec + rec)

        sweep_records[tau] = {
            "tau": float(tau), "f1": float(f1), "tp": tot_tp, "fp": tot_fp, "fn": tot_fn,
            "prec": float(prec), "rec": float(rec), "admitted": admitted, "retained_pos": retained_pos
        }

        if f1 > best_f1 + 1e-12 or (abs(f1 - best_f1) <= 1e-12 and (best_tau is None or tau > best_tau)):
            best_f1 = f1
            best_tau = tau
            best_rec = sweep_records[tau]

    return best_tau, best_f1, best_rec, sweep_records


def train_lovo_mlp(X_feat, y_act, v_act, design_vids, hidden_layer_sizes, seed):
    n_samples = len(y_act)
    oof_scores = np.full(n_samples, np.nan)
    fold_n_iters = []

    for vid in design_vids:
        test_mask = (v_act == vid)
        train_mask = ~test_mask

        X_tr, y_tr = X_feat[train_mask], y_act[train_mask]
        X_te = X_feat[test_mask]

        if len(X_te) == 0 or len(np.unique(y_tr)) < 2:
            continue

        m_tr = X_tr.mean(axis=0)
        s_tr = X_tr.std(axis=0)
        s_tr[s_tr < 1e-6] = 1.0

        X_tr_std = (X_tr - m_tr) / s_tr
        X_te_std = (X_te - m_tr) / s_tr

        pos_idx = np.where(y_tr == 1)[0]
        neg_idx = np.where(y_tr == 0)[0]
        rng = np.random.default_rng(seed * 1000 + vid)
        oversampled_pos = rng.choice(pos_idx, size=len(neg_idx), replace=True)
        bal_idx = np.concatenate([neg_idx, oversampled_pos])
        rng.shuffle(bal_idx)

        clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation="relu", alpha=0.05, max_iter=350, random_state=seed)
        clf.fit(X_tr_std[bal_idx], y_tr[bal_idx])
        oof_scores[test_mask] = clf.predict_proba(X_te_std)[:, 1]
        fold_n_iters.append(int(clf.n_iter_))

    assert np.isfinite(oof_scores).all(), f"[FATAL] Non-finite scores generated for seed {seed}!"
    conv = {"mean_iters": float(np.mean(fold_n_iters)), "cap_hits": int(np.sum(np.array(fold_n_iters) >= 350))}
    return oof_scores, conv


def main():
    driver_sha256 = compute_sha256(Path(__file__))
    print("=" * 115)
    print("MARSEN PHASE B2: NATIVE-ROI CONTEXT SCALE OPTIMIZATION (v4, P-1864, P-1877, P-1887, P-1888, P-1894)")
    print(f"Driver Script SHA-256: {driver_sha256}")
    print("=" * 115)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[*] Compute Target: {device} ({device_name})")

    with open(SPLIT_MANIFEST_JSON, "r", encoding="utf-8") as f:
        design_vids = sorted(json.load(f)["_meta"]["train_videos"])

    # P-1888: Hard cohort assertion
    assert design_vids == EXPECTED_DESIGN_VIDS, f"[FATAL] Cohort mismatch: {design_vids} != {EXPECTED_DESIGN_VIDS}"
    print(f"[+] P-1888 Guard Certified: Design cohort strictly matches 10 sequences: {design_vids}")

    cached_queries = torch.load(QUERY_DATASET_PTH, map_location="cpu")["extracted_queries"]
    frame_bundle = torch.load(FRAME_TENSORS_PTH, map_location="cpu")
    frame_records = frame_bundle["frame_records"]

    active_indices = [idx for idx, q in enumerate(cached_queries) if q["is_active"]]
    active_queries = [cached_queries[idx] for idx in active_indices]
    n_active = len(active_queries)
    y_act = np.array([q["label_op_util"] for q in active_queries], dtype=np.int32)
    v_act = np.array([q["video_id"] for q in active_queries], dtype=np.int32)

    # P-1888: Strict query count and prevalence verification
    assert n_active == 1915, f"[FATAL] Expected 1,915 active queries, got {n_active}"
    assert int(y_act.sum()) == 92, f"[FATAL] Expected 92 positives, got {int(y_act.sum())}"
    assert sorted(np.unique(v_act).tolist()) == EXPECTED_DESIGN_VIDS, "[FATAL] Query video cohort mismatch!"
    print(f"[+] P-1888 Population Certified: Exactly 1,915 active queries, 92 positives (4.80% prevalence).")

    frame_dict = {fr["image_id"]: fr for fr in frame_records}
    for q in cached_queries:
        fr = frame_dict[q["image_id"]]
        cx, cy, w, h = fr["temp_boxes"][q["query_id"]].tolist()
        q["pred_box_norm"] = (cx, cy, w, h)
        q["img_w"] = fr["img_w"]
        q["img_h"] = fr["img_h"]

    branch_bundle = torch.load(CACHED_BRANCHES_PTH, map_location="cpu")
    feat_meta_15 = branch_bundle["branches"]["metadata"][:, :15]
    assert feat_meta_15.shape == (n_active, 15)

    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        coco_full = json.load(f)
    cat_ids = sorted(c["id"] for c in coco_full["categories"])
    label_to_cat_id = {i: cat_id for i, cat_id in enumerate(cat_ids)}

    active_key_set = set((q["image_id"], q["query_id"]) for q in active_queries)
    positive_key_set = set((q["image_id"], q["query_id"]) for q in active_queries if q["label_op_util"] == 1)

    base_frame_cache = {}
    m0_tp, m0_fp, m0_fn = 0, 0, 0
    for fr in frame_records:
        tpb, fpb, fnb = evaluate_operational_top100(
            fr["d_logits"], fr["d_boxes"], fr["gt_boxes_px"], fr["gt_cats"], label_to_cat_id, fr["img_w"], fr["img_h"]
        )
        base_frame_cache[fr["image_id"]] = (tpb, fpb, fnb)
        m0_tp += tpb
        m0_fp += fpb
        m0_fn += fnb
    calculated_base_f1 = 2 * m0_tp / max(1e-7, 2 * m0_tp + m0_fp + m0_fn)
    assert abs(calculated_base_f1 - BASE_M0_F1) < 1e-5

    # --------------------------------------------------------------------------
    # 1. P-1877, P-1887 & P-1894 CROP EXTRACTION & DIAGNOSTIC ENGINE
    # --------------------------------------------------------------------------
    fov_features = {}
    cache_valid = False

    if CACHED_MULTI_FOV_PTH.exists():
        print(f"\n[+] Verifying Multi-FOV Feature Cache: {CACHED_MULTI_FOV_PTH.resolve()}")
        c_bundle = torch.load(CACHED_MULTI_FOV_PTH, map_location="cpu")
        if (
            isinstance(c_bundle, dict)
            and c_bundle.get("driver_sha256") == driver_sha256
            and c_bundle.get("n_active") == n_active
            and c_bundle.get("crop_min_side_norm") == CROP_MIN_SIDE_NORM
            and all(f"fov_{scale}" in c_bundle for scale in FOV_SCALES)
            and "floor_binding_rates" in c_bundle
            and "pairwise_identity_rates" in c_bundle
        ):
            fov_features = {scale: c_bundle[f"fov_{scale}"] for scale in FOV_SCALES}
            floor_binding_rates = c_bundle["floor_binding_rates"]
            pairwise_identity_rates = c_bundle["pairwise_identity_rates"]
            cache_valid = True
            print("[+] P-1877 / P-1887 Certified Features and Diagnostics Ingested from Cache.")

    if not cache_valid:
        print("\n[*] Initializing Frozen ResNet-18 Stem for Multi-FOV Extraction...")
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        encoder = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2,
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
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

        def get_crop_pixel_box(img_size, box_norm, expansion_factor):
            cx, cy, w, h = box_norm
            unclipped_side = max(w, h) * expansion_factor
            floor_bound = (unclipped_side < CROP_MIN_SIDE_NORM)
            side = max(CROP_MIN_SIDE_NORM, unclipped_side)

            x1_norm = max(0.0, cx - side / 2.0)
            y1_norm = max(0.0, cy - side / 2.0)
            x2_norm = min(1.0, cx + side / 2.0)
            y2_norm = min(1.0, cy + side / 2.0)

            W, H = img_size
            px_box = (
                max(0, int(x1_norm * W)),
                max(0, int(y1_norm * H)),
                min(W, max(int(x1_norm * W) + 1, int(x2_norm * W))),
                min(H, max(int(y1_norm * H) + 1, int(y2_norm * H)))
            )
            return px_box, floor_bound

        unique_iids = sorted(list(set(q["image_id"] for q in active_queries)))
        img_cache = {}
        print(f"[*] Preloading {len(unique_iids)} Target Frames from SeaClips.zip...")

        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zip_index = {name.replace("\\", "/").strip("/"): name for name in zf.namelist()}
            for name in zf.namelist():
                norm = name.replace("\\", "/").strip("/")
                parts = norm.split("/")
                if len(parts) >= 2:
                    zip_index["/".join(parts[-2:])] = name

            for iid in unique_iids:
                rel_path = img_meta_by_id[iid]["file_name"].replace("\\", "/").strip("/")
                target_key = "/".join(rel_path.split("/")[-2:]) if len(rel_path.split("/")) >= 2 else rel_path

                # P-1894: Robust lookup with exact endswith fallback
                zip_entry = zip_index.get(rel_path) or zip_index.get(target_key)
                if zip_entry is None:
                    for name in zf.namelist():
                        if name.replace("\\", "/").endswith(rel_path):
                            zip_entry = name
                            break
                assert zip_entry is not None, f"[FATAL] Cannot find {rel_path} in SeaClips.zip"

                with zf.open(zip_entry) as f:
                    img_cache[iid] = Image.open(io.BytesIO(f.read())).convert("RGB")

        # Extract crops, compute feature embeddings, and evaluate P-1887 diagnostics
        pixel_boxes_by_scale = {}
        floor_binding_rates = {}

        for scale in FOV_SCALES:
            print(f"[*] Extracting native crops for FOV Scale {scale:.1f}x across {n_active:,} queries...")
            crop_tensors = []
            scale_px_boxes = []
            floor_hits = 0

            for q in active_queries:
                img_nat = img_cache[q["image_id"]]
                px_box, is_floor = get_crop_pixel_box(img_nat.size, q["pred_box_norm"], expansion_factor=scale)
                crop_pil = img_nat.crop(px_box)
                crop_tensors.append(img_transform(crop_pil))
                scale_px_boxes.append(px_box)
                if is_floor:
                    floor_hits += 1

            pixel_boxes_by_scale[scale] = scale_px_boxes
            floor_binding_rates[scale] = (floor_hits / n_active) * 100.0

            feat_list = []
            batch_size = 64
            with torch.no_grad():
                for i in range(0, len(crop_tensors), batch_size):
                    batch = torch.stack(crop_tensors[i:i + batch_size]).to(device)
                    feats = encoder(batch).cpu().numpy()
                    feat_list.append(feats)

            fov_features[scale] = np.concatenate(feat_list, axis=0).astype(np.float32)
            assert fov_features[scale].shape == (n_active, 128)

        # Compute pairwise identical crop box rates
        pairwise_identity_rates = {}
        adjacent_pairs = [(1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (1.0, 3.0)]
        for s1, s2 in adjacent_pairs:
            matches = sum(1 for i in range(n_active) if pixel_boxes_by_scale[s1][i] == pixel_boxes_by_scale[s2][i])
            pairwise_identity_rates[f"{s1}x_vs_{s2}x"] = (matches / n_active) * 100.0

        # P-1877: Verification check against certified 1.5x cache
        visual_cache_15 = torch.load(CACHED_VISUAL_FEATURES_PTH, map_location="cpu")["feat_visual"].astype(np.float32)
        max_15_diff = float(np.abs(fov_features[1.5] - visual_cache_15).max())
        print(f"[*] P-1877 Crop Parity Check (Re-extracted 1.5x vs. Certified Cache): max |diff| = {max_15_diff:.2e}")
        assert max_15_diff < 1e-4, f"[FATAL] Re-extracted 1.5x features deviated from certified cache: {max_15_diff} >= 1e-4!"
        print("[+] P-1877 Certified: Re-extracted 1.5x exactly matches frozen cache. Crop parity guaranteed.")

        save_dict = {
            "driver_sha256": driver_sha256,
            "n_active": n_active,
            "crop_size": CROP_SIZE,
            "crop_min_side_norm": CROP_MIN_SIDE_NORM,
            "encoder": ENCODER_NAME,
            "floor_binding_rates": floor_binding_rates,
            "pairwise_identity_rates": pairwise_identity_rates
        }
        for scale in FOV_SCALES:
            save_dict[f"fov_{scale}"] = fov_features[scale]
        torch.save(save_dict, CACHED_MULTI_FOV_PTH)
        print(f"[+] Multi-FOV Features and Diagnostics Saved: {CACHED_MULTI_FOV_PTH.resolve()}")

    # --------------------------------------------------------------------------
    # P-1887: EFFECTIVE-FOV & CROP-FLOOR DIAGNOSTIC REPORT
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("P-1887: EFFECTIVE-FOV & CROP-FLOOR DIAGNOSTIC REPORT")
    print("=" * 115)
    print("1. Crop-Floor Binding Rate (Proposals where max(w, h) * scale < 0.025):")
    for scale in FOV_SCALES:
        rate = floor_binding_rates[scale] if isinstance(scale, float) else floor_binding_rates[float(scale)]
        print(f"   - Scale {scale:.1f}x : {rate:5.1f}% of active proposals pinned by 0.025 floor")

    print("\n2. Pairwise Identical Integer Pixel Crop Box Match Rates:")
    for pair_key, match_rate in pairwise_identity_rates.items():
        print(f"   - Pair {pair_key:<12} : {match_rate:5.1f}% of queries produce 100% IDENTICAL pixel crops")

    # --------------------------------------------------------------------------
    # 2. EVALUATE NATIVE FOV EXPANSION SCALES (TOPOLOGY: 143 -> 64 -> 1, 9,281 PARAMS)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("P-1864: EVALUATING NATIVE FOV SCALES ON BASE B (WINNING TOPOLOGY: 143 -> 64 -> 1, 9,281 PARAMS)")
    print("=" * 115)

    fov_results = defaultdict(dict)
    scale_names = {
        1.0: "FOV 1.0x",
        1.5: "FOV 1.5x (Anchor)",
        2.0: "FOV 2.0x",
        3.0: "FOV 3.0x"
    }

    for scale in FOV_SCALES:
        name = scale_names[scale]
        feat_hr_scale = fov_features[scale]
        X_scale = np.concatenate([feat_hr_scale, feat_meta_15], axis=-1)
        assert X_scale.shape == (n_active, 143)

        print(f"\n[*] Evaluating: {name:<24} (Dim=143, Hidden=64, Params=9,281)")
        print(f"    {'Seed':<6} | {'Pooled AP':<12} | {'ROC-AUC':<10} | {'Exact tau*':<14} | {'Best F1*':<12} | {'Delta F1 vs M0'}")
        print("    " + "-" * 75)

        for s in SEEDS:
            oof_s, conv_s = train_lovo_mlp(X_scale, y_act, v_act, design_vids, hidden_layer_sizes=WINNING_HIDDEN, seed=s)
            ap_s = float(average_precision_score(y_act, oof_s))
            auc_s = float(roc_auc_score(y_act, oof_s))

            exact_taus = get_globally_exact_thresholds(oof_s)
            score_map_s = { (q["image_id"], q["query_id"]): float(oof_s[i]) for i, q in enumerate(active_queries) }
            best_tau_s, best_f1_s, best_rec_s, _ = run_detector_sweep_fast(
                score_map_s, frame_records, active_key_set, positive_key_set, label_to_cat_id, exact_taus, base_frame_cache
            )

            fov_results[scale][s] = {
                "ap": ap_s, "auc": auc_s, "tau_star": best_tau_s, "f1_star": best_f1_s,
                "delta_f1_m0": best_f1_s - BASE_M0_F1, "best_rec": best_rec_s, "conv": conv_s
            }
            print(f"    Seed {s:<2d} | {ap_s:10.4f}   | {auc_s:8.4f}   | {best_tau_s:12.6f}   | {best_f1_s:10.6f}   | {best_f1_s - BASE_M0_F1:+12.4f}")

    # Anchor parity check on 1.5x anchor
    mean_15_ap = np.mean([fov_results[1.5][s]["ap"] for s in SEEDS])
    print(f"\n[*] Anchor Check: Evaluated FOV 1.5x Mean AP: {mean_15_ap:.4f} (Phase B1 Winner: {EXPECTED_BASE_H64_AP:.4f})")
    assert abs(mean_15_ap - EXPECTED_BASE_H64_AP) < 1e-3, "[FATAL] FOV 1.5x anchor failed parity against Phase B1 winner!"
    print("[+] Anchor Verified: FOV 1.5x replicates Phase B1 winning baseline with exact parity.")

    # --------------------------------------------------------------------------
    # FOV SCALE SYNTHESIS & SELECTION
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("NATIVE FOV SCALE SYNTHESIS (RANKED BY MEAN POOLED AP ACROSS SEEDS 41, 42, 43)")
    print("=" * 115)
    print(f"{'Rank':<5} | {'FOV Representation':<24} | {'Mean AP':<10} | {'AP Std':<10} | {'Seed 41 AP':<12} | {'Seed 42 AP':<12} | {'Seed 43 AP':<12} | {'Mean F1*':<10} | {'Mean dF1':<10}")
    print("-" * 118)

    summary_list = []
    base_15_aps = [fov_results[1.5][s]["ap"] for s in SEEDS]

    for scale in FOV_SCALES:
        aps = [fov_results[scale][s]["ap"] for s in SEEDS]
        f1s = [fov_results[scale][s]["f1_star"] for s in SEEDS]
        df1s = [fov_results[scale][s]["delta_f1_m0"] for s in SEEDS]
        aucs = [fov_results[scale][s]["auc"] for s in SEEDS]

        d_aps = [aps[i] - base_15_aps[i] for i in range(len(SEEDS))]

        summary_list.append({
            "scale": scale,
            "name": scale_names[scale],
            "mean_ap": float(np.mean(aps)),
            "std_ap": float(np.std(aps)),
            "aps": aps,
            "mean_f1": float(np.mean(f1s)),
            "std_f1": float(np.std(f1s)),
            "f1s": f1s,
            "mean_df1": float(np.mean(df1s)),
            "mean_dap_vs_baseline": float(np.mean(d_aps)),
            "seed_daps_vs_15": d_aps,
            "mean_auc": float(np.mean(aucs))
        })

    summary_list.sort(key=lambda x: x["mean_ap"], reverse=True)

    for rank, item in enumerate(summary_list, 1):
        star = " <-- BEST TESTED FOV SCALE" if rank == 1 else ""
        print(f"#{rank:<3d} | {item['name']:<24} | {item['mean_ap']:8.4f}   | {item['std_ap']:8.4f}   | {item['aps'][0]:10.4f}   | {item['aps'][1]:10.4f}   | {item['aps'][2]:10.4f}   | {item['mean_f1']:8.6f} | {item['mean_df1']:+8.4f}   {star}")

    winning_fov = summary_list[0]
    print("\n" + "=" * 115)
    print(f"[+] Best Tested Native Context Scale: {winning_fov['name']}")
    print(f"    - Expansion Multiplier: {winning_fov['scale']}x")
    print(f"    - Mean OOF AP:          {winning_fov['mean_ap']:.4f} +/- {winning_fov['std_ap']:.4f}")
    print(f"    - Mean Best F1*:        {winning_fov['mean_f1']:.6f} ({winning_fov['mean_df1']:+.4f} over M0)")
    print(f"    - Delta vs 1.5x Anchor: {winning_fov['mean_dap_vs_baseline']:+.4f} AP")
    print(f"    - Paired Seed Deltas:   Seed 41: {winning_fov['seed_daps_vs_15'][0]:+.4f} | Seed 42: {winning_fov['seed_daps_vs_15'][1]:+.4f} | Seed 43: {winning_fov['seed_daps_vs_15'][2]:+.4f}")
    print("=" * 115)

    results_payload = {
        "_meta": {
            "artifact_name": "marsen_roi_scale_results_v4",
            "driver_sha256": driver_sha256,
            "baseline_m0_f1": float(BASE_M0_F1),
            "reference_seed": REFERENCE_SEED,
            "winning_scale": float(winning_fov["scale"]),
            "winning_scale_name": winning_fov["name"],
            "winning_mean_ap": float(winning_fov["mean_ap"]),
            "winning_mean_f1": float(winning_fov["mean_f1"]),
            "paired_deltas_vs_15": winning_fov["seed_daps_vs_15"],
            "floor_binding_rates": floor_binding_rates,
            "pairwise_identity_rates": pairwise_identity_rates
        },
        "ranked_fov_summary": summary_list,
        "seed_runs": {
            str(scale): {str(s): fov_results[scale][s] for s in SEEDS} for scale in FOV_SCALES
        }
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    res_hash = compute_sha256(RESULTS_JSON)
    print(f"[+] ROI Scale Results Frozen: {RESULTS_JSON.resolve()}")
    print(f"    Manifest SHA-256:         {res_hash}")
    print("=" * 115)


if __name__ == "__main__":
    main()