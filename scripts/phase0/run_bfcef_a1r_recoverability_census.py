"""
D:/MARIVISION/H0_workbench/eval_tools/run_bfcef_a1r_recoverability_census.py
MARSEN B-FCEF: Stage A1R - Native Recoverability & Passive Cue Census
(D-4396..D-4410, P-2260..P-2266)

Executes passive structural census across all 20,075 frames (0% detector perturbation):
  1. Produces exactly one record per each of the 1,576 utility-positive temporal proposals (|P_U| = 1576).
  2. Dual native matching:
     - j_any = argmax_j IoU_j, I_any = max_j IoU_j (best geometry regardless of class)
     - j_same = argmax_{j: c_hat_j == c_t} IoU_j, I_same = max_{j: c_hat_j == c_t} IoU_j
  3. Classifies each utility event into 5 mutually exclusive, exhaustive categories:
     - Cat 1: I_same >= 0.30 (Native Spatial + Class Support)
       * Cat 1A: Event-level check confirms >= 1 same-class support query in native Top-100
       * Cat 1B: All same-class support queries fall outside native Top-100 (True Rank Suppression)
     - Cat 2: I_any >= 0.30 and I_same < 0.30 (Classification-Limited Proxy)
     - Cat 3: I_same >= 0.10 and I_any < 0.30 (Localization-Limited Proxy)
     - Cat 4: I_any >= 0.10 and I_same < 0.10 (Combined Class + Loc Proxy)
     - Cat 5: I_any < 0.10 (No Close Native Geometric Support)
     Hard-asserts: sum(Cat_k) == 1576, and Cat 1A + Cat 1B == Cat 1.
  4. Profiles general native reference population stratified by rank tiers (1-25, 26-100, 101-300).
  5. Computes normalized multi-label Bernoulli entropy H_B in [0, 1].
  6. Freezes certified summary and itemized records to JSON.
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
from collections import defaultdict
import torch
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

# Path Constants
ROOT_DIR = Path("D:/MARIVISION")
OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
SHARD_DIR = OUT_DIR / "full_split_shards"

CENSUS_JSON = OUT_DIR / "marsen_b2_production_census_56shards.json"
TRAIN_JSON = ROOT_DIR / "seaclips-train.json"
A1R_SUMMARY_OUT_JSON = OUT_DIR / "bfcef_a1r_recoverability_census_summary.json"
A1R_RECORDS_OUT_JSON = OUT_DIR / "bfcef_a1r_records_1576.json"

NUM_VIDEOS = 47
EXPECTED_TOTAL_QUERIES = 211712
EXPECTED_TOTAL_WINDOWS = 20075
TOTAL_ORACLE_RESCUES = 1576
NUM_CLASSES = 7

FROZEN_CENSUS_SHA256 = "a20a734d668a9078bdfe274cb09fa5a47ad9c6a3a9d7cef2d1f99b926623fbc5"


def compute_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def box_iou_matrix_xywh(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Vectorized pairwise IoU between boxes1 (N, 4) and boxes2 (M, 4) in normalized cxcywh."""
    b1_x1 = boxes1[:, 0] - boxes1[:, 2] / 2.0
    b1_y1 = boxes1[:, 1] - boxes1[:, 3] / 2.0
    b1_x2 = boxes1[:, 0] + boxes1[:, 2] / 2.0
    b1_y2 = boxes1[:, 1] + boxes1[:, 3] / 2.0

    b2_x1 = boxes2[:, 0] - boxes2[:, 2] / 2.0
    b2_y1 = boxes2[:, 1] - boxes2[:, 3] / 2.0
    b2_x2 = boxes2[:, 0] + boxes2[:, 2] / 2.0
    b2_y2 = boxes2[:, 1] + boxes2[:, 3] / 2.0

    ix1 = torch.max(b1_x1.unsqueeze(1), b2_x1.unsqueeze(0))
    iy1 = torch.max(b1_y1.unsqueeze(1), b2_y1.unsqueeze(0))
    ix2 = torch.min(b1_x2.unsqueeze(1), b2_x2.unsqueeze(0))
    iy2 = torch.min(b1_y2.unsqueeze(1), b2_y2.unsqueeze(0))

    iw = torch.clamp(ix2 - ix1, min=0.0)
    ih = torch.clamp(iy2 - iy1, min=0.0)
    inter = iw * ih

    area1 = (boxes1[:, 2] * boxes1[:, 3]).unsqueeze(1)
    area2 = (boxes2[:, 2] * boxes2[:, 3]).unsqueeze(0)
    union = area1 + area2 - inter

    return torch.where(union > 0, inter / union, torch.zeros_like(inter))


def compute_normalized_bernoulli_entropy(probs: torch.Tensor, eps: float = 1e-7) -> float:
    """Computes normalized multi-label Bernoulli entropy H_B in [0.0, 1.0]."""
    p = torch.clamp(probs, min=eps, max=1.0 - eps)
    h_per_class = -(p * torch.log2(p) + (1.0 - p) * torch.log2(1.0 - p))
    return float(torch.mean(h_per_class).item())


def main():
    t_start = time.time()
    driver_sha = compute_sha256(Path(__file__))

    print("=" * 115)
    print("MARSEN B-FCEF: STAGE A1R - NATIVE RECOVERABILITY & PASSIVE CUE CENSUS")
    print(f"Driver Script SHA-256: {driver_sha}")
    print("Population Invariants: |P_U| = 1,576 Utility Events | 20,075 Target Frames")
    print("=" * 115, flush=True)

    # 1. Provenance Verification
    print("[*] Running Cryptographic Provenance Preflight Gate...", flush=True)
    assert CENSUS_JSON.exists(), f"[FATAL] Missing census JSON at {CENSUS_JSON}"
    assert compute_sha256(CENSUS_JSON) == FROZEN_CENSUS_SHA256, "[FATAL] Census SHA mismatch"

    with open(CENSUS_JSON, "r", encoding="utf-8") as f:
        census = json.load(f)
    frozen_shard_ledger = census["shard_sha256_ledger"]

    for vid in range(1, NUM_VIDEOS + 1):
        p = SHARD_DIR / f"marsen_b2_shard_train_vid{vid}.pth"
        assert p.exists(), f"[FATAL] Missing shard {p.name}"
        assert compute_sha256(p) == frozen_shard_ledger[p.name], f"[FATAL] Shard drift on {p.name}"
    print("    [+] Provenance verified: Census and all 47 Train shards bitwise certified.\n", flush=True)

    # 2. Ingest Categories
    print("[*] Loading Ground Truth Category Map from seaclips-train.json...", flush=True)
    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        train_coco = json.load(f)
    sorted_cats = sorted(train_coco["categories"], key=lambda x: x["id"])
    cat_names = [c["name"] for c in sorted_cats]
    print(f"    [+] Registered {len(cat_names)} categories: {cat_names}\n", flush=True)

    # 3. Census Accumulators
    records_1576 = []
    category_counts = defaultdict(int)
    cat1_sub_counts = {"Cat 1A: Native Top-100 Member": 0, "Cat 1B: Rank/Score-Suppressed": 0}
    scale_cross_tab = defaultdict(lambda: defaultdict(int))
    class_cross_tab = defaultdict(lambda: defaultdict(int))

    all_i_any_values = []
    all_i_same_values = []

    # Stratified Native Population Storage (Tier 1, Tier 2, Tier 3)
    stratified_native_cues = {
        "Tier1_Rank1_25": {"scores": [], "margins": [], "entropies": [], "areas": []},
        "Tier2_Rank26_100": {"scores": [], "margins": [], "entropies": [], "areas": []},
        "Tier3_Rank101_300": {"scores": [], "margins": [], "entropies": [], "areas": []}
    }

    # Frame-Level Context Taps
    frame_context_taps = {
        "temporal_proposal_density": [],
        "detector_activation_energy": []
    }

    total_scanned_frames = 0
    total_active_queries = 0

    print("[*] Processing 47 Train Shards for Vectorized Dual Matching & Stratified Sampling...", flush=True)

    for vid in range(1, NUM_VIDEOS + 1):
        shard_path = SHARD_DIR / f"marsen_b2_shard_train_vid{vid}.pth"
        shard = torch.load(shard_path, map_location="cpu")
        total_active_queries += len(shard["active_queries"])

        active_q_by_img = defaultdict(list)
        for q in shard["active_queries"]:
            active_q_by_img[q["image_id"]].append(q)

        for fr in shard.get("frame_records", []):
            total_scanned_frames += 1
            f_idx = total_scanned_frames
            iid = fr["image_id"]
            img_w = float(fr["img_w"])
            img_h = float(fr["img_h"])

            d_logits = fr["d_logits"]  # (300, 7)
            d_boxes = fr["d_boxes"]    # (300, 4)
            d_probs = d_logits.sigmoid()
            d_scores, d_pred_cats = torch.max(d_probs, dim=-1)

            # Flat Top-100 Indices
            flat_probs = d_probs.flatten()
            top100_flat_indices = torch.topk(flat_probs, min(100, flat_probs.numel())).indices

            # Frame-Level Context Taps
            n_temp = int(fr.get("n_temp", 0))
            frame_context_taps["temporal_proposal_density"].append(float(n_temp) / 100.0)
            frame_context_taps["detector_activation_energy"].append(float(d_probs.sum().item()))

            # Stratified Native Sampling (Deterministic per Tier)
            native_ranks = torch.argsort(-d_scores)
            idx_t1 = int(native_ranks[(f_idx * 7) % 25].item())
            idx_t2 = int(native_ranks[25 + (f_idx * 11) % 75].item())
            idx_t3 = int(native_ranks[100 + (f_idx * 13) % 200].item())

            for tier_name, q_idx in [("Tier1_Rank1_25", idx_t1), ("Tier2_Rank26_100", idx_t2), ("Tier3_Rank101_300", idx_t3)]:
                box_px_area = float((d_boxes[q_idx, 2] * img_w) * (d_boxes[q_idx, 3] * img_h))
                top2_m = float((torch.topk(d_logits[q_idx], 2)[0][0] - torch.topk(d_logits[q_idx], 2)[0][1]).item())
                stratified_native_cues[tier_name]["scores"].append(float(d_scores[q_idx].item()))
                stratified_native_cues[tier_name]["margins"].append(top2_m)
                stratified_native_cues[tier_name]["entropies"].append(compute_normalized_bernoulli_entropy(d_probs[q_idx]))
                stratified_native_cues[tier_name]["areas"].append(box_px_area)

            # Process Oracle Utility-Positive Events for this Frame
            frame_act_q = active_q_by_img.get(iid, [])
            rescue_events = [q for q in frame_act_q if int(q["label_op_util"]) == 1]
            if len(rescue_events) == 0:
                continue

            temp_boxes = fr["temp_boxes"]
            temp_logits = fr["temp_logits"]

            # Vectorized Pairwise IoU Matrix: (300, K_rescues)
            rescue_locs = [int(q["query_id"]) for q in rescue_events]
            t_boxes_rescues = temp_boxes[rescue_locs]
            t_logits_rescues = temp_logits[rescue_locs]

            iou_matrix = box_iou_matrix_xywh(d_boxes, t_boxes_rescues)  # (300, K)

            for k_idx, q_ev in enumerate(rescue_events):
                q_id = rescue_locs[k_idx]
                t_box = t_boxes_rescues[k_idx]
                t_logit = t_logits_rescues[k_idx]

                # Semantic class and score integrity assertions (CHL-2068, D-4406)
                if "pred_class" in q_ev:
                    c_t = int(q_ev["pred_class"])
                    assert int(torch.argmax(t_logit).item()) == c_t, "[FATAL] argmax(t_logit) != pred_class"
                else:
                    c_t = int(torch.argmax(t_logit).item())

                if "max_score" in q_ev:
                    s_temp = float(q_ev["max_score"])
                    assert abs(float(t_logit.sigmoid().max().item()) - s_temp) < 1e-4, "[FATAL] max(sigmoid) != max_score"
                else:
                    s_temp = float(torch.max(t_logit.sigmoid()).item())

                # Area & Scale Bin
                bw_px = float(t_box[2] * img_w)
                bh_px = float(t_box[3] * img_h)
                area_px = bw_px * bh_px
                if area_px < 32.0 * 32.0:
                    scale_bin = "Small (<32²)"
                elif area_px < 96.0 * 96.0:
                    scale_bin = "Medium (32²-96²)"
                else:
                    scale_bin = "Large (>96²)"

                # 1. Best Geometry Regardless of Class (j_any, I_any)
                ious_k = iou_matrix[:, k_idx]
                i_any, j_any = float(torch.max(ious_k).item()), int(torch.argmax(ious_k).item())

                # 2. Best Query whose Predicted Class Equals c_t (j_same, I_same)
                same_class_mask = (d_pred_cats == c_t)
                if same_class_mask.any():
                    same_class_ious = torch.where(same_class_mask, ious_k, torch.zeros_like(ious_k))
                    i_same = float(torch.max(same_class_ious).item())
                    j_same = int(torch.argmax(same_class_ious).item())
                else:
                    i_same = 0.0
                    j_same = None

                all_i_any_values.append(i_any)
                all_i_same_values.append(i_same)

                # 3. Categorization with Event-Level Cat 1A/1B Top-100 Test (D-4407..D-4410)
                sub_category = None
                in_top100_flag = False
                n_support_030 = 0
                n_support_top100 = 0

                if i_same >= 0.30:
                    category = "Cat 1: Native Spatial + Class Support (I_same >= 0.30)"
                    
                    # Event-Level Test across ALL same-class spatial support queries (D-4409, D-4410)
                    support_mask = same_class_mask & (ious_k >= 0.30)
                    support_query_ids = torch.nonzero(support_mask, as_tuple=False).flatten()
                    support_flat_ids = support_query_ids * NUM_CLASSES + c_t
                    support_top100_mask = (support_flat_ids[:, None] == top100_flat_indices[None, :]).any(dim=1)

                    in_top100_flag = bool(support_top100_mask.any().item())
                    n_support_030 = int(support_query_ids.numel())
                    n_support_top100 = int(support_top100_mask.sum().item())

                    if in_top100_flag:
                        sub_category = "Cat 1A: Native Top-100 Member"
                        cat1_sub_counts["Cat 1A: Native Top-100 Member"] += 1
                    else:
                        sub_category = "Cat 1B: Rank/Score-Suppressed"
                        cat1_sub_counts["Cat 1B: Rank/Score-Suppressed"] += 1

                    # Representative metadata bound strictly to j_same (P-2266, CLM-3635)
                    j_rec = j_same
                    p_target_class = float(d_probs[j_same, c_t].item())
                    flat_target_rank = int((flat_probs > p_target_class).sum().item()) + 1
                    top2_native = torch.topk(d_logits[j_same], 2)[0]
                    native_margin = float((top2_native[0] - top2_native[1]).item())
                    native_h_b = compute_normalized_bernoulli_entropy(d_probs[j_same])
                    s_native = float(d_scores[j_same].item())
                    query_rank = int((d_scores > s_native).sum().item()) + 1
                    delta_score = s_temp - s_native

                elif i_any >= 0.30:
                    category = "Cat 2: Classification-Limited Proxy (I_any >= 0.30, I_same < 0.30)"
                    j_rec = j_any
                    p_target_class = float(d_probs[j_any, c_t].item())
                    flat_target_rank = int((flat_probs > p_target_class).sum().item()) + 1
                    target_flat_idx = j_any * NUM_CLASSES + c_t
                    in_top100_flag = bool((top100_flat_indices == target_flat_idx).any().item())
                    top2_native = torch.topk(d_logits[j_any], 2)[0]
                    native_margin = float((top2_native[0] - top2_native[1]).item())
                    native_h_b = compute_normalized_bernoulli_entropy(d_probs[j_any])
                    s_native = float(d_scores[j_any].item())
                    query_rank = int((d_scores > s_native).sum().item()) + 1
                    delta_score = s_temp - s_native

                elif i_same >= 0.10:
                    category = "Cat 3: Localization-Limited Proxy (0.10 <= I_same, I_any < 0.30)"
                    j_rec = j_same
                    p_target_class = float(d_probs[j_same, c_t].item())
                    flat_target_rank = int((flat_probs > p_target_class).sum().item()) + 1
                    target_flat_idx = j_same * NUM_CLASSES + c_t
                    in_top100_flag = bool((top100_flat_indices == target_flat_idx).any().item())
                    top2_native = torch.topk(d_logits[j_same], 2)[0]
                    native_margin = float((top2_native[0] - top2_native[1]).item())
                    native_h_b = compute_normalized_bernoulli_entropy(d_probs[j_same])
                    s_native = float(d_scores[j_same].item())
                    query_rank = int((d_scores > s_native).sum().item()) + 1
                    delta_score = s_temp - s_native

                elif i_any >= 0.10:
                    category = "Cat 4: Combined Class + Loc Proxy (0.10 <= I_any < 0.30, I_same < 0.10)"
                    j_rec = j_any
                    p_target_class = float(d_probs[j_any, c_t].item())
                    flat_target_rank = int((flat_probs > p_target_class).sum().item()) + 1
                    target_flat_idx = j_any * NUM_CLASSES + c_t
                    in_top100_flag = bool((top100_flat_indices == target_flat_idx).any().item())
                    top2_native = torch.topk(d_logits[j_any], 2)[0]
                    native_margin = float((top2_native[0] - top2_native[1]).item())
                    native_h_b = compute_normalized_bernoulli_entropy(d_probs[j_any])
                    s_native = float(d_scores[j_any].item())
                    query_rank = int((d_scores > s_native).sum().item()) + 1
                    delta_score = s_temp - s_native

                else:
                    category = "Cat 5: No Close Native Geometric Support (I_any < 0.10)"
                    j_rec = j_any
                    p_target_class = float(d_probs[j_any, c_t].item())
                    flat_target_rank = int((flat_probs > p_target_class).sum().item()) + 1
                    target_flat_idx = j_any * NUM_CLASSES + c_t
                    in_top100_flag = bool((top100_flat_indices == target_flat_idx).any().item())
                    top2_native = torch.topk(d_logits[j_any], 2)[0]
                    native_margin = float((top2_native[0] - top2_native[1]).item())
                    native_h_b = compute_normalized_bernoulli_entropy(d_probs[j_any])
                    s_native = float(d_scores[j_any].item())
                    query_rank = int((d_scores > s_native).sum().item()) + 1
                    delta_score = s_temp - s_native

                category_counts[category] += 1
                scale_cross_tab[category][scale_bin] += 1
                class_cross_tab[category][cat_names[c_t]] += 1

                records_1576.append({
                    "event_id": len(records_1576) + 1,
                    "video_id": int(q_ev["video_id"]),
                    "image_id": int(iid),
                    "query_id": q_id,
                    "category_name": cat_names[c_t],
                    "scale_bin": scale_bin,
                    "area_px": round(area_px, 2),
                    "temporal_score": round(s_temp, 4),
                    "best_any_iou": round(i_any, 4),
                    "best_any_query_id": j_any,
                    "best_any_pred_class": cat_names[int(d_pred_cats[j_any].item())],
                    "best_same_class_iou": round(i_same, 4),
                    "best_same_query_id": j_same if j_same is not None else -1,
                    "target_class_native_score": round(p_target_class, 4),
                    "target_class_flat_rank": flat_target_rank,
                    "is_target_in_native_top100": in_top100_flag,
                    "n_sameclass_support_iou030": n_support_030,
                    "n_sameclass_support_top100": n_support_top100,
                    "native_query_rank": query_rank,
                    "native_top2_margin": round(native_margin, 4),
                    "native_bernoulli_entropy": round(native_h_b, 4),
                    "score_delta": round(delta_score, 4),
                    "recoverability_category": category,
                    "cat1_sub_category": sub_category
                })

    # Hard-Assert Invariants (D-4399, CLM-3634)
    assert total_scanned_frames == EXPECTED_TOTAL_WINDOWS, f"[FATAL] Frame count mismatch: {total_scanned_frames}"
    assert total_active_queries == EXPECTED_TOTAL_QUERIES, f"[FATAL] Active query count mismatch: {total_active_queries}"
    assert len(records_1576) == TOTAL_ORACLE_RESCUES, f"[FATAL] Rescue count: {len(records_1576)} != {TOTAL_ORACLE_RESCUES}"
    assert sum(category_counts.values()) == TOTAL_ORACLE_RESCUES, "[FATAL] Invariant sum(Cat_k) == 1576 violated!"
    assert (cat1_sub_counts["Cat 1A: Native Top-100 Member"] + cat1_sub_counts["Cat 1B: Rank/Score-Suppressed"]) == category_counts["Cat 1: Native Spatial + Class Support (I_same >= 0.30)"], "[FATAL] Cat 1A + Cat 1B != Cat 1!"

    print(f"[+] Verified Population Invariants: Exactly {len(records_1576)} records generated across 20,075 frames.\n")

    # =========================================================================
    # RECOVERABILITY DECOMPOSITION REPORT (SUM == 1,576)
    # =========================================================================
    print("=" * 115)
    print("RECOVERABILITY DECOMPOSITION OF ALL 1,576 UTILITY-POSITIVE TEMPORAL EVENTS")
    print("=" * 115)
    print(f"{'Recoverability Category':<62} | {'Count':<6} | {'% of 1576':<10} | {'Small':<7} | {'Medium':<7} | {'Large'}")
    print("-" * 115)

    ordered_categories = [
        "Cat 1: Native Spatial + Class Support (I_same >= 0.30)",
        "Cat 2: Classification-Limited Proxy (I_any >= 0.30, I_same < 0.30)",
        "Cat 3: Localization-Limited Proxy (0.10 <= I_same, I_any < 0.30)",
        "Cat 4: Combined Class + Loc Proxy (0.10 <= I_any < 0.30, I_same < 0.10)",
        "Cat 5: No Close Native Geometric Support (I_any < 0.10)"
    ]

    total_in_native_reach = 0
    for cat in ordered_categories:
        cnt = category_counts[cat]
        pct = (cnt / TOTAL_ORACLE_RESCUES) * 100.0
        n_s = scale_cross_tab[cat]["Small (<32²)"]
        n_m = scale_cross_tab[cat]["Medium (32²-96²)"]
        n_l = scale_cross_tab[cat]["Large (>96²)"]
        if "Cat 5" not in cat:
            total_in_native_reach += cnt
        print(f"{cat:<62} | {cnt:>6} | {pct:>9.2f}% | {n_s:>7} | {n_m:>7} | {n_l:>7}")
        if cat.startswith("Cat 1:"):
            c1a = cat1_sub_counts["Cat 1A: Native Top-100 Member"]
            c1b = cat1_sub_counts["Cat 1B: Rank/Score-Suppressed"]
            print(f"  --> Cat 1A: Target in Native Top-100 (Operational/Competition) | {c1a:>6} | {(c1a/TOTAL_ORACLE_RESCUES)*100:>9.2f}% |")
            print(f"  --> Cat 1B: True Rank/Score-Suppressed (Outside Top-100)       | {c1b:>6} | {(c1b/TOTAL_ORACLE_RESCUES)*100:>9.2f}% |")

    print("-" * 115)
    pct_reach = (total_in_native_reach / TOTAL_ORACLE_RESCUES) * 100.0
    print(f"[*] Native Slot Support (Cats 1-4):      {total_in_native_reach:>6} / {TOTAL_ORACLE_RESCUES} ({pct_reach:.2f}%)")
    print(f"[*] No Close Native Support (Cat 5):     {category_counts['Cat 5: No Close Native Geometric Support (I_any < 0.10)']:>6} / {TOTAL_ORACLE_RESCUES} ({100.0 - pct_reach:.2f}%)\n")

    # =========================================================================
    # IoU SENSITIVITY PROFILE & EMPIRICAL PERCENTILES (P-2265)
    # =========================================================================
    print("=" * 115)
    print("IoU SENSITIVITY PROFILE & PERCENTILES ACROSS ALL 1,576 UTILITY-POSITIVE EVENTS (P-2265)")
    print("=" * 115)

    arr_i_any = np.array(all_i_any_values)
    arr_i_same = np.array(all_i_same_values)

    p_any_005 = (arr_i_any < 0.05).mean() * 100.0
    p_any_010 = (arr_i_any < 0.10).mean() * 100.0
    p_any_020 = (arr_i_any < 0.20).mean() * 100.0
    p_any_030 = (arr_i_any >= 0.30).mean() * 100.0
    p_any_050 = (arr_i_any >= 0.50).mean() * 100.0

    p_same_005 = (arr_i_same < 0.05).mean() * 100.0
    p_same_010 = (arr_i_same < 0.10).mean() * 100.0
    p_same_020 = (arr_i_same < 0.20).mean() * 100.0
    p_same_030 = (arr_i_same >= 0.30).mean() * 100.0

    print("[*] Cumulative Tail Fractions:")
    print(f"    - Best Any-Class IoU (I_any):   P(<0.05)={p_any_005:>5.2f}% | P(<0.10)={p_any_010:>5.2f}% | P(<0.20)={p_any_020:>5.2f}% | P(>=0.30)={p_any_030:>5.2f}% | P(>=0.50)={p_any_050:>5.2f}%")
    print(f"    - Best Same-Class IoU (I_same): P(<0.05)={p_same_005:>5.2f}% | P(<0.10)={p_same_010:>5.2f}% | P(<0.20)={p_same_020:>5.2f}% | P(>=0.30)={p_same_030:>5.2f}%")

    p_any_q = [float(np.percentile(arr_i_any, p)) for p in [10, 25, 50, 75, 90]]
    p_same_q = [float(np.percentile(arr_i_same, p)) for p in [10, 25, 50, 75, 90]]
    print("\n[*] Empirical Quantiles (p10 / p25 / p50 / p75 / p90):")
    print(f"    - Best Any-Class IoU (I_any):   {p_any_q[0]:.4f} / {p_any_q[1]:.4f} / {p_any_q[2]:.4f} / {p_any_q[3]:.4f} / {p_any_q[4]:.4f}")
    print(f"    - Best Same-Class IoU (I_same): {p_same_q[0]:.4f} / {p_same_q[1]:.4f} / {p_same_q[2]:.4f} / {p_same_q[3]:.4f} / {p_same_q[4]:.4f}\n")

    # =========================================================================
    # STRATIFIED NATIVE REFERENCE POPULATION PERCENTILES
    # =========================================================================
    print("=" * 115)
    print("STRATIFIED NATIVE REFERENCE POPULATION DISTRIBUTIONS (EMPIRICAL PERCENTILES)")
    print("=" * 115)
    print(f"{'Rank Tier':<20} | {'Max Score (p10 / p50 / p90)':<30} | {'Margin (p10 / p50 / p90)':<28} | {'Entropy H_B (p10 / p50 / p90)'}")
    print("-" * 115)

    for tier_name, data in stratified_native_cues.items():
        s_arr = np.array(data["scores"])
        m_arr = np.array(data["margins"])
        e_arr = np.array(data["entropies"])
        s_p = f"{np.percentile(s_arr, 10):.3f} / {np.percentile(s_arr, 50):.3f} / {np.percentile(s_arr, 90):.3f}"
        m_p = f"{np.percentile(m_arr, 10):.2f} / {np.percentile(m_arr, 50):.2f} / {np.percentile(m_arr, 90):.2f}"
        e_p = f"{np.percentile(e_arr, 10):.3f} / {np.percentile(e_arr, 50):.3f} / {np.percentile(e_arr, 90):.3f}"
        print(f"{tier_name:<20} | {s_p:<30} | {m_p:<28} | {e_p}")

    dens_arr = np.array(frame_context_taps["temporal_proposal_density"])
    nrg_arr = np.array(frame_context_taps["detector_activation_energy"])
    print("\n[*] Frame-Level Context Taps (All 20,075 Target Frames):")
    print(f"    - Temporal Proposal Density (p10 / p50 / p90):  {np.percentile(dens_arr, 10):.3f} / {np.percentile(dens_arr, 50):.3f} / {np.percentile(dens_arr, 90):.3f}")
    print(f"    - Detector Activation Energy (p10 / p50 / p90): {np.percentile(nrg_arr, 10):.2f} / {np.percentile(nrg_arr, 50):.2f} / {np.percentile(nrg_arr, 90):.2f}")

    # =========================================================================
    # FREEZE ARTIFACTS
    # =========================================================================
    summary_payload = {
        "_meta": {
            "driver_sha256": driver_sha,
            "total_frames": total_scanned_frames,
            "total_oracle_rescues": TOTAL_ORACLE_RESCUES,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "recoverability_decomposition": {
            cat: {
                "count": category_counts[cat],
                "percentage": round((category_counts[cat] / TOTAL_ORACLE_RESCUES) * 100.0, 2),
                "scale_breakdown": dict(scale_cross_tab[cat]),
                "class_breakdown": dict(class_cross_tab[cat])
            } for cat in ordered_categories
        },
        "cat1_sub_decomposition": {
            k: {
                "count": cat1_sub_counts[k],
                "percentage": round((cat1_sub_counts[k] / TOTAL_ORACLE_RESCUES) * 100.0, 2)
            } for k in cat1_sub_counts
        },
        "iou_sensitivity": {
            "fractions_any": {"lt_005": p_any_005, "lt_010": p_any_010, "lt_020": p_any_020, "ge_030": p_any_030, "ge_050": p_any_050},
            "fractions_same": {"lt_005": p_same_005, "lt_010": p_same_010, "lt_020": p_same_020, "ge_030": p_same_030},
            "quantiles_any_p10_p25_p50_p75_p90": p_any_q,
            "quantiles_same_p10_p25_p50_p75_p90": p_same_q
        },
        "stratified_native_reference_percentiles": {
            tier: {
                "scores_p10_p50_p90": [float(np.percentile(d["scores"], p)) for p in [10, 50, 90]],
                "margins_p10_p50_p90": [float(np.percentile(d["margins"], p)) for p in [10, 50, 90]],
                "entropies_p10_p50_p90": [float(np.percentile(d["entropies"], p)) for p in [10, 50, 90]]
            } for tier, d in stratified_native_cues.items()
        },
        "frame_context_percentiles": {
            "temporal_proposal_density": [float(np.percentile(dens_arr, p)) for p in [10, 50, 90]],
            "detector_activation_energy": [float(np.percentile(nrg_arr, p)) for p in [10, 50, 90]]
        }
    }

    with open(A1R_SUMMARY_OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    with open(A1R_RECORDS_OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records_1576, f, indent=2)

    print("\n" + "=" * 115)
    print(f"[+] BFCEF-A1R EXECUTION COMPLETED IN {time.time() - t_start:.2f}s")
    print(f"[+] Summary JSON: {A1R_SUMMARY_OUT_JSON.resolve()}")
    print(f"[+] Itemized Records (1,576 rows): {A1R_RECORDS_OUT_JSON.resolve()}")
    print("=" * 115 + "\n", flush=True)


if __name__ == "__main__":
    main()