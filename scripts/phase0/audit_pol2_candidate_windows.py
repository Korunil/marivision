"""
D:/MARIVISION/H0_workbench/eval_tools/audit_pol2_candidate_windows.py
Phase-II Proof-of-Launch Stage 2 (PoL-2): Pre-Flight Candidate Window Opportunity Audit

Strict Compliance Checklist:
  - D-3326 / D-3330: Produces a candidate window opportunity audit; does NOT freeze clips yet.
  - D-3327 / D-3328: Decomposes opportunity into FASTPATH (CA at t-1) vs. MEMORY_INCREMENTAL (no CA at t-1, CA in t-2..t-5).
  - D-3329: Tracks raw (s > 0.00) and operational (s >= 0.20) opportunities separately.
  - D-3331: Asserts that target L=5 backward chains map strictly to frames t-1..t-5 of the candidate window.
  - D-3332: Hard-asserts live TRAIN JSON hash, M0 checkpoint hash, approved links hash, 20,310 images, and baseline counts.
  - Metadata consistency: Audits sensor, weather, sea-state, and reflections across all 6 window frames.
"""

import json
import hashlib
import csv
from pathlib import Path
from collections import defaultdict
from pycocotools.coco import COCO

TRAIN_JSON = Path("D:/MARIVISION/seaclips-train.json")
OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
CACHE_FILE = OUT_DIR / "cache_m0_train_raw_top100.json"
LINKS_JSON = OUT_DIR / "pol0_pseudo_links_v1.json"

OUT_CANDIDATES_JSON = OUT_DIR / "pol2_candidate_windows_v1.json"
OUT_CANDIDATES_CSV = OUT_DIR / "pol2_candidate_windows_v1.csv"

EXPECTED_LINKS_SHA256 = "e132f0a5782e0a53da858db80e64db0b879652057eeea9c149af2735cb25a2fb"
EXPECTED_M0_HASH = "28e6288c3f6769c2d5275aa77ddb248735b55a09cd46a6b570e3f117890b268b"
EVAL_IOU_THRESHOLD = 0.30
OPERATING_SCORE_THRESHOLD = 0.20


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


def main():
    print("=" * 115)
    print("PROOF-OF-LAUNCH STAGE 2 (PoL-2): CANDIDATE WINDOW OPPORTUNITY AUDIT")
    print(f"Target Dataset:   {TRAIN_JSON.resolve()}")
    print(f"Target Links Map: {LINKS_JSON.resolve()}")
    print(f"Target M0 Cache:  {CACHE_FILE.resolve()}")
    print("=" * 115)

    assert TRAIN_JSON.exists(), f"[FATAL] Missing {TRAIN_JSON}"
    assert LINKS_JSON.exists(), f"[FATAL] Missing {LINKS_JSON}"
    assert CACHE_FILE.exists(), f"[FATAL] Missing {CACHE_FILE}"

    train_json_hash = compute_sha256(TRAIN_JSON)
    links_json_hash = compute_sha256(LINKS_JSON)

    # --------------------------------------------------------------------------
    # STEP 1: HARD PROVENANCE & BASELINE INVENTORY ASSERTIONS (D-3332)
    # --------------------------------------------------------------------------
    print("\n--- STEP 1: VERIFYING PROVENANCE & REPRODUCING BASELINE INVENTORY ---")
    assert links_json_hash == EXPECTED_LINKS_SHA256, (
        f"[FATAL] Approved links hash mismatch!\n"
        f"  Expected: {EXPECTED_LINKS_SHA256}\n"
        f"  Found:    {links_json_hash}"
    )

    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache_data = json.load(f)
    cache_meta = cache_data.get("_meta", {})
    assert cache_meta.get("train_json_sha256") == train_json_hash, "[FATAL] Prediction cache train JSON mismatch!"
    assert cache_meta.get("m0_checkpoint_sha256") == EXPECTED_M0_HASH, "[FATAL] M0 Checkpoint SHA mismatch!"

    with open(LINKS_JSON, "r", encoding="utf-8") as f:
        links_data = json.load(f)
    links_meta = links_data.get("_meta", {})
    assert links_meta.get("train_json_sha256") == train_json_hash, "[FATAL] Links JSON train JSON mismatch!"

    raw_detections = {int(k): v for k, v in cache_data["detections"].items()}
    frozen_links = links_data["links"]
    assert len(frozen_links) == 65542, f"[FATAL] Link count mismatch: {len(frozen_links)} != 65,542"

    coco = COCO(str(TRAIN_JSON))
    img_ids = sorted(coco.getImgIds())
    ann_ids = sorted(coco.getAnnIds())
    cats = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    ann_lookup = {a["id"]: a for a in coco.loadAnns(ann_ids)}

    assert len(img_ids) == 20310, f"[FATAL] Expected 20,310 frames, found {len(img_ids)}"
    assert len(ann_ids) == 65542, f"[FATAL] Expected 65,542 annotations, found {len(ann_ids)}"
    assert set(raw_detections.keys()) == set(img_ids), "[FATAL] Prediction cache image ID set does not match dataset!"

    # Verify baseline in-sample miss counts
    total_4px_gt = 0
    total_4px_misses = 0
    for a in ann_lookup.values():
        img_info = coco.loadImgs([a["image_id"]])[0]
        sx = 640.0 / float(img_info["width"])
        sy = 640.0 / float(img_info["height"])
        min_dim = min(a["bbox"][2] * sx, a["bbox"][3] * sy)
        if min_dim < 4.0:
            total_4px_gt += 1
            preds = raw_detections.get(a["image_id"], [])
            covered = any(
                p["category_id"] == a["category_id"] and box_iou(p["bbox"], a["bbox"]) >= EVAL_IOU_THRESHOLD
                for p in preds
            )
            if not covered:
                total_4px_misses += 1

    assert total_4px_gt == 33751, f"[FATAL] Warped <4px GT mismatch: {total_4px_gt} != 33,751"
    assert total_4px_misses == 12372, f"[FATAL] Warped <4px M0 misses mismatch: {total_4px_misses} != 12,372"
    print(f"[+] Provenance and Baseline Invariants Certified:")
    print(f"    - Live Train JSON SHA-256: {train_json_hash} (VERIFIED)")
    print(f"    - Approved Links SHA-256:  {links_json_hash} (VERIFIED)")
    print(f"    - Certified M0 SHA-256:    {EXPECTED_M0_HASH} (VERIFIED)")
    print(f"    - Baseline <4px Invariants: Exactly {total_4px_gt:,} GT and {total_4px_misses:,} CA misses verified.")

    # --------------------------------------------------------------------------
    # STEP 2: ORGANIZE CHRONOLOGICAL VIDEO FRAMES & METADATA
    # --------------------------------------------------------------------------
    print("\n--- STEP 2: SCANNING 6-FRAME WINDOWS & VERIFYING L=5 LINK CHAINS ---")
    video_frames = defaultdict(list)
    img_meta = {}

    for img_id in img_ids:
        info = coco.loadImgs([img_id])[0]
        v_id = int(info["video_id"])
        f_id = int(info["frame_id"])
        img_meta[img_id] = {
            "width": float(info["width"]), "height": float(info["height"]),
            "video_id": v_id, "frame_id": f_id, "file_name": info["file_name"],
            "sensor": info.get("camera_sensor", "unknown"),
            "weather": info.get("weather", "unknown"),
            "sea_state": info.get("sea_state", "unknown"),
            "reflections": info.get("water_reflections", "unknown"),
        }
        video_frames[v_id].append((f_id, img_id))

    for v_id in video_frames:
        video_frames[v_id].sort(key=lambda x: x[0])

    candidate_windows = []
    total_valid_windows = 0

    for v_id, frames in sorted(video_frames.items()):
        if len(frames) < 6:
            continue

        for i in range(len(frames) - 5):
            window = frames[i:i + 6]
            f_ids = [f[0] for f in window]
            i_ids = [f[1] for f in window]

            # Require strictly consecutive frame_ids
            if f_ids != list(range(f_ids[0], f_ids[0] + 6)):
                continue

            total_valid_windows += 1
            final_iid = i_ids[-1]
            meta_final = img_meta[final_iid]
            w_scale = 640.0 / meta_final["width"]
            h_scale = 640.0 / meta_final["height"]

            # Audit metadata constancy across all 6 frames
            sensors = {img_meta[iid]["sensor"] for iid in i_ids}
            weathers = {img_meta[iid]["weather"] for iid in i_ids}
            sea_states = {img_meta[iid]["sea_state"] for iid in i_ids}
            reflections = {img_meta[iid]["reflections"] for iid in i_ids}

            final_anns = coco.loadAnns(coco.getAnnIds(imgIds=[final_iid]))
            preds_final = raw_detections.get(final_iid, [])

            opportunity_targets = []

            for a in final_anns:
                bw, bh = a["bbox"][2], a["bbox"][3]
                min_dim = min(bw * w_scale, bh * h_scale)
                if min_dim >= 4.0:
                    continue  # Target the <4px failure regime

                cid = a["category_id"]
                covered_at_t = any(
                    p["category_id"] == cid and box_iou(p["bbox"], a["bbox"]) >= EVAL_IOU_THRESHOLD
                    for p in preds_final
                )
                if covered_at_t:
                    continue  # Target baseline misses only

                # Step 3: Verify strict backward chain mapping to window frames (D-3331)
                chain_aids = []
                chain_iids = []
                chain_fids = []
                curr_aid = str(a["id"])

                for step in range(5):
                    link_entry = frozen_links.get(curr_aid)
                    if link_entry and link_entry["status"] == "LINKED" and link_entry["parent_id"] is not None:
                        p_aid = link_entry["parent_id"]
                        p_ann = ann_lookup[p_aid]
                        p_meta = img_meta[p_ann["image_id"]]

                        chain_aids.append(p_aid)
                        chain_iids.append(p_ann["image_id"])
                        chain_fids.append(p_meta["frame_id"])
                        curr_aid = str(p_aid)
                    else:
                        break

                if len(chain_aids) != 5:
                    continue  # Incomplete chain

                # D-3331: Chain frames must match window frames [t-1, t-2, t-3, t-4, t-5]
                expected_fids = f_ids[4::-1]  # [f4, f3, f2, f1, f0]
                expected_iids = i_ids[4::-1]
                if chain_fids != expected_fids or chain_iids != expected_iids:
                    continue  # Chain does not map to this candidate window

                # Interrogate M0 predictions for each parent k in [1..5]
                ca_raw = [False] * 5
                scores = [0.0] * 5
                ca_s20 = [False] * 5

                for k_idx, p_aid in enumerate(chain_aids):
                    p_ann = ann_lookup[p_aid]
                    p_preds = raw_detections.get(p_ann["image_id"], [])
                    matching_preds = [
                        d for d in p_preds
                        if d["category_id"] == cid and box_iou(d["bbox"], p_ann["bbox"]) >= EVAL_IOU_THRESHOLD
                    ]
                    if matching_preds:
                        ca_raw[k_idx] = True
                        top_s = max(d["score"] for d in matching_preds)
                        scores[k_idx] = top_s
                        if top_s >= OPERATING_SCORE_THRESHOLD:
                            ca_s20[k_idx] = True

                # D-3327, D-3328: Decompose into FASTPATH vs MEMORY_INCREMENTAL
                is_fastpath_raw = ca_raw[0]
                is_fastpath_s20 = ca_s20[0]

                is_mem_incr_raw = (not ca_raw[0]) and any(ca_raw[1:])
                is_mem_incr_s20 = (not ca_s20[0]) and any(ca_s20[1:])

                is_persist_ge2_raw = (sum(ca_raw) >= 2)
                is_persist_ge2_s20 = (sum(ca_s20) >= 2)

                # Include if target provides any valid opportunity
                if any(ca_raw):
                    opportunity_targets.append({
                        "ann_id": a["id"],
                        "category": cats[cid],
                        "min_dim_warp": min_dim,
                        "ca_raw_k": ca_raw,
                        "scores_k": scores,
                        "ca_s20_k": ca_s20,
                        "fastpath_raw": is_fastpath_raw,
                        "fastpath_s20": is_fastpath_s20,
                        "mem_incr_raw": is_mem_incr_raw,
                        "mem_incr_s20": is_mem_incr_s20,
                        "persist_ge2_raw": is_persist_ge2_raw,
                        "persist_ge2_s20": is_persist_ge2_s20,
                        "chain_ann_ids": chain_aids,
                    })

            if len(opportunity_targets) > 0:
                candidate_windows.append({
                    "video_id": v_id,
                    "frame_start": f_ids[0],
                    "frame_end": f_ids[-1],
                    "frame_ids": f_ids,
                    "image_ids": i_ids,
                    "sensor": "/".join(sorted(sensors)),
                    "weather": "/".join(sorted(weathers)),
                    "sea_state": "/".join(sorted(sea_states)),
                    "reflections": "/".join(sorted(reflections)),
                    "n_total_opp_raw": len(opportunity_targets),
                    "n_fastpath_raw": sum(1 for t in opportunity_targets if t["fastpath_raw"]),
                    "n_fastpath_s20": sum(1 for t in opportunity_targets if t["fastpath_s20"]),
                    "n_mem_incr_raw": sum(1 for t in opportunity_targets if t["mem_incr_raw"]),
                    "n_mem_incr_s20": sum(1 for t in opportunity_targets if t["mem_incr_s20"]),
                    "n_persist_raw": sum(1 for t in opportunity_targets if t["persist_ge2_raw"]),
                    "n_persist_s20": sum(1 for t in opportunity_targets if t["persist_ge2_s20"]),
                    "targets": opportunity_targets,
                })

    print(f"[+] Scanned {total_valid_windows:,} strictly consecutive 6-frame windows across {len(video_frames)} videos.")
    print(f"[+] Identified {len(candidate_windows):,} candidate windows containing >=1 linkable <4px miss with historical coverage.")

    # --------------------------------------------------------------------------
    # STEP 3: AGGREGATE OPPORTUNITY DISTRIBUTION PER VIDEO
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("CANDIDATE WINDOW OPPORTUNITY AUDIT: PER-VIDEO AGGREGATE SUMMARY")
    print("Evaluates availability of Fast-Path (t-1) vs. Memory-Incremental (t-2..t-5) targets across videos")
    print("=" * 115)
    print(f"{'Video ID':<10} | {'Windows':<9} | {'Sensor':<10} | {'FastPath (raw / s20)':<22} | {'MemIncr (raw / s20)':<22} | {'Persist>=2 (raw / s20)':<24} | {'Categories'}")
    print("-" * 115)

    video_summary = defaultdict(lambda: {
        "windows": 0, "sensor": "", "max_fp_raw": 0, "max_fp_s20": 0,
        "max_mi_raw": 0, "max_mi_s20": 0, "max_p2_raw": 0, "max_p2_s20": 0,
        "cats": set(), "best_win": None
    })

    for w in candidate_windows:
        v_id = w["video_id"]
        v_rec = video_summary[v_id]
        v_rec["windows"] += 1
        v_rec["sensor"] = w["sensor"]
        v_rec["max_fp_raw"] = max(v_rec["max_fp_raw"], w["n_fastpath_raw"])
        v_rec["max_fp_s20"] = max(v_rec["max_fp_s20"], w["n_fastpath_s20"])
        v_rec["max_mi_raw"] = max(v_rec["max_mi_raw"], w["n_mem_incr_raw"])
        v_rec["max_mi_s20"] = max(v_rec["max_mi_s20"], w["n_mem_incr_s20"])
        v_rec["max_p2_raw"] = max(v_rec["max_p2_raw"], w["n_persist_raw"])
        v_rec["max_p2_s20"] = max(v_rec["max_p2_s20"], w["n_persist_s20"])
        for t in w["targets"]:
            v_rec["cats"].add(t["category"])

        # Track the window with the most balanced opportunity profile
        score_balance = (w["n_fastpath_s20"] > 0) + (w["n_mem_incr_s20"] > 0) + (w["n_total_opp_raw"])
        if v_rec["best_win"] is None or score_balance > v_rec.get("best_score", -1):
            v_rec["best_win"] = w
            v_rec["best_score"] = score_balance

    # Sort videos by combined s >= 0.20 opportunity capacity
    sorted_vids = sorted(
        video_summary.items(),
        key=lambda x: (x[1]["max_mi_s20"] > 0, x[1]["max_fp_s20"] + x[1]["max_mi_s20"], x[1]["windows"]),
        reverse=True
    )

    for v_id, stats in sorted_vids:
        cat_str = ", ".join(sorted(stats["cats"]))
        fp_str = f"{stats['max_fp_raw']:2d} / {stats['max_fp_s20']:2d}"
        mi_str = f"{stats['max_mi_raw']:2d} / {stats['max_mi_s20']:2d}"
        p2_str = f"{stats['max_p2_raw']:2d} / {stats['max_p2_s20']:2d}"
        print(f"video_{v_id:<4d} | {stats['windows']:<9,d} | {stats['sensor']:<10} | {fp_str:<22} | {mi_str:<22} | {p2_str:<24} | {cat_str}")

    # --------------------------------------------------------------------------
    # STEP 4: CATEGORY-LEVEL OPPORTUNITY AVAILABILITY
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("CATEGORY OPPORTUNITY DISTRIBUTION ACROSS ALL CANDIDATE WINDOWS")
    print("=" * 115)
    print(f"{'Category':<18} | {'Total Target Instances':<24} | {'FastPath (s20)':<18} | {'MemIncr (s20)':<18} | {'Persist>=2 (s20)':<18}")
    print("-" * 105)

    cat_counts = defaultdict(lambda: {"total": 0, "fp_s20": 0, "mi_s20": 0, "p2_s20": 0})
    for w in candidate_windows:
        for t in w["targets"]:
            c = t["category"]
            cat_counts[c]["total"] += 1
            if t["fastpath_s20"]: cat_counts[c]["fp_s20"] += 1
            if t["mem_incr_s20"]: cat_counts[c]["mi_s20"] += 1
            if t["persist_ge2_s20"]: cat_counts[c]["p2_s20"] += 1

    for cname in sorted(cats.values()):
        cc = cat_counts[cname]
        print(f"{cname:<18} | {cc['total']:<24,d} | {cc['fp_s20']:<18,d} | {cc['mi_s20']:<18,d} | {cc['p2_s20']:<18,d}")

    # --------------------------------------------------------------------------
    # STEP 5: EXPORT CANDIDATE AUDIT ARTIFACTS (D-3330)
    # --------------------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Master Candidate JSON
    master_candidate_payload = {
        "_meta": {
            "audit_name": "pol2_candidate_windows_v1",
            "train_json": str(TRAIN_JSON.resolve()),
            "train_json_sha256": train_json_hash,
            "links_json_sha256": links_json_hash,
            "m0_checkpoint_sha256": EXPECTED_M0_HASH,
            "total_candidate_windows": len(candidate_windows),
            "status": "CANDIDATE_AUDIT_COMPLETE_AWAITING_SELECTION_RULE"
        },
        "windows": candidate_windows
    }

    with open(OUT_CANDIDATES_JSON, "w", encoding="utf-8") as f:
        json.dump(master_candidate_payload, f, indent=2)

    cand_sha256 = compute_sha256(OUT_CANDIDATES_JSON)

    # Flattened CSV for rapid inspection
    csv_rows = []
    for w in candidate_windows:
        csv_rows.append({
            "video_id": w["video_id"],
            "frame_start": w["frame_start"],
            "frame_end": w["frame_end"],
            "sensor": w["sensor"],
            "weather": w["weather"],
            "sea_state": w["sea_state"],
            "reflections": w["reflections"],
            "n_total_opp_raw": w["n_total_opp_raw"],
            "n_fastpath_raw": w["n_fastpath_raw"],
            "n_fastpath_s20": w["n_fastpath_s20"],
            "n_mem_incr_raw": w["n_mem_incr_raw"],
            "n_mem_incr_s20": w["n_mem_incr_s20"],
            "n_persist_raw": w["n_persist_raw"],
            "n_persist_s20": w["n_persist_s20"],
        })

    with open(OUT_CANDIDATES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\n" + "=" * 115)
    print("CANDIDATE WINDOW OPPORTUNITY AUDIT COMPLETE (D-3330)")
    print(f"Master Candidate JSON: {OUT_CANDIDATES_JSON.resolve()}")
    print(f"Candidate CSV:         {OUT_CANDIDATES_CSV.resolve()}")
    print(f"Candidate SHA-256:     {cand_sha256}")
    print("=" * 115)


if __name__ == "__main__":
    main()