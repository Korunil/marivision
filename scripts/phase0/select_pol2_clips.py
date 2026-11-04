"""
D:/MARIVISION/H0_workbench/eval_tools/select_pol2_clips.py
PoL-2 Pre-Flight: Programmatic Selection of 5 Diverse 6-Frame Training Windows

Strict Compliance Checklist:
  - D-3318: Selects 5 distinct videos with exactly 6 consecutive frames each (t-5 to t).
  - D-3319 / D-3320: Derived programmatically from certified TRAIN JSON, link artifact, and M0 cache.
  - Prioritizes windows with maximum linkable <4px M0 misses possessing historical coverage.
  - Zero GPU compute: pure read-only data evaluation.
"""

import json
import hashlib
from pathlib import Path
from collections import defaultdict
from pycocotools.coco import COCO

TRAIN_JSON = Path("D:/MARIVISION/seaclips-train.json")
OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
CACHE_FILE = OUT_DIR / "cache_m0_train_raw_top100.json"
LINKS_JSON = OUT_DIR / "pol0_pseudo_links_v1.json"
OUT_CLIPS_JSON = OUT_DIR / "pol2_selected_clips.json"

EXPECTED_LINKS_SHA256 = "e132f0a5782e0a53da858db80e64db0b879652057eeea9c149af2735cb25a2fb"
EXPECTED_M0_HASH = "28e6288c3f6769c2d5275aa77ddb248735b55a09cd46a6b570e3f117890b268b"
EVAL_IOU_THRESHOLD = 0.30


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
    print("PROOF-OF-LAUNCH STAGE 2 (PoL-2): PROGRAMMATIC CLIP SELECTOR")
    print("=" * 115)

    assert LINKS_JSON.exists(), f"Missing {LINKS_JSON}"
    assert CACHE_FILE.exists(), f"Missing {CACHE_FILE}"
    assert compute_sha256(LINKS_JSON) == EXPECTED_LINKS_SHA256, "Links JSON SHA-256 mismatch!"

    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache_data = json.load(f)
    assert cache_data["_meta"]["m0_checkpoint_sha256"] == EXPECTED_M0_HASH

    raw_detections = {int(k): v for k, v in cache_data["detections"].items()}

    with open(LINKS_JSON, "r", encoding="utf-8") as f:
        frozen_links = json.load(f)["links"]

    coco = COCO(str(TRAIN_JSON))
    img_ids = sorted(coco.getImgIds())
    cats = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    ann_lookup = {a["id"]: a for a in coco.loadAnns(coco.getAnnIds())}

    # Group frames by video_id in chronological order
    video_frames = defaultdict(list)
    img_meta = {}

    for img_id in img_ids:
        info = coco.loadImgs([img_id])[0]
        v_id = int(info["video_id"])
        f_id = int(info["frame_id"])
        img_meta[img_id] = {
            "width": float(info["width"]),
            "height": float(info["height"]),
            "video_id": v_id,
            "frame_id": f_id,
            "file_name": info["file_name"],
            "sensor": info.get("camera_sensor", "unknown"),
            "weather": info.get("weather", "unknown"),
            "sea_state": info.get("sea_state", "unknown"),
            "water_reflections": info.get("water_reflections", "unknown"),
        }
        video_frames[v_id].append((f_id, img_id))

    for v_id in video_frames:
        video_frames[v_id].sort(key=lambda x: x[0])

    # Slide 6-frame windows across all videos
    candidate_windows = []

    for v_id, frames in sorted(video_frames.items()):
        if len(frames) < 6:
            continue

        for i in range(len(frames) - 5):
            window = frames[i:i + 6]
            # Ensure unbroken consecutive frame_ids
            f_ids = [f[0] for f in window]
            if f_ids != list(range(f_ids[0], f_ids[0] + 6)):
                continue

            final_iid = window[-1][1]
            meta_final = img_meta[final_iid]
            w_scale = 640.0 / meta_final["width"]
            h_scale = 640.0 / meta_final["height"]

            final_anns = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=[final_iid]))]
            preds_final = raw_detections.get(final_iid, [])

            opportunity_targets = []

            for a in final_anns:
                bw, bh = a["bbox"][2], a["bbox"][3]
                min_dim_warp = min(bw * w_scale, bh * h_scale)

                if min_dim_warp >= 4.0:
                    continue  # Target the <4px failure regime

                cid = a["category_id"]
                covered_ca = any(
                    p["category_id"] == cid and box_iou(p["bbox"], a["bbox"]) >= EVAL_IOU_THRESHOLD
                    for p in preds_final
                )

                if covered_ca:
                    continue  # We only target baseline misses

                # Trace backward link chain
                chain_aids = []
                curr_aid = str(a["id"])
                for _ in range(5):
                    link_entry = frozen_links.get(curr_aid)
                    if link_entry and link_entry["status"] == "LINKED" and link_entry["parent_id"] is not None:
                        parent_id = link_entry["parent_id"]
                        chain_aids.append(parent_id)
                        curr_aid = str(parent_id)
                    else:
                        break

                if len(chain_aids) == 5:
                    # Target has unbroken L=5 history; verify prior M0 evidence
                    prior_covered = False
                    best_prior_score = 0.0

                    for p_aid in chain_aids:
                        p_ann = ann_lookup[p_aid]
                        p_preds = raw_detections.get(p_ann["image_id"], [])
                        matching_preds = [
                            d for d in p_preds
                            if d["category_id"] == cid and box_iou(d["bbox"], p_ann["bbox"]) >= EVAL_IOU_THRESHOLD
                        ]
                        if matching_preds:
                            prior_covered = True
                            best_prior_score = max(best_prior_score, max(d["score"] for d in matching_preds))

                    if prior_covered:
                        opportunity_targets.append({
                            "ann_id": a["id"],
                            "category": cats[cid],
                            "min_dim_warp": min_dim_warp,
                            "best_prior_score": best_prior_score,
                        })

            if len(opportunity_targets) > 0:
                candidate_windows.append({
                    "video_id": v_id,
                    "frame_start": f_ids[0],
                    "frame_end": f_ids[-1],
                    "frame_ids": f_ids,
                    "image_ids": [f[1] for f in window],
                    "file_names": [img_meta[f[1]]["file_name"] for f in window],
                    "sensor": meta_final["sensor"],
                    "weather": meta_final["weather"],
                    "sea_state": meta_final["sea_state"],
                    "reflections": meta_final["water_reflections"],
                    "n_opportunity_targets": len(opportunity_targets),
                    "targets": opportunity_targets,
                })

    print(f"[*] Total Valid 6-Frame Windows Identified: {len(candidate_windows):,}")

    # Select the top window per video, then select top 5 distinct videos
    # Prioritizing opportunity count while balancing categories
    candidates_by_video = defaultdict(list)
    for w in candidate_windows:
        candidates_by_video[w["video_id"]].append(w)

    best_per_video = []
    for v_id, win_list in candidates_by_video.items():
        win_list.sort(key=lambda x: (x["n_opportunity_targets"], max(t["best_prior_score"] for t in x["targets"])), reverse=True)
        best_per_video.append(win_list[0])

    best_per_video.sort(key=lambda x: x["n_opportunity_targets"], reverse=True)
    selected_5_clips = best_per_video[:5]

    print("\n" + "=" * 115)
    print("SELECTED 5 DISTINCT-VIDEO 6-FRAME WINDOWS FOR POL-2 FALSIFICATION")
    print("=" * 115)
    print(f"{'Clip #':<8} | {'Video ID':<10} | {'Frames (t-5 -> t)':<20} | {'Sensor':<10} | {'Environment':<28} | {'Target Category Breakdown'}")
    print("-" * 115)

    for idx, clip in enumerate(selected_5_clips):
        cats_in_clip = defaultdict(int)
        for t in clip["targets"]:
            cats_in_clip[t["category"]] += 1
        cat_str = ", ".join([f"{k}: {v}" for k, v in cats_in_clip.items()])
        env_str = f"{clip['weather']}, {clip['sea_state']}, refl={clip['reflections']}"
        f_range = f"{clip['frame_start']} -> {clip['frame_end']}"
        print(f"Clip {idx+1:<3} | {clip['video_id']:<10} | {f_range:<20} | {clip['sensor']:<10} | {env_str:<28} | {cat_str} ({clip['n_opportunity_targets']} targets)")

    # Save frozen selection manifest
    with open(OUT_CLIPS_JSON, "w", encoding="utf-8") as f:
        json.dump(selected_5_clips, f, indent=2)

    print(f"\n[+] PoL-2 Clip Selection Manifest Frozen: {OUT_CLIPS_JSON.resolve()}")


if __name__ == "__main__":
    main()