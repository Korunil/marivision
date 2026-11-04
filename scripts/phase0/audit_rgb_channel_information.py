"""
D:/MARIVISION/H0_workbench/eval_tools/audit_rgb_channel_information.py
Phase-II Proof-of-Launch: Native-Resolution RGB & Chromatic Information Audit

Evaluates:
  1. Target vs. Local Background Contrast across R, G, B, Luminance (Y),
     Opponent Channels (R-G, B - (R+G)/2), and Normalized Chromaticity (r, g, b).
  2. Contrast Dilution: Measures Michelson contrast at Native Resolution vs.
     after the official non-aspect 640x640 warp.
  3. Stratification across warped size bins (<4px, [4,8)px, [8,16)px, >=16px),
     categories, camera sensors, and reflection metadata.
  4. Exports: pol0_rgb_contrast_summary.json and pol0_rgb_contrast_strata.csv.
"""

import os
import sys
import json
import zipfile
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

ZIP_PATH = Path("D:/MARIVISION/SeaClips.zip")
TRAIN_JSON = Path("D:/MARIVISION/seaclips-train.json")
OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")

# Evaluate across a statistically representative multi-video sample (e.g. 1,000 frames)
# to maintain rapid execution while covering all sensors and categories.
SAMPLE_FRAME_LIMIT = 1200


def get_image_file(zip_ref, file_name):
    clean = file_name.replace("\\", "/").strip().lstrip("/")
    candidates = [
        clean,
        f"train/{clean}",
        clean.split("train/")[-1],
        f"SeaClips/train/{clean.split('train/')[-1]}",
        f"SeaClips/{clean}"
    ]
    for c in candidates:
        try:
            with zip_ref.open(c) as f:
                return Image.open(f).convert("RGB")
        except KeyError:
            continue
    return None


def compute_michelson(target_val, surround_val, eps=1e-5):
    return abs(target_val - surround_val) / (target_val + surround_val + eps)


def main():
    print("=" * 115)
    print("PROOF-OF-LAUNCH: NATIVE-RESOLUTION RGB & CHROMATIC CONTRAST AUDIT")
    print(f"Dataset Target: {TRAIN_JSON.resolve()} | Archive: {ZIP_PATH.resolve()}")
    print("=" * 115)

    assert TRAIN_JSON.exists(), f"[FATAL] Missing {TRAIN_JSON}"
    assert ZIP_PATH.exists(), f"[FATAL] Missing {ZIP_PATH}"

    coco = COCO(str(TRAIN_JSON))
    img_ids = sorted(coco.getImgIds())
    cats = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}

    # Step evenly across dataset to ensure multi-video coverage
    stride = max(1, len(img_ids) // SAMPLE_FRAME_LIMIT)
    sampled_img_ids = img_ids[::stride][:SAMPLE_FRAME_LIMIT]
    print(f"[*] Auditing {len(sampled_img_ids):,} representative frames across videos (stride = {stride})...")

    zip_ref = zipfile.ZipFile(ZIP_PATH, "r")

    # Data collection containers
    contrast_records = []
    size_bins = ["<4px", "[4,8)px", "[8,16)px", ">=16px"]

    processed_boxes = 0
    skipped_boxes = 0

    for idx, img_id in enumerate(sampled_img_ids):
        info = coco.loadImgs([img_id])[0]
        fname = info["file_name"]
        img = get_image_file(zip_ref, fname)
        if img is None:
            continue

        W_orig, H_orig = img.size
        img_np = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3] in [0, 1]

        # Precompute warped 640x640 image to directly test warp dilution
        img_warp_np = np.array(img.resize((640, 640), Image.BILINEAR), dtype=np.float32) / 255.0

        s_x = 640.0 / W_orig
        s_y = 640.0 / H_orig

        sensor = info.get("camera_sensor", "unknown")
        refl = info.get("water_reflections", "unknown")
        vid = fname.split("/")[0]

        anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))

        for a in anns:
            bx, by, bw, bh = a["bbox"]
            if bw < 1.0 or bh < 1.0:
                skipped_boxes += 1
                continue

            # Native target coordinates
            x1, y1 = int(round(bx)), int(round(by))
            x2, y2 = int(round(bx + bw)), int(round(by + bh))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W_orig, x2), min(H_orig, y2)

            if (x2 - x1) < 1 or (y2 - y1) < 1:
                skipped_boxes += 1
                continue

            # Surrounding background ring: expand box by 1.5x, pad by at least 3 pixels
            pad_x = max(3, int(round(bw * 0.5)))
            pad_y = max(3, int(round(bh * 0.5)))
            sx1, sy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            sx2, sy2 = min(W_orig, x2 + pad_x), min(H_orig, y2 + pad_y)

            # Native crops
            target_crop = img_np[y1:y2, x1:x2]
            surround_crop = img_np[sy1:sy2, sx1:sx2]

            # Compute surround mask (exclude target region from surround ring)
            mask_surround = np.ones((sy2 - sy1, sx2 - sx1), dtype=bool)
            rel_y1, rel_y2 = y1 - sy1, y2 - sy1
            rel_x1, rel_x2 = x1 - sx1, x2 - sx1
            mask_surround[rel_y1:rel_y2, rel_x1:rel_x2] = False

            if not np.any(mask_surround):
                skipped_boxes += 1
                continue

            # Target channel means
            t_r = float(np.mean(target_crop[:, :, 0]))
            t_g = float(np.mean(target_crop[:, :, 1]))
            t_b = float(np.mean(target_crop[:, :, 2]))

            # Surround channel means
            s_r = float(np.mean(surround_crop[mask_surround, 0]))
            s_g = float(np.mean(surround_crop[mask_surround, 1]))
            s_b = float(np.mean(surround_crop[mask_surround, 2]))

            # Luminance & Opponents
            t_y = 0.299 * t_r + 0.587 * t_g + 0.114 * t_b
            s_y_lum = 0.299 * s_r + 0.587 * s_g + 0.114 * s_b

            t_org = t_r - t_g
            s_org = s_r - s_g

            t_oby = t_b - 0.5 * (t_r + t_g)
            s_oby = s_b - 0.5 * (s_r + s_g)

            # Normalized chromaticity
            t_sum = t_r + t_g + t_b + 1e-6
            s_sum = s_r + s_g + s_b + 1e-6
            t_norm_r = t_r / t_sum
            s_norm_r = s_r / s_sum

            # Warped Coordinates & Warped Michelson Contrast
            wx1, wy1 = int(round(bx * s_x)), int(round(by * s_y))
            wx2, wy2 = int(round((bx + bw) * s_x)), int(round((by + bh) * s_y))
            wx1, wy1 = max(0, wx1), max(0, wy1)
            wx2, wy2 = min(640, wx2), min(640, wy2)

            wpad_x = max(2, int(round((wx2 - wx1) * 0.5)))
            wpad_y = max(2, int(round((wy2 - wy1) * 0.5)))
            wsx1, wsy1 = max(0, wx1 - wpad_x), max(0, wy1 - wpad_y)
            wsx2, wsy2 = min(640, wx2 + wpad_x), min(640, wy2 + wpad_y)

            target_warp_crop = img_warp_np[wy1:wy2, wx1:wx2]
            surround_warp_crop = img_warp_np[wsy1:wsy2, wsx1:wsx2]

            w_mask_surround = np.ones((wsy2 - wsy1, wsx2 - wsx1), dtype=bool)
            w_rel_y1, w_rel_y2 = wy1 - wsy1, wy2 - wsy1
            w_rel_x1, w_rel_x2 = wx1 - wsx1, wx2 - wsx1
            w_mask_surround[w_rel_y1:w_rel_y2, w_rel_x1:w_rel_x2] = False

            if target_warp_crop.size > 0 and np.any(w_mask_surround):
                wt_r = float(np.mean(target_warp_crop[:, :, 0]))
                ws_r = float(np.mean(surround_warp_crop[w_mask_surround, 0]))
                wt_y = 0.299 * wt_r + 0.587 * float(np.mean(target_warp_crop[:, :, 1])) + 0.114 * float(np.mean(target_warp_crop[:, :, 2]))
                ws_y = 0.299 * ws_r + 0.587 * float(np.mean(surround_warp_crop[w_mask_surround, 1])) + 0.114 * float(np.mean(surround_warp_crop[w_mask_surround, 2]))
                m_red_warp = compute_michelson(wt_r, ws_r)
                m_lum_warp = compute_michelson(wt_y, ws_y)
            else:
                m_red_warp = 0.0
                m_lum_warp = 0.0

            # Size binning under warp
            min_dim_warp = min(bw * s_x, bh * s_y)
            if min_dim_warp < 4.0: sb = "<4px"
            elif min_dim_warp < 8.0: sb = "[4,8)px"
            elif min_dim_warp < 16.0: sb = "[8,16)px"
            else: sb = ">=16px"

            m_red_native = compute_michelson(t_r, s_r)
            m_green_native = compute_michelson(t_g, s_g)
            m_blue_native = compute_michelson(t_b, s_b)
            m_lum_native = compute_michelson(t_y, s_y_lum)

            contrast_records.append({
                "category": cats[a["category_id"]],
                "size_bin": sb,
                "sensor": sensor,
                "reflections": refl,
                "video": vid,
                "min_dim_warp": min_dim_warp,
                # Absolute channel differences
                "delta_r": abs(t_r - s_r),
                "delta_g": abs(t_g - s_g),
                "delta_b": abs(t_b - s_b),
                "delta_lum": abs(t_y - s_y_lum),
                "delta_org": abs(t_org - s_org),
                "delta_norm_r": abs(t_norm_r - s_norm_r),
                # Michelson contrast
                "michelson_r_native": m_red_native,
                "michelson_g_native": m_green_native,
                "michelson_b_native": m_blue_native,
                "michelson_lum_native": m_lum_native,
                "michelson_r_warp": m_red_warp,
                "michelson_lum_warp": m_lum_warp,
                # Contrast retention ratio (Warp / Native)
                "retention_r": (m_red_warp / (m_red_native + 1e-5)),
                "retention_lum": (m_lum_warp / (m_lum_native + 1e-5))
            })
            processed_boxes += 1

        if (idx + 1) % 200 == 0:
            print(f"    Progress: [{idx+1:4d}/{len(sampled_img_ids):4d}] frames processed ({processed_boxes:,} boxes analyzed)...")

    zip_ref.close()
    print(f"[+] Extraction complete: {processed_boxes:,} valid boxes analyzed ({skipped_boxes:,} degenerate boxes skipped).")

    # --------------------------------------------------------------------------
    # TASK 1: CHANNEL CONTRAST COMPARISON BY SIZE BIN
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("TASK 1: NATIVE CHANNEL CONTRAST BY WARPED SIZE BIN (Testing Red vs. Blue/Green & Luminance)")
    print("=" * 115)
    print(f"{'Size Bin (Warped)':<18} | {'N Boxes':<8} | {'Michelson R':<14} | {'Michelson G':<14} | {'Michelson B':<14} | {'Michelson Lum':<15} | {'Opponent |R-G|':<14}")
    print("-" * 115)

    size_summary = {}
    for sb in size_bins:
        sub = [r for r in contrast_records if r["size_bin"] == sb]
        n_box = len(sub)
        if n_box == 0: continue

        med_r = float(np.median([r["michelson_r_native"] for r in sub]))
        med_g = float(np.median([r["michelson_g_native"] for r in sub]))
        med_b = float(np.median([r["michelson_b_native"] for r in sub]))
        med_y = float(np.median([r["michelson_lum_native"] for r in sub]))
        med_org = float(np.median([r["delta_org"] for r in sub]))

        size_summary[sb] = {
            "n": n_box, "med_r": med_r, "med_g": med_g, "med_b": med_b,
            "med_lum": med_y, "med_org": med_org
        }
        print(f"{sb:<18} | {n_box:<8,d} | {med_r:6.4f}{'':<8} | {med_g:6.4f}{'':<8} | {med_b:6.4f}{'':<8} | {med_y:6.4f}{'':<9} | {med_org:6.4f}")

    # --------------------------------------------------------------------------
    # TASK 2: CONTRAST DILUTION (NATIVE VS. 640x640 WARP)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("TASK 2: CONTRAST DILUTION ANALYSIS (How Much Signal Does 640x640 Warp Destroy?)")
    print("=" * 115)
    print(f"{'Size Bin (Warped)':<18} | {'Native Michelson R':<20} | {'Warped Michelson R':<20} | {'Red Retention Ratio':<22} | {'Lum Retention Ratio':<20}")
    print("-" * 115)

    dilution_summary = {}
    for sb in size_bins:
        sub = [r for r in contrast_records if r["size_bin"] == sb]
        if not sub: continue

        nat_r = float(np.median([r["michelson_r_native"] for r in sub]))
        wrp_r = float(np.median([r["michelson_r_warp"] for r in sub]))
        ret_r = float(np.median([r["retention_r"] for r in sub]))
        ret_y = float(np.median([r["retention_lum"] for r in sub]))

        dilution_summary[sb] = {
            "nat_r": nat_r, "wrp_r": wrp_r, "ret_r": ret_r, "ret_lum": ret_y
        }
        print(f"{sb:<18} | {nat_r:6.4f}{'':<14} | {wrp_r:6.4f}{'':<14} | {ret_r*100:5.1f}% (Signal Kept){'':<4} | {ret_y*100:5.1f}% (Signal Kept)")

    # --------------------------------------------------------------------------
    # TASK 3: PER-CATEGORY RED & OPPONENT DOMINANCE
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("TASK 3: CATEGORY DECOMPOSITION: RED CHANNEL VS. LUMINANCE CONTRAST (<8px Objects)")
    print("=" * 115)
    print(f"{'Category':<18} | {'<8px Boxes':<12} | {'Native Michelson R':<20} | {'Native Michelson Lum':<22} | {'Red / Lum Contrast Advantage':<28}")
    print("-" * 115)

    cat_summary = {}
    for cname in sorted(cats.values()):
        sub = [r for r in contrast_records if r["category"] == cname and r["min_dim_warp"] < 8.0]
        n_box = len(sub)
        if n_box < 10: continue

        med_r = float(np.median([r["michelson_r_native"] for r in sub]))
        med_y = float(np.median([r["michelson_lum_native"] for r in sub]))
        advantage = ((med_r - med_y) / (med_y + 1e-5)) * 100.0

        cat_summary[cname] = {"n": n_box, "med_r": med_r, "med_lum": med_y, "advantage_pct": advantage}
        print(f"{cname:<18} | {n_box:<12,d} | {med_r:6.4f}{'':<14} | {med_y:6.4f}{'':<16} | {advantage:+6.1f}%")

    # --------------------------------------------------------------------------
    # TASK 4: SENSOR & REFLECTION COMPARISON
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("TASK 4: SENSOR & REFLECTION METADATA STRATIFICATION (<8px Objects)")
    print("=" * 115)

    def print_meta_contrast(dim_name):
        dim_vals = sorted(list(set(r[dim_name] for r in contrast_records)))
        print(f"--- Stratum: {dim_name.upper()} ---")
        print(f"  {'Condition':<18} | {'N (<8px)':<10} | {'Michelson R':<14} | {'Michelson B':<14} | {'Opponent |R-G|':<16}")
        print("  " + "-" * 80)
        for v in dim_vals:
            sub = [r for r in contrast_records if r[dim_name] == v and r["min_dim_warp"] < 8.0]
            if len(sub) < 10: continue
            mr = np.median([r["michelson_r_native"] for r in sub])
            mb = np.median([r["michelson_b_native"] for r in sub])
            morg = np.median([r["delta_org"] for r in sub])
            print(f"  {v:<18} | {len(sub):<10,d} | {mr:6.4f}{'':<8} | {mb:6.4f}{'':<8} | {morg:6.4f}")
        print()

    print_meta_contrast("sensor")
    print_meta_contrast("reflections")

    # --------------------------------------------------------------------------
    # EXPORT ARTIFACTS
    # --------------------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "pol0_rgb_contrast_summary.json"
    csv_path = OUT_DIR / "pol0_rgb_contrast_strata.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(contrast_records[0].keys()))
        writer.writeheader()
        writer.writerows(contrast_records)

    summary_export = {
        "_meta": {
            "audit_version": "PoL-0_RGB_Contrast_Audit_v1.0",
            "frames_audited": len(sampled_img_ids),
            "total_boxes_analyzed": processed_boxes
        },
        "contrast_by_size": size_summary,
        "dilution_by_size": dilution_summary,
        "category_advantage": cat_summary
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_export, f, indent=2)

    print("=" * 115)
    print(f"[+] RGB audit records written successfully:")
    print(f"    1. {json_path}")
    print(f"    2. {csv_path}")
    print("=" * 115)


if __name__ == "__main__":
    main()