"""
D:/MARIVISION/H0_workbench/eval_tools/inspect_margin_resolved_qa.py
Phase-II Proof-of-Launch Stage 0 (PoL-0): Part 2 - Supplementary MARGIN_RESOLVED QA

Strict Compliance Checklist:
  - D-3212 / D-3216: Strictly read-only against pol0_pseudo_links_v1.json; SHA-256 preserved.
  - D-3213 / P-1503: Hard assertions on total_linked == 57020 and recognized link bases.
  - D-3214 / P-1504: Renamed to LOW_MARGIN (15 lowest stored effective-margin cases).
  - D-3215 / P-1505: Cleans QA_SUPP_DIR prior to rendering; asserts 100% render completeness.
  - Verified SHA-256: Asserts e132f0a5782e0a53da858db80e64db0b879652057eeea9c149af2735cb25a2fb before and after.
"""

import os
import sys
import json
import hashlib
import zipfile
import csv
import random
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image, ImageDraw
from pycocotools.coco import COCO

ZIP_PATH = Path("D:/MARIVISION/SeaClips.zip")
TRAIN_JSON = Path("D:/MARIVISION/seaclips-train.json")
OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
LINKS_JSON = OUT_DIR / "pol0_pseudo_links_v1.json"
QA_SUPP_DIR = OUT_DIR / "qa_crops_margin_resolved"
QA_SUPP_MANIFEST = OUT_DIR / "pol0_margin_resolved_qa_manifest.csv"

CANDIDATE_SHA256 = "e132f0a5782e0a53da858db80e64db0b879652057eeea9c149af2735cb25a2fb"
RANDOM_SEED = 42


def compute_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def get_image_from_zip(zip_ref, file_name):
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


def main():
    print("=" * 115)
    print("PROOF-OF-LAUNCH STAGE 0 (PoL-0): SUPPLEMENTARY QA AUDIT FOR MARGIN_RESOLVED LINKS")
    print(f"Candidate Link Map: {LINKS_JSON.resolve()}")
    print("=" * 115)

    assert LINKS_JSON.exists(), f"[FATAL] Missing {LINKS_JSON}"
    assert TRAIN_JSON.exists(), f"[FATAL] Missing {TRAIN_JSON}"
    assert ZIP_PATH.exists(), f"[FATAL] Missing {ZIP_PATH}"

    # Pre-execution candidate hash verification (D-3212)
    initial_hash = compute_sha256(LINKS_JSON)
    assert initial_hash == CANDIDATE_SHA256, (
        f"[FATAL] Pre-execution candidate hash mismatch!\n"
        f"  Expected: {CANDIDATE_SHA256}\n"
        f"  Found:    {initial_hash}"
    )
    print(f"[+] Initial Candidate Hash Verified: {initial_hash}")

    # Read-only ingestion
    with open(LINKS_JSON, "r", encoding="utf-8") as f:
        payload = json.load(f)

    links_dict = payload.get("links", {})
    assert len(links_dict) == 65542, f"[FATAL] Expected 65,542 link records, found {len(links_dict)}"

    # --------------------------------------------------------------------------
    # TASK 1: TALLY & ASSERT ACCEPTED LINKS BY LINK_BASIS (D-3213, P-1503)
    # --------------------------------------------------------------------------
    basis_counts = defaultdict(int)
    cat_basis_counts = defaultdict(lambda: defaultdict(int))
    margin_resolved_pool = []

    for aid_str, rec in links_dict.items():
        status = rec.get("status")
        if status == "LINKED":
            basis = rec.get("link_basis", "UNKNOWN")
            basis_counts[basis] += 1
            cname = rec.get("category_name", "unknown")
            cat_basis_counts[cname][basis] += 1

            if basis == "MARGIN_RESOLVED":
                margin_val = rec.get("margin")
                d_norm_val = rec.get("d_norm")
                if margin_val is not None:
                    margin_resolved_pool.append({
                        "ann_id": int(aid_str),
                        "parent_id": int(rec["parent_id"]),
                        "category": cname,
                        "video_id": rec.get("video_id"),
                        "frame_t": rec.get("frame_id"),
                        "frame_t1": rec.get("parent_frame_id"),
                        "image_id": rec.get("image_id"),
                        "parent_image_id": rec.get("parent_image_id"),
                        "bbox_t": rec.get("bbox_t"),
                        "bbox_t1": rec.get("parent_bbox_t1"),
                        "d_norm": float(d_norm_val) if d_norm_val is not None else 0.0,
                        "margin": float(margin_val)
                    })

    total_linked = sum(basis_counts.values())

    # Hard assertions on linkage invariants (D-3213, P-1503)
    assert total_linked == 57020, f"[FATAL] Expected 57,020 LINKED records, found {total_linked}"
    unexpected_bases = set(basis_counts) - {"MUTUAL_SINGLETON", "MARGIN_RESOLVED"}
    assert not unexpected_bases, f"[FATAL] Unexpected LINKED bases: {unexpected_bases}"

    n_singleton = basis_counts.get("MUTUAL_SINGLETON", 0)
    n_margin = basis_counts.get("MARGIN_RESOLVED", 0)
    assert n_singleton + n_margin == 57020, f"[FATAL] Sum of bases mismatch: {n_singleton} + {n_margin} != 57,020"
    assert len(margin_resolved_pool) == n_margin, f"[FATAL] Pool length mismatch: {len(margin_resolved_pool)} != {n_margin}"

    print(f"\n[+] Total Accepted LINKED Annotations Certified: {total_linked:,}")
    print(f"    - MUTUAL_SINGLETON: {n_singleton:6,d} ({n_singleton / total_linked * 100:5.2f}%)")
    print(f"    - MARGIN_RESOLVED:  {n_margin:6,d} ({n_margin / total_linked * 100:5.2f}%)")

    print("\n--- Breakdown by Category ---")
    print(f"{'Category':<18} | {'Total Linked':<14} | {'MUTUAL_SINGLETON':<20} | {'MARGIN_RESOLVED':<20}")
    print("-" * 75)
    for cname in sorted(cat_basis_counts.keys()):
        tot_c = sum(cat_basis_counts[cname].values())
        s_c = cat_basis_counts[cname].get("MUTUAL_SINGLETON", 0)
        m_c = cat_basis_counts[cname].get("MARGIN_RESOLVED", 0)
        print(f"{cname:<18} | {tot_c:<14,d} | {s_c:5,d} ({s_c/tot_c*100:5.1f}%)         | {m_c:5,d} ({m_c/tot_c*100:5.1f}%)")

    # --------------------------------------------------------------------------
    # TASK 2: GENERATE SUPPLEMENTARY VISUAL QA CROPS IF MARGIN_RESOLVED > 0
    # --------------------------------------------------------------------------
    if n_margin == 0:
        print("\n[*] MARGIN_RESOLVED count is exactly 0.")
        print("    100% of accepted links are MUTUAL_SINGLETON. No supplementary competition QA required.")
    else:
        print(f"\n[*] Generating Supplementary Visual QA for MARGIN_RESOLVED ({n_margin:,} candidate records)...")

        # Clean directory before rendering (D-3215, P-1505)
        if QA_SUPP_DIR.exists():
            for p in QA_SUPP_DIR.glob("*.png"):
                try:
                    p.unlink()
                except Exception:
                    pass
        else:
            QA_SUPP_DIR.mkdir(parents=True, exist_ok=True)

        # Subset A: 15 lowest stored effective-margin cases (D-3214, P-1504)
        sorted_by_margin = sorted(margin_resolved_pool, key=lambda x: x["margin"])
        low_margin_cases = sorted_by_margin[:15]
        low_margin_aids = {x["ann_id"] for x in low_margin_cases}

        # Subset B: 15 stratified cases across remaining pool
        remaining_pool = [x for x in margin_resolved_pool if x["ann_id"] not in low_margin_aids]
        rng = random.Random(RANDOM_SEED)

        buckets = defaultdict(list)
        for item in remaining_pool:
            buckets[(item["category"], item["video_id"])].append(item)

        stratified_cases = []
        b_keys = sorted(list(buckets.keys()))
        rng.shuffle(b_keys)

        while len(stratified_cases) < 15 and b_keys:
            for k in list(b_keys):
                if buckets[k]:
                    stratified_cases.append(buckets[k].pop(rng.randint(0, len(buckets[k]) - 1)))
                    if len(stratified_cases) == 15:
                        break
                else:
                    b_keys.remove(k)

        supp_sample = []
        for item in low_margin_cases:
            supp_sample.append((item, "LOW_MARGIN"))
        for item in stratified_cases:
            supp_sample.append((item, "STRATIFIED"))

        print(f"    Selected {len(supp_sample)} test cases: {len(low_margin_cases)} lowest effective-margin + {len(stratified_cases)} stratified.")

        coco = COCO(str(TRAIN_JSON))
        zip_ref = zipfile.ZipFile(ZIP_PATH, "r")
        manifest_rows = []

        def crop_with_box(img, box, color):
            W, H = img.size
            bx, by, bw, bh = box
            pad = max(40, int(max(bw, bh) * 1.5))
            x1, y1 = max(0, int(bx - pad)), max(0, int(by - pad))
            x2, y2 = min(W, int(bx + bw + pad)), min(H, int(by + bh + pad))
            crop = img.crop((x1, y1, x2, y2)).resize((240, 240), Image.BILINEAR)

            sx = 240.0 / (x2 - x1)
            sy = 240.0 / (y2 - y1)
            rx1 = (bx - x1) * sx
            ry1 = (by - y1) * sy
            rx2 = rx1 + bw * sx
            ry2 = ry1 + bh * sy

            draw = ImageDraw.Draw(crop)
            draw.rectangle([rx1, ry1, rx2, ry2], outline=color, width=3)
            return crop

        for idx, (item, sample_type) in enumerate(supp_sample):
            info_curr = coco.loadImgs([item["image_id"]])[0]
            info_prior = coco.loadImgs([item["parent_image_id"]])[0]

            img_curr = get_image_from_zip(zip_ref, info_curr["file_name"])
            img_prior = get_image_from_zip(zip_ref, info_prior["file_name"])

            assert img_curr is not None, f"[FATAL] Missing image in zip: {info_curr['file_name']}"
            assert img_prior is not None, f"[FATAL] Missing image in zip: {info_prior['file_name']}"

            crop_p = crop_with_box(img_prior, item["bbox_t1"], "red")
            crop_c = crop_with_box(img_curr, item["bbox_t"], "lime")

            comp = Image.new("RGB", (510, 280), (25, 25, 25))
            comp.paste(crop_p, (10, 30))
            comp.paste(crop_c, (260, 30))

            draw = ImageDraw.Draw(comp)
            disp_vid = f"video_{item['video_id']:02d}" if isinstance(item['video_id'], int) else str(item['video_id'])
            header = f"MARGIN_RESOLVED #{idx+1:02d} ({sample_type[:3]}) | {disp_vid} | {item['category']}"
            sub_header = f"d_norm={item['d_norm']:.4f} | margin={item['margin']:.4f}"
            draw.text((12, 6), header, fill=(240, 240, 240))
            draw.text((12, 18), sub_header, fill=(200, 200, 200))
            draw.text((15, 252), f"t-1 (Frame {item['frame_t1']})", fill=(255, 100, 100))
            draw.text((265, 252), f"t (Frame {item['frame_t']})", fill=(100, 255, 100))

            out_name = f"margin_resolved_{idx+1:02d}.png"
            out_file = QA_SUPP_DIR / out_name
            comp.save(out_file)

            manifest_rows.append({
                "sample_id": f"MARGIN_RESOLVED_{idx+1:02d}",
                "sample_type": sample_type,
                "video_id": item["video_id"],
                "category": item["category"],
                "ann_t": item["ann_id"],
                "ann_t1": item["parent_id"],
                "frame_t": item["frame_t"],
                "frame_t1": item["frame_t1"],
                "d_norm": item["d_norm"],
                "margin": item["margin"],
                "crop_file": str(out_file.resolve())
            })

        zip_ref.close()

        # Assert render completeness (D-3215, P-1505)
        assert len(manifest_rows) == len(supp_sample), (
            f"[FATAL] Rendered only {len(manifest_rows)}/{len(supp_sample)} selected QA cases"
        )

        with open(QA_SUPP_MANIFEST, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            w.writeheader()
            w.writerows(manifest_rows)

        print(f"[+] Certified: Rendered exactly {len(manifest_rows)}/{len(supp_sample)} supplementary QA composites to: {QA_SUPP_DIR}")
        print(f"[+] Manifest exported to: {QA_SUPP_MANIFEST}")

    # --------------------------------------------------------------------------
    # FINAL HASH VERIFICATION (D-3212: Guarantee Zero Artifact Mutation)
    # --------------------------------------------------------------------------
    final_hash = compute_sha256(LINKS_JSON)
    assert final_hash == CANDIDATE_SHA256, (
        f"[FATAL] Post-execution candidate hash corrupted!\n"
        f"  Expected: {CANDIDATE_SHA256}\n"
        f"  Found:    {final_hash}"
    )
    print("\n" + "=" * 115)
    print(f"[+] Hash Integrity Confirmed: {LINKS_JSON.name} is completely unaltered.")
    print(f"    Candidate SHA-256: {final_hash}")
    print("=" * 115)


if __name__ == "__main__":
    main()