"""
D:/MARIVISION/H0_workbench/eval_tools/build_temporal_pseudolinks.py
Phase-II Proof-of-Launch Stage 0 (PoL-0): Part 2 [Stage A] - GT Pseudo-Link Builder & QA

Strict Compliance Checklist:
  - D-3192: Authoritative video_id used for stream grouping; filename-derived vid retained only for display.
  - D-3193: Asserts frame_id uniqueness within every video stream.
  - D-3184: Strict De Morgan ambiguity logic (margin >= 0.008 AND margin >= 0.35 * d_best).
  - D-3196: Distinguishes MUTUAL_SINGLETON (margin: None) from MARGIN_RESOLVED (margin: float).
  - D-3186 / D-3187: Deterministic stratified visual QA (30 LINKED + 30 AMBIGUOUS) rendered to qa_crops/.
  - D-3189: Validates adjacent frame image dimensions before computing normalized metrics.
  - D-3178: Certified 4-state partition across all 65,542 annotations.
  - D-3194: Declares artifact as candidate awaiting QA approval (no M0 predictions touched).
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
QA_DIR = OUT_DIR / "qa_crops"
LINKS_JSON = OUT_DIR / "pol0_pseudo_links_v1.json"
QA_MANIFEST = OUT_DIR / "pol0_pseudo_link_qa_manifest.csv"

RANDOM_SEED = 42


def compute_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def get_video_id_from_path(file_name):
    clean = str(file_name).replace("\\", "/")
    for part in clean.split("/"):
        if part.startswith("video_"):
            return part
    return "unknown"


def box_iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[0] + b1[2], b2[0] + b2[2])
    y2 = min(b1[1] + b1[3], b2[1] + b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = b1[2] * b1[3] + b2[2] * b2[3] - inter
    return inter / union if union > 0.0 else 0.0


def box_center_dist_norm(b1, b2, W1, H1, W2, H2):
    c1 = np.array([(b1[0] + b1[2] / 2.0) / W1, (b1[1] + b1[3] / 2.0) / H1])
    c2 = np.array([(b2[0] + b2[2] / 2.0) / W2, (b2[1] + b2[3] / 2.0) / H2])
    return float(np.linalg.norm(c1 - c2))


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
    print("PROOF-OF-LAUNCH STAGE 0 (PoL-0): PART 2 [STAGE A] - GT PSEUDO-LINK BUILDER & QA")
    print(f"Dataset Target: {TRAIN_JSON.resolve()}")
    print("=" * 115)

    assert TRAIN_JSON.exists(), f"[FATAL] Missing {TRAIN_JSON}"
    assert ZIP_PATH.exists(), f"[FATAL] Missing {ZIP_PATH}"

    train_json_hash = compute_sha256(TRAIN_JSON)

    coco = COCO(str(TRAIN_JSON))
    img_ids = sorted(coco.getImgIds())
    ann_ids = sorted(coco.getAnnIds())
    cats = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}

    assert len(img_ids) == 20310, f"Expected 20,310 frames, found {len(img_ids)}"
    assert len(ann_ids) == 65542, f"Expected 65,542 annotations, found {len(ann_ids)}"

    # --------------------------------------------------------------------------
    # STEP 1: AUTHORITATIVE VIDEO & FRAME_ID CHRONOLOGY (D-3192, D-3193)
    # --------------------------------------------------------------------------
    video_streams = defaultdict(list)
    img_meta = {}

    for img_id in img_ids:
        info = coco.loadImgs([img_id])[0]
        fname = info["file_name"]
        
        # D-3192: Authoritative video_id from schema
        vid = int(info["video_id"])
        frame_id = int(info["frame_id"])
        W = float(info["width"])
        H = float(info["height"])

        img_meta[img_id] = {
            "video_id": vid, "frame_id": frame_id, "fname": fname,
            "display_vid": get_video_id_from_path(fname),
            "W": W, "H": H, "sensor": info.get("camera_sensor", "unknown")
        }
        video_streams[vid].append((frame_id, img_id))

    # D-3193: Assert frame_id uniqueness within every video stream
    for vid, frames in video_streams.items():
        frame_ids_list = [f[0] for f in frames]
        assert len(frame_ids_list) == len(set(frame_ids_list)), f"[FATAL] Duplicate frame_id detected in video {vid}!"

    anns_by_img = defaultdict(list)
    ann_lookup = {}
    for a in coco.loadAnns(ann_ids):
        anns_by_img[a["image_id"]].append(a)
        ann_lookup[a["id"]] = a

    print(f"[*] Chronology Indexed across {len(video_streams)} distinct videos (Total frames: {len(img_ids):,}).")
    print(f"    frame_id uniqueness asserted for all sequences.")

    # --------------------------------------------------------------------------
    # STEP 2: CONSTRUCT PSEUDO-LINKS WITH DE MORGAN AMBIGUITY LOGIC (D-3184, D-3196)
    # --------------------------------------------------------------------------
    print("\n[*] Constructing normalized pseudo-links (t -> t-1)...")

    link_results = {}
    partition_counts = {
        "NO_PRIOR_FRAME": 0,
        "NO_LINK_CANDIDATE": 0,
        "AMBIGUOUS": 0,
        "LINKED": 0
    }
    cat_partition_counts = defaultdict(lambda: defaultdict(int))

    linked_candidate_pool = []
    ambiguous_candidate_pool = []

    for vid, frames in sorted(video_streams.items()):
        frames_sorted = sorted(frames, key=lambda x: x[0])
        fid_to_iid = {f[0]: f[1] for f in frames_sorted}

        for frame_id, curr_iid in frames_sorted:
            prior_fid = frame_id - 1
            curr_anns = anns_by_img[curr_iid]

            # Case 1: Sequence Head / Boundary (D-3178)
            if prior_fid not in fid_to_iid:
                for a in curr_anns:
                    partition_counts["NO_PRIOR_FRAME"] += 1
                    cat_partition_counts[cats[a["category_id"]]]["NO_PRIOR_FRAME"] += 1
                    link_results[a["id"]] = {
                        "status": "NO_PRIOR_FRAME", "parent_id": None,
                        "d_norm": None, "margin": None, "link_basis": None
                    }
                continue

            prior_iid = fid_to_iid[prior_fid]
            prior_anns = anns_by_img[prior_iid]

            W_curr = img_meta[curr_iid]["W"]
            H_curr = img_meta[curr_iid]["H"]
            W_prior = img_meta[prior_iid]["W"]
            H_prior = img_meta[prior_iid]["H"]

            # D-3189: Assert identical dimensions between adjacent frames
            assert W_curr == W_prior and H_curr == H_prior, f"Dimension shift in video {vid}: ({W_curr}, {H_curr}) vs ({W_prior}, {H_prior})"

            if not prior_anns:
                for a in curr_anns:
                    partition_counts["NO_LINK_CANDIDATE"] += 1
                    cat_partition_counts[cats[a["category_id"]]]["NO_LINK_CANDIDATE"] += 1
                    link_results[a["id"]] = {
                        "status": "NO_LINK_CANDIDATE", "parent_id": None,
                        "d_norm": None, "margin": None, "link_basis": None
                    }
                continue

            # Process candidates strictly per category
            for cid in cats:
                cname = cats[cid]
                A_c = [a for a in curr_anns if a["category_id"] == cid]
                B_c = [b for b in prior_anns if b["category_id"] == cid]

                if not A_c:
                    continue
                if not B_c:
                    for a in A_c:
                        partition_counts["NO_LINK_CANDIDATE"] += 1
                        cat_partition_counts[cname]["NO_LINK_CANDIDATE"] += 1
                        link_results[a["id"]] = {
                            "status": "NO_LINK_CANDIDATE", "parent_id": None,
                            "d_norm": None, "margin": None, "link_basis": None
                        }
                    continue

                forward_cand = defaultdict(list)
                backward_cand = defaultdict(list)

                for a in A_c:
                    w_n = a["bbox"][2] / W_curr
                    h_n = a["bbox"][3] / H_curr
                    dim_n = max(w_n, h_n)

                    r_gate_norm = max(0.015, 1.5 * dim_n) if dim_n < 0.025 else max(0.025, 0.75 * dim_n)
                    area_a = max(1e-6, w_n * h_n)

                    for b in B_c:
                        w_nb = b["bbox"][2] / W_prior
                        h_nb = b["bbox"][3] / H_prior
                        area_b = max(1e-6, w_nb * h_nb)
                        r_area = max(area_a, area_b) / min(area_a, area_b)

                        d_norm = box_center_dist_norm(a["bbox"], b["bbox"], W_curr, H_curr, W_prior, H_prior)
                        iou = box_iou(a["bbox"], b["bbox"])

                        if (d_norm <= r_gate_norm or iou >= 0.15) and r_area <= 3.0:
                            forward_cand[a["id"]].append((d_norm, b["id"]))
                            backward_cand[b["id"]].append((d_norm, a["id"]))

                # Evaluate mutual uniqueness with De Morgan logic (D-3184, D-3196)
                for a in A_c:
                    aid = a["id"]
                    cands = forward_cand.get(aid, [])

                    if len(cands) == 0:
                        partition_counts["NO_LINK_CANDIDATE"] += 1
                        cat_partition_counts[cname]["NO_LINK_CANDIDATE"] += 1
                        link_results[aid] = {
                            "status": "NO_LINK_CANDIDATE", "parent_id": None,
                            "d_norm": None, "margin": None, "link_basis": None
                        }
                    elif len(cands) == 1:
                        d_best, bid = cands[0]
                        rev_cands = backward_cand.get(bid, [])

                        if len(rev_cands) == 1 and rev_cands[0][1] == aid:
                            # D-3196: Mutual singleton receives margin: None
                            partition_counts["LINKED"] += 1
                            cat_partition_counts[cname]["LINKED"] += 1
                            link_results[aid] = {
                                "status": "LINKED", "parent_id": bid,
                                "d_norm": float(d_best), "margin": None,
                                "link_basis": "MUTUAL_SINGLETON"
                            }
                            linked_candidate_pool.append((aid, bid, vid, cname, d_best, None, "MUTUAL_SINGLETON"))
                        elif len(rev_cands) > 1:
                            rev_cands.sort(key=lambda x: x[0])
                            best_rev_d = rev_cands[0][0]
                            margin_rev = rev_cands[1][0] - rev_cands[0][0]

                            # Strict De Morgan logic: BOTH absolute and relative margin must hold
                            margin_rev_ok = (margin_rev >= 0.008) and (margin_rev >= 0.35 * best_rev_d)

                            if rev_cands[0][1] == aid and margin_rev_ok:
                                partition_counts["LINKED"] += 1
                                cat_partition_counts[cname]["LINKED"] += 1
                                link_results[aid] = {
                                    "status": "LINKED", "parent_id": bid,
                                    "d_norm": float(d_best), "margin": float(margin_rev),
                                    "link_basis": "MARGIN_RESOLVED"
                                }
                                linked_candidate_pool.append((aid, bid, vid, cname, d_best, margin_rev, "MARGIN_RESOLVED"))
                            else:
                                partition_counts["AMBIGUOUS"] += 1
                                cat_partition_counts[cname]["AMBIGUOUS"] += 1
                                link_results[aid] = {
                                    "status": "AMBIGUOUS", "parent_id": None,
                                    "d_norm": float(d_best), "margin": float(margin_rev),
                                    "link_basis": "REVERSE_AMBIGUITY"
                                }
                                ambiguous_candidate_pool.append((aid, bid, vid, cname, d_best, margin_rev, "REVERSE_AMBIGUITY"))
                        else:
                            partition_counts["AMBIGUOUS"] += 1
                            cat_partition_counts[cname]["AMBIGUOUS"] += 1
                            link_results[aid] = {
                                "status": "AMBIGUOUS", "parent_id": None,
                                "d_norm": float(d_best), "margin": 0.0,
                                "link_basis": "MISSING_REVERSE"
                            }
                            ambiguous_candidate_pool.append((aid, bid, vid, cname, d_best, 0.0, "MISSING_REVERSE"))
                    else:
                        cands.sort(key=lambda x: x[0])
                        d_best, bid = cands[0]
                        margin_fwd = cands[1][0] - cands[0][0]

                        margin_fwd_ok = (margin_fwd >= 0.008) and (margin_fwd >= 0.35 * d_best)

                        if margin_fwd_ok:
                            rev_cands = backward_cand.get(bid, [])
                            if len(rev_cands) == 1 and rev_cands[0][1] == aid:
                                partition_counts["LINKED"] += 1
                                cat_partition_counts[cname]["LINKED"] += 1
                                link_results[aid] = {
                                    "status": "LINKED", "parent_id": bid,
                                    "d_norm": float(d_best), "margin": float(margin_fwd),
                                    "link_basis": "MARGIN_RESOLVED"
                                }
                                linked_candidate_pool.append((aid, bid, vid, cname, d_best, margin_fwd, "MARGIN_RESOLVED"))
                            elif len(rev_cands) > 1:
                                rev_cands.sort(key=lambda x: x[0])
                                best_rev_d = rev_cands[0][0]
                                margin_rev = rev_cands[1][0] - rev_cands[0][0]

                                margin_rev_ok = (margin_rev >= 0.008) and (margin_rev >= 0.35 * best_rev_d)

                                if rev_cands[0][1] == aid and margin_rev_ok:
                                    eff_margin = min(margin_fwd, margin_rev)
                                    partition_counts["LINKED"] += 1
                                    cat_partition_counts[cname]["LINKED"] += 1
                                    link_results[aid] = {
                                        "status": "LINKED", "parent_id": bid,
                                        "d_norm": float(d_best), "margin": float(eff_margin),
                                        "link_basis": "MARGIN_RESOLVED"
                                    }
                                    linked_candidate_pool.append((aid, bid, vid, cname, d_best, eff_margin, "MARGIN_RESOLVED"))
                                else:
                                    eff_margin = min(margin_fwd, margin_rev)
                                    partition_counts["AMBIGUOUS"] += 1
                                    cat_partition_counts[cname]["AMBIGUOUS"] += 1
                                    link_results[aid] = {
                                        "status": "AMBIGUOUS", "parent_id": None,
                                        "d_norm": float(d_best), "margin": float(eff_margin),
                                        "link_basis": "MUTUAL_COMPETITION"
                                    }
                                    ambiguous_candidate_pool.append((aid, bid, vid, cname, d_best, eff_margin, "MUTUAL_COMPETITION"))
                            else:
                                partition_counts["AMBIGUOUS"] += 1
                                cat_partition_counts[cname]["AMBIGUOUS"] += 1
                                link_results[aid] = {
                                    "status": "AMBIGUOUS", "parent_id": None,
                                    "d_norm": float(d_best), "margin": 0.0,
                                    "link_basis": "MISSING_REVERSE"
                                }
                                ambiguous_candidate_pool.append((aid, bid, vid, cname, d_best, 0.0, "MISSING_REVERSE"))
                        else:
                            partition_counts["AMBIGUOUS"] += 1
                            cat_partition_counts[cname]["AMBIGUOUS"] += 1
                            link_results[aid] = {
                                "status": "AMBIGUOUS", "parent_id": None,
                                "d_norm": float(d_best), "margin": float(margin_fwd),
                                "link_basis": "FORWARD_AMBIGUITY"
                            }
                            ambiguous_candidate_pool.append((aid, bid, vid, cname, d_best, margin_fwd, "FORWARD_AMBIGUITY"))

    # Assert exact 4-state partition across all 65,542 annotations (D-3178)
    sum_partition = sum(partition_counts.values())
    assert sum_partition == 65542, f"[FATAL] Partition mismatch: {sum_partition} != 65,542"
    assert len(link_results) == 65542, f"[FATAL] Link dictionary length mismatch: {len(link_results)}"

    print(f"\n[+] 4-State GT Pseudo-Linking Partition Certified (Sum = {sum_partition:,}):")
    print(f"    1. LINKED (Unambiguous 1-to-1): {partition_counts['LINKED']:6,d} ({partition_counts['LINKED']/65542*100:5.2f}%)")
    print(f"    2. NO_LINK_CANDIDATE:          {partition_counts['NO_LINK_CANDIDATE']:6,d} ({partition_counts['NO_LINK_CANDIDATE']/65542*100:5.2f}%)")
    print(f"    3. AMBIGUOUS (Strictly Excluded):{partition_counts['AMBIGUOUS']:6,d} ({partition_counts['AMBIGUOUS']/65542*100:5.2f}%)")
    print(f"    4. NO_PRIOR_FRAME (Boundary):   {partition_counts['NO_PRIOR_FRAME']:6,d} ({partition_counts['NO_PRIOR_FRAME']/65542*100:5.2f}%)")

    # Category breakdown diagnostic printout
    print("\n--- Pseudo-Link Partition Breakdown by Category ---")
    print(f"{'Category':<18} | {'Total Boxes':<12} | {'Linked (1-to-1)':<16} | {'No Candidate':<16} | {'Ambiguous':<14} | {'Boundary (No Prior)':<20}")
    print("-" * 105)
    for cname in sorted(cats.values()):
        sub_c = cat_partition_counts[cname]
        tot_c = sum(sub_c.values())
        print(f"{cname:<18} | {tot_c:<12,d} | {sub_c['LINKED']:5,d} ({sub_c['LINKED']/tot_c*100:4.1f}%)   | {sub_c['NO_LINK_CANDIDATE']:5,d} ({sub_c['NO_LINK_CANDIDATE']/tot_c*100:4.1f}%)   | {sub_c['AMBIGUOUS']:5,d} ({sub_c['AMBIGUOUS']/tot_c*100:4.1f}%) | {sub_c['NO_PRIOR_FRAME']:5,d} ({sub_c['NO_PRIOR_FRAME']/tot_c*100:4.1f}%)")

    # --------------------------------------------------------------------------
    # STEP 3: SERIALIZE CANDIDATE LINK ARTIFACT (D-3185, D-3194)
    # --------------------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    links_export = {}
    for aid, res in link_results.items():
        ann = ann_lookup[aid]
        iid = ann["image_id"]
        meta_t = img_meta[iid]

        parent_aid = res["parent_id"]
        parent_ann = ann_lookup[parent_aid] if parent_aid is not None else None
        parent_meta = img_meta[parent_ann["image_id"]] if parent_ann is not None else None

        links_export[str(aid)] = {
            "status": res["status"],
            "parent_id": parent_aid,
            "d_norm": res["d_norm"],
            "margin": res["margin"],
            "link_basis": res["link_basis"],
            "category_id": ann["category_id"],
            "category_name": cats[ann["category_id"]],
            "video_id": meta_t["video_id"],
            "frame_id": meta_t["frame_id"],
            "image_id": iid,
            "bbox_t": ann["bbox"],
            "parent_frame_id": parent_meta["frame_id"] if parent_meta else None,
            "parent_image_id": parent_ann["image_id"] if parent_ann else None,
            "parent_bbox_t1": parent_ann["bbox"] if parent_ann else None
        }

    candidate_link_payload = {
        "_meta": {
            "artifact_name": "pol0_pseudo_links_v1_candidate",
            "train_json": str(TRAIN_JSON.resolve()),
            "train_json_sha256": train_json_hash,
            "total_annotations": len(links_export),
            "partition_counts": partition_counts,
            "ambiguity_threshold_norm": 0.008,
            "relative_margin_threshold": 0.35,
            "status": "CANDIDATE_AWAITING_VISUAL_QA_APPROVAL"
        },
        "links": links_export
    }

    with open(LINKS_JSON, "w", encoding="utf-8") as f:
        json.dump(candidate_link_payload, f, indent=2)

    candidate_links_hash = compute_sha256(LINKS_JSON)

    # --------------------------------------------------------------------------
    # STEP 4: STRATIFIED VISUAL QA COMPOSITING (D-3186, D-3187, P-1490)
    # --------------------------------------------------------------------------
    print("\n[*] Generating Stratified Visual QA Composites (30 LINKED + 30 AMBIGUOUS)...")
    QA_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(RANDOM_SEED)

    def sample_stratified(pool, target_n):
        buckets = defaultdict(list)
        for item in pool:
            # Bucket by (category, video_id)
            buckets[(item[3], item[2])].append(item)

        sample = []
        keys = sorted(list(buckets.keys()))
        rng.shuffle(keys)

        while len(sample) < target_n and keys:
            for k in list(keys):
                if buckets[k]:
                    sample.append(buckets[k].pop(rng.randint(0, len(buckets[k]) - 1)))
                    if len(sample) == target_n:
                        break
                else:
                    keys.remove(k)
        return sample

    qa_linked = sample_stratified(linked_candidate_pool, 30)
    qa_ambiguous = sample_stratified(ambiguous_candidate_pool, 30)

    qa_manifest_rows = []

    try:
        zip_ref = zipfile.ZipFile(ZIP_PATH, "r")

        def render_crop_composite(item, idx, status_label):
            aid, bid, vid, cat, d_norm, margin, basis = item
            ann_curr = ann_lookup[aid]
            ann_prior = ann_lookup[bid]

            info_curr = coco.loadImgs([ann_curr["image_id"]])[0]
            info_prior = coco.loadImgs([ann_prior["image_id"]])[0]

            img_curr = get_image_from_zip(zip_ref, info_curr["file_name"])
            img_prior = get_image_from_zip(zip_ref, info_prior["file_name"])

            if img_curr is None or img_prior is None:
                return None

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

            crop_p = crop_with_box(img_prior, ann_prior["bbox"], "red")
            crop_c = crop_with_box(img_curr, ann_curr["bbox"], "lime")

            comp = Image.new("RGB", (510, 280), (25, 25, 25))
            comp.paste(crop_p, (10, 30))
            comp.paste(crop_c, (260, 30))

            draw = ImageDraw.Draw(comp)
            disp_vid = img_meta[ann_curr["image_id"]]["display_vid"]
            header = f"{status_label} #{idx+1:02d} | {disp_vid} | {cat} | d_norm={d_norm:.4f}"
            if margin is not None:
                header += f" | margin={margin:.4f}"
            else:
                header += " | margin=None (Singleton)"
            draw.text((12, 10), header, fill=(240, 240, 240))
            draw.text((15, 252), f"t-1 (Prior Frame {info_prior['frame_id']})", fill=(255, 100, 100))
            draw.text((265, 252), f"t (Current Frame {info_curr['frame_id']})", fill=(100, 255, 100))

            out_name = f"{status_label.lower()}_{idx+1:02d}.png"
            out_file = QA_DIR / out_name
            comp.save(out_file)

            return {
                "sample_id": f"{status_label}_{idx+1:02d}",
                "status": status_label, "basis": basis,
                "video_id": vid, "category": cat,
                "sensor": img_meta[ann_curr["image_id"]]["sensor"],
                "ann_t": aid, "ann_t1": bid,
                "frame_t": info_curr["frame_id"], "frame_t1": info_prior["frame_id"],
                "d_norm": d_norm, "margin": margin,
                "crop_file": str(out_file.resolve())
            }

        for idx, item in enumerate(qa_linked):
            res = render_crop_composite(item, idx, "LINKED")
            if res: qa_manifest_rows.append(res)

        for idx, item in enumerate(qa_ambiguous):
            res = render_crop_composite(item, idx, "AMBIGUOUS")
            if res: qa_manifest_rows.append(res)

        zip_ref.close()

        with open(QA_MANIFEST, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(qa_manifest_rows[0].keys()))
            w.writeheader()
            w.writerows(qa_manifest_rows)

        print(f"[+] Rendered {len(qa_manifest_rows)} side-by-side QA crop sheets to: {QA_DIR}")
        print(f"[+] Visual QA Manifest saved to: {QA_MANIFEST}")

    except Exception as e:
        print(f"[!] Warning: QA rendering encountered issue: {e}")

    print("\n" + "=" * 115)
    print("STAGE A COMPLETE: CANDIDATE PSEUDO-LINK ARTIFACT CREATED — AWAITING VISUAL QA / HASH APPROVAL")
    print(f"Candidate Link Artifact: {LINKS_JSON}")
    print(f"Candidate SHA-256:        {candidate_links_hash}")
    print("=" * 115)


if __name__ == "__main__":
    main()