"""
D:/MARIVISION/H0_workbench/eval_tools/audit_dynamics_composition.py
Hardened Pre-M2 Dynamics Stratum Composition Audit
- Asserts exactly 2,396 valid contiguous validation delta frames.
- Asserts zero unmapped COCO images (hard failure on missing keys).
- Exports immutable artifact: val_horizon_dynamics_composition.json.
- Identifies top 20 frames by m_t for model-independent qualitative pre-selection.
"""

import json
from pathlib import Path
from collections import defaultdict
from pycocotools.coco import COCO

VAL_ANN = Path("D:/MARIVISION/seaclips-val.json")
BINS_JSON = Path("D:/MARIVISION/H0_workbench/horizon_tracks/val_horizon_dynamics_bins.json")
OUT_JSON = Path("D:/MARIVISION/H0_workbench/horizon_tracks/val_horizon_dynamics_composition.json")


def canonical_val_key(path_value):
    s = str(path_value).replace("\\", "/").strip("/")
    parts = [p for p in s.split("/") if p]
    lower = [p.lower() for p in parts]
    if "val" in lower:
        idx = len(lower) - 1 - lower[::-1].index("val")
        return "/".join(parts[idx:])
    if len(parts) >= 2:
        return "val/" + "/".join(parts[-2:])
    return "val/" + s


def main():
    assert VAL_ANN.exists(), f"Missing: {VAL_ANN}"
    assert BINS_JSON.exists(), f"Missing: {BINS_JSON}"

    coco = COCO(str(VAL_ANN))
    with open(BINS_JSON, "r", encoding="utf-8") as f:
        bins_data = json.load(f)

    tertiles = bins_data["tertiles"]["composite"]
    q33 = tertiles[0]  # 0.000361
    q67 = tertiles[1]  # 0.000943
    frame_records = bins_data["frames"]

    cat_map = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}

    # Canonical mapping of COCO image records
    img_name_to_id = {}
    for img_id in coco.getImgIds():
        img_info = coco.loadImgs([img_id])[0]
        key = canonical_val_key(img_info["file_name"])
        img_name_to_id[key] = img_id

    strata_stats = {
        "Low":    {"frames": 0, "boxes": 0, "classes": defaultdict(int), "scales": {"S": 0, "M": 0, "L": 0}},
        "Medium": {"frames": 0, "boxes": 0, "classes": defaultdict(int), "scales": {"S": 0, "M": 0, "L": 0}},
        "High":   {"frames": 0, "boxes": 0, "classes": defaultdict(int), "scales": {"S": 0, "M": 0, "L": 0}},
    }

    evaluated_count = 0
    missing_keys = []
    ranked_frames = []

    for full_key, rec in frame_records.items():
        if not rec.get("valid_delta", False):
            continue

        evaluated_count += 1
        d_c = rec["d_c"]
        d_s = rec["d_s"]
        m_t = (d_c**2 + d_s**2)**0.5

        key = canonical_val_key(full_key)
        img_id = img_name_to_id.get(key)
        if img_id is None:
            missing_keys.append((full_key, key))
            continue

        ranked_frames.append({"key": key, "m_t": m_t, "d_c": d_c, "d_s": d_s})

        if m_t <= q33:
            s_name = "Low"
        elif m_t <= q67:
            s_name = "Medium"
        else:
            s_name = "High"

        strata_stats[s_name]["frames"] += 1

        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            w, h = ann["bbox"][2], ann["bbox"][3]
            if w <= 0 or h <= 0:
                continue

            strata_stats[s_name]["boxes"] += 1
            cid = ann["category_id"]
            strata_stats[s_name]["classes"][cid] += 1

            area = ann.get("area", w * h)
            if area < 32**2:
                strata_stats[s_name]["scales"]["S"] += 1
            elif area <= 96**2:
                strata_stats[s_name]["scales"]["M"] += 1
            else:
                strata_stats[s_name]["scales"]["L"] += 1

    # Strict invariant assertions
    assert evaluated_count == 2396, f"Expected 2,396 valid dynamics frames, found {evaluated_count}"
    assert not missing_keys, f"{len(missing_keys)} frames could not be mapped to COCO. Examples: {missing_keys[:5]}"

    # Sort descending by m_t
    ranked_frames.sort(key=lambda x: x["m_t"], reverse=True)

    print("=" * 105)
    print(f"PRE-M2 DYNAMICS STRATUM COMPOSITION AUDIT ({evaluated_count:,} Contiguous Delta Frames)")
    print(f"Locked Tertiles: q33 = {q33:.6f} | q67 = {q67:.6f}")
    print("=" * 105)
    print(f"{'Stratum':<10} | {'Frames':<8} | {'GT Boxes':<10} | {'Small (<32^2)':<15} | {'Medium (32-96^2)':<17} | {'Large (>96^2)':<15}")
    print("-" * 105)
    for s_name in ["Low", "Medium", "High"]:
        st = strata_stats[s_name]
        print(f"{s_name:<10} | {st['frames']:<8} | {st['boxes']:<10} | {st['scales']['S']:<15} | {st['scales']['M']:<17} | {st['scales']['L']:<15}")
    print("-" * 105)

    print("\nPer-Category Instance Support across Dynamics Strata:")
    print(f"{'Category':<20} | {'Low Dynamics':<15} | {'Medium Dynamics':<17} | {'High Dynamics':<15} | {'Total':<10}")
    print("-" * 85)
    for cid, cname in sorted(cat_map.items()):
        cnt_low = strata_stats["Low"]["classes"][cid]
        cnt_med = strata_stats["Medium"]["classes"][cid]
        cnt_high = strata_stats["High"]["classes"][cid]
        tot = cnt_low + cnt_med + cnt_high
        print(f"{cname:<20} | {cnt_low:<15} | {cnt_med:<17} | {cnt_high:<15} | {tot:<10}")
    print("=" * 85)

    print("\nTop 20 Validation Frames Ranked Strictly by m_t (For Qualitative Pre-Selection):")
    print(f"{'Rank':<5} | {'Frame Key':<35} | {'m_t':<10} | {'|Delta c|':<10} | {'|Delta s|':<10}")
    print("-" * 80)
    for idx, rf in enumerate(ranked_frames[:20], 1):
        print(f"{idx:<5} | {rf['key']:<35} | {rf['m_t']:.6f}   | {rf['d_c']:.6f}   | {rf['d_s']:.6f}")
    print("=" * 80)

    # Persist immutable artifact
    export_payload = {
        "evaluated_frames": evaluated_count,
        "tertiles": {"q33": q33, "q67": q67},
        "strata": {
            s: {
                "frames": strata_stats[s]["frames"],
                "boxes": strata_stats[s]["boxes"],
                "scales": strata_stats[s]["scales"],
                "classes": {cat_map[cid]: cnt for cid, cnt in strata_stats[s]["classes"].items()}
            }
            for s in ["Low", "Medium", "High"]
        },
        "top_ranked_frames": ranked_frames[:50]
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)
    print(f"[+] Successfully exported audited composition ledger to: {OUT_JSON.name}")


if __name__ == "__main__":
    main()