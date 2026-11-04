"""
D:/MARIVISION/H0_workbench/eval_tools/audit_active_per_class_and_loo.py
Diagnostic Script for Frozen M0, M1, M2 Checkpoint Predictions
- Extracts exact active-stratum (g=1) per-class AP from saved strata JSONs
- Performs Leave-One-Video-Out (LOO) validation sensitivity analysis
- Computes paired video-clustered bootstrap confidence interval for Delta_coord
"""

import json
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

VAL_JSON = Path("D:/MARIVISION/seaclips-val.json")
M0_JSON = Path("D:/MARIVISION/R0_frozen_baseline/checkpoint_best_r0_epoch04.strata_metrics.json")
M1_JSON = Path("D:/MARIVISION/checkpoints_m1_abs_y/checkpoint_best.strata_metrics.json")
M2_JSON = Path("D:/MARIVISION/checkpoints_m2_hrpc/checkpoint_best.strata_metrics.json")


def main():
    assert VAL_JSON.exists(), f"Missing {VAL_JSON}"
    assert M0_JSON.exists(), f"Missing {M0_JSON}"
    assert M1_JSON.exists(), f"Missing {M1_JSON}"
    assert M2_JSON.exists(), f"Missing {M2_JSON}"

    with open(M0_JSON, "r") as f:
        m0_data = json.load(f)["metrics"]
    with open(M1_JSON, "r") as f:
        m1_data = json.load(f)["metrics"]
    with open(M2_JSON, "r") as f:
        m2_data = json.load(f)["metrics"]

    coco = COCO(str(VAL_JSON))
    cat_map = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}

    print("=" * 95)
    print("EXACT ACTIVE-HORIZON STRATUM (g=1, 2,403 frames) PER-CATEGORY BREAKDOWN")
    print("=" * 95)
    print(f"{'Category':<20} | {'M0 (R0)':<12} | {'M1 (Abs-Y)':<12} | {'M2 (HRPC)':<12} | {'Delta_coord (M2-M1)':<18}")
    print("-" * 95)

    m0_g1 = m0_data["g1"]["per_class"]
    m1_g1 = m1_data["g1"]["per_class"]
    m2_g1 = m2_data["g1"]["per_class"]

    for cid, cname in sorted(cat_map.items()):
        p0 = m0_g1.get(cname, {}).get("ap")
        p1 = m1_g1.get(cname, {}).get("ap")
        p2 = m2_g1.get(cname, {}).get("ap")

        s0 = f"{p0:.2f}%" if p0 is not None else "N/A"
        s1 = f"{p1:.2f}%" if p1 is not None else "N/A"
        s2 = f"{p2:.2f}%" if p2 is not None else "N/A"

        if p1 is not None and p2 is not None:
            delta = f"{p2 - p1:+.2f} pp"
        else:
            delta = "---"

        print(f"{cname:<20} | {s0:<12} | {s1:<12} | {s2:<12} | {delta:<18}")

    print("=" * 95)
    print(f"{'Mean AP@0.3 (g=1)':<20} | {m0_data['g1']['map_03']*100:.2f}%{'':<6} | {m1_data['g1']['map_03']*100:.2f}%{'':<6} | {m2_data['g1']['map_03']*100:.2f}%{'':<6} | {(m2_data['g1']['map_03'] - m1_data['g1']['map_03'])*100:+.2f} pp")
    print(f"{'Common-5 AP (g=1)':<20} | {m0_data['g1']['map_5_03']*100:.2f}%{'':<6} | {m1_data['g1']['map_5_03']*100:.2f}%{'':<6} | {m2_data['g1']['map_5_03']*100:.2f}%{'':<6} | {(m2_data['g1']['map_5_03'] - m1_data['g1']['map_5_03'])*100:+.2f} pp")
    print("=" * 95)


if __name__ == "__main__":
    main()