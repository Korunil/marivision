r"""m2_train_target_census.py - READ-ONLY, CPU-only, TRAIN annotations only, no images (MARIVISION-HR-P2, 2026-10-28).

Question: did M2 arm T's tile sampling change WHAT the detector was taught, class by class? Arm T over-predicts the
small classes (OBJECT, MARINE_MARKER, ANIMAL) on background and object parts. One explanation is a shift in the
training-target class mix: small objects survive inside 640x360 tiles, while large objects (SHIP, SAILING_VESSEL) are
often clipped below 50% visibility and dropped, or lie outside the sampled window.

It computes the EXPECTED training targets per sample, exactly as m2_train_r0_finetune.py builds them (its own
clip_boxes_to_window + pipeline windows; each of the 16 windows equally likely), for:
  arm F   whole frame every time
  arm T   0.5 x whole frame + 0.5 x mean over the 16 windows
and reports per class: expected boxes per sample, share of all targets, dropped-as-background boxes, and the share of
samples with no target at all.

Run from D:\MARIVISION\R0_frozen_baseline (imports m2_train_r0_finetune.py):
  python -W error m2_train_target_census.py --coco-train D:\MARIVISION\seaclips-train.json
Writes nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = ("ANIMAL", "BOAT", "MARINE_MARKER", "OBJECT", "LEISURE_VEHICLE", "SAILING_VESSEL", "SHIP")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco-train", default="D:/MARIVISION/seaclips-train.json")
    ap.add_argument("--pipeline-dir", default="D:/MARIVISION/H0_workbench/eval_tools")
    ap.add_argument("--p-tile", type=float, default=0.5)
    ap.add_argument("--min-visible", type=float, default=0.5)
    a = ap.parse_args()
    if "train" not in Path(a.coco_train).name.lower():
        raise SystemExit("[SPLIT BLOCK] TRAIN annotations only.")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import m2_train_r0_finetune as M2

    raw = json.loads(Path(a.coco_train).read_text(encoding="utf-8"))
    cat_ids = sorted(c["id"] for c in raw["categories"])
    c2l = {c: i for i, c in enumerate(cat_ids)}  # identical to SeaClipsDetection.cat_id_to_label
    anns = defaultdict(list)
    for x in raw["annotations"]:  # R0's dataset keeps every annotation (crowd included); so does this census
        anns[int(x["image_id"])].append(x)
    frame_cnt, tile_cnt, tile_drop = Counter(), Counter(), Counter()
    n_img, frame_empty, tile_empty = 0, 0, 0.0
    win_cache = {}
    for im in raw["images"]:
        iid, w, h = int(im["id"]), int(im["width"]), int(im["height"])
        n_img += 1
        a_img = [x for x in anns[iid] if x["bbox"][2] > 0 and x["bbox"][3] > 0]
        for x in a_img:
            frame_cnt[c2l[x["category_id"]]] += 1
        frame_empty += int(not a_img)
        if (w, h) not in win_cache:
            win_cache[(w, h)] = M2.stage_c_windows(a.pipeline_dir, w, h)
        wins = win_cache[(w, h)]
        for win in wins:
            _b, labels, _d = M2.clip_boxes_to_window(a_img, win, c2l, a.min_visible)
            for lb in labels:
                tile_cnt[lb] += 1.0 / len(wins)
            tile_empty += (1.0 if not labels else 0.0) / len(wins)
            # which classes were dropped (partially visible) in this window
            x0, y0, x1, y1 = win
            for x in a_img:
                bx, by, bw, bh = [float(v) for v in x["bbox"]]
                ix0, iy0, ix1, iy1 = max(bx, x0), max(by, y0), min(bx + bw, x1), min(by + bh, y1)
                if ix1 > ix0 and iy1 > iy0 and (ix1 - ix0) * (iy1 - iy0) < a.min_visible * bw * bh:
                    tile_drop[c2l[x["category_id"]]] += 1.0 / len(wins)
    p = a.p_tile
    armT = {k: (1 - p) * frame_cnt[k] + p * tile_cnt[k] for k in range(7)}
    totF, totT = sum(frame_cnt.values()), sum(armT.values())
    print(f"[CENSUS] TRAIN frames {n_img}; p_tile {p}; min_visible {a.min_visible}")
    print(f"{'class':>16} | {'F boxes/sample':>14} | {'F share':>7} | {'tile boxes/tile':>15} | {'T boxes/sample':>14} | "
          f"{'T share':>7} | {'T/F share':>9} | {'dropped/tile':>12}")
    for k in range(7):
        fs = frame_cnt[k] / max(1, totF)
        ts = armT[k] / max(1e-9, totT)
        print(f"{CLASSES[k]:>16} | {frame_cnt[k] / n_img:14.3f} | {fs:7.1%} | {tile_cnt[k] / n_img:15.3f} | "
              f"{armT[k] / n_img:14.3f} | {ts:7.1%} | {(ts / fs if fs else float('nan')):9.2f} | {tile_drop[k] / n_img:12.4f}")
    print(f"{'ALL':>16} | {totF / n_img:14.3f} | {'':7} | {sum(tile_cnt.values()) / n_img:15.3f} | {totT / n_img:14.3f}")
    print(f"[CENSUS] samples with no target: arm F {frame_empty / n_img:.1%} | tiles {tile_empty / n_img:.1%} | "
          f"arm T {(1 - p) * frame_empty / n_img + p * tile_empty / n_img:.1%}")


if __name__ == "__main__":
    main()
