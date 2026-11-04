r"""diagnose_truncated_pass_tail.py — READ-ONLY, CPU-only (MARIVISION-HR-P2, 2026-10-27).

Question: did V13 Run 1's last optimizer steps (pass 6 cut at 1,127 of ~1,568 steps, pos_ema 19.8 vs ~7) see an
unrepresentative slice of TRAIN? It rebuilds the exact training schedules from the TRAIN COCO JSON (no images decoded,
nothing written) and reports, per window of optimizer steps, which videos and classes the detector was fed.

  A. V13 Run 1, pass 6: repeat-factor sampler, frame stride 3, epoch_offset 5, truncated at 1,127 steps (what actually ran)
  B. v5, pass 6: size-only sampler, frame stride 3, epoch_offset 5, full pass (what v5's final steps saw)
  C. V13.6 fix: the same truncated pass but with evenly spaced micro-batches

Run from eval_tools (imports train_marivision_hr_p2_arms.py from the same folder):
  python -W error diagnose_truncated_pass_tail.py --coco-train D:\MARIVISION\seaclips-train.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import train_marivision_hr_p2_arms as R


def describe_windows(sched, anns_by_image, vid_of, bs, ga, window_steps, label, tail_steps=10):
    tiles_per_step = bs * ga
    n_steps = len(sched) // tiles_per_step
    print(f"\n=== {label}: {n_steps} optimizer steps, {len(sched)} tiles ===")
    print(f"{'steps':>13} | {'videos':>6} | {'top video (share)':>20} | {'GT/tile':>7} | {'ANIMAL GT%':>10} | {'MARKER GT%':>10} | {'empty tiles%':>12}")
    rows = []
    for w0 in range(0, n_steps, window_steps):
        w1 = min(n_steps, w0 + window_steps)
        tiles = sched[w0 * tiles_per_step: w1 * tiles_per_step]
        vids = Counter(vid_of[int(iid)] for iid, _ in tiles)
        cls = Counter()
        n_gt = 0
        n_empty = 0
        for iid, cw in tiles:
            _, labels, _ = R.project_gt_boxes_into_crop_with_ignore(anns_by_image.get(int(iid), []), cw)
            n_gt += len(labels)
            if len(labels) == 0:
                n_empty += 1
            for lb in labels.tolist():
                cls[R.CLASS_NAMES[int(lb)]] += 1
        top_v, top_n = vids.most_common(1)[0]
        tot = max(1, n_gt)
        row = (w0 + 1, w1, len(vids), top_v, top_n / len(tiles), n_gt / len(tiles), cls["ANIMAL"] / tot, cls["MARINE_MARKER"] / tot, n_empty / len(tiles))
        rows.append(row)
        print(f"{row[0]:>5}-{row[1]:<7} | {row[2]:>6} | {('v' + str(row[3])):>9} ({row[4]:6.1%}) | {row[5]:7.2f} | {row[6]:10.1%} | {row[7]:10.1%} | {row[8]:12.1%}")
    last = sched[max(0, (n_steps - tail_steps)) * tiles_per_step: n_steps * tiles_per_step]
    gts = [len(R.project_gt_boxes_into_crop_with_ignore(anns_by_image.get(int(i), []), c)[1]) for i, c in last]
    vids = Counter(vid_of[int(i)] for i, _ in last)
    print(f"  last {tail_steps} steps: GT/tile={np.mean(gts):.2f}, videos={dict(vids.most_common(4))}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco-train", required=True)
    ap.add_argument("--arm", default="e")
    ap.add_argument("--frame-stride", type=int, default=3)
    ap.add_argument("--epoch-offset", type=int, default=5, help="Pass index (0-based) of the truncated pass; Run 1 pass 6 = 5")
    ap.add_argument("--truncate-steps", type=int, default=1127, help="Optimizer steps actually run in that pass")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--window-steps", type=int, default=100)
    a = ap.parse_args()
    if "test" in Path(a.coco_train).name.lower():
        raise SystemExit("[SPLIT BLOCK] TEST split is locked.")
    raw = json.loads(Path(a.coco_train).read_text(encoding="utf-8"))
    images = {int(im["id"]): im for im in raw["images"]}
    anns = defaultdict(list)
    for x in raw.get("annotations", []):
        if int(x.get("iscrowd", 0)) == 0:
            anns[int(x["image_id"])].append(x)
    vid_of = {iid: int(im.get("video_id", 0)) for iid, im in images.items()}
    tps = a.batch_size * a.grad_accum

    s13 = R.build_slicing_aided_training_schedule(images, anns, arm=a.arm, frame_stride=a.frame_stride,
                                                  epoch_offset=a.epoch_offset, interleave_across_videos=True,
                                                  repeat_factor_sampling=True, repeat_factor_threshold=0.10)
    s5 = R.build_slicing_aided_training_schedule(images, anns, arm=a.arm, frame_stride=a.frame_stride,
                                                 epoch_offset=a.epoch_offset, interleave_across_videos=True)
    print(f"[SCHEDULES] V13 pass: {len(s13) // tps} steps (truncated run used {a.truncate_steps}); v5 pass: {len(s5) // tps} steps")
    describe_windows(s13, anns, vid_of, a.batch_size, a.grad_accum, a.window_steps, "V13 full pass 6 (reference)")
    describe_windows(s13[: a.truncate_steps * tps], anns, vid_of, a.batch_size, a.grad_accum, a.window_steps,
                     f"A. V13 Run 1 pass 6 AS RUN (prefix, {a.truncate_steps} steps)")
    describe_windows(s5, anns, vid_of, a.batch_size, a.grad_accum, a.window_steps, "B. v5 pass 6 (full)")
    fixed = R.select_evenly_spaced_microbatches(s13, a.batch_size, a.truncate_steps * a.grad_accum)
    describe_windows(fixed, anns, vid_of, a.batch_size, a.grad_accum, a.window_steps,
                     f"C. V13.6 fix: evenly spaced ({a.truncate_steps} steps)")


if __name__ == "__main__":
    main()
