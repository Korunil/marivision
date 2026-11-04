"""
D:/MARIVISION/H0_workbench/eval_tools/precompute_horizon_dynamics_bins.py
Pre-registers Horizon-Dynamics tertile boundaries on the frozen validation set.
- Calculates:
    c_t = (eta_L + eta_R) / 2      (Vertical elevation proxy)
    s_t = eta_R - eta_L            (Slope / roll proxy)
    |Delta c_t|, |Delta s_t|        (Instantaneous shocks)
- Enforces strict contiguity: never computes temporal deltas across g=0 frames or video boundaries
"""

import json
from pathlib import Path
import numpy as np

VAL_TRACKS = Path("D:/MARIVISION/H0_workbench/horizon_tracks/seaclips_horizon_tracks_val_semantic_v2.json")
OUT_BINS = Path("D:/MARIVISION/H0_workbench/horizon_tracks/val_horizon_dynamics_bins.json")


def main():
    with open(VAL_TRACKS, "r", encoding="utf-8") as f:
        tracks = json.load(f)

    delta_c_list = []
    delta_s_list = []
    frame_records = {}

    for video_key, v_info in sorted(tracks.items()):
        split_name, video_name = video_key.split("/", 1)
        frames_dict = v_info.get("frames", {})
        sorted_frame_names = sorted(frames_dict.keys())

        prev_c = None
        prev_s = None

        for fn in sorted_frame_names:
            eta_L, eta_R, g = frames_dict[fn]
            # full_key = f"{split_name}/{fn.replace('\\\\', '/').strip('/')}"
            fn = fn.replace("\\", "/").strip("/")
            full_key = f"{split_name}/{fn}"


            if g == 1.0:
                c_t = (eta_L + eta_R) / 2.0
                s_t = eta_R - eta_L

                if prev_c is not None and prev_s is not None:
                    d_c = abs(c_t - prev_c)
                    d_s = abs(s_t - prev_s)
                    delta_c_list.append(d_c)
                    delta_s_list.append(d_s)
                    frame_records[full_key] = {"c": c_t, "s": s_t, "d_c": d_c, "d_s": d_s, "valid_delta": True}
                else:
                    # Start of a new contiguous g=1 segment
                    frame_records[full_key] = {"c": c_t, "s": s_t, "d_c": 0.0, "d_s": 0.0, "valid_delta": False}

                prev_c = c_t
                prev_s = s_t
            else:
                # Reset continuity on unavailable frames
                prev_c = None
                prev_s = None
                frame_records[full_key] = {"c": 0.5, "s": 0.0, "d_c": 0.0, "d_s": 0.0, "valid_delta": False}

    d_c_arr = np.array(delta_c_list)
    d_s_arr = np.array(delta_s_list)

    # Compute tertile thresholds (33.33% and 66.67% quantiles)
    c_tertiles = [float(np.percentile(d_c_arr, 33.33)), float(np.percentile(d_c_arr, 66.67))]
    s_tertiles = [float(np.percentile(d_s_arr, 33.33)), float(np.percentile(d_s_arr, 66.67))]

    # Composite dynamic index: Euclidean norm of vertical and slope change
    composite_arr = np.sqrt(d_c_arr**2 + d_s_arr**2)
    comp_tertiles = [float(np.percentile(composite_arr, 33.33)), float(np.percentile(composite_arr, 66.67))]

    print("=" * 85)
    print("PRE-REGISTERED VALIDATION HORIZON DYNAMICS QUANTILE BREAKDOWN")
    print(f"Total contiguous delta samples: {len(d_c_arr):,}")
    print("-" * 85)
    print(f"Vertical Elevation Shock (|Delta c_t|):")
    print(f"  T1 (33.3%): {c_tertiles[0]:.6f} | T2 (66.7%): {c_tertiles[1]:.6f} | Max: {float(d_c_arr.max()):.6f}")
    print(f"Slope / Roll Shock (|Delta s_t|):")
    print(f"  T1 (33.3%): {s_tertiles[0]:.6f} | T2 (66.7%): {s_tertiles[1]:.6f} | Max: {float(d_s_arr.max()):.6f}")
    print(f"Composite Dynamic Magnitude (sqrt(d_c^2 + d_s^2)):")
    print(f"  T1 (33.3%): {comp_tertiles[0]:.6f} | T2 (66.7%): {comp_tertiles[1]:.6f} | Max: {float(composite_arr.max()):.6f}")
    print("=" * 85)

    out_data = {
        "num_valid_deltas": len(d_c_arr),
        "tertiles": {
            "delta_c": c_tertiles,
            "delta_s": s_tertiles,
            "composite": comp_tertiles
        },
        "frames": frame_records
    }

    with open(OUT_BINS, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    print(f"[+] Locked pre-registered dynamics ledger to: {OUT_BINS.name}")


if __name__ == "__main__":
    main()