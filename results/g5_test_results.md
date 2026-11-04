# G5 TEST results (G5T.0) -- single evaluation, all 7,030 TEST frames

**Primary: routed mAP@0.3 over 3 seeds = 35.29 ± 1.09** (seeds 35.29, 36.39, 34.21)

| vs (paper, mean ± SD, n = 3) | paper | diff | Welch t | df | one-tailed p | significant at 0.05/3 |
|---|---|---|---|---|---|---|
| TransVOD++ Swin-B | 31.95 ± 3.28 | +3.34 | 1.676 | 2.44 | 0.1064 | no |
| Def. DETR Swin-B | 31.28 ± 3.72 | +4.01 | 1.794 | 2.34 | 0.0980 | no |
| Def. DETR R50 | 22.86 ± 0.91 | +12.43 | 15.144 | 3.87 | 0.0001 | YES |

Secondary, unrouted (G4F tiled only): 38.16 (seeds 36.59, 39.33, 38.56)

| model | variant | mAP@0.3 | small | medium | large | mAPca@0.3 | mAP@0.5 |
|---|---|---|---|---|---|---|---|
| seed42 | plain | 20.58 | 15.12 | 49.77 | 81.83 | 38.94 | 10.78 |
| seed42 | tiled | 36.59 | 31.22 | 63.36 | 85.50 | 57.70 | 25.27 |
| seed42 | routed | 35.29 | 29.96 | 63.36 | 85.50 | 57.62 | 22.08 |
| seed43 | plain | 25.45 | 16.38 | 60.65 | 90.33 | 40.15 | 16.44 |
| seed43 | tiled | 39.33 | 30.16 | 71.12 | 90.57 | 56.42 | 28.88 |
| seed43 | routed | 36.39 | 27.23 | 71.12 | 90.57 | 56.12 | 24.83 |
| seed44 | plain | 25.32 | 19.25 | 57.51 | 83.55 | 42.63 | 12.83 |
| seed44 | tiled | 38.56 | 31.92 | 71.44 | 84.42 | 58.05 | 26.09 |
| seed44 | routed | 34.21 | 27.56 | 71.44 | 84.42 | 57.65 | 22.03 |
| r0e15 | plain | 20.72 | 16.79 | 49.83 | 52.02 | 39.73 | 10.14 |
| r0e04 | plain | 19.04 | 14.85 | 30.30 | 51.76 | 37.23 | 11.36 |

| model | variant | ANIMAL | BOAT | LEISURE_VEHICLE | MARINE_MARKER | OBJECT | SAILING_VESSEL | SHIP |
|---|---|---|---|---|---|---|---|---|
| seed42 | plain | 8.54 | 47.42 | 2.12 | 11.34 | 12.28 | 42.87 | 19.47 |
| seed42 | tiled | 39.30 | 64.33 | 21.14 | 19.63 | 28.92 | 65.23 | 17.56 |
| seed42 | routed | 30.20 | 64.33 | 21.14 | 19.63 | 28.92 | 65.23 | 17.56 |
| seed43 | plain | 23.32 | 43.92 | 5.43 | 6.84 | 7.86 | 44.76 | 46.02 |
| seed43 | tiled | 50.75 | 64.89 | 24.48 | 13.70 | 17.94 | 61.20 | 42.34 |
| seed43 | routed | 30.20 | 64.89 | 24.48 | 13.70 | 17.94 | 61.20 | 42.34 |
| seed44 | plain | 37.46 | 44.77 | 4.60 | 13.20 | 10.98 | 41.64 | 24.61 |
| seed44 | tiled | 60.65 | 63.31 | 29.29 | 17.56 | 18.93 | 57.29 | 22.86 |
| seed44 | routed | 30.20 | 63.31 | 29.29 | 17.56 | 18.93 | 57.29 | 22.86 |
| r0e15 | plain | 30.20 | 40.43 | 12.42 | 5.92 | 10.72 | 38.44 | 6.89 |
| r0e04 | plain | 12.79 | 44.24 | 11.52 | 6.63 | 10.92 | 34.42 | 12.78 |

Export throughput (s/frame, RTX 3060, includes image decode): {"seed42": 1.9234605224508987, "seed43": 1.943289608060042, "seed44": 1.9744473408499619, "r0e15": 0.12884373159530663, "r0e04": 0.11260388833530258, "system_routed_per_frame_est": {"seed42": 2.0523042540462053, "seed43": 2.0721333396553487, "seed44": 2.1032910724452685}}

Inputs: {"seed42": {"ckpt_sha256": "b393a1a57b399aa5be4d92099f11c12641b40e1f926884d0f12424469f52fef6", "n_frames": 7030}, "seed43": {"ckpt_sha256": "2211b3062fa693658cb50d2d6e2e6624c8eb1f3d43e98fa2f7212a3cc7ac2871", "n_frames": 7030}, "seed44": {"ckpt_sha256": "44bbd0e27bb272f52bdce64f216c47c084466f5928ce0de6006a299fe1bcd463", "n_frames": 7030}, "r0e15": {"ckpt_sha256": "64f9b4dee2b7a4875fc018e12877efdd3fb1e1d0f3f0a3b77ec8302b1cf9dac0", "n_frames": 7030}, "r0e04": {"ckpt_sha256": "9e4abee6b3948d6ef54ad5864820985e0072146702d7ba1a003a60ac1f76ed4f", "n_frames": 7030}}
