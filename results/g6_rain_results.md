# G6 robustness: rain (G6R.0) -- single scoring, all TEST frames

Corruption: {"corruption": "rain", "albumentations": "2.0.8", "params": {"p": 1.0, "rain_type": "drizzle", "drop_width": 10, "drop_length": 20, "drop_color": [100, 100, 100]}, "seed": "set_random_seed(seed_base + image id)", "seed_base": 1000000, "g6_version": "G6R.0", "g6_sha256": "954f0e85516849701e300da1cf62a9bea53ce6257b81993291dc79112eb1712f"}

| system | mAP@0.3 (corrupted) | clean (G5) | change |
|---|---|---|---|
| ours routed (G5 primary), 3 seeds | 20.34 ± 3.77 (16.13, 21.52, 23.38) | 35.29 | -14.95 |
| ours unrouted, 3 seeds | 22.17 ± 5.52 (16.28, 22.99, 27.23) | 38.16 | -15.99 |
| r0e15 plain (ours) | 5.66 | 20.72 | -15.05 |
| r0e04 plain (ours) | 7.26 | 19.04 | -11.78 |
| Def. DETR R50 (paper, mean of 3 seeds) | 5.52 | 22.86 | -17.34 |
| Def. DETR Swin-B (paper, mean of 3 seeds) | 23.71 | 31.28 | -7.57 |
| TransVOD++ Swin-B (paper, mean of 3 seeds) | 24.47 | 31.95 | -7.48 |

The paper publishes no SDs for the corrupted sets, so no significance test is made.

| model | variant | mAP@0.3 | small | medium | large | mAPca@0.3 | mAP@0.5 |
|---|---|---|---|---|---|---|---|
| seed42 | plain | 10.36 | 6.34 | 25.53 | 75.18 | 17.77 | 4.98 |
| seed42 | tiled | 16.28 | 12.51 | 32.47 | 80.11 | 12.58 | 10.96 |
| seed42 | routed | 16.13 | 12.10 | 24.02 | 80.11 | 12.41 | 10.20 |
| seed43 | plain | 12.86 | 6.80 | 31.93 | 78.64 | 19.92 | 8.06 |
| seed43 | tiled | 22.99 | 15.64 | 48.18 | 79.79 | 26.90 | 16.64 |
| seed43 | routed | 21.52 | 14.27 | 35.90 | 79.79 | 26.17 | 15.47 |
| seed44 | plain | 13.72 | 7.56 | 37.62 | 77.79 | 22.83 | 7.66 |
| seed44 | tiled | 27.23 | 20.28 | 52.74 | 79.09 | 32.68 | 19.18 |
| seed44 | routed | 23.38 | 16.42 | 42.59 | 79.09 | 32.02 | 16.08 |
| r0e15 | plain | 5.66 | 4.89 | 10.09 | 39.35 | 8.19 | 2.78 |
| r0e04 | plain | 7.26 | 5.33 | 12.75 | 21.28 | 6.86 | 3.66 |

| model | variant | ANIMAL | BOAT | LEISURE_VEHICLE | MARINE_MARKER | OBJECT | SAILING_VESSEL | SHIP |
|---|---|---|---|---|---|---|---|---|
| seed42 | plain | 2.05 | 25.37 | 2.25 | 6.37 | 1.21 | 16.07 | 19.20 |
| seed42 | tiled | 9.46 | 46.83 | 5.38 | 10.53 | 14.24 | 9.80 | 17.70 |
| seed42 | routed | 8.41 | 46.83 | 5.38 | 10.53 | 14.24 | 9.80 | 17.70 |
| seed43 | plain | 4.98 | 25.36 | 2.88 | 3.67 | 0.67 | 19.08 | 33.39 |
| seed43 | tiled | 18.69 | 48.57 | 24.19 | 8.56 | 6.48 | 24.47 | 29.99 |
| seed43 | routed | 8.41 | 48.57 | 24.19 | 8.56 | 6.48 | 24.47 | 29.99 |
| seed44 | plain | 8.48 | 24.03 | 2.94 | 6.93 | 0.98 | 23.96 | 28.73 |
| seed44 | tiled | 35.40 | 48.18 | 26.58 | 11.99 | 8.36 | 32.96 | 27.16 |
| seed44 | routed | 8.41 | 48.18 | 26.58 | 11.99 | 8.36 | 32.96 | 27.16 |
| r0e15 | plain | 8.41 | 12.85 | 7.34 | 0.95 | 1.26 | 6.30 | 2.51 |
| r0e04 | plain | 7.42 | 20.72 | 9.37 | 2.52 | 1.19 | 4.07 | 5.57 |
