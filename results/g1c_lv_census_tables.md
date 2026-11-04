# G1c LV census (G1c.0)

TRAIN: 20310 images, 65542 boxes; LV boxes 600 in 2 videos

| Scale (COCO, native px) | TRAIN LV | TEST LV (public) |
|---|---|---|
| small | 0 | 9912 |
| medium | 600 | 2203 |
| large | 0 | 0 |

Percentiles 5 / 25 / 50 / 75 / 95:

- √area px: 45.4 / 47.9 / 58.9 / 78.5 / 88.3
- min side px: 39.0 / 41.0 / 44.0 / 50.0 / 60.0
- min side after 640 warp px: 13.0 / 13.7 / 14.7 / 16.7 / 20.0
- aspect w/h: 0.29 / 0.35 / 0.60 / 0.85 / 0.98
- shrink factor to reach small: 0.36 / 0.41 / 0.55 / 0.67 / 0.71
- min side once just small px: 17.3 / 19.0 / 24.6 / 29.5 / 31.7

| Video | boxes | frames w/ LV | frames | median √area | sensor | weather | sea | scale |
|---|---|---|---|---|---|---|---|---|
| video_18 | 300 | 300 | 300 | 47.9 | Axis | sunny | smooth | {"medium": 300} |
| video_23 | 300 | 300 | 300 | 78.5 | Axis | sunny | smooth | {"medium": 300} |

Co-occurring classes in LV frames: {"SAILING_VESSEL": 5375, "BOAT": 1803, "MARINE_MARKER": 5035, "ANIMAL": 728, "OBJECT": 11}
LV y-centre (5/50/95): {5: 0.4888888888888889, 50: 0.530787037037037, 95: 0.5671296296296297}
Reference classes: {"BOAT": {"n": 6261, "median_sqrt_area": 26.40075756488817, "small_share": 0.5789809934515253}, "SAILING_VESSEL": {"n": 26551, "median_sqrt_area": 24.331050121192877, "small_share": 0.565515423147904}, "OBJECT": {"n": 15364, "median_sqrt_area": 5.656854249492381, "small_share": 0.9876985160114553}}

Scale mix, all classes:

- ANIMAL: {"small": 2302, "medium": 54}
- BOAT: {"small": 3625, "large": 588, "medium": 2048}
- LEISURE_VEHICLE: {"medium": 600}
- MARINE_MARKER: {"small": 11895, "medium": 708}
- OBJECT: {"small": 15175, "medium": 189}
- SAILING_VESSEL: {"small": 15015, "large": 2607, "medium": 8929}
- SHIP: {"small": 905, "medium": 517, "large": 385}
