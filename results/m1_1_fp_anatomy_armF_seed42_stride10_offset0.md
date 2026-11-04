# FP anatomy (score >= 0.05, 1-to-1 @ IoU 0.30) - armF_seed42_stride10_offset0

Frames 432; checkpoint epoch M2_armF_step10155. Tiled = V0 (M1 merge).

### Plain R0

| predicted class | TP | DUPLICATE | WRONG_CLASS | PART_OF_OBJECT | LOCALIZATION | BACKGROUND | FP total | FP >= median TP score (by bucket) |
|---|---|---|---|---|---|---|---|---|
| ANIMAL | 4 | 1 | 103 | 0 | 2 | 24 | 130 | - |
| BOAT | 73 | 2 | 139 | 2 | 1 | 63 | 207 | WRON 4, PART 1, BACK 1 |
| LEISURE_VEHICLE | 0 | 0 | 5 | 0 | 0 | 0 | 5 | - |
| MARINE_MARKER | 89 | 5 | 85 | 2 | 10 | 13 | 115 | DUPL 3, WRON 49, LOCA 2, BACK 4 |
| OBJECT | 62 | 0 | 206 | 3 | 10 | 301 | 520 | WRON 2, LOCA 4, BACK 2 |
| SAILING_VESSEL | 142 | 13 | 171 | 3 | 0 | 96 | 283 | WRON 30 |
| SHIP | 60 | 1 | 27 | 2 | 0 | 10 | 40 | - |

### Tiled R0 (V0)

| predicted class | TP | DUPLICATE | WRONG_CLASS | PART_OF_OBJECT | LOCALIZATION | BACKGROUND | FP total | FP >= median TP score (by bucket) |
|---|---|---|---|---|---|---|---|---|
| ANIMAL | 4 | 2 | 351 | 7 | 1 | 148 | 509 | BACK 2 |
| BOAT | 93 | 18 | 277 | 8 | 2 | 132 | 437 | DUPL 1, WRON 13, PART 1, BACK 12 |
| LEISURE_VEHICLE | 0 | 0 | 36 | 0 | 0 | 0 | 36 | - |
| MARINE_MARKER | 272 | 26 | 111 | 9 | 15 | 71 | 232 | DUPL 8, WRON 15, PART 1, BACK 12 |
| OBJECT | 66 | 1 | 332 | 7 | 14 | 309 | 663 | WRON 8, LOCA 1 |
| SAILING_VESSEL | 143 | 14 | 463 | 33 | 0 | 202 | 712 | WRON 74 |
| SHIP | 62 | 2 | 73 | 9 | 0 | 18 | 102 | WRON 1 |

### Tiled R0 (V0) by leading source stage (B = full frame, C = 640x360 tile)

| predicted class | TP | DUPLICATE | WRONG_CLASS | PART_OF_OBJECT | LOCALIZATION | BACKGROUND | FP total | FP >= median TP score (by bucket) |
|---|---|---|---|---|---|---|---|---|
| ANIMAL @B | 3 | 1 | 82 | 0 | 1 | 21 | 105 | - |
| ANIMAL @C | 1 | 1 | 269 | 7 | 0 | 127 | 404 | BACK 2 |
| BOAT @B | 50 | 1 | 132 | 2 | 1 | 59 | 195 | WRON 2, PART 1, BACK 1 |
| BOAT @C | 43 | 17 | 145 | 6 | 1 | 73 | 242 | DUPL 1, WRON 11, BACK 11 |
| LEISURE_VEHICLE @B | 0 | 0 | 5 | 0 | 0 | 0 | 5 | - |
| LEISURE_VEHICLE @C | 0 | 0 | 31 | 0 | 0 | 0 | 31 | - |
| MARINE_MARKER @B | 54 | 4 | 83 | 2 | 6 | 13 | 108 | DUPL 1, WRON 14, BACK 1 |
| MARINE_MARKER @C | 218 | 22 | 28 | 7 | 9 | 58 | 124 | DUPL 7, WRON 1, PART 1, BACK 11 |
| OBJECT @B | 66 | 0 | 174 | 3 | 6 | 212 | 395 | WRON 2, LOCA 1 |
| OBJECT @C | 0 | 1 | 158 | 4 | 8 | 97 | 268 | WRON 6 |
| SAILING_VESSEL @B | 121 | 11 | 133 | 2 | 0 | 89 | 235 | WRON 32 |
| SAILING_VESSEL @C | 22 | 3 | 330 | 31 | 0 | 113 | 477 | WRON 42 |
| SHIP @B | 59 | 1 | 26 | 2 | 0 | 9 | 38 | - |
| SHIP @C | 3 | 1 | 47 | 7 | 0 | 9 | 64 | WRON 1 |
