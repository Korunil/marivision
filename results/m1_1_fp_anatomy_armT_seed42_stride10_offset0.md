# FP anatomy (score >= 0.05, 1-to-1 @ IoU 0.30) - armT_seed42_stride10_offset0

Frames 432; checkpoint epoch M2_armT_step10155. Tiled = V0 (M1 merge).

### Plain R0

| predicted class | TP | DUPLICATE | WRONG_CLASS | PART_OF_OBJECT | LOCALIZATION | BACKGROUND | FP total | FP >= median TP score (by bucket) |
|---|---|---|---|---|---|---|---|---|
| ANIMAL | 2 | 2 | 119 | 7 | 0 | 111 | 239 | WRON 11, BACK 1 |
| BOAT | 54 | 2 | 127 | 7 | 0 | 28 | 164 | WRON 13, PART 1, BACK 1 |
| LEISURE_VEHICLE | 0 | 0 | 49 | 0 | 0 | 0 | 49 | - |
| MARINE_MARKER | 190 | 39 | 199 | 78 | 18 | 126 | 460 | DUPL 11, WRON 72, PART 7, LOCA 5, BACK 13 |
| OBJECT | 95 | 43 | 637 | 95 | 93 | 1649 | 2517 | WRON 89, PART 15, LOCA 5, BACK 100 |
| SAILING_VESSEL | 127 | 42 | 74 | 14 | 0 | 47 | 177 | DUPL 4, WRON 22 |
| SHIP | 75 | 2 | 46 | 3 | 0 | 39 | 90 | - |

### Tiled R0 (V0)

| predicted class | TP | DUPLICATE | WRONG_CLASS | PART_OF_OBJECT | LOCALIZATION | BACKGROUND | FP total | FP >= median TP score (by bucket) |
|---|---|---|---|---|---|---|---|---|
| ANIMAL | 2 | 0 | 348 | 31 | 0 | 224 | 603 | WRON 52, BACK 8 |
| BOAT | 120 | 43 | 212 | 37 | 31 | 122 | 445 | DUPL 3, WRON 16, PART 1, BACK 5 |
| LEISURE_VEHICLE | 0 | 0 | 152 | 2 | 0 | 3 | 157 | - |
| MARINE_MARKER | 502 | 74 | 459 | 480 | 72 | 675 | 1760 | DUPL 12, WRON 67, PART 17, LOCA 7, BACK 49 |
| OBJECT | 100 | 18 | 1013 | 269 | 129 | 6193 | 7622 | WRON 95, PART 12, LOCA 6, BACK 93 |
| SAILING_VESSEL | 142 | 66 | 307 | 187 | 1 | 270 | 831 | WRON 43, BACK 2 |
| SHIP | 78 | 6 | 165 | 21 | 0 | 50 | 242 | WRON 1 |

### Tiled R0 (V0) by leading source stage (B = full frame, C = 640x360 tile)

| predicted class | TP | DUPLICATE | WRONG_CLASS | PART_OF_OBJECT | LOCALIZATION | BACKGROUND | FP total | FP >= median TP score (by bucket) |
|---|---|---|---|---|---|---|---|---|
| ANIMAL @B | 2 | 0 | 105 | 7 | 0 | 101 | 213 | WRON 14, BACK 1 |
| ANIMAL @C | 0 | 0 | 243 | 24 | 0 | 123 | 390 | WRON 38, BACK 7 |
| BOAT @B | 30 | 0 | 115 | 6 | 0 | 25 | 146 | WRON 10, PART 1, BACK 1 |
| BOAT @C | 90 | 43 | 97 | 31 | 31 | 97 | 299 | DUPL 3, WRON 6, BACK 4 |
| LEISURE_VEHICLE @B | 0 | 0 | 39 | 0 | 0 | 0 | 39 | - |
| LEISURE_VEHICLE @C | 0 | 0 | 113 | 2 | 0 | 3 | 118 | - |
| MARINE_MARKER @B | 81 | 12 | 134 | 67 | 10 | 111 | 334 | DUPL 1, WRON 29, PART 3, LOCA 1, BACK 4 |
| MARINE_MARKER @C | 421 | 62 | 325 | 413 | 62 | 564 | 1426 | DUPL 11, WRON 38, PART 14, LOCA 6, BACK 45 |
| OBJECT @B | 77 | 4 | 369 | 58 | 51 | 1013 | 1495 | WRON 72, PART 10, LOCA 3, BACK 61 |
| OBJECT @C | 23 | 14 | 644 | 211 | 78 | 5180 | 6127 | WRON 23, PART 2, LOCA 3, BACK 32 |
| SAILING_VESSEL @B | 106 | 41 | 63 | 6 | 0 | 41 | 151 | WRON 18 |
| SAILING_VESSEL @C | 36 | 25 | 244 | 181 | 1 | 229 | 680 | WRON 25, BACK 2 |
| SHIP @B | 74 | 0 | 43 | 3 | 0 | 39 | 85 | - |
| SHIP @C | 4 | 6 | 122 | 18 | 0 | 11 | 157 | WRON 1 |
