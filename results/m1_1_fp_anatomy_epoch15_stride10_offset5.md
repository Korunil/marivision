# FP anatomy (score >= 0.05, 1-to-1 @ IoU 0.30) - epoch15_stride10_offset5

Frames 426; checkpoint epoch 15. Tiled = V0 (M1 merge).

### Plain R0

| predicted class | TP | DUPLICATE | WRONG_CLASS | PART_OF_OBJECT | LOCALIZATION | BACKGROUND | FP total | FP >= median TP score (by bucket) |
|---|---|---|---|---|---|---|---|---|
| ANIMAL | 2 | 2 | 108 | 1 | 0 | 17 | 128 | - |
| BOAT | 78 | 5 | 181 | 4 | 0 | 76 | 266 | WRON 17, BACK 2 |
| LEISURE_VEHICLE | 0 | 0 | 2 | 0 | 0 | 0 | 2 | - |
| MARINE_MARKER | 81 | 4 | 83 | 0 | 10 | 21 | 118 | DUPL 2, WRON 35, LOCA 5, BACK 5 |
| OBJECT | 63 | 2 | 225 | 8 | 21 | 377 | 633 | LOCA 6, BACK 2 |
| SAILING_VESSEL | 139 | 16 | 134 | 3 | 0 | 74 | 227 | WRON 30 |
| SHIP | 60 | 5 | 23 | 1 | 0 | 4 | 33 | WRON 1 |

### Tiled R0 (V0)

| predicted class | TP | DUPLICATE | WRONG_CLASS | PART_OF_OBJECT | LOCALIZATION | BACKGROUND | FP total | FP >= median TP score (by bucket) |
|---|---|---|---|---|---|---|---|---|
| ANIMAL | 2 | 1 | 337 | 10 | 0 | 125 | 473 | - |
| BOAT | 94 | 27 | 324 | 14 | 4 | 137 | 506 | DUPL 4, WRON 22, LOCA 1, BACK 10 |
| LEISURE_VEHICLE | 0 | 0 | 47 | 0 | 0 | 2 | 49 | - |
| MARINE_MARKER | 242 | 18 | 119 | 2 | 19 | 71 | 229 | DUPL 4, WRON 19, LOCA 2, BACK 11 |
| OBJECT | 67 | 0 | 364 | 9 | 22 | 332 | 727 | WRON 6, LOCA 2 |
| SAILING_VESSEL | 140 | 16 | 415 | 34 | 0 | 181 | 646 | DUPL 1, WRON 84, BACK 1 |
| SHIP | 60 | 1 | 73 | 3 | 0 | 10 | 87 | WRON 2 |

### Tiled R0 (V0) by leading source stage (B = full frame, C = 640x360 tile)

| predicted class | TP | DUPLICATE | WRONG_CLASS | PART_OF_OBJECT | LOCALIZATION | BACKGROUND | FP total | FP >= median TP score (by bucket) |
|---|---|---|---|---|---|---|---|---|
| ANIMAL @B | 2 | 0 | 83 | 1 | 0 | 17 | 101 | - |
| ANIMAL @C | 0 | 1 | 254 | 9 | 0 | 108 | 372 | - |
| BOAT @B | 53 | 1 | 165 | 4 | 0 | 68 | 238 | WRON 5, BACK 1 |
| BOAT @C | 41 | 26 | 159 | 10 | 4 | 69 | 268 | DUPL 4, WRON 17, LOCA 1, BACK 9 |
| LEISURE_VEHICLE @B | 0 | 0 | 2 | 0 | 0 | 0 | 2 | - |
| LEISURE_VEHICLE @C | 0 | 0 | 45 | 0 | 0 | 2 | 47 | - |
| MARINE_MARKER @B | 57 | 2 | 84 | 0 | 6 | 21 | 113 | WRON 16, BACK 2 |
| MARINE_MARKER @C | 185 | 16 | 35 | 2 | 13 | 50 | 116 | DUPL 4, WRON 3, LOCA 2, BACK 9 |
| OBJECT @B | 66 | 0 | 182 | 7 | 16 | 239 | 444 | LOCA 2 |
| OBJECT @C | 1 | 0 | 182 | 2 | 6 | 93 | 283 | WRON 6 |
| SAILING_VESSEL @B | 110 | 15 | 103 | 3 | 0 | 70 | 191 | DUPL 1, WRON 31 |
| SAILING_VESSEL @C | 30 | 1 | 312 | 31 | 0 | 111 | 455 | WRON 53, BACK 1 |
| SHIP @B | 59 | 1 | 22 | 1 | 0 | 4 | 28 | WRON 1 |
| SHIP @C | 1 | 0 | 51 | 2 | 0 | 6 | 59 | WRON 1 |
