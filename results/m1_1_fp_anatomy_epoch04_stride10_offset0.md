# FP anatomy (score >= 0.05, 1-to-1 @ IoU 0.30) - epoch04_stride10_offset0

Frames 432; checkpoint epoch 4. Tiled = V0 (M1 merge).

### Plain R0

| predicted class | TP | DUPLICATE | WRONG_CLASS | PART_OF_OBJECT | LOCALIZATION | BACKGROUND | FP total | FP >= median TP score (by bucket) |
|---|---|---|---|---|---|---|---|---|
| ANIMAL | 4 | 4 | 551 | 14 | 3 | 919 | 1491 | BACK 1 |
| BOAT | 114 | 144 | 353 | 97 | 15 | 529 | 1138 | DUPL 2, WRON 4, PART 1, BACK 11 |
| LEISURE_VEHICLE | 0 | 0 | 10 | 0 | 0 | 1 | 11 | - |
| MARINE_MARKER | 240 | 163 | 276 | 126 | 124 | 536 | 1225 | DUPL 33, WRON 21, PART 9, LOCA 28, BACK 32 |
| OBJECT | 125 | 107 | 698 | 142 | 118 | 3408 | 4473 | WRON 7, PART 3, LOCA 32, BACK 74 |
| SAILING_VESSEL | 139 | 32 | 461 | 27 | 1 | 687 | 1208 | WRON 27, PART 1 |
| SHIP | 66 | 5 | 1 | 0 | 0 | 3 | 9 | - |

### Tiled R0 (V0)

| predicted class | TP | DUPLICATE | WRONG_CLASS | PART_OF_OBJECT | LOCALIZATION | BACKGROUND | FP total | FP >= median TP score (by bucket) |
|---|---|---|---|---|---|---|---|---|
| ANIMAL | 4 | 3 | 914 | 165 | 3 | 3764 | 4849 | BACK 1 |
| BOAT | 123 | 85 | 560 | 98 | 46 | 919 | 1708 | DUPL 2, WRON 15, PART 1, BACK 10 |
| LEISURE_VEHICLE | 0 | 0 | 144 | 2 | 0 | 14 | 160 | - |
| MARINE_MARKER | 518 | 73 | 295 | 142 | 159 | 1142 | 1811 | DUPL 1, WRON 13, PART 5, LOCA 21, BACK 30 |
| OBJECT | 123 | 29 | 1010 | 233 | 101 | 5105 | 6478 | DUPL 1, WRON 15, PART 2, LOCA 23, BACK 49 |
| SAILING_VESSEL | 143 | 42 | 872 | 135 | 6 | 1306 | 2361 | WRON 70 |
| SHIP | 66 | 0 | 13 | 2 | 0 | 2 | 17 | - |

### Tiled R0 (V0) by leading source stage (B = full frame, C = 640x360 tile)

| predicted class | TP | DUPLICATE | WRONG_CLASS | PART_OF_OBJECT | LOCALIZATION | BACKGROUND | FP total | FP >= median TP score (by bucket) |
|---|---|---|---|---|---|---|---|---|
| ANIMAL @B | 4 | 2 | 285 | 9 | 3 | 615 | 914 | BACK 1 |
| ANIMAL @C | 0 | 1 | 629 | 156 | 0 | 3149 | 3935 | - |
| BOAT @B | 84 | 11 | 267 | 66 | 8 | 364 | 716 | WRON 3, PART 1, BACK 7 |
| BOAT @C | 39 | 74 | 293 | 32 | 38 | 555 | 992 | DUPL 2, WRON 12, BACK 3 |
| LEISURE_VEHICLE @B | 0 | 0 | 6 | 0 | 0 | 1 | 7 | - |
| LEISURE_VEHICLE @C | 0 | 0 | 138 | 2 | 0 | 13 | 153 | - |
| MARINE_MARKER @B | 138 | 36 | 213 | 98 | 71 | 383 | 801 | DUPL 1, WRON 6, PART 1, LOCA 11, BACK 4 |
| MARINE_MARKER @C | 380 | 37 | 82 | 44 | 88 | 759 | 1010 | WRON 7, PART 4, LOCA 10, BACK 26 |
| OBJECT @B | 118 | 21 | 393 | 105 | 53 | 2072 | 2644 | DUPL 1, WRON 5, PART 2, LOCA 18, BACK 46 |
| OBJECT @C | 5 | 8 | 617 | 128 | 48 | 3033 | 3834 | WRON 10, LOCA 5, BACK 3 |
| SAILING_VESSEL @B | 110 | 24 | 188 | 22 | 1 | 467 | 702 | WRON 27 |
| SAILING_VESSEL @C | 33 | 18 | 684 | 113 | 5 | 839 | 1659 | WRON 43 |
| SHIP @B | 66 | 0 | 1 | 0 | 0 | 2 | 3 | - |
| SHIP @C | 0 | 0 | 12 | 2 | 0 | 0 | 14 | - |
