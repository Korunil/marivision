# ANIMAL diagnosis (G4FU.0)

## 1. ANIMAL GT per video

| split | v02 | v09 | v18 | v21 | v22 | v42 |
|---|---|---|---|---|---|---|
| train (total 2356) | 383 | 191 | 728 | 391 | 133 | 530 |
| val_full (total 253) | 0 | 253 | 0 | 0 | 0 | 0 |
| val_off0 (total 27) | 0 | 27 | 0 | 0 | 0 | 0 |
| val_off5 (total 25) | 0 | 25 | 0 | 0 | 0 | 0 |

## 2. TRAIN tile-training exposure per class (M2 rule, 16 Stage C windows)

| class | TRAIN boxes | share full frames | share tile targets | kept windows / box | never kept in a window | dropped-as-background / box | median size at net input, full frame px | median size, tile px |
|---|---|---|---|---|---|---|---|---|
| ANIMAL | 2356 | 0.036 | 0.022 | 1.63 | 0.000 | 0.04 | 4.7 | 13.3 |
| BOAT | 6261 | 0.096 | 0.088 | 2.45 | 0.000 | 0.46 | 15.0 | 38.1 |
| MARINE_MARKER | 12603 | 0.192 | 0.172 | 2.36 | 0.000 | 0.10 | 3.7 | 11.2 |
| OBJECT | 15364 | 0.234 | 0.253 | 2.85 | 0.000 | 0.10 | 3.5 | 10.1 |
| LEISURE_VEHICLE | 600 | 0.009 | 0.005 | 1.52 | 0.000 | 1.44 | 26.2 | 64.6 |
| SAILING_VESSEL | 26551 | 0.405 | 0.430 | 2.81 | 0.000 | 0.66 | 14.3 | 42.1 |
| SHIP | 1807 | 0.028 | 0.031 | 2.94 | 0.000 | 0.50 | 12.8 | 36.3 |

## 3. Fate of the offset-0 ANIMAL GT boxes per model (score >= 0.05, IoU >= 0.30)

| model | mode | ANIMAL AP30 | FOUND_AS_BOAT | FOUND_AS_OBJECT | FOUND_RIGHT_CLASS | MISSED | SEEN_LOW_SCORE | ANIMAL preds >= 0.05 (score q25/50/75/max) | best ANIMAL score on ANIMAL GT (q25/50/75/max) |
|---|---|---|---|---|---|---|---|---|---|
| G4F_e15 (ConvNeXt-B + tiles, 15 ep, fp32) | plain | 0.0003 | 0 | 0 | 1 | 26 | 0 | 1313: 0.057 / 0.069 / 0.095 / max 0.840 | 0.000 / 0.000 / 0.000 / max 0.140 |
| G4F_e15 (ConvNeXt-B + tiles, 15 ep, fp32) | tiled | 0.0005 | 0 | 0 | 2 | 23 | 2 | 1474: 0.059 / 0.076 / 0.123 / max 0.874 | 0.000 / 0.000 / 0.000 / max 0.269 |
| R0_e15 (R50, 15 ep (R0)) | plain | 0.1426 | 0 | 0 | 4 | 23 | 0 | 147: 0.061 / 0.075 / 0.097 / max 0.774 | 0.000 / 0.000 / 0.000 / max 0.774 |
| R0_e15 (R50, 15 ep (R0)) | tiled | 0.0673 | 0 | 0 | 4 | 23 | 0 | 506: 0.063 / 0.085 / 0.145 / max 0.866 | 0.000 / 0.000 / 0.000 / max 0.828 |
| R0_e04 (R50, 4 ep (R0, VAL-selected)) | plain | 0.0896 | 0 | 0 | 4 | 23 | 0 | 1495: 0.059 / 0.072 / 0.101 / max 0.729 | 0.000 / 0.000 / 0.000 / max 0.729 |
| R0_e04 (R50, 4 ep (R0, VAL-selected)) | tiled | 0.0805 | 0 | 0 | 4 | 23 | 0 | 4853: 0.065 / 0.088 / 0.137 / max 0.729 | 0.000 / 0.000 / 0.000 / max 0.729 |
| M2_armF (R0e15 + 1 ep full frames) | plain | 0.1386 | 0 | 0 | 4 | 23 | 0 | 134: 0.062 / 0.079 / 0.098 / max 0.839 | 0.000 / 0.000 / 0.000 / max 0.839 |
| M2_armF (R0e15 + 1 ep full frames) | tiled | 0.0902 | 0 | 0 | 4 | 23 | 0 | 513: 0.063 / 0.086 / 0.143 / max 0.873 | 0.000 / 0.000 / 0.000 / max 0.873 |
| M2_armT (R0e15 + 1 ep with tiles) | plain | 0.0051 | 0 | 1 | 2 | 23 | 1 | 241: 0.055 / 0.060 / 0.071 / max 0.263 | 0.000 / 0.000 / 0.000 / max 0.106 |
| M2_armT (R0e15 + 1 ep with tiles) | tiled | 0.0013 | 0 | 1 | 2 | 24 | 0 | 605: 0.056 / 0.063 / 0.083 / max 0.315 | 0.000 / 0.000 / 0.000 / max 0.106 |
| P_R50_fp32 (pilot R50 fp32, no tiles) | plain | 0.0640 | 0 | 0 | 4 | 23 | 0 | 980: 0.060 / 0.076 / 0.111 / max 0.408 | 0.000 / 0.000 / 0.000 / max 0.408 |
| P_R50_fp32 (pilot R50 fp32, no tiles) | tiled | 0.0373 | 0 | 0 | 6 | 21 | 0 | 3830: 0.060 / 0.077 / 0.112 / max 0.443 | 0.000 / 0.000 / 0.000 / max 0.408 |
| P_R50_bf16 (pilot R50 bf16, no tiles) | plain | 0.0012 | 0 | 0 | 4 | 23 | 0 | 1019: 0.061 / 0.077 / 0.112 / max 0.406 | 0.000 / 0.000 / 0.000 / max 0.188 |
| P_R50_bf16 (pilot R50 bf16, no tiles) | tiled | 0.0003 | 0 | 0 | 4 | 23 | 0 | 3900: 0.064 / 0.084 / 0.126 / max 0.438 | 0.000 / 0.000 / 0.000 / max 0.194 |
| P_CNB (pilot ConvNeXt-B, no tiles) | plain | 0.0214 | 0 | 0 | 3 | 24 | 0 | 1111: 0.060 / 0.076 / 0.101 / max 0.404 | 0.000 / 0.000 / 0.000 / max 0.402 |
| P_CNB (pilot ConvNeXt-B, no tiles) | tiled | 0.0157 | 1 | 0 | 7 | 19 | 0 | 2377: 0.063 / 0.084 / 0.122 / max 0.471 | 0.000 / 0.000 / 0.033 / max 0.402 |
| P_CNBT (pilot ConvNeXt-B + tiles) | plain | 0.0010 | 0 | 0 | 3 | 23 | 1 | 4403: 0.058 / 0.071 / 0.094 / max 0.350 | 0.000 / 0.000 / 0.000 / max 0.189 |
| P_CNBT (pilot ConvNeXt-B + tiles) | tiled | 0.0008 | 0 | 0 | 3 | 24 | 0 | 5214: 0.059 / 0.072 / 0.097 / max 0.364 | 0.000 / 0.000 / 0.000 / max 0.196 |
