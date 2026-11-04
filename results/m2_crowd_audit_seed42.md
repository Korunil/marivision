# Crowd-annotation audit - seed42

### TRAIN: annotations by class

| class | non-crowd | crowd (iscrowd=1) | median crowd box area px^2 |
|---|---|---|---|
| ANIMAL | 2356 | 0 | - |
| BOAT | 6261 | 0 | - |
| MARINE_MARKER | 12603 | 0 | - |
| OBJECT | 15364 | 0 | - |
| LEISURE_VEHICLE | 600 | 0 | - |
| SAILING_VESSEL | 26551 | 0 | - |
| SHIP | 1807 | 0 | - |
| total | 65542 | 0 | |

### VAL: annotations by class

| class | non-crowd | crowd (iscrowd=1) | median crowd box area px^2 |
|---|---|---|---|
| ANIMAL | 253 | 0 | - |
| BOAT | 1725 | 0 | - |
| MARINE_MARKER | 7401 | 0 | - |
| OBJECT | 1835 | 0 | - |
| LEISURE_VEHICLE | 0 | 0 | - |
| SAILING_VESSEL | 1403 | 0 | - |
| SHIP | 1082 | 0 | - |
| total | 13699 | 0 | |

### VAL evaluated frames (432): crowd boxes per video

| video | crowd boxes | frames with crowd |
|---|---|---|
| v01 | 0 | 0 |
| v02 | 0 | 0 |
| v03 | 0 | 0 |
| v04 | 0 | 0 |
| v05 | 0 | 0 |
| v06 | 0 | 0 |
| v07 | 0 | 0 |
| v08 | 0 | 0 |
| v09 | 0 | 0 |

### Background FPs (score >= 0.05) lying on a VAL crowd box

| variant | background FPs | on a crowd box | share | of these: OBJECT | MARKER |
|---|---|---|---|---|---|
| R0e04_plain | 6083 | 0 | 0.0% | 0 | 0 |
| R0e04_tiled | 12252 | 0 | 0.0% | 0 | 0 |
| R0e15_plain | 616 | 0 | 0.0% | 0 | 0 |
| R0e15_tiled | 930 | 0 | 0.0% | 0 | 0 |
| F_plain | 507 | 0 | 0.0% | 0 | 0 |
| F_tiled | 880 | 0 | 0.0% | 0 | 0 |
| T_plain | 2000 | 0 | 0.0% | 0 | 0 |
| T_tiled | 7537 | 0 | 0.0% | 0 | 0 |

### pycocotools macro mAP@0.3 on the same frames: A = crowd dropped (HR-P2 protocol), B = crowd as ignore (standard COCOeval)

| variant | A macro | B macro | B - A | ANIMAL A/B | BOAT A/B | MARINE_MARKER A/B | OBJECT A/B | SAILING_VESSEL A/B | SHIP A/B |
|---|---|---|---|---|---|---|---|---|---|
| R0e04_plain | 0.4241 | 0.4241 | +0.0000 | 0.0896/0.0896 | 0.4522/0.4522 | 0.1928/0.1928 | 0.236/0.236 | 0.7342/0.7342 | 0.8399/0.8399 |
| R0e04_tiled | 0.4344 | 0.4344 | +0.0000 | 0.0805/0.0805 | 0.4654/0.4654 | 0.5136/0.5136 | 0.2483/0.2483 | 0.4931/0.4931 | 0.8054/0.8054 |
| R0e15_plain | 0.3647 | 0.3647 | +0.0000 | 0.1426/0.1426 | 0.3587/0.3587 | 0.1186/0.1186 | 0.2272/0.2272 | 0.7611/0.7611 | 0.5802/0.5802 |
| R0e15_tiled | 0.3459 | 0.3459 | +0.0000 | 0.0673/0.0673 | 0.3605/0.3605 | 0.3766/0.3766 | 0.2254/0.2254 | 0.5008/0.5008 | 0.545/0.545 |
| F_plain | 0.3657 | 0.3657 | +0.0000 | 0.1386/0.1386 | 0.3462/0.3462 | 0.1266/0.1266 | 0.2274/0.2274 | 0.7743/0.7743 | 0.5812/0.5812 |
| F_tiled | 0.3524 | 0.3524 | +0.0000 | 0.0902/0.0902 | 0.3458/0.3458 | 0.4026/0.4026 | 0.2239/0.2239 | 0.5016/0.5016 | 0.5501/0.5501 |
| T_plain | 0.314 | 0.314 | +0.0000 | 0.0051/0.0051 | 0.2577/0.2577 | 0.1632/0.1632 | 0.1081/0.1081 | 0.6947/0.6947 | 0.6554/0.6554 |
| T_tiled | 0.3797 | 0.3797 | +0.0000 | 0.0013/0.0013 | 0.4735/0.4735 | 0.4396/0.4396 | 0.1063/0.1063 | 0.6516/0.6516 | 0.606/0.606 |

Primary pair under B: T_tiled 0.3797 vs F_plain 0.3657 (delta +0.0140); under A: 0.3797 vs 0.3657.
