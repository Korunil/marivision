# G4c ANIMAL anatomy and class routing (G4C.0)

VAL videos with ANIMAL GT in offsets 0 + 5: [9]

## Q4 routing (frozen rule) and plain vs tiled

| model | plain macro off0 | tiled macro off0 | tiled macro off5 | ANIMAL pooled plain | ANIMAL pooled tiled | SELF route macro off0 / off5 / ANIMAL pooled | R0 route macro off0 / off5 / ANIMAL pooled |
|---|---|---|---|---|---|---|---|
| R0_e15 | 0.3647 | 0.3459 | 0.3459 | 0.1160 | 0.0792 | 0.3585 / 0.3459 / 0.1160 | 0.3585 / 0.3459 / 0.1160 |
| G4F_e15 | 0.4301 | 0.5011 | 0.5157 | 0.0001 | 0.0005 | 0.5011 / 0.5156 / 0.0001 | 0.5248 / 0.5305 / 0.1160 |
| CTRL | 0.4099 | 0.4912 | 0.5134 | 0.0000 | 0.0005 | 0.4911 / 0.5132 / 0.0000 | 0.5149 / 0.5281 / 0.1160 |
| NOTILE | 0.4125 | 0.4781 | 0.4973 | 0.0001 | 0.0005 | 0.4781 / 0.4972 / 0.0001 | 0.5018 / 0.5120 / 0.1160 |
| RFS | 0.4155 | 0.4916 | 0.5090 | 0.0000 | 0.0005 | 0.4915 / 0.5088 / 0.0000 | 0.5153 / 0.5237 / 0.1160 |

Rule: for G4F_e15, SELF if its pooled ANIMAL AP >= 0.05, else R0 if >= 0.05, else none.  The other 5 classes are unchanged by construction (asserted).

**G5 INFERENCE ROUTING: R0**

## Q2 fate of the pooled ANIMAL GT (score >= 0.05, IoU >= 0.30)

| model | mode | FOUND_RIGHT_CLASS | MISSED | SEEN_LOW_SCORE |
|---|---|---|---|---|
| R0_e15 | plain | 6 | 46 | 0 |
| R0_e15 | tiled | 6 | 46 | 0 |
| G4F_e15 | plain | 1 | 51 | 0 |
| G4F_e15 | tiled | 4 | 46 | 2 |
| CTRL | plain | 1 | 51 | 0 |
| CTRL | tiled | 4 | 46 | 2 |
| NOTILE | plain | 1 | 51 | 0 |
| NOTILE | tiled | 4 | 46 | 2 |
| RFS | plain | 1 | 51 | 0 |
| RFS | tiled | 4 | 47 | 1 |

## Q1 anatomy of ANIMAL predictions >= 0.05 (offsets 0 + 5)

| model | mode | TP | duplicate | wrong class | on crowd | background | FP in ANIMAL videos | FP in other videos | FP >= 0.3 | tile-only FP | full-frame-backed FP | best TP score | FP above best TP | wrong class hits GT of |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0_e15 | plain | 6 | 5 | 213 | 0 | 53 | 10 | 261 | 3 | – | – | 0.852 | 0 | {"SHIP": 11, "SAILING_VESSEL": 3, "MARINE_MARKER": 145, "BOAT": 54} |
| R0_e15 | tiled | 6 | 4 | 682 | 0 | 289 | 40 | 935 | 57 | 531 | 444 | 0.886 | 0 | {"BOAT": 105, "SHIP": 11, "SAILING_VESSEL": 11, "MARINE_MARKER": 493, "OBJECT": 62} |
| G4F_e15 | plain | 1 | 0 | 782 | 0 | 1758 | 15 | 2525 | 162 | – | – | 0.140 | 307 | {"SHIP": 118, "SAILING_VESSEL": 8, "MARINE_MARKER": 518, "BOAT": 129, "OBJECT": 9} |
| G4F_e15 | tiled | 4 | 0 | 1147 | 0 | 1750 | 15 | 2882 | 368 | 639 | 2258 | 0.351 | 341 | {"SHIP": 130, "BOAT": 162, "SAILING_VESSEL": 13, "MARINE_MARKER": 833, "OBJECT": 9} |
| CTRL | plain | 1 | 0 | 730 | 0 | 1113 | 7 | 1836 | 179 | – | – | 0.105 | 462 | {"SHIP": 120, "BOAT": 115, "SAILING_VESSEL": 13, "MARINE_MARKER": 478, "OBJECT": 4} |
| CTRL | tiled | 4 | 0 | 1132 | 0 | 1122 | 7 | 2247 | 430 | 530 | 1724 | 0.444 | 351 | {"SHIP": 135, "BOAT": 157, "SAILING_VESSEL": 16, "MARINE_MARKER": 820, "OBJECT": 4} |
| NOTILE | plain | 1 | 0 | 723 | 0 | 1165 | 12 | 1876 | 179 | – | – | 0.116 | 387 | {"SHIP": 98, "SAILING_VESSEL": 6, "MARINE_MARKER": 507, "BOAT": 103, "OBJECT": 9} |
| NOTILE | tiled | 4 | 0 | 1078 | 0 | 1205 | 11 | 2272 | 386 | 561 | 1722 | 0.457 | 288 | {"SHIP": 107, "SAILING_VESSEL": 8, "MARINE_MARKER": 812, "BOAT": 144, "OBJECT": 7} |
| RFS | plain | 1 | 0 | 909 | 0 | 1482 | 9 | 2382 | 217 | – | – | 0.133 | 402 | {"SHIP": 165, "BOAT": 125, "SAILING_VESSEL": 2, "MARINE_MARKER": 616, "OBJECT": 1} |
| RFS | tiled | 4 | 0 | 1291 | 0 | 1670 | 12 | 2949 | 521 | 815 | 2146 | 0.544 | 350 | {"SHIP": 180, "BOAT": 180, "SAILING_VESSEL": 1, "MARINE_MARKER": 929, "OBJECT": 1} |

FP by video (tiled):

- R0_e15: {"v01": {"BACKGROUND": 39, "WRONG_CLASS": 10}, "v02": {"WRONG_CLASS": 2, "BACKGROUND": 4}, "v03": {"BACKGROUND": 96, "WRONG_CLASS": 389}, "v04": {"WRONG_CLASS": 74, "BACKGROUND": 33}, "v05": {"BACKGROUND": 3, "WRONG_CLASS": 19}, "v06": {"BACKGROUND": 46, "WRONG_CLASS": 110}, "v07": {"BACKGROUND": 32, "WRONG_CLASS": 75}, "v08": {"BACKGROUND": 3}, "v09": {"BACKGROUND": 33, "WRONG_CLASS": 3, "TP": 6, "DUPLICATE": 4}}
- G4F_e15: {"v01": {"BACKGROUND": 65, "WRONG_CLASS": 53}, "v02": {"BACKGROUND": 7, "WRONG_CLASS": 1}, "v03": {"BACKGROUND": 1160, "WRONG_CLASS": 726}, "v04": {"WRONG_CLASS": 95, "BACKGROUND": 164}, "v05": {"BACKGROUND": 85, "WRONG_CLASS": 92}, "v06": {"BACKGROUND": 23, "WRONG_CLASS": 49}, "v07": {"WRONG_CLASS": 122, "BACKGROUND": 231}, "v08": {"BACKGROUND": 9}, "v09": {"BACKGROUND": 6, "TP": 4, "WRONG_CLASS": 9}}
- CTRL: {"v01": {"BACKGROUND": 58, "WRONG_CLASS": 68}, "v02": {"BACKGROUND": 2, "WRONG_CLASS": 1}, "v03": {"WRONG_CLASS": 708, "BACKGROUND": 790}, "v04": {"WRONG_CLASS": 94, "BACKGROUND": 96}, "v05": {"BACKGROUND": 24, "WRONG_CLASS": 85}, "v06": {"WRONG_CLASS": 51, "BACKGROUND": 15}, "v07": {"WRONG_CLASS": 121, "BACKGROUND": 128}, "v08": {"BACKGROUND": 6}, "v09": {"TP": 4, "BACKGROUND": 3, "WRONG_CLASS": 4}}
- NOTILE: {"v01": {"BACKGROUND": 56, "WRONG_CLASS": 37}, "v02": {"BACKGROUND": 3, "WRONG_CLASS": 1}, "v03": {"BACKGROUND": 844, "WRONG_CLASS": 695}, "v04": {"WRONG_CLASS": 86, "BACKGROUND": 102}, "v05": {"BACKGROUND": 28, "WRONG_CLASS": 78}, "v06": {"BACKGROUND": 11, "WRONG_CLASS": 52}, "v07": {"WRONG_CLASS": 122, "BACKGROUND": 149}, "v08": {"BACKGROUND": 8}, "v09": {"WRONG_CLASS": 7, "TP": 4, "BACKGROUND": 4}}
- RFS: {"v01": {"BACKGROUND": 60, "WRONG_CLASS": 107}, "v02": {"WRONG_CLASS": 1, "BACKGROUND": 1}, "v03": {"BACKGROUND": 1094, "WRONG_CLASS": 790}, "v04": {"WRONG_CLASS": 104, "BACKGROUND": 184}, "v05": {"BACKGROUND": 90, "WRONG_CLASS": 82}, "v06": {"BACKGROUND": 12, "WRONG_CLASS": 58}, "v07": {"WRONG_CLASS": 148, "BACKGROUND": 211}, "v08": {"BACKGROUND": 7}, "v09": {"BACKGROUND": 11, "TP": 4, "WRONG_CLASS": 1}}

## Q3 ANIMAL size census (native px; network-input px at 640, full frame and Stage C tile)

| split | video | boxes | frames | small / medium / large | native sqrt(area) q10/50/90 | net full q10/50/90 | net tile q10/50/90 | share < 8 px native |
|---|---|---|---|---|---|---|---|---|
| train | v02 | 383 | 343 | 346 / 37 / 0 | 12.6 / 15.6 / 23.4 | 5.6 / 6.9 / 10.4 | 16.9 / 20.7 / 31.1 | 0.0026 |
| train | v09 | 191 | 191 | 191 / 0 / 0 | 2.8 / 3.5 / 3.9 | 1.7 / 2.1 / 2.3 | 5.0 / 6.2 / 6.9 | 1.0000 |
| train | v18 | 728 | 286 | 728 / 0 / 0 | 8.9 / 10.0 / 12.0 | 4.0 / 4.4 / 5.3 | 11.9 / 13.3 / 16.0 | 0.0206 |
| train | v21 | 391 | 391 | 391 / 0 / 0 | 4.0 / 4.9 / 6.9 | 2.4 / 2.9 / 4.1 | 7.1 / 8.7 / 12.3 | 1.0000 |
| train | v22 | 133 | 133 | 133 / 0 / 0 | 5.1 / 6.0 / 6.5 | 2.2 / 2.7 / 2.9 | 6.7 / 8.0 / 8.6 | 1.0000 |
| train | v42 | 530 | 282 | 513 / 17 / 0 | 8.8 / 13.9 / 19.3 | 5.3 / 8.3 / 11.5 | 15.7 / 24.6 / 34.4 | 0.0698 |
| train | ALL | 2356 | 1626 | 2302 / 54 / 0 | 4.0 / 10.0 / 16.9 | 2.4 / 4.7 / 8.9 | 7.1 / 14.0 / 26.6 | 0.3260 |
| val_full | v09 | 253 | 226 | 251 / 2 / 0 | 7.3 / 9.4 / 14.6 | 3.3 / 4.2 / 6.5 | 9.8 / 12.5 / 19.5 | 0.2174 |
| val_full | ALL | 253 | 226 | 251 / 2 / 0 | 7.3 / 9.4 / 14.6 | 3.3 / 4.2 / 6.5 | 9.8 / 12.5 / 19.5 | 0.2174 |
| val_off0_off5 | v09 | 52 | 46 | 52 / 0 / 0 | 7.5 / 9.4 / 16.4 | 3.3 / 4.2 / 7.3 | 10.0 / 12.6 / 21.9 | 0.2308 |
| val_off0_off5 | ALL | 52 | 46 | 52 / 0 / 0 | 7.5 / 9.4 / 16.4 | 3.3 / 4.2 / 7.3 | 10.0 / 12.6 / 21.9 | 0.2308 |
| TEST (public bins only) | – | 932 | – | 928 / 4 / 0 | – | – | – | – |
