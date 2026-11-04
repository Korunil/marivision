# M2 class diagnostic - seed42

432 VAL frames; GT outcome counts (score >= 0.05, IoU >= 0.30)

### GT class ANIMAL (n = 27)

| outcome | R0e15_plain | F_plain | F_tiled | T_plain | T_tiled |
|---|---|---|---|---|---|
| FOUND_RIGHT_CLASS | 4 | 4 | 4 | 2 | 2 |
| FOUND_AS_OBJECT | 0 | 0 | 0 | 1 | 1 |
| MISSED | 23 | 23 | 23 | 23 | 24 |
| SEEN_LOW_SCORE | 0 | 0 | 0 | 1 | 0 |

### GT class OBJECT (n = 186)

| outcome | R0e15_plain | F_plain | F_tiled | T_plain | T_tiled |
|---|---|---|---|---|---|
| FOUND_RIGHT_CLASS | 62 | 62 | 66 | 95 | 100 |
| FOUND_AS_MARINE_MARKER | 0 | 0 | 0 | 0 | 1 |
| FOUND_AS_SAILING_VESSEL | 0 | 0 | 3 | 0 | 1 |
| MISSED | 100 | 100 | 95 | 73 | 65 |
| SEEN_LOW_SCORE | 24 | 24 | 22 | 18 | 19 |

### GT class BOAT (n = 173)

| outcome | R0e15_plain | F_plain | F_tiled | T_plain | T_tiled |
|---|---|---|---|---|---|
| FOUND_RIGHT_CLASS | 82 | 73 | 93 | 54 | 120 |
| FOUND_AS_ANIMAL | 0 | 0 | 1 | 0 | 0 |
| FOUND_AS_MARINE_MARKER | 4 | 7 | 1 | 12 | 12 |
| FOUND_AS_OBJECT | 2 | 2 | 2 | 36 | 13 |
| FOUND_AS_SAILING_VESSEL | 0 | 2 | 1 | 1 | 3 |
| MISSED | 61 | 67 | 49 | 69 | 21 |
| SEEN_LOW_SCORE | 24 | 22 | 26 | 1 | 4 |

### GT class MARINE_MARKER (n = 747)

| outcome | R0e15_plain | F_plain | F_tiled | T_plain | T_tiled |
|---|---|---|---|---|---|
| FOUND_RIGHT_CLASS | 83 | 89 | 272 | 190 | 502 |
| FOUND_AS_ANIMAL | 2 | 2 | 18 | 1 | 18 |
| FOUND_AS_BOAT | 7 | 6 | 10 | 0 | 0 |
| FOUND_AS_OBJECT | 49 | 45 | 31 | 43 | 21 |
| FOUND_AS_SAILING_VESSEL | 49 | 50 | 126 | 34 | 19 |
| MISSED | 489 | 482 | 246 | 418 | 177 |
| SEEN_LOW_SCORE | 68 | 73 | 44 | 61 | 10 |

### GT class SAILING_VESSEL (n = 143)

| outcome | R0e15_plain | F_plain | F_tiled | T_plain | T_tiled |
|---|---|---|---|---|---|
| FOUND_RIGHT_CLASS | 142 | 142 | 143 | 127 | 142 |
| FOUND_AS_LEISURE_VEHICLE | 0 | 0 | 0 | 1 | 0 |
| MISSED | 1 | 1 | 0 | 13 | 0 |
| SEEN_LOW_SCORE | 0 | 0 | 0 | 2 | 1 |

### GT class SHIP (n = 110)

| outcome | R0e15_plain | F_plain | F_tiled | T_plain | T_tiled |
|---|---|---|---|---|---|
| FOUND_RIGHT_CLASS | 61 | 60 | 62 | 75 | 78 |
| FOUND_AS_ANIMAL | 0 | 1 | 1 | 0 | 0 |
| FOUND_AS_BOAT | 2 | 3 | 2 | 2 | 1 |
| FOUND_AS_OBJECT | 7 | 4 | 4 | 26 | 23 |
| FOUND_AS_SAILING_VESSEL | 1 | 0 | 0 | 0 | 0 |
| MISSED | 27 | 29 | 31 | 6 | 8 |
| SEEN_LOW_SCORE | 12 | 13 | 10 | 1 | 0 |

### Predictions with score >= 0.05, per predicted class: count and score quartiles (25/50/75/max)

| predicted class | R0e15_plain | F_plain | F_tiled | T_plain | T_tiled |
|---|---|---|---|---|---|
| ANIMAL | 147: 0.061 / 0.075 / 0.097 / max 0.774 | 134: 0.062 / 0.079 / 0.098 / max 0.839 | 513: 0.063 / 0.086 / 0.143 / max 0.873 | 241: 0.055 / 0.060 / 0.071 / max 0.263 | 605: 0.056 / 0.063 / 0.083 / max 0.315 |
| BOAT | 338: 0.066 / 0.091 / 0.214 / max 0.960 | 280: 0.061 / 0.086 / 0.186 / max 0.960 | 530: 0.063 / 0.090 / 0.197 / max 0.998 | 218: 0.065 / 0.095 / 0.278 / max 0.933 | 565: 0.069 / 0.109 / 0.275 / max 0.970 |
| MARINE_MARKER | 208: 0.058 / 0.077 / 0.169 / max 0.935 | 204: 0.063 / 0.083 / 0.170 / max 0.919 | 504: 0.074 / 0.133 / 0.454 / max 0.999 | 650: 0.060 / 0.080 / 0.119 / max 0.912 | 2262: 0.063 / 0.083 / 0.136 / max 0.988 |
| OBJECT | 726: 0.072 / 0.124 / 0.367 / max 0.960 | 582: 0.076 / 0.142 / 0.426 / max 0.960 | 729: 0.077 / 0.155 / 0.443 / max 0.999 | 2612: 0.064 / 0.095 / 0.224 / max 0.964 | 7722: 0.059 / 0.074 / 0.119 / max 0.999 |
| LEISURE_VEHICLE | 3: 0.055 / 0.058 / 0.058 / max 0.060 | 5: 0.053 / 0.054 / 0.055 / max 0.075 | 36: 0.055 / 0.062 / 0.077 / max 0.232 | 49: 0.062 / 0.075 / 0.093 / max 0.120 | 157: 0.061 / 0.073 / 0.113 / max 0.295 |
| SAILING_VESSEL | 374: 0.076 / 0.673 / 0.941 / max 0.988 | 425: 0.072 / 0.204 / 0.944 / max 0.991 | 855: 0.076 / 0.208 / 0.920 / max 0.999 | 304: 0.077 / 0.589 / 0.873 / max 0.978 | 973: 0.066 / 0.115 / 0.602 / max 0.999 |
| SHIP | 95: 0.080 / 0.321 / 0.760 / max 0.976 | 100: 0.073 / 0.376 / 0.874 / max 0.980 | 164: 0.065 / 0.125 / 0.690 / max 0.988 | 165: 0.061 / 0.091 / 0.604 / max 0.959 | 320: 0.058 / 0.079 / 0.167 / max 0.997 |

### ANIMAL and OBJECT GT boxes per video (evidence behind those two AP values)

| video | frames | ANIMAL GT | OBJECT GT |
|---|---|---|---|
| v01 | 61 | 0 | 0 |
| v02 | 61 | 0 | 0 |
| v03 | 91 | 0 | 5 |
| v04 | 30 | 0 | 0 |
| v05 | 49 | 0 | 49 |
| v06 | 46 | 0 | 46 |
| v07 | 30 | 0 | 0 |
| v08 | 34 | 0 | 26 |
| v09 | 30 | 27 | 60 |
