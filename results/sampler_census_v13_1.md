# MARIVISION-HR-P2 V13.1 Sampler Census (TRAIN only, CPU, no images decoded)

frame_stride=5 passes=6 bs=4 ga=4

Repeat factors (category_id -> r_c): {'1': 1.1176, '2': 1.0, '3': 1.0, '4': 1.0, '5': 1.8398, '6': 1.0, '7': 1.1613}

## Arm (b)

| Sampler | Tiles | Opt steps | ANIMAL GT | BOAT GT | MARINE_MARKER GT | OBJECT GT | LEISURE_VEHICLE GT | SAILING_VESSEL GT | SHIP GT | StageC MARKER+ANIMAL tiles |
|---|---|---|---|---|---|---|---|---|---|---|
| v5_size_only | 24396 | 1522 | 2828 | 7520 | 15131 | 18450 | 720 | 31868 | 2171 | 0 |

## Arm (d)

| Sampler | Tiles | Opt steps | ANIMAL GT | BOAT GT | MARINE_MARKER GT | OBJECT GT | LEISURE_VEHICLE GT | SAILING_VESSEL GT | SHIP GT | StageC MARKER+ANIMAL tiles |
|---|---|---|---|---|---|---|---|---|---|---|
| v5_size_only | 85951 | 5369 | 5091 | 10161 | 32257 | 47019 | 1018 | 61793 | 3774 | 232 |
| v12_class_balanced | 100135 | 6257 | 8975 (1.76x) | 18542 (1.82x) | 36029 (1.12x) | 48794 (1.04x) | 1457 (1.43x) | 70040 (1.13x) | 8358 (2.21x) | 527 |
| v13_repeat_factor | 90125 | 5631 | 5508 (1.08x) | 10906 (1.07x) | 34603 (1.07x) | 48604 (1.03x) | 1719 (1.69x) | 63087 (1.02x) | 7601 (2.01x) | 377 |

## Arm (e)

| Sampler | Tiles | Opt steps | ANIMAL GT | BOAT GT | MARINE_MARKER GT | OBJECT GT | LEISURE_VEHICLE GT | SAILING_VESSEL GT | SHIP GT | StageC MARKER+ANIMAL tiles |
|---|---|---|---|---|---|---|---|---|---|---|
| v5_size_only | 85951 | 5369 | 5091 | 10161 | 32257 | 47019 | 1018 | 61793 | 3774 | 232 |
| v12_class_balanced | 100135 | 6257 | 8975 (1.76x) | 18542 (1.82x) | 36029 (1.12x) | 48794 (1.04x) | 1457 (1.43x) | 70040 (1.13x) | 8358 (2.21x) | 527 |
| v13_repeat_factor | 90125 | 5631 | 5508 (1.08x) | 10906 (1.07x) | 34603 (1.07x) | 48604 (1.03x) | 1719 (1.69x) | 63087 (1.02x) | 7601 (2.01x) | 377 |

