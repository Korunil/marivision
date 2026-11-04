# M1.1 merge variants - epoch04_stride10_offset0

| variant | macro AP30 plain | macro AP30 tiled | pooled delta | CI95 | videos > 0 | MARKER | SAILING_VESSEL | ANIMAL | SHIP | CA-AP50 | recall@0.3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V0_M1_DEFAULT | 0.4241 | 0.4344 | 0.0103 | [-0.01868, 0.08552] | 4/9 | 0.5136 | 0.4931 | 0.0805 | 0.8054 | 0.3714 | 0.7569 |
| (plain per class) | | | | | | 0.193 | 0.7342 | 0.0896 | 0.8398 | 0.1986 | 0.5924 |

Pre-declared development choice (highest pooled tiled macro AP30, ties -> lower number): **V0_M1_DEFAULT** (only meaningful on the development cache, offset 5).
