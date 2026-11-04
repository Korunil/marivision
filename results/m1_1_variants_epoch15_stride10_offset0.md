# M1.1 merge variants - epoch15_stride10_offset0

| variant | macro AP30 plain | macro AP30 tiled | pooled delta | CI95 | videos > 0 | MARKER | SAILING_VESSEL | ANIMAL | SHIP | CA-AP50 | recall@0.3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V0_M1_DEFAULT | 0.3647 | 0.3459 | -0.0188 | [-0.00781, 0.08918] | 5/9 | 0.3766 | 0.5008 | 0.0673 | 0.545 | 0.4072 | 0.6753 |
| (plain per class) | | | | | | 0.1186 | 0.7611 | 0.1426 | 0.5802 | 0.224 | 0.4942 |

Pre-declared development choice (highest pooled tiled macro AP30, ties -> lower number): **V0_M1_DEFAULT** (only meaningful on the development cache, offset 5).
