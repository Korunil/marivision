# M1.1 merge variants - epoch15_stride10_offset5

| variant | macro AP30 plain | macro AP30 tiled | pooled delta | CI95 | videos > 0 | MARKER | SAILING_VESSEL | ANIMAL | SHIP | CA-AP50 | recall@0.3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V0_M1_DEFAULT | 0.3578 | 0.3459 | -0.0119 | [-0.01197, 0.08153] | 5/9 | 0.3761 | 0.4938 | 0.0891 | 0.5526 | 0.4096 | 0.675 |
| V1_SCALE_ROUTED | 0.3578 | 0.342 | -0.0158 | [-0.01431, 0.0804] | 5/9 | 0.3763 | 0.471 | 0.0891 | 0.5513 | 0.4078 | 0.675 |
| V2_PLAIN_PLUS_SMALL | 0.3578 | 0.3452 | -0.0126 | [-0.01087, 0.07271] | 4/9 | 0.3752 | 0.5333 | 0.0891 | 0.5525 | 0.3958 | 0.6676 |
| (plain per class) | | | | | | 0.129 | 0.7815 | 0.0891 | 0.5877 | 0.2371 | 0.504 |

Pre-declared development choice (highest pooled tiled macro AP30, ties -> lower number): **V0_M1_DEFAULT** (only meaningful on the development cache, offset 5).
