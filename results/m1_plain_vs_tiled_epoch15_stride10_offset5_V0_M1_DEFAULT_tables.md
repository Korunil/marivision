# M1 plain vs tiled R0 - epoch15_stride10_offset5_V0_M1_DEFAULT

426 VAL frames, 1363 GT, checkpoint epoch 15 (sha 64f9b4dee2b7), pycocotools parity: PASSED

**Primary endpoint (AP30_mAP tiled - plain, native protocol (no cut, top-100)): NOT_MET** - pooled delta -0.0119, 95% CI [-0.01197, 0.08153], 5/9 videos positive (rule: ci95_low > 0 AND >= 6 videos positive)

| metric | plain native | tiled native | plain s>=0.05 | tiled s>=0.05 |
|---|---|---|---|---|
| AP30_mAP | 0.3578 | 0.3459 | 0.3351 | 0.316 |
| AP50_mAP | 0.2793 | 0.2915 | 0.2665 | 0.2713 |
| AP30_gt_weighted_mAP | 0.2704 | 0.372 | 0.229 | 0.2994 |
| AP50_gt_weighted_mAP | 0.1882 | 0.2929 | 0.1686 | 0.2446 |
| mAP_class_agnostic_50 | 0.2371 | 0.4096 | 0.231 | 0.4 |
| AP_S_at_50_mAP | 0.1332 | 0.1894 | 0.1231 | 0.17 |
| recall_1to1_iou30 | 0.504 | 0.675 | 0.3103 | 0.4439 |
| recall_1to1_iou50 | 0.361 | 0.5598 | 0.2465 | 0.3822 |
| fp_per_frame_1to1_iou50 | 98.85 | 98.21 | 3.51 | 6.58 |

| AP30 per class | plain native | tiled native |
|---|---|---|
| ANIMAL | 0.0891 | 0.0891 |
| BOAT | 0.3467 | 0.3412 |
| MARINE_MARKER | 0.129 | 0.3761 |
| OBJECT | 0.2126 | 0.2225 |
| SAILING_VESSEL | 0.7815 | 0.4938 |
| SHIP | 0.5877 | 0.5526 |

| recall@0.3 by area bin (640-scale) | plain native | tiled native |
|---|---|---|
| 01_LE_4px2 | 0.5139 | 0.625 |
| 02_4_to_6px2 | 0.0366 | 0.1463 |
| 03_6_to_9px2 | 0.136 | 0.368 |
| 04_9_to_25px2 | 0.3043 | 0.6008 |
| 05_25_to_150px2 | 0.805 | 0.9164 |
| 06_GT_150px2 | 0.8694 | 0.8612 |

| bootstrap tiled - plain | op | pooled delta | CI95 | videos > 0 | L2VO mean |
|---|---|---|---|---|---|
| AP30_mAP | native | -0.0119 | [-0.01197, 0.08153] | 5/9 | 0.00086 |
| AP50_mAP | native | 0.0122 | [0.00824, 0.12176] | 5/9 | 0.02233 |
| mAP_class_agnostic_50 | native | 0.1725 | [-0.01968, 0.16249] | 4/9 | 0.01261 |
| overall_gt_recall_30 | native | 0.171 | [0.00703, 0.13875] | 4/9 | 0.02186 |
| overall_gt_recall_50 | native | 0.1988 | [0.03417, 0.16828] | 5/9 | 0.05096 |
| AP30_mAP | s005 | -0.0191 | [0.00122, 0.09349] | 6/9 | 0.00954 |
| AP50_mAP | s005 | 0.0048 | [0.01273, 0.11935] | 6/9 | 0.02233 |
| mAP_class_agnostic_50 | s005 | 0.169 | [-0.01954, 0.16228] | 4/9 | 0.0126 |
| overall_gt_recall_30 | s005 | 0.1336 | [0.01917, 0.1341] | 6/9 | 0.03249 |
| overall_gt_recall_50 | s005 | 0.1357 | [0.03191, 0.1385] | 6/9 | 0.04867 |

Per-video AP30 deltas (native): {"v01": 0.0246, "v02": -0.02, "v03": 0.1122, "v04": -0.0573, "v05": 0.0075, "v06": 0.0808, "v07": 0.1764, "v08": 0.0, "v09": -0.0296}
