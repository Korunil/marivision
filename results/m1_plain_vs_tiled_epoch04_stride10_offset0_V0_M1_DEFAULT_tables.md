# M1 plain vs tiled R0 - epoch04_stride10_offset0_V0_M1_DEFAULT

432 VAL frames, 1386 GT, checkpoint epoch 4 (sha 28e6288c3f67), pycocotools parity: PASSED

**Primary endpoint (AP30_mAP tiled - plain, native protocol (no cut, top-100)): NOT_MET** - pooled delta 0.0103, 95% CI [-0.01868, 0.08552], 4/9 videos positive (rule: ci95_low > 0 AND >= 6 videos positive)

| metric | plain native | tiled native | plain s>=0.05 | tiled s>=0.05 |
|---|---|---|---|---|
| AP30_mAP | 0.4241 | 0.4344 | 0.3801 | 0.3977 |
| AP50_mAP | 0.2977 | 0.3369 | 0.2593 | 0.3045 |
| AP30_gt_weighted_mAP | 0.3363 | 0.4846 | 0.3097 | 0.4652 |
| AP50_gt_weighted_mAP | 0.2115 | 0.3376 | 0.1912 | 0.3208 |
| mAP_class_agnostic_50 | 0.1986 | 0.3714 | 0.1974 | 0.3702 |
| AP_S_at_50_mAP | 0.1761 | 0.2321 | 0.1102 | 0.1886 |
| recall_1to1_iou30 | 0.5924 | 0.7569 | 0.4964 | 0.7049 |
| recall_1to1_iou50 | 0.4105 | 0.5909 | 0.3333 | 0.5491 |
| fp_per_frame_1to1_iou50 | 98.68 | 98.1 | 22.64 | 40.74 |

| AP30 per class | plain native | tiled native |
|---|---|---|
| ANIMAL | 0.0896 | 0.0805 |
| BOAT | 0.4522 | 0.4654 |
| MARINE_MARKER | 0.193 | 0.5136 |
| OBJECT | 0.236 | 0.2483 |
| SAILING_VESSEL | 0.7342 | 0.4931 |
| SHIP | 0.8398 | 0.8054 |

| recall@0.3 by area bin (640-scale) | plain native | tiled native |
|---|---|---|
| 01_LE_4px2 | 0.2927 | 0.378 |
| 02_4_to_6px2 | 0.0465 | 0.3953 |
| 03_6_to_9px2 | 0.3333 | 0.5659 |
| 04_9_to_25px2 | 0.4853 | 0.8212 |
| 05_25_to_150px2 | 0.8537 | 0.8836 |
| 06_GT_150px2 | 0.8857 | 0.8041 |

| bootstrap tiled - plain | op | pooled delta | CI95 | videos > 0 | L2VO mean |
|---|---|---|---|---|---|
| AP30_mAP | native | 0.0103 | [-0.01868, 0.08552] | 4/9 | -0.00764 |
| AP50_mAP | native | 0.0392 | [0.01219, 0.14788] | 6/9 | 0.0261 |
| mAP_class_agnostic_50 | native | 0.1728 | [-0.01409, 0.1761] | 5/9 | 0.0087 |
| overall_gt_recall_30 | native | 0.1645 | [-0.00293, 0.13834] | 4/9 | 0.00839 |
| overall_gt_recall_50 | native | 0.1804 | [-0.00714, 0.15878] | 5/9 | 0.025 |
| AP30_mAP | s005 | 0.0176 | [-0.01707, 0.08417] | 4/9 | -0.00821 |
| AP50_mAP | s005 | 0.0452 | [0.01459, 0.14073] | 6/9 | 0.0295 |
| mAP_class_agnostic_50 | s005 | 0.1728 | [-0.01214, 0.17746] | 5/9 | 0.00974 |
| overall_gt_recall_30 | s005 | 0.2085 | [0.00826, 0.17086] | 5/9 | 0.02342 |
| overall_gt_recall_50 | s005 | 0.2158 | [0.02776, 0.18953] | 5/9 | 0.05453 |

Per-video AP30 deltas (native): {"v01": 0.0294, "v02": -0.0465, "v03": 0.1578, "v04": -0.0374, "v05": 0.0656, "v06": -0.0386, "v07": 0.1737, "v08": 0.0, "v09": -0.026}
