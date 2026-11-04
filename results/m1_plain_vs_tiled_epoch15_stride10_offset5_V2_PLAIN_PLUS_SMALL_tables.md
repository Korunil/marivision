# M1 plain vs tiled R0 - epoch15_stride10_offset5_V2_PLAIN_PLUS_SMALL

426 VAL frames, 1363 GT, checkpoint epoch 15 (sha 64f9b4dee2b7), pycocotools parity: PASSED

**Primary endpoint (AP30_mAP tiled - plain, native protocol (no cut, top-100)): NOT_MET** - pooled delta -0.0126, 95% CI [-0.01087, 0.07271], 4/9 videos positive (rule: ci95_low > 0 AND >= 6 videos positive)

| metric | plain native | tiled native | plain s>=0.05 | tiled s>=0.05 |
|---|---|---|---|---|
| AP30_mAP | 0.3578 | 0.3452 | 0.3351 | 0.3159 |
| AP50_mAP | 0.2793 | 0.2866 | 0.2665 | 0.2676 |
| AP30_gt_weighted_mAP | 0.2704 | 0.3698 | 0.229 | 0.2986 |
| AP50_gt_weighted_mAP | 0.1882 | 0.2791 | 0.1686 | 0.2337 |
| mAP_class_agnostic_50 | 0.2371 | 0.3958 | 0.231 | 0.3848 |
| AP_S_at_50_mAP | 0.1332 | 0.1686 | 0.1231 | 0.1511 |
| recall_1to1_iou30 | 0.504 | 0.6676 | 0.3103 | 0.438 |
| recall_1to1_iou50 | 0.361 | 0.5334 | 0.2465 | 0.3639 |
| fp_per_frame_1to1_iou50 | 98.85 | 98.29 | 3.51 | 6.65 |

| AP30 per class | plain native | tiled native |
|---|---|---|
| ANIMAL | 0.0891 | 0.0891 |
| BOAT | 0.3467 | 0.3431 |
| MARINE_MARKER | 0.129 | 0.3752 |
| OBJECT | 0.2126 | 0.1777 |
| SAILING_VESSEL | 0.7815 | 0.5333 |
| SHIP | 0.5877 | 0.5525 |

| recall@0.3 by area bin (640-scale) | plain native | tiled native |
|---|---|---|
| 01_LE_4px2 | 0.5139 | 0.5694 |
| 02_4_to_6px2 | 0.0366 | 0.1463 |
| 03_6_to_9px2 | 0.136 | 0.384 |
| 04_9_to_25px2 | 0.3043 | 0.5891 |
| 05_25_to_150px2 | 0.805 | 0.9102 |
| 06_GT_150px2 | 0.8694 | 0.8612 |

| bootstrap tiled - plain | op | pooled delta | CI95 | videos > 0 | L2VO mean |
|---|---|---|---|---|---|
| AP30_mAP | native | -0.0126 | [-0.01087, 0.07271] | 4/9 | -0.00591 |
| AP50_mAP | native | 0.0073 | [0.00523, 0.10442] | 5/9 | 0.01527 |
| mAP_class_agnostic_50 | native | 0.1587 | [-0.04706, 0.14708] | 4/9 | -0.0082 |
| overall_gt_recall_30 | native | 0.1636 | [0.00392, 0.13057] | 4/9 | 0.01736 |
| overall_gt_recall_50 | native | 0.1724 | [0.02353, 0.14542] | 5/9 | 0.03519 |
| AP30_mAP | s005 | -0.0192 | [-0.00012, 0.08606] | 6/9 | 0.0068 |
| AP50_mAP | s005 | 0.0011 | [0.00909, 0.10523] | 6/9 | 0.01639 |
| mAP_class_agnostic_50 | s005 | 0.1538 | [-0.04797, 0.14459] | 4/9 | -0.0105 |
| overall_gt_recall_30 | s005 | 0.1277 | [0.01575, 0.12337] | 6/9 | 0.02241 |
| overall_gt_recall_50 | s005 | 0.1174 | [0.02274, 0.12028] | 6/9 | 0.03133 |

Per-video AP30 deltas (native): {"v01": 0.0311, "v02": -0.0215, "v03": 0.1096, "v04": -0.0151, "v05": 0.006, "v06": -0.0085, "v07": 0.1763, "v08": 0.0, "v09": -0.0334}
