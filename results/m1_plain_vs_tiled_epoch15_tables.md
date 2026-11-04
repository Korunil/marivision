# M1 plain vs tiled R0 - epoch15

432 VAL frames, 1386 GT, checkpoint epoch 15 (sha 64f9b4dee2b7), pycocotools parity: PASSED

**Primary endpoint (AP30_mAP tiled - plain, native protocol (no cut, top-100)): NOT_MET** - pooled delta -0.0188, 95% CI [-0.00781, 0.08918], 5/9 videos positive (rule: ci95_low > 0 AND >= 6 videos positive)

| metric | plain native | tiled native | plain s>=0.05 | tiled s>=0.05 |
|---|---|---|---|---|
| AP30_mAP | 0.3647 | 0.3459 | 0.3458 | 0.32 |
| AP50_mAP | 0.2703 | 0.2849 | 0.2598 | 0.2671 |
| AP30_gt_weighted_mAP | 0.2665 | 0.3745 | 0.2312 | 0.3114 |
| AP50_gt_weighted_mAP | 0.1803 | 0.2953 | 0.1628 | 0.2508 |
| mAP_class_agnostic_50 | 0.224 | 0.4072 | 0.2164 | 0.4 |
| AP_S_at_50_mAP | 0.1348 | 0.1944 | 0.1262 | 0.1781 |
| recall_1to1_iou30 | 0.4942 | 0.6753 | 0.3139 | 0.4632 |
| recall_1to1_iou50 | 0.3651 | 0.5606 | 0.2453 | 0.3983 |
| fp_per_frame_1to1_iou50 | 98.83 | 98.2 | 3.6 | 6.7 |

| AP30 per class | plain native | tiled native |
|---|---|---|
| ANIMAL | 0.1426 | 0.0673 |
| BOAT | 0.3586 | 0.3605 |
| MARINE_MARKER | 0.1186 | 0.3766 |
| OBJECT | 0.2272 | 0.2254 |
| SAILING_VESSEL | 0.7611 | 0.5008 |
| SHIP | 0.5802 | 0.545 |

| recall@0.3 by area bin (640-scale) | plain native | tiled native |
|---|---|---|
| 01_LE_4px2 | 0.4878 | 0.5732 |
| 02_4_to_6px2 | 0.0233 | 0.2209 |
| 03_6_to_9px2 | 0.1705 | 0.3566 |
| 04_9_to_25px2 | 0.2829 | 0.6149 |
| 05_25_to_150px2 | 0.794 | 0.8985 |
| 06_GT_150px2 | 0.8612 | 0.8571 |

| bootstrap tiled - plain | op | pooled delta | CI95 | videos > 0 | L2VO mean |
|---|---|---|---|---|---|
| AP30_mAP | native | -0.0188 | [-0.00781, 0.08918] | 5/9 | 0.00116 |
| AP50_mAP | native | 0.0146 | [0.01522, 0.13414] | 5/9 | 0.03557 |
| mAP_class_agnostic_50 | native | 0.1832 | [0.01174, 0.19224] | 5/9 | 0.03359 |
| overall_gt_recall_30 | native | 0.1811 | [0.01517, 0.15304] | 5/9 | 0.03358 |
| overall_gt_recall_50 | native | 0.1955 | [0.03175, 0.16891] | 5/9 | 0.04595 |
| AP30_mAP | s005 | -0.0258 | [0.00036, 0.0971] | 6/9 | 0.0077 |
| AP50_mAP | s005 | 0.0073 | [0.01704, 0.12863] | 6/9 | 0.03397 |
| mAP_class_agnostic_50 | s005 | 0.1836 | [0.01139, 0.19468] | 5/9 | 0.03364 |
| overall_gt_recall_30 | s005 | 0.1493 | [0.01998, 0.14204] | 6/9 | 0.02941 |
| overall_gt_recall_50 | s005 | 0.153 | [0.03564, 0.15241] | 6/9 | 0.05198 |

Per-video AP30 deltas (native): {"v01": 0.0343, "v02": -0.008, "v03": 0.1106, "v04": -0.0131, "v05": 0.0209, "v06": 0.0346, "v07": 0.2097, "v08": -0.0004, "v09": -0.0602}
