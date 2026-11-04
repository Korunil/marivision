# M1 plain vs tiled R0 - epoch15_stride10_offset5_V1_SCALE_ROUTED

426 VAL frames, 1363 GT, checkpoint epoch 15 (sha 64f9b4dee2b7), pycocotools parity: PASSED

**Primary endpoint (AP30_mAP tiled - plain, native protocol (no cut, top-100)): NOT_MET** - pooled delta -0.0158, 95% CI [-0.01431, 0.0804], 5/9 videos positive (rule: ci95_low > 0 AND >= 6 videos positive)

| metric | plain native | tiled native | plain s>=0.05 | tiled s>=0.05 |
|---|---|---|---|---|
| AP30_mAP | 0.3578 | 0.342 | 0.3351 | 0.3121 |
| AP50_mAP | 0.2793 | 0.2876 | 0.2665 | 0.2674 |
| AP30_gt_weighted_mAP | 0.2704 | 0.3697 | 0.229 | 0.2971 |
| AP50_gt_weighted_mAP | 0.1882 | 0.2907 | 0.1686 | 0.2424 |
| mAP_class_agnostic_50 | 0.2371 | 0.4078 | 0.231 | 0.3984 |
| AP_S_at_50_mAP | 0.1332 | 0.1894 | 0.1231 | 0.17 |
| recall_1to1_iou30 | 0.504 | 0.675 | 0.3103 | 0.4439 |
| recall_1to1_iou50 | 0.361 | 0.5598 | 0.2465 | 0.3822 |
| fp_per_frame_1to1_iou50 | 98.85 | 98.21 | 3.51 | 6.55 |

| AP30 per class | plain native | tiled native |
|---|---|---|
| ANIMAL | 0.0891 | 0.0891 |
| BOAT | 0.3467 | 0.3419 |
| MARINE_MARKER | 0.129 | 0.3763 |
| OBJECT | 0.2126 | 0.2225 |
| SAILING_VESSEL | 0.7815 | 0.471 |
| SHIP | 0.5877 | 0.5513 |

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
| AP30_mAP | native | -0.0158 | [-0.01431, 0.0804] | 5/9 | -0.00096 |
| AP50_mAP | native | 0.0083 | [0.00587, 0.12068] | 5/9 | 0.02043 |
| mAP_class_agnostic_50 | native | 0.1707 | [-0.02175, 0.16198] | 4/9 | 0.01077 |
| overall_gt_recall_30 | native | 0.171 | [0.00703, 0.13875] | 4/9 | 0.02186 |
| overall_gt_recall_50 | native | 0.1988 | [0.03417, 0.16828] | 5/9 | 0.05096 |
| AP30_mAP | s005 | -0.023 | [-0.00107, 0.09284] | 6/9 | 0.00771 |
| AP50_mAP | s005 | 0.0009 | [0.0099, 0.11825] | 6/9 | 0.0205 |
| mAP_class_agnostic_50 | s005 | 0.1674 | [-0.02205, 0.16151] | 4/9 | 0.01061 |
| overall_gt_recall_30 | s005 | 0.1336 | [0.01917, 0.1341] | 6/9 | 0.03249 |
| overall_gt_recall_50 | s005 | 0.1357 | [0.03191, 0.1385] | 6/9 | 0.04867 |

Per-video AP30 deltas (native): {"v01": 0.0252, "v02": -0.0225, "v03": 0.1122, "v04": -0.0573, "v05": 0.0075, "v06": 0.0808, "v07": 0.1764, "v08": 0.0, "v09": -0.0404}
