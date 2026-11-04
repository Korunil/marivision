# G2 results (G2.0.1)

Checkpoint: `D:\MARIVISION\G2_runs\main\expert_last.pth`  holdout: None

## Primary: paired class decision on R0-covered VAL GT boxes

GT 13699, covered 6141

| | R0 | Expert |
|---|---|---|
| accuracy | 0.6935 | 0.6475 |
| balanced accuracy | 0.7218 | 0.7042 |

Δ balanced accuracy -0.0177; fixed 386, broken 669; videos not worse 6/9

| Class | recall R0 | recall expert |
|---|---|---|
| ANIMAL | 0.8500 | 1.0000 |
| BOAT | 0.9246 | 0.8418 |
| MARINE_MARKER | 0.4214 | 0.3611 |
| OBJECT | 1.0000 | 1.0000 |
| SAILING_VESSEL | 1.0000 | 1.0000 |
| SHIP | 0.1349 | 0.0221 |

| Video | n | acc R0 | acc expert |
|---|---|---|---|
| video_01 | 805 | 0.4820 | 0.3640 |
| video_02 | 275 | 1.0000 | 1.0000 |
| video_03 | 1492 | 0.3613 | 0.1937 |
| video_04 | 467 | 0.9036 | 0.9914 |
| video_05 | 1143 | 0.7122 | 0.7122 |
| video_06 | 372 | 0.7796 | 0.9892 |
| video_07 | 440 | 0.8795 | 0.7432 |
| video_08 | 331 | 1.0000 | 1.0000 |
| video_09 | 816 | 0.9963 | 1.0000 |

Expert balanced accuracy on all GT boxes (oracle boxes, incl. not covered): 0.5668
LV predicted on covered boxes (all wrong on VAL): 59

**PRIMARY VERDICT: FAIL**

## Secondary: mAP30 on 432 VAL frames (stride 10, offset 0), R0 entities rescored

| Arm | mAP30 | Δ | videos + |
|---|---|---|---|
| A0 | 0.4241 | +0.0000 | – |
| E1 | 0.2909 | -0.1332 | 4 |
| E2 | 0.3508 | -0.0733 | 5 |

| Video | A0 | E1 | E2 |
|---|---|---|---|
| video_01 | 0.7117 | 0.3767 | 0.6394 |
| video_02 | 0.9413 | 0.9413 | 0.9413 |
| video_03 | 0.1100 | 0.1194 | 0.1427 |
| video_04 | 0.9584 | 0.9870 | 0.9866 |
| video_05 | 0.7534 | 0.5381 | 0.7155 |
| video_06 | 0.3153 | 0.3312 | 0.3383 |
| video_07 | 0.4284 | 0.4144 | 0.4709 |
| video_08 | 0.5000 | 0.5000 | 0.5000 |
| video_09 | 0.6445 | 0.6470 | 0.6455 |

Parity: rebuilt A0 = raw export on these frames: True (0.424109); raw export full-VAL mAP30 0.424488

LV guard: {"lv_argmax_on_confident_val_entities": 6}
