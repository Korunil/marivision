# G4b pre-declaration: restoring ANIMAL without losing the tile gains (2026-10-31)

- **Script:** `g4b_animal_finetune.py`, version G4B.0, SHA-256 cb199b02bf5dde64ed7f1e96b7c270091554b529100af30fc233545a516864cf. The sandbox selftest passes.
- **Where it runs:** from `D:\MARIVISION\R0_frozen_baseline`.
- **Reused:** it imports `g4_pilot.py` (G4P.0) and `g4_full.py` (G4F.0.1), both SHA-checked: model build, loss, tile dataset, export and atomic save.
- **Approval:** the Director approved G4b on 2026-10-31 ("we can't lose those points to ANIMAL").

## Why
- G4F epoch 15 passed its primary endpoint (+0.136 on offset 0, 7/9 videos; replicated at +0.158, 6/9, on offset 5).
- But its ANIMAL head collapsed:
  - VAL ANIMAL AP 0.0005 vs R0's 0.143;
  - best ANIMAL score on any VAL bird 0.14–0.27;
  - about 1,400 false ANIMAL predictions.
- **Paired evidence (M2 armF vs armT) blames tile training.** ANIMAL's share of training targets falls from 3.6% (full frames) to 2.2% (tiles).
- **TEST stakes:** 932 ANIMAL boxes, and ANIMAL is the paper R50's best class (52.8 AP). A collapse there could cost up to about 7.5 TEST mAP.

## Arms
All three arms are identical except the named factor:
- **Init:** `g4f_epoch15.pth`.
- **Optimizer:** fresh AdamW at 0.1 × the R0 LRs (2e-5 core / 2e-6 backbone / 2e-6 linear_proj), weight decay 1e-4, clip 0.1.
- **Schedule:** 10,155 steps at batch 2, fp32, seed 42.
- **Frame order:** the same base permutation of all 20,310 TRAIN frames.

| Arm | Data change | Basis |
|---|---|---|
| CTRL | none (G4F recipe: M2 tiles, p_tile 0.5) | separates the effect of one more epoch from the fixes |
| NOTILE | full frames only (p_tile 0) | M2 armF (no tiles) kept ANIMAL at 0.139; armT (tiles) fell to 0.005 |
| RFS | p_tile 0.5 + ANIMAL repeat-factor sampling + ANIMAL-aware window choice (below) | Repeat-factor sampling for rare classes (Gupta et al., LVIS, CVPR 2019) |

**RFS details**
- The repeat factor is r = max(1, √(t / f)), with t = 0.3 and f the share of TRAIN frames containing ANIMAL, applied with stochastic rounding. The selection is ANIMAL-only, so the effect can be attributed.
- When a frame containing ANIMAL is tiled, the window is drawn uniformly among windows in which an ANIMAL box is ≥ 50% visible.
- t = 0.3 is pragmatic and untuned.

## Scoring
- Inference is fp32, tiled (17-tile V0 merge) and plain, on 432 offset-0 + 432 offset-5 VAL frames, with pycocotools AP@0.3.
- **References:** G4F e15 and R0 e15 (existing caches).

## Pass rule (frozen)
An arm **passes** only if all three hold:
1. ANIMAL AP30 (tiled, offset 0 + offset 5 pooled, 52 GT in v09) **≥ 0.05**;
2. **and ≥ CTRL + 0.03**;
3. **and** the offset-0 tiled macro AP30 of the 5 other VAL classes **≥ CTRL − 0.01**.

**Winner:** the passing arm with the highest offset-0 tiled macro AP30. A tie goes to NOTILE, the simpler arm.

**Reported, not decision inputs:**
- ANIMAL GT scored ≥ 0.3;
- best ANIMAL score on GT (median / max);
- ANIMAL false positives ≥ 0.05;
- per-class AP;
- plain-inference results.

**Consequences**
- **Winner:** its recipe change becomes part of the G5 recipe. The paper reports it as a G4b amendment.
- **No winner:** G5 does not start, and ANIMAL is diagnosed further.

## Known limits
- **VAL measures ANIMAL weakly:** all 253 VAL ANIMAL GT are in one video (v09), and every model, R0 included, misses most of them. This selection uses VAL frames, so only G5's single TEST evaluation is an independent check.
- **One epoch of fine-tuning is short.** A failure here does not prove that a longer schedule would fail.

## Proof before the runs
**Selftest** (sandbox CPU, synthetic zip with ANIMAL). It checks:
- sampling: CTRL and NOTILE share the same order; RFS has r = 1.73 when f = 0.1; ANIMAL sample share 0.100 → 0.159; deterministic;
- the ANIMAL indicators against a hand-computed case;
- the verdict rule;
- RFS ANIMAL frames always carry an ANIMAL target;
- all 3 arms: strict init, LR × 0.1, finite steps, refusal to overwrite;
- export at offsets 0 and 5 → score → verdict → report, end to end.

**Smoke** (GPU). It checks:
- 40 steps per arm: s/step, sampling stats, ANIMAL targets seen, tile fraction (NOTILE must be 0);
- export parity of the rebuilt e15 checkpoint against the existing e15 offset-0 cache.

## Selftest and smoke on the workstation (2026-10-31)
**Selftest: all 5 checks printed `[ok]`.**
- The script then raised `PermissionError` while deleting its temporary folder. Windows keeps the synthetic `SeaClips.zip` locked, because the selftest's datasets still hold open zip handles.
- This is a cleanup fault in the selftest only; every assertion had already passed. No code change was made, and the SHA is unchanged.

**Smoke: PASS.**

| Check | Result |
|---|---|
| e15 init SHA | b393a1a5…fef6 |
| ANIMAL frames | 1,626 of 20,310 (f = 0.080) |
| RFS repeat factor | 1.94; expected ANIMAL sample share ≈ 0.145 over the epoch, vs 0.080 |
| NOTILE tile fraction | 0 |
| fp32 s/step | about 0.77 |
| Export parity of the rebuilt e15 checkpoint | 51 tiles, 0.0 px |
| Projection | 2.23 h per arm, 8.2 h in total |

## Results (2026-10-31): no winner. G5 does not start.

**Training:** all 3 arms completed in about 129 min each, at 0.76 s/step with an 8.03 GB peak.

| Arm | ANIMAL targets seen in the epoch |
|---|---|
| CTRL | 1,318 |
| NOTILE | 2,356 |
| RFS | 3,931 (sample share 0.143) |

**Exports:** 6 caches written (offset 0: 432 frames; offset 5: 426 frames).

| Cache | SHA-256 |
|---|---|
| CTRL offset 0 | c3099060… |
| CTRL offset 5 | 032c1049… |
| NOTILE offset 0 | bd5c6dc5… |
| NOTILE offset 5 | 1ee781b5… |
| RFS offset 0 | f83e9aa6… |
| RFS offset 5 | 71c9eeb2… |

**Results.** All columns are AP30 with tiled inference. Macros are on offset 0; ANIMAL is pooled over offsets 0 + 5.

| Model | Macro | Macro, 5 classes without ANIMAL | ANIMAL | ANIMAL GT scored ≥ 0.3 | Best ANIMAL score on GT (max) | ANIMAL FP ≥ 0.05 |
|---|---|---|---|---|---|---|
| R0 e15 | 0.3459 | 0.4016 | 0.0792 | 6/52 | 0.886 | 971 |
| G4F e15 | 0.5011 | 0.6013 | 0.0005 | 1/52 | 0.351 | 2,897 |
| CTRL | 0.4912 | 0.5893 | 0.0005 | 2/52 | 0.444 | 2,254 |
| NOTILE | 0.4781 | 0.5737 | 0.0005 | 1/52 | 0.457 | 2,283 |
| RFS | 0.4916 | 0.5898 | 0.0005 | 3/52 | 0.544 | 2,961 |

**Verdict against the frozen rule**
- **NOTILE fails:** ANIMAL is not fixed, and the guard fails (0.5737 < 0.5793).
- **RFS fails:** ANIMAL is not fixed; the guard passes.

**Readings**
- **Reference check:** the R0 e15 tiled macro is 0.3459 at both offsets, which is a coincidence. It is already recorded in M1.1 (offset 0 = M1.0's 0.3459; offset 5 V0 = 0.3459). The G4F primary used R0 e15 *plain* (0.3647).
- **One epoch of full-frame fine-tuning does not reverse the collapse.** M2 armF had shown that full-frame training *prevents* the collapse from the R0 init; G4b shows that it does not *reverse* it once a model has been tile-trained for 15 epochs.
- **RFS raises ANIMAL confidence on the real birds** (best score on a bird 0.35 → 0.54; 3/52 above 0.3), **but also raises the ANIMAL false positives** (2,961). The failure is precision: false positives outrank the birds.
- **The arms differ by up to ±0.07 per class** (SHIP 0.67–0.75), so single-run per-class VAL differences of that size are noise.
