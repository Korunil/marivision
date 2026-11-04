# G4 pilot pre-declaration: four small-budget arms, staged (2026-10-28)

- **Script:** `g4_pilot.py`, version G4P.0, SHA-256 3128fbf2985c7aa26fc5d38ec334e5cc67ac8281947c8b2da5932e31bdcd496b. The sandbox selftest passes.
- **Where it runs:** from `D:\MARIVISION\R0_frozen_baseline`, using R0's frozen model, dataset and `r0_preflight` code, as M2 did.
- **Reused unchanged and SHA-checked at start:**
  - `m2_train_r0_finetune.py` (M2.1, 8711c96c…): tile dataset and box clipping.
  - `m1_r0_tiled_inference.py` (M1.0, 73044625…): tile schedule and R0 eval transform.
- **Scoring:** the unchanged `m2_evaluate.py` (M2E.0): same evaluator, pycocotools parity, 9-video bootstrap and FP anatomy as M2.
- **Data:** TEST is never read.

## Why a pilot
- G3 showed a ConvNeXt-B (in22k) Deformable-DETR costs about the same per step as R0 on the RTX 3060.
- Before committing 1–3 GPU-days, the Director asked for a cheap paired test of two things:
  1. ConvNeXt-B with tiles against ConvNeXt-B without tiles;
  2. then the winner against R50.
- The precision effect (bf16 vs fp32) is measured at the same scale, so that a later long run can be compared fairly with R0 epoch 15 (fp32).

## Arms (each comparison differs in exactly one factor)

| Arm | Backbone | Precision | Tiles | Stage |
|---|---|---|---|---|
| CNB_bf16 | ConvNeXt-B in22k (timm) | bf16 autocast | no | 1 |
| CNBT_bf16 | ConvNeXt-B in22k | bf16 | M2 recipe: p_tile 0.5, min_visible 0.5 | 1 |
| R50_bf16 | R50 (R0) | bf16 | no | 2 |
| R50_fp32 | R50 (R0) | fp32 (R0's setting) | no | 2 |

## Common recipe (frozen)
**Data and schedule**
- Every 5th TRAIN frame of every video, sorted by frame_id: 4,086 frames from all 47 videos (4,062 was my earlier estimate; the smoke counted 4,086).
- 10,155 optimizer steps at batch 2, which equals one R0 epoch of steps.
- One seeded frame order, identical for every arm (seed 42).
- R0 transforms, with horizontal flip.

**Optimisation (R0's own)**
- AdamW parameter groups and learning rates: 2e-4, backbone 2e-5, linear_proj ×0.1.
- Weight decay 1e-4, gradient clip 0.1.
- The LR drops ×0.1 at 80% of the steps (step 8,124), for every arm.

**Initialisation**
- R50 arms: the COCO Deformable-DETR checkpoint minus the class head, exactly as R0.
- ConvNeXt arms: COCO transformer, queries and box head. The backbone is timm in22k with its stem and first stage frozen. The input projections and 7-class head are new.
- The bf16 arms use autocast on the forward pass only. Loss and optimizer run in fp32, with no loss scaling.

**Checkpoints and inference**
- Checkpoints are saved at 50% and 100%. **Only 100% is used for decisions.** The 50% checkpoint serves only the trend check.
- Inference is the same for every arm: fp32, the 17-tile M1 schedule on the 432 VAL frames (stride 10, offset 0), cached in the M1.1 tile-cache schema.
- *Plain* means the full-frame tile. *Tiled* means the V0 merge.

## Decision rules (frozen)
**Stage 1: tiles or not**
- Scoring: `m2_evaluate.py` with `--cache-T CNBT@100 --cache-F CNB@100`. Its PRIMARY row is `T_tiled_vs_F_plain`.
- **Tiles win only if all three hold:**
  1. pooled Δ macro AP30 ≥ +0.02;
  2. ≥ 5 of 9 videos positive;
  3. from CNBT@50 to CNBT@100 (tiled), the growth factor of BACKGROUND false positives is ≤ the growth factor of TPs.
     - These come from the FP anatomy of a second scoring run with the @50 caches.
- **Otherwise CNB (no tiles) is the winner.** It is the simpler model.
- The tie case is recorded for TEST/LV reasons, but **does not change the winner**.

**Stage 2: backbone and precision**
- **Precision:** `--cache-T R50_bf16@100 --cache-F R50_fp32@100`, row `T_plain_vs_F_plain`.
  - The long run uses bf16 unless Δ(bf16 − fp32) ≤ −0.01. In that case it uses fp32.
  - If −0.01 < Δ ≤ −0.005 and ≤ 3 of 9 videos are positive, the result is **ambiguous**. A full R0 bf16 reference run is then put to the Director as an option.
- **Backbone:** `--cache-T winner@100 --cache-F R50_bf16@100`. Each arm uses its natural inference: CNB plain, CNBT tiled.
  - **It passes only if** pooled Δ ≥ +0.02 and ≥ 5 of 9 videos are positive.
  - **If it fails, no GPU-days are committed.** We first diagnose, e.g. whether the new input projections need longer.

**Reported but not used for any decision:**
- pilot vs R0 epoch 15 and R0 epoch 04 (budget mismatch, so for information only);
- per-class AP, recall by size, FP anatomy;
- bootstrap 95% CIs.

## Known limits
- **Direction, not magnitude.** A pilot gain does not predict the size of a full-run gain.
- **It is biased against ConvNeXt-B**, whose input projections start untrained. A negative backbone result is therefore weaker evidence than a positive one.
- **It is one seed.** R0 swings about ±5 points between epochs, which is why there is a +0.02 floor and a video-count rule.
- **VAL has no LEISURE_VEHICLE**, so tiles' main TEST benefit (small LV) is invisible here.

## Proof before the runs
1. **selftest** (CPU, synthetic zip, against R0-folder code). It checks:
   - each comparison differs in exactly one factor;
   - the LR schedule;
   - every 5th frame per video;
   - an identical seeded frame order;
   - all arms build, with a complete optimizer partition;
   - COCO transfer;
   - finite bf16 training steps for R50 and for ConvNeXt + tiles, with the same frame order in both;
   - exact checkpoint rebuild;
   - that the export writes the M1.1 schema and m2_evaluate's plain/V0 builders read it.

   Selftest training runs at 256 px for CPU memory; the real path is 640.
2. **smoke** (GPU, real data). It checks:
   - 20 real steps per arm with the real loader: s/step, peak memory and the same frame order;
   - bf16 vs fp32 loss on identical weights and batches, for R50 and ConvNeXt-B (max relative difference ≤ 5%);
   - **export parity:** this export path on the R0 epoch-15 checkpoint must reproduce the existing R0e15 tile cache tile by tile (M1.1 tolerance) on 3 frames;
   - projections of training and export time.

## Smoke result (workstation, 2026-10-28; G4P.0, SHA 3128fbf2…496b): PASS
- **Selftest:** passed on the workstation.
- **Subset:** 4,086 frames.
- **Frame order:** the same first image ids in all four arms.

| Arm | s/step (with data loading) | Peak GB | Projected 10,155 steps |
|---|---|---|---|
| R50_fp32 | 0.448 | 4.8 | 1.26 h |
| R50_bf16 | 0.424 | 4.7 | 1.20 h |
| CNB_bf16 | 0.496 | 6.6 | 1.40 h |
| CNBT_bf16 | 0.505 | 6.6 | 1.43 h |

- **bf16 vs fp32 loss**, same weights and batches:
  - R50: max relative difference 4.6%, inside the 5% limit but close to it. Likely Hungarian-matching flips on near-ties.
  - ConvNeXt-B: 0.3%.
  - Stage 2 measures the precision effect on mAP directly.
- **Export parity against the existing R0 epoch-15 tile cache:** 3 frames × 17 tiles, 0 failed tiles, 0.0 px maximum box deviation. The G4 export path reproduces the M1.1 cache exactly.
- **Export time:** ConvNeXt-B 2.2 s/frame, so about 16 min per 432-frame cache.
- **Stage 1 total:** about 2.8 h of training, 4 exports (about 1.1 h) and 2 CPU scorings.

## Stage 1 result (2026-10-28): **tiles win (CNBT_bf16)**
**Runs**
- Both arms trained for 10,155 steps at 0.48 s/step, with peak memory 6.58 GB and no non-finite loss.
- All 4 exports completed in about 14 min each, and pycocotools parity passed for every variant.
- Cache SHAs:

| Cache | SHA-256 |
|---|---|
| CNB@100 | e07c424b…afdbd |
| CNBT@100 | 8834b4e9…1cb9 |
| CNB@50 | bab9c707…59d1 |
| CNBT@50 | 9f56734b…d091 |

**Pre-declared rule, applied to `m2_results_g4s1_at100` and `m2_results_g4s1_at50`**

| Condition | Required | Observed | Met |
|---|---|---|---|
| Δ macro AP30, CNBT tiled − CNB plain | ≥ +0.02 | **+0.0857** (0.5471 vs 0.4614) | yes |
| Videos positive | ≥ 5/9 | **5/9**: v01 +0.01, v03 +0.16, v05 +0.13, v06 +0.26, v07 +0.12; v02 −0.06, v04 −0.04, v08 −0.03, v09 −0.15 | yes |
| BACKGROUND FP growth ≤ TP growth, CNBT tiled from @50 to @100 | BG factor ≤ TP factor | BG 20,023 → 11,387 (×0.57); TP 852 → 1,109 (×1.30) | yes |

**Decision: CNBT_bf16 (ConvNeXt-B + tiles, tiled inference) is the stage 1 winner.**

## Caveats (reported, not decision inputs)
- **The evidence is moderate.** M2's stricter bootstrap rule is NOT met: the 95% CI is [−0.035, 0.126], with 5/9 videos rather than 6. Four videos are negative, v09 the most.
- **ANIMAL is near 0 in both ConvNeXt arms**: CNB 0.021, CNBT 0.001, against R0e04's 0.090. All 27 ANIMAL GT are in v09, which is the most negative video. The same collapse appeared in M2's tile arm.
- **A tile-trained model needs tiled inference.** CNBT plain is *below* CNB plain (−0.0225, 2/9 videos), as in M2. The deployable model is therefore CNBT + 17-tile V0 inference, at about 2.2 s/frame on the RTX 3060.
- **The 50% checkpoints were taken before the LR drop**, so the @50 → @100 trend mixes training length with the drop. All arms improved strongly: CNB plain 0.361 → 0.461, CNBT tiled 0.280 → 0.547.
- **For information only (budget mismatch, not paired by recipe):**
  - CNB plain at pilot budget (0.461) already exceeds the fully trained R0e04 (0.424, VAL-selected) and R0e15 (0.365).
  - CNBT tiled (0.547) is the highest VAL macro AP30 of the programme, at +0.123 over R0e04.
  - The backbone effect has not yet been tested paired; stage 2 does that.

## Stage 2 (as pre-declared)
- **Training:** R50_bf16 and R50_fp32, with exports @100 only (the stage 2 rules need no trend).
- **Precision:** `--cache-T R50_bf16@100 --cache-F R50_fp32@100`, row `T_plain_vs_F_plain`.
- **Backbone, the decision row:** `--cache-T CNBT@100 --cache-F R50_bf16@100`, row `T_tiled_vs_F_plain`, each arm at its natural inference.
- **Added before any stage 2 result, reported and not a decision input:** `--cache-T CNB@100 --cache-F R50_bf16@100`, row `T_plain_vs_F_plain`. It is the backbone effect alone, without tiles, which the paper's ablation needs.

## Stage 2 result (2026-10-29)
**Runs**
- R50_bf16: 10,155 steps at 0.405 s/step, peak 4.66 GB.
- R50_fp32: 10,155 steps at 0.454 s/step, peak 4.82 GB.
- Exports: about 10 min each.
  - R50_bf16@100 SHA 141e00e3…628b.
  - R50_fp32@100 SHA 7eff6290…7228.
- pycocotools parity passed in all three scorings.

**Precision** (`g4s2_precision`, row `T_plain_vs_F_plain`, R50 bf16 − R50 fp32)

| | bf16 | fp32 | Δ | Videos + |
|---|---|---|---|---|
| plain, the decision row | 0.3922 | 0.4213 | **−0.0291** (CI −0.060…0.048) | 5/9 |
| tiled (information only) | 0.4357 | 0.4123 | +0.0234 | 6/9 |

**Decision (frozen rule: bf16 unless Δ ≤ −0.01): the long run uses fp32.**
- The sign flips between plain and tiled inference, so the precision effect is probably mostly noise at one seed.
- The rule is applied as written.

**Backbone** (`g4s2_backbone`, row `T_tiled_vs_F_plain`, CNBT tiled − R50_bf16 plain)
- **Δ = +0.1549** (0.5471 vs 0.3922), with **5/9 videos** positive.
- **The rule (≥ +0.02 and ≥ 5/9) is met: PASS.**
- Per-video deltas: v01 +0.02, v02 +0.01, v03 +0.12, v04 −0.07, v05 −0.01, v06 +0.37, v07 +0.06, v08 −0.02, v09 −0.01.
  - Their mean is about +0.05, far below the pooled +0.155. The pooled gain is class-level, mainly SHIP 0.67 → 0.98, SAILING_VESSEL 0.81 → 0.95 and MARINE_MARKER 0.26 → 0.60.
  - Per video, the gain is concentrated in v06 and v03. The bootstrap CI is [−0.014, 0.142]; M2's stricter rule is not met.
- Against R50 **fp32** plain (0.4213; an unpaired difference, information only): +0.126.

**Backbone alone, without tiles** (`g4s2_backbone_only`, row `T_plain_vs_F_plain`, CNB plain − R50_bf16 plain; added before stage 2 results, not a decision input)
- +0.0692, 5/9 videos (0.4614 vs 0.3922).
- Against R50 fp32 plain (unpaired): +0.040.

**ANIMAL**
- ANIMAL is near 0 for *every* pilot arm, R50 included: R50_bf16 0.001, R50_fp32 0.064.
- The collapse is therefore a pilot-budget / subset effect (about 470 TRAIN ANIMAL boxes in the 1-in-5 subset), not a ConvNeXt or tiles effect.
- VAL ANIMAL is not measurable on these frames: all 27 GT are in v09.

## Pilot conclusion
- Winner: **ConvNeXt-B + tiles (CNBT), tiled inference, trained in fp32.**
- Evidence: pooled gains of +0.086 over the same backbone without tiles, and +0.155 over R50 at the same budget. Both meet the pre-declared rules at 5/9 videos.
- Per-video evidence is moderate, and no bootstrap CI excludes 0. **The long run and the 3-seed G5 must confirm this.**
