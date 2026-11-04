# G4 full-run pre-declaration: ConvNeXt-B + tiles, fp32, 15 epochs (2026-10-29)

- **Script:** `g4_full.py`, version G4F.0, SHA-256 b1e86c76094fe4f391a53fb2c2af28423d847a96435877d7179a6a796f74e487. The sandbox selftest passes.
- **Where it runs:** from `D:\MARIVISION\R0_frozen_baseline`.
- **Reused:** it imports `g4_pilot.py` (G4P.0, SHA-checked) for the model build, COCO transfer, loss, tile dataset and export conventions. The pilot's export path was verified against the R0 epoch-15 cache.
- **Approval:** the Director approved the full 15-epoch schedule on 2026-10-29, choosing it because it keeps the comparison with the paper's recipe exact.

## Recipe (frozen)
| Item | Setting |
|---|---|
| Model | R0's Deformable-DETR with a ConvNeXt-B in22k backbone (timm, stem + first stage frozen) |
| Init | COCO Deformable-DETR transformer / queries / box head; new backbone projections and 7-class head (as in the pilot) |
| Data | all 20,310 TRAIN frames; M2 tile sampling p_tile 0.5, min_visible 0.5; flip 0.5; R0 transforms (640×640) |
| Schedule | 15 epochs × 10,155 steps, batch 2 (R0 / paper: 15 epochs) |
| Optimizer | AdamW, R0 groups: 2e-4 / backbone 2e-5 / linear_proj ×0.1; weight decay 1e-4; clip 0.1 |
| LR | ×0.1 after epochs 11 and 14, i.e. R0's `MultiStepLR([11, 14])`: epochs 1–11 at 2e-4, 12–14 at 2e-5, 15 at 2e-6 |
| Precision | fp32 (pilot precision rule) |
| Seed | 42. Epoch *e* uses the seeded permutation and RNG (42·1000 + *e*), so a resumed epoch replays exactly. CPU selftest: max \|Δparam\| 0 after stop/resume. |
| Checkpoints | Model-only after every epoch (`g4f_epochNN.pth`), plus `resume_latest.pth` (model + optimizer), written atomically |

## Per-epoch selection data (inline, after each checkpoint)
- After each epoch, the model runs tiled and plain inference on the **432 VAL frames at stride 10, offset 5**. These frames are disjoint from the offset-0 evaluation frames and come from the same 9 videos.
- The cache uses the M1.1 schema. Plain = the full-frame tile; tiled = the 17-tile V0 merge.
- pycocotools macro AP30 is computed for both and written to `curve.json`.
- An evaluation failure never stops training. A missing cache can be re-exported with `--mode export`.

## Endpoints (frozen before any full-run result)
**Primary: no selection**
- Epoch 15 (tiled) vs R0 epoch 15 (plain).
- Frames: 432 VAL frames at offset 0.
- Scored by the unchanged `m2_evaluate.py`: `--cache-T e15@offset0`, with the pilot R50_fp32@100 cache passed as `--cache-F` only because the tool requires one. Its comparisons are not interpreted.
- The row read is `T_tiled_vs_R0e15_plain`.
- **MET** means M2's rule: bootstrap 95% CI low > 0 **and** ≥ 6/9 videos positive.
- **Supportive** means the pilot's rule: Δ ≥ +0.02 **and** ≥ 5/9 videos positive.

**Secondary: the paper's protocol (VAL-selected epoch)**
- The epoch is selected by the highest **offset-5** tiled macro AP30, with ties going to the earlier epoch.
- It is compared, on offset 0, with R0 epoch 04 (plain). R0 epoch 04 was selected on full VAL, so its number carries more selection optimism than ours.
- The row read is `T_tiled_vs_R0e04_plain`.
- If the selected epoch is not 15, a second `m2_evaluate` run uses `--cache-T best@offset0 --cache-F e15@offset0`. Its `T_tiled_vs_F_tiled` row then shows best vs last.

**Reported, not decision inputs:**
- the full offset-5 curve;
- per-class AP, FP anatomy and recall by size;
- plain inference of the same checkpoints.

**Next steps after this run**
- **MET or supportive:** proceed to the G5 pre-declaration (3 seeds, a single TEST evaluation, Welch/Bonferroni as in the paper). The 3 seeds also decide whether the claim holds.
- **Neither:** stop and diagnose before any further GPU-days.

## Known limits
- **One seed.** G5 supplies the seeds.
- **VAL has no LEISURE_VEHICLE.** The LV augmentation (G1c D1–D4) is not part of this run.
- **ANIMAL is not measurable** on offset-0 frames: all 27 GT are in v09.
- **The VAL → TEST level transfer is weak** (R0: about 0.42 on VAL vs 22.9 on TEST). The VAL gains are relative evidence only.

## Proof before the run
**Selftest** (sandbox CPU, synthetic zip, against R0-folder code). It checks:
- the LR schedule equals `MultiStepLR([11, 14])`;
- the per-epoch seeded order;
- the best-epoch rule;
- 2 CPU epochs with inline export + scoring;
- **stop after epoch 1 + resume is bit-identical to the uninterrupted run**;
- an epoch checkpoint rebuilt from disk re-exports the inline cache exactly;
- a fresh run refuses to overwrite an existing run folder.

**Smoke** (GPU, real data). It checks:
- 2 × 40-step "epochs" with inline evaluation (3 frames);
- stop/resume on GPU: epoch-2 loss within 5% and |Δparam| < 1e-2, since GPU kernels are not bit-deterministic;
- evaluation memory with the training state still resident;
- export parity of this export path on the R0 epoch-15 checkpoint against the existing cache;
- fp32 s/step (steps 21–40, excluding worker start-up);
- projections and checkpoint size.

## Amendment G4F.0.1 and audit additions (2026-10-29, before the full run)
**Smoke result for G4F.0: FAIL on the resume check only.** Everything else passed:
- inline evaluation;
- export parity vs the R0 epoch-15 cache (51 tiles, 0.0 px);
- evaluation peak 3.5 GB;
- fp32 at 0.777 s/step, projecting 36.5 h for 15 epochs;
- epoch checkpoints of 0.38 GB.

**Diagnosis: a fault in the check, not in resume.**
- The check compared an uninterrupted GPU run with a resumed one, using a fixed |Δparam| < 1e-2. It measured 0.0129.
- Two *fresh, identical* GPU runs already diverge after 40 steps: epoch-1 average loss 23.03 vs 22.68. `grid_sample` backward is not bit-deterministic.
- The CPU selftest shows resume is bit-exact (|Δ| = 0, also on the workstation).

**G4F.0.1** (SHA-256 c1fbed7f03bbfaac525c422a00b85a436d582116d4b1b2e72d2d69472021479d) changes only the smoke check. It now requires:
- the resume file holds epoch 1, with 40 optimizer steps on every parameter, and weights equal to `g4f_epoch01.pth`;
- the same epoch-2 frame order and LR factor;
- uninterrupted-vs-resumed drift ≤ 5 × the fresh-vs-fresh epoch-1 drift.

Training code is unchanged. The sandbox selftest passes.

**Audit additions** (see the decision record "MARIVISION decision record and parameter audit")
- **Offset-5 replication added, reported only:** epoch 15 vs R0 epoch 15 on the offset-5 frames, pooled and per video, with the same pycocotools scorer. At epoch 15 no selection touches offset 5. The frames are correlated with offset 0, so this is a replication, not an independent test.
- **Corrected wording:**
  - stride 10 and offsets 0 and 5 are our pre-registered subsets (V13, M1.1), not the paper's protocol;
  - the paper does not document checkpoint selection;
  - R0's LR milestones (epochs 11 and 14) were an assumption of the R0 reproduction.

## Training result (completed 2026-10-30/31): offset-5 curve
- **Run:** all 15 epochs completed, at about 132 min of training plus about 14 min of evaluation per epoch. Evaluation peak was 3.5 GB.

| Epoch | LR factor | Avg loss | Plain AP30 (off5) | Tiled AP30 (off5) |
|---|---|---|---|---|
| 1 | 1 | 9.87 | 0.394 | 0.453 |
| 2 | 1 | 8.14 | 0.312 | 0.402 |
| 3 | 1 | 7.78 | 0.357 | 0.488 |
| 4 | 1 | 7.34 | 0.315 | 0.371 |
| 5 | 1 | 7.09 | 0.383 | 0.474 |
| 6 | 1 | 7.01 | 0.292 | 0.382 |
| 7 | 1 | 6.75 | 0.298 | 0.405 |
| 8 | 1 | 6.65 | 0.255 | 0.372 |
| 9 | 1 | 6.48 | 0.378 | 0.461 |
| 10 | 1 | 6.43 | 0.279 | 0.399 |
| 11 | 1 | 6.30 | 0.340 | 0.427 |
| 12 | 0.1 | 4.71 | 0.416 | 0.505 |
| 13 | 0.1 | 4.38 | 0.432 | 0.509 |
| 14 | 0.1 | 4.29 | 0.441 | **0.517** |
| 15 | 0.01 | 4.07 | 0.440 | 0.516 |

**Selected epoch** (frozen rule: highest offset-5 tiled score, ties go to the earlier epoch): **14**. Epoch 14 vs epoch 15 is +0.001, which is within noise.

**Observations (descriptive only)**
- **At the full learning rate (epochs 1–11)**, VAL scores swing by ±0.06 between epochs, while training loss falls smoothly.
- **After the first drop (epoch 12)** the scores stabilise at 0.505–0.517. Epochs 14 and 15 are a plateau, with no late overfitting. R0, by contrast, fell from 0.424 at epoch 4 to 0.365 at epoch 15.
- **Per class at epoch 15 (tiled, offset 5):**

  | Class | AP30 |
  |---|---|
  | SAILING_VESSEL | 0.946 |
  | SHIP | 0.780 |
  | MARINE_MARKER | 0.520 |
  | BOAT | 0.469 |
  | OBJECT | 0.379 |
  | ANIMAL | 0.001 |

  **ANIMAL is near 0 even on full TRAIN.** So the collapse is not a pilot-subset effect, which contradicts the G4 pilot interpretation. It needs a diagnosis before G5.
- **These offset-5 numbers can't be compared** with any offset-0 number in earlier tables. The endpoints are read on offset 0.

## Endpoint results (2026-10-31; offset 0, 432 frames, pycocotools parity passed)
| Comparison | Ours | Baseline | Δ macro AP30 | Bootstrap 95% CI | Videos + | Verdict |
|---|---|---|---|---|---|---|
| **Primary:** e15 tiled vs R0 e15 plain | 0.5011 | 0.3647 | **+0.136** | [−0.001, 0.197] | **7/9** | **SUPPORTIVE**; MET missed by 0.001 on the CI low |
| Secondary: e14 (selected) tiled vs R0 e04 plain | 0.5050 | 0.4241 | +0.081 | [−0.057, 0.183] | 6/9 | supportive rule met; reported only |
| e14 vs e15 (both tiled) | 0.5050 | 0.5011 | +0.004 | [−0.005, 0.008] | 5/9 | equal within noise |

**Primary per-video Δ:** v01 +0.05, v02 −0.08, v03 +0.13, v04 +0.03, v05 +0.39, v06 +0.15, v07 +0.27, v08 +0.01, v09 −0.11.

**Per class (e15 tiled vs R0 e15 plain)**

| Class | Ours | R0 e15 |
|---|---|---|
| SAILING_VESSEL | 0.928 | 0.761 |
| SHIP | 0.746 | 0.580 |
| MARINE_MARKER | 0.537 | 0.119 |
| BOAT | 0.435 | 0.359 |
| OBJECT | 0.361 | 0.227 |
| **ANIMAL** | **0.0005** | 0.143 |

**What drives the shortfall**
- The ANIMAL collapse alone costs about 0.024 of macro AP (0.143 / 6).
- All 27 offset-0 ANIMAL GT are in v09, which is the most negative video.

**Other measures**
- Recall@0.3: 0.70 vs 0.49.
- FP anatomy at score ≥ 0.05: 872 TP vs 434 for R0, with 2,840 BACKGROUND and 2,478 WRONG_CLASS.

**Decision (as pre-declared):** the primary is supportive, so we **proceed to the G5 pre-declaration**.
- Before G5's seeds are spent, the ANIMAL collapse must be diagnosed: it is one of 7 TEST classes (928 small TEST boxes).
- If the fix changes the recipe, G5 must use the fixed recipe and say so.

## Follow-up analyses (2026-10-31; g4f_followup.py G4FU.0, SHA 07dbf9ce…881d; CPU only)
### Offset-5 replication (reported only)
- **Gates:** all 4 parity gates passed. G4F matches curve.json exactly, and R0 e15 matches the M1.1 report.
- **Result:** G4F e15 tiled 0.5157 vs R0 e15 plain 0.3578, **Δ +0.158, 6/9 videos**.
- **Negative videos:** v02 −0.05 and v04 −0.02, both near ceiling (R0 0.996); v09 −0.08, the ANIMAL video.
- **Conclusion:** consistent with the offset-0 primary result (+0.136, 7/9).

### ANIMAL diagnosis
1. **VAL can barely measure ANIMAL.**
   - All 253 VAL ANIMAL GT are in v09: 27 at offset 0, 25 at offset 5. TRAIN has 2,356 boxes across 6 videos.
   - Every model misses 19–24 of the 27 offset-0 boxes, R0 included; they are about 4.7 px at the full-frame network input.
   - R0 e15's 0.143 AP comes from 4 boxes found with high ANIMAL scores (max 0.77).
2. **Our model is not confident on ANIMAL and fires many false ANIMAL predictions.**
   - It finds 1–2 boxes, and its best ANIMAL score on any ANIMAL GT is 0.14 plain and 0.27 tiled.
   - It makes 1,313–1,474 ANIMAL predictions at ≥ 0.05, against 147 for R0 e15.
3. **Tile training causes it (paired evidence).** M2 armF and armT are the same R0 e15 model given one more epoch, without and with tiles.

   | | M2 armF (no tiles) | M2 armT (tiles) |
   |---|---|---|
   | ANIMAL AP | 0.139 | 0.005 |
   | Max ANIMAL score on GT | 0.84 | 0.11 |
   | Found as ANIMAL | 4 | 2 |

   The pilot's tile arms show the same pattern.
4. **Mechanism candidate (TRAIN exposure under the M2 tile rule):**
   - ANIMAL's share of training targets falls from 3.6% in full frames to 2.2% in tiles, the largest relative drop of any class. ANIMAL is kept in 1.63 windows per box, against 2.4–2.9 for the other classes.
   - ANIMAL size at the network input changes from 4.7 px (full frame) to 13.3 px (tile).
   - So tiles both under-sample ANIMAL and teach it at a different scale.
5. **The TEST risk is large.** TEST has 932 ANIMAL boxes. The paper's R50 scores 52.8 ANIMAL AP, its best class. A low-confidence ANIMAL head could cost up to about 7.5 TEST mAP points. VAL cannot tell us how large the loss will be.
