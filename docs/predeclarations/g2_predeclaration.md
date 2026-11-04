# G2 pre-declaration: crop + context class expert vs R0 on the same VAL boxes (2026-10-28)

- **Script:** `g2_crop_expert.py`, version G2.0, SHA-256 b48d94cc67ed43b50c991c4a7de1c66d2c7ef2d69fcf836f86fb0610d46c2327
- **Inputs:**
  - TRAIN and VAL jsons.
  - `SeaClips.zip`, read-only and never extracted.
  - R0 epoch04 full-VAL export: 426,600 detections = 4,266 frames × top-100.
- **Excluded:** TEST is not used. The cache and the run directory are dataset-derived and never leave the workstation.

## Question
Given a box, does a classifier trained only on TRAIN ground truth name the class on *unseen VAL videos* better than R0 does on the same box?

Why this is the next test:
- G1b-0 showed R0's cross-video class errors are persistent per object. On the g = 1 subset, 27.6% of covered GT boxes are wrong, and a vote along the track fixes only about 5%.
- G1a showed geometry alone does not carry over between videos.
- What is left to try is appearance plus context, learned by a model that never sees R0's errors.

## Expert (all settings frozen in `CFG`)
**Input view**
- A square context window centred on the box, with side S = clip(4·max(w, h), 64, 1024) native px, taken from native-resolution frames.
- It is stored at 1.5 S (192 px), so that scale and shift jitter have room.
- The network sees S resized to 128 px.

**Model and training**
- Torchvision ConvNeXt-T, ImageNet-1k pretrained, with a 7-way head.
- AdamW: learning rate 2e-4, weight decay 0.05, 200-step warmup, then cosine decay.
- Batch 64 with AMP, label smoothing 0.1, 8 × 30,000 samples (3,750 steps).
- The *last* checkpoint is kept. There is no VAL selection, so no winner's curse.

**Data and sampling**
- TRAIN GT boxes from every 2nd frame (about 32.8k crops).
- Class-balanced sampler: class share ∝ n_c^0.5.

**Augmentation**
- Scale jitter ×0.85–1.15 and centre jitter ±8% of S.
- Horizontal flip.
- Brightness, contrast and saturation each ±30%, plus 10% greyscale and light noise.

**LV design (G1c D1/D2)**
- With probability 0.7, an LV sample is replaced by a shrink + paste version:
  - The LV is shrunk (never enlarged) to a √area drawn from the TRAIN BOAT + SAILING_VESSEL √area distribution.
  - It is pasted with feathered blending into a 256 px native background window. Windows come from TRAIN videos *without* LV, near the horizon band (0.45–0.62 H), and contain no GT box.
- **Geometry metadata:** none, because G1a failed.

## Primary readout and pass rule
**Covered set**
- VAL GT boxes (all 9 videos) for which R0 has an entity with IoU ≥ 0.3 and obj ≥ 0.3. When several qualify, the highest-obj entity is used; this is the G1b-0 rule.
- An entity is a group of same-box detections, i.e. one Deformable-DETR query.

**What is compared**
- R0's class = argmax of that entity's class scores.
- The expert's class = argmax on a crop of **the same R0 entity box**, not the GT box. This is the deployment situation and keeps the comparison paired.
- Any LV prediction on VAL is counted as wrong, because VAL has no LV.

**PASS requires all of:**
1. Balanced accuracy (expert − R0) ≥ +0.05, over the classes present in the covered set.
2. Expert accuracy ≥ R0 accuracy in at least ⅔ of the VAL videos with ≥ 30 covered boxes.

**Also reported:**
- Plain accuracy and per-class recall for both.
- Per-video accuracy.
- Confusion matrices.
- Fixed and broken counts.
- Expert balanced accuracy on *all* GT boxes using the GT box itself (oracle localisation, including boxes R0 misses).

## Secondary readout (descriptive; informs the G4 design, cannot rescue a primary fail)
**Frames:** the 432 VAL frames at stride 10, offset 0 (1,386 GT, same protocol as V13/M2).

**Arms** (all R0 entities rescored; top-100 per frame; COCO mAP@0.3):

| Arm | Rule |
|---|---|
| A0 | Raw R0 |
| E1 | obj · normmax(p_expert) |
| E2 | obj · normmax(0.5 p_R0 + 0.5 p_expert) |

**Parity gate:** the rebuilt A0 must equal the raw export's mAP on the same frames (tolerance 1e-9). The full-VAL raw mAP30 is also printed; 0.4245 is expected.

**Reported:** pooled and per-video Δ.

## Guards
- **LV false positives (G1c D4):** the number of confident VAL entities (obj ≥ 0.3) the expert labels LV, and LV predictions on covered boxes. All of these are wrong.
- **Input gates:** TRAIN 20,310 / 65,542; VAL 4,266 / 13,699; export 426,600 and ≤ 100 per frame; 432 frames / 1,386 GT.
- **Refusals:** the build and train modes refuse to overwrite an existing cache or checkpoint.

## LV held-out check (G1c D3; descriptive only)
- Trained twice, with `--holdout-video video_23` and `--holdout-video video_18`.
- Held-out videos are excluded from crops and LV patches. LV videos are never used as background sources.
- Readouts:
  - LV recall on the held-out video's native crops.
  - LV recall at fixed √area of 32, 16 and 8 px, pasted onto VAL backgrounds with a fixed seed.
- With only two objects, this is an indication, not a test.

## Order of operations (proof before any long run)
1. **selftest** (CPU): crop geometry; LV shrink + paste size; offset-0 subset; sampler shares; paired-readout arithmetic; rescoring identity (E1 fed R0's own scores = A0); end-to-end build → train → eval on a synthetic zip (expert beats a 40%-wrong R0); LV held-out path. Spawn-mode DataLoader workers were also checked.
2. **smoke** (GPU, a few real frames: 4 from each of 8 TRAIN videos including both LV videos, and 20 VAL frames):
   - Real zip reading and the mini cache.
   - Pretrained-weight download.
   - Overfit test: 64 crops, 150 steps, train accuracy ≥ 0.9.
   - Real training loop with workers.
   - Peak memory < 10 GB.
   - Full eval path.
   - Projections of build time, training time and cache size.
3. **build**, then **train**, then **eval**. Expected: about 9–10 GB of cache and under an hour of GPU time. The smoke report gives the actual projections.

## Consequences
**PASS**
- The class expert becomes the core of G4.
- E1/E2 show whether a simple rescoring already lifts mAP.
- Next is a backbone feasibility smoke (G3) for a stronger expert and proposer.

**FAIL**
- ImageNet-pretrained appearance + context at this capacity does not transfer across SeaClips videos either.
- The class-confusion problem is then a representation-capacity problem (Swin-B scale), or a genuine label-ambiguity problem.
- Next: inspect the expert's confusion matrix and the oracle-box accuracy before choosing between a stronger backbone (G3) and class-definition analysis (e.g. OBJECT vs. BOAT at distance).

## Amendment G2.0.1 (2026-10-28, before training; no change to the method)
- **What happened:** `--mode train` failed on the workstation before any training step (`PIL _imaging` DLL load error). torchvision imported PIL *after* torch.
- **Fix:** `from PIL import Image` now comes at module top, before any torch import. Nothing else changed.
- **New script:** G2.0.1, SHA-256 2dd48cd657432406199fe1c5245fad5ce3d1ef65a6a0a1df66b2777b101e46db.
- **The cache built by G2.0 stays valid**, because the build code is unchanged:
  - Build time: 1,058 s.
  - 32,819 TRAIN crops, 300 LV patches, 2,979 TRAIN backgrounds.
  - 13,699 VAL GT crops, of which 6,141 are R0-covered: these form the primary comparison set.
  - 19,973 R0 entities on the 432 frames, and 495 VAL backgrounds.

## Results (run 2026-10-28; G2.0.1, SHA 2dd48cd6…46db; 3,750 steps in 321 s; cache from G2.0)
- **Parity:** passed. The rebuilt A0 equals the raw export on the 432 frames (0.424109). Raw full-VAL mAP30 is 0.424488, matching the expected 0.4245.

| Primary, paired on 6,141 R0-covered VAL GT | R0 | Expert |
|---|---|---|
| Accuracy | 0.694 | 0.647 |
| Balanced accuracy | 0.722 | 0.704 |

**Verdict: FAIL.**
- Δ balanced accuracy = −0.018, against the required +0.05.
- The expert was not worse in 6 of 9 videos, which meets the video condition, but the accuracy condition fails.
- It fixed 386 boxes and broke 669.

**Where the errors are**
- R0's 1,882 argmax errors are almost all in two classes:
  - MARINE_MARKER: 1,064 wrong, read as SAILING_VESSEL 454, OBJECT 452, BOAT 107.
  - SHIP: 628 wrong, read as BOAT 562.
- Two videos carry most of them: video_03 (R0 accuracy 0.36) and video_01 (0.48).
- The expert adds two error modes of its own:
  - MARKER → ANIMAL 421 and BOAT → ANIMAL 133. Small blobs, once scale-normalised, look like birds; the sampler had raised ANIMAL to about 8% of training samples.
  - SHIP recall collapses (0.13 → 0.02); SHIP is read as BOAT 492 and as OBJECT 332.
- Expert balanced accuracy on GT boxes over *all* VAL GT is 0.567.

**Secondary (mAP30 on 432 frames)**

| Arm | mAP30 | Δ |
|---|---|---|
| A0 | 0.4241 | – |
| E1 | 0.2909 | −0.133 |
| E2 | 0.3508 | −0.073 |

- SHIP AP collapses from 0.840 to 0.102 (E1) and 0.395 (E2).

**LV guard**
- 59 covered VAL boxes and 6 confident VAL entities were labelled LV. All are false positives.

## Interpretation
1. **A post-hoc, crop-based class expert does not carry over between videos better than R0 does.** That holds even with ImageNet pretraining, native-resolution context and class balancing.
2. **mAP depends on how each class's scores rank across frames, more than on which class wins on each box.** Key evidence:
   - R0's SHIP argmax recall on covered boxes is only 0.13, yet its SHIP AP on the 432 frames is 0.84.
   - R0 is multi-label (sigmoid, top-100). It emits SHIP as a *secondary* class on the same box, and AP ranks SHIP entries across frames.
   - Replacing or blending R0's class distribution destroys that information.
   - So "wrong-class" argmax errors are much less costly than they look. The wrong-class FPs that do hurt are the ones that outrank true positives *within* a class.
   - A post-hoc classifier would need *calibrated, ranked per-class scores* that beat R0's, not better argmax accuracy.
3. **Scale-normalised crops discard the cue that separates SHIP from BOAT and MARKER from ANIMAL.** That cue is absolute apparent size, judged together with appearance and scene.
   - It explains both of the expert's new error modes.
   - It is consistent with G1a, where geometry alone was weak and SHIP recall was 0 from geometry alone.
   - The detector sees both appearance and size; the crop expert sees only appearance.
4. **The concentration in video_01 and video_03 suggests video-specific shift**, such as unfamiliar marker types, or label ambiguity. The Director can inspect this locally. Images are not shared.

## Consequences
- **The decomposed "proposer + post-hoc class expert" route is closed.** It joins temporal class voting (G1b-0), geometry and horizon cues (G1a), and in-sample second stages (B-FCEF-F) in the ledger.
- **The LV held-out runs (D3) are skipped.** Their purpose was to validate the expert, which is now closed. The LV question moves into detector training (G1c D1–D4).
- **The next gate is G3**, an end-to-end detector with a stronger backbone. It keeps the multi-label per-class ranking that G2 shows mAP depends on, and adds native resolution (the only positive VAL evidence, PoL-4 and M2) and LV augmentation. This matches where the paper's gains come from (Swin-B +8.4 mAP on TEST, mostly SHIP, BOAT and SAILING_VESSEL).
- **For the paper:** a negative result with a mechanism. On SeaClips, cross-video errors are not fixed by a better crop classifier, because mAP@0.3 rewards how each class's scores rank across frames, and that ranking comes from the detector's joint, scale-aware scoring.
