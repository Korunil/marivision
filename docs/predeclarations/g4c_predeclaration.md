# G4c pre-declaration: ANIMAL anatomy and class routing (2026-10-31)

- **Script:** `g4c_animal_anatomy.py`, version G4C.0, SHA-256 fb4e8249bba2c128a264796c0f55490d8e790a678687e3c3ec503343ae6315c5. The sandbox selftest passes (`-W error`).
- **Where it runs:** from `D:\MARIVISION\H0_workbench\eval_tools`, next to `g4f_followup.py` (G4FU.0, SHA-checked 07dbf9ce…881d). It reuses that script's cache → plain/tiled prediction path, pycocotools AP@0.3 and GT-fate rule unchanged.
- **Approval:** the Director approved G4c on 2026-10-31.
- **Compute:** CPU only and read-only. No model, no images, no GPU, no TEST labels. TEST enters only as the paper's public size bins (supplement Table 1: ANIMAL 928 small / 4 medium / 0 large).
- **Inputs:** 10 VAL tile caches at offsets 0 and 5 for R0 e15, G4F e15 and G4b CTRL / NOTILE / RFS. The six G4b caches are pinned by full SHA; a mismatch refuses. The SHAs of all inputs are written to the report.

## Why
G4b found no winner. No arm restored ANIMAL (pooled AP about 0.0005 in every G4-family model, vs 0.079 for R0 e15 tiled). RFS raised ANIMAL confidence on real birds but also raised the false ANIMAL detections. Before any further training, we need to know where the failure sits.

## Questions (descriptive, no pass rule)
1. **Q1, anatomy of ANIMAL predictions ≥ 0.05** (offsets 0 + 5 pooled, plain and tiled). Each prediction is classified, greedily by score:
   - TP;
   - duplicate;
   - wrong class (and which GT class it lands on);
   - on a crowd box;
   - background.

   Also reported:
   - whether it falls in v09 (the only VAL video with ANIMAL) or another video;
   - for tiled predictions, whether a same-class full-frame prediction backs it (IoU ≥ 0.5) or it is tile-only;
   - the count scoring ≥ 0.3;
   - the number of false positives scoring above the best true ANIMAL match.
2. **Q2, fate of each pooled ANIMAL GT** (g4f_followup rule, score ≥ 0.05, IoU ≥ 0.3): found as ANIMAL / found as another class / low score / missed.
3. **Q3, ANIMAL size census.** For TRAIN per video vs VAL: √area at native resolution and at the 640 network input (full frame and a Stage C tile), the COCO size bins, and the share below 8 px. The TEST public bins are given for comparison.
4. **Q4, class routing.** For each G4-family model X, the five non-ANIMAL classes come from X tiled, and the ANIMAL detections come from:
   - **SELF:** X's own plain (full-frame) output;
   - **R0:** R0 e15's plain output.

   AP is computed per class, so the five other classes are unchanged by construction; the script asserts this to 1e-12.

## Frozen rule (Q4 only): G5 inference configuration
For X = G4F e15:
- **SELF** if its pooled ANIMAL AP30 ≥ 0.05;
- otherwise **R0** if its pooled ANIMAL AP30 ≥ 0.05;
- otherwise **none**.

SELF ranks first because it needs only one model.

## Honest caveats
- **This rule was written after seeing R0's VAL ANIMAL numbers** (R0 e15 plain 0.143 on offset 0), so the R0 route is expected to clear 0.05. It is an engineering fallback, not a blind selection.
- **TEST is the real test.** G5 will report routed and unrouted TEST results side by side and state the routing's history in the paper.
- **Inference cost:** routing to R0 adds one R50 full-frame pass per frame (on top of 17 ConvNeXt-B tiles); latency will be reported.
- **Q1–Q3 are descriptive** and only shape the next design. Nothing in them starts a GPU run without the Director's approval.
- **VAL ANIMAL is one video (v09), 52 GT in the pooled subsets.**

## Proof before the run
**Selftest** (sandbox CPU, synthetic caches). It checks:
- the anatomy against a hand-built case: TP / duplicate / wrong class (→ GT class) / crowd / background, video, tile-only vs full-frame-backed, ≥ 0.3, FP above the best TP;
- that routing keeps the other classes and swaps only ANIMAL;
- the rule order SELF > R0 > none, including a missing value;
- end to end with two synthetic models over offsets 0 + 5:
  - fates;
  - the R0 route reaching ANIMAL AP 1.0 with the other classes asserted unchanged;
  - census sizes against a hand computation;
  - the report;
  - refusal to overwrite an existing report.

## Outputs
`eval_tools\g4c_reports\g4c_report.md` and `.json`. These are dataset-derived and stay on the workstation.

## Results (2026-10-31): routing rule → **R0**

**Q4 routing.** All columns are AP30 macro; ANIMAL is pooled over offsets 0 + 5.

| Model (tiled) | Unrouted off0 / off5 | R0-routed off0 / off5 | ANIMAL pooled: unrouted / SELF / R0 |
|---|---|---|---|
| G4F e15 | 0.5011 / 0.5157 | **0.5248 / 0.5305** | 0.0005 / 0.0001 / 0.1160 |
| CTRL | 0.4912 / 0.5134 | 0.5149 / 0.5281 | 0.0005 / 0.0000 / 0.1160 |
| NOTILE | 0.4781 / 0.4973 | 0.5018 / 0.5120 | 0.0005 / 0.0001 / 0.1160 |
| RFS | 0.4916 / 0.5090 | 0.5153 / 0.5237 | 0.0005 / 0.0000 / 0.1160 |

- **SELF fails:** G4F's own full-frame ANIMAL AP is 0.0001.
- **R0 passes,** at 0.116 ≥ 0.05.
- **G5 inference configuration:** G4F tiled for the 5 other classes + R0 e15 plain for ANIMAL.
- **Reference:** R0 e15 plain macro 0.3647, tiled 0.3459.
- **Backbone alone:** G4F e15 plain macro is 0.4301 vs R0 e15 plain 0.3647, so the backbone gives +0.065 without tiled inference.

**Q2 and Q1: the G4 ANIMAL failure is precision, not recall**
- **Recall is similar.** G4F tiled finds 4/52 birds and R0 finds 6/52; 46 are missed by both. VAL birds have a median √area of 9.4 px native and 4.2 px at the 640 input.
- **Ranking is what differs.** R0 has 0 false positives above its best true bird (0.85–0.89). G4F tiled has 341 above its best (0.35).
- **G4F tiled emits 2,897 ANIMAL false positives ≥ 0.05**, 2,882 of them outside v09, and 368 of them ≥ 0.3:
  - **wrong class:** 1,147, mostly on MARINE_MARKER (833), then BOAT 162 and SHIP 130;
  - **background:** 1,750.
- **The false positives concentrate in video v03:** 1,886 of 2,897.
- **This is not a tiled-inference artefact.** 2,258 of the 2,897 tiled false positives are backed by a same-class full-frame prediction, and the plain output alone has 2,540 false positives. The tile-trained ConvNeXt has learned to call tiny blobs (buoys, clutter) ANIMAL.
- **The G4b arms only reshuffle these numbers:** RFS 2,961; CTRL 2,254; NOTILE 2,283.

**Q3: ANIMAL census**
- **TRAIN is heterogeneous:** 6 videos. Three hold very small birds (median native √area 3.5 / 4.9 / 6.0 px; 100% < 8 px). Three hold larger ones (10–16 px).
- **Overall, 33% of TRAIN ANIMAL boxes are < 8 px native.** VAL v09 has a median of 9.4 px, with 22% < 8 px.
- **TEST:** 928 of 932 boxes are small. The finer TEST distribution is unknown, and R50 reaches 52.8 AP there.
- **At the 640 full-frame input, TRAIN ANIMAL is a median of 4.7 px** (q10 2.4 px). The labels themselves teach "a few-pixel blob = ANIMAL", which plausibly drives the marker confusion once tiles add scale variety.

## Director's design intent (2026-10-31), recorded before G5
"When our detector score is below R0 then it should select R0 detection."

This is formalised as **per-class source selection**, to be frozen in the G5 pre-declaration:
- **Choosing the source:** for each class, the source is whichever of G4F tiled and R0 e15 plain has the higher VAL AP30 (offsets 0 + 5 pooled). The choice is made once on seed 42, then applied unchanged to every seed and to TEST.
- **Applied to the existing numbers,** only ANIMAL routes to R0 (offset 0: R0 0.143 vs G4F 0.0005). G4F wins on BOAT, MARINE_MARKER, OBJECT, SAILING_VESSEL and SHIP. This reproduces the G4c rule exactly.
- **LEISURE_VEHICLE has no VAL boxes:** with no evidence either way, it defaults to the primary detector (G4F).
- **Not adopted: comparing individual detection scores between the two models.** Their scores are not calibrated to each other: on the same birds, G4F peaks at 0.35 and R0 at 0.85–0.89. A raw score comparison would be arbitrary.
