# G5 pre-declaration: three seeds and the single TEST evaluation (2026-10-31)

**Approval:** the Director approved G5 on 2026-10-31 ("Alright let's do it"). This covers about 80 GPU hours for seeds 43 and 44, the TEST inventory, and the TEST evaluation once every model is frozen.

**Status:** Parts A and B are frozen now. Part C (the TEST evaluation tool) is written and selftested on VAL before TEST is touched. It gets its own SHA and is appended here before it runs.

## A. The system under test (frozen)

| Component | Definition |
|---|---|
| Detector | G4F recipe: ConvNeXt-B in22k in R0's Deformable-DETR, COCO transfer, M2 tiles p 0.5, fp32, 15 epochs, R0 optimiser / schedule (g4_full.py G4F.0.1, SHA c1fbed7f…479d) |
| Seeds | 42 (existing G4F run), **43**, **44** (`g5_seed_train.py`; only the seed changes) |
| Checkpoint | **Final epoch 15** of each seed. No VAL checkpoint selection. |
| Inference | 17-tile V0 merge (Stage B full frame + 16 Stage C windows), fp32 |
| Per-class source | For each class, whichever of G4F tiled and R0 e15 plain has the higher VAL AP30 (offsets 0 + 5 pooled), decided once on seed 42 (G4c) |
| Decided routing | ANIMAL ← R0 e15 plain (`checkpoint_epoch_15.pth`); BOAT, MARINE_MARKER, OBJECT, SAILING_VESSEL, SHIP ← G4F tiled; LEISURE_VEHICLE ← G4F tiled (no VAL evidence, so it defaults to the primary detector) |
| R0 for routing | One frozen R0 e15 model for all three seeds |

**Honest caveats**
- **The routing was chosen after seeing VAL numbers** (G4c), so it is a fallback, not a blind selection.
- **ANIMAL comes from the same R0 model in every seed,** so its across-seed variance is zero. The 3-seed SD therefore slightly understates the system's true variance.

## B. Seeds 43 and 44 (frozen)

**Script:** `g5_seed_train.py` G5S.0, SHA-256 624d313df628208a1a06770b431036558423444d57551b59014c316ebfaf500e. The sandbox selftest passes. It runs from `R0_frozen_baseline` and SHA-checks `g4_full.py` and `g4_pilot.py`. It calls `g4_full.run_training` unchanged, with `RECIPE` identical except for `seed`. The output goes to `D:\MARIVISION\G5\seed{S}\run`.

**Per seed:**
- 15 epochs at about 2.1 h each;
- an inline offset-5 VAL curve, reported only (about 14 min per epoch);
- about 34–37 h in total.

**After training:**
- an e15 VAL export at offsets 0 and 5, reported as a sanity check, not a selection;
- `resume` restarts an unfinished epoch from `resume_latest.pth`.

**Proof before the runs:**
- **Selftest:** recipe identity apart from the seed; seed checks; two seeds trained into separate folders; the seed carried in the checkpoint; different models; refusals.
- **Smoke, per seed (GPU, about 10 min):**
  - the wrapper's seed-42 recipe and epoch-1 frame order equal the real G4F run;
  - 40 real steps with seed S, with the seed and frame order verified in the checkpoint;
  - inline eval;
  - refusal to overwrite;
  - memory < 11 GB;
  - disk space;
  - the real run folder not yet started;
  - projections.

## C. TEST evaluation (protocol frozen; tool to follow)

1. **Before:** `g5_test_inventory.py` G5I.0 (SHA-256 029c8e5393a78ea41a708cb8c6d9d10937b615586c1be475f3bc4337b9a33395; the selftest passes) counts TEST frames from the zip file list only. It opens no label file and decodes no image. It sizes the export.
2. **Frames:** every frame of the 18 TEST videos, as in the paper.
3. **Metric, matching the paper:**
   - COCO-style AP at IoU 0.30 (pycocotools, maxDets 100) on original images;
   - **mAP@0.3 = mean over all 7 classes**, including LEISURE_VEHICLE;
   - per-class AP;
   - small / medium / large AP;
   - class-agnostic mAPca@0.3;
   - mAP@0.5.
4. **Runs, each once:**
   - seeds 42 / 43 / 44, routed (**primary**) and unrouted (secondary);
   - R0 e04 and R0 e15 plain (our R50 references);
   - latency per frame on the RTX 3060.
5. **Primary endpoint.** The 3-seed mean routed TEST mAP@0.3 against the paper's published mean ± SD (n = 3):
   - TransVOD++ 31.95 ± 3.28;
   - Swin-B Def-DETR 31.28 ± 3.72;
   - R50 Def-DETR 22.86 ± 0.91.

   The test is one-tailed Welch (Welch–Satterthwaite df) computed from summary statistics, as in the paper, with Bonferroni correction over these 3 comparisons (α = 0.05 / 3).
   - **Claim "outperforms TransVOD++"** only if that comparison is significant.
   - **"Higher mean, not significant"** is reported as such.
6. **Secondary:**
   - routed vs unrouted per seed (the ANIMAL effect on TEST);
   - per-class AP against the paper's Table;
   - robustness on the paper's rain / sun-flare TEST copies (optional, declared before running).
7. **Locks:**
   - TEST is read only by the Part C tool, after the six seed caches (G4F e15 at offsets 0 and 5 already exist) and the seed checkpoints are frozen;
   - no change to any model, routing or threshold after TEST is seen;
   - every result is reported, whatever it shows;
   - TEST predictions and weights stay on the workstation (licence).

## Known limits
- VAL → TEST level transfer is weak (R0 is about 0.42 on VAL vs 22.9 on TEST).
- LEISURE_VEHICLE has no VAL data, and on TEST it is mostly small objects, an unseen scale (G1c).
- Our tiled inference costs about 2 s per frame, against single-pass 640 for the paper's models. Latency is reported next to accuracy.

## Proof results on the workstation (2026-10-31)

**TEST inventory (G5I.0; SHA verified; selftest PASS)**
- **Zip contents:** 7,030 TEST frames in 18 videos (300–900 frames per video); TRAIN 20,310 frames / 47 videos; VAL 4,266 / 9.
- **Projected TEST export:** about 3.7 h per seed with 17 tiles, plus about 0.3 h for R0 plain; about 11.5 h in total.
- **Locked TEST label file:** `seaclips-test.json`, SHA-256 524569f50e9a5820e449cfcb27c34744f43de8bc53acd1b07d7a6dc2701c195d. Its bytes were hashed but never parsed. The Part C tool must refuse any other hash.
- **Other annotation files:** TRAIN JSON SHA edc32bdb…9b35 and VAL JSON SHA 8d3fca68…a324.

**Seed smoke runs (G5S.0; SHA verified; selftest PASS): SMOKE PASS for both seeds 43 and 44**
- **Recipe identity:** the wrapper's seed-42 recipe and epoch-1 frame order equal the stored G4F run.
- **Seed applied:** each checkpoint carries its seed and that seed's frame order.
- **Speed and memory:** 0.77 s per step; training peak 8.03 GB, eval peak 3.5 GB.
- **Refusals and folders:** a fresh run over an existing folder is refused, and the real run folders do not exist yet.
- **Disk:** a checkpoint is 0.38 GB and a resume file 1.13 GB. Both seeds need 14.5 GB; 946 GB is free.
- **Projection:** about 36 h per seed, about 72 h for both.

## Part C tool (written 2026-11-02, before any TEST read)

**Script:** `g5_test_eval.py` G5T.0, SHA-256 ecb20f5681ce914ca25c5a9811c790d58685d748f5a1d0fd8069adc37df12adf. The sandbox selftest passes.
- **Where it runs:** from `R0_frozen_baseline`. It SHA-checks `g4_full.py` and `g4_pilot.py`.
- **Locks:**
  - every TEST mode needs `--test-unlock`;
  - the TEST label file must hash to 524569f5…195d;
  - TEST's class mapping must equal VAL's;
  - each seed checkpoint must be epoch 15 of its declared seed;
  - `score` refuses unless all 5 exports (seeds 42/43/44, R0 e15, R0 e04) are complete and match their checkpoints;
  - `score` writes `g5_test_scored.lock` and never runs a second time.
- **Export:** the same tile schedule, transform and box format as `g4_full.export_with_model`, checked tile by tile in the selftest. It writes resumable parts of 250 frames. R0 models export only the full-frame tile.
- **Scorer:**
  - pycocotools with maxDets 100: mAP@0.3, per class, small / medium / large, mAP@0.5, class-agnostic mAPca@0.3;
  - its plain and tiled values equal `g4_full.score_cache`;
  - routing swaps only ANIMAL, asserted per class.
- **Statistics:** one-tailed Welch from summary statistics. The Student-t tail is checked against tables; Bonferroni α = 0.05/3.
- **Proof on VAL** (`--mode proof`; GPU, about 10 min; TEST untouched):
  - this tool's export reproduces the existing G4F e15 and R0 e15 VAL caches on 20 frames;
  - its scorer reproduces the recorded VAL numbers within 6e-4: G4F e15 tiled 0.5011, routed 0.5248, R0 e15 plain 0.3647, R0 e04 plain 0.4241.

**What the frozen statistics imply** (computed now, before TEST): with the paper's SD of 3.28 at n = 3, "significantly better than TransVOD++" at α = 0.05/3 needs a 3-seed mean of about **40 mAP@0.3** (39.8–40.9 for a seed SD of 1–3), not just above 31.95. A mean between 32 and 40 will be reported as "higher mean, not significant".

## Seed 43 on VAL (2026-11-02; reported only, no decision)
- **Offset-5 tiled macro, epoch 15:** seed 43 0.411 vs seed 42 0.516.
- **Plain:** 0.299 vs 0.440. Training loss is about the same (4.10 vs 4.07).

| Class (off5 tiled, e15) | Seed 42 | Seed 43 | Δ |
|---|---|---|---|
| ANIMAL | 0.001 | 0.002 | ≈0 (collapsed in both) |
| BOAT | 0.469 | 0.313 | −0.156 |
| MARINE_MARKER | 0.520 | 0.489 | −0.031 |
| OBJECT | 0.379 | 0.237 | −0.142 |
| SAILING_VESSEL | 0.946 | 0.844 | −0.102 |
| SHIP | 0.780 | 0.580 | −0.200 |

**Readings**
- **The seed gap sits in the vessel-type classes** (BOAT, OBJECT, SAILING_VESSEL, SHIP), the classes whose cross-video classification the roadmap had already identified as the core failure. MARINE_MARKER is stable.
- **Within a seed, epochs 12–15 vary by about 0.02,** so the gap between seeds (about 0.10) is real seed variance, not evaluation noise.
- **The ANIMAL collapse repeats in seed 43,** so it is a property of the recipe, not of one seed. This supports the routing.

## Seed 44 finished (2026-11-03; VAL, reported only)

**Offset-5 VAL, epoch 15.** AP30; the R0 e15 references are plain 0.358 and tiled 0.346.

| | Seed 42 | Seed 43 | Seed 44 | Mean ± SD |
|---|---|---|---|---|
| Tiled macro | 0.516 | 0.411 | 0.437 | **0.455 ± 0.055** |
| Plain macro | 0.440 | 0.299 | 0.363 | 0.367 ± 0.071 |
| BOAT | 0.469 | 0.313 | 0.546 | 0.443 |
| MARINE_MARKER | 0.520 | 0.489 | 0.479 | 0.496 |
| OBJECT | 0.379 | 0.237 | 0.238 | 0.285 |
| SAILING_VESSEL | 0.946 | 0.844 | 0.716 | 0.835 |
| SHIP | 0.780 | 0.580 | 0.643 | 0.668 |
| ANIMAL | 0.001 | 0.002 | 0.001 | collapsed in all 3 |

- **Every seed's tiled result is above R0 e15 plain** (+0.158 / +0.053 / +0.079), a mean of +0.097.
- **The plain mean (0.367) is roughly equal to R0 plain (0.358).** The gain comes from tiled inference on top of the stronger backbone, not from the backbone alone.
- **Seed 42 is the best of the three.** The G4 decisions, which were made on seed 42, were therefore made on an optimistic draw. This is recorded for the paper.
- **MARINE_MARKER is the most stable class across seeds; SAILING_VESSEL, SHIP and BOAT vary most.**
- **ANIMAL collapses in all three seeds.**
- **Routing adds the same ANIMAL term to every seed.** On offset 5 that is about +0.015 macro (G4c: 0.5157 → 0.5305), so the routed VAL values would be about 0.53 / 0.43 / 0.45.

## Part C proof on the workstation (2026-11-03): PROOF PASS
- **Script check:** the SHA verified (ecb20f56…2adf) and the selftest passed.
- **Export parity on 20 VAL frames:**
  - seed 42: 340 tiles, 0 mismatches, max box deviation 0.0 px;
  - R0 e15 plain: 0 unmatched detections, max 0.009 px.
- **Scorer parity on the full offset-0 VAL caches:**

  | | Tool | Recorded |
  |---|---|---|
  | G4F e15 tiled | 0.50115 | 0.5011 |
  | G4F e15 routed | 0.52482 | 0.5248 |
  | R0 e15 plain | 0.36473 | 0.3647 |
  | R0 e04 plain | 0.42409 | 0.4241 |

- **TEST label file:** the SHA matches the lock.
- **Measured export speed:** 1.94 s/frame for the 17-tile seed, 0.13 s/frame for R0 plain. That projects TEST at 3.8 h per seed and 0.25 h per R0 model.
- **Noted for the report:** routing lowers the class-agnostic mAPca@0.3 on VAL (tiled 0.529 → routed 0.491), because R0's ANIMAL boxes are added on top of G4F's boxes for the same objects. mAPca is a secondary metric, and both values will be reported.

## G5 TEST RESULT (2026-11-04): single evaluation, 7,030 frames, G5T.0, lock written

**Inputs (checkpoint SHAs):**
- seed 42 b393a1a5…fef6;
- seed 43 2211b306…2871;
- seed 44 44bbd0e2…d463;
- R0 e15 64f9b4de…dac0;
- R0 e04 9e4abee6…ed4f.

### Primary (pre-declared): routed system, mAP@0.3 = **35.29 ± 1.09** (seeds 35.29 / 36.39 / 34.21)

| vs paper (mean ± SD, n = 3) | Diff | Welch t | df | One-tailed p | Significant at 0.05/3 |
|---|---|---|---|---|---|
| TransVOD++ Swin-B 31.95 ± 3.28 | +3.34 | 1.68 | 2.44 | 0.106 | no |
| Def. DETR Swin-B 31.28 ± 3.72 | +4.01 | 1.79 | 2.34 | 0.098 | no |
| Def. DETR R50 22.86 ± 0.91 | +12.43 | 15.14 | 3.87 | 0.0001 | **YES** |

**Verdict under the frozen rule:**
- The pre-declared system has a **higher mean than every model in the SeaClips paper**.
- It is **significantly better than R50**.
- It is **not significantly better than TransVOD++ or Swin-B**. "Higher mean, not significant" is the claim.

### Secondary (pre-declared): unrouted G4F tiled = **38.16 ± 1.41** (36.59 / 39.33 / 38.56)
- **Routing hurt on TEST in every seed:** −1.30 / −2.94 / −4.35.
- **Why:** on TEST, the G4F tiled ANIMAL AP is 39.3 / 50.8 / 60.7, against R0 e15's 30.2. VAL ANIMAL (one video, 52 tiny birds) did not represent TEST ANIMAL.
- **This is exactly the risk recorded in G4c's caveats.** The routing decision was wrong. It stays in the record and in the paper as a lesson: a per-class selection made on an unrepresentative validation class transferred negatively.
- **Exploratory only** (not pre-declared as a hypothesis test, so it is not a claim): Welch for unrouted vs TransVOD++ p = 0.032 and vs Swin-B p = 0.035. Neither is below 0.0167, so even the better variant is not significant after Bonferroni.

### Other TEST observations (descriptive)

**Tiled inference is the main source of the gain.**

| Seed | Plain | Tiled | Gain |
|---|---|---|---|
| 42 | 20.6 | 36.6 | +16.0 |
| 43 | 25.5 | 39.3 | +13.9 |
| 44 | 25.3 | 38.6 | +13.2 |

**Small-object AP of 30–32 (tiled)** is above every model in the paper (best: RT-DETR 24.98, Swin-B 22.93).

**Per class (tiled, 3 seeds) vs TransVOD++:**

| Class | Ours | TransVOD++ | |
|---|---|---|---|
| BOAT | 63–65 | 48.1 | better |
| SAILING_VESSEL | 57–65 | 36.9 | better |
| LEISURE_VEHICLE | 21–29 | 10.6 | better, despite no small LV in TRAIN |
| OBJECT | 18–29 | 3.1 | better |
| ANIMAL | 39–61 | 59.8 | |
| MARINE_MARKER | 14–20 | 16.6 | about equal |
| **SHIP** | **17.6 / 42.3 / 22.9** | **48.6** | **worse and unstable** |

SHIP is the weakest class, consistent with the VAL seed analysis: vessel-type classification across videos is fragile.

**Our R0 on TEST:** e15 20.72 and e04 19.04, both below the paper's R50 at 22.86 ± 0.91. Our evaluator and checkpoints therefore do not inflate R50. The internal same-pipeline gain of the routed system over our R0 e15 is +14.6.

**Other metrics:**
- mAP@0.5: routed 22.0–24.8, tiled 25.3–28.9 (paper: TransVOD++ 23.50, Swin-B 21.70).
- Latency: about 1.92–1.97 s/frame tiled, about 2.05–2.10 s/frame routed, on the RTX 3060. The paper's models use a single 640 pass.

**Limits of the comparison**
- The paper's numbers come from its authors' pipeline; ours come from our pycocotools evaluator. The protocol matches as documented (AP@0.3 over 7 classes, original-image boxes, all TEST frames). The internal R0 reference is the same-pipeline anchor.
- Only 3 seeds.
- Rain and sun-flare robustness were not run.
