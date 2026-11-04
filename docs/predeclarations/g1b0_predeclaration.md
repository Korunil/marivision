# G1b-0 pre-declaration: track-level class vote on R0 VAL predictions (2026-10-28)

- **Script:** `g1b0_track_class_vote.py`, version G1b-0.0, SHA-256 e173df9ea963c6832b2e8fe09e59124ad83ddaead2870ed62d54addcaff92956
- **Inputs:**
  - `cache_dt_m0_g1.json`: R0 epoch04 predictions, top-100 per frame, made by `audit_video05_and_gateoff.py`.
  - `seaclips-val.json`.
- **Scope:** the dense g = 1 VAL stratum, 2,403 frames from 6 videos (03, 04, 05, 06, 08, 09).
- **Excluded:** no model, no images, no TEST, no fitted parameters.

## Question
Is R0's cross-video wrong-class error *temporal flicker* or *persistent*?
- **Flicker:** the right class wins in neighbouring frames of the same object. A causal track vote would then fix it at no cost.
- **Persistent:** the object is misread consistently. A vote cannot help, and the class expert (G2) must bring new information.

## Method (all hyperparameters frozen before the run)
**Entities**
- Detections in a frame that share an identical bbox come from the same Deformable-DETR query. Each such group is one entity with a class-score vector s.
- obj = max s, and p = s / obj.

**Causal linking (per video, in frame order)**
- Entities are processed in descending obj. Entities with obj < 0.05 are passed through unchanged.
- Each entity takes the nearest live track, with the gate: centre distance / max(√(w·h) of the track box, 4 px) ≤ 1.0.
- The area ratio must lie within [¼, 4].
- One entity per track per frame. A track dies after a frame gap greater than 3.

**Arms**

| Arm | Rule |
|---|---|
| A0 | Identity: the cache rebuilt from entities. It must equal the raw cache exactly. |
| A1 (primary) | s′ = obj · p̂, where p̂ is the obj-weighted mean of p over the last 15 track members (current included), normalised so its max is 1 |
| A2 | Blend: s′ = obj · normmax(0.5 p + 0.5 p̂) |
| A3 | Offline whole-track vote (uses future frames) |

**Evaluation**
- Per frame, keep the top 100 (box, class, score) entries.
- COCO mAP at IoU 0.3, maxDets 100: the same protocol as the H0 parity scripts.
- Reported for the pooled stratum and per video, plus per-class AP, class-agnostic AP and AR.
- Per-video values come from masked re-accumulation of one evaluation. The selftest checks them against separate per-video evaluations.
  - Passing a reduced `params.imgIds` to pycocotools `accumulate()` is **not** used: it indexes the wrong images (a pycocotools bug found while building this).

**Diagnostics**
- For each GT box, take the highest-obj entity with IoU ≥ 0.3 and obj ≥ 0.3. Count whether its argmax class is right or wrong before each arm, and how many are fixed or broken.
- The class-switch rate between consecutive members of each track (obj ≥ 0.3).
- Track counts and lengths.

## Parity gates (any failure stops the run; nothing is interpreted)
1. Cache metadata: epoch 4, unconditioned, 2,403 images. The image-id hash must equal the stored hash.
2. At most 100 detections per image. The videos must be exactly 03, 04, 05, 06, 08 and 09.
3. Raw-cache mAP30 must match the M0 sidecar: within 1e-4 if the sidecar is found, otherwise 0.4109 ± 6e-5.
4. A0 rebuild equals raw exactly, for both mAP and mAPca. No entity splits.

## Pass rule (primary arm A1 only; A2 and A3 cannot rescue a fail)
- Δ mAP30 (A1 − A0) ≥ +0.010.
- Positive in ≥ 4 of 6 videos.
- Worst per-video Δ ≥ −0.010.

**Why mAPca is reported but not gated:** the vote raises the scores of secondary-class duplicates of the same box. Class-agnostic AP therefore drops mechanically even when the vote is right. The selftest showed this: +0.38 mAP alongside −0.06 mAPca.

## Consequences
**PASS**
- A causal track vote becomes a G4 component: a free post-process.
- The class expert's error budget is reduced by the fixed share.
- A3 − A1 shows what an offline (non-causal) variant would add.

**FAIL**
- Wrong-class error is persistent per object, not flicker.
- This closes the "temporal smoothing of class" route. It joins learned temporal fusion (2F, 2M, 3V, MARSEN) in the ledger.
- It strengthens the case that G2 must add new appearance or context information.
- If A3 passes while A1 fails: exploratory only. It is recorded, not adopted.

## Known limits
- g = 1 frames only (6 of 9 VAL videos). The cache was built for the H0 work.
- Classes below R0's top-100 for a query are invisible to the vote.
- R0 epoch04 was VAL-selected. The paired Δ is unaffected by the level, but this carries winner's-curse optimism.

## Selftest (sandbox, `-W error`): PASS
- **Metric:** perfect detections give mAP 1.
- **Identity:** the rebuild equals raw.
- **Per-video:** re-accumulation equals separate evaluations.
- **Positive control (random flicker):**
  - A0 0.578 → A1 0.793 (A3 0.930).
  - 6/6 videos positive.
  - 18/18 object tracks recovered.
- **Negative control (persistent misread):** no gain. Not a pass.
- **Null control (stable classes):** no change. Not a pass.
- **Causality:** perturbing frames after t leaves A1 scores at t unchanged.
- **Top-k and gate:** checks pass.
- **Dry run:** a full-size synthetic cache in the real schema (233k detections, 2,403 images) passes all gates. It takes about 2 minutes on CPU.

## Results (run 2026-10-28; script SHA e173df9e…2956; cache SHA 4d9f7e17…76b0; val SHA 8d3fca68…a324)
- **Gates:** all passed. The raw cache gave 0.410851, matching the M0 sidecar. The A0 rebuild was identical. No pycocotools warnings.

| Arm | mAP30 | Δ | Videos + | Worst video Δ | Meets rule |
|---|---|---|---|---|---|
| A0 | 0.4109 | – | – | – | – |
| **A1 (primary)** | 0.4236 | +0.0128 | **3/6** | **−0.021** (video_06) | **No** |
| A2 (blend) | 0.4315 | +0.0206 | 4/6 | −0.000 | Yes; exploratory, cannot rescue |
| A3 (offline) | 0.4261 | +0.0152 | 4/6 | −0.000 | Yes; exploratory, cannot rescue |

**Verdict: FAIL** (A1 has only 3/6 videos positive, and video_06 loses 0.021).

## Interpretation
**Wrong-class error is persistent, not flicker**
- Of the 4,282 GT boxes covered by a confident entity, 1,182 (27.6%) carry the wrong argmax class.
- The causal vote fixes 57 of them (4.8%) and breaks 44 right ones.
- Along tracks, the class changes in only 4.1% of consecutive confident pairs.
- So R0 misreads an object the same way frame after frame.

**The pooled gain is one class in one video**
- SHIP AP goes 0.661 → 0.758 (A1) and 0.783 (A2). All g = 1 SHIP boxes are in video_05.
- SHIP alone contributes about +0.016 to the pooled mean (+0.098 / 6 classes). The other five classes net about −0.004.
- ANIMAL halves (0.036 → 0.017).
- The gain comes from re-ranking scores *within* SHIP, not from fixing argmax classes.

**Videos 08 and 09 are unchanged**
- Their classes are already stable along tracks, so the vote has nothing to change.

## Consequences (as pre-declared)
- The "temporal smoothing of class" route is closed. It joins learned temporal fusion (2F, 2M, 3V, MARSEN) in the ledger.
- Wrong-class error is a *per-object, per-video systematic* misreading. Fixing it needs new appearance or context information, which is exactly what G2 tests.
- **A2 (blend) is recorded, not adopted.** It may be re-proposed as a free post-process in G4 only under a fresh pre-declaration, calibrated on VAL offset-5 and judged on offset-0. The evidence for it is one class in one video.
- **For the paper:** on SeaClips, R0's cross-video class errors are temporally consistent. Only about 5% are temporally correctable. This is direct evidence that the bottleneck is representation, not temporal aggregation, and it is consistent with TransVOD++ gaining little over Swin-B (31.95 vs 31.28).
