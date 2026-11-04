# G1a pre-declaration: box geometry as a class cue (2026-10-28)

- **Script:** `g1a_geometry_class_cue.py`, version G1a.0, SHA-256 64171e40c7d9da34aba2e467b820fc1e4454cc939d46356361f4b1c50b2db468
- **Data:** TRAIN only, CPU only. Annotations plus the curated TRAIN horizon tracks. No images, no models, no VAL, no TEST.

## Why this design changed from the original "size ratio" idea
- The curated horizon sits close to H/2: median 0.50 on TRAIN.
- A non-linear learner given raw size (h) and image position (y_bottom) can already compute h / (y_bottom − 0.5). The "H/2 ratio" therefore adds no information beyond size + position.
- The only genuinely new information a horizon can add is where it departs from H/2 (roll, pitch, raised or lowered horizons).
- G1a therefore asks two separate questions:
  1. Is box geometry (size, position) informative about class across *unseen videos*? This decides whether the class expert receives geometry metadata, since a crop classifier discards absolute size and position.
  2. Does the curated horizon add anything beyond plain position?
- Prior evidence lowers expectations:
  - In H0, horizon-relative positional conditioning was no better than an abs-y control.
  - In MARSEN, a horizon cue that included h/Δy gave no lift. That was a different task, with a Hough-detected horizon.

## Features
All features are normalized by image H and W. The bottom edge is the contact point. The horizon is roll-corrected at the box x.

| Set | Contents |
|---|---|
| A, size | log(h/H), log(w/H), log(w/h) |
| B, size + position | A + y_bottom, y_centre, x_centre |
| C, B + curated horizon | B + d_bottom, d_centre, sign(d_bottom), log(h/H) − log(\|d_bottom\| + 0.01), roll, horizon mean |

- A and B are evaluated on all 65,542 TRAIN boxes (47 videos, 7 classes).
- A, B and C are evaluated on the 18,494 boxes in frames with a visible horizon (g = 1; 24 videos, 6 classes).

## Protocol
- **Folds:** 5 deterministic folds split by video (greedy balance on box count). Every video sits in exactly one test fold.
- **Learners:** HistGradientBoosting is primary (300 iterations, learning rate 0.08, 31 leaves, L2 = 1, seed 0). Standardized multinomial logistic regression is secondary.
- **Training weights:** class-balanced.
- **Metrics:** balanced accuracy over the classes present in each test fold, plus macro-F1. Both are hand-computed.

## Pass rules (frozen)
| Rule | Condition (primary learner) | Consequence |
|---|---|---|
| R1 | All-frame B pooled balanced accuracy > chance (1/7) + 0.10 | Geometry is informative |
| R2 | All-frame mean(B − A) ≥ +0.02 and B > A in ≥ 4/5 folds | Position adds to size |
| R3 | g = 1 mean(C − B) ≥ +0.02 and C > B in ≥ 4/5 folds | Curated horizon adds |

**What follows from each outcome**
- **If R1 and R2 pass:** the class expert receives geometry metadata (size and position).
- **If R1 passes but R2 fails:** the class expert receives size metadata only.
- **If R3 fails:** horizon features are closed for classification as well. The paper reports that H/2 suffices on stabilized cameras.
- **If R3 passes:** a TEST-time horizon estimator becomes a requirement. Curated horizons cannot be used on TEST.

**What G1a cannot establish**
- It measures geometry *alone*, not geometry's marginal value on top of appearance. That is tested in G2.
- TRAIN class mix differs from TEST, notably LEISURE_VEHICLE, which is small in TEST but absent from TRAIN at that scale.

## Selftest (sandbox, `-W error`): PASS
- Hand-computed features check, including a roll-corrected horizon.
- Metric checks.
- Video-disjoint, deterministic folds.
- Positive control: a varying horizon with a ratio signal, where C beats B.
- Negative control: a fixed horizon at 0.5, where C does not pass R3.
- Null control: near chance.
- The run mode was dry-run on a synthetic dataset in the real schema.

The workstation selftest also passed. The SHA was verified with certutil.

## Results (run 2026-10-28; script SHA 64171e40…db468; train JSON SHA edc32bdb…9b35; tracks SHA 4e4a1151…dd32)
- Inventory check passed: 65,542 boxes, 18,494 of them in frames with a visible horizon (g = 1).

| Learner | Set | Pooled bal. acc | Fold bal. acc |
|---|---|---|---|
| HGB | all_A | 0.236 | 0.14–0.43 |
| HGB | all_B | 0.238 | 0.26–0.42 |
| HGB | g1_A / g1_B / g1_C | 0.207 / 0.246 / 0.278 | wide |
| LogReg | all_A / all_B | 0.232 / 0.236 | 0.23–0.32 |
| LogReg | g1_A / g1_B / g1_C | 0.362 / 0.394 / 0.322 | 0.22–0.63 |

**Verdict (primary learner, HGB): R1 FAIL, R2 FAIL, R3 FAIL.**
- **R1:** all-frame B scored 0.238 against chance 0.143, a margin of +0.095 when +0.10 was required.
- **R2:** Δ(B − A) = +0.033, positive in only 3/5 folds.
- **R3:** Δ(C − B) = +0.065, positive in only 3/5 folds, driven by two folds (+0.24, +0.12). The logistic learner gives Δ = −0.040.

## Interpretation
- Geometry alone is only weakly informative across unseen videos.
- The mapping from geometry to class is unstable: it changes from video to video with camera height, zoom and scene.
- The video-disjoint folds swing by ±0.15, and the flexible learner (HGB) does *worse* than logistic regression on g = 1.
- In per-class recall (HGB, all_B), SHIP and LEISURE_VEHICLE are 0. ANIMAL and MARINE_MARKER are below 0.07.

## Consequences (as pre-declared)
- Geometry is not carried forward as a design element of the class expert.
- G2's primary arm uses appearance plus context only. Size/position metadata may appear solely as a pre-declared ablation arm.
- Horizon features are now closed for classification as well. Together with H0 (detector input) and MARSEN (verifier cue), that is three independent negatives.
- For the paper: on stabilized SeaClips cameras, horizon geometry adds nothing beyond plain image position, and image geometry itself transfers poorly across videos.
