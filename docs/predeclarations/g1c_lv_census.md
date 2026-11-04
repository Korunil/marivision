# G1c: LEISURE_VEHICLE census on TRAIN (2026-10-28)

- **Script:** `g1c_lv_census.py`, version G1c.0, SHA-256 31396a974613d7c8a023340a52bbcc61e0008985082c502ba2fabdedfed2e31b
- **Data:** TRAIN annotations only (SHA edc32bdb…9b35; inventory 20,310 images / 65,542 boxes). CPU only. No images, no VAL, no TEST labels.
- **Comparison data:** public TEST counts from the paper supplement, Table 1.
- **Type of gate:** a census, not a hypothesis test. It fixes the design of the LV augmentation and validation.

## Findings
**1. TRAIN LV is effectively two objects**
- There are 600 boxes in 2 videos, video_18 and video_23. Each video has 300 frames, and each frame contains exactly one LV.
- Both videos come from the same camera and conditions: Axis sensor, sunny, smooth sea.
- The paper defines LV as non-motorized vehicles (rowboat, windsurfer). TRAIN aspect ratios (w/h, 5–95%) of 0.29–0.98 fit upright shapes such as windsurfers or paddlers.
- The paper says the split could not be made independent for LV, and that some videos show the same objects from other viewpoints or times. We do not check this against TEST, and nothing may depend on it.

**2. The scale gap is total**

| Scale | TRAIN | TEST |
|---|---|---|
| Small | 0% | 82% (9,912) |
| Medium | 600 | 2,203 |

- TRAIN LV √area is 45–88 px (5–95%), and its min side is 39–60 px at native resolution.
- Shrinking a box *just* to the COCO-small boundary needs a factor of 0.36–0.71. TEST small LV may be far smaller than that boundary.
- For comparison, the median √area of small-heavy vessel classes in TRAIN is about 24–26 px (BOAT 26.4, SAILING_VESSEL 24.3).

**3. The 640 resize makes this worse**
- After the 640×640 warp, TRAIN LV min side is already only 13–20 px.
- An LV shrunk to small scale would be about 6–11 px after the warp, and smaller ones fall below 4 px.
- Native or tiled resolution is therefore a prerequisite for LV, which supports the native-resolution proposer.

**4. Context confound**
- LV frames are crowded scenes. Each frame holds on average about 9 SAILING_VESSEL, 8 MARINE_MARKER and 3 BOAT boxes.
- A model can learn "LV = that scene" or "LV = that camera" instead of the object itself.

**5. Annotation check**
- The `area` field equals w·h exactly for every LV box (max relative difference 0). The scale bins are therefore consistent with the paper's.

## Why LV matters
- On TEST, every published model is weak on LV: R50 14.23 AP, Swin-B 16.85, TransVOD++ 10.63.
- LV is 1/7 of mAP. Raising LV AP by +20 lifts overall mAP by about +2.9. That is roughly a third of the +9 gap to TransVOD++.
- No VAL tuning is possible, because VAL contains no LV.

## Design consequences (adopted for G2 and G4)
**D1. Scale augmentation (fixed without looking at TEST)**
- Paste LV instances with a target √area drawn from the empirical TRAIN √area distribution of BOAT + SAILING_VESSEL. Vessel-like classes are a TEST-free stand-in for "how small water vessels appear in SeaClips".
- The shrink factor is capped at 1, so LV is never enlarged.
- Use antialiased downsampling, followed by mild blur and noise matched to the target scale.

**D2. Context augmentation**
- Copy-paste LV crops into water regions of *other* TRAIN videos covering all sensors, weathers and sea states. Use alpha/feathered blending, placed near the horizon band (TRAIN LV y-centre is 0.49–0.57 of the frame height).
- Add photometric jitter, so that camera and scene stop being predictive of the class.

**D3. LV validation (the only TEST-free option)**
- Use 2-fold leave-one-LV-video-out on TRAIN: train without video_23 and test on video_23, then swap.
- Test LV crops at native size *and* synthetically shrunk to the D1 scales.
- This readout is descriptive only. With only two objects, it cannot be a powered test.

**D4. LV false-positive guard**
- VAL contains no LV, so every LV prediction on VAL is a false positive.
- Any LV-augmented model must report its VAL LV false-positive count and its BOAT/SAILING_VESSEL AP change against the non-augmented arm.
- This guards against LV training stealing small-vessel detections, which would add more of the wrong-class errors that are already our main failure mode.
