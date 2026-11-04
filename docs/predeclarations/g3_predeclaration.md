# G3 pre-declaration: backbone feasibility smoke on the RTX 3060 (2026-10-28)

- **Script:** `g3_backbone_smoke.py`, version G3.0, SHA-256 d0df3725956fca4e138c59ba070923e57e950f04c8246fa0ea298918d372ba21
- **Data:** 2 TRAIN frames from the zip (read-only): the first GT frame of video_01, and an LV frame from video_18.
- **Excluded:** no VAL, no TEST, and no training run.
- **Repository:** the Deformable-DETR repository is **not modified**. The timm backbone is swapped in at run time (`model.backbone[0]`, `model.input_proj`).

## Why G3
G2 showed that mAP@0.3 depends on how the detector ranks each class's scores across frames, and that a post-hoc classifier destroys that ranking. Every remaining lever has to live inside the detector:
- a stronger backbone (the paper's Swin-B gains +8.4 TEST mAP over R50);
- native resolution (PoL-4, M2);
- LV augmentation (G1c).

G3 decides *which* backbone a later, separately pre-declared training run may use, and what it costs.

## Candidates
- R50, the R0 control (torchvision path, unchanged).
- ConvNeXt-T, -S and -B (timm `fb_in22k`).
- Swin-T and Swin-B (timm `ms_in22k`, `strict_img_size=False`).

Every candidate gets the same adaptation:
- Stride-8, 16 and 32 features feed R0's 4-level Deformable-DETR. The 4th level uses a stride-2 conv, as in R0.
- The stem and first stage are frozen, mirroring R0's frozen conv1/layer1.
- The transformer, queries and box head transfer from the COCO Deformable-DETR checkpoint.
- The backbone, input projections and the 7-class head are new.
- R0's optimizer groups, learning rates (2e-4 / 2e-5 / ×0.1) and grad clip of 0.1 are used.

## Measurements (per candidate, real frames, pretrained weights)
- Settings: 640×640 batch 2 in fp32 and in AMP, plus 1024×1024 batch 2 in AMP as the native-resolution proxy.
- Median s/iter over 8 iterations, after 2 warmup iterations.
- Peak GPU memory.
- Finite loss and gradients.
- Overfit on the 2 frames: 40 steps, AMP. The loss must fall by ≥ 30%.
- Projection of 15 epochs × 10,155 iterations. This is GPU compute only; data loading is excluded.
- Failures (OOM, timm version issues) are recorded per candidate, and the run continues.

## Selection rule (frozen)
**Feasible** means all of:
- finite loss and gradients;
- an overfit loss drop of ≥ 30%;
- peak memory ≤ 11.0 GB at 640×640 batch 2, in fp32 or AMP.

**Recommendation:** among feasible candidates, the highest-capacity backbone whose projected 15-epoch AMP time at 640 is ≤ 4× the R50 fp32 time.

The full table is reported either way. **The Director sets the GPU budget.**

## Selftest (sandbox, `-W error`, against an unmodified upstream Deformable-DETR with the pytorch MSDA backend): PASS
- The timm wrapper gives the right channels and strides. Swin's NHWC output is converted to NCHW. The stem is frozen.
- The swap rebuilds `input_proj` with the right in-channels. COCO transfer loads transformer tensors exactly, and backbone, input_proj and class head are skipped.
- CPU train steps give finite loss and gradients. The overfit drop is 0.83.
- The R50 control path works.
- The selection rule is checked on a hand-made table.

**Only two known warnings from the unmodified repository are silenced:**
- torchvision's `pretrained=` deprecation;
- `torch.meshgrid` indexing.

All other warnings still raise.

## Amendment G3.0.1 (2026-10-28, before any measurement; method unchanged)
The workstation run of G3.0 stopped before measuring anything.

**1. `import timm` failed under `-W error`**
- timm optionally imports `wandb`. The installed wandb raises a pydantic `UnsupportedFieldAttributeWarning` at import and leaves a temp dir behind.
- Fix: `wandb` is blocked via `sys.modules["wandb"] = None`. timm handles the resulting ImportError, and wandb is never used.

**2. The live `SeaClipsDetection` has no `.images` attribute**
- Fix: frames are now picked through `ds.coco`.
- Every pick is verified against the `image_id` the dataset itself returns. A mismatch stops the run.

**New script:** G3.0.1, SHA-256 0a526711aa7554a4c37d1337c108a32ffc1e13d0886579e7a1fa9418c0cd00f8. The sandbox selftest passes.

## Results (run 2026-10-28; G3.0.1, SHA 0a526711…00f8; timm 1.0.26, torch 2.5.1, RTX 3060)
- **Run note:** with `HF_HUB_DISABLE_SYMLINKS_WARNING=1` set (an environment variable only; no script change), all candidates built.
- **A first attempt** downloaded every weight file but stopped at build: `-W error` turned the huggingface_hub Windows "no symlinks" notice into an exception.

| Backbone | Params M (backbone) | fp32 640 s/it · GB | AMP 640 s/it · GB | AMP 1024 s/it · GB | Overfit drop | 15 ep AMP 640 h |
|---|---|---|---|---|---|---|
| R50 (R0) | 40 (23) | 0.461 · 4.8 | 0.405 · 4.7 | 0.850 · 10.8 | 0.60 | 17.2 |
| ConvNeXt-T | 41 (28) | 0.500 · 5.4 | 0.383 · 4.7 | 0.919 · 10.8 ¹ | 0.59 | 16.2 |
| ConvNeXt-S | 63 (49) | 0.623 · 6.7 | 0.425 · 5.6 | 2.123 · 12.6 ² | 0.61 | 18.0 |
| **ConvNeXt-B** | 101 (88) | 0.763 · 8.0 | **0.462 · 6.6** | 11.98 · 14.1 ² | 0.60 | **19.6** |
| Swin-T | 41 (28) | 0.483 · 5.5 | 0.394 · 4.8 | 1.123 · 11.1 | 0.57 | 16.7 |
| Swin-B | 101 (87) | 0.728 · 8.1 | 0.508 · 6.9 | 4.039 · 15.0 ² | 0.60 | 21.5 |

¹ One fp16 step produced non-finite gradients. GradScaler skips such steps, but it is a warning about fp16 numerics.

² Peak memory is above the 12 GB of physical VRAM. The Windows driver spilled into shared system memory, which explains the 5–25× slowdown. **Full-frame 1024 is not viable for S/B backbones.**

**Verdict (frozen rule):**
- All six are feasible at 640×640 batch 2.
- **Recommended: ConvNeXt-B in22k.** It is the largest feasible backbone, at 1.00× the R50 fp32 time.

**Interpretation**
- At 640 the transformer dominates the cost, with the pytorch MSDA backend. A B-size backbone costs about the same per step as R50, either fp32 R0 (×1.0) or AMP R50 (×1.14).
- The paper's backbone class (Swin-B) is therefore affordable on the RTX 3060.
- **Native resolution has to come from tiles at 640** (the M2 window recipe), not from larger full frames.
- **fp16 AMP showed an overflow at 1024.** The training run should use bf16 autocast, which Ampere supports and which needs no loss scaling, after a short fp32-vs-bf16 loss-parity check in its smoke.
- **All projections exclude data loading.**
