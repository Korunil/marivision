# Modifications to Deformable DETR

MARIVISION runs on a modified copy of **Deformable DETR** (Zhu et al., ICLR 2021),
<https://github.com/fundamentalvision/Deformable-DETR>, licensed under the Apache License 2.0 (SenseTime).

This repository does not redistribute the upstream code. It contains:

- `patches/deformable_detr.patch`: the exact changes to five upstream files, as a unified diff;
- the files that were added to the working copy (listed below).

The modifications are by Anil Kumar Korupoju (2026) and are distributed under the Apache License 2.0.

- **Upstream base:** commit `11169a60c33333af00a4849f1808023eba96a931` (the head of `main`).
- **Verification:** applying the patch to a fresh clone of that commit reproduces all five modified files with the
  SHA-256 digests listed below (checked on 2026-11-11).
- **Weights are not included.** The COCO checkpoint (`r50_deformable_detr-checkpoint.pth`) is downloaded from the
  upstream release page. SeaClips-trained weights are not redistributed.

## How to rebuild the working copy

```
git clone https://github.com/fundamentalvision/Deformable-DETR deformable_detr
cd deformable_detr
git checkout 11169a60c33333af00a4849f1808023eba96a931
git apply ../patches/deformable_detr.patch
```

Then copy the added files into the same folder (see "Added files") and set `R0_MSDA_BACKEND=pytorch` before running any
script.

## Modified files

| # | File | SHA-256 (first 16) | Change | Why |
|---|---|---|---|---|
| 1 | `models/ops/modules/ms_deform_attn.py` | `bae81c40ff27c5ec` | The environment variable `R0_MSDA_BACKEND` selects the attention backend. `cuda` (the default) calls the compiled extension and raises a clear error if it is not built. `pytorch` calls the pure-PyTorch reference `ms_deform_attn_core_pytorch` directly. Any other value raises an error. | Runs on Windows without building the CUDA extension. All MARIVISION results were produced with `R0_MSDA_BACKEND=pytorch`. |
| 2 | `models/ops/functions/ms_deform_attn_func.py` | `bb5dc70810978091` | The import of the compiled `MultiScaleDeformableAttention` extension is wrapped in `try/except ImportError`; `MSDeformAttnFunction.forward` raises a `RuntimeError` if the extension is missing. The argument lists were also re-wrapped over several lines (no change in behaviour). | Lets the package import when the extension is not built. |
| 3 | `models/deformable_detr.py` | `6556f1e75981f30b` | `build()` takes the number of classes from `args.num_classes_override` when it is set. `DeformableDETR.forward` accepts an optional `horizons` argument, which this file does not use. One trailing comma was removed (no change in behaviour). | Seven SeaClips classes instead of COCO's 91. The `horizons` argument remains from an early horizon-conditioning experiment; no reported model uses it. |
| 4 | `util/misc.py` | `c48e9c86e0bd4c55` | If `torchvision.ops.misc._NewEmptyTensorOp` cannot be imported, an equivalent class is defined locally. | The upstream version test `float(torchvision.__version__[:3]) < 0.5` reads recent versions such as "0.20" as 0.2 and then imports a class that newer torchvision has removed. |
| 5 | `models/segmentation.py` | `ae1c60426f9b945e` | `from PIL import Image` is moved above `import torch`. | Import order required on the workstation used. The segmentation head itself is not used. |

## Added files

| File | SHA-256 (first 16) | Purpose |
|---|---|---|
| `datasets/seaclips_detection.py` | `d2efe6f475921d35` | SeaClips loader in COCO format; reads images directly from `SeaClips.zip` |
| `r0_preflight.py` | `3b19a1107f17ed00` | Builds the argument set of the ResNet-50 reference detector (seven classes) and checks the environment |
| `train_seaclips_r0_baseline.py` | `cdb3a1f35b8436f9` | Trains the ResNet-50 reference detector |
| `resume_seaclips_r0_baseline.py` | `f24c336c59746536` | Resumes that training from a checkpoint |
| `export_r0_val_predictions.py` | `078dea6b696ce4cb` | Exports validation predictions of the reference detector |
| `models/hrpc.py` | `918d1798c21b153b` | Horizon-relative positional conditioning module from an early experiment; not imported by any modified upstream file and not used by any reported model |

The experiment scripts that run from the same folder (`m1_*.py`, `m2_*.py`, `g4*.py`, `g5_*.py`, `g6_*.py`) are in
`scripts/`, with their digests in `scripts/MANIFEST.sha256`.

## Upstream files absent from the working copy

`LICENSE`, `README.md`, `benchmark.py`, `requirements.txt`, `docs/changelog.md` and the five shell scripts in `configs/`
were not present in the working copy. None is needed to run MARIVISION. Cloning upstream as shown above restores them.

## Behaviour notes for anyone reproducing

- **Determinism:** on GPU, the pure-PyTorch deformable attention uses `grid_sample`, whose backward pass is not
  bit-deterministic. Resumed training is therefore statistically equivalent to uninterrupted training, not identical.
- **Speed:** the pure-PyTorch backend is slower than the CUDA extension. All timings reported in the paper use it.
- **Unchanged:** the matcher, the losses, the post-processing (top 100 query–class pairs per image) and the COCO
  evaluation helpers.
