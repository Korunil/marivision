# MARIVISION

A pre-registered study of resolution-preserving tiled detection on the
[SeaClips](#dataset) maritime video benchmark, run on a single 12 GB GPU.

> **Status:** code and records accompany the manuscript *"Resolution-preserving tiled detection on SeaClips: a
> pre-registered study of what transfers to unseen maritime videos, and what does not"* (in preparation).

## Results at a glance (SeaClips TEST, mAP@0.3, mean ± SD over 3 seeds)

| System | mAP@0.3 |
|---|---|
| Pre-declared system (ANIMAL routed to R50) | 35.29 ± 1.09 |
| Pre-declared secondary (unrouted) | 38.16 ± 1.41 |
| Our R50 reproduction (epoch 15, single run) | 20.72 |

**Comparison with the published baselines:** TransVOD++ Swin-B 31.95 ± 3.28, Deformable-DETR Swin-B 31.28 ± 3.72,
Deformable-DETR R50 22.86 ± 0.91.

- Under the benchmark's one-tailed Welch test with Bonferroni correction, the primary system differs significantly
  only from Deformable-DETR R50.
- The differences from the two Swin-B models are not significant (p = 0.106 and 0.098).
- The full results, the fourteen negative or partial results and the dataset properties are in the manuscript and in
  `results/`.

## What is here and what is not

| Included | Not included (dataset licence) |
|---|---|
| Every gate script, as it was run, with SHA-256 (`scripts/`, `MANIFEST.sha256`) | SeaClips images or annotations |
| Pre-declarations with frozen pass rules (`docs/predeclarations/`) | Per-frame predictions or caches |
| Aggregate result tables (`results/`) | Trained weights |
| The manuscript source (`paper/`) | |

To reproduce, obtain SeaClips from its maintainers under their terms. The scripts expect a Deformable-DETR checkout and
the paths described in each script's header. They were run as-is on Windows (Python 3.11, RTX 3060), and the paths are
hard-coded to the original machine. A path-configurable release is planned.

## How the study was run

1. Each gate has a pre-declaration: hypothesis, arms, numeric pass rule and the consequence of each outcome. It also
   records the SHA-256 of the script that will run.
2. Each script has `--mode selftest` (synthetic data, CPU) and, for GPU work, `--mode smoke` (a few real steps). Both
   must pass before a long run.
3. The TEST label file was hash-locked without being parsed. The TEST tool scores once and then writes a lock file.

## Dataset

SeaClips (Denk et al., WACV 2026). The dataset is gated and research-only; see its licence.

## Citation

See `CITATION.cff`.

## Licence

Code: MIT (see `LICENSE`). Records and documents: CC BY 4.0.
