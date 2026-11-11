# MARIVISION

Code and experiment records for the paper

> **MARIVISION: Resolution-preserving detection of small surface objects in maritime video**
> Anil Kumar Korupoju, Indian Register of Shipping. Submitted to *Ocean Engineering*, 2026.

MARIVISION is a detector for small surface objects (buoys, small craft, debris, seabirds) in maritime video. It
combines Deformable DETR with a ConvNeXt-B backbone, training on a mixture of full frames and native-resolution
windows, and inference on 17 views per frame. It was developed and evaluated on the
[SeaClips](#dataset) benchmark, on a single NVIDIA RTX 3060 (12 GB).

## Results (SeaClips test split, 18 unseen videos, mAP@0.3, mean ± SD over 3 seeds)

| System | Clean | Synthetic rain | Synthetic sun flare |
|---|---|---|---|
| **MARIVISION** | **38.16 ± 1.41** | 22.17 ± 5.52 | 37.90 ± 1.02 |
| **MARIVISION + class routing** (ANIMAL from the ResNet-50 reference) | **35.29 ± 1.09** | 20.34 ± 3.77 | 35.04 ± 0.95 |
| Same weights, full-frame view only | 23.78 ± 2.78 | 12.31 ± 1.75 | 23.41 ± 2.39 |
| ResNet-50 reference (this repository, Epoch 15, one run) | 20.72 | 5.66 | 20.44 |
| TransVOD++ Swin-B (published) | 31.95 ± 3.28 | 24.47 | 32.64 |
| Deformable DETR Swin-B (published) | 31.28 ± 3.72 | 23.71 | 30.11 |
| Deformable DETR ResNet-50 (published) | 22.86 ± 0.91 | 5.52 | 18.16 |

How to read these numbers:

- **Significance.** Under the benchmark's one-tailed Welch test with Bonferroni correction (α = 0.0167), both
  configurations are significantly above Deformable DETR ResNet-50 (p = 0.0001). The higher means than TransVOD++ and
  Swin-B Deformable DETR are **not** significant (p = 0.032 and 0.035 for MARIVISION; 0.106 and 0.098 with class
  routing).
- **Class routing transferred negatively.** The routing rule was chosen on one validation video and lowered test
  mAP@0.3 in every seed (by 1.30 to 4.35 points). Both configurations were fixed before the test split was used.
- **Cost.** 17 views per frame take about 2 s per frame on the RTX 3060.
- **Rain.** Synthetic rain reduces accuracy by 42%, more than for the published Swin-B models.
- **Different pipelines.** Published values come from the benchmark authors' pipeline (Denk et al., WACV 2026); the
  corrupted-set values have no published standard deviations.

## How the design was reached

The paper tests six hypotheses, each against a criterion fixed before the test was run. The table maps them to the
scripts and records here. Script and record names use the working names of the project (see the glossary below).

| Paper | Hypothesis | Outcome on SeaClips | Scripts | Records (`docs/predeclarations/`) |
|---|---|---|---|---|
| Sec. 4.2 | H1 Scene geometry (horizon, box geometry) | Not supported | `scripts/phase0/`, `scripts/g1/g1a_*` | `h0_horizon_record`, `g1a_predeclaration` |
| Sec. 4.3 | H2 Temporal context (fusion, trajectory voting) | Not supported | `scripts/phase0/`, `scripts/g1/g1b0_*` | `g1b0_predeclaration` |
| Sec. 4.4 | H3 Native resolution at inference | Recall up, mAP unchanged | `scripts/m1/` | `v13_1_lead_audit_record`, `m1_1_predeclaration` |
| Sec. 4.5 | H4 Post-hoc class correction | Not supported | `scripts/phase0/`, `scripts/m2/m2_single_label_cache.py`, `scripts/g2/` | `m2_sl_predeclaration`, `g2_predeclaration` |
| Sec. 4.6 | H5 Native-resolution training (ResNet-50) | Recall up, class errors dominate | `scripts/m2/` | `m2_predeclaration` |
| Sec. 4.7 | Leisure-vehicle audit | Dataset constraint | `scripts/g1/g1c_*` | `g1c_lv_census` |
| Sec. 5 | H6 Capacity + native-resolution training | Supported | `scripts/g3/`, `scripts/g4/` | `g3_`, `g4_pilot_`, `g4_full_`, `g4b_`, `g4c_predeclaration` |
| Sec. 7 | Test evaluation, three seeds | — | `scripts/g5/` | `g5_predeclaration` |
| Sec. 7.6 | Rain and sun flare | — | `scripts/g6/` | `g6_predeclaration` |

The detector itself is defined by `scripts/g4/g4_pilot.py` (model, window sampling), `scripts/g4/g4_full.py`
(training schedule) and `scripts/common/marivision_hr_p2_pipeline.py` (views and fusion).

### Glossary: working names in the scripts and records

| Here | In the paper |
|---|---|
| R0, R50 | ResNet-50 reference detector |
| G4F | MARIVISION (ConvNeXt-B, 15-epoch run) |
| tile, tiled, Stage C window; plain, Stage B | native-resolution window, 17-view inference; full-frame view |
| unrouted / routed; secondary / primary | MARIVISION / MARIVISION + class routing |
| offset 0 / offset 5 | evaluation subset / selection subset of the validation split |
| gate, arm, pre-declaration | test, variant, criterion fixed in advance |
| M1, M2, G1–G6 | working names of the experiments (see the table above) |
| AP30 on a 0–1 scale | mAP@0.3 in percent |

## Repository layout

| Path | Content |
|---|---|
| `scripts/` | Every experiment script as it was run; `MANIFEST.sha256` lists their SHA-256 digests |
| `scripts/common/` | Views and fusion pipeline, dataset loader, reference-detector tools |
| `patches/`, `NOTICE_MODIFICATIONS.md` | Changes to Deformable DETR, as a patch against upstream |
| `docs/predeclarations/` | The record of each experiment: hypothesis, variants, criterion, code digest, outcome |
| `docs/RUNNING.md` | Environment, data layout, order of runs, run times |
| `results/` | Aggregate result tables (see `results/README.md`) |
| `tools/` | Manifest, frame-size census and patch-listing utilities |

## Reproducing

1. Obtain SeaClips from its maintainers under their terms (see [Dataset](#dataset)).
2. Follow `docs/RUNNING.md`. Each script has a `selftest` mode (synthetic data, CPU) and, for GPU work, a `smoke`
   mode; run both before any long run.
3. Check the digests: `python tools/make_manifest.py` and compare with `MANIFEST.sha256`.

The scripts were run on Windows (Python 3.11, PyTorch 2.5.1) with paths of the original machine; adjust the paths in
each script header. One training run of MARIVISION takes about 34–37 hours on the RTX 3060, and a test evaluation
about 4 hours per seed.

## What is not included

In line with the SeaClips licence, this repository contains **no** dataset images, annotations, per-frame predictions,
prediction caches or trained weights. Only code, records and aggregate tables are released.

## Dataset

SeaClips: F. Denk, C. Rankl, S. Almouahed, D. Moser, R. Sablatnig, "SeaClips: A Video Dataset for Maritime Object
Detection", WACV 2026. The dataset is gated and limited to research on maritime obstacle detection; see its licence.

## Development note

The code and the manuscript were developed with the assistance of generative AI tools (Claude, Anthropic; Gemini,
Google), as declared in the paper. The author directed the work, reviewed the code and results, and is responsible for
the content.

## Citation

See `CITATION.cff`. Until the paper is published, please cite it as a manuscript submitted to *Ocean Engineering*.

## Licence

Apache License 2.0 (see `LICENSE` and `NOTICE`). The modifications to Deformable DETR (Apache-2.0, SenseTime) are
described in `NOTICE_MODIFICATIONS.md`.
