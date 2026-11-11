# Running the MARIVISION scripts

Every script in `scripts/` is the exact file that was run, with its hash in `MANIFEST.sha256` and in its
pre-declaration (`docs/predeclarations/`). To keep those hashes valid, the files are released unchanged. This page explains
the environment, data layout and working folders they expect.

> **Data are not included.** SeaClips (images and annotations) must be obtained from its maintainers under their
> licence. No predictions or trained weights are distributed.

## 1. Environment

| Package | Version used | Needed for |
|---|---|---|
| Python | 3.11 (Windows, conda env) | all |
| PyTorch / torchvision | 2.5.1 / matching | all GPU work |
| timm | 1.0.26 | ConvNeXt / Swin backbones (G3 onwards) |
| pycocotools | any recent | all scoring |
| numpy, scipy, Pillow | recent | all |
| albumentations, albucore | 2.0.8, 0.0.24 | G6 only |
| opencv-python / -headless | 5.0.0.93 | G6 only |

GPU: one NVIDIA RTX 3060 (12 GB). Peak use was 8.0 GB in training and 3.5 GB in inference.

**Windows note:** in some conda environments, `PIL` must be imported before `torch`, or PIL's `_imaging` DLL fails to
load. Every script therefore starts with `from PIL import Image`.

**Environment variables**, set in every shell before running GPU scripts:

```
set R0_MSDA_BACKEND=pytorch                 # pure-PyTorch deformable attention (see deformable_detr/NOTICE_MODIFICATIONS.md)
set HF_HUB_DISABLE_SYMLINKS_WARNING=1       # avoids a Windows warning that -W error would turn into an exception
```

All runs use `python -W error ...`, so any unexpected warning stops the run.

## 2. Data and checkpoint layout

The scripts use fixed paths under a root folder `D:\MARIVISION`. The constants are at the top of each script.

```
D:\MARIVISION\
├── SeaClips.zip                       # images; read directly from the zip, never extracted
├── seaclips-train.json                # SHA-256 edc32bdb...9b35
├── seaclips-val.json                  # SHA-256 8d3fca68...a324
├── seaclips-test.json                 # SHA-256 524569f5...195d  (locked; read only by G5 / G6 TEST modes)
├── r50_deformable_detr-checkpoint.pth # COCO checkpoint from the Deformable DETR release
├── checkpoints_r0\checkpoint_epoch_XX.pth   # R0 baseline, epochs 1-15
├── R0_frozen_baseline\                # = deformable_detr/ + the "R0-folder" scripts below
├── H0_workbench\eval_tools\           # = scripts/common/marivision_hr_p2_pipeline.py + the "eval-tools" scripts below
├── H0_workbench\g2\, H0_workbench\g3\ # G2 and G3 working folders
├── G4_pilot\, G4_full\, G4b\, G5\     # outputs written by the scripts
```

**Using another root or operating system.** Edit the path constants at the top of each script, and record that the
file no longer matches its published hash. For example, a Linux copy can replace `D:/MARIVISION` with a local path.

## 3. Two working folders

Several scripts check that their companion files sit next to them, so they must be **run from** a specific folder.

| Working folder | Contents to place there | Scripts run from it |
|---|---|---|
| **R0 folder**: a copy of `deformable_detr/` | `deformable_detr/*` (with `r0_preflight.py`, `datasets/seaclips_detection.py`) | `m1/m1_r0_tiled_inference.py`, `m1/m1_1_r0_tile_cache.py`, `m1/m1_parity_probe.py`, `m2/m2_train_r0_finetune.py`, `g4/g4_pilot.py`, `g4/g4_full.py`, `g4/g4b_animal_finetune.py`, `g5/g5_seed_train.py`, `g5/g5_test_eval.py`, `g6/g6_robustness.py` |
| **eval-tools folder** | `scripts/common/marivision_hr_p2_pipeline.py` | `m1/m1_compare_plain_vs_tiled.py`, `m1/m1_1_merge_variants.py`, `m1/m1_1_fp_anatomy.py`, `m2/m2_evaluate.py`, `m2/m2_*` diagnostics, `g1/*`, `g4/g4f_followup.py`, `g4/g4c_animal_anatomy.py`, `g5/g5_test_inventory.py` |
| G2 / G3 folders | none | `g2/g2_crop_expert.py`, `g3/g3_backbone_smoke.py` |

The R0-folder scripts find the pipeline through `--pipeline-dir` (default `D:/MARIVISION/H0_workbench/eval_tools`).
Some R0-folder scripts import a sibling (e.g. `g4_full.py` imports `g4_pilot.py`; `g6` imports `g5_test_eval.py`).
These imports are SHA-checked, and a mismatch refuses to run.

## 4. Order of execution

Every script supports `--mode selftest`, which uses synthetic data on the CPU. Run it first: it must print
`SELFTEST PASS`. GPU scripts also have `--mode smoke`, which runs a few real steps; it must pass before any long run.
The pass rules each gate was judged by are in its pre-declaration.

| Phase | Gate | Script(s) and modes | Main output |
|---|---|---|---|
| Baseline | R0 | `python train_seaclips_r0_baseline.py` in the R0 folder (settings from `r0_preflight.py`, 15 epochs; `resume_seaclips_r0_baseline.py` continues an interrupted run) `<confirm arguments>` | `checkpoints_r0\` |
| A | M1 | `m1_r0_tiled_inference.py --mode selftest \| export`; `m1_compare_plain_vs_tiled.py --mode compare` | tiled VAL caches, M1 report |
| A | M1.1 | `m1_1_r0_tile_cache.py --mode export`; `m1_1_merge_variants.py --mode score`; `m1_1_fp_anatomy.py --mode run` | `R0_frozen_baseline\m1_1_tile_cache\` |
| A | M2 | `m2_train_r0_finetune.py --mode smoke \| train`; `m2_evaluate.py --mode score`; `m2_oracle_bounds.py`, `m2_class_diagnostic.py`, `m2_crowd_audit.py --mode run` | M2 reports |
| A | M2-SL | `m2_single_label_cache.py --mode convert`, then `m2_evaluate.py --mode score` | SL caches and report |
| B | G1a | `g1a_geometry_class_cue.py --mode run` | G1a report (TRAIN only) |
| B | G1b-0 | `g1b0_track_class_vote.py --mode run` | G1b-0 report |
| B | G1c | `g1c_lv_census.py --mode run` | `g1c_reports\` |
| B | G2 | `g2_crop_expert.py --mode smoke`, then `build`, `train`, `eval` | G2 report |
| C | G3 | `g3_backbone_smoke.py --mode smoke` | backbone table |
| C | G4 pilot | `g4_pilot.py --mode smoke \| train --arm A \| export --arm A`; scoring with `m2_evaluate.py` | `G4_pilot\` |
| C | G4 full | `g4_full.py --mode smoke \| train [--resume] \| export --epoch K --offset 0\|5 \| curve` | `G4_full\run\` |
| C | G4 follow-up | `g4f_followup.py --mode animal \| off5` | `g4f_reports\` |
| C | G4b | `g4b_animal_finetune.py --mode smoke \| train --arm A \| export --arm A --offset 0\|5 \| score` | `G4b\run\` |
| C | G4c | `g4c_animal_anatomy.py --mode run` | `g4c_reports\` |
| D | G5 seeds | `g5_seed_train.py --mode smoke --seed S \| train --seed S [--resume] \| curve --seed S` (S = 43, 44) | `G5\seed4X\run\` |
| D | G5 inventory | `g5_test_inventory.py --mode run` (zip file list only) | `G5\g5_test_inventory.md` |
| D | G5 TEST | `g5_test_eval.py --mode proof`; then `--mode export --model M --test-unlock` for M in r0e15, r0e04, seed42, seed43, seed44; then `--mode score --test-unlock` (once) | `G5\g5_test_results.md` |
| D | G6 | `g6_robustness.py --mode proof`; `--mode export --corruption C --model M --test-unlock`; `--mode score --corruption C --test-unlock` (once per corruption) | `G5\robust\<C>\` |

## 5. Run times on one RTX 3060

| Job | Time |
|---|---|
| R0 / G4F / G5 seed training | about 2.15 h per epoch, plus 0.23 h inline VAL export; about 36 h for 15 epochs |
| G4b, one fine-tune arm (1 epoch) | about 2.2 h |
| Tiled export, 432 VAL frames | about 14 min |
| Tiled TEST export (7,030 frames) | about 3.8 h per model (rain 3.9 h, sun flare 4.2 h) |
| Full-frame R0 TEST export | about 0.25 h |
| TEST scoring (CPU, single core) | about 1–2 h |

## 6. TEST lock
TEST is read only by `g5_test_eval.py` and `g6_robustness.py`, and only in modes that require `--test-unlock`. Both refuse:
- a label file whose SHA-256 differs from the lock;
- a checkpoint that is not the declared epoch-15 checkpoint of its seed;
- a class mapping that differs from VAL's.

Scoring refuses to run before all models are exported, and writes a lock file after its single run. **These locks were
part of the study design. Re-scoring TEST in your own reproduction is your choice, but it is not the protocol reported
in the paper.**

## 7. What is deliberately not reproducible from this repository
- **SeaClips data,** for licence reasons.
- **Trained weights and per-frame predictions,** for licence reasons.
- **Phase 0 (H0, temporal adapters, MARSEN, B-FCEF-F),** which predates the pre-declaration procedure. Its scripts are
  in `scripts/phase0/` where released, without recorded hashes.
