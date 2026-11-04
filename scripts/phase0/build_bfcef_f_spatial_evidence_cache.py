from __future__ import annotations

"""
B-FCEF-F derived spatial-evidence cache builder.

Purpose
-------
Run the frozen R0 detector once over the authoritative 20,075-frame Train
universe and persist only the compact, parameter-free spatial evidence needed
by the permanent S and L experts:

    scale_spatial : [Q, 21]
    loc_spatial   : [Q, 24]

The script never trains or modifies R0. Every frame is reconciled against the
certified canonical E0 cache before a derived record is accepted.

Source of truth / provenance chain
----------------------------------
- frozen R0 checkpoint SHA-256
- frozen production census SHA-256
- census-declared Train JSON SHA-256
- all 47 production shard SHA-256 entries
- canonical E0 cache manifest internal roots
- all 47 canonical E0 cache shard SHA-256 entries
- live-vs-canonical R0 parity on every processed frame

The derived manifest records the exact input manifest SHA and this builder's
SHA so later S0/S1 runs can be pinned to an immutable evidence source.
"""

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import torch


# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------
ROOT_DIR = Path("D:/MARIVISION")
EVAL_DIR = ROOT_DIR / "H0_workbench" / "eval_tools"
SHARD_DIR = EVAL_DIR / "full_split_shards"
SCREENING_CONFIG_PATH = EVAL_DIR / "screening_config_s0.json"
PRODUCTION_SPATIAL_SOURCE = EVAL_DIR / "bfcef_f" / "spatial_evidence.py"
E0_CACHE_DIR = EVAL_DIR / "e0_cache"
DERIVED_ROOT = EVAL_DIR / "bfcef_f_derived_cache"
OUTPUT_DIR = DERIVED_ROOT / "spatial_v2"

CENSUS_JSON = EVAL_DIR / "marsen_b2_production_census_56shards.json"
TRAIN_JSON = ROOT_DIR / "seaclips-train.json"
ZIP_PATH = ROOT_DIR / "SeaClips.zip"
R0_CHECKPOINT_PATH = ROOT_DIR / "checkpoints_r0" / "checkpoint_best.pth"
CANONICAL_CACHE_MANIFEST = E0_CACHE_DIR / "bfcef_e0_cache_manifest.json"
SUBSTRATE_SCRIPT = EVAL_DIR / "run_bfcef_e0_substrate.py"

FROZEN_R0_CKPT_SHA256 = (
    "28e6288c3f6769c2d5275aa77ddb248735b55a09cd46a6b570e3f117890b268b"
)
FROZEN_CENSUS_SHA256 = (
    "a20a734d668a9078bdfe274cb09fa5a47ad9c6a3a9d7cef2d1f99b926623fbc5"
)

NUM_VIDEOS = 47
EXPECTED_TOTAL_FRAMES = 20_075
EXPECTED_NUM_QUERIES = 300
SPATIAL_SCHEMA_VERSION = "bfcef_f_spatial_v2_production_s21_l24"

# Frozen evidence sampling definition for the first architecture-conformant
# spatial evidence cache.
SCALE_RING_EXPANSION = 1.50
LOCALIZATION_SIDE_SAMPLES = 5
LOCALIZATION_BOUNDARY_OFFSET = 0.05

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from bfcef_f.config import BFCEFFScreeningConfig
from bfcef_f.spatial_evidence import (
    extract_localization_spatial_evidence,
    extract_scale_spatial_evidence,
)


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------
def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _assert_tensor_close(
    observed: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    label: str,
    image_id: int,
) -> float:
    if observed.shape != expected.shape:
        raise AssertionError(
            f"[FATAL] {label} shape mismatch on image {image_id}: "
            f"{tuple(observed.shape)} != {tuple(expected.shape)}"
        )

    delta = float(
        (
            observed.float().cpu()
            - expected.float().cpu()
        )
        .abs()
        .max()
        .item()
    )

    if delta > atol:
        raise AssertionError(
            f"[FATAL] {label} drift on image {image_id}: "
            f"{delta:.3e} > {atol:.1e}"
        )

    return delta


def _find_deformable_detr_root() -> Path:
    candidates = [
        ROOT_DIR / "Deformable-DETR",
        ROOT_DIR / "deformable_detr",
    ]

    for candidate in candidates:
        if (
            candidate.exists()
            and (candidate / "main.py").exists()
            and (candidate / "models" / "deformable_detr.py").exists()
            and (
                candidate
                / "datasets"
                / "seaclips_detection.py"
            ).exists()
        ):
            return candidate

    raise AssertionError(
        "[FATAL] Cannot locate authoritative "
        "Deformable-DETR repository."
    )


# -----------------------------------------------------------------------------
# Compact spatial descriptor extraction
# -----------------------------------------------------------------------------
# IMPORTANT: there is exactly ONE executable definition of S/L spatial evidence:
# bfcef_f.spatial_evidence.  This cache builder delegates to it instead of
# maintaining a second implementation that can silently drift while preserving
# the same [Q,21]/[Q,24] shapes.

def compute_scale_spatial_descriptor(
    boxes: torch.Tensor,
    feature_maps: Dict[str, torch.Tensor],
    ring_expansion: float = SCALE_RING_EXPANSION,
) -> torch.Tensor:
    return extract_scale_spatial_evidence(
        boxes_cxcywh=boxes,
        feature_maps=feature_maps,
        ring_expansion=ring_expansion,
    )


def compute_localization_spatial_descriptor(
    boxes: torch.Tensor,
    feature_maps: Dict[str, torch.Tensor],
    side_samples: int = LOCALIZATION_SIDE_SAMPLES,
    offset_fraction: float = LOCALIZATION_BOUNDARY_OFFSET,
) -> torch.Tensor:
    return extract_localization_spatial_evidence(
        boxes_cxcywh=boxes,
        feature_maps=feature_maps,
        side_samples=side_samples,
        offset_frac=offset_fraction,
    )


# -----------------------------------------------------------------------------
# Provenance + authoritative R0 initialization
# -----------------------------------------------------------------------------
def verify_source_provenance():
    for path, label in [
        (
            CENSUS_JSON,
            "census",
        ),
        (
            TRAIN_JSON,
            "Train JSON",
        ),
        (
            ZIP_PATH,
            "SeaClips ZIP",
        ),
        (
            R0_CHECKPOINT_PATH,
            "R0 checkpoint",
        ),
        (
            CANONICAL_CACHE_MANIFEST,
            "canonical cache manifest",
        ),
        (
            SUBSTRATE_SCRIPT,
            "R0 substrate script",
        ),
    ]:
        if not path.exists():
            raise AssertionError(
                f"[FATAL] Missing {label}: "
                f"{path}"
            )

    if (
        compute_sha256(
            CENSUS_JSON
        )
        != FROZEN_CENSUS_SHA256
    ):
        raise AssertionError(
            "[FATAL] Production census "
            "SHA drift"
        )

    if (
        compute_sha256(
            R0_CHECKPOINT_PATH
        )
        != FROZEN_R0_CKPT_SHA256
    ):
        raise AssertionError(
            "[FATAL] R0 checkpoint "
            "SHA drift"
        )

    census = json.loads(
        CENSUS_JSON.read_text(
            encoding="utf-8"
        )
    )

    train_json_sha = (
        census["_meta"]
        ["live_provenance"]
        ["train_json_sha256"]
    )

    if (
        compute_sha256(
            TRAIN_JSON
        )
        != train_json_sha
    ):
        raise AssertionError(
            "[FATAL] Train JSON SHA drift"
        )

    source_ledger = (
        census[
            "shard_sha256_ledger"
        ]
    )

    for vid in range(
        1,
        NUM_VIDEOS + 1,
    ):
        name = (
            "marsen_b2_shard_train_"
            f"vid{vid}.pth"
        )

        path = (
            SHARD_DIR
            / name
        )

        if not path.exists():
            raise AssertionError(
                "[FATAL] Missing "
                f"production shard: {name}"
            )

        if (
            compute_sha256(path)
            != source_ledger[name]
        ):
            raise AssertionError(
                "[FATAL] Production "
                f"shard SHA drift: {name}"
            )

    canonical = json.loads(
        CANONICAL_CACHE_MANIFEST
        .read_text(
            encoding="utf-8"
        )
    )

    meta = canonical["_meta"]

    if (
        meta[
            "actual_total_frames"
        ]
        != EXPECTED_TOTAL_FRAMES
    ):
        raise AssertionError(
            "[FATAL] Canonical cache "
            "frame-count drift"
        )

    if (
        meta["num_videos"]
        != NUM_VIDEOS
    ):
        raise AssertionError(
            "[FATAL] Canonical cache "
            "video-count drift"
        )

    if (
        meta[
            "r0_checkpoint_sha256"
        ]
        != FROZEN_R0_CKPT_SHA256
    ):
        raise AssertionError(
            "[FATAL] Canonical cache "
            "points to a different "
            "R0 checkpoint"
        )

    if (
        meta[
            "census_sha256"
        ]
        != FROZEN_CENSUS_SHA256
    ):
        raise AssertionError(
            "[FATAL] Canonical cache "
            "points to a different "
            "census"
        )

    if (
        meta[
            "train_json_sha256"
        ]
        != train_json_sha
    ):
        raise AssertionError(
            "[FATAL] Canonical cache "
            "Train JSON provenance "
            "mismatch"
        )

    cache_ledger = canonical[
        "cache_shard_sha256_ledger"
    ]

    if (
        len(cache_ledger)
        != NUM_VIDEOS
    ):
        raise AssertionError(
            "[FATAL] Canonical cache "
            "shard ledger is not "
            "47 files"
        )

    for (
        name,
        expected_sha,
    ) in cache_ledger.items():
        path = E0_CACHE_DIR / name

        if not path.exists():
            raise AssertionError(
                "[FATAL] Missing "
                f"canonical cache shard: "
                f"{name}"
            )

        if (
            compute_sha256(path)
            != expected_sha
        ):
            raise AssertionError(
                "[FATAL] Canonical cache "
                f"shard SHA drift: {name}"
            )

    return (
        census,
        canonical,
        compute_sha256(
            CANONICAL_CACHE_MANIFEST
        ),
        train_json_sha,
    )


def build_frozen_r0():
    deformable_root = (
        _find_deformable_detr_root()
    )

    for p in reversed(
        [
            EVAL_DIR,
            deformable_root,
            ROOT_DIR,
        ]
    ):
        p_str = str(p)

        if p_str in sys.path:
            sys.path.remove(p_str)

        sys.path.insert(
            0,
            p_str,
        )

    from run_bfcef_e0_substrate import (
        load_authoritative_r0_args,
    )

    from models.deformable_detr import (
        build as build_deformable_detr,
    )

    args = (
        load_authoritative_r0_args()
    )

    args.device = str(DEVICE)

    model, _, _ = (
        build_deformable_detr(
            args
        )
    )

    checkpoint = torch.load(
        R0_CHECKPOINT_PATH,
        map_location="cpu",
    )

    state = checkpoint.get(
        "model",
        checkpoint,
    )

    cleaned = {
        (
            k[7:]
            if k.startswith(
                "module."
            )
            else k
        ): v
        for k, v in state.items()
    }

    model.load_state_dict(
        cleaned,
        strict=True,
    )

    model.to(DEVICE)
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    return (
        model,
        deformable_root,
    )


# -----------------------------------------------------------------------------
# Main build
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Verify provenance and "
            "process a deterministic "
            "small sample without "
            "writing the full cache."
        ),
    )

    args = parser.parse_args()

    t0 = time.time()

    driver_sha = compute_sha256(
        Path(__file__)
    )

    if not SCREENING_CONFIG_PATH.exists():
        raise AssertionError(
            f"[FATAL] Missing screening config: {SCREENING_CONFIG_PATH}"
        )

    screening = BFCEFFScreeningConfig.from_json(
        SCREENING_CONFIG_PATH
    )
    screening_config_sha = compute_sha256(
        SCREENING_CONFIG_PATH
    )

    if not PRODUCTION_SPATIAL_SOURCE.exists():
        raise AssertionError(
            f"[FATAL] Missing production spatial source: {PRODUCTION_SPATIAL_SOURCE}"
        )
    production_spatial_sha = compute_sha256(
        PRODUCTION_SPATIAL_SOURCE
    )

    print(
        "=" * 112
    )
    print(
        "B-FCEF-F DERIVED SPATIAL "
        "EVIDENCE CACHE BUILDER"
    )
    print(
        f"Driver SHA-256: {driver_sha}"
    )
    print(
        f"Device:        {DEVICE}"
    )
    print(
        f"Schema:        "
        f"{SPATIAL_SCHEMA_VERSION}"
    )
    print(
        f"Spatial source: {PRODUCTION_SPATIAL_SOURCE}"
    )
    print(
        f"Spatial SHA:    {production_spatial_sha}"
    )
    print(
        "=" * 112,
        flush=True,
    )

    (
        census,
        canonical_manifest,
        canonical_manifest_sha,
        train_json_sha,
    ) = verify_source_provenance()

    print(
        "[+] Provenance PASS: "
        "census, checkpoint, "
        "Train JSON, 47 production "
        "shards and 47 canonical "
        "cache shards"
    )

    print(
        "[+] Canonical cache "
        "manifest SHA-256: "
        f"{canonical_manifest_sha}"
    )

    (
        model,
        deformable_root,
    ) = build_frozen_r0()

    from datasets.seaclips_detection import (
        SeaClipsDetection,
    )
    from util.misc import collate_fn

    dataset = SeaClipsDetection(
        ZIP_PATH,
        TRAIN_JSON,
        is_train=False,
        random_flip=False,
    )

    # ------------------------------------------------------------------
    # Authoritative image_id -> dataset-index mapping
    #
    # SeaClipsDetection revisions used across the certified R0 pipeline
    # expose image IDs through one of three equivalent interfaces:
    #
    #   1. dataset.ids
    #   2. dataset.img_ids
    #   3. dataset.coco.dataset["images"]
    #
    # This follows the same compatibility contract already used by the
    # certified canonical E0 cache builder. Only the public wrapper
    # attribute differs; image_id semantics remain unchanged.
    # ------------------------------------------------------------------

    if hasattr(
        dataset,
        "ids",
    ):
        dataset_image_ids = [
            int(iid)
            for iid in dataset.ids
        ]

        dataset_id_source = (
            "dataset.ids"
        )

    elif hasattr(
        dataset,
        "img_ids",
    ):
        dataset_image_ids = [
            int(iid)
            for iid in dataset.img_ids
        ]

        dataset_id_source = (
            "dataset.img_ids"
        )

    elif (
        hasattr(
            dataset,
            "coco",
        )
        and hasattr(
            dataset.coco,
            "dataset",
        )
        and isinstance(
            dataset.coco.dataset,
            dict,
        )
        and "images"
        in dataset.coco.dataset
    ):
        dataset_image_ids = [
            int(
                image["id"]
            )
            for image
            in dataset.coco.dataset[
                "images"
            ]
        ]

        dataset_id_source = (
            'dataset.coco.dataset["images"]'
        )

    else:
        raise AssertionError(
            "[FATAL] Authoritative "
            "SeaClipsDetection exposes "
            "none of the certified image-ID "
            "interfaces: dataset.ids, "
            "dataset.img_ids, or "
            'dataset.coco.dataset["images"]'
        )

    iid_to_ds_idx = {
        int(iid): idx
        for idx, iid in enumerate(
            dataset_image_ids
        )
    }

    if (
        len(iid_to_ds_idx)
        != len(dataset_image_ids)
    ):
        raise AssertionError(
            "[FATAL] Duplicate image_id "
            "values detected while building "
            "SeaClipsDetection index."
        )

    if (
        len(iid_to_ds_idx)
        != len(dataset)
    ):
        raise AssertionError(
            "[FATAL] SeaClipsDetection "
            "image-ID cardinality mismatch: "
            f"{len(iid_to_ds_idx)} IDs "
            f"for dataset length "
            f"{len(dataset)}."
        )

    print(
        "[+] SeaClipsDetection image-ID "
        "mapping: "
        f"{dataset_id_source} "
        f"({len(iid_to_ds_idx):,} IDs)"
    )

    taps: Dict[
        str,
        torch.Tensor,
    ] = {}

    def transformer_hook(
        module,
        inp,
        outp,
    ):
        srcs = inp[0]

        taps["P3"] = (
            srcs[0].detach()
        )
        taps["P4"] = (
            srcs[1].detach()
        )
        taps["P5"] = (
            srcs[2].detach()
        )

        hs = outp[0]
        init_ref = outp[1]
        inter_refs = outp[2]

        last = (
            hs.shape[0] - 1
        )

        taps["q_native"] = (
            hs[last].detach()
        )

        if last == 0:
            taps["ref_pts"] = (
                init_ref.detach()
            )
        else:
            taps["ref_pts"] = (
                inter_refs[
                    last - 1
                ].detach()
            )

    handle = (
        model.transformer
        .register_forward_hook(
            transformer_hook
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    derived_ledger: Dict[
        str,
        str,
    ] = {}

    total_frames = 0

    max_dz = 0.0
    max_db = 0.0
    max_dq = 0.0
    max_dr = 0.0

    preflight_videos = {
        1,
        12,
        24,
        36,
        47,
    }

    try:
        for vid in range(
            1,
            NUM_VIDEOS + 1,
        ):
            if (
                args.preflight_only
                and vid
                not in preflight_videos
            ):
                continue

            production_path = (
                SHARD_DIR
                / (
                    "marsen_b2_shard_"
                    f"train_vid{vid}.pth"
                )
            )

            canonical_path = (
                E0_CACHE_DIR
                / (
                    "bfcef_e0_cache_"
                    f"vid{vid}.pth"
                )
            )

            production = torch.load(
                production_path,
                map_location="cpu",
            )

            source_frames = (
                production[
                    "frame_records"
                ]
            )

            canonical_records = (
                torch.load(
                    canonical_path,
                    map_location="cpu",
                )
            )

            if (
                len(source_frames)
                != len(
                    canonical_records
                )
            ):
                raise AssertionError(
                    "[FATAL] Video "
                    f"{vid}: production/"
                    "canonical frame "
                    "count mismatch"
                )

            if args.preflight_only:
                n = len(
                    canonical_records
                )

                frame_indices = sorted(
                    set(
                        [
                            0,
                            n // 2,
                            n - 1,
                        ]
                    )
                )
            else:
                frame_indices = list(
                    range(
                        len(
                            canonical_records
                        )
                    )
                )

            out_records: List[
                dict
            ] = []

            t_vid = time.time()

            with torch.no_grad():
                for seq_idx in (
                    frame_indices
                ):
                    source_fr = (
                        source_frames[
                            seq_idx
                        ]
                    )

                    cached_fr = (
                        canonical_records[
                            seq_idx
                        ]
                    )

                    image_id = int(
                        cached_fr[
                            "image_id"
                        ]
                    )

                    if (
                        int(
                            source_fr[
                                "image_id"
                            ]
                        )
                        != image_id
                    ):
                        raise AssertionError(
                            "[FATAL] Video "
                            f"{vid} seq "
                            f"{seq_idx}: "
                            "production/"
                            "canonical "
                            "image_id "
                            "mismatch"
                        )

                    if (
                        int(
                            cached_fr[
                                "video_id"
                            ]
                        )
                        != vid
                    ):
                        raise AssertionError(
                            "[FATAL] Video "
                            "identity mismatch "
                            "in canonical cache"
                        )

                    if (
                        int(
                            cached_fr[
                                "seq_idx"
                            ]
                        )
                        != seq_idx
                    ):
                        raise AssertionError(
                            "[FATAL] Video "
                            f"{vid}: canonical "
                            "seq_idx mismatch"
                        )

                    if (
                        image_id
                        not in iid_to_ds_idx
                    ):
                        raise AssertionError(
                            "[FATAL] "
                            f"image_id "
                            f"{image_id} "
                            "absent from "
                            "Train dataset"
                        )

                    item = dataset[
                        iid_to_ds_idx[
                            image_id
                        ]
                    ]

                    samples, _ = (
                        collate_fn(
                            [item]
                        )
                    )

                    samples = (
                        samples.to(
                            DEVICE
                        )
                    )

                    taps.clear()

                    out_live = model(
                        samples
                    )

                    required = {
                        "P3",
                        "P4",
                        "P5",
                        "q_native",
                        "ref_pts",
                    }

                    if (
                        set(taps)
                        != required
                    ):
                        raise AssertionError(
                            "[FATAL] "
                            "Transformer tap "
                            "contract mismatch: "
                            f"{sorted(taps)}"
                        )

                    z_live = (
                        out_live[
                            "pred_logits"
                        ][0]
                        .detach()
                    )

                    b_live = (
                        out_live[
                            "pred_boxes"
                        ][0]
                        .detach()
                    )

                    q_live = (
                        taps[
                            "q_native"
                        ][0]
                        .detach()
                    )

                    r_live = (
                        taps[
                            "ref_pts"
                        ][0]
                        .detach()
                    )

                    dz = (
                        _assert_tensor_close(
                            z_live,
                            cached_fr[
                                "d_logits"
                            ],
                            atol=1e-4,
                            label="R0 logits",
                            image_id=image_id,
                        )
                    )

                    db = (
                        _assert_tensor_close(
                            b_live,
                            cached_fr[
                                "d_boxes"
                            ],
                            atol=1e-4,
                            label="R0 boxes",
                            image_id=image_id,
                        )
                    )

                    dq = (
                        _assert_tensor_close(
                            q_live,
                            cached_fr[
                                "q_native"
                            ],
                            atol=1e-5,
                            label=(
                                "R0 query latent"
                            ),
                            image_id=image_id,
                        )
                    )

                    dr = (
                        _assert_tensor_close(
                            r_live,
                            cached_fr[
                                "ref_pts"
                            ],
                            atol=1e-6,
                            label=(
                                "R0 reference "
                                "points"
                            ),
                            image_id=image_id,
                        )
                    )

                    max_dz = max(
                        max_dz,
                        dz,
                    )
                    max_db = max(
                        max_db,
                        db,
                    )
                    max_dq = max(
                        max_dq,
                        dq,
                    )
                    max_dr = max(
                        max_dr,
                        dr,
                    )

                    feature_maps = {
                        "P3": taps["P3"],
                        "P4": taps["P4"],
                        "P5": taps["P5"],
                    }

                    scale_spatial = (
                        compute_scale_spatial_descriptor(
                            b_live,
                            feature_maps,
                            ring_expansion=(
                                screening.scale_ring_expansion
                            ),
                        )
                    )

                    loc_spatial = (
                        compute_localization_spatial_descriptor(
                            b_live,
                            feature_maps,
                            side_samples=(
                                screening.localization_side_samples
                            ),
                            offset_fraction=(
                                screening.localization_boundary_offset
                            ),
                        )
                    )

                    if not (
                        torch.isfinite(
                            scale_spatial
                        ).all()
                    ):
                        raise AssertionError(
                            "[FATAL] "
                            "Non-finite S "
                            "spatial evidence "
                            f"on image "
                            f"{image_id}"
                        )

                    if not (
                        torch.isfinite(
                            loc_spatial
                        ).all()
                    ):
                        raise AssertionError(
                            "[FATAL] "
                            "Non-finite L "
                            "spatial evidence "
                            f"on image "
                            f"{image_id}"
                        )

                    out_records.append(
                        {
                            "image_id": (
                                image_id
                            ),
                            "video_id": (
                                vid
                            ),
                            "seq_idx": (
                                seq_idx
                            ),
                            "scale_spatial": (
                                scale_spatial
                                .float()
                                .cpu()
                            ),
                            "loc_spatial": (
                                loc_spatial
                                .float()
                                .cpu()
                            ),
                        }
                    )

                    total_frames += 1

            if not args.preflight_only:
                out_name = (
                    "bfcef_f_spatial_"
                    "evidence_"
                    f"vid{vid}.pth"
                )

                out_path = (
                    OUTPUT_DIR
                    / out_name
                )

                torch.save(
                    out_records,
                    out_path,
                )

                derived_ledger[
                    out_name
                ] = compute_sha256(
                    out_path
                )

                print(
                    f"[+] Video "
                    f"{vid:02d}/47: "
                    f"{len(out_records):>4} "
                    "frames -> "
                    f"{out_name} "
                    f"({time.time() - t_vid:.1f}s)"
                )

            else:
                print(
                    "[+] Preflight "
                    f"video {vid:02d}: "
                    f"{len(out_records)} "
                    "deterministic "
                    "frames PASS"
                )

            del (
                production,
                source_frames,
                canonical_records,
                out_records,
            )

            gc.collect()

            if (
                DEVICE.type
                == "cuda"
            ):
                torch.cuda.empty_cache()

    finally:
        handle.remove()

    if args.preflight_only:
        expected = (
            3
            * len(
                preflight_videos
            )
        )

        if (
            total_frames
            != expected
        ):
            raise AssertionError(
                "[FATAL] Spatial "
                "preflight frame "
                f"count {total_frames} "
                f"!= {expected}"
            )

        print(
            "\n[+] SPATIAL "
            "DERIVED-EVIDENCE "
            "PREFLIGHT PASS"
        )

        print(
            "    Frames checked:             "
            f"{total_frames}"
        )

        print(
            "    Max live/cache logit drift: "
            f"{max_dz:.3e}"
        )

        print(
            "    Max live/cache box drift:   "
            f"{max_db:.3e}"
        )

        print(
            "    Max live/cache query drift: "
            f"{max_dq:.3e}"
        )

        print(
            "    Max live/cache ref drift:   "
            f"{max_dr:.3e}"
        )

        return

    if (
        total_frames
        != EXPECTED_TOTAL_FRAMES
    ):
        raise AssertionError(
            "[FATAL] Derived spatial "
            f"frame count {total_frames} "
            f"!= {EXPECTED_TOTAL_FRAMES}"
        )

    if (
        len(derived_ledger)
        != NUM_VIDEOS
    ):
        raise AssertionError(
            "[FATAL] Derived spatial "
            "cache does not contain "
            "47 shards"
        )

    manifest = {
        "_meta": {
            "schema_version": (
                SPATIAL_SCHEMA_VERSION
            ),
            "builder_sha256": (
                driver_sha
            ),
            "production_spatial_source_sha256": (
                production_spatial_sha
            ),
            "screening_config_sha256": (
                screening_config_sha
            ),
            "r0_checkpoint_sha256": (
                FROZEN_R0_CKPT_SHA256
            ),
            "census_sha256": (
                FROZEN_CENSUS_SHA256
            ),
            "train_json_sha256": (
                train_json_sha
            ),
            "canonical_cache_manifest_sha256": (
                canonical_manifest_sha
            ),
            "canonical_cache_builder_sha256": (
                canonical_manifest[
                    "_meta"
                ][
                    "cache_driver_sha256"
                ]
            ),
            "num_videos": (
                NUM_VIDEOS
            ),
            "total_frames": (
                total_frames
            ),
            "num_queries": (
                EXPECTED_NUM_QUERIES
            ),
            "scale_spatial_dim": 21,
            "localization_spatial_dim": 24,
            "scale_ring_expansion": (
                screening.scale_ring_expansion
            ),
            "localization_side_samples": (
                screening.localization_side_samples
            ),
            "localization_boundary_offset": (
                screening.localization_boundary_offset
            ),
            "max_live_cache_logit_drift": (
                max_dz
            ),
            "max_live_cache_box_drift": (
                max_db
            ),
            "max_live_cache_query_drift": (
                max_dq
            ),
            "max_live_cache_reference_drift": (
                max_dr
            ),
            "device": (
                str(DEVICE)
            ),
            "deformable_detr_root": (
                str(
                    deformable_root
                    .resolve()
                )
            ),
            "timestamp": (
                time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            ),
        },
        "derived_spatial_shard_sha256_ledger": (
            derived_ledger
        ),
    }

    manifest_path = (
        OUTPUT_DIR
        / (
            "bfcef_f_spatial_"
            "evidence_manifest.json"
        )
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest_sha = (
        compute_sha256(
            manifest_path
        )
    )

    print(
        "\n"
        + "=" * 112
    )

    print(
        "[+] B-FCEF-F SPATIAL "
        "DERIVED-EVIDENCE CACHE "
        "COMPLETE"
    )

    print(
        f"[+] Frames:       "
        f"{total_frames:,}"
    )

    print(
        f"[+] Manifest:     "
        f"{manifest_path}"
    )

    print(
        f"[+] Manifest SHA: "
        f"{manifest_sha}"
    )

    print(
        f"[+] Runtime:      "
        f"{(time.time() - t0) / 60.0:.2f} min"
    )

    print(
        "=" * 112
    )


if __name__ == "__main__":
    main()