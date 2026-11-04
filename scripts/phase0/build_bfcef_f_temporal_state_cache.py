from __future__ import annotations

"""
B-FCEF-F deterministic K-step temporal-state cache builder.

This script does NOT perform a new association search.

It deterministically chains the already-certified first-order E0 temporal
association cache via `matched_prev_index`, preserving the exact causal
association substrate while exposing the richer typed temporal state required
by the permanent T expert.

Per frame/query, up to K_max past steps are stored:

    disp_history           [Q,K,2]
    score_persistence      [Q,K]
    class_consistency      [Q,K]
    appearance_similarity  [Q,K]
    association_confidence [Q,K]
    same_slot              [Q,K]
    valid                  [Q,K]
    history_query_index    [Q,K]

Query summaries:

    track_age              [Q,1]
    velocity               [Q,2]
    acceleration           [Q,2]
    global_motion          [Q,2]

The exact E0 one-step temporal cue/proxy contract is also reproduced as:

    legacy_cues_t          [Q,8]
    n_t_proxy              [Q]
    r_t_proxy              [Q]

so T-Final retains the old evidence while gaining explicit causal history.
"""

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault(
    "OMP_NUM_THREADS",
    "1",
)
os.environ.setdefault(
    "MKL_NUM_THREADS",
    "1",
)
os.environ.setdefault(
    "OPENBLAS_NUM_THREADS",
    "1",
)

import torch
import torch.nn.functional as F


ROOT_DIR = Path(
    "D:/MARIVISION"
)

EVAL_DIR = (
    ROOT_DIR
    / "H0_workbench"
    / "eval_tools"
)

E0_CACHE_DIR = (
    EVAL_DIR
    / "e0_cache"
)

DERIVED_ROOT = (
    EVAL_DIR
    / "bfcef_f_derived_cache"
)

CANONICAL_CACHE_MANIFEST = (
    E0_CACHE_DIR
    / "bfcef_e0_cache_manifest.json"
)

TEMPORAL_CACHE_MANIFEST = (
    E0_CACHE_DIR
    / (
        "bfcef_e0_temporal_"
        "cache_manifest.json"
    )
)

R0_CHECKPOINT_PATH = (
    ROOT_DIR
    / "checkpoints_r0"
    / "checkpoint_best.pth"
)

CENSUS_JSON = (
    EVAL_DIR
    / (
        "marsen_b2_production_"
        "census_56shards.json"
    )
)

FROZEN_R0_CKPT_SHA256 = (
    "28e6288c3f6769c2d5275aa77ddb2487"
    "35b55a09cd46a6b570e3f117890b268b"
)

FROZEN_CENSUS_SHA256 = (
    "a20a734d668a9078bdfe274cb09fa5a47"
    "ad9c6a3a9d7cef2d1f99b926623fbc5"
)

FROZEN_ASSOCIATION_VERSION = (
    "t1_hungarian_v1_"
    "geom07iou03center_"
    "fusion05g03a02c_"
    "iou015_cost065"
)

NUM_VIDEOS = 47
EXPECTED_TOTAL_FRAMES = 20_075
EXPECTED_NUM_QUERIES = 300

TEMPORAL_SCHEMA_VERSION = (
    "bfcef_f_temporal_state_v1"
)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def compute_sha256(
    path: Path,
) -> str:
    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:
        while True:
            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def _ensure_tensor(
    value: torch.Tensor,
    *,
    shape: Tuple[int, ...]
    | None = None,
    dtype: torch.dtype
    | None = None,
    label: str,
) -> torch.Tensor:
    if not isinstance(
        value,
        torch.Tensor,
    ):
        raise AssertionError(
            f"[FATAL] {label} "
            "is not a torch.Tensor"
        )

    if (
        shape is not None
        and tuple(value.shape)
        != tuple(shape)
    ):
        raise AssertionError(
            f"[FATAL] {label} shape "
            f"{tuple(value.shape)} "
            f"!= {tuple(shape)}"
        )

    if (
        dtype is not None
        and value.dtype != dtype
    ):
        raise AssertionError(
            f"[FATAL] {label} dtype "
            f"{value.dtype} != {dtype}"
        )

    return value


# -----------------------------------------------------------------------------
# Provenance verification
# -----------------------------------------------------------------------------
def verify_source_provenance():
    for path, label in [
        (
            CANONICAL_CACHE_MANIFEST,
            "canonical cache manifest",
        ),
        (
            TEMPORAL_CACHE_MANIFEST,
            "temporal cache manifest",
        ),
        (
            R0_CHECKPOINT_PATH,
            "R0 checkpoint",
        ),
        (
            CENSUS_JSON,
            "production census",
        ),
    ]:
        if not path.exists():
            raise AssertionError(
                f"[FATAL] Missing "
                f"{label}: {path}"
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

    if (
        compute_sha256(
            CENSUS_JSON
        )
        != FROZEN_CENSUS_SHA256
    ):
        raise AssertionError(
            "[FATAL] Production "
            "census SHA drift"
        )

    canonical_sha = (
        compute_sha256(
            CANONICAL_CACHE_MANIFEST
        )
    )

    temporal_sha = (
        compute_sha256(
            TEMPORAL_CACHE_MANIFEST
        )
    )

    canonical = json.loads(
        CANONICAL_CACHE_MANIFEST
        .read_text(
            encoding="utf-8"
        )
    )

    temporal = json.loads(
        TEMPORAL_CACHE_MANIFEST
        .read_text(
            encoding="utf-8"
        )
    )

    c_meta = canonical[
        "_meta"
    ]

    if (
        c_meta[
            "actual_total_frames"
        ]
        != EXPECTED_TOTAL_FRAMES
    ):
        raise AssertionError(
            "[FATAL] Canonical cache "
            "total-frame drift"
        )

    if (
        c_meta[
            "num_videos"
        ]
        != NUM_VIDEOS
    ):
        raise AssertionError(
            "[FATAL] Canonical cache "
            "video-count drift"
        )

    if (
        c_meta[
            "r0_checkpoint_sha256"
        ]
        != FROZEN_R0_CKPT_SHA256
    ):
        raise AssertionError(
            "[FATAL] Canonical cache "
            "R0 provenance mismatch"
        )

    if (
        c_meta[
            "census_sha256"
        ]
        != FROZEN_CENSUS_SHA256
    ):
        raise AssertionError(
            "[FATAL] Canonical cache "
            "census provenance "
            "mismatch"
        )

    t_meta = temporal[
        "_meta"
    ]

    if (
        t_meta[
            "input_cache_manifest_sha256"
        ]
        != canonical_sha
    ):
        raise AssertionError(
            "[FATAL] First-order "
            "temporal cache was not "
            "built from the current "
            "canonical cache manifest"
        )

    if (
        t_meta[
            "association_version"
        ]
        != FROZEN_ASSOCIATION_VERSION
    ):
        raise AssertionError(
            "[FATAL] First-order "
            "temporal association-"
            "version drift"
        )

    if (
        t_meta[
            "actual_total_frames"
        ]
        != EXPECTED_TOTAL_FRAMES
    ):
        raise AssertionError(
            "[FATAL] First-order "
            "temporal cache total-"
            "frame drift"
        )

    if (
        t_meta[
            "num_videos"
        ]
        != NUM_VIDEOS
    ):
        raise AssertionError(
            "[FATAL] First-order "
            "temporal cache video-"
            "count drift"
        )

    c_ledger = canonical[
        "cache_shard_sha256_ledger"
    ]

    t_ledger = temporal[
        "temporal_cache_shard_sha256_ledger"
    ]

    if (
        len(c_ledger)
        != NUM_VIDEOS
        or len(t_ledger)
        != NUM_VIDEOS
    ):
        raise AssertionError(
            "[FATAL] Expected "
            "47 canonical + 47 "
            "temporal source shards"
        )

    for (
        name,
        expected_sha,
    ) in c_ledger.items():
        path = (
            E0_CACHE_DIR
            / name
        )

        if not path.exists():
            raise AssertionError(
                "[FATAL] Missing "
                "canonical cache "
                f"shard: {name}"
            )

        if (
            compute_sha256(path)
            != expected_sha
        ):
            raise AssertionError(
                "[FATAL] Canonical "
                f"cache SHA drift: "
                f"{name}"
            )

    for (
        name,
        expected_sha,
    ) in t_ledger.items():
        path = (
            E0_CACHE_DIR
            / name
        )

        if not path.exists():
            raise AssertionError(
                "[FATAL] Missing "
                "first-order temporal "
                f"cache shard: {name}"
            )

        if (
            compute_sha256(path)
            != expected_sha
        ):
            raise AssertionError(
                "[FATAL] Temporal "
                f"cache SHA drift: "
                f"{name}"
            )

    return (
        canonical,
        temporal,
        canonical_sha,
        temporal_sha,
    )


# -----------------------------------------------------------------------------
# Temporal-state construction
# -----------------------------------------------------------------------------
def _query_scores(
    logits: torch.Tensor,
) -> torch.Tensor:
    return (
        logits
        .sigmoid()
        .max(dim=-1)
        .values
    )


def _class_consistency(
    curr_logits: torch.Tensor,
    prev_logits: torch.Tensor,
) -> torch.Tensor:
    """
    Cosine consistency of sigmoid
    class-probability vectors.

    Range is clipped to [0,1].
    """

    curr_p = (
        curr_logits.sigmoid()
    )

    prev_p = (
        prev_logits.sigmoid()
    )

    cos = F.cosine_similarity(
        curr_p,
        prev_p,
        dim=-1,
        eps=1e-8,
    )

    return cos.clamp(
        0.0,
        1.0,
    )


def _legacy_temporal_cues(
    canonical_record: dict,
    temporal_record: dict,
):
    """
    Reproduce the certified E0
    temporal cue/proxy semantics.

    returns:

        cues_t       [Q,8]
        n_t_proxy    [Q]
        r_t_proxy    [Q]
        availability [Q]
    """

    d_boxes = (
        canonical_record[
            "d_boxes"
        ]
        .float()
    )

    d_logits = (
        canonical_record[
            "d_logits"
        ]
        .float()
    )

    scores = _query_scores(
        d_logits
    )

    matched_boxes = (
        temporal_record[
            "matched_boxes"
        ]
        .float()
    )

    match_conf = (
        temporal_record[
            "match_confidence"
        ]
        .float()
    )

    app_sim = (
        temporal_record[
            "appearance_similarity"
        ]
        .float()
    )

    avail = (
        match_conf
        > 0.10
    ).float()

    disp_x = (
        d_boxes[:, 0]
        - matched_boxes[:, 0]
    ) * 5.0

    disp_y = (
        d_boxes[:, 1]
        - matched_boxes[:, 1]
    ) * 5.0

    disp_mag = torch.sqrt(
        disp_x ** 2
        + disp_y ** 2
        + 1e-6
    )

    r_proxy = (
        torch.clamp(
            match_conf
            * app_sim,
            0.1,
            1.0,
        )
        * avail
    )

    n_proxy = (
        torch.clamp(
            disp_mag
            + (
                1.0
                - scores
            ),
            0.0,
            1.0,
        )
        * avail
    )

    cues = torch.stack(
        [
            disp_x,
            disp_y,
            disp_mag,
            match_conf,
            app_sim,
            scores,
            n_proxy,
            r_proxy,
        ],
        dim=-1,
    ) * avail.unsqueeze(-1)

    return (
        cues,
        n_proxy,
        r_proxy,
        avail,
    )


def build_frame_kstep_state(
    canonical_records: List[dict],
    temporal_records: List[dict],
    frame_idx: int,
    k_max: int,
) -> dict:
    current = canonical_records[
        frame_idx
    ]

    q_count = int(
        current[
            "d_boxes"
        ].shape[0]
    )

    if (
        q_count
        != EXPECTED_NUM_QUERIES
    ):
        raise AssertionError(
            "[FATAL] Expected "
            f"{EXPECTED_NUM_QUERIES} "
            f"queries, got {q_count}"
        )

    dtype = torch.float32

    disp_history = torch.zeros(
        q_count,
        k_max,
        2,
        dtype=dtype,
    )

    score_persistence = (
        torch.zeros(
            q_count,
            k_max,
            dtype=dtype,
        )
    )

    class_consistency = (
        torch.zeros(
            q_count,
            k_max,
            dtype=dtype,
        )
    )

    appearance_similarity = (
        torch.zeros(
            q_count,
            k_max,
            dtype=dtype,
        )
    )

    association_confidence = (
        torch.zeros(
            q_count,
            k_max,
            dtype=dtype,
        )
    )

    same_slot = torch.zeros(
        q_count,
        k_max,
        dtype=dtype,
    )

    valid = torch.zeros(
        q_count,
        k_max,
        dtype=torch.bool,
    )

    history_query_index = (
        torch.full(
            (
                q_count,
                k_max,
            ),
            -1,
            dtype=torch.int64,
        )
    )

    q_current = torch.arange(
        q_count,
        dtype=torch.int64,
    )

    chain_alive = torch.ones(
        q_count,
        dtype=torch.bool,
    )

    for step in range(
        k_max
    ):
        curr_frame_idx = (
            frame_idx
            - step
        )

        prev_frame_idx = (
            curr_frame_idx
            - 1
        )

        if (
            prev_frame_idx
            < 0
        ):
            break

        curr_canonical = (
            canonical_records[
                curr_frame_idx
            ]
        )

        prev_canonical = (
            canonical_records[
                prev_frame_idx
            ]
        )

        curr_temporal = (
            temporal_records[
                curr_frame_idx
            ]
        )

        matched_prev = (
            _ensure_tensor(
                curr_temporal[
                    "matched_prev_index"
                ],
                shape=(
                    q_count,
                ),
                dtype=torch.int64,
                label=(
                    "matched_prev_index "
                    f"frame "
                    f"{curr_frame_idx}"
                ),
            )
        )

        match_conf = (
            _ensure_tensor(
                curr_temporal[
                    "match_confidence"
                ],
                shape=(
                    q_count,
                ),
                label=(
                    "match_confidence "
                    f"frame "
                    f"{curr_frame_idx}"
                ),
            )
            .float()
        )

        app_sim = (
            _ensure_tensor(
                curr_temporal[
                    "appearance_similarity"
                ],
                shape=(
                    q_count,
                ),
                label=(
                    "appearance_similarity "
                    f"frame "
                    f"{curr_frame_idx}"
                ),
            )
            .float()
        )

        safe_curr = (
            q_current.clamp(
                0,
                q_count - 1,
            )
        )

        prev_q = (
            matched_prev[
                safe_curr
            ]
        )

        step_valid = (
            chain_alive
            & (prev_q >= 0)
            & (prev_q < q_count)
        )

        safe_prev = (
            prev_q.clamp(
                0,
                q_count - 1,
            )
        )

        curr_boxes = (
            curr_canonical[
                "d_boxes"
            ]
            .float()[
                safe_curr
            ]
        )

        prev_boxes = (
            prev_canonical[
                "d_boxes"
            ]
            .float()[
                safe_prev
            ]
        )

        curr_logits = (
            curr_canonical[
                "d_logits"
            ]
            .float()[
                safe_curr
            ]
        )

        prev_logits = (
            prev_canonical[
                "d_logits"
            ]
            .float()[
                safe_prev
            ]
        )

        step_disp = (
            curr_boxes[:, :2]
            - prev_boxes[:, :2]
        )

        curr_score = (
            _query_scores(
                curr_logits
            )
        )

        prev_score = (
            _query_scores(
                prev_logits
            )
        )

        step_score_persistence = (
            1.0
            - (
                curr_score
                - prev_score
            ).abs()
        ).clamp(
            0.0,
            1.0,
        )

        step_class_consistency = (
            _class_consistency(
                curr_logits,
                prev_logits,
            )
        )

        step_conf = (
            match_conf[
                safe_curr
            ]
            .clamp(
                0.0,
                1.0,
            )
        )

        step_app = (
            app_sim[
                safe_curr
            ]
            .clamp(
                -1.0,
                1.0,
            )
        )

        step_same_slot = (
            safe_prev
            == safe_curr
        ).float()

        mask_f = (
            step_valid.float()
        )

        disp_history[
            :,
            step,
        ] = (
            step_disp
            * mask_f[:, None]
        )

        score_persistence[
            :,
            step,
        ] = (
            step_score_persistence
            * mask_f
        )

        class_consistency[
            :,
            step,
        ] = (
            step_class_consistency
            * mask_f
        )

        appearance_similarity[
            :,
            step,
        ] = (
            step_app
            * mask_f
        )

        association_confidence[
            :,
            step,
        ] = (
            step_conf
            * mask_f
        )

        same_slot[
            :,
            step,
        ] = (
            step_same_slot
            * mask_f
        )

        valid[
            :,
            step,
        ] = (
            step_valid
        )

        history_query_index[
            :,
            step,
        ] = torch.where(
            step_valid,
            safe_prev,
            torch.full_like(
                safe_prev,
                -1,
            ),
        )

        q_current = torch.where(
            step_valid,
            safe_prev,
            q_current,
        )

        chain_alive = (
            step_valid
        )

    # Validity is constructed
    # causally. Once a chain dies,
    # all older positions remain
    # invalid.
    track_age = (
        valid.sum(
            dim=1,
            keepdim=True,
        )
        .float()
    )

    valid_f = valid.float()

    denom = (
        valid_f
        .sum(
            dim=1,
            keepdim=True,
        )
        .clamp_min(1.0)
    )

    velocity = (
        (
            disp_history
            * valid_f.unsqueeze(-1)
        )
        .sum(dim=1)
        / denom
    )

    acceleration = torch.zeros(
        q_count,
        2,
        dtype=dtype,
    )

    if k_max >= 2:
        accel_valid = (
            valid[:, 0]
            & valid[:, 1]
        )

        acceleration[
            accel_valid
        ] = (
            disp_history[
                accel_valid,
                0,
            ]
            - disp_history[
                accel_valid,
                1,
            ]
        )

    global_motion_vec = (
        torch.zeros(
            2,
            dtype=dtype,
        )
    )

    if k_max > 0:
        reliable = (
            valid[:, 0]
            & (
                association_confidence[
                    :,
                    0,
                ]
                > 0.10
            )
        )

        if (
            int(
                reliable
                .sum()
                .item()
            )
            > 0
        ):
            global_motion_vec = (
                disp_history[
                    reliable,
                    0,
                ]
                .median(
                    dim=0
                )
                .values
            )

    global_motion = (
        global_motion_vec
        .unsqueeze(0)
        .expand(
            q_count,
            2,
        )
        .clone()
    )

    (
        legacy_cues_t,
        n_t_proxy,
        r_t_proxy,
        legacy_available,
    ) = _legacy_temporal_cues(
        current,
        temporal_records[
            frame_idx
        ],
    )

    result = {
        "legacy_cues_t": (
            legacy_cues_t
        ),
        "n_t_proxy": (
            n_t_proxy
        ),
        "r_t_proxy": (
            r_t_proxy
        ),
        "legacy_available": (
            legacy_available
        ),
        "disp_history": (
            disp_history
        ),
        "score_persistence": (
            score_persistence
        ),
        "class_consistency": (
            class_consistency
        ),
        "appearance_similarity": (
            appearance_similarity
        ),
        "association_confidence": (
            association_confidence
        ),
        "same_slot": (
            same_slot
        ),
        "valid": (
            valid
        ),
        "history_query_index": (
            history_query_index
        ),
        "track_age": (
            track_age
        ),
        "velocity": (
            velocity
        ),
        "acceleration": (
            acceleration
        ),
        "global_motion": (
            global_motion
        ),
    }

    expected_shapes = {
        "legacy_cues_t": (
            q_count,
            8,
        ),
        "n_t_proxy": (
            q_count,
        ),
        "r_t_proxy": (
            q_count,
        ),
        "legacy_available": (
            q_count,
        ),
        "disp_history": (
            q_count,
            k_max,
            2,
        ),
        "score_persistence": (
            q_count,
            k_max,
        ),
        "class_consistency": (
            q_count,
            k_max,
        ),
        "appearance_similarity": (
            q_count,
            k_max,
        ),
        "association_confidence": (
            q_count,
            k_max,
        ),
        "same_slot": (
            q_count,
            k_max,
        ),
        "valid": (
            q_count,
            k_max,
        ),
        "history_query_index": (
            q_count,
            k_max,
        ),
        "track_age": (
            q_count,
            1,
        ),
        "velocity": (
            q_count,
            2,
        ),
        "acceleration": (
            q_count,
            2,
        ),
        "global_motion": (
            q_count,
            2,
        ),
    }

    for key, shape in (
        expected_shapes.items()
    ):
        if (
            tuple(
                result[key].shape
            )
            != shape
        ):
            raise AssertionError(
                "[FATAL] Temporal "
                f"state {key} shape "
                f"{tuple(result[key].shape)} "
                f"!= {shape}"
            )

        if (
            result[key]
            .dtype
            .is_floating_point
            and not torch.isfinite(
                result[key]
            ).all()
        ):
            raise AssertionError(
                "[FATAL] Non-finite "
                "temporal state "
                f"values in {key}"
            )

    return result


def verify_frame_alignment(
    canonical_records: List[dict],
    temporal_records: List[dict],
    vid: int,
) -> None:
    if (
        len(canonical_records)
        != len(temporal_records)
    ):
        raise AssertionError(
            "[FATAL] Video "
            f"{vid}: canonical/"
            "temporal record-count "
            "mismatch"
        )

    for idx, (
        canonical,
        temporal,
    ) in enumerate(
        zip(
            canonical_records,
            temporal_records,
        )
    ):
        if (
            int(
                canonical[
                    "image_id"
                ]
            )
            != int(
                temporal[
                    "image_id"
                ]
            )
        ):
            raise AssertionError(
                "[FATAL] Video "
                f"{vid} seq {idx}: "
                "image_id mismatch"
            )

        if (
            int(
                canonical[
                    "video_id"
                ]
            )
            != vid
            or int(
                temporal[
                    "video_id"
                ]
            )
            != vid
        ):
            raise AssertionError(
                "[FATAL] Video "
                f"{vid} seq {idx}: "
                "video_id mismatch"
            )

        if (
            int(
                canonical[
                    "seq_idx"
                ]
            )
            != idx
            or int(
                temporal[
                    "seq_idx"
                ]
            )
            != idx
        ):
            raise AssertionError(
                "[FATAL] Video "
                f"{vid}: non-canonical "
                f"seq_idx at position "
                f"{idx}"
            )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--k-max",
        type=int,
        default=7,
        help=(
            "Maximum causal history "
            "depth to materialize. "
            "S0 may slice any "
            "K <= K_max."
        ),
    )

    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Verify provenance and "
            "deterministic K-step "
            "construction on "
            "representative videos "
            "only."
        ),
    )

    args = parser.parse_args()

    if args.k_max < 1:
        raise ValueError(
            "--k-max must be >= 1"
        )

    t0 = time.time()

    builder_sha = compute_sha256(
        Path(__file__)
    )

    (
        canonical_manifest,
        temporal_manifest,
        canonical_sha,
        temporal_sha,
    ) = verify_source_provenance()

    output_dir = (
        DERIVED_ROOT
        / (
            "temporal_kmax"
            f"{args.k_max}_v1"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 112
    )

    print(
        "B-FCEF-F DETERMINISTIC "
        "K-STEP TEMPORAL-STATE "
        "CACHE BUILDER"
    )

    print(
        "Builder SHA-256:              "
        f"{builder_sha}"
    )

    print(
        "Canonical manifest SHA-256:   "
        f"{canonical_sha}"
    )

    print(
        "Temporal manifest SHA-256:    "
        f"{temporal_sha}"
    )

    print(
        "Association version:          "
        f"{FROZEN_ASSOCIATION_VERSION}"
    )

    print(
        "K_max:                        "
        f"{args.k_max}"
    )

    print(
        "=" * 112,
        flush=True,
    )

    output_ledger: Dict[
        str,
        str,
    ] = {}

    total_frames = 0
    total_valid_links = 0
    total_possible_links = 0

    preflight_videos = {
        1,
        12,
        24,
        36,
        47,
    }

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

        canonical_path = (
            E0_CACHE_DIR
            / (
                "bfcef_e0_cache_"
                f"vid{vid}.pth"
            )
        )

        temporal_path = (
            E0_CACHE_DIR
            / (
                "bfcef_e0_temporal_"
                f"cache_vid{vid}.pth"
            )
        )

        canonical_records = (
            torch.load(
                canonical_path,
                map_location="cpu",
            )
        )

        temporal_records = (
            torch.load(
                temporal_path,
                map_location="cpu",
            )
        )

        verify_frame_alignment(
            canonical_records,
            temporal_records,
            vid,
        )

        if args.preflight_only:
            n = len(
                canonical_records
            )

            frame_indices = sorted(
                set(
                    [
                        0,
                        (
                            1
                            if n > 1
                            else 0
                        ),
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

        for frame_idx in (
            frame_indices
        ):
            state_1 = (
                build_frame_kstep_state(
                    canonical_records,
                    temporal_records,
                    frame_idx,
                    args.k_max,
                )
            )

            if args.preflight_only:
                state_2 = (
                    build_frame_kstep_state(
                        canonical_records,
                        temporal_records,
                        frame_idx,
                        args.k_max,
                    )
                )

                for key in state_1:
                    if not torch.equal(
                        state_1[key],
                        state_2[key],
                    ):
                        raise AssertionError(
                            "[FATAL] "
                            "Non-deterministic "
                            "temporal state: "
                            f"video {vid} "
                            f"frame {frame_idx} "
                            f"key {key}"
                        )

            valid_links = int(
                state_1[
                    "valid"
                ]
                .sum()
                .item()
            )

            total_valid_links += (
                valid_links
            )

            total_possible_links += (
                EXPECTED_NUM_QUERIES
                * args.k_max
            )

            canonical = (
                canonical_records[
                    frame_idx
                ]
            )

            out_records.append(
                {
                    "image_id": int(
                        canonical[
                            "image_id"
                        ]
                    ),
                    "video_id": (
                        vid
                    ),
                    "seq_idx": (
                        frame_idx
                    ),
                    **state_1,
                }
            )

            total_frames += 1

        if args.preflight_only:
            print(
                "[+] Preflight "
                f"video {vid:02d}: "
                f"{len(out_records)} "
                "deterministic "
                "frames PASS"
            )

        else:
            out_name = (
                "bfcef_f_temporal_"
                "state_kmax"
                f"{args.k_max}_"
                f"vid{vid}.pth"
            )

            out_path = (
                output_dir
                / out_name
            )

            torch.save(
                out_records,
                out_path,
            )

            output_ledger[
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

        del (
            canonical_records,
            temporal_records,
            out_records,
        )

        gc.collect()

    if args.preflight_only:
        expected_per_vid = 4

        expected_frames = (
            expected_per_vid
            * len(
                preflight_videos
            )
        )

        if (
            total_frames
            != expected_frames
        ):
            raise AssertionError(
                "[FATAL] Temporal "
                "preflight frames "
                f"{total_frames} "
                f"!= {expected_frames}"
            )

        coverage = (
            total_valid_links
            / max(
                1,
                total_possible_links,
            )
        )

        print(
            "\n[+] K-STEP TEMPORAL-"
            "STATE PREFLIGHT PASS"
        )

        print(
            "    Frames checked:      "
            f" {total_frames}"
        )

        print(
            "    Valid-link coverage: "
            f" {coverage:.4%}"
        )

        return

    if (
        total_frames
        != EXPECTED_TOTAL_FRAMES
    ):
        raise AssertionError(
            "[FATAL] Temporal-state "
            f"frame count {total_frames} "
            f"!= {EXPECTED_TOTAL_FRAMES}"
        )

    if (
        len(output_ledger)
        != NUM_VIDEOS
    ):
        raise AssertionError(
            "[FATAL] Temporal-state "
            "output does not contain "
            "47 shards"
        )

    coverage = (
        total_valid_links
        / max(
            1,
            total_possible_links,
        )
    )

    manifest = {
        "_meta": {
            "schema_version": (
                TEMPORAL_SCHEMA_VERSION
            ),
            "builder_sha256": (
                builder_sha
            ),
            "r0_checkpoint_sha256": (
                FROZEN_R0_CKPT_SHA256
            ),
            "census_sha256": (
                FROZEN_CENSUS_SHA256
            ),
            "canonical_cache_manifest_sha256": (
                canonical_sha
            ),
            "first_order_temporal_manifest_sha256": (
                temporal_sha
            ),
            "association_version": (
                FROZEN_ASSOCIATION_VERSION
            ),
            "source_iou_threshold": (
                temporal_manifest[
                    "_meta"
                ][
                    "iou_threshold"
                ]
            ),
            "source_max_cost_threshold": (
                temporal_manifest[
                    "_meta"
                ][
                    "max_cost_threshold"
                ]
            ),
            "source_fusion_weights": (
                temporal_manifest[
                    "_meta"
                ][
                    "fusion_weights"
                ]
            ),
            "k_max": (
                args.k_max
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
            "valid_link_coverage": (
                coverage
            ),
            "history_semantics": (
                "causal_chain_over_"
                "certified_matched_"
                "prev_index"
            ),
            "global_motion_semantics": (
                "median_current_step_"
                "displacement_conf_gt_"
                "0.10"
            ),
            "timestamp": (
                time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            ),
        },
        "derived_temporal_state_shard_sha256_ledger": (
            output_ledger
        ),
    }

    manifest_path = (
        output_dir
        / (
            "bfcef_f_temporal_"
            "state_manifest.json"
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
        "[+] B-FCEF-F K-STEP "
        "TEMPORAL-STATE CACHE "
        "COMPLETE"
    )

    print(
        f"[+] Frames:              "
        f"{total_frames:,}"
    )

    print(
        "[+] Valid-link coverage: "
        f"{coverage:.4%}"
    )

    print(
        f"[+] Manifest:            "
        f"{manifest_path}"
    )

    print(
        f"[+] Manifest SHA:        "
        f"{manifest_sha}"
    )

    print(
        f"[+] Runtime:             "
        f"{(time.time() - t0) / 60.0:.2f} min"
    )

    print(
        "=" * 112
    )


if __name__ == "__main__":
    main()