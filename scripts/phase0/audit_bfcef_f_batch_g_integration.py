from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch


# =====================================================================
# PATHS
# =====================================================================

ROOT = Path(
    "D:/MARIVISION"
)

EVAL = (
    ROOT
    / "H0_workbench"
    / "eval_tools"
)

E0 = (
    EVAL
    / "e0_cache"
)

DERIVED = (
    EVAL
    / "bfcef_f_derived_cache"
)

CAN_MANIFEST = (
    E0
    / "bfcef_e0_cache_manifest.json"
)

FIRST_TEMP_MANIFEST = (
    E0
    / "bfcef_e0_temporal_cache_manifest.json"
)

SPATIAL_DIR = (
    DERIVED
    / "spatial_v2"
)

SPATIAL_MANIFEST = (
    SPATIAL_DIR
    / "bfcef_f_spatial_evidence_manifest.json"
)

TEMP_DIR = (
    DERIVED
    / "temporal_kmax7_v1"
)

TEMP_MANIFEST = (
    TEMP_DIR
    / "bfcef_f_temporal_state_manifest.json"
)

SCREENING_CONFIG = (
    EVAL
    / "screening_config_s0.json"
)

SPATIAL_SOURCE = (
    EVAL
    / "bfcef_f"
    / "spatial_evidence.py"
)


# =====================================================================
# FROZEN ROOTS
# =====================================================================

EXPECTED_CAN_SHA = (
    "210ad12de4ebe02cdbe000956972955ae"
    "58ceb8f0d222b703787f33a49af03cd"
)

EXPECTED_FIRST_TEMP_SHA = (
    "e4754d9c82c60836d817e49ceb552018"
    "07c22405677558bb2a69b3e96a6bd286"
)

EXPECTED_SPATIAL_MANIFEST_SHA = (
    "8899edf3eb3b46428dc3a2186058063c"
    "74045d14e0c75744fc0a0793daefa9b5"
)

EXPECTED_TEMP_MANIFEST_SHA = (
    "d84d2b75cc043da855c1490a66e727b6"
    "77c7c70cb7ab881c7c08b2308bf3e985"
)

EXPECTED_SPATIAL_BUILDER_SHA = (
    "84cae19d886f628bbdecea5151d07b3c"
    "f9ec20a18e8725098baf42ba64901cdf"
)

EXPECTED_TEMP_BUILDER_SHA = (
    "dd4bf0bc13e175ca9e28ded4e0d8978f"
    "666480e4445a7807b3b7642d94b579b5"
)

EXPECTED_SPATIAL_SOURCE_SHA = (
    "e2bb7899025408d51a09aa2c289c3288"
    "312ae07bd9e10c442dbed62d796c73ce"
)

EXPECTED_SPATIAL_SCHEMA = (
    "bfcef_f_spatial_v2_production_s21_l24"
)

EXPECTED_TEMP_SCHEMA = (
    "bfcef_f_temporal_state_v1"
)

EXPECTED_ASSOCIATION = (
    "t1_hungarian_v1_"
    "geom07iou03center_"
    "fusion05g03a02c_"
    "iou015_cost065"
)


NUM_VIDEOS = 47
TOTAL_FRAMES = 20_075
Q = 300

KMAX = 7
K_S0 = 3

SAMPLE_VIDEOS = (
    1,
    12,
    24,
    36,
    47,
)


# =====================================================================
# HELPERS
# =====================================================================

def sha256(
    path: Path,
) -> str:

    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(
                chunk
            )

    return h.hexdigest()


def require_sha(
    path: Path,
    expected: str,
    label: str,
) -> None:

    if not path.exists():

        raise AssertionError(
            "[FATAL] Missing "
            f"{label}: {path}"
        )

    actual = sha256(
        path
    )

    if actual != expected:

        raise AssertionError(
            f"[FATAL] {label} SHA drift:\n"
            f"  expected={expected}\n"
            f"  actual  ={actual}"
        )


def load_json(
    path: Path,
) -> dict[str, Any]:

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def assert_finite(
    name: str,
    tensor: torch.Tensor,
) -> None:

    if (
        tensor.dtype.is_floating_point
        and not torch.isfinite(
            tensor
        ).all()
    ):

        raise AssertionError(
            "[FATAL] Non-finite "
            f"values in {name}"
        )


# =====================================================================
# MANIFEST AUDIT
# =====================================================================

def audit_manifests():

    require_sha(
        CAN_MANIFEST,
        EXPECTED_CAN_SHA,
        "canonical manifest",
    )

    require_sha(
        FIRST_TEMP_MANIFEST,
        EXPECTED_FIRST_TEMP_SHA,
        "first-order temporal manifest",
    )

    require_sha(
        SPATIAL_MANIFEST,
        EXPECTED_SPATIAL_MANIFEST_SHA,
        "derived spatial manifest",
    )

    require_sha(
        TEMP_MANIFEST,
        EXPECTED_TEMP_MANIFEST_SHA,
        "derived K-step temporal manifest",
    )

    canonical = load_json(
        CAN_MANIFEST
    )

    first_temporal = load_json(
        FIRST_TEMP_MANIFEST
    )

    spatial = load_json(
        SPATIAL_MANIFEST
    )

    temporal = load_json(
        TEMP_MANIFEST
    )

    cm = canonical[
        "_meta"
    ]

    fm = first_temporal[
        "_meta"
    ]

    sm = spatial[
        "_meta"
    ]

    tm = temporal[
        "_meta"
    ]

    # -------------------------------------------------------------
    # Canonical roots
    # -------------------------------------------------------------

    if (
        int(
            cm[
                "actual_total_frames"
            ]
        )
        != TOTAL_FRAMES
        or int(
            cm[
                "num_videos"
            ]
        )
        != NUM_VIDEOS
    ):

        raise AssertionError(
            "[FATAL] Canonical "
            "manifest cardinality drift."
        )

    if (
        int(
            fm[
                "actual_total_frames"
            ]
        )
        != TOTAL_FRAMES
        or int(
            fm[
                "num_videos"
            ]
        )
        != NUM_VIDEOS
    ):

        raise AssertionError(
            "[FATAL] First-order "
            "temporal manifest "
            "cardinality drift."
        )

    # -------------------------------------------------------------
    # Spatial manifest
    # -------------------------------------------------------------

    if (
        sm[
            "schema_version"
        ]
        != EXPECTED_SPATIAL_SCHEMA
    ):

        raise AssertionError(
            "[FATAL] Spatial schema drift: "
            f"{sm['schema_version']}"
        )

    if (
        sm[
            "builder_sha256"
        ]
        != EXPECTED_SPATIAL_BUILDER_SHA
    ):

        raise AssertionError(
            "[FATAL] Spatial builder "
            "SHA mismatch inside manifest."
        )

    if (
        sm[
            "production_spatial_source_sha256"
        ]
        != EXPECTED_SPATIAL_SOURCE_SHA
    ):

        raise AssertionError(
            "[FATAL] Production spatial "
            "source SHA mismatch."
        )

    if (
        sha256(
            SPATIAL_SOURCE
        )
        != EXPECTED_SPATIAL_SOURCE_SHA
    ):

        raise AssertionError(
            "[FATAL] Current "
            "bfcef_f/spatial_evidence.py "
            "has drifted after cache generation."
        )

    current_screening_sha = (
        sha256(
            SCREENING_CONFIG
        )
    )

    if (
        sm[
            "screening_config_sha256"
        ]
        != current_screening_sha
    ):

        raise AssertionError(
            "[FATAL] Spatial cache "
            "is not bound to the current "
            "screening_config_s0.json."
        )

    if (
        sm[
            "canonical_cache_manifest_sha256"
        ]
        != EXPECTED_CAN_SHA
    ):

        raise AssertionError(
            "[FATAL] Spatial canonical "
            "source mismatch."
        )

    if (
        int(
            sm[
                "num_videos"
            ]
        )
        != NUM_VIDEOS
        or int(
            sm[
                "total_frames"
            ]
        )
        != TOTAL_FRAMES
    ):

        raise AssertionError(
            "[FATAL] Spatial "
            "manifest cardinality drift."
        )

    if (
        int(
            sm[
                "num_queries"
            ]
        )
        != Q
    ):

        raise AssertionError(
            "[FATAL] Spatial "
            "num_queries drift."
        )

    if (
        int(
            sm[
                "scale_spatial_dim"
            ]
        )
        != 21
        or int(
            sm[
                "localization_spatial_dim"
            ]
        )
        != 24
    ):

        raise AssertionError(
            "[FATAL] Spatial descriptor "
            "dimension drift."
        )

    for key in (
        "max_live_cache_logit_drift",
        "max_live_cache_box_drift",
        "max_live_cache_query_drift",
        "max_live_cache_reference_drift",
    ):

        if (
            float(
                sm[
                    key
                ]
            )
            >= 1.0e-4
        ):

            raise AssertionError(
                "[FATAL] Spatial full-run "
                "R0 parity failed: "
                f"{key}={sm[key]}"
            )

    # -------------------------------------------------------------
    # Temporal manifest
    # -------------------------------------------------------------

    if (
        tm[
            "schema_version"
        ]
        != EXPECTED_TEMP_SCHEMA
    ):

        raise AssertionError(
            "[FATAL] Temporal schema drift: "
            f"{tm['schema_version']}"
        )

    if (
        tm[
            "builder_sha256"
        ]
        != EXPECTED_TEMP_BUILDER_SHA
    ):

        raise AssertionError(
            "[FATAL] Temporal builder "
            "SHA mismatch."
        )

    if (
        tm[
            "canonical_cache_manifest_sha256"
        ]
        != EXPECTED_CAN_SHA
    ):

        raise AssertionError(
            "[FATAL] K-step temporal "
            "canonical source mismatch."
        )

    if (
        tm[
            "first_order_temporal_manifest_sha256"
        ]
        != EXPECTED_FIRST_TEMP_SHA
    ):

        raise AssertionError(
            "[FATAL] K-step temporal "
            "first-order source mismatch."
        )

    if (
        tm[
            "association_version"
        ]
        != EXPECTED_ASSOCIATION
    ):

        raise AssertionError(
            "[FATAL] Temporal "
            "association-version drift."
        )

    if (
        int(
            tm[
                "k_max"
            ]
        )
        != KMAX
    ):

        raise AssertionError(
            "[FATAL] Temporal Kmax drift."
        )

    if (
        int(
            tm[
                "num_videos"
            ]
        )
        != NUM_VIDEOS
        or int(
            tm[
                "total_frames"
            ]
        )
        != TOTAL_FRAMES
    ):

        raise AssertionError(
            "[FATAL] Temporal "
            "manifest cardinality drift."
        )

    if (
        int(
            tm[
                "num_queries"
            ]
        )
        != Q
    ):

        raise AssertionError(
            "[FATAL] Temporal "
            "num_queries drift."
        )

    spatial_ledger = (
        spatial[
            "derived_spatial_shard_sha256_ledger"
        ]
    )

    temporal_ledger = (
        temporal[
            "derived_temporal_state_shard_sha256_ledger"
        ]
    )

    if (
        len(
            spatial_ledger
        )
        != NUM_VIDEOS
    ):

        raise AssertionError(
            "[FATAL] Spatial manifest "
            "does not contain exactly "
            "47 shard hashes."
        )

    if (
        len(
            temporal_ledger
        )
        != NUM_VIDEOS
    ):

        raise AssertionError(
            "[FATAL] Temporal manifest "
            "does not contain exactly "
            "47 shard hashes."
        )

    print(
        "[+] Manifest / provenance roots PASS"
    )

    print(
        "    Spatial manifest: "
        f"{EXPECTED_SPATIAL_MANIFEST_SHA}"
    )

    print(
        "    Temporal manifest: "
        f"{EXPECTED_TEMP_MANIFEST_SHA}"
    )

    print(
        "    Full temporal "
        "valid-link coverage: "
        f"{float(tm['valid_link_coverage']):.4%}"
    )

    return (
        spatial,
        temporal,
    )


# =====================================================================
# TEMPORAL RECORD CONTRACT
# =====================================================================

TEMP_SHAPES = {

    "legacy_cues_t":
        (
            Q,
            8,
        ),

    "n_t_proxy":
        (
            Q,
        ),

    "r_t_proxy":
        (
            Q,
        ),

    "legacy_available":
        (
            Q,
        ),

    "disp_history":
        (
            Q,
            KMAX,
            2,
        ),

    "score_persistence":
        (
            Q,
            KMAX,
        ),

    "class_consistency":
        (
            Q,
            KMAX,
        ),

    "appearance_similarity":
        (
            Q,
            KMAX,
        ),

    "association_confidence":
        (
            Q,
            KMAX,
        ),

    "same_slot":
        (
            Q,
            KMAX,
        ),

    "valid":
        (
            Q,
            KMAX,
        ),

    "history_query_index":
        (
            Q,
            KMAX,
        ),

    "track_age":
        (
            Q,
            1,
        ),

    "velocity":
        (
            Q,
            2,
        ),

    "acceleration":
        (
            Q,
            2,
        ),

    "global_motion":
        (
            Q,
            2,
        ),
}


# =====================================================================
# COMPLETE 20,075-FRAME DISK AUDIT
# =====================================================================

def audit_all_shards(
    spatial_manifest,
    temporal_manifest,
):

    spatial_ledger = (
        spatial_manifest[
            "derived_spatial_shard_sha256_ledger"
        ]
    )

    temporal_ledger = (
        temporal_manifest[
            "derived_temporal_state_shard_sha256_ledger"
        ]
    )

    total = 0

    for vid in range(
        1,
        NUM_VIDEOS + 1,
    ):

        canonical_path = (
            E0
            / (
                "bfcef_e0_cache_"
                f"vid{vid}.pth"
            )
        )

        spatial_path = (
            SPATIAL_DIR
            / (
                "bfcef_f_spatial_"
                "evidence_"
                f"vid{vid}.pth"
            )
        )

        temporal_path = (
            TEMP_DIR
            / (
                "bfcef_f_temporal_"
                f"state_kmax{KMAX}_"
                f"vid{vid}.pth"
            )
        )

        if (
            spatial_path.name
            not in spatial_ledger
        ):

            raise AssertionError(
                "[FATAL] Missing "
                "spatial ledger entry "
                f"for video {vid}."
            )

        if (
            temporal_path.name
            not in temporal_ledger
        ):

            raise AssertionError(
                "[FATAL] Missing "
                "temporal ledger entry "
                f"for video {vid}."
            )

        if (
            sha256(
                spatial_path
            )
            != spatial_ledger[
                spatial_path.name
            ]
        ):

            raise AssertionError(
                "[FATAL] Spatial "
                "shard SHA drift: "
                f"video {vid}."
            )

        if (
            sha256(
                temporal_path
            )
            != temporal_ledger[
                temporal_path.name
            ]
        ):

            raise AssertionError(
                "[FATAL] Temporal "
                "shard SHA drift: "
                f"video {vid}."
            )

        canonical = torch.load(
            canonical_path,
            map_location="cpu",
        )

        spatial = torch.load(
            spatial_path,
            map_location="cpu",
        )

        temporal = torch.load(
            temporal_path,
            map_location="cpu",
        )

        n = len(
            canonical
        )

        if (
            len(
                spatial
            )
            != n
            or len(
                temporal
            )
            != n
        ):

            raise AssertionError(
                "[FATAL] Length mismatch "
                f"video={vid}: "
                f"canonical={n}, "
                f"spatial={len(spatial)}, "
                f"temporal={len(temporal)}"
            )

        total += n

        for idx, (
            canonical_record,
            spatial_record,
            temporal_record,
        ) in enumerate(
            zip(
                canonical,
                spatial,
                temporal,
            )
        ):

            image_id = int(
                canonical_record[
                    "image_id"
                ]
            )

            seq_idx = int(
                canonical_record[
                    "seq_idx"
                ]
            )

            video_id = int(
                canonical_record[
                    "video_id"
                ]
            )

            if (
                video_id != vid
                or seq_idx != idx
            ):

                raise AssertionError(
                    "[FATAL] Canonical "
                    "identity drift: "
                    f"vid={vid}, idx={idx}"
                )

            for label, record in (
                (
                    "spatial",
                    spatial_record,
                ),
                (
                    "temporal",
                    temporal_record,
                ),
            ):

                if (
                    int(
                        record[
                            "image_id"
                        ]
                    )
                    != image_id
                    or int(
                        record[
                            "video_id"
                        ]
                    )
                    != vid
                    or int(
                        record[
                            "seq_idx"
                        ]
                    )
                    != idx
                ):

                    raise AssertionError(
                        "[FATAL] "
                        f"{label} alignment "
                        "mismatch: "
                        f"vid={vid}, "
                        f"idx={idx}"
                    )

            # -----------------------------------------------------
            # Spatial contract
            # -----------------------------------------------------

            if (
                tuple(
                    spatial_record[
                        "scale_spatial"
                    ].shape
                )
                != (
                    Q,
                    21,
                )
            ):

                raise AssertionError(
                    "[FATAL] scale_spatial "
                    "shape drift: "
                    f"vid={vid}, idx={idx}, "
                    "shape="
                    f"{tuple(spatial_record['scale_spatial'].shape)}"
                )

            if (
                tuple(
                    spatial_record[
                        "loc_spatial"
                    ].shape
                )
                != (
                    Q,
                    24,
                )
            ):

                raise AssertionError(
                    "[FATAL] loc_spatial "
                    "shape drift: "
                    f"vid={vid}, idx={idx}, "
                    "shape="
                    f"{tuple(spatial_record['loc_spatial'].shape)}"
                )

            assert_finite(
                (
                    "scale_spatial "
                    f"vid={vid} idx={idx}"
                ),
                spatial_record[
                    "scale_spatial"
                ],
            )

            assert_finite(
                (
                    "loc_spatial "
                    f"vid={vid} idx={idx}"
                ),
                spatial_record[
                    "loc_spatial"
                ],
            )

            # -----------------------------------------------------
            # Temporal contract
            # -----------------------------------------------------

            for (
                key,
                expected_shape,
            ) in TEMP_SHAPES.items():

                if (
                    key
                    not in temporal_record
                ):

                    raise AssertionError(
                        "[FATAL] Missing "
                        f"temporal key {key}: "
                        f"vid={vid}, idx={idx}"
                    )

                tensor = (
                    temporal_record[
                        key
                    ]
                )

                if (
                    tuple(
                        tensor.shape
                    )
                    != expected_shape
                ):

                    raise AssertionError(
                        "[FATAL] Temporal "
                        f"{key} shape drift: "
                        f"vid={vid}, idx={idx}, "
                        f"{tuple(tensor.shape)} "
                        "!= "
                        f"{expected_shape}"
                    )

                assert_finite(
                    (
                        f"temporal {key} "
                        f"vid={vid} idx={idx}"
                    ),
                    tensor,
                )

            history_query_index = (
                temporal_record[
                    "history_query_index"
                ]
                .long()
            )

            valid = (
                temporal_record[
                    "valid"
                ]
                .bool()
            )

            if (
                int(
                    history_query_index
                    .min()
                    .item()
                )
                < -1
                or int(
                    history_query_index
                    .max()
                    .item()
                )
                >= Q
            ):

                raise AssertionError(
                    "[FATAL] "
                    "history_query_index "
                    "out of range: "
                    f"vid={vid}, idx={idx}"
                )

            if torch.any(
                history_query_index[
                    ~valid
                ]
                != -1
            ):

                raise AssertionError(
                    "[FATAL] Invalid "
                    "temporal slots do not "
                    "carry query index -1: "
                    f"vid={vid}, idx={idx}"
                )

        print(
            "[+] Video "
            f"{vid:02d}/47 "
            "disk alignment PASS: "
            f"{n:4d} frames"
        )

        del (
            canonical,
            spatial,
            temporal,
        )

    if (
        total
        != TOTAL_FRAMES
    ):

        raise AssertionError(
            "[FATAL] Audited "
            f"frame total {total} "
            f"!= {TOTAL_FRAMES}"
        )

    print(
        "[+] Full-disk shard audit PASS: "
        f"{total:,} frames across "
        "47 videos"
    )


# =====================================================================
# PERSISTED CONSUMER AUDIT
# =====================================================================

def audit_consumer_samples():

    if (
        str(
            EVAL
        )
        in sys.path
    ):
        sys.path.remove(
            str(
                EVAL
            )
        )

    sys.path.insert(
        0,
        str(
            EVAL
        ),
    )

    from bfcef_f_s0_data import (
        assert_temporal_k_equivalence,
        frame_inputs_from_records,
        temporal_state_from_kmax_record,
    )

    from build_bfcef_f_temporal_state_cache import (
        build_frame_kstep_state,
    )

    sample_count = 0

    for vid in SAMPLE_VIDEOS:

        canonical_path = (
            E0
            / (
                "bfcef_e0_cache_"
                f"vid{vid}.pth"
            )
        )

        first_order_path = (
            E0
            / (
                "bfcef_e0_temporal_"
                f"cache_vid{vid}.pth"
            )
        )

        spatial_path = (
            SPATIAL_DIR
            / (
                "bfcef_f_spatial_"
                "evidence_"
                f"vid{vid}.pth"
            )
        )

        temporal_path = (
            TEMP_DIR
            / (
                "bfcef_f_temporal_"
                f"state_kmax{KMAX}_"
                f"vid{vid}.pth"
            )
        )

        canonical = torch.load(
            canonical_path,
            map_location="cpu",
        )

        first_order = torch.load(
            first_order_path,
            map_location="cpu",
        )

        spatial = torch.load(
            spatial_path,
            map_location="cpu",
        )

        temporal = torch.load(
            temporal_path,
            map_location="cpu",
        )

        indices = sorted(
            set(
                (
                    0,
                    len(
                        canonical
                    )
                    // 2,
                    len(
                        canonical
                    )
                    - 1,
                )
            )
        )

        for idx in indices:

            derived_temporal = (
                temporal[
                    idx
                ]
            )

            # -----------------------------------------------------
            # Persisted Kmax7 -> K3 consumer conversion.
            # -----------------------------------------------------

            sliced = (
                temporal_state_from_kmax_record(
                    derived_temporal,
                    K=K_S0,
                    device=torch.device(
                        "cpu"
                    ),
                )
            )

            # -----------------------------------------------------
            # Independent direct K3 reconstruction from the
            # canonical + frozen first-order temporal substrate.
            # -----------------------------------------------------

            direct_k3 = (
                build_frame_kstep_state(
                    canonical,
                    first_order,
                    idx,
                    K_S0,
                )
            )

            assert_temporal_k_equivalence(
                sliced,
                direct_k3,
                K=K_S0,
                atol=0.0,
            )

            # -----------------------------------------------------
            # Exercise the exact persisted spatial + temporal
            # S0 input-assembly function.
            # -----------------------------------------------------

            X = (
                frame_inputs_from_records(
                    canonical[
                        idx
                    ],

                    scale_spatial=(
                        spatial[
                            idx
                        ][
                            "scale_spatial"
                        ]
                    ),

                    loc_spatial=(
                        spatial[
                            idx
                        ][
                            "loc_spatial"
                        ]
                    ),

                    temporal_kmax_record=(
                        derived_temporal
                    ),

                    K=K_S0,

                    device=torch.device(
                        "cpu"
                    ),
                )
            )

            if (
                tuple(
                    X.scene_q.shape
                )
                != (
                    1,
                    5,
                )
            ):

                raise AssertionError(
                    "[FATAL] Scene-Q "
                    "consumer shape drift: "
                    f"vid={vid}, idx={idx}"
                )

            if (
                tuple(
                    X.temporal
                    .disp_history
                    .shape
                )
                != (
                    Q,
                    K_S0,
                    2,
                )
            ):

                raise AssertionError(
                    "[FATAL] Consumer "
                    "K=3 temporal shape drift: "
                    f"vid={vid}, idx={idx}"
                )

            sample_count += 1

        print(
            "[+] Consumer sample "
            f"video {vid:02d}: "
            f"{len(indices)} "
            "disk-loaded frames PASS"
        )

        del (
            canonical,
            first_order,
            spatial,
            temporal,
        )

    if (
        sample_count
        != 15
    ):

        raise AssertionError(
            "[FATAL] Expected "
            "15 consumer samples, "
            f"got {sample_count}"
        )

    print(
        "[+] Persisted "
        "producer -> disk -> consumer "
        "audit PASS: "
        "15 deterministic frames, "
        "exact Kmax7 -> K3 equivalence"
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    print(
        "=" * 116
    )

    print(
        "B-FCEF-F RUN_06 "
        "BATCH-G PERSISTED-CACHE "
        "INTEGRATION AUDIT"
    )

    print(
        "TRAIN-ONLY DERIVED CACHE AUDIT "
        "-- NO R0 INFERENCE "
        "-- NO S0 TRAINING"
    )

    print(
        "=" * 116
    )

    (
        spatial_manifest,
        temporal_manifest,
    ) = audit_manifests()

    audit_all_shards(
        spatial_manifest,
        temporal_manifest,
    )

    audit_consumer_samples()

    print(
        "=" * 116
    )

    print(
        "[PASS] RUN_06 BATCH-G "
        "PERSISTED-CACHE INTEGRATION "
        "AUDIT COMPLETE"
    )

    print(
        "47 spatial + 47 temporal "
        "derived shards certified "
        "across 20,075 Train frames."
    )

    print(
        "S0 has NOT been started."
    )

    print(
        "=" * 116
    )


if __name__ == "__main__":
    main()