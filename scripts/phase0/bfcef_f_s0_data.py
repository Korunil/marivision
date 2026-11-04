from __future__ import annotations

from typing import Any, Dict

import torch

from bfcef_f.contracts import (
    FrameExpertInputs,
    TemporalState,
)


def _to(
    tensor: torch.Tensor,
    device: torch.device,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    out = tensor.to(device)

    if dtype is not None:
        out = out.to(dtype=dtype)

    return out

SCENE_Q_FIELDS = (
    "detector_query_activity_density",
    "detector_activation_energy",
    "coarse_feature_energy",
    "coarse_feature_variance",
    "visibility_proxy",
)


def _scene_q_to_tensor(
    scene_q,
    device: torch.device,
) -> torch.Tensor:
    """
    Convert the certified canonical Scene-Q representation to [1,5].

    Historical/canonical cache representation:
        dict with five named scalar fields.

    Production B-FCEF-F representation:
        float32 tensor [1,5].

    Field order is frozen explicitly here. Never rely on dict insertion
    order for semantic feature layout.
    """

    if torch.is_tensor(
        scene_q
    ):
        result = (
            scene_q
            .to(
                device=device,
                dtype=torch.float32,
            )
        )

        if result.ndim == 1:
            result = (
                result
                .unsqueeze(0)
            )

    elif isinstance(
        scene_q,
        dict,
    ):
        actual_keys = set(
            scene_q.keys()
        )

        expected_keys = set(
            SCENE_Q_FIELDS
        )

        if (
            actual_keys
            != expected_keys
        ):
            missing = sorted(
                expected_keys
                - actual_keys
            )

            extra = sorted(
                actual_keys
                - expected_keys
            )

            raise AssertionError(
                "[FATAL] Scene-Q schema mismatch. "
                f"Missing={missing}, "
                f"Extra={extra}"
            )

        result = torch.tensor(
            [
                [
                    float(
                        scene_q[
                            field
                        ]
                    )
                    for field
                    in SCENE_Q_FIELDS
                ]
            ],
            device=device,
            dtype=torch.float32,
        )

    else:
        raise TypeError(
            "[FATAL] Unsupported Scene-Q "
            f"type: {type(scene_q).__name__}"
        )

    if (
        result.shape
        != (
            1,
            5,
        )
    ):
        raise AssertionError(
            "[FATAL] Scene-Q tensor "
            "must have shape [1,5], "
            f"got {tuple(result.shape)}"
        )

    if not torch.isfinite(
        result
    ).all():
        raise FloatingPointError(
            "[FATAL] Scene-Q contains "
            "non-finite values."
        )

    return result

def temporal_state_from_kmax_record(
    record: Dict[str, Any],
    *,
    K: int,
    device: torch.device,
) -> TemporalState:
    """
    Convert a materialized Kmax temporal record to configured K.

    IMPORTANT
    ---------
    This does NOT merely slice history tensors.

    K-dependent summaries are recomputed after slicing so a K=3
    consumer cannot receive information from history positions 4..Kmax.

    The recomputation matches the certified temporal builder:
      track_age
      velocity
      acceleration
      global_motion
    """

    if K < 1:
        raise ValueError(
            "K must be >= 1."
        )

    disp_all = record[
        "disp_history"
    ]

    if (
        disp_all.ndim != 3
        or disp_all.shape[-1] != 2
    ):
        raise ValueError(
            "disp_history must be "
            "[Q,Kmax,2], got "
            f"{tuple(disp_all.shape)}"
        )

    k_max = int(
        disp_all.shape[1]
    )

    if K > k_max:
        raise ValueError(
            f"Requested K={K} exceeds "
            f"materialized Kmax={k_max}."
        )

    disp = _to(
        record[
            "disp_history"
        ][:, :K],
        device,
        dtype=torch.float32,
    )

    score_persistence = _to(
        record[
            "score_persistence"
        ][:, :K],
        device,
        dtype=torch.float32,
    )

    class_consistency = _to(
        record[
            "class_consistency"
        ][:, :K],
        device,
        dtype=torch.float32,
    )

    appearance_similarity = _to(
        record[
            "appearance_similarity"
        ][:, :K],
        device,
        dtype=torch.float32,
    )

    association_confidence = _to(
        record[
            "association_confidence"
        ][:, :K],
        device,
        dtype=torch.float32,
    )

    same_slot = _to(
        record[
            "same_slot"
        ][:, :K],
        device,
        dtype=torch.float32,
    )

    valid_bool = _to(
        record[
            "valid"
        ][:, :K],
        device,
    ).bool()

    valid = (
        valid_bool.float()
    )

    q_count = int(
        disp.shape[0]
    )

    # ------------------------------------------------------------
    # Recompute K-dependent summaries.
    # ------------------------------------------------------------

    track_age = (
        valid
        .sum(
            dim=1,
            keepdim=True,
        )
        .float()
    )

    denom = (
        valid
        .sum(
            dim=1,
            keepdim=True,
        )
        .clamp_min(
            1.0
        )
    )

    velocity = (
        (
            disp
            * valid.unsqueeze(-1)
        )
        .sum(
            dim=1
        )
        / denom
    )

    acceleration = torch.zeros(
        q_count,
        2,
        device=device,
        dtype=torch.float32,
    )

    if K >= 2:
        accel_valid = (
            valid_bool[:, 0]
            & valid_bool[:, 1]
        )

        acceleration[
            accel_valid
        ] = (
            disp[
                accel_valid,
                0,
            ]
            - disp[
                accel_valid,
                1,
            ]
        )

    # Certified builder defines global motion from the most recent
    # reliable displacement only, so this is independent of older K.
    global_motion_vec = torch.zeros(
        2,
        device=device,
        dtype=torch.float32,
    )

    reliable = (
        valid_bool[:, 0]
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
            disp[
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

    legacy_key = (
        "legacy_cues_t"
        if "legacy_cues_t"
        in record
        else "legacy_cues"
    )

    return TemporalState(
        legacy_cues=_to(
            record[
                legacy_key
            ],
            device,
            dtype=torch.float32,
        ),

        disp_history=disp,

        score_persistence=(
            score_persistence
        ),

        class_consistency=(
            class_consistency
        ),

        appearance_similarity=(
            appearance_similarity
        ),

        association_confidence=(
            association_confidence
        ),

        same_slot=(
            same_slot
        ),

        valid=(
            valid
        ),

        track_age=(
            track_age
        ),

        velocity=(
            velocity
        ),

        acceleration=(
            acceleration
        ),

        global_motion=(
            global_motion
        ),
    )


def assert_temporal_k_equivalence(
    from_kmax: TemporalState,
    direct_k_record: Dict[
        str,
        Any,
    ],
    *,
    K: int,
    atol: float = 0.0,
) -> None:
    """
    Verify that:

        build Kmax=7
        -> consumer K=3

    is exactly equivalent to:

        certified builder run directly at K=3

    including stored K-dependent summary channels.
    """

    device = (
        from_kmax
        .disp_history
        .device
    )

    legacy_direct = (
        direct_k_record[
            "legacy_cues_t"
        ]
        if "legacy_cues_t"
        in direct_k_record
        else direct_k_record[
            "legacy_cues"
        ]
    )

    direct_map = {
        "legacy_cues":
            _to(
                legacy_direct,
                device,
                dtype=torch.float32,
            ),

        "disp_history":
            _to(
                direct_k_record[
                    "disp_history"
                ],
                device,
                dtype=torch.float32,
            ),

        "score_persistence":
            _to(
                direct_k_record[
                    "score_persistence"
                ],
                device,
                dtype=torch.float32,
            ),

        "class_consistency":
            _to(
                direct_k_record[
                    "class_consistency"
                ],
                device,
                dtype=torch.float32,
            ),

        "appearance_similarity":
            _to(
                direct_k_record[
                    "appearance_similarity"
                ],
                device,
                dtype=torch.float32,
            ),

        "association_confidence":
            _to(
                direct_k_record[
                    "association_confidence"
                ],
                device,
                dtype=torch.float32,
            ),

        "same_slot":
            _to(
                direct_k_record[
                    "same_slot"
                ],
                device,
                dtype=torch.float32,
            ),

        "valid":
            _to(
                direct_k_record[
                    "valid"
                ],
                device,
            ).float(),

        "track_age":
            _to(
                direct_k_record[
                    "track_age"
                ],
                device,
                dtype=torch.float32,
            ),

        "velocity":
            _to(
                direct_k_record[
                    "velocity"
                ],
                device,
                dtype=torch.float32,
            ),

        "acceleration":
            _to(
                direct_k_record[
                    "acceleration"
                ],
                device,
                dtype=torch.float32,
            ),

        "global_motion":
            _to(
                direct_k_record[
                    "global_motion"
                ],
                device,
                dtype=torch.float32,
            ),
    }

    if (
        direct_map[
            "disp_history"
        ].shape[1]
        != K
    ):
        raise AssertionError(
            "[FATAL] Direct-K record "
            "history depth "
            f"{direct_map['disp_history'].shape[1]} "
            f"!= K={K}"
        )

    for (
        name,
        expected,
    ) in direct_map.items():

        observed = getattr(
            from_kmax,
            name,
        )

        if (
            observed.shape
            != expected.shape
        ):
            raise AssertionError(
                f"[FATAL] K-slice "
                f"{name} shape mismatch: "
                f"{tuple(observed.shape)} "
                "!= "
                f"{tuple(expected.shape)}"
            )

        delta = float(
            (
                observed.float()
                - expected.float()
            )
            .abs()
            .max()
            .item()
        )

        if delta > atol:
            raise AssertionError(
                f"[FATAL] K-slice "
                f"{name} drift: "
                f"{delta:.3e} "
                f"> {atol:.1e}"
            )


def frame_inputs_from_records(
    canonical: Dict[
        str,
        Any,
    ],
    *,
    scale_spatial: torch.Tensor,
    loc_spatial: torch.Tensor,
    temporal_kmax_record: Dict[
        str,
        Any,
    ],
    K: int,
    device: torch.device,
) -> FrameExpertInputs:
    """
    Authoritative S0 frame-consumer assembly.

    The same function can consume:
      - live spatial descriptors during this smoke test;
      - persisted Batch-G spatial descriptors after cache generation.
    """

    temporal = (
        temporal_state_from_kmax_record(
            temporal_kmax_record,
            K=K,
            device=device,
        )
    )

    q_count = int(
        canonical[
            "d_boxes"
        ].shape[0]
    )

    raw_proxies = {
        "S": {
            "N": _to(
                canonical[
                    "n_s_proxy"
                ],
                device,
                dtype=torch.float32,
            ).reshape(
                q_count,
                1,
            ),
            "R": _to(
                canonical[
                    "r_s_proxy"
                ],
                device,
                dtype=torch.float32,
            ).reshape(
                q_count,
                1,
            ),
        },

        "L": {
            "N": _to(
                canonical[
                    "n_l_proxy"
                ],
                device,
                dtype=torch.float32,
            ).reshape(
                q_count,
                1,
            ),
            "R": _to(
                canonical[
                    "r_l_proxy"
                ],
                device,
                dtype=torch.float32,
            ).reshape(
                q_count,
                1,
            ),
        },

        "C": {
            "N": _to(
                canonical[
                    "n_c_proxy"
                ],
                device,
                dtype=torch.float32,
            ).reshape(
                q_count,
                1,
            ),
            "R": _to(
                canonical[
                    "r_c_proxy"
                ],
                device,
                dtype=torch.float32,
            ).reshape(
                q_count,
                1,
            ),
        },

        "T": {
            "N": _to(
                temporal_kmax_record[
                    "n_t_proxy"
                ],
                device,
                dtype=torch.float32,
            ).reshape(
                q_count,
                1,
            ),
            "R": _to(
                temporal_kmax_record[
                    "r_t_proxy"
                ],
                device,
                dtype=torch.float32,
            ).reshape(
                q_count,
                1,
            ),
        },
    }

    X = FrameExpertInputs(
        cues_s=_to(
            canonical[
                "cues_s"
            ],
            device,
            dtype=torch.float32,
        ),

        cues_l=_to(
            canonical[
                "cues_l"
            ],
            device,
            dtype=torch.float32,
        ),

        cues_c=_to(
            canonical[
                "cues_c"
            ],
            device,
            dtype=torch.float32,
        ),

        scale_spatial=(
            scale_spatial
            .to(
                device=device,
                dtype=torch.float32,
            )
        ),

        loc_spatial=(
            loc_spatial
            .to(
                device=device,
                dtype=torch.float32,
            )
        ),

        temporal=(
            temporal
        ),

        raw_proxies=(
            raw_proxies
        ),

        scene_q=(
            _scene_q_to_tensor(
                canonical[
                    "scene_q"
                ],
                device,
            )
        ),
    )

    if (
        X.scale_spatial.shape
        != (
            q_count,
            21,
        )
    ):
        raise AssertionError(
            "[FATAL] scale_spatial "
            f"shape {tuple(X.scale_spatial.shape)} "
            f"!= {(q_count, 21)}"
        )

    if (
        X.loc_spatial.shape
        != (
            q_count,
            24,
        )
    ):
        raise AssertionError(
            "[FATAL] loc_spatial "
            f"shape {tuple(X.loc_spatial.shape)} "
            f"!= {(q_count, 24)}"
        )

    if (
        X.temporal
        .disp_history
        .shape
        != (
            q_count,
            K,
            2,
        )
    ):
        raise AssertionError(
            "[FATAL] Temporal K slice "
            "did not reach configured shape."
        )

    return X