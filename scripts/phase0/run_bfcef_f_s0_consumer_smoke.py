from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

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

ADAPTERS = (
    EVAL
    / "e0_adapters"
)

TRAIN_JSON = (
    ROOT
    / "seaclips-train.json"
)

ZIP_PATH = (
    ROOT
    / "SeaClips.zip"
)

CKPT = (
    ROOT
    / "checkpoints_r0"
    / "checkpoint_best.pth"
)

CAN_MANIFEST = (
    E0
    / "bfcef_e0_cache_manifest.json"
)

TEMP_MANIFEST = (
    E0
    / "bfcef_e0_temporal_cache_manifest.json"
)

FOLD_MANIFEST = (
    ADAPTERS
    / "bfcef_e0_5fold_manifest.json"
)

CONFIG = (
    EVAL
    / "screening_config_s0.json"
)


R0_SHA = (
    "28e6288c3f6769c2d5275aa77ddb2487"
    "35b55a09cd46a6b570e3f117890b268b"
)

CAN_SHA = (
    "210ad12de4ebe02cdbe000956972955ae"
    "58ceb8f0d222b703787f33a49af03cd"
)

TEMP_SHA = (
    "e4754d9c82c60836d817e49ceb552018"
    "07c22405677558bb2a69b3e96a6bd286"
)

FOLD_SHA = (
    "c923f8abd5833996331566582f568b2b"
    "0e9ad1d7d17f606c8ddae693b6b8a103"
)


NUM_VIDEOS = 47
NUM_QUERIES = 300
NUM_CLASSES = 7

EXPECTED_FRAMES = (
    20_075
)

KMAX = 7

DEVICE = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)


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
            f"[FATAL] Missing "
            f"{label}: {path}"
        )

    actual = sha256(
        path
    )

    if actual != expected:
        raise AssertionError(
            f"[FATAL] {label} "
            "SHA drift:\n"
            f"expected={expected}\n"
            f"actual  ={actual}"
        )


def find_deformable_root() -> Path:

    for p in (
        ROOT
        / "Deformable-DETR",

        ROOT
        / "deformable_detr",
    ):
        if (
            (
                p
                / "models"
                / "deformable_detr.py"
            ).exists()
            and (
                p
                / "datasets"
                / "seaclips_detection.py"
            ).exists()
        ):
            return p

    raise AssertionError(
        "[FATAL] Cannot locate "
        "authoritative Deformable-DETR "
        "repository."
    )


def dataset_index(
    dataset,
):

    if hasattr(
        dataset,
        "ids",
    ):
        ids = list(
            dataset.ids
        )
        source = (
            "dataset.ids"
        )

    elif hasattr(
        dataset,
        "img_ids",
    ):
        ids = list(
            dataset.img_ids
        )
        source = (
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
        and "images"
        in dataset.coco.dataset
    ):
        ids = [
            int(
                image[
                    "id"
                ]
            )
            for image
            in dataset.coco.dataset[
                "images"
            ]
        ]

        source = (
            'dataset.coco.dataset["images"]'
        )

    else:
        raise AssertionError(
            "[FATAL] No certified "
            "SeaClips image-ID interface."
        )

    mapping = {
        int(value): index
        for index, value
        in enumerate(ids)
    }

    if (
        len(mapping)
        != len(ids)
    ):
        raise AssertionError(
            "[FATAL] Duplicate "
            "dataset image IDs."
        )

    print(
        "[+] Dataset image-ID "
        f"mapping: {source} "
        f"({len(mapping):,} IDs)"
    )

    return mapping


def load_folds():

    require_sha(
        FOLD_MANIFEST,
        FOLD_SHA,
        "5-fold manifest",
    )

    payload = json.loads(
        FOLD_MANIFEST
        .read_text(
            encoding="utf-8"
        )
    )

    folds = {
        int(k): [
            int(v)
            for v in values
        ]
        for k, values
        in payload[
            "folds"
        ].items()
    }

    capacities = [
        len(
            folds[i]
        )
        for i
        in range(5)
    ]

    if capacities != [
        10,
        10,
        9,
        9,
        9,
    ]:
        raise AssertionError(
            "[FATAL] Fold capacities "
            f"drift: {capacities}"
        )

    union = sorted(
        video
        for values
        in folds.values()
        for video
        in values
    )

    if union != list(
        range(
            1,
            48,
        )
    ):
        raise AssertionError(
            "[FATAL] Fold partition "
            "is not exact disjoint "
            "union 1..47."
        )

    oof = sorted(
        folds[0]
    )

    train = sorted(
        set(
            range(
                1,
                48,
            )
        )
        - set(oof)
    )

    if (
        set(train)
        & set(oof)
    ):
        raise AssertionError(
            "[FATAL] Fold leakage."
        )

    print(
        "[+] Fold 0 isolation PASS: "
        f"{len(train)} train / "
        f"{len(oof)} OOF videos"
    )

    print(
        f"    OOF videos: {oof}"
    )

    return (
        train,
        oof,
    )


def weighted_detection_loss(
    criterion,
    logits: torch.Tensor,
    boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
) -> torch.Tensor:

    loss_dict = criterion(
        {
            "pred_logits":
                logits.unsqueeze(0),

            "pred_boxes":
                boxes.unsqueeze(0),
        },
        [
            {
                "boxes":
                    gt_boxes,

                "labels":
                    gt_labels,
            }
        ],
    )

    terms = [
        value
        * criterion.weight_dict[
            key
        ]
        for key, value
        in loss_dict.items()
        if key
        in criterion.weight_dict
    ]

    if not terms:
        raise AssertionError(
            "[FATAL] Official criterion "
            "returned no weighted terms."
        )

    return sum(
        terms
    )


def snapshot(
    module,
):
    return {
        name:
            parameter
            .detach()
            .cpu()
            .clone()

        for name, parameter
        in module.named_parameters()
    }


def max_delta(
    before,
    module,
) -> float:

    maximum = 0.0

    for (
        name,
        parameter,
    ) in module.named_parameters():

        delta = float(
            (
                parameter
                .detach()
                .cpu()
                - before[
                    name
                ]
            )
            .abs()
            .max()
            .item()
        )

        maximum = max(
            maximum,
            delta,
        )

    return maximum


def main():

    print(
        "=" * 112
    )

    print(
        "B-FCEF-F S0 CONSUMER / "
        "TRAINABILITY SMOKE -- "
        "REAL TRAIN FRAME ONLY"
    )

    print(
        "NO FULL CACHE GENERATION. "
        "NO FIVE-FOLD S0 TRAINING. "
        "NO VAL/TEST ACCESS."
    )

    print(
        "=" * 112
    )

    print(
        f"Device: {DEVICE}"
    )

    # ============================================================
    # 1. Frozen provenance + fold isolation
    # ============================================================

    require_sha(
        CKPT,
        R0_SHA,
        "R0 checkpoint",
    )

    require_sha(
        CAN_MANIFEST,
        CAN_SHA,
        "canonical cache manifest",
    )

    require_sha(
        TEMP_MANIFEST,
        TEMP_SHA,
        "temporal association manifest",
    )

    can_manifest = json.loads(
        CAN_MANIFEST
        .read_text(
            encoding="utf-8"
        )
    )

    temp_manifest = json.loads(
        TEMP_MANIFEST
        .read_text(
            encoding="utf-8"
        )
    )

    if (
        can_manifest[
            "_meta"
        ][
            "actual_total_frames"
        ]
        != EXPECTED_FRAMES
    ):
        raise AssertionError(
            "[FATAL] Canonical "
            "frame count drift."
        )

    if (
        temp_manifest[
            "_meta"
        ][
            "actual_total_frames"
        ]
        != EXPECTED_FRAMES
    ):
        raise AssertionError(
            "[FATAL] Temporal "
            "frame count drift."
        )

    (
        train_vids,
        oof_vids,
    ) = load_folds()

    smoke_vid = (
        train_vids[0]
    )

    if (
        smoke_vid
        in oof_vids
    ):
        raise AssertionError(
            "[FATAL] Smoke video "
            "is OOF."
        )

    can_path = (
        E0
        / (
            "bfcef_e0_cache_"
            f"vid{smoke_vid}.pth"
        )
    )

    temp_path = (
        E0
        / (
            "bfcef_e0_temporal_"
            f"cache_vid{smoke_vid}.pth"
        )
    )

    if (
        sha256(
            can_path
        )
        != can_manifest[
            "cache_shard_sha256_ledger"
        ][
            can_path.name
        ]
    ):
        raise AssertionError(
            "[FATAL] Selected "
            "canonical shard SHA drift."
        )

    if (
        sha256(
            temp_path
        )
        != temp_manifest[
            "temporal_cache_shard_sha256_ledger"
        ][
            temp_path.name
        ]
    ):
        raise AssertionError(
            "[FATAL] Selected "
            "temporal shard SHA drift."
        )

    canonical_records = torch.load(
        can_path,
        map_location="cpu",
    )

    temporal_records = torch.load(
        temp_path,
        map_location="cpu",
    )

    if (
        len(
            canonical_records
        )
        != len(
            temporal_records
        )
    ):
        raise AssertionError(
            "[FATAL] Canonical/"
            "temporal length mismatch."
        )

    frame_idx = (
        len(
            canonical_records
        )
        // 2
    )

    canonical = (
        canonical_records[
            frame_idx
        ]
    )

    if (
        int(
            canonical[
                "video_id"
            ]
        )
        != smoke_vid
        or int(
            canonical[
                "seq_idx"
            ]
        )
        != frame_idx
    ):
        raise AssertionError(
            "[FATAL] Smoke frame "
            "alignment mismatch."
        )

    print(
        "[+] Selected fold-0 "
        "TRAIN frame: "
        f"video={smoke_vid}, "
        f"seq_idx={frame_idx}, "
        "image_id="
        f"{int(canonical['image_id'])}"
    )

    # ============================================================
    # 2. Load authoritative production package.
    # ============================================================

    deformable_root = (
        find_deformable_root()
    )

    for path in (
        str(ROOT),
        str(deformable_root),
        str(EVAL),
    ):
        if path in sys.path:
            sys.path.remove(
                path
            )

        sys.path.insert(
            0,
            path,
        )

    from bfcef_f.config import (
        BFCEFFScreeningConfig,
    )

    from bfcef_f.contracts import (
        CanonicalObjectState,
    )

    from bfcef_f.model import (
        BFCEFFModel,
    )

    from bfcef_f.spatial_evidence import (
        extract_localization_spatial_evidence,
        extract_scale_spatial_evidence,
    )

    from bfcef_f_s0_data import (
        assert_temporal_k_equivalence,
        frame_inputs_from_records,
        temporal_state_from_kmax_record,
    )

    from build_bfcef_f_temporal_state_cache import (
        build_frame_kstep_state,
    )

    from r0_preflight import (
        get_authoritative_r0_args,
    )

    from models.deformable_detr import (
        build as build_deformable_detr,
    )

    from datasets.seaclips_detection import (
        SeaClipsDetection,
    )

    from util.misc import (
        collate_fn,
    )

    config = (
        BFCEFFScreeningConfig
        .from_json(
            CONFIG
        )
    )

    K = int(
        config.temporal_memory_K
    )

    if (
        K != 3
        or K > KMAX
    ):
        raise AssertionError(
            "[FATAL] Frozen S0 smoke "
            "expects K=3 <= Kmax=7, "
            f"got K={K}"
        )

    # ============================================================
    # 3. Kmax=7 -> K=3 exact consumer certification.
    # ============================================================

    state7 = (
        build_frame_kstep_state(
            canonical_records,
            temporal_records,
            frame_idx,
            KMAX,
        )
    )

    state3_direct = (
        build_frame_kstep_state(
            canonical_records,
            temporal_records,
            frame_idx,
            K,
        )
    )

    temporal_k3 = (
        temporal_state_from_kmax_record(
            state7,
            K=K,
            device=DEVICE,
        )
    )

    assert_temporal_k_equivalence(
        temporal_k3,
        state3_direct,
        K=K,
        atol=0.0,
    )

    print(
        "[+] Kmax=7 -> K=3 "
        "consumer semantics PASS, "
        "including recomputed summaries."
    )

    # ============================================================
    # 4. Build frozen real R0.
    # ============================================================

    args = (
        get_authoritative_r0_args()
    )

    args.device = str(
        DEVICE
    )

    if (
        int(
            args.num_queries
        )
        != NUM_QUERIES
        or bool(
            args.two_stage
        )
        or bool(
            args.with_box_refine
        )
    ):
        raise AssertionError(
            "[FATAL] Authoritative "
            "R0 args drift."
        )

    (
        r0_model,
        criterion,
        _,
    ) = build_deformable_detr(
        args
    )

    checkpoint = torch.load(
        CKPT,
        map_location="cpu",
    )

    state_dict = checkpoint.get(
        "model",
        checkpoint,
    )

    state_dict = {
        (
            key[7:]
            if key.startswith(
                "module."
            )
            else key
        ):
            value

        for key, value
        in state_dict.items()
    }

    r0_model.load_state_dict(
        state_dict,
        strict=True,
    )

    r0_model.to(
        DEVICE
    )

    r0_model.eval()

    criterion.to(
        DEVICE
    )

    for parameter in (
        r0_model.parameters()
    ):
        parameter.requires_grad_(
            False
        )

    # ============================================================
    # 5. One real Train-frame live feature tap.
    # ============================================================

    taps = {}

    def transformer_hook(
        module,
        inp,
        outp,
    ):
        srcs = inp[0]

        taps[
            "P3"
        ] = srcs[
            0
        ].detach()

        taps[
            "P4"
        ] = srcs[
            1
        ].detach()

        taps[
            "P5"
        ] = srcs[
            2
        ].detach()

        hs = outp[0]
        init_ref = outp[1]
        inter_refs = outp[2]

        last = (
            hs.shape[0]
            - 1
        )

        taps[
            "q"
        ] = (
            hs[
                last
            ]
            .detach()
        )

        if last == 0:
            taps[
                "r"
            ] = (
                init_ref
                .detach()
            )
        else:
            taps[
                "r"
            ] = (
                inter_refs[
                    last - 1
                ]
                .detach()
            )

    handle = (
        r0_model
        .transformer
        .register_forward_hook(
            transformer_hook
        )
    )

    dataset = SeaClipsDetection(
        ZIP_PATH,
        TRAIN_JSON,
        is_train=False,
        random_flip=False,
    )

    iid_to_idx = (
        dataset_index(
            dataset
        )
    )

    image_id = int(
        canonical[
            "image_id"
        ]
    )

    if (
        image_id
        not in iid_to_idx
    ):
        raise AssertionError(
            "[FATAL] Smoke image "
            "not in Train dataset."
        )

    item = dataset[
        iid_to_idx[
            image_id
        ]
    ]

    samples, _ = collate_fn(
        [
            item
        ]
    )

    samples = samples.to(
        DEVICE
    )

    with torch.no_grad():
        live = r0_model(
            samples
        )

    handle.remove()

    z_live = (
        live[
            "pred_logits"
        ][0]
        .detach()
    )

    b_live = (
        live[
            "pred_boxes"
        ][0]
        .detach()
    )

    q_live = (
        taps[
            "q"
        ][0]
        .detach()
    )

    r_live = (
        taps[
            "r"
        ][0]
        .detach()
    )

    z = (
        canonical[
            "d_logits"
        ]
        .to(
            DEVICE
        )
        .float()
    )

    b = (
        canonical[
            "d_boxes"
        ]
        .to(
            DEVICE
        )
        .float()
    )

    q = (
        canonical[
            "q_native"
        ]
        .to(
            DEVICE
        )
        .float()
    )

    r = (
        canonical[
            "ref_pts"
        ]
        .to(
            DEVICE
        )
        .float()
    )

    def drift(
        observed,
        expected,
    ):
        return float(
            (
                observed.float()
                - expected.float()
            )
            .abs()
            .max()
            .item()
        )

    dz = drift(
        z_live,
        z,
    )

    db = drift(
        b_live,
        b,
    )

    dq = drift(
        q_live,
        q,
    )

    dr = drift(
        r_live,
        r,
    )

    if max(
        dz,
        db,
        dq,
        dr,
    ) >= 1e-4:
        raise AssertionError(
            "[FATAL] Live/cache "
            "R0 drift: "
            f"dz={dz}, "
            f"db={db}, "
            f"dq={dq}, "
            f"dr={dr}"
        )

    print(
        "[+] Real R0 parity PASS: "
        f"dz={dz:.3e}, "
        f"db={db:.3e}, "
        f"dq={dq:.3e}, "
        f"dr={dr:.3e}"
    )

    feature_maps = {
        "P3":
            taps[
                "P3"
            ],

        "P4":
            taps[
                "P4"
            ],

        "P5":
            taps[
                "P5"
            ],
    }

    with torch.no_grad():

        scale_spatial = (
            extract_scale_spatial_evidence(
                b,
                feature_maps,
                ring_expansion=(
                    config
                    .scale_ring_expansion
                ),
            )
        )

        loc_spatial = (
            extract_localization_spatial_evidence(
                b,
                feature_maps,
                side_samples=(
                    config
                    .localization_side_samples
                ),
                offset_frac=(
                    config
                    .localization_boundary_offset
                ),
            )
        )

    X = (
        frame_inputs_from_records(
            canonical,

            scale_spatial=(
                scale_spatial
            ),

            loc_spatial=(
                loc_spatial
            ),

            temporal_kmax_record=(
                state7
            ),

            K=K,

            device=DEVICE,
        )
    )

    capabilities = {
        "base": True
    }

    for name in (
        "scale",
        "localization",
        "semantic",
        "temporal",
    ):
        capabilities[
            name
        ] = True

        capabilities[
            f"{name}.cls"
        ] = True

        capabilities[
            f"{name}.box"
        ] = True

    O = CanonicalObjectState(
        h=q,
        z=z,
        b=b,
        r=r,

        s=(
            z.sigmoid()
            .max(
                dim=-1
            )
            .values
        ),

        F_maps=(
            feature_maps
        ),

        native_handle={
            "source":
                "S0_smoke",

            "image_id":
                image_id,
        },

        capabilities=(
            capabilities
        ),
    )

    # ============================================================
    # 6. Build B-FCEF-F using certified package.
    # ============================================================

    model = BFCEFFModel(
        native_class_embed=(
            r0_model
            .class_embed[
                -1
            ]
        ),

        native_bbox_embed=(
            r0_model
            .bbox_embed[
                -1
            ]
        ),

        candidate_topk=100,

        config=(
            config
        ),

        num_classes=(
            NUM_CLASSES
        ),
    ).to(
        DEVICE
    )

    if any(
        parameter.requires_grad
        for parameter
        in model.adapter.parameters()
    ):
        raise AssertionError(
            "[FATAL] Native "
            "adapter heads are "
            "not frozen."
        )

    native_cls_before = (
        snapshot(
            model
            .adapter
            .class_embed
        )
    )

    native_box_before = (
        snapshot(
            model
            .adapter
            .bbox_embed
        )
    )

    trainable = [
        parameter
        for parameter
        in model.parameters()
        if parameter.requires_grad
    ]

    # Smoke-only optimizer:
    # no weight decay so zero-gradient Gate/Router parameters cannot
    # move merely from decoupled AdamW weight decay.
    optimizer = torch.optim.AdamW(
        trainable,
        lr=1e-4,
        weight_decay=0.0,
    )

    # ============================================================
    # 7. One REAL forward / backward / update.
    # ============================================================

    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    (
        z_star,
        b_star,
        info,
    ) = model(
        O,
        X,
        context_mode="unknown",
    )

    if info.get(
        "bypassed",
        True,
    ):
        raise AssertionError(
            "[FATAL] Training "
            "incorrectly took "
            "hard bypass."
        )

    for key in (
        "expert_residual_cls",
        "expert_residual_box",
        "cls_posteriors",
        "box_raw",
    ):
        if key not in info:
            raise AssertionError(
                "[FATAL] Missing "
                "training output: "
                f"{key}"
            )

    for expert in (
        "S",
        "L",
        "C",
        "T",
    ):
        if (
            info[
                "cls_posteriors"
            ][
                expert
            ].shape
            != (
                NUM_QUERIES,
                NUM_CLASSES,
                9,
            )
        ):
            raise AssertionError(
                "[FATAL] "
                f"{expert} "
                "cls posterior "
                "shape drift."
            )

        if (
            info[
                "box_raw"
            ][
                expert
            ].shape
            != (
                NUM_QUERIES,
                5,
            )
        ):
            raise AssertionError(
                "[FATAL] "
                f"{expert} "
                "box raw "
                "shape drift."
            )

    gt_boxes = (
        canonical[
            "gt_boxes_norm"
        ]
        .to(
            DEVICE
        )
        .float()
    )

    gt_labels = (
        canonical[
            "gt_labels"
        ]
        .to(
            DEVICE
        )
        .long()
    )

    main_loss = (
        weighted_detection_loss(
            criterion,
            z_star,
            b_star,
            gt_boxes,
            gt_labels,
        )
    )

    # ------------------------------------------------------------
    # Deep expert supervision smoke.
    #
    # This is deliberately task-decoupled and bypasses the Gate/Router.
    # It proves all eight S/L/C/T cls/box residual projectors can learn
    # from the real cached/live data path.
    #
    # The coefficient below is smoke-only. It is NOT yet the frozen
    # scientific S0 loss weighting.
    # ------------------------------------------------------------

    deep_losses = []

    residual_cls = (
        info[
            "expert_residual_cls"
        ]
    )

    residual_box = (
        info[
            "expert_residual_box"
        ]
    )

    for expert_index in range(
        4
    ):

        z_expert = (
            model
            .adapter
            .decode_cls(
                O.h
                + residual_cls[
                    :,
                    expert_index,
                ]
            )
        )

        b_expert = (
            model
            .adapter
            .decode_box(
                O.h
                + residual_box[
                    :,
                    expert_index,
                ],
                O.r,
            )
        )

        expert_loss = (
            weighted_detection_loss(
                criterion,
                z_expert,
                b_expert,
                gt_boxes,
                gt_labels,
            )
        )

        deep_losses.append(
            expert_loss
        )

    deep_loss = (
        torch.stack(
            deep_losses
        )
        .mean()
    )

    smoke_loss = (
        main_loss
        + deep_loss
    )

    if not torch.isfinite(
        smoke_loss
    ):
        raise AssertionError(
            "[FATAL] Non-finite "
            "smoke loss."
        )

    smoke_loss.backward()

    expert_modules = {
        "S":
            model.expert_s,

        "L":
            model.expert_l,

        "C":
            model.expert_c,

        "T":
            model.expert_t,
    }

    projector_before = {
        expert: {
            "cls":
                snapshot(
                    module.proj_cls
                ),

            "box":
                snapshot(
                    module.proj_box
                ),
        }

        for expert, module
        in expert_modules.items()
    }

    gradient_report = {}

    for (
        expert,
        module,
    ) in expert_modules.items():

        cls_grad = (
            0.0
            if (
                module
                .proj_cls
                .weight
                .grad
                is None
            )
            else float(
                module
                .proj_cls
                .weight
                .grad
                .norm()
                .item()
            )
        )

        box_grad = (
            0.0
            if (
                module
                .proj_box
                .weight
                .grad
                is None
            )
            else float(
                module
                .proj_box
                .weight
                .grad
                .norm()
                .item()
            )
        )

        if (
            cls_grad <= 0.0
            or box_grad <= 0.0
        ):
            raise AssertionError(
                "[FATAL] "
                f"{expert} projector "
                "gradient missing: "
                f"cls={cls_grad:.3e}, "
                f"box={box_grad:.3e}"
            )

        gradient_report[
            expert
        ] = (
            cls_grad,
            box_grad,
        )

    torch.nn.utils.clip_grad_norm_(
        trainable,
        max_norm=1.0,
    )

    optimizer.step()

    for (
        expert,
        module,
    ) in expert_modules.items():

        cls_delta = (
            max_delta(
                projector_before[
                    expert
                ][
                    "cls"
                ],
                module.proj_cls,
            )
        )

        box_delta = (
            max_delta(
                projector_before[
                    expert
                ][
                    "box"
                ],
                module.proj_box,
            )
        )

        if (
            cls_delta <= 0.0
            or box_delta <= 0.0
        ):
            raise AssertionError(
                "[FATAL] "
                f"{expert} "
                "projector did "
                "not update."
            )

    native_cls_delta = (
        max_delta(
            native_cls_before,
            model
            .adapter
            .class_embed,
        )
    )

    native_box_delta = (
        max_delta(
            native_box_before,
            model
            .adapter
            .bbox_embed,
        )
    )

    if (
        native_cls_delta
        != 0.0
        or native_box_delta
        != 0.0
    ):
        raise AssertionError(
            "[FATAL] Frozen "
            "native heads changed."
        )

    print(
        "[+] One-step REAL "
        "train smoke PASS: "
        f"main={float(main_loss.item()):.6f}, "
        f"deep={float(deep_loss.item()):.6f}"
    )

    for (
        expert,
        (
            cls_grad,
            box_grad,
        ),
    ) in gradient_report.items():

        print(
            f"    {expert}: "
            "grad(proj_cls)="
            f"{cls_grad:.3e}, "
            "grad(proj_box)="
            f"{box_grad:.3e}"
        )

    print(
        "[+] Frozen native heads PASS: "
        f"cls_delta={native_cls_delta:.3e}, "
        f"box_delta={native_box_delta:.3e}"
    )

    # ============================================================
    # 8. Post-update exact hard-eval fallback.
    # ============================================================

    model.eval()

    with torch.no_grad():

        (
            z_eval,
            b_eval,
            eval_info,
        ) = model(
            O,
            X,
            context_mode="unknown",
        )

    if not eval_info.get(
        "bypassed",
        False,
    ):
        raise AssertionError(
            "[FATAL] Neutral "
            "hard-gate eval "
            "did not bypass."
        )

    if not torch.equal(
        z_eval,
        O.z,
    ):
        raise AssertionError(
            "[FATAL] Hard-eval "
            "baseline logit "
            "fallback is not "
            "bitwise exact."
        )

    if not torch.equal(
        b_eval,
        O.b,
    ):
        raise AssertionError(
            "[FATAL] Hard-eval "
            "baseline box "
            "fallback is not "
            "bitwise exact."
        )

    print(
        "[+] Exact post-update "
        "hard-eval fallback PASS."
    )

    print(
        "=" * 112
    )

    print(
        "[PASS] S0 CONSUMER / "
        "TRAINABILITY SMOKE COMPLETE"
    )

    print(
        "No derived cache written. "
        "No S0 checkpoint written. "
        "Val/Test not accessed."
    )

    print(
        "=" * 112
    )


if __name__ == "__main__":
    main()