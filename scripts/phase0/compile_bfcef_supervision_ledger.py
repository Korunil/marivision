from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Sequence,
    Set,
    Tuple,
)

import numpy as np
import torch


# ============================================================================
# THREAD / CPU CONTROL
# ============================================================================

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


# ============================================================================
# PROJECT PATHS
# ============================================================================

ROOT_DIR = Path("D:/MARIVISION")

EVAL_DIR = (
    ROOT_DIR
    / "H0_workbench"
    / "eval_tools"
)

DEFAULT_MASTER = (
    EVAL_DIR
    / "e0_adapters"
    / "bfcef_e0_oof_ledger_index.json"
)

DEFAULT_D_REPORT = (
    EVAL_DIR
    / "d_audit"
    / "bfcef_d0_d5_environmental_evidence_report.json"
)

DEFAULT_BASELINE_PROFILE = (
    Path(__file__).resolve().parent
    / "baseline_profile_r0.json"
)

DEFAULT_OUT = (
    EVAL_DIR
    / "g1_router"
    / "supervision_ledger_chatgpt"
)


# ============================================================================
# CRYPTOGRAPHIC ROOTS
# ============================================================================

FROZEN_R0_CHECKPOINT_SHA256 = (
    "28e6288c3f6769c2d5275aa77ddb248735b55a09cd46a6b570e3f117890b268b"
)

FROZEN_E0_MASTER_SHA256 = (
    "722b5febc9e849af1120fadcacfeffcef5230025ef6dfd9221dfbdc30094c3cf"
)

FROZEN_D_REPORT_SHA256 = (
    "67d47fbd5534acfebcc542b7b064084db210f52877171bc1562b9057b2274b5b"
)


EXPERTS = (
    "S",
    "L",
    "C",
    "T",
)


# ============================================================================
# FROZEN D1 TARGETS
# ============================================================================

FROZEN_D1_TARGETS: Dict[
    str,
    Dict[str, Dict[str, int]],
] = {
    "S": {
        "cls_only": {
            "tp": 50314,
            "rescued": 25,
            "lost": 452,
            "fp_added": 106,
            "fp_removed": 10675,
        },
        "box_only": {
            "tp": 49413,
            "rescued": 1196,
            "lost": 2524,
            "fp_added": 3090,
            "fp_removed": 1762,
        },
        "full": {
            "tp": 48931,
            "rescued": 1133,
            "lost": 2943,
            "fp_added": 3059,
            "fp_removed": 12245,
        },
    },

    "L": {
        "cls_only": {
            "tp": 50636,
            "rescued": 101,
            "lost": 206,
            "fp_added": 334,
            "fp_removed": 8688,
        },
        "box_only": {
            "tp": 51313,
            "rescued": 2019,
            "lost": 1447,
            "fp_added": 1982,
            "fp_removed": 2554,
        },
        "full": {
            "tp": 51201,
            "rescued": 2071,
            "lost": 1611,
            "fp_added": 2214,
            "fp_removed": 11133,
        },
    },

    "C": {
        "cls_only": {
            "tp": 50388,
            "rescued": 100,
            "lost": 453,
            "fp_added": 89,
            "fp_removed": 13913,
        },
        "box_only": {
            "tp": 48127,
            "rescued": 1740,
            "lost": 4354,
            "fp_added": 5021,
            "fp_removed": 2407,
        },
        "full": {
            "tp": 47740,
            "rescued": 1737,
            "lost": 4738,
            "fp_added": 4879,
            "fp_removed": 16055,
        },
    },

    "T": {
        "cls_only": {
            "tp": 50455,
            "rescued": 1,
            "lost": 287,
            "fp_added": 80,
            "fp_removed": 9087,
        },
        "box_only": {
            "tp": 49921,
            "rescued": 863,
            "lost": 1683,
            "fp_added": 2140,
            "fp_removed": 1320,
        },
        "full": {
            "tp": 49619,
            "rescued": 830,
            "lost": 1952,
            "fp_added": 2138,
            "fp_removed": 10309,
        },
    },
}


# ============================================================================
# GENERIC HELPERS
# ============================================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def load_json(path: Path) -> Any:
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def require_file(
    path: Path,
    label: str,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"[FATAL] Missing {label}: {path}"
        )


# ============================================================================
# PROFILE-DRIVEN EVALUATOR POLICY
# ============================================================================

@dataclass(frozen=True)
class EvalPolicy:
    topk: int

    score_threshold: float
    iou_threshold: float

    num_classes: int

    total_frames: int
    total_gt: int

    tp_base: int
    fp_base: int
    fn_base: int

    f1_base: float

    @classmethod
    def from_profile(
        cls,
        path: Path,
    ) -> "EvalPolicy":
        payload = load_json(path)

        policy = payload["evaluator_policy"]
        oracle = payload["canonical_oracle"]

        return cls(
            topk=int(
                policy["topk"]
            ),

            score_threshold=float(
                policy["score_threshold"]
            ),

            iou_threshold=float(
                policy["match_iou_threshold"]
            ),

            num_classes=int(
                policy["num_classes"]
            ),

            total_frames=int(
                oracle["total_frames"]
            ),

            total_gt=int(
                oracle["total_gt"]
            ),

            tp_base=int(
                oracle["tp_base"]
            ),

            fp_base=int(
                oracle["fp_base"]
            ),

            fn_base=int(
                oracle["fn_base"]
            ),

            f1_base=float(
                oracle["f1_base"]
            ),
        )


# ============================================================================
# IOU
# ============================================================================

def cxcywh_iou_matrix(
    boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
) -> torch.Tensor:
    """
    Pairwise IoU for normalized cxcywh boxes.

    Returns:
        [Q, G]
    """

    if (
        boxes.ndim != 2
        or boxes.shape[-1] != 4
    ):
        raise ValueError(
            "boxes must have shape [Q,4]"
        )

    if (
        gt_boxes.ndim != 2
        or gt_boxes.shape[-1] != 4
    ):
        raise ValueError(
            "gt_boxes must have shape [G,4]"
        )

    if gt_boxes.numel() == 0:
        return boxes.new_zeros(
            (
                boxes.shape[0],
                0,
            )
        )

    bxy1 = (
        boxes[:, :2]
        - boxes[:, 2:] / 2.0
    )

    bxy2 = (
        boxes[:, :2]
        + boxes[:, 2:] / 2.0
    )

    gxy1 = (
        gt_boxes[:, :2]
        - gt_boxes[:, 2:] / 2.0
    )

    gxy2 = (
        gt_boxes[:, :2]
        + gt_boxes[:, 2:] / 2.0
    )

    inter_xy1 = torch.maximum(
        bxy1[:, None, :],
        gxy1[None, :, :],
    )

    inter_xy2 = torch.minimum(
        bxy2[:, None, :],
        gxy2[None, :, :],
    )

    inter_wh = (
        inter_xy2
        - inter_xy1
    ).clamp(
        min=0.0
    )

    inter = (
        inter_wh[..., 0]
        * inter_wh[..., 1]
    )

    b_area = (
        boxes[:, 2]
        * boxes[:, 3]
    )[:, None]

    g_area = (
        gt_boxes[:, 2]
        * gt_boxes[:, 3]
    )[None, :]

    union = (
        b_area
        + g_area
        - inter
    )

    return torch.where(
        union > 0,
        inter / union,
        torch.zeros_like(inter),
    )


# ============================================================================
# TOP-K EXTRACTION
# ============================================================================

def topk_from_logits(
    logits: torch.Tensor,
    boxes: torch.Tensor,
    p: EvalPolicy,
) -> Dict[str, torch.Tensor]:
    probs = logits.sigmoid()

    return topk_from_flat_probs(
        probs.flatten(),
        boxes,
        p,
    )


def topk_from_flat_probs(
    flat_probs: torch.Tensor,
    boxes: torch.Tensor,
    p: EvalPolicy,
) -> Dict[str, torch.Tensor]:
    scores, flat_idx = torch.topk(
        flat_probs,
        min(
            p.topk,
            flat_probs.numel(),
        ),
        largest=True,
        sorted=True,
    )

    q_idx = (
        flat_idx
        // p.num_classes
    )

    c_idx = (
        flat_idx
        % p.num_classes
    )

    selected_boxes = boxes[
        q_idx
    ]

    operational = (
        scores
        >= p.score_threshold
    )

    return {
        "all_scores": scores,
        "all_q": q_idx,
        "all_c": c_idx,
        "all_boxes": selected_boxes,

        "op_scores": scores[
            operational
        ],
        "op_q": q_idx[
            operational
        ],
        "op_c": c_idx[
            operational
        ],
        "op_boxes": selected_boxes[
            operational
        ],
    }


# ============================================================================
# MATCHERS
# ============================================================================

def greedy_match_from_iou(
    pred: Mapping[
        str,
        torch.Tensor,
    ],
    gt_labels: torch.Tensor,
    iou_qg: torch.Tensor,
    p: EvalPolicy,
) -> Dict[str, Any]:
    """
    Exact score-ordered same-class greedy matcher using a cached
    query-to-GT IoU matrix.
    """

    matched_gt: Set[int] = set()

    det_to_gt: Dict[
        Tuple[int, int],
        Tuple[
            int,
            float,
            float,
        ],
    ] = {}

    fp_pairs: Set[
        Tuple[int, int]
    ] = set()

    scores = pred["op_scores"]
    q_idx = pred["op_q"]
    c_idx = pred["op_c"]

    for d in range(
        scores.numel()
    ):
        q = int(
            q_idx[d]
        )

        c = int(
            c_idx[d]
        )

        best_gt = None
        best_iou = -1.0

        for g in range(
            gt_labels.numel()
        ):
            if g in matched_gt:
                continue

            if (
                int(gt_labels[g])
                != c
            ):
                continue

            iou = float(
                iou_qg[q, g]
            )

            if (
                iou
                >= p.iou_threshold
                and iou
                > best_iou
            ):
                best_iou = iou
                best_gt = g

        pair = (
            q,
            c,
        )

        if best_gt is None:
            fp_pairs.add(
                pair
            )

        else:
            matched_gt.add(
                best_gt
            )

            det_to_gt[
                pair
            ] = (
                best_gt,
                best_iou,
                float(
                    scores[d]
                ),
            )

    tp = len(
        matched_gt
    )

    fp = len(
        fp_pairs
    )

    fn = (
        int(
            gt_labels.numel()
        )
        - tp
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,

        "matched_gt": matched_gt,

        "det_to_gt": det_to_gt,

        "fp_pairs": fp_pairs,
    }


def brute_match(
    pred: Mapping[
        str,
        torch.Tensor,
    ],
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    p: EvalPolicy,
) -> Dict[str, Any]:
    """
    Independent brute reference matcher operating directly on selected
    detection boxes.
    """

    iou = cxcywh_iou_matrix(
        pred["op_boxes"],
        gt_boxes,
    )

    matched_gt: Set[int] = set()

    det_to_gt: Dict[
        Tuple[int, int],
        Tuple[
            int,
            float,
            float,
        ],
    ] = {}

    fp_pairs: Set[
        Tuple[int, int]
    ] = set()

    for d in range(
        pred["op_scores"].numel()
    ):
        q = int(
            pred["op_q"][d]
        )

        c = int(
            pred["op_c"][d]
        )

        best_gt = None
        best_iou = -1.0

        for g in range(
            gt_labels.numel()
        ):
            if g in matched_gt:
                continue

            if (
                int(gt_labels[g])
                != c
            ):
                continue

            value = float(
                iou[d, g]
            )

            if (
                value
                >= p.iou_threshold
                and value
                > best_iou
            ):
                best_iou = value
                best_gt = g

        pair = (
            q,
            c,
        )

        if best_gt is None:
            fp_pairs.add(
                pair
            )

        else:
            matched_gt.add(
                best_gt
            )

            det_to_gt[
                pair
            ] = (
                best_gt,
                best_iou,
                float(
                    pred["op_scores"][d]
                ),
            )

    tp = len(
        matched_gt
    )

    fp = len(
        fp_pairs
    )

    return {
        "tp": tp,
        "fp": fp,

        "fn": (
            int(
                gt_labels.numel()
            )
            - tp
        ),

        "matched_gt": matched_gt,

        "det_to_gt": det_to_gt,

        "fp_pairs": fp_pairs,
    }


# ============================================================================
# EXACTNESS HELPERS
# ============================================================================

def pair_list(
    pred: Mapping[
        str,
        torch.Tensor,
    ],
    operational: bool,
) -> List[
    Tuple[int, int]
]:
    q = (
        pred["op_q"]
        if operational
        else pred["all_q"]
    )

    c = (
        pred["op_c"]
        if operational
        else pred["all_c"]
    )

    return list(
        zip(
            q.cpu().tolist(),
            c.cpu().tolist(),
        )
    )


def assert_match_equivalent(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    context: str,
) -> None:
    for key in (
        "tp",
        "fp",
        "fn",
    ):
        if (
            a[key]
            != b[key]
        ):
            raise AssertionError(
                f"{context}: "
                f"{key} mismatch "
                f"{a[key]} != {b[key]}"
            )

    if (
        a["matched_gt"]
        != b["matched_gt"]
    ):
        raise AssertionError(
            f"{context}: "
            "matched-GT set mismatch"
        )

    if (
        a["fp_pairs"]
        != b["fp_pairs"]
    ):
        raise AssertionError(
            f"{context}: "
            "FP-pair set mismatch"
        )

    if (
        set(
            a["det_to_gt"].keys()
        )
        !=
        set(
            b["det_to_gt"].keys()
        )
    ):
        raise AssertionError(
            f"{context}: "
            "det_to_gt key mismatch"
        )

    for key in a[
        "det_to_gt"
    ]:
        aa = a[
            "det_to_gt"
        ][key]

        bb = b[
            "det_to_gt"
        ][key]

        if (
            aa[0] != bb[0]
            or not math.isclose(
                aa[1],
                bb[1],
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ):
            raise AssertionError(
                f"{context}: "
                f"det_to_gt mismatch "
                f"for {key}: "
                f"{aa} != {bb}"
            )


# ============================================================================
# OOF MASTER / SHARD ACCESS
# ============================================================================

def iter_video_entries(
    master: Mapping[
        str,
        Any,
    ],
) -> Iterable[
    Tuple[
        int,
        int,
        Mapping[
            str,
            Any,
        ],
    ]
]:
    folds = master.get(
        "folds"
    )

    if not isinstance(
        folds,
        Mapping,
    ):
        raise KeyError(
            "Master index "
            "missing 'folds'."
        )

    for fold_name in sorted(
        folds.keys()
    ):
        fold_idx = int(
            str(
                fold_name
            ).replace(
                "fold_",
                "",
            )
        )

        videos = folds[
            fold_name
        ][
            "videos"
        ]

        for vid_str in sorted(
            videos.keys(),
            key=lambda x: int(x),
        ):
            yield (
                fold_idx,
                int(vid_str),
                videos[
                    vid_str
                ],
            )


def load_shard(
    master_path: Path,
    meta: Mapping[
        str,
        Any,
    ],
) -> List[
    Mapping[
        str,
        Any,
    ]
]:
    path = (
        master_path.parent
        / str(
            meta["file"]
        )
    )

    require_file(
        path,
        "OOF expert shard",
    )

    if (
        "sha256" in meta
        and sha256_file(path)
        != meta["sha256"]
    ):
        raise AssertionError(
            "[FATAL] OOF shard "
            f"SHA mismatch: {path}"
        )

    records = torch.load(
        path,
        map_location="cpu",
    )

    if (
        isinstance(
            records,
            Mapping,
        )
        and "records"
        in records
    ):
        records = records[
            "records"
        ]

    if (
        "num_records" in meta
        and len(records)
        != int(
            meta["num_records"]
        )
    ):
        raise AssertionError(
            "[FATAL] record-count "
            f"mismatch for {path}"
        )

    return records


def frame_base(
    fr: Mapping[
        str,
        Any,
    ],
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    required = (
        "z_base",
        "b_base",
        "gt_boxes_norm",
        "gt_labels",
    )

    missing = [
        k
        for k in required
        if k not in fr
    ]

    if missing:
        raise KeyError(
            "OOF frame missing "
            f"keys: {missing}"
        )

    return (
        fr["z_base"].float(),
        fr["b_base"].float(),
        fr["gt_boxes_norm"].float(),
        fr["gt_labels"].long(),
    )


def expert_variant(
    fr: Mapping[
        str,
        Any,
    ],
    expert: str,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
]:
    z_key = (
        f"z_{expert.lower()}"
    )

    b_key = (
        f"b_{expert.lower()}"
    )

    if (
        z_key not in fr
        or b_key not in fr
    ):
        raise KeyError(
            "Frame missing "
            f"{z_key}/{b_key}"
        )

    return (
        fr[z_key].float(),
        fr[b_key].float(),
    )


# ============================================================================
# DETERMINISTIC LOCAL EXACTNESS PREFLIGHT
# ============================================================================

def deterministic_preflight_frames(
    master_path: Path,
    master: Mapping[
        str,
        Any,
    ],
) -> List[
    Tuple[
        int,
        int,
        Mapping[
            str,
            Any,
        ],
    ]
]:
    by_fold: Dict[
        int,
        List[
            Tuple[
                int,
                Mapping[
                    str,
                    Any,
                ],
            ]
        ],
    ] = {}

    for (
        fold_idx,
        vid,
        meta,
    ) in iter_video_entries(
        master
    ):
        by_fold.setdefault(
            fold_idx,
            [],
        ).append(
            (
                vid,
                meta,
            )
        )

    chosen = []

    for fold_idx in sorted(
        by_fold.keys()
    ):
        videos = by_fold[
            fold_idx
        ]

        vid, meta = videos[
            len(videos) // 2
        ]

        records = load_shard(
            master_path,
            meta,
        )

        if not records:
            continue

        record = records[
            len(records) // 2
        ]

        chosen.append(
            (
                fold_idx,
                vid,
                record,
            )
        )

    return chosen


def run_local_exactness_preflight(
    master_path: Path,
    master: Mapping[
        str,
        Any,
    ],
    p: EvalPolicy,
) -> None:
    frames = (
        deterministic_preflight_frames(
            master_path,
            master,
        )
    )

    if not frames:
        raise AssertionError(
            "No frames available "
            "for F0-L preflight."
        )

    print(
        "[*] F0-L exactness preflight: "
        f"{len(frames)} deterministic "
        "frames across folds"
    )

    comparisons = 0

    for (
        fold_idx,
        vid,
        fr,
    ) in frames:
        (
            z_base,
            b_base,
            gt_boxes,
            gt_labels,
        ) = frame_base(fr)

        # Immutable references.
        base_flat = (
            z_base
            .sigmoid()
            .flatten()
            .clone()
        )

        base_iou_qg = (
            cxcywh_iou_matrix(
                b_base,
                gt_boxes,
            )
        )

        base_pred = (
            topk_from_flat_probs(
                base_flat,
                b_base,
                p,
            )
        )

        active_queries = sorted(
            set(
                base_pred[
                    "all_q"
                ].tolist()
            )
        )

        if (
            len(
                active_queries
            )
            <= 5
        ):
            q_tests = (
                active_queries
            )
        else:
            q_tests = [
                active_queries[0],
                active_queries[
                    len(
                        active_queries
                    ) // 2
                ],
                active_queries[-1],
            ]

        for expert in EXPERTS:
            (
                z_exp,
                b_exp,
            ) = expert_variant(
                fr,
                expert,
            )

            p_exp = z_exp.sigmoid()

            for q in q_tests:
                # ------------------------------------------------------
                # CLASSIFICATION BRUTE REPLAY
                # ------------------------------------------------------

                z_local = (
                    z_base.clone()
                )

                z_local[q] = (
                    z_exp[q]
                )

                pred_brute = (
                    topk_from_logits(
                        z_local,
                        b_base,
                        p,
                    )
                )

                match_brute = (
                    brute_match(
                        pred_brute,
                        gt_boxes,
                        gt_labels,
                        p,
                    )
                )

                # ------------------------------------------------------
                # CLASSIFICATION CACHED EXACT REPLAY
                # ------------------------------------------------------

                flat_fast = (
                    base_flat.clone()
                )

                start = (
                    q
                    * p.num_classes
                )

                flat_fast[
                    start:
                    start
                    + p.num_classes
                ] = p_exp[q]

                pred_fast = (
                    topk_from_flat_probs(
                        flat_fast,
                        b_base,
                        p,
                    )
                )

                match_fast = (
                    greedy_match_from_iou(
                        pred_fast,
                        gt_labels,
                        base_iou_qg,
                        p,
                    )
                )

                if (
                    pair_list(
                        pred_brute,
                        False,
                    )
                    !=
                    pair_list(
                        pred_fast,
                        False,
                    )
                ):
                    raise AssertionError(
                        f"F{fold_idx} "
                        f"V{vid} "
                        f"{expert} q{q} "
                        "cls Top-K identity "
                        "mismatch"
                    )

                if (
                    pair_list(
                        pred_brute,
                        True,
                    )
                    !=
                    pair_list(
                        pred_fast,
                        True,
                    )
                ):
                    raise AssertionError(
                        f"F{fold_idx} "
                        f"V{vid} "
                        f"{expert} q{q} "
                        "cls operational "
                        "identity mismatch"
                    )

                assert_match_equivalent(
                    match_brute,
                    match_fast,
                    (
                        f"F{fold_idx} "
                        f"V{vid} "
                        f"{expert} "
                        f"q{q} cls"
                    ),
                )

                # ------------------------------------------------------
                # BOX BRUTE REPLAY
                # ------------------------------------------------------

                b_local = (
                    b_base.clone()
                )

                b_local[q] = (
                    b_exp[q]
                )

                pred_box_brute = (
                    topk_from_logits(
                        z_base,
                        b_local,
                        p,
                    )
                )

                match_box_brute = (
                    brute_match(
                        pred_box_brute,
                        gt_boxes,
                        gt_labels,
                        p,
                    )
                )

                # ------------------------------------------------------
                # BOX CACHED EXACT REPLAY
                # ------------------------------------------------------

                pred_box_fast = dict(
                    base_pred
                )

                pred_box_fast[
                    "all_boxes"
                ] = b_local[
                    pred_box_fast[
                        "all_q"
                    ]
                ]

                op_mask = (
                    pred_box_fast[
                        "all_scores"
                    ]
                    >= p.score_threshold
                )

                pred_box_fast[
                    "op_scores"
                ] = pred_box_fast[
                    "all_scores"
                ][
                    op_mask
                ]

                pred_box_fast[
                    "op_q"
                ] = pred_box_fast[
                    "all_q"
                ][
                    op_mask
                ]

                pred_box_fast[
                    "op_c"
                ] = pred_box_fast[
                    "all_c"
                ][
                    op_mask
                ]

                pred_box_fast[
                    "op_boxes"
                ] = pred_box_fast[
                    "all_boxes"
                ][
                    op_mask
                ]

                iou_fast = (
                    base_iou_qg.clone()
                )

                iou_fast[
                    q:
                    q + 1
                ] = cxcywh_iou_matrix(
                    b_exp[
                        q:
                        q + 1
                    ],
                    gt_boxes,
                )

                match_box_fast = (
                    greedy_match_from_iou(
                        pred_box_fast,
                        gt_labels,
                        iou_fast,
                        p,
                    )
                )

                if (
                    pair_list(
                        pred_box_brute,
                        False,
                    )
                    !=
                    pair_list(
                        pred_box_fast,
                        False,
                    )
                ):
                    raise AssertionError(
                        f"F{fold_idx} "
                        f"V{vid} "
                        f"{expert} q{q} "
                        "box Top-K identity "
                        "mismatch"
                    )

                if (
                    pair_list(
                        pred_box_brute,
                        True,
                    )
                    !=
                    pair_list(
                        pred_box_fast,
                        True,
                    )
                ):
                    raise AssertionError(
                        f"F{fold_idx} "
                        f"V{vid} "
                        f"{expert} q{q} "
                        "box operational "
                        "identity mismatch"
                    )

                assert_match_equivalent(
                    match_box_brute,
                    match_box_fast,
                    (
                        f"F{fold_idx} "
                        f"V{vid} "
                        f"{expert} "
                        f"q{q} box"
                    ),
                )

                comparisons += 2

    print(
        "[+] F0-L EXACTNESS PREFLIGHT "
        f"PASS: {comparisons} local "
        "cls/box comparisons matched "
        "brute replay exactly."
    )


# ============================================================================
# FULL F0-G EXPERT-v0 CERTIFICATION
# ============================================================================

def empty_attr():
    return {
        expert: {
            variant: {
                "tp": 0,
                "fp": 0,

                "rescued": 0,
                "lost": 0,

                "fp_added": 0,
                "fp_removed": 0,

                "retained": 0,
            }
            for variant in (
                "cls_only",
                "box_only",
                "full",
            )
        }
        for expert in EXPERTS
    }


def accumulate_variant(
    attr: MutableMapping[
        str,
        int,
    ],
    base: Mapping[
        str,
        Any,
    ],
    variant: Mapping[
        str,
        Any,
    ],
) -> None:
    attr["tp"] += int(
        variant["tp"]
    )

    attr["fp"] += int(
        variant["fp"]
    )

    attr[
        "fp_added"
    ] += len(
        variant["fp_pairs"]
        - base["fp_pairs"]
    )

    attr[
        "fp_removed"
    ] += len(
        base["fp_pairs"]
        - variant["fp_pairs"]
    )

    all_gt = (
        base["matched_gt"]
        | variant["matched_gt"]
    )

    for g in all_gt:
        in_base = (
            g
            in base["matched_gt"]
        )

        in_variant = (
            g
            in variant["matched_gt"]
        )

        if (
            not in_base
            and in_variant
        ):
            attr[
                "rescued"
            ] += 1

        elif (
            in_base
            and not in_variant
        ):
            attr[
                "lost"
            ] += 1

        elif (
            in_base
            and in_variant
        ):
            attr[
                "retained"
            ] += 1


def certify_v0_global(
    master_path: Path,
    master: Mapping[
        str,
        Any,
    ],
    p: EvalPolicy,
) -> None:
    base_tp = 0
    base_fp = 0
    base_fn = 0
    base_gt = 0
    frames = 0

    attr = empty_attr()

    print(
        "[*] Running full F0-G "
        "frozen Expert-v0 "
        "certification across "
        "all OOF frames..."
    )

    for (
        fold_idx,
        vid,
        meta,
    ) in iter_video_entries(
        master
    ):
        records = load_shard(
            master_path,
            meta,
        )

        for fr in records:
            (
                z_base,
                b_base,
                gt_boxes,
                gt_labels,
            ) = frame_base(fr)

            base_pred = (
                topk_from_logits(
                    z_base,
                    b_base,
                    p,
                )
            )

            base_match = (
                brute_match(
                    base_pred,
                    gt_boxes,
                    gt_labels,
                    p,
                )
            )

            base_tp += (
                base_match["tp"]
            )

            base_fp += (
                base_match["fp"]
            )

            base_fn += (
                base_match["fn"]
            )

            base_gt += int(
                gt_labels.numel()
            )

            frames += 1

            for expert in EXPERTS:
                (
                    z_exp,
                    b_exp,
                ) = expert_variant(
                    fr,
                    expert,
                )

                variants = {
                    "cls_only":
                        topk_from_logits(
                            z_exp,
                            b_base,
                            p,
                        ),

                    "box_only":
                        topk_from_logits(
                            z_base,
                            b_exp,
                            p,
                        ),

                    "full":
                        topk_from_logits(
                            z_exp,
                            b_exp,
                            p,
                        ),
                }

                for (
                    name,
                    pred,
                ) in variants.items():
                    variant_match = (
                        brute_match(
                            pred,
                            gt_boxes,
                            gt_labels,
                            p,
                        )
                    )

                    accumulate_variant(
                        attr[
                            expert
                        ][
                            name
                        ],
                        base_match,
                        variant_match,
                    )

        print(
            f"    fold={fold_idx} "
            f"video={vid}: "
            f"cumulative "
            f"frames={frames:,}"
        )

    if (
        frames
        != p.total_frames
        or base_gt
        != p.total_gt
        or base_tp
        != p.tp_base
        or base_fp
        != p.fp_base
        or base_fn
        != p.fn_base
    ):
        raise AssertionError(
            "Base oracle mismatch: "
            f"frames="
            f"{frames}/"
            f"{p.total_frames}, "
            f"GT="
            f"{base_gt}/"
            f"{p.total_gt}, "
            f"TP="
            f"{base_tp}/"
            f"{p.tp_base}, "
            f"FP="
            f"{base_fp}/"
            f"{p.fp_base}, "
            f"FN="
            f"{base_fn}/"
            f"{p.fn_base}"
        )

    f1 = (
        2.0
        * base_tp
        / max(
            1.0,
            (
                2.0
                * base_tp
                + base_fp
                + base_fn
            ),
        )
    )

    if not math.isclose(
        f1,
        p.f1_base,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise AssertionError(
            "F1 mismatch: "
            f"{f1} != "
            f"{p.f1_base}"
        )

    for expert in EXPERTS:
        for variant in (
            "cls_only",
            "box_only",
            "full",
        ):
            obs = attr[
                expert
            ][
                variant
            ]

            tgt = (
                FROZEN_D1_TARGETS[
                    expert
                ][
                    variant
                ]
            )

            for key in (
                "tp",
                "rescued",
                "lost",
                "fp_added",
                "fp_removed",
            ):
                if (
                    obs[key]
                    != tgt[key]
                ):
                    raise AssertionError(
                        "D1 mismatch "
                        f"{expert}/"
                        f"{variant}/"
                        f"{key}: "
                        f"{obs[key]} "
                        f"!= {tgt[key]}"
                    )

            if (
                obs["fp_added"]
                - obs["fp_removed"]
                !=
                obs["fp"]
                - p.fp_base
            ):
                raise AssertionError(
                    "FP conservation "
                    "mismatch "
                    f"{expert}/"
                    f"{variant}"
                )

            print(
                "    PASS "
                f"{expert}/"
                f"{variant}: "
                f"TP="
                f"{obs['tp']:,}, "
                f"rescue/loss="
                f"{obs['rescued']:,}/"
                f"{obs['lost']:,}, "
                f"FP add/remove="
                f"{obs['fp_added']:,}/"
                f"{obs['fp_removed']:,}"
            )

    print(
        "[+] F0-G CERTIFICATION PASS: "
        "Base oracle and all 12 frozen "
        "D1 variants match exactly."
    )


# ============================================================================
# FINAL-EXPERT LOCAL SUPERVISION COMPILATION
# ============================================================================

def local_transition_state(
    pair: Tuple[int, int],
    match: Mapping[
        str,
        Any,
    ],
) -> int:
    if pair in match[
        "det_to_gt"
    ]:
        return 0  # TP

    if pair in match[
        "fp_pairs"
    ]:
        return 1  # FP

    return 2      # NONOP


def best_same_class_iou(
    q: int,
    gt_labels: torch.Tensor,
    iou_qg: torch.Tensor,
    class_ids: Sequence[int],
) -> Dict[int, float]:
    out = {}

    for c in class_ids:
        idx = (
            gt_labels
            == int(c)
        ).nonzero(
            as_tuple=False
        ).flatten()

        if idx.numel():
            out[
                int(c)
            ] = float(
                iou_qg[
                    q,
                    idx,
                ].max()
            )

        else:
            out[
                int(c)
            ] = 0.0

    return out


def compile_final_local(
    master_path: Path,
    master: Mapping[
        str,
        Any,
    ],
    p: EvalPolicy,
    out_root: Path,
    baseline_profile_path: Path,
    d_report_path: Path,
) -> Path:
    source_sha = sha256_file(
        master_path
    )

    driver_sha = sha256_file(
        Path(__file__)
    )

    baseline_profile_sha = (
        sha256_file(
            baseline_profile_path
        )
    )

    d_report_sha = (
        sha256_file(
            d_report_path
        )
    )

    namespace = (
        out_root
        / (
            f"{source_sha[:16]}_"
            f"{driver_sha[:12]}"
        )
    )

    # Never overwrite a previous ledger.
    namespace.mkdir(
        parents=True,
        exist_ok=False,
    )

    shard_hashes = {}

    transition = {
        expert:
            np.zeros(
                (3, 3),
                dtype=np.int64,
            )
        for expert
        in EXPERTS
    }

    total_frames = 0

    for (
        fold_idx,
        vid,
        meta,
    ) in iter_video_entries(
        master
    ):
        records = load_shard(
            master_path,
            meta,
        )

        out_records = []

        for fr in records:
            (
                z_base,
                b_base,
                gt_boxes,
                gt_labels,
            ) = frame_base(fr)

            base_pred = (
                topk_from_logits(
                    z_base,
                    b_base,
                    p,
                )
            )

            base_match = (
                brute_match(
                    base_pred,
                    gt_boxes,
                    gt_labels,
                    p,
                )
            )

            base_pairs = set(
                pair_list(
                    base_pred,
                    False,
                )
            )

            Q_B = sorted(
                set(
                    base_pred[
                        "all_q"
                    ].cpu().tolist()
                )
            )

            p_base = (
                z_base.sigmoid()
            )

            iou_base_qg = (
                cxcywh_iou_matrix(
                    b_base,
                    gt_boxes,
                )
            )

            expert_records = {}

            for expert in EXPERTS:
                (
                    z_exp,
                    b_exp,
                ) = expert_variant(
                    fr,
                    expert,
                )

                p_exp = (
                    z_exp.sigmoid()
                )

                cls_local = {}
                box_local = {}

                for q in Q_B:
                    # --------------------------------------------------
                    # CLASSIFICATION LOCAL COUNTERFACTUAL
                    # --------------------------------------------------

                    z_local = (
                        z_base.clone()
                    )

                    z_local[q] = (
                        z_exp[q]
                    )

                    pred_cls = (
                        topk_from_logits(
                            z_local,
                            b_base,
                            p,
                        )
                    )

                    match_cls = (
                        brute_match(
                            pred_cls,
                            gt_boxes,
                            gt_labels,
                            p,
                        )
                    )

                    after_pairs = set(
                        pair_list(
                            pred_cls,
                            False,
                        )
                    )

                    hist = np.zeros(
                        (3, 3),
                        dtype=np.int16,
                    )

                    class_rows = {}

                    for c in range(
                        p.num_classes
                    ):
                        pair = (
                            q,
                            c,
                        )

                        s0 = (
                            local_transition_state(
                                pair,
                                base_match,
                            )
                        )

                        s1 = (
                            local_transition_state(
                                pair,
                                match_cls,
                            )
                        )

                        hist[
                            s0,
                            s1,
                        ] += 1

                        transition[
                            expert
                        ][
                            s0,
                            s1,
                        ] += 1

                        score_base = float(
                            p_base[
                                q,
                                c,
                            ]
                        )

                        score_after = float(
                            p_exp[
                                q,
                                c,
                            ]
                        )

                        class_rows[
                            c
                        ] = {
                            "state_base":
                                s0,

                            "state_after":
                                s1,

                            "score_base":
                                score_base,

                            "score_after":
                                score_after,

                            "delta_score":
                                (
                                    score_after
                                    - score_base
                                ),

                            "base_in_candidate_set":
                                (
                                    pair
                                    in base_pairs
                                ),

                            "after_in_candidate_set":
                                (
                                    pair
                                    in after_pairs
                                ),

                            "base_operational":
                                (
                                    pair
                                    in base_match[
                                        "det_to_gt"
                                    ]
                                    or pair
                                    in base_match[
                                        "fp_pairs"
                                    ]
                                ),

                            "after_operational":
                                (
                                    pair
                                    in match_cls[
                                        "det_to_gt"
                                    ]
                                    or pair
                                    in match_cls[
                                        "fp_pairs"
                                    ]
                                ),
                        }

                    cls_local[q] = {
                        "class_transitions":
                            class_rows,

                        "hist_3x3":
                            hist.tolist(),

                        "n_preserve":
                            int(
                                hist[0, 0]
                            ),

                        "n_loss":
                            int(
                                hist[0, 1]
                                + hist[0, 2]
                            ),

                        "n_prune":
                            int(
                                hist[1, 2]
                            ),

                        "n_add":
                            int(
                                hist[2, 1]
                            ),

                        "n_fp_to_tp":
                            int(
                                hist[1, 0]
                            ),

                        "n_rescue":
                            int(
                                hist[2, 0]
                            ),
                    }

                    # --------------------------------------------------
                    # BOX LOCAL COUNTERFACTUAL
                    # --------------------------------------------------

                    b_local = (
                        b_base.clone()
                    )

                    b_local[q] = (
                        b_exp[q]
                    )

                    pred_box = (
                        topk_from_logits(
                            z_base,
                            b_local,
                            p,
                        )
                    )

                    match_box = (
                        brute_match(
                            pred_box,
                            gt_boxes,
                            gt_labels,
                            p,
                        )
                    )

                    base_gt_q = {
                        value[0]
                        for key, value
                        in base_match[
                            "det_to_gt"
                        ].items()
                        if key[0] == q
                    }

                    after_gt_q = {
                        value[0]
                        for key, value
                        in match_box[
                            "det_to_gt"
                        ].items()
                        if key[0] == q
                    }

                    iou_after_qg = (
                        iou_base_qg.clone()
                    )

                    iou_after_qg[
                        q:
                        q + 1
                    ] = (
                        cxcywh_iou_matrix(
                            b_exp[
                                q:
                                q + 1
                            ],
                            gt_boxes,
                        )
                    )

                    # Store continuous same-class localization
                    # progress even when the IoU remains below 0.30.
                    class_ids = list(
                        range(
                            p.num_classes
                        )
                    )

                    base_best = (
                        best_same_class_iou(
                            q,
                            gt_labels,
                            iou_base_qg,
                            class_ids,
                        )
                    )

                    after_best = (
                        best_same_class_iou(
                            q,
                            gt_labels,
                            iou_after_qg,
                            class_ids,
                        )
                    )

                    box_local[q] = {
                        "retained_gt_ids":
                            sorted(
                                base_gt_q
                                & after_gt_q
                            ),

                        "lost_gt_ids":
                            sorted(
                                base_gt_q
                                - after_gt_q
                            ),

                        "rescued_gt_ids":
                            sorted(
                                after_gt_q
                                - base_gt_q
                            ),

                        "best_same_class_iou_base":
                            base_best,

                        "best_same_class_iou_after":
                            after_best,

                        "best_same_class_delta_iou": {
                            c:
                                (
                                    after_best[c]
                                    - base_best[c]
                                )
                            for c
                            in class_ids
                        },
                    }

                expert_records[
                    expert
                ] = {
                    "cls_local":
                        cls_local,

                    "box_local":
                        box_local,
                }

            out_records.append(
                {
                    "image_id":
                        int(
                            fr["image_id"]
                        ),

                    "seq_idx":
                        int(
                            fr["seq_idx"]
                        ),

                    "video_id":
                        vid,

                    "fold_idx":
                        fold_idx,

                    "C_B":
                        pair_list(
                            base_pred,
                            False,
                        ),

                    "Q_B":
                        Q_B,

                    "experts":
                        expert_records,
                }
            )

            total_frames += 1

        shard_name = (
            "bfcef_f_supervision_"
            f"fold{fold_idx}_"
            f"vid{vid}.pth"
        )

        shard_path = (
            namespace
            / shard_name
        )

        torch.save(
            {
                "records":
                    out_records,

                "source_master_sha256":
                    source_sha,
            },
            shard_path,
        )

        shard_hashes[
            shard_name
        ] = sha256_file(
            shard_path
        )

        print(
            "    compiled "
            f"fold={fold_idx} "
            f"video={vid} "
            f"frames="
            f"{len(out_records)} "
            f"cumulative="
            f"{total_frames:,}"
        )

        del (
            out_records,
            records,
        )

        gc.collect()

    index = {
        "schema":
            "bfcef_f_gate_supervision_v1",

        "source_master_sha256":
            source_sha,

        "source_master_path":
            str(
                master_path.resolve()
            ),

        "compiler_sha256":
            driver_sha,

        "baseline_profile_sha256":
            baseline_profile_sha,

        "d_report_sha256":
            d_report_sha,

        "total_frames":
            total_frames,

        "transition_matrices": {
            expert:
                transition[
                    expert
                ].tolist()
            for expert
            in EXPERTS
        },

        "shards":
            shard_hashes,
    }

    index_path = (
        namespace
        / "bfcef_f_gate_supervision_index.json"
    )

    index_path.write_text(
        json.dumps(
            index,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "[+] Final-expert "
        "F0-L supervision "
        f"index: {index_path}"
    )

    print(
        "[+] SHA-256: "
        f"{sha256_file(index_path)}"
    )

    return index_path


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "B-FCEF-F F0 "
            "Supervision Compiler"
        )
    )

    parser.add_argument(
        "--master-index",
        type=Path,
        default=DEFAULT_MASTER,
    )

    parser.add_argument(
        "--baseline-profile",
        type=Path,
        default=DEFAULT_BASELINE_PROFILE,
    )

    parser.add_argument(
        "--d-report",
        type=Path,
        default=DEFAULT_D_REPORT,
    )

    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT,
    )

    modes = (
        parser
        .add_mutually_exclusive_group(
            required=True
        )
    )

    modes.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Run deterministic "
            "brute-vs-cached "
            "F0-L exactness "
            "preflight only."
        ),
    )

    modes.add_argument(
        "--certify-v0",
        action="store_true",
        help=(
            "Run F0-L preflight "
            "then full frozen "
            "Expert-v0 F0-G "
            "certification."
        ),
    )

    modes.add_argument(
        "--compile-final",
        action="store_true",
        help=(
            "Compile full local "
            "Gate supervision "
            "for the supplied "
            "Final-Expert OOF ledger."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    for path, label in (
        (
            args.master_index,
            "OOF master index",
        ),
        (
            args.baseline_profile,
            "baseline profile",
        ),
        (
            args.d_report,
            "D0-D5+ report",
        ),
    ):
        require_file(
            path,
            label,
        )

    # --------------------------------------------------------------
    # D-report provenance
    # --------------------------------------------------------------

    d_sha = sha256_file(
        args.d_report
    )

    if (
        d_sha
        != FROZEN_D_REPORT_SHA256
    ):
        raise AssertionError(
            "[FATAL] D-report "
            "SHA mismatch: "
            f"{d_sha}"
        )

    # --------------------------------------------------------------
    # R0 checkpoint provenance
    # --------------------------------------------------------------

    profile = load_json(
        args.baseline_profile
    )

    checkpoint_path = Path(
        profile[
            "checkpoint"
        ][
            "path"
        ]
    )

    require_file(
        checkpoint_path,
        "R0 checkpoint",
    )

    checkpoint_sha = (
        sha256_file(
            checkpoint_path
        )
    )

    if (
        checkpoint_sha
        != profile[
            "checkpoint"
        ][
            "sha256"
        ]
        or checkpoint_sha
        != FROZEN_R0_CHECKPOINT_SHA256
    ):
        raise AssertionError(
            "[FATAL] R0 "
            "checkpoint SHA "
            "mismatch: "
            f"{checkpoint_sha}"
        )

    # --------------------------------------------------------------
    # Master ledger provenance
    # --------------------------------------------------------------

    master_sha = sha256_file(
        args.master_index
    )

    master = load_json(
        args.master_index
    )

    policy = (
        EvalPolicy.from_profile(
            args.baseline_profile
        )
    )

    if (
        args.preflight_only
        or args.certify_v0
    ):
        if (
            master_sha
            != FROZEN_E0_MASTER_SHA256
        ):
            raise AssertionError(
                "[FATAL] frozen "
                "E0 master SHA "
                "mismatch: "
                f"{master_sha}"
            )

    print(
        "=" * 100
    )

    print(
        "B-FCEF-F F0 "
        "SUPERVISION COMPILER"
    )

    print(
        f"Master: "
        f"{args.master_index}"
    )

    print(
        f"Master SHA: "
        f"{master_sha}"
    )

    print(
        f"D-report SHA: "
        f"{d_sha}"
    )

    print(
        f"R0 checkpoint SHA: "
        f"{checkpoint_sha}"
    )

    print(
        "=" * 100
    )

    # Exact replay semantics are always checked before either
    # certification or local-ledger compilation.
    run_local_exactness_preflight(
        args.master_index,
        master,
        policy,
    )

    if args.preflight_only:
        return

    if args.certify_v0:
        certify_v0_global(
            args.master_index,
            master,
            policy,
        )
        return

    if args.compile_final:
        args.out_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        compile_final_local(
            args.master_index,
            master,
            policy,
            args.out_root,
            args.baseline_profile,
            args.d_report,
        )

        return


if __name__ == "__main__":
    main()