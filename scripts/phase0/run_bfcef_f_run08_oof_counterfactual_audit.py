from __future__ import annotations

"""
B-FCEF-F RUN_08 -- Final-Expert OOF Counterfactual Audit
========================================================

Read-only audit over the complete 20,075-frame Train OOF substrate produced by
RUN_07. No training, no model inference, no Val/Test access, and no U/H label
compilation.

Audit conditions
----------------
Base plus, for each Final Expert S/L/C/T:
  - cls-only: expert logits + baseline boxes
  - box-only: baseline logits + expert boxes
  - full:     expert logits + expert boxes

Frozen evaluator semantics (copied from certified G0/F0-G contract)
------------------------------------------------------------------
  - sigmoid(logits), flatten [300 x 7], global Top-100 query-class pairs
  - COCO evaluator receives all Top-100 detections
  - operational detector uses score >= 0.20
  - same-class greedy GT matching at IoU >= 0.30
  - operational FP identity is (q_idx, class_idx)

The script first hard-reproduces the frozen R0 oracle:
  GT=64,670, TP=50,741, FP=52,801, FN=13,929,
  F1=0.6032982189..., mAP@0.30=0.731700 within 1e-4.

Outputs
-------
D:/MARIVISION/H0_workbench/eval_tools/bfcef_f_s0/run08_final_expert_oof_audit_v1/
  run08_final_expert_oof_audit_report.json
  run08_global_metrics.csv
  run08_complementarity.json

No file under the RUN_07 OOF root is modified.
"""

import contextlib
import copy
import csv
import gc
import hashlib
import io
import json
import math
import os
import sys
import time
from PIL import Image
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path("D:/MARIVISION")
EVAL = ROOT / "H0_workbench" / "eval_tools"
E0 = EVAL / "e0_cache"
FOLD_DIR = EVAL / "e0_adapters"
S0_ROOT = EVAL / "bfcef_f_s0" / "final_expert_crossfit_v1"
OUT = EVAL / "bfcef_f_s0" / "run08_final_expert_oof_audit_v1"

TRAIN_JSON = ROOT / "seaclips-train.json"
CANONICAL_MANIFEST = E0 / "bfcef_e0_cache_manifest.json"
FOLD_MANIFEST = FOLD_DIR / "bfcef_e0_5fold_manifest.json"
OOF_MANIFEST = S0_ROOT / "final_expert_oof_manifest.json"
FIXED_MATCH_MANIFEST = S0_ROOT / "fixed_match_v1" / "bfcef_f_s0a_fixed_match_manifest.json"

# Locate authoritative Deformable-DETR repository for CocoEvaluator only.
DEFORMABLE_ROOT = None
for _candidate in (ROOT / "Deformable-DETR", ROOT / "deformable_detr"):
    if (
        _candidate.exists()
        and (_candidate / "datasets" / "coco_eval.py").exists()
        and (_candidate / "models" / "deformable_detr.py").exists()
    ):
        DEFORMABLE_ROOT = _candidate
        break

if DEFORMABLE_ROOT is None:
    raise AssertionError("[FATAL] Cannot locate authoritative Deformable-DETR repository.")

for _p in reversed([EVAL, DEFORMABLE_ROOT, ROOT]):
    _s = str(_p)
    if _s in sys.path:
        sys.path.remove(_s)
    sys.path.insert(0, _s)

from datasets.coco_eval import CocoEvaluator  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402


# =============================================================================
# FROZEN ROOTS / CONTRACT
# =============================================================================

EXPECTED_OOF_MANIFEST_SHA = (
    "1c2569f1871b475e4aa71d20e505c8bb738d9b3ed7d4df185787c1c5546c6001"
)
EXPECTED_RUN07_TRAINER_SHA = (
    "a2455d2aa37f91f3a7c01c9fa307bf06026741115d46a0cd9e391642bbed5009"
)
EXPECTED_CANONICAL_MANIFEST_SHA = (
    "210ad12de4ebe02cdbe000956972955ae58ceb8f0d222b703787f33a49af03cd"
)
EXPECTED_FOLD_MANIFEST_SHA = (
    "c923f8abd5833996331566582f568b2b0e9ad1d7d17f606c8ddae693b6b8a103"
)
EXPECTED_FIXED_MATCH_MANIFEST_SHA = (
    "405fe3419336718605a341065186eaeaa8ddc662d374759840334223d17dc862"
)
EXPECTED_R0_SHA = (
    "28e6288c3f6769c2d5275aa77ddb248735b55a09cd46a6b570e3f117890b268b"
)
EXPECTED_SPATIAL_MANIFEST_SHA = (
    "8899edf3eb3b46428dc3a2186058063c74045d14e0c75744fc0a0793daefa9b5"
)
EXPECTED_TEMPORAL_MANIFEST_SHA = (
    "d84d2b75cc043da855c1490a66e727b677c7c70cb7ab881c7c08b2308bf3e985"
)
EXPECTED_FIRST_TEMP_MANIFEST_SHA = (
    "e4754d9c82c60836d817e49ceb55201807c22405677558bb2a69b3e96a6bd286"
)
EXPECTED_SCREENING_CONFIG_SHA = (
    "f5a03e543c13360b50b38ec5850862cf36d2801d90b6c24b1a04f794e2281c26"
)
EXPECTED_OOF_SCHEMA = "bfcef_f_s0a_final_expert_oof_v1"
EXPECTED_OBJECTIVE = "s0a_final_expert_fixed_baseline_match_task_decoupled_v1"
EXPECTED_EXPERT_ORDER = ("S", "L", "C", "T")

NUM_VIDEOS = 47
NUM_FOLDS = 5
NUM_QUERIES = 300
NUM_CLASSES = 7
EXPECTED_TOTAL_FRAMES = 20_075
EXPECTED_TOTAL_GT = 64_670

CERTIFIED_BASE_TP = 50_741
CERTIFIED_BASE_FP = 52_801
CERTIFIED_BASE_FN = 13_929
CERTIFIED_BASE_F1 = 2.0 * CERTIFIED_BASE_TP / (
    2.0 * CERTIFIED_BASE_TP + CERTIFIED_BASE_FP + CERTIFIED_BASE_FN
)
CERTIFIED_BASE_MAP = 0.731700

TOPK = 100
SCORE_THRESHOLD = 0.20
IOU_THRESHOLD = 0.30

CAMERAS = ("e-CAM", "FLIR", "Axis")
SCALE_NAMES = ("small", "medium", "large")
MODES = ("cls", "box", "full")
EXPERTS = EXPECTED_EXPERT_ORDER
VARIANTS = tuple(f"{e}_{m}" for e in EXPERTS for m in MODES)

# Historical Expert-v0 results are reference-only, never assertions for Final Experts.
HISTORICAL_V0 = {
    "S_full": {"map": 0.7112, "tp": 48931, "fp": 43615, "fn": 15739,
               "rescued": 1133, "lost": 2943, "fp_added": 3059, "fp_removed": 12245},
    "L_full": {"map": 0.7385, "tp": 51201, "fp": 43882, "fn": 13469,
               "rescued": 2071, "lost": 1611, "fp_added": 2214, "fp_removed": 11133},
    "C_full": {"map": 0.6952, "tp": 47740, "fp": 41625, "fn": 16930,
               "rescued": 1737, "lost": 4738, "fp_added": 4879, "fp_removed": 16055},
    "T_full": {"map": 0.7213, "tp": 49619, "fp": 44630, "fn": 15051,
               "rescued": 830, "lost": 1952, "fp_added": 2138, "fp_removed": 10309},
}

DEVICE = torch.device("cpu")


# =============================================================================
# GENERIC HELPERS
# =============================================================================


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_sha(path: Path, expected: str, label: str) -> None:
    if not path.exists():
        raise AssertionError(f"[FATAL] Missing {label}: {path}")
    actual = sha256(path)
    if actual != expected:
        raise AssertionError(
            f"[FATAL] {label} SHA drift:\n"
            f"  expected={expected}\n"
            f"  actual  ={actual}"
        )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def finite_tensor(name: str, x: torch.Tensor) -> None:
    if x.dtype.is_floating_point and not torch.isfinite(x).all():
        raise AssertionError(f"[FATAL] Non-finite tensor: {name}")


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    return safe_div(2.0 * tp, 2.0 * tp + fp + fn)


def precision_from_counts(tp: int, fp: int) -> float:
    return safe_div(tp, tp + fp)


def recall_from_counts(tp: int, fn: int) -> float:
    return safe_div(tp, tp + fn)


def new_event_counts() -> dict:
    return {
        "rescued": 0,
        "lost": 0,
        "retained": 0,
        "fp_added": 0,
        "fp_removed": 0,
    }


def new_base_counts() -> dict:
    return {"gt": 0, "tp": 0, "fp": 0, "fn": 0}


def box_cxcywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = b.unbind(-1)
    return torch.stack(
        [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h],
        dim=-1,
    )


def box_iou_norm_cxcywh(b1: Sequence[float], b2: Sequence[float]) -> float:
    b1_x1 = b1[0] - b1[2] / 2.0
    b1_y1 = b1[1] - b1[3] / 2.0
    b1_x2 = b1[0] + b1[2] / 2.0
    b1_y2 = b1[1] + b1[3] / 2.0

    b2_x1 = b2[0] - b2[2] / 2.0
    b2_y1 = b2[1] - b2[3] / 2.0
    b2_x2 = b2[0] + b2[2] / 2.0
    b2_y2 = b2[1] + b2[3] / 2.0

    ix = max(0.0, min(b1_x2, b2_x2) - max(b1_x1, b2_x1))
    iy = max(0.0, min(b1_y2, b2_y2) - max(b1_y1, b2_y1))
    inter = ix * iy
    union = b1[2] * b1[3] + b2[2] * b2[3] - inter
    return inter / union if union > 0.0 else 0.0


def extract_top100_predictions(logits: torch.Tensor, boxes: torch.Tensor) -> dict:
    """Certified G0/F0-G global flattened Top-100 semantics."""
    if tuple(logits.shape) != (NUM_QUERIES, NUM_CLASSES):
        raise AssertionError(f"[FATAL] logits shape drift: {tuple(logits.shape)}")
    if tuple(boxes.shape) != (NUM_QUERIES, 4):
        raise AssertionError(f"[FATAL] boxes shape drift: {tuple(boxes.shape)}")

    probs = logits.sigmoid()
    flat_probs = probs.flatten()
    scores, flat_idx = torch.topk(
        flat_probs,
        min(TOPK, flat_probs.numel()),
        largest=True,
        sorted=True,
    )
    q_idx = flat_idx // NUM_CLASSES
    c_idx = flat_idx % NUM_CLASSES
    chosen_boxes = boxes[q_idx]
    valid = scores >= SCORE_THRESHOLD

    return {
        "all_scores": scores,
        "all_q_idx": q_idx,
        "all_c_idx": c_idx,
        "all_boxes": chosen_boxes,
        "op_scores": scores[valid],
        "op_q_idx": q_idx[valid],
        "op_c_idx": c_idx[valid],
        "op_boxes": chosen_boxes[valid],
    }


def match_operational_detections_to_gt(
    op_scores: torch.Tensor,
    op_c_idx: torch.Tensor,
    op_boxes_norm: torch.Tensor,
    op_q_idx: torch.Tensor,
    gt_boxes_norm: torch.Tensor,
    gt_labels: torch.Tensor,
) -> Dict[str, Any]:
    """Exact certified greedy same-class operational matcher."""
    num_gt = int(gt_boxes_norm.shape[0])
    num_det = int(op_scores.shape[0])

    matched_gt_indices: Set[int] = set()
    matched_det_indices: Set[int] = set()
    gt_to_det: Dict[int, Tuple[int, float, float, int, int]] = {}

    if num_gt > 0 and num_det > 0:
        gt_boxes_list = gt_boxes_norm.tolist()
        gt_labels_list = gt_labels.tolist()
        det_boxes_list = op_boxes_norm.tolist()
        det_classes_list = op_c_idx.tolist()
        det_scores_list = op_scores.tolist()
        det_q_list = op_q_idx.tolist()

        for d_idx in range(num_det):
            c_idx = det_classes_list[d_idx]
            det_box = det_boxes_list[d_idx]
            score = det_scores_list[d_idx]
            q_idx = det_q_list[d_idx]

            best_iou = -1.0
            best_gt_idx = None

            for g_idx in range(num_gt):
                if g_idx in matched_gt_indices:
                    continue
                if gt_labels_list[g_idx] != c_idx:
                    continue
                iou = box_iou_norm_cxcywh(det_box, gt_boxes_list[g_idx])
                if iou >= IOU_THRESHOLD and iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

            if best_gt_idx is not None:
                matched_gt_indices.add(best_gt_idx)
                matched_det_indices.add(d_idx)
                gt_to_det[best_gt_idx] = (
                    d_idx,
                    best_iou,
                    score,
                    q_idx,
                    c_idx,
                )

    fp_dict = {}
    for d_idx in range(num_det):
        if d_idx not in matched_det_indices:
            key = (int(op_q_idx[d_idx].item()), int(op_c_idx[d_idx].item()))
            fp_dict[key] = {
                "score": float(op_scores[d_idx].item()),
                "box": [float(x) for x in op_boxes_norm[d_idx].tolist()],
            }

    tp_count = len(matched_gt_indices)
    fp_count = len(fp_dict)
    fn_count = num_gt - tp_count

    return {
        "matched_gt_indices": matched_gt_indices,
        "matched_det_indices": matched_det_indices,
        "gt_to_det": gt_to_det,
        "fp_dict": fp_dict,
        "tp": tp_count,
        "fp": fp_count,
        "fn": fn_count,
        "num_det": num_det,
        "num_gt": num_gt,
    }


def gt_scale_name(box_norm: torch.Tensor, image_w: float, image_h: float) -> str:
    area = float(box_norm[2].item()) * image_w * float(box_norm[3].item()) * image_h
    if area < 32.0 ** 2:
        return "small"
    if area < 96.0 ** 2:
        return "medium"
    return "large"


def quiet_finalize_evaluator(ev: CocoEvaluator) -> np.ndarray:
    ev.synchronize_between_processes()
    with contextlib.redirect_stdout(io.StringIO()):
        ev.accumulate()
        ev.summarize()
    return ev.coco_eval["bbox"].stats.copy()


def per_class_ap(ev: CocoEvaluator) -> List[float]:
    precision = ev.coco_eval["bbox"].eval["precision"]
    result = []
    for k in range(NUM_CLASSES):
        p_k = precision[0, :, k, 0, 2]
        valid = p_k[p_k > -1]
        result.append(float(np.mean(valid)) if len(valid) else 0.0)
    return result


# =============================================================================
# PROVENANCE / INVENTORY
# =============================================================================


def parse_folds() -> Dict[int, List[int]]:
    payload = load_json(FOLD_MANIFEST)
    folds = {int(k): [int(v) for v in vals] for k, vals in payload["folds"].items()}
    if sorted(folds) != list(range(NUM_FOLDS)):
        raise AssertionError("[FATAL] Fold keys drift.")
    if [len(folds[f]) for f in range(NUM_FOLDS)] != [10, 10, 9, 9, 9]:
        raise AssertionError("[FATAL] Fold capacities drift.")
    union = sorted(v for vals in folds.values() for v in vals)
    if union != list(range(1, NUM_VIDEOS + 1)):
        raise AssertionError("[FATAL] Fold partition is not exact union 1..47.")
    return folds


def fold_of_video(folds: Mapping[int, Sequence[int]]) -> Dict[int, int]:
    result = {}
    for fold, vids in folds.items():
        for vid in vids:
            if vid in result:
                raise AssertionError(f"[FATAL] Video {vid} appears in multiple folds.")
            result[vid] = fold
    if sorted(result) != list(range(1, NUM_VIDEOS + 1)):
        raise AssertionError("[FATAL] Fold/video mapping incomplete.")
    return result


def canonical_path(vid: int) -> Path:
    return E0 / f"bfcef_e0_cache_vid{vid}.pth"


def oof_path(vid: int, video_to_fold: Mapping[int, int]) -> Path:
    fold = int(video_to_fold[vid])
    return S0_ROOT / f"fold{fold}" / f"bfcef_f_s0a_oof_vid{vid}.pth"


def verify_provenance() -> Tuple[dict, dict, Dict[int, List[int]], Dict[int, int]]:
    print("[*] Verifying RUN_07 + canonical provenance roots...")

    require_sha(OOF_MANIFEST, EXPECTED_OOF_MANIFEST_SHA, "RUN_07 final OOF manifest")
    require_sha(CANONICAL_MANIFEST, EXPECTED_CANONICAL_MANIFEST_SHA, "canonical cache manifest")
    require_sha(FOLD_MANIFEST, EXPECTED_FOLD_MANIFEST_SHA, "5-fold manifest")
    require_sha(FIXED_MATCH_MANIFEST, EXPECTED_FIXED_MATCH_MANIFEST_SHA, "fixed-match manifest")

    oof_manifest = load_json(OOF_MANIFEST)
    can_manifest = load_json(CANONICAL_MANIFEST)
    folds = parse_folds()
    v2f = fold_of_video(folds)

    om = oof_manifest["_meta"]
    expected_meta = {
        "schema_version": EXPECTED_OOF_SCHEMA,
        "objective_version": EXPECTED_OBJECTIVE,
        "trainer_sha256": EXPECTED_RUN07_TRAINER_SHA,
        "r0_checkpoint_sha256": EXPECTED_R0_SHA,
        "canonical_manifest_sha256": EXPECTED_CANONICAL_MANIFEST_SHA,
        "first_order_temporal_manifest_sha256": EXPECTED_FIRST_TEMP_MANIFEST_SHA,
        "spatial_manifest_sha256": EXPECTED_SPATIAL_MANIFEST_SHA,
        "temporal_manifest_sha256": EXPECTED_TEMPORAL_MANIFEST_SHA,
        "fold_manifest_sha256": EXPECTED_FOLD_MANIFEST_SHA,
        "screening_config_sha256": EXPECTED_SCREENING_CONFIG_SHA,
        "fixed_match_manifest_sha256": EXPECTED_FIXED_MATCH_MANIFEST_SHA,
        "num_folds": NUM_FOLDS,
        "num_videos": NUM_VIDEOS,
        "total_oof_frames": EXPECTED_TOTAL_FRAMES,
        "expert_order": list(EXPECTED_EXPERT_ORDER),
        "oof_dtype": "float32",
    }
    for key, value in expected_meta.items():
        if om.get(key) != value:
            raise AssertionError(
                f"[FATAL] RUN_07 OOF manifest metadata drift for {key}: "
                f"{om.get(key)!r} != {value!r}"
            )

    if int(can_manifest["_meta"]["actual_total_frames"]) != EXPECTED_TOTAL_FRAMES:
        raise AssertionError("[FATAL] Canonical manifest frame cardinality drift.")
    if int(can_manifest["_meta"]["num_videos"]) != NUM_VIDEOS:
        raise AssertionError("[FATAL] Canonical manifest video cardinality drift.")
    expected_train_sha = can_manifest["_meta"].get("train_json_sha256")
    if not expected_train_sha or sha256(TRAIN_JSON) != expected_train_sha:
        raise AssertionError("[FATAL] seaclips-train.json SHA is not bound to the canonical manifest.")

    # Canonical shards.
    can_ledger = can_manifest["cache_shard_sha256_ledger"]
    if len(can_ledger) != NUM_VIDEOS:
        raise AssertionError("[FATAL] Canonical shard ledger cardinality drift.")
    for vid in range(1, NUM_VIDEOS + 1):
        p = canonical_path(vid)
        expected = can_ledger.get(p.name)
        if expected is None:
            raise AssertionError(f"[FATAL] Missing canonical ledger entry {p.name}")
        if sha256(p) != expected:
            raise AssertionError(f"[FATAL] Canonical shard SHA drift: {p.name}")

    # RUN_07 checkpoint/certificate/OOF ledgers.
    for ledger_name in (
        "fold_checkpoint_sha256_ledger",
        "fold_certificate_sha256_ledger",
        "oof_shard_sha256_ledger",
    ):
        ledger = oof_manifest[ledger_name]
        expected_count = NUM_FOLDS if ledger_name != "oof_shard_sha256_ledger" else NUM_VIDEOS
        if len(ledger) != expected_count:
            raise AssertionError(
                f"[FATAL] {ledger_name} count {len(ledger)} != {expected_count}"
            )
        for rel, expected in ledger.items():
            p = S0_ROOT / rel
            if not p.exists():
                raise AssertionError(f"[FATAL] Missing RUN_07 artifact: {p}")
            if sha256(p) != expected:
                raise AssertionError(f"[FATAL] RUN_07 artifact SHA drift: {rel}")

    print("[+] Provenance PASS: canonical 47 shards + RUN_07 5 checkpoints + 5 certificates + 47 OOF shards")
    print(f"    OOF manifest SHA: {EXPECTED_OOF_MANIFEST_SHA}")
    print(f"    Fixed-match SHA:  {EXPECTED_FIXED_MATCH_MANIFEST_SHA}")

    return oof_manifest, can_manifest, folds, v2f


# =============================================================================
# OOF RECORD INTEGRITY
# =============================================================================


def baseline_candidate_queries(base_logits: torch.Tensor) -> List[int]:
    flat_idx = torch.topk(
        base_logits.sigmoid().flatten(),
        TOPK,
        largest=True,
        sorted=True,
    ).indices
    q_idx = flat_idx // NUM_CLASSES
    return sorted(set(int(x) for x in q_idx.tolist()))


def verify_oof_record(
    canonical: Mapping[str, Any],
    oof: Mapping[str, Any],
    vid: int,
    idx: int,
) -> None:
    image_id = int(canonical["image_id"])
    if (
        int(canonical["video_id"]) != vid
        or int(canonical["seq_idx"]) != idx
        or int(oof["image_id"]) != image_id
        or int(oof["video_id"]) != vid
        or int(oof["seq_idx"]) != idx
    ):
        raise AssertionError(
            f"[FATAL] Canonical/OOF alignment drift vid={vid}, idx={idx}, image={image_id}"
        )

    base_logits = canonical["d_logits"].float()
    base_boxes = canonical["d_boxes"].float()
    expert_logits = oof["expert_cls_logits"].float()
    expert_boxes = oof["expert_box_boxes"].float()
    l_iou = oof["l_expected_iou"].float()

    if tuple(expert_logits.shape) != (4, NUM_QUERIES, NUM_CLASSES):
        raise AssertionError(
            f"[FATAL] OOF expert_cls_logits shape drift vid={vid}, idx={idx}: {tuple(expert_logits.shape)}"
        )
    if tuple(expert_boxes.shape) != (4, NUM_QUERIES, 4):
        raise AssertionError(
            f"[FATAL] OOF expert_box_boxes shape drift vid={vid}, idx={idx}: {tuple(expert_boxes.shape)}"
        )
    if tuple(l_iou.shape) != (NUM_QUERIES,):
        raise AssertionError(
            f"[FATAL] OOF L expected-IoU shape drift vid={vid}, idx={idx}: {tuple(l_iou.shape)}"
        )

    finite_tensor("OOF expert logits", expert_logits)
    finite_tensor("OOF expert boxes", expert_boxes)
    finite_tensor("OOF L expected IoU", l_iou)

    if torch.any(l_iou < 0.0) or torch.any(l_iou > 1.0):
        raise AssertionError(f"[FATAL] L expected-IoU out of [0,1] vid={vid}, idx={idx}")

    persisted = sorted(int(x) for x in oof["candidate_queries"].tolist())
    recomputed = baseline_candidate_queries(base_logits)
    if persisted != recomputed:
        raise AssertionError(
            f"[FATAL] Candidate-query provenance drift vid={vid}, idx={idx}: "
            f"persisted={persisted}, recomputed={recomputed}"
        )

    mask = torch.zeros(NUM_QUERIES, dtype=torch.bool)
    if persisted:
        mask[torch.tensor(persisted, dtype=torch.long)] = True
    outside = ~mask

    # Exact architecture contract: residuals are masked to candidate query slots.
    for k, expert in enumerate(EXPERTS):
        if not torch.equal(expert_logits[k, outside], base_logits[outside]):
            diff = float((expert_logits[k, outside] - base_logits[outside]).abs().max().item())
            raise AssertionError(
                f"[FATAL] {expert} cls changed outside BCI query set vid={vid}, idx={idx}, maxdiff={diff:.3e}"
            )
        if not torch.equal(expert_boxes[k, outside], base_boxes[outside]):
            diff = float((expert_boxes[k, outside] - base_boxes[outside]).abs().max().item())
            raise AssertionError(
                f"[FATAL] {expert} box changed outside BCI query set vid={vid}, idx={idx}, maxdiff={diff:.3e}"
            )


def run_full_oof_integrity(v2f: Mapping[int, int]) -> Dict[int, int]:
    print("[*] Full 20,075-frame OOF structural / BCI integrity audit...")
    frames_by_video = {}
    total = 0

    for vid in range(1, NUM_VIDEOS + 1):
        can = torch.load(canonical_path(vid), map_location="cpu")
        oof = torch.load(oof_path(vid, v2f), map_location="cpu")
        if len(can) != len(oof):
            raise AssertionError(
                f"[FATAL] Canonical/OOF length mismatch vid={vid}: {len(can)} != {len(oof)}"
            )

        for idx, (c, o) in enumerate(zip(can, oof)):
            verify_oof_record(c, o, vid, idx)

        frames_by_video[vid] = len(can)
        total += len(can)
        if vid % 10 == 0 or vid == NUM_VIDEOS:
            print(f"    [+] OOF integrity videos 1..{vid:02d}/47 PASS")

        del can, oof
        gc.collect()

    if total != EXPECTED_TOTAL_FRAMES:
        raise AssertionError(f"[FATAL] OOF integrity total {total} != {EXPECTED_TOTAL_FRAMES}")

    print(
        "[+] OOF integrity PASS: 47 videos / 20,075 frames; finite tensors; "
        "candidate provenance exact; outside-BCI predictions bitwise baseline"
    )
    return frames_by_video


# =============================================================================
# METADATA / EVALUATOR SETUP
# =============================================================================


def load_metadata():
    if not TRAIN_JSON.exists():
        raise AssertionError(f"[FATAL] Missing Train JSON: {TRAIN_JSON}")

    coco_gt = COCO(str(TRAIN_JSON))
    train = load_json(TRAIN_JSON)

    img_native_size = {
        int(im["id"]): (float(im["width"]), float(im["height"]))
        for im in train["images"]
    }
    img_to_video = {
        int(im["id"]): int(im["video_id"])
        for im in train["images"]
    }

    video_to_camera = {}
    for v in train["videos"]:
        vid = int(v.get("id", v.get("video_id")))
        sensor = str(v["camera_sensor"]).strip()
        if sensor not in CAMERAS:
            raise AssertionError(f"[FATAL] Unknown camera sensor: {sensor}")
        video_to_camera[vid] = sensor

    if set(video_to_camera) != set(range(1, NUM_VIDEOS + 1)):
        raise AssertionError("[FATAL] Train video/camera inventory is not exactly 1..47.")

    cat_ids = sorted(coco_gt.getCatIds())
    if len(cat_ids) != NUM_CLASSES:
        raise AssertionError(f"[FATAL] Expected 7 COCO categories, found {len(cat_ids)}")
    cat_id_to_name = {int(c["id"]): c["name"] for c in train["categories"]}
    class_names = [cat_id_to_name[cid] for cid in cat_ids]
    cat_id_tensor = torch.tensor(cat_ids, dtype=torch.long)

    coco_gt_ca = copy.deepcopy(coco_gt)
    coco_gt_ca.dataset["categories"] = [{"id": 1, "name": "object"}]
    for ann in coco_gt_ca.dataset["annotations"]:
        ann["category_id"] = 1
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt_ca.createIndex()

    return (
        coco_gt,
        coco_gt_ca,
        img_native_size,
        img_to_video,
        video_to_camera,
        cat_id_tensor,
        class_names,
    )


def make_evaluators(coco_gt, coco_gt_ca):
    ev = CocoEvaluator(coco_gt, ["bbox"])
    ev_ca = CocoEvaluator(coco_gt_ca, ["bbox"])
    ev.coco_eval["bbox"].params.iouThrs = np.array([IOU_THRESHOLD])
    ev_ca.coco_eval["bbox"].params.iouThrs = np.array([IOU_THRESHOLD])
    return ev, ev_ca


def add_coco_prediction(
    pred_map: MutableMapping[int, dict],
    pred_map_ca: MutableMapping[int, dict],
    img_id: int,
    pred: Mapping[str, torch.Tensor],
    img_w: float,
    img_h: float,
    cat_id_tensor: torch.Tensor,
) -> None:
    boxes_xyxy_norm = box_cxcywh_to_xyxy(pred["all_boxes"])
    scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=boxes_xyxy_norm.dtype)
    scaled_boxes = boxes_xyxy_norm * scale
    cat_labels = cat_id_tensor[pred["all_c_idx"]]

    pred_map[img_id] = {
        "boxes": scaled_boxes,
        "scores": pred["all_scores"],
        "labels": cat_labels,
    }
    pred_map_ca[img_id] = {
        "boxes": scaled_boxes,
        "scores": pred["all_scores"],
        "labels": torch.ones_like(cat_labels),
    }


# =============================================================================
# BASELINE ORACLE PASS
# =============================================================================


def make_base_strata(class_names: Sequence[str]) -> dict:
    return {
        "by_fold": {str(f): new_base_counts() for f in range(NUM_FOLDS)},
        "by_camera": {c: new_base_counts() for c in CAMERAS},
        "by_class": {str(c): {**new_base_counts(), "name": class_names[c]} for c in range(NUM_CLASSES)},
        "by_scale": {s: new_base_counts() for s in SCALE_NAMES},
        "by_video": {str(v): new_base_counts() for v in range(1, NUM_VIDEOS + 1)},
    }


def evaluate_baseline(
    folds: Mapping[int, Sequence[int]],
    v2f: Mapping[int, int],
    coco_gt,
    coco_gt_ca,
    img_native_size: Mapping[int, Tuple[float, float]],
    img_to_video: Mapping[int, int],
    video_to_camera: Mapping[int, str],
    cat_id_tensor: torch.Tensor,
    class_names: Sequence[str],
):
    print("=" * 118)
    print("PASS A -- FROZEN R0 BASELINE ORACLE IDENTITY")
    print("=" * 118)

    ev, ev_ca = make_evaluators(coco_gt, coco_gt_ca)
    base_cache: Dict[int, dict] = {}
    strata = make_base_strata(class_names)

    total_tp = total_fp = total_fn = total_gt = 0
    seen_images: Set[int] = set()

    for vid in range(1, NUM_VIDEOS + 1):
        fold = int(v2f[vid])
        camera = video_to_camera[vid]
        records = torch.load(canonical_path(vid), map_location="cpu")
        v_preds = {}
        v_preds_ca = {}

        for idx, fr in enumerate(records):
            img_id = int(fr["image_id"])
            if img_id in seen_images:
                raise AssertionError(f"[FATAL] Duplicate canonical image_id {img_id}")
            seen_images.add(img_id)
            if int(fr["video_id"]) != vid or int(fr["seq_idx"]) != idx:
                raise AssertionError(f"[FATAL] Canonical identity drift vid={vid}, idx={idx}")
            if img_to_video.get(img_id) != vid:
                raise AssertionError(f"[FATAL] Train JSON video mismatch image={img_id}")

            z = fr["d_logits"].float()
            b = fr["d_boxes"].float()
            gt_boxes = fr["gt_boxes_norm"].float()
            gt_labels = fr["gt_labels"].long()
            pred = extract_top100_predictions(z, b)

            img_w, img_h = img_native_size[img_id]
            add_coco_prediction(
                v_preds,
                v_preds_ca,
                img_id,
                pred,
                img_w,
                img_h,
                cat_id_tensor,
            )

            m = match_operational_detections_to_gt(
                pred["op_scores"],
                pred["op_c_idx"],
                pred["op_boxes"],
                pred["op_q_idx"],
                gt_boxes,
                gt_labels,
            )

            num_gt = int(gt_labels.numel())
            total_gt += num_gt
            total_tp += m["tp"]
            total_fp += m["fp"]
            total_fn += m["fn"]

            frame_scales = [
                gt_scale_name(gt_boxes[g], img_w, img_h)
                for g in range(num_gt)
            ]

            # Frame-level baseline cache required by all 12 counterfactual passes.
            base_cache[img_id] = {
                "matched_gt_indices": m["matched_gt_indices"],
                "gt_to_det": m["gt_to_det"],
                "fp_keys": set(m["fp_dict"].keys()),
                "tp": m["tp"],
                "fp": m["fp"],
                "fn": m["fn"],
                "num_gt": num_gt,
                "gt_scales": frame_scales,
                "fold": fold,
                "camera": camera,
                "video_id": vid,
            }

            for key in (
                strata["by_fold"][str(fold)],
                strata["by_camera"][camera],
                strata["by_video"][str(vid)],
            ):
                key["gt"] += num_gt
                key["tp"] += m["tp"]
                key["fp"] += m["fp"]
                key["fn"] += m["fn"]

            matched = m["matched_gt_indices"]
            for g in range(num_gt):
                cls = int(gt_labels[g].item())
                scale_name = frame_scales[g]
                for bucket in (
                    strata["by_class"][str(cls)],
                    strata["by_scale"][scale_name],
                ):
                    bucket["gt"] += 1
                    if g in matched:
                        bucket["tp"] += 1
                    else:
                        bucket["fn"] += 1

        ev.update(v_preds)
        ev_ca.update(v_preds_ca)
        del records, v_preds, v_preds_ca
        gc.collect()

    if len(seen_images) != EXPECTED_TOTAL_FRAMES:
        raise AssertionError(
            f"[FATAL] Baseline processed {len(seen_images)} images != {EXPECTED_TOTAL_FRAMES}"
        )

    stats = quiet_finalize_evaluator(ev)
    stats_ca = quiet_finalize_evaluator(ev_ca)
    map03 = float(stats[0])
    mapca03 = float(stats_ca[0])
    pclass = per_class_ap(ev)
    f1 = f1_from_counts(total_tp, total_fp, total_fn)

    checks = [
        ("GT", total_gt, EXPECTED_TOTAL_GT),
        ("TP", total_tp, CERTIFIED_BASE_TP),
        ("FP", total_fp, CERTIFIED_BASE_FP),
        ("FN", total_fn, CERTIFIED_BASE_FN),
    ]
    for label, actual, expected in checks:
        if actual != expected:
            raise AssertionError(f"[FATAL] Baseline {label} {actual} != frozen {expected}")
    if abs(f1 - CERTIFIED_BASE_F1) > 1e-12:
        raise AssertionError(
            f"[FATAL] Baseline F1 drift {f1:.12f} != {CERTIFIED_BASE_F1:.12f}"
        )
    if abs(map03 - CERTIFIED_BASE_MAP) > 1e-4:
        raise AssertionError(
            f"[FATAL] Baseline mAP@.30 drift {map03:.6f} != {CERTIFIED_BASE_MAP:.6f}"
        )

    print("[PASS] CANONICAL BASELINE ORACLE IDENTITY")
    print(
        f"    GT={total_gt:,} TP={total_tp:,} FP={total_fp:,} FN={total_fn:,} "
        f"F1={f1:.10f} mAP@.30={map03:.6f} mAPca@.30={mapca03:.6f}"
    )

    baseline = {
        "gt": total_gt,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision_from_counts(total_tp, total_fp),
        "recall": recall_from_counts(total_tp, total_fn),
        "f1": f1,
        "map30": map03,
        "mapca30": mapca03,
        "ap_s30": float(stats[3]),
        "ap_m30": float(stats[4]),
        "ap_l30": float(stats[5]),
        "per_class_ap30": {str(i): pclass[i] for i in range(NUM_CLASSES)},
        "strata": strata,
    }

    del ev, ev_ca
    gc.collect()
    return baseline, base_cache


# =============================================================================
# VARIANT COUNTERFACTUAL PASS
# =============================================================================


def make_transition_record(class_names: Sequence[str]) -> dict:
    return {
        **new_event_counts(),
        "by_fold": {str(f): new_event_counts() for f in range(NUM_FOLDS)},
        "by_camera": {c: new_event_counts() for c in CAMERAS},
        "by_class": {str(c): {**new_event_counts(), "name": class_names[c]} for c in range(NUM_CLASSES)},
        "by_scale": {s: new_event_counts() for s in SCALE_NAMES},
        "by_video": {str(v): new_event_counts() for v in range(1, NUM_VIDEOS + 1)},
        "delta_iou_retained": [],
        "delta_score_retained": [],
    }


def select_variant_tensors(
    variant: str,
    base_logits: torch.Tensor,
    base_boxes: torch.Tensor,
    oof_record: Mapping[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    expert, mode = variant.split("_", 1)
    k = EXPERTS.index(expert)
    z_exp = oof_record["expert_cls_logits"][k].float()
    b_exp = oof_record["expert_box_boxes"][k].float()

    if mode == "cls":
        return z_exp, base_boxes
    if mode == "box":
        return base_logits, b_exp
    if mode == "full":
        return z_exp, b_exp
    raise AssertionError(f"[FATAL] Unknown variant mode: {variant}")


def summarize_continuous(values: Sequence[float]) -> dict:
    if not values:
        return {"n": 0, "mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def evaluate_variant(
    variant: str,
    v2f: Mapping[int, int],
    coco_gt,
    coco_gt_ca,
    img_native_size: Mapping[int, Tuple[float, float]],
    video_to_camera: Mapping[int, str],
    cat_id_tensor: torch.Tensor,
    class_names: Sequence[str],
    base_cache: Mapping[int, dict],
    baseline: Mapping[str, Any],
):
    print("=" * 118)
    print(f"EVALUATING FINAL-EXPERT OOF VARIANT: {variant}")
    print("=" * 118)
    t0 = time.time()

    ev, ev_ca = make_evaluators(coco_gt, coco_gt_ca)
    tr = make_transition_record(class_names)

    total_tp = total_fp = total_fn = 0
    op_by_fold = {str(f): {"tp": 0, "fp": 0, "fn": 0} for f in range(NUM_FOLDS)}
    op_by_video = {str(v): {"tp": 0, "fp": 0, "fn": 0} for v in range(1, NUM_VIDEOS + 1)}

    rescue_ids: Set[Tuple[int, int]] = set()
    loss_ids: Set[Tuple[int, int]] = set()

    for vid in range(1, NUM_VIDEOS + 1):
        fold = int(v2f[vid])
        camera = video_to_camera[vid]
        can = torch.load(canonical_path(vid), map_location="cpu")
        oof = torch.load(oof_path(vid, v2f), map_location="cpu")
        if len(can) != len(oof):
            raise AssertionError(f"[FATAL] Variant alignment length mismatch vid={vid}")

        v_preds = {}
        v_preds_ca = {}

        for idx, (fr, oo) in enumerate(zip(can, oof)):
            img_id = int(fr["image_id"])
            base_logits = fr["d_logits"].float()
            base_boxes = fr["d_boxes"].float()
            gt_boxes = fr["gt_boxes_norm"].float()
            gt_labels = fr["gt_labels"].long()

            z, b = select_variant_tensors(variant, base_logits, base_boxes, oo)
            pred = extract_top100_predictions(z, b)

            img_w, img_h = img_native_size[img_id]
            add_coco_prediction(
                v_preds,
                v_preds_ca,
                img_id,
                pred,
                img_w,
                img_h,
                cat_id_tensor,
            )

            m = match_operational_detections_to_gt(
                pred["op_scores"],
                pred["op_c_idx"],
                pred["op_boxes"],
                pred["op_q_idx"],
                gt_boxes,
                gt_labels,
            )

            total_tp += m["tp"]
            total_fp += m["fp"]
            total_fn += m["fn"]
            for bucket in (op_by_fold[str(fold)], op_by_video[str(vid)]):
                bucket["tp"] += m["tp"]
                bucket["fp"] += m["fp"]
                bucket["fn"] += m["fn"]

            base = base_cache[img_id]
            base_matched = base["matched_gt_indices"]
            exp_matched = m["matched_gt_indices"]
            base_gt_to_det = base["gt_to_det"]
            exp_gt_to_det = m["gt_to_det"]
            frame_scales = base["gt_scales"]
            num_gt = int(gt_labels.numel())

            for g in range(num_gt):
                cls = int(gt_labels[g].item())
                scale_name = frame_scales[g]
                is_base = g in base_matched
                is_exp = g in exp_matched

                buckets = (
                    tr,
                    tr["by_fold"][str(fold)],
                    tr["by_camera"][camera],
                    tr["by_class"][str(cls)],
                    tr["by_scale"][scale_name],
                    tr["by_video"][str(vid)],
                )

                if (not is_base) and is_exp:
                    for bucket in buckets:
                        bucket["rescued"] += 1
                    rescue_ids.add((img_id, g))

                elif is_base and (not is_exp):
                    for bucket in buckets:
                        bucket["lost"] += 1
                    loss_ids.add((img_id, g))

                elif is_base and is_exp:
                    for bucket in buckets:
                        bucket["retained"] += 1
                    b_d = base_gt_to_det[g]
                    e_d = exp_gt_to_det[g]
                    tr["delta_iou_retained"].append(float(e_d[1] - b_d[1]))
                    tr["delta_score_retained"].append(float(e_d[2] - b_d[2]))

            base_fp_keys = base["fp_keys"]
            exp_fp_keys = set(m["fp_dict"].keys())
            removed = base_fp_keys - exp_fp_keys
            added = exp_fp_keys - base_fp_keys

            for bucket in (
                tr,
                tr["by_fold"][str(fold)],
                tr["by_camera"][camera],
                tr["by_video"][str(vid)],
            ):
                bucket["fp_removed"] += len(removed)
                bucket["fp_added"] += len(added)

            for q_idx, c_idx in removed:
                tr["by_class"][str(int(c_idx))]["fp_removed"] += 1
            for q_idx, c_idx in added:
                tr["by_class"][str(int(c_idx))]["fp_added"] += 1

        ev.update(v_preds)
        ev_ca.update(v_preds_ca)

        if vid % 10 == 0 or vid == NUM_VIDEOS:
            print(f"    [+] {variant}: videos 1..{vid:02d}/47 processed")

        del can, oof, v_preds, v_preds_ca
        gc.collect()

    stats = quiet_finalize_evaluator(ev)
    stats_ca = quiet_finalize_evaluator(ev_ca)
    pclass = per_class_ap(ev)

    # Global conservation invariants: identical to certified G0 event semantics.
    if tr["retained"] + tr["lost"] != CERTIFIED_BASE_TP:
        raise AssertionError(
            f"[FATAL] {variant}: retained+lost != Base TP: "
            f"{tr['retained']}+{tr['lost']} != {CERTIFIED_BASE_TP}"
        )
    if tr["retained"] + tr["rescued"] != total_tp:
        raise AssertionError(
            f"[FATAL] {variant}: retained+rescued != variant TP"
        )
    if tr["rescued"] - tr["lost"] != total_tp - CERTIFIED_BASE_TP:
        raise AssertionError(
            f"[FATAL] {variant}: rescue-loss conservation failed"
        )
    if tr["fp_added"] - tr["fp_removed"] != total_fp - CERTIFIED_BASE_FP:
        raise AssertionError(
            f"[FATAL] {variant}: FP add-remove conservation failed"
        )

    # Stronger per-fold / per-video conservation using the already-certified
    # baseline strata from Pass A (no redundant cache rereads).
    for fold in range(NUM_FOLDS):
        key = str(fold)
        base_fold = baseline["strata"]["by_fold"][key]
        tf = tr["by_fold"][key]
        of = op_by_fold[key]
        if tf["rescued"] - tf["lost"] != of["tp"] - base_fold["tp"]:
            raise AssertionError(f"[FATAL] {variant}: fold {fold} TP conservation failed")
        if tf["fp_added"] - tf["fp_removed"] != of["fp"] - base_fold["fp"]:
            raise AssertionError(f"[FATAL] {variant}: fold {fold} FP conservation failed")

    for vid in range(1, NUM_VIDEOS + 1):
        key = str(vid)
        base_video = baseline["strata"]["by_video"][key]
        tv = tr["by_video"][key]
        ov = op_by_video[key]
        if tv["rescued"] - tv["lost"] != ov["tp"] - base_video["tp"]:
            raise AssertionError(f"[FATAL] {variant}: video {vid} TP conservation failed")
        if tv["fp_added"] - tv["fp_removed"] != ov["fp"] - base_video["fp"]:
            raise AssertionError(f"[FATAL] {variant}: video {vid} FP conservation failed")

    # Convert raw continuous arrays into compact summaries before serialization.
    delta_iou_summary = summarize_continuous(tr.pop("delta_iou_retained"))
    delta_score_summary = summarize_continuous(tr.pop("delta_score_retained"))

    metrics = {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision_from_counts(total_tp, total_fp),
        "recall": recall_from_counts(total_tp, total_fn),
        "f1": f1_from_counts(total_tp, total_fp, total_fn),
        "map30": float(stats[0]),
        "mapca30": float(stats_ca[0]),
        "ap_s30": float(stats[3]),
        "ap_m30": float(stats[4]),
        "ap_l30": float(stats[5]),
        "per_class_ap30": {str(i): pclass[i] for i in range(NUM_CLASSES)},
        "by_fold_operational": op_by_fold,
        "by_video_operational": op_by_video,
    }

    transition = {
        **tr,
        "net_tp": tr["rescued"] - tr["lost"],
        "net_fp": tr["fp_added"] - tr["fp_removed"],
        "protection_loss_rate": safe_div(tr["lost"], CERTIFIED_BASE_TP),
        "retained_delta_iou": delta_iou_summary,
        "retained_delta_score": delta_score_summary,
    }

    print(
        f"[+] {variant} COMPLETE: mAP={metrics['map30']:.4f}, "
        f"TP={total_tp:,}, FP={total_fp:,}, FN={total_fn:,}, F1={metrics['f1']:.4f}; "
        f"rescue={tr['rescued']:,}, loss={tr['lost']:,}, "
        f"FPadd={tr['fp_added']:,}, FPremove={tr['fp_removed']:,}; "
        f"runtime={(time.time()-t0)/60.0:.2f} min"
    )

    del ev, ev_ca
    gc.collect()

    return metrics, transition, rescue_ids, loss_ids


# =============================================================================
# COMPLEMENTARITY
# =============================================================================


def pairwise_overlap(sets_by_expert: Mapping[str, Set[Tuple[int, int]]]) -> dict:
    result = {}
    for i, a in enumerate(EXPERTS):
        for b in EXPERTS[i + 1:]:
            sa = sets_by_expert[a]
            sb = sets_by_expert[b]
            inter = len(sa & sb)
            union = len(sa | sb)
            result[f"{a}-{b}"] = {
                "intersection": inter,
                "union": union,
                "jaccard": safe_div(inter, union),
            }
    return result


def complementarity_summary(
    event_sets: Mapping[str, Set[Tuple[int, int]]],
    mode: str,
) -> dict:
    sets_by_expert = {e: event_sets[f"{e}_{mode}"] for e in EXPERTS}
    union = set().union(*sets_by_expert.values())

    counter = Counter()
    unique_by_expert = {e: 0 for e in EXPERTS}
    for item in union:
        owners = [e for e in EXPERTS if item in sets_by_expert[e]]
        counter[len(owners)] += 1
        if len(owners) == 1:
            unique_by_expert[owners[0]] += 1

    return {
        "mode": mode,
        "union_count": len(union),
        "by_expert_count": {e: len(sets_by_expert[e]) for e in EXPERTS},
        "unique_by_expert": unique_by_expert,
        "exactly_n_experts": {str(n): int(counter.get(n, 0)) for n in range(1, 5)},
        "fraction_exactly_one": safe_div(counter.get(1, 0), len(union)),
        "pairwise": pairwise_overlap(sets_by_expert),
    }


# =============================================================================
# REPORTING
# =============================================================================


def print_global_table(baseline: Mapping[str, Any], metrics: Mapping[str, Mapping[str, Any]]) -> None:
    print("\n" + "=" * 138)
    print("RUN_08 TABLE 1 -- GLOBAL FINAL-EXPERT OOF COUNTERFACTUAL METRICS")
    print("=" * 138)
    print(
        f"{'Variant':<10} | {'mAP@.3':>7} | {'DmAP':>7} | {'TP':>7} | {'FP':>7} | {'FN':>7} | "
        f"{'Prec':>7} | {'Recall':>7} | {'F1':>7} | {'DF1':>7} | {'AP_S':>7} | {'AP_M':>7} | {'AP_L':>7}"
    )
    print("-" * 138)

    rows = [("Base", baseline)] + [(v, metrics[v]) for v in VARIANTS]
    for name, m in rows:
        dmap = float(m["map30"]) - float(baseline["map30"])
        df1 = float(m["f1"]) - float(baseline["f1"])
        print(
            f"{name:<10} | {m['map30']:>7.4f} | {dmap:>+7.4f} | {m['tp']:>7,} | {m['fp']:>7,} | {m['fn']:>7,} | "
            f"{m['precision']:>7.3f} | {m['recall']:>7.3f} | {m['f1']:>7.4f} | {df1:>+7.4f} | "
            f"{m['ap_s30']:>7.4f} | {m['ap_m30']:>7.4f} | {m['ap_l30']:>7.4f}"
        )
    print("=" * 138)


def print_transition_table(transitions: Mapping[str, Mapping[str, Any]]) -> None:
    print("\n" + "=" * 112)
    print("RUN_08 TABLE 2 -- GLOBAL GT/FP OUTCOME TRANSITIONS")
    print("=" * 112)
    print(
        f"{'Variant':<10} | {'Rescue':>7} | {'Lost':>7} | {'NetTP':>7} | {'Retained':>9} | "
        f"{'FPadd':>7} | {'FPrem':>7} | {'NetFP':>7} | {'LossRate':>8}"
    )
    print("-" * 112)
    for v in VARIANTS:
        t = transitions[v]
        print(
            f"{v:<10} | {t['rescued']:>7,} | {t['lost']:>7,} | {t['net_tp']:>+7,} | {t['retained']:>9,} | "
            f"{t['fp_added']:>7,} | {t['fp_removed']:>7,} | {t['net_fp']:>+7,} | {t['protection_loss_rate']:>8.2%}"
        )
    print("=" * 112)


def print_complementarity_table(complementarity: Mapping[str, Any]) -> None:
    print("\n" + "=" * 110)
    print("RUN_08 TABLE 3 -- EXPERT COMPLEMENTARITY (RESCUED BASELINE FNs)")
    print("=" * 110)
    print(
        f"{'Mode':<8} | {'Union':>7} | {'Unique S':>8} | {'Unique L':>8} | {'Unique C':>8} | {'Unique T':>8} | "
        f"{'Exactly1':>8} | {'Frac1':>7}"
    )
    print("-" * 110)
    for mode in MODES:
        c = complementarity["rescues"][mode]
        print(
            f"{mode:<8} | {c['union_count']:>7,} | {c['unique_by_expert']['S']:>8,} | "
            f"{c['unique_by_expert']['L']:>8,} | {c['unique_by_expert']['C']:>8,} | "
            f"{c['unique_by_expert']['T']:>8,} | {c['exactly_n_experts']['1']:>8,} | "
            f"{c['fraction_exactly_one']:>7.2%}"
        )
    print("=" * 110)


def write_global_csv(path: Path, baseline: Mapping[str, Any], metrics: Mapping[str, Mapping[str, Any]], transitions):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "variant", "map30", "delta_map30", "mapca30", "ap_s30", "ap_m30", "ap_l30",
        "tp", "fp", "fn", "precision", "recall", "f1", "delta_f1",
        "rescued", "lost", "net_tp", "retained", "fp_added", "fp_removed", "net_fp",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "variant": "Base",
            "map30": baseline["map30"],
            "delta_map30": 0.0,
            "mapca30": baseline["mapca30"],
            "ap_s30": baseline["ap_s30"],
            "ap_m30": baseline["ap_m30"],
            "ap_l30": baseline["ap_l30"],
            "tp": baseline["tp"], "fp": baseline["fp"], "fn": baseline["fn"],
            "precision": baseline["precision"], "recall": baseline["recall"],
            "f1": baseline["f1"], "delta_f1": 0.0,
            "rescued": "", "lost": "", "net_tp": "", "retained": "",
            "fp_added": "", "fp_removed": "", "net_fp": "",
        })
        for v in VARIANTS:
            m = metrics[v]
            t = transitions[v]
            writer.writerow({
                "variant": v,
                "map30": m["map30"],
                "delta_map30": m["map30"] - baseline["map30"],
                "mapca30": m["mapca30"],
                "ap_s30": m["ap_s30"],
                "ap_m30": m["ap_m30"],
                "ap_l30": m["ap_l30"],
                "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
                "precision": m["precision"], "recall": m["recall"],
                "f1": m["f1"], "delta_f1": m["f1"] - baseline["f1"],
                "rescued": t["rescued"], "lost": t["lost"], "net_tp": t["net_tp"],
                "retained": t["retained"], "fp_added": t["fp_added"],
                "fp_removed": t["fp_removed"], "net_fp": t["net_fp"],
            })


# =============================================================================
# SELF-TEST
# =============================================================================


def self_test() -> None:
    # Two GTs, three detections: verify class-aware greedy matching and FP identity.
    gt_boxes = torch.tensor([
        [0.25, 0.25, 0.20, 0.20],
        [0.75, 0.75, 0.20, 0.20],
    ], dtype=torch.float32)
    gt_labels = torch.tensor([0, 1], dtype=torch.long)
    scores = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)
    classes = torch.tensor([0, 1, 0], dtype=torch.long)
    boxes = torch.tensor([
        [0.25, 0.25, 0.20, 0.20],
        [0.75, 0.75, 0.20, 0.20],
        [0.50, 0.50, 0.10, 0.10],
    ], dtype=torch.float32)
    queries = torch.tensor([1, 2, 3], dtype=torch.long)
    m = match_operational_detections_to_gt(scores, classes, boxes, queries, gt_boxes, gt_labels)
    assert m["tp"] == 2 and m["fp"] == 1 and m["fn"] == 0
    assert set(m["fp_dict"]) == {(3, 0)}

    # Complementarity sanity.
    dummy = {
        "S_full": {(1, 0), (2, 0)},
        "L_full": {(2, 0), (3, 0)},
        "C_full": {(4, 0)},
        "T_full": set(),
    }
    c = complementarity_summary(dummy, "full")
    assert c["union_count"] == 4
    assert c["exactly_n_experts"]["1"] == 3
    assert c["exactly_n_experts"]["2"] == 1
    print("[PASS] RUN_08 INTERNAL SELF-TEST")


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 118)
    print("B-FCEF-F RUN_08 -- FINAL-EXPERT OOF COUNTERFACTUAL AUDIT")
    print("READ ONLY | TRAIN ONLY | NO MODEL INFERENCE | NO U/H COMPILATION | NO VAL/TEST")
    print("=" * 118)
    print(f"Driver SHA-256: {sha256(Path(__file__))}")
    print(f"RUN_07 OOF root: {OOF_MANIFEST}")

    self_test()

    oof_manifest, can_manifest, folds, v2f = verify_provenance()
    run_full_oof_integrity(v2f)

    (
        coco_gt,
        coco_gt_ca,
        img_native_size,
        img_to_video,
        video_to_camera,
        cat_id_tensor,
        class_names,
    ) = load_metadata()

    baseline, base_cache = evaluate_baseline(
        folds,
        v2f,
        coco_gt,
        coco_gt_ca,
        img_native_size,
        img_to_video,
        video_to_camera,
        cat_id_tensor,
        class_names,
    )

    metrics = {}
    transitions = {}
    rescue_sets = {}
    loss_sets = {}

    for variant in VARIANTS:
        m, t, rescues, losses = evaluate_variant(
            variant,
            v2f,
            coco_gt,
            coco_gt_ca,
            img_native_size,
            video_to_camera,
            cat_id_tensor,
            class_names,
            base_cache,
            baseline,
        )
        metrics[variant] = m
        transitions[variant] = t
        rescue_sets[variant] = rescues
        loss_sets[variant] = losses

    complementarity = {
        "rescues": {mode: complementarity_summary(rescue_sets, mode) for mode in MODES},
        "losses": {mode: complementarity_summary(loss_sets, mode) for mode in MODES},
    }

    print_global_table(baseline, metrics)
    print_transition_table(transitions)
    print_complementarity_table(complementarity)

    report_path = OUT / "run08_final_expert_oof_audit_report.json"
    csv_path = OUT / "run08_global_metrics.csv"
    comp_path = OUT / "run08_complementarity.json"

    report = {
        "_meta": {
            "schema_version": "bfcef_f_run08_final_expert_oof_audit_v1",
            "driver_sha256": sha256(Path(__file__)),
            "run07_oof_manifest_sha256": EXPECTED_OOF_MANIFEST_SHA,
            "run07_trainer_sha256": EXPECTED_RUN07_TRAINER_SHA,
            "canonical_manifest_sha256": EXPECTED_CANONICAL_MANIFEST_SHA,
            "fold_manifest_sha256": EXPECTED_FOLD_MANIFEST_SHA,
            "fixed_match_manifest_sha256": EXPECTED_FIXED_MATCH_MANIFEST_SHA,
            "total_videos": NUM_VIDEOS,
            "total_frames": EXPECTED_TOTAL_FRAMES,
            "total_gt": EXPECTED_TOTAL_GT,
            "data_scope": "Train only; no Val/Test paths are referenced",
            "evaluator_semantics": {
                "ranking": "sigmoid logits -> flattened 300x7 -> global Top-100",
                "coco_predictions": "all Top-100",
                "operational_score_threshold": SCORE_THRESHOLD,
                "operational_match": "same-class greedy",
                "operational_iou_threshold": IOU_THRESHOLD,
                "fp_identity": "(q_idx, class_idx)",
            },
            "variants": list(VARIANTS),
            "runtime_minutes": (time.time() - t_start) / 60.0,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "class_names": {str(i): class_names[i] for i in range(NUM_CLASSES)},
        "baseline": baseline,
        "final_expert_variants": metrics,
        "transitions": transitions,
        "complementarity": complementarity,
        "historical_expert_v0_reference_only": HISTORICAL_V0,
    }
    save_json(report_path, report)
    save_json(comp_path, complementarity)
    write_global_csv(csv_path, baseline, metrics, transitions)

    print("\n" + "=" * 118)
    print("[PASS] RUN_08 FINAL-EXPERT OOF COUNTERFACTUAL AUDIT COMPLETE")
    print(f"[+] Report:          {report_path}")
    print(f"[+] Report SHA:      {sha256(report_path)}")
    print(f"[+] Global CSV:      {csv_path}")
    print(f"[+] CSV SHA:         {sha256(csv_path)}")
    print(f"[+] Complementarity: {comp_path}")
    print(f"[+] Complement SHA:  {sha256(comp_path)}")
    print(f"[+] Runtime:         {(time.time() - t_start) / 60.0:.2f} min")
    print("[+] FINAL U/H LABEL COMPILATION HAS NOT BEEN STARTED.")
    print("=" * 118)


if __name__ == "__main__":
    main()
