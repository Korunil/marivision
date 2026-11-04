from __future__ import annotations

"""
B-FCEF-F RUN_09A
================
Final OOF Local-Counterfactual Consequence Compiler.

Purpose
-------
Compile the exact Final-Expert local marginal consequences required to train
the permanent B-FCEF-F self-assessment -> U/H -> task Gates -> Router stack.

This compiler uses ONLY:
  * frozen canonical R0 cache
  * RUN_07 honest Final-Expert OOF outputs
  * frozen five-fold partition
  * frozen RUN_08 provenance roots

It performs NO training and NO Val/Test access.

Classification local intervention
---------------------------------
For expert k and candidate query q:

    B + k_cls^(q)

replaces ONLY q's seven logits with the Final OOF expert logits. All other
query logits and all 300 boxes remain Base. For each class c, the exact
transition

    {TP, FP, NONOP}_Base -> {TP, FP, NONOP}_Local

is preserved as one of 9 row-major states.

Box local intervention
----------------------
For expert k and candidate query q:

    B + k_box^(q)

replaces ONLY q's box. Scores/order remain Base. Because DETR query q shares
one physical box across classes, all q-class operational hypotheses are
examined and complete Base/After match identities are retained.

Efficiency without semantic approximation
-----------------------------------------
The production compiler uses exact local identities but avoids repeatedly
replaying all 300x7 scores:
  * cls: exact Top-100 is reconstructed from Base Top-107 + q's seven new scores
  * box: Base Top-100 ordering is unchanged
  * only the target q-class matching prefixes are evaluated

A deterministic brute-force preflight compares the optimized result against
the original full F0-L replay semantics before full compilation.

Outputs
-------
bfcef_f_s0/run09_final_local_consequences_v1/
  fold{f}/bfcef_f_run09_consequence_vid{vid}.pth
  run09_final_consequence_manifest.json
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import torch


# =============================================================================
# PATHS / ROOTS
# =============================================================================

ROOT = Path("D:/MARIVISION")
EVAL = ROOT / "H0_workbench" / "eval_tools"
E0 = EVAL / "e0_cache"
FOLD_DIR = EVAL / "e0_adapters"
S0A = EVAL / "bfcef_f_s0" / "final_expert_crossfit_v1"
RUN08 = EVAL / "bfcef_f_s0" / "run08_final_expert_oof_audit_v1"
OUT = EVAL / "bfcef_f_s0" / "run09_final_local_consequences_v1"

CANONICAL_MANIFEST = E0 / "bfcef_e0_cache_manifest.json"
FOLD_MANIFEST = FOLD_DIR / "bfcef_e0_5fold_manifest.json"
OOF_MANIFEST = S0A / "final_expert_oof_manifest.json"
RUN08_REPORT = RUN08 / "run08_final_expert_oof_audit_report.json"

EXPECTED_CANONICAL_SHA = "210ad12de4ebe02cdbe000956972955ae58ceb8f0d222b703787f33a49af03cd"
EXPECTED_FOLD_SHA = "c923f8abd5833996331566582f568b2b0e9ad1d7d17f606c8ddae693b6b8a103"
EXPECTED_OOF_SHA = "1c2569f1871b475e4aa71d20e505c8bb738d9b3ed7d4df185787c1c5546c6001"
EXPECTED_RUN08_SHA = "b91dab20f426dbc36189e6127202059811091ed1b2abfd13c0fd8a86b000c5ff"

SCHEMA = "bfcef_f_run09_final_local_consequence_v1"
NUM_VIDEOS = 47
NUM_FOLDS = 5
NUM_QUERIES = 300
NUM_CLASSES = 7
TOTAL_FRAMES = 20_075
TOPK = 100
TOPK_MARGIN = TOPK + NUM_CLASSES
SCORE_THRESHOLD = 0.20
IOU_THRESHOLD = 0.30
EXPERTS = ("S", "L", "C", "T")

TP = 0
FP = 1
NONOP = 2

BENEFICIAL_CLS = {3, 5, 6}  # FP->TP, FP->NONOP, NONOP->TP
HARMFUL_CLS = {1, 2, 7}     # TP->FP, TP->NONOP, NONOP->FP


# =============================================================================
# HELPERS
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
            f"[FATAL] {label} SHA drift:\n  expected={expected}\n  actual  ={actual}"
        )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def box_iou_cxcywh(a: Sequence[float], b: Sequence[float]) -> float:
    ax1 = a[0] - 0.5 * a[2]
    ay1 = a[1] - 0.5 * a[3]
    ax2 = a[0] + 0.5 * a[2]
    ay2 = a[1] + 0.5 * a[3]

    bx1 = b[0] - 0.5 * b[2]
    by1 = b[1] - 0.5 * b[3]
    bx2 = b[0] + 0.5 * b[2]
    by2 = b[1] + 0.5 * b[3]

    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0.0 else 0.0


def parse_folds() -> Dict[int, List[int]]:
    payload = load_json(FOLD_MANIFEST)
    folds = {int(k): [int(v) for v in vals] for k, vals in payload["folds"].items()}
    if sorted(folds) != list(range(NUM_FOLDS)):
        raise AssertionError("[FATAL] Fold keys drift.")
    if [len(folds[i]) for i in range(NUM_FOLDS)] != [10, 10, 9, 9, 9]:
        raise AssertionError("[FATAL] Fold capacities drift.")
    if sorted(v for vals in folds.values() for v in vals) != list(range(1, 48)):
        raise AssertionError("[FATAL] Fold partition drift.")
    return folds


def video_to_fold_map(folds: Dict[int, List[int]]) -> Dict[int, int]:
    result = {}
    for f, vids in folds.items():
        for v in vids:
            if v in result:
                raise AssertionError("[FATAL] Duplicate video fold assignment.")
            result[v] = f
    return result


def canonical_path(vid: int) -> Path:
    return E0 / f"bfcef_e0_cache_vid{vid}.pth"


def oof_path(fold: int, vid: int) -> Path:
    return S0A / f"fold{fold}" / f"bfcef_f_s0a_oof_vid{vid}.pth"


def out_shard_path(fold: int, vid: int) -> Path:
    return OUT / f"fold{fold}" / f"bfcef_f_run09_consequence_vid{vid}.pth"


def verify_roots(folds: Dict[int, List[int]]) -> dict:
    require_sha(CANONICAL_MANIFEST, EXPECTED_CANONICAL_SHA, "canonical manifest")
    require_sha(FOLD_MANIFEST, EXPECTED_FOLD_SHA, "5-fold manifest")
    require_sha(OOF_MANIFEST, EXPECTED_OOF_SHA, "RUN_07 OOF manifest")
    require_sha(RUN08_REPORT, EXPECTED_RUN08_SHA, "RUN_08 report")

    can_manifest = load_json(CANONICAL_MANIFEST)
    oof_manifest = load_json(OOF_MANIFEST)

    if int(can_manifest["_meta"]["actual_total_frames"]) != TOTAL_FRAMES:
        raise AssertionError("[FATAL] Canonical cardinality drift.")
    if int(oof_manifest["_meta"]["total_oof_frames"]) != TOTAL_FRAMES:
        raise AssertionError("[FATAL] OOF cardinality drift.")
    if tuple(oof_manifest["_meta"]["expert_order"]) != EXPERTS:
        raise AssertionError("[FATAL] OOF expert order drift.")

    can_ledger = can_manifest["cache_shard_sha256_ledger"]
    oof_ledger = oof_manifest["oof_shard_sha256_ledger"]

    for vid in range(1, 48):
        cp = canonical_path(vid)
        if sha256(cp) != can_ledger[cp.name]:
            raise AssertionError(f"[FATAL] Canonical shard drift vid={vid}")

    for f, vids in folds.items():
        for vid in vids:
            p = oof_path(f, vid)
            key = f"fold{f}/{p.name}"
            if sha256(p) != oof_ledger[key]:
                raise AssertionError(f"[FATAL] OOF shard drift fold={f}, vid={vid}")

    print("[+] Provenance PASS: canonical + RUN_07 OOF + RUN_08 report")
    return {"canonical": can_manifest, "oof": oof_manifest}


# =============================================================================
# FROZEN OPERATIONAL MATCHING
# =============================================================================

def extract_top100(logits: torch.Tensor, boxes: torch.Tensor) -> dict:
    probs = logits.sigmoid()
    scores, flat = torch.topk(probs.flatten(), TOPK)
    q = flat // NUM_CLASSES
    c = flat % NUM_CLASSES
    valid = scores >= SCORE_THRESHOLD
    return {
        "all_scores": scores,
        "all_flat": flat,
        "all_q": q,
        "all_c": c,
        "all_boxes": boxes[q],
        "op_scores": scores[valid],
        "op_q": q[valid],
        "op_c": c[valid],
        "op_boxes": boxes[q][valid],
    }


def match_full(pred: dict, gt_boxes: torch.Tensor, gt_labels: torch.Tensor) -> dict:
    matched_gt: Set[int] = set()
    det_to_gt: Dict[Tuple[int, int], Tuple[int, float, float]] = {}
    fp_pairs: Set[Tuple[int, int]] = set()

    gtb = gt_boxes.tolist()
    gtl = gt_labels.tolist()
    scores = pred["op_scores"].tolist()
    qs = pred["op_q"].tolist()
    cs = pred["op_c"].tolist()
    boxes = pred["op_boxes"].tolist()

    for d in range(len(scores)):
        q = int(qs[d])
        c = int(cs[d])
        best_iou = -1.0
        best_gt = None
        for g in range(len(gtb)):
            if g in matched_gt or int(gtl[g]) != c:
                continue
            iou = box_iou_cxcywh(boxes[d], gtb[g])
            if iou >= IOU_THRESHOLD and iou > best_iou:
                best_iou = iou
                best_gt = g
        if best_gt is None:
            fp_pairs.add((q, c))
        else:
            matched_gt.add(best_gt)
            det_to_gt[(q, c)] = (int(best_gt), float(best_iou), float(scores[d]))

    return {
        "det_to_gt": det_to_gt,
        "fp_pairs": fp_pairs,
        "matched_gt": matched_gt,
    }


def baseline_prefix_available(pred: dict, gt_boxes: torch.Tensor, gt_labels: torch.Tensor) -> dict:
    """Available same-class GTs immediately BEFORE each operational pair."""
    matched_by_class: Dict[int, Set[int]] = {c: set() for c in range(NUM_CLASSES)}
    available_before: Dict[Tuple[int, int], Tuple[int, ...]] = {}

    gtb = gt_boxes.tolist()
    gtl = [int(x) for x in gt_labels.tolist()]
    scores = pred["all_scores"].tolist()
    qs = pred["all_q"].tolist()
    cs = pred["all_c"].tolist()
    boxes = pred["all_boxes"].tolist()

    for d in range(TOPK):
        if float(scores[d]) < SCORE_THRESHOLD:
            continue
        q, c = int(qs[d]), int(cs[d])
        avail = tuple(
            g for g in range(len(gtb))
            if gtl[g] == c and g not in matched_by_class[c]
        )
        available_before[(q, c)] = avail

        best_iou = -1.0
        best_gt = None
        for g in avail:
            iou = box_iou_cxcywh(boxes[d], gtb[g])
            if iou >= IOU_THRESHOLD and iou > best_iou:
                best_iou = iou
                best_gt = g
        if best_gt is not None:
            matched_by_class[c].add(best_gt)

    return available_before


# =============================================================================
# EXACT OPTIMIZED LOCAL CLASSIFICATION
# =============================================================================

def local_top100_for_query(
    base_top_margin_scores: torch.Tensor,
    base_top_margin_flat: torch.Tensor,
    q: int,
    new_q_scores: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    margin_q = base_top_margin_flat // NUM_CLASSES
    keep = margin_q != int(q)

    scores = torch.cat([base_top_margin_scores[keep], new_q_scores.float()], dim=0)
    new_flat = torch.arange(
        q * NUM_CLASSES,
        (q + 1) * NUM_CLASSES,
        dtype=torch.long,
    )
    flat = torch.cat([base_top_margin_flat[keep], new_flat], dim=0)

    top_scores, order = torch.topk(scores, TOPK)
    top_flat = flat[order]
    return top_scores, top_flat // NUM_CLASSES, top_flat % NUM_CLASSES


def target_pair_state(
    top_scores: torch.Tensor,
    top_q: torch.Tensor,
    top_c: torch.Tensor,
    target_q: int,
    target_c: int,
    boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
) -> int:
    target_positions = torch.nonzero(
        (top_q == int(target_q)) & (top_c == int(target_c)),
        as_tuple=False,
    ).reshape(-1)

    if target_positions.numel() == 0:
        return NONOP

    pos = int(target_positions[0].item())
    if float(top_scores[pos].item()) < SCORE_THRESHOLD:
        return NONOP

    matched_gt: Set[int] = set()
    gtb = gt_boxes.tolist()
    gtl = [int(x) for x in gt_labels.tolist()]
    boxes_list = boxes.tolist()

    for d in range(pos + 1):
        if float(top_scores[d].item()) < SCORE_THRESHOLD:
            continue
        c = int(top_c[d].item())
        if c != int(target_c):
            continue
        q = int(top_q[d].item())

        best_iou = -1.0
        best_gt = None
        for g in range(len(gtb)):
            if g in matched_gt or gtl[g] != c:
                continue
            iou = box_iou_cxcywh(boxes_list[q], gtb[g])
            if iou >= IOU_THRESHOLD and iou > best_iou:
                best_iou = iou
                best_gt = g

        if q == int(target_q):
            return TP if best_gt is not None else FP

        if best_gt is not None:
            matched_gt.add(best_gt)

    raise AssertionError("[FATAL] Target pair state resolution fell through.")


# =============================================================================
# BOX LOCAL MATCH FOR q
# =============================================================================

def box_after_match_for_pair(
    q: int,
    c: int,
    new_box: torch.Tensor,
    base_pred: dict,
    prefix_available: dict,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
) -> Tuple[int, float]:
    pair = (int(q), int(c))
    if pair not in prefix_available:
        return -1, 0.0

    # Pair is operational. Prior assignments are unchanged because only q's
    # box changes at q's own position.
    avail = prefix_available[pair]
    gtb = gt_boxes.tolist()
    box = new_box.tolist()

    best_iou = -1.0
    best_gt = None
    for g in avail:
        iou = box_iou_cxcywh(box, gtb[g])
        if iou >= IOU_THRESHOLD and iou > best_iou:
            best_iou = iou
            best_gt = g

    if best_gt is None:
        return -1, 0.0
    return int(best_gt), float(best_iou)


# =============================================================================
# ONE FRAME
# =============================================================================

def compile_frame(
    can: dict,
    oof: dict,
    fold: int,
    brute_check: bool = False,
) -> Tuple[dict, dict]:
    if (
        int(can["image_id"]) != int(oof["image_id"])
        or int(can["video_id"]) != int(oof["video_id"])
        or int(can["seq_idx"]) != int(oof["seq_idx"])
    ):
        raise AssertionError("[FATAL] Canonical/OOF frame identity drift.")

    z_base = can["d_logits"].float()
    b_base = can["d_boxes"].float()
    gt_boxes = can["gt_boxes_norm"].float()
    gt_labels = can["gt_labels"].long()

    expert_logits = oof["expert_cls_logits"].float()
    expert_boxes = oof["expert_box_boxes"].float()

    if expert_logits.shape != (4, NUM_QUERIES, NUM_CLASSES):
        raise AssertionError("[FATAL] OOF logits shape drift.")
    if expert_boxes.shape != (4, NUM_QUERIES, 4):
        raise AssertionError("[FATAL] OOF boxes shape drift.")

    candidate_queries = oof["candidate_queries"].long()
    if candidate_queries.ndim != 1:
        raise AssertionError("[FATAL] candidate_queries must be 1-D.")

    base_pred = extract_top100(z_base, b_base)
    base_match = match_full(base_pred, gt_boxes, gt_labels)
    prefix_available = baseline_prefix_available(base_pred, gt_boxes, gt_labels)

    base_probs = z_base.sigmoid()
    margin_scores, margin_flat = torch.topk(base_probs.flatten(), TOPK_MARGIN)

    base_state = {}
    for q in candidate_queries.tolist():
        for c in range(NUM_CLASSES):
            pair = (int(q), int(c))
            if pair in base_match["det_to_gt"]:
                s = TP
            elif pair in base_match["fp_pairs"]:
                s = FP
            else:
                s = NONOP
            base_state[pair] = s

    experts_out = {}
    local_summary = {}

    for e_idx, expert in enumerate(EXPERTS):
        cls_transition = torch.empty(
            candidate_queries.numel(), NUM_CLASSES, dtype=torch.uint8
        )
        cls_score_delta = torch.empty(
            candidate_queries.numel(), NUM_CLASSES, dtype=torch.float32
        )

        box_base_gt = torch.full(
            (candidate_queries.numel(), NUM_CLASSES), -1, dtype=torch.int16
        )
        box_after_gt = torch.full_like(box_base_gt, -1)
        box_iou_base = torch.zeros(
            candidate_queries.numel(), NUM_CLASSES, dtype=torch.float32
        )
        box_iou_after = torch.zeros_like(box_iou_base)

        box_rescue = torch.zeros(candidate_queries.numel(), 1, dtype=torch.float32)
        box_loss = torch.zeros_like(box_rescue)
        box_retain = torch.zeros_like(box_rescue)
        box_delta_iou = torch.zeros_like(box_rescue)
        u_box = torch.zeros_like(box_rescue)
        h_box = torch.zeros_like(box_rescue)

        cls_counts = Counter()
        box_counts = Counter()

        for qi, q_t in enumerate(candidate_queries):
            q = int(q_t.item())

            # ---------------------------------------------------------
            # Classification local marginal
            # ---------------------------------------------------------
            new_scores = expert_logits[e_idx, q].sigmoid()
            l_scores, l_q, l_c = local_top100_for_query(
                margin_scores, margin_flat, q, new_scores
            )

            for c in range(NUM_CLASSES):
                s0 = int(base_state[(q, c)])
                s1 = target_pair_state(
                    l_scores, l_q, l_c, q, c, b_base, gt_boxes, gt_labels
                )
                state = s0 * 3 + s1
                cls_transition[qi, c] = state
                cls_score_delta[qi, c] = float(
                    new_scores[c].item() - base_probs[q, c].item()
                )
                cls_counts[state] += 1

            # ---------------------------------------------------------
            # Box local marginal -- exact q match identities
            # ---------------------------------------------------------
            base_gt_set: Set[int] = set()
            after_gt_set: Set[int] = set()

            for c in range(NUM_CLASSES):
                pair = (q, c)
                if pair in base_match["det_to_gt"]:
                    g, iou, _ = base_match["det_to_gt"][pair]
                    box_base_gt[qi, c] = int(g)
                    box_iou_base[qi, c] = float(iou)
                    base_gt_set.add(int(g))

                g_after, iou_after = box_after_match_for_pair(
                    q=q,
                    c=c,
                    new_box=expert_boxes[e_idx, q],
                    base_pred=base_pred,
                    prefix_available=prefix_available,
                    gt_boxes=gt_boxes,
                    gt_labels=gt_labels,
                )
                if g_after >= 0:
                    box_after_gt[qi, c] = int(g_after)
                    box_iou_after[qi, c] = float(iou_after)
                    after_gt_set.add(int(g_after))

            retained = base_gt_set & after_gt_set
            lost = base_gt_set - after_gt_set
            rescued = after_gt_set - base_gt_set

            r = float(bool(rescued))
            l = float(bool(lost))
            t = float(bool(retained))

            box_rescue[qi, 0] = r
            box_loss[qi, 0] = l
            box_retain[qi, 0] = t

            affected = sorted(base_gt_set | after_gt_set)
            if affected:
                shifts = []
                for g in affected:
                    gt = gt_boxes[g].tolist()
                    shifts.append(
                        box_iou_cxcywh(expert_boxes[e_idx, q].tolist(), gt)
                        - box_iou_cxcywh(b_base[q].tolist(), gt)
                    )
                d_iou = float(sum(shifts) / len(shifts))
            else:
                d_iou = 0.0

            d_iou = max(-1.0, min(1.0, d_iou))
            box_delta_iou[qi, 0] = d_iou

            u = max(r, max(d_iou, 0.0))
            h = max(l, max(-d_iou, 0.0))
            u_box[qi, 0] = u
            h_box[qi, 0] = h

            box_counts["rescue"] += int(r)
            box_counts["loss"] += int(l)
            box_counts["retain"] += int(t)
            box_counts["affected"] += int(bool(affected))

            # ---------------------------------------------------------
            # Deterministic brute-force certification on requested
            # sample frames.
            # ---------------------------------------------------------
            if brute_check:
                z_local = z_base.clone()
                z_local[q] = expert_logits[e_idx, q]
                brute_cls_pred = extract_top100(z_local, b_base)
                brute_cls_match = match_full(brute_cls_pred, gt_boxes, gt_labels)

                for c in range(NUM_CLASSES):
                    pair = (q, c)
                    if pair in brute_cls_match["det_to_gt"]:
                        bstate = TP
                    elif pair in brute_cls_match["fp_pairs"]:
                        bstate = FP
                    else:
                        bstate = NONOP
                    opt_state = int(cls_transition[qi, c].item()) % 3
                    if bstate != opt_state:
                        raise AssertionError(
                            f"[FATAL] Optimized cls local replay mismatch "
                            f"expert={expert} q={q} c={c}: {opt_state}!={bstate}"
                        )

                b_local = b_base.clone()
                b_local[q] = expert_boxes[e_idx, q]
                brute_box_pred = extract_top100(z_base, b_local)
                brute_box_match = match_full(brute_box_pred, gt_boxes, gt_labels)

                brute_base_q = {
                    int(info[0])
                    for (qq, cc), info in base_match["det_to_gt"].items()
                    if int(qq) == q
                }
                brute_after_q = {
                    int(info[0])
                    for (qq, cc), info in brute_box_match["det_to_gt"].items()
                    if int(qq) == q
                }
                if brute_base_q != base_gt_set or brute_after_q != after_gt_set:
                    raise AssertionError(
                        f"[FATAL] Optimized box local replay mismatch "
                        f"expert={expert} q={q}: "
                        f"base {base_gt_set}!={brute_base_q}, "
                        f"after {after_gt_set}!={brute_after_q}"
                    )

        beneficial = torch.zeros_like(cls_transition, dtype=torch.float32)
        harmful = torch.zeros_like(cls_transition, dtype=torch.float32)
        for state in BENEFICIAL_CLS:
            beneficial += (cls_transition == state).float()
        for state in HARMFUL_CLS:
            harmful += (cls_transition == state).float()

        u_cls = beneficial.mean(dim=-1, keepdim=True)
        h_cls = harmful.mean(dim=-1, keepdim=True)

        experts_out[expert] = {
            "cls_transition": cls_transition,
            "cls_score_delta": cls_score_delta,
            "u_cls_target": u_cls,
            "h_cls_target": h_cls,

            "box_base_gt_idx": box_base_gt,
            "box_after_gt_idx": box_after_gt,
            "box_iou_base": box_iou_base,
            "box_iou_after": box_iou_after,
            "box_rescue_target": box_rescue,
            "box_loss_target": box_loss,
            "box_retain_target": box_retain,
            "box_delta_iou_target": box_delta_iou,
            "u_box_target": u_box,
            "h_box_target": h_box,
        }

        local_summary[expert] = {
            "cls_transition_counts": {str(k): int(v) for k, v in sorted(cls_counts.items())},
            "box_counts": {k: int(v) for k, v in box_counts.items()},
        }

    frame = {
        "image_id": int(can["image_id"]),
        "video_id": int(can["video_id"]),
        "seq_idx": int(can["seq_idx"]),
        "fold_idx": int(fold),
        "candidate_queries": candidate_queries.to(torch.int16),
        "experts": experts_out,
    }
    return frame, local_summary


# =============================================================================
# PREFLIGHT
# =============================================================================

def preflight(folds: Dict[int, List[int]], v2f: Dict[int, int]) -> None:
    print("[*] RUN_09A optimized-vs-brute exactness preflight")
    sample_videos = (1, 12, 24, 36, 47)
    comparisons = 0

    for vid in sample_videos:
        fold = v2f[vid]
        can_records = torch.load(canonical_path(vid), map_location="cpu")
        oof_records = torch.load(oof_path(fold, vid), map_location="cpu")
        for idx in sorted({0, len(can_records) // 2, len(can_records) - 1}):
            can = can_records[idx]
            oof = oof_records[idx]

            # Full brute comparison over only three deterministic candidate
            # queries to keep certification cheap but nontrivial.
            q_all = oof["candidate_queries"].long()
            choose = sorted({0, q_all.numel() // 2, q_all.numel() - 1})
            q_subset = q_all[choose]

            oof_small = dict(oof)
            oof_small["candidate_queries"] = q_subset

            compile_frame(can, oof_small, fold, brute_check=True)
            comparisons += int(q_subset.numel()) * len(EXPERTS)

        print(f"    [+] video {vid:02d} preflight PASS")

    print(
        f"[PASS] RUN_09A LOCAL-REPLAY EXACTNESS: "
        f"{comparisons} expert-query comparisons, all 7 cls states + box match sets exact"
    )


# =============================================================================
# FULL COMPILE
# =============================================================================

def existing_shard_valid(path: Path, fold: int, vid: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = torch.load(path, map_location="cpu")
        meta = payload["_meta"]
        return (
            meta.get("schema_version") == SCHEMA
            and int(meta.get("fold_idx")) == fold
            and int(meta.get("video_id")) == vid
            and meta.get("oof_manifest_sha256") == EXPECTED_OOF_SHA
            and meta.get("run08_report_sha256") == EXPECTED_RUN08_SHA
        )
    except Exception:
        return False


def merge_counter(dst: dict, src: dict) -> None:
    for expert in EXPERTS:
        for k, v in src[expert]["cls_transition_counts"].items():
            dst[expert]["cls_transition_counts"][str(k)] += int(v)
        for k, v in src[expert]["box_counts"].items():
            dst[expert]["box_counts"][k] += int(v)


def summary_from_records(records: list) -> dict:
    summary = {
        e: {
            "cls_transition_counts": defaultdict(int),
            "box_counts": defaultdict(int),
        }
        for e in EXPERTS
    }
    for fr in records:
        for e in EXPERTS:
            tr = fr["experts"][e]["cls_transition"].reshape(-1).tolist()
            for x in tr:
                summary[e]["cls_transition_counts"][str(int(x))] += 1

            ex = fr["experts"][e]
            summary[e]["box_counts"]["rescue"] += int(ex["box_rescue_target"].sum().item())
            summary[e]["box_counts"]["loss"] += int(ex["box_loss_target"].sum().item())
            summary[e]["box_counts"]["retain"] += int(ex["box_retain_target"].sum().item())
            affected = (
                (ex["box_rescue_target"] > 0)
                | (ex["box_loss_target"] > 0)
                | (ex["box_retain_target"] > 0)
            )
            summary[e]["box_counts"]["affected"] += int(affected.sum().item())

    return {
        e: {
            "cls_transition_counts": dict(summary[e]["cls_transition_counts"]),
            "box_counts": dict(summary[e]["box_counts"]),
        }
        for e in EXPERTS
    }


def compile_all(folds: Dict[int, List[int]], v2f: Dict[int, int], resume: bool) -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    ledger = {}
    per_video = {}
    global_summary = {
        e: {
            "cls_transition_counts": defaultdict(int),
            "box_counts": defaultdict(int),
        }
        for e in EXPERTS
    }

    total_frames = 0
    total_candidate_queries = 0
    t0 = time.time()

    for vid in range(1, 48):
        fold = v2f[vid]
        out_path = out_shard_path(fold, vid)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if resume and existing_shard_valid(out_path, fold, vid):
            payload = torch.load(out_path, map_location="cpu")
            records = payload["records"]
            summary = payload["_meta"]["summary"]
            print(f"[+] Reusing certified RUN_09A shard vid={vid:02d}: {len(records):4d} frames")
        else:
            can_records = torch.load(canonical_path(vid), map_location="cpu")
            oof_records = torch.load(oof_path(fold, vid), map_location="cpu")
            if len(can_records) != len(oof_records):
                raise AssertionError(f"[FATAL] vid={vid} canonical/OOF length mismatch")

            records = []
            for idx, (can, oof) in enumerate(zip(can_records, oof_records)):
                fr, _ = compile_frame(can, oof, fold, brute_check=False)
                records.append(fr)
                if (idx + 1) % 200 == 0:
                    print(
                        f"    vid={vid:02d} frames={idx+1:4d}/{len(can_records)}"
                    )

            summary = summary_from_records(records)
            payload = {
                "_meta": {
                    "schema_version": SCHEMA,
                    "video_id": vid,
                    "fold_idx": fold,
                    "num_frames": len(records),
                    "canonical_manifest_sha256": EXPECTED_CANONICAL_SHA,
                    "oof_manifest_sha256": EXPECTED_OOF_SHA,
                    "run08_report_sha256": EXPECTED_RUN08_SHA,
                    "summary": summary,
                },
                "records": records,
            }
            torch.save(payload, out_path)
            print(
                f"[+] Compiled vid={vid:02d}: {len(records):4d} frames -> {out_path.name}"
            )

        shard_sha = sha256(out_path)
        ledger[f"fold{fold}/{out_path.name}"] = shard_sha
        per_video[str(vid)] = {
            "fold": fold,
            "file": f"fold{fold}/{out_path.name}",
            "sha256": shard_sha,
            "num_frames": len(records),
            "summary": summary,
        }

        total_frames += len(records)
        total_candidate_queries += sum(
            int(fr["candidate_queries"].numel()) for fr in records
        )

        for e in EXPERTS:
            for k, v in summary[e]["cls_transition_counts"].items():
                global_summary[e]["cls_transition_counts"][str(k)] += int(v)
            for k, v in summary[e]["box_counts"].items():
                global_summary[e]["box_counts"][k] += int(v)

        del records

    if total_frames != TOTAL_FRAMES:
        raise AssertionError(
            f"[FATAL] RUN_09A frame total {total_frames} != {TOTAL_FRAMES}"
        )
    if len(ledger) != NUM_VIDEOS:
        raise AssertionError("[FATAL] RUN_09A shard count != 47")

    global_summary_json = {
        e: {
            "cls_transition_counts": {
                str(k): int(v)
                for k, v in sorted(
                    global_summary[e]["cls_transition_counts"].items(),
                    key=lambda kv: int(kv[0]),
                )
            },
            "box_counts": {
                k: int(v)
                for k, v in sorted(global_summary[e]["box_counts"].items())
            },
        }
        for e in EXPERTS
    }

    manifest = {
        "_meta": {
            "schema_version": SCHEMA,
            "driver_sha256": sha256(Path(__file__)),
            "canonical_manifest_sha256": EXPECTED_CANONICAL_SHA,
            "fold_manifest_sha256": EXPECTED_FOLD_SHA,
            "oof_manifest_sha256": EXPECTED_OOF_SHA,
            "run08_report_sha256": EXPECTED_RUN08_SHA,
            "num_videos": NUM_VIDEOS,
            "total_frames": total_frames,
            "total_candidate_queries": total_candidate_queries,
            "num_classes": NUM_CLASSES,
            "cls_state_semantics": {
                "0": "TP->TP",
                "1": "TP->FP",
                "2": "TP->NONOP",
                "3": "FP->TP",
                "4": "FP->FP",
                "5": "FP->NONOP",
                "6": "NONOP->TP",
                "7": "NONOP->FP",
                "8": "NONOP->NONOP",
            },
            "u_cls_target": "mean over 7 classes of {FP->TP, FP->NONOP, NONOP->TP}",
            "h_cls_target": "mean over 7 classes of {TP->FP, TP->NONOP, NONOP->FP}",
            "u_box_target": "max(any rescue, positive mean affected-GT delta IoU)",
            "h_box_target": "max(any loss, negative mean affected-GT delta IoU magnitude)",
            "runtime_minutes": (time.time() - t0) / 60.0,
        },
        "shard_sha256_ledger": ledger,
        "per_video": per_video,
        "global_summary": global_summary_json,
    }

    manifest_path = OUT / "run09_final_consequence_manifest.json"
    save_json(manifest_path, manifest)

    print("=" * 118)
    print("[PASS] RUN_09A FINAL LOCAL CONSEQUENCE COMPILATION COMPLETE")
    print(f"[+] Frames:            {total_frames:,}")
    print(f"[+] Candidate queries: {total_candidate_queries:,}")
    print(f"[+] Manifest:          {manifest_path}")
    print(f"[+] Manifest SHA:      {sha256(manifest_path)}")
    print("=" * 118)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    print("=" * 118)
    print("B-FCEF-F RUN_09A -- FINAL OOF LOCAL-COUNTERFACTUAL CONSEQUENCE COMPILER")
    print("TRAIN ONLY | NO TRAINING | NO VAL/TEST | FINAL EXPERT OOF ONLY")
    print("=" * 118)

    folds = parse_folds()
    v2f = video_to_fold_map(folds)
    verify_roots(folds)

    if args.preflight_only:
        preflight(folds, v2f)
        return

    compile_all(folds, v2f, resume=args.resume)


if __name__ == "__main__":
    main()
