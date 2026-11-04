from __future__ import annotations

"""
B-FCEF-F RUN_09D
================
Calibrated Integrated Mechanism Qualification -- META-FOLD 0.

READ ONLY. NO TRAINING. NO CALIBRATION. NO VAL/TEST.

Frozen inputs
-------------
- RUN_07 Final Expert fold-0 checkpoint
- RUN_09B downstream mechanism checkpoint
- RUN_09C train-derived calibrated hard thresholds

Variants
--------
Base
Full
ClsOnly
BoxOnly
MinusS
MinusL
MinusC
MinusT

Metrics
-------
Certified SeaClips Fold-0:
- mAP@0.30
- category-agnostic mAP@0.30
- AP_S / AP_M / AP_L
- per-class AP@0.30
- operational TP / FP / FN / Precision / Recall / F1
- GT rescue / loss / retain
- FP add / remove

Mechanism diagnostics (Full)
----------------------------
- hard mask density by expert/task
- simplex route mass pi_0, pi_S, pi_L, pi_C, pi_T
- expert mass = 1-pi_0 quantiles
- fractions above meaningful route-mass cutoffs
- exact / tolerance-based output-change rates
- Top-100 and operational candidate-set changes
- hard bypass fraction

Acceptance gates
----------------
1. Calibrated checkpoint/report SHA exact.
2. Fold-0 video inventory exact.
3. Base operational counts exactly reproduce RUN_09C.
4. Full operational counts exactly reproduce RUN_09C.
5. No fold-0 labels are used for any tuning (this script has no tuning path).
"""

import contextlib
import csv
import gc
import hashlib
import io
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("R0_MSDA_BACKEND", "pytorch")

import numpy as np
import torch

ROOT = Path("D:/MARIVISION")
EVAL = ROOT / "H0_workbench" / "eval_tools"
RUN09C = EVAL / "bfcef_f_s0" / "run09c_threshold_calibration_v1"
CAL_CKPT = RUN09C / "meta_fold0_uh_gate_router_calibrated.pth"
CAL_REPORT = RUN09C / "run09c_threshold_calibration_report.json"
OUT = EVAL / "bfcef_f_s0" / "run09d_calibrated_qualification_v1"

EXPECTED_CAL_CKPT_SHA = "a1debfaadcc7f754319c63386ac93d993dc2a036cefbdf4e1ee70a567b517deb"
EXPECTED_CAL_REPORT_SHA = "9527b40a2ba546310f15d3e8dc205e50f870788fb01468de315038a224751093"

META_FOLD = 0
EXPERTS = ("S", "L", "C", "T")
CAPABILITY_NAME = {
    "S": "scale",
    "L": "localization",
    "C": "semantic",
    "T": "temporal",
}

VARIANTS = (
    "Base",
    "Full",
    "ClsOnly",
    "BoxOnly",
    "MinusS",
    "MinusL",
    "MinusC",
    "MinusT",
)

EXPECTED_FOLD0_VIDEOS = [6, 7, 14, 16, 17, 24, 36, 37, 40, 42]
EXPECTED_BASE_COUNTS = {"tp": 8918, "fp": 17137, "fn": 6177}
EXPECTED_FULL_COUNTS = {"tp": 8861, "fp": 13509, "fn": 6234}

ROUTE_CUTOFFS = (0.01, 0.05, 0.10, 0.25)
CHANGE_TOLS = (0.0, 1e-7, 1e-5, 1e-3)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if str(EVAL) not in sys.path:
    sys.path.insert(0, str(EVAL))

# Authoritative local implementation and certified evaluator semantics.
import train_bfcef_f_run09_joint_uh_gate_router as r9  # noqa: E402
import run_bfcef_f_run08_oof_counterfactual_audit as r8  # noqa: E402


# =============================================================================
# GENERIC
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


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def metrics_from_counts(tp: int, fp: int, fn: int) -> dict:
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn),
        "f1": safe_div(2 * tp, 2 * tp + fp + fn),
    }


def percentile_dict(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": 0.0, "p01": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    a = np.asarray(values, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "p01": float(np.percentile(a, 1)),
        "p10": float(np.percentile(a, 10)),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }


# =============================================================================
# MODEL LOAD
# =============================================================================

def load_calibrated_downstream(model, payload: dict) -> None:
    for expert in EXPERTS:
        model.self_assessment[expert].load_state_dict(
            payload["self_assessment"][expert], strict=True
        )
        model.gate_cls[expert].load_state_dict(
            payload["gate_cls"][expert], strict=True
        )
        model.gate_box[expert].load_state_dict(
            payload["gate_box"][expert], strict=True
        )
    model.coord_cls.load_state_dict(payload["coord_cls"], strict=True)
    model.coord_box.load_state_dict(payload["coord_box"], strict=True)


def initialize_model():
    (
        model,
        criterion,
        CanonicalObjectState,
        frame_inputs_from_records,
    ) = r9.initialize_runtime()

    payload = torch.load(CAL_CKPT, map_location="cpu")
    load_calibrated_downstream(model, payload)

    # Fold-0 Final Experts are the honest RUN_07 model for all Fold-0 videos.
    r9.load_source_fold(model, META_FOLD)
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    return model, CanonicalObjectState, frame_inputs_from_records, payload


# =============================================================================
# VARIANT FORWARD
# =============================================================================

def set_expert_capability(O, expert: str, enabled: bool) -> None:
    name = CAPABILITY_NAME[expert]
    O.capabilities[name] = bool(enabled)
    O.capabilities[f"{name}.cls"] = bool(enabled)
    O.capabilities[f"{name}.box"] = bool(enabled)


def production_forward(model, O, X, disabled_expert: str | None = None):
    if disabled_expert is not None:
        set_expert_capability(O, disabled_expert, False)

    try:
        z, b, info = model(
            O,
            X,
            m_ext=None,
            m_hat_vis=None,
            m_avail=None,
            context_mode="auto",
        )
    finally:
        if disabled_expert is not None:
            set_expert_capability(O, disabled_expert, True)

    return z, b, info


def make_variants(model, O, X):
    z_full, b_full, info_full = production_forward(model, O, X)

    outputs = {
        "Base": (O.z, O.b),
        "Full": (z_full, b_full),
        "ClsOnly": (z_full, O.b),
        "BoxOnly": (O.z, b_full),
    }

    for expert in EXPERTS:
        z, b, _ = production_forward(model, O, X, disabled_expert=expert)
        outputs[f"Minus{expert}"] = (z, b)

    return outputs, info_full


# =============================================================================
# DIAGNOSTICS
# =============================================================================

class FullDiagnostics:
    def __init__(self):
        self.frames = 0
        self.bypass_frames = 0
        self.candidate_queries = 0

        self.hard_cls = np.zeros(4, dtype=np.int64)
        self.hard_box = np.zeros(4, dtype=np.int64)

        self.pi_cls = [[] for _ in range(5)]
        self.pi_box = [[] for _ in range(5)]
        self.expert_mass_cls = []
        self.expert_mass_box = []

        self.change_cls = {tol: 0 for tol in CHANGE_TOLS}
        self.change_box = {tol: 0 for tol in CHANGE_TOLS}
        self.change_cls_frames = {tol: 0 for tol in CHANGE_TOLS}
        self.change_box_frames = {tol: 0 for tol in CHANGE_TOLS}

        self.top100_pair_changed_frames = 0
        self.operational_pair_changed_frames = 0
        self.top100_pair_symdiff = 0
        self.operational_pair_symdiff = 0

    def update(self, O, z_full, b_full, info, pred_base, pred_full):
        self.frames += 1
        if bool(info.get("bypassed", False)):
            self.bypass_frames += 1

        candidate_set = info["candidate_set"]
        q = torch.as_tensor(candidate_set.queries, device=z_full.device, dtype=torch.long)
        self.candidate_queries += int(q.numel())

        if q.numel() > 0:
            if "m_cls" in info:
                hc = info["m_cls"][q].detach().cpu().numpy()
                hb = info["m_box"][q].detach().cpu().numpy()
                self.hard_cls += (hc > 0).sum(axis=0)
                self.hard_box += (hb > 0).sum(axis=0)

            if "pi_cls" in info:
                pc = info["pi_cls"][q].detach().float().cpu().numpy()
                pb = info["pi_box"][q].detach().float().cpu().numpy()

                for j in range(5):
                    self.pi_cls[j].extend(pc[:, j].tolist())
                    self.pi_box[j].extend(pb[:, j].tolist())

                self.expert_mass_cls.extend((1.0 - pc[:, 0]).tolist())
                self.expert_mass_box.extend((1.0 - pb[:, 0]).tolist())

            dz = (z_full[q] - O.z[q]).abs().amax(dim=-1).detach().cpu().numpy()
            db = (b_full[q] - O.b[q]).abs().amax(dim=-1).detach().cpu().numpy()

            for tol in CHANGE_TOLS:
                c_cls = int((dz > tol).sum())
                c_box = int((db > tol).sum())
                self.change_cls[tol] += c_cls
                self.change_box[tol] += c_box
                self.change_cls_frames[tol] += int(c_cls > 0)
                self.change_box_frames[tol] += int(c_box > 0)

        base_top = set(zip(
            pred_base["all_q_idx"].tolist(),
            pred_base["all_c_idx"].tolist(),
        ))
        full_top = set(zip(
            pred_full["all_q_idx"].tolist(),
            pred_full["all_c_idx"].tolist(),
        ))
        d = len(base_top.symmetric_difference(full_top))
        self.top100_pair_symdiff += d
        self.top100_pair_changed_frames += int(d > 0)

        base_op = set(zip(
            pred_base["op_q_idx"].tolist(),
            pred_base["op_c_idx"].tolist(),
        ))
        full_op = set(zip(
            pred_full["op_q_idx"].tolist(),
            pred_full["op_c_idx"].tolist(),
        ))
        d = len(base_op.symmetric_difference(full_op))
        self.operational_pair_symdiff += d
        self.operational_pair_changed_frames += int(d > 0)

    def finalize(self):
        routes = {}
        for task, values, expert_mass in (
            ("cls", self.pi_cls, self.expert_mass_cls),
            ("box", self.pi_box, self.expert_mass_box),
        ):
            routes[task] = {
                "pi_0": percentile_dict(values[0]),
                "pi_S": percentile_dict(values[1]),
                "pi_L": percentile_dict(values[2]),
                "pi_C": percentile_dict(values[3]),
                "pi_T": percentile_dict(values[4]),
                "expert_mass_1_minus_pi0": percentile_dict(expert_mass),
                "expert_mass_fraction_above": {
                    str(c): float(np.mean(np.asarray(expert_mass) > c))
                    if expert_mass else 0.0
                    for c in ROUTE_CUTOFFS
                },
            }

        return {
            "frames": self.frames,
            "hard_bypass_frames": self.bypass_frames,
            "hard_bypass_fraction": safe_div(self.bypass_frames, self.frames),
            "candidate_queries": self.candidate_queries,
            "hard_mask_active_count": {
                "cls": {e: int(self.hard_cls[i]) for i, e in enumerate(EXPERTS)},
                "box": {e: int(self.hard_box[i]) for i, e in enumerate(EXPERTS)},
            },
            "hard_mask_active_fraction_of_candidates": {
                "cls": {
                    e: safe_div(int(self.hard_cls[i]), self.candidate_queries)
                    for i, e in enumerate(EXPERTS)
                },
                "box": {
                    e: safe_div(int(self.hard_box[i]), self.candidate_queries)
                    for i, e in enumerate(EXPERTS)
                },
            },
            "routes": routes,
            "output_change": {
                "cls_query_count_above_tol": {
                    str(t): int(self.change_cls[t]) for t in CHANGE_TOLS
                },
                "box_query_count_above_tol": {
                    str(t): int(self.change_box[t]) for t in CHANGE_TOLS
                },
                "cls_query_fraction_above_tol": {
                    str(t): safe_div(self.change_cls[t], self.candidate_queries)
                    for t in CHANGE_TOLS
                },
                "box_query_fraction_above_tol": {
                    str(t): safe_div(self.change_box[t], self.candidate_queries)
                    for t in CHANGE_TOLS
                },
                "cls_frame_fraction_above_tol": {
                    str(t): safe_div(self.change_cls_frames[t], self.frames)
                    for t in CHANGE_TOLS
                },
                "box_frame_fraction_above_tol": {
                    str(t): safe_div(self.change_box_frames[t], self.frames)
                    for t in CHANGE_TOLS
                },
            },
            "top100_pair_changed_frame_fraction": safe_div(
                self.top100_pair_changed_frames, self.frames
            ),
            "operational_pair_changed_frame_fraction": safe_div(
                self.operational_pair_changed_frames, self.frames
            ),
            "mean_top100_pair_symmetric_difference_per_frame": safe_div(
                self.top100_pair_symdiff, self.frames
            ),
            "mean_operational_pair_symmetric_difference_per_frame": safe_div(
                self.operational_pair_symdiff, self.frames
            ),
        }


# =============================================================================
# QUALIFICATION
# =============================================================================

def new_transition():
    return {
        "rescued": 0,
        "lost": 0,
        "retained": 0,
        "fp_added": 0,
        "fp_removed": 0,
    }


def finalize_variant_evaluator(ev, ev_ca, class_names):
    stats = r8.quiet_finalize_evaluator(ev)
    stats_ca = r8.quiet_finalize_evaluator(ev_ca)
    pclass = r8.per_class_ap(ev)

    return {
        "map30": float(stats[0]),
        "mapca30": float(stats_ca[0]),
        "ap_s30": float(stats[3]),
        "ap_m30": float(stats[4]),
        "ap_l30": float(stats[5]),
        "per_class_ap30": {
            class_names[i]: float(pclass[i]) for i in range(len(class_names))
        },
    }


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 126)
    print("B-FCEF-F RUN_09D -- CALIBRATED INTEGRATED MECHANISM QUALIFICATION")
    print("META-FOLD 0 ONLY | READ ONLY | NO TRAINING | NO CALIBRATION | NO VAL/TEST")
    print("=" * 126)

    require_sha(CAL_CKPT, EXPECTED_CAL_CKPT_SHA, "RUN_09C calibrated checkpoint")
    require_sha(CAL_REPORT, EXPECTED_CAL_REPORT_SHA, "RUN_09C report")
    print("[+] RUN_09C provenance PASS")

    folds = r9.parse_folds()
    if sorted(folds[META_FOLD]) != EXPECTED_FOLD0_VIDEOS:
        raise AssertionError(
            f"[FATAL] Fold-0 inventory drift: {sorted(folds[META_FOLD])}"
        )

    model, CanonicalObjectState, frame_inputs_from_records, cal_payload = initialize_model()

    (
        coco_gt,
        coco_gt_ca,
        img_native_size,
        img_to_video,
        video_to_camera,
        cat_id_tensor,
        class_names,
    ) = r8.load_metadata()

    # One class-aware and one category-agnostic evaluator per variant.
    evaluators = {
        v: r8.make_evaluators(coco_gt, coco_gt_ca) for v in VARIANTS
    }

    op_counts = {
        v: {"tp": 0, "fp": 0, "fn": 0} for v in VARIANTS
    }
    transitions = {
        v: new_transition() for v in VARIANTS if v != "Base"
    }
    by_video = {
        v: {
            str(vid): {"tp": 0, "fp": 0, "fn": 0}
            for vid in EXPECTED_FOLD0_VIDEOS
        }
        for v in VARIANTS
    }

    class_support = Counter()
    diag = FullDiagnostics()
    seen_images = set()

    with torch.no_grad():
        for vid in EXPECTED_FOLD0_VIDEOS:
            canonical, spatial, temporal, labels = r9.load_video_all(vid, META_FOLD)

            video_pred_maps = {v: {} for v in VARIANTS}
            video_pred_maps_ca = {v: {} for v in VARIANTS}

            for idx in range(len(canonical)):
                can = canonical[idx]
                O, X = r9.s0a.make_O_X(
                    can,
                    spatial[idx],
                    temporal[idx],
                    CanonicalObjectState,
                    frame_inputs_from_records,
                )

                outputs, info_full = make_variants(model, O, X)

                gt_boxes = can["gt_boxes_norm"].float()
                gt_labels = can["gt_labels"].long()
                img_id = int(can["image_id"])

                if img_id in seen_images:
                    raise AssertionError(f"[FATAL] Duplicate Fold-0 image_id={img_id}")
                seen_images.add(img_id)

                for c in gt_labels.tolist():
                    class_support[int(c)] += 1

                frame_pred = {}
                frame_match = {}

                for variant in VARIANTS:
                    z, b = outputs[variant]
                    pred = r8.extract_top100_predictions(
                        z.detach().cpu().float(),
                        b.detach().cpu().float(),
                    )
                    frame_pred[variant] = pred

                    img_w, img_h = img_native_size[img_id]
                    r8.add_coco_prediction(
                        video_pred_maps[variant],
                        video_pred_maps_ca[variant],
                        img_id,
                        pred,
                        img_w,
                        img_h,
                        cat_id_tensor,
                    )

                    m = r8.match_operational_detections_to_gt(
                        pred["op_scores"],
                        pred["op_c_idx"],
                        pred["op_boxes"],
                        pred["op_q_idx"],
                        gt_boxes,
                        gt_labels,
                    )
                    frame_match[variant] = m

                    op_counts[variant]["tp"] += int(m["tp"])
                    op_counts[variant]["fp"] += int(m["fp"])
                    op_counts[variant]["fn"] += int(m["fn"])

                    bv = by_video[variant][str(vid)]
                    bv["tp"] += int(m["tp"])
                    bv["fp"] += int(m["fp"])
                    bv["fn"] += int(m["fn"])

                base_m = frame_match["Base"]

                for variant in VARIANTS:
                    if variant == "Base":
                        continue
                    vm = frame_match[variant]
                    tr = transitions[variant]

                    bgt = base_m["matched_gt_indices"]
                    vgt = vm["matched_gt_indices"]
                    tr["rescued"] += len(vgt - bgt)
                    tr["lost"] += len(bgt - vgt)
                    tr["retained"] += len(bgt & vgt)

                    bfp = set(base_m["fp_dict"])
                    vfp = set(vm["fp_dict"])
                    tr["fp_added"] += len(vfp - bfp)
                    tr["fp_removed"] += len(bfp - vfp)

                diag.update(
                    O,
                    outputs["Full"][0],
                    outputs["Full"][1],
                    info_full,
                    frame_pred["Base"],
                    frame_pred["Full"],
                )

            # Stream evaluators once per video.
            for variant in VARIANTS:
                ev, ev_ca = evaluators[variant]
                ev.update(video_pred_maps[variant])
                ev_ca.update(video_pred_maps_ca[variant])

            print(
                f"    [+] Fold0 video {vid:02d} complete: "
                f"{len(canonical):4d} frames"
            )

            del canonical, spatial, temporal, labels
            del video_pred_maps, video_pred_maps_ca
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # Hard operational identity gates against RUN_09C
    # -----------------------------------------------------------------
    for k, expected in EXPECTED_BASE_COUNTS.items():
        if op_counts["Base"][k] != expected:
            raise AssertionError(
                f"[FATAL] Fold0 Base {k} drift: "
                f"{op_counts['Base'][k]} != {expected}"
            )

    for k, expected in EXPECTED_FULL_COUNTS.items():
        if op_counts["Full"][k] != expected:
            raise AssertionError(
                f"[FATAL] Fold0 Full {k} drift: "
                f"{op_counts['Full'][k]} != {expected}"
            )

    print("[PASS] RUN_09C OPERATIONAL IDENTITY: Base + Full counts exact")

    # -----------------------------------------------------------------
    # Finalize COCO metrics
    # -----------------------------------------------------------------
    results = {}

    for variant in VARIANTS:
        ev, ev_ca = evaluators[variant]
        coco_metrics = finalize_variant_evaluator(
            ev, ev_ca, class_names
        )
        result = {
            **metrics_from_counts(**op_counts[variant]),
            **coco_metrics,
            "by_video_operational": by_video[variant],
        }

        if variant != "Base":
            tr = transitions[variant]
            tr["net_tp"] = tr["rescued"] - tr["lost"]
            tr["net_fp"] = tr["fp_added"] - tr["fp_removed"]

            if tr["retained"] + tr["lost"] != op_counts["Base"]["tp"]:
                raise AssertionError(
                    f"[FATAL] {variant} retained+lost conservation failed"
                )
            if tr["retained"] + tr["rescued"] != op_counts[variant]["tp"]:
                raise AssertionError(
                    f"[FATAL] {variant} retained+rescued conservation failed"
                )
            if tr["net_tp"] != (
                op_counts[variant]["tp"] - op_counts["Base"]["tp"]
            ):
                raise AssertionError(
                    f"[FATAL] {variant} TP conservation failed"
                )
            if tr["net_fp"] != (
                op_counts[variant]["fp"] - op_counts["Base"]["fp"]
            ):
                raise AssertionError(
                    f"[FATAL] {variant} FP conservation failed"
                )

        results[variant] = result

        del ev, ev_ca
        gc.collect()

    diagnostics = diag.finalize()

    # -----------------------------------------------------------------
    # Print qualification tables
    # -----------------------------------------------------------------
    base_r = results["Base"]

    print("\n" + "=" * 154)
    print("RUN_09D TABLE 1 -- FOLD-0 CALIBRATED INTEGRATED QUALIFICATION")
    print("=" * 154)
    print(
        f"{'Variant':<9} | {'mAP@.3':>7} | {'DmAP':>7} | {'mAPca':>7} | "
        f"{'TP':>6} | {'FP':>7} | {'FN':>6} | {'Prec':>7} | {'Rec':>7} | "
        f"{'F1':>7} | {'DF1':>7} | {'AP_S':>7} | {'AP_M':>7} | {'AP_L':>7}"
    )
    print("-" * 154)

    for v in VARIANTS:
        m = results[v]
        print(
            f"{v:<9} | {m['map30']:7.4f} | {m['map30']-base_r['map30']:+7.4f} | "
            f"{m['mapca30']:7.4f} | {m['tp']:6,d} | {m['fp']:7,d} | {m['fn']:6,d} | "
            f"{m['precision']:7.3f} | {m['recall']:7.3f} | {m['f1']:7.4f} | "
            f"{m['f1']-base_r['f1']:+7.4f} | {m['ap_s30']:7.4f} | "
            f"{m['ap_m30']:7.4f} | {m['ap_l30']:7.4f}"
        )
    print("=" * 154)

    print("\n" + "=" * 118)
    print("RUN_09D TABLE 2 -- OPERATIONAL CONSEQUENCE ATTRIBUTION")
    print("=" * 118)
    print(
        f"{'Variant':<9} | {'Rescue':>7} | {'Lost':>7} | {'NetTP':>7} | "
        f"{'Retain':>7} | {'FPadd':>7} | {'FPrem':>7} | {'NetFP':>7}"
    )
    print("-" * 118)
    for v in VARIANTS:
        if v == "Base":
            continue
        t = transitions[v]
        print(
            f"{v:<9} | {t['rescued']:7,d} | {t['lost']:7,d} | {t['net_tp']:+7,d} | "
            f"{t['retained']:7,d} | {t['fp_added']:7,d} | {t['fp_removed']:7,d} | "
            f"{t['net_fp']:+7,d}"
        )
    print("=" * 118)

    print("\n" + "=" * 118)
    print("RUN_09D TABLE 3 -- FULL MECHANISM ROUTE / INTERVENTION DIAGNOSTICS")
    print("=" * 118)
    print(
        f"Hard bypass fraction: {diagnostics['hard_bypass_fraction']:.4%} "
        f"({diagnostics['hard_bypass_frames']}/{diagnostics['frames']} frames)"
    )
    print(f"Candidate queries: {diagnostics['candidate_queries']:,}")
    print(
        "Hard cls active fractions: "
        + ", ".join(
            f"{e}={diagnostics['hard_mask_active_fraction_of_candidates']['cls'][e]:.2%}"
            for e in EXPERTS
        )
    )
    print(
        "Hard box active fractions: "
        + ", ".join(
            f"{e}={diagnostics['hard_mask_active_fraction_of_candidates']['box'][e]:.2%}"
            for e in EXPERTS
        )
    )

    for task in ("cls", "box"):
        rd = diagnostics["routes"][task]
        pi0 = rd["pi_0"]
        mass = rd["expert_mass_1_minus_pi0"]
        print(
            f"{task.upper()} pi0: mean={pi0['mean']:.4f}, p10={pi0['p10']:.4f}, "
            f"p50={pi0['p50']:.4f}, p90={pi0['p90']:.4f}, p99={pi0['p99']:.4f}"
        )
        print(
            f"{task.upper()} expert mass 1-pi0: mean={mass['mean']:.4f}, "
            f"p50={mass['p50']:.4f}, p90={mass['p90']:.4f}, p99={mass['p99']:.4f}; "
            + ", ".join(
                f">{cut:.2f}={rd['expert_mass_fraction_above'][str(cut)]:.2%}"
                for cut in ROUTE_CUTOFFS
            )
        )

    oc = diagnostics["output_change"]
    print(
        "CLS candidate output change fractions: "
        + ", ".join(
            f">{tol:g}={oc['cls_query_fraction_above_tol'][str(tol)]:.2%}"
            for tol in CHANGE_TOLS
        )
    )
    print(
        "BOX candidate output change fractions: "
        + ", ".join(
            f">{tol:g}={oc['box_query_fraction_above_tol'][str(tol)]:.2%}"
            for tol in CHANGE_TOLS
        )
    )
    print(
        f"Top100 pair-set changed frames: "
        f"{diagnostics['top100_pair_changed_frame_fraction']:.2%}; "
        f"operational pair-set changed frames: "
        f"{diagnostics['operational_pair_changed_frame_fraction']:.2%}"
    )
    print("=" * 118)

    print("\n" + "=" * 118)
    print("RUN_09D TABLE 4 -- PER-CLASS AP@0.30")
    print("=" * 118)
    print(f"{'Class':<20} | {'Support':>8} | {'Base':>8} | {'Full':>8} | {'Delta':>8}")
    print("-" * 118)
    for i, name in enumerate(class_names):
        b = results["Base"]["per_class_ap30"][name]
        f = results["Full"]["per_class_ap30"][name]
        print(
            f"{name:<20} | {class_support[i]:8,d} | {b:8.4f} | {f:8.4f} | {f-b:+8.4f}"
        )
    print("=" * 118)

    # Leave-one-out contribution relative to Full.
    leave_one_out = {}
    full = results["Full"]
    for e in EXPERTS:
        m = results[f"Minus{e}"]
        leave_one_out[e] = {
            "delta_map30_without_expert_vs_full": m["map30"] - full["map30"],
            "delta_f1_without_expert_vs_full": m["f1"] - full["f1"],
            "delta_tp_without_expert_vs_full": m["tp"] - full["tp"],
            "delta_fp_without_expert_vs_full": m["fp"] - full["fp"],
        }

    report = {
        "_meta": {
            "schema_version": "bfcef_f_run09d_calibrated_integrated_qualification_v1",
            "driver_sha256": sha256(Path(__file__)),
            "calibrated_checkpoint_sha256": EXPECTED_CAL_CKPT_SHA,
            "run09c_report_sha256": EXPECTED_CAL_REPORT_SHA,
            "meta_fold": META_FOLD,
            "heldout_videos": EXPECTED_FOLD0_VIDEOS,
            "read_only": True,
            "no_training": True,
            "no_calibration": True,
            "no_val_test": True,
            "runtime_minutes": (time.time() - t0) / 60.0,
            "evaluator_semantics": {
                "ranking": "sigmoid logits -> flattened 300x7 -> global Top100",
                "coco_predictions": "all Top100",
                "operational_score_threshold": 0.20,
                "operational_iou_threshold": 0.30,
                "operational_match": "same-class greedy",
            },
        },
        "class_names": class_names,
        "class_support_fold0": {
            class_names[i]: int(class_support[i]) for i in range(len(class_names))
        },
        "variants": results,
        "transitions": transitions,
        "leave_one_out_vs_full": leave_one_out,
        "full_mechanism_diagnostics": diagnostics,
    }

    report_path = OUT / "run09d_calibrated_integrated_qualification.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    csv_path = OUT / "run09d_variant_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "variant", "map30", "delta_map30", "mapca30",
            "ap_s30", "ap_m30", "ap_l30",
            "tp", "fp", "fn", "precision", "recall", "f1", "delta_f1",
            "rescued", "lost", "retained", "fp_added", "fp_removed",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for v in VARIANTS:
            m = results[v]
            t = transitions.get(v, {})
            w.writerow({
                "variant": v,
                "map30": m["map30"],
                "delta_map30": m["map30"] - base_r["map30"],
                "mapca30": m["mapca30"],
                "ap_s30": m["ap_s30"],
                "ap_m30": m["ap_m30"],
                "ap_l30": m["ap_l30"],
                "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
                "precision": m["precision"], "recall": m["recall"],
                "f1": m["f1"], "delta_f1": m["f1"] - base_r["f1"],
                "rescued": t.get("rescued", ""),
                "lost": t.get("lost", ""),
                "retained": t.get("retained", ""),
                "fp_added": t.get("fp_added", ""),
                "fp_removed": t.get("fp_removed", ""),
            })

    print("\n" + "=" * 126)
    print("[PASS] RUN_09D CALIBRATED INTEGRATED MECHANISM QUALIFICATION COMPLETE")
    print(f"[+] Report: {report_path}")
    print(f"[+] Report SHA: {sha256(report_path)}")
    print(f"[+] CSV: {csv_path}")
    print(f"[+] CSV SHA: {sha256(csv_path)}")
    print("[+] No training, no calibration, no Val/Test access.")
    print("=" * 126)


if __name__ == "__main__":
    main()
