from __future__ import annotations

"""
B-FCEF-F RUN_09C
================
Leakage-safe hard-threshold calibration for the completed RUN_09B meta-fold-0 pilot.

Why this exists
---------------
RUN_09B trained with continuous g_soft but inference uses:

    mask_hard = 1[g_soft > mask_thresh]
    g_hard    = mask_hard * g_soft

The completed meta-fold-0 pilot reached 100% hard bypass because the inherited
0.10 thresholds never opened. This script asks the decisive question:

    Did g_soft learn useful ranking below 0.10?

Calibration rule
----------------
For each of the 8 permanent task gates (S/L/C/T x cls/box):

  * collect g_soft ONLY on the meta-training videos (folds 1-4)
  * collect the exact Final local signed consequence target:
        net_cls = U_cls - H_cls
        net_box = U_box - H_box
  * choose the highest threshold that maximizes cumulative signed local utility
        sum(net | g_soft > threshold)
    (highest threshold wins ties, preserving sparsity)
  * if no positive cumulative utility exists, leave that gate effectively closed

Fold 0 labels/outcomes are NEVER used to choose thresholds.

After calibration:
  * set only the eight mask_thresh values
  * perform exact hard-gate Fold-0 evaluation
  * save a calibrated downstream checkpoint + report

No training, no Val/Test, no architecture change.
"""

import gc
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List
from PIL import Image

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("R0_MSDA_BACKEND", "pytorch")

import torch

ROOT = Path("D:/MARIVISION")
EVAL = ROOT / "H0_workbench" / "eval_tools"
RUN09B_DIR = EVAL / "bfcef_f_s0" / "run09_joint_uh_gate_router_pilot_v1"
RUN09A_DIR = EVAL / "bfcef_f_s0" / "run09_final_local_consequences_v1"

PILOT_CKPT = RUN09B_DIR / "meta_fold0_uh_gate_router.pth"
CONSEQUENCE_MANIFEST = RUN09A_DIR / "run09_final_consequence_manifest.json"

EXPECTED_PILOT_SHA = "255adc392398e24838c7c76395832be5790e51dbe1b99423b2f7b1f41ddbb458"
EXPECTED_CONSEQUENCE_SHA = "f6dcf39a073d7c9fe27201dbe8c38d658cb1b0093f5ba1797a9740314dc70b72"

OUT_DIR = EVAL / "bfcef_f_s0" / "run09c_threshold_calibration_v1"
REPORT_PATH = OUT_DIR / "run09c_threshold_calibration_report.json"
CALIBRATED_CKPT = OUT_DIR / "meta_fold0_uh_gate_router_calibrated.pth"

EXPERTS = ("S", "L", "C", "T")
META_FOLD = 0
BINS = 10_000

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if str(EVAL) not in sys.path:
    sys.path.insert(0, str(EVAL))

import train_bfcef_f_run09_joint_uh_gate_router as r9  # noqa: E402


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


def load_downstream_checkpoint(model, payload: dict) -> None:
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


def empty_stat():
    return {
        "count": torch.zeros(BINS + 1, dtype=torch.long),
        "net_sum": torch.zeros(BINS + 1, dtype=torch.float64),
        "pos_count": torch.zeros(BINS + 1, dtype=torch.long),
        "neg_count": torch.zeros(BINS + 1, dtype=torch.long),
    }


def update_stat(stat, g: torch.Tensor, net: torch.Tensor):
    g = g.detach().float().cpu().reshape(-1).clamp(0.0, 1.0)
    net = net.detach().float().cpu().reshape(-1)

    idx = torch.clamp((g * BINS).floor().long(), min=0, max=BINS)

    stat["count"] += torch.bincount(idx, minlength=BINS + 1)
    stat["net_sum"] += torch.bincount(
        idx, weights=net.double(), minlength=BINS + 1
    )
    stat["pos_count"] += torch.bincount(
        idx, weights=(net > 0).long(), minlength=BINS + 1
    ).long()
    stat["neg_count"] += torch.bincount(
        idx, weights=(net < 0).long(), minlength=BINS + 1
    ).long()


def quantile_from_hist(count: torch.Tensor, q: float) -> float:
    total = int(count.sum().item())
    if total <= 0:
        return 0.0
    target = max(1, int(math.ceil(q * total)))
    cs = torch.cumsum(count, dim=0)
    idx = int(torch.nonzero(cs >= target, as_tuple=False)[0].item())
    return idx / BINS


def calibrate_stat(stat) -> dict:
    count = stat["count"]
    net_sum = stat["net_sum"]
    pos = stat["pos_count"]
    neg = stat["neg_count"]

    # threshold t=i/BINS activates bins i+1...BINS because inference is g > t
    cum_net = torch.flip(torch.cumsum(torch.flip(net_sum, dims=[0]), dim=0), dims=[0])
    cum_count = torch.flip(torch.cumsum(torch.flip(count, dims=[0]), dim=0), dims=[0])
    cum_pos = torch.flip(torch.cumsum(torch.flip(pos, dims=[0]), dim=0), dims=[0])
    cum_neg = torch.flip(torch.cumsum(torch.flip(neg, dims=[0]), dim=0), dims=[0])

    best_utility = 0.0
    best_i = BINS  # closed
    best_count = 0
    best_pos = 0
    best_neg = 0

    # i threshold activates >i, so use cumulative at i+1.
    for i in range(BINS - 1, -1, -1):
        utility = float(cum_net[i + 1].item())
        n = int(cum_count[i + 1].item())
        p = int(cum_pos[i + 1].item())
        ng = int(cum_neg[i + 1].item())

        # Highest threshold wins utility ties -> sparse/conservative.
        if utility > best_utility + 1e-12:
            best_utility = utility
            best_i = i
            best_count = n
            best_pos = p
            best_neg = ng

    if best_utility <= 0.0 or best_count == 0:
        threshold = 0.9999
        status = "CLOSED_NO_POSITIVE_TRAIN_UTILITY"
    else:
        threshold = best_i / BINS
        status = "CALIBRATED_OPEN"

    total = int(count.sum().item())
    max_bin = int(torch.nonzero(count > 0, as_tuple=False)[-1].item()) if total else 0

    return {
        "threshold": float(threshold),
        "status": status,
        "best_cumulative_signed_utility": float(best_utility),
        "activated_train_queries": int(best_count),
        "activated_train_fraction": best_count / max(1, total),
        "positive_target_queries_among_active": int(best_pos),
        "negative_target_queries_among_active": int(best_neg),
        "active_positive_fraction": best_pos / max(1, best_pos + best_neg),
        "g_soft_distribution": {
            "p50": quantile_from_hist(count, 0.50),
            "p90": quantile_from_hist(count, 0.90),
            "p95": quantile_from_hist(count, 0.95),
            "p99": quantile_from_hist(count, 0.99),
            "max_approx": max_bin / BINS,
        },
        "total_train_candidate_queries": total,
    }


def save_calibrated_checkpoint(model, original_payload, calibration, eval_result):
    payload = {
        "_meta": dict(original_payload["_meta"]),
        "calibration": calibration,
        "heldout_eval_after_calibration": eval_result,
        "source_pilot_checkpoint_sha256": EXPECTED_PILOT_SHA,
        "run09_consequence_manifest_sha256": EXPECTED_CONSEQUENCE_SHA,
    }
    payload["_meta"]["schema_version"] = (
        "bfcef_f_run09c_calibrated_meta_fold0_v1"
    )

    payload["self_assessment"] = {
        e: {k: v.detach().cpu() for k, v in model.self_assessment[e].state_dict().items()}
        for e in EXPERTS
    }
    payload["gate_cls"] = {
        e: {k: v.detach().cpu() for k, v in model.gate_cls[e].state_dict().items()}
        for e in EXPERTS
    }
    payload["gate_box"] = {
        e: {k: v.detach().cpu() for k, v in model.gate_box[e].state_dict().items()}
        for e in EXPERTS
    }
    payload["coord_cls"] = {
        k: v.detach().cpu() for k, v in model.coord_cls.state_dict().items()
    }
    payload["coord_box"] = {
        k: v.detach().cpu() for k, v in model.coord_box.state_dict().items()
    }

    torch.save(payload, CALIBRATED_CKPT)


def main():
    print("=" * 118)
    print("B-FCEF-F RUN_09C -- LEAKAGE-SAFE HARD-THRESHOLD CALIBRATION")
    print("META-TRAIN FOLDS 1-4 ONLY FOR CALIBRATION | FOLD 0 HELD OUT")
    print("NO TRAINING | NO VAL/TEST | NO ARCHITECTURE CHANGE")
    print("=" * 118)

    require_sha(PILOT_CKPT, EXPECTED_PILOT_SHA, "RUN_09B pilot checkpoint")
    require_sha(
        CONSEQUENCE_MANIFEST,
        EXPECTED_CONSEQUENCE_SHA,
        "RUN_09A consequence manifest",
    )

    folds = r9.parse_folds()
    v2f = r9.video_fold_map(folds)
    train_videos = sorted(set(range(1, 48)) - set(folds[META_FOLD]))

    (
        model,
        criterion,
        CanonicalObjectState,
        frame_inputs_from_records,
    ) = r9.initialize_runtime()

    ckpt = torch.load(PILOT_CKPT, map_location="cpu")
    load_downstream_checkpoint(model, ckpt)
    model.eval()

    original_thresholds = {
        "cls": {
            e: float(model.gate_cls[e].mask_thresh.detach().cpu().item())
            for e in EXPERTS
        },
        "box": {
            e: float(model.gate_box[e].mask_thresh.detach().cpu().item())
            for e in EXPERTS
        },
    }

    print(f"[+] Original cls thresholds: {original_thresholds['cls']}")
    print(f"[+] Original box thresholds: {original_thresholds['box']}")

    stats = {
        "cls": {e: empty_stat() for e in EXPERTS},
        "box": {e: empty_stat() for e in EXPERTS},
    }

    with torch.no_grad():
        for source_fold in sorted({v2f[v] for v in train_videos}):
            r9.load_source_fold(model, source_fold)
            vids = [v for v in train_videos if v2f[v] == source_fold]

            for vid in vids:
                canonical, spatial, temporal, labels = r9.load_video_all(
                    vid, source_fold
                )

                for idx in range(len(canonical)):
                    O, X = r9.s0a.make_O_X(
                        canonical[idx],
                        spatial[idx],
                        temporal[idx],
                        CanonicalObjectState,
                        frame_inputs_from_records,
                    )
                    bundle = r9.expert_bundle(model, O, X)
                    out = r9.mechanism_forward(model, O, X, bundle, hard=False)

                    q_idx = labels[idx]["candidate_queries"].long().to(DEVICE)

                    for k, e in enumerate(EXPERTS):
                        ex = labels[idx]["experts"][e]
                        net_cls = (
                            ex["u_cls_target"].float()
                            - ex["h_cls_target"].float()
                        ).to(DEVICE)
                        net_box = (
                            ex["u_box_target"].float()
                            - ex["h_box_target"].float()
                        ).to(DEVICE)

                        update_stat(
                            stats["cls"][e],
                            out["g_cls"][q_idx, k],
                            net_cls,
                        )
                        update_stat(
                            stats["box"][e],
                            out["g_box"][q_idx, k],
                            net_box,
                        )

                print(
                    f"    [+] collected calibration scores vid={vid:02d} "
                    f"(source fold {source_fold})"
                )

                del canonical, spatial, temporal, labels
                gc.collect()
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

    calibration = {
        "meta_fold": META_FOLD,
        "calibration_videos": train_videos,
        "heldout_videos": sorted(folds[META_FOLD]),
        "selection_rule": (
            "highest threshold maximizing cumulative signed local utility "
            "sum(U-H | g_soft > threshold); fold0 excluded"
        ),
        "original_thresholds": original_thresholds,
        "cls": {},
        "box": {},
    }

    print("=" * 118)
    print("CALIBRATION RESULTS")
    print("=" * 118)

    for task in ("cls", "box"):
        for e in EXPERTS:
            result = calibrate_stat(stats[task][e])
            calibration[task][e] = result
            print(
                f"[+] {task.upper()} {e}: "
                f"old={original_thresholds[task][e]:.4f} "
                f"new={result['threshold']:.4f} "
                f"status={result['status']} "
                f"active_train={result['activated_train_fraction']:.4%} "
                f"p95={result['g_soft_distribution']['p95']:.4f} "
                f"p99={result['g_soft_distribution']['p99']:.4f} "
                f"max~={result['g_soft_distribution']['max_approx']:.4f}"
            )

    # Apply ONLY train-derived thresholds.
    for e in EXPERTS:
        model.gate_cls[e].mask_thresh.data.fill_(
            float(calibration["cls"][e]["threshold"])
        )
        model.gate_box[e].mask_thresh.data.fill_(
            float(calibration["box"][e]["threshold"])
        )

    print("=" * 118)
    print("HELD-OUT FOLD-0 HARD EVALUATION WITH TRAIN-DERIVED THRESHOLDS")
    print("=" * 118)

    eval_result = r9.hard_eval_fold0(
        model,
        CanonicalObjectState,
        frame_inputs_from_records,
        folds,
        v2f,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_calibrated_checkpoint(
        model, ckpt, calibration, eval_result
    )

    report = {
        "_meta": {
            "schema_version": "bfcef_f_run09c_threshold_calibration_v1",
            "source_pilot_checkpoint_sha256": EXPECTED_PILOT_SHA,
            "run09_consequence_manifest_sha256": EXPECTED_CONSEQUENCE_SHA,
            "calibration_used_fold0_labels": False,
            "no_training": True,
        },
        "calibration": calibration,
        "fold0_hard_eval": eval_result,
        "calibrated_checkpoint": {
            "path": str(CALIBRATED_CKPT),
            "sha256": sha256(CALIBRATED_CKPT),
        },
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 118)
    print("[PASS] RUN_09C THRESHOLD CALIBRATION + HELD-OUT EVALUATION COMPLETE")
    print(
        "[+] Fold0 Base:       "
        f"TP={eval_result['base']['tp']:,} "
        f"FP={eval_result['base']['fp']:,} "
        f"FN={eval_result['base']['fn']:,} "
        f"F1={eval_result['base']['f1']:.6f}"
    )
    print(
        "[+] Fold0 Integrated: "
        f"TP={eval_result['integrated']['tp']:,} "
        f"FP={eval_result['integrated']['fp']:,} "
        f"FN={eval_result['integrated']['fn']:,} "
        f"F1={eval_result['integrated']['f1']:.6f}"
    )
    print(f"[+] Hard bypass fraction: {eval_result['bypass_fraction']:.4%}")
    print(f"[+] Active cls by expert: {eval_result['active_cls_by_expert']}")
    print(f"[+] Active box by expert: {eval_result['active_box_by_expert']}")
    print(f"[+] Report: {REPORT_PATH}")
    print(f"[+] Report SHA: {sha256(REPORT_PATH)}")
    print(f"[+] Calibrated checkpoint SHA: {sha256(CALIBRATED_CKPT)}")
    print("[+] Fold0 labels were NOT used for calibration.")
    print("=" * 118)


if __name__ == "__main__":
    main()
