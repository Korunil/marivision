
from __future__ import annotations

"""
RUN_09F Stage 1: Train-only learned Editor pilot.

Frozen proposal: RUN09C calibrated Full.
Control: RUN09D Base/Full.
Train Editor: source folds 1-3.
Inner diagnostic: source fold 4.
Pilot evaluation: source fold 0, once.

No Validation or Test.
"""

import argparse
import gc
import hashlib
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("R0_MSDA_BACKEND", "pytorch")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path("D:/MARIVISION")
EVAL = ROOT / "H0_workbench" / "eval_tools"
S0 = EVAL / "bfcef_f_s0"
RUN09C = S0 / "run09c_threshold_calibration_v1"
CKPT = RUN09C / "meta_fold0_uh_gate_router_calibrated.pth"
STAGE0 = (
    S0 / "run09f_final_resilient_pilot_v1"
    / "run09f_stage0_preflight.json"
)
OUT = S0 / "run09f_editor_pilot_v1"

EXPERTS = ("S", "L", "C", "T")

CKPT_SHA = (
    "a1debfaadcc7f754319c63386ac93d993dc2a036cefbdf4e1ee70a567b517deb"
)
STAGE0_SHA = (
    "50544a414132c7d1cecea87c066c8ed4fb4aefa14a5065854959f2c47646c5af"
)
SOURCE_SHA = (
    "1c2569f1871b475e4aa71d20e505c8bb738d9b3ed7d4df185787c1c5546c6001"
)

F0_BASE = {"tp": 8918, "fp": 17137, "fn": 6177}
F0_FULL = {"tp": 8861, "fp": 13509, "fn": 6234}

PCLS, RCLS = 16, 8
PBOX, RBOX = 11, 8

if str(EVAL) not in sys.path:
    sys.path.insert(0, str(EVAL))

# Permit pure synthetic tests without local SeaClips imports.
if "--self-test-only" not in sys.argv:
    import train_bfcef_f_run09_joint_uh_gate_router as r9
    import run_bfcef_f_run08_oof_counterfactual_audit as r8


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gate(path, expected):
    assert path.exists(), f"Missing {path}"
    actual = sha(path)
    assert actual == expected, (
        f"SHA DRIFT: {path}\n"
        f"expected={expected}\nactual={actual}"
    )


def read_only_inputs():
    gate(CKPT, CKPT_SHA)
    gate(STAGE0, STAGE0_SHA)
    gate(
        r9.s0a.OUT / "final_expert_oof_manifest.json",
        SOURCE_SHA,
    )
    report = json.loads(STAGE0.read_text())
    inventory = report["inventory_reconciliation"]

    assert inventory["cached_vs_raw_seen_discrepancy"] == 0
    assert not inventory["sample_mismatched_images"]
    assert report["replay_failure_count"] == 0
    assert report["total_consequence_replays"] == 310

    print("[PASS] Immutable RUN09C, Stage0 and RUN07 provenance")


def initialize():
    model, _, CO, FX = r9.initialize_runtime()
    payload = torch.load(
        CKPT, map_location="cpu", weights_only=False
    )
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
    model.coord_cls.load_state_dict(
        payload["coord_cls"], strict=True
    )
    model.coord_box.load_state_dict(
        payload["coord_box"], strict=True
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model, CO, FX


def pi_or_baseline(info, key):
    result = info.get(key)
    if result is None:
        assert info.get("bypassed", False), (
            "Router mass missing outside exact bypass."
        )
        pi = torch.zeros(300, 5)
        pi[:, 0] = 1.0
        return pi
    return result.detach().cpu().float()


def match(z, b, frame):
    prediction = r8.extract_top100_predictions(z, b)
    result = r8.match_operational_detections_to_gt(
        prediction["op_scores"],
        prediction["op_c_idx"],
        prediction["op_boxes"],
        prediction["op_q_idx"],
        frame["gt_boxes_norm"].float(),
        frame["gt_labels"].long(),
    )
    return prediction, result


def utility(accepted, reverted):
    """Global utility of keeping proposed action vs reverting."""
    ma = set(accepted["matched_gt_indices"])
    mr = set(reverted["matched_gt_indices"])
    fa = set(accepted["fp_dict"])
    fr = set(reverted["fp_dict"])

    rescued = len(ma - mr)
    lost = len(mr - ma)
    removed_fp = len(fr - fa)
    added_fp = len(fa - fr)

    score = (
        4 * (rescued - lost)
        + removed_fp - added_fp
    )
    return score, (rescued, lost, removed_fp, added_fp)


def features(z0, b0, z1, b1, oof, pic, pib, q, c=None):
    """Inference-available features only: no labels/GT."""
    if c is not None:
        a = float(z0[q, c].sigmoid())
        d = float(z1[q, c].sigmoid())
        box = b0[q]

        proposer = [
            a, d, d-a, abs(d-a),
            float(z0[q].sigmoid().max()),
            float(box[2] * box[3]),
            float(box[2] / box[3].clamp_min(1e-5)),
            float(pic[q, 0]),
        ]
        proposer += [float(i == c) for i in range(7)]
        proposer += [float(z1[q].sigmoid().max())]

        dif = (
            oof["expert_cls_logits"][:, q, c].float()
            - z0[q, c]
        ).clamp(-10, 10)
        reviewer = [float(torch.tanh(x / 3)) for x in dif]
        reviewer += [float(x > 0) for x in dif]

        assert len(proposer) == PCLS
        assert len(reviewer) == RCLS

    else:
        delta = (b1[q] - b0[q]).float()
        box = b0[q]
        proposer = [
            float(z0[q].sigmoid().max()),
            float(z1[q].sigmoid().max()),
            float(box[2] * box[3]),
            float(box[2] / box[3].clamp_min(1e-5)),
        ]
        proposer += delta.tolist()
        proposer += [
            float(delta.abs().max()),
            float(pib[q, 0]),
            float((delta ** 2).sum().sqrt()),
        ]

        dif = (
            oof["expert_box_boxes"][:, q].float()
            - b0[q]
        )
        reviewer = [
            float(dif[k].abs().max()) for k in range(4)
        ]
        reviewer += [
            float(dif[k, 0]) for k in range(4)
        ]
        assert len(proposer) == PBOX
        assert len(reviewer) == RBOX

    return proposer, reviewer


class Editor(nn.Module):
    def __init__(self, task, collaborative):
        super().__init__()
        self.task = task
        self.collaborative = collaborative
        dim = PCLS if task == "cls" else PBOX

        if collaborative:
            reviewer_dim = RCLS if task == "cls" else RBOX
            self.review = nn.Sequential(
                nn.Linear(reviewer_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
            )
            dim += 8

        self.editor = nn.Sequential(
            nn.Linear(dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, p, r):
        if self.collaborative:
            p = torch.cat([p, self.review(r)], dim=-1)
        return self.editor(p).flatten()


def fit(task, collaborative, examples, epochs=18):
    train = [x for x in examples if x[4] in (1, 2, 3)]
    positives = sum(x[2] for x in train)
    negatives = len(train) - positives

    if len(train) < 60 or min(positives, negatives) < 15:
        print(
            f"[WARN] {task} supervision insufficient: "
            f"N={len(train)}, pos={positives}, neg={negatives}. "
            "Preserving Full for this task."
        )
        return None, {
            "train": len(train),
            "pos": positives,
            "neg": negatives,
            "bypass": True,
        }

    torch.manual_seed(
        20261008 + 3 * (task == "box") + int(collaborative)
    )
    model = Editor(task, collaborative)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=8e-4, weight_decay=1e-4
    )

    p = torch.tensor(
        [x[0] for x in train], dtype=torch.float32
    )
    r = torch.tensor(
        [x[1] for x in train], dtype=torch.float32
    )
    y = torch.tensor(
        [x[2] for x in train], dtype=torch.float32
    )

    pm, ps = p.mean(0), p.std(0).clamp_min(0.05)
    if task == "cls":
        pm[8:15] = 0.0
        ps[8:15] = 1.0
    rm, rs = r.mean(0), r.std(0).clamp_min(0.05)

    model.register_buffer("pm", pm)
    model.register_buffer("ps", ps)
    model.register_buffer("rm", rm)
    model.register_buffer("rs", rs)

    p, r = (p-pm)/ps, (r-rm)/rs
    weight = torch.tensor([
        min(4.0, max(0.25, negatives/max(1, positives)))
    ])

    indices = torch.arange(len(y))
    for epoch in range(epochs):
        generator = torch.Generator().manual_seed(
            20261008 + epoch
        )
        shuffle = indices[
            torch.randperm(len(indices), generator=generator)
        ]
        for ids in shuffle.split(512):
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(
                model(p[ids], r[ids]), y[ids],
                pos_weight=weight,
            )
            loss.backward()
            optimizer.step()

    model.eval()
    assert all(
        torch.isfinite(v).all() for v in model.parameters()
    )
    return model, {
        "train": len(train),
        "pos": positives,
        "neg": negatives,
        "bypass": False,
    }


def probs(editor, rows):
    if editor is None:
        return None
    if not rows:
        return torch.zeros(0)
    p = torch.tensor(
        [x[0] for x in rows], dtype=torch.float32
    )
    r = torch.tensor(
        [x[1] for x in rows], dtype=torch.float32
    )
    with torch.no_grad():
        return editor(
            (p-editor.pm)/editor.ps,
            (r-editor.rm)/editor.rs,
        ).sigmoid()


def sample_indices(z0, b0, z1, b1, queries, task, n):
    if task == "cls":
        dif = (z1.sigmoid()-z0.sigmoid()).abs()
        candidate = []

        for q in queries:
            for c in range(7):
                delta = float(dif[q, c])
                near = min(
                    abs(float(z0[q, c].sigmoid())-0.20),
                    abs(float(z1[q, c].sigmoid())-0.20),
                )
                if delta > 1e-7:
                    candidate.append((q, c, delta, near))

        selected = sorted(
            candidate, key=lambda x: -x[2]
        )[:n//2]
        selected += sorted(
            candidate, key=lambda x: (x[3], -x[2])
        )[:n-n//2]
        return list(dict.fromkeys(
            (x[0], x[1]) for x in selected
        ))

    candidate = [
        (q, float((b1[q]-b0[q]).abs().max()))
        for q in queries
    ]
    return [
        (q, None)
        for q, delta in sorted(
            candidate, key=lambda x: -x[1]
        )[:n]
        if delta > 1e-7
    ]


def collect_one(
    frame, oof, z0, b0, z1, b1, info,
    fold, video, quota
):
    _, full = match(z1, b1, frame)
    pic = info["pi_cls"]
    pib = info["pi_box"]
    queries = [
        int(q) for q in oof["candidate_queries"].tolist()
    ]
    rows = {"cls": [], "box": []}

    for task in ("cls", "box"):
        for q, c in sample_indices(
            z0, b0, z1, b1, queries, task, quota
        ):
            zrev, brev = z1.clone(), b1.clone()
            if task == "cls":
                zrev[q, c] = z0[q, c]
            else:
                brev[q] = b0[q]

            _, reverted = match(zrev, brev, frame)
            score, consequence = utility(full, reverted)
            if score == 0:
                continue

            p, r = features(
                z0, b0, z1, b1, oof, pic, pib, q, c
            )
            rows[task].append((
                p, r, int(score > 0),
                video, fold, consequence
            ))
    return rows


def run_data(model, CO, FX, folds, max_per_video, quota):
    gathered = {"cls": [], "box": []}
    stats = Counter()

    with torch.no_grad():
        for fold in (1, 2, 3, 4):
            r9.load_source_fold(model, fold)
            for video in folds[fold]:
                can, spa, tmp, _ = r9.load_video_all(
                    video, fold
                )
                oof = torch.load(
                    r9.s0a.fold_oof_path(fold, video),
                    map_location="cpu",
                    weights_only=False,
                )
                assert len(can) == len(oof)

                count = min(max_per_video, len(can))
                indices = np.linspace(
                    0, len(can)-1, count, dtype=int
                )

                for idx in sorted(set(indices.tolist())):
                    O, X = r9.s0a.make_O_X(
                        can[idx], spa[idx], tmp[idx],
                        CO, FX
                    )
                    z, b, info = model(
                        O, X,
                        m_ext=None,
                        m_hat_vis=None,
                        m_avail=None,
                        context_mode="auto",
                    )

                    z0 = O.z.detach().cpu().float()
                    b0 = O.b.detach().cpu().float()
                    z1 = z.detach().cpu().float()
                    b1 = b.detach().cpu().float()
                    info = {
                        "pi_cls": pi_or_baseline(info, "pi_cls"),
                        "pi_box": pi_or_baseline(info, "pi_box"),
                    }

                    rows = collect_one(
                        can[idx], oof[idx],
                        z0, b0, z1, b1, info,
                        fold, video, quota
                    )
                    for task in rows:
                        gathered[task].extend(rows[task])
                        stats[f"fold{fold}_{task}"] += len(rows[task])

                print(
                    f"[+] Supervision fold={fold} "
                    f"video={video:02d}, frames={len(indices)}"
                )
                del can, spa, tmp, oof
                gc.collect()

    return gathered, dict(stats)


def apply_editors(
    z0, b0, z1, b1, oof, info, editors,
    threshold=0.25,
):
    z, b = z1.clone(), b1.clone()
    queries = [
        int(q) for q in oof["candidate_queries"].tolist()
    ]

    # Restrict review to the union of endpoint Top100 pairs.
    pred0 = r8.extract_top100_predictions(z0, b0)
    pred1 = r8.extract_top100_predictions(z1, b1)
    relevant = set(zip(
        pred0["all_q_idx"].tolist(),
        pred0["all_c_idx"].tolist(),
    ))
    relevant.update(zip(
        pred1["all_q_idx"].tolist(),
        pred1["all_c_idx"].tolist(),
    ))
    relevant_q = {q for q, _ in relevant}

    pic = pi_or_baseline(info, "pi_cls")
    pib = pi_or_baseline(info, "pi_box")

    counters = {
        "cls_reject": 0,
        "box_reject": 0,
        "cls_eligible": 0,
        "box_eligible": 0,
    }

    for task in ("cls", "box"):
        editor = editors[task]
        if editor is None:
            continue

        slots, rows = [], []

        if task == "cls":
            for q in queries:
                for c in range(7):
                    if (q, c) not in relevant:
                        continue
                    if abs(float(z1[q, c]-z0[q, c])) < 1e-7:
                        continue
                    slots.append((q, c))
                    p, r = features(
                        z0, b0, z1, b1, oof, pic, pib, q, c
                    )
                    rows.append((p, r, None, 0, 0, None))
        else:
            for q in queries:
                if q not in relevant_q:
                    continue
                if float((b1[q]-b0[q]).abs().max()) < 1e-7:
                    continue
                slots.append((q, None))
                p, r = features(
                    z0, b0, z1, b1, oof, pic, pib, q, None
                )
                rows.append((p, r, None, 0, 0, None))

        if not rows:
            continue

        confidence = probs(editor, rows)
        counters[f"{task}_eligible"] += len(slots)

        for (q, c), approval in zip(
            slots, confidence.tolist()
        ):
            if approval < threshold:
                if task == "cls":
                    z[q, c] = z0[q, c]
                else:
                    b[q] = b0[q]
                counters[f"{task}_reject"] += 1

    return z, b, counters


def metrics(counts):
    t, f, n = counts["tp"], counts["fp"], counts["fn"]
    return {
        "tp": t, "fp": f, "fn": n,
        "precision": t/max(t+f, 1),
        "recall": t/max(t+n, 1),
        "f1": 2*t/max(2*t+f+n, 1),
    }


def evaluate_fold0(model, CO, FX, folds, editors, threshold):
    coco, coco_ca, size, _, _, cat, classes = r8.load_metadata()
    variants = (
        "Base", "Full",
        "ProposerEditor", "CollaborativeEditor"
    )
    evaluators = {
        v: r8.make_evaluators(coco, coco_ca)
        for v in variants
    }
    counts = {v: Counter() for v in variants}
    events = {
        v: Counter() for v in variants if v != "Base"
    }
    interventions = {
        v: Counter()
        for v in ("ProposerEditor", "CollaborativeEditor")
    }

    assert sorted(folds[0]) == [
        6, 7, 14, 16, 17, 24, 36, 37, 40, 42
    ]
    r9.load_source_fold(model, 0)

    with torch.no_grad():
        for video in folds[0]:
            can, spa, tmp, _ = r9.load_video_all(video, 0)
            oof = torch.load(
                r9.s0a.fold_oof_path(0, video),
                map_location="cpu",
                weights_only=False,
            )
            prediction_maps = {v: {} for v in variants}
            agnostic_maps = {v: {} for v in variants}

            for i in range(len(can)):
                fr = can[i]
                O, X = r9.s0a.make_O_X(
                    fr, spa[i], tmp[i], CO, FX
                )
                z, b, info = model(
                    O, X,
                    m_ext=None,
                    m_hat_vis=None,
                    m_avail=None,
                    context_mode="auto",
                )

                z0 = O.z.detach().cpu().float()
                b0 = O.b.detach().cpu().float()
                z1 = z.detach().cpu().float()
                b1 = b.detach().cpu().float()

                outputs = {
                    "Base": (z0, b0),
                    "Full": (z1, b1),
                }
                for mode in (
                    "ProposerEditor",
                    "CollaborativeEditor",
                ):
                    ze, be, summary = apply_editors(
                        z0, b0, z1, b1,
                        oof[i], info, editors[mode], threshold
                    )
                    outputs[mode] = (ze, be)
                    interventions[mode].update(summary)

                matches = {}
                for variant in variants:
                    zi, bi = outputs[variant]
                    pred, match_result = match(zi, bi, fr)
                    matches[variant] = match_result

                    counts[variant].update({
                        "tp": match_result["tp"],
                        "fp": match_result["fp"],
                        "fn": match_result["fn"],
                    })
                    img = int(fr["image_id"])
                    W, H = size[img]
                    r8.add_coco_prediction(
                        prediction_maps[variant],
                        agnostic_maps[variant],
                        img, pred, W, H, cat,
                    )

                for variant in events:
                    a = matches[variant]
                    baseline = matches["Base"]
                    fa = set(a["fp_dict"])
                    fb = set(baseline["fp_dict"])

                    events[variant].update({
                        "rescued": len(
                            a["matched_gt_indices"]
                            - baseline["matched_gt_indices"]
                        ),
                        "lost": len(
                            baseline["matched_gt_indices"]
                            - a["matched_gt_indices"]
                        ),
                        "fp_add": len(fa-fb),
                        "fp_remove": len(fb-fa),
                    })

            for variant in variants:
                ev, ev_ca = evaluators[variant]
                ev.update(prediction_maps[variant])
                ev_ca.update(agnostic_maps[variant])

            print(
                f"[+] Fold0 evaluated video {video:02d}: "
                f"{len(can)} frames"
            )
            del can, spa, tmp, oof
            gc.collect()

    result = {}
    for variant in variants:
        ev, ev_ca = evaluators[variant]
        stats = r8.quiet_finalize_evaluator(ev)
        stats_ca = r8.quiet_finalize_evaluator(ev_ca)

        result[variant] = {
            **metrics(counts[variant]),
            "map30": float(stats[0]),
            "mapca30": float(stats_ca[0]),
            "ap_s30": float(stats[3]),
            "ap_m30": float(stats[4]),
            "ap_l30": float(stats[5]),
            "per_class_ap30": dict(zip(
                classes, r8.per_class_ap(ev)
            )),
            "events": (
                dict(events[variant]) if variant in events else {}
            ),
        }

    assert all(
        result["Base"][k] == v for k, v in F0_BASE.items()
    ), "RUN09D BASE identity failed"
    assert all(
        result["Full"][k] == v for k, v in F0_FULL.items()
    ), "RUN09D FULL identity failed"

    assert abs(result["Base"]["map30"]-0.7592) < 0.0003
    assert abs(result["Full"]["map30"]-0.7595) < 0.0003

    return result, {
        k: dict(v) for k, v in interventions.items()
    }


def self_test():
    assert utility(
        {"matched_gt_indices": {1}, "fp_dict": {}},
        {"matched_gt_indices": set(), "fp_dict": {(3,2): 0}},
    )[0] == 5
    assert utility(
        {"matched_gt_indices": set(), "fp_dict": {(3,2): 0}},
        {"matched_gt_indices": {1}, "fp_dict": {}},
    )[0] == -5

    editor = Editor("cls", True)
    assert editor(
        torch.randn(2, PCLS), torch.randn(2, RCLS)
    ).shape == (2,)

    z = torch.randn(300, 7)
    box = torch.rand(300, 4)
    zn = z.clone()
    bn = box.clone()
    zn[2, 1] += 1
    bn[4, 0] += 0.1
    pi = torch.full((300, 5), 0.2)
    oof = {
        "expert_cls_logits": torch.zeros(4, 300, 7),
        "expert_box_boxes": torch.zeros(4, 300, 4),
    }

    fc = features(z, box, zn, bn, oof, pi, pi, 2, 1)
    fb = features(z, box, zn, bn, oof, pi, pi, 4, None)
    assert len(fc[0]) == PCLS and len(fc[1]) == RCLS
    assert len(fb[0]) == PBOX and len(fb[1]) == RBOX

    print("[PASS] Stage1 synthetic utility/editor/feature tests")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--self-test-only", action="store_true"
    )
    parser.add_argument(
        "--frames-per-video", type=int, default=24
    )
    parser.add_argument(
        "--probes-per-task", type=int, default=14
    )
    parser.add_argument(
        "--reject-below", type=float, default=0.25
    )
    args = parser.parse_args()

    self_test()
    if args.self_test_only:
        return

    assert 8 <= args.frames_per_video <= 60
    assert 4 <= args.probes_per_task <= 32
    assert args.reject_below == 0.25, (
        "Pilot threshold is frozen. Do not tune Fold0."
    )

    random.seed(20261008)
    np.random.seed(20261008)
    torch.manual_seed(20261008)

    read_only_inputs()
    folds = r9.parse_folds()
    assert len(folds) == 5

    model, CO, FX = initialize()
    data, stats = run_data(
        model, CO, FX, folds,
        args.frames_per_video,
        args.probes_per_task,
    )

    editors = {}
    support = {}
    for mode in (
        "ProposerEditor",
        "CollaborativeEditor",
    ):
        editors[mode] = {}
        support[mode] = {}
        for task in ("cls", "box"):
            ed, info = fit(
                task,
                mode == "CollaborativeEditor",
                data[task],
            )
            editors[mode][task] = ed
            support[mode][task] = info

    # Fold4 is diagnostic only. No tuning or model selection.
    diagnostics = {}
    for mode in editors:
        diagnostics[mode] = {}
        for task in ("cls", "box"):
            rows = [x for x in data[task] if x[4] == 4]
            ed = editors[mode][task]

            if ed is None:
                diagnostics[mode][task] = {
                    "n": len(rows),
                    "bypass": True,
                }
                continue

            probability = probs(ed, rows)
            labels = torch.tensor([x[2] for x in rows])
            decision = probability >= args.reject_below

            diagnostics[mode][task] = {
                "n": len(rows),
                "pos": int(labels.sum()),
                "reject_fraction": (
                    float((~decision).float().mean())
                    if len(probability) else 0.0
                ),
                "prediction_accuracy": (
                    float(
                        ((probability > 0.5) == labels)
                        .float().mean()
                    )
                    if len(probability) else None
                ),
            }

    print("[+] Fold4 diagnostic:", diagnostics)

    final, interventions = evaluate_fold0(
        model, CO, FX, folds, editors, args.reject_below
    )

    OUT.mkdir(parents=True, exist_ok=True)
    checkpoints = {}
    for mode in editors:
        path = OUT / (
            mode.lower() + "_editor_stage1.pth"
        )
        torch.save({
            "schema": "run09f_stage1_v1",
            "source_checkpoint_sha256": CKPT_SHA,
            "editor_state": {
                task: (
                    editors[mode][task].state_dict()
                    if editors[mode][task] is not None else None
                )
                for task in ("cls", "box")
            },
            "support": support[mode],
            "train_source_folds": [1, 2, 3],
            "inner_fold": 4,
            "heldout_fold": 0,
            "fixed_reject_threshold": args.reject_below,
        }, path)

        checkpoints[mode] = {
            "name": path.name,
            "sha256": sha(path),
        }

    report = {
        "_meta": {
            "schema": "bfcef_f_run09f_editor_stage1_pilot_v1",
            "train_only": True,
            "test_embargo": True,
            "stage0_sha256": STAGE0_SHA,
            "run09c_checkpoint_sha256": CKPT_SHA,
            "stage1_driver_sha256": sha(Path(__file__)),
            "video_training_folds": [1, 2, 3],
            "inner_diagnostic_fold": 4,
            "pilot_fold": 0,
            "no_fold0_tuning": True,
            "note": (
                "Provisional-route editorial veto pilot; "
                "production pre-route authority requires "
                "separate integration."
            ),
        },
        "settings": vars(args),
        "global_supervision_counts": stats,
        "editor_training_support": support,
        "fold4_diagnostics": diagnostics,
        "fold0_metrics": final,
        "editor_interventions": interventions,
        "editor_checkpoints": checkpoints,
    }

    report_path = OUT / "run09f_stage1_editor_pilot.json"
    report_path.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("=" * 110)
    print("RUN_09F STAGE 1 -- CONTROLLED FOLD0 EDITOR COMPARISON")
    print("=" * 110)
    print(
        f"{'Variant':<21} {'mAP':>8} {'mAPca':>8} "
        f"{'TP':>7} {'FP':>7} {'FN':>7} {'F1':>8}"
    )
    for name, m in final.items():
        print(
            f"{name:<21} {m['map30']:>8.4f} "
            f"{m['mapca30']:>8.4f} {m['tp']:>7} "
            f"{m['fp']:>7} {m['fn']:>7} {m['f1']:>8.4f}"
        )
    print("[+] Report:", report_path)
    print("[+] SHA-256:", sha(report_path))
    print(
        "[PASS] Stage 1 pilot complete. "
        "STOP before further training or Val/Test."
    )


if __name__ == "__main__":
    main()
