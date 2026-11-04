
from __future__ import annotations

"""RUN_09F Stage 0: Train-only inventory and consequential-replay preflight."""

import argparse
import collections
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch

ROOT = Path("D:/MARIVISION")
EVAL = ROOT / "H0_workbench" / "eval_tools"
TRAIN = ROOT / "seaclips-train.json"
S0 = EVAL / "bfcef_f_s0"
OOF = S0 / "final_expert_crossfit_v1"
LOCAL = S0 / "run09_final_local_consequences_v1"
OUT = S0 / "run09f_final_resilient_pilot_v1"

EXPECTED = {
    OOF / "final_expert_oof_manifest.json":
        "1c2569f1871b475e4aa71d20e505c8bb738d9b3ed7d4df185787c1c5546c6001",
    LOCAL / "run09_final_consequence_manifest.json":
        "f6dcf39a073d7c9fe27201dbe8c38d658cb1b0093f5ba1797a9740314dc70b72",
    S0 / "run09e_editor_readiness_v1" / "run09e_editor_readiness_audit.json":
        "7de6e240dc929b2e71112b99c35a2e364d308ef3765e592856afa1098af5941e",
}

EXPERTS = ("S", "L", "C", "T")

# Class states:
# 0=TP, 1=FP, 2=NONOP
# transition = 3 * before_state + after_state
TP_DELTA = (0, -1, -1, 1, 0, 0, 1, 0, 0)

EVENTS = {
    1: "TP_to_FP",
    2: "TP_loss",
    3: "FP_to_TP",
    5: "FP_prune",
    6: "NONOP_rescue",
    7: "FP_created",
}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def self_test():
    assert list(TP_DELTA) == [
        0, -1, -1, 1, 0, 0, 1, 0, 0
    ]
    assert (1 // 3, 1 % 3) == (0, 1)

    x, y, w, h = 10, 20, 30, 40
    W, H = 100, 100
    box = np.array([
        (x + w / 2) / W,
        (y + h / 2) / H,
        w / W,
        h / H,
    ])
    assert np.allclose(box, [.25, .4, .3, .4])
    print("[PASS] Synthetic transition and coordinate tests")


def convert_ann(ann, dims):
    x, y, w, h = map(float, ann["bbox"])
    W, H = dims
    return np.array([
        (x + w / 2) / W,
        (y + h / 2) / H,
        w / W,
        h / H,
    ], dtype=np.float64)


def compare_annotations(raw, cached, tol=2e-3):
    if len(raw) != len(cached):
        return False, "different annotation count"

    remaining = list(range(len(raw)))

    for cls, box in cached:
        same = [j for j in remaining if raw[j][0] == cls]
        if not same:
            return False, "category mismatch"

        errors = [
            np.max(np.abs(raw[j][1] - box))
            for j in same
        ]
        nearest = int(np.argmin(errors))
        if errors[nearest] > tol:
            return False, (
                f"box-coordinate mismatch "
                f"max_error={errors[nearest]:.5g}"
            )
        remaining.remove(same[nearest])

    return not remaining, "OK"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--self-test-only", action="store_true"
    )
    parser.add_argument(
        "--per-fold-event-quota", type=int, default=2
    )
    args = parser.parse_args()

    self_test()
    if args.self_test_only:
        return

    if args.per_fold_event_quota not in range(1, 5):
        raise ValueError("Quota must be between 1 and 4")

    for path, expected in EXPECTED.items():
        assert path.exists(), f"[FATAL] Missing {path}"
        assert sha(path) == expected, (
            f"[FATAL] Provenance SHA drift: {path}"
        )

    if str(EVAL) not in sys.path:
        sys.path.insert(0, str(EVAL))

    import run_bfcef_f_run08_oof_counterfactual_audit as r8

    folds = r8.parse_folds()
    v2f = {
        v: f for f, videos in folds.items() for v in videos
    }

    raw = json.loads(TRAIN.read_text(encoding="utf-8"))
    categories = sorted(
        raw["categories"], key=lambda x: int(x["id"])
    )

    expected_names = [
        "ANIMAL", "BOAT", "MARINE_MARKER", "OBJECT",
        "LEISURE_VEHICLE", "SAILING_VESSEL", "SHIP"
    ]
    assert [x["name"] for x in categories] == expected_names

    id2idx = {
        int(c["id"]): i for i, c in enumerate(categories)
    }
    images = {
        int(im["id"]): im for im in raw["images"]
    }

    annotations = collections.defaultdict(list)
    for a in raw["annotations"]:
        cid = int(a["category_id"])
        iid = int(a["image_id"])
        assert cid in id2idx and iid in images
        im = images[iid]

        annotations[iid].append((
            id2idx[cid],
            convert_ann(a, (im["width"], im["height"]))
        ))

    observed = set()
    cached_counts = collections.Counter()
    raw_seen_counts = collections.Counter()
    mismatches = []
    event_slots = {}
    all_frames = 0

    print("=" * 110)
    print("RUN_09F STAGE 0A -- ANNOTATION RECONCILIATION")
    print("=" * 110)

    for vid in range(1, 48):
        fold = v2f[vid]

        can = torch.load(
            r8.canonical_path(vid),
            map_location="cpu",
            weights_only=False
        )
        oo = torch.load(
            OOF / f"fold{fold}" /
            f"bfcef_f_s0a_oof_vid{vid}.pth",
            map_location="cpu",
            weights_only=False
        )
        ll = torch.load(
            LOCAL / f"fold{fold}" /
            f"bfcef_f_run09_consequence_vid{vid}.pth",
            map_location="cpu",
            weights_only=False
        )["records"]

        assert len(can) == len(oo) == len(ll)
        all_frames += len(can)

        for frame_idx, (fr, of, lf) in enumerate(
            zip(can, oo, ll)
        ):
            iid = int(fr["image_id"])
            assert iid not in observed and iid in images
            assert int(images[iid]["video_id"]) == vid
            assert iid == int(of["image_id"])
            assert iid == int(lf["image_id"])
            observed.add(iid)

            q = lf["candidate_queries"].long().tolist()
            assert q == of["candidate_queries"].long().tolist()

            gt_cls = fr["gt_labels"].long().tolist()
            gt_boxes = fr["gt_boxes_norm"].float().numpy()

            cached = [
                (int(c), b.astype(np.float64))
                for c, b in zip(gt_cls, gt_boxes)
            ]
            original = annotations[iid]

            for c, _ in cached:
                cached_counts[c] += 1
            for c, _ in original:
                raw_seen_counts[c] += 1

            ok, reason = compare_annotations(
                original, cached
            )
            if not ok and len(mismatches) < 30:
                mismatches.append({
                    "video": vid,
                    "image_id": iid,
                    "cached": len(cached),
                    "raw": len(original),
                    "reason": reason,
                })

            # Identify consequential counterfactuals
            # from immutable RUN09A labels.
            for k, expert in enumerate(EXPERTS):
                transitions = (
                    lf["experts"][expert]["cls_transition"]
                    .long().numpy()
                )

                for state, event in EVENTS.items():
                    slot = (fold, expert, "cls", event)
                    positions = np.argwhere(
                        transitions == state
                    )
                    selected = event_slots.setdefault(slot, [])

                    for qi, c in positions:
                        if len(selected) >= args.per_fold_event_quota:
                            break
                        selected.append((
                            vid, frame_idx,
                            int(q[int(qi)]), int(c)
                        ))

                for field, event in [
                    ("box_rescue_target", "box_rescue"),
                    ("box_loss_target", "box_loss"),
                ]:
                    slot = (fold, expert, "box", event)
                    selected = event_slots.setdefault(slot, [])

                    values = (
                        lf["experts"][expert][field]
                        .reshape(-1)
                    )
                    hits = (values > 0).nonzero(
                        as_tuple=False
                    ).reshape(-1).tolist()

                    needed = max(
                        0,
                        args.per_fold_event_quota - len(selected)
                    )
                    for qi in hits[:needed]:
                        selected.append((
                            vid, frame_idx, int(q[qi]), -1
                        ))

        print(
            f"[+] Inventory video {vid:02d}: "
            f"{len(can)} frames"
        )
        del can, oo, ll

    unseen = set(images) - observed
    original_on_seen = sum(raw_seen_counts.values())
    original_on_unseen = sum(
        len(annotations[i]) for i in unseen
    )
    cached_total = sum(cached_counts.values())
    raw_total = len(raw["annotations"])

    assert raw_total == (
        original_on_seen + original_on_unseen
    )
    assert all_frames == 20075
    assert cached_total == 64670
    assert raw_total == 65542

    inventory = {
        "raw_images": len(images),
        "cached_images": len(observed),
        "raw_annotations": raw_total,
        "cached_gt": cached_total,
        "raw_minus_cached": raw_total - cached_total,
        "raw_annotations_on_cached_images": original_on_seen,
        "raw_annotations_on_uncached_images": original_on_unseen,
        "cached_vs_raw_seen_discrepancy":
            original_on_seen - cached_total,
        "sample_mismatched_images": mismatches,
        "sample_uncached_images": [
            {
                "image_id": iid,
                "video": int(images[iid]["video_id"]),
                "annotations": len(annotations[iid]),
            }
            for iid in sorted(unseen)[:30]
        ],
        "cached_count_by_class": {
            categories[i]["name"]: n
            for i, n in cached_counts.items()
        },
        "raw_count_by_class_on_cached_images": {
            categories[i]["name"]: n
            for i, n in raw_seen_counts.items()
        },
    }

    print("[+] Annotation reconciliation:")
    for key in [
        "raw_images", "cached_images",
        "raw_annotations", "cached_gt",
        "raw_minus_cached",
        "raw_annotations_on_uncached_images",
        "cached_vs_raw_seen_discrepancy",
    ]:
        print(f"    {key}: {inventory[key]}")

    print("=" * 110)
    print("RUN_09F STAGE 0B -- CONSEQUENTIAL GLOBAL REPLAYS")
    print("=" * 110)

    by_video = collections.defaultdict(list)
    for slot, cases in event_slots.items():
        for vid, frame_idx, q, c in cases:
            by_video[vid].append((
                slot, frame_idx, q, c
            ))

    summary = collections.defaultdict(
        collections.Counter
    )
    failures = []

    for vid in sorted(by_video):
        fold = v2f[vid]
        can = torch.load(
            r8.canonical_path(vid),
            map_location="cpu",
            weights_only=False
        )
        oo = torch.load(
            OOF / f"fold{fold}" /
            f"bfcef_f_s0a_oof_vid{vid}.pth",
            map_location="cpu",
            weights_only=False
        )
        ll = torch.load(
            LOCAL / f"fold{fold}" /
            f"bfcef_f_run09_consequence_vid{vid}.pth",
            map_location="cpu",
            weights_only=False
        )["records"]

        for slot, frame_idx, q, c in by_video[vid]:
            fr = can[frame_idx]
            of = oo[frame_idx]
            lf = ll[frame_idx]

            z = fr["d_logits"].float()
            b = fr["d_boxes"].float()
            gt_b = fr["gt_boxes_norm"].float()
            gt_l = fr["gt_labels"].long()

            def match(znew, bnew):
                pred = r8.extract_top100_predictions(
                    znew, bnew
                )
                m = r8.match_operational_detections_to_gt(
                    pred["op_scores"],
                    pred["op_c_idx"],
                    pred["op_boxes"],
                    pred["op_q_idx"],
                    gt_b, gt_l
                )
                return m

            baseline = match(z, b)
            _, expert, task, event = slot
            ki = EXPERTS.index(expert)

            qi = (
                lf["candidate_queries"]
                .long().tolist().index(q)
            )

            if task == "cls":
                z1 = z.clone()
                z1[q] = (
                    of["expert_cls_logits"][ki, q]
                    .float()
                )
                after = match(z1, b)

                trans = (
                    lf["experts"][expert]["cls_transition"]
                    [qi].long().tolist()
                )

                observed_state = (
                    0 if any(
                        d[3] == q and d[4] == c
                        for d in after["gt_to_det"].values()
                    )
                    else (
                        1 if (q, c) in after["fp_dict"]
                        else 2
                    )
                )

                if observed_state != trans[c] % 3:
                    failures.append({
                        "video": vid,
                        "frame": frame_idx,
                        "expert": expert,
                        "event": event,
                        "expected_state": trans[c] % 3,
                        "actual_state": observed_state,
                    })

                local_delta = sum(
                    TP_DELTA[state] for state in trans
                )

            else:
                b1 = b.clone()
                b1[q] = (
                    of["expert_box_boxes"][ki, q]
                    .float()
                )
                after = match(z, b1)

                local_delta = (
                    int(
                        lf["experts"][expert]
                        ["box_rescue_target"][qi].item()
                    )
                    - int(
                        lf["experts"][expert]
                        ["box_loss_target"][qi].item()
                    )
                )

                for cidx in range(7):
                    matched = [
                        g for g, d
                        in after["gt_to_det"].items()
                        if d[3] == q and d[4] == cidx
                    ]
                    actual_gt = (
                        matched[0] if matched else -1
                    )
                    expected_gt = int(
                        lf["experts"][expert]
                        ["box_after_gt_idx"][qi, cidx]
                        .item()
                    )
                    if actual_gt != expected_gt:
                        failures.append({
                            "video": vid,
                            "frame": frame_idx,
                            "expert": expert,
                            "event": event,
                            "class": cidx,
                            "expected_gt": expected_gt,
                            "actual_gt": actual_gt,
                        })

            global_delta = (
                after["tp"] - baseline["tp"]
            )

            group = f"{task}/{event}"
            summary[group]["replays"] += 1
            summary[group]["global_tp_change"] += int(
                global_delta != 0
            )
            summary[group]["local_vs_global_delta_mismatch"] += int(
                global_delta != local_delta
            )
            summary[group]["global_fp_change"] += int(
                after["fp"] != baseline["fp"]
            )

        print(
            f"[+] Consequential replay video {vid:02d}: "
            f"{len(by_video[vid])} cases"
        )

    report = {
        "_meta": {
            "schema": "bfcef_f_run09f_stage0_v1",
            "train_only": True,
            "test_embargo": True,
            "source_sha256": sha(Path(__file__)),
        },
        "inventory_reconciliation": inventory,
        "consequence_replay_by_event": {
            k: dict(v) for k, v in summary.items()
        },
        "total_consequence_replays": sum(
            v["replays"] for v in summary.values()
        ),
        "replay_failures": failures[:30],
        "replay_failure_count": len(failures),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    report_path = OUT / "run09f_stage0_preflight.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("=" * 110)
    print("[+] Stage 0 report:", report_path)
    print("[+] Report SHA:", sha(report_path))

    print("[+] Global replay results:")
    for name, counts in summary.items():
        print(f"    {name}: {dict(counts)}")

    if mismatches or original_on_seen != cached_total:
        raise AssertionError(
            "[BLOCK] Cached-image annotations differ from "
            "raw Train. Reconcile before Editor training."
        )

    if failures:
        raise AssertionError(
            "[BLOCK] Local/global replay identity mismatch."
        )

    if not summary or not any(
        r["global_tp_change"] > 0
        for r in summary.values()
    ):
        raise AssertionError(
            "[BLOCK] No consequential TP-changing "
            "global cases were replayed."
        )

    print(
        "[PASS] RUN_09F Stage 0: "
        "inventory reconciled and targeted replays certified"
    )
    print(
        "STOP: Review Stage 0 before Editor model training."
    )


if __name__ == "__main__":
    main()
