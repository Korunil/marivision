from __future__ import annotations
"""RUN_10B: frozen RUN09C fold-excluded calibration + RUN09D qualification.

Run folds 1..4 only. NO training. NO official Validation/Test.
The exact RUN09C update_stat/calibrate_stat functions and RUN09D evaluator
semantics are reused, not reimplemented.
"""
import argparse
import gc
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("R0_MSDA_BACKEND", "pytorch")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = Path("D:/MARIVISION")
EVAL = ROOT / "H0_workbench" / "eval_tools"
S0 = EVAL / "bfcef_f_s0"
INPUT = S0 / "run10_frozen_meta_train_v1"
OUT = S0 / "run10b_fold_excluded_calibration_v1"
EXPERTS = ("S", "L", "C", "T")
VARIANTS = ("Base", "Full", "ClsOnly", "BoxOnly", "MinusS", "MinusL", "MinusC", "MinusT")
RUN09C_SOURCE_SHA = "40f7f66dfeaa2e0841da911324bbfdd3de6391293ea5359d9c9d61013bbdccb3"
RUN09D_SOURCE_SHA = "5e15f4c02dd99e450de22b54db085277f1c2cfcd43c6482e17ab93f0f687d812"
RUN09B_SOURCE_SHA = "d51399ed4eaea068dbcfd0a330ad87b01fe277ce907df9e3121af460a514976e"
RUN10A_DRIVER_SHA = "8ebbbc1d736f3572a6e3a7aa94c1c24e132d9e366685679ba30b3932a2d6c762"
OOF_SHA = "1c2569f1871b475e4aa71d20e505c8bb738d9b3ed7d4df185787c1c5546c6001"
CONSEQUENCE_SHA = "f6dcf39a073d7c9fe27201dbe8c38d658cb1b0093f5ba1797a9740314dc70b72"
KNOWN_CKPT = {
    1: "f685ded588e9aca499b8440044bf39acc7a3c5c95e88f08c4cac93f124ffa2fd",
    2: "85367d941a6f9fbdecf6465b30684d2ed51ecfd20abde1bbb39e7dd6b97a19b1",
    3: "f54097709045b0a177779aff63b58c641497e0fbcb5f1e0dd9dabd4445134b53",
    4: "6bad50e1d96a9578764f8340ac0a531f16c533d14ba3752dfdf5b34a2e467313",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check(ok, message):
    if not ok:
        raise AssertionError("[FATAL] " + message)


def verify_sha(path, expected, label):
    check(Path(path).is_file(), f"Missing {label}: {path}")
    actual = sha(path)
    check(actual == expected, f"{label} SHA drift: expected {expected}, actual {actual}")


def load_sources():
    if str(EVAL) not in sys.path:
        sys.path.insert(0, str(EVAL))
    import torch
    import audit_bfcef_f_run09c_threshold_calibration as c
    import audit_bfcef_f_run09d_calibrated_qualification as d
    import train_bfcef_f_run09_joint_uh_gate_router as r9
    import run_bfcef_f_run08_oof_counterfactual_audit as r8

    verify_sha(Path(c.__file__), RUN09C_SOURCE_SHA, "RUN09C calibration source")
    verify_sha(Path(d.__file__), RUN09D_SOURCE_SHA, "RUN09D qualification source")
    verify_sha(Path(r9.__file__), RUN09B_SOURCE_SHA, "RUN09B trainer source")
    verify_sha(r9.OOF_MANIFEST, OOF_SHA, "RUN07 expert OOF manifest")
    verify_sha(r9.CONSEQUENCE_MANIFEST, CONSEQUENCE_SHA, "RUN09A consequence manifest")
    check(c.BINS == 10000 and c.EXPERTS == EXPERTS, "RUN09C algorithm constants changed")
    check(d.VARIANTS == VARIANTS, "RUN09D variant set changed")
    check(r9.WARMUP_EPOCHS == 1 and r9.JOINT_EPOCHS == 2, "Training contract changed")
    r9.s0a.verify_source_and_roots()
    return torch, c, d, r9, r8


def paths(fold):
    return (
        INPUT / f"meta_fold{fold}_run09b_raw.pth",
        INPUT / f"meta_fold{fold}_training_certificate.json",
        OUT / f"meta_fold{fold}_calibrated.pth",
        OUT / f"meta_fold{fold}_qualification.json",
    )


def preflight(fold, r9):
    folds = r9.parse_folds()
    check(sorted(folds) == [0, 1, 2, 3, 4], "Unexpected fold IDs")
    all_ids = [v for vals in folds.values() for v in vals]
    check(len(all_ids) == 47 and sorted(all_ids) == list(range(1, 48)), "Invalid 47-video partition")
    held = sorted(folds[fold])
    train = sorted(set(range(1, 48)) - set(held))
    check(len(held) in (9, 10) and len(train) in (37, 38), "Unexpected fold sizes")
    ckpt, cert_path, cal_path, report_path = paths(fold)
    check(cert_path.exists(), f"Missing RUN10A certificate: {cert_path}")
    cert = json.loads(cert_path.read_text(encoding="utf-8"))
    check(cert["target_meta_fold"] == fold, "Certificate target-fold drift")
    check(cert["train_videos"] == train and cert["heldout_videos"] == held, "Certificate video inventory drift")
    check(not set(train).intersection(held), "Calibration/evaluation leakage")
    check(cert["source_trainer_sha256"] == RUN09B_SOURCE_SHA, "Certificate source drift")
    check(cert["driver_sha256"] == RUN10A_DRIVER_SHA, "RUN10A driver drift")
    check(cert["run07_oof_manifest_sha256"] == OOF_SHA, "Certificate OOF root drift")
    check(cert["run09_consequence_manifest_sha256"] == CONSEQUENCE_SHA, "Certificate local consequence root drift")
    check(cert["calibration_status"] == "NOT_CALIBRATED", "Raw checkpoint was already calibrated")
    check(cert["heldout_evaluation_performed"] is False, "RUN10A evaluation flag unexpected")
    check(cert["official_validation_accessed"] is False and cert["official_test_accessed"] is False, "Embargo flag unexpected")
    check(len(cert["history"]) == 3, "Expected warmup + two joint epochs")
    check([h["name"].split()[1] for h in cert["history"]] == ["WARMUP", "JOINT", "JOINT"], "Epoch history drift")
    verify_sha(ckpt, KNOWN_CKPT[fold], f"RUN10A fold {fold} raw checkpoint")
    check(cert["checkpoint_sha256"] == KNOWN_CKPT[fold], "Checkpoint/certificate mismatch")
    return folds, train, held, ckpt, cal_path, report_path


def verify_state_preservation(raw, calibrated, torch):
    for task in ("self_assessment", "gate_cls", "gate_box"):
        check(raw[task].keys() == calibrated[task].keys(), f"{task} expert keys drift")
        for e in raw[task]:
            check(raw[task][e].keys() == calibrated[task][e].keys(), f"{task} {e} param keys drift")
            for key, value in raw[task][e].items():
                if task in ("gate_cls", "gate_box") and key == "mask_thresh":
                    continue
                check(torch.equal(value, calibrated[task][e][key]), f"Unexpected weight change in {task}.{e}.{key}")
    for task in ("coord_cls", "coord_box"):
        check(raw[task].keys() == calibrated[task].keys(), f"{task} parameter keys drift")
        for key, value in raw[task].items():
            check(torch.equal(value, calibrated[task][key]), f"Unexpected coordinator drift {task}.{key}")


def calibrate(fold, model, CO, FX, train, v2f, torch, c, r9):
    stats = {task: {e: c.empty_stat() for e in EXPERTS} for task in ("cls", "box")}
    original = {
        task: {e: float(getattr(model, "gate_" + task)[e].mask_thresh.detach().cpu().item()) for e in EXPERTS}
        for task in ("cls", "box")
    }
    used = []
    with torch.no_grad():
        for source_fold in sorted({v2f[v] for v in train}):
            check(source_fold != fold, "Target source fold entered calibration")
            r9.load_source_fold(model, source_fold)
            for video in [v for v in train if v2f[v] == source_fold]:
                check(video not in used, f"Repeated calibration video {video}")
                can, spa, tmp, labels = r9.load_video_all(video, source_fold)
                check(len(can) == len(spa) == len(tmp) == len(labels), "Calibrator input alignment drift")
                for i in range(len(can)):
                    O, X = r9.s0a.make_O_X(can[i], spa[i], tmp[i], CO, FX)
                    bundle = r9.expert_bundle(model, O, X)
                    out = r9.mechanism_forward(model, O, X, bundle, hard=False)
                    q = labels[i]["candidate_queries"].long().to(r9.DEVICE)
                    for k, expert in enumerate(EXPERTS):
                        ex = labels[i]["experts"][expert]
                        for task, gain, harm in (
                            ("cls", "u_cls_target", "h_cls_target"),
                            ("box", "u_box_target", "h_box_target"),
                        ):
                            net = (ex[gain].float() - ex[harm].float()).to(r9.DEVICE)
                            c.update_stat(stats[task][expert], out["g_" + task][q, k], net)
                used.append(video)
                print(f"    [+] Fold{fold} calibration video {video:02d}, source fold={source_fold}, frames={len(can)}", flush=True)
                del can, spa, tmp, labels
                gc.collect()
                if r9.DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
    check(sorted(used) == train, "Calibration did not cover exact training-video inventory")
    results = {
        "meta_fold": fold, "calibration_videos": train, "heldout_videos": sorted(set(range(1,48)) - set(train)),
        "selection_rule": "RUN09C original: highest threshold maximizing cumulative signed local utility sum(U-H | g_soft > threshold); target fold excluded",
        "bins": c.BINS, "original_thresholds": original,
        "cls": {}, "box": {},
    }
    for task in ("cls", "box"):
        for expert in EXPERTS:
            row = c.calibrate_stat(stats[task][expert])
            results[task][expert] = row
            gate = getattr(model, "gate_" + task)[expert]
            gate.mask_thresh.data.fill_(float(row["threshold"]))
            check(float(gate.mask_thresh.item()) >= 0.0, "Invalid threshold")
            print(f"[+] Fold{fold} {task.upper()} {expert}: tau={row['threshold']:.4f}, {row['status']}, active_train={row['activated_train_fraction']:.3%}", flush=True)
    return results


def evaluate(fold, model, CO, FX, folds, torch, d, r9, r8):
    held = sorted(folds[fold])
    r9.load_source_fold(model, fold)
    model.eval()
    (coco, coco_ca, size, _, _, cat, classes) = r8.load_metadata()
    evaluators = {v: r8.make_evaluators(coco, coco_ca) for v in VARIANTS}
    counts = {v: Counter() for v in VARIANTS}
    by_video = {v: {} for v in VARIANTS}
    trans = {v: d.new_transition() for v in VARIANTS if v != "Base"}
    diag = d.FullDiagnostics()
    support = Counter()
    image_ids = set()
    with torch.no_grad():
        for video in held:
            can, spa, tmp, _ = r9.load_video_all(video, fold)
            pred_maps = {v: {} for v in VARIANTS}
            pred_maps_ca = {v: {} for v in VARIANTS}
            vcounts = {v: Counter() for v in VARIANTS}
            for i in range(len(can)):
                fr = can[i]
                O, X = r9.s0a.make_O_X(fr, spa[i], tmp[i], CO, FX)
                variants, info = d.make_variants(model, O, X)
                iid = int(fr["image_id"])
                check(iid not in image_ids, "Duplicate image ID")
                image_ids.add(iid)
                gt_b, gt_l = fr["gt_boxes_norm"].float(), fr["gt_labels"].long()
                support.update(gt_l.tolist())
                predictions, matches = {}, {}
                for variant, (z, b) in variants.items():
                    pred = r8.extract_top100_predictions(z.detach().cpu().float(), b.detach().cpu().float())
                    predictions[variant] = pred
                    w, h = size[iid]
                    r8.add_coco_prediction(pred_maps[variant], pred_maps_ca[variant], iid, pred, w, h, cat)
                    m = r8.match_operational_detections_to_gt(
                        pred["op_scores"], pred["op_c_idx"], pred["op_boxes"], pred["op_q_idx"], gt_b, gt_l
                    )
                    matches[variant] = m
                    for key in ("tp", "fp", "fn"):
                        counts[variant][key] += int(m[key])
                        vcounts[variant][key] += int(m[key])
                b = matches["Base"]
                for variant in trans:
                    m = matches[variant]
                    tr = trans[variant]
                    tr["rescued"] += len(m["matched_gt_indices"] - b["matched_gt_indices"])
                    tr["lost"] += len(b["matched_gt_indices"] - m["matched_gt_indices"])
                    tr["retained"] += len(b["matched_gt_indices"] & m["matched_gt_indices"])
                    tr["fp_added"] += len(set(m["fp_dict"]) - set(b["fp_dict"]))
                    tr["fp_removed"] += len(set(b["fp_dict"]) - set(m["fp_dict"]))
                diag.update(O, variants["Full"][0], variants["Full"][1], info, predictions["Base"], predictions["Full"])
            for variant in VARIANTS:
                ev, ev_ca = evaluators[variant]
                ev.update(pred_maps[variant])
                ev_ca.update(pred_maps_ca[variant])
                by_video[variant][str(video)] = dict(vcounts[variant])
            print(f"    [+] Fold{fold} qualification video {video:02d}, frames={len(can)}", flush=True)
            del can, spa, tmp
            gc.collect()
            if r9.DEVICE.type == "cuda":
                torch.cuda.empty_cache()
    result = {}
    for variant in VARIANTS:
        ev, ev_ca = evaluators[variant]
        count = counts[variant]
        m = {
            **d.metrics_from_counts(**{k: count[k] for k in ("tp", "fp", "fn")}),
            **d.finalize_variant_evaluator(ev, ev_ca, classes),
            "by_video_operational": by_video[variant],
        }
        if variant != "Base":
            tr = trans[variant]
            tr["net_tp"] = tr["rescued"] - tr["lost"]
            tr["net_fp"] = tr["fp_added"] - tr["fp_removed"]
            check(tr["retained"] + tr["lost"] == counts["Base"]["tp"], "GT retention conservation")
            check(tr["retained"] + tr["rescued"] == count["tp"], "GT rescue conservation")
            check(tr["net_fp"] == count["fp"] - counts["Base"]["fp"], "FP conservation")
            check(tr["net_tp"] == count["tp"] - counts["Base"]["tp"], "TP conservation")
        result[variant] = m
    check(len(image_ids) > 0, "No held-out frames evaluated")
    diag_report = diag.finalize()
    check(diag_report["frames"] == len(image_ids), "Diagnostic frame census drift")
    return result, trans, diag_report, {classes[i]: int(support[i]) for i in range(len(classes))}


def run_one(fold, preflight_only=False):
    torch, c, d, r9, r8 = load_sources()
    folds, train, held, raw_path, cal_path, report_path = preflight(fold, r9)
    ckpt = torch.load(raw_path, map_location="cpu", weights_only=False)
    check(ckpt["_meta"]["meta_fold"] == fold, "Embedded checkpoint fold mismatch")
    check(ckpt["_meta"]["run07_oof_manifest_sha256"] == OOF_SHA, "Embedded OOF root drift")
    check(ckpt["_meta"]["run09_consequence_manifest_sha256"] == CONSEQUENCE_SHA, "Embedded consequence root drift")
    check(not cal_path.exists() and not report_path.exists(), "Existing RUN10B artifacts: refuse overwrite")
    print(f"[PASS] Fold{fold} raw checkpoint, certificate, and all frozen source hashes; train={len(train)}, held={len(held)}", flush=True)
    if preflight_only:
        return
    model, _, CO, FX = r9.initialize_runtime()
    c.load_downstream_checkpoint(model, ckpt)
    model.eval()
    thresholds = calibrate(fold, model, CO, FX, train, r9.video_fold_map(folds), torch, c, r9)
    # First reuse exact original RUN09C hard-count evaluation (target fold
    # selected by RUN09B's META_FOLD, which was global fold 0 in original).
    old_fold = r9.META_FOLD
    r9.META_FOLD = fold
    try:
        basic = r9.hard_eval_fold0(model, CO, FX, folds, r9.video_fold_map(folds))
        check(basic["heldout_videos"] == held, "RUN09C original hard evaluation used incorrect fold")
        variants, transitions, diagnostics, support = evaluate(fold, model, CO, FX, folds, torch, d, r9, r8)
    finally:
        r9.META_FOLD = old_fold
    for variant in ("Base", "Full"):
        key = "base" if variant == "Base" else "integrated"
        check(all(variants[variant][k] == basic[key][k] for k in ("tp", "fp", "fn")), f"RUN09C/RUN09D operational count disagreement: {variant}")
    # Save only downstream weights; never incorporate held-out GT into parameters.
    calibrated = {
        "_meta": {**ckpt["_meta"], "schema_version": "bfcef_f_run10b_calibrated_v1"},
        "calibration": thresholds,
        "heldout_eval_after_calibration": basic,
        "source_pilot_checkpoint_sha256": KNOWN_CKPT[fold],
        "run09_consequence_manifest_sha256": CONSEQUENCE_SHA,
    }
    for task in ("self_assessment", "gate_cls", "gate_box"):
        calibrated[task] = {
            e: {k: t.detach().cpu().clone() for k, t in getattr(model, task)[e].state_dict().items()}
            for e in EXPERTS
        }
    for task in ("coord_cls", "coord_box"):
        calibrated[task] = {k: t.detach().cpu().clone() for k, t in getattr(model, task).state_dict().items()}
    verify_state_preservation(ckpt, calibrated, torch)
    for task in ("cls", "box"):
        for e in EXPERTS:
            actual = float(calibrated["gate_" + task][e]["mask_thresh"].item())
            check(abs(actual - thresholds[task][e]["threshold"]) < 1e-6, "Saved threshold mismatch")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(calibrated, cal_path)
    report = {
        "_meta": {
            "schema_version": "bfcef_f_run10b_v1", "meta_fold": fold,
            "train_videos": train, "heldout_videos": held,
            "source_calibrator_sha256": RUN09C_SOURCE_SHA,
            "source_qualifier_sha256": RUN09D_SOURCE_SHA,
            "run10a_raw_checkpoint_sha256": KNOWN_CKPT[fold],
            "calibrated_checkpoint_sha256": sha(cal_path),
            "driver_sha256": sha(Path(__file__)),
            "calibration_uses_heldout_labels": False,
            "only_gate_thresholds_changed": True,
            "official_validation_accessed": False,
            "official_test_accessed": False,
            "limitations": "Conditional downstream meta-OOF only; common R0 trained on all Train videos. RUN09C quantized histogram threshold rule preserved verbatim.",
        },
        "calibration": thresholds,
        "run09c_original_hard_counts": basic,
        "variants": variants,
        "transitions": transitions,
        "full_mechanism_diagnostics": diagnostics,
        "class_support": support,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=" * 110)
    print(f"[PASS] RUN10B FOLD {fold}: calibrated and qualified without retraining")
    print(f"Calibration videos: {len(train)}; held-out videos: {held}")
    print("%-12s %9s %9s %8s %8s %8s %8s" % ("Variant", "mAP30", "mAPca", "TP", "FP", "FN", "F1"))
    for v in ("Base", "Full"):
        m = variants[v]
        print(f"{v:<12} {m['map30']:>9.4f} {m['mapca30']:>9.4f} {m['tp']:>8} {m['fp']:>8} {m['fn']:>8} {m['f1']:>8.4f}")
    print("Checkpoint:", cal_path, "SHA256:", sha(cal_path))
    print("Report:", report_path, "SHA256:", sha(report_path))
    print("No Validation or Test; freeze both artifacts.")
    print("=" * 110)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, choices=(1, 2, 3, 4), required=True)
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args()
    check(args.fold in KNOWN_CKPT, "Unsupported fold")
    run_one(args.fold, args.preflight_only)


if __name__ == "__main__":
    main()
