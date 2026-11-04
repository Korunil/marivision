
from __future__ import annotations

"""
B-FCEF-F RUN_10A
Frozen architecture, remaining meta-folds 1-4.

Reuses certified RUN09B:
- Same S/L/C/T + U/H + Gates + Coordinator + Router
- Same optimizer, losses, seed, epochs
- No Editor
- No target-fold labels in training
- No calibration, held-out evaluation, Validation or Test
"""

import argparse
import gc
import os
import sys
from pathlib import Path

os.environ.setdefault("R0_MSDA_BACKEND", "pytorch")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import torch

EVAL = Path("D:/MARIVISION/H0_workbench/eval_tools")
OUT = EVAL / "bfcef_f_s0" / "run10_frozen_meta_train_v1"


# RUN09B source variation verified against the original:
# 1. Added: from PIL import Image
# 2. Removed two explanatory comment lines.
# Executable AST is identical after excluding the PIL import.
# No model/training/evaluation algorithm changes.

EXPECTED_R9_SHA = (
    "d51399ed4eaea068dbcfd0a330ad87b01fe277ce907df9e3121af460a514976e"
)

EXPECTED_OOF_SHA = (
    "1c2569f1871b475e4aa71d20e505c8bb738d9b3ed7d4df185787c1c5546c6001"
)
EXPECTED_LOCAL_SHA = (
    "f6dcf39a073d7c9fe27201dbe8c38d658cb1b0093f5ba1797a9740314dc70b72"
)
EXPECTED_F0_CAL_SHA = (
    "a1debfaadcc7f754319c63386ac93d993dc2a036cefbdf4e1ee70a567b517deb"
)

if str(EVAL) not in sys.path:
    sys.path.insert(0, str(EVAL))

import train_bfcef_f_run09_joint_uh_gate_router as r9


def require(condition, message):
    if not condition:
        raise AssertionError("[FATAL] " + message)


def file_hash(path):
    path = Path(path)
    require(path.is_file(), f"Missing source: {path}")
    return r9.sha256(path)


def source_preflight():
    require(
        file_hash(Path(r9.__file__)) == EXPECTED_R9_SHA,
        "Original RUN09B source changed."
    )
    require(
        file_hash(r9.OOF_MANIFEST) == EXPECTED_OOF_SHA,
        "Final-Expert OOF manifest drift."
    )
    require(
        file_hash(r9.CONSEQUENCE_MANIFEST) == EXPECTED_LOCAL_SHA,
        "RUN09A consequence manifest drift."
    )

    cal = (
        EVAL / "bfcef_f_s0"
        / "run09c_threshold_calibration_v1"
        / "meta_fold0_uh_gate_router_calibrated.pth"
    )
    require(
        file_hash(cal) == EXPECTED_F0_CAL_SHA,
        "Frozen Fold0 reference checkpoint drift."
    )

    r9.s0a.verify_source_and_roots()

    require(
        r9.WARMUP_EPOCHS == 1 and r9.JOINT_EPOCHS == 2,
        "Training epochs changed."
    )
    require(
        r9.SEED == 20261009,
        "Training seed changed."
    )
    require(
        r9.LR_SA == 2e-4 and r9.LR_GATE == 1e-4,
        "Learning rates changed."
    )
    require(
        tuple(r9.EXPERTS) == ("S", "L", "C", "T"),
        "Expert roster changed."
    )

    print("[PASS] Frozen source and training contract")


def fold_contract(target):
    folds = r9.parse_folds()
    require(
        sorted(folds) == [0, 1, 2, 3, 4],
        "Fold inventory changed."
    )

    all_videos = [
        int(v) for group in folds.values() for v in group
    ]
    require(
        len(all_videos) == 47
        and sorted(all_videos) == list(range(1, 48)),
        "Train videos must partition 1..47."
    )

    heldout = sorted(folds[target])
    train = sorted(set(range(1, 48)) - set(heldout))

    require(
        len(set(train) & set(heldout)) == 0,
        "Target fold leaked into training."
    )

    return folds, r9.video_fold_map(folds), train, heldout


def paths(target):
    return (
        OUT / f"meta_fold{target}_run09b_raw.pth",
        OUT / f"meta_fold{target}_training_certificate.json",
    )


def verify_completed(target, train, heldout):
    checkpoint, certificate = paths(target)

    if not checkpoint.exists() and not certificate.exists():
        return False

    require(
        checkpoint.exists() and certificate.exists(),
        "Partial checkpoint/certificate pair found."
    )

    report = r9.load_json(certificate)

    require(
        report["target_meta_fold"] == target,
        "Target fold mismatch."
    )
    require(
        report["train_videos"] == train
        and report["heldout_videos"] == heldout,
        "Training/held-out inventory mismatch."
    )
    require(
        file_hash(checkpoint) == report["checkpoint_sha256"],
        "Checkpoint SHA mismatch."
    )
    require(
        report["source_trainer_sha256"] == EXPECTED_R9_SHA,
        "Trainer version mismatch."
    )
    require(
        report["calibration_status"] == "NOT_CALIBRATED",
        "Unknown calibration phase."
    )

    payload = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )

    meta = payload["_meta"]
    require(meta["meta_fold"] == target, "Embedded fold drift.")
    require(
        meta["run07_oof_manifest_sha256"] == EXPECTED_OOF_SHA,
        "OOF lineage mismatch."
    )
    require(
        meta["run09_consequence_manifest_sha256"]
        == EXPECTED_LOCAL_SHA,
        "Consequence lineage mismatch."
    )

    for key in ("self_assessment", "gate_cls", "gate_box"):
        require(
            sorted(payload[key]) == sorted(r9.EXPERTS),
            f"Incomplete {key} state."
        )

    require(
        "coord_cls" in payload and "coord_box" in payload,
        "Missing coordinator state."
    )

    return True


def assert_frozen_modules(model):
    frozen = (
        "expert_s", "expert_l", "expert_c",
        "expert_t", "context_bus", "adapter"
    )

    for name in frozen:
        module = getattr(model, name)
        require(
            all(not p.requires_grad for p in module.parameters()),
            f"Frozen module {name} became trainable."
        )


def train_fold(target, smoke_only=False, resume=False):
    source_preflight()

    folds, v2f, train, heldout = fold_contract(target)

    # The original RUN09B smoke uses META_FOLD to exclude
    # held-out videos. No original function is edited.
    r9.META_FOLD = target

    print("=" * 110)
    print(f"RUN_10A FROZEN MECHANISM -- META-FOLD {target}")
    print(f"Training videos: {len(train)}")
    print(f"Held-out videos: {heldout}")
    print("RUN09B: 1 warm-up + 2 joint epochs")
    print("TRAIN ONLY -- NO CALIBRATION -- NO VAL/TEST")
    print("=" * 110)

    if not smoke_only and verify_completed(target, train, heldout):
        if resume:
            print(f"[SKIP] Fold {target} checkpoint already certified")
            return
        raise AssertionError(
            "Output already exists. Use --resume to skip."
        )

    manifest = r9.load_json(r9.CONSEQUENCE_MANIFEST)

    model, criterion, CO, FX = r9.initialize_runtime()
    r9.set_seed(r9.SEED)

    r9.exact_mechanism_smoke(
        model, criterion, CO, FX,
        folds, v2f, manifest
    )

    print(
        "[PASS] OOF reproduction, forward parity, "
        "gradient and fallback smoke"
    )

    if smoke_only:
        return

    # The smoke test changes temporary model state.
    # Reconstruct a pristine training model.
    del model, criterion
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model, criterion, CO, FX = r9.initialize_runtime()
    r9.set_seed(r9.SEED)

    weights = r9.aggregate_training_counts(manifest, train)
    history = []

    # Original RUN09B warm-up phase.
    r9.set_training_modes(model, joint=False)
    assert_frozen_modules(model)
    optimizer = r9.make_optimizer(model, joint=False)

    for epoch in range(r9.WARMUP_EPOCHS):
        result = r9.run_epoch(
            epoch_name=f"F{target} WARMUP {epoch+1}/1",
            model=model,
            criterion=criterion,
            CanonicalObjectState=CO,
            frame_inputs_from_records=FX,
            train_videos=train,
            v2f=v2f,
            weights=weights,
            joint=False,
            seed=r9.SEED + epoch,
            optimizer=optimizer,
        )
        history.append(result)
        print(f"[+] {result['name']} complete")

    # Original RUN09B joint phase.
    r9.set_training_modes(model, joint=True)
    assert_frozen_modules(model)
    optimizer = r9.make_optimizer(model, joint=True)

    for epoch in range(r9.JOINT_EPOCHS):
        result = r9.run_epoch(
            epoch_name=f"F{target} JOINT {epoch+1}/2",
            model=model,
            criterion=criterion,
            CanonicalObjectState=CO,
            frame_inputs_from_records=FX,
            train_videos=train,
            v2f=v2f,
            weights=weights,
            joint=True,
            seed=r9.SEED + 100 + epoch,
            optimizer=optimizer,
        )
        history.append(result)
        print(f"[+] {result['name']} complete")

    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint, certificate = paths(target)

    require(
        not checkpoint.exists() and not certificate.exists(),
        "Refusing to overwrite existing artifacts."
    )

    payload = {
        "_meta": {
            "schema_version": "bfcef_f_run10a_run09b_raw_v1",
            "meta_fold": target,
            "heldout_videos": heldout,
            "train_videos": train,
            "run07_oof_manifest_sha256": EXPECTED_OOF_SHA,
            "run09_consequence_manifest_sha256": EXPECTED_LOCAL_SHA,
            "run09b_source_sha256": EXPECTED_R9_SHA,
            "warmup_epochs": r9.WARMUP_EPOCHS,
            "joint_epochs": r9.JOINT_EPOCHS,
            "lr_sa": r9.LR_SA,
            "lr_gate": r9.LR_GATE,
            "seed": r9.SEED,
            "history": history,
            "status": "RAW_UNCALIBRATED",
        },
        "self_assessment": {
            e: {
                k: v.detach().cpu()
                for k, v in m.state_dict().items()
            }
            for e, m in model.self_assessment.items()
        },
        "gate_cls": {
            e: {
                k: v.detach().cpu()
                for k, v in m.state_dict().items()
            }
            for e, m in model.gate_cls.items()
        },
        "gate_box": {
            e: {
                k: v.detach().cpu()
                for k, v in m.state_dict().items()
            }
            for e, m in model.gate_box.items()
        },
        "coord_cls": {
            k: v.detach().cpu()
            for k, v in model.coord_cls.state_dict().items()
        },
        "coord_box": {
            k: v.detach().cpu()
            for k, v in model.coord_box.state_dict().items()
        },
    }

    torch.save(payload, checkpoint)

    report = {
        "_schema": "run10a_train_cert_v1",
        "target_meta_fold": target,
        "train_videos": train,
        "heldout_videos": heldout,
        "checkpoint_sha256": file_hash(checkpoint),
        "source_trainer_sha256": EXPECTED_R9_SHA,
        "driver_sha256": file_hash(Path(__file__)),
        "run07_oof_manifest_sha256": EXPECTED_OOF_SHA,
        "run09_consequence_manifest_sha256": EXPECTED_LOCAL_SHA,
        "calibration_status": "NOT_CALIBRATED",
        "heldout_evaluation_performed": False,
        "official_validation_accessed": False,
        "official_test_accessed": False,
        "history": history,
    }

    r9.save_json(certificate, report)

    require(
        verify_completed(target, train, heldout),
        "Post-save certificate verification failed."
    )

    print("=" * 110)
    print(f"[PASS] RUN_10A META-FOLD {target} TRAINING CERTIFIED")
    print(f"Checkpoint: {checkpoint}")
    print(f"SHA-256: {report['checkpoint_sha256']}")
    print("RAW UNCALIBRATED -- NO HELD-OUT EVALUATION")
    print("=" * 110)


def self_test():
    assert EXPECTED_R9_SHA != EXPECTED_OOF_SHA
    assert tuple(range(1, 5)) == (1, 2, 3, 4)
    assert "test" not in str(OUT).lower()
    print("[PASS] RUN_10A configuration self-test")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold", type=int,
        choices=(1, 2, 3, 4), required=True
    )
    parser.add_argument(
        "--smoke-only", action="store_true"
    )
    parser.add_argument(
        "--resume", action="store_true"
    )
    args = parser.parse_args()

    self_test()
    train_fold(args.fold, args.smoke_only, args.resume)


if __name__ == "__main__":
    main()
