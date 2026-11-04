from __future__ import annotations

"""
B-FCEF-F RUN_07 / S0-A Final-Expert five-fold cross-fit.

Scientific purpose
------------------
Train only the permanent S/L/C/T Final Experts plus the shared Physical
Context Bus on the frozen 47-video Train universe, then export honest OOF
expert predictions for the later Final-Expert counterfactual / U-H compiler.

This is a reference/trainability cross-fit stage, NOT architecture selection.
No official validation/test data are accessed.

Frozen S0-A objective
---------------------
For each frame, compute one immutable Hungarian assignment I_B from the
frozen R0 baseline logits/boxes. Reuse I_B for every expert and every epoch.

For each expert k in {S,L,C,T}:
  * cls branch: native_cls(h + candidate_mask * delta_h_cls_k)
                -> official Deformable-DETR classification loss under I_B
  * box branch: native_box(h + candidate_mask * delta_h_box_k, reference)
                -> official L1 + GIoU losses under I_B

The L specialist's expected-IoU head also receives a small realized-IoU MSE
auxiliary target under the same fixed assignment.

Trainable in S0-A
-----------------
  * expert_s
  * expert_l
  * expert_c
  * expert_t
  * context_bus

Frozen in S0-A
--------------
  * R0 native classification / bbox heads
  * self-assessment heads
  * cls/box TSK gates
  * coordination heads
  * simplex router

OOF export
----------
Each video is predicted only by the fold model whose training set excluded
that video. Expert decoded logits and boxes are persisted as float32 to avoid
half-precision threshold/matching edge drift in the subsequent exact replay.
"""

import argparse
import copy
import gc
import hashlib
import inspect
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import torch
import torch.nn.functional as F


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path("D:/MARIVISION")
EVAL = ROOT / "H0_workbench" / "eval_tools"
E0 = EVAL / "e0_cache"
DERIVED = EVAL / "bfcef_f_derived_cache"
SPATIAL_DIR = DERIVED / "spatial_v2"
TEMP_DIR = DERIVED / "temporal_kmax7_v1"
FOLD_DIR = EVAL / "e0_adapters"

OUT = EVAL / "bfcef_f_s0" / "final_expert_crossfit_v1"
MATCH_DIR = OUT / "fixed_match_v1"

R0_CHECKPOINT = ROOT / "checkpoints_r0" / "checkpoint_best.pth"
CANONICAL_MANIFEST = E0 / "bfcef_e0_cache_manifest.json"
FIRST_TEMP_MANIFEST = E0 / "bfcef_e0_temporal_cache_manifest.json"
SPATIAL_MANIFEST = SPATIAL_DIR / "bfcef_f_spatial_evidence_manifest.json"
TEMP_MANIFEST = TEMP_DIR / "bfcef_f_temporal_state_manifest.json"
FOLD_MANIFEST = FOLD_DIR / "bfcef_e0_5fold_manifest.json"
SCREENING_CONFIG = EVAL / "screening_config_s0.json"
S0_DATA_SOURCE = EVAL / "bfcef_f_s0_data.py"


# =============================================================================
# FROZEN PROVENANCE ROOTS
# =============================================================================

FROZEN_R0_SHA = (
    "28e6288c3f6769c2d5275aa77ddb248735b55a09cd46a6b570e3f117890b268b"
)
FROZEN_CAN_SHA = (
    "210ad12de4ebe02cdbe000956972955ae58ceb8f0d222b703787f33a49af03cd"
)
FROZEN_FIRST_TEMP_SHA = (
    "e4754d9c82c60836d817e49ceb55201807c22405677558bb2a69b3e96a6bd286"
)
FROZEN_SPATIAL_SHA = (
    "8899edf3eb3b46428dc3a2186058063c74045d14e0c75744fc0a0793daefa9b5"
)
FROZEN_TEMP_SHA = (
    "d84d2b75cc043da855c1490a66e727b677c7c70cb7ab881c7c08b2308bf3e985"
)
FROZEN_FOLD_SHA = (
    "c923f8abd5833996331566582f568b2b0e9ad1d7d17f606c8ddae693b6b8a103"
)
FROZEN_SCREENING_CONFIG_SHA = (
    "f5a03e543c13360b50b38ec5850862cf36d2801d90b6c24b1a04f794e2281c26"
)

# Core source SHAs from BFCEF_F_CORRECTED_2026-10-07.zip, which is the tree
# that was promoted and certified by RUN_00 through RUN_06.
FROZEN_CORE_SOURCE_SHA = {
    "bfcef_f/model.py": "e63cc37a6c3c31c8563fa560047c3fc7a29237d676a735993e9cd1cead034006",
    "bfcef_f/bci.py": "357b55d3b9b675c0b6074224b45ff7c1b74204c51a1d4f0c0ff2f5a735b20ed9",
    "bfcef_f/context_bus.py": "9426bee1f7d32960d33508f4b4fa81a4a433ed311ea48517f6fc69ce54e4ff69",
    "bfcef_f/experts/scale.py": "a526955008d689df6e04e7cc71fd901a957c5815f9f21a779bfaacc8f0847997",
    "bfcef_f/experts/localization.py": "d2e65e351e89e7c60aa5b94466dfbddd2fbced02201a7e35daeb32ce9186c416",
    "bfcef_f/experts/semantic.py": "8b072947127f0c12f665b9ca06ef4330d2003d903f4da6bf4486b0b3b93aec58",
    "bfcef_f/experts/temporal.py": "dd104bd82f3f4e1cd485b4335ddc1cacd58f3f439cbde1e5ca932a6eb13e8c11",
    "bfcef_f/adapters/deformable_detr.py": "ce55a8192b23960973288fc431f7aac88fdacc8fddccf017573fc4c772312979",
}


# =============================================================================
# S0-A CONTRACT
# =============================================================================

OBJECTIVE_VERSION = "s0a_final_expert_fixed_baseline_match_task_decoupled_v1"
MATCH_SCHEMA = "bfcef_f_s0a_fixed_baseline_hungarian_v1"
OOF_SCHEMA = "bfcef_f_s0a_final_expert_oof_v1"

EXPERT_ORDER = ("S", "L", "C", "T")

NUM_VIDEOS = 47
NUM_FOLDS = 5
NUM_QUERIES = 300
NUM_CLASSES = 7
EXPECTED_TOTAL_FRAMES = 20_075
K_S0 = 3

EPOCHS = 2
LR = 2.0e-4
WEIGHT_DECAY = 1.0e-4
GRAD_CLIP = 0.1
L_IOU_AUX_WEIGHT = 0.25
SEED = 20261007

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# =============================================================================
# BASIC HELPERS
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


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cpu_state_dict(module: torch.nn.Module) -> dict:
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}


def count_trainable(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# =============================================================================
# SOURCE / MANIFEST VERIFICATION
# =============================================================================


def verify_source_and_roots() -> dict:
    require_sha(R0_CHECKPOINT, FROZEN_R0_SHA, "R0 checkpoint")
    require_sha(CANONICAL_MANIFEST, FROZEN_CAN_SHA, "canonical manifest")
    require_sha(
        FIRST_TEMP_MANIFEST,
        FROZEN_FIRST_TEMP_SHA,
        "first-order temporal manifest",
    )
    require_sha(SPATIAL_MANIFEST, FROZEN_SPATIAL_SHA, "spatial manifest")
    require_sha(TEMP_MANIFEST, FROZEN_TEMP_SHA, "K-step temporal manifest")
    require_sha(FOLD_MANIFEST, FROZEN_FOLD_SHA, "5-fold manifest")
    require_sha(
        SCREENING_CONFIG,
        FROZEN_SCREENING_CONFIG_SHA,
        "screening_config_s0.json",
    )

    if not S0_DATA_SOURCE.exists():
        raise AssertionError(f"[FATAL] Missing certified S0 consumer: {S0_DATA_SOURCE}")

    for rel, expected in FROZEN_CORE_SOURCE_SHA.items():
        require_sha(EVAL / rel, expected, rel)

    can = load_json(CANONICAL_MANIFEST)
    spatial = load_json(SPATIAL_MANIFEST)
    temporal = load_json(TEMP_MANIFEST)

    if int(can["_meta"]["actual_total_frames"]) != EXPECTED_TOTAL_FRAMES:
        raise AssertionError("[FATAL] Canonical frame cardinality drift.")
    if int(spatial["_meta"]["total_frames"]) != EXPECTED_TOTAL_FRAMES:
        raise AssertionError("[FATAL] Spatial frame cardinality drift.")
    if int(temporal["_meta"]["total_frames"]) != EXPECTED_TOTAL_FRAMES:
        raise AssertionError("[FATAL] Temporal frame cardinality drift.")
    if int(temporal["_meta"]["k_max"]) < K_S0:
        raise AssertionError("[FATAL] Temporal Kmax cannot satisfy S0 K=3.")

    # Bind all 94 derived files once before training. RUN_06 already proved
    # their full semantic alignment; this startup pass catches post-RUN_06
    # mutation without repeating the entire integration audit.
    spatial_ledger = spatial["derived_spatial_shard_sha256_ledger"]
    temporal_ledger = temporal["derived_temporal_state_shard_sha256_ledger"]

    if len(spatial_ledger) != NUM_VIDEOS or len(temporal_ledger) != NUM_VIDEOS:
        raise AssertionError("[FATAL] Derived shard ledger cardinality drift.")

    for name, expected in spatial_ledger.items():
        require_sha(SPATIAL_DIR / name, expected, f"spatial shard {name}")

    for name, expected in temporal_ledger.items():
        require_sha(TEMP_DIR / name, expected, f"temporal shard {name}")

    print("[+] Provenance/source PASS: corrected core tree + frozen manifests + 94 derived shards")
    print(f"    S0 consumer SHA-256: {sha256(S0_DATA_SOURCE)}")

    return {
        "canonical": can,
        "spatial": spatial,
        "temporal": temporal,
    }


def parse_folds() -> Dict[int, List[int]]:
    payload = load_json(FOLD_MANIFEST)
    folds = {int(k): [int(v) for v in values] for k, values in payload["folds"].items()}

    if sorted(folds.keys()) != [0, 1, 2, 3, 4]:
        raise AssertionError("[FATAL] Fold keys drift.")

    capacities = [len(folds[i]) for i in range(NUM_FOLDS)]
    if capacities != [10, 10, 9, 9, 9]:
        raise AssertionError(f"[FATAL] Fold capacities drift: {capacities}")

    union = sorted(v for values in folds.values() for v in values)
    if union != list(range(1, NUM_VIDEOS + 1)):
        raise AssertionError("[FATAL] Fold partition is not exact disjoint union 1..47.")

    return folds


# =============================================================================
# IMPORTS / AUTHORITATIVE NATIVE HEADS + CRITERION
# =============================================================================


def find_deformable_root() -> Path:
    for p in (ROOT / "Deformable-DETR", ROOT / "deformable_detr"):
        if (
            (p / "models" / "deformable_detr.py").exists()
            and (p / "util" / "box_ops.py").exists()
        ):
            return p
    raise AssertionError("[FATAL] Cannot locate authoritative Deformable-DETR repository.")


def prepare_imports():
    deformable_root = find_deformable_root()

    # EVAL first so certified local bfcef_f wins; Deformable root supplies
    # models/util; ROOT remains last fallback.
    for p in (str(ROOT), str(deformable_root), str(EVAL)):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)

    from bfcef_f.config import BFCEFFScreeningConfig
    from bfcef_f.contracts import CanonicalObjectState
    from bfcef_f.model import BFCEFFModel
    from bfcef_f_s0_data import frame_inputs_from_records
    from r0_preflight import get_authoritative_r0_args
    from models.deformable_detr import build as build_deformable_detr
    from util import box_ops

    return (
        BFCEFFScreeningConfig,
        CanonicalObjectState,
        BFCEFFModel,
        frame_inputs_from_records,
        get_authoritative_r0_args,
        build_deformable_detr,
        box_ops,
    )


def build_native_templates_and_criterion(
    get_authoritative_r0_args,
    build_deformable_detr,
):
    args = get_authoritative_r0_args()

    if int(args.num_queries) != NUM_QUERIES:
        raise AssertionError("[FATAL] R0 num_queries drift.")
    if bool(args.two_stage):
        raise AssertionError("[FATAL] R0 two_stage drift.")
    if bool(args.with_box_refine):
        raise AssertionError("[FATAL] R0 with_box_refine drift.")

    # No live R0 inference is needed. Build/load on CPU, copy only the native
    # final heads, then release the full detector before S0-A training.
    args.device = "cpu"
    r0_model, criterion, _ = build_deformable_detr(args)

    checkpoint = torch.load(R0_CHECKPOINT, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    r0_model.load_state_dict(state, strict=True)
    r0_model.eval()

    native_cls_template = copy.deepcopy(r0_model.class_embed[-1]).cpu().eval()
    native_box_template = copy.deepcopy(r0_model.bbox_embed[-1]).cpu().eval()

    for p in native_cls_template.parameters():
        p.requires_grad_(False)
    for p in native_box_template.parameters():
        p.requires_grad_(False)

    criterion.to(DEVICE)
    criterion.eval()

    del r0_model, checkpoint, state
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return native_cls_template, native_box_template, criterion


# =============================================================================
# DATA ACCESS
# =============================================================================


def video_paths(vid: int) -> Tuple[Path, Path, Path]:
    return (
        E0 / f"bfcef_e0_cache_vid{vid}.pth",
        SPATIAL_DIR / f"bfcef_f_spatial_evidence_vid{vid}.pth",
        TEMP_DIR / f"bfcef_f_temporal_state_kmax7_vid{vid}.pth",
    )


def load_video_triplet(vid: int):
    can_path, spatial_path, temporal_path = video_paths(vid)

    canonical = torch.load(can_path, map_location="cpu")
    spatial = torch.load(spatial_path, map_location="cpu")
    temporal = torch.load(temporal_path, map_location="cpu")

    if not (len(canonical) == len(spatial) == len(temporal)):
        raise AssertionError(
            f"[FATAL] Video {vid} length mismatch: "
            f"canonical={len(canonical)}, spatial={len(spatial)}, temporal={len(temporal)}"
        )

    for idx in (0, len(canonical) - 1):
        image_id = int(canonical[idx]["image_id"])
        for label, record in (("spatial", spatial[idx]), ("temporal", temporal[idx])):
            if (
                int(record["image_id"]) != image_id
                or int(record["video_id"]) != vid
                or int(record["seq_idx"]) != idx
            ):
                raise AssertionError(
                    f"[FATAL] {label} edge alignment drift vid={vid}, idx={idx}"
                )

    return canonical, spatial, temporal


def make_O_X(
    canonical_record: dict,
    spatial_record: dict,
    temporal_record: dict,
    CanonicalObjectState,
    frame_inputs_from_records,
):
    z = canonical_record["d_logits"].to(DEVICE, dtype=torch.float32)
    b = canonical_record["d_boxes"].to(DEVICE, dtype=torch.float32)
    h = canonical_record["q_native"].to(DEVICE, dtype=torch.float32)
    r = canonical_record["ref_pts"].to(DEVICE, dtype=torch.float32)

    O = CanonicalObjectState(
        h=h,
        z=z,
        b=b,
        r=r,
        s=z.sigmoid().max(dim=-1).values,
        F_maps={},
        native_handle={
            "source": "s0a_final_expert_crossfit",
            "image_id": int(canonical_record["image_id"]),
        },
        capabilities={
            "base": True,
            "scale": True,
            "scale.cls": True,
            "scale.box": True,
            "localization": True,
            "localization.cls": True,
            "localization.box": True,
            "semantic": True,
            "semantic.cls": True,
            "semantic.box": True,
            "temporal": True,
            "temporal.cls": True,
            "temporal.box": True,
        },
    )

    X = frame_inputs_from_records(
        canonical_record,
        scale_spatial=spatial_record["scale_spatial"],
        loc_spatial=spatial_record["loc_spatial"],
        temporal_kmax_record=temporal_record,
        K=K_S0,
        device=DEVICE,
    )

    return O, X


# =============================================================================
# MODEL / FINAL EXPERT FORWARD
# =============================================================================


def construct_model(
    native_cls_template,
    native_box_template,
    config,
    BFCEFFModel,
):
    model = BFCEFFModel(
        native_class_embed=copy.deepcopy(native_cls_template),
        native_bbox_embed=copy.deepcopy(native_box_template),
        candidate_topk=100,
        config=config,
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    # Fail closed on the corrected C-Expert interface: h,cues,z,boxes,context.
    sig = inspect.signature(model.expert_c.forward)
    names = list(sig.parameters.keys())
    if names != ["h", "cues_c", "z_base", "boxes", "context"]:
        raise AssertionError(
            "[FATAL] Corrected C-Expert signature drift: "
            f"got {names}, expected ['h','cues_c','z_base','boxes','context']"
        )

    # Freeze every architectural component first.
    for p in model.parameters():
        p.requires_grad_(False)

    # S0-A learns Final Experts + shared Context Bus only.
    for module in (
        model.expert_s,
        model.expert_l,
        model.expert_c,
        model.expert_t,
        model.context_bus,
    ):
        for p in module.parameters():
            p.requires_grad_(True)

    # Native heads remain frozen, including through parent train().
    if any(p.requires_grad for p in model.adapter.parameters()):
        raise AssertionError("[FATAL] Native adapter is trainable in S0-A.")

    # Explicitly ensure all later-stage components are frozen.
    frozen_groups = (
        model.self_assessment,
        model.gate_cls,
        model.gate_box,
        model.coord_cls,
        model.coord_box,
        model.router,
    )
    for module in frozen_groups:
        if any(p.requires_grad for p in module.parameters()):
            raise AssertionError("[FATAL] Later-stage gate/router component is trainable in S0-A.")

    return model


def final_expert_forward(model, O, X):
    # No external metadata are injected in S0-A. AUTO therefore uses the
    # learned visual estimator / Scene-Q path, with configured training-time
    # uncertainty augmentation and deterministic eval-time behavior.
    C_t = model.context_bus(
        m_ext=None,
        m_hat_vis=None,
        scene_q=X.scene_q,
        m_avail=None,
        mode="auto",
    )

    dS_cls, dS_box, _ = model.expert_s(
        O.h,
        X.cues_s,
        O.b,
        X.scale_spatial,
        C_t,
    )

    dL_cls, dL_box, _, L_expected_iou = model.expert_l(
        O.h,
        X.cues_l,
        O.b,
        O.r,
        X.loc_spatial,
        C_t,
    )

    dC_cls, dC_box, _ = model.expert_c(
        O.h,
        X.cues_c,
        O.z,
        O.b,
        C_t,
    )

    dT_cls, dT_box, _ = model.expert_t(
        O.h,
        X.temporal,
        C_t,
    )

    delta_cls = torch.stack(
        [dS_cls, dL_cls, dC_cls, dT_cls],
        dim=1,
    )
    delta_box = torch.stack(
        [dS_box, dL_box, dC_box, dT_box],
        dim=1,
    )

    candidate_set = model.bci.extract(O.z)
    candidate_mask = torch.zeros(
        NUM_QUERIES,
        1,
        device=O.h.device,
        dtype=O.h.dtype,
    )
    if candidate_set.queries:
        candidate_mask[candidate_set.queries] = 1.0

    delta_cls = delta_cls * candidate_mask.unsqueeze(1)
    delta_box = delta_box * candidate_mask.unsqueeze(1)

    return delta_cls, delta_box, L_expected_iou, candidate_mask


# =============================================================================
# FIXED BASELINE HUNGARIAN ASSIGNMENT CACHE
# =============================================================================


def match_manifest_path() -> Path:
    return MATCH_DIR / "bfcef_f_s0a_fixed_match_manifest.json"


def match_shard_path(vid: int) -> Path:
    return MATCH_DIR / f"bfcef_f_s0a_fixed_match_vid{vid}.pth"


def verify_match_cache() -> bool:
    manifest_path = match_manifest_path()
    if not manifest_path.exists():
        return False

    try:
        payload = load_json(manifest_path)
        meta = payload["_meta"]
        checks = {
            "schema_version": MATCH_SCHEMA,
            "canonical_manifest_sha256": FROZEN_CAN_SHA,
            "r0_checkpoint_sha256": FROZEN_R0_SHA,
            "num_videos": NUM_VIDEOS,
            "total_frames": EXPECTED_TOTAL_FRAMES,
            "num_queries": NUM_QUERIES,
        }
        for key, value in checks.items():
            if meta.get(key) != value:
                return False

        ledger = payload["shard_sha256_ledger"]
        if len(ledger) != NUM_VIDEOS:
            return False

        for name, expected in ledger.items():
            p = MATCH_DIR / name
            if not p.exists() or sha256(p) != expected:
                return False

        return True
    except Exception:
        return False


def build_match_cache(criterion) -> None:
    MATCH_DIR.mkdir(parents=True, exist_ok=True)

    ledger: Dict[str, str] = {}
    total = 0
    t0 = time.time()

    print("[*] Building immutable baseline Hungarian assignment cache once...")

    with torch.no_grad():
        for vid in range(1, NUM_VIDEOS + 1):
            can_path, _, _ = video_paths(vid)
            records = torch.load(can_path, map_location="cpu")
            output = []

            for idx, record in enumerate(records):
                z = record["d_logits"].to(DEVICE, dtype=torch.float32)
                b = record["d_boxes"].to(DEVICE, dtype=torch.float32)
                targets = [{
                    "boxes": record["gt_boxes_norm"].to(DEVICE, dtype=torch.float32),
                    "labels": record["gt_labels"].to(DEVICE, dtype=torch.long),
                }]

                indices = criterion.matcher(
                    {
                        "pred_logits": z.unsqueeze(0),
                        "pred_boxes": b.unsqueeze(0),
                    },
                    targets,
                )

                src_idx, tgt_idx = indices[0]

                output.append({
                    "image_id": int(record["image_id"]),
                    "video_id": vid,
                    "seq_idx": idx,
                    "src_idx": src_idx.detach().cpu().to(torch.int32),
                    "tgt_idx": tgt_idx.detach().cpu().to(torch.int32),
                })
                total += 1

            path = match_shard_path(vid)
            torch.save(output, path)
            ledger[path.name] = sha256(path)

            print(
                f"    [+] Match cache video {vid:02d}/47: "
                f"{len(output):4d} frames"
            )

            del records, output
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    if total != EXPECTED_TOTAL_FRAMES:
        raise AssertionError(
            f"[FATAL] Fixed-match frame total {total} != {EXPECTED_TOTAL_FRAMES}"
        )

    manifest = {
        "_meta": {
            "schema_version": MATCH_SCHEMA,
            "canonical_manifest_sha256": FROZEN_CAN_SHA,
            "r0_checkpoint_sha256": FROZEN_R0_SHA,
            "num_videos": NUM_VIDEOS,
            "total_frames": total,
            "num_queries": NUM_QUERIES,
            "matching_semantics": "one_frozen_R0_Hungarian_assignment_per_frame_reused_for_all_experts",
        },
        "shard_sha256_ledger": ledger,
    }
    save_json(match_manifest_path(), manifest)

    print(
        "[+] Fixed baseline match cache COMPLETE: "
        f"SHA={sha256(match_manifest_path())}, "
        f"runtime={(time.time() - t0) / 60.0:.2f} min"
    )


def ensure_match_cache(criterion) -> None:
    if verify_match_cache():
        print(
            "[+] Reusing certified fixed baseline match cache: "
            f"{sha256(match_manifest_path())}"
        )
        return

    build_match_cache(criterion)

    if not verify_match_cache():
        raise AssertionError("[FATAL] Fixed-match cache failed self-verification.")


# =============================================================================
# LOSS UNDER FIXED BASELINE ASSIGNMENT
# =============================================================================


def make_indices(match_record: dict):
    return [(
        match_record["src_idx"].to(DEVICE, dtype=torch.long),
        match_record["tgt_idx"].to(DEVICE, dtype=torch.long),
    )]


def task_decoupled_loss(
    model,
    criterion,
    box_ops,
    O,
    canonical_record: dict,
    delta_cls,
    delta_box,
    L_expected_iou,
    match_record: dict,
):
    indices = make_indices(match_record)

    targets = [{
        "boxes": canonical_record["gt_boxes_norm"].to(DEVICE, dtype=torch.float32),
        "labels": canonical_record["gt_labels"].to(DEVICE, dtype=torch.long),
    }]

    # Exact single-process analogue of SetCriterion.forward normalisation.
    num_boxes = float(max(1, int(targets[0]["labels"].numel())))

    w_ce = float(criterion.weight_dict.get("loss_ce", 1.0))
    w_bbox = float(criterion.weight_dict.get("loss_bbox", 1.0))
    w_giou = float(criterion.weight_dict.get("loss_giou", 1.0))

    cls_terms = []
    box_terms = []
    decoded_box = []

    for expert_index in range(4):
        z_k = model.adapter.decode_cls(
            O.h + delta_cls[:, expert_index]
        )

        b_k = model.adapter.decode_box(
            O.h + delta_box[:, expert_index],
            O.r,
        )
        decoded_box.append(b_k)

        cls_output = {
            "pred_logits": z_k.unsqueeze(0),
            "pred_boxes": O.b.unsqueeze(0),
        }
        box_output = {
            "pred_logits": O.z.unsqueeze(0),
            "pred_boxes": b_k.unsqueeze(0),
        }

        cls_dict = criterion.loss_labels(
            cls_output,
            targets,
            indices,
            num_boxes,
            log=False,
        )
        box_dict = criterion.loss_boxes(
            box_output,
            targets,
            indices,
            num_boxes,
        )

        cls_terms.append(w_ce * cls_dict["loss_ce"])
        box_terms.append(
            w_bbox * box_dict["loss_bbox"]
            + w_giou * box_dict["loss_giou"]
        )

    cls_loss = torch.stack(cls_terms).mean()
    box_loss = torch.stack(box_terms).mean()

    # L expected-IoU auxiliary under the same fixed baseline assignment.
    src_idx, tgt_idx = indices[0]
    if int(src_idx.numel()) > 0:
        pred_l = decoded_box[1][src_idx]
        gt_l = targets[0]["boxes"][tgt_idx]

        iou_matrix, _ = box_ops.box_iou(
            box_ops.box_cxcywh_to_xyxy(pred_l),
            box_ops.box_cxcywh_to_xyxy(gt_l),
        )
        realized_iou = torch.diagonal(iou_matrix).detach()
        predicted_iou = L_expected_iou[src_idx].reshape(-1)
        l_iou_loss = F.mse_loss(predicted_iou, realized_iou)
    else:
        l_iou_loss = L_expected_iou.sum() * 0.0

    total = cls_loss + box_loss + L_IOU_AUX_WEIGHT * l_iou_loss
    return total, cls_loss, box_loss, l_iou_loss


# =============================================================================
# EXACT-OBJECTIVE PREFLIGHT
# =============================================================================


def run_exact_objective_preflight(
    folds,
    config,
    native_cls_template,
    native_box_template,
    criterion,
    box_ops,
    BFCEFFModel,
    CanonicalObjectState,
    frame_inputs_from_records,
):
    print("[*] RUN_07 exact-objective preflight: one real fold-0 Train frame")

    train_videos = sorted(set(range(1, 48)) - set(folds[0]))
    vid = train_videos[0]
    canonical, spatial, temporal = load_video_triplet(vid)
    idx = len(canonical) // 2

    O, X = make_O_X(
        canonical[idx],
        spatial[idx],
        temporal[idx],
        CanonicalObjectState,
        frame_inputs_from_records,
    )

    set_seed(SEED)
    model = construct_model(
        native_cls_template,
        native_box_template,
        config,
        BFCEFFModel,
    )
    model.train()

    # Native-head reconstruction must still agree with the cached baseline.
    with torch.no_grad():
        z_native = model.adapter.decode_cls(O.h)
        b_native = model.adapter.decode_box(O.h, O.r)
        dz = float((z_native - O.z).abs().max().item())
        db = float((b_native - O.b).abs().max().item())

    if dz >= 1e-4 or db >= 1e-4:
        raise AssertionError(
            f"[FATAL] RUN_07 native-head parity failed: dz={dz:.3e}, db={db:.3e}"
        )

    # Direct baseline match for preflight only. Full run materializes/reuses it.
    with torch.no_grad():
        targets = [{
            "boxes": canonical[idx]["gt_boxes_norm"].to(DEVICE, dtype=torch.float32),
            "labels": canonical[idx]["gt_labels"].to(DEVICE, dtype=torch.long),
        }]
        indices = criterion.matcher(
            {
                "pred_logits": O.z.unsqueeze(0),
                "pred_boxes": O.b.unsqueeze(0),
            },
            targets,
        )

    match_record = {
        "src_idx": indices[0][0].detach().cpu().to(torch.int32),
        "tgt_idx": indices[0][1].detach().cpu().to(torch.int32),
    }

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=LR,
        weight_decay=0.0,  # smoke-only: avoid decay-only motion
    )
    optimizer.zero_grad(set_to_none=True)

    d_cls, d_box, l_expected_iou, candidate_mask = final_expert_forward(model, O, X)
    total, cls_loss, box_loss, l_iou_loss = task_decoupled_loss(
        model,
        criterion,
        box_ops,
        O,
        canonical[idx],
        d_cls,
        d_box,
        l_expected_iou,
        match_record,
    )

    if not torch.isfinite(total):
        raise AssertionError("[FATAL] RUN_07 preflight produced non-finite loss.")

    total.backward()

    grad_report = {}
    for name, module in (
        ("S", model.expert_s),
        ("L", model.expert_l),
        ("C", model.expert_c),
        ("T", model.expert_t),
    ):
        cls_grad = module.proj_cls.weight.grad
        box_grad = module.proj_box.weight.grad
        cls_norm = 0.0 if cls_grad is None else float(cls_grad.norm().item())
        box_norm = 0.0 if box_grad is None else float(box_grad.norm().item())
        if cls_norm <= 0.0 or box_norm <= 0.0:
            raise AssertionError(
                f"[FATAL] RUN_07 missing {name} projector gradient: "
                f"cls={cls_norm:.3e}, box={box_norm:.3e}"
            )
        grad_report[name] = (cls_norm, box_norm)

    torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
    optimizer.step()

    candidate_count = int(candidate_mask.sum().item())
    if candidate_count <= 0 or candidate_count > NUM_QUERIES:
        raise AssertionError(f"[FATAL] Invalid BCI candidate-query count: {candidate_count}")

    print(
        "[+] RUN_07 exact-objective PREFLIGHT PASS: "
        f"video={vid}, seq={idx}, candidates={candidate_count}, "
        f"native_dz={dz:.3e}, native_db={db:.3e}, "
        f"total={float(total.item()):.6f}, "
        f"cls={float(cls_loss.item()):.6f}, "
        f"box={float(box_loss.item()):.6f}, "
        f"L-IoU={float(l_iou_loss.item()):.6f}"
    )
    for name, (gc_, gb_) in grad_report.items():
        print(f"    {name}: grad_cls={gc_:.3e}, grad_box={gb_:.3e}")

    del model, optimizer, canonical, spatial, temporal, O, X
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


# =============================================================================
# FOLD PATHS / RESUME
# =============================================================================


def fold_dir(fold: int) -> Path:
    return OUT / f"fold{fold}"


def fold_checkpoint_path(fold: int) -> Path:
    return fold_dir(fold) / f"final_expert_fold{fold}.pth"


def fold_certificate_path(fold: int) -> Path:
    return fold_dir(fold) / f"fold{fold}_complete.json"


def fold_oof_path(fold: int, vid: int) -> Path:
    return fold_dir(fold) / f"bfcef_f_s0a_oof_vid{vid}.pth"


def checkpoint_matches_contract(path: Path, fold: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = torch.load(path, map_location="cpu")
        meta = payload["_meta"]
        expected = {
            "objective_version": OBJECTIVE_VERSION,
            "fold": fold,
            "epochs": EPOCHS,
            "r0_checkpoint_sha256": FROZEN_R0_SHA,
            "canonical_manifest_sha256": FROZEN_CAN_SHA,
            "spatial_manifest_sha256": FROZEN_SPATIAL_SHA,
            "temporal_manifest_sha256": FROZEN_TEMP_SHA,
            "fold_manifest_sha256": FROZEN_FOLD_SHA,
            "screening_config_sha256": FROZEN_SCREENING_CONFIG_SHA,
        }
        return all(meta.get(k) == v for k, v in expected.items())
    except Exception:
        return False


def certified_fold_complete(fold: int, oof_videos: List[int]) -> bool:
    cert_path = fold_certificate_path(fold)
    ckpt_path = fold_checkpoint_path(fold)

    if not cert_path.exists() or not checkpoint_matches_contract(ckpt_path, fold):
        return False

    try:
        cert = load_json(cert_path)
        meta = cert["_meta"]
        if meta.get("checkpoint_sha256") != sha256(ckpt_path):
            return False
        if meta.get("objective_version") != OBJECTIVE_VERSION:
            return False
        if meta.get("oof_videos") != oof_videos:
            return False

        ledger = cert["oof_shard_sha256_ledger"]
        expected_names = {fold_oof_path(fold, vid).name for vid in oof_videos}
        if set(ledger.keys()) != expected_names:
            return False

        for name, expected in ledger.items():
            p = fold_dir(fold) / name
            if not p.exists() or sha256(p) != expected:
                return False

        return True
    except Exception:
        return False


def load_fold_checkpoint_into_model(path: Path, model) -> dict:
    payload = torch.load(path, map_location="cpu")
    model.expert_s.load_state_dict(payload["expert_s"], strict=True)
    model.expert_l.load_state_dict(payload["expert_l"], strict=True)
    model.expert_c.load_state_dict(payload["expert_c"], strict=True)
    model.expert_t.load_state_dict(payload["expert_t"], strict=True)
    model.context_bus.load_state_dict(payload["context_bus"], strict=True)
    return payload


# =============================================================================
# TRAIN FOLD
# =============================================================================


def train_fold(
    fold: int,
    folds,
    config,
    native_cls_template,
    native_box_template,
    criterion,
    box_ops,
    BFCEFFModel,
    CanonicalObjectState,
    frame_inputs_from_records,
):
    oof_videos = sorted(folds[fold])
    train_videos = sorted(set(range(1, NUM_VIDEOS + 1)) - set(oof_videos))

    if set(train_videos) & set(oof_videos):
        raise AssertionError("[FATAL] Fold leakage.")

    set_seed(SEED + fold)
    model = construct_model(
        native_cls_template,
        native_box_template,
        config,
        BFCEFFModel,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    fold_dir(fold).mkdir(parents=True, exist_ok=True)

    print("=" * 116)
    print(
        f"S0-A FINAL EXPERT CROSS-FIT | FOLD {fold} | "
        f"TRAIN {len(train_videos)} videos | OOF {len(oof_videos)} videos"
    )
    print(f"OOF videos: {oof_videos}")
    print(f"Trainable parameters: {count_trainable(model):,}")
    print("=" * 116)

    epoch_summaries = []

    for epoch in range(EPOCHS):
        model.train()

        rng = random.Random(SEED + fold * 1000 + epoch)
        video_order = list(train_videos)
        rng.shuffle(video_order)

        sum_total = 0.0
        sum_cls = 0.0
        sum_box = 0.0
        sum_iou = 0.0
        frames = 0
        t_epoch = time.time()

        for video_position, vid in enumerate(video_order, start=1):
            canonical, spatial, temporal = load_video_triplet(vid)
            matches = torch.load(match_shard_path(vid), map_location="cpu")

            if len(matches) != len(canonical):
                raise AssertionError(f"[FATAL] Fixed-match length mismatch vid={vid}")

            frame_order = list(range(len(canonical)))
            rng.shuffle(frame_order)

            for idx in frame_order:
                O, X = make_O_X(
                    canonical[idx],
                    spatial[idx],
                    temporal[idx],
                    CanonicalObjectState,
                    frame_inputs_from_records,
                )

                optimizer.zero_grad(set_to_none=True)

                d_cls, d_box, l_expected_iou, _ = final_expert_forward(model, O, X)

                total, cls_loss, box_loss, l_iou_loss = task_decoupled_loss(
                    model,
                    criterion,
                    box_ops,
                    O,
                    canonical[idx],
                    d_cls,
                    d_box,
                    l_expected_iou,
                    matches[idx],
                )

                if not torch.isfinite(total):
                    raise AssertionError(
                        f"[FATAL] Non-finite S0-A loss fold={fold}, epoch={epoch+1}, "
                        f"vid={vid}, idx={idx}"
                    )

                total.backward()
                torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
                optimizer.step()

                sum_total += float(total.detach().item())
                sum_cls += float(cls_loss.detach().item())
                sum_box += float(box_loss.detach().item())
                sum_iou += float(l_iou_loss.detach().item())
                frames += 1

                if frames % 500 == 0:
                    print(
                        f"    fold={fold} epoch={epoch+1}/{EPOCHS} "
                        f"frames={frames:,} "
                        f"mean_total={sum_total/frames:.5f} "
                        f"mean_cls={sum_cls/frames:.5f} "
                        f"mean_box={sum_box/frames:.5f}"
                    )

                del O, X, d_cls, d_box, l_expected_iou
                del total, cls_loss, box_loss, l_iou_loss

            print(
                f"    [+] fold={fold} epoch={epoch+1}/{EPOCHS} "
                f"video {video_position:02d}/{len(video_order)} "
                f"(vid={vid:02d}) complete"
            )

            del canonical, spatial, temporal, matches
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        if frames <= 0:
            raise AssertionError("[FATAL] Empty S0-A training epoch.")

        summary = {
            "epoch": epoch + 1,
            "frames": frames,
            "mean_total_loss": sum_total / frames,
            "mean_cls_loss": sum_cls / frames,
            "mean_box_loss": sum_box / frames,
            "mean_l_iou_loss": sum_iou / frames,
            "runtime_minutes": (time.time() - t_epoch) / 60.0,
        }
        epoch_summaries.append(summary)

        print(
            f"[+] Fold {fold} epoch {epoch+1}/{EPOCHS} COMPLETE: "
            f"frames={frames:,}, total={summary['mean_total_loss']:.6f}, "
            f"cls={summary['mean_cls_loss']:.6f}, "
            f"box={summary['mean_box_loss']:.6f}, "
            f"L-IoU={summary['mean_l_iou_loss']:.6f}, "
            f"runtime={summary['runtime_minutes']:.2f} min"
        )

    ckpt_path = fold_checkpoint_path(fold)
    checkpoint_payload = {
        "_meta": {
            "objective_version": OBJECTIVE_VERSION,
            "fold": fold,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "l_iou_aux_weight": L_IOU_AUX_WEIGHT,
            "seed": SEED + fold,
            "context_mode_train": "auto",
            "temporal_K": K_S0,
            "r0_checkpoint_sha256": FROZEN_R0_SHA,
            "canonical_manifest_sha256": FROZEN_CAN_SHA,
            "spatial_manifest_sha256": FROZEN_SPATIAL_SHA,
            "temporal_manifest_sha256": FROZEN_TEMP_SHA,
            "fold_manifest_sha256": FROZEN_FOLD_SHA,
            "screening_config_sha256": FROZEN_SCREENING_CONFIG_SHA,
            "s0_data_source_sha256": sha256(S0_DATA_SOURCE),
            "fixed_match_manifest_sha256": sha256(match_manifest_path()),
            "train_videos": train_videos,
            "oof_videos": oof_videos,
            "epoch_summaries": epoch_summaries,
        },
        "expert_s": cpu_state_dict(model.expert_s),
        "expert_l": cpu_state_dict(model.expert_l),
        "expert_c": cpu_state_dict(model.expert_c),
        "expert_t": cpu_state_dict(model.expert_t),
        "context_bus": cpu_state_dict(model.context_bus),
    }
    torch.save(checkpoint_payload, ckpt_path)

    print(
        f"[+] Fold {fold} checkpoint frozen: {ckpt_path.name} "
        f"SHA={sha256(ckpt_path)}"
    )

    return model, ckpt_path, train_videos, oof_videos, epoch_summaries


# =============================================================================
# OOF EXPORT / CERTIFICATE
# =============================================================================


def export_oof_fold(
    fold: int,
    model,
    oof_videos: List[int],
    CanonicalObjectState,
    frame_inputs_from_records,
):
    model.eval()
    ledger: Dict[str, str] = {}
    total = 0

    print(f"[*] Exporting fold {fold} honest OOF Final-Expert predictions...")

    with torch.no_grad():
        for vid in oof_videos:
            canonical, spatial, temporal = load_video_triplet(vid)
            output = []

            for idx in range(len(canonical)):
                O, X = make_O_X(
                    canonical[idx],
                    spatial[idx],
                    temporal[idx],
                    CanonicalObjectState,
                    frame_inputs_from_records,
                )

                d_cls, d_box, l_expected_iou, candidate_mask = final_expert_forward(
                    model, O, X
                )

                cls_logits = []
                box_boxes = []

                for expert_index in range(4):
                    z_k = model.adapter.decode_cls(
                        O.h + d_cls[:, expert_index]
                    )
                    b_k = model.adapter.decode_box(
                        O.h + d_box[:, expert_index],
                        O.r,
                    )

                    if not torch.isfinite(z_k).all() or not torch.isfinite(b_k).all():
                        raise AssertionError(
                            f"[FATAL] Non-finite OOF output fold={fold}, vid={vid}, idx={idx}"
                        )

                    cls_logits.append(z_k.detach().cpu().float())
                    box_boxes.append(b_k.detach().cpu().float())

                expert_cls_logits = torch.stack(cls_logits, dim=0)
                expert_box_boxes = torch.stack(box_boxes, dim=0)

                if expert_cls_logits.shape != (4, NUM_QUERIES, NUM_CLASSES):
                    raise AssertionError("[FATAL] OOF cls tensor shape drift.")
                if expert_box_boxes.shape != (4, NUM_QUERIES, 4):
                    raise AssertionError("[FATAL] OOF box tensor shape drift.")

                candidate_queries = (
                    torch.nonzero(
                        candidate_mask.reshape(-1) > 0.5,
                        as_tuple=False,
                    )
                    .reshape(-1)
                    .detach()
                    .cpu()
                    .to(torch.int16)
                )

                output.append({
                    "image_id": int(canonical[idx]["image_id"]),
                    "video_id": vid,
                    "seq_idx": idx,
                    "candidate_queries": candidate_queries,
                    "expert_cls_logits": expert_cls_logits,
                    "expert_box_boxes": expert_box_boxes,
                    "l_expected_iou": l_expected_iou.detach().cpu().reshape(-1).float(),
                })
                total += 1

                del O, X, d_cls, d_box, l_expected_iou, candidate_mask
                del cls_logits, box_boxes, expert_cls_logits, expert_box_boxes

            path = fold_oof_path(fold, vid)
            torch.save(output, path)
            ledger[path.name] = sha256(path)

            print(
                f"    [+] fold={fold} OOF video {vid:02d}: "
                f"{len(output):4d} frames -> {path.name}"
            )

            del canonical, spatial, temporal, output
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    return ledger, total


def write_fold_certificate(
    fold: int,
    checkpoint_path: Path,
    oof_videos: List[int],
    oof_ledger: dict,
    oof_frames: int,
):
    payload = torch.load(checkpoint_path, map_location="cpu")
    epoch_summaries = payload["_meta"]["epoch_summaries"]

    certificate = {
        "_meta": {
            "objective_version": OBJECTIVE_VERSION,
            "oof_schema": OOF_SCHEMA,
            "fold": fold,
            "epochs": EPOCHS,
            "checkpoint_sha256": sha256(checkpoint_path),
            "r0_checkpoint_sha256": FROZEN_R0_SHA,
            "canonical_manifest_sha256": FROZEN_CAN_SHA,
            "spatial_manifest_sha256": FROZEN_SPATIAL_SHA,
            "temporal_manifest_sha256": FROZEN_TEMP_SHA,
            "fold_manifest_sha256": FROZEN_FOLD_SHA,
            "screening_config_sha256": FROZEN_SCREENING_CONFIG_SHA,
            "s0_data_source_sha256": sha256(S0_DATA_SOURCE),
            "fixed_match_manifest_sha256": sha256(match_manifest_path()),
            "oof_videos": oof_videos,
            "oof_frames": oof_frames,
            "epoch_summaries": epoch_summaries,
        },
        "oof_shard_sha256_ledger": oof_ledger,
    }

    cert_path = fold_certificate_path(fold)
    save_json(cert_path, certificate)

    print(
        f"[+] Fold {fold} certificate frozen: {cert_path.name} "
        f"SHA={sha256(cert_path)}"
    )


# =============================================================================
# FINAL OOF MANIFEST
# =============================================================================


def finalize_oof_manifest(folds: Dict[int, List[int]]) -> None:
    oof_ledger: Dict[str, str] = {}
    checkpoint_ledger: Dict[str, str] = {}
    certificate_ledger: Dict[str, str] = {}

    seen_videos = set()
    total_frames = 0

    for fold in range(NUM_FOLDS):
        oof_videos = sorted(folds[fold])
        if not certified_fold_complete(fold, oof_videos):
            raise AssertionError(f"[FATAL] Fold {fold} is not certified complete.")

        ckpt = fold_checkpoint_path(fold)
        cert = fold_certificate_path(fold)
        cert_payload = load_json(cert)

        checkpoint_ledger[f"fold{fold}/{ckpt.name}"] = sha256(ckpt)
        certificate_ledger[f"fold{fold}/{cert.name}"] = sha256(cert)

        for vid in oof_videos:
            if vid in seen_videos:
                raise AssertionError("[FATAL] Duplicate OOF video across folds.")
            seen_videos.add(vid)

            path = fold_oof_path(fold, vid)
            expected = cert_payload["oof_shard_sha256_ledger"][path.name]
            if sha256(path) != expected:
                raise AssertionError(f"[FATAL] OOF SHA drift fold={fold}, vid={vid}")

            records = torch.load(path, map_location="cpu")
            total_frames += len(records)
            oof_ledger[f"fold{fold}/{path.name}"] = expected

    if seen_videos != set(range(1, NUM_VIDEOS + 1)):
        raise AssertionError("[FATAL] OOF videos do not cover exactly 1..47.")
    if total_frames != EXPECTED_TOTAL_FRAMES:
        raise AssertionError(
            f"[FATAL] Final OOF frame total {total_frames} != {EXPECTED_TOTAL_FRAMES}"
        )

    manifest = {
        "_meta": {
            "schema_version": OOF_SCHEMA,
            "objective_version": OBJECTIVE_VERSION,
            "trainer_sha256": sha256(Path(__file__)),
            "r0_checkpoint_sha256": FROZEN_R0_SHA,
            "canonical_manifest_sha256": FROZEN_CAN_SHA,
            "first_order_temporal_manifest_sha256": FROZEN_FIRST_TEMP_SHA,
            "spatial_manifest_sha256": FROZEN_SPATIAL_SHA,
            "temporal_manifest_sha256": FROZEN_TEMP_SHA,
            "fold_manifest_sha256": FROZEN_FOLD_SHA,
            "screening_config_sha256": FROZEN_SCREENING_CONFIG_SHA,
            "s0_data_source_sha256": sha256(S0_DATA_SOURCE),
            "fixed_match_manifest_sha256": sha256(match_manifest_path()),
            "num_folds": NUM_FOLDS,
            "num_videos": NUM_VIDEOS,
            "total_oof_frames": total_frames,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "l_iou_aux_weight": L_IOU_AUX_WEIGHT,
            "seed_base": SEED,
            "context_mode_train": "auto",
            "context_mode_oof": "auto_eval",
            "temporal_K": K_S0,
            "candidate_profile": "R0 flattened 300x7 sigmoid global Top-100 query membership",
            "expert_order": list(EXPERT_ORDER),
            "oof_dtype": "float32",
            "oof_semantics": (
                "each video predicted only by the fold model whose training set excluded that video"
            ),
        },
        "fold_checkpoint_sha256_ledger": checkpoint_ledger,
        "fold_certificate_sha256_ledger": certificate_ledger,
        "oof_shard_sha256_ledger": oof_ledger,
    }

    path = OUT / "final_expert_oof_manifest.json"
    save_json(path, manifest)

    print("=" * 116)
    print("[PASS] S0-A FINAL EXPERT 5-FOLD CROSSFIT COMPLETE")
    print(f"[+] OOF frames:    {total_frames:,}")
    print(f"[+] OOF videos:    {len(seen_videos)}")
    print(f"[+] Manifest:      {path}")
    print(f"[+] Manifest SHA:  {sha256(path)}")
    print("=" * 116)


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run the exact S0-A objective on one real Train frame and write no S0 outputs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse the fixed-match cache and skip certified completed folds.",
    )
    args = parser.parse_args()

    print("=" * 116)
    print("B-FCEF-F RUN_07 / S0-A FINAL-EXPERT 5-FOLD CROSSFIT")
    print(f"Objective:   {OBJECTIVE_VERSION}")
    print(f"Device:      {DEVICE}")
    print(f"Epochs:      {EPOCHS}")
    print(f"Optimizer:   AdamW lr={LR} wd={WEIGHT_DECAY} grad_clip={GRAD_CLIP}")
    print(f"Temporal K:  {K_S0}")
    print("=" * 116)

    verify_source_and_roots()
    folds = parse_folds()

    (
        BFCEFFScreeningConfig,
        CanonicalObjectState,
        BFCEFFModel,
        frame_inputs_from_records,
        get_authoritative_r0_args,
        build_deformable_detr,
        box_ops,
    ) = prepare_imports()

    config = BFCEFFScreeningConfig.from_json(SCREENING_CONFIG)
    if int(config.temporal_memory_K) != K_S0:
        raise AssertionError(
            f"[FATAL] screening_config_s0 temporal_memory_K={config.temporal_memory_K} != {K_S0}"
        )

    native_cls_template, native_box_template, criterion = (
        build_native_templates_and_criterion(
            get_authoritative_r0_args,
            build_deformable_detr,
        )
    )

    if args.preflight_only:
        run_exact_objective_preflight(
            folds,
            config,
            native_cls_template,
            native_box_template,
            criterion,
            box_ops,
            BFCEFFModel,
            CanonicalObjectState,
            frame_inputs_from_records,
        )
        print("[PASS] RUN_07 S0-A EXACT-OBJECTIVE PREFLIGHT COMPLETE")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    ensure_match_cache(criterion)

    print(f"[+] RUN_07 trainer SHA-256: {sha256(Path(__file__))}")
    print(f"[+] Fixed-match manifest SHA-256: {sha256(match_manifest_path())}")

    for fold in range(NUM_FOLDS):
        oof_videos = sorted(folds[fold])

        if args.resume and certified_fold_complete(fold, oof_videos):
            print(f"[+] Fold {fold}: certified complete -> SKIP")
            continue

        checkpoint = fold_checkpoint_path(fold)

        if args.resume and checkpoint_matches_contract(checkpoint, fold):
            # Training finished previously; do not waste time retraining just
            # because OOF export/certification was interrupted.
            print(
                f"[+] Fold {fold}: valid trained checkpoint exists but fold is not certified; "
                "resuming from OOF export."
            )
            model = construct_model(
                native_cls_template,
                native_box_template,
                config,
                BFCEFFModel,
            )
            load_fold_checkpoint_into_model(checkpoint, model)
            epoch_summaries = torch.load(checkpoint, map_location="cpu")["_meta"]["epoch_summaries"]
        else:
            model, checkpoint, _, oof_videos, epoch_summaries = train_fold(
                fold,
                folds,
                config,
                native_cls_template,
                native_box_template,
                criterion,
                box_ops,
                BFCEFFModel,
                CanonicalObjectState,
                frame_inputs_from_records,
            )

        oof_ledger, oof_frames = export_oof_fold(
            fold,
            model,
            oof_videos,
            CanonicalObjectState,
            frame_inputs_from_records,
        )

        write_fold_certificate(
            fold,
            checkpoint,
            oof_videos,
            oof_ledger,
            oof_frames,
        )

        del model
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    finalize_oof_manifest(folds)


if __name__ == "__main__":
    main()
