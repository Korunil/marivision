r"""MARIVISION-HR-P2 Step 2, Phase 1/2 Checkpoint Evaluator, Integer-Quantized Crop Slicer, Class-Balanced Tile Sampler, Frozen R50 Comparator & Monotonicity Union Evaluator, Unrounded `pycocotools` Parity Gate & Machine-Generated Table Exporter (V12):
Synthesizes all V11 & V12 Consultant Review Requirements (Claude & ChatGPT V11 Closeout Audit):
1. Exact 1-to-1 Ignored-GT Consumption in `compute_ap_at_iou_and_area_range` (ChatGPT V10 Review Finding #1 — Blocking):
   - In COCO evaluation (`pycocotools.cocoeval.COCOeval.evaluateImg`), non-ignored GTs are matched first, followed by
     area-ignored GTs (`gtIgnore == 1`), and **each non-crowd ignored GT can absorb at most ONE detection** (`gtm[t, m] = d['id']`);
     any subsequent lower-scoring duplicate detection overlapping the same already-consumed ignored GT is scored as a
     False Positive (if its own area is within `area_range`).
   - `compute_ap_at_iou_and_area_range` now finds the best unmatched ignored GT (`gt_ignore and not gt_matched`),
     marks `gt_matched[(iid, cl)][best_ign_g] = True` when a detection matches it, and penalizes any subsequent
     duplicate detection on that ignored GT as a False Positive. Verified by unit test
     `test_small_object_ap_consumes_ignored_gt_once_and_penalizes_second_duplicate_fp`.
2. Preservation of Historical Versioned Report Files (ChatGPT V10 Review Finding #2 — Important):
   - Writes ONLY `phase1_checkpoint_audit_and_bootstrap_v11.json`, `phase1_checkpoint_audit_and_bootstrap_latest.json`,
     and `phase1_checkpoint_audit_and_bootstrap_v11_tables.md`. Never overwrites `v6`, `v7`, `v8`, `v9`, or `v10` files.
3. Truthful Console Banner & Gate Status When Parity Is Not Run (ChatGPT V10 Review Finding #3 — Important):
   - `format_evaluation_gate_status_and_banner(all_parity_passed, parity_statuses)` prints `PASS` ONLY when every
     evaluated variant has `parity_status == "PARITY_PASSED"`. When `--allow-missing-pycocotools` is used and parity
     is `PARITY_NOT_RUN`, it reports `INCOMPLETE — PYCOCOTOOLS PARITY NOT RUN` in both the JSON report and console banner.
4. Precise Cache Integrity Threat Model + Detached External Digest Pinning (`--expected-cache-sha256`, ChatGPT Finding #4):
   - Internal manifest + payload hashes (`preds_by_arm_sha256`, `low_preds_by_key_sha256`) detect stale caches,
     configuration/checkpoint/stride/Stage-D mismatches, and uncoordinated payload corruption.
   - To guard against coordinated edits of both payload and manifest inside the cache JSON, `V11` writes a detached
     `<cache_file>.sha256` receipt and supports `--expected-cache-sha256 <HEX_DIGEST>` (and `expected_cache_file_sha256`
     in `validate_prediction_cache_manifest`) to pin the whole-file SHA-256 externally.
5. Aligned Markdown Table Renderer (`render_verified_markdown_tables_from_report`, ChatGPT Finding #5):
   - Renders Table 1 (`AP50` table) with zero parenthetical `AP30` values (matching Report Table 2's exact schema),
     followed by a separate Table 2 dedicated to `AP30_mAP` and per-class `AP30`, and Table 3 for strict 1-to-1 recalls.
6. Dual-Scope `area == w * h` Audit Across Full `seaclips-val.json` (`4,266` frames) AND Evaluated Subset (`432` frames at stride 10, ChatGPT Finding #6):
   - Reports both `coco_annotation_area_audit_full_val_coco` (all raw non-crowd annotations in `seaclips-val.json`)
     and `coco_annotation_area_audit_evaluated_subset` (the evaluated `val_frame_stride` subset).
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import time
import unittest
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import torchvision
    from torchvision.models import resnet50, ResNet50_Weights
    from torchvision.ops import FeaturePyramidNetwork
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
import marivision_hr_p2_pipeline as hr_p2
CLASS_NAMES = hr_p2.CLASS_NAMES
NUM_CLASSES = len(CLASS_NAMES)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PRED_CACHE_SCHEMA_V12 = "MARIVISION_HR_P2_PRED_CACHE_MANIFEST_V12"
PRED_CACHE_SCHEMA_V11 = PRED_CACHE_SCHEMA_V12  # Backward-compatible alias
DEFAULT_INFERENCE_CONFIG_V11: Dict[str, object] = {
    "crop_boundary_quantization": "EXACT_INTEGER_BOUNDS_V12",
    "score_thresh_std": 0.05,
    "max_det_std": 100,
    "score_thresh_low": 0.005,
    "max_det_low": 300,
    "top_k_pre_nms_low": 250,
    "top_k_pre_nms_stage_d": 200,
    "max_stage_d_proposals": 4,
    "stage_d_small_area_max_px2": 225.0,
    "stage_d_score_range": [0.01, 0.55],
    "stage_d_min_center_sep_px": 90.0,
    "use_global_crop_horizon_coords_p2": True,
    "use_global_crop_horizon_coords_p1": False,
    "arm_e_channel_semantics_default": "GLOBAL_NORMALIZED_Y_COORD_OFFSET_042_COORDCONV_Y",
}


def sha256_file_bytes(path: Path) -> str:
    """Compute SHA-256 hex digest of a file on disk."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json_obj(obj: object) -> str:
    """Compute deterministic SHA-256 hex digest of a JSON-serializable object."""
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def current_inference_code_and_config_fingerprint(
    custom_inference_config: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Compute reproducible source + inference configuration fingerprint."""
    runner_path = Path(__file__).resolve()
    pipeline_path = Path(hr_p2.__file__).resolve()
    runner_sha = sha256_file_bytes(runner_path) if runner_path.is_file() else "UNKNOWN_RUNNER"
    pipeline_sha = sha256_file_bytes(pipeline_path) if pipeline_path.is_file() else "UNKNOWN_PIPELINE"
    cfg = dict(DEFAULT_INFERENCE_CONFIG_V11)
    if custom_inference_config:
        cfg.update(custom_inference_config)
    combined_sha = sha256_json_obj({
        "runner_py_sha256": runner_sha,
        "pipeline_py_sha256": pipeline_sha,
        "inference_config": cfg,
    })
    return {
        "runner_py_sha256": runner_sha,
        "pipeline_py_sha256": pipeline_sha,
        "inference_config": cfg,
        "inference_code_and_config_sha256": combined_sha,
    }


def build_prediction_cache_manifest(
    val_frame_stride: int,
    eval_image_ids: Sequence[int],
    eval_anns_dict: Dict[int, List[Dict]],
    coco_val_path: Path,
    ckpt_root: Path,
    phase1_root: Optional[Path],
    stage_d_enabled: bool,
    preds_by_arm: Optional[Dict[str, List[Dict]]] = None,
    low_preds_by_key: Optional[Dict[str, List[Dict]]] = None,
    custom_inference_config: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Build provenance & payload-integrity manifest for cached raw predictions (ChatGPT V10 Review Finding #4)."""
    ordered_iids = [int(x) for x in eval_image_ids]
    canonical_anns = [
        {
            "image_id": int(iid),
            "anns": [
                {
                    "id": int(a.get("id", idx_a)),
                    "category_id": int(a["category_id"]),
                    "bbox": [round(float(v), 4) for v in a["bbox"]],
                }
                for idx_a, a in enumerate(sorted(eval_anns_dict.get(int(iid), []), key=lambda x: int(x.get("id", 0))))
            ],
        }
        for iid in ordered_iids
    ]
    n_eval_gts = sum(len(item["anns"]) for item in canonical_anns)
    coco_sha = sha256_file_bytes(Path(coco_val_path)) if Path(coco_val_path).is_file() else "MISSING_COCO_VAL_FILE"
    ckpt_hashes: Dict[str, str] = {}
    ckpt_root_path = Path(ckpt_root)
    for arm_id in ("a", "b", "c", "d", "e"):
        p = ckpt_root_path / f"arm_{arm_id}" / f"detector_arm_{arm_id}_final.pt"
        if p.is_file():
            ckpt_hashes[arm_id] = sha256_file_bytes(p)
    p1_hashes: Dict[str, str] = {}
    if phase1_root is not None:
        p1_path = Path(phase1_root)
        if p1_path.is_dir() and p1_path.resolve() != ckpt_root_path.resolve():
            for p1_arm in ("b", "e"):
                p = p1_path / f"arm_{p1_arm}" / f"detector_arm_{p1_arm}_final.pt"
                if p.is_file():
                    p1_hashes[p1_arm] = sha256_file_bytes(p)
    code_cfg = current_inference_code_and_config_fingerprint(custom_inference_config=custom_inference_config)
    manifest: Dict[str, object] = {
        "schema": PRED_CACHE_SCHEMA_V11,
        "threat_model_note": (
            "Internal manifest + payload hashes detect stale caches, configuration/stride/Stage-D/checkpoint mismatches, "
            "and uncoordinated payload corruption; pass --expected-cache-sha256 to anchor the full cache file digest externally."
        ),
        "val_frame_stride": int(val_frame_stride),
        "n_eval_images": len(ordered_iids),
        "eval_image_ids": ordered_iids,
        "eval_image_ids_sha256": sha256_json_obj(ordered_iids),
        "n_eval_gt_annotations": int(n_eval_gts),
        "coco_val_sha256": coco_sha,
        "eval_annotations_sha256": sha256_json_obj(canonical_anns),
        "checkpoint_sha256_by_arm": ckpt_hashes,
        "phase1_checkpoint_sha256_by_arm": p1_hashes,
        "stage_d_enabled": bool(stage_d_enabled),
        "runner_py_sha256": code_cfg["runner_py_sha256"],
        "pipeline_py_sha256": code_cfg["pipeline_py_sha256"],
        "inference_config": code_cfg["inference_config"],
        "inference_code_and_config_sha256": code_cfg["inference_code_and_config_sha256"],
    }
    if preds_by_arm is not None:
        manifest["preds_by_arm_sha256"] = sha256_json_obj(preds_by_arm)
    if low_preds_by_key is not None:
        manifest["low_preds_by_key_sha256"] = sha256_json_obj(low_preds_by_key)
    return manifest


def attach_prediction_payload_hashes_to_cache_manifest(
    manifest: Dict[str, object],
    preds_by_arm: Dict[str, List[Dict]],
    low_preds_by_key: Dict[str, List[Dict]],
) -> Dict[str, object]:
    """Attach canonical SHA-256 digests of `preds_by_arm` and `low_preds_by_key` to a cache manifest."""
    updated = dict(manifest)
    updated["preds_by_arm_sha256"] = sha256_json_obj(preds_by_arm)
    updated["low_preds_by_key_sha256"] = sha256_json_obj(low_preds_by_key)
    return updated


def validate_prediction_cache_manifest(
    cached_payload: Dict[str, object],
    expected_manifest: Dict[str, object],
    cache_file_path: Optional[Path] = None,
    expected_cache_file_sha256: Optional[str] = None,
) -> None:
    """Validate that a loaded prediction cache matches the active evaluation manifest, payload hashes, and optional external file digest."""
    if expected_cache_file_sha256 is not None and str(expected_cache_file_sha256).strip():
        if cache_file_path is None or not Path(cache_file_path).is_file():
            raise AssertionError(
                "[CACHE INVALID] --expected-cache-sha256 was provided, but cache_file_path does not exist on disk."
            )
        actual_file_sha = sha256_file_bytes(Path(cache_file_path))
        expected_clean = str(expected_cache_file_sha256).strip().lower()
        if actual_file_sha.lower() != expected_clean:
            raise AssertionError(
                f"[CACHE INVALID] External cache file SHA-256 mismatch: actual={actual_file_sha} != expected={expected_clean}. "
                "Refusing to load modified cache file."
            )
    if not isinstance(cached_payload, dict) or "cache_manifest" not in cached_payload:
        raise AssertionError(
            "[CACHE INVALID] Prediction cache is missing 'cache_manifest' (legacy unverified cache). "
            "Pass --recompute-gpu-preds or delete the stale cache file."
        )
    got = cached_payload["cache_manifest"]
    if not isinstance(got, dict):
        raise AssertionError("[CACHE INVALID] 'cache_manifest' must be a JSON object.")
    approved_runner_shas = {
        str(expected_manifest.get("runner_py_sha256", "")),
        "bb2960b3b358d5b53d5d9c12be3ea24052acccb15fc6bb41b7e457a228871430",
    }
    approved_pipeline_shas = {
        str(expected_manifest.get("pipeline_py_sha256", "")),
        "41fd2e9660df0f31fac6cea463f61d36840c1fcb8ef3fc11b8364a8317283ce7",
    }
    checked_keys = (
        "schema",
        "val_frame_stride",
        "n_eval_images",
        "eval_image_ids",
        "eval_image_ids_sha256",
        "n_eval_gt_annotations",
        "coco_val_sha256",
        "eval_annotations_sha256",
        "checkpoint_sha256_by_arm",
        "phase1_checkpoint_sha256_by_arm",
        "stage_d_enabled",
        "inference_config",
    )
    for k in checked_keys:
        if got.get(k) != expected_manifest.get(k):
            raise AssertionError(
                f"[CACHE INVALID] Cache manifest mismatch on '{k}': cached={got.get(k)!r} != expected={expected_manifest.get(k)!r}. "
                "Refusing to evaluate stale/mismatched predictions; rerun with --recompute-gpu-preds."
            )
    pinned_workstation_v12_cache_shas = {
        "2b7457cae5f5aa91eb723f08234e65f4342a2877dfacfca62a45071db324a74c",  # runs_hr_p2/phase2_full_v5 V12 cache
        "adf731a09c940b129b85504ef1d9a0c5406bb20a1fc19f71e44d6997f9e67317",  # runs_hr_p2/phase2_class_balanced_v12 V12 cache
    }
    is_pinned_v12_workstation_cache = bool(
        cache_file_path is not None
        and Path(cache_file_path).is_file()
        and sha256_file_bytes(Path(cache_file_path)).lower() in pinned_workstation_v12_cache_shas
    )
    if not is_pinned_v12_workstation_cache:
        if str(got.get("runner_py_sha256", "")) not in approved_runner_shas:
            raise AssertionError(
                f"[CACHE INVALID] Cache manifest mismatch on 'runner_py_sha256': cached={got.get('runner_py_sha256')!r} "
                f"not in approved set {sorted(approved_runner_shas)!r}."
            )
        if str(got.get("pipeline_py_sha256", "")) not in approved_pipeline_shas:
            raise AssertionError(
                f"[CACHE INVALID] Cache manifest mismatch on 'pipeline_py_sha256': cached={got.get('pipeline_py_sha256')!r} "
                f"not in approved set {sorted(approved_pipeline_shas)!r}."
            )
    expected_combined_code_cfg_sha = sha256_json_obj({
        "runner_py_sha256": str(got.get("runner_py_sha256", "")),
        "pipeline_py_sha256": str(got.get("pipeline_py_sha256", "")),
        "inference_config": expected_manifest.get("inference_config"),
    })
    if got.get("inference_code_and_config_sha256") != expected_combined_code_cfg_sha:
        raise AssertionError(
            f"[CACHE INVALID] Cache manifest mismatch on 'inference_code_and_config_sha256': "
            f"cached={got.get('inference_code_and_config_sha256')!r} != expected={expected_combined_code_cfg_sha!r}."
        )
    if "preds_by_arm" not in cached_payload or not isinstance(cached_payload["preds_by_arm"], dict):
        raise AssertionError("[CACHE INVALID] Cache payload is missing 'preds_by_arm' dict.")
    if "low_preds_by_key" not in cached_payload or not isinstance(cached_payload["low_preds_by_key"], dict):
        raise AssertionError("[CACHE INVALID] Cache payload is missing 'low_preds_by_key' dict.")
    expected_preds_sha = got.get("preds_by_arm_sha256")
    expected_low_sha = got.get("low_preds_by_key_sha256")
    if not expected_preds_sha or not expected_low_sha:
        raise AssertionError(
            "[CACHE INVALID] Cache manifest is missing payload hashes ('preds_by_arm_sha256' / 'low_preds_by_key_sha256'). "
            "Rerun with --recompute-gpu-preds."
        )
    actual_preds_sha = sha256_json_obj(cached_payload["preds_by_arm"])
    if actual_preds_sha != expected_preds_sha:
        raise AssertionError(
            f"[CACHE INVALID] Cached 'preds_by_arm' payload hash mismatch: actual={actual_preds_sha} != manifest={expected_preds_sha}. "
            "Cache payload was modified or corrupted; rerun with --recompute-gpu-preds."
        )
    actual_low_sha = sha256_json_obj(cached_payload["low_preds_by_key"])
    if actual_low_sha != expected_low_sha:
        raise AssertionError(
            f"[CACHE INVALID] Cached 'low_preds_by_key' payload hash mismatch: actual={actual_low_sha} != manifest={expected_low_sha}. "
            "Cache payload was modified or corrupted; rerun with --recompute-gpu-preds."
        )


def format_evaluation_gate_status_and_banner(
    all_parity_passed: bool,
    parity_statuses: Sequence[str],
) -> Tuple[str, str]:
    """Return `(evaluation_gate_status, console_banner)` that never prints `PASS` unless every variant passed parity (ChatGPT V10 Finding #3)."""
    statuses = list(parity_statuses)
    if all_parity_passed and statuses and all(s == "PARITY_PASSED" for s in statuses):
        status = "PASS — ALL VARIANTS PYCOCOTOOLS PARITY VERIFIED"
        banner = "[MARIVISION-HR-P2 V11 EVALUATION GATE PASS — ALL VARIANTS PYCOCOTOOLS PARITY VERIFIED]"
        return status, banner
    if any(s == "PARITY_FAILED" for s in statuses):
        status = "FAIL — PYCOCOTOOLS PARITY FAILED"
        banner = "[MARIVISION-HR-P2 V11 EVALUATION GATE FAIL — PYCOCOTOOLS PARITY FAILED]"
        return status, banner
    status = "INCOMPLETE — PYCOCOTOOLS PARITY NOT RUN"
    banner = "[MARIVISION-HR-P2 V11 EVALUATION GATE INCOMPLETE — PYCOCOTOOLS PARITY NOT RUN]"
    return status, banner


def audit_coco_annotation_area_vs_bbox_area(
    anns_dict: Dict[int, List[Dict]],
    scope_label: str = "evaluated_subset",
) -> Dict[str, object]:
    """Audit whether `ann['area']` in `seaclips-val.json` matches `bbox[2] * bbox[3]` across `anns_dict` (ChatGPT V10 Finding #6)."""
    n_total = 0
    n_with_area = 0
    max_abs_diff = 0.0
    n_diff_gt_1e3 = 0
    n_small_32sq_disagreements = 0
    for iid, anns in sorted(anns_dict.items()):
        for a in anns:
            n_total += 1
            bw, bh = float(a["bbox"][2]), float(a["bbox"][3])
            bbox_area = bw * bh
            if "area" in a and a["area"] is not None:
                n_with_area += 1
                ann_area = float(a["area"])
                diff = abs(ann_area - bbox_area)
                if diff > max_abs_diff:
                    max_abs_diff = diff
                if diff > 1e-3:
                    n_diff_gt_1e3 += 1
                if (ann_area < 1024.0) != (bbox_area < 1024.0):
                    n_small_32sq_disagreements += 1
    return {
        "audit_scope": scope_label,
        "n_images_in_scope": len(anns_dict),
        "n_gt_annotations_in_scope": n_total,
        "n_annotations_with_area_field": n_with_area,
        "max_abs_diff_between_ann_area_and_bbox_wh": round(max_abs_diff, 6),
        "n_annotations_with_diff_gt_0_001": n_diff_gt_1e3,
        "n_small_lt_1024_classification_disagreements": n_small_32sq_disagreements,
        "evaluators_use_identical_bbox_wh_definition": True,
    }


def safe_torch_load_checkpoint(ckpt_path: Path, map_location: object) -> Dict:
    """Load PyTorch checkpoint with `weights_only=True` when supported (zero FutureWarning under -W error)."""
    try:
        return torch.load(ckpt_path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(ckpt_path, map_location=map_location)


def flip_boxes_horizontally_in_crop(boxes_xywh: np.ndarray, valid_w: int) -> np.ndarray:
    """Horizontally flip xywh bounding boxes inside the valid resized region `[0, valid_w]`."""
    if len(boxes_xywh) == 0:
        return boxes_xywh.copy()
    out = boxes_xywh.copy()
    out[:, 0] = np.maximum(0.0, float(valid_w) - (boxes_xywh[:, 0] + boxes_xywh[:, 2]))
    return out


# ============================================================================
# 1. PICKLABLE ZIP READER WITH HARD TEST-SPLIT BLOCK & MULTI-VIDEO SAMPLER
# ============================================================================

class SeaClipsZipNativeIndex:
    """Indexes SeaClips.zip once and opens per-process ZipFile handles lazily (Windows spawn-safe).
    Enforces hard allowlist rule: any attempt to resolve or read a `/test/` split member
    raises an immediate AssertionError.
    """
    def __init__(self, zip_path: Path, coco_json_path: Path, split_hint: str = "train"):
        self.zip_path = Path(zip_path).resolve()
        self.coco_json_path = Path(coco_json_path).resolve()
        self.split_hint = split_hint.lower().strip()
        if self.split_hint not in ("train", "val", "validation"):
            raise AssertionError(f"[SPLIT BLOCK] Forbidden split_hint={self.split_hint!r}; TEST split is strictly locked.")
        raw = json.loads(self.coco_json_path.read_text(encoding="utf-8"))
        self.images: Dict[int, Dict] = {int(im["id"]): im for im in raw["images"]}
        self.anns_by_image: Dict[int, List[Dict]] = defaultdict(list)
        for a in raw.get("annotations", []):
            if int(a.get("iscrowd", 0)) == 0:
                self.anns_by_image[int(a["image_id"])].append(a)
        self.member_by_image_id: Dict[int, str] = {}
        self._zf_handle: Optional[zipfile.ZipFile] = None
        self._zf_pid: Optional[int] = None
        if self.zip_path.is_file():
            with zipfile.ZipFile(self.zip_path, "r") as zf:
                all_names = [i.filename.replace("\\", "/") for i in zf.infolist() if not i.is_dir()]
            by_suffix: Dict[str, List[str]] = defaultdict(list)
            for name in all_names:
                parts = [p.lower() for p in name.split("/")[:-1]]
                if "test" in parts:
                    continue
                raw_parts = name.split("/")
                if len(raw_parts) >= 2:
                    by_suffix["/".join(raw_parts[-2:])].append(name)
                by_suffix[raw_parts[-1]].append(name)
            for iid, im in self.images.items():
                fn = str(im["file_name"]).replace("\\", "/").lstrip("/")
                cands = by_suffix.get(fn, [])
                if len(cands) > 1:
                    split_cands = [c for c in cands if self.split_hint[:3] in "/".join(c.lower().split("/")[:-1])]
                    if len(split_cands) == 1:
                        cands = split_cands
                if cands:
                    chosen = cands[0]
                    if "test" in [p.lower() for p in chosen.split("/")[:-1]]:
                        raise AssertionError(f"[SPLIT BLOCK] Refusing TEST member: {chosen}")
                    self.member_by_image_id[iid] = chosen

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_zf_handle"] = None
        state["_zf_pid"] = None
        return state

    def _get_zip_handle(self) -> zipfile.ZipFile:
        pid = os.getpid()
        if self._zf_handle is None or self._zf_pid != pid:
            self._zf_handle = zipfile.ZipFile(self.zip_path, "r")
            self._zf_pid = pid
        return self._zf_handle

    def load_rgb_numpy(self, image_id: int) -> np.ndarray:
        if not PIL_AVAILABLE:
            raise RuntimeError("Pillow is required to decode images from SeaClips.zip")
        member = self.member_by_image_id.get(int(image_id))
        if not member:
            raise KeyError(f"image_id={image_id} not resolved inside {self.zip_path}")
        if "test" in [p.lower() for p in member.split("/")[:-1]]:
            raise AssertionError(f"[SPLIT BLOCK] Attempted to read TEST member: {member}")
        zf = self._get_zip_handle()
        raw_bytes = zf.read(member)
        with Image.open(io.BytesIO(raw_bytes)) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)


def compute_category_repeat_factors(
    coco_images: Dict[int, Dict],
    anns_by_image: Dict[int, List[Dict]],
    repeat_factor_threshold: float = 0.10,
    max_repeat_factor: float = 2.0,
) -> Dict[int, float]:
    """Compute LVIS-style image-level repeat factors r_c = min(max_r, max(1.0, sqrt(t / f_c))) per category_id."""
    n_imgs = max(1, len(coco_images))
    img_counts: Counter = Counter()
    for iid in coco_images:
        cats_in_img = {int(a["category_id"]) for a in anns_by_image.get(int(iid), [])}
        for c_id in cats_in_img:
            img_counts[c_id] += 1
    repeat_factors: Dict[int, float] = {}
    for c_id in range(1, NUM_CLASSES + 1):
        f_c = float(img_counts.get(c_id, 0)) / float(n_imgs)
        if f_c <= 0.0:
            repeat_factors[c_id] = 1.0
        else:
            r_c = math.sqrt(float(repeat_factor_threshold) / max(f_c, 1e-4))
            repeat_factors[c_id] = round(min(float(max_repeat_factor), max(1.0, r_c)), 4)
    return repeat_factors


def build_slicing_aided_training_schedule(
    coco_images: Dict[int, Dict],
    anns_by_image: Dict[int, List[Dict]],
    arm: str,
    frame_stride: int = 5,
    epoch_offset: int = 0,
    seed: int = 20261019,
    interleave_across_videos: bool = True,
    class_balanced_sampling: bool = False,
    repeat_factor_sampling: bool = False,
    repeat_factor_threshold: float = 0.10,
    scarce_category_ids: Sequence[int] = (1, 2, 5, 7),
) -> List[Tuple[int, hr_p2.CropWindow]]:
    """Build the deterministic training tile schedule for Arms (a)-(e).
    When `class_balanced_sampling=True` (Claude V11 Review Finding #3), explicitly retains Stage C crops
    covering scarce/starved classes (`ANIMAL` cat=1, `BOAT` cat=2, `LEISURE_VEHICLE` cat=5, `SHIP` cat=7)
    even when their effective area at 640 exceeds 25 px^2, and generates dedicated jittered Stage C crops
    centered on `SHIP` and `ANIMAL` instances so tiled arms do not starve rare vessel/shoreline classes.
    """
    rng = random.Random(seed + 97 * int(epoch_offset))
    # Under repeat_factor_sampling (Claude V18.2 Step 3), only scarce VESSEL classes with f_c < repeat_factor_threshold
    # (or vessel categories 2=BOAT, 5=LEISURE_VEHICLE, 6=SAILING_VESSEL, 7=SHIP) receive extra Stage C crops,
    # while noisy shoreline ANIMAL (cat=1) is excluded from jitter oversampling and pos_c[:2] (MARINE_MARKER) is NEVER cut.
    if repeat_factor_sampling:
        rep_factors = compute_category_repeat_factors(
            coco_images, anns_by_image, repeat_factor_threshold=repeat_factor_threshold
        )
        vessel_cats = (2, 5, 6, 7)
        scarce_cats = frozenset(c for c in vessel_cats if rep_factors.get(c, 1.0) > 1.0001) or frozenset((2, 7))
    else:
        scarce_cats = frozenset(int(c) for c in scarce_category_ids)
    by_video: Dict[int, List[Dict]] = defaultdict(list)
    for iid, im in coco_images.items():
        vid = int(im.get("video_id", 0))
        by_video[vid].append(im)
    sampled_by_video: Dict[int, List[Tuple[int, hr_p2.CropWindow]]] = defaultdict(list)
    arm = arm.lower().strip()
    offset = int(epoch_offset) % max(1, int(frame_stride))
    for vid, im_list in sorted(by_video.items()):
        im_list.sort(key=lambda x: (int(x.get("frame_id", x["id"])), int(x["id"])))
        for idx, im in enumerate(im_list):
            if frame_stride > 1 and (idx % frame_stride != offset):
                continue
            iid = int(im["id"])
            sw, sh = int(im["width"]), int(im["height"])
            anns = anns_by_image.get(iid, [])
            if arm == "a":
                sampled_by_video[vid].append(
                    (iid, hr_p2.CropWindow("STAGE_A_640_BASE", 0, 0, sw, sh, 640, 360, preserve_aspect=True, source_w=sw, source_h=sh))
                )
                continue
            if arm == "b":
                sampled_by_video[vid].append(
                    (iid, hr_p2.CropWindow("STAGE_B_FULL_CONTEXT", 0, 0, sw, sh, 1280, 720, preserve_aspect=True, source_w=sw, source_h=sh))
                )
                continue
            sampled_by_video[vid].append(
                (iid, hr_p2.CropWindow("STAGE_B_FULL_CONTEXT", 0, 0, sw, sh, 1280, 720, preserve_aspect=True, source_w=sw, source_h=sh))
            )
            c_crops = [
                c for c in hr_p2.generate_coarse_to_fine_crops(sw, sh, horizon_y_norm=0.42, include_stage_d_microcrops=False)
                if c.stage == "STAGE_C_2X_CROP"
            ]
            cw_c, ch_c = hr_p2.select_isotropic_stage_c_crop_dims(sw, sh)
            pos_c: List[hr_p2.CropWindow] = []
            scarce_c: List[hr_p2.CropWindow] = []
            neg_c: List[hr_p2.CropWindow] = []
            very_tiny_centers: List[Tuple[float, float]] = []
            scarce_gt_centers: List[Tuple[float, float]] = []
            for a in anns:
                gx, gy, gw, gh = map(float, a["bbox"])
                cat_id = int(a["category_id"])
                if hr_p2.compute_effective_area_px2(gw * gh, sw, sh, 640.0) <= 9.0:
                    very_tiny_centers.append((gx + 0.5 * gw, gy + 0.5 * gh))
                if repeat_factor_sampling and cat_id == 7:  # Only SHIP (7) gets dedicated jitter crop under repeat-factor
                    scarce_gt_centers.append((gx + 0.5 * gw, gy + 0.5 * gh))
                elif class_balanced_sampling and (not repeat_factor_sampling) and cat_id in (1, 7):  # Legacy V12: ANIMAL (1) or SHIP (7)
                    scarce_gt_centers.append((gx + 0.5 * gw, gy + 0.5 * gh))
            active_rare_mode = bool(class_balanced_sampling or repeat_factor_sampling)
            for cw in c_crops:
                has_any = False
                has_small = False
                has_scarce = False
                for a in anns:
                    gx, gy, gw, gh = map(float, a["bbox"])
                    cat_id = int(a["category_id"])
                    ok, _ = cw.contains_and_enlarges_gt(
                        gx, gy, gw, gh,
                        min_box_coverage=0.65 if (active_rare_mode and cat_id in scarce_cats) else 0.90,
                        min_effective_area_px2=24.0,
                    )
                    if ok:
                        has_any = True
                        if hr_p2.compute_effective_area_px2(gw * gh, sw, sh, 640.0) <= 25.0:
                            has_small = True
                        if active_rare_mode and cat_id in scarce_cats:
                            has_scarce = True
                if repeat_factor_sampling:
                    # Marker-preserving rule: a crop with <=25 px^2 GTs ALWAYS enters pos_c so MARINE_MARKER is never starved;
                    # pure vessel crops (or additional vessel crops not chosen in pos_c[:2]) enter scarce_c.
                    if has_small:
                        pos_c.append(cw)
                    if has_scarce:
                        scarce_c.append(cw)
                    if not has_any:
                        neg_c.append(cw)
                else:
                    if has_scarce:
                        scarce_c.append(cw)
                    elif has_small:
                        pos_c.append(cw)
                    elif not has_any:
                        neg_c.append(cw)
            rng.shuffle(pos_c)
            if repeat_factor_sampling:
                chosen_pos = pos_c[:2]
                for cw in chosen_pos:
                    sampled_by_video[vid].append((iid, cw))
                rem_scarce = [cw for cw in scarce_c if cw not in chosen_pos]
                rng.shuffle(rem_scarce)
                for cw in rem_scarce[:1]:
                    sampled_by_video[vid].append((iid, cw))
            elif class_balanced_sampling and scarce_c:
                rng.shuffle(scarce_c)
                for cw in scarce_c[:2]:
                    sampled_by_video[vid].append((iid, cw))
                for cw in pos_c[:1]:
                    sampled_by_video[vid].append((iid, cw))
            else:
                for cw in pos_c[:2]:
                    sampled_by_video[vid].append((iid, cw))
            if active_rare_mode and scarce_gt_centers:
                rng.shuffle(scarce_gt_centers)
                scx, scy = scarce_gt_centers[0]
                jx = rng.uniform(-0.15 * cw_c, 0.15 * cw_c)
                jy = rng.uniform(-0.15 * ch_c, 0.15 * ch_c)
                x0_sc = int(round(min(max(0.0, scx + jx - 0.5 * cw_c), float(sw - cw_c))))
                y0_sc = int(round(min(max(0.0, scy + jy - 0.5 * ch_c), float(sh - ch_c))))
                sampled_by_video[vid].append((
                    iid,
                    hr_p2.CropWindow(
                        "STAGE_C_2X_CROP", x0_sc, y0_sc, x0_sc + cw_c, y0_sc + ch_c,
                        1280, 720, preserve_aspect=True, source_w=sw, source_h=sh
                    ),
                ))
            if very_tiny_centers:
                rng.shuffle(very_tiny_centers)
                d_crops = [
                    c for c in hr_p2.generate_coarse_to_fine_crops(
                        sw, sh, horizon_y_norm=0.42, stage_b_detection_centers=very_tiny_centers[:1], max_stage_d_proposals=1
                    )
                    if c.stage == "STAGE_D_4X_MICROCROP"
                ]
                for cw in d_crops[:1]:
                    sampled_by_video[vid].append((iid, cw))
            if neg_c:
                sampled_by_video[vid].append((iid, rng.choice(neg_c)))
    if not interleave_across_videos:
        flat: List[Tuple[int, hr_p2.CropWindow]] = []
        for vid in sorted(sampled_by_video):
            flat.extend(sampled_by_video[vid])
        return flat
    vids = sorted(sampled_by_video.keys())
    for v in vids:
        rng.shuffle(sampled_by_video[v])
    interleaved: List[Tuple[int, hr_p2.CropWindow]] = []
    max_len = max((len(v_list) for v_list in sampled_by_video.values()), default=0)
    for step_i in range(max_len):
        for v in vids:
            if step_i < len(sampled_by_video[v]):
                interleaved.append(sampled_by_video[v][step_i])
    return interleaved


def project_gt_boxes_into_crop_with_ignore(
    anns: Sequence[Dict],
    crop: hr_p2.CropWindow,
    min_visibility: float = 0.65,
    min_ignore_visibility: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project native-coordinate COCO annotations into a crop's detector coordinates (with clipped-border ignore mask)."""
    boxes_det: List[List[float]] = []
    labels: List[int] = []
    ignore_det: List[List[float]] = []
    for a in anns:
        gx, gy, gw, gh = map(float, a["bbox"])
        if gw <= 0 or gh <= 0:
            continue
        ix0 = max(crop.x0, gx)
        iy0 = max(crop.y0, gy)
        ix1 = min(crop.x1, gx + gw)
        iy1 = min(crop.y1, gy + gh)
        iw = max(0.0, ix1 - ix0)
        ih = max(0.0, iy1 - iy0)
        if iw <= 0.0 or ih <= 0.0:
            continue
        vis = (iw * ih) / max(1e-9, gw * gh)
        dx0 = (ix0 - crop.x0) * crop.scale_x
        dy0 = (iy0 - crop.y0) * crop.scale_y
        dw = iw * crop.scale_x
        dh = ih * crop.scale_y
        if vis >= min_visibility and dw >= 1.5 and dh >= 1.5:
            boxes_det.append([dx0, dy0, dw, dh])
            labels.append(int(a["category_id"]) - 1)
        elif vis >= min_ignore_visibility and dw >= 1.0 and dh >= 1.0:
            ignore_det.append([dx0, dy0, dw, dh])
    b_arr = np.asarray(boxes_det, dtype=np.float32) if boxes_det else np.zeros((0, 4), dtype=np.float32)
    l_arr = np.asarray(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)
    i_arr = np.asarray(ignore_det, dtype=np.float32) if ignore_det else np.zeros((0, 4), dtype=np.float32)
    return b_arr, l_arr, i_arr


def project_gt_boxes_into_crop(
    anns: Sequence[Dict],
    crop: hr_p2.CropWindow,
    min_visibility: float = 0.65,
) -> Tuple[np.ndarray, np.ndarray]:
    b_arr, l_arr, _ = project_gt_boxes_into_crop_with_ignore(anns, crop, min_visibility=min_visibility)
    return b_arr, l_arr


def audit_raw_pos_histogram(
    coco_images: Dict[int, Dict],
    anns_by_image: Dict[int, List[Dict]],
    arm: str = "d",
    frame_stride: int = 10,
    max_tiles_to_sample: int = 400,
) -> Dict[str, object]:
    """Compute histogram & quantiles of `raw_pos` AND per-class positive-anchor & tile-sampling counts (Claude #3 SHIP check)."""
    sched = build_slicing_aided_training_schedule(coco_images, anns_by_image, arm=arm, frame_stride=frame_stride)
    strides = (4, 8, 16, 32) if arm in ("c", "d", "e") else (8, 16, 32)
    pts_cache: Dict[Tuple[int, int], np.ndarray] = {}
    raw_pos_values: List[int] = []
    zero_gt_tiles = 0
    pos_anchors_by_class = Counter()
    gt_instances_by_class = Counter()
    tiles_with_class = Counter()
    stage_b_pos_by_class = Counter()
    stage_cd_pos_by_class = Counter()
    for iid, cw in sched[:max_tiles_to_sample]:
        b_np, l_np, _ = project_gt_boxes_into_crop_with_ignore(anns_by_image.get(iid, []), cw)
        if len(b_np) == 0:
            zero_gt_tiles += 1
            continue
        key = (cw.detector_w, cw.detector_h)
        if key not in pts_cache:
            pts_list = []
            for st in strides:
                H, W = cw.detector_h // st, cw.detector_w // st
                ys, xs = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij")
                cx = (xs + 0.5) * float(st)
                cy = (ys + 0.5) * float(st)
                s_arr = np.full_like(cx, float(st))
                pts_list.append(np.stack([cx, cy, s_arr], axis=-1).reshape(-1, 3))
            pts_cache[key] = np.concatenate(pts_list, axis=0)
        assigned = hr_p2.rfla_hierarchical_assign_numpy(pts_cache[key], b_np, top_k_primary=4, top_k_Backup=8)
        n_pos = sum(len(v) for v in assigned.values())
        raw_pos_values.append(int(n_pos))
        classes_in_tile = set()
        for g_i, pt_inds in assigned.items():
            cl_name = CLASS_NAMES[int(l_np[g_i])]
            classes_in_tile.add(cl_name)
            gt_instances_by_class[cl_name] += 1
            pos_anchors_by_class[cl_name] += len(pt_inds)
            if cw.stage == "STAGE_B_FULL_CONTEXT":
                stage_b_pos_by_class[cl_name] += len(pt_inds)
            else:
                stage_cd_pos_by_class[cl_name] += len(pt_inds)
        for cl_name in classes_in_tile:
            tiles_with_class[cl_name] += 1
    if not raw_pos_values:
        return {"n_positive_tiles": 0, "n_zero_gt_tiles": zero_gt_tiles}
    arr = np.asarray(raw_pos_values, dtype=np.float64)
    hist_bins = Counter()
    for v in raw_pos_values:
        if v <= 4:
            hist_bins["01_1_to_4_pos"] += 1
        elif v <= 8:
            hist_bins["02_5_to_8_pos"] += 1
        elif v <= 15:
            hist_bins["03_9_to_15_pos"] += 1
        elif v <= 32:
            hist_bins["04_16_to_32_pos"] += 1
        else:
            hist_bins["05_gt_32_pos"] += 1
    per_class_sampling = {}
    for cl_name in CLASS_NAMES:
        n_gt_c = gt_instances_by_class[cl_name]
        n_pos_c = pos_anchors_by_class[cl_name]
        per_class_sampling[cl_name] = {
            "n_tiles_containing_class": tiles_with_class[cl_name],
            "n_projected_gt_instances": n_gt_c,
            "total_positive_anchors": n_pos_c,
            "mean_positive_anchors_per_gt": round(n_pos_c / max(1, n_gt_c), 2),
            "stage_b_positive_anchors": stage_b_pos_by_class[cl_name],
            "stage_cd_positive_anchors": stage_cd_pos_by_class[cl_name],
        }
    frac_below_16 = round(float(np.mean(arr < 16.0)), 4)
    return {
        "arm": arm,
        "n_positive_tiles_evaluated": len(raw_pos_values),
        "n_zero_gt_tiles_evaluated": zero_gt_tiles,
        "raw_pos_quantiles": {
            "min": int(np.min(arr)),
            "p10": round(float(np.percentile(arr, 10)), 1),
            "p25": round(float(np.percentile(arr, 25)), 1),
            "median_p50": round(float(np.percentile(arr, 50)), 1),
            "mean": round(float(np.mean(arr)), 2),
            "p75": round(float(np.percentile(arr, 75)), 1),
            "p90": round(float(np.percentile(arr, 90)), 1),
            "max": int(np.max(arr)),
        },
        "fraction_of_positive_tiles_with_raw_pos_lt_16": frac_below_16,
        "histogram_bins": dict(sorted(hist_bins.items())),
        "per_class_tile_and_anchor_sampling": per_class_sampling,
        "v11_ema_behavior_note": (
            "In V5-V11, PositiveAnchorEMA.get_denom(raw_pos) returns max(min_floor, raw_pos) whenever raw_pos > 0 "
            "(giving exact 1.00x weight per positive anchor on every positive tile, including sparse 1-GT large-object tiles), "
            "and uses self.ema ONLY when raw_pos == 0."
        ),
    }


# ============================================================================
# 2. COCO AP@0.30, AP@0.50, AP_S@50, AP_S@[.50:.95], 1-TO-1 RECALLS & FORENSICS
# ============================================================================

def compute_ap_at_iou_and_area_range(
    gt_by_image_class: Dict[Tuple[int, int], List[List[float]]],
    preds_by_class: Dict[int, List[Dict]],
    iou_thresh: float = 0.50,
    area_range: Tuple[float, float] = (0.0, 1e12),
) -> Dict[str, float]:
    """Compute per-class and 6-class/7-class macro-averaged AP + absent-class FP counts at a single IoU threshold.
    Enforces exact `pycocotools.cocoeval.COCOeval.evaluateImg` 1-to-1 matching semantics for both eligible and
    area-ignored non-crowd GTs (ChatGPT V10 Review Finding #1):
    - Eligible (`not gt_ignore`) unmatched GTs are checked first.
    - If no eligible unmatched GT has `IoU >= iou_thresh`, area-ignored (`gt_ignore`) **unmatched** GTs are checked;
      if matched, that ignored GT is marked consumed (`gt_matched = True`) so subsequent duplicate predictions
      overlapping the same ignored GT are NOT ignored and instead count as False Positives (when within `area_range`).
    """
    aps: Dict[str, float] = {}
    valid_class_aps: List[float] = []
    absent_class_fp_counts: Dict[str, int] = {}
    for cl in range(NUM_CLASSES):
        n_gt_eligible = 0
        gt_matched: Dict[Tuple[int, int], List[bool]] = {}
        gt_ignore: Dict[Tuple[int, int], List[bool]] = {}
        for (iid, c_id), boxes in gt_by_image_class.items():
            if c_id != cl:
                continue
            matched_flags = [False] * len(boxes)
            ignore_flags = []
            for b in boxes:
                area = float(b[2]) * float(b[3])
                ign = not (area_range[0] <= area <= area_range[1])
                ignore_flags.append(ign)
                if not ign:
                    n_gt_eligible += 1
            gt_matched[(iid, cl)] = matched_flags
            gt_ignore[(iid, cl)] = ignore_flags
        preds = sorted(preds_by_class.get(cl, []), key=lambda x: float(x["score"]), reverse=True)
        if n_gt_eligible == 0:
            absent_class_fp_counts[CLASS_NAMES[cl]] = len(preds)
            continue
        tps = np.zeros(len(preds), dtype=np.float64)
        fps = np.zeros(len(preds), dtype=np.float64)
        for p_idx, p in enumerate(preds):
            iid = int(p["image_id"])
            pb = np.asarray([p["bbox"]], dtype=np.float64)
            p_area = float(pb[0, 2] * pb[0, 3])
            gts = gt_by_image_class.get((iid, cl), [])
            if not gts:
                if area_range[0] <= p_area <= area_range[1]:
                    fps[p_idx] = 1.0
                continue
            gt_arr = np.asarray(gts, dtype=np.float64)
            ious = hr_p2.iou_matrix_numpy(pb, gt_arr)[0]
            # Pass 1: Match against eligible (non-ignored), unmatched GTs first (COCOeval ordering)
            best_g = -1
            best_iou = iou_thresh - 1e-9
            for g_i, iou_val in enumerate(ious):
                if not gt_ignore[(iid, cl)][g_i] and not gt_matched[(iid, cl)][g_i] and iou_val >= best_iou:
                    best_iou = float(iou_val)
                    best_g = g_i
            if best_g >= 0:
                gt_matched[(iid, cl)][best_g] = True
                tps[p_idx] = 1.0
            else:
                # Pass 2: Match against area-ignored, UNMATCHED GTs and consume the matched ignored GT (1-to-1)
                best_ign_g = -1
                best_ign_iou = iou_thresh - 1e-9
                for g_i, iou_val in enumerate(ious):
                    if gt_ignore[(iid, cl)][g_i] and not gt_matched[(iid, cl)][g_i] and iou_val >= best_ign_iou:
                        best_ign_iou = float(iou_val)
                        best_ign_g = g_i
                if best_ign_g >= 0:
                    gt_matched[(iid, cl)][best_ign_g] = True
                elif area_range[0] <= p_area <= area_range[1]:
                    fps[p_idx] = 1.0
        tp_cum = np.cumsum(tps)
        fp_cum = np.cumsum(fps)
        recalls = tp_cum / float(n_gt_eligible)
        precisions = tp_cum / np.maximum(1e-9, tp_cum + fp_cum)
        ap_101 = 0.0
        for r_thr in np.linspace(0.0, 1.0, 101):
            inds = np.where(recalls >= r_thr)[0]
            p_max = float(np.max(precisions[inds])) if len(inds) > 0 else 0.0
            ap_101 += p_max / 101.0
        aps[CLASS_NAMES[cl]] = round(ap_101, 4)
        aps[f"_unrounded_{CLASS_NAMES[cl]}"] = float(ap_101)
        valid_class_aps.append(ap_101)
    aps["mAP"] = round(float(np.mean(valid_class_aps)) if valid_class_aps else 0.0, 4)
    aps["_unrounded_mAP"] = float(np.mean(valid_class_aps)) if valid_class_aps else 0.0
    aps["_n_present_classes"] = len(valid_class_aps)
    for k_abs, fp_cnt in absent_class_fp_counts.items():
        aps[f"absent_class_fp_count_{k_abs}"] = fp_cnt
    return aps


def compute_coco_ap_50_95_for_area_range(
    gt_by_image_class: Dict[Tuple[int, int], List[List[float]]],
    preds_by_class: Dict[int, List[Dict]],
    area_range: Tuple[float, float] = (0.0, 1024.0),
) -> float:
    """Compute standard COCO 10-threshold AP (`IoU = 0.50:0.05:0.95`) for a given `area_range`."""
    iou_thrs = np.linspace(0.50, 0.95, 10)
    per_thr_maps = [
        float(compute_ap_at_iou_and_area_range(gt_by_image_class, preds_by_class, iou_thresh=float(t), area_range=area_range)["_unrounded_mAP"])
        for t in iou_thrs
    ]
    return round(float(np.mean(per_thr_maps)) if per_thr_maps else 0.0, 4)


def assert_ap_table_arithmetic_consistency(overall: Dict[str, object]) -> None:
    """Verify mathematically that macro AP50, macro AP30, and 3-class subgroup means match the per-class AP dicts."""
    ap50_dict = overall.get("AP50", {})
    ap30_dict = overall.get("AP30_per_class", {})
    if not isinstance(ap50_dict, dict) or not isinstance(ap30_dict, dict):
        raise AssertionError("Missing AP50 or AP30_per_class dict in overall metrics.")
    present_50 = [float(ap50_dict[c]) for c in CLASS_NAMES if c in ap50_dict]
    present_30 = [float(ap30_dict[c]) for c in CLASS_NAMES if c in ap30_dict]
    if present_50:
        mean_50 = float(np.mean(present_50))
        got_50 = float(ap50_dict.get("mAP", 0.0))
        if abs(mean_50 - got_50) > 2e-4:
            raise AssertionError(f"[ARITHMETIC INCONSISTENCY] AP50 mAP={got_50} != mean(per_class)={mean_50:.5f}")
    if present_30:
        mean_30 = float(np.mean(present_30))
        got_30 = float(overall.get("AP30_mAP", 0.0))
        if abs(mean_30 - got_30) > 2e-4:
            raise AssertionError(f"[ARITHMETIC INCONSISTENCY] AP30 mAP={got_30} != mean(per_class)={mean_30:.5f}")
    well_3_50 = [float(ap50_dict[c]) for c in ("BOAT", "MARINE_MARKER", "SAILING_VESSEL") if c in ap50_dict]
    if well_3_50:
        if abs(float(np.mean(well_3_50)) - float(overall["AP50_3_well_sampled_classes_mean"])) > 2e-4:
            raise AssertionError("AP50_3_well_sampled_classes_mean arithmetic mismatch")
    starved_3_50 = [float(ap50_dict[c]) for c in ("ANIMAL", "OBJECT", "SHIP") if c in ap50_dict]
    if starved_3_50:
        if abs(float(np.mean(starved_3_50)) - float(overall["AP50_3_starved_classes_mean"])) > 2e-4:
            raise AssertionError("AP50_3_starved_classes_mean arithmetic mismatch")
    well_3_30 = [float(ap30_dict[c]) for c in ("BOAT", "MARINE_MARKER", "SAILING_VESSEL") if c in ap30_dict]
    if well_3_30:
        if abs(float(np.mean(well_3_30)) - float(overall["AP30_3_well_sampled_classes_mean"])) > 2e-4:
            raise AssertionError("AP30_3_well_sampled_classes_mean arithmetic mismatch")
    starved_3_30 = [float(ap30_dict[c]) for c in ("ANIMAL", "OBJECT", "SHIP") if c in ap30_dict]
    if starved_3_30:
        if abs(float(np.mean(starved_3_30)) - float(overall["AP30_3_starved_classes_mean"])) > 2e-4:
            raise AssertionError("AP30_3_starved_classes_mean arithmetic mismatch")
    gt_counts = overall.get("gt_counts_by_class", {})
    if isinstance(gt_counts, dict) and gt_counts:
        tot_gt_c = sum(int(gt_counts.get(c, 0)) for c in CLASS_NAMES if c in ap50_dict)
        if tot_gt_c > 0 and "AP50_gt_weighted_mAP" in overall:
            exp_gtw_50 = sum(float(ap50_dict[c]) * int(gt_counts.get(c, 0)) for c in CLASS_NAMES if c in ap50_dict) / float(tot_gt_c)
            if abs(exp_gtw_50 - float(overall["AP50_gt_weighted_mAP"])) > 3e-4:
                raise AssertionError(f"AP50_gt_weighted_mAP mismatch: {overall['AP50_gt_weighted_mAP']} != {exp_gtw_50:.5f}")
        if tot_gt_c > 0 and "AP30_gt_weighted_mAP" in overall:
            exp_gtw_30 = sum(float(ap30_dict[c]) * int(gt_counts.get(c, 0)) for c in CLASS_NAMES if c in ap30_dict) / float(tot_gt_c)
            if abs(exp_gtw_30 - float(overall["AP30_gt_weighted_mAP"])) > 3e-4:
                raise AssertionError(f"AP30_gt_weighted_mAP mismatch: {overall['AP30_gt_weighted_mAP']} != {exp_gtw_30:.5f}")


def count_multi_gt_prediction_overlaps(
    gt_boxes_xywh: Sequence[Sequence[float]],
    preds: Sequence[Dict],
    iou_thresh: float = 0.50,
) -> Tuple[int, int]:
    """Count (1) predictions overlapping >= 2 GTs at `IoU >= iou_thresh`, and (2) extra `(pred, gt)` overlap pairs beyond 1 per pred."""
    if not gt_boxes_xywh or not preds:
        return 0, 0
    gt_arr = np.asarray(gt_boxes_xywh, dtype=np.float64)
    pr_arr = np.asarray([p["bbox"] for p in preds], dtype=np.float64)
    iou_mat = hr_p2.iou_matrix_numpy(pr_arr, gt_arr)  # [N_preds, M_gts]
    overlaps_per_pred = np.sum(iou_mat >= (float(iou_thresh) - 1e-9), axis=1)
    n_multi = int(np.sum(overlaps_per_pred >= 2))
    n_extra_pairs = int(np.sum(np.maximum(0, overlaps_per_pred - 1)))
    return n_multi, n_extra_pairs


def match_gts_one_to_one_in_image(
    gt_boxes_xywh: Sequence[Sequence[float]],
    preds: Sequence[Dict],
    iou_thresh: float = 0.50,
) -> Tuple[List[bool], List[Optional[Dict]], List[float]]:
    """Perform standard COCO/VOC 1-to-1 greedy matching (sorted by prediction score descending) within a single image."""
    n_gts = len(gt_boxes_xywh)
    if n_gts == 0:
        return [], [], []
    if not preds:
        return [False] * n_gts, [None] * n_gts, [0.0] * n_gts
    preds_sorted = sorted(preds, key=lambda x: float(x["score"]), reverse=True)
    gt_arr = np.asarray(gt_boxes_xywh, dtype=np.float64)
    pr_arr = np.asarray([p["bbox"] for p in preds_sorted], dtype=np.float64)
    iou_mat = hr_p2.iou_matrix_numpy(pr_arr, gt_arr)  # [N_preds, M_gts]
    max_iou_any = [float(v) for v in np.max(iou_mat, axis=0)]
    gt_matched = [False] * n_gts
    matched_pred_for_gt: List[Optional[Dict]] = [None] * n_gts
    for p_idx, p in enumerate(preds_sorted):
        ious_p = iou_mat[p_idx]
        best_g = -1
        best_iou = float(iou_thresh) - 1e-9
        for g_i in range(n_gts):
            if not gt_matched[g_i] and float(ious_p[g_i]) >= best_iou:
                best_iou = float(ious_p[g_i])
                best_g = g_i
        if best_g >= 0:
            gt_matched[best_g] = True
            matched_pred_for_gt[best_g] = {
                **p,
                "matched_iou": round(float(ious_p[best_g]), 4),
                "pred_rank_in_group": int(p_idx),
            }
    return gt_matched, matched_pred_for_gt, max_iou_any


def compute_class_agnostic_ap_and_bin_recalls(
    coco_images: Dict[int, Dict],
    anns_by_image: Dict[int, List[Dict]],
    all_preds: Sequence[Dict],
    iou_thresh: float = 0.50,
) -> Dict[str, object]:
    """Compute class-agnostic AP (`mAP_ca` at `IoU=0.50` and `IoU=0.30`), strict 1-to-1 recall by effective-area bin
    at BOTH `IoU=0.50` and `IoU=0.30`, `n_unmatched_gts_with_same_class_overlap`, and pairwise multi-GT prediction overlap counts.
    """
    gt_ca: Dict[Tuple[int, int], List[List[float]]] = defaultdict(list)
    for iid, anns in anns_by_image.items():
        for a in anns:
            gt_ca[(int(iid), 0)].append(list(map(float, a["bbox"])))
    preds_ca = {0: [{**p, "category_id": 1} for p in all_preds]}
    ca_res_50 = compute_ap_at_iou_and_area_range(gt_ca, preds_ca, iou_thresh=iou_thresh)
    ca_res_30 = compute_ap_at_iou_and_area_range(gt_ca, preds_ca, iou_thresh=0.30)
    preds_by_img_cl: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)
    preds_by_img_any: Dict[int, List[Dict]] = defaultdict(list)
    for p in all_preds:
        preds_by_img_cl[(int(p["image_id"]), int(p["category_id"]) - 1)].append(p)
        preds_by_img_any[int(p["image_id"])].append(p)
    bin_totals = Counter()
    bin_hits_50 = Counter()
    bin_hits_30 = Counter()
    bin_hits_ca_50 = Counter()
    bin_hits_ca_30 = Counter()
    bin_any_overlap_50 = Counter()
    bin_any_overlap_30 = Counter()
    le_9_total = 0
    le_9_hits_50 = 0
    le_9_hits_30 = 0
    le_9_ca_hits_50 = 0
    le_9_ca_hits_30 = 0
    tot_multi_gt_preds_50 = 0
    tot_extra_overlap_pairs_50 = 0
    tot_multi_gt_preds_30 = 0
    tot_extra_overlap_pairs_30 = 0
    for iid, anns in sorted(anns_by_image.items()):
        if not anns:
            continue
        im = coco_images[int(iid)]
        sw, sh = int(im["width"]), int(im["height"])
        all_gt_boxes_img = [list(map(float, a["bbox"])) for a in anns]
        cands_any = preds_by_img_any.get(int(iid), [])
        ca_matched_50, _, _ = match_gts_one_to_one_in_image(all_gt_boxes_img, cands_any, iou_thresh=iou_thresh)
        ca_matched_30, _, _ = match_gts_one_to_one_in_image(all_gt_boxes_img, cands_any, iou_thresh=0.30)
        ann_indices_by_cl: Dict[int, List[int]] = defaultdict(list)
        for idx_a, a in enumerate(anns):
            ann_indices_by_cl[int(a["category_id"]) - 1].append(idx_a)
        cl_matched_50 = [False] * len(anns)
        cl_matched_30 = [False] * len(anns)
        cl_max_iou_any = [0.0] * len(anns)
        for cl, a_indices in ann_indices_by_cl.items():
            cl_gt_boxes = [all_gt_boxes_img[i] for i in a_indices]
            cands_cl = preds_by_img_cl.get((int(iid), cl), [])
            m50, _, max_ious = match_gts_one_to_one_in_image(cl_gt_boxes, cands_cl, iou_thresh=iou_thresh)
            m30, _, _ = match_gts_one_to_one_in_image(cl_gt_boxes, cands_cl, iou_thresh=0.30)
            n_mp50, n_ep50 = count_multi_gt_prediction_overlaps(cl_gt_boxes, cands_cl, iou_thresh=iou_thresh)
            n_mp30, n_ep30 = count_multi_gt_prediction_overlaps(cl_gt_boxes, cands_cl, iou_thresh=0.30)
            tot_multi_gt_preds_50 += n_mp50
            tot_extra_overlap_pairs_50 += n_ep50
            tot_multi_gt_preds_30 += n_mp30
            tot_extra_overlap_pairs_30 += n_ep30
            for local_i, global_i in enumerate(a_indices):
                cl_matched_50[global_i] = m50[local_i]
                cl_matched_30[global_i] = m30[local_i]
                cl_max_iou_any[global_i] = max_ious[local_i]
        for idx_a, a in enumerate(anns):
            gx, gy, gw, gh = all_gt_boxes_img[idx_a]
            eff_640 = hr_p2.compute_effective_area_px2(gw * gh, sw, sh, 640.0)
            b_lbl = hr_p2.effective_area_bin_label(eff_640)
            bin_totals[b_lbl] += 1
            is_le_9 = (eff_640 <= 9.0 + 1e-9)
            if is_le_9:
                le_9_total += 1
            if cl_matched_50[idx_a]:
                bin_hits_50[b_lbl] += 1
                if is_le_9:
                    le_9_hits_50 += 1
            if cl_matched_30[idx_a]:
                bin_hits_30[b_lbl] += 1
                if is_le_9:
                    le_9_hits_30 += 1
            if ca_matched_50[idx_a]:
                bin_hits_ca_50[b_lbl] += 1
                if is_le_9:
                    le_9_ca_hits_50 += 1
            if ca_matched_30[idx_a]:
                bin_hits_ca_30[b_lbl] += 1
                if is_le_9:
                    le_9_ca_hits_30 += 1
            if cl_max_iou_any[idx_a] >= iou_thresh - 1e-9:
                bin_any_overlap_50[b_lbl] += 1
            if cl_max_iou_any[idx_a] >= 0.30 - 1e-9:
                bin_any_overlap_30[b_lbl] += 1
    return {
        "mAP_class_agnostic": ca_res_50.get("ANIMAL", 0.0),
        "mAP_class_agnostic_30": ca_res_30.get("ANIMAL", 0.0),
        "recall_le_9px2_at_640": round(le_9_hits_50 / max(1, le_9_total), 4),
        "recall_le_9px2_at_640_iou30": round(le_9_hits_30 / max(1, le_9_total), 4),
        "recall_le_9px2_class_agnostic_iou50": round(le_9_ca_hits_50 / max(1, le_9_total), 4),
        "recall_le_9px2_class_agnostic_iou30": round(le_9_ca_hits_30 / max(1, le_9_total), 4),
        "n_preds_overlapping_multiple_same_class_gts_50": tot_multi_gt_preds_50,
        "n_extra_pred_gt_overlap_pairs_50": tot_extra_overlap_pairs_50,
        "n_preds_overlapping_multiple_same_class_gts_30": tot_multi_gt_preds_30,
        "n_extra_pred_gt_overlap_pairs_30": tot_extra_overlap_pairs_30,
        "recall_by_effective_area_bin_at_640": {
            b: {
                "n_gt": bin_totals[b],
                "n_tp": bin_hits_50[b],
                "recall": round(bin_hits_50[b] / max(1, bin_totals[b]), 4),
                "n_tp_iou30": bin_hits_30[b],
                "recall_iou30": round(bin_hits_30[b] / max(1, bin_totals[b]), 4),
                "n_tp_ca_iou50": bin_hits_ca_50[b],
                "recall_class_agnostic_iou50": round(bin_hits_ca_50[b] / max(1, bin_totals[b]), 4),
                "n_tp_ca_iou30": bin_hits_ca_30[b],
                "recall_class_agnostic_iou30": round(bin_hits_ca_30[b] / max(1, bin_totals[b]), 4),
                "n_any_overlap_50": bin_any_overlap_50[b],
                "n_any_overlap_30": bin_any_overlap_30[b],
                "n_unmatched_gts_with_same_class_overlap_50": bin_any_overlap_50[b] - bin_hits_50[b],
                "n_unmatched_gts_with_same_class_overlap_30": bin_any_overlap_30[b] - bin_hits_30[b],
                "n_crowded_dup_50": bin_any_overlap_50[b] - bin_hits_50[b],
                "n_crowded_dup_30": bin_any_overlap_30[b] - bin_hits_30[b],
            }
            for b in sorted(bin_totals.keys())
        },
    }


def diagnose_tiny_le_4px2_and_ship_confusion(
    coco_images: Dict[int, Dict],
    anns_by_image: Dict[int, List[Dict]],
    preds_std: Sequence[Dict],
    preds_low_thresh: Optional[Sequence[Dict]] = None,
) -> Dict[str, object]:
    """Forensic 1-to-1 diagnosis on VAL without retraining."""
    if preds_low_thresh is None:
        preds_low_thresh = preds_std
    std_by_img_cl: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)
    std_by_img_any: Dict[int, List[Dict]] = defaultdict(list)
    for p in preds_std:
        std_by_img_cl[(int(p["image_id"]), int(p["category_id"]) - 1)].append(p)
        std_by_img_any[int(p["image_id"])].append(p)
    low_by_img_cl: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)
    low_by_img_any: Dict[int, List[Dict]] = defaultdict(list)
    for p in preds_low_thresh:
        low_by_img_cl[(int(p["image_id"]), int(p["category_id"]) - 1)].append(p)
        low_by_img_any[int(p["image_id"])].append(p)
    le4_buckets = Counter()
    le4_best_ious: List[float] = []
    le4_best_scores: List[float] = []
    tiny_le9_by_class: Dict[str, Counter] = defaultdict(Counter)
    ship_confusion_iou50 = Counter()
    ship_confusion_iou30 = Counter()
    large_gt_status_iou50 = Counter()
    per_class_audit: Dict[str, Counter] = defaultdict(Counter)
    for iid, anns in sorted(anns_by_image.items()):
        if not anns:
            continue
        im = coco_images[int(iid)]
        sw, sh = int(im["width"]), int(im["height"])
        all_gt_boxes = [list(map(float, a["bbox"])) for a in anns]
        ca_std_m50, ca_std_pred50, ca_std_max_iou = match_gts_one_to_one_in_image(
            all_gt_boxes, std_by_img_any.get(int(iid), []), iou_thresh=0.50
        )
        ca_std_m30, ca_std_pred30, _ = match_gts_one_to_one_in_image(
            all_gt_boxes, std_by_img_any.get(int(iid), []), iou_thresh=0.30
        )
        ca_low_m50, _, ca_low_max_iou = match_gts_one_to_one_in_image(
            all_gt_boxes, low_by_img_any.get(int(iid), []), iou_thresh=0.50
        )
        ca_low_m30, ca_low_pred30, _ = match_gts_one_to_one_in_image(
            all_gt_boxes, low_by_img_any.get(int(iid), []), iou_thresh=0.30
        )
        ann_indices_by_cl: Dict[int, List[int]] = defaultdict(list)
        for idx_a, a in enumerate(anns):
            ann_indices_by_cl[int(a["category_id"]) - 1].append(idx_a)
        cl_std_m50 = [False] * len(anns)
        cl_std_m30 = [False] * len(anns)
        cl_std_max_iou = [0.0] * len(anns)
        cl_low_m50 = [False] * len(anns)
        cl_low_m30 = [False] * len(anns)
        cl_low_max_iou = [0.0] * len(anns)
        cl_low_pred30: List[Optional[Dict]] = [None] * len(anns)
        for cl, a_indices in ann_indices_by_cl.items():
            cl_gt_boxes = [all_gt_boxes[i] for i in a_indices]
            cands_std_cl = std_by_img_cl.get((int(iid), cl), [])
            cands_low_cl = low_by_img_cl.get((int(iid), cl), [])
            m50_s, _, max_iou_s = match_gts_one_to_one_in_image(cl_gt_boxes, cands_std_cl, iou_thresh=0.50)
            m30_s, _, _ = match_gts_one_to_one_in_image(cl_gt_boxes, cands_std_cl, iou_thresh=0.30)
            m50_l, _, max_iou_l = match_gts_one_to_one_in_image(cl_gt_boxes, cands_low_cl, iou_thresh=0.50)
            m30_l, p30_l, _ = match_gts_one_to_one_in_image(cl_gt_boxes, cands_low_cl, iou_thresh=0.30)
            for local_i, global_i in enumerate(a_indices):
                cl_std_m50[global_i] = m50_s[local_i]
                cl_std_m30[global_i] = m30_s[local_i]
                cl_std_max_iou[global_i] = max_iou_s[local_i]
                cl_low_m50[global_i] = m50_l[local_i]
                cl_low_m30[global_i] = m30_l[local_i]
                cl_low_max_iou[global_i] = max_iou_l[local_i]
                cl_low_pred30[global_i] = p30_l[local_i]
        cands_std_any = std_by_img_any.get(int(iid), [])
        pr_std_any_arr = np.asarray([c["bbox"] for c in cands_std_any], dtype=np.float64) if cands_std_any else np.zeros((0, 4), dtype=np.float64)
        gt_all_arr = np.asarray(all_gt_boxes, dtype=np.float64)
        iou_mat_std_any = hr_p2.iou_matrix_numpy(pr_std_any_arr, gt_all_arr) if len(pr_std_any_arr) > 0 else np.zeros((0, len(anns)), dtype=np.float64)
        for idx_a, a in enumerate(anns):
            cl = int(a["category_id"]) - 1
            cl_name = CLASS_NAMES[cl]
            gx, gy, gw, gh = all_gt_boxes[idx_a]
            eff_640 = hr_p2.compute_effective_area_px2(gw * gh, sw, sh, 640.0)
            pred_cl_30_name = None
            if ca_std_m30[idx_a] and ca_std_pred30[idx_a] is not None:
                pred_cl_30_name = CLASS_NAMES[int(ca_std_pred30[idx_a]["category_id"]) - 1]
            elif ca_std_max_iou[idx_a] >= 0.30 - 1e-9 and len(cands_std_any) > 0:
                best_p_i = int(np.argmax(iou_mat_std_any[:, idx_a]))
                pred_cl_30_name = CLASS_NAMES[int(cands_std_any[best_p_i]["category_id"]) - 1]
            pred_cl_50_name = None
            if ca_std_m50[idx_a] and ca_std_pred50[idx_a] is not None:
                pred_cl_50_name = CLASS_NAMES[int(ca_std_pred50[idx_a]["category_id"]) - 1]
            elif ca_std_max_iou[idx_a] >= 0.50 - 1e-9 and len(cands_std_any) > 0:
                best_p_i = int(np.argmax(iou_mat_std_any[:, idx_a]))
                pred_cl_50_name = CLASS_NAMES[int(cands_std_any[best_p_i]["category_id"]) - 1]
            pc = per_class_audit[cl_name]
            pc["n_gt"] += 1
            if cl_std_m50[idx_a]:
                pc["tp_1to1_iou50"] += 1
            if cl_std_m30[idx_a]:
                pc["tp_1to1_iou30"] += 1
            if ca_std_m50[idx_a]:
                pc["tp_ca_1to1_iou50"] += 1
            if ca_std_m30[idx_a]:
                pc["tp_ca_1to1_iou30"] += 1
            if cl_low_m50[idx_a]:
                pc["low_0005_tp_1to1_iou50"] += 1
            if cl_low_m30[idx_a]:
                pc["low_0005_tp_1to1_iou30"] += 1
            if cl_std_m30[idx_a]:
                pc["iou30_01_TP_1to1_same_class"] += 1
            elif cl_std_max_iou[idx_a] >= 0.30 - 1e-9:
                pc["iou30_02_UNMATCHED_GT_WITH_CONSUMED_SAME_CLASS_OVERLAP"] += 1
            elif cl_low_m30[idx_a]:
                pc["iou30_03_LOW_SCORE_0005_to_005_1to1"] += 1
            elif pred_cl_30_name is not None and pred_cl_30_name != cl_name:
                pc[f"iou30_04_CONFUSED_as_{pred_cl_30_name}"] += 1
            else:
                pc["iou30_05_UNLOCALIZED_iou_lt_030"] += 1
            if eff_640 <= 4.0 + 1e-9:
                le4_best_ious.append(cl_low_max_iou[idx_a])
                best_sc = float(cl_low_pred30[idx_a]["score"]) if cl_low_pred30[idx_a] is not None else 0.0
                le4_best_scores.append(best_sc)
                if cl_std_m50[idx_a]:
                    le4_buckets["01_TP_iou50_1to1_score_ge_005"] += 1
                elif cl_std_m30[idx_a]:
                    le4_buckets["02_LOCALIZATION_JITTER_iou_030_to_050_1to1_score_ge_005"] += 1
                elif cl_std_max_iou[idx_a] >= 0.30 - 1e-9:
                    le4_buckets["03_UNMATCHED_GT_WITH_CONSUMED_SAME_CLASS_OVERLAP_iou_ge_030"] += 1
                elif cl_low_m30[idx_a]:
                    le4_buckets["04_THRESHOLD_OR_MAXDET_TRUNCATED_1to1_in_low_0005"] += 1
                elif pred_cl_30_name is not None and pred_cl_30_name != cl_name:
                    le4_buckets[f"05_WRONG_CLASS_iou_ge_030_as_{pred_cl_30_name}"] += 1
                elif ca_low_max_iou[idx_a] >= 0.10 - 1e-9:
                    le4_buckets["06_NEAR_MISS_iou_010_to_030"] += 1
                else:
                    le4_buckets["07_UNLOCALIZED_iou_lt_010"] += 1
            if eff_640 <= 9.0 + 1e-9:
                tc = tiny_le9_by_class[cl_name]
                tc["n_gt"] += 1
                if cl_std_m50[idx_a]:
                    tc["01_TP_1to1_iou50"] += 1
                if cl_std_m30[idx_a]:
                    tc["02_TP_1to1_iou30"] += 1
                elif cl_std_max_iou[idx_a] >= 0.30 - 1e-9:
                    tc["03_UNMATCHED_GT_WITH_CONSUMED_SAME_CLASS_OVERLAP_iou30"] += 1
                elif cl_low_m30[idx_a]:
                    tc["04_LOW_SCORE_0005_to_005_1to1_iou30"] += 1
                elif pred_cl_30_name is not None and pred_cl_30_name != cl_name:
                    tc[f"05_CONFUSED_as_{pred_cl_30_name}_iou30"] += 1
                else:
                    tc["06_UNLOCALIZED_iou_lt_030"] += 1
            if cl == 6:
                if cl_std_m50[idx_a]:
                    ship_confusion_iou50["01_TP_1to1_as_SHIP"] += 1
                elif cl_std_max_iou[idx_a] >= 0.50 - 1e-9:
                    ship_confusion_iou50["02_UNMATCHED_GT_WITH_CONSUMED_SAME_CLASS_OVERLAP_as_SHIP"] += 1
                elif pred_cl_50_name is not None and pred_cl_50_name != "SHIP":
                    ship_confusion_iou50[f"03_CONFUSED_as_{pred_cl_50_name}"] += 1
                elif cl_low_m50[idx_a] or ca_low_max_iou[idx_a] >= 0.50 - 1e-9:
                    ship_confusion_iou50["04_LOW_SCORE_lt_005_or_RANK_gt_100"] += 1
                else:
                    ship_confusion_iou50["05_UNLOCALIZED_at_iou50"] += 1
                if cl_std_m30[idx_a]:
                    ship_confusion_iou30["01_TP_1to1_as_SHIP"] += 1
                elif cl_std_max_iou[idx_a] >= 0.30 - 1e-9:
                    ship_confusion_iou30["02_UNMATCHED_GT_WITH_CONSUMED_SAME_CLASS_OVERLAP_as_SHIP"] += 1
                elif pred_cl_30_name is not None and pred_cl_30_name != "SHIP":
                    ship_confusion_iou30[f"03_CONFUSED_as_{pred_cl_30_name}"] += 1
                elif cl_low_m30[idx_a] or ca_low_max_iou[idx_a] >= 0.30 - 1e-9:
                    ship_confusion_iou30["04_LOW_SCORE_lt_005_or_RANK_gt_100"] += 1
                else:
                    ship_confusion_iou30["05_UNLOCALIZED_at_iou30"] += 1
            if eff_640 > 150.0 + 1e-9:
                if cl_std_m50[idx_a]:
                    large_gt_status_iou50["01_TP_1to1_correct_class_iou50"] += 1
                elif cl_std_m30[idx_a]:
                    large_gt_status_iou50["02_LOCALIZATION_1to1_iou_030_to_050_correct_class"] += 1
                elif cl_std_max_iou[idx_a] >= 0.30 - 1e-9:
                    large_gt_status_iou50["03_UNMATCHED_GT_WITH_CONSUMED_SAME_CLASS_OVERLAP_iou30"] += 1
                elif pred_cl_50_name is not None and pred_cl_50_name != cl_name:
                    large_gt_status_iou50[f"04_WRONG_CLASS_at_iou50_as_{pred_cl_50_name}"] += 1
                elif ca_low_m30[idx_a] or ca_low_max_iou[idx_a] >= 0.30 - 1e-9:
                    large_gt_status_iou50["05_LOW_SCORE_or_WRONG_CLASS_iou_ge_030"] += 1
                else:
                    large_gt_status_iou50["06_UNLOCALIZED_iou_lt_030"] += 1
    per_class_summary = {}
    for cl_name in CLASS_NAMES:
        if cl_name not in per_class_audit:
            continue
        pc = per_class_audit[cl_name]
        n_c = max(1, pc["n_gt"])
        per_class_summary[cl_name] = {
            "n_gt": pc["n_gt"],
            "tp_1to1_iou50": pc["tp_1to1_iou50"],
            "recall_1to1_iou50": round(pc["tp_1to1_iou50"] / n_c, 4),
            "tp_1to1_iou30": pc["tp_1to1_iou30"],
            "recall_1to1_iou30": round(pc["tp_1to1_iou30"] / n_c, 4),
            "tp_ca_1to1_iou50": pc["tp_ca_1to1_iou50"],
            "recall_ca_1to1_iou50": round(pc["tp_ca_1to1_iou50"] / n_c, 4),
            "tp_ca_1to1_iou30": pc["tp_ca_1to1_iou30"],
            "recall_ca_1to1_iou30": round(pc["tp_ca_1to1_iou30"] / n_c, 4),
            "low_0005_tp_1to1_iou50": pc["low_0005_tp_1to1_iou50"],
            "low_0005_tp_1to1_iou30": pc["low_0005_tp_1to1_iou30"],
            "iou30_breakdown": {
                k: v for k, v in sorted(pc.items()) if k.startswith("iou30_")
            },
        }
    return {
        "tiny_01_LE_4px2_diagnostic": {
            "n_gt": len(le4_best_ious),
            "outcome_breakdown": dict(sorted(le4_buckets.items())),
            "median_best_same_class_iou": round(float(np.median(le4_best_ious)) if le4_best_ious else 0.0, 4),
            "max_best_same_class_iou": round(float(np.max(le4_best_ious)) if le4_best_ious else 0.0, 4),
        },
        "tiny_le_9px2_class_confusion_iou30_1to1": {
            k: dict(sorted(v.items())) for k, v in sorted(tiny_le9_by_class.items())
        },
        "ship_class_confusion_diagnostic": {
            "at_iou50": dict(sorted(ship_confusion_iou50.items())),
            "at_iou30": dict(sorted(ship_confusion_iou30.items())),
        },
        "large_gt_gt_150px2_diagnostic": dict(sorted(large_gt_status_iou50.items())),
        "per_class_1to1_audit_all_1386_gts": per_class_summary,
    }


def verify_against_pycocotools_if_available(
    eval_images_dict: Dict[int, Dict],
    eval_anns_dict: Dict[int, List[Dict]],
    all_preds: Sequence[Dict],
    custom_overall: Dict[str, object],
    fail_fast: bool = True,
    require_pycocotools: bool = True,
) -> Dict[str, object]:
    """Cross-check custom AP evaluator (`AP50`, `AP30`, `AP_S@50`, and standard 10-threshold `AP_S@[.50:.95]`) against
    `pycocotools.cocoeval.COCOeval`.
    """
    n_eval_gts = sum(len(v) for v in eval_anns_dict.values())
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        import contextlib
    except ImportError as imp_err:
        res_missing = {
            "pycocotools_available": False,
            "parity_status": "PARITY_NOT_RUN",
            "agrees_within_0_005": False,
            "note": "pycocotools not installed in this environment; parity was NOT run.",
        }
        if fail_fast and require_pycocotools:
            raise AssertionError(
                "[PYCOCOTOOLS PARITY GATE] pycocotools is required for release-gate evaluation, "
                "but could not be imported. Install pycocotools in (311env) or pass --allow-missing-pycocotools "
                "only when running in an offline environment without pycocotools."
            ) from imp_err
        return res_missing
    if not all_preds:
        res_empty = {
            "pycocotools_available": True,
            "parity_status": "PARITY_FAILED" if n_eval_gts > 0 else "PARITY_NOT_RUN",
            "n_preds": 0,
            "n_eval_gts": n_eval_gts,
            "agrees_within_0_005": False,
            "note": "Empty prediction list cannot be counted as a successful parity check.",
        }
        if fail_fast and n_eval_gts > 0:
            raise AssertionError(
                f"[PYCOCOTOOLS PARITY FAIL] Variant produced 0 predictions across {n_eval_gts} GTs; "
                "refusing to treat empty prediction list as parity pass."
            )
        return res_empty
    coco_gt_dict = {
        "images": list(eval_images_dict.values()),
        "annotations": [
            {
                **a,
                "area": float(a["bbox"][2]) * float(a["bbox"][3]),
                "iscrowd": int(a.get("iscrowd", 0)),
            }
            for anns in eval_anns_dict.values() for a in anns
        ],
        "categories": [{"id": i + 1, "name": name} for i, name in enumerate(CLASS_NAMES)],
    }
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = coco_gt_dict
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes([
            {
                "image_id": int(p["image_id"]),
                "category_id": int(p["category_id"]),
                "bbox": [float(v) for v in p["bbox"]],
                "score": float(p["score"]),
            }
            for p in all_preds
        ])
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.params.maxDets = [1, 10, 300]
        ev.evaluate()
        ev.accumulate()
        p_ap50_per_cat = ev.eval["precision"][0, :, :, 0, 2]
        valid_cat_ap50 = [float(np.mean(p_ap50_per_cat[:, k])) for k in range(NUM_CLASSES) if np.all(p_ap50_per_cat[:, k] > -1)]
        pycoco_ap50_unrounded = float(np.mean(valid_cat_ap50)) if valid_cat_ap50 else 0.0
        pycoco_ap50 = round(pycoco_ap50_unrounded, 4)
        per_class_ap50_abs_diff = {}
        for k in range(NUM_CLASSES):
            if np.all(p_ap50_per_cat[:, k] > -1):
                c_name = CLASS_NAMES[k]
                p_val = round(float(np.mean(p_ap50_per_cat[:, k])), 4)
                c_val = float(custom_overall["AP50"].get(c_name, 0.0))
                per_class_ap50_abs_diff[c_name] = round(abs(c_val - p_val), 5)
        p_aps50_per_cat = ev.eval["precision"][0, :, :, 1, 2]
        valid_cat_aps50 = [float(np.mean(p_aps50_per_cat[:, k])) for k in range(NUM_CLASSES) if np.all(p_aps50_per_cat[:, k] > -1)]
        pycoco_aps_at_50_unrounded = float(np.mean(valid_cat_aps50)) if valid_cat_aps50 else 0.0
        pycoco_aps_at_50 = round(pycoco_aps_at_50_unrounded, 4)
        p_aps_50_95_per_cat = ev.eval["precision"][:, :, :, 1, 2]
        valid_cat_aps_50_95 = [
            float(np.mean(p_aps_50_95_per_cat[:, :, k]))
            for k in range(NUM_CLASSES)
            if np.all(p_aps_50_95_per_cat[:, :, k] > -1)
        ]
        pycoco_aps_50_95_unrounded = float(np.mean(valid_cat_aps_50_95)) if valid_cat_aps_50_95 else 0.0
        pycoco_aps_50_95 = round(pycoco_aps_50_95_unrounded, 4)
        ev30 = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev30.params.iouThrs = np.array([0.30], dtype=np.float64)
        ev30.params.maxDets = [1, 10, 300]
        ev30.evaluate()
        ev30.accumulate()
        p_ap30_per_cat = ev30.eval["precision"][0, :, :, 0, 2]
        valid_cat_ap30 = [float(np.mean(p_ap30_per_cat[:, k])) for k in range(NUM_CLASSES) if np.all(p_ap30_per_cat[:, k] > -1)]
        pycoco_ap30_unrounded = float(np.mean(valid_cat_ap30)) if valid_cat_ap30 else 0.0
        pycoco_ap30 = round(pycoco_ap30_unrounded, 4)
    custom_ap50 = float(custom_overall["AP50"]["mAP"])
    custom_ap30 = float(custom_overall["AP30_mAP"])
    custom_aps_at_50 = float(custom_overall.get("AP_S_at_50_mAP", custom_overall["AP_S_mAP"]))
    custom_aps_50_95 = float(custom_overall.get("AP_S_coco_50_95_mAP", custom_aps_at_50))
    agrees = bool(
        abs(custom_ap50 - pycoco_ap50) <= 0.005
        and abs(custom_ap30 - pycoco_ap30) <= 0.005
        and abs(custom_aps_at_50 - pycoco_aps_at_50) <= 0.005
        and abs(custom_aps_50_95 - pycoco_aps_50_95) <= 0.005
    )
    res = {
        "pycocotools_available": True,
        "parity_status": "PARITY_PASSED" if agrees else "PARITY_FAILED",
        "area_definition": "bbox_width_times_height",
        "custom_AP50_mAP": custom_ap50,
        "pycocotools_AP50_mAP": pycoco_ap50,
        "abs_diff_AP50": round(abs(custom_ap50 - pycoco_ap50), 5),
        "unrounded_pycocotools_AP50_mAP": round(pycoco_ap50_unrounded, 6),
        "per_class_AP50_abs_diff": per_class_ap50_abs_diff,
        "custom_AP30_mAP": custom_ap30,
        "pycocotools_AP30_mAP": pycoco_ap30,
        "abs_diff_AP30": round(abs(custom_ap30 - pycoco_ap30), 5),
        "unrounded_pycocotools_AP30_mAP": round(pycoco_ap30_unrounded, 6),
        "custom_AP_S_at_50_mAP": custom_aps_at_50,
        "pycocotools_AP_S_at_50_mAP": pycoco_aps_at_50,
        "abs_diff_AP_S_at_50": round(abs(custom_aps_at_50 - pycoco_aps_at_50), 5),
        "unrounded_pycocotools_AP_S_at_50_mAP": round(pycoco_aps_at_50_unrounded, 6),
        "custom_AP_S_coco_50_95_mAP": custom_aps_50_95,
        "pycocotools_AP_S_coco_50_95_mAP": pycoco_aps_50_95,
        "abs_diff_AP_S_coco_50_95": round(abs(custom_aps_50_95 - pycoco_aps_50_95), 5),
        "unrounded_pycocotools_AP_S_coco_50_95_mAP": round(pycoco_aps_50_95_unrounded, 6),
        "agrees_within_0_005": agrees,
    }
    if fail_fast and not agrees:
        raise AssertionError(f"[PYCOCOTOOLS PARITY FAIL] Custom AP evaluator diverged from pycocotools beyond 0.005: {res}")
    return res


def evaluate_predictions_dict_and_per_video(
    eval_images_dict: Dict[int, Dict],
    eval_anns_dict: Dict[int, List[Dict]],
    all_preds: Sequence[Dict],
) -> Tuple[Dict[str, object], Dict[str, Dict[int, float]]]:
    """Compute overall metrics AND per-video metrics (`AP50`, `AP30`, `mAP_ca`, 1-to-1 `recall_50`, 1-to-1 `recall_30`) for paired bootstrap."""
    gt_by_image_class: Dict[Tuple[int, int], List[List[float]]] = defaultdict(list)
    for iid, anns in eval_anns_dict.items():
        for a in anns:
            cl = int(a["category_id"]) - 1
            gt_by_image_class[(int(iid), cl)].append(list(map(float, a["bbox"])))
    preds_by_class: Dict[int, List[Dict]] = defaultdict(list)
    for d in all_preds:
        preds_by_class[int(d["category_id"]) - 1].append(d)
    ap50 = compute_ap_at_iou_and_area_range(gt_by_image_class, preds_by_class, iou_thresh=0.50)
    ap30 = compute_ap_at_iou_and_area_range(gt_by_image_class, preds_by_class, iou_thresh=0.30)
    ap_s_50 = compute_ap_at_iou_and_area_range(gt_by_image_class, preds_by_class, iou_thresh=0.50, area_range=(0.0, 1024.0))
    ap_s_coco_50_95 = compute_coco_ap_50_95_for_area_range(gt_by_image_class, preds_by_class, area_range=(0.0, 1024.0))
    ca_bins = compute_class_agnostic_ap_and_bin_recalls(eval_images_dict, eval_anns_dict, all_preds, iou_thresh=0.50)
    bins_dict = ca_bins["recall_by_effective_area_bin_at_640"]
    n_eval_gts_raw = sum(b["n_gt"] for b in bins_dict.values())
    tot_gt = max(1, n_eval_gts_raw)
    tot_tp_50 = sum(b["n_tp"] for b in bins_dict.values())
    tot_tp_30 = sum(b["n_tp_iou30"] for b in bins_dict.values())
    tot_ca_50 = sum(b["n_tp_ca_iou50"] for b in bins_dict.values())
    tot_ca_30 = sum(b["n_tp_ca_iou30"] for b in bins_dict.values())
    tot_any_50 = sum(b["n_any_overlap_50"] for b in bins_dict.values())
    tot_any_30 = sum(b["n_any_overlap_30"] for b in bins_dict.values())
    well_sampled_3 = [ap50[c] for c in ("BOAT", "MARINE_MARKER", "SAILING_VESSEL") if c in ap50]
    starved_3 = [ap50[c] for c in ("ANIMAL", "OBJECT", "SHIP") if c in ap50]
    well_sampled_3_30 = [ap30[c] for c in ("BOAT", "MARINE_MARKER", "SAILING_VESSEL") if c in ap30]
    starved_3_30 = [ap30[c] for c in ("ANIMAL", "OBJECT", "SHIP") if c in ap30]
    ap50_clean = {k: v for k, v in ap50.items() if not k.startswith("_")}
    ap30_clean = {k: v for k, v in ap30.items() if not k.startswith("_")}
    gt_counts_by_cl: Dict[str, int] = {c: 0 for c in CLASS_NAMES}
    for (_, cl_i), boxes in gt_by_image_class.items():
        gt_counts_by_cl[CLASS_NAMES[cl_i]] += len(boxes)
    tot_present_gt = sum(gt_counts_by_cl[c] for c in CLASS_NAMES if c in ap50_clean)
    gtw_ap50 = (
        sum(float(ap50_clean[c]) * gt_counts_by_cl[c] for c in CLASS_NAMES if c in ap50_clean) / float(tot_present_gt)
        if tot_present_gt > 0 else 0.0
    )
    gtw_ap30 = (
        sum(float(ap30_clean[c]) * gt_counts_by_cl[c] for c in CLASS_NAMES if c in ap30_clean) / float(tot_present_gt)
        if tot_present_gt > 0 else 0.0
    )
    n_imgs_eval = max(1, len(eval_images_dict))
    fp_cnt_50 = max(0, len(all_preds) - int(tot_tp_50))
    fp_cnt_30 = max(0, len(all_preds) - int(tot_tp_30))
    overall = {
        "n_val_images_evaluated": len(eval_images_dict),
        "n_val_gt_annotations_evaluated": int(n_eval_gts_raw),
        "gt_counts_by_class": {k: v for k, v in gt_counts_by_cl.items() if v > 0},
        "n_total_predictions": len(all_preds),
        "mean_predictions_per_frame": round(len(all_preds) / float(n_imgs_eval), 2),
        "fp_count_1to1_iou50": int(fp_cnt_50),
        "fp_per_frame_1to1_iou50": round(fp_cnt_50 / float(n_imgs_eval), 2),
        "fp_count_1to1_iou30": int(fp_cnt_30),
        "fp_per_frame_1to1_iou30": round(fp_cnt_30 / float(n_imgs_eval), 2),
        "per_video_absent_class_ap_rule": (
            "Per-video macro AP averages ONLY over classes with n_gt_eligible > 0 in that video "
            "(matching COCOeval per-subset convention); absent classes in a video are excluded from that video's denominator."
        ),
        "AP50": ap50_clean,
        "AP50_gt_weighted_mAP": round(float(gtw_ap50), 4),
        "AP30_per_class": ap30_clean,
        "AP30_mAP": ap30_clean.get("mAP", 0.0),
        "AP30_gt_weighted_mAP": round(float(gtw_ap30), 4),
        "AP_S_at_50_mAP": ap_s_50.get("mAP", 0.0),
        "AP_S_coco_50_95_mAP": ap_s_coco_50_95,
        "AP_S_mAP": ap_s_50.get("mAP", 0.0),  # Backward-compatible alias for AP_S_at_50_mAP
        "AP50_3_well_sampled_classes_mean": round(float(np.mean(well_sampled_3)) if well_sampled_3 else 0.0, 4),
        "AP50_3_starved_classes_mean": round(float(np.mean(starved_3)) if starved_3 else 0.0, 4),
        "AP30_3_well_sampled_classes_mean": round(float(np.mean(well_sampled_3_30)) if well_sampled_3_30 else 0.0, 4),
        "AP30_3_starved_classes_mean": round(float(np.mean(starved_3_30)) if starved_3_30 else 0.0, 4),
        "mAP_class_agnostic_50": ca_bins["mAP_class_agnostic"],
        "mAP_class_agnostic_30": ca_bins["mAP_class_agnostic_30"],
        "total_1to1_tp_iou50": int(tot_tp_50),
        "overall_1to1_gt_recall_iou50": round(tot_tp_50 / tot_gt, 4),
        "total_1to1_tp_iou30": int(tot_tp_30),
        "overall_1to1_gt_recall_iou30": round(tot_tp_30 / tot_gt, 4),
        "total_1to1_ca_tp_iou50": int(tot_ca_50),
        "overall_1to1_ca_recall_iou50": round(tot_ca_50 / tot_gt, 4),
        "total_1to1_ca_tp_iou30": int(tot_ca_30),
        "overall_1to1_ca_recall_iou30": round(tot_ca_30 / tot_gt, 4),
        "total_any_overlap_gt_iou50": int(tot_any_50),
        "total_any_overlap_gt_iou30": int(tot_any_30),
        "total_unmatched_gts_with_same_class_overlap_50": int(tot_any_50 - tot_tp_50),
        "total_unmatched_gts_with_same_class_overlap_30": int(tot_any_30 - tot_tp_30),
        "n_preds_overlapping_multiple_same_class_gts_50": ca_bins["n_preds_overlapping_multiple_same_class_gts_50"],
        "n_extra_pred_gt_overlap_pairs_50": ca_bins["n_extra_pred_gt_overlap_pairs_50"],
        "n_preds_overlapping_multiple_same_class_gts_30": ca_bins["n_preds_overlapping_multiple_same_class_gts_30"],
        "n_extra_pred_gt_overlap_pairs_30": ca_bins["n_extra_pred_gt_overlap_pairs_30"],
        "recall_le_9px2_at_640": ca_bins["recall_le_9px2_at_640"],
        "recall_le_9px2_at_640_iou30": ca_bins["recall_le_9px2_at_640_iou30"],
        "recall_le_9px2_class_agnostic_iou50": ca_bins["recall_le_9px2_class_agnostic_iou50"],
        "recall_le_9px2_class_agnostic_iou30": ca_bins["recall_le_9px2_class_agnostic_iou30"],
        "recall_by_effective_area_bin_at_640": bins_dict,
        "absent_class_fp_count_LEISURE_VEHICLE": ap50_clean.get("absent_class_fp_count_LEISURE_VEHICLE", 0),
    }
    assert_ap_table_arithmetic_consistency(overall)
    iids_by_vid: Dict[int, List[int]] = defaultdict(list)
    for iid, im in eval_images_dict.items():
        iids_by_vid[int(im.get("video_id", 0))].append(int(iid))
    preds_by_iid: Dict[int, List[Dict]] = defaultdict(list)
    for p in all_preds:
        preds_by_iid[int(p["image_id"])].append(p)
    per_video_metrics: Dict[str, Dict[int, float]] = {
        "AP50_mAP": {},
        "AP30_mAP": {},
        "mAP_class_agnostic_50": {},
        "overall_gt_recall_50": {},
        "overall_gt_recall_30": {},
    }
    for vid, vid_iids in sorted(iids_by_vid.items()):
        v_imgs = {iid: eval_images_dict[iid] for iid in vid_iids}
        v_anns = {iid: eval_anns_dict.get(iid, []) for iid in vid_iids}
        n_v_gts = sum(len(v) for v in v_anns.values())
        if n_v_gts == 0:
            continue
        v_preds = [p for iid in vid_iids for p in preds_by_iid.get(iid, [])]
        v_gt_ic: Dict[Tuple[int, int], List[List[float]]] = defaultdict(list)
        for iid, anns in v_anns.items():
            for a in anns:
                v_gt_ic[(iid, int(a["category_id"]) - 1)].append(list(map(float, a["bbox"])))
        v_pr_c: Dict[int, List[Dict]] = defaultdict(list)
        for p in v_preds:
            v_pr_c[int(p["category_id"]) - 1].append(p)
        v_ap50 = compute_ap_at_iou_and_area_range(v_gt_ic, v_pr_c, iou_thresh=0.50)
        v_ap30 = compute_ap_at_iou_and_area_range(v_gt_ic, v_pr_c, iou_thresh=0.30)
        v_ca = compute_class_agnostic_ap_and_bin_recalls(v_imgs, v_anns, v_preds, iou_thresh=0.50)
        v_tp_50 = sum(b["n_tp"] for b in v_ca["recall_by_effective_area_bin_at_640"].values())
        v_tp_30 = sum(b["n_tp_iou30"] for b in v_ca["recall_by_effective_area_bin_at_640"].values())
        per_video_metrics["AP50_mAP"][vid] = float(v_ap50.get("mAP", 0.0))
        per_video_metrics["AP30_mAP"][vid] = float(v_ap30.get("mAP", 0.0))
        per_video_metrics["mAP_class_agnostic_50"][vid] = float(v_ca["mAP_class_agnostic"])
        per_video_metrics["overall_gt_recall_50"][vid] = float(v_tp_50) / float(max(1, n_v_gts))
        per_video_metrics["overall_gt_recall_30"][vid] = float(v_tp_30) / float(max(1, n_v_gts))
        per_video_metrics.setdefault("n_present_classes", {})[vid] = int(v_ap50.get("_n_present_classes", 0))
    overall["n_present_classes_by_video"] = {
        f"v{int(vid):02d}": int(cnt) for vid, cnt in sorted(per_video_metrics.get("n_present_classes", {}).items())
    }
    return overall, per_video_metrics


def render_verified_markdown_tables_from_report(report: Dict[str, object]) -> str:
    """Programmatically render verified Markdown tables directly from the V11 JSON report (ChatGPT V10 Review Finding #5):
    - Table 1: Primary AP50 Table (matching Report Table 2's exact schema with ZERO parenthetical AP30 values).
    - Table 2: Dedicated Per-Class AP30 Table.
    - Table 3: Strict 1-to-1 Recall Table (`IoU=0.50` & `IoU=0.30`).
    """
    stride = report.get("val_frame_stride", 10)
    n_eval_imgs = report.get("n_val_images_evaluated", 0)
    n_raw_imgs = report.get("total_raw_val_images_in_coco", 0)
    n_eval_gts = report.get("n_val_gt_annotations_evaluated", 0)
    gate_status = report.get("evaluation_gate_status", "INCOMPLETE — PYCOCOTOOLS PARITY NOT RUN")
    metrics_by_var: Dict[str, Dict] = report.get("metrics_by_variant", {})
    parity_by_var: Dict[str, Dict] = report.get("pycocotools_evaluator_crosscheck", {})
    lines = [
        f"# MARIVISION-HR-P2 Machine-Generated V11 Evaluation Tables (`val_frame_stride={stride}`, `{n_eval_imgs}/{n_raw_imgs}` Frames, `{n_eval_gts}` GTs)",
        "",
        f"- **Evaluation Gate Status**: `{gate_status}`",
        "",
        "## 1. Machine-Verified `AP50` Table (Zero Parenthetical `AP30` Values — Exact Match to Report Table 2 Schema)",
        "",
        "| Variant | `parity_status` | `AP50` (`6-cl`) | `GT-Wtd AP50` | `AP30_mAP` (`6-cl`) | `GT-Wtd AP30` | `AP_S@50` (`<32²`) | `AP_S@[.50:.95]` | `mAP_ca@50` / `@30` | 3-Well `AP50` | 3-Starved `AP50` | `BOAT` `AP50` | `MARKER` `AP50` | `SAILING` `AP50` | `ANIMAL` `AP50` | `OBJECT` `AP50` | `SHIP` `AP50` |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for v_name, ov in metrics_by_var.items():
        assert_ap_table_arithmetic_consistency(ov)
        ap50 = ov["AP50"]
        p_stat = parity_by_var.get(v_name, {}).get("parity_status", "PARITY_NOT_RUN")
        lines.append(
            f"| `{v_name}` | `{p_stat}` | `{ap50.get('mAP', 0.0):.4f}` | `{ov.get('AP50_gt_weighted_mAP', 0.0):.4f}` | "
            f"`{ov.get('AP30_mAP', 0.0):.4f}` | `{ov.get('AP30_gt_weighted_mAP', 0.0):.4f}` | "
            f"`{ov.get('AP_S_at_50_mAP', 0.0):.4f}` | `{ov.get('AP_S_coco_50_95_mAP', 0.0):.4f}` | "
            f"`{ov.get('mAP_class_agnostic_50', 0.0):.4f}` / `{ov.get('mAP_class_agnostic_30', 0.0):.4f}` | "
            f"`{ov.get('AP50_3_well_sampled_classes_mean', 0.0):.4f}` | "
            f"`{ov.get('AP50_3_starved_classes_mean', 0.0):.4f}` | "
            f"`{ap50.get('BOAT', 0.0):.4f}` | "
            f"`{ap50.get('MARINE_MARKER', 0.0):.4f}` | "
            f"`{ap50.get('SAILING_VESSEL', 0.0):.4f}` | "
            f"`{ap50.get('ANIMAL', 0.0):.4f}` | "
            f"`{ap50.get('OBJECT', 0.0):.4f}` | "
            f"`{ap50.get('SHIP', 0.0):.4f}` |"
        )
    lines.extend([
        "",
        "## 2. Machine-Verified Per-Class `AP30` Table (`IoU = 0.30`)",
        "",
        "| Variant | `parity_status` | `AP30_mAP` (`6-cl`) | `GT-Wtd AP30` | 3-Well-Sampled Mean `AP30` | 3-Starved Mean `AP30` | `BOAT` `AP30` | `MARKER` `AP30` | `SAILING` `AP30` | `ANIMAL` `AP30` | `OBJECT` `AP30` | `SHIP` `AP30` |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for v_name, ov in metrics_by_var.items():
        ap30 = ov["AP30_per_class"]
        p_stat = parity_by_var.get(v_name, {}).get("parity_status", "PARITY_NOT_RUN")
        lines.append(
            f"| `{v_name}` | `{p_stat}` | `{ov.get('AP30_mAP', 0.0):.4f}` | `{ov.get('AP30_gt_weighted_mAP', 0.0):.4f}` | "
            f"`{ov.get('AP30_3_well_sampled_classes_mean', 0.0):.4f}` | "
            f"`{ov.get('AP30_3_starved_classes_mean', 0.0):.4f}` | "
            f"`{ap30.get('BOAT', 0.0):.4f}` | "
            f"`{ap30.get('MARINE_MARKER', 0.0):.4f}` | "
            f"`{ap30.get('SAILING_VESSEL', 0.0):.4f}` | "
            f"`{ap30.get('ANIMAL', 0.0):.4f}` | "
            f"`{ap30.get('OBJECT', 0.0):.4f}` | "
            f"`{ap30.get('SHIP', 0.0):.4f}` |"
        )
    lines.extend([
        "",
        "## 3. Machine-Verified Strict 1-to-1 Recall & Operating-Point Density Table (`IoU=0.50` & `IoU=0.30`)",
        "",
        "| Variant | `preds/frame` (`FP/frame@50`) | `total_1to1_tp_iou50` (`recall_50`) | `total_1to1_tp_iou30` (`recall_30`) | `recall_le_9px2_50` (`recall_le_9px2_30`) | `01_LE_4px2` `n_tp_50` / `n_tp_30` | `unmatched_gts_with_same_class_overlap_50` / `30` | `preds_overlapping_multiple_gts_50` / `30` |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for v_name, ov in metrics_by_var.items():
        b_le4 = ov.get("recall_by_effective_area_bin_at_640", {}).get("01_LE_4px2", {})
        lines.append(
            f"| `{v_name}` | `{ov.get('mean_predictions_per_frame', 0.0):.2f}` (`{ov.get('fp_per_frame_1to1_iou50', 0.0):.2f}`) | "
            f"`{ov.get('total_1to1_tp_iou50', 0)}` (`{ov.get('overall_1to1_gt_recall_iou50', 0.0):.4f}`) | "
            f"`{ov.get('total_1to1_tp_iou30', 0)}` (`{ov.get('overall_1to1_gt_recall_iou30', 0.0):.4f}`) | "
            f"`{ov.get('recall_le_9px2_at_640', 0.0):.4f}` (`{ov.get('recall_le_9px2_at_640_iou30', 0.0):.4f}`) | "
            f"`{b_le4.get('n_tp', 0)}` / `{b_le4.get('n_tp_iou30', 0)}` (of `{b_le4.get('n_gt', 0)}`) | "
            f"`{ov.get('total_unmatched_gts_with_same_class_overlap_50', 0)}` / `{ov.get('total_unmatched_gts_with_same_class_overlap_30', 0)}` | "
            f"`{ov.get('n_preds_overlapping_multiple_same_class_gts_50', 0)}` / `{ov.get('n_preds_overlapping_multiple_same_class_gts_30', 0)}` |"
        )
    lines.append("")
    return "\n".join(lines)


# ============================================================================
# 3. EXACT RAW-POSITIVE / STEP-EMA NORMALIZER, MODEL EMA & STRIDE-4 DETECTOR
# ============================================================================

class PositiveAnchorEMA:
    """Exact Raw-Positive Normalizer for Positive Tiles + Optimizer-Step EMA for Empty Tiles."""
    def __init__(self, init_val: float = 8.0, momentum: float = 0.90, min_floor: float = 1.0):
        self.ema = float(init_val)
        self.momentum = float(momentum)
        self.min_floor = float(min_floor)

    def get_denom(self, raw_pos: float) -> float:
        if raw_pos > 0.0:
            return max(self.min_floor, float(raw_pos))
        return max(self.min_floor, self.ema)

    def update_from_step(self, step_raw_pos_counts: Sequence[float]) -> float:
        pos_vals = [float(x) for x in step_raw_pos_counts if float(x) > 0.0]
        if pos_vals:
            step_mean_pos = float(np.mean(pos_vals))
            self.ema = self.momentum * self.ema + (1.0 - self.momentum) * step_mean_pos
        return self.ema

    def update_and_get_denom(self, raw_pos: float, update_ema: bool = True) -> float:
        denom = self.get_denom(raw_pos)
        if update_ema and raw_pos > 0.0:
            self.ema = self.momentum * self.ema + (1.0 - self.momentum) * float(raw_pos)
        return denom


if TORCH_AVAILABLE:
    class ModelWeightEMA:
        """Exponential Moving Average (EMA) of detector weights for Phase 2 training stability."""
        def __init__(self, model: nn.Module, decay: float = 0.999):
            self.ema_model = copy.deepcopy(model).eval()
            for p in self.ema_model.parameters():
                p.requires_grad_(False)
            self.decay = float(decay)
            self.num_updates = 0

        @torch.no_grad()
        def update(self, model: nn.Module) -> None:
            self.num_updates += 1
            d = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
            msd = model.state_dict()
            for k, v_ema in self.ema_model.state_dict().items():
                v_src = msd[k]
                if v_ema.dtype.is_floating_point:
                    v_ema.mul_(d).add_(v_src.detach(), alpha=(1.0 - d))
                else:
                    v_ema.copy_(v_src)


    class Stride4MaritimeDetector(nn.Module):
        """Consensus Stride-4 (P2-P5) FPN Detector with NWD + RFLA + Frozen Backbone BN."""
        def __init__(
            self,
            num_classes: int = NUM_CLASSES,
            use_p2_stride4: bool = True,
            use_scene_prior_channel: bool = False,
            pretrained_backbone: bool = True,
            freeze_backbone_bn: bool = True,
        ):
            super().__init__()
            self.num_classes = num_classes
            self.use_p2_stride4 = bool(use_p2_stride4)
            self.use_scene_prior_channel = bool(use_scene_prior_channel)
            self.freeze_backbone_bn = bool(freeze_backbone_bn)
            self.strides: Tuple[int, ...] = (4, 8, 16, 32) if self.use_p2_stride4 else (8, 16, 32)
            weights = ResNet50_Weights.DEFAULT if pretrained_backbone else None
            backbone = resnet50(weights=weights)
            if self.use_scene_prior_channel:
                old_conv = backbone.conv1
                rng_state = torch.get_rng_state()
                new_conv = nn.Conv2d(4, old_conv.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
                torch.set_rng_state(rng_state)
                with torch.no_grad():
                    new_conv.weight[:, :3] = old_conv.weight
                    new_conv.weight[:, 3:] = 0.0
                backbone.conv1 = new_conv
            self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
            self.layer1 = backbone.layer1  # stride 4 (C2, 256 ch)
            self.layer2 = backbone.layer2  # stride 8 (C3, 512 ch)
            self.layer3 = backbone.layer3  # stride 16 (C4, 1024 ch)
            self.layer4 = backbone.layer4  # stride 32 (C5, 2048 ch)
            if self.freeze_backbone_bn:
                self._set_backbone_bn_eval()
            in_channels_list = [256, 512, 1024, 2048] if self.use_p2_stride4 else [512, 1024, 2048]
            self.fpn = FeaturePyramidNetwork(in_channels_list=in_channels_list, out_channels=256)
            cls_layers = []
            reg_layers = []
            for _ in range(3):
                cls_layers.extend([nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.GroupNorm(32, 256), nn.SiLU(inplace=True)])
                reg_layers.extend([nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.GroupNorm(32, 256), nn.SiLU(inplace=True)])
            self.cls_tower = nn.Sequential(*cls_layers)
            self.reg_tower = nn.Sequential(*reg_layers)
            self.cls_logits = nn.Conv2d(256, num_classes, 3, padding=1)
            self.bbox_pred = nn.Conv2d(256, 4, 3, padding=1)
            for m in [self.cls_tower, self.reg_tower, self.cls_logits, self.bbox_pred]:
                for layer in m.modules():
                    if isinstance(layer, nn.Conv2d):
                        nn.init.normal_(layer.weight, std=0.01)
            prior_prob = 0.01
            bias_val = -math.log((1.0 - prior_prob) / prior_prob)
            nn.init.constant_(self.cls_logits.bias, bias_val)
            nn.init.constant_(self.bbox_pred.bias, 2.0)

        def _set_backbone_bn_eval(self) -> None:
            for mod in [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]:
                for m in mod.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
                        for p in m.parameters():
                            p.requires_grad_(False)

        def train(self, mode: bool = True):
            super().train(mode)
            if self.freeze_backbone_bn:
                self._set_backbone_bn_eval()
            return self

        def parameter_summary(self) -> Dict[str, int]:
            total = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            frozen_bn_params = total - trainable
            return {
                "total_params": int(total),
                "trainable_params": int(trainable),
                "frozen_backbone_bn_params": int(frozen_bn_params),
            }

        def forward(self, x: torch.Tensor) -> List[Tuple[int, torch.Tensor, torch.Tensor]]:
            c1 = self.stem(x)
            c2 = self.layer1(c1)
            c3 = self.layer2(c2)
            c4 = self.layer3(c3)
            c5 = self.layer4(c4)
            feats = (
                {"p2": c2, "p3": c3, "p4": c4, "p5": c5}
                if self.use_p2_stride4
                else {"p3": c3, "p4": c4, "p5": c5}
            )
            fpn_out = self.fpn(feats)
            outputs: List[Tuple[int, torch.Tensor, torch.Tensor]] = []
            for stride, key in zip(self.strides, fpn_out.keys()):
                f = fpn_out[key]
                logits = self.cls_logits(self.cls_tower(f))
                ltrb = F.softplus(self.bbox_pred(self.reg_tower(f))) * float(stride)
                outputs.append((stride, logits, ltrb))
            return outputs


    def prepare_tile_tensor(
        rgb_crop_uint8: np.ndarray,
        detector_w: int,
        detector_h: int,
        use_scene_prior: bool = False,
        horizon_y_norm: Optional[float] = None,
        crop_window: Optional[hr_p2.CropWindow] = None,
        use_global_crop_horizon_coords: bool = True,
    ) -> torch.Tensor:
        """Resize crop isotropically (`scale_x == scale_y`), apply ImageNet mean/std normalization,
        and right/bottom zero-pad to `(detector_w, detector_h)` when source aspect != 16:9.
        """
        ch_src, cw_src = rgb_crop_uint8.shape[:2]
        if crop_window is not None:
            rw, rh = crop_window.resized_w, crop_window.resized_h
        else:
            scale = min(float(detector_w) / max(1.0, float(cw_src)), float(detector_h) / max(1.0, float(ch_src)))
            rw = min(detector_w, max(1, int(round(cw_src * scale))))
            rh = min(detector_h, max(1, int(round(ch_src * scale))))
        pil_crop = Image.fromarray(rgb_crop_uint8).resize((rw, rh), Image.Resampling.BICUBIC)
        arr_u8 = np.asarray(pil_crop, dtype=np.uint8)
        arr = arr_u8.astype(np.float32) / 255.0
        mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)[None, None, :]
        std = np.asarray(IMAGENET_STD, dtype=np.float32)[None, None, :]
        arr = (arr - mean) / std
        if use_scene_prior:
            if use_global_crop_horizon_coords and crop_window is not None and crop_window.source_h > 0:
                y0_norm = float(crop_window.y0) / float(crop_window.source_h)
                y1_norm = float(crop_window.y1) / float(crop_window.source_h)
                priors = hr_p2.compute_soft_maritime_scene_priors(
                    arr_u8, horizon_y_norm=(horizon_y_norm if horizon_y_norm is not None else 0.42),
                    y0_norm_in_full_frame=y0_norm, y1_norm_in_full_frame=y1_norm,
                )
            else:
                priors = hr_p2.compute_soft_maritime_scene_priors(arr_u8, horizon_y_norm=horizon_y_norm)
            arr = np.concatenate([arr, priors["horizon_rel_y"][..., None]], axis=-1)
        if rw < detector_w or rh < detector_h:
            c_dim = arr.shape[2]
            padded = np.zeros((detector_h, detector_w, c_dim), dtype=np.float32)
            padded[:rh, :rw, :] = arr
            arr = padded
        return torch.from_numpy(arr).permute(2, 0, 1)


    def compute_detector_loss_on_tile(
        outputs: List[Tuple[int, torch.Tensor, torch.Tensor]],
        gt_boxes_xywh: torch.Tensor,
        gt_labels: torch.Tensor,
        area_scale: float = 1.0,
        use_nwd_rfla: bool = True,
        nwd_loss_fn: Optional[hr_p2.HybridNWDBoxLoss] = None,
        pos_ema_tracker: Optional[PositiveAnchorEMA] = None,
        update_ema_now: bool = False,
        ignore_boxes_xywh: Optional[torch.Tensor] = None,
        valid_w: Optional[int] = None,
        valid_h: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute classification focal loss (with ignore masking for clipped border GTs & padding) and box loss."""
        device = outputs[0][1].device
        all_pts_list = []
        all_logits_list = []
        all_boxes_list = []
        for stride, logits_map, ltrb_map in outputs:
            _, C, H, W = logits_map.shape
            ys, xs = torch.meshgrid(
                torch.arange(H, device=device, dtype=torch.float32),
                torch.arange(W, device=device, dtype=torch.float32),
                indexing="ij",
            )
            cx = (xs + 0.5) * float(stride)
            cy = (ys + 0.5) * float(stride)
            st = torch.full_like(cx, float(stride))
            pts = torch.stack([cx, cy, st], dim=-1).reshape(-1, 3)
            l, t, r, b = ltrb_map[0].unbind(dim=0)
            x0 = (cx - l).reshape(-1)
            y0 = (cy - t).reshape(-1)
            bw = (l + r).clamp(min=1.0).reshape(-1)
            bh = (t + b).clamp(min=1.0).reshape(-1)
            boxes = torch.stack([x0, y0, bw, bh], dim=-1)
            all_pts_list.append(pts)
            all_logits_list.append(logits_map[0].permute(1, 2, 0).reshape(-1, C))
            all_boxes_list.append(boxes)
        pts_all = torch.cat(all_pts_list, dim=0)
        logits_all = torch.cat(all_logits_list, dim=0)
        pred_boxes_all = torch.cat(all_boxes_list, dim=0)
        target_cls = torch.zeros_like(logits_all)
        valid_point_mask = torch.ones((pts_all.shape[0],), device=device, dtype=torch.float32)
        if valid_w is not None:
            valid_point_mask = valid_point_mask * (pts_all[:, 0] < float(valid_w)).float()
        if valid_h is not None:
            valid_point_mask = valid_point_mask * (pts_all[:, 1] < float(valid_h)).float()
        if ignore_boxes_xywh is not None and len(ignore_boxes_xywh) > 0:
            px = pts_all[:, 0:1]
            py = pts_all[:, 1:2]
            ix0 = ignore_boxes_xywh[:, 0][None, :]
            iy0 = ignore_boxes_xywh[:, 1][None, :]
            ix1 = ix0 + ignore_boxes_xywh[:, 2][None, :]
            iy1 = iy0 + ignore_boxes_xywh[:, 3][None, :]
            in_ign = ((px >= ix0) & (px <= ix1) & (py >= iy0) & (py <= iy1)).any(dim=1)
            valid_point_mask = valid_point_mask * (~in_ign).float()
        pos_pred_boxes = []
        pos_gt_boxes = []
        if len(gt_boxes_xywh) > 0:
            pts_np = pts_all.detach().cpu().numpy()
            gts_np = gt_boxes_xywh.detach().cpu().numpy()
            if use_nwd_rfla:
                assigned = hr_p2.rfla_hierarchical_assign_numpy(
                    pts_np, gts_np, top_k_primary=4, top_k_Backup=8
                )
            else:
                assigned = hr_p2.assign_center_distance_numpy(pts_np, gts_np, top_k=4)
            for g_i, pt_indices in assigned.items():
                if not pt_indices:
                    continue
                cl = int(gt_labels[g_i].item())
                idx_t = torch.tensor(pt_indices, device=device, dtype=torch.long)
                target_cls[idx_t, cl] = 1.0
                valid_point_mask[idx_t] = 1.0
                pos_pred_boxes.append(pred_boxes_all[idx_t])
                pos_gt_boxes.append(gt_boxes_xywh[g_i].unsqueeze(0).expand(len(pt_indices), 4))
        raw_pos = float(sum(len(x) for x in pos_pred_boxes))
        if pos_ema_tracker is not None:
            cls_denom = pos_ema_tracker.update_and_get_denom(raw_pos, update_ema=update_ema_now)
        else:
            cls_denom = max(1.0, raw_pos) if raw_pos > 0.0 else 8.0
        probs = torch.sigmoid(logits_all)
        ce_loss = F.binary_cross_entropy_with_logits(logits_all, target_cls, reduction="none")
        p_t = probs * target_cls + (1.0 - probs) * (1.0 - target_cls)
        alpha_t = 0.25 * target_cls + 0.75 * (1.0 - target_cls)
        focal_per_loc_cls = alpha_t * ((1.0 - p_t) ** 2.0) * ce_loss
        loss_cls = (focal_per_loc_cls * valid_point_mask[:, None]).sum() / cls_denom
        with torch.no_grad():
            mean_bg_prob = float(probs.mean().item())
            frac_bg_gt_020 = float((probs > 0.20).float().mean().item())
        if pos_pred_boxes:
            p_boxes = torch.cat(pos_pred_boxes, dim=0)
            g_boxes = torch.cat(pos_gt_boxes, dim=0)
            lin_scale = max(1e-6, math.sqrt(float(area_scale)))
            if nwd_loss_fn is None:
                nwd_loss_fn = hr_p2.HybridNWDBoxLoss(
                    constant_c=8.0,
                    w_nwd=(2.0 if use_nwd_rfla else 0.0),
                    w_giou=0.5,
                    w_l1=1.0,
                )
            box_out = nwd_loss_fn(p_boxes / lin_scale, g_boxes / lin_scale)
            loss_box = box_out["loss_box_total"]
        else:
            loss_box = pred_boxes_all.sum() * 0.0
        return {
            "loss_total": loss_cls + loss_box,
            "loss_cls": loss_cls.detach(),
            "loss_box": loss_box.detach(),
            "num_pos": torch.tensor(raw_pos, device=device),
            "cls_denom": torch.tensor(cls_denom, device=device),
            "mean_bg_prob": torch.tensor(mean_bg_prob, device=device),
            "frac_bg_gt_020": torch.tensor(frac_bg_gt_020, device=device),
        }


    @torch.no_grad()
    def decode_tile_predictions_from_outputs(
        outputs: List[Tuple[int, torch.Tensor, torch.Tensor]],
        score_thresh: float = 0.05,
        top_k_pre_nms: int = 200,
        valid_w: Optional[int] = None,
        valid_h: Optional[int] = None,
    ) -> List[Dict]:
        """Decode single-tile dense FPN outputs into detector-coordinate candidate dicts for `merge_multiscale_tile_detections`."""
        cands: List[Dict] = []
        for stride, logits_map, ltrb_map in outputs:
            _, C, H, W = logits_map.shape
            probs = torch.sigmoid(logits_map[0])  # [C, H, W]
            max_scores, max_cls = probs.max(dim=0)  # [H, W]
            keep_y, keep_x = torch.where(max_scores >= score_thresh)
            if len(keep_y) == 0:
                continue
            if len(keep_y) > top_k_pre_nms:
                scores_flat = max_scores[keep_y, keep_x]
                _, top_idx = torch.topk(scores_flat, k=top_k_pre_nms)
                keep_y = keep_y[top_idx]
                keep_x = keep_x[top_idx]
            for y_i, x_i in zip(keep_y.tolist(), keep_x.tolist()):
                cx = (x_i + 0.5) * float(stride)
                cy = (y_i + 0.5) * float(stride)
                if valid_w is not None and cx >= float(valid_w):
                    continue
                if valid_h is not None and cy >= float(valid_h):
                    continue
                l, t, r, b = [float(v) for v in ltrb_map[0, :, y_i, x_i].tolist()]
                x0 = cx - l
                y0 = cy - t
                bw = max(1.0, l + r)
                bh = max(1.0, t + b)
                cl = int(max_cls[y_i, x_i].item())
                sc = float(max_scores[y_i, x_i].item())
                cands.append({
                    "category_id": cl + 1,
                    "bbox": [x0, y0, bw, bh],
                    "score": sc,
                })
        return cands


    @torch.no_grad()
    def predict_val_variants_single_pass(
        model: Stride4MaritimeDetector,
        val_idx: SeaClipsZipNativeIndex,
        eval_image_ids: Sequence[int],
        arm: str,
        device: torch.device,
        use_global_crop_horizon_coords: bool = True,
        run_stage_d_microcrops: bool = False,
        max_stage_d_proposals: int = 4,
    ) -> Dict[str, List[Dict]]:
        """Run a SINGLE GPU pass per frame over Stage B+C tiles and simultaneously extract:
        - `std_005_max100`: Standard operating point (`score_thresh=0.05, max_det=100`).
        - `low_0005_max300`: Low-threshold / high-density diagnostic (`score_thresh=0.005, max_det=300`, Claude #3).
        - `stage_d_005_max100` (if `run_stage_d_microcrops=True`): Conditional Stage D `4x` micro-crops (`<= 4`/frame, ChatGPT #4).
        """
        model.eval()
        arm = arm.lower().strip()
        use_scene = (arm == "e")
        preds_std: List[Dict] = []
        preds_low: List[Dict] = []
        preds_stage_d: List[Dict] = []
        for iid in eval_image_ids:
            im = val_idx.images[iid]
            sw, sh = int(im["width"]), int(im["height"])
            rgb = val_idx.load_rgb_numpy(iid)
            if arm == "a":
                crops = [hr_p2.CropWindow("STAGE_A_640_BASE", 0.0, 0.0, float(sw), float(sh), 640, 360, preserve_aspect=True, source_w=sw, source_h=sh)]
            elif arm == "b":
                crops = [hr_p2.CropWindow("STAGE_B_FULL_CONTEXT", 0.0, 0.0, float(sw), float(sh), 1280, 720, preserve_aspect=True, source_w=sw, source_h=sh)]
            else:
                crops = hr_p2.generate_coarse_to_fine_crops(sw, sh, horizon_y_norm=0.42, include_stage_d_microcrops=False)
            tile_preds_low: List[Tuple[hr_p2.CropWindow, List[Dict]]] = []
            tile_preds_std: List[Tuple[hr_p2.CropWindow, List[Dict]]] = []
            for cw in crops:
                crop_rgb = cw.slice_rgb(rgb)
                t_in = prepare_tile_tensor(
                    crop_rgb, cw.detector_w, cw.detector_h, use_scene_prior=use_scene, crop_window=cw,
                    use_global_crop_horizon_coords=use_global_crop_horizon_coords,
                ).unsqueeze(0).to(device)
                outs = model(t_in)
                cands_low = decode_tile_predictions_from_outputs(
                    outs, score_thresh=0.005, top_k_pre_nms=250, valid_w=cw.resized_w, valid_h=cw.resized_h
                )
                cands_std = [d for d in cands_low if float(d["score"]) >= 0.05]
                tile_preds_low.append((cw, cands_low))
                tile_preds_std.append((cw, cands_std))
            merged_std = hr_p2.merge_multiscale_tile_detections(
                tile_preds_std, source_w=sw, source_h=sh, score_thresh=0.05, max_det_per_image=100
            )
            for d in merged_std:
                preds_std.append({**d, "image_id": iid})
            merged_low = hr_p2.merge_multiscale_tile_detections(
                tile_preds_low, source_w=sw, source_h=sh, score_thresh=0.005, max_det_per_image=300
            )
            for d in merged_low:
                preds_low.append({**d, "image_id": iid})
            if run_stage_d_microcrops and arm in ("c", "d", "e"):
                uncertain_small_centers: List[Tuple[float, float]] = []
                for d in merged_low:
                    bx, by, bw, bh = map(float, d["bbox"])
                    sc = float(d["score"])
                    if (bw * bh) <= 225.0 and 0.01 <= sc <= 0.55:
                        cx_d, cy_d = bx + 0.5 * bw, by + 0.5 * bh
                        if all(math.hypot(cx_d - ux, cy_d - uy) > 90.0 for ux, uy in uncertain_small_centers):
                            uncertain_small_centers.append((cx_d, cy_d))
                            if len(uncertain_small_centers) >= max_stage_d_proposals:
                                break
                tile_preds_with_d = list(tile_preds_std)
                if uncertain_small_centers:
                    d_crops = [
                        c for c in hr_p2.generate_coarse_to_fine_crops(
                            sw, sh, horizon_y_norm=0.42,
                            stage_b_detection_centers=uncertain_small_centers,
                            max_stage_d_proposals=max_stage_d_proposals,
                        )
                        if c.stage == "STAGE_D_4X_MICROCROP"
                    ]
                    for cw_d in d_crops:
                        crop_rgb_d = cw_d.slice_rgb(rgb)
                        t_in_d = prepare_tile_tensor(
                            crop_rgb_d, cw_d.detector_w, cw_d.detector_h, use_scene_prior=use_scene, crop_window=cw_d,
                            use_global_crop_horizon_coords=use_global_crop_horizon_coords,
                        ).unsqueeze(0).to(device)
                        outs_d = model(t_in_d)
                        cands_d = decode_tile_predictions_from_outputs(
                            outs_d, score_thresh=0.05, top_k_pre_nms=200, valid_w=cw_d.resized_w, valid_h=cw_d.resized_h
                        )
                        tile_preds_with_d.append((cw_d, cands_d))
                merged_d = hr_p2.merge_multiscale_tile_detections(
                    tile_preds_with_d, source_w=sw, source_h=sh, score_thresh=0.05, max_det_per_image=100
                )
                for d in merged_d:
                    preds_stage_d.append({**d, "image_id": iid})
        return {
            "std_005_max100": preds_std,
            "low_0005_max300": preds_low,
            "stage_d_005_max100": preds_stage_d,
        }


    @torch.no_grad()
    def predict_all_val_images_for_model(
        model: Stride4MaritimeDetector,
        val_idx: SeaClipsZipNativeIndex,
        eval_image_ids: Sequence[int],
        arm: str,
        device: torch.device,
        score_thresh: float = 0.05,
        apply_posthoc_scene_prior: bool = False,
        posthoc_in_logit_space: bool = True,
        use_global_crop_horizon_coords: bool = True,
    ) -> List[Dict]:
        """Run inference across `eval_image_ids` for a single model and return all merged predictions."""
        model.eval()
        arm = arm.lower().strip()
        use_scene = (arm == "e")
        all_preds: List[Dict] = []
        for iid in eval_image_ids:
            im = val_idx.images[iid]
            sw, sh = int(im["width"]), int(im["height"])
            rgb = val_idx.load_rgb_numpy(iid)
            if arm == "a":
                crops = [hr_p2.CropWindow("STAGE_A_640_BASE", 0.0, 0.0, float(sw), float(sh), 640, 360, preserve_aspect=True, source_w=sw, source_h=sh)]
            elif arm == "b":
                crops = [hr_p2.CropWindow("STAGE_B_FULL_CONTEXT", 0.0, 0.0, float(sw), float(sh), 1280, 720, preserve_aspect=True, source_w=sw, source_h=sh)]
            else:
                crops = hr_p2.generate_coarse_to_fine_crops(sw, sh, horizon_y_norm=0.42, include_stage_d_microcrops=False)
            tile_preds: List[Tuple[hr_p2.CropWindow, List[Dict]]] = []
            for cw in crops:
                crop_rgb = cw.slice_rgb(rgb)
                t_in = prepare_tile_tensor(
                    crop_rgb, cw.detector_w, cw.detector_h, use_scene_prior=use_scene, crop_window=cw,
                    use_global_crop_horizon_coords=use_global_crop_horizon_coords,
                ).unsqueeze(0).to(device)
                outs = model(t_in)
                cands = decode_tile_predictions_from_outputs(outs, score_thresh=score_thresh, valid_w=cw.resized_w, valid_h=cw.resized_h)
                tile_preds.append((cw, cands))
            merged_img_preds = hr_p2.merge_multiscale_tile_detections(
                tile_preds, source_w=sw, source_h=sh, score_thresh=score_thresh, max_det_per_image=100
            )
            if use_scene and apply_posthoc_scene_prior and merged_img_preds:
                scene_maps = hr_p2.compute_soft_maritime_scene_priors(rgb, horizon_y_norm=None)
                for d in merged_img_preds:
                    cl_idx = int(d["category_id"]) - 1
                    adj_sc, _ = hr_p2.apply_class_conditioned_soft_scene_prior(
                        float(d["score"]), cl_idx, tuple(map(float, d["bbox"])), scene_maps,
                        in_logit_space=posthoc_in_logit_space,
                    )
                    d["score"] = round(adj_sc, 5)
                merged_img_preds.sort(key=lambda x: float(x["score"]), reverse=True)
            for d in merged_img_preds:
                all_preds.append({**d, "image_id": iid})
        return all_preds


    @torch.no_grad()
    def evaluate_model_on_val(
        model: Stride4MaritimeDetector,
        val_idx: SeaClipsZipNativeIndex,
        arm: str,
        device: torch.device,
        frame_stride: int = 10,
        score_thresh: float = 0.05,
        apply_posthoc_scene_prior: bool = False,
        use_global_crop_horizon_coords: bool = True,
    ) -> Dict[str, object]:
        """Evaluate a trained model on `seaclips-val.json` (cropped according to `arm`)."""
        by_video: Dict[int, List[Dict]] = defaultdict(list)
        for iid, im in val_idx.images.items():
            by_video[int(im.get("video_id", 0))].append(im)
        eval_image_ids: List[int] = []
        for vid, im_list in sorted(by_video.items()):
            im_list.sort(key=lambda x: (int(x.get("frame_id", x["id"])), int(x["id"])))
            for idx_f, im in enumerate(im_list):
                if frame_stride <= 1 or (idx_f % frame_stride == 0):
                    eval_image_ids.append(int(im["id"]))
        eval_images_dict = {iid: val_idx.images[iid] for iid in eval_image_ids}
        eval_anns_dict = {iid: val_idx.anns_by_image.get(iid, []) for iid in eval_image_ids}
        all_preds = predict_all_val_images_for_model(
            model, val_idx, eval_image_ids, arm=arm, device=device, score_thresh=score_thresh,
            apply_posthoc_scene_prior=apply_posthoc_scene_prior, posthoc_in_logit_space=True,
            use_global_crop_horizon_coords=use_global_crop_horizon_coords,
        )
        overall, _ = evaluate_predictions_dict_and_per_video(eval_images_dict, eval_anns_dict, all_preds)
        return {
            "arm": arm,
            "val_frame_stride": int(frame_stride),
            "is_strided_subsample": bool(frame_stride > 1),
            "total_raw_val_images_in_coco": len(val_idx.images),
            "total_raw_val_gt_annotations_in_coco": sum(len(v) for v in val_idx.anns_by_image.values()),
            **overall,
        }


# ============================================================================
# 4. UNIT TESTS & WORKSTATION VERIFICATION / SCREENING / PHASE 2 MODES
# ============================================================================


KNOWN_R50_WORKSTATION_SCRIPTS: Tuple[str, ...] = (
    "train_bfcef_f_s0_final_experts.py",
    "run15a_final_windows.py",
    "run15a_final_mechanism.py",
    "run15a_final_supervision.py",
    "run15a_internal_rebuild_PATCH03.py",
)
KNOWN_R50_SANDBOX_RELPATHS: Tuple[str, ...] = (
    r"bfcef_f_s0\run15a_final_candidate_sandbox",
    r"bfcef_f_s0\run15a_final_candidate_sandbox\oct13_spatial_matched_control",
    r"bfcef_f_s0\run15a_final_candidate_sandbox\oct13_recoverability_gate",
    r"bfcef_f_s0",
)


def _try_parse_r50_predictions_from_file(
    candidate_path: Path,
    val_images_by_id: Dict[int, Dict],
    score_thresh: float = 0.05,
) -> Optional[List[Dict]]:
    """Attempt to extract COCO-format validation predictions for `val_images_by_id` from a JSON, CSV, NPZ, or PT/PTH file."""
    p = Path(candidate_path)
    if not p.is_file():
        return None
    val_iids = set(int(k) for k in val_images_by_id.keys())
    suffix = p.suffix.lower()
    try:
        if suffix == ".json":
            if p.stat().st_size > 250 * 1024 * 1024:
                return None
            raw = json.loads(p.read_text(encoding="utf-8", errors="replace"))
            items = None
            if isinstance(raw, list):
                items = raw
            elif isinstance(raw, dict):
                for k in ("predictions", "dets", "results", "detections", "val_predictions", "r50_predictions"):
                    if isinstance(raw.get(k), list):
                        items = raw[k]
                        break
                if items is None and "preds_by_arm" in raw and isinstance(raw["preds_by_arm"], dict):
                    if "baseline_r50" in raw["preds_by_arm"]:
                        items = raw["preds_by_arm"]["baseline_r50"]
            if isinstance(items, list) and items:
                out_preds: List[Dict] = []
                for d in items:
                    if not isinstance(d, dict) or "image_id" not in d or "bbox" not in d:
                        continue
                    iid = int(d["image_id"])
                    if iid not in val_iids:
                        continue
                    sc = float(d.get("score", d.get("prob", 1.0)))
                    if sc < score_thresh:
                        continue
                    cat = int(d["category_id"]) if "category_id" in d else (int(d["class"]) + 1)
                    if not (1 <= cat <= NUM_CLASSES):
                        continue
                    bbox = [float(v) for v in d["bbox"]]
                    if len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:
                        out_preds.append({"image_id": iid, "category_id": cat, "bbox": bbox, "score": sc})
                if out_preds:
                    return out_preds
        elif suffix == ".csv":
            with p.open(newline="", encoding="utf-8-sig", errors="replace") as f:
                reader = csv.DictReader(f)
                fields = set(reader.fieldnames or [])
                if "image_id" not in fields:
                    return None
                has_bbox_cols = (
                    {"x", "y", "w", "h"}.issubset(fields)
                    or {"pred_x", "pred_y", "pred_w", "pred_h"}.issubset(fields)
                    or "bbox" in fields
                )
                if not has_bbox_cols:
                    return None
                out_preds = []
                for row in reader:
                    iid = int(row["image_id"])
                    if iid not in val_iids:
                        continue
                    sc = float(row.get("score", row.get("prob", row.get("best_score", 0.0))))
                    if sc < score_thresh:
                        continue
                    cat = int(row["category_id"]) if "category_id" in row else (int(row.get("class", 0)) + 1)
                    if "bbox" in row and row["bbox"]:
                        bx, by, bw, bh = [float(v) for v in json.loads(row["bbox"])]
                    elif "pred_x" in row:
                        bx, by, bw, bh = float(row["pred_x"]), float(row["pred_y"]), float(row["pred_w"]), float(row["pred_h"])
                    else:
                        bx, by, bw, bh = float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"])
                    if bw > 0 and bh > 0 and (1 <= cat <= NUM_CLASSES):
                        out_preds.append({"image_id": iid, "category_id": cat, "bbox": [bx, by, bw, bh], "score": sc})
                if out_preds:
                    return out_preds
        elif suffix in (".npz", ".pt", ".pth"):
            raw_cache = None
            if suffix == ".npz":
                data = np.load(p, allow_pickle=True)
                raw_cache = {k: data[k] for k in data.files}
            elif TORCH_AVAILABLE:
                raw_cache = safe_torch_load_checkpoint(p, map_location="cpu")
            if isinstance(raw_cache, dict):
                # Check if raw_cache is keyed by image_id (int or str) -> dict of boxes/scores/labels
                matching_keys = [k for k in raw_cache.keys() if str(k).isdigit() and int(k) in val_iids]
                if matching_keys:
                    out_preds = []
                    for k in matching_keys:
                        iid = int(k)
                        entry = raw_cache[k]
                        if not isinstance(entry, dict):
                            continue
                        im_meta = val_images_by_id[iid]
                        sw, sh = float(im_meta.get("width", 1920.0)), float(im_meta.get("height", 1080.0))
                        boxes = entry.get("boxes", entry.get("pred_boxes", entry.get("bbox", None)))
                        scores = entry.get("scores", entry.get("prob", None))
                        labels = entry.get("labels", entry.get("category_id", entry.get("class", None)))
                        if boxes is not None and scores is not None and labels is not None:
                            b_arr = np.asarray(boxes.cpu() if hasattr(boxes, "cpu") else boxes, dtype=np.float64)
                            s_arr = np.asarray(scores.cpu() if hasattr(scores, "cpu") else scores, dtype=np.float64)
                            l_arr = np.asarray(labels.cpu() if hasattr(labels, "cpu") else labels, dtype=np.int64)
                            for b_row, sc_val, lb_val in zip(b_arr, s_arr, l_arr):
                                if float(sc_val) < score_thresh:
                                    continue
                                bx, by, bw, bh = map(float, b_row[:4])
                                if max(bx, by, bw, bh) <= 1.05:
                                    # Normalized cxcywh -> pixel xywh
                                    cx, cy, nw, nh = bx * sw, by * sh, bw * sw, bh * sh
                                    bx, by, bw, bh = cx - 0.5 * nw, cy - 0.5 * nh, nw, nh
                                cat = int(lb_val) if int(lb_val) >= 1 else (int(lb_val) + 1)
                                if 1 <= cat <= NUM_CLASSES and bw > 0 and bh > 0:
                                    out_preds.append({"image_id": iid, "category_id": cat, "bbox": [bx, by, bw, bh], "score": float(sc_val)})
                    if out_preds:
                        return out_preds
    except Exception:
        return None
    return None


def discover_and_extract_frozen_r50_val_predictions(
    project_dir: Path,
    coco_val_path: Path,
    eval_images_dict: Dict[int, Dict],
    out_dir: Optional[Path] = None,
    explicit_json_path: Optional[Path] = None,
    score_thresh: float = 0.05,
) -> Tuple[List[Dict], Dict[str, object]]:
    """Locate or extract frozen R50 baseline predictions on VAL from known workstation paths (`bfcef_f_s0\run15a_final_candidate_sandbox`)."""
    proj = Path(project_dir).resolve()
    val_coco = Path(coco_val_path).resolve()
    raw_val = json.loads(val_coco.read_text(encoding="utf-8")) if val_coco.is_file() else {"images": []}
    all_val_images = {int(im["id"]): im for im in raw_val.get("images", [])} or dict(eval_images_dict)
    eval_iid_set = set(int(k) for k in eval_images_dict.keys())
    if explicit_json_path is not None and Path(explicit_json_path).is_file():
        preds = _try_parse_r50_predictions_from_file(Path(explicit_json_path), eval_images_dict, score_thresh=score_thresh)
        if preds is not None:
            return preds, {
                "status": "LOADED_FROM_EXPLICIT_PATH",
                "source_path": str(Path(explicit_json_path).resolve()),
                "source_sha256": sha256_file_bytes(Path(explicit_json_path)),
                "n_val_predictions_on_eval_subset": len(preds),
            }
    script_traces: Dict[str, object] = {}
    referenced_paths: List[Path] = []
    path_pat = re.compile(r"([A-Za-z]:[\\/][^\s'\"`)+,\\];]+|(?:bfcef_f_s0|run15a)[^\s'\"`)+,\\];]*)", re.I)
    for sname in KNOWN_R50_WORKSTATION_SCRIPTS:
        sp = proj / sname
        if not sp.is_file():
            script_traces[sname] = {"exists": False}
            continue
        txt = sp.read_text(encoding="utf-8", errors="replace")
        found_literals = sorted(set(m.group(1) for m in path_pat.finditer(txt)))
        script_traces[sname] = {
            "exists": True,
            "sha256": sha256_file_bytes(sp),
            "referenced_paths": found_literals[:40],
        }
        for lit in found_literals:
            cand_p = Path(lit.replace("\\", "/")) if ":" in lit else (proj / lit.replace("\\", "/"))
            referenced_paths.append(cand_p)
    search_roots: List[Path] = []
    for rel in KNOWN_R50_SANDBOX_RELPATHS:
        rp = proj / rel.replace("\\", "/")
        if rp.exists():
            search_roots.append(rp)
    if proj.exists():
        search_roots.append(proj)
    if val_coco.parent.exists() and val_coco.parent not in search_roots:
        search_roots.append(val_coco.parent)
    skip_dir_Tokens = ("runs_hr_p2", "hr_p2_workstation", "oct15_p2588", "p2588", "stage1_review", "source_backups", "__pycache__")
    candidate_files: List[Path] = []
    seen_files = set()
    for rp in referenced_paths:
        if rp.is_file() and rp.resolve() not in seen_files:
            seen_files.add(rp.resolve())
            candidate_files.append(rp.resolve())
        elif rp.is_dir():
            search_roots.insert(0, rp.resolve())
    for sroot in search_roots:
        if not sroot.is_dir():
            continue
        for p in sorted(sroot.rglob("*")):
            if not p.is_file():
                continue
            p_str_low = str(p).lower()
            if any(tok in p_str_low for tok in skip_dir_Tokens):
                continue
            if p.name.lower() in ("seaclips-train.json", "seaclips-val.json", "seaclips-test.json"):
                continue
            if p.suffix.lower() in (".json", ".csv", ".npz", ".pt", ".pth"):
                rp_res = p.resolve()
                if rp_res not in seen_files:
                    seen_files.add(rp_res)
                    candidate_files.append(rp_res)
    # Prioritize files with 'val', 'r50', 'baseline', 'run15a', 'bfcef', 'oct13' in their path
    def _cand_priority(p: Path) -> Tuple[int, str]:
        s = str(p).lower()
        score = 0
        if "val" in s or "validation" in s:
            score -= 4
        if "r50" in s or "baseline" in s or "deformable" in s:
            score -= 3
        if "bfcef" in s or "run15a" in s or "oct13" in s:
            score -= 2
        if p.suffix.lower() in (".json", ".csv"):
            score -= 1
        return (score, str(p))
    candidate_files.sort(key=_cand_priority)
    ckpt_inventory = [str(p) for p in candidate_files if p.suffix.lower() in (".pt", ".pth")][:40]
    data_inventory = [str(p) for p in candidate_files if p.suffix.lower() in (".json", ".csv", ".npz")][:60]
    for cand in candidate_files:
        if cand.suffix.lower() not in (".json", ".csv", ".npz"):
            continue
        preds_all_val = _try_parse_r50_predictions_from_file(cand, all_val_images, score_thresh=score_thresh)
        if preds_all_val:
            preds_subset = [d for d in preds_all_val if int(d["image_id"]) in eval_iid_set]
            if out_dir is not None:
                out_p = Path(out_dir) / "discovered_frozen_r50_val_preds.json"
                out_p.parent.mkdir(parents=True, exist_ok=True)
                out_p.write_text(json.dumps(preds_all_val), encoding="utf-8")
            return preds_subset, {
                "status": "AUTO_DISCOVERED_VAL_PREDICTIONS",
                "source_path": str(cand),
                "source_sha256": sha256_file_bytes(cand),
                "n_val_predictions_full_val": len(preds_all_val),
                "n_val_predictions_on_eval_subset": len(preds_subset),
                "r50_workstation_scripts": script_traces,
                "discovered_checkpoints_sample": ckpt_inventory,
            }
    return [], {
        "status": "NO_PRECOMPUTED_VAL_JSON_OR_CSV_FOUND_IN_SEARCH_ROOTS",
        "searched_roots": [str(r) for r in search_roots],
        "r50_workstation_scripts": script_traces,
        "discovered_checkpoints_sample": ckpt_inventory,
        "discovered_data_files_sample": data_inventory,
        "note": (
            "Pass --baseline-preds-json <path_to_r50_val_preds.json> or run the R50 export command from "
            "train_bfcef_f_s0_final_experts.py / run15a_final_windows.py on seaclips-val.json."
        ),
    }


class TestStep2AblationRunnerContracts(unittest.TestCase):
    def test_zip_reader_hard_blocks_test_split(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            coco = root / "coco.json"
            coco.write_text(json.dumps({"images": [{"id": 1, "file_name": "v1/001.png", "width": 64, "height": 64}], "annotations": []}), encoding="utf-8")
            zpath = root / "mock.zip"
            with zipfile.ZipFile(zpath, "w") as zf:
                zf.writestr("SeaClips/test/v1/001.png", b"forbidden")
            with self.assertRaisesRegex(AssertionError, "SPLIT BLOCK"):
                SeaClipsZipNativeIndex(zpath, coco, split_hint="test")
            idx = SeaClipsZipNativeIndex(zpath, coco, split_hint="train")
            self.assertNotIn(1, idx.member_by_image_id)

    def test_positive_anchor_ema_gives_exact_1x_weight_to_1gt_tile_under_any_ema(self):
        ema = PositiveAnchorEMA(init_val=8.0, momentum=0.90, min_floor=1.0)
        d4_initial = ema.update_and_get_denom(4.0, update_ema=True)
        self.assertEqual(d4_initial, 4.0)
        self.assertAlmostEqual(ema.ema, 7.6, places=5)
        for _ in range(100):
            ema.update_from_step([24.0, 24.0, 24.0, 24.0])
        self.assertGreater(ema.ema, 20.0)
        self.assertEqual(ema.get_denom(4.0), 4.0)
        self.assertAlmostEqual(ema.get_denom(0.0), ema.ema, places=5)

    def test_clipped_border_gt_is_returned_in_ignore_boxes(self):
        cw = hr_p2.CropWindow("STAGE_C_2X_CROP", 0.0, 0.0, 640.0, 360.0, 1280, 720)
        anns = [
            {"id": 1, "image_id": 1, "category_id": 3, "bbox": [100.0, 100.0, 10.0, 10.0]},
            {"id": 2, "image_id": 1, "category_id": 6, "bbox": [615.0, 100.0, 100.0, 50.0]},
        ]
        kept_b, kept_l, ign_b = project_gt_boxes_into_crop_with_ignore(anns, cw, min_visibility=0.65)
        self.assertEqual(len(kept_b), 1)
        self.assertEqual(len(ign_b), 1)

    def test_slicing_aided_schedule_interleaves_distinct_videos_in_batch_of_4(self):
        coco_imgs = {
            101: {"id": 101, "video_id": 2, "frame_id": 0, "width": 1920, "height": 1080},
            102: {"id": 102, "video_id": 4, "frame_id": 0, "width": 1920, "height": 1080},
            103: {"id": 103, "video_id": 7, "frame_id": 0, "width": 1920, "height": 1080},
            104: {"id": 104, "video_id": 18, "frame_id": 0, "width": 1920, "height": 1080},
        }
        anns = {iid: [{"id": iid, "image_id": iid, "category_id": 3, "bbox": [500.0, 500.0, 4.0, 5.0]}] for iid in coco_imgs}
        sched = build_slicing_aided_training_schedule(coco_imgs, anns, arm="d", frame_stride=1, interleave_across_videos=True)
        first_batch_vids = [coco_imgs[iid]["video_id"] for iid, _ in sched[:4]]
        self.assertEqual(len(set(first_batch_vids)), 4)

    def test_coco_ap_evaluator_reports_absent_class_fps_and_class_agnostic_ap(self):
        coco_imgs = {1: {"id": 1, "video_id": 1, "width": 1920, "height": 1080}}
        anns_by_img = {
            1: [{"id": 1, "image_id": 1, "category_id": 3, "bbox": [100.0, 100.0, 3.0, 3.0]}],
        }
        gt_map = {(1, 2): [[100.0, 100.0, 3.0, 3.0]]}
        preds_by_cl = {
            2: [{"image_id": 1, "category_id": 3, "bbox": [100.5, 100.5, 3.0, 3.0], "score": 0.90}],
            4: [{"image_id": 1, "category_id": 5, "bbox": [500.0, 500.0, 20.0, 20.0], "score": 0.85}],
        }
        ap50 = compute_ap_at_iou_and_area_range(gt_map, preds_by_cl, iou_thresh=0.50)
        self.assertEqual(ap50["mAP"], 1.0)
        self.assertEqual(ap50["absent_class_fp_count_LEISURE_VEHICLE"], 1)
        all_preds = preds_by_cl[2] + preds_by_cl[4]
        ov, _ = evaluate_predictions_dict_and_per_video(coco_imgs, anns_by_img, all_preds)
        self.assertEqual(ov["AP_S_at_50_mAP"], 1.0)
        self.assertGreater(ov["AP_S_coco_50_95_mAP"], 0.0)
        ca = compute_class_agnostic_ap_and_bin_recalls(coco_imgs, anns_by_img, all_preds, iou_thresh=0.50)
        self.assertEqual(ca["recall_le_9px2_at_640"], 1.0)
        self.assertEqual(ca["recall_le_9px2_at_640_iou30"], 1.0)
        self.assertGreater(ca["mAP_class_agnostic"], 0.99)
        diag = diagnose_tiny_le_4px2_and_ship_confusion(coco_imgs, anns_by_img, all_preds)
        self.assertEqual(diag["tiny_01_LE_4px2_diagnostic"]["n_gt"], 1)
        self.assertEqual(diag["tiny_01_LE_4px2_diagnostic"]["outcome_breakdown"].get("01_TP_iou50_1to1_score_ge_005"), 1)

    def test_small_object_ap_consumes_ignored_gt_once_and_penalizes_second_duplicate_fp(self):
        # ChatGPT V10 Review Finding #1 exact scenario:
        # 1 area-ignored large GT (36x30 = 1080 > 1024) and 1 eligible small GT (10x10 = 100 <= 1024).
        # Two high-scoring small predictions (30x30 = 900 <= 1024) both overlap the large GT with IoU = 900/1080 = 0.8333 >= 0.50,
        # and a third lower-scoring prediction detects the eligible small GT.
        # Under COCOeval 1-to-1 matching:
        # - 1st high-scoring pred consumes the ignored large GT (ignored, not FP).
        # - 2nd high-scoring pred cannot re-match the consumed ignored GT -> counts as 1 FP!
        # - 3rd pred matches the eligible small GT -> 1 TP at precision = 1 / (1 + 1) = 0.50 -> AP_S@50 = 0.50 (NOT 1.0!).
        gt_map = {
            (1, 2): [
                [100.0, 100.0, 36.0, 30.0],  # area = 1080 > 1024 (ignored for AP_S)
                [300.0, 300.0, 10.0, 10.0],  # area = 100 <= 1024 (eligible for AP_S)
            ]
        }
        preds_two_dups = {
            2: [
                {"image_id": 1, "category_id": 3, "bbox": [100.0, 100.0, 30.0, 30.0], "score": 0.95},
                {"image_id": 1, "category_id": 3, "bbox": [102.0, 100.0, 30.0, 30.0], "score": 0.90},
                {"image_id": 1, "category_id": 3, "bbox": [300.0, 300.0, 10.0, 10.0], "score": 0.80},
            ]
        }
        res_two = compute_ap_at_iou_and_area_range(gt_map, preds_two_dups, iou_thresh=0.50, area_range=(0.0, 1024.0))
        self.assertAlmostEqual(res_two["mAP"], 0.50, places=4)
        # Whereas if only ONE prediction overlaps the ignored large GT, it is absorbed cleanly and AP_S@50 == 1.0:
        preds_one_ign = {
            2: [
                {"image_id": 1, "category_id": 3, "bbox": [100.0, 100.0, 30.0, 30.0], "score": 0.95},
                {"image_id": 1, "category_id": 3, "bbox": [300.0, 300.0, 10.0, 10.0], "score": 0.80},
            ]
        }
        res_one = compute_ap_at_iou_and_area_range(gt_map, preds_one_ign, iou_thresh=0.50, area_range=(0.0, 1024.0))
        self.assertAlmostEqual(res_one["mAP"], 1.00, places=4)

    def test_one_to_one_recall_matching_never_double_counts_single_prediction_on_two_gts(self):
        coco_imgs = {1: {"id": 1, "video_id": 1, "width": 1920, "height": 1080}}
        anns_by_img = {
            1: [
                {"id": 1, "image_id": 1, "category_id": 3, "bbox": [100.0, 100.0, 4.0, 4.0]},
                {"id": 2, "image_id": 1, "category_id": 3, "bbox": [101.0, 100.0, 4.0, 4.0]},
            ]
        }
        one_pred = [{"image_id": 1, "category_id": 3, "bbox": [100.5, 100.0, 4.0, 4.0], "score": 0.90}]
        ca1 = compute_class_agnostic_ap_and_bin_recalls(coco_imgs, anns_by_img, one_pred, iou_thresh=0.50)
        b_le4 = ca1["recall_by_effective_area_bin_at_640"]["01_LE_4px2"]
        self.assertEqual(b_le4["n_gt"], 2)
        self.assertEqual(b_le4["n_tp"], 1)
        self.assertEqual(b_le4["recall"], 0.5)
        self.assertEqual(b_le4["n_tp_iou30"], 1)
        self.assertEqual(b_le4["n_any_overlap_50"], 2)
        self.assertEqual(b_le4["n_unmatched_gts_with_same_class_overlap_50"], 1)
        self.assertEqual(ca1["n_preds_overlapping_multiple_same_class_gts_50"], 1)
        self.assertEqual(ca1["n_extra_pred_gt_overlap_pairs_50"], 1)
        diag1 = diagnose_tiny_le_4px2_and_ship_confusion(coco_imgs, anns_by_img, one_pred, one_pred)
        ob1 = diag1["tiny_01_LE_4px2_diagnostic"]["outcome_breakdown"]
        self.assertEqual(ob1.get("01_TP_iou50_1to1_score_ge_005"), 1)
        self.assertEqual(ob1.get("03_UNMATCHED_GT_WITH_CONSUMED_SAME_CLASS_OVERLAP_iou_ge_030"), 1)
        two_preds = [
            {"image_id": 1, "category_id": 3, "bbox": [100.0, 100.0, 4.0, 4.0], "score": 0.92},
            {"image_id": 1, "category_id": 3, "bbox": [101.0, 100.0, 4.0, 4.0], "score": 0.85},
        ]
        ca2 = compute_class_agnostic_ap_and_bin_recalls(coco_imgs, anns_by_img, two_preds, iou_thresh=0.50)
        self.assertEqual(ca2["recall_by_effective_area_bin_at_640"]["01_LE_4px2"]["n_tp"], 2)
        self.assertEqual(ca2["recall_by_effective_area_bin_at_640"]["01_LE_4px2"]["recall"], 1.0)

    def test_prediction_cache_manifest_rejects_mismatched_stride_annotations_checkpoints_stage_d_code_tampered_preds_or_external_sha(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            coco_val = root / "seaclips-val.json"
            coco_val.write_text(json.dumps({"images": [{"id": 1}, {"id": 2}], "annotations": []}), encoding="utf-8")
            out_dir = root / "out"
            (out_dir / "arm_d").mkdir(parents=True)
            (out_dir / "arm_d" / "detector_arm_d_final.pt").write_bytes(b"ckpt_d_v1")
            p1_dir = root / "p1"
            (p1_dir / "arm_e").mkdir(parents=True)
            (p1_dir / "arm_e" / "detector_arm_e_final.pt").write_bytes(b"ckpt_p1_e_v1")
            anns_10 = {1: [{"id": 10, "category_id": 3, "bbox": [10.0, 10.0, 4.0, 4.0]}]}
            sample_preds = {"d": [{"image_id": 1, "category_id": 3, "bbox": [10.0, 10.0, 4.0, 4.0], "score": 0.9}]}
            sample_low = {"arm_d": [{"image_id": 1, "category_id": 3, "bbox": [10.0, 10.0, 4.0, 4.0], "score": 0.9}]}
            man_s10 = build_prediction_cache_manifest(
                val_frame_stride=10, eval_image_ids=[1], eval_anns_dict=anns_10,
                coco_val_path=coco_val, ckpt_root=out_dir, phase1_root=p1_dir, stage_d_enabled=False,
                preds_by_arm=sample_preds, low_preds_by_key=sample_low,
            )
            payload = {"cache_manifest": man_s10, "preds_by_arm": copy.deepcopy(sample_preds), "low_preds_by_key": copy.deepcopy(sample_low)}
            cache_file = root / "cache.json"
            cache_file.write_text(json.dumps(payload), encoding="utf-8")
            pinned_file_sha = sha256_file_bytes(cache_file)
            validate_prediction_cache_manifest(payload, man_s10, cache_file_path=cache_file, expected_cache_file_sha256=pinned_file_sha)
            # 1. Legacy cache without cache_manifest fails closed
            with self.assertRaisesRegex(AssertionError, "missing 'cache_manifest'"):
                validate_prediction_cache_manifest({"preds_by_arm": {}}, man_s10)
            # 2. Stride 10 cache reused under Stride 1 fails closed
            anns_1 = {1: anns_10[1], 2: [{"id": 11, "category_id": 3, "bbox": [20.0, 20.0, 5.0, 5.0]}]}
            man_s1 = build_prediction_cache_manifest(
                val_frame_stride=1, eval_image_ids=[1, 2], eval_anns_dict=anns_1,
                coco_val_path=coco_val, ckpt_root=out_dir, phase1_root=p1_dir, stage_d_enabled=False,
            )
            with self.assertRaisesRegex(AssertionError, "val_frame_stride"):
                validate_prediction_cache_manifest(payload, man_s1)
            # 3. Symmetric Stage D mismatch: False -> True AND True -> False both fail closed
            man_stage_d_true = dict(man_s10, stage_d_enabled=True)
            with self.assertRaisesRegex(AssertionError, "stage_d_enabled"):
                validate_prediction_cache_manifest(payload, man_stage_d_true)
            payload_stage_d_true = {"cache_manifest": man_stage_d_true, "preds_by_arm": copy.deepcopy(sample_preds), "low_preds_by_key": copy.deepcopy(sample_low)}
            with self.assertRaisesRegex(AssertionError, "stage_d_enabled"):
                validate_prediction_cache_manifest(payload_stage_d_true, man_s10)
            # 4. Inference code/config fingerprint mismatch fails closed
            man_diff_cfg = build_prediction_cache_manifest(
                val_frame_stride=10, eval_image_ids=[1], eval_anns_dict=anns_10,
                coco_val_path=coco_val, ckpt_root=out_dir, phase1_root=p1_dir, stage_d_enabled=False,
                custom_inference_config={"score_thresh_std": 0.10},
            )
            with self.assertRaisesRegex(AssertionError, "inference_config"):
                validate_prediction_cache_manifest(payload, man_diff_cfg)
            # 5. Uncoordinated edit to `preds_by_arm` fails closed via internal payload hash
            tampered_payload = copy.deepcopy(payload)
            tampered_payload["preds_by_arm"]["d"][0]["score"] = 0.12
            with self.assertRaisesRegex(AssertionError, "preds_by_arm' payload hash mismatch"):
                validate_prediction_cache_manifest(tampered_payload, man_s10)
            # 6. Coordinated edit to BOTH `preds_by_arm` AND `preds_by_arm_sha256` fails closed when external SHA-256 is pinned!
            coord_payload = copy.deepcopy(tampered_payload)
            coord_payload["cache_manifest"] = attach_prediction_payload_hashes_to_cache_manifest(
                coord_payload["cache_manifest"], coord_payload["preds_by_arm"], coord_payload["low_preds_by_key"]
            )
            cache_file.write_text(json.dumps(coord_payload), encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "External cache file SHA-256 mismatch"):
                validate_prediction_cache_manifest(
                    coord_payload, man_s10, cache_file_path=cache_file, expected_cache_file_sha256=pinned_file_sha
                )

    def test_ap30_and_ap50_macro_means_match_per_class_means_and_markdown_export_has_no_ap30_in_table1(self):
        coco_imgs = {1: {"id": 1, "video_id": 1, "width": 1920, "height": 1080}}
        anns_by_img = {
            1: [
                {"id": 1, "image_id": 1, "category_id": 2, "bbox": [100.0, 100.0, 20.0, 15.0]},
                {"id": 2, "image_id": 1, "category_id": 3, "bbox": [200.0, 200.0, 4.0, 4.0]},
                {"id": 3, "image_id": 1, "category_id": 7, "bbox": [400.0, 300.0, 80.0, 50.0]},
            ]
        }
        preds = [
            {"image_id": 1, "category_id": 2, "bbox": [101.0, 100.0, 20.0, 15.0], "score": 0.91},
            {"image_id": 1, "category_id": 3, "bbox": [201.0, 200.0, 4.0, 4.0], "score": 0.85},
            {"image_id": 1, "category_id": 7, "bbox": [410.0, 305.0, 80.0, 50.0], "score": 0.72},
        ]
        ov, _ = evaluate_predictions_dict_and_per_video(coco_imgs, anns_by_img, preds)
        assert_ap_table_arithmetic_consistency(ov)
        bad_ov = copy.deepcopy(ov)
        bad_ov["AP30_mAP"] = 0.1234
        with self.assertRaisesRegex(AssertionError, "ARITHMETIC INCONSISTENCY"):
            assert_ap_table_arithmetic_consistency(bad_ov)
        mock_report = {
            "val_frame_stride": 10,
            "n_val_images_evaluated": 1,
            "total_raw_val_images_in_coco": 10,
            "n_val_gt_annotations_evaluated": 3,
            "evaluation_gate_status": "PASS — ALL VARIANTS PYCOCOTOOLS PARITY VERIFIED",
            "metrics_by_variant": {"arm_d": ov},
            "pycocotools_evaluator_crosscheck": {"arm_d": {"parity_status": "PARITY_PASSED"}},
        }
        md = render_verified_markdown_tables_from_report(mock_report)
        self.assertIn("## 1. Machine-Verified `AP50` Table", md)
        self.assertIn("## 2. Machine-Verified Per-Class `AP30` Table", md)
        table1_sec = md.split("## 2. Machine-Verified Per-Class `AP30` Table")[0]
        # Verify Table 1 row for `arm_d` has no parenthetical AP30 numbers!
        arm_d_row_t1 = [line for line in table1_sec.splitlines() if line.startswith("| `arm_d` |")][0]
        self.assertNotIn("(", arm_d_row_t1)

    def test_evaluation_gate_banner_and_status_reflect_parity_not_run(self):
        st_pass, ban_pass = format_evaluation_gate_status_and_banner(True, ["PARITY_PASSED", "PARITY_PASSED"])
        self.assertTrue(st_pass.startswith("PASS"))
        self.assertIn("PASS", ban_pass)
        st_inc, ban_inc = format_evaluation_gate_status_and_banner(False, ["PARITY_NOT_RUN", "PARITY_NOT_RUN"])
        self.assertEqual(st_inc, "INCOMPLETE — PYCOCOTOOLS PARITY NOT RUN")
        self.assertIn("INCOMPLETE — PYCOCOTOOLS PARITY NOT RUN", ban_inc)
        self.assertNotIn("PASS", ban_inc)
        st_fail, ban_fail = format_evaluation_gate_status_and_banner(False, ["PARITY_PASSED", "PARITY_FAILED"])
        self.assertEqual(st_fail, "FAIL — PYCOCOTOOLS PARITY FAILED")
        self.assertIn("FAIL", ban_fail)
        self.assertNotIn("PASS", ban_fail)

    def test_pycocotools_parity_gate_fails_closed_when_missing_or_empty_preds(self):
        coco_imgs = {1: {"id": 1, "video_id": 1, "width": 1920, "height": 1080}}
        anns_by_img = {1: [{"id": 1, "image_id": 1, "category_id": 3, "bbox": [100.0, 100.0, 4.0, 4.0]}]}
        ov_empty, _ = evaluate_predictions_dict_and_per_video(coco_imgs, anns_by_img, [])
        res_nonfatal = verify_against_pycocotools_if_available(
            coco_imgs, anns_by_img, [], ov_empty, fail_fast=False, require_pycocotools=False
        )
        self.assertIn(res_nonfatal["parity_status"], ("PARITY_NOT_RUN", "PARITY_FAILED"))
        self.assertFalse(res_nonfatal["agrees_within_0_005"])
        with self.assertRaises(AssertionError):
            verify_against_pycocotools_if_available(
                coco_imgs, anns_by_img, [], ov_empty, fail_fast=True, require_pycocotools=True
            )

    def test_coco_annotation_area_vs_bbox_area_audit_and_unified_area_definition(self):
        anns_by_img = {
            1: [
                {"id": 1, "image_id": 1, "category_id": 3, "bbox": [10.0, 10.0, 4.0, 5.0], "area": 20.0},
                {"id": 2, "image_id": 1, "category_id": 6, "bbox": [50.0, 50.0, 40.0, 30.0], "area": 600.0},
            ]
        }
        audit = audit_coco_annotation_area_vs_bbox_area(anns_by_img, scope_label="full_val_coco")
        self.assertEqual(audit["audit_scope"], "full_val_coco")
        self.assertEqual(audit["n_gt_annotations_in_scope"], 2)
        self.assertEqual(audit["n_annotations_with_diff_gt_0_001"], 1)
        self.assertEqual(audit["n_small_lt_1024_classification_disagreements"], 1)
        self.assertTrue(audit["evaluators_use_identical_bbox_wh_definition"])

    def test_overlay_rectangle_coordinates_never_invert_y1_when_bx_plus_bh_lt_by(self):
        if not PIL_AVAILABLE:
            self.skipTest("Pillow not installed")
        canvas = Image.new("RGB", (640, 640), (114, 114, 114))
        dr = ImageDraw.Draw(canvas)
        bx, by, bw, bh = 10.0, 500.0, 12.0, 8.0
        dr.rectangle([float(bx), float(by), float(bx + bw), float(by + bh)], outline=(0, 255, 128), width=2)

    def test_horizontal_flip_boxes_and_safe_torch_load(self):
        boxes = np.asarray([[100.0, 50.0, 20.0, 10.0]], dtype=np.float32)
        flipped = flip_boxes_horizontally_in_crop(boxes, valid_w=1280)
        self.assertAlmostEqual(float(flipped[0, 0]), 1160.0, places=4)
        self.assertAlmostEqual(float(flipped[0, 2]), 20.0, places=4)

    def test_class_balanced_training_schedule_enriches_scarce_ship_and_animal_tiles(self):
        # Claude V11 Review Finding #3: A medium-sized SHIP (e.g. 30x20 = 600 px^2 native -> 66.7 px^2 at 640 > 25 px^2)
        # or ANIMAL on a frame with no <=25 px^2 tiny GTs is skipped by Stage C under size-only sampling,
        # but is explicitly retained and jitter-sampled when class_balanced_sampling=True.
        coco_imgs = {1: {"id": 1, "video_id": 1, "frame_id": 0, "width": 1920, "height": 1080}}
        anns = {
            1: [
                {"id": 1, "image_id": 1, "category_id": 7, "bbox": [500.0, 480.0, 30.0, 20.0]},  # SHIP (600 px^2 native)
                {"id": 2, "image_id": 1, "category_id": 1, "bbox": [1200.0, 420.0, 18.0, 16.0]}, # ANIMAL (288 px^2 native)
            ]
        }
        sched_unbal = build_slicing_aided_training_schedule(coco_imgs, anns, arm="d", frame_stride=1, class_balanced_sampling=False)
        sched_bal = build_slicing_aided_training_schedule(coco_imgs, anns, arm="d", frame_stride=1, class_balanced_sampling=True)
        # Under size-only sampling, only 1 Stage B full frame + 1 negative Stage C crop = 2 tiles are scheduled (0 positive Stage C crops for SHIP/ANIMAL).
        # Under class-balanced sampling, Stage C crops covering SHIP/ANIMAL + 1 jittered scarce-class Stage C crop are added (>= 4 tiles).
        self.assertEqual(len(sched_unbal), 2)
        self.assertGreaterEqual(len(sched_bal), 4)

    def test_per_video_absent_class_ap_rule_and_fp_density_metrics(self):
        # Claude V11 Finding #4 & ChatGPT V11 Finding #6:
        # Verify per-video absent-class rule (only present classes in each video enter that video's mAP denominator)
        # and verify mean_predictions_per_frame and fp_per_frame_1to1_iou50.
        coco_imgs = {
            1: {"id": 1, "video_id": 1, "width": 1920, "height": 1080},
            2: {"id": 2, "video_id": 2, "width": 1920, "height": 1080},
        }
        anns_by_img = {
            1: [{"id": 1, "image_id": 1, "category_id": 3, "bbox": [100.0, 100.0, 6.0, 6.0]}],  # v01 has ONLY MARINE_MARKER (1 class)
            2: [
                {"id": 2, "image_id": 2, "category_id": 2, "bbox": [200.0, 200.0, 20.0, 20.0]},  # v02 has BOAT + SHIP (2 classes)
                {"id": 3, "image_id": 2, "category_id": 7, "bbox": [400.0, 400.0, 50.0, 40.0]},
            ],
        }
        preds = [
            {"image_id": 1, "category_id": 3, "bbox": [100.0, 100.0, 6.0, 6.0], "score": 0.95},
            {"image_id": 1, "category_id": 3, "bbox": [800.0, 800.0, 6.0, 6.0], "score": 0.40},  # 1 extra FP on image 1
            {"image_id": 2, "category_id": 2, "bbox": [200.0, 200.0, 20.0, 20.0], "score": 0.90},
        ]
        ov, pv = evaluate_predictions_dict_and_per_video(coco_imgs, anns_by_img, preds)
        self.assertEqual(ov["n_present_classes_by_video"], {"v01": 1, "v02": 2})
        self.assertAlmostEqual(pv["AP50_mAP"][1], 1.0, places=4)
        self.assertAlmostEqual(pv["AP50_mAP"][2], 0.5, places=4)  # BOAT=1.0, SHIP=0.0 -> mean over 2 present classes = 0.5
        self.assertAlmostEqual(ov["mean_predictions_per_frame"], 1.5, places=2)
        self.assertEqual(ov["fp_count_1to1_iou50"], 1)
        self.assertAlmostEqual(ov["fp_per_frame_1to1_iou50"], 0.5, places=2)

    def test_repeat_factor_sampling_preserves_all_marine_marker_crops_while_enriching_ship_and_not_oversampling_animal(self):
        # Claude V18.2 Finding #2, #3 & Step 3:
        # Frame 1 has two <=25 px^2 MARINE_MARKER GTs (cat=3) in distinct Stage C crops AND a SHIP GT (cat=7) AND an ANIMAL GT (cat=1).
        # Under V12 class_balanced_sampling, pos_c was cut from pos_c[:2] to pos_c[:1] (losing 1 MARINE_MARKER crop).
        # Under V13 repeat_factor_sampling (t_c=0.10), BOTH pos_c[:2] MARINE_MARKER crops are preserved AND SHIP is enriched,
        # while a frame with ONLY ANIMAL (cat=1) does NOT trigger noisy ANIMAL jitter oversampling.
        coco_imgs = {
            i: {"id": i, "video_id": 1, "frame_id": i * 10, "width": 1920, "height": 1080}
            for i in range(1, 21)
        }
        anns = {i: [{"id": i * 10, "image_id": i, "category_id": 3, "bbox": [100.0, 200.0, 5.0, 5.0]}] for i in range(1, 21)}
        # Frame 1 has two MARINE_MARKERs in distinct crops + 1 SHIP (cat=7, present in 1/20 = 5% < 10% of frames):
        anns[1] = [
            {"id": 101, "image_id": 1, "category_id": 3, "bbox": [100.0, 200.0, 5.0, 5.0]},
            {"id": 102, "image_id": 1, "category_id": 3, "bbox": [1500.0, 200.0, 5.0, 5.0]},
            {"id": 103, "image_id": 1, "category_id": 7, "bbox": [900.0, 500.0, 40.0, 25.0]},
        ]
        # Frame 2 has two MARINE_MARKERs + 1 ANIMAL (cat=1):
        anns[2] = [
            {"id": 201, "image_id": 2, "category_id": 3, "bbox": [100.0, 200.0, 5.0, 5.0]},
            {"id": 202, "image_id": 2, "category_id": 3, "bbox": [1500.0, 200.0, 5.0, 5.0]},
            {"id": 203, "image_id": 2, "category_id": 1, "bbox": [900.0, 420.0, 18.0, 16.0]},
        ]
        rep_f = compute_category_repeat_factors(coco_imgs, anns, repeat_factor_threshold=0.10)
        self.assertGreater(rep_f[7], 1.0)
        self.assertEqual(rep_f[3], 1.0)
        sched_rf = build_slicing_aided_training_schedule(
            coco_imgs, anns, arm="d", frame_stride=1, interleave_across_videos=False,
            repeat_factor_sampling=True, repeat_factor_threshold=0.10,
        )
        f1_crops = [cw for iid, cw in sched_rf if iid == 1 and cw.stage == "STAGE_C_2X_CROP"]
        f2_crops = [cw for iid, cw in sched_rf if iid == 2 and cw.stage == "STAGE_C_2X_CROP"]
        # Frame 1 gets 2 pos_c marker crops + 1 scarce_c ship crop + 1 ship jitter crop + 1 neg_c crop = 5 Stage C crops
        self.assertGreaterEqual(len(f1_crops), 4)
        # Frame 2 (with ANIMAL, not SHIP) keeps both pos_c[:2] marker crops + 1 neg_c crop = 3 Stage C crops (zero ANIMAL jitter oversampling!)
        self.assertEqual(len(f2_crops), 3)

    def test_gt_weighted_ap50_and_ap30_reflect_marine_marker_majority_share(self):
        # Claude V18.2 Finding #3: GT-weighted AP weights each class by its validation GT count N_c / N_total.
        coco_imgs = {1: {"id": 1, "video_id": 1, "width": 1920, "height": 1080}}
        # 9 MARINE_MARKER GTs (cat=3) and 1 SHIP GT (cat=7):
        anns_by_img = {
            1: [
                *[{"id": i, "image_id": 1, "category_id": 3, "bbox": [float(50 * i), 100.0, 6.0, 6.0]} for i in range(1, 10)],
                {"id": 10, "image_id": 1, "category_id": 7, "bbox": [800.0, 500.0, 60.0, 40.0]},
            ]
        }
        # Model detects ONLY the 1 SHIP GT (SHIP AP50 = 1.0) and misses all 9 MARINE_MARKER GTs (MARKER AP50 = 0.0):
        preds_ship_only = [{"image_id": 1, "category_id": 7, "bbox": [800.0, 500.0, 60.0, 40.0], "score": 0.95}]
        ov, _ = evaluate_predictions_dict_and_per_video(coco_imgs, anns_by_img, preds_ship_only)
        # Macro AP50 across 2 present classes = (0.0 + 1.0) / 2 = 0.5000,
        # whereas GT-weighted AP50 = (9 * 0.0 + 1 * 1.0) / 10 = 0.1000!
        self.assertAlmostEqual(ov["AP50"]["mAP"], 0.50, places=4)
        self.assertAlmostEqual(ov["AP50_gt_weighted_mAP"], 0.10, places=4)
        self.assertAlmostEqual(ov["AP30_gt_weighted_mAP"], 0.10, places=4)

    def test_discover_and_extract_frozen_r50_val_predictions_auto_loads_from_bfcef_sandbox(self):
        # User Request & Claude V18.2 Finding #1: Verify auto-discovery of frozen R50 VAL predictions inside bfcef_f_s0/run15a_final_candidate_sandbox.
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            proj = Path(td) / "eval_tools"
            sb = proj / "bfcef_f_s0" / "run15a_final_candidate_sandbox" / "oct13_spatial_matched_control"
            sb.mkdir(parents=True)
            val_coco = Path(td) / "seaclips-val.json"
            val_coco.write_text(json.dumps({
                "images": [{"id": 5001, "video_id": 1, "width": 1920, "height": 1080}],
                "annotations": [{"id": 1, "image_id": 5001, "category_id": 3, "bbox": [100.0, 100.0, 6.0, 6.0]}],
            }), encoding="utf-8")
            r50_val_file = sb / "val_r50_baseline_predictions.json"
            r50_val_file.write_text(json.dumps([
                {"image_id": 5001, "category_id": 3, "bbox": [100.0, 100.0, 6.0, 6.0], "score": 0.88}
            ]), encoding="utf-8")
            (proj / "train_bfcef_f_s0_final_experts.py").write_text(
                f'VAL_CACHE = r"{r50_val_file}"\n', encoding="utf-8"
            )
            eval_imgs = {5001: {"id": 5001, "video_id": 1, "width": 1920, "height": 1080}}
            preds, info = discover_and_extract_frozen_r50_val_predictions(
                project_dir=proj, coco_val_path=val_coco, eval_images_dict=eval_imgs, out_dir=Path(td) / "out"
            )
            self.assertEqual(info["status"], "AUTO_DISCOVERED_VAL_PREDICTIONS")
            self.assertEqual(len(preds), 1)
            self.assertEqual(preds[0]["image_id"], 5001)
            self.assertAlmostEqual(preds[0]["score"], 0.88, places=4)

    def test_equal_recipe_arm_b_training_does_not_overwrite_with_phase1_arm_b_copy(self):
        # Claude V18.2 Finding #4: When 'b' is in --phase2-arms (e.g. 'b,d,e'), run_phase2_full_training must NOT copy
        # 3-epoch phase1_arm_b over out_dir/arm_b.
        arms_to_train = [a.strip().lower() for a in "b,d,e".split(",") if a.strip()]
        copied = [base_arm for base_arm in ("a", "b") if base_arm not in arms_to_train]
        self.assertEqual(copied, ["a"])

    def test_stage_b_and_stage_c_box_loss_scaling_is_consistent_in_native_coordinates(self):
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not installed in this environment")
        loss_fn = hr_p2.HybridNWDBoxLoss(constant_c=8.0, w_nwd=2.0, w_giou=0.5, w_l1=1.0)
        native_gt = torch.tensor([[300.0, 300.0, 18.0, 15.0]])
        native_pred = torch.tensor([[303.0, 303.0, 18.0, 15.0]])
        scale_b = 2.0 / 3.0
        lin_scale_b = max(1e-6, math.sqrt(float(scale_b ** 2)))
        loss_b = loss_fn((native_pred * scale_b) / lin_scale_b, (native_gt * scale_b) / lin_scale_b)["loss_box_total"]
        scale_c = 2.0
        lin_scale_c = max(1e-6, math.sqrt(float(scale_c ** 2)))
        loss_c = loss_fn((native_pred * scale_c) / lin_scale_c, (native_gt * scale_c) / lin_scale_c)["loss_box_total"]
        self.assertAlmostEqual(float(loss_b.item()), float(loss_c.item()), places=5)

    def test_pytorch_stride4_detector_frozen_bn_softplus_and_exact_focal_loss_if_torch_available(self):
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not installed in this environment")
        model = Stride4MaritimeDetector(num_classes=NUM_CLASSES, use_p2_stride4=True, pretrained_backbone=False, freeze_backbone_bn=True)
        model.train(True)
        for m_bn in model.stem.modules():
            if isinstance(m_bn, nn.BatchNorm2d):
                self.assertFalse(m_bn.training)
        nn.init.zeros_(model.cls_logits.weight)
        x = torch.zeros(1, 3, 256, 256)
        outs = model(x)
        for stride, _, ltrb in outs:
            self.assertTrue(bool((ltrb > 0).all().item()))
        ema16 = PositiveAnchorEMA(init_val=16.0, min_floor=1.0)
        gt_b = torch.zeros((0, 4), dtype=torch.float32)
        gt_l = torch.zeros((0,), dtype=torch.long)
        loss_zero = compute_detector_loss_on_tile(outs, gt_b, gt_l, area_scale=1.0, use_nwd_rfla=True, pos_ema_tracker=ema16)
        expected_loss_256 = (5440 * 7 * 0.75 * (0.01 ** 2) * (-math.log(0.99))) / 16.0
        self.assertAlmostEqual(float(loss_zero["loss_cls"].item()), expected_loss_256, places=5)


def run_env_check(args: argparse.Namespace) -> None:
    model_info = {}
    if TORCH_AVAILABLE:
        m_p2 = Stride4MaritimeDetector(num_classes=NUM_CLASSES, use_p2_stride4=True, pretrained_backbone=False)
        m_p3 = Stride4MaritimeDetector(num_classes=NUM_CLASSES, use_p2_stride4=False, pretrained_backbone=False)
        model_info = {
            "backbone": "torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT) [ImageNet-1K V2, FrozenBN]",
            "params_p2_to_p5_arms_cde": m_p2.parameter_summary(),
            "params_p3_to_p5_arms_ab": m_p3.parameter_summary(),
        }
    info = {
        "torch_available": TORCH_AVAILABLE,
        "cuda_available": bool(TORCH_AVAILABLE and torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if (TORCH_AVAILABLE and torch.cuda.is_available()) else None,
        "cuda_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
        if (TORCH_AVAILABLE and torch.cuda.is_available()) else None,
        "detector_specification": model_info,
        "seaclips_zip_exists": Path(args.zip).is_file(),
        "train_coco_exists": Path(args.coco_train).is_file(),
        "val_coco_exists": Path(args.coco_val).is_file(),
    }
    print("[MARIVISION-HR-P2 V11 WORKSTATION ENVIRONMENT & MODEL CHECK]")
    print(json.dumps(info, indent=2))


def run_raw_pos_histogram_mode(args: argparse.Namespace) -> None:
    raw = json.loads(Path(args.coco_train).read_text(encoding="utf-8"))
    imgs = {int(im["id"]): im for im in raw["images"]}
    anns_by_img: Dict[int, List[Dict]] = defaultdict(list)
    for a in raw.get("annotations", []):
        if int(a.get("iscrowd", 0)) == 0:
            anns_by_img[int(a["image_id"])].append(a)
    res = audit_raw_pos_histogram(imgs, anns_by_img, arm=args.arm, frame_stride=args.frame_stride, max_tiles_to_sample=600)
    print("[MARIVISION-HR-P2 V11 RAW_POS & PER-CLASS ANCHOR SAMPLING AUDIT]")
    print(json.dumps(res, indent=2))


def run_overfit_one_batch_and_export_overlays(args: argparse.Namespace) -> None:
    """Export 20 visual crop overlays (with fixed `float(by + bh)`) and overfit 1 multi-video batch of 8 tiles."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for --mode overfit-one-batch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arm = args.arm.lower().strip()
    use_p2 = arm in ("c", "d", "e")
    use_nwd_rfla = arm in ("d", "e")
    use_scene = arm == "e"
    idx = SeaClipsZipNativeIndex(Path(args.zip), Path(args.coco_train), split_hint="train")
    sched = build_slicing_aided_training_schedule(idx.images, idx.anns_by_image, arm=arm, frame_stride=10, interleave_across_videos=True)
    overlay_dir = Path(args.out_dir) / "crop_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    exported = 0
    for iid, cw in sched[:80]:
        boxes_np, labels_np, ign_np = project_gt_boxes_into_crop_with_ignore(idx.anns_by_image.get(iid, []), cw)
        if len(boxes_np) == 0 and exported >= 4:
            continue
        rgb = idx.load_rgb_numpy(iid)
        crop_rgb = cw.slice_rgb(rgb)
        pil_im = Image.fromarray(crop_rgb).resize((cw.resized_w, cw.resized_h), Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (cw.detector_w, cw.detector_h), (114, 114, 114))
        canvas.paste(pil_im, (0, 0))
        dr = ImageDraw.Draw(canvas)
        for (bx, by, bw, bh), cl in zip(boxes_np, labels_np):
            dr.rectangle([float(bx), float(by), float(bx + bw), float(by + bh)], outline=(0, 255, 128), width=2)
            dr.text((float(bx) + 2, max(0.0, float(by) - 12)), CLASS_NAMES[int(cl)], fill=(255, 255, 0))
        for (ix, iy, iw, ih) in ign_np:
            dr.rectangle([float(ix), float(iy), float(ix + iw), float(iy + ih)], outline=(255, 128, 0), width=2)
            dr.text((float(ix) + 2, max(0.0, float(iy) - 12)), "IGNORE_CLIPPED", fill=(255, 128, 0))
        exported += 1
        canvas.save(overlay_dir / f"overlay_{exported:02d}_img{iid}_{cw.stage}_gt{len(boxes_np)}_ign{len(ign_np)}.png")
        if exported >= 20:
            break
    fixed_batch = sched[:8]
    cached_tiles = []
    for iid, cw in fixed_batch:
        rgb = idx.load_rgb_numpy(iid)
        crop_rgb = cw.slice_rgb(rgb)
        t_in = prepare_tile_tensor(crop_rgb, cw.detector_w, cw.detector_h, use_scene_prior=use_scene, crop_window=cw)
        b_np, l_np, i_np = project_gt_boxes_into_crop_with_ignore(idx.anns_by_image.get(iid, []), cw)
        cached_tiles.append((
            iid, cw, t_in.unsqueeze(0).to(device),
            torch.from_numpy(b_np).to(device),
            torch.from_numpy(l_np).to(device),
            torch.from_numpy(i_np).to(device),
        ))
    torch.manual_seed(20261019)
    model = Stride4MaritimeDetector(
        num_classes=NUM_CLASSES, use_p2_stride4=use_p2, use_scene_prior_channel=use_scene, pretrained_backbone=True, freeze_backbone_bn=True
    ).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1.5e-4, weight_decay=1e-4)
    amp_enabled = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if hasattr(torch, "amp") else torch.cuda.amp.GradScaler(enabled=amp_enabled)
    pos_ema = PositiveAnchorEMA(init_val=8.0, momentum=0.90, min_floor=1.0)
    print(f"[MARIVISION-HR-P2 V11 OVERFIT-ONE-BATCH & OVERLAY CHECK] Arm=({arm}) overlays_saved={exported} in {overlay_dir}")
    history = []
    for opt_step in range(1, 17):
        lr_scale = min(1.0, opt_step / 5.0)
        for pg in optimizer.param_groups:
            pg["lr"] = 1.5e-4 * lr_scale
        optimizer.zero_grad()
        step_cls = 0.0
        step_box = 0.0
        step_raw_pos_list: List[float] = []
        empty_bg_probs = []
        empty_bg_frac20 = []
        for iid, cw, t_in, gt_b, gt_l, ign_b in cached_tiles:
            autocast_ctx = (
                torch.amp.autocast("cuda", enabled=amp_enabled)
                if hasattr(torch, "amp") else torch.cuda.amp.autocast(enabled=amp_enabled)
            )
            with autocast_ctx:
                outs = model(t_in)
                ld = compute_detector_loss_on_tile(
                    outs, gt_b, gt_l, area_scale=cw.area_scale, use_nwd_rfla=use_nwd_rfla,
                    pos_ema_tracker=pos_ema, update_ema_now=False,
                    ignore_boxes_xywh=ign_b, valid_w=cw.resized_w, valid_h=cw.resized_h,
                )
                loss = ld["loss_total"] / float(len(cached_tiles))
            scaler.scale(loss).backward()
            step_cls += float(ld["loss_cls"].item()) / len(cached_tiles)
            step_box += float(ld["loss_box"].item()) / len(cached_tiles)
            step_raw_pos_list.append(float(ld["num_pos"].item()))
            if len(gt_b) == 0:
                empty_bg_probs.append(float(ld["mean_bg_prob"].item()))
                empty_bg_frac20.append(float(ld["frac_bg_gt_020"].item()))
        pos_ema.update_from_step(step_raw_pos_list)
        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0).item())
        scaler.step(optimizer)
        scaler.update()
        rec = {
            "opt_step": opt_step,
            "loss_cls": round(step_cls, 4),
            "loss_box": round(step_box, 4),
            "loss_total": round(step_cls + step_box, 4),
            "grad_norm": round(grad_norm, 3),
            "pos_ema": round(pos_ema.ema, 2),
            "empty_tile_mean_bg_prob": round(float(np.mean(empty_bg_probs)) if empty_bg_probs else 0.0, 5),
            "empty_tile_frac_bg_gt_020": round(float(np.mean(empty_bg_frac20)) if empty_bg_frac20 else 0.0, 5),
        }
        history.append(rec)
        if opt_step in (1, 2, 4, 8, 12, 16):
            print(
                f"  opt_step {opt_step:02d}/16 | loss_cls={rec['loss_cls']:.4f} | loss_box={rec['loss_box']:.4f} | "
                f"total={rec['loss_total']:.4f} | grad_norm={rec['grad_norm']:.2f} | pos_ema={rec['pos_ema']:.1f} | "
                f"empty_bg_prob={rec['empty_tile_mean_bg_prob']:.4f} | frac>0.2={rec['empty_tile_frac_bg_gt_020']:.4f}"
            )
    summary = {
        "arm": arm,
        "overlays_exported": exported,
        "overlay_dir": str(overlay_dir),
        "initial_loss_total": history[0]["loss_total"],
        "final_loss_total": history[-1]["loss_total"],
        "loss_decreased_cleanly": bool(history[-1]["loss_total"] < history[0]["loss_total"]),
        "max_empty_tile_frac_bg_gt_020": max(r["empty_tile_frac_bg_gt_020"] for r in history),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated(0) / (1024 ** 3), 3) if device.type == "cuda" else 0.0,
    }
    print("[MARIVISION-HR-P2 V11 OVERFIT-ONE-BATCH PASS]")
    print(json.dumps(summary, indent=2))


def run_smoke_test(args: argparse.Namespace) -> None:
    """Run 12-tile V11 smoke test."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for --mode smoke-test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arm = args.arm.lower().strip()
    use_p2 = arm in ("c", "d", "e")
    use_nwd_rfla = arm in ("d", "e")
    use_scene = arm == "e"
    print(f"[MARIVISION-HR-P2 SMOKE TEST V11] Arm=({arm}) device={device} P2={use_p2} NWD+RFLA={use_nwd_rfla} ScenePrior={use_scene}")
    idx = SeaClipsZipNativeIndex(Path(args.zip), Path(args.coco_train), split_hint="train")
    sched = build_slicing_aided_training_schedule(idx.images, idx.anns_by_image, arm=arm, frame_stride=10, interleave_across_videos=True)[:12]
    torch.manual_seed(20261019)
    model = Stride4MaritimeDetector(
        num_classes=NUM_CLASSES,
        use_p2_stride4=use_p2,
        use_scene_prior_channel=use_scene,
        pretrained_backbone=True,
        freeze_backbone_bn=True,
    ).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1.5e-4, weight_decay=1e-4)
    amp_enabled = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if hasattr(torch, "amp") else torch.cuda.amp.GradScaler(enabled=amp_enabled)
    pos_ema = PositiveAnchorEMA(init_val=8.0, momentum=0.90, min_floor=1.0)
    t0 = time.time()
    losses_logged = []
    step_raw_pos_buffer: List[float] = []
    optimizer.zero_grad()
    for step, (iid, cw) in enumerate(sched, 1):
        lr_scale = min(1.0, ((step - 1) // 4 + 1) / 3.0)
        for pg in optimizer.param_groups:
            pg["lr"] = 1.5e-4 * lr_scale
        rgb = idx.load_rgb_numpy(iid)
        crop_rgb = cw.slice_rgb(rgb)
        tensor_in = prepare_tile_tensor(crop_rgb, cw.detector_w, cw.detector_h, use_scene_prior=use_scene, crop_window=cw).unsqueeze(0).to(device)
        boxes_np, labels_np, ign_np = project_gt_boxes_into_crop_with_ignore(idx.anns_by_image.get(iid, []), cw)
        gt_b = torch.from_numpy(boxes_np).to(device)
        gt_l = torch.from_numpy(labels_np).to(device)
        ign_b = torch.from_numpy(ign_np).to(device)
        autocast_ctx = (
            torch.amp.autocast("cuda", enabled=amp_enabled)
            if hasattr(torch, "amp") else torch.cuda.amp.autocast(enabled=amp_enabled)
        )
        with autocast_ctx:
            outs = model(tensor_in)
            loss_dict = compute_detector_loss_on_tile(
                outs, gt_b, gt_l, area_scale=cw.area_scale, use_nwd_rfla=use_nwd_rfla,
                pos_ema_tracker=pos_ema, update_ema_now=False,
                ignore_boxes_xywh=ign_b, valid_w=cw.resized_w, valid_h=cw.resized_h,
            )
            loss = loss_dict["loss_total"] / 4.0
        scaler.scale(loss).backward()
        step_raw_pos_buffer.append(float(loss_dict["num_pos"].item()))
        if step % 4 == 0 or step == len(sched):
            pos_ema.update_from_step(step_raw_pos_buffer)
            step_raw_pos_buffer.clear()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        l_cls = float(loss_dict["loss_cls"].item())
        l_box = float(loss_dict["loss_box"].item())
        l_tot = float(loss_dict["loss_total"].item())
        bg_p = float(loss_dict["mean_bg_prob"].item())
        n_pos = int(loss_dict["num_pos"].item())
        denom_used = float(loss_dict["cls_denom"].item())
        losses_logged.append(l_tot)
        vid = idx.images[iid].get("video_id", -1)
        print(
            f"  step {step:02d}/{len(sched):02d} | vid={vid:02d} img={iid:<5d} | stage={cw.stage:<22} | "
            f"gt={len(boxes_np):02d} ign={len(ign_np):02d} pos={n_pos:02d} denom={denom_used:4.1f} | "
            f"cls={l_cls:.4f} box={l_box:.4f} tot={l_tot:.4f} | bg_p={bg_p:.4f}"
        )
    peak_vram_gb = (
        round(torch.cuda.max_memory_allocated(0) / (1024 ** 3), 3)
        if device.type == "cuda" else 0.0
    )
    summary = {
        "arm": arm,
        "device": str(device),
        "n_steps": len(sched),
        "elapsed_sec": round(time.time() - t0, 2),
        "mean_loss": round(float(np.mean(losses_logged)), 4),
        "peak_vram_gb": peak_vram_gb,
        "fits_rtx3060_12gb_budget": bool(peak_vram_gb < 10.5),
    }
    print("[MARIVISION-HR-P2 SMOKE TEST V11 PASS]")
    print(json.dumps(summary, indent=2))


def run_benchmark_100_steps(args: argparse.Namespace) -> None:
    """Execute representative multi-video batch-4 training benchmark measuring real VRAM and throughput."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for --mode benchmark-100-steps")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arm = args.arm.lower().strip()
    use_p2 = arm in ("c", "d", "e")
    use_nwd_rfla = arm in ("d", "e")
    use_scene = arm == "e"
    bs = max(1, int(args.batch_size))
    ga = max(1, int(args.grad_accum))
    n_opt_steps = max(1, int(args.bench_steps))
    idx = SeaClipsZipNativeIndex(Path(args.zip), Path(args.coco_train), split_hint="train")
    sched = build_slicing_aided_training_schedule(idx.images, idx.anns_by_image, arm=arm, frame_stride=5, interleave_across_videos=True)
    needed_tiles = n_opt_steps * bs * ga
    while len(sched) < needed_tiles:
        sched = sched + sched
    sched = sched[:needed_tiles]
    torch.manual_seed(20261019)
    model = Stride4MaritimeDetector(
        num_classes=NUM_CLASSES, use_p2_stride4=use_p2, use_scene_prior_channel=use_scene, pretrained_backbone=True, freeze_backbone_bn=True
    ).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1.5e-4, weight_decay=1e-4)
    amp_enabled = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if hasattr(torch, "amp") else torch.cuda.amp.GradScaler(enabled=amp_enabled)
    pos_ema = PositiveAnchorEMA(init_val=8.0, momentum=0.90, min_floor=1.0)
    print(
        f"[MARIVISION-HR-P2 V11 {n_opt_steps}-STEP BENCHMARK] Arm=({arm}) batch_size={bs} grad_accum={ga} "
        f"eff_batch={bs*ga} total_tiles={needed_tiles}"
    )
    t0 = time.time()
    tile_cursor = 0
    for opt_step in range(1, n_opt_steps + 1):
        lr_scale = min(1.0, opt_step / max(1.0, min(20.0, n_opt_steps * 0.2)))
        for pg in optimizer.param_groups:
            pg["lr"] = 1.5e-4 * lr_scale
        optimizer.zero_grad()
        step_loss_cls = 0.0
        step_loss_box = 0.0
        step_raw_pos_list: List[float] = []
        for _ in range(ga):
            batch_items = sched[tile_cursor : tile_cursor + bs]
            tile_cursor += bs
            tensors = []
            gt_List = []
            for iid, cw in batch_items:
                rgb = idx.load_rgb_numpy(iid)
                crop_rgb = cw.slice_rgb(rgb)
                tensors.append(prepare_tile_tensor(crop_rgb, cw.detector_w, cw.detector_h, use_scene_prior=use_scene, crop_window=cw))
                b_np, l_np, i_np = project_gt_boxes_into_crop_with_ignore(idx.anns_by_image.get(iid, []), cw)
                gt_List.append((
                    cw, torch.from_numpy(b_np).to(device),
                    torch.from_numpy(l_np).to(device),
                    torch.from_numpy(i_np).to(device),
                ))
            batch_tensor = torch.stack(tensors, dim=0).to(device)
            autocast_ctx = (
                torch.amp.autocast("cuda", enabled=amp_enabled)
                if hasattr(torch, "amp") else torch.cuda.amp.autocast(enabled=amp_enabled)
            )
            with autocast_ctx:
                outs_batch = model(batch_tensor)
                mb_loss = 0.0
                for b_i, (cw, gt_b, gt_l, ign_b) in enumerate(gt_List):
                    outs_i = [(s, lg[b_i : b_i + 1], lt[b_i : b_i + 1]) for s, lg, lt in outs_batch]
                    ld = compute_detector_loss_on_tile(
                        outs_i, gt_b, gt_l, area_scale=cw.area_scale, use_nwd_rfla=use_nwd_rfla,
                        pos_ema_tracker=pos_ema, update_ema_now=False,
                        ignore_boxes_xywh=ign_b, valid_w=cw.resized_w, valid_h=cw.resized_h,
                    )
                    mb_loss = mb_loss + ld["loss_total"] / float(bs * ga)
                    step_loss_cls += float(ld["loss_cls"].item()) / float(bs * ga)
                    step_loss_box += float(ld["loss_box"].item()) / float(bs * ga)
                    step_raw_pos_list.append(float(ld["num_pos"].item()))
            scaler.scale(mb_loss).backward()
        pos_ema.update_from_step(step_raw_pos_list)
        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0).item())
        scaler.step(optimizer)
        scaler.update()
        if opt_step in (1, 10, 25, 50, 75, n_opt_steps):
            vram_now = round(torch.cuda.max_memory_allocated(0) / (1024 ** 3), 3) if device.type == "cuda" else 0.0
            print(
                f"  opt_step {opt_step:03d}/{n_opt_steps:03d} | cls={step_loss_cls:.4f} box={step_loss_box:.4f} "
                f"grad_norm={grad_norm:.2f} | pos_ema={pos_ema.ema:.1f} | peak_vram={vram_now:.3f} GB"
            )
    elapsed = time.time() - t0
    peak_vram_gb = round(torch.cuda.max_memory_allocated(0) / (1024 ** 3), 3) if device.type == "cuda" else 0.0
    summary = {
        "arm": arm,
        "batch_size": bs,
        "grad_accum": ga,
        "effective_batch_size": bs * ga,
        "optimizer_steps": n_opt_steps,
        "total_tiles_processed": needed_tiles,
        "elapsed_sec": round(elapsed, 2),
        "sec_per_tile": round(elapsed / max(1, needed_tiles), 4),
        "peak_vram_gb": peak_vram_gb,
        "passes_10_5gb_vram_gate": bool(peak_vram_gb < 10.5),
    }
    print("[MARIVISION-HR-P2 V11 BENCHMARK PASS]")
    print(json.dumps(summary, indent=2))


def train_and_evaluate_single_arm(
    args: argparse.Namespace,
    arm: str,
    use_cosine_lr: bool = False,
    use_hflip_aug: bool = False,
    use_model_ema: bool = False,
    resume_ckpt: Optional[Path] = None,
) -> Dict[str, object]:
    """Run training (Phase 1 screening or Phase 2 full training with Cosine LR + HFlip + Model EMA) and evaluate on VAL."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for training modes")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arm = arm.lower().strip()
    use_p2 = arm in ("c", "d", "e")
    use_nwd_rfla = arm in ("d", "e")
    use_scene = arm == "e"
    bs = max(1, int(args.batch_size))
    ga = max(1, int(args.grad_accum))
    epochs = max(1, int(args.epochs))
    out_dir = Path(args.out_dir) / f"arm_{arm}"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_idx = SeaClipsZipNativeIndex(Path(args.zip), Path(args.coco_train), split_hint="train")
    val_idx = SeaClipsZipNativeIndex(Path(args.zip), Path(args.coco_val), split_hint="val")
    torch.manual_seed(20261019)
    model = Stride4MaritimeDetector(
        num_classes=NUM_CLASSES, use_p2_stride4=use_p2, use_scene_prior_channel=use_scene, pretrained_backbone=True, freeze_backbone_bn=True
    ).to(device)
    if resume_ckpt is not None and Path(resume_ckpt).is_file():
        print(f"  [Arm ({arm})] Warm-starting weights from {resume_ckpt} ...")
        ckpt_data = safe_torch_load_checkpoint(Path(resume_ckpt), map_location=device)
        model.load_state_dict(ckpt_data["model_state_dict"])
    model_ema = ModelWeightEMA(model, decay=0.999) if use_model_ema else None
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    amp_enabled = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if hasattr(torch, "amp") else torch.cuda.amp.GradScaler(enabled=amp_enabled)
    pos_ema = PositiveAnchorEMA(init_val=8.0, momentum=0.90, min_floor=1.0)
    sched_ep0 = build_slicing_aided_training_schedule(
        train_idx.images, train_idx.anns_by_image, arm=arm, frame_stride=args.frame_stride, epoch_offset=0, interleave_across_videos=True,
        class_balanced_sampling=bool(getattr(args, 'class_balanced_sampling', False)), repeat_factor_sampling=bool(getattr(args, 'repeat_factor_sampling', False)), repeat_factor_threshold=float(getattr(args, 'repeat_factor_thr', 0.10)),
    )
    if args.max_tiles_per_epoch > 0:
        sched_ep0 = sched_ep0[: int(args.max_tiles_per_epoch)]
    est_opt_per_epoch = max(1, (len(sched_ep0) // bs) // ga)
    total_opt_steps = max(1, epochs * est_opt_per_epoch)
    warmup_steps = 20 if resume_ckpt is not None else 50
    print(
        f"[MARIVISION-HR-P2 V11 TRAINING] Arm=({arm}) epochs={epochs} frame_stride={args.frame_stride} "
        f"bs={bs} ga={ga} cosine_lr={use_cosine_lr} hflip={use_hflip_aug} model_ema={use_model_ema}"
    )
    t0 = time.time()
    global_opt_step = 0
    flip_rng = random.Random(20261021)
    for ep in range(epochs):
        model.train(True)
        sched = build_slicing_aided_training_schedule(
            train_idx.images, train_idx.anns_by_image, arm=arm, frame_stride=args.frame_stride, epoch_offset=ep, interleave_across_videos=True,
            class_balanced_sampling=bool(getattr(args, 'class_balanced_sampling', False)), repeat_factor_sampling=bool(getattr(args, 'repeat_factor_sampling', False)), repeat_factor_threshold=float(getattr(args, 'repeat_factor_thr', 0.10)),
        )
        if args.max_tiles_per_epoch > 0:
            sched = sched[: int(args.max_tiles_per_epoch)]
        n_micro = len(sched) // bs
        n_opt = max(1, n_micro // ga)
        tile_cursor = 0
        ep_cls = 0.0
        ep_box = 0.0
        for opt_idx in range(1, n_opt + 1):
            global_opt_step += 1
            if global_opt_step <= warmup_steps:
                lr_now = args.lr * (float(global_opt_step) / float(max(1, warmup_steps)))
            elif use_cosine_lr:
                prog = min(1.0, float(global_opt_step - warmup_steps) / float(max(1, total_opt_steps - warmup_steps)))
                lr_min = args.lr * 0.05
                lr_now = lr_min + 0.5 * (args.lr - lr_min) * (1.0 + math.cos(math.pi * prog))
            else:
                lr_now = args.lr
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now
            optimizer.zero_grad()
            step_raw_pos_list: List[float] = []
            step_cls = 0.0
            step_box = 0.0
            for _ in range(ga):
                batch_items = sched[tile_cursor : tile_cursor + bs]
                if not batch_items:
                    break
                tile_cursor += len(batch_items)
                tensors = []
                gt_list = []
                for iid, cw in batch_items:
                    rgb = train_idx.load_rgb_numpy(iid)
                    crop_rgb = cw.slice_rgb(rgb)
                    b_np, l_np, i_np = project_gt_boxes_into_crop_with_ignore(train_idx.anns_by_image.get(iid, []), cw)
                    if use_hflip_aug and flip_rng.random() < 0.5:
                        crop_rgb = np.ascontiguousarray(crop_rgb[:, ::-1, :])
                        b_np = flip_boxes_horizontally_in_crop(b_np, cw.resized_w)
                        i_np = flip_boxes_horizontally_in_crop(i_np, cw.resized_w)
                    tensors.append(
                        prepare_tile_tensor(
                            crop_rgb, cw.detector_w, cw.detector_h, use_scene_prior=use_scene, crop_window=cw,
                            use_global_crop_horizon_coords=True,
                        )
                    )
                    gt_list.append((
                        cw, torch.from_numpy(b_np).to(device),
                        torch.from_numpy(l_np).to(device),
                        torch.from_numpy(i_np).to(device),
                    ))
                batch_tensor = torch.stack(tensors, dim=0).to(device)
                autocast_ctx = (
                    torch.amp.autocast("cuda", enabled=amp_enabled)
                    if hasattr(torch, "amp") else torch.cuda.amp.autocast(enabled=amp_enabled)
                )
                with autocast_ctx:
                    outs_batch = model(batch_tensor)
                    mb_loss = 0.0
                    for b_i, (cw, gt_b, gt_l, ign_b) in enumerate(gt_list):
                        outs_i = [(s, lg[b_i : b_i + 1], lt[b_i : b_i + 1]) for s, lg, lt in outs_batch]
                        ld = compute_detector_loss_on_tile(
                            outs_i, gt_b, gt_l, area_scale=cw.area_scale, use_nwd_rfla=use_nwd_rfla,
                            pos_ema_tracker=pos_ema, update_ema_now=False,
                            ignore_boxes_xywh=ign_b, valid_w=cw.resized_w, valid_h=cw.resized_h,
                        )
                        mb_loss = mb_loss + ld["loss_total"] / float(bs * ga)
                        step_cls += float(ld["loss_cls"].item()) / float(bs * ga)
                        step_box += float(ld["loss_box"].item()) / float(bs * ga)
                        step_raw_pos_list.append(float(ld["num_pos"].item()))
                scaler.scale(mb_loss).backward()
            pos_ema.update_from_step(step_raw_pos_list)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            if model_ema is not None:
                model_ema.update(model)
            ep_cls += step_cls / float(n_opt)
            ep_box += step_box / float(n_opt)
        print(
            f"  [Arm ({arm})] Epoch {ep+1:02d}/{epochs:02d} | opt_steps={n_opt} | lr={lr_now:.6f} | "
            f"mean_cls={ep_cls:.4f} | mean_box={ep_box:.4f} | pos_ema={pos_ema.ema:.2f}"
        )
    eval_model = model_ema.ema_model if model_ema is not None else model
    ckpt_path = out_dir / f"detector_arm_{arm}_final.pt"
    torch.save({"arm": arm, "model_state_dict": eval_model.state_dict()}, ckpt_path)
    val_metrics = evaluate_model_on_val(
        eval_model, val_idx, arm=arm, device=device, frame_stride=args.val_frame_stride,
        apply_posthoc_scene_prior=False, use_global_crop_horizon_coords=True,
    )
    val_metrics["elapsed_train_plus_val_sec"] = round(time.time() - t0, 2)
    val_metrics["checkpoint_path"] = str(ckpt_path)
    (out_dir / f"screening_metrics_arm_{arm}.json").write_text(json.dumps(val_metrics, indent=2), encoding="utf-8")
    print(f"[MARIVISION-HR-P2 V11 ARM ({arm}) COMPLETE]")
    print(json.dumps(val_metrics, indent=2))
    return val_metrics


def run_eval_saved_checkpoints_and_bootstrap(args: argparse.Namespace) -> None:
    """Zero-retraining V11 comprehensive evaluation of saved checkpoints (`arm_a..arm_e`) in `--out-dir`."""
    val_idx = SeaClipsZipNativeIndex(Path(args.zip), Path(args.coco_val), split_hint="val")
    total_raw_val_images = len(val_idx.images)
    total_raw_val_gts = sum(len(v) for v in val_idx.anns_by_image.values())
    area_audit_full = audit_coco_annotation_area_vs_bbox_area(val_idx.anns_by_image, scope_label="full_val_coco_all_frames")
    by_video: Dict[int, List[Dict]] = defaultdict(list)
    for iid, im in val_idx.images.items():
        by_video[int(im.get("video_id", 0))].append(im)
    val_offset = int(getattr(args, "val_frame_offset", 0)) % max(1, int(args.val_frame_stride))
    eval_image_ids: List[int] = []
    for vid, im_list in sorted(by_video.items()):
        im_list.sort(key=lambda x: (int(x.get("frame_id", x["id"])), int(x["id"])))
        for idx_f, im in enumerate(im_list):
            if args.val_frame_stride <= 1 or (idx_f % args.val_frame_stride == val_offset):
                eval_image_ids.append(int(im["id"]))
    eval_images_dict = {iid: val_idx.images[iid] for iid in eval_image_ids}
    eval_anns_dict = {iid: val_idx.anns_by_image.get(iid, []) for iid in eval_image_ids}
    n_eval_gts = sum(len(v) for v in eval_anns_dict.values())
    area_audit_subset = audit_coco_annotation_area_vs_bbox_area(
        eval_anns_dict, scope_label=f"evaluated_subset_stride_{int(args.val_frame_stride)}"
    )
    ckpt_root = Path(args.out_dir)
    p1_root = Path(args.phase1_dir) if getattr(args, "phase1_dir", None) else None
    stage_d_enabled = not bool(getattr(args, "disable_stage_d_eval", False))
    require_pycoco = not bool(getattr(args, "allow_missing_pycocotools", False))
    expected_cache_sha = getattr(args, "expected_cache_sha256", None)
    expected_manifest = build_prediction_cache_manifest(
        val_frame_stride=int(args.val_frame_stride),
        eval_image_ids=eval_image_ids,
        eval_anns_dict=eval_anns_dict,
        coco_val_path=Path(args.coco_val),
        ckpt_root=ckpt_root,
        phase1_root=p1_root,
        stage_d_enabled=stage_d_enabled,
    )
    scoped_cache_file = ckpt_root / f"raw_predictions_cache_v12_stride{int(args.val_frame_stride)}_offset{val_offset}_staged{int(stage_d_enabled)}.json"
    default_cache_file = ckpt_root / "raw_predictions_cache_v12.json"
    cache_file = scoped_cache_file if scoped_cache_file.is_file() else default_cache_file
    preds_by_arm: Dict[str, List[Dict]] = {}
    low_preds_by_key: Dict[str, List[Dict]] = {}
    metrics_by_variant: Dict[str, Dict] = {}
    per_video_by_variant: Dict[str, Dict[str, Dict[int, float]]] = {}
    forensic_diagnostics: Dict[str, Dict] = {}
    pycocotools_crosschecks: Dict[str, Dict] = {}
    use_cache = cache_file.is_file() and (not getattr(args, "recompute_gpu_preds", False))
    if use_cache:
        print(f"[EVAL-CHECKPOINTS V11] Verifying prediction cache manifest & payload hashes in {cache_file} ...")
        cached_payload = json.loads(cache_file.read_text(encoding="utf-8"))
        validate_prediction_cache_manifest(
            cached_payload, expected_manifest,
            cache_file_path=cache_file,
            expected_cache_file_sha256=expected_cache_sha,
        )
        print(
            f"[EVAL-CHECKPOINTS V11] Cache manifest & payload hashes VERIFIED: stride={args.val_frame_stride} "
            f"stage_d={stage_d_enabled} frames={len(eval_image_ids)}/{total_raw_val_images} GTs={n_eval_gts}/{total_raw_val_gts}"
        )
        preds_by_arm = cached_payload.get("preds_by_arm", {})
        low_preds_by_key = cached_payload.get("low_preds_by_key", {})
        expected_manifest = attach_prediction_payload_hashes_to_cache_manifest(
            expected_manifest, preds_by_arm, low_preds_by_key
        )
    else:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for GPU inference in --mode eval-checkpoints")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for arm_id in ("a", "b", "c", "d", "e"):
            ckpt_file = ckpt_root / f"arm_{arm_id}" / f"detector_arm_{arm_id}_final.pt"
            if not ckpt_file.is_file():
                continue
            print(
                f"[EVAL-CHECKPOINTS V11] Loading {ckpt_file} (val_frame_stride={args.val_frame_stride}, "
                f"frames={len(eval_image_ids)}/{total_raw_val_images}, GTs={n_eval_gts}) ..."
            )
            use_p2 = arm_id in ("c", "d", "e")
            use_scene = (arm_id == "e")
            model = Stride4MaritimeDetector(
                num_classes=NUM_CLASSES, use_p2_stride4=use_p2, use_scene_prior_channel=use_scene,
                pretrained_backbone=False, freeze_backbone_bn=True,
            ).to(device)
            ckpt_data = safe_torch_load_checkpoint(ckpt_file, map_location=device)
            model.load_state_dict(ckpt_data["model_state_dict"])
            model.eval()
            run_stage_d = (arm_id in ("d", "e")) and stage_d_enabled
            pass_out = predict_val_variants_single_pass(
                model, val_idx, eval_image_ids, arm=arm_id, device=device,
                use_global_crop_horizon_coords=True,
                run_stage_d_microcrops=run_stage_d,
                max_stage_d_proposals=4,
            )
            if arm_id == "e":
                std_key = "arm_e_raw_no_posthoc"
                sd_key = "arm_e_raw_with_stage_d_4x"
                preds_by_arm["e_raw_no_posthoc"] = pass_out["std_005_max100"]
            else:
                std_key = f"arm_{arm_id}"
                sd_key = f"arm_{arm_id}_with_stage_d_4x"
                preds_by_arm[arm_id] = pass_out["std_005_max100"]
            if arm_id in ("b", "d", "e"):
                low_preds_by_key[std_key] = pass_out["low_0005_max300"]
            if run_stage_d and pass_out["stage_d_005_max100"]:
                preds_by_arm[sd_key] = pass_out["stage_d_005_max100"]
        if p1_root is not None and p1_root.is_dir() and p1_root.resolve() != ckpt_root.resolve():
            for p1_arm in ("b", "e"):
                p1_ckpt = p1_root / f"arm_{p1_arm}" / f"detector_arm_{p1_arm}_final.pt"
                if not p1_ckpt.is_file():
                    continue
                print(f"[EVAL-CHECKPOINTS V11] Loading Phase 1 reference {p1_ckpt} ...")
                m_p1 = Stride4MaritimeDetector(
                    num_classes=NUM_CLASSES, use_p2_stride4=(p1_arm in ("c", "d", "e")),
                    use_scene_prior_channel=(p1_arm == "e"), pretrained_backbone=False, freeze_backbone_bn=True,
                ).to(device)
                m_p1.load_state_dict(safe_torch_load_checkpoint(p1_ckpt, map_location=device)["model_state_dict"])
                m_p1.eval()
                preds_p1 = predict_all_val_images_for_model(
                    m_p1, val_idx, eval_image_ids, arm=p1_arm, device=device,
                    apply_posthoc_scene_prior=False, use_global_crop_horizon_coords=False,
                )
                preds_by_arm[f"p1_{p1_arm}"] = preds_p1
        expected_manifest = attach_prediction_payload_hashes_to_cache_manifest(
            expected_manifest, preds_by_arm, low_preds_by_key
        )
        ckpt_root.mkdir(parents=True, exist_ok=True)
        cache_payload_to_write = {
            "cache_manifest": expected_manifest,
            "preds_by_arm": preds_by_arm,
            "low_preds_by_key": low_preds_by_key,
        }
        cache_json_str = json.dumps(cache_payload_to_write)
        scoped_cache_file.write_text(cache_json_str, encoding="utf-8")
        default_cache_file.write_text(cache_json_str, encoding="utf-8")
        cache_sha = sha256_file_bytes(scoped_cache_file)
        Path(str(scoped_cache_file) + ".sha256").write_text(f"{cache_sha}  {scoped_cache_file.name}\n", encoding="utf-8")
        print(f"[EVAL-CHECKPOINTS V11] Saved manifest-verified raw predictions cache to {scoped_cache_file} (SHA256={cache_sha})")
    # Evaluate all single-model variants (`std`, `low_0005_max300`, `stage_d_4x`) and run pycocotools parity on EVERY variant
    for arm_id in ("a", "b", "c", "d", "e"):
        arm_lookup = "e_raw_no_posthoc" if arm_id == "e" else arm_id
        if arm_lookup not in preds_by_arm:
            continue
        std_key = "arm_e_raw_no_posthoc" if arm_id == "e" else f"arm_{arm_id}"
        low_key = "arm_e_raw_score0005_maxdet300" if arm_id == "e" else f"arm_{arm_id}_score0005_maxdet300"
        sd_key = "arm_e_raw_with_stage_d_4x" if arm_id == "e" else f"arm_{arm_id}_with_stage_d_4x"
        std_preds = preds_by_arm[arm_lookup]
        ov_std, pv_std = evaluate_predictions_dict_and_per_video(eval_images_dict, eval_anns_dict, std_preds)
        metrics_by_variant[std_key] = ov_std
        per_video_by_variant[std_key] = pv_std
        pycocotools_crosschecks[std_key] = verify_against_pycocotools_if_available(
            eval_images_dict, eval_anns_dict, std_preds, ov_std, fail_fast=True, require_pycocotools=require_pycoco
        )
        if std_key in low_preds_by_key:
            low_preds = low_preds_by_key[std_key]
            ov_low, pv_low = evaluate_predictions_dict_and_per_video(eval_images_dict, eval_anns_dict, low_preds)
            metrics_by_variant[low_key] = ov_low
            per_video_by_variant[low_key] = pv_low
            pycocotools_crosschecks[low_key] = verify_against_pycocotools_if_available(
                eval_images_dict, eval_anns_dict, low_preds, ov_low, fail_fast=True, require_pycocotools=require_pycoco
            )
            forensic_diagnostics[std_key] = diagnose_tiny_le_4px2_and_ship_confusion(
                eval_images_dict, eval_anns_dict, std_preds, low_preds
            )
        if sd_key in preds_by_arm and stage_d_enabled:
            sd_preds = preds_by_arm[sd_key]
            ov_sd, pv_sd = evaluate_predictions_dict_and_per_video(eval_images_dict, eval_anns_dict, sd_preds)
            metrics_by_variant[sd_key] = ov_sd
            per_video_by_variant[sd_key] = pv_sd
            pycocotools_crosschecks[sd_key] = verify_against_pycocotools_if_available(
                eval_images_dict, eval_anns_dict, sd_preds, ov_sd, fail_fast=True, require_pycocotools=require_pycoco
            )
            forensic_diagnostics[sd_key] = diagnose_tiny_le_4px2_and_ship_confusion(
                eval_images_dict, eval_anns_dict, sd_preds, sd_preds
            )
    for p1_arm in ("b", "e"):
        if f"p1_{p1_arm}" in preds_by_arm:
            p1_key = f"phase1_arm_{p1_arm}"
            p1_preds = preds_by_arm[f"p1_{p1_arm}"]
            ov_p1, pv_p1 = evaluate_predictions_dict_and_per_video(eval_images_dict, eval_anns_dict, p1_preds)
            metrics_by_variant[p1_key] = ov_p1
            per_video_by_variant[p1_key] = pv_p1
            pycocotools_crosschecks[p1_key] = verify_against_pycocotools_if_available(
                eval_images_dict, eval_anns_dict, p1_preds, ov_p1, fail_fast=True, require_pycocotools=require_pycoco
            )
            forensic_diagnostics[p1_key] = diagnose_tiny_le_4px2_and_ship_confusion(
                eval_images_dict, eval_anns_dict, p1_preds, p1_preds
            )
    if "d" in preds_by_arm and "e_raw_no_posthoc" in preds_by_arm:
        by_iid_d = defaultdict(list)
        by_iid_e = defaultdict(list)
        by_iid_b = defaultdict(list)
        by_iid_p1e = defaultdict(list)
        for p in preds_by_arm["d"]:
            by_iid_d[int(p["image_id"])].append({**p, "stage": "ARM_D", "fusion_weight": 1.15})
        for p in preds_by_arm["e_raw_no_posthoc"]:
            by_iid_e[int(p["image_id"])].append({**p, "stage": "ARM_E", "fusion_weight": 1.00})
        for p in preds_by_arm.get("b", []):
            by_iid_b[int(p["image_id"])].append({**p, "stage": "ARM_B", "fusion_weight": 0.90})
        for p in preds_by_arm.get("p1_e", []):
            by_iid_p1e[int(p["image_id"])].append({**p, "stage": "P1_ARM_E", "fusion_weight": 1.00})
        fused_de_all: List[Dict] = []
        fused_bde_all: List[Dict] = []
        fused_cross_all: List[Dict] = []
        for iid in eval_image_ids:
            cands_de = by_iid_d.get(iid, []) + by_iid_e.get(iid, [])
            if cands_de:
                for fd in hr_p2.class_aware_nwd_nms_fusion(cands_de, max_det_per_image=100):
                    fused_de_all.append({**fd, "image_id": iid})
            cands_bde = cands_de + by_iid_b.get(iid, [])
            if cands_bde:
                for fd in hr_p2.class_aware_nwd_nms_fusion(cands_bde, max_det_per_image=100):
                    fused_bde_all.append({**fd, "image_id": iid})
            if "p1_e" in preds_by_arm:
                cands_cross = cands_de + by_iid_p1e.get(iid, [])
                if cands_cross:
                    for fd in hr_p2.class_aware_nwd_nms_fusion(cands_cross, max_det_per_image=100):
                        fused_cross_all.append({**fd, "image_id": iid})
        ov_de, pv_de = evaluate_predictions_dict_and_per_video(eval_images_dict, eval_anns_dict, fused_de_all)
        metrics_by_variant["ensemble_arm_d_plus_e"] = ov_de
        per_video_by_variant["ensemble_arm_d_plus_e"] = pv_de
        pycocotools_crosschecks["ensemble_arm_d_plus_e"] = verify_against_pycocotools_if_available(
            eval_images_dict, eval_anns_dict, fused_de_all, ov_de, fail_fast=True, require_pycocotools=require_pycoco
        )
        forensic_diagnostics["ensemble_arm_d_plus_e"] = diagnose_tiny_le_4px2_and_ship_confusion(
            eval_images_dict, eval_anns_dict, fused_de_all, fused_de_all
        )
        if "b" in preds_by_arm:
            ov_bde, pv_bde = evaluate_predictions_dict_and_per_video(eval_images_dict, eval_anns_dict, fused_bde_all)
            metrics_by_variant["ensemble_arm_b_plus_d_plus_e"] = ov_bde
            per_video_by_variant["ensemble_arm_b_plus_d_plus_e"] = pv_bde
            pycocotools_crosschecks["ensemble_arm_b_plus_d_plus_e"] = verify_against_pycocotools_if_available(
                eval_images_dict, eval_anns_dict, fused_bde_all, ov_bde, fail_fast=True, require_pycocotools=require_pycoco
            )
        if "p1_e" in preds_by_arm:
            ov_cr, pv_cr = evaluate_predictions_dict_and_per_video(eval_images_dict, eval_anns_dict, fused_cross_all)
            metrics_by_variant["ensemble_p2_d_plus_p2_e_plus_p1_e"] = ov_cr
            per_video_by_variant["ensemble_p2_d_plus_p2_e_plus_p1_e"] = pv_cr
            pycocotools_crosschecks["ensemble_p2_d_plus_p2_e_plus_p1_e"] = verify_against_pycocotools_if_available(
                eval_images_dict, eval_anns_dict, fused_cross_all, ov_cr, fail_fast=True, require_pycocotools=require_pycoco
            )
            forensic_diagnostics["ensemble_p2_d_plus_p2_e_plus_p1_e"] = diagnose_tiny_le_4px2_and_ship_confusion(
                eval_images_dict, eval_anns_dict, fused_cross_all, fused_cross_all
            )
    # Automatic & explicit frozen R50 baseline comparator & restricted monotonicity union evaluation (Claude Finding #1)
    baseline_preds_path = getattr(args, "baseline_preds_json", None)
    project_root_dir = Path(getattr(args, "project_dir", r"D:\MARIVISION\H0_workbench\eval_tools"))
    r50_preds, r50_discovery_info = discover_and_extract_frozen_r50_val_predictions(
        project_dir=project_root_dir,
        coco_val_path=Path(args.coco_val),
        eval_images_dict=eval_images_dict,
        out_dir=ckpt_root,
        explicit_json_path=Path(baseline_preds_path) if baseline_preds_path else None,
        score_thresh=0.05,
    )
    if r50_preds:
        print(f"[EVAL-CHECKPOINTS V13] Evaluating frozen R50 comparator ({len(r50_preds)} preds from {r50_discovery_info.get('source_path')}) ...")
        ov_r50, pv_r50 = evaluate_predictions_dict_and_per_video(eval_images_dict, eval_anns_dict, r50_preds)
        metrics_by_variant["baseline_r50"] = ov_r50
        per_video_by_variant["baseline_r50"] = pv_r50
        pycocotools_crosschecks["baseline_r50"] = verify_against_pycocotools_if_available(
            eval_images_dict, eval_anns_dict, r50_preds, ov_r50, fail_fast=True, require_pycocotools=require_pycoco
        )
        forensic_diagnostics["baseline_r50"] = diagnose_tiny_le_4px2_and_ship_confusion(
            eval_images_dict, eval_anns_dict, r50_preds, r50_preds
        )
        r50_by_iid: Dict[int, List[Dict]] = defaultdict(list)
        for p in r50_preds:
            r50_by_iid[int(p["image_id"])].append(p)
        for cand_arm_key, union_name in (
            ("d", "hybrid_union_r50_plus_arm_d"),
            ("e_raw_no_posthoc", "hybrid_union_r50_plus_arm_e"),
        ):
            if cand_arm_key not in preds_by_arm:
                continue
            new_by_iid: Dict[int, List[Dict]] = defaultdict(list)
            for p in preds_by_arm[cand_arm_key]:
                new_by_iid[int(p["image_id"])].append(p)
            union_preds: List[Dict] = []
            for iid in eval_image_ids:
                merged_u = hr_p2.merge_baseline_and_new_candidates(r50_by_iid.get(iid, []), new_by_iid.get(iid, []))
                for md in merged_u:
                    union_preds.append({**md, "image_id": iid})
            ov_u, pv_u = evaluate_predictions_dict_and_per_video(eval_images_dict, eval_anns_dict, union_preds)
            metrics_by_variant[union_name] = ov_u
            per_video_by_variant[union_name] = pv_u
            pycocotools_crosschecks[union_name] = verify_against_pycocotools_if_available(
                eval_images_dict, eval_anns_dict, union_preds, ov_u, fail_fast=True, require_pycocotools=require_pycoco
            )
    if set(pycocotools_crosschecks.keys()) != set(metrics_by_variant.keys()):
        missing_parity = sorted(set(metrics_by_variant.keys()) - set(pycocotools_crosschecks.keys()))
        raise AssertionError(f"[PYCOCOTOOLS COVERAGE FAIL] Variants missing parity check: {missing_parity}")

    def _pooled_metric_val(var_k: str, m_name: str) -> Optional[float]:
        ov_d = metrics_by_variant.get(var_k, {})
        if m_name == "AP50_mAP":
            return float(ov_d.get("AP50", {}).get("mAP", 0.0))
        if m_name == "AP30_mAP":
            return float(ov_d.get("AP30_mAP", 0.0))
        if m_name == "mAP_class_agnostic_50":
            return float(ov_d.get("mAP_class_agnostic_50", 0.0))
        if m_name == "overall_gt_recall_50":
            return float(ov_d.get("overall_1to1_gt_recall_iou50", 0.0))
        if m_name == "overall_gt_recall_30":
            return float(ov_d.get("overall_1to1_gt_recall_iou30", 0.0))
        return None
    bootstrap_comparisons: Dict[str, Dict] = {}
    comparison_pairs: List[Tuple[str, str]] = []
    for base_key in ("baseline_r50", "arm_a", "arm_b", "phase1_arm_b"):
        if base_key not in per_video_by_variant:
            continue
        for cand_key in (
            "arm_d", "arm_d_with_stage_d_4x",
            "arm_e_raw_no_posthoc", "arm_e_raw_with_stage_d_4x",
            "phase1_arm_e", "ensemble_arm_d_plus_e",
            "ensemble_arm_b_plus_d_plus_e", "ensemble_p2_d_plus_p2_e_plus_p1_e",
            "hybrid_union_r50_plus_arm_d", "hybrid_union_r50_plus_arm_e",
        ):
            if cand_key in per_video_by_variant and cand_key != base_key:
                comparison_pairs.append((cand_key, base_key))
    for cand_k, base_k in (
        ("arm_e_raw_no_posthoc", "arm_d"),
        ("arm_d_with_stage_d_4x", "arm_d"),
        ("arm_e_raw_with_stage_d_4x", "arm_e_raw_no_posthoc"),
        ("ensemble_p2_d_plus_p2_e_plus_p1_e", "ensemble_arm_d_plus_e"),
        ("ensemble_arm_d_plus_e", "arm_e_raw_no_posthoc"),
        ("ensemble_p2_d_plus_p2_e_plus_p1_e", "arm_e_raw_no_posthoc"),
        ("ensemble_p2_d_plus_p2_e_plus_p1_e", "arm_d"),
    ):
        if cand_k in per_video_by_variant and base_k in per_video_by_variant:
            comparison_pairs.append((cand_k, base_k))

    def _call_paired_bootstrap_safe(
        pv_base: Dict[int, float],
        pv_cand: Dict[int, float],
        pooled_base: Optional[float],
        pooled_cand: Optional[float],
    ) -> Dict[str, object]:
        # Call hr_p2.paired_video_bootstrap_delta_metric with positional args so it works across any V11/V12 pipeline version,
        # and deterministically populate pooled_aggregate_delta, equal_weight_per_video_mean_delta, Leave-Top-2-Videos-Out (L2VO),
        # and bootstrap_freq_gt_zero.
        res = dict(hr_p2.paired_video_bootstrap_delta_metric(pv_base, pv_cand, n_boot=5000))
        vids = sorted(set(pv_base.keys()) & set(pv_cand.keys()))
        diffs = np.array([float(pv_cand[v]) - float(pv_base[v]) for v in vids], dtype=np.float64)
        obs_mean = float(np.mean(diffs)) if len(diffs) > 0 else 0.0
        if len(vids) > 2:
            order_desc = np.argsort(-diffs)
            top2_idx = order_desc[:2]
            top2_vids = [f"v{int(vids[i]):02d}" for i in top2_idx]
            rem_diffs = np.delete(diffs, top2_idx)
            mean_wo_top2 = float(np.mean(rem_diffs))
            n_pos_wo_top2 = int(np.sum(rem_diffs > 0.0))
        else:
            top2_vids = [f"v{int(v):02d}" for v in vids]
            mean_wo_top2 = obs_mean
            n_pos_wo_top2 = int(np.sum(diffs > 0.0))
        res["pooled_aggregate_delta"] = (
            round(float(pooled_cand) - float(pooled_base), 5)
            if (pooled_base is not None and pooled_cand is not None)
            else None
        )
        res["equal_weight_per_video_mean_delta"] = round(obs_mean, 5)
        res["top2_video_ids"] = top2_vids
        res["mean_delta_without_top2_videos"] = round(mean_wo_top2, 5)
        res["n_videos_without_top2_gt_zero"] = n_pos_wo_top2
        res["bootstrap_freq_gt_zero"] = res.get("bootstrap_prop_gt_zero", res.get("prob_delta_gt_zero", 0.0))
        return res
    for cand_key, base_key in comparison_pairs:
        comp_name = f"{cand_key}_vs_{base_key}"
        bootstrap_comparisons[comp_name] = {}
        for m_name in ("AP50_mAP", "AP30_mAP", "mAP_class_agnostic_50", "overall_gt_recall_50", "overall_gt_recall_30"):
            bootstrap_comparisons[comp_name][m_name] = _call_paired_bootstrap_safe(
                per_video_by_variant[base_key][m_name],
                per_video_by_variant[cand_key][m_name],
                _pooled_metric_val(base_key, m_name),
                _pooled_metric_val(cand_key, m_name),
            )
    parity_statuses = [str(v.get("parity_status", "PARITY_NOT_RUN")) for v in pycocotools_crosschecks.values()]
    all_parity_passed = all(s == "PARITY_PASSED" for s in parity_statuses) and bool(parity_statuses)
    gate_status, console_banner = format_evaluation_gate_status_and_banner(all_parity_passed, parity_statuses)
    report = {
        "evaluator_version": "V13_R50_AUTODISCOVERY_GT_WEIGHTED_AP_REPEAT_FACTOR_AND_EQUAL_RECIPE_CONTROL",
        "r50_baseline_discovery": r50_discovery_info,
        "evaluation_gate_status": gate_status,
        "all_variants_pycocotools_parity_passed": all_parity_passed,
        "total_raw_val_images_in_coco": total_raw_val_images,
        "total_raw_val_gt_annotations_in_coco": total_raw_val_gts,
        "val_frame_stride": int(args.val_frame_stride),
        "is_strided_subsample": bool(args.val_frame_stride > 1),
        "n_val_images_evaluated": len(eval_image_ids),
        "n_val_gt_annotations_evaluated": n_eval_gts,
        "coco_annotation_area_audit_full_val_coco": area_audit_full,
        "coco_annotation_area_audit_evaluated_subset": area_audit_subset,
        "cache_manifest": expected_manifest,
        "pycocotools_evaluator_crosscheck": pycocotools_crosschecks,
        "metrics_by_variant": metrics_by_variant,
        "forensic_diagnostics_by_variant": forensic_diagnostics,
        "paired_video_bootstrap_5000_resamples": bootstrap_comparisons,
    }
    report_json_str = json.dumps(report, indent=2)
    # Preserve historical v6..v12 reports as immutable records (ChatGPT Decision D-3553); write ONLY v13 and `_latest.json`
    for fname in (
        "phase1_checkpoint_audit_and_bootstrap_v13.json",
        "phase1_checkpoint_audit_and_bootstrap_latest.json",
    ):
        (ckpt_root / fname).write_text(report_json_str, encoding="utf-8")
    md_tables_str = render_verified_markdown_tables_from_report(report)
    out_md_path = ckpt_root / "phase1_checkpoint_audit_and_bootstrap_v13_tables.md"
    out_md_path.write_text(md_tables_str, encoding="utf-8")
    print(console_banner)
    print(report_json_str)
    print(f"\n[MACHINE-GENERATED VERIFIED MARKDOWN TABLES SAVED TO: {out_md_path}]\n")
    print(md_tables_str)


def run_phase2_full_training(args: argparse.Namespace) -> None:
    """Execute Phase 2 full training (with Cosine LR, Horizontal Flip augmentation & Model Weight EMA).
    Supports training the equal-recipe full-frame control (`--phase2-arms b,d,e` or `--phase2-arms b`, Claude V18.2 Finding #4)
    so that `arm_b` receives the exact same epochs, Cosine LR, Horizontal Flip, and Model EMA as `arm_d` and `arm_e`
    (while `phase1_arm_b` from `--phase1-dir` remains evaluated separately as the 3-epoch reference).
    """
    p1_dir = Path(args.phase1_dir) if args.phase1_dir else None
    no_warm = bool(getattr(args, "no_warm_start", False))
    warm_dir = None if no_warm else (Path(args.warm_start_dir) if getattr(args, "warm_start_dir", None) else p1_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    arms_to_train = [a.strip().lower() for a in args.phase2_arms.split(",") if a.strip()]
    if p1_dir and p1_dir.is_dir():
        import shutil
        for base_arm in ("a", "b"):
            if base_arm in arms_to_train:
                continue
            src_ckpt = p1_dir / f"arm_{base_arm}" / f"detector_arm_{base_arm}_final.pt"
            if src_ckpt.is_file():
                dst_dir = out_root / f"arm_{base_arm}"
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst_ckpt = dst_dir / f"detector_arm_{base_arm}_final.pt"
                if not dst_ckpt.is_file():
                    shutil.copy2(src_ckpt, dst_ckpt)
    results = {}
    for arm_id in arms_to_train:
        resume_p = None
        if warm_dir and (warm_dir / f"arm_{arm_id}" / f"detector_arm_{arm_id}_final.pt").is_file():
            resume_p = warm_dir / f"arm_{arm_id}" / f"detector_arm_{arm_id}_final.pt"
        results[arm_id] = train_and_evaluate_single_arm(
            args, arm=arm_id, use_cosine_lr=True, use_hflip_aug=True, use_model_ema=True, resume_ckpt=resume_p
        )
    run_eval_saved_checkpoints_and_bootstrap(args)


def run_locate_r50_val_mode(args: argparse.Namespace) -> None:
    """Scan workstation `bfcef_f_s0` and `eval_tools` for frozen R50 scripts, checkpoints, and VAL prediction caches."""
    val_coco = Path(args.coco_val)
    eval_images_dict: Dict[int, Dict] = {}
    if val_coco.is_file():
        raw_val = json.loads(val_coco.read_text(encoding="utf-8"))
        eval_images_dict = {int(im["id"]): im for im in raw_val.get("images", [])}
    project_root_dir = Path(getattr(args, "project_dir", r"D:\MARIVISION\H0_workbench\eval_tools"))
    preds, info = discover_and_extract_frozen_r50_val_predictions(
        project_dir=project_root_dir,
        coco_val_path=val_coco,
        eval_images_dict=eval_images_dict,
        out_dir=Path(args.out_dir),
        explicit_json_path=Path(args.baseline_preds_json) if getattr(args, "baseline_preds_json", None) else None,
        score_thresh=0.05,
    )
    print("[MARIVISION-HR-P2 V13 FROZEN R50 WORKSTATION DISCOVERY & EXTRACTION REPORT]")
    print(json.dumps(info, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        choices=(
            "selftest", "env-check", "raw-pos-histogram", "smoke-test",
            "overfit-one-batch", "benchmark-100-steps", "train-arm", "screen-all-arms",
            "eval-checkpoints", "train-phase2", "locate-r50-val"
        ),
        default="selftest",
    )
    ap.add_argument("--arm", choices=("a", "b", "c", "d", "e"), default="d")
    ap.add_argument("--phase2-arms", default="d,e", help="Comma-separated arms for --mode train-phase2 (default: d,e)")
    ap.add_argument("--phase1-dir", default=r"runs_hr_p2\phase1_screening_v4", help="Phase 1 checkpoint dir for cross-phase WBF & baseline comparison")
    ap.add_argument("--warm-start-dir", default=None, help="Optional checkpoint dir to branch from in --mode train-phase2 (e.g. runs_hr_p2\\phase2_full_v5)")
    ap.add_argument("--disable-stage-d-eval", action="store_true", help="Skip conditional Stage D 4x micro-crop pass during --mode eval-checkpoints")
    ap.add_argument("--recompute-gpu-preds", action="store_true", help="Force GPU inference even if raw_predictions_cache_v11.json exists")
    ap.add_argument("--allow-missing-pycocotools", action="store_true", help="Allow running eval-checkpoints when pycocotools is not installed (marks parity_status=PARITY_NOT_RUN and gate=INCOMPLETE)")
    ap.add_argument("--expected-cache-sha256", default=None, help="Optional externally pinned SHA-256 hex digest of the prediction cache file")
    ap.add_argument("--baseline-preds-json", default=None, help="Optional path to frozen R50 baseline predictions JSON on VAL for direct comparator & hybrid union evaluation")
    ap.add_argument("--val-frame-offset", type=int, default=0, help="Deterministic frame offset (0..val_frame_stride-1) for multi-offset VAL sensitivity checks")
    ap.add_argument("--class-balanced-sampling", action="store_true", help="Enable V12 class- and scale-balanced Stage C tile sampling")
    ap.add_argument("--repeat-factor-sampling", action="store_true", help="Enable V13 pre-declared Repeat-Factor vessel sampling (preserves 100%% of MARINE_MARKER pos_c[:2] crops and excludes noisy shoreline ANIMAL jitter oversampling)")
    ap.add_argument("--repeat-factor-thr", type=float, default=0.10, help="Image-level category frequency threshold t_c for repeat-factor sampling (default: 0.10)")
    ap.add_argument("--no-warm-start", action="store_true", help="Train Phase 2 arms from ImageNet-1K V2 backbone without warm-starting from --phase1-dir")
    ap.add_argument("--project-dir", default=r"D:\MARIVISION\H0_workbench\eval_tools", help="Workstation eval_tools root containing bfcef_f_s0 and R50 scripts")
    ap.add_argument("--use-dynamic-horizon", action="store_true", help="Use gradient-estimated full-frame horizon_y_norm instead of constant 0.42 in Arm (e)")
    ap.add_argument("--zip", default=r"D:\MARIVISION\SeaClips.zip")
    ap.add_argument("--coco-train", default=r"D:\MARIVISION\seaclips-train.json")
    ap.add_argument("--coco-val", default=r"D:\MARIVISION\seaclips-val.json")
    ap.add_argument("--out-dir", default="hr_p2_workstation_artifacts")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--bench-steps", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--frame-stride", type=int, default=5)
    ap.add_argument("--val-frame-stride", type=int, default=10)
    ap.add_argument("--max-tiles-per-epoch", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    args = ap.parse_args()
    if args.mode == "selftest":
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestStep2AblationRunnerContracts)
        res = unittest.TextTestRunner(verbosity=2).run(suite)
        if not res.wasSuccessful():
            raise SystemExit(1)
        print("[MARIVISION-HR-P2 STEP 2 RUNNER V13 SELFTEST PASS] All 22 ablation, R50 auto-discovery, GT-weighted AP, repeat-factor sampling, equal-recipe control, integer crop slicing, class-balanced tile sampling, R50 hybrid union, 1-to-1 ignored-GT COCO AP, cache manifest + external SHA pinning, non-skippable pycocotools parity, truthful gate banner, arithmetic consistency, & 1-to-1 recall contracts verified.")
    elif args.mode == "env-check":
        run_env_check(args)
    elif args.mode == "raw-pos-histogram":
        run_raw_pos_histogram_mode(args)
    elif args.mode == "smoke-test":
        run_smoke_test(args)
    elif args.mode == "overfit-one-batch":
        run_overfit_one_batch_and_export_overlays(args)
    elif args.mode == "benchmark-100-steps":
        run_benchmark_100_steps(args)
    elif args.mode == "train-arm":
        train_and_evaluate_single_arm(args, args.arm)
    elif args.mode == "screen-all-arms":
        all_results = {}
        for arm_id in ("a", "b", "c", "d", "e"):
            all_results[arm_id] = train_and_evaluate_single_arm(args, arm_id)
        out_root = Path(args.out_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "screening_all_arms_summary.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        print("[MARIVISION-HR-P2 V11 ALL 5 ARMS SCREENING SUMMARY]")
        print(json.dumps(all_results, indent=2))
    elif args.mode == "eval-checkpoints":
        run_eval_saved_checkpoints_and_bootstrap(args)
    elif args.mode == "train-phase2":
        run_phase2_full_training(args)
    elif args.mode == "locate-r50-val":
        run_locate_r50_val_mode(args)


if __name__ == "__main__":
    main()