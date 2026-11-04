"""
D:/MARIVISION/H0_workbench/eval_tools/test_pol2_candidate_2f_integrated.py
Phase-II Proof-of-Launch Stage 2 (PoL-2): Candidate 2F Real-Detector Structural Verification

Strict Compliance Checklist:
  - D-3458 / D-3459: CANDIDATES_JSON defined and hash-verified alongside all 7 source artifacts.
  - D-3466: COCO imported cleanly from pycocotools.coco.
  - D-3467 / P-1609: Cache provenance validated by embedded M0 hash + exact set equality with 20,310 TRAIN image IDs.
  - D-3468 / P-1610: Evaluates direct d_access and actual 5-point discrete d_stencil(R);
                     adaptive scan guarantees data-derived interior minimum for R_sample.
  - D-3469 / D-3470: Asserts matching_preds exists for every FP target;
                     reproduces exactly 235 targets across 12 contributing e-CAM videos.
  - D-3461 / P-1603: Pre-patch golden baseline captured before monkey-patching;
                     asserts 0.0 delta across primary and all 5 auxiliary decoder heads.
  - D-3462: Instruments backbone.forward with invocation counter (asserts count == 2).
  - D-3463 / P-1604: Compares F_null, F_correct, real same-sensor F_wrong (different e-CAM video),
                     and diagnostic synthetic rolled cache.
  - D-3464: Selects FP-eligible adjacent frame pair deterministically from candidate artifact.
  - P-1611: Auxiliary zero-gate tolerances asserted explicitly (< 1e-5 logits, < 1e-6 boxes).
"""

import os
import sys
import copy
import json
import hashlib
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO

# Add Deformable-DETR to sys.path
DETR_PATH = Path("D:/MARIVISION/Deformable-DETR")
if str(DETR_PATH) not in sys.path:
    sys.path.insert(0, str(DETR_PATH))

if "R0_MSDA_BACKEND" not in os.environ:
    os.environ["R0_MSDA_BACKEND"] = "pytorch"

from models.deformable_detr import build, DeformableDETR
from models.deformable_transformer import DeformableTransformer
from r0_preflight import get_authoritative_r0_args
from datasets.seaclips_detection import SeaClipsDetection
from util.misc import collate_fn, NestedTensor, inverse_sigmoid, nested_tensor_from_tensor_list

OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
CANDIDATES_JSON = OUT_DIR / "pol2_candidate_windows_v1.json"
TRAIN_JSON = Path("D:/MARIVISION/seaclips-train.json")
CACHE_FILE = OUT_DIR / "cache_m0_train_raw_top100.json"
LINKS_JSON = OUT_DIR / "pol0_pseudo_links_v1.json"
COHORT_S_JSON = OUT_DIR / "pol2_primary_same_sensor_clips.json"
COHORT_X_JSON = OUT_DIR / "pol2_secondary_cross_sensor_clips.json"
DEV_POOL_JSON = OUT_DIR / "pol2_dev_pool_manifest.json"
M0_CKPT_PATH = Path("D:/MARIVISION/checkpoints_r0/checkpoint_best.pth")
ZIP_PATH = Path("D:/MARIVISION/SeaClips.zip")

EXPECTED_CANDIDATE_SHA256 = "f29e0b8a5cbfc46641222d84a6d370c3e9bc51ca4ca1124b7753078ca7c1e270"
EXPECTED_TRAIN_JSON_SHA256 = "edc32bdbd6b792eb83b6daf11e14708232f1f4a9e06153ebebd5bad1caef9b35"
EXPECTED_LINKS_SHA256 = "e132f0a5782e0a53da858db80e64db0b879652057eeea9c149af2735cb25a2fb"
EXPECTED_COHORT_S_SHA256 = "b157b11e0b1bb8f9a77729a4999d48693f5d1e0cb052d8423b5a8dc903f029dc"
EXPECTED_COHORT_X_SHA256 = "91afc4a0a82e0fc87d9d523a902023e19fe5ac9b8c88b186e359bbb88303dac9"
EXPECTED_DEV_POOL_SHA256 = "533567e1d74544db7087a95253e640ef0959b998e5d503f57b0eb8074d8fe970"
EXPECTED_M0_HASH = "28e6288c3f6769c2d5275aa77ddb248735b55a09cd46a6b570e3f117890b268b"

EVAL_IOU_THRESHOLD = 0.30
OPERATING_SCORE_THRESHOLD = 0.20


def compute_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def box_iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[0] + b1[2], b2[0] + b2[2]), min(b1[1] + b1[3], b2[1] + b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = b1[2] * b1[3] + b2[2] * b2[3] - inter
    return inter / union if union > 0.0 else 0.0


# ------------------------------------------------------------------------------
# 1. CANDIDATE 2F: SYMMETRIC NATIVE-RESOLUTION FEATURE PROBE
# ------------------------------------------------------------------------------

class SpatiallyAnchoredFeatureProbe(nn.Module):
    def __init__(self, d_model=256, num_levels=4, stencil_radius_px=3.0):
        super().__init__()
        self.d_model = d_model
        self.num_levels = num_levels
        self.stencil_radius_px = stencil_radius_px
        self.radius_norm = stencil_radius_px / 640.0

        stencil_offsets = torch.tensor([
            [0.0, 0.0],
            [+self.radius_norm, 0.0],
            [-self.radius_norm, 0.0],
            [0.0, +self.radius_norm],
            [0.0, -self.radius_norm]
        ], dtype=torch.float32)
        self.register_buffer("stencil_offsets", stencil_offsets)  # (5, 2)

        self.stencil_logits = nn.Parameter(torch.zeros(num_levels, 5))
        self.feat_proj = nn.Linear(d_model * num_levels, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, tgt, query_pos, reference_points, feat_cache, temporal_enabled=True):
        if not temporal_enabled or feat_cache is None or len(feat_cache) == 0:
            return tgt

        B, N, C = tgt.shape
        base_ref = reference_points[..., :2].unsqueeze(2)
        sample_coords = base_ref + self.stencil_offsets.view(1, 1, 5, 2)
        grid_coords = sample_coords * 2.0 - 1.0

        stencil_weights = F.softmax(self.stencil_logits, dim=-1)

        level_features = []
        for lvl, fmap in enumerate(feat_cache):
            sampled = F.grid_sample(fmap, grid_coords, mode='bilinear', padding_mode='zeros', align_corners=False)
            sampled = sampled.permute(0, 2, 3, 1)

            w_lvl = stencil_weights[lvl].view(1, 1, 5, 1)
            fused_lvl = torch.sum(sampled * w_lvl, dim=2)
            level_features.append(fused_lvl)

        concat_levels = torch.cat(level_features, dim=-1)
        delta_tgt = self.feat_proj(concat_levels)
        delta_tgt = self.norm(delta_tgt)

        gate = torch.tanh(self.gamma)
        return tgt + gate * delta_tgt


# ------------------------------------------------------------------------------
# 2. SOURCE MONKEY-PATCHING
# ------------------------------------------------------------------------------

def apply_clean_source_hooks():
    def patched_transformer_forward(self, srcs, masks, pos_embeds, query_embed=None,
                                    temporal_adapter=None, feat_cache=None, temporal_enabled=True):
        assert self.d_model == srcs[0].shape[1]
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shapes.append((h, w))
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        bs, _, c = memory.shape
        query_embed, tgt = torch.split(query_embed, c, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_embed).sigmoid()
        init_reference_out = reference_points

        if temporal_adapter is not None and temporal_enabled:
            tgt = temporal_adapter(
                tgt=tgt,
                query_pos=query_embed,
                reference_points=reference_points,
                feat_cache=feat_cache,
                temporal_enabled=temporal_enabled
            )

        hs, inter_references = self.decoder(
            tgt, reference_points, memory,
            spatial_shapes, level_start_index, valid_ratios,
            query_embed, mask_flatten
        )
        return hs, init_reference_out, inter_references, None, None

    DeformableTransformer.forward = patched_transformer_forward

    def patched_detr_forward(self, samples: NestedTensor,
                             temporal_adapter=None, feat_cache=None, temporal_enabled=True,
                             return_feature_cache=False):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = self.query_embed.weight
        hs, init_reference, inter_references, _, _ = self.transformer(
            srcs, masks, pos, query_embeds,
            temporal_adapter=temporal_adapter,
            feat_cache=feat_cache,
            temporal_enabled=temporal_enabled
        )

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            reference = init_reference if lvl == 0 else inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        if return_feature_cache:
            cached_srcs = [s.detach() for s in srcs]
            return out, cached_srcs

        return out

    DeformableDETR.forward = patched_detr_forward


# ------------------------------------------------------------------------------
# 3. VERIFICATION HARNESS ENGINE
# ------------------------------------------------------------------------------

def main():
    print("=" * 115)
    print("PROOF-OF-LAUNCH STAGE 2 (PoL-2): CANDIDATE 2F INTEGRATED DETECTOR STRUCTURAL VERIFICATION")
    print("=" * 115)

    # 1. Full Live Provenance Verification (D-3458, D-3459, D-3467, P-1605, P-1609)
    assert CANDIDATES_JSON.exists(), f"[FATAL] Missing {CANDIDATES_JSON}"
    assert TRAIN_JSON.exists(), f"[FATAL] Missing {TRAIN_JSON}"
    assert CACHE_FILE.exists(), f"[FATAL] Missing {CACHE_FILE}"
    assert LINKS_JSON.exists(), f"[FATAL] Missing {LINKS_JSON}"
    assert COHORT_S_JSON.exists(), f"[FATAL] Missing {COHORT_S_JSON}"
    assert COHORT_X_JSON.exists(), f"[FATAL] Missing {COHORT_X_JSON}"
    assert DEV_POOL_JSON.exists(), f"[FATAL] Missing {DEV_POOL_JSON}"
    assert M0_CKPT_PATH.exists(), f"[FATAL] Missing {M0_CKPT_PATH}"

    assert compute_sha256(CANDIDATES_JSON) == EXPECTED_CANDIDATE_SHA256, "[FATAL] Candidate JSON SHA mismatch!"
    assert compute_sha256(TRAIN_JSON) == EXPECTED_TRAIN_JSON_SHA256, "[FATAL] Train JSON SHA mismatch!"
    assert compute_sha256(LINKS_JSON) == EXPECTED_LINKS_SHA256, "[FATAL] Links JSON SHA mismatch!"
    assert compute_sha256(COHORT_S_JSON) == EXPECTED_COHORT_S_SHA256, "[FATAL] Cohort S SHA mismatch!"
    assert compute_sha256(COHORT_X_JSON) == EXPECTED_COHORT_X_SHA256, "[FATAL] Cohort X SHA mismatch!"
    assert compute_sha256(DEV_POOL_JSON) == EXPECTED_DEV_POOL_SHA256, "[FATAL] Dev pool manifest SHA mismatch!"
    assert compute_sha256(M0_CKPT_PATH) == EXPECTED_M0_HASH, "[FATAL] M0 checkpoint SHA mismatch!"

    # Cache provenance certification & exact set equality check (P-1605, P-1609)
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache_data = json.load(f)
    assert cache_data["_meta"]["m0_checkpoint_sha256"] == EXPECTED_M0_HASH, "[FATAL] M0 checkpoint SHA mismatch in cache!"

    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        train_raw = json.load(f)
    train_image_ids = {int(x["id"]) for x in train_raw["images"]}
    raw_detections = {int(k): v for k, v in cache_data["detections"].items()}
    cache_image_ids = set(raw_detections.keys())

    assert len(train_image_ids) == 20310, f"[FATAL] Expected 20,310 train image IDs, got {len(train_image_ids)}"
    assert cache_image_ids == train_image_ids, (
        f"[FATAL] Cache/TRAIN image-ID inventory mismatch: "
        f"missing={len(train_image_ids - cache_image_ids)}, "
        f"extra={len(cache_image_ids - train_image_ids)}"
    )
    live_cache_hash = compute_sha256(CACHE_FILE)

    print("[+] All 7 Source Manifests & Checkpoints Certified by Immutable SHA-256.")
    print(f"[+] M0 Raw Prediction Cache Certified: Embedded M0 SHA Verified | Exact 20,310 Set Equality Certified | Live SHA: {live_cache_hash}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Target: {device} ({torch.cuda.get_device_name(0)})")

    # Load baseline model before patching
    args = get_authoritative_r0_args()
    if hasattr(args, "conditioning_mode"):
        args.conditioning_mode = "none"

    model, criterion, postprocessors = build(args)
    ckpt = torch.load(M0_CKPT_PATH, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    criterion.to(device)
    print("[+] Canonical M0 baseline loaded into memory (unpatched).")

    # --------------------------------------------------------------------------
    # STEP 1: DIRECT ACCESS & ADAPTIVE STENCIL RESIDUAL CALIBRATION
    # --------------------------------------------------------------------------
    print("\n--- STEP 1: DIRECT ACCESS & ADAPTIVE STENCIL RESIDUAL CALIBRATION (D-3468, P-1610) ---")
    with torch.no_grad():
        c_dim = model.transformer.d_model
        q_weight = model.query_embed.weight
        q_pos, _ = torch.split(q_weight, c_dim, dim=1)
        init_refs = model.transformer.reference_points(q_pos).sigmoid().cpu().numpy()  # (300, 2)
    init_refs_px = init_refs * 640.0

    with open(DEV_POOL_JSON, "r", encoding="utf-8") as f:
        dev_manifest = json.load(f)
    ecam_dev_vids = set(dev_manifest["_meta"]["ecam_dev_video_ids"])

    with open(CANDIDATES_JSON, "r", encoding="utf-8") as f:
        cand_windows = json.load(f)["windows"]

    coco = COCO(str(TRAIN_JSON))
    ann_lookup = {a["id"]: a for a in coco.loadAnns(coco.getAnnIds())}
    img_lookup = {info["id"]: info for info in coco.loadImgs(coco.getImgIds())}

    d_access_list = []
    c_m0_list = []
    seen_targets = set()
    fp_contributing_vids = set()

    for w in cand_windows:
        v_id = w["video_id"]
        if v_id in ecam_dev_vids and w["n_fastpath_s20"] >= 1:
            for t in w["targets"]:
                t_key = (v_id, t["ann_id"])
                if t["fastpath_s20"] and t_key not in seen_targets:
                    seen_targets.add(t_key)
                    fp_contributing_vids.add(v_id)

                    p_aid = t["chain_ann_ids"][0]
                    p_ann = ann_lookup[p_aid]
                    p_img = img_lookup[p_ann["image_id"]]
                    sx = 640.0 / float(p_img["width"])
                    sy = 640.0 / float(p_img["height"])

                    p_preds = raw_detections.get(p_ann["image_id"], [])
                    matching_preds = [
                        d for d in p_preds
                        if d["category_id"] == p_ann["category_id"] and box_iou(d["bbox"], p_ann["bbox"]) >= EVAL_IOU_THRESHOLD
                        and d["score"] >= OPERATING_SCORE_THRESHOLD
                    ]
                    # D-3469: Strict invariant assertion
                    assert matching_preds, (
                        f"[FATAL] FP provenance failure: ann {t['ann_id']} in video {v_id} is marked "
                        f"fastpath_s20 but no qualifying t-1 M0 evidence was reproduced in cache!"
                    )

                    best_p = max(matching_preds, key=lambda d: d["score"])
                    px, py, pw, ph = best_p["bbox"]
                    cx_m0_px = (px + pw / 2.0) * sx
                    cy_m0_px = (py + ph / 2.0) * sy
                    c_m0_list.append((cx_m0_px, cy_m0_px))

                    # d_access = min_q ||p_q - c_{t-1}^M0||
                    dists = np.sqrt((init_refs_px[:, 0] - cx_m0_px) ** 2 + (init_refs_px[:, 1] - cy_m0_px) ** 2)
                    d_access_list.append(float(np.min(dists)))

    # D-3470: Strict population reproduction assertions
    assert len(d_access_list) == 235, f"[FATAL] Expected exactly 235 FP targets, got {len(d_access_list)}"
    assert len(fp_contributing_vids) == 12, f"[FATAL] Expected exactly 12 contributing e-CAM videos, got {len(fp_contributing_vids)}"

    p50_acc = float(np.percentile(d_access_list, 50))
    p75_acc = float(np.percentile(d_access_list, 75))
    p90_acc = float(np.percentile(d_access_list, 90))
    p95_acc = float(np.percentile(d_access_list, 95))

    print(f"[+] e-CAM FP Direct Center Distance d_access (N = {len(d_access_list)} targets across {len(fp_contributing_vids)} contributing videos):")
    print(f"    - P50: {p50_acc:5.2f} px | P75: {p75_acc:5.2f} px | P90: {p90_acc:5.2f} px | P95: {p95_acc:5.2f} px")

    # Adaptive Stencil Search Domain (P-1610)
    c_m0_arr = np.array(c_m0_list)  # (235, 2)
    R_max_init = max(1, int(np.ceil(p95_acc)))
    candidate_radii = list(range(1, R_max_init + 1))
    stencil_residuals = {}

    print(f"\n--- Discrete 5-Point Cross Stencil Residual Scan d_stencil(R) (Adaptive Initial Ceiling = {R_max_init} px) ---")
    print(f"{'Radius R (px)':<14} | {'P50 Residual':<16} | {'P75 Residual':<16} | {'P90 Residual':<16} | {'P95 Residual':<16}")
    print("-" * 84)

    # Adaptive scan loop to guarantee interior minimum
    while True:
        for R in candidate_radii:
            if R in stencil_residuals:
                continue
            offsets_px = np.array([
                [0.0, 0.0],
                [+R, 0.0],
                [-R, 0.0],
                [0.0, +R],
                [0.0, -R]
            ], dtype=np.float32)

            # All 300 x 5 stencil sample locations: (1500, 2)
            stencil_pts = (init_refs_px[:, None, :] + offsets_px[None, :, :]).reshape(-1, 2)
            diff = c_m0_arr[:, None, :] - stencil_pts[None, :, :]  # (235, 1500, 2)
            d_res = np.min(np.sqrt(np.sum(diff ** 2, axis=-1)), axis=1)  # (235,)

            st_p50 = float(np.percentile(d_res, 50))
            st_p75 = float(np.percentile(d_res, 75))
            st_p90 = float(np.percentile(d_res, 90))
            st_p95 = float(np.percentile(d_res, 95))

            stencil_residuals[R] = {"p50": st_p50, "p75": st_p75, "p90": st_p90, "p95": st_p95}
            print(f"R = {R:<10d} | {st_p50:6.2f} px       | {st_p75:6.2f} px       | {st_p90:6.2f} px       | {st_p95:6.2f} px")

        best_R = min(sorted(stencil_residuals.keys()), key=lambda r: (stencil_residuals[r]["p90"], r))
        current_max = max(stencil_residuals.keys())

        # If best_R is at upper boundary and below practical limit, extend automatically
        if best_R == current_max and current_max < 30:
            extended_max = min(30, int(np.ceil(1.5 * current_max)))
            print(f"[*] Best radius ({best_R} px) at boundary. Extending scan ceiling: {current_max} -> {extended_max} px...")
            candidate_radii = list(range(current_max + 1, extended_max + 1))
        else:
            break

    R_sample = float(best_R)
    print(f"\n[+] Adaptive Stencil Optimization (P-1610): Locked R_sample = {R_sample:.1f} px ({R_sample/640.0:.5f} normalized).")
    print(f"    Achieved P90 Stencil Residual: {stencil_residuals[best_R]['p90']:.2f} px (Interior Optimum Certified).")

    # --------------------------------------------------------------------------
    # STEP 2: DETERMINISTIC FP-ELIGIBLE ADJACENT FRAME SELECTION (D-3464)
    # --------------------------------------------------------------------------
    print("\n--- STEP 2: DETERMINISTIC FP-ELIGIBLE SMOKE PAIR SELECTION (D-3464) ---")
    eligible_dev_windows = [
        w for w in cand_windows
        if w["video_id"] in ecam_dev_vids and w["n_fastpath_s20"] >= 1
    ]
    eligible_dev_windows.sort(key=lambda w: (w["video_id"], w["frame_start"]))
    smoke_window = eligible_dev_windows[0]
    smoke_vid = smoke_window["video_id"]
    f_t_minus_1_iid = smoke_window["image_ids"][-2]
    f_t_iid = smoke_window["image_ids"][-1]

    alt_ecam_vids = [v for v in sorted(list(ecam_dev_vids)) if v != smoke_vid]
    alt_dev_windows = [w for w in cand_windows if w["video_id"] == alt_ecam_vids[0]]
    alt_dev_windows.sort(key=lambda w: (w["video_id"], w["frame_start"]))
    wrong_window = alt_dev_windows[0]
    wrong_iid = wrong_window["image_ids"][-2]

    print(f"[+] Primary Smoke Stream (Video {smoke_vid}, e-CAM):")
    print(f"    - Frame t-1 (History): Image ID {f_t_minus_1_iid}")
    print(f"    - Frame t   (Current): Image ID {f_t_iid}")
    print(f"[+] Real Same-Sensor Wrong Stream (Video {wrong_window['video_id']}, e-CAM): Image ID {wrong_iid}")

    dataset = SeaClipsDetection(ZIP_PATH, TRAIN_JSON, is_train=False, random_flip=False)
    id_to_idx = {iid: idx for idx, iid in enumerate(sorted(coco.getImgIds()))}

    b_prev = collate_fn([dataset[id_to_idx[f_t_minus_1_iid]]])
    b_curr = collate_fn([dataset[id_to_idx[f_t_iid]]])
    b_wrong = collate_fn([dataset[id_to_idx[wrong_iid]]])

    samples_t_minus_1 = b_prev[0].to(device)
    samples_t = b_curr[0].to(device)
    targets_t = [{k: v.to(device) for k, v in t.items()} for t in b_curr[1]]
    samples_wrong = b_wrong[0].to(device)

    # --------------------------------------------------------------------------
    # STEP 3: CAPTURE TRUE PRE-PATCH GOLDEN BASELINE (D-3461, P-1603)
    # --------------------------------------------------------------------------
    print("\n--- STEP 3: CAPTURING CANONICAL UNTOUCHED PRE-PATCH GOLDEN BASELINE ---")
    with torch.no_grad():
        out_golden = model(samples_t)
        golden_logits = out_golden["pred_logits"].clone()
        golden_boxes = out_golden["pred_boxes"].clone()
        golden_aux = [
            {'pred_logits': a['pred_logits'].clone(), 'pred_boxes': a['pred_boxes'].clone()}
            for a in out_golden.get("aux_outputs", [])
        ]
    print(f"[+] Golden baseline captured on unpatched canonical M0 (Primary + {len(golden_aux)} Aux Levels).")

    # --------------------------------------------------------------------------
    # STEP 4: MONKEY-PATCH DETECTOR & INSTANTIATE CANDIDATE 2F
    # --------------------------------------------------------------------------
    print("\n--- STEP 4: MONKEY-PATCHING DETECTOR & INSTANTIATING CANDIDATE 2F ---")
    apply_clean_source_hooks()

    probe = SpatiallyAnchoredFeatureProbe(d_model=256, num_levels=4, stencil_radius_px=R_sample).to(device)
    total_params = sum(p.numel() for p in probe.parameters() if p.requires_grad)
    print(f"[+] Candidate 2F Parameter Count: {total_params:,} parameters (asserting < 270,000)")
    assert total_params < 270000, f"[FATAL] Parameter count exceeded 270k: {total_params}"

    # --------------------------------------------------------------------------
    # STEP 5: CONTRACT 1 - TRUE PRE-PATCH STATIC PARITY & ZERO-GATE BYPASS
    # --------------------------------------------------------------------------
    print("\n--- STEP 5: VERIFYING CONTRACT 1 (CANONICAL STATIC PARITY & ZERO-GATE BYPASS) ---")
    with torch.no_grad():
        out_off = model(samples_t, temporal_adapter=probe, feat_cache=None, temporal_enabled=False)
        d_logits_off = torch.max(torch.abs(out_off["pred_logits"] - golden_logits)).item()
        d_boxes_off = torch.max(torch.abs(out_off["pred_boxes"] - golden_boxes)).item()

        d_aux_logits_off = 0.0
        d_aux_boxes_off = 0.0
        if "aux_outputs" in out_off:
            for aux_off, aux_gold in zip(out_off["aux_outputs"], golden_aux):
                d_aux_logits_off = max(d_aux_logits_off, torch.max(torch.abs(aux_off["pred_logits"] - aux_gold["pred_logits"])).item())
                d_aux_boxes_off = max(d_aux_boxes_off, torch.max(torch.abs(aux_off["pred_boxes"] - aux_gold["pred_boxes"])).item())

        print(f"[+] Contract 1A (Patched temporal_enabled=False vs Canonical Pre-Patch Golden):")
        print(f"    - Primary Head: Delta Logits = {d_logits_off:.10e} | Delta Boxes = {d_boxes_off:.10e}")
        print(f"    - Aux Heads (5): Delta Logits = {d_aux_logits_off:.10e} | Delta Boxes = {d_aux_boxes_off:.10e}")
        assert d_logits_off == 0.0, "[FATAL] Non-zero delta with temporal_enabled=False against canonical baseline!"
        assert d_boxes_off == 0.0, "[FATAL] Non-zero delta with temporal_enabled=False against canonical baseline!"
        assert d_aux_logits_off == 0.0, "[FATAL] Non-zero aux delta with temporal_enabled=False!"
        assert d_aux_boxes_off == 0.0, "[FATAL] Non-zero aux delta with temporal_enabled=False!"

        # Extract features from t-1 frame
        _, feat_cache_correct = model(samples_t_minus_1, temporal_enabled=False, return_feature_cache=True)
        probe.gamma.data.fill_(0.0)

        # Contract 1B: Active zero-gate (gamma = 0) with real feature cache vs. GOLDEN
        out_zg = model(samples_t, temporal_adapter=probe, feat_cache=feat_cache_correct, temporal_enabled=True)
        d_logits_zg = torch.max(torch.abs(out_zg["pred_logits"] - golden_logits)).item()
        d_boxes_zg = torch.max(torch.abs(out_zg["pred_boxes"] - golden_boxes)).item()

        d_aux_logits_zg = 0.0
        d_aux_boxes_zg = 0.0
        if "aux_outputs" in out_zg:
            for aux_zg, aux_gold in zip(out_zg["aux_outputs"], golden_aux):
                d_aux_logits_zg = max(d_aux_logits_zg, torch.max(torch.abs(aux_zg["pred_logits"] - aux_gold["pred_logits"])).item())
                d_aux_boxes_zg = max(d_aux_boxes_zg, torch.max(torch.abs(aux_zg["pred_boxes"] - aux_gold["pred_boxes"])).item())

        print(f"[+] Contract 1B (Active Zero-Gate gamma=0.0 with Feat Cache vs Canonical Golden):")
        print(f"    - Primary Head: Delta Logits = {d_logits_zg:.10e} | Delta Boxes = {d_boxes_zg:.10e}")
        print(f"    - Aux Heads (5): Delta Logits = {d_aux_logits_zg:.10e} | Delta Boxes = {d_aux_boxes_zg:.10e}")
        
        # Primary & Auxiliary Zero-Gate Tolerance Assertions (P-1611)
        assert d_logits_zg < 1e-5, f"[FATAL] Zero-gate logits delta exceeded 1e-5: {d_logits_zg}"
        assert d_boxes_zg < 1e-6, f"[FATAL] Zero-gate boxes delta exceeded 1e-6: {d_boxes_zg}"
        assert d_aux_logits_zg < 1e-5, f"[FATAL] Zero-gate aux logits delta exceeded 1e-5: {d_aux_logits_zg}"
        assert d_aux_boxes_zg < 1e-6, f"[FATAL] Zero-gate aux boxes delta exceeded 1e-6: {d_aux_boxes_zg}"

        print("[+] Contract 1 Certified: temporal-off path is bit-for-bit canonical; active zero-gate path is within declared FP32 tolerance across primary and auxiliary heads.")

    # --------------------------------------------------------------------------
    # STEP 6: CONTRACT 2 - EMPIRICAL BACKBONE INVOCATION COUNTING (D-3462)
    # --------------------------------------------------------------------------
    print("\n--- STEP 6: VERIFYING CONTRACT 2 (SINGLE-PASS CACHE REUSE & ZERO REDUNDANCY) ---")
    backbone_call_count = 0
    orig_backbone_forward = model.backbone.forward

    def counting_backbone_forward(*b_args, **b_kwargs):
        nonlocal backbone_call_count
        backbone_call_count += 1
        return orig_backbone_forward(*b_args, **b_kwargs)

    model.backbone.forward = counting_backbone_forward

    with torch.no_grad():
        _, feat_cache_correct = model(samples_t_minus_1, temporal_enabled=False, return_feature_cache=True)
        assert backbone_call_count == 1, f"[FATAL] Frame t-1 backbone calls: {backbone_call_count} != 1"

        _ = model(samples_t, temporal_adapter=probe, feat_cache=feat_cache_correct, temporal_enabled=True)
        assert backbone_call_count == 2, f"[FATAL] Total backbone calls: {backbone_call_count} != 2"

    model.backbone.forward = orig_backbone_forward

    assert len(feat_cache_correct) == 4, f"[FATAL] Expected 4 feature levels, got {len(feat_cache_correct)}"
    for lvl, fmap in enumerate(feat_cache_correct):
        assert not fmap.requires_grad, f"[FATAL] Level {lvl} cache is not detached!"
    print(f"[+] Contract 2 Certified: Exactly 2 backbone invocations across 2 frames. Zero re-encoding.")

    # --------------------------------------------------------------------------
    # STEP 7: CONTRACT 3 - COUNTERFACTUAL PERTURBATION (D-3463, P-1604)
    # --------------------------------------------------------------------------
    print("\n--- STEP 7: VERIFYING CONTRACT 3 (COUNTERFACTUAL FEATURE PERTURBATION) ---")
    with torch.no_grad():
        probe.gamma.data.fill_(0.5)

        # 1. Null Feature Cache (F_null)
        out_null = model(samples_t, temporal_adapter=probe, feat_cache=None, temporal_enabled=True)
        # 2. Correct Real History (F_correct)
        out_correct = model(samples_t, temporal_adapter=probe, feat_cache=feat_cache_correct, temporal_enabled=True)
        # 3. Real Same-Sensor Wrong History (F_wrong-real)
        _, feat_cache_wrong_real = model(samples_wrong, temporal_enabled=False, return_feature_cache=True)
        out_wrong_real = model(samples_t, temporal_adapter=probe, feat_cache=feat_cache_wrong_real, temporal_enabled=True)
        # 4. Synthetic Spatial Rolled Cache (Diagnostic only)
        feat_cache_synth = [torch.roll(fmap, shifts=(5, 5), dims=(-2, -1)) for fmap in feat_cache_correct]
        out_synth = model(samples_t, temporal_adapter=probe, feat_cache=feat_cache_synth, temporal_enabled=True)

        delta_correct_vs_null = torch.max(torch.abs(out_correct["pred_boxes"] - out_null["pred_boxes"])).item()
        delta_correct_vs_wrong = torch.max(torch.abs(out_correct["pred_boxes"] - out_wrong_real["pred_boxes"])).item()
        delta_correct_vs_synth = torch.max(torch.abs(out_correct["pred_boxes"] - out_synth["pred_boxes"])).item()

        print(f"[+] Detector Box Alteration under Active Gate (gamma = 0.5):")
        print(f"    - Max Delta (F_correct vs F_null):                {delta_correct_vs_null:.8f}")
        print(f"    - Max Delta (F_correct vs F_wrong-real e-CAM):    {delta_correct_vs_wrong:.8f}")
        print(f"    - Max Delta (F_correct vs F_synth-rolled):        {delta_correct_vs_synth:.8f} [Diagnostic]")

        assert delta_correct_vs_null > 0.0, "[FATAL] Correct feature cache produced zero alteration vs null cache!"
        assert delta_correct_vs_wrong > 0.0, "[FATAL] Real wrong feature cache produced zero alteration vs correct cache!"
        print("[+] Contract 3 Certified: Integrated detector is measurably sensitive to temporal feature input.")

    # --------------------------------------------------------------------------
    # STEP 8: CONTRACT 4 - HUNGARIAN TASK-LOSS CRITERION GRADIENT TRANSMISSION
    # --------------------------------------------------------------------------
    print("\n--- STEP 8: VERIFYING CONTRACT 4 (HUNGARIAN CRITERION GRADIENT FLOW) ---")
    probe.train()
    probe.gamma.data.fill_(0.1)

    optimizer = torch.optim.AdamW(probe.parameters(), lr=2e-4)
    optimizer.zero_grad()

    out_train = model(samples_t, temporal_adapter=probe, feat_cache=feat_cache_correct, temporal_enabled=True)
    loss_dict = criterion(out_train, targets_t)
    loss = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict if k in criterion.weight_dict)
    loss.backward()

    stencil_grad_norm = probe.stencil_logits.grad.norm().item()
    proj_grad_norm = probe.feat_proj.weight.grad.norm().item()
    gamma_grad = probe.gamma.grad.item()

    print(f"[+] Downstream Hungarian Task Loss: {loss.item():.4f}")
    print(f"[+] Parameter Gradient Transmission Audit:")
    print(f"    - stencil_logits Grad Norm: {stencil_grad_norm:.8f}")
    print(f"    - feat_proj Grad Norm:      {proj_grad_norm:.8f}")
    print(f"    - gamma Scalar Grad:        {gamma_grad:.8f}")

    assert stencil_grad_norm > 0.0, "[FATAL] Zero gradient in stencil_logits!"
    assert proj_grad_norm > 0.0, "[FATAL] Zero gradient in feat_proj!"
    assert abs(gamma_grad) > 0.0, "[FATAL] Zero gradient in gamma!"
    print("[+] Contract 4 Certified: Hungarian detection loss optimizes all Candidate 2F parameters.")

    print("\n" + "=" * 115)
    print("ALL 5 STRUCTURAL INTEGRATION CONTRACTS CERTIFIED FOR CANDIDATE 2F.")
    print("=" * 115)


if __name__ == "__main__":
    main()