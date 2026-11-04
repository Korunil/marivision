"""
D:/MARIVISION/H0_workbench/eval_tools/train_pol2_candidate2f_micro.py
Phase-II Proof-of-Launch Stage 2 (PoL-2F): Multi-Video Learned Micro-Falsification Harness

Strict Compliance Checklist:
  - D-3482: Reuses Candidate-1 optimizer recipe (AdamW, weights lr=2e-4, gamma lr=1e-2, wd=1e-4).
  - D-3512 / P-1631: Exact top-100 flattened query x class ranking semantics matching frozen M0 cache.
  - D-3522: Validates calibration static miss reproduction (0/40) under top-100 evaluator before training.
  - D-3513 / D-3514 / P-1633: Trains on frames k=1..5 (5 updates/epoch), updating directly on frame t.
  - D-3506 / P-1626: Hard E* = 0 gate: terminates before touching test data if budget is zero.
  - D-3519 / P-1636 / P-1638: Retrains all 3 seeds on 10 videos, saves and hashes all 3 final models
                              to pol2_candidate2f_final_models.pth BEFORE loading or evaluating test data.
  - D-3520 / P-1637: Freezes final 10-video refit schedule to pol2_candidate2f_final_refit_schedule.json.
  - D-3507 / D-3508 / P-1628: Predeclared video-level mechanism gate: Delta_corr > 0 and Delta_static > 0
                              on both video_7 and video_30 independently.
  - D-3509: Asserts static condition D reproduces exactly 0/40 rescues on test targets before temporal analysis.
  - D-3527 / P-1641: Asserts live evaluated test targets strictly match n_expected_tgts per video.
  - D-3526 / P-1640: Explicit wrong-donor lineage: records donor eval frame/image and donor prev frame/image.
  - D-3521 / P-1639: Descriptive target-rescue analysis: per-seed counts, means, delta, and multi-seed frequency.
  - P-1627: Stores exact live split-manifest SHA, cache SHA, checkpoint SHA, and probe weights SHA.
"""

import os
import sys
import copy
import json
import hashlib
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import defaultdict
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
DEV_POOL_JSON = OUT_DIR / "pol2_dev_pool_manifest.json"
SPLIT_MANIFEST_JSON = OUT_DIR / "pol2_candidate2f_micro_split.json"
M0_CKPT_PATH = Path("D:/MARIVISION/checkpoints_r0/checkpoint_best.pth")
ZIP_PATH = Path("D:/MARIVISION/SeaClips.zip")

INIT_WEIGHTS_PTH = OUT_DIR / "pol2_candidate2f_initializations.pth"
SCHEDULE_JSON = OUT_DIR / "pol2_candidate2f_train_schedule.json"
REFIT_SCHEDULE_JSON = OUT_DIR / "pol2_candidate2f_final_refit_schedule.json"
FINAL_MODELS_PTH = OUT_DIR / "pol2_candidate2f_final_models.pth"
RESULTS_JSON = OUT_DIR / "pol2_candidate2f_micro_results.json"

EXPECTED_CANDIDATE_SHA256 = "f29e0b8a5cbfc46641222d84a6d370c3e9bc51ca4ca1124b7753078ca7c1e270"
EXPECTED_TRAIN_JSON_SHA256 = "edc32bdbd6b792eb83b6daf11e14708232f1f4a9e06153ebebd5bad1caef9b35"
EXPECTED_LINKS_SHA256 = "e132f0a5782e0a53da858db80e64db0b879652057eeea9c149af2735cb25a2fb"
EXPECTED_DEV_POOL_SHA256 = "533567e1d74544db7087a95253e640ef0959b998e5d503f57b0eb8074d8fe970"
EXPECTED_SPLIT_SHA256 = "646a8be84aea0ff94a5f4642d81a0210000b8e34e6fded72687d55639c96ecb1"
EXPECTED_M0_HASH = "28e6288c3f6769c2d5275aa77ddb248735b55a09cd46a6b570e3f117890b268b"

SEEDS = [41, 42, 43]
MAX_CALIB_EPOCHS = 25
STENCIL_RADIUS_PX = 18.0
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------------------
# 1. CANDIDATE 2F MODULE & SURGICAL SOURCE HOOKS
# ------------------------------------------------------------------------------

class SpatiallyAnchoredFeatureProbe(nn.Module):
    def __init__(self, d_model=256, num_levels=4, stencil_radius_px=18.0):
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
        self.register_buffer("stencil_offsets", stencil_offsets)

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
# 2. EXACT TOP-100 RESCUE EVALUATION HELPER (P-1631)
# ------------------------------------------------------------------------------

def evaluate_video_frame(model, criterion, samples, targets_dict, eval_targets, feat_cache, temporal_adapter, temporal_enabled=True):
    """
    Evaluates a single frame using canonical DETR top-100 query x class ranking semantics (P-1631).
    Targets in eval_targets have bboxes in original image coordinates [bx, by, bw, bh].
    Predictions are scaled to original image coordinates using targets_dict['orig_size'].
    """
    out = model(samples, temporal_adapter=temporal_adapter, feat_cache=feat_cache, temporal_enabled=temporal_enabled)
    loss_dict = criterion(out, [targets_dict])
    loss = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict if k in criterion.weight_dict).item()

    # Exact Top-100 DETR ranking semantics (P-1631)
    prob = out["pred_logits"][0].sigmoid()  # (300, num_classes)
    num_classes = prob.shape[-1]
    topk_scores, topk_idx = torch.topk(prob.flatten(), 100)
    query_idx = torch.div(topk_idx, num_classes, rounding_mode="floor")
    topk_labels = topk_idx % num_classes
    topk_boxes = out["pred_boxes"][0][query_idx]

    # Convert to original image coordinates xywh
    img_h, img_w = targets_dict["orig_size"][0].item(), targets_dict["orig_size"][1].item()
    pred_x1 = (topk_boxes[:, 0] - topk_boxes[:, 2] / 2.0) * img_w
    pred_y1 = (topk_boxes[:, 1] - topk_boxes[:, 3] / 2.0) * img_h
    pred_w = topk_boxes[:, 2] * img_w
    pred_h = topk_boxes[:, 3] * img_h
    pred_boxes_xywh = torch.stack([pred_x1, pred_y1, pred_w, pred_h], dim=-1).cpu().numpy()
    pred_scores_np = topk_scores.cpu().numpy()
    pred_cats_np = topk_labels.cpu().numpy()

    rescued_map = {}
    for t in eval_targets:
        aid = t["ann_id"]
        gt_box = t["bbox"]  # [x, y, w, h] in original image coordinates
        gt_cat = t["category_id"]

        is_rescued = False
        for p_idx in range(100):
            if pred_scores_np[p_idx] >= OPERATING_SCORE_THRESHOLD and pred_cats_np[p_idx] == gt_cat:
                iou = box_iou(gt_box, pred_boxes_xywh[p_idx])
                if iou >= EVAL_IOU_THRESHOLD:
                    is_rescued = True
                    break
        rescued_map[aid] = is_rescued

    return loss, rescued_map


# ------------------------------------------------------------------------------
# 3. MAIN HARNESS ENGINE
# ------------------------------------------------------------------------------

def main():
    print("=" * 115)
    print("PROOF-OF-LAUNCH STAGE 2 (PoL-2F): MULTI-VIDEO LEARNED MICRO-FALSIFICATION HARNESS")
    print("=" * 115)

    # 1. Full Live Provenance Verification (P-1627)
    assert CANDIDATES_JSON.exists(), f"[FATAL] Missing {CANDIDATES_JSON}"
    assert TRAIN_JSON.exists(), f"[FATAL] Missing {TRAIN_JSON}"
    assert CACHE_FILE.exists(), f"[FATAL] Missing {CACHE_FILE}"
    assert LINKS_JSON.exists(), f"[FATAL] Missing {LINKS_JSON}"
    assert DEV_POOL_JSON.exists(), f"[FATAL] Missing {DEV_POOL_JSON}"
    assert SPLIT_MANIFEST_JSON.exists(), f"[FATAL] Missing {SPLIT_MANIFEST_JSON}"
    assert M0_CKPT_PATH.exists(), f"[FATAL] Missing {M0_CKPT_PATH}"

    cand_hash = compute_sha256(CANDIDATES_JSON)
    train_hash = compute_sha256(TRAIN_JSON)
    links_hash = compute_sha256(LINKS_JSON)
    dev_pool_hash = compute_sha256(DEV_POOL_JSON)
    split_hash = compute_sha256(SPLIT_MANIFEST_JSON)
    m0_hash = compute_sha256(M0_CKPT_PATH)

    assert cand_hash == EXPECTED_CANDIDATE_SHA256, "[FATAL] Candidate SHA mismatch!"
    assert train_hash == EXPECTED_TRAIN_JSON_SHA256, "[FATAL] Train JSON SHA mismatch!"
    assert links_hash == EXPECTED_LINKS_SHA256, "[FATAL] Links SHA mismatch!"
    assert dev_pool_hash == EXPECTED_DEV_POOL_SHA256, "[FATAL] Dev pool SHA mismatch!"
    assert split_hash == EXPECTED_SPLIT_SHA256, "[FATAL] Split manifest SHA mismatch!"
    assert m0_hash == EXPECTED_M0_HASH, "[FATAL] M0 checkpoint SHA mismatch!"

    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache_data = json.load(f)
    assert cache_data["_meta"]["m0_checkpoint_sha256"] == EXPECTED_M0_HASH, "[FATAL] M0 checkpoint SHA mismatch in cache!"

    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        train_raw = json.load(f)
    train_image_ids = {int(x["id"]) for x in train_raw["images"]}
    raw_detections = {int(k): v for k, v in cache_data["detections"].items()}
    assert len(train_image_ids) == 20310
    assert set(raw_detections.keys()) == train_image_ids, "[FATAL] Cache image ID set mismatch against TRAIN JSON!"
    live_cache_hash = compute_sha256(CACHE_FILE)

    with open(SPLIT_MANIFEST_JSON, "r", encoding="utf-8") as f:
        split_meta = json.load(f)["_meta"]

    train_vids = split_meta["train_videos"]
    calib_vids = split_meta["calib_videos"]
    test_vids = split_meta["test_videos"]

    print("[+] All Upstream Manifests & Prediction Cache Certified by Immutable SHA-256.")
    print(f"[+] Loaded Frozen 8/2/2 Partition ({split_meta['artifact_name']}):")
    print(f"    - Split Manifest SHA-256:          {split_hash}")
    print(f"    - Training Pool (8 Videos):        {train_vids} ({split_meta['n_train_targets']} targets)")
    print(f"    - Calibration Set (2 Videos):     {calib_vids} ({split_meta['n_calib_targets']} targets, {split_meta['n_calib_unique_frames']} frames)")
    print(f"    - Optimization-Untouched Test:     {test_vids} ({split_meta['n_test_targets']} targets, {split_meta['n_test_unique_frames']} frames)")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Target: {device} ({torch.cuda.get_device_name(0)})")

    apply_clean_source_hooks()

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
    print("[+] Baseline M0 loaded; all backbone, transformer, and head parameters strictly frozen.")

    dataset = SeaClipsDetection(ZIP_PATH, TRAIN_JSON, is_train=False, random_flip=False)
    coco = dataset.coco
    img_ids = sorted(coco.getImgIds())
    id_to_idx = {iid: idx for idx, iid in enumerate(img_ids)}

    with open(CANDIDATES_JSON, "r", encoding="utf-8") as f:
        all_windows = json.load(f)["windows"]

    windows_by_vid = defaultdict(list)
    for w in all_windows:
        if w["video_id"] in train_vids + calib_vids + test_vids and w["n_fastpath_s20"] >= 1:
            windows_by_vid[w["video_id"]].append(w)

    for v in train_vids:
        assert len(windows_by_vid[v]) >= 1, f"[FATAL] Video {v} has no training windows!"

    # --------------------------------------------------------------------------
    # STAGE 0: REPRODUCIBILITY LOCK & SCHEDULES (P-1620, P-1621, P-1637)
    # --------------------------------------------------------------------------
    print("\n--- STAGE 0: PRE-FLIGHT REPRODUCIBILITY LOCK (P-1620, P-1621, P-1637) ---")
    initial_probe_weights = {}
    train_schedule = {}
    refit_schedule = {}
    all_10_vids = sorted(train_vids + calib_vids)

    for s in SEEDS:
        set_seed(s)
        probe_init = SpatiallyAnchoredFeatureProbe(d_model=256, num_levels=4, stencil_radius_px=STENCIL_RADIUS_PX)
        probe_init.gamma.data.fill_(0.0)
        initial_probe_weights[str(s)] = {k: v.detach().cpu().clone() for k, v in probe_init.state_dict().items()}

        # 1. 8-video calibration training window schedule (P-1621)
        rng = np.random.default_rng(s * 1000 + 42)
        sched_epochs = []
        for e in range(1, MAX_CALIB_EPOCHS + 1):
            epoch_wins = {}
            for v in train_vids:
                n_avail = len(windows_by_vid[v])
                chosen_idx = int(rng.integers(0, n_avail))
                epoch_wins[str(v)] = {
                    "window_index": chosen_idx,
                    "frame_start": windows_by_vid[v][chosen_idx]["frame_start"],
                    "frame_end": windows_by_vid[v][chosen_idx]["frame_end"],
                    "image_ids": windows_by_vid[v][chosen_idx]["image_ids"]
                }
            sched_epochs.append(epoch_wins)
        train_schedule[str(s)] = sched_epochs

        # 2. 10-video final refit window schedule (P-1637)
        rng_refit = np.random.default_rng(s * 2000 + 84)
        refit_epochs = []
        for e in range(1, MAX_CALIB_EPOCHS + 1):
            refit_wins = {}
            for v in all_10_vids:
                n_avail = len(windows_by_vid[v])
                chosen_idx = int(rng_refit.integers(0, n_avail))
                refit_wins[str(v)] = {
                    "window_index": chosen_idx,
                    "frame_start": windows_by_vid[v][chosen_idx]["frame_start"],
                    "frame_end": windows_by_vid[v][chosen_idx]["frame_end"],
                    "image_ids": windows_by_vid[v][chosen_idx]["image_ids"]
                }
            refit_epochs.append(refit_wins)
        refit_schedule[str(s)] = refit_epochs

    torch.save(initial_probe_weights, INIT_WEIGHTS_PTH)
    init_hash = compute_sha256(INIT_WEIGHTS_PTH)
    print(f"[+] Initial Probe Weights Frozen (CPU):   {INIT_WEIGHTS_PTH.resolve()}")
    print(f"    Initial Weights SHA-256:               {init_hash}")

    with open(SCHEDULE_JSON, "w", encoding="utf-8") as f:
        json.dump(train_schedule, f, indent=2)
    sched_hash = compute_sha256(SCHEDULE_JSON)
    print(f"[+] Training Window Schedule Frozen:       {SCHEDULE_JSON.resolve()}")
    print(f"    Window Schedule SHA-256:               {sched_hash}")

    with open(REFIT_SCHEDULE_JSON, "w", encoding="utf-8") as f:
        json.dump(refit_schedule, f, indent=2)
    refit_sched_hash = compute_sha256(REFIT_SCHEDULE_JSON)
    print(f"[+] Final Refit Window Schedule Frozen:    {REFIT_SCHEDULE_JSON.resolve()}")
    print(f"    Refit Schedule SHA-256:                {refit_sched_hash}")

    # Load calibration frame records with explicit chronological ordering (P-1630)
    with open(SPLIT_MANIFEST_JSON, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    calib_tgts_by_img = defaultdict(list)
    for t in split_data["calibration_targets"]:
        calib_tgts_by_img[t["image_id"]].append(t)

    calib_frame_data = defaultdict(list)
    for iid, tgts in calib_tgts_by_img.items():
        img_info = coco.loadImgs([iid])[0]
        v_id = img_info["video_id"]
        fid = img_info["frame_id"]
        prev_imgs = [m for m in coco.loadImgs(coco.getImgIds()) if m["video_id"] == v_id and m["frame_id"] == fid - 1]
        assert len(prev_imgs) == 1, f"[FATAL] Missing prior frame for calib image {iid}"
        prev_iid = prev_imgs[0]["id"]
        calib_frame_data[v_id].append({
            "image_id": iid,
            "frame_id": fid,
            "prev_image_id": prev_iid,
            "targets": tgts
        })

    for v in calib_vids:
        calib_frame_data[v].sort(key=lambda x: x["frame_id"])

    print(f"[+] Loaded Chronologically Ordered Calibration Evaluation Sets:")
    print(f"    - video_{calib_vids[0]}: {len(calib_frame_data[calib_vids[0]])} unique evaluation frames")
    print(f"    - video_{calib_vids[1]}: {len(calib_frame_data[calib_vids[1]])} unique evaluation frames")

    # --------------------------------------------------------------------------
    # PRE-FLIGHT NON-TEST EVALUATOR REPRODUCTION CHECK (D-3522, P-1632)
    # --------------------------------------------------------------------------
    print("\n--- PRE-FLIGHT NON-TEST EVALUATOR REPRODUCTION CHECK (D-3522, P-1632) ---")
    probe_test = SpatiallyAnchoredFeatureProbe(d_model=256, num_levels=4, stencil_radius_px=STENCIL_RADIUS_PX).to(device)
    probe_test.gamma.data.fill_(0.0)
    probe_test.eval()

    with torch.no_grad():
        calib_resc_D_total = 0
        calib_tgts_total = 0
        for v in calib_vids:
            for fd in calib_frame_data[v]:
                s_curr = collate_fn([dataset[id_to_idx[fd["image_id"]]]])
                tgt_dict = {k: v_t.to(device) for k, v_t in s_curr[1][0].items()}
                _, r_map = evaluate_video_frame(model, criterion, s_curr[0].to(device), tgt_dict, fd["targets"], None, None, False)
                for t in fd["targets"]:
                    calib_tgts_total += 1
                    if r_map.get(t["ann_id"], False):
                        calib_resc_D_total += 1

    print(f"[+] Pre-Flight Static Condition D Rescues on Calibration Targets: {calib_resc_D_total}/{calib_tgts_total}")
    assert calib_resc_D_total == 0, (
        f"[FATAL] Live top-100 evaluator rescued {calib_resc_D_total} targets on static condition D! "
        f"Evaluator does not reproduce frozen M0 miss semantics."
    )
    print("[+] Frozen M0 miss-status reproduction verified for all 40 calibration targets under canonical top-100 evaluator (D-3522).")

    # --------------------------------------------------------------------------
    # STAGE 1: MULTI-SCENE BUDGET CALIBRATION (0..25 EPOCHS) (P-1613, D-3513)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("STAGE 1: MULTI-SCENE BUDGET CALIBRATION (8 TRAIN VIDEOS -> 2 CALIB VIDEOS)")
    print("Optimization: 5 Synchronized Updates per Epoch (k=1..5, updating directly on frame t) (D-3513)")
    print("=" * 115)

    tri_seed_calib_loss_curves = {}
    tri_seed_calib_rescue_curves = {}
    tri_seed_first_grads = {}

    for seed in SEEDS:
        print(f"\n[*] Initializing Calibration Scan for Seed {seed}...")
        set_seed(seed)

        probe = SpatiallyAnchoredFeatureProbe(d_model=256, num_levels=4, stencil_radius_px=STENCIL_RADIUS_PX).to(device)
        probe.load_state_dict(initial_probe_weights[str(seed)])
        probe.to(device)

        # Two-LR AdamW recipe
        optimizer = torch.optim.AdamW([
            {'params': [probe.feat_proj.weight, probe.feat_proj.bias, probe.stencil_logits, probe.norm.weight, probe.norm.bias], 'lr': 2e-4},
            {'params': [probe.gamma], 'lr': 1e-2}
        ], weight_decay=1e-4)

        calib_loss_curve = []
        calib_rescue_curve = []

        # --- EVALUATE EPOCH 0 (Untrained Base Baseline) ---
        probe.eval()
        with torch.no_grad():
            v_losses_0 = []
            rescued_0_total = 0
            total_tgts_0 = 0

            for v in calib_vids:
                v_frame_losses = []
                for fd in calib_frame_data[v]:
                    s_curr = collate_fn([dataset[id_to_idx[fd["image_id"]]]])
                    s_prev = collate_fn([dataset[id_to_idx[fd["prev_image_id"]]]])

                    _, feat_cache_prev = model(s_prev[0].to(device), temporal_enabled=False, return_feature_cache=True)
                    tgt_dict = {k: v_t.to(device) for k, v_t in s_curr[1][0].items()}

                    loss_f, resc_m = evaluate_video_frame(
                        model, criterion, s_curr[0].to(device), tgt_dict, fd["targets"],
                        feat_cache=feat_cache_prev, temporal_adapter=probe, temporal_enabled=True
                    )
                    v_frame_losses.append(loss_f)

                    for t in fd["targets"]:
                        total_tgts_0 += 1
                        if resc_m.get(t["ann_id"], False):
                            rescued_0_total += 1

                v_losses_0.append(float(np.mean(v_frame_losses)))

            macro_loss_0 = float(np.mean(v_losses_0))
            calib_loss_curve.append(macro_loss_0)
            calib_rescue_curve.append(rescued_0_total)

        print(f"  Epoch 0 (Untrained Base) | Macro Calib Loss: {macro_loss_0:.4f} (v{calib_vids[0]}: {v_losses_0[0]:.4f}, v{calib_vids[1]}: {v_losses_0[1]:.4f}) | Gate: {probe.gamma.item():.4f} | Rescued: {rescued_0_total}/{total_tgts_0}")

        # --- RUN 25 SYNCHRONIZED TRAINING EPOCHS (5 STEPS/EPOCH, k=1..5) ---
        first_step_grads = None

        for epoch in range(1, MAX_CALIB_EPOCHS + 1):
            probe.train()
            sched_epoch = train_schedule[str(seed)][epoch - 1]

            vid_batches = {}
            for v in train_vids:
                w_info = sched_epoch[str(v)]
                batches = []
                for iid in w_info["image_ids"]:
                    sample = dataset[id_to_idx[iid]]
                    b = collate_fn([sample])
                    batches.append((
                        b[0].to(device),
                        [{k: v_t.to(device) for k, v_t in t.items()} for t in b[1]]
                    ))
                vid_batches[v] = batches

            # Frame 0 initializes detached cache
            running_caches = {}
            with torch.no_grad():
                for v in train_vids:
                    s0, _ = vid_batches[v][0]
                    _, fc = model(s0, temporal_enabled=False, return_feature_cache=True)
                    running_caches[v] = fc

            # 5 updates per epoch (frames k=1..5, training directly on opportunity frame 5)
            for k in range(1, 6):
                optimizer.zero_grad()
                next_caches = {}

                for v in train_vids:
                    sk, tk = vid_batches[v][k]
                    fc_in = running_caches[v]

                    out_k, next_fc = model(sk, temporal_adapter=probe, feat_cache=fc_in, temporal_enabled=True, return_feature_cache=True)
                    next_caches[v] = next_fc

                    loss_dict = criterion(out_k, tk)
                    loss_v = sum(loss_dict[term] * criterion.weight_dict[term] for term in loss_dict if term in criterion.weight_dict)
                    (loss_v / float(len(train_vids))).backward()

                if epoch == 1 and k == 1:
                    first_step_grads = {
                        "gamma_grad": float(probe.gamma.grad.item()) if probe.gamma.grad is not None else 0.0,
                        "stencil_grad_norm": float(probe.stencil_logits.grad.norm().item()) if probe.stencil_logits.grad is not None else 0.0,
                        "feat_proj_grad_norm": float(probe.feat_proj.weight.grad.norm().item()) if probe.feat_proj.weight.grad is not None else 0.0
                    }
                    print(f"    [*] Epoch 1 Step 1 Gradients (Accumulated across 8 Videos):")
                    print(f"        - dL/d_gamma:             {first_step_grads['gamma_grad']:+8.6f}")
                    print(f"        - ||dL/d_stencil||:       {first_step_grads['stencil_grad_norm']:8.6f}")
                    print(f"        - ||dL/d_feat_proj||:     {first_step_grads['feat_proj_grad_norm']:8.6f}")

                optimizer.step()
                running_caches = next_caches

            # Evaluate on Calibration Videos
            probe.eval()
            with torch.no_grad():
                v_losses_e = []
                rescued_e_total = 0

                for v in calib_vids:
                    v_frame_losses = []
                    for fd in calib_frame_data[v]:
                        s_curr = collate_fn([dataset[id_to_idx[fd["image_id"]]]])
                        s_prev = collate_fn([dataset[id_to_idx[fd["prev_image_id"]]]])

                        _, fc_prev = model(s_prev[0].to(device), temporal_enabled=False, return_feature_cache=True)
                        tgt_dict = {k: v_t.to(device) for k, v_t in s_curr[1][0].items()}

                        loss_f, resc_m = evaluate_video_frame(
                            model, criterion, s_curr[0].to(device), tgt_dict, fd["targets"],
                            feat_cache=fc_prev, temporal_adapter=probe, temporal_enabled=True
                        )
                        v_frame_losses.append(loss_f)

                        for t in fd["targets"]:
                            if resc_m.get(t["ann_id"], False):
                                rescued_e_total += 1

                    v_losses_e.append(float(np.mean(v_frame_losses)))

                macro_loss_e = float(np.mean(v_losses_e))
                calib_loss_curve.append(macro_loss_e)
                calib_rescue_curve.append(rescued_e_total)

            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:<3d} (Trained)        | Macro Calib Loss: {macro_loss_e:.4f} (v{calib_vids[0]}: {v_losses_e[0]:.4f}, v{calib_vids[1]}: {v_losses_e[1]:.4f}) | Gate: {probe.gamma.item():.4f} | Rescued: {rescued_e_total}/{total_tgts_0}")

        tri_seed_calib_loss_curves[seed] = calib_loss_curve
        tri_seed_calib_rescue_curves[seed] = calib_rescue_curve
        tri_seed_first_grads[seed] = first_step_grads

    # --- BUDGET SELECTION RULE (D-3490) ---
    mean_macro_losses = []
    for e_idx in range(MAX_CALIB_EPOCHS + 1):
        avg_l = float(np.mean([tri_seed_calib_loss_curves[s][e_idx] for s in SEEDS]))
        mean_macro_losses.append(avg_l)

    min_mean_calib = min(mean_macro_losses)
    optimal_epoch = mean_macro_losses.index(min_mean_calib)

    print("\n" + "=" * 115)
    print("TRI-SEED MULTI-VIDEO CALIBRATION SCAN RESULTS (EPOCHS 0..25)")
    print("=" * 115)
    print(f"{'Epoch':<8} | {'Seed 41 Macro':<18} | {'Seed 42 Macro':<18} | {'Seed 43 Macro':<18} | {'Tri-Seed Mean':<18} | {'Status'}")
    print("-" * 96)
    for e in range(MAX_CALIB_EPOCHS + 1):
        lbl = "Base (0)" if e == 0 else f"Epoch {e:<3d}"
        l41 = tri_seed_calib_loss_curves[41][e]
        l42 = tri_seed_calib_loss_curves[42][e]
        l43 = tri_seed_calib_loss_curves[43][e]
        m_l = mean_macro_losses[e]
        marker = " <--- OPTIMAL (E*)" if e == optimal_epoch else ""
        print(f"{lbl:<8} | {l41:<18.4f} | {l42:<18.4f} | {l43:<18.4f} | {m_l:<18.4f} | {marker}")

    print("\n" + "=" * 115)
    print(f"[+] Multi-Video Calibration Budget Locked: E* = {optimal_epoch} Epochs")
    print(f"    Epoch 0 Untrained Baseline Mean Loss:   {mean_macro_losses[0]:.4f}")
    print(f"    Optimal Calibration Mean Loss:          {min_mean_calib:.4f} (Delta = {min_mean_calib - mean_macro_losses[0]:+.4f})")
    print("=" * 115)

    # --------------------------------------------------------------------------
    # STAGE 2 HARD GATE: IF E* == 0, TERMINATE CLEANLY (D-3506, P-1626)
    # --------------------------------------------------------------------------
    if optimal_epoch == 0:
        print("\n" + "=" * 115)
        print("[!] CALIBRATION RESULT: ZERO-TRAINING BASELINE OPTIMAL (E* = 0) (D-3506).")
        print("    Multi-video training across 8 videos failed to earn a positive training budget.")
        print("    No positive epoch reduced held-out calibration loss on video_16 and video_46 relative to Epoch 0.")
        print("    UNTOUCHED TEST SET WILL NOT BE SPENT. Exiting cleanly without performing inference on video_7 or video_30.")
        print("=" * 115)

        results_payload = {
            "_meta": {
                "artifact_name": "pol2_candidate2f_micro_results_v1",
                "status": "CALIBRATION_FAILED_NO_POSITIVE_BUDGET",
                "split_manifest_sha256": split_hash,
                "initial_probe_weights_sha256": init_hash,
                "train_schedule_sha256": sched_hash,
                "final_refit_schedule_sha256": refit_sched_hash,
                "m0_prediction_cache_sha256": live_cache_hash,
                "m0_checkpoint_sha256": EXPECTED_M0_HASH,
                "optimal_budget_E_star": 0,
                "min_mean_calib_loss": min_mean_calib,
                "epoch_0_calib_loss": mean_macro_losses[0],
                "stencil_radius_px": STENCIL_RADIUS_PX,
                "seeds": SEEDS
            },
            "tri_seed_calib_loss_curves": {str(k): v for k, v in tri_seed_calib_loss_curves.items()},
            "tri_seed_calib_rescue_curves": {str(k): v for k, v in tri_seed_calib_rescue_curves.items()},
            "tri_seed_first_grads": {str(k): v for k, v in tri_seed_first_grads.items()},
            "test_results_per_seed": {},
            "video_level_decision": {}
        }
        with open(RESULTS_JSON, "w", encoding="utf-8") as f:
            json.dump(results_payload, f, indent=2)
        res_hash = compute_sha256(RESULTS_JSON)
        print(f"[+] Calibration Failure Record Frozen: {RESULTS_JSON.resolve()}")
        print(f"    Record SHA-256:                    {res_hash}\n")
        return

    # --------------------------------------------------------------------------
    # STAGE 2A: RETRAIN ALL 3 SEEDS & FREEZE MODELS BEFORE TEST INFERENCE (D-3519, P-1636, P-1638)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("STAGE 2A: RETRAINING ALL 3 SEEDS ON 10 VIDEOS & FREEZING MODELS (P-1636, P-1638)")
    print(f"Training on 10 Non-Test Videos for exactly E* = {optimal_epoch} Epochs BEFORE Test Set Opening")
    print("=" * 115)

    trained_probes_state = {}

    for seed in SEEDS:
        print(f"\n[*] Training Final Model for Seed {seed} on 10 Videos ({optimal_epoch} Epochs)...")
        set_seed(seed)

        probe = SpatiallyAnchoredFeatureProbe(d_model=256, num_levels=4, stencil_radius_px=STENCIL_RADIUS_PX).to(device)
        probe.load_state_dict(initial_probe_weights[str(seed)])
        probe.to(device)

        optimizer = torch.optim.AdamW([
            {'params': [probe.feat_proj.weight, probe.feat_proj.bias, probe.stencil_logits, probe.norm.weight, probe.norm.bias], 'lr': 2e-4},
            {'params': [probe.gamma], 'lr': 1e-2}
        ], weight_decay=1e-4)

        for epoch in range(1, optimal_epoch + 1):
            probe.train()
            sched_refit_epoch = refit_schedule[str(seed)][epoch - 1]

            vid_batches_10 = {}
            for v in all_10_vids:
                w_info = sched_refit_epoch[str(v)]
                batches = []
                for iid in w_info["image_ids"]:
                    b = collate_fn([dataset[id_to_idx[iid]]])
                    batches.append((
                        b[0].to(device),
                        [{k: v_t.to(device) for k, v_t in t.items()} for t in b[1]]
                    ))
                vid_batches_10[v] = batches

            running_caches_10 = {}
            with torch.no_grad():
                for v in all_10_vids:
                    s0, _ = vid_batches_10[v][0]
                    _, fc = model(s0, temporal_enabled=False, return_feature_cache=True)
                    running_caches_10[v] = fc

            # 5 updates per epoch (frames k=1..5)
            for k in range(1, 6):
                optimizer.zero_grad()
                next_caches_10 = {}
                for v in all_10_vids:
                    sk, tk = vid_batches_10[v][k]
                    fc_in = running_caches_10[v]
                    out_k, next_fc = model(sk, temporal_adapter=probe, feat_cache=fc_in, temporal_enabled=True, return_feature_cache=True)
                    next_caches_10[v] = next_fc

                    loss_dict = criterion(out_k, tk)
                    loss_v = sum(loss_dict[term] * criterion.weight_dict[term] for term in loss_dict if term in criterion.weight_dict)
                    (loss_v / float(len(all_10_vids))).backward()

                optimizer.step()
                running_caches_10 = next_caches_10

        trained_probes_state[str(seed)] = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}
        print(f"[+] Final Training Complete for Seed {seed} | Final Gate (gamma): {probe.gamma.item():.4f}")

    # Freeze final models artifact BEFORE any test inference (D-3519, P-1638)
    torch.save(trained_probes_state, FINAL_MODELS_PTH)
    final_models_hash = compute_sha256(FINAL_MODELS_PTH)
    print(f"\n[+] All 3 Final Candidate-2F Models Frozen (CPU): {FINAL_MODELS_PTH.resolve()}")
    print(f"    Final Models SHA-256:                         {final_models_hash}")
    print("[+] Model Optimization Frozen: All weights fixed before optimization-untouched test evaluation.")

    # --------------------------------------------------------------------------
    # STAGE 2B: UNTOUCHED TEST EVALUATION (D-3519, D-3507, D-3508, D-3528)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("STAGE 2B: OPTIMIZATION-UNTOUCHED DEVELOPMENT TEST FALSIFICATION")
    print(f"Held-Out Current-Video Endpoints: video_{test_vids[0]} and video_{test_vids[1]}")
    print("=" * 115)

    test_tgts_by_img = defaultdict(list)
    for t in split_data["test_targets"]:
        test_tgts_by_img[t["image_id"]].append(t)

    test_frame_data = defaultdict(list)
    for iid, tgts in test_tgts_by_img.items():
        img_info = coco.loadImgs([iid])[0]
        v_id = img_info["video_id"]
        fid = img_info["frame_id"]
        prev_imgs = [m for m in coco.loadImgs(coco.getImgIds()) if m["video_id"] == v_id and m["frame_id"] == fid - 1]
        assert len(prev_imgs) == 1, f"[FATAL] Missing prior frame for test image {iid}"
        prev_iid = prev_imgs[0]["id"]
        prev_fid = prev_imgs[0]["frame_id"]
        test_frame_data[v_id].append({
            "image_id": iid,
            "frame_id": fid,
            "prev_image_id": prev_iid,
            "prev_frame_id": prev_fid,
            "targets": tgts
        })

    # Explicit chronological sort (P-1630)
    for v in test_vids:
        test_frame_data[v].sort(key=lambda x: x["frame_id"])

    v_test_a, v_test_b = test_vids[0], test_vids[1]
    n_fa = len(test_frame_data[v_test_a])
    n_fb = len(test_frame_data[v_test_b])
    n_ta = sum(len(fd["targets"]) for fd in test_frame_data[v_test_a])
    n_tb = sum(len(fd["targets"]) for fd in test_frame_data[v_test_b])

    print(f"[+] Loaded Chronologically Ordered Untouched Test Streams:")
    print(f"    - video_{v_test_a}: {n_fa} unique frames ({n_ta} targets)")
    print(f"    - video_{v_test_b}: {n_fb} unique frames ({n_tb} targets)")
    assert n_ta + n_tb == 40, f"[FATAL] Total test targets: {n_ta + n_tb} != 40"

    # Precompute correct previous-frame feature caches
    caches_a = []
    with torch.no_grad():
        for fd in test_frame_data[v_test_a]:
            s_prev = collate_fn([dataset[id_to_idx[fd["prev_image_id"]]]])
            _, fc = model(s_prev[0].to(device), temporal_enabled=False, return_feature_cache=True)
            caches_a.append(fc)

        caches_b = []
        for fd in test_frame_data[v_test_b]:
            s_prev = collate_fn([dataset[id_to_idx[fd["prev_image_id"]]]])
            _, fc = model(s_prev[0].to(device), temporal_enabled=False, return_feature_cache=True)
            caches_b.append(fc)

    # D-3526 / P-1640: Explicit unambiguous wrong-donor lineage
    wrong_cache_mapping = {
        str(v_test_a): [
            {
                "curr_frame_id": fd["frame_id"],
                "curr_image_id": fd["image_id"],
                "wrong_donor_eval_frame_id": test_frame_data[v_test_b][idx % n_fb]["frame_id"],
                "wrong_donor_eval_image_id": test_frame_data[v_test_b][idx % n_fb]["image_id"],
                "wrong_donor_prev_frame_id": test_frame_data[v_test_b][idx % n_fb]["prev_frame_id"],
                "wrong_donor_prev_image_id": test_frame_data[v_test_b][idx % n_fb]["prev_image_id"]
            }
            for idx, fd in enumerate(test_frame_data[v_test_a])
        ],
        str(v_test_b): [
            {
                "curr_frame_id": fd["frame_id"],
                "curr_image_id": fd["image_id"],
                "wrong_donor_eval_frame_id": test_frame_data[v_test_a][idx % n_fa]["frame_id"],
                "wrong_donor_eval_image_id": test_frame_data[v_test_a][idx % n_fa]["image_id"],
                "wrong_donor_prev_frame_id": test_frame_data[v_test_a][idx % n_fa]["prev_frame_id"],
                "wrong_donor_prev_image_id": test_frame_data[v_test_a][idx % n_fa]["prev_image_id"]
            }
            for idx, fd in enumerate(test_frame_data[v_test_b])
        ]
    }

    test_results_per_seed = {}
    seed_rescued_aids_correct = defaultdict(set)
    seed_rescued_aids_wrong = defaultdict(set)

    for seed in SEEDS:
        probe = SpatiallyAnchoredFeatureProbe(d_model=256, num_levels=4, stencil_radius_px=STENCIL_RADIUS_PX).to(device)
        probe.load_state_dict(trained_probes_state[str(seed)])
        probe.to(device)
        probe.eval()

        seed_test_record = {}
        with torch.no_grad():
            for v, f_data, c_correct, c_wrong, n_expected_tgts in [
                (v_test_a, test_frame_data[v_test_a], caches_a, caches_b, n_ta),
                (v_test_b, test_frame_data[v_test_b], caches_b, caches_a, n_tb)
            ]:
                losses_D, losses_Fcorr, losses_Fwrong = [], [], []
                resc_D_count, resc_Fcorr_count, resc_Fwrong_count = 0, 0, 0
                n_v_tgts = 0

                for idx, fd in enumerate(f_data):
                    s_curr = collate_fn([dataset[id_to_idx[fd["image_id"]]]])
                    tgt_dict = {k: v_t.to(device) for k, v_t in s_curr[1][0].items()}

                    fc_corr = c_correct[idx]
                    fc_wr = c_wrong[idx % len(c_wrong)]

                    # Condition 1: Static Baseline D
                    l_d, r_d = evaluate_video_frame(model, criterion, s_curr[0].to(device), tgt_dict, fd["targets"], None, None, False)
                    # Condition 2: F_correct
                    l_corr, r_corr = evaluate_video_frame(model, criterion, s_curr[0].to(device), tgt_dict, fd["targets"], fc_corr, probe, True)
                    # Condition 3: F_wrong
                    l_wr, r_wr = evaluate_video_frame(model, criterion, s_curr[0].to(device), tgt_dict, fd["targets"], fc_wr, probe, True)

                    losses_D.append(l_d)
                    losses_Fcorr.append(l_corr)
                    losses_Fwrong.append(l_wr)

                    for t in fd["targets"]:
                        aid = t["ann_id"]
                        n_v_tgts += 1
                        if r_d.get(aid, False): resc_D_count += 1
                        if r_corr.get(aid, False):
                            resc_Fcorr_count += 1
                            seed_rescued_aids_correct[seed].add(aid)
                        if r_wr.get(aid, False):
                            resc_Fwrong_count += 1
                            seed_rescued_aids_wrong[seed].add(aid)

                # D-3509: Strict static baseline miss reproduction invariant
                assert resc_D_count == 0, (
                    f"[FATAL] Static Condition D rescued {resc_D_count} targets on video {v}! "
                    f"Expected exactly 0 rescues on predeclared M0 misses."
                )

                # D-3527 / P-1641: Target count invariant assertion
                assert n_v_tgts == n_expected_tgts, (
                    f"[FATAL] Evaluated {n_v_tgts} targets on video {v}, expected {n_expected_tgts}"
                )

                seed_test_record[v] = {
                    "loss_D": float(np.mean(losses_D)),
                    "loss_F_correct": float(np.mean(losses_Fcorr)),
                    "loss_F_wrong": float(np.mean(losses_Fwrong)),
                    "delta_corr_vs_wrong": float(np.mean(losses_Fwrong) - np.mean(losses_Fcorr)),
                    "delta_corr_vs_D": float(np.mean(losses_D) - np.mean(losses_Fcorr)),
                    "rescued_D": resc_D_count,
                    "rescued_F_correct": resc_Fcorr_count,
                    "rescued_F_wrong": resc_Fwrong_count,
                    "total_targets": n_v_tgts
                }

        test_results_per_seed[seed] = seed_test_record

        rec_a = seed_test_record[v_test_a]
        rec_b = seed_test_record[v_test_b]
        macro_d_loss = (rec_a["delta_corr_vs_wrong"] + rec_b["delta_corr_vs_wrong"]) / 2.0
        tot_resc_corr = rec_a["rescued_F_correct"] + rec_b["rescued_F_correct"]
        tot_resc_wr = rec_a["rescued_F_wrong"] + rec_b["rescued_F_wrong"]

        print(f"\n  Seed {seed} Test Results (Gamma = {probe.gamma.item():.4f}):")
        print(f"    - video_{v_test_a}: L(corr)={rec_a['loss_F_correct']:.4f}, L(wrong)={rec_a['loss_F_wrong']:.4f}, L(D)={rec_a['loss_D']:.4f} | Rescued: corr={rec_a['rescued_F_correct']}/{n_ta}, wrong={rec_a['rescued_F_wrong']}/{n_ta}, D=0/{n_ta}")
        print(f"    - video_{v_test_b}: L(corr)={rec_b['loss_F_correct']:.4f}, L(wrong)={rec_b['loss_F_wrong']:.4f}, L(D)={rec_b['loss_D']:.4f} | Rescued: corr={rec_b['rescued_F_correct']}/{n_tb}, wrong={rec_b['rescued_F_wrong']}/{n_tb}, D=0/{n_tb}")
        print(f"    - Macro Loss Delta [L(wrong) - L(corr)]: {macro_d_loss:+.4f} (Positive = Correct Correspondence Wins)")
        print(f"    - Pooled Rescued Targets: F_correct = {tot_resc_corr}/40 | F_wrong = {tot_resc_wr}/40 | Static D = 0/40")

    # --------------------------------------------------------------------------
    # LAYER 1: PREDECLARED VIDEO-LEVEL MECHANISM GATE (D-3507, D-3508, D-3528)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("LAYER 1: PREDECLARED VIDEO-LEVEL MECHANISM GATE (D-3507, D-3508, D-3528)")
    print("Held-Out Current-Video Endpoints: video_7 and video_30 | Formal Gate: Delta_corr > 0 and Delta_static > 0 on both")
    print("=" * 115)

    video_decision = {}
    passed_mechanism = True

    for v in [v_test_a, v_test_b]:
        mean_l_corr = float(np.mean([test_results_per_seed[s][v]["loss_F_correct"] for s in SEEDS]))
        mean_l_wrong = float(np.mean([test_results_per_seed[s][v]["loss_F_wrong"] for s in SEEDS]))
        mean_l_d = float(np.mean([test_results_per_seed[s][v]["loss_D"] for s in SEEDS]))

        delta_corr_v = mean_l_wrong - mean_l_corr
        delta_static_v = mean_l_d - mean_l_corr

        resc_corr_mean = float(np.mean([test_results_per_seed[s][v]["rescued_F_correct"] for s in SEEDS]))
        resc_wrong_mean = float(np.mean([test_results_per_seed[s][v]["rescued_F_wrong"] for s in SEEDS]))
        v_total_tgts = test_results_per_seed[SEEDS[0]][v]["total_targets"]

        corr_pass = delta_corr_v > 0.0
        static_pass = delta_static_v > 0.0
        v_pass = corr_pass and static_pass
        if not v_pass:
            passed_mechanism = False

        video_decision[str(v)] = {
            "mean_loss_F_correct": mean_l_corr,
            "mean_loss_F_wrong": mean_l_wrong,
            "mean_loss_D": mean_l_d,
            "delta_corr": delta_corr_v,
            "delta_static": delta_static_v,
            "delta_corr_positive": corr_pass,
            "delta_static_positive": static_pass,
            "mean_rescued_F_correct": resc_corr_mean,
            "mean_rescued_F_wrong": resc_wrong_mean,
            "rescued_D": 0,
            "total_targets": v_total_tgts,
            "mechanism_verdict": "PASS" if v_pass else "FAIL"
        }

        print(f"video_{v:<4d} (N = {v_total_tgts} targets):")
        print(f"  - Tri-Seed Mean Losses: L(corr) = {mean_l_corr:.4f} | L(wrong) = {mean_l_wrong:.4f} | L(D) = {mean_l_d:.4f}")
        print(f"  - Delta_corr   [L(wrong) - L(corr)]: {delta_corr_v:+.6f} -> {'PASS (>0)' if corr_pass else 'FAIL (<=0)'}")
        print(f"  - Delta_static [L(D) - L(corr)]:     {delta_static_v:+.6f} -> {'PASS (>0)' if static_pass else 'FAIL (<=0)'}")
        print(f"  - Mean Rescued Targets: F_correct = {resc_corr_mean:.2f}/{v_total_tgts} | F_wrong = {resc_wrong_mean:.2f}/{v_total_tgts} | Static D = 0/{v_total_tgts}")
        print(f"  - Video Mechanism Verdict: {video_decision[str(v)]['mechanism_verdict']}")

    print("\n" + "-" * 115)
    if passed_mechanism:
        mechanism_verdict_str = "PASS (Correspondence-sensitive temporal benefit demonstrated under the predeclared wrong-history control on both optimization-untouched development videos)"
        print(f"[+] MECHANISTIC GATE: {mechanism_verdict_str}.")
        print("    Both unseen test videos independently satisfy L(F_correct) < L(F_wrong) and L(F_correct) < L(D).")
    else:
        mechanism_verdict_str = "FAIL (At least one held-out video failed directional loss criteria)"
        print(f"[!] MECHANISTIC GATE: {mechanism_verdict_str}.")
    print("-" * 115)

    # --------------------------------------------------------------------------
    # LAYER 2: OPERATIONAL TARGET RESCUE DESCRIPTIVE REPORT (D-3521, P-1639)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("LAYER 2: OPERATIONAL TARGET RESCUE DESCRIPTIVE REPORT (D-3521, P-1639)")
    print("Evaluates recovery of the 40 targeted <4px M0 misses under IoU >= 0.30, score >= 0.20, correct class")
    print("=" * 115)

    r_corr_by_seed = {s: len(seed_rescued_aids_correct[s]) for s in SEEDS}
    r_wrong_by_seed = {s: len(seed_rescued_aids_wrong[s]) for s in SEEDS}

    mean_r_corr = float(np.mean(list(r_corr_by_seed.values())))
    mean_r_wrong = float(np.mean(list(r_wrong_by_seed.values())))
    delta_r = mean_r_corr - mean_r_wrong

    all_test_aids = {t["ann_id"] for t in split_data["test_targets"]}
    assert len(all_test_aids) == 40

    corr_freq = {aid: sum(1 for s in SEEDS if aid in seed_rescued_aids_correct[s]) for aid in all_test_aids}
    wrong_freq = {aid: sum(1 for s in SEEDS if aid in seed_rescued_aids_wrong[s]) for aid in all_test_aids}

    n_corr_1of3 = sum(1 for cnt in corr_freq.values() if cnt == 1)
    n_corr_2of3 = sum(1 for cnt in corr_freq.values() if cnt == 2)
    n_corr_3of3 = sum(1 for cnt in corr_freq.values() if cnt == 3)
    n_corr_any = sum(1 for cnt in corr_freq.values() if cnt >= 1)

    n_wrong_1of3 = sum(1 for cnt in wrong_freq.values() if cnt == 1)
    n_wrong_2of3 = sum(1 for cnt in wrong_freq.values() if cnt == 2)
    n_wrong_3of3 = sum(1 for cnt in wrong_freq.values() if cnt == 3)
    n_wrong_any = sum(1 for cnt in wrong_freq.values() if cnt >= 1)

    print(f"Per-Seed Rescued Targets (out of 40):")
    for s in SEEDS:
        print(f"  - Seed {s}: F_correct = {r_corr_by_seed[s]:2d}/40 | F_wrong = {r_wrong_by_seed[s]:2d}/40 | Static D = 0/40")
    print(f"Tri-Seed Rescue Means: F_correct = {mean_r_corr:.2f}/40 | F_wrong = {mean_r_wrong:.2f}/40 | Delta_R = {delta_r:+.2f}")

    print(f"\nAnnotation-Level Target Consistency across Seeds:")
    print(f"  - F_correct: Recovered in >=1 seed = {n_corr_any:2d}/40 | Exactly 1 seed = {n_corr_1of3:2d} | Exactly 2 seeds = {n_corr_2of3:2d} | All 3 seeds = {n_corr_3of3:2d}")
    print(f"  - F_wrong:   Recovered in >=1 seed = {n_wrong_any:2d}/40 | Exactly 1 seed = {n_wrong_1of3:2d} | Exactly 2 seeds = {n_wrong_2of3:2d} | All 3 seeds = {n_wrong_3of3:2d}")

    operational_report = {
        "rescued_correct_by_seed": r_corr_by_seed,
        "rescued_wrong_by_seed": r_wrong_by_seed,
        "rescued_D_by_seed": {s: 0 for s in SEEDS},
        "mean_rescued_correct": mean_r_corr,
        "mean_rescued_wrong": mean_r_wrong,
        "delta_rescued_mean": delta_r,
        "consistency_correct": {"recovered_in_any_seed": n_corr_any, "in_1_of_3_seeds": n_corr_1of3, "in_2_of_3_seeds": n_corr_2of3, "in_all_3_seeds": n_corr_3of3},
        "consistency_wrong": {"recovered_in_any_seed": n_wrong_any, "in_1_of_3_seeds": n_wrong_1of3, "in_2_of_3_seeds": n_wrong_2of3, "in_all_3_seeds": n_wrong_3of3}
    }

    # --------------------------------------------------------------------------
    # 4. SERIALIZE COMPREHENSIVE RESULTS MANIFEST (P-1627)
    # --------------------------------------------------------------------------
    results_payload = {
        "_meta": {
            "artifact_name": "pol2_candidate2f_micro_results_v1",
            "status": "COMPLETED",
            "split_manifest_sha256": split_hash,
            "initial_probe_weights_sha256": init_hash,
            "train_schedule_sha256": sched_hash,
            "final_refit_schedule_sha256": refit_sched_hash,
            "final_models_sha256": final_models_hash,
            "m0_prediction_cache_sha256": live_cache_hash,
            "m0_checkpoint_sha256": EXPECTED_M0_HASH,
            "optimal_budget_E_star": optimal_epoch,
            "min_mean_calib_loss": min_mean_calib,
            "stencil_radius_px": STENCIL_RADIUS_PX,
            "seeds": SEEDS,
            "mechanism_verdict": mechanism_verdict_str
        },
        "tri_seed_calib_loss_curves": {str(k): v for k, v in tri_seed_calib_loss_curves.items()},
        "tri_seed_calib_rescue_curves": {str(k): v for k, v in tri_seed_calib_rescue_curves.items()},
        "tri_seed_first_grads": {str(k): v for k, v in tri_seed_first_grads.items()},
        "test_results_per_seed": {str(s): {str(v): data for v, data in test_results_per_seed[s].items()} for s in SEEDS},
        "video_level_decision": video_decision,
        "operational_rescue_report": operational_report,
        "wrong_cache_mapping": wrong_cache_mapping
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    res_hash = compute_sha256(RESULTS_JSON)
    print("\n" + "=" * 115)
    print(f"[+] Candidate 2F Micro-Falsification Record Frozen: {RESULTS_JSON.resolve()}")
    print(f"    Immutable Results SHA-256:                     {res_hash}\n")


if __name__ == "__main__":
    main()