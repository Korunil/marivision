"""
D:/MARIVISION/H0_workbench/eval_tools/audit_pol2_candidate2f_rescue_corrected.py
PoL-2F Post-Flight: Corrected Category-Space Operational Target Rescue Audit & Loss Decomposition

Strict Compliance Checklist:
  - D-3538 / D-3539 / P-1644: Loads frozen final models (SHA 40614ca8...) without retraining;
                               maps internal zero-based class indices to dataset category IDs via label_to_cat_id.
  - D-3546 / P-1648: Asserts whole-file SHA-256 for cache_m0_train_raw_top100.json (d9816083...).
  - D-3547 / P-1649: Reconstructs and asserts exact identity of wrong_cache_mapping against original results.
  - D-3522 / D-3509: Checks static condition D miss reproduction on calibration (40) and test (40) targets.
  - D-3527: Asserts live evaluated test target count strictly matches n_expected_tgts per video.
  - D-3548 / D-3549 / P-1650 / P-1651: Video-level loss decomposition (primary and all auxiliary heads).
  - D-3550 / P-1652: Serializes exact rescued annotation IDs per seed and full ann_id -> rescue_frequency map.
  - D-3540: References original results manifest (SHA 4fc31dd9...) and exports corrected audit artifact.
"""

import os
import sys
import json
import hashlib
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
TRAIN_JSON = Path("D:/MARIVISION/seaclips-train.json")
CACHE_FILE = OUT_DIR / "cache_m0_train_raw_top100.json"
DEV_POOL_JSON = OUT_DIR / "pol2_dev_pool_manifest.json"
SPLIT_MANIFEST_JSON = OUT_DIR / "pol2_candidate2f_micro_split.json"
RESULTS_JSON = OUT_DIR / "pol2_candidate2f_micro_results.json"
FINAL_MODELS_PTH = OUT_DIR / "pol2_candidate2f_final_models.pth"
M0_CKPT_PATH = Path("D:/MARIVISION/checkpoints_r0/checkpoint_best.pth")
ZIP_PATH = Path("D:/MARIVISION/SeaClips.zip")

CORRECTED_AUDIT_JSON = OUT_DIR / "pol2_candidate2f_rescue_corrected.json"

EXPECTED_TRAIN_JSON_SHA256 = "edc32bdbd6b792eb83b6daf11e14708232f1f4a9e06153ebebd5bad1caef9b35"
EXPECTED_CACHE_SHA256 = "d9816083760d698935b44e9165522e709b490c3af29e710d75b8b0d2d90210e0"
EXPECTED_DEV_POOL_SHA256 = "533567e1d74544db7087a95253e640ef0959b998e5d503f57b0eb8074d8fe970"
EXPECTED_SPLIT_SHA256 = "646a8be84aea0ff94a5f4642d81a0210000b8e34e6fded72687d55639c96ecb1"
EXPECTED_RESULTS_SHA256 = "4fc31dd9a76c99eab421202f48cbd83800871d74de4764c57d8579ec4c84072b"
EXPECTED_FINAL_MODELS_SHA256 = "40614ca849ad719aa7be409d620ed301fc34dfd78750f9bac5179f1f322d41f1"
EXPECTED_M0_HASH = "28e6288c3f6769c2d5275aa77ddb248735b55a09cd46a6b570e3f117890b268b"

SEEDS = [41, 42, 43]
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


# ------------------------------------------------------------------------------
# 1. CANDIDATE 2F MODULE & SOURCE HOOKS
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
# 2. COMPLETE LOSS DECOMPOSITION & CORRECTED RESCUE EVALUATOR (P-1644, P-1650)
# ------------------------------------------------------------------------------

def evaluate_video_frame_corrected(model, criterion, samples, targets_dict, eval_targets, label_to_cat_id, feat_cache, temporal_adapter, temporal_enabled=True):
    out = model(samples, temporal_adapter=temporal_adapter, feat_cache=feat_cache, temporal_enabled=temporal_enabled)
    loss_dict = criterion(out, [targets_dict])
    weighted = {k: loss_dict[k] * criterion.weight_dict[k] for k in loss_dict if k in criterion.weight_dict}
    loss_total = sum(weighted.values()).item()

    # Primary head terms
    loss_ce_primary = weighted.get('loss_ce', torch.tensor(0.0)).item()
    loss_bbox_primary = weighted.get('loss_bbox', torch.tensor(0.0)).item()
    loss_giou_primary = weighted.get('loss_giou', torch.tensor(0.0)).item()
    loss_primary_total = loss_ce_primary + loss_bbox_primary + loss_giou_primary

    # All heads (primary + auxiliary)
    loss_ce_all = sum(v.item() for k, v in weighted.items() if k == "loss_ce" or k.startswith("loss_ce_"))
    loss_bbox_all = sum(v.item() for k, v in weighted.items() if k == "loss_bbox" or k.startswith("loss_bbox_"))
    loss_giou_all = sum(v.item() for k, v in weighted.items() if k == "loss_giou" or k.startswith("loss_giou_"))

    # Assert complete accounting
    assert abs(loss_total - (loss_ce_all + loss_bbox_all + loss_giou_all)) < 1e-4

    loss_breakdown = {
        "loss_total": loss_total,
        "primary": {
            "loss_total": loss_primary_total,
            "loss_ce": loss_ce_primary,
            "loss_bbox": loss_bbox_primary,
            "loss_giou": loss_giou_primary
        },
        "all_heads": {
            "loss_total": loss_total,
            "loss_ce": loss_ce_all,
            "loss_bbox": loss_bbox_all,
            "loss_giou": loss_giou_all
        }
    }

    # Top-100 DETR ranking
    prob = out["pred_logits"][0].sigmoid()
    num_classes = prob.shape[-1]
    topk_scores, topk_idx = torch.topk(prob.flatten(), 100)
    query_idx = torch.div(topk_idx, num_classes, rounding_mode="floor")
    topk_labels = topk_idx % num_classes
    topk_boxes = out["pred_boxes"][0][query_idx]

    pred_cat_ids = np.array([label_to_cat_id[int(lbl.item())] for lbl in topk_labels])

    img_h, img_w = targets_dict["orig_size"][0].item(), targets_dict["orig_size"][1].item()
    pred_x1 = (topk_boxes[:, 0] - topk_boxes[:, 2] / 2.0) * img_w
    pred_y1 = (topk_boxes[:, 1] - topk_boxes[:, 3] / 2.0) * img_h
    pred_w = topk_boxes[:, 2] * img_w
    pred_h = topk_boxes[:, 3] * img_h
    pred_boxes_xywh = torch.stack([pred_x1, pred_y1, pred_w, pred_h], dim=-1).cpu().numpy()
    pred_scores_np = topk_scores.cpu().numpy()

    rescued_map = {}
    for t in eval_targets:
        aid = t["ann_id"]
        gt_box = t["bbox"]
        gt_cat = t["category_id"]

        is_rescued = False
        for p_idx in range(100):
            if pred_scores_np[p_idx] >= OPERATING_SCORE_THRESHOLD and pred_cat_ids[p_idx] == gt_cat:
                iou = box_iou(gt_box, pred_boxes_xywh[p_idx])
                if iou >= EVAL_IOU_THRESHOLD:
                    is_rescued = True
                    break
        rescued_map[aid] = is_rescued

    return loss_breakdown, rescued_map


# ------------------------------------------------------------------------------
# 3. MAIN AUDIT ENGINE
# ------------------------------------------------------------------------------

def main():
    print("=" * 115)
    print("PoL-2F POST-FLIGHT: CORRECTED CATEGORY-SPACE RESCUE AUDIT & COMPLETE LOSS DECOMPOSITION")
    print("=" * 115)

    # 1. Full Live Provenance Assertions (D-3546, P-1648)
    assert TRAIN_JSON.exists(), f"[FATAL] Missing {TRAIN_JSON}"
    assert CACHE_FILE.exists(), f"[FATAL] Missing {CACHE_FILE}"
    assert DEV_POOL_JSON.exists(), f"[FATAL] Missing {DEV_POOL_JSON}"
    assert SPLIT_MANIFEST_JSON.exists(), f"[FATAL] Missing {SPLIT_MANIFEST_JSON}"
    assert RESULTS_JSON.exists(), f"[FATAL] Missing {RESULTS_JSON}"
    assert FINAL_MODELS_PTH.exists(), f"[FATAL] Missing {FINAL_MODELS_PTH}"
    assert M0_CKPT_PATH.exists(), f"[FATAL] Missing {M0_CKPT_PATH}"

    assert compute_sha256(TRAIN_JSON) == EXPECTED_TRAIN_JSON_SHA256, "[FATAL] Train JSON SHA mismatch!"
    assert compute_sha256(CACHE_FILE) == EXPECTED_CACHE_SHA256, "[FATAL] Prediction cache SHA mismatch!"
    assert compute_sha256(DEV_POOL_JSON) == EXPECTED_DEV_POOL_SHA256, "[FATAL] Dev pool SHA mismatch!"
    assert compute_sha256(SPLIT_MANIFEST_JSON) == EXPECTED_SPLIT_SHA256, "[FATAL] Split manifest SHA mismatch!"
    assert compute_sha256(RESULTS_JSON) == EXPECTED_RESULTS_SHA256, "[FATAL] Results manifest SHA mismatch!"
    assert compute_sha256(FINAL_MODELS_PTH) == EXPECTED_FINAL_MODELS_SHA256, "[FATAL] Final models SHA mismatch!"
    assert compute_sha256(M0_CKPT_PATH) == EXPECTED_M0_HASH, "[FATAL] M0 checkpoint SHA mismatch!"

    # Cache inventory verification (exact set equality)
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache_data = json.load(f)
    assert cache_data["_meta"]["m0_checkpoint_sha256"] == EXPECTED_M0_HASH

    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        train_raw = json.load(f)
    train_image_ids = {int(x["id"]) for x in train_raw["images"]}
    raw_detections = {int(k): v for k, v in cache_data["detections"].items()}
    assert len(train_image_ids) == 20310
    assert set(raw_detections.keys()) == train_image_ids

    print("[+] All 7 Source Artifacts & Checkpoints Certified by Immutable SHA-256 (D-3546):")
    print(f"    - Frozen Split Manifest SHA:        {EXPECTED_SPLIT_SHA256}")
    print(f"    - Frozen Final Models SHA:          {EXPECTED_FINAL_MODELS_SHA256}")
    print(f"    - Original Results Manifest SHA:    {EXPECTED_RESULTS_SHA256}")
    print(f"    - Raw M0 Prediction Cache SHA:      {EXPECTED_CACHE_SHA256} (20,310 Set Equality Certified)")

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

    dataset = SeaClipsDetection(ZIP_PATH, TRAIN_JSON, is_train=False, random_flip=False)
    coco = dataset.coco
    id_to_idx = {iid: idx for idx, iid in enumerate(sorted(coco.getImgIds()))}

    # Canonical label_to_cat_id mapping (P-1644)
    if hasattr(dataset, "label_to_cat_id"):
        label_to_cat_id = dataset.label_to_cat_id
    else:
        cat_ids = sorted(coco.getCatIds())
        label_to_cat_id = {i: cat_id for i, cat_id in enumerate(cat_ids)}
    
    print("\n--- Certified Label-to-Category Mapping (P-1644) ---")
    cats = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    for lbl, cid in sorted(label_to_cat_id.items()):
        print(f"  Internal Model Label {lbl} -> Dataset Category ID {cid} ({cats[cid]})")

    with open(SPLIT_MANIFEST_JSON, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        orig_results = json.load(f)

    calib_vids = split_data["_meta"]["calib_videos"]
    test_vids = split_data["_meta"]["test_videos"]

    # --------------------------------------------------------------------------
    # STEP 1: VERIFY CALIBRATION STATIC MISS REPRODUCTION (D-3522)
    # --------------------------------------------------------------------------
    print("\n--- STEP 1: CALIBRATION STATIC MISS REPRODUCTION AUDIT (D-3522) ---")
    calib_tgts_by_img = defaultdict(list)
    for t in split_data["calibration_targets"]:
        calib_tgts_by_img[t["image_id"]].append(t)

    calib_frame_data = defaultdict(list)
    for iid, tgts in calib_tgts_by_img.items():
        img_info = coco.loadImgs([iid])[0]
        v_id = img_info["video_id"]
        fid = img_info["frame_id"]
        prev_imgs = [m for m in coco.loadImgs(coco.getImgIds()) if m["video_id"] == v_id and m["frame_id"] == fid - 1]
        assert len(prev_imgs) == 1, f"[FATAL] Non-unique or missing prior frame for image {iid}"
        calib_frame_data[v_id].append({
            "image_id": iid,
            "frame_id": fid,
            "prev_image_id": prev_imgs[0]["id"],
            "targets": tgts
        })

    for v in calib_vids:
        calib_frame_data[v].sort(key=lambda x: x["frame_id"])

    with torch.no_grad():
        calib_resc_D_total = 0
        calib_tgts_total = 0
        for v in calib_vids:
            for fd in calib_frame_data[v]:
                s_curr = collate_fn([dataset[id_to_idx[fd["image_id"]]]])
                tgt_dict = {k: v_t.to(device) for k, v_t in s_curr[1][0].items()}
                _, r_map = evaluate_video_frame_corrected(
                    model, criterion, s_curr[0].to(device), tgt_dict, fd["targets"],
                    label_to_cat_id, None, None, False
                )
                for t in fd["targets"]:
                    calib_tgts_total += 1
                    if r_map.get(t["ann_id"], False):
                        calib_resc_D_total += 1

    print(f"[+] Static Condition D Rescues on Calibration Targets: {calib_resc_D_total}/{calib_tgts_total}")
    assert calib_resc_D_total == 0, (
        f"[FATAL] Static Condition D rescued {calib_resc_D_total} calibration targets under corrected category mapping!"
    )
    print("[+] Frozen M0 miss-status reproduction certified for all 40 calibration targets under corrected top-100 evaluator (D-3522).")

    # --------------------------------------------------------------------------
    # STEP 2: TEST STREAM EXTRACTION & COUNTERFACTUAL MAPPING ASSERTION (D-3547, P-1649)
    # --------------------------------------------------------------------------
    print("\n--- STEP 2: TEST STREAM & COUNTERFACTUAL MAPPING INVARIANT (D-3547, P-1649) ---")
    test_tgts_by_img = defaultdict(list)
    for t in split_data["test_targets"]:
        test_tgts_by_img[t["image_id"]].append(t)

    test_frame_data = defaultdict(list)
    for iid, tgts in test_tgts_by_img.items():
        img_info = coco.loadImgs([iid])[0]
        v_id = img_info["video_id"]
        fid = img_info["frame_id"]
        prev_imgs = [m for m in coco.loadImgs(coco.getImgIds()) if m["video_id"] == v_id and m["frame_id"] == fid - 1]
        assert len(prev_imgs) == 1, f"[FATAL] Non-unique or missing prior frame for test image {iid}"
        test_frame_data[v_id].append({
            "image_id": iid,
            "frame_id": fid,
            "prev_image_id": prev_imgs[0]["id"],
            "prev_frame_id": prev_imgs[0]["frame_id"],
            "targets": tgts
        })

    for v in test_vids:
        test_frame_data[v].sort(key=lambda x: x["frame_id"])

    v_test_a, v_test_b = test_vids[0], test_vids[1]
    n_fa = len(test_frame_data[v_test_a])
    n_fb = len(test_frame_data[v_test_b])
    n_ta = sum(len(fd["targets"]) for fd in test_frame_data[v_test_a])
    n_tb = sum(len(fd["targets"]) for fd in test_frame_data[v_test_b])

    # Reconstruct unambiguous wrong cache mapping and verify exact equality with original results
    reconstructed_wrong_mapping = {
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

    assert reconstructed_wrong_mapping == orig_results["wrong_cache_mapping"], (
        "[FATAL] Reconstructed wrong_cache_mapping does not match original results manifest!"
    )
    print(f"[+] Counterfactual Invariant Certified: Reconstructed wrong-cache mapping matches original run identically (D-3547).")

    # Audit test static miss reproduction
    with torch.no_grad():
        test_resc_D_total = 0
        test_tgts_total = 0
        for v in test_vids:
            for fd in test_frame_data[v]:
                s_curr = collate_fn([dataset[id_to_idx[fd["image_id"]]]])
                tgt_dict = {k: v_t.to(device) for k, v_t in s_curr[1][0].items()}
                _, r_map = evaluate_video_frame_corrected(
                    model, criterion, s_curr[0].to(device), tgt_dict, fd["targets"],
                    label_to_cat_id, None, None, False
                )
                for t in fd["targets"]:
                    test_tgts_total += 1
                    if r_map.get(t["ann_id"], False):
                        test_resc_D_total += 1

    print(f"[+] Static Condition D Rescues on Test Targets: {test_resc_D_total}/{test_tgts_total}")
    assert test_resc_D_total == 0, (
        f"[FATAL] Static Condition D rescued {test_resc_D_total} test targets under corrected category mapping!"
    )
    print("[+] Frozen M0 miss-status reproduction certified for all 40 test targets under corrected top-100 evaluator (D-3509).")

    # --------------------------------------------------------------------------
    # STEP 3: CORRECTED OPERATIONAL RESCUE AUDIT ON FROZEN MODELS (P-1644, P-1650)
    # --------------------------------------------------------------------------
    print("\n--- STEP 3: CORRECTED OPERATIONAL RESCUE AUDIT ON FROZEN MODELS ---")
    trained_models = torch.load(FINAL_MODELS_PTH, map_location="cpu")

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

    corrected_seed_records = {}
    seed_rescued_aids_corr = defaultdict(set)
    seed_rescued_aids_wr = defaultdict(set)

    # Hierarchical loss storage: seed -> video -> condition -> list of frame breakdowns (P-1651)
    loss_breakdowns_hierarchical = {
        s: {
            v: {"D": [], "F_correct": [], "F_wrong": []}
            for v in [v_test_a, v_test_b]
        }
        for s in SEEDS
    }

    for seed in SEEDS:
        probe = SpatiallyAnchoredFeatureProbe(d_model=256, num_levels=4, stencil_radius_px=STENCIL_RADIUS_PX).to(device)
        probe.load_state_dict(trained_models[str(seed)])
        probe.to(device)
        probe.eval()

        seed_rec = {}
        with torch.no_grad():
            for v, f_data, c_correct, c_wrong, n_expected_tgts in [
                (v_test_a, test_frame_data[v_test_a], caches_a, caches_b, n_ta),
                (v_test_b, test_frame_data[v_test_b], caches_b, caches_a, n_tb)
            ]:
                r_corr_cnt = 0
                r_wr_cnt = 0
                n_v_tgts = 0

                for idx, fd in enumerate(f_data):
                    s_curr = collate_fn([dataset[id_to_idx[fd["image_id"]]]])
                    tgt_dict = {k: v_t.to(device) for k, v_t in s_curr[1][0].items()}

                    fc_corr = c_correct[idx]
                    fc_wr = c_wrong[idx % len(c_wrong)]

                    l_d, _ = evaluate_video_frame_corrected(model, criterion, s_curr[0].to(device), tgt_dict, fd["targets"], label_to_cat_id, None, None, False)
                    l_corr, r_corr = evaluate_video_frame_corrected(model, criterion, s_curr[0].to(device), tgt_dict, fd["targets"], label_to_cat_id, fc_corr, probe, True)
                    l_wr, r_wr = evaluate_video_frame_corrected(model, criterion, s_curr[0].to(device), tgt_dict, fd["targets"], label_to_cat_id, fc_wr, probe, True)

                    loss_breakdowns_hierarchical[seed][v]["D"].append(l_d)
                    loss_breakdowns_hierarchical[seed][v]["F_correct"].append(l_corr)
                    loss_breakdowns_hierarchical[seed][v]["F_wrong"].append(l_wr)

                    for t in fd["targets"]:
                        aid = t["ann_id"]
                        n_v_tgts += 1
                        if r_corr.get(aid, False):
                            r_corr_cnt += 1
                            seed_rescued_aids_corr[seed].add(aid)
                        if r_wr.get(aid, False):
                            r_wr_cnt += 1
                            seed_rescued_aids_wr[seed].add(aid)

                # D-3527: Assert live evaluated target count matches expected
                assert n_v_tgts == n_expected_tgts, f"[FATAL] Evaluated {n_v_tgts} targets on video {v}, expected {n_expected_tgts}"

                seed_rec[v] = {
                    "rescued_D": 0,
                    "rescued_F_correct": r_corr_cnt,
                    "rescued_F_wrong": r_wr_cnt,
                    "total_targets": n_expected_tgts
                }

        corrected_seed_records[seed] = seed_rec

        tot_corr = seed_rec[v_test_a]["rescued_F_correct"] + seed_rec[v_test_b]["rescued_F_correct"]
        tot_wr = seed_rec[v_test_a]["rescued_F_wrong"] + seed_rec[v_test_b]["rescued_F_wrong"]
        print(f"  Seed {seed} Corrected Rescues (Gamma = {probe.gamma.item():.4f}):")
        print(f"    - video_{v_test_a} ({n_ta} tgts): F_correct = {seed_rec[v_test_a]['rescued_F_correct']} | F_wrong = {seed_rec[v_test_a]['rescued_F_wrong']} | D = 0")
        print(f"    - video_{v_test_b} ({n_tb} tgts): F_correct = {seed_rec[v_test_b]['rescued_F_correct']} | F_wrong = {seed_rec[v_test_b]['rescued_F_wrong']} | D = 0")
        print(f"    - Pooled Total:         F_correct = {tot_corr:2d}/40 | F_wrong = {tot_wr:2d}/40 | D = 0/40")

    # --------------------------------------------------------------------------
    # STEP 4: MULTI-SEED RESCUE CONSISTENCY & LOSS DECOMPOSITION (P-1639, P-1650, P-1651)
    # --------------------------------------------------------------------------
    all_test_aids = {t["ann_id"] for t in split_data["test_targets"]}
    corr_freq = {aid: sum(1 for s in SEEDS if aid in seed_rescued_aids_corr[s]) for aid in all_test_aids}
    wr_freq = {aid: sum(1 for s in SEEDS if aid in seed_rescued_aids_wr[s]) for aid in all_test_aids}

    print("\n" + "=" * 115)
    print("CORRECTED OPERATIONAL TARGET RESCUE REPORT (D-3521, P-1639)")
    print("=" * 115)
    mean_corr_resc = float(np.mean([len(seed_rescued_aids_corr[s]) for s in SEEDS]))
    mean_wr_resc = float(np.mean([len(seed_rescued_aids_wr[s]) for s in SEEDS]))
    print(f"Tri-Seed Target Rescue Means (out of 40):")
    print(f"  - Condition D (Static):        0.00 / 40 (0.0%)")
    print(f"  - Condition F_correct (True):  {mean_corr_resc:.2f} / 40 ({mean_corr_resc/40.0*100:.1f}%)")
    print(f"  - Condition F_wrong   (Swap):  {mean_wr_resc:.2f} / 40 ({mean_wr_resc/40.0*100:.1f}%)")
    print(f"  - Delta (Correct - Wrong):     {mean_corr_resc - mean_wr_resc:+.2f} targets")

    print(f"\nAnnotation-Level Target Consistency across Seeds:")
    print(f"  - F_correct: In >=1 seed = {sum(1 for c in corr_freq.values() if c >= 1):2d} | 1 seed = {sum(1 for c in corr_freq.values() if c == 1):2d} | 2 seeds = {sum(1 for c in corr_freq.values() if c == 2):2d} | All 3 seeds = {sum(1 for c in corr_freq.values() if c == 3):2d}")
    print(f"  - F_wrong:   In >=1 seed = {sum(1 for c in wr_freq.values() if c >= 1):2d} | 1 seed = {sum(1 for c in wr_freq.values() if c == 1):2d} | 2 seeds = {sum(1 for c in wr_freq.values() if c == 2):2d} | All 3 seeds = {sum(1 for c in wr_freq.values() if c == 3):2d}")

    # Compute video-level and video-macro loss decompositions (D-3549, P-1651)
    def summarize_breakdowns(breakdown_list):
        return {
            "loss_total": float(np.mean([b["loss_total"] for b in breakdown_list])),
            "primary": {
                "loss_total": float(np.mean([b["primary"]["loss_total"] for b in breakdown_list])),
                "loss_ce": float(np.mean([b["primary"]["loss_ce"] for b in breakdown_list])),
                "loss_bbox": float(np.mean([b["primary"]["loss_bbox"] for b in breakdown_list])),
                "loss_giou": float(np.mean([b["primary"]["loss_giou"] for b in breakdown_list]))
            },
            "all_heads": {
                "loss_total": float(np.mean([b["all_heads"]["loss_total"] for b in breakdown_list])),
                "loss_ce": float(np.mean([b["all_heads"]["loss_ce"] for b in breakdown_list])),
                "loss_bbox": float(np.mean([b["all_heads"]["loss_bbox"] for b in breakdown_list])),
                "loss_giou": float(np.mean([b["all_heads"]["loss_giou"] for b in breakdown_list]))
            }
        }

    video_decompositions = {}
    for v in [v_test_a, v_test_b]:
        video_decompositions[str(v)] = {}
        for cond in ["D", "F_correct", "F_wrong"]:
            # Combine all seed evaluations for this video
            all_v_b = []
            for s in SEEDS:
                all_v_b.extend(loss_breakdowns_hierarchical[s][v][cond])
            video_decompositions[str(v)][cond] = summarize_breakdowns(all_v_b)

    # 2-video macro decomposition
    macro_decomposition = {}
    for cond in ["D", "F_correct", "F_wrong"]:
        macro_decomposition[cond] = {
            "loss_total": (video_decompositions[str(v_test_a)][cond]["loss_total"] + video_decompositions[str(v_test_b)][cond]["loss_total"]) / 2.0,
            "primary": {
                term: (video_decompositions[str(v_test_a)][cond]["primary"][term] + video_decompositions[str(v_test_b)][cond]["primary"][term]) / 2.0
                for term in ["loss_total", "loss_ce", "loss_bbox", "loss_giou"]
            },
            "all_heads": {
                term: (video_decompositions[str(v_test_a)][cond]["all_heads"][term] + video_decompositions[str(v_test_b)][cond]["all_heads"][term]) / 2.0
                for term in ["loss_total", "loss_ce", "loss_bbox", "loss_giou"]
            }
        }

    print("\n" + "=" * 115)
    print("VIDEO-LEVEL & MACRO LOSS DECOMPOSITION (D-3548, D-3549, P-1650, P-1651)")
    print("=" * 115)
    for v_lbl, v_key in [(f"video_{v_test_a} ({n_ta} frames)", str(v_test_a)), (f"video_{v_test_b} ({n_tb} frames)", str(v_test_b)), ("2-Video Macro Mean", "macro")]:
        d_src = video_decompositions[v_key] if v_key != "macro" else macro_decomposition
        print(f"\n[{v_lbl}] All-Heads Loss Family Summary:")
        print(f"{'Condition':<14} | {'Total Loss':<14} | {'Loss CE (All)':<18} | {'Loss Bbox (All)':<18} | {'Loss GIoU (All)':<18}")
        print("-" * 88)
        for cond in ["D", "F_correct", "F_wrong"]:
            ah = d_src[cond]["all_heads"]
            print(f"{cond:<14} | {ah['loss_total']:<14.4f} | {ah['loss_ce']:<18.4f} | {ah['loss_bbox']:<18.4f} | {ah['loss_giou']:<18.4f}")

        d_ce_wr = d_src["F_wrong"]["all_heads"]["loss_ce"] - d_src["F_correct"]["all_heads"]["loss_ce"]
        d_bb_wr = d_src["F_wrong"]["all_heads"]["loss_bbox"] - d_src["F_correct"]["all_heads"]["loss_bbox"]
        d_gi_wr = d_src["F_wrong"]["all_heads"]["loss_giou"] - d_src["F_correct"]["all_heads"]["loss_giou"]
        print(f"  Delta [F_wrong - F_correct]: CE = {d_ce_wr:+.6f} | Bbox = {d_bb_wr:+.6f} | GIoU = {d_gi_wr:+.6f}")

    # --------------------------------------------------------------------------
    # STEP 5: SERIALIZE CORRECTED AUDIT MANIFEST (D-3540, D-3550, P-1652)
    # --------------------------------------------------------------------------
    target_rescue_frequency = {
        "F_correct": {str(aid): corr_freq[aid] for aid in sorted(all_test_aids)},
        "F_wrong": {str(aid): wr_freq[aid] for aid in sorted(all_test_aids)}
    }
    rescued_aids_by_seed = {
        "F_correct": {str(s): sorted(list(seed_rescued_aids_corr[s])) for s in SEEDS},
        "F_wrong": {str(s): sorted(list(seed_rescued_aids_wr[s])) for s in SEEDS}
    }

    audit_payload = {
        "_meta": {
            "artifact_name": "pol2_candidate2f_rescue_corrected_v1",
            "role": "CORRECTED_OPERATIONAL_RESCUE_AUDIT_SUPERSEDING_RESCUE_IN_4fc31dd9",
            "original_results_manifest_sha256": EXPECTED_RESULTS_SHA256,
            "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
            "final_models_sha256": EXPECTED_FINAL_MODELS_SHA256,
            "m0_checkpoint_sha256": EXPECTED_M0_HASH,
            "m0_prediction_cache_sha256": EXPECTED_CACHE_SHA256,
            "correction_type": "canonical_top100_label_to_cat_id_mapping",
            "calibration_miss_reproduction": "VERIFIED_0_RESCUES_ON_40_TARGETS",
            "test_static_miss_reproduction": "VERIFIED_0_RESCUES_ON_40_TARGETS",
            "counterfactual_mapping_verified": True,
            "seeds": SEEDS
        },
        "corrected_seed_records": {str(s): corrected_seed_records[s] for s in SEEDS},
        "tri_seed_rescue_means": {
            "condition_D": 0.0,
            "condition_F_correct": mean_corr_resc,
            "condition_F_wrong": mean_wr_resc,
            "delta_mean_correct_vs_wrong": mean_corr_resc - mean_wr_resc
        },
        "target_consistency": {
            "correct_in_any_seed": sum(1 for c in corr_freq.values() if c >= 1),
            "correct_in_1_of_3": sum(1 for c in corr_freq.values() if c == 1),
            "correct_in_2_of_3": sum(1 for c in corr_freq.values() if c == 2),
            "correct_in_all_3": sum(1 for c in corr_freq.values() if c == 3),
            "wrong_in_any_seed": sum(1 for c in wr_freq.values() if c >= 1),
            "wrong_in_1_of_3": sum(1 for c in wr_freq.values() if c == 1),
            "wrong_in_2_of_3": sum(1 for c in wr_freq.values() if c == 2),
            "wrong_in_all_3": sum(1 for c in wr_freq.values() if c == 3)
        },
        "target_rescue_frequency": target_rescue_frequency,
        "rescued_aids_by_seed": rescued_aids_by_seed,
        "video_decompositions": video_decompositions,
        "macro_decomposition": macro_decomposition
    }

    with open(CORRECTED_AUDIT_JSON, "w", encoding="utf-8") as f:
        json.dump(audit_payload, f, indent=2)

    corr_hash = compute_sha256(CORRECTED_AUDIT_JSON)
    print("\n" + "=" * 115)
    print(f"[+] Corrected Rescue Audit Manifest Frozen: {CORRECTED_AUDIT_JSON.resolve()}")
    print(f"    Immutable Manifest SHA-256:             {corr_hash}")
    print("=" * 115)


if __name__ == "__main__":
    main()