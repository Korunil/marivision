"""
D:/MARIVISION/H0_workbench/eval_tools/run_pol2_candidate2m_gate_sweep.py
Candidate 2M Epoch-8 Scalar Gate Calibration Sweep & Admission Forensics (Patched v3)

Strict Compliance Checklist:
  - D-3660: Freezes Epoch 8 as the optimal calibration checkpoint (13/44 rescues).
  - P-1726: Sweeps scalar temporal gate gamma over {10, 4, 2, 1, 0, -1, -2, -3, -4, -5, -6, -8} on calibration only.
  - P-1737: Certifies that Epoch-8 deterministic replay reproduces v3 behavior (13/44 correct, 0/44 shifted).
  - P-1738: Persists and hashes the exact Epoch-8 adapter state to pol2_candidate2m_seed41_epoch8.pth.
  - P-1739: Serializes complete operational metrics (tp, fp, fn, precision, recall) for each gamma.
  - P-1740: Separates raw Top-100 query occupancy from thresholded active predictions (s >= 0.20).
  - P-1734: Canonical six-output admission forensics helper with 7-tuple return matching all call sites.
  - P-1735: Dedicated baseline evaluation helper computing exact pooled F1_D on calibration frames.
  - P-1736: Strict Epoch-8 reproduction guards (category-aware, <1e-4 tolerance, unique GT indices).
  - P-1731: Admissibility tie-break uses -gamma (prioritizes stronger suppression).
  - Fail-Fast Gate: If no gamma passes, Candidate 2M is closed permanently; protected videos 26/40 remain unspent.
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
from collections import deque, defaultdict
from pycocotools.coco import COCO

# Add Deformable-DETR to sys.path
DETR_PATH = Path("D:/MARIVISION/Deformable-DETR")
if str(DETR_PATH) not in sys.path:
    sys.path.insert(0, str(DETR_PATH))

if "R0_MSDA_BACKEND" not in os.environ:
    os.environ["R0_MSDA_BACKEND"] = "pytorch"

from models.deformable_detr import build, DeformableDETR
from models.deformable_transformer import DeformableTransformer, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from r0_preflight import get_authoritative_r0_args
from datasets.seaclips_detection import SeaClipsDetection
from util.misc import collate_fn, NestedTensor, inverse_sigmoid, nested_tensor_from_tensor_list

OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
CANDIDATES_JSON = OUT_DIR / "pol2_candidate_windows_v1.json"
TRAIN_JSON = Path("D:/MARIVISION/seaclips-train.json")
CACHE_FILE = OUT_DIR / "cache_m0_train_raw_top100.json"
DEV_POOL_JSON = OUT_DIR / "pol2_dev_pool_manifest.json"
SPLIT_MANIFEST_JSON = OUT_DIR / "pol2_candidate2m_micro_split.json"
TOKEN_CACHE_PTH = OUT_DIR / "cache_m0_candidate2m_design_calib.pth"
M0_CKPT_PATH = Path("D:/MARIVISION/checkpoints_r0/checkpoint_best.pth")
ZIP_PATH = Path("D:/MARIVISION/SeaClips.zip")

EPOCH8_MODEL_PTH = OUT_DIR / "pol2_candidate2m_seed41_epoch8.pth"
GATE_SWEEP_RESULTS_JSON = OUT_DIR / "pol2_candidate2m_gate_sweep_results.json"

EXPECTED_CANDIDATE_SHA256 = "f29e0b8a5cbfc46641222d84a6d370c3e9bc51ca4ca1124b7753078ca7c1e270"
EXPECTED_TRAIN_JSON_SHA256 = "edc32bdbd6b792eb83b6daf11e14708232f1f4a9e06153ebebd5bad1caef9b35"
EXPECTED_CACHE_SHA256 = "d9816083760d698935b44e9165522e709b490c3af29e710d75b8b0d2d90210e0"
EXPECTED_DEV_POOL_SHA256 = "533567e1d74544db7087a95253e640ef0959b998e5d503f57b0eb8074d8fe970"
EXPECTED_SPLIT_SHA256 = "7da471d2793fe3369508200dec50422bb76492860c67a1a61871645471856772"
EXPECTED_TOKEN_CACHE_SHA256 = "d1e6407722c59968f87e0ae9cfdead28cacf89924dcd765ca41cf8993da4dd4c"
EXPECTED_M0_HASH = "28e6288c3f6769c2d5275aa77ddb248735b55a09cd46a6b570e3f117890b268b"

MEMORY_AGES = [2, 3, 4, 5]
MAX_TOKENS_PER_AGE = 8
SCORE_THRESHOLD = 0.20
EVAL_IOU_THRESHOLD = 0.30
LAMBDA_MI = 1.0
MAX_GRAD_NORM = 0.1
TARGET_EPOCH = 8
GAMMA_GRID = [10.0, 4.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -8.0]


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
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_shifted_memory_bank(mb):
    shifted = []
    for tok in mb:
        t_s = copy.deepcopy(tok)
        cx, cy, w, h = t_s["bbox"]
        t_s["bbox"] = [(cx + 0.25) % 1.0, (cy + 0.25) % 1.0, w, h]
        shifted.append(t_s)
    return shifted


# ------------------------------------------------------------------------------
# 1. CANDIDATE 2M RESIDUAL MODULE (P-1696)
# ------------------------------------------------------------------------------

class HistoricalObjectTokenMemory(nn.Module):
    def __init__(self, d_model=256, num_classes=7, max_age=6):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes

        self.cls_proj = nn.Linear(num_classes, 64)
        self.age_embed = nn.Embedding(max_age + 1, 64)
        self.norm = nn.LayerNorm(d_model + 64 + 1 + 64)
        self.token_proj = nn.Linear(d_model + 64 + 1 + 64, d_model)
        self.pos_proj = nn.Linear(d_model, d_model)

        self.gamma = nn.Parameter(torch.tensor(10.0, dtype=torch.float32), requires_grad=False)

    def instantiate_temporal_queries(self, memory_tokens, device):
        if not memory_tokens or len(memory_tokens) == 0:
            return None, None, None

        K = len(memory_tokens)
        z_stack = torch.stack([m["z"].to(device) for m in memory_tokens], dim=0)
        logits_stack = torch.stack([m["logits"].to(device) for m in memory_tokens], dim=0)
        scores_stack = torch.tensor([[m["score"]] for m in memory_tokens], device=device, dtype=torch.float32)
        ages_stack = torch.tensor([m["age"] for m in memory_tokens], device=device, dtype=torch.long)

        ref_coords = torch.tensor([[m["bbox"][0], m["bbox"][1]] for m in memory_tokens], device=device, dtype=torch.float32)
        ref_coords = ref_coords.clamp(min=1e-4, max=1.0 - 1e-4)

        cls_feat = F.relu(self.cls_proj(logits_stack))
        age_feat = self.age_embed(ages_stack)

        concat_feat = torch.cat([z_stack, cls_feat, scores_stack, age_feat], dim=-1)
        delta_tgt = self.token_proj(self.norm(concat_feat))
        tgt_mem = z_stack + delta_tgt

        inv_ref = inverse_sigmoid(ref_coords)
        pos_proj_in = torch.cat([inv_ref, inv_ref.new_zeros((K, self.d_model - 2))], dim=-1)
        pos_mem = self.pos_proj(pos_proj_in)

        return tgt_mem, pos_mem, ref_coords


def init_identity_preserving(adapter: HistoricalObjectTokenMemory):
    with torch.no_grad():
        adapter.token_proj.weight.zero_()
        adapter.token_proj.bias.zero_()
        adapter.pos_proj.weight.zero_()
        adapter.pos_proj.bias.zero_()
        adapter.norm.weight.fill_(1.0)
        adapter.norm.bias.zero_()
        nn.init.kaiming_uniform_(adapter.cls_proj.weight, a=1.0)
        adapter.cls_proj.bias.fill_(0.01)
        nn.init.normal_(adapter.age_embed.weight, std=0.02)
        adapter.gamma.data.fill_(10.0)


# ------------------------------------------------------------------------------
# 2. SOURCE HOOKS & ASYMMETRIC ATTENTION MASKING (D-3582, P-1729)
# ------------------------------------------------------------------------------

def apply_candidate2m_source_hooks():
    def patched_layer_forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None, attn_mask=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1), attn_mask=attn_mask)[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt = self.forward_ffn(tgt)
        return tgt

    DeformableTransformerDecoderLayer.forward = patched_layer_forward

    def patched_decoder_forward(self, tgt, reference_points, memory,
                                spatial_shapes, level_start_index, valid_ratios,
                                query_pos=None, src_padding_mask=None, attn_mask=None):
        output = tgt
        intermediate = []
        intermediate_reference_points = []

        for layer_id, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]

            output = layer(output, query_pos, reference_points_input, memory,
                           spatial_shapes, level_start_index, src_padding_mask, attn_mask=attn_mask)

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points

    DeformableTransformerDecoder.forward = patched_decoder_forward

    def patched_transformer_forward(self, srcs, masks, pos_embeds, query_embed=None,
                                    temporal_adapter=None, memory_bank=None, temporal_enabled=True):
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
        query_embed_canon, tgt_canon = torch.split(query_embed, c, dim=1)
        query_embed_canon = query_embed_canon.unsqueeze(0).expand(bs, -1, -1)
        tgt_canon = tgt_canon.unsqueeze(0).expand(bs, -1, -1)
        ref_canon = self.reference_points(query_embed_canon).sigmoid()

        attn_mask = None
        k_mem_count = 0

        if temporal_enabled and temporal_adapter is not None and memory_bank is not None:
            tgt_mem, pos_mem, ref_mem = temporal_adapter.instantiate_temporal_queries(
                memory_bank[0] if isinstance(memory_bank, list) and len(memory_bank) > 0 else memory_bank,
                device=memory.device
            )
            if tgt_mem is not None:
                k_mem_count = tgt_mem.shape[0]
                tgt_all = torch.cat([tgt_canon, tgt_mem.unsqueeze(0).expand(bs, -1, -1)], dim=1)
                pos_all = torch.cat([query_embed_canon, pos_mem.unsqueeze(0).expand(bs, -1, -1)], dim=1)
                ref_all = torch.cat([ref_canon, ref_mem.unsqueeze(0).expand(bs, -1, -1)], dim=1)

                total_q = 300 + k_mem_count
                attn_mask = torch.zeros((total_q, total_q), dtype=torch.float32, device=memory.device)
                attn_mask[:300, 300:] = float('-inf')
            else:
                tgt_all, pos_all, ref_all = tgt_canon, query_embed_canon, ref_canon
        else:
            tgt_all, pos_all, ref_all = tgt_canon, query_embed_canon, ref_canon

        hs, inter_references = self.decoder(
            tgt_all, ref_all, memory,
            spatial_shapes, level_start_index, valid_ratios,
            pos_all, mask_flatten, attn_mask=attn_mask
        )
        return hs, ref_all, inter_references, k_mem_count

    DeformableTransformer.forward = patched_transformer_forward

    def patched_detr_forward(self, samples: NestedTensor,
                             temporal_adapter=None, memory_bank=None, temporal_enabled=True,
                             return_decoder_tokens=False):
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
        hs, init_reference, inter_references, k_mem_count = self.transformer(
            srcs, masks, pos, query_embeds,
            temporal_adapter=temporal_adapter,
            memory_bank=memory_bank,
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

            if temporal_enabled and temporal_adapter is not None and k_mem_count > 0:
                gate_scale = torch.sigmoid(temporal_adapter.gamma)
                logits_canon = outputs_class[:, :300, :]
                logits_mem = outputs_class[:, 300:, :]
                logits_mem_gated = logits_mem + torch.log(gate_scale + 1e-7)
                outputs_class = torch.cat([logits_canon, logits_mem_gated], dim=1)

            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        if return_decoder_tokens:
            return out, hs[-1]

        return out

    DeformableDETR.forward = patched_detr_forward


# ------------------------------------------------------------------------------
# 3. MATCHED-POSITIVE AUXILIARY LOSS IMPLEMENTATION (P-1716, P-1720)
# ------------------------------------------------------------------------------

def compute_matched_positive_temporal_loss(matcher, criterion, out_temp, targets_list):
    def loss_single_layer(preds):
        matches = matcher(preds, targets_list)
        loss_ce = torch.tensor(0.0, device=preds["pred_logits"].device)
        loss_bbox = torch.tensor(0.0, device=preds["pred_boxes"].device)
        loss_giou = torch.tensor(0.0, device=preds["pred_boxes"].device)

        total_matched = 0
        for b_idx, (src_idx, tgt_idx) in enumerate(matches):
            if len(src_idx) == 0:
                continue
            total_matched += len(src_idx)

            matched_logits = preds["pred_logits"][b_idx, src_idx]
            target_labels = targets_list[b_idx]["labels"][tgt_idx]
            target_onehot = torch.zeros_like(matched_logits)
            target_onehot.scatter_(1, target_labels.unsqueeze(-1), 1.0)

            ce = F.binary_cross_entropy_with_logits(matched_logits, target_onehot, reduction="none")
            p = matched_logits.sigmoid()
            p_t = p * target_onehot + (1 - p) * (1 - target_onehot)
            alpha = getattr(criterion, 'focal_alpha', 0.25)
            alpha_t = alpha * target_onehot + (1 - alpha) * (1 - target_onehot)
            loss_ce = loss_ce + (alpha_t * ce * ((1 - p_t) ** 2)).sum()

            matched_boxes = preds["pred_boxes"][b_idx, src_idx]
            target_boxes = targets_list[b_idx]["boxes"][tgt_idx]
            loss_bbox = loss_bbox + F.l1_loss(matched_boxes, target_boxes, reduction="sum")
            loss_giou = loss_giou + (1.0 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(matched_boxes), box_cxcywh_to_xyxy(target_boxes)))).sum()

        num_boxes = max(1, total_matched)
        w_ce = criterion.weight_dict.get("loss_ce", 2.0)
        w_bbox = criterion.weight_dict.get("loss_bbox", 5.0)
        w_giou = criterion.weight_dict.get("loss_giou", 2.0)

        layer_losses = {
            "loss_ce": (loss_ce / num_boxes) * w_ce,
            "loss_bbox": (loss_bbox / num_boxes) * w_bbox,
            "loss_giou": (loss_giou / num_boxes) * w_giou
        }
        layer_total = layer_losses["loss_ce"] + layer_losses["loss_bbox"] + layer_losses["loss_giou"]
        return layer_total, layer_losses, total_matched

    total_loss, primary_losses, n_matched = loss_single_layer(out_temp)
    if "aux_outputs" in out_temp:
        for aux_preds in out_temp["aux_outputs"]:
            l_aux, _, _ = loss_single_layer(aux_preds)
            total_loss = total_loss + l_aux

    return total_loss, primary_losses, n_matched


# ------------------------------------------------------------------------------
# 4. BASELINE & ADMISSION FORENSICS EVALUATION HELPERS (P-1734, P-1735, P-1740)
# ------------------------------------------------------------------------------

def evaluate_baseline_operational(model, samples, targets_dict, eval_targets, label_to_cat_id):
    """
    P-1735: Evaluates unaugmented baseline M0 over 300 canonical queries.
    Returns: rescued_map, frame_tp, frame_fp, frame_fn, base_pairs
    """
    out_d = model(samples, temporal_enabled=False)
    prob_d = out_d["pred_logits"][0].sigmoid()
    num_classes = prob_d.shape[-1]
    topk_scores, topk_idx = torch.topk(prob_d.flatten(), 100)
    query_indices = torch.div(topk_idx, num_classes, rounding_mode="floor")
    class_labels = topk_idx % num_classes
    boxes_ranked = out_d["pred_boxes"][0][query_indices]

    base_pairs = set(zip(query_indices.tolist(), class_labels.tolist()))
    pred_cat_ids = np.array([label_to_cat_id[int(lbl.item())] for lbl in class_labels])

    img_h, img_w = targets_dict["orig_size"][0].item(), targets_dict["orig_size"][1].item()
    pred_x1 = (boxes_ranked[:, 0] - boxes_ranked[:, 2] / 2.0) * img_w
    pred_y1 = (boxes_ranked[:, 1] - boxes_ranked[:, 3] / 2.0) * img_h
    pred_w = boxes_ranked[:, 2] * img_w
    pred_h = boxes_ranked[:, 3] * img_h
    pred_boxes_xywh = torch.stack([pred_x1, pred_y1, pred_w, pred_h], dim=-1).cpu().numpy()
    pred_scores_np = topk_scores.cpu().numpy()

    rescued_map = {}
    for t in eval_targets:
        aid = t["ann_id"]
        gt_box = t["bbox"]
        gt_cat = t["category_id"]

        is_rescued = False
        for p_idx in range(100):
            if pred_scores_np[p_idx] >= SCORE_THRESHOLD and pred_cat_ids[p_idx] == gt_cat:
                if box_iou(gt_box, pred_boxes_xywh[p_idx]) >= EVAL_IOU_THRESHOLD:
                    is_rescued = True
                    break
        rescued_map[aid] = is_rescued

    gt_boxes_norm = targets_dict["boxes"].cpu()
    gt_labels_raw = targets_dict["labels"].cpu()
    n_gt = len(gt_labels_raw)

    gt_boxes_xywh = []
    gt_cats_list = []
    for g_idx in range(n_gt):
        gcx, gcy, gw, gh = gt_boxes_norm[g_idx].tolist()
        gx1 = (gcx - gw / 2.0) * img_w
        gy1 = (gcy - gh / 2.0) * img_h
        gt_boxes_xywh.append([gx1, gy1, gw * img_w, gh * img_h])
        gt_cats_list.append(label_to_cat_id[gt_labels_raw[g_idx].item()])

    gt_matched = [False] * n_gt
    frame_tp = 0; frame_fp = 0
    for p_idx in range(100):
        if pred_scores_np[p_idx] >= SCORE_THRESHOLD:
            p_box = pred_boxes_xywh[p_idx]
            p_cat = pred_cat_ids[p_idx]
            best_g = -1
            best_iou = -1.0
            for g_idx in range(n_gt):
                if not gt_matched[g_idx] and gt_cats_list[g_idx] == p_cat:
                    iou_g = box_iou(gt_boxes_xywh[g_idx], p_box)
                    if iou_g >= EVAL_IOU_THRESHOLD and iou_g > best_iou:
                        best_iou = iou_g
                        best_g = g_idx
            if best_g >= 0:
                gt_matched[best_g] = True
                frame_tp += 1
            else:
                frame_fp += 1

    frame_fn = n_gt - sum(gt_matched)
    return rescued_map, frame_tp, frame_fp, frame_fn, base_pairs


def evaluate_frame_admission_forensics(model, samples, targets_dict, eval_targets, label_to_cat_id, memory_bank, adapter, base_pairs=None):
    """
    P-1734 / P-1740: Canonical seven-output admission forensics helper.
    Returns: rescued_map, frame_tp, frame_fp, frame_fn, displaced_count, temporal_entries_top100, temporal_entries_ge_s20
    """
    out_aug = model(samples, temporal_adapter=adapter, memory_bank=memory_bank, temporal_enabled=True)

    prob_aug = out_aug["pred_logits"][0].sigmoid()
    num_classes = prob_aug.shape[-1]
    topk_aug_scores, topk_aug_idx = torch.topk(prob_aug.flatten(), 100)
    topk_aug_q_idx = torch.div(topk_aug_idx, num_classes, rounding_mode="floor")
    topk_aug_c_idx = topk_aug_idx % num_classes
    boxes_ranked = out_aug["pred_boxes"][0][topk_aug_q_idx]

    aug_pairs = set(zip(topk_aug_q_idx.tolist(), topk_aug_c_idx.tolist()))
    displaced_count = len(base_pairs.difference(aug_pairs)) if base_pairs is not None else 0

    temporal_entries_top100 = sum(1 for q in topk_aug_q_idx.tolist() if q >= 300)
    temporal_entries_ge_s20 = sum(1 for q, s in zip(topk_aug_q_idx.tolist(), topk_aug_scores.tolist()) if q >= 300 and s >= SCORE_THRESHOLD)

    pred_cat_ids = np.array([label_to_cat_id[int(lbl.item())] for lbl in topk_aug_c_idx])

    img_h, img_w = targets_dict["orig_size"][0].item(), targets_dict["orig_size"][1].item()
    pred_x1 = (boxes_ranked[:, 0] - boxes_ranked[:, 2] / 2.0) * img_w
    pred_y1 = (boxes_ranked[:, 1] - boxes_ranked[:, 3] / 2.0) * img_h
    pred_w = boxes_ranked[:, 2] * img_w
    pred_h = boxes_ranked[:, 3] * img_h
    pred_boxes_xywh = torch.stack([pred_x1, pred_y1, pred_w, pred_h], dim=-1).cpu().numpy()
    pred_scores_np = topk_aug_scores.cpu().numpy()

    rescued_map = {}
    for t in eval_targets:
        aid = t["ann_id"]
        gt_box = t["bbox"]
        gt_cat = t["category_id"]

        is_rescued = False
        for p_idx in range(100):
            if pred_scores_np[p_idx] >= SCORE_THRESHOLD and pred_cat_ids[p_idx] == gt_cat:
                if box_iou(gt_box, pred_boxes_xywh[p_idx]) >= EVAL_IOU_THRESHOLD:
                    is_rescued = True
                    break
        rescued_map[aid] = is_rescued

    gt_boxes_norm = targets_dict["boxes"].cpu()
    gt_labels_raw = targets_dict["labels"].cpu()
    n_gt = len(gt_labels_raw)

    gt_boxes_xywh = []
    gt_cats_list = []
    for g_idx in range(n_gt):
        gcx, gcy, gw, gh = gt_boxes_norm[g_idx].tolist()
        gx1 = (gcx - gw / 2.0) * img_w
        gy1 = (gcy - gh / 2.0) * img_h
        gt_boxes_xywh.append([gx1, gy1, gw * img_w, gh * img_h])
        gt_cats_list.append(label_to_cat_id[gt_labels_raw[g_idx].item()])

    gt_matched = [False] * n_gt
    frame_tp = 0; frame_fp = 0
    for p_idx in range(100):
        if pred_scores_np[p_idx] >= SCORE_THRESHOLD:
            p_box = pred_boxes_xywh[p_idx]
            p_cat = pred_cat_ids[p_idx]
            best_g = -1
            best_iou = -1.0
            for g_idx in range(n_gt):
                if not gt_matched[g_idx] and gt_cats_list[g_idx] == p_cat:
                    iou_g = box_iou(gt_boxes_xywh[g_idx], p_box)
                    if iou_g >= EVAL_IOU_THRESHOLD and iou_g > best_iou:
                        best_iou = iou_g
                        best_g = g_idx
            if best_g >= 0:
                gt_matched[best_g] = True
                frame_tp += 1
            else:
                frame_fp += 1

    frame_fn = n_gt - sum(gt_matched)
    return rescued_map, frame_tp, frame_fp, frame_fn, displaced_count, temporal_entries_top100, temporal_entries_ge_s20


# ------------------------------------------------------------------------------
# 5. MAIN GATE SWEEP ENGINE (P-1726, P-1727, P-1728, P-1731, P-1737..P-1740)
# ------------------------------------------------------------------------------

def main():
    driver_sha256 = compute_sha256(Path(__file__))
    print("=" * 115)
    print("CANDIDATE 2M EPOCH-8 SCALAR GATE CALIBRATION SWEEP (PATCHED v3)")
    print(f"Driver Script SHA-256: {driver_sha256}")
    print("=" * 115)

    for p_path in [CANDIDATES_JSON, TRAIN_JSON, CACHE_FILE, DEV_POOL_JSON, SPLIT_MANIFEST_JSON, M0_CKPT_PATH]:
        assert p_path.exists(), f"[FATAL] Missing {p_path}"

    assert compute_sha256(CANDIDATES_JSON) == EXPECTED_CANDIDATE_SHA256
    assert compute_sha256(TRAIN_JSON) == EXPECTED_TRAIN_JSON_SHA256
    assert compute_sha256(CACHE_FILE) == EXPECTED_CACHE_SHA256
    assert compute_sha256(DEV_POOL_JSON) == EXPECTED_DEV_POOL_SHA256
    assert compute_sha256(SPLIT_MANIFEST_JSON) == EXPECTED_SPLIT_SHA256
    assert compute_sha256(M0_CKPT_PATH) == EXPECTED_M0_HASH

    print("[+] All Upstream Manifests & M0 Checkpoint Certified by Immutable SHA-256.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Target: {device} ({torch.cuda.get_device_name(0)})")

    dataset = SeaClipsDetection(ZIP_PATH, TRAIN_JSON, is_train=False, random_flip=False)
    coco = dataset.coco
    cats = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    ann_lookup = {a["id"]: a for a in coco.loadAnns(coco.getAnnIds())}
    img_lookup = {info["id"]: info for info in coco.loadImgs(coco.getImgIds())}
    id_to_idx = {iid: idx for idx, iid in enumerate(sorted(coco.getImgIds()))}

    if hasattr(dataset, "label_to_cat_id"):
        label_to_cat_id = dataset.label_to_cat_id
    else:
        cat_ids = sorted(coco.getCatIds())
        label_to_cat_id = {i: cat_id for i, cat_id in enumerate(cat_ids)}
    cat_id_to_label = {v: k for k, v in label_to_cat_id.items()}

    with open(SPLIT_MANIFEST_JSON, "r", encoding="utf-8") as f:
        split_meta = json.load(f)["_meta"]
    calib_vids = sorted(split_meta["calib_videos"])
    design_vids = sorted(split_meta["train_videos"])

    apply_candidate2m_source_hooks()

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

    assert TOKEN_CACHE_PTH.exists(), f"[FATAL] Missing frozen token cache {TOKEN_CACHE_PTH}!"
    cache_hash = compute_sha256(TOKEN_CACHE_PTH)
    assert cache_hash == EXPECTED_TOKEN_CACHE_SHA256, "[FATAL] Token cache SHA mismatch!"
    hist_token_cache = torch.load(TOKEN_CACHE_PTH, map_location="cpu")
    print(f"[+] Loaded Frozen Token Cache (SHA: {cache_hash}): {len(hist_token_cache)} frames.")

    with open(CANDIDATES_JSON, "r", encoding="utf-8") as f:
        cand_windows = json.load(f)["windows"]

    windows_by_vid = defaultdict(list)
    for w in cand_windows:
        v_id = w["video_id"]
        if v_id in calib_vids and w.get("n_mem_incr_s20", w.get("n_memincr_s20", 0)) >= 1:
            windows_by_vid[v_id].append(w)

    def get_memory_bank_for_window(window, cache_dict):
        curr_fid = window["frame_ids"][-1]
        memory_bank = []
        for f_idx in range(5):
            hist_iid = window["image_ids"][f_idx]
            hist_fid = window["frame_ids"][f_idx]
            age = curr_fid - hist_fid
            if age in MEMORY_AGES and hist_iid in cache_dict:
                for tok in cache_dict[hist_iid]:
                    t_c = copy.deepcopy(tok)
                    t_c["age"] = age
                    memory_bank.append(t_c)
        return memory_bank

    def build_deduplicated_eval_stream(video_windows):
        windows_by_target_iid = defaultdict(list)
        for w in video_windows:
            curr_iid = w["image_ids"][-1]
            windows_by_target_iid[curr_iid].append(w)

        eval_items = []
        for curr_iid, w_list in windows_by_target_iid.items():
            primary_w = w_list[0]
            for w in w_list[1:]:
                assert w["image_ids"] == primary_w["image_ids"]
                assert w["frame_ids"] == primary_w["frame_ids"]
            seen_aids = set()
            merged_eval_targets = []
            for w in w_list:
                for t in w["targets"]:
                    if t.get("mem_incr_s20", t.get("memincr_s20", False)):
                        aid = t["ann_id"]
                        if aid not in seen_aids:
                            seen_aids.add(aid)
                            merged_eval_targets.append({
                                "ann_id": aid,
                                "category_id": ann_lookup[aid]["category_id"],
                                "bbox": ann_lookup[aid]["bbox"]
                            })
            if merged_eval_targets:
                eval_items.append({
                    "window": primary_w,
                    "image_id": curr_iid,
                    "eval_targets": merged_eval_targets
                })
        return eval_items

    stream_calib_33 = build_deduplicated_eval_stream(windows_by_vid[33])
    stream_calib_46 = build_deduplicated_eval_stream(windows_by_vid[46])

    # P-1732 / P-1735: Recompute exact pooled baseline F1_D and memoize base_pairs
    print("\n" + "=" * 115)
    print("RECOMPUTING EXACT POOLED BASELINE F1_D & MEMOIZING BASELINE TOP-100 DETECTIONS (P-1732, P-1735)")
    print("=" * 115)

    total_tp_base = 0; total_fp_base = 0; total_fn_base = 0
    rescued_D_total = 0
    baseline_pairs_cache = {}

    with torch.no_grad():
        for stream_curr in [stream_calib_33, stream_calib_46]:
            for item in stream_curr:
                s_c = collate_fn([dataset[id_to_idx[item["image_id"]]]])
                tgt_dict = {"boxes": s_c[1][0]["boxes"].to(device), "labels": s_c[1][0]["labels"].to(device), "orig_size": s_c[1][0]["orig_size"]}
                r_map, tpd, fpd, fnd, base_pairs = evaluate_baseline_operational(model, s_c[0].to(device), tgt_dict, item["eval_targets"], label_to_cat_id)
                total_tp_base += tpd; total_fp_base += fpd; total_fn_base += fnd
                baseline_pairs_cache[item["image_id"]] = base_pairs
                for t in item["eval_targets"]:
                    if r_map.get(t["ann_id"], False):
                        rescued_D_total += 1

    prec_base = total_tp_base / max(1, total_tp_base + total_fp_base)
    rec_base = total_tp_base / max(1, total_tp_base + total_fn_base)
    exact_baseline_f1_D = 2 * prec_base * rec_base / max(1e-7, prec_base + rec_base)
    print(f"[+] Exact Pooled Baseline F1_D: {exact_baseline_f1_D:.6f} (TP={total_tp_base}, FP={total_fp_base}, FN={total_fn_base}, Rescued_D={rescued_D_total}/44)")

    # Reproduce seed 41 screening run through Epoch 8 (P-1733)
    print("\n" + "=" * 115)
    print("REPRODUCING SEED 41 SCREENING RUN THROUGH EPOCH 8 (P-1733)")
    print("=" * 115)

    SCREEN_SEED = 41
    set_seed(SCREEN_SEED)

    adapter = HistoricalObjectTokenMemory(d_model=256, num_classes=7, max_age=6).to(device)
    init_identity_preserving(adapter)
    optimizer = torch.optim.AdamW([p for p in adapter.parameters() if p.requires_grad], lr=2e-4, weight_decay=1e-4)

    design_windows_by_vid = defaultdict(list)
    for w in cand_windows:
        v_id = w["video_id"]
        if v_id in design_vids and w.get("n_mem_incr_s20", w.get("n_memincr_s20", 0)) >= 1:
            design_windows_by_vid[v_id].append(w)

    rng_train = np.random.default_rng(SCREEN_SEED * 1000 + 42)

    for epoch in range(1, TARGET_EPOCH + 1):
        adapter.train()
        for vid in design_vids:
            w_list = design_windows_by_vid[vid]
            chosen_w = w_list[int(rng_train.integers(0, len(w_list)))]

            target_iid = chosen_w["image_ids"][-1]
            s_t = collate_fn([dataset[id_to_idx[target_iid]]])
            samples_t = s_t[0].to(device)
            tgt_all_t = {"boxes": s_t[1][0]["boxes"].to(device), "labels": s_t[1][0]["labels"].to(device)}

            mi_tgts_t = [t for t in chosen_w["targets"] if t.get("mem_incr_s20", t.get("memincr_s20", False))]
            if not mi_tgts_t: continue

            img_h, img_w = s_t[1][0]["orig_size"][0].item(), s_t[1][0]["orig_size"][1].item()
            mi_indices = []

            # P-1733: Category-aware matching, strict box tolerance (<1e-4), and unique GT checks
            for t_mi in mi_tgts_t:
                ann = ann_lookup[t_mi["ann_id"]]
                target_label = cat_id_to_label[ann["category_id"]]
                bx, by, bw, bh = ann["bbox"]
                t_box = torch.tensor([(bx+bw/2.0)/img_w, (by+bh/2.0)/img_h, bw/img_w, bh/img_h], device=device)

                candidate_mask = tgt_all_t["labels"] == target_label
                assert candidate_mask.any(), f"[FATAL] Category mismatch during target resolution in window {chosen_w['frame_start']}!"
                candidate_idxs = candidate_mask.nonzero().flatten()
                errs = (tgt_all_t["boxes"][candidate_idxs] - t_box).abs().sum(-1)
                assert errs.min().item() < 1e-4, f"[FATAL] Target mapping error {errs.min().item():.6e} exceeds 1e-4 in window {chosen_w['frame_start']}!"
                mi_indices.append(candidate_idxs[errs.argmin()].item())

            assert len(mi_indices) == len(set(mi_indices)), f"[FATAL] Duplicate GT collision during target resolution in window {chosen_w['frame_start']}!"

            tgt_mi_t = {"boxes": tgt_all_t["boxes"][mi_indices], "labels": tgt_all_t["labels"][mi_indices]}
            mb_train = get_memory_bank_for_window(chosen_w, hist_token_cache)
            if not mb_train: continue

            optimizer.zero_grad()
            out_tr = model(samples_t, temporal_adapter=adapter, memory_bank=[mb_train], temporal_enabled=True)
            out_temp_tr = {"pred_logits": out_tr["pred_logits"][:, 300:, :], "pred_boxes": out_tr["pred_boxes"][:, 300:, :]}
            if "aux_outputs" in out_tr:
                out_temp_tr["aux_outputs"] = [{"pred_logits": a["pred_logits"][:, 300:, :], "pred_boxes": a["pred_boxes"][:, 300:, :]} for a in out_tr["aux_outputs"]]

            loss_aux, _, n_m = compute_matched_positive_temporal_loss(criterion.matcher, criterion, out_temp_tr, [tgt_mi_t])
            if n_m > 0:
                loss_total = LAMBDA_MI * loss_aux
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_([p for p in adapter.parameters() if p.requires_grad], max_norm=MAX_GRAD_NORM)
                optimizer.step()

        print(f"    [+] Completed Epoch {epoch} / {TARGET_EPOCH} (Seed 41)")

    # P-1738: Save and hash exact Epoch 8 adapter state
    epoch8_state = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
    torch.save({
        "seed": SCREEN_SEED,
        "epoch": TARGET_EPOCH,
        "adapter_state": epoch8_state,
        "source_driver_sha256": driver_sha256,
    }, EPOCH8_MODEL_PTH)
    epoch8_sha = compute_sha256(EPOCH8_MODEL_PTH)
    print(f"[+] Epoch 8 Adapter State Saved & Frozen: {EPOCH8_MODEL_PTH.resolve()}")
    print(f"    Epoch-8 Adapter SHA-256:             {epoch8_sha}")

    # --------------------------------------------------------------------------
    # EXECUTE SCALAR GATE SWEEP ON EPOCH 8 ADAPTER (P-1726 THROUGH P-1740)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("EXECUTING SCALAR GATE SWEEP ON CALIBRATION SET (v33 & v46, N = 44 TARGETS)")
    print("=" * 115)
    print(f"{'Gamma':<8} | {'Sigmoid(g)':<12} | {'Rescued Corr':<16} | {'Rescued Shifted':<18} | {'Delta R':<10} | {'Displaced Top100':<18} | {'Temporal Entry':<16} | {'Temporal s>=.20':<16} | {'F1_corr':<10} | {'Gate Status'}")
    print("-" * 158)

    sweep_results = {}
    ad_sweep = HistoricalObjectTokenMemory(d_model=256, num_classes=7, max_age=6).to(device)
    ad_sweep.load_state_dict(epoch8_state)
    ad_sweep.eval()

    for g_val in GAMMA_GRID:
        ad_sweep.gamma.data.fill_(g_val)
        sig_g = torch.sigmoid(ad_sweep.gamma).item()

        rescued_corr = 0
        rescued_shifted = 0
        total_tp = 0; total_fp = 0; total_fn = 0
        total_displaced = 0
        total_temp_entries = 0
        total_temp_ge_s20 = 0

        with torch.no_grad():
            for v_curr, stream_curr in [(33, stream_calib_33), (46, stream_calib_46)]:
                for item in stream_curr:
                    s_c = collate_fn([dataset[id_to_idx[item["image_id"]]]])
                    tgt_dict = {"boxes": s_c[1][0]["boxes"].to(device), "labels": s_c[1][0]["labels"].to(device), "orig_size": s_c[1][0]["orig_size"]}

                    mb_corr = get_memory_bank_for_window(item["window"], hist_token_cache)
                    mb_shifted = make_shifted_memory_bank(mb_corr)
                    base_p = baseline_pairs_cache[item["image_id"]]

                    # P-1734: 7-tuple return matching helper
                    res_c, tpc, fpc, fnc, disp_c, temp_e_c, temp_s20_c = evaluate_frame_admission_forensics(
                        model, s_c[0].to(device), tgt_dict, item["eval_targets"], label_to_cat_id, [mb_corr], ad_sweep, base_pairs=base_p
                    )
                    res_s, _, _, _, _, _, _ = evaluate_frame_admission_forensics(
                        model, s_c[0].to(device), tgt_dict, item["eval_targets"], label_to_cat_id, [mb_shifted], ad_sweep, base_pairs=base_p
                    )

                    total_tp += tpc; total_fp += fpc; total_fn += fnc
                    total_displaced += disp_c
                    total_temp_entries += temp_e_c
                    total_temp_ge_s20 += temp_s20_c

                    for t in item["eval_targets"]:
                        aid = t["ann_id"]
                        if res_c.get(aid, False): rescued_corr += 1
                        if res_s.get(aid, False): rescued_shifted += 1

        prec = total_tp / max(1, total_tp + total_fp)
        rec = total_tp / max(1, total_tp + total_fn)
        f1_corr = 2 * prec * rec / max(1e-7, prec + rec)
        delta_r = rescued_corr - rescued_shifted

        # P-1737: Certification check at gamma = +10.0
        if g_val == 10.0:
            assert rescued_corr == 13, f"[FATAL] Epoch-8 replay failed to reproduce v3 rescue! Expected 13, got {rescued_corr}"
            assert rescued_shifted == 0, f"[FATAL] Epoch-8 replay produced non-zero shifted rescue: {rescued_shifted}"
            assert abs(f1_corr - 0.2162) < 0.01, f"[FATAL] Epoch-8 replay F1 diverged from v3! Expected ~0.2162, got {f1_corr:.4f}"
            print(f"    [+] P-1737 Certified: Epoch-8 replay reproduced v3 behavior (13/44 correct, 0/44 shifted, F1={f1_corr:.4f}).")

        # P-1728 Admissibility against exact baseline F1_D
        is_admissible = (delta_r > 0 and rescued_corr > rescued_D_total and f1_corr >= exact_baseline_f1_D)
        status_lbl = "ADMISSIBLE" if is_admissible else "REJECTED"

        # P-1739: Complete metrics serialization
        sweep_results[str(g_val)] = {
            "gamma": g_val,
            "sigmoid_gamma": sig_g,
            "rescued_corr": rescued_corr,
            "rescued_shifted": rescued_shifted,
            "delta_r": delta_r,
            "displaced_top100": total_displaced,
            "temporal_entries_top100": total_temp_entries,
            "temporal_entries_ge_s20": total_temp_ge_s20,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": float(prec),
            "recall": float(rec),
            "f1_corr": float(f1_corr),
            "admissible": is_admissible
        }

        print(f"{g_val:+6.1f}   | {sig_g:10.4f}   | {rescued_corr:2d} / 44 ({rescued_corr/44*100:4.1f}%)    | {rescued_shifted:2d} / 44 ({rescued_shifted/44*100:4.1f}%)    | {delta_r:+3d}       | {total_displaced:16d}   | {total_temp_entries:16d} | {total_temp_ge_s20:16d} | {f1_corr:.4f}    | {status_lbl}")

    admissible_keys = [k for k, v in sweep_results.items() if v["admissible"]]

    print("\n" + "=" * 115)
    print("SCALAR GATE CALIBRATION SWEEP VERDICT (P-1728, P-1731)")
    print("=" * 115)
    if len(admissible_keys) > 0:
        # P-1731: Corrected tie-break: max delta_r, then max f1_corr, then min gamma (-gamma)
        best_gamma_key = max(admissible_keys, key=lambda k: (sweep_results[k]["delta_r"], sweep_results[k]["f1_corr"], -sweep_results[k]["gamma"]))
        best_g_rec = sweep_results[best_gamma_key]
        print(f"[+] ADMISSIBLE GAMMA FOUND: gamma* = {best_g_rec['gamma']} (Sigmoid: {best_g_rec['sigmoid_gamma']:.4f})")
        print(f"    - Correct Rescue: {best_g_rec['rescued_corr']} / 44 | Shifted Rescue: {best_g_rec['rescued_shifted']} / 44 (Delta_R = {best_g_rec['delta_r']:+d})")
        print(f"    - Whole-Detector F1: {best_g_rec['f1_corr']:.4f} (Exact Baseline F1_D: {exact_baseline_f1_D:.4f})")
        print(f"    - Canonical Displacements: {best_g_rec['displaced_top100']} | Thresholded Temporal Entries: {best_g_rec['temporal_entries_ge_s20']}")
        gate_status = "SUCCESS_ADMISSIBLE_GAMMA_FOUND"
    else:
        has_positive_margin = any(v["delta_r"] > 0 for v in sweep_results.values())
        has_improvement_over_D = any(v["rescued_corr"] > rescued_D_total for v in sweep_results.values())
        if not has_positive_margin:
            gate_status = "CALIBRATION_FAILED_MECHANISM_GATE"
        elif not has_improvement_over_D:
            gate_status = "CALIBRATION_FAILED_RESCUE_GATE"
        else:
            gate_status = "CALIBRATION_FAILED_QUALITY_GUARDRAIL"

        print(f"[!] NO ADMISSIBLE GAMMA FOUND: {gate_status}.")
        print("    Candidate 2M is PERMANENTLY CLOSED. Protected test videos 26 and 40 remain unspent.")

    results_payload = {
        "_meta": {
            "artifact_name": "pol2_candidate2m_gate_sweep_results_v3",
            "status": gate_status,
            "driver_sha256": driver_sha256,
            "token_cache_sha256": cache_hash,
            "epoch8_adapter_sha256": epoch8_sha,
            "optimal_target_epoch": TARGET_EPOCH,
            "exact_baseline_f1_D": float(exact_baseline_f1_D),
            "admissible_gamma_found": len(admissible_keys) > 0
        },
        "sweep_results": sweep_results
    }

    with open(GATE_SWEEP_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    res_hash = compute_sha256(GATE_SWEEP_RESULTS_JSON)
    print(f"\n[+] Gate Sweep Results Frozen: {GATE_SWEEP_RESULTS_JSON.resolve()}")
    print(f"    Results Manifest SHA-256:   {res_hash}\n")


if __name__ == "__main__":
    main()