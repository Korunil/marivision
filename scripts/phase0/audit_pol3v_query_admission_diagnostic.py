"""
D:/MARIVISION/H0_workbench/eval_tools/audit_pol3v_query_admission_diagnostic.py
Candidate 3V Pre-Flight: Design-Only Query Admission Diagnostic & Admission Simulation (v7)

Strict Compliance Checklist:
  - D-3675 / D-3676: Strictly restricted to the 10 design videos (Calibration & Test UNTOUCHED).
  - P-1757: Iterative backward pruning until convergence (fixed point): guarantees irreducible final utility set.
  - P-1754: Zero GT leakage: curr_score, class_agreement, and canonical_support derived from inference-available pred_class.
  - P-1755: Greedy forward joint-compatible admission (Delta TP > 0 and Delta FP <= 0).
  - P-1750: Evaluates canonical_support strictly from unaugmented baseline out_d detector outputs.
  - P-1753: Exact Bernoulli entropy for independent sigmoid class probabilities.
  - P-1751: Operational admission simulation evaluates exact whole-detector F1 with admitted queries.
  - Same-History Assertions: Enforces identical historical image/frame chains when merging same-frame windows.
  - Active-Only Primary Audit: Primary LOVO probe evaluates threshold-active temporal queries (s >= 0.20).
  - LOVO Training Guards: Degenerate single-class folds excluded from pooled total counts and AUC calculations.
  - Tie-Aware Metrics: Uses tie-aware ROC-AUC and distinct-threshold PR-AUC.
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
from collections import defaultdict
from pycocotools.coco import COCO

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

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
EPOCH8_MODEL_PTH = OUT_DIR / "pol2_candidate2m_seed41_epoch8.pth"
M0_CKPT_PATH = Path("D:/MARIVISION/checkpoints_r0/checkpoint_best.pth")
ZIP_PATH = Path("D:/MARIVISION/SeaClips.zip")

DIAGNOSTIC_JSON = OUT_DIR / "pol3v_query_admission_diagnostic.json"

EXPECTED_CANDIDATE_SHA256 = "f29e0b8a5cbfc46641222d84a6d370c3e9bc51ca4ca1124b7753078ca7c1e270"
EXPECTED_TRAIN_JSON_SHA256 = "edc32bdbd6b792eb83b6daf11e14708232f1f4a9e06153ebebd5bad1caef9b35"
EXPECTED_CACHE_SHA256 = "d9816083760d698935b44e9165522e709b490c3af29e710d75b8b0d2d90210e0"
EXPECTED_DEV_POOL_SHA256 = "533567e1d74544db7087a95253e640ef0959b998e5d503f57b0eb8074d8fe970"
EXPECTED_SPLIT_SHA256 = "7da471d2793fe3369508200dec50422bb76492860c67a1a61871645471856772"
EXPECTED_TOKEN_CACHE_SHA256 = "d1e6407722c59968f87e0ae9cfdead28cacf89924dcd765ca41cf8993da4dd4c"
EXPECTED_EPOCH8_ADAPTER_SHA256 = "aea024e2c1072fb5f46d1d376ca1b75c0cdab81a20681ac9958b5d6c0b5f00cf"
EXPECTED_M0_HASH = "28e6288c3f6769c2d5275aa77ddb248735b55a09cd46a6b570e3f117890b268b"

MEMORY_AGES = [2, 3, 4, 5]
SCORE_THRESHOLD = 0.20
EVAL_IOU_THRESHOLD = 0.30


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
# 1. CANDIDATE 2M RESIDUAL MODULE (FROZEN EPOCH 8 CHECKPOINT)
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
                reference_points_input = reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
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

        if return_decoder_tokens:
            return out, hs[-1]
        return out

    DeformableDETR.forward = patched_detr_forward


# ------------------------------------------------------------------------------
# 2. HOMOLOGOUS EVIDENCE FEATURE EXTRACTION (15 FEATURES) (P-1750, P-1753, P-1754)
# ------------------------------------------------------------------------------

FEATURE_NAMES = [
    "hist_score",
    "hist_age",
    "hist_class_entropy",
    "hist_class_margin",
    "hist_box_area",
    "curr_score",
    "curr_class_entropy",
    "curr_class_margin",
    "class_agreement",
    "hidden_cos_sim",
    "hidden_l2_dist",
    "box_center_disp",
    "box_scale_ratio",
    "box_iou_hist",
    "canonical_support"
]


def extract_query_evidence_features(tok, h_i_t, b_i_t_norm, logits_i_t, base_d_boxes_px, base_d_logits, img_w, img_h):
    s_hist = float(tok["score"])
    age_hist = float(tok["age"])
    p_hist = torch.sigmoid(tok["logits"]).cpu()
    p_hist_sort, _ = torch.sort(p_hist, descending=True)

    eps = 1e-8
    hist_entropy = float(
        -(p_hist * torch.log(p_hist.clamp(min=eps)) + (1.0 - p_hist) * torch.log((1.0 - p_hist).clamp(min=eps))).sum().item()
    )
    hist_margin = float((p_hist_sort[0] - p_hist_sort[1]).item())
    hist_area = float(tok["bbox"][2] * tok["bbox"][3])

    p_curr = torch.sigmoid(logits_i_t).cpu()
    p_curr_sort, p_curr_idx = torch.sort(p_curr, descending=True)

    # P-1754: Derived strictly from top predicted class without GT leakage
    pred_class = int(p_curr_idx[0].item())
    curr_score = float(p_curr_sort[0].item())
    curr_entropy = float(
        -(p_curr * torch.log(p_curr.clamp(min=eps)) + (1.0 - p_curr) * torch.log((1.0 - p_curr).clamp(min=eps))).sum().item()
    )
    curr_margin = float((p_curr_sort[0] - p_curr_sort[1]).item())

    hist_pred_class = int(torch.argmax(p_hist).item())
    class_agreement = 1.0 if (hist_pred_class == pred_class) else 0.0

    z_hist = tok["z"].to(h_i_t.device)
    cos_sim = float(F.cosine_similarity(h_i_t.unsqueeze(0), z_hist.unsqueeze(0)).item())
    l2_dist = float(torch.norm(h_i_t - z_hist, p=2).item())

    cx_curr, cy_curr, w_curr, h_curr = b_i_t_norm.tolist()
    cx_hist, cy_hist, w_hist, h_hist = tok["bbox"]
    center_disp = float(np.sqrt((cx_curr - cx_hist) ** 2 + (cy_curr - cy_hist) ** 2))
    scale_ratio = float(np.log(max(1e-6, w_curr * h_curr) / max(1e-6, w_hist * h_hist)))

    box_curr_px = [
        (cx_curr - w_curr / 2.0) * img_w,
        (cy_curr - h_curr / 2.0) * img_h,
        w_curr * img_w,
        h_curr * img_h
    ]
    box_hist_px = [
        (cx_hist - w_hist / 2.0) * img_w,
        (cy_hist - h_hist / 2.0) * img_h,
        w_hist * img_w,
        h_hist * img_h
    ]
    iou_hist = float(box_iou(box_curr_px, box_hist_px))

    # P-1750: Local subthreshold canonical support from baseline out_d for the predicted class
    canonical_support = 0.0
    for q_c in range(300):
        b_c_px = base_d_boxes_px[q_c]
        if box_iou(box_curr_px, b_c_px) >= 0.10:
            c_score = float(torch.sigmoid(base_d_logits[q_c, pred_class]).item())
            if c_score > canonical_support:
                canonical_support = c_score

    feat_vec = [
        s_hist, age_hist, hist_entropy, hist_margin, hist_area,
        curr_score, curr_entropy, curr_margin, class_agreement,
        cos_sim, l2_dist, center_disp, scale_ratio, iou_hist,
        canonical_support
    ]
    return feat_vec, box_curr_px, pred_class, curr_score


# ------------------------------------------------------------------------------
# 3. TIE-AWARE METRICS & LOGISTIC REGRESSION PROBE
# ------------------------------------------------------------------------------

def compute_roc_pr_auc(y_true, y_scores):
    y_true = np.asarray(y_true, dtype=np.int32)
    y_scores = np.asarray(y_scores, dtype=np.float64)
    pos_mask = (y_true == 1)
    n_pos = int(pos_mask.sum())
    n_neg = int(len(y_true) - n_pos)

    if n_pos == 0 or n_neg == 0:
        return 0.5, float(n_pos / max(1, len(y_true)))

    if SKLEARN_AVAILABLE:
        try:
            roc_auc = float(roc_auc_score(y_true, y_scores))
            pr_auc = float(average_precision_score(y_true, y_scores))
            return roc_auc, pr_auc
        except Exception:
            pass

    temp = np.argsort(y_scores)
    sorted_scores = y_scores[temp]
    unique_scores, inverse_indices, counts = np.unique(sorted_scores, return_inverse=True, return_counts=True)
    tie_sums = np.bincount(inverse_indices, weights=np.arange(1, len(y_scores) + 1))
    tie_avgs = tie_sums / counts
    ranks = tie_avgs[np.unique(y_scores, return_inverse=True)[1]]

    pos_ranks = ranks[pos_mask]
    u = np.sum(pos_ranks) - (n_pos * (n_pos + 1)) / 2.0
    roc_auc = float(u / (n_pos * n_neg))

    desc_idx = np.argsort(-y_scores)
    y_sorted = y_true[desc_idx]
    scores_sorted = y_scores[desc_idx]

    threshold_mask = np.empty(len(scores_sorted), dtype=bool)
    threshold_mask[:-1] = scores_sorted[:-1] != scores_sorted[1:]
    threshold_mask[-1] = True

    tps = np.cumsum(y_sorted)[threshold_mask]
    fps = np.cumsum(1 - y_sorted)[threshold_mask]
    recalls = tps / n_pos
    precisions = tps / (tps + fps)

    recalls = np.concatenate(([0.0], recalls))
    precisions = np.concatenate(([precisions[0] if len(precisions) > 0 else 1.0], precisions))
    pr_auc = float(np.sum((recalls[1:] - recalls[:-1]) * precisions[1:]))

    return roc_auc, pr_auc


def fit_logistic_regression(X_train, y_train, l2_reg=1.0, max_iter=250):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    N, D = X_train.shape
    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(1)

    pos_count = max(1.0, float(y_train.sum()))
    neg_count = max(1.0, float((1.0 - y_train).sum()))
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=device)

    w = torch.zeros((D, 1), dtype=torch.float32, device=device, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([w, b], lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        logits = torch.matmul(X_t, w) + b
        bce = F.binary_cross_entropy_with_logits(logits, y_t, pos_weight=pos_weight)
        reg = 0.5 * l2_reg * torch.sum(w ** 2)
        loss = bce + reg
        loss.backward()
        return loss

    optimizer.step(closure)
    return w.detach().cpu().numpy().flatten(), float(b.detach().cpu().item())


# ------------------------------------------------------------------------------
# 4. OPERATIONAL TOP-100 DETECTOR EVALUATOR
# ------------------------------------------------------------------------------

def evaluate_operational_top100(logits, boxes, gt_boxes_px, gt_cats, label_to_cat_id, img_w, img_h):
    prob = logits.sigmoid()
    num_classes = prob.shape[-1]
    topk_scores, topk_idx = torch.topk(prob.flatten(), min(100, prob.numel()))
    q_idx = torch.div(topk_idx, num_classes, rounding_mode="floor")
    c_idx = topk_idx % num_classes
    boxes_ranked = boxes[q_idx]

    pred_cat_ids = np.array([label_to_cat_id[int(lbl.item())] for lbl in c_idx])
    pred_boxes_px = []
    for b in boxes_ranked:
        cx, cy, w, h = b.tolist()
        pred_boxes_px.append([(cx - w / 2.0) * img_w, (cy - h / 2.0) * img_h, w * img_w, h * img_h])
    pred_scores = topk_scores.cpu().numpy()

    n_gt = len(gt_cats)
    gt_matched = [False] * n_gt
    tp = 0
    fp = 0

    for p_idx in range(len(pred_scores)):
        if pred_scores[p_idx] >= SCORE_THRESHOLD:
            p_box = pred_boxes_px[p_idx]
            p_cat = pred_cat_ids[p_idx]
            best_iou = -1.0
            best_g = -1
            for g_idx in range(n_gt):
                if not gt_matched[g_idx] and gt_cats[g_idx] == p_cat:
                    iou_g = box_iou(gt_boxes_px[g_idx], p_box)
                    if iou_g >= EVAL_IOU_THRESHOLD and iou_g > best_iou:
                        best_iou = iou_g
                        best_g = g_idx
            if best_g >= 0:
                gt_matched[best_g] = True
                tp += 1
            else:
                fp += 1

    fn = n_gt - sum(gt_matched)
    return tp, fp, fn, gt_matched


# ------------------------------------------------------------------------------
# 5. MAIN DIAGNOSTIC ENGINE (P-1743 THROUGH P-1757)
# ------------------------------------------------------------------------------

def main():
    driver_sha256 = compute_sha256(Path(__file__))
    print("=" * 115)
    print("CANDIDATE 3V PRE-FLIGHT: DESIGN-ONLY QUERY ADMISSION DIAGNOSTIC & SIMULATION (v7)")
    print(f"Driver Script SHA-256: {driver_sha256}")
    print("=" * 115)

    for p_path in [CANDIDATES_JSON, TRAIN_JSON, CACHE_FILE, DEV_POOL_JSON, SPLIT_MANIFEST_JSON, TOKEN_CACHE_PTH, EPOCH8_MODEL_PTH, M0_CKPT_PATH]:
        assert p_path.exists(), f"[FATAL] Missing {p_path}"

    assert compute_sha256(CANDIDATES_JSON) == EXPECTED_CANDIDATE_SHA256
    assert compute_sha256(TRAIN_JSON) == EXPECTED_TRAIN_JSON_SHA256
    assert compute_sha256(CACHE_FILE) == EXPECTED_CACHE_SHA256
    assert compute_sha256(DEV_POOL_JSON) == EXPECTED_DEV_POOL_SHA256
    assert compute_sha256(SPLIT_MANIFEST_JSON) == EXPECTED_SPLIT_SHA256
    assert compute_sha256(TOKEN_CACHE_PTH) == EXPECTED_TOKEN_CACHE_SHA256
    assert compute_sha256(EPOCH8_MODEL_PTH) == EXPECTED_EPOCH8_ADAPTER_SHA256
    assert compute_sha256(M0_CKPT_PATH) == EXPECTED_M0_HASH

    print("[+] All Upstream Manifests, Token Cache, and Epoch-8 Adapter Certified by Immutable SHA-256.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Target: {device} ({torch.cuda.get_device_name(0)})")

    dataset = SeaClipsDetection(ZIP_PATH, TRAIN_JSON, is_train=False, random_flip=False)
    coco = dataset.coco
    cats = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    ann_lookup = {a["id"]: a for a in coco.loadAnns(coco.getAnnIds())}
    id_to_idx = {iid: idx for idx, iid in enumerate(sorted(coco.getImgIds()))}

    if hasattr(dataset, "label_to_cat_id"):
        label_to_cat_id = dataset.label_to_cat_id
    else:
        cat_ids = sorted(coco.getCatIds())
        label_to_cat_id = {i: cat_id for i, cat_id in enumerate(cat_ids)}

    with open(SPLIT_MANIFEST_JSON, "r", encoding="utf-8") as f:
        split_meta = json.load(f)["_meta"]
    design_vids = sorted(split_meta["train_videos"])
    print(f"[+] Loaded Clean Design Split: 10 Videos {design_vids} ({split_meta['n_train_targets']} MI Targets)")
    print(f"    (Calibration Videos 33/46 & Protected Test Videos 26/40 STRICTLY EXCLUDED)")

    apply_candidate2m_source_hooks()

    args = get_authoritative_r0_args()
    if hasattr(args, "conditioning_mode"):
        args.conditioning_mode = "none"

    model, criterion, _ = build(args)
    ckpt = torch.load(M0_CKPT_PATH, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    hist_token_cache = torch.load(TOKEN_CACHE_PTH, map_location="cpu")
    epoch8_ckpt = torch.load(EPOCH8_MODEL_PTH, map_location="cpu")
    adapter = HistoricalObjectTokenMemory(d_model=256, num_classes=7, max_age=6).to(device)
    adapter.load_state_dict(epoch8_ckpt["adapter_state"])
    adapter.gamma.data.fill_(10.0)
    adapter.eval()
    print("[+] Loaded Frozen Candidate-2M Epoch-8 Adapter (aea024e2...) with Open Gate (gamma = +10.0).")

    with open(CANDIDATES_JSON, "r", encoding="utf-8") as f:
        cand_windows = json.load(f)["windows"]

    design_windows = [
        w for w in cand_windows
        if w["video_id"] in design_vids and w.get("n_mem_incr_s20", w.get("n_memincr_s20", 0)) >= 1
    ]

    windows_by_iid = defaultdict(list)
    for w in design_windows:
        windows_by_iid[w["image_ids"][-1]].append(w)

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

    print(f"\n[*] Extracted {len(windows_by_iid)} Unique Target Frames across 10 Design Videos.")
    print("    Executing Greedy Forward Admission + P-1757 Iterative Backward Pruning to Convergence...")

    all_extracted_queries = []
    video_query_counts = defaultdict(lambda: {"op_util": 0, "rescue_pot": 0, "total": 0, "active": 0})
    baseline_totals = {"tp": 0, "fp": 0, "fn": 0}
    simulation_totals = {"tp": 0, "fp": 0, "fn": 0}

    total_gt_objects = 0
    total_m0_misses = 0

    with torch.no_grad():
        for count, (curr_iid, w_list) in enumerate(windows_by_iid.items(), 1):
            primary_w = w_list[0]

            for w in w_list[1:]:
                assert w["image_ids"] == primary_w["image_ids"], f"[FATAL] History image_ids mismatch across same-current-frame windows in video {primary_w['video_id']}!"
                assert w["frame_ids"] == primary_w["frame_ids"], f"[FATAL] History frame_ids mismatch across same-current-frame windows in video {primary_w['video_id']}!"

            vid = primary_w["video_id"]
            s_c = collate_fn([dataset[id_to_idx[curr_iid]]])
            samples = s_c[0].to(device)
            tgt_dict = s_c[1][0]
            img_h, img_w = tgt_dict["orig_size"][0].item(), tgt_dict["orig_size"][1].item()

            gt_boxes_norm = tgt_dict["boxes"].cpu()
            gt_labels_raw = tgt_dict["labels"].cpu()
            n_gt = len(gt_labels_raw)
            total_gt_objects += n_gt

            gt_boxes_px = []
            gt_cats = []
            for g_idx in range(n_gt):
                gcx, gcy, gw, gh = gt_boxes_norm[g_idx].tolist()
                gx1 = (gcx - gw / 2.0) * img_w
                gy1 = (gcy - gh / 2.0) * img_h
                gt_boxes_px.append([gx1, gy1, gw * img_w, gh * img_h])
                gt_cats.append(label_to_cat_id[gt_labels_raw[g_idx].item()])

            # Step 1: Baseline D Operational Evaluation (300 Canonical Queries)
            out_d = model(samples, temporal_enabled=False)
            tpd, fpd, fnd, d_matched_gt = evaluate_operational_top100(
                out_d["pred_logits"][0], out_d["pred_boxes"][0], gt_boxes_px, gt_cats, label_to_cat_id, img_w, img_h
            )
            baseline_totals["tp"] += tpd
            baseline_totals["fp"] += fpd
            baseline_totals["fn"] += fnd

            missed_gt_indices = set([g_idx for g_idx in range(n_gt) if not d_matched_gt[g_idx]])
            total_m0_misses += len(missed_gt_indices)

            # Pure baseline out_d canonical evidence for canonical_support (P-1750)
            d_boxes_norm = out_d["pred_boxes"][0]
            base_d_boxes_px = []
            for b in d_boxes_norm:
                cx, cy, w, h = b.tolist()
                base_d_boxes_px.append([(cx - w / 2.0) * img_w, (cy - h / 2.0) * img_h, w * img_w, h * img_h])
            base_d_logits = out_d["pred_logits"][0]

            # Step 2: Augmented Candidate 2M Forward
            mb_corr = get_memory_bank_for_window(primary_w, hist_token_cache)
            if not mb_corr:
                simulation_totals["tp"] += tpd
                simulation_totals["fp"] += fpd
                simulation_totals["fn"] += fnd
                continue

            out_aug, hs_last = model(samples, temporal_adapter=adapter, memory_bank=[mb_corr], temporal_enabled=True, return_decoder_tokens=True)
            K_mem = len(mb_corr)

            temp_logits = out_aug["pred_logits"][0, 300:]
            temp_boxes_norm = out_aug["pred_boxes"][0, 300:]
            temp_hs = hs_last[0, 300:]

            # Step 3: P-1754 Leakage-Free Feature Extraction
            frame_queries = []
            for q_t in range(K_mem):
                tok = mb_corr[q_t]
                h_i_t = temp_hs[q_t]
                b_i_t = temp_boxes_norm[q_t]
                l_i_t = temp_logits[q_t]

                feat_vec, box_px, pred_class, max_s = extract_query_evidence_features(
                    tok, h_i_t, b_i_t, l_i_t, base_d_boxes_px, base_d_logits, img_w, img_h
                )
                frame_queries.append({
                    "query_id": q_t,
                    "features": feat_vec,
                    "box_px": box_px,
                    "pred_class": pred_class,
                    "max_score": max_s,
                    "is_active": (max_s >= SCORE_THRESHOLD)
                })

            # Secondary Label: Geometric Rescue Potential (Rescues an M0 miss via any class entry)
            prob_temp = temp_logits.sigmoid().cpu()
            flat_pairs = []
            for q_t in range(K_mem):
                for c_t in range(7):
                    s_val = float(prob_temp[q_t, c_t].item())
                    flat_pairs.append((s_val, q_t, c_t))

            flat_pairs.sort(key=lambda x: x[0], reverse=True)
            claimed_miss_gt = set()
            query_rescue_potential = {q_t: 0 for q_t in range(K_mem)}
            query_diag_util_class = {q_t: frame_queries[q_t]["pred_class"] for q_t in range(K_mem)}

            for s_val, q_t, c_t in flat_pairs:
                if s_val < SCORE_THRESHOLD:
                    break
                p_box = frame_queries[q_t]["box_px"]
                p_cat = label_to_cat_id[c_t]

                matched_gt_idx = -1
                best_iou = -1.0
                for g_idx in range(n_gt):
                    if gt_cats[g_idx] == p_cat:
                        iou_g = box_iou(gt_boxes_px[g_idx], p_box)
                        if iou_g >= EVAL_IOU_THRESHOLD and iou_g > best_iou:
                            best_iou = iou_g
                            matched_gt_idx = g_idx

                if matched_gt_idx >= 0:
                    if (matched_gt_idx in missed_gt_indices) and (matched_gt_idx not in claimed_miss_gt):
                        if query_rescue_potential[q_t] == 0:
                            claimed_miss_gt.add(matched_gt_idx)
                            query_rescue_potential[q_t] = 1
                            query_diag_util_class[q_t] = c_t

            # Step 4: P-1755 Greedy Forward Admission
            sorted_q_indices = sorted(range(K_mem), key=lambda q: frame_queries[q]["max_score"], reverse=True)
            admitted_queries_list = []
            curr_joint_logits = out_d["pred_logits"][0].clone()
            curr_joint_boxes = out_d["pred_boxes"][0].clone()
            curr_tp = tpd
            curr_fp = fpd

            for q_t in sorted_q_indices:
                cand_logits = torch.cat([curr_joint_logits, temp_logits[q_t:q_t+1]], dim=0)
                cand_boxes = torch.cat([curr_joint_boxes, temp_boxes_norm[q_t:q_t+1]], dim=0)

                tp_cand, fp_cand, fn_cand, _ = evaluate_operational_top100(
                    cand_logits, cand_boxes, gt_boxes_px, gt_cats, label_to_cat_id, img_w, img_h
                )

                if (tp_cand > curr_tp) and (fp_cand <= curr_fp):
                    admitted_queries_list.append(q_t)
                    curr_joint_logits = cand_logits
                    curr_joint_boxes = cand_boxes
                    curr_tp = tp_cand
                    curr_fp = fp_cand

            # P-1757: Iterative Backward Redundancy Pruning to Fixed-Point Convergence
            final_admitted_set = set(admitted_queries_list)
            while True:
                pruned_in_pass = False
                current_ordered_admitted = [q for q in admitted_queries_list if q in final_admitted_set]

                for q_t in reversed(current_ordered_admitted):
                    if len(final_admitted_set) <= 1 and curr_tp > tpd:
                        break

                    test_set = [q for q in current_ordered_admitted if q != q_t and q in final_admitted_set]
                    if test_set:
                        rem_temp_logits = torch.stack([temp_logits[q] for q in test_set], dim=0)
                        rem_temp_boxes = torch.stack([temp_boxes_norm[q] for q in test_set], dim=0)
                        cand_logits_without = torch.cat([out_d["pred_logits"][0], rem_temp_logits], dim=0)
                        cand_boxes_without = torch.cat([out_d["pred_boxes"][0], rem_temp_boxes], dim=0)
                    else:
                        cand_logits_without = out_d["pred_logits"][0]
                        cand_boxes_without = out_d["pred_boxes"][0]

                    tp_without, fp_without, _, _ = evaluate_operational_top100(
                        cand_logits_without, cand_boxes_without, gt_boxes_px, gt_cats, label_to_cat_id, img_w, img_h
                    )

                    # If removing q_t preserves full TP and does not increase FP, prune it permanently
                    if (tp_without == curr_tp) and (fp_without <= curr_fp):
                        final_admitted_set.remove(q_t)
                        curr_fp = fp_without
                        pruned_in_pass = True

                if not pruned_in_pass:
                    break

            query_operational_utility = {q_t: (1 if q_t in final_admitted_set else 0) for q_t in range(K_mem)}

            # Operational Admission Simulation evaluates exact performance of final_admitted_set
            sim_logits = torch.cat([out_d["pred_logits"][0], temp_logits], dim=0)
            sim_boxes = torch.cat([out_d["pred_boxes"][0], temp_boxes_norm], dim=0)

            for q_t in range(K_mem):
                if q_t not in final_admitted_set:
                    sim_logits[300 + q_t, :] = -1e9

            tpo, fpo, fno, _ = evaluate_operational_top100(
                sim_logits, sim_boxes, gt_boxes_px, gt_cats, label_to_cat_id, img_w, img_h
            )
            simulation_totals["tp"] += tpo
            simulation_totals["fp"] += fpo
            simulation_totals["fn"] += fno

            for q_t in range(K_mem):
                q_info = frame_queries[q_t]
                lbl_op = query_operational_utility[q_t]
                lbl_resc = query_rescue_potential[q_t]
                diag_cls = query_diag_util_class[q_t]

                video_query_counts[vid]["total"] += 1
                if q_info["is_active"]:
                    video_query_counts[vid]["active"] += 1

                if lbl_op == 1:
                    video_query_counts[vid]["op_util"] += 1
                if lbl_resc == 1:
                    video_query_counts[vid]["rescue_pot"] += 1

                all_extracted_queries.append({
                    "video_id": vid,
                    "image_id": curr_iid,
                    "query_id": q_t,
                    "label_op_util": lbl_op, # Primary label: fixed-point non-redundant operational utility
                    "label_rescue_pot": lbl_resc, # Secondary label: geometric miss rescue potential
                    "diagnostic_utility_class": diag_cls,
                    "pred_class": q_info["pred_class"],
                    "is_active": q_info["is_active"],
                    "max_score": q_info["max_score"],
                    "features": q_info["features"]
                })

            if count % 50 == 0 or count == len(windows_by_iid):
                print(f"    Processed {count}/{len(windows_by_iid)} frames...")

    # --------------------------------------------------------------------------
    # 6. OPERATIONAL ADMISSION SIMULATION HEADROOM REPORT
    # --------------------------------------------------------------------------
    prec_d = baseline_totals["tp"] / max(1, baseline_totals["tp"] + baseline_totals["fp"])
    rec_d = baseline_totals["tp"] / max(1, baseline_totals["tp"] + baseline_totals["fn"])
    f1_d = 2 * prec_d * rec_d / max(1e-7, prec_d + rec_d)

    prec_sim = simulation_totals["tp"] / max(1, simulation_totals["tp"] + simulation_totals["fp"])
    rec_sim = simulation_totals["tp"] / max(1, simulation_totals["tp"] + simulation_totals["fn"])
    f1_sim = 2 * prec_sim * rec_sim / max(1e-7, prec_sim + rec_sim)

    total_queries = len(all_extracted_queries)
    active_queries = [q for q in all_extracted_queries if q["is_active"]]
    n_active = len(active_queries)

    total_op_util = sum(1 for q in all_extracted_queries if q["label_op_util"] == 1)
    total_rescue_pot = sum(1 for q in all_extracted_queries if q["label_rescue_pot"] == 1)

    active_op_util = sum(1 for q in active_queries if q["label_op_util"] == 1)
    active_rescue_pot = sum(1 for q in active_queries if q["label_rescue_pot"] == 1)

    print("\n" + "=" * 115)
    print("PHASE B: FIXED-POINT CONVERGED ADMISSION SIMULATION (P-1751, P-1755, P-1757)")
    print("=" * 115)
    print(f"[*] Total Extracted Queries: {total_queries:,} | Threshold-Active (s >= 0.20): {n_active:,} ({n_active/total_queries*100:4.1f}%)")
    print(f"    - Minimal Operational Utility (Primary Label, Delta TP > 0 & Delta FP <= 0): {total_op_util:5d} ({total_op_util/total_queries*100:5.2f}% of all, {active_op_util/n_active*100:5.2f}% of active)")
    print(f"    - Geometric Rescue Potential  (Secondary Label, rescues M0 miss):             {total_rescue_pot:5d} ({total_rescue_pot/total_queries*100:5.2f}% of all, {active_rescue_pot/n_active*100:5.2f}% of active)")
    print(f"\n[*] Operational Detector Comparison ({len(windows_by_iid)} Frames, Exact Top-100 Rerun):")
    print(f"    - Baseline M0 (D):       TP = {baseline_totals['tp']:4d}, FP = {baseline_totals['fp']:4d}, FN = {baseline_totals['fn']:4d} | Prec = {prec_d:.4f}, Rec = {rec_d:.4f}, F1 = {f1_d:.4f}")
    print(f"    - Admission Simulation:  TP = {simulation_totals['tp']:4d}, FP = {simulation_totals['fp']:4d}, FN = {simulation_totals['fn']:4d} | Prec = {prec_sim:.4f}, Rec = {rec_sim:.4f}, F1 = {f1_sim:.4f}")
    print(f"    - Simulated Headroom:    Delta F1 = {f1_sim - f1_d:+.4f} ({f1_d:.4f} -> {f1_sim:.4f}) | Delta Recall = {rec_sim - rec_d:+.4f}")

    # --------------------------------------------------------------------------
    # 7. PRIMARY LOVO PROBE AUDIT (THRESHOLD-ACTIVE QUERIES s >= 0.20, LABEL = OP_UTIL)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("PHASE C: PRIMARY LOVO PROBE AUDIT (THRESHOLD-ACTIVE QUERIES, LABEL = IRREDUCIBLE OP UTILITY)")
    print("=" * 115)

    X_act = np.array([q["features"] for q in active_queries])
    y_act = np.array([q["label_op_util"] for q in active_queries])
    v_act = np.array([q["video_id"] for q in active_queries])

    pos_mask = (y_act == 1)
    neg_mask = (y_act == 0)

    print(f"{'Feature Name':<24} | {'Mean (Op Util, y=1)':<20} | {'Mean (Non-Util, y=0)':<20} | {'Delta [Pos - Neg]':<18} | {'Active Single AUC'}")
    print("-" * 112)

    single_feat_aucs_active = {}
    for f_idx, fname in enumerate(FEATURE_NAMES):
        m_pos = float(np.mean(X_act[pos_mask, f_idx])) if pos_mask.any() else 0.0
        m_neg = float(np.mean(X_act[neg_mask, f_idx])) if neg_mask.any() else 0.0
        d_val = m_pos - m_neg
        auc_f, _ = compute_roc_pr_auc(y_act, X_act[:, f_idx])
        eff_auc = max(auc_f, 1.0 - auc_f)
        single_feat_aucs_active[fname] = eff_auc
        print(f"{fname:<24} | {m_pos:+18.4f}   | {m_neg:+18.4f}   | {d_val:+18.4f}   | {eff_auc:10.4f} ({'Admit' if auc_f >= 0.5 else 'Suppress'})")

    lovo_probs_act = np.full(len(y_act), np.nan)
    per_video_metrics = {}
    evaluable_test_indices = []

    for vid in design_vids:
        test_mask = (v_act == vid)
        train_mask = ~test_mask

        X_tr, y_tr = X_act[train_mask], y_act[train_mask]
        X_te = X_act[test_mask]
        y_te = y_act[test_mask]
        n_v = len(y_te)
        n_pos_v = int(y_te.sum())

        if len(np.unique(y_tr)) < 2:
            print(f"    [!] Warning: Training fold for held-out video_{vid} is single-class. Marking fold non-evaluable.")
            per_video_metrics[vid] = {
                "n_active": n_v,
                "n_pos": n_pos_v,
                "pos_pct": float(n_pos_v / max(1, n_v) * 100.0),
                "roc_auc": None,
                "pr_auc": None,
                "status": "DEGENERATE_TRAIN_FOLD"
            }
            continue

        mean_tr = np.mean(X_tr, axis=0)
        std_tr = np.std(X_tr, axis=0)
        std_tr[std_tr < 1e-6] = 1.0

        X_tr_std = (X_tr - mean_tr) / std_tr
        X_te_std = (X_te - mean_tr) / std_tr

        w, b = fit_logistic_regression(X_tr_std, y_tr, l2_reg=1.0)
        test_logits = np.dot(X_te_std, w) + b
        p_te = 1.0 / (1.0 + np.exp(-np.clip(test_logits, -15.0, 15.0)))
        lovo_probs_act[test_mask] = p_te
        evaluable_test_indices.extend(np.where(test_mask)[0].tolist())

        if n_pos_v > 0 and n_pos_v < n_v:
            auc_v, pr_v = compute_roc_pr_auc(y_te, p_te)
        else:
            auc_v, pr_v = np.nan, np.nan

        per_video_metrics[vid] = {
            "n_active": n_v,
            "n_pos": n_pos_v,
            "pos_pct": float(n_pos_v / max(1, n_v) * 100.0),
            "roc_auc": float(auc_v) if not np.isnan(auc_v) else None,
            "pr_auc": float(pr_v) if not np.isnan(pr_v) else None,
            "status": "EVALUATED"
        }

    print("\n" + "=" * 115)
    print(f"{'Held-Out Video':<16} | {'Active Queries':<16} | {'Operational TPs':<18} | {'Prevalence':<12} | {'Test Fold ROC-AUC':<20} | {'Test Fold PR-AUC'}")
    print("-" * 110)
    evaluable_aucs = []
    evaluable_prs = []
    for vid in design_vids:
        vm = per_video_metrics[vid]
        auc_str = f"{vm['roc_auc']:.4f}" if vm['roc_auc'] is not None else "N/A (single-class)"
        pr_str = f"{vm['pr_auc']:.4f}" if vm['pr_auc'] is not None else "N/A"
        if vm['roc_auc'] is not None:
            evaluable_aucs.append(vm['roc_auc'])
            evaluable_prs.append(vm['pr_auc'])
        print(f"video_{vid:<10d} | {vm['n_active']:16d} | {vm['n_pos']:18d} | {vm['pos_pct']:10.2f}% | {auc_str:<20} | {pr_str}")

    macro_mean_auc = float(np.mean(evaluable_aucs)) if evaluable_aucs else 0.5
    macro_median_auc = float(np.median(evaluable_aucs)) if evaluable_aucs else 0.5
    macro_mean_pr = float(np.mean(evaluable_prs)) if evaluable_prs else 0.0
    macro_median_pr = float(np.median(evaluable_prs)) if evaluable_prs else 0.0

    eval_idx_arr = np.array(evaluable_test_indices)
    if len(eval_idx_arr) > 0 and len(np.unique(y_act[eval_idx_arr])) > 1:
        pooled_auc_act, pooled_pr_act = compute_roc_pr_auc(y_act[eval_idx_arr], lovo_probs_act[eval_idx_arr])
        base_prev_act = float(np.mean(y_act[eval_idx_arr]))
        n_eval_act = len(eval_idx_arr)
        n_eval_pos = int(y_act[eval_idx_arr].sum())
    else:
        pooled_auc_act, pooled_pr_act = 0.5, 0.0
        base_prev_act = 0.0
        n_eval_act = 0
        n_eval_pos = 0

    print("-" * 110)
    print(f"{'MACRO MEAN':<16} | {'-':<16} | {'-':<18} | {'-':<12} | {macro_mean_auc:10.4f}           | {macro_mean_pr:10.4f}")
    print(f"{'MACRO MEDIAN':<16} | {'-':<16} | {'-':<18} | {'-':<12} | {macro_median_auc:10.4f}           | {macro_median_pr:10.4f}")
    print(f"{'POOLED TOTAL':<16} | {n_eval_act:16d} | {n_eval_pos:18d} | {base_prev_act*100:10.2f}% | {pooled_auc_act:10.4f}           | {pooled_pr_act:10.4f} (Lift: {pooled_pr_act/max(1e-6, base_prev_act):.2f}x)")

    # --------------------------------------------------------------------------
    # 8. SECONDARY DIAGNOSTIC: ALL-QUERY AUDIT
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("PHASE D: SECONDARY AUDIT (ALL EXTRACTED QUERIES INCLUDING INACTIVE s < 0.20)")
    print("=" * 115)

    X_all = np.array([q["features"] for q in all_extracted_queries])
    y_all = np.array([q["label_op_util"] for q in all_extracted_queries])
    v_all = np.array([q["video_id"] for q in all_extracted_queries])

    lovo_probs_all = np.full(len(y_all), np.nan)
    evaluable_all_indices = []

    for vid in design_vids:
        test_mask = (v_all == vid)
        train_mask = ~test_mask
        X_tr, y_tr = X_all[train_mask], y_all[train_mask]
        X_te = X_all[test_mask]

        if len(np.unique(y_tr)) < 2:
            continue

        mean_tr = np.mean(X_tr, axis=0)
        std_tr = np.std(X_tr, axis=0)
        std_tr[std_tr < 1e-6] = 1.0

        w, b = fit_logistic_regression((X_tr - mean_tr) / std_tr, y_tr, l2_reg=1.0)
        test_logits = np.dot((X_te - mean_tr) / std_tr, w) + b
        lovo_probs_all[test_mask] = 1.0 / (1.0 + np.exp(-np.clip(test_logits, -15.0, 15.0)))
        evaluable_all_indices.extend(np.where(test_mask)[0].tolist())

    eval_all_arr = np.array(evaluable_all_indices)
    if len(eval_all_arr) > 0 and len(np.unique(y_all[eval_all_arr])) > 1:
        pooled_auc_all, pooled_pr_all = compute_roc_pr_auc(y_all[eval_all_arr], lovo_probs_all[eval_all_arr])
        base_prev_all = float(np.mean(y_all[eval_all_arr]))
    else:
        pooled_auc_all, pooled_pr_all = 0.5, 0.0
        base_prev_all = 0.0

    print(f"[*] All-Query LOVO Pooled Performance: ROC-AUC = {pooled_auc_all:.4f} | PR-AUC = {pooled_pr_all:.4f} (Base Prev = {base_prev_all:.4f}, Lift = {pooled_pr_all/max(1e-6, base_prev_all):.2f}x)")

    diagnostic_payload = {
        "_meta": {
            "artifact_name": "pol3v_query_admission_diagnostic_v7",
            "driver_sha256": driver_sha256,
            "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
            "token_cache_sha256": EXPECTED_TOKEN_CACHE_SHA256,
            "epoch8_adapter_sha256": EXPECTED_EPOCH8_ADAPTER_SHA256,
            "total_design_videos_audited": len(design_vids),
            "total_design_frames_audited": len(windows_by_iid),
            "total_temporal_queries_extracted": total_queries,
            "total_threshold_active_queries": n_active
        },
        "query_distribution": {
            "total_queries": total_queries,
            "total_active_queries": n_active,
            "operational_utility_count": total_op_util,
            "rescue_potential_count": total_rescue_pot,
            "active_operational_utility_count": active_op_util,
            "active_rescue_potential_count": active_rescue_pot
        },
        "incremental_query_admission_simulation": {
            "baseline_m0_tp": baseline_totals["tp"],
            "baseline_m0_fp": baseline_totals["fp"],
            "baseline_m0_fn": baseline_totals["fn"],
            "baseline_m0_f1": float(f1_d),
            "simulated_tp": simulation_totals["tp"],
            "simulated_fp": simulation_totals["fp"],
            "simulated_fn": simulation_totals["fn"],
            "simulated_f1": float(f1_sim),
            "delta_f1_headroom": float(f1_sim - f1_d)
        },
        "primary_active_lovo_probe": {
            "evaluated_queries_count": n_eval_act,
            "evaluated_positives_count": n_eval_pos,
            "pooled_roc_auc": float(pooled_auc_act),
            "pooled_pr_auc": float(pooled_pr_act),
            "macro_mean_roc_auc": macro_mean_auc,
            "macro_median_roc_auc": macro_median_auc,
            "macro_mean_pr_auc": macro_mean_pr,
            "macro_median_pr_auc": macro_median_pr,
            "per_video_metrics": per_video_metrics,
            "single_feature_aucs": single_feat_aucs_active
        },
        "secondary_all_query_lovo_probe": {
            "pooled_roc_auc": float(pooled_auc_all),
            "pooled_pr_auc": float(pooled_pr_all),
            "base_prevalence": float(base_prev_all)
        }
    }

    with open(DIAGNOSTIC_JSON, "w", encoding="utf-8") as f:
        json.dump(diagnostic_payload, f, indent=2)

    diag_hash = compute_sha256(DIAGNOSTIC_JSON)
    print("\n" + "=" * 115)
    print(f"[+] Candidate 3V Admission Diagnostic Manifest Frozen: {DIAGNOSTIC_JSON.resolve()}")
    print(f"    Manifest SHA-256:                                 {diag_hash}")
    print("=" * 115)


if __name__ == "__main__":
    main()