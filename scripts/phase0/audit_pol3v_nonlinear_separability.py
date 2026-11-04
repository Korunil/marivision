"""
D:/MARIVISION/H0_workbench/eval_tools/audit_pol3v_nonlinear_separability.py
Candidate 3V Pre-Flight: Label-Intersection Audit & Nonlinear Scalar Separability Benchmark (v2)

Strict Compliance Checklist:
  - D-3687 / D-3688 / D-3689 / D-3690: Design split ONLY; calibration and protected test remain untouched.
  - P-1760: Versioned/schema-validated dataset cache (pol3v_design_queries_dataset_v7.pth).
  - P-1761: Fair, consistent class balancing across all 4 model training folds:
            * Logistic Regression: class_weight="balanced"
            * Shallow MLP: deterministic training-fold minority oversampling
            * GBDT: inverse-frequency sample_weights
            * Random Forest: class_weight="balanced"
  - P-1762: Comparative benchmark against linear baseline and PR lift (no arbitrary binary thresholds).
  - P-1758: Evaluates exact set intersection between operational utility and geometric rescue potential.
  - P-1754: Zero GT leakage: 15 features derived from inference-available pred_class and baseline out_d.
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

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

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

QUERY_DATASET_PTH = OUT_DIR / "pol3v_design_queries_dataset_v7.pth"
NONLINEAR_RESULTS_JSON = OUT_DIR / "pol3v_nonlinear_separability_results.json"

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
# 1. CANDIDATE 2M MODULE ARCHITECTURE (EPOCH-8 CHECKPOINT)
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
# 2. FEATURE EXTRACTION & TOP-100 EVALUATION HELPERS
# ------------------------------------------------------------------------------

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
# 3. DATASET EXTRACTION ENGINE (P-1760 PROVENANCE VALIDATION)
# ------------------------------------------------------------------------------

def load_or_extract_query_dataset(driver_sha256):
    if QUERY_DATASET_PTH.exists():
        print(f"[+] Loading Cached Query Dataset: {QUERY_DATASET_PTH.resolve()}")
        cached_data = torch.load(QUERY_DATASET_PTH, map_location="cpu")

        # P-1760: Strict schema and count validation
        valid_schema = (
            isinstance(cached_data, dict)
            and cached_data.get("version") == "v7"
            and "extracted_queries" in cached_data
            and len(cached_data["extracted_queries"]) == 2045
            and sum(1 for q in cached_data["extracted_queries"] if q.get("is_active")) == 1915
            and sum(1 for q in cached_data["extracted_queries"] if q.get("label_op_util") == 1) == 92
            and sum(1 for q in cached_data["extracted_queries"] if q.get("label_rescue_pot") == 1) == 96
            and len(cached_data["extracted_queries"][0].get("features", [])) == 15
        )

        if valid_schema:
            print(f"[+] Cache Schema Validated (Version: v7, 2,045 total, 1,915 active, 92 op_util, 96 rescue_pot).")
            return cached_data["extracted_queries"]
        else:
            print("[!] Cached dataset failed schema or count assertions. Purging stale cache and re-extracting...")

    print("[*] Re-extracting 2,045 queries across 10 design videos under strict v7 invariants...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = SeaClipsDetection(ZIP_PATH, TRAIN_JSON, is_train=False, random_flip=False)
    coco = dataset.coco
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

    all_extracted_queries = []

    with torch.no_grad():
        for count, (curr_iid, w_list) in enumerate(windows_by_iid.items(), 1):
            primary_w = w_list[0]
            for w in w_list[1:]:
                assert w["image_ids"] == primary_w["image_ids"]
                assert w["frame_ids"] == primary_w["frame_ids"]

            vid = primary_w["video_id"]
            s_c = collate_fn([dataset[id_to_idx[curr_iid]]])
            samples = s_c[0].to(device)
            tgt_dict = s_c[1][0]
            img_h, img_w = tgt_dict["orig_size"][0].item(), tgt_dict["orig_size"][1].item()

            gt_boxes_norm = tgt_dict["boxes"].cpu()
            gt_labels_raw = tgt_dict["labels"].cpu()
            n_gt = len(gt_labels_raw)

            gt_boxes_px = []
            gt_cats = []
            for g_idx in range(n_gt):
                gcx, gcy, gw, gh = gt_boxes_norm[g_idx].tolist()
                gx1 = (gcx - gw / 2.0) * img_w
                gy1 = (gcy - gh / 2.0) * img_h
                gt_boxes_px.append([gx1, gy1, gw * img_w, gh * img_h])
                gt_cats.append(label_to_cat_id[gt_labels_raw[g_idx].item()])

            out_d = model(samples, temporal_enabled=False)
            tpd, fpd, fnd, d_matched_gt = evaluate_operational_top100(
                out_d["pred_logits"][0], out_d["pred_boxes"][0], gt_boxes_px, gt_cats, label_to_cat_id, img_w, img_h
            )
            missed_gt_indices = set([g_idx for g_idx in range(n_gt) if not d_matched_gt[g_idx]])

            d_boxes_norm = out_d["pred_boxes"][0]
            base_d_boxes_px = []
            for b in d_boxes_norm:
                cx, cy, w, h = b.tolist()
                base_d_boxes_px.append([(cx - w / 2.0) * img_w, (cy - h / 2.0) * img_h, w * img_w, h * img_h])
            base_d_logits = out_d["pred_logits"][0]

            mb_corr = get_memory_bank_for_window(primary_w, hist_token_cache)
            if not mb_corr:
                continue

            out_aug, hs_last = model(samples, temporal_adapter=adapter, memory_bank=[mb_corr], temporal_enabled=True, return_decoder_tokens=True)
            K_mem = len(mb_corr)

            temp_logits = out_aug["pred_logits"][0, 300:]
            temp_boxes_norm = out_aug["pred_boxes"][0, 300:]
            temp_hs = hs_last[0, 300:]

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

            prob_temp = temp_logits.sigmoid().cpu()
            flat_pairs = []
            for q_t in range(K_mem):
                for c_t in range(7):
                    s_val = float(prob_temp[q_t, c_t].item())
                    flat_pairs.append((s_val, q_t, c_t))

            flat_pairs.sort(key=lambda x: x[0], reverse=True)
            claimed_miss_gt = set()
            query_rescue_potential = {q_t: 0 for q_t in range(K_mem)}

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

            sorted_q_indices = sorted(range(K_mem), key=lambda q: frame_queries[q]["max_score"], reverse=True)
            admitted_queries_list = []
            curr_joint_logits = out_d["pred_logits"][0].clone()
            curr_joint_boxes = out_d["pred_boxes"][0].clone()
            curr_tp = tpd
            curr_fp = fpd

            for q_t in sorted_q_indices:
                cand_logits = torch.cat([curr_joint_logits, temp_logits[q_t:q_t+1]], dim=0)
                cand_boxes = torch.cat([curr_joint_boxes, temp_boxes_norm[q_t:q_t+1]], dim=0)
                tp_cand, fp_cand, _, _ = evaluate_operational_top100(
                    cand_logits, cand_boxes, gt_boxes_px, gt_cats, label_to_cat_id, img_w, img_h
                )
                if (tp_cand > curr_tp) and (fp_cand <= curr_fp):
                    admitted_queries_list.append(q_t)
                    curr_joint_logits = cand_logits
                    curr_joint_boxes = cand_boxes
                    curr_tp = tp_cand
                    curr_fp = fp_cand

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

                    if (tp_without == curr_tp) and (fp_without <= curr_fp):
                        final_admitted_set.remove(q_t)
                        curr_fp = fp_without
                        pruned_in_pass = True

                if not pruned_in_pass:
                    break

            for q_t in range(K_mem):
                q_info = frame_queries[q_t]
                lbl_op = 1 if q_t in final_admitted_set else 0
                lbl_resc = query_rescue_potential[q_t]

                all_extracted_queries.append({
                    "video_id": vid,
                    "image_id": curr_iid,
                    "query_id": q_t,
                    "label_op_util": lbl_op,
                    "label_rescue_pot": lbl_resc,
                    "pred_class": q_info["pred_class"],
                    "is_active": q_info["is_active"],
                    "max_score": q_info["max_score"],
                    "features": q_info["features"]
                })

            if count % 50 == 0 or count == len(windows_by_iid):
                print(f"    Processed {count}/{len(windows_by_iid)} frames...")

    assert len(all_extracted_queries) == 2045, f"Expected 2045 queries, got {len(all_extracted_queries)}"
    assert sum(1 for q in all_extracted_queries if q["is_active"]) == 1915, "Expected 1915 active queries"
    assert sum(1 for q in all_extracted_queries if q["label_op_util"] == 1) == 92, "Expected 92 op_util queries"
    assert sum(1 for q in all_extracted_queries if q["label_rescue_pot"] == 1) == 96, "Expected 96 rescue_pot queries"

    torch.save({
        "version": "v7",
        "driver_sha256": driver_sha256,
        "feature_names": FEATURE_NAMES,
        "extracted_queries": all_extracted_queries
    }, QUERY_DATASET_PTH)
    print(f"[+] Saved Certified v7 Query Dataset Cache: {QUERY_DATASET_PTH.resolve()}")
    return all_extracted_queries


# ------------------------------------------------------------------------------
# 4. MAIN BENCHMARK ENGINE (P-1758, P-1761, P-1762)
# ------------------------------------------------------------------------------

def main():
    driver_sha256 = compute_sha256(Path(__file__))
    print("=" * 115)
    print("CANDIDATE 3V PRE-FLIGHT: LABEL-INTERSECTION AUDIT & NONLINEAR SEPARABILITY BENCHMARK (v2)")
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

    all_queries = load_or_extract_query_dataset(driver_sha256)
    active_queries = [q for q in all_queries if q["is_active"]]

    with open(SPLIT_MANIFEST_JSON, "r", encoding="utf-8") as f:
        design_vids = sorted(json.load(f)["_meta"]["train_videos"])

    # --------------------------------------------------------------------------
    # P-1758: LABEL-INTERSECTION AUDIT
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("P-1758: LABEL-INTERSECTION AUDIT (OPERATIONAL UTILITY vs RESCUE POTENTIAL)")
    print("=" * 115)

    def compute_intersection_stats(query_list, set_name):
        n_total = len(query_list)
        n_both = sum(1 for q in query_list if q["label_op_util"] == 1 and q["label_rescue_pot"] == 1)
        n_op_only = sum(1 for q in query_list if q["label_op_util"] == 1 and q["label_rescue_pot"] == 0)
        n_resc_only = sum(1 for q in query_list if q["label_op_util"] == 0 and q["label_rescue_pot"] == 1)
        n_neither = sum(1 for q in query_list if q["label_op_util"] == 0 and q["label_rescue_pot"] == 0)

        n_op_total = sum(1 for q in query_list if q["label_op_util"] == 1)
        n_resc_total = sum(1 for q in query_list if q["label_rescue_pot"] == 1)
        conv_rate = (n_both / max(1, n_resc_total)) * 100.0

        print(f"[*] Label Intersection: {set_name} (N = {n_total:,})")
        print(f"    - Both (Operational Utility ∩ Rescue Potential): {n_both:5d} ({n_both/n_total*100:5.2f}%)")
        print(f"    - Operational Utility ONLY (op_util \\ rescue_pot):   {n_op_only:5d} ({n_op_only/n_total*100:5.2f}%)")
        print(f"    - Rescue Potential ONLY    (rescue_pot \\ op_util):   {n_resc_only:5d} ({n_resc_only/n_total*100:5.2f}%)")
        print(f"    - Neither (Non-Admissible Negatives):             {n_neither:5d} ({n_neither/n_total*100:5.2f}%)")
        print(f"    - Operational Conversion: {n_both} / {n_resc_total} geometric rescues survive pruning into operational TPs ({conv_rate:.2f}%)\n")

        return {
            "n_total": n_total,
            "both": n_both,
            "op_only": n_op_only,
            "resc_only": n_resc_only,
            "neither": n_neither,
            "op_total": n_op_total,
            "resc_total": n_resc_total,
            "conversion_rate_pct": float(conv_rate)
        }

    intersection_all = compute_intersection_stats(all_queries, "All Extracted Temporal Queries")
    intersection_active = compute_intersection_stats(active_queries, "Threshold-Active Temporal Queries (s >= 0.20)")

    # --------------------------------------------------------------------------
    # P-1759 & P-1761: NONLINEAR SCALAR SEPARABILITY AUDIT WITH BALANCED TRAINING
    # --------------------------------------------------------------------------
    print("=" * 115)
    print("P-1759 / P-1761: NONLINEAR SCALAR SEPARABILITY BENCHMARK (BALANCED LOVO TRAINING FOLDS)")
    print("=" * 115)

    X_act = np.array([q["features"] for q in active_queries])
    y_act = np.array([q["label_op_util"] for q in active_queries])
    v_act = np.array([q["video_id"] for q in active_queries])

    models_list = [
        "Logistic Regression (Linear)",
        "Shallow MLP (1x32, ReLU)",
        "Gradient Boosted Trees (GBDT)",
        "Random Forest (RF, depth=4)"
    ]

    benchmark_results = {}

    for m_name in models_list:
        print(f"\n[*] Evaluating Model: {m_name}...")
        lovo_probs = np.full(len(y_act), np.nan)
        fold_metrics = {}
        evaluable_indices = []

        for vid in design_vids:
            test_mask = (v_act == vid)
            train_mask = ~test_mask

            X_tr, y_tr = X_act[train_mask], y_act[train_mask]
            X_te = X_act[test_mask]
            y_te = y_act[test_mask]
            n_v = len(y_te)
            n_pos_v = int(y_te.sum())

            if len(np.unique(y_tr)) < 2:
                fold_metrics[vid] = {"n_active": n_v, "n_pos": n_pos_v, "roc_auc": None, "pr_auc": None, "status": "DEGENERATE_TRAIN_FOLD"}
                continue

            # Standardize based strictly on training fold statistics
            mean_tr = np.mean(X_tr, axis=0)
            std_tr = np.std(X_tr, axis=0)
            std_tr[std_tr < 1e-6] = 1.0

            X_tr_std = (X_tr - mean_tr) / std_tr
            X_te_std = (X_te - mean_tr) / std_tr

            # P-1761: Balanced Training Optimization per model family
            if m_name == "Logistic Regression (Linear)":
                clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=350, random_state=42)
                clf.fit(X_tr_std, y_tr)
                p_te = clf.predict_proba(X_te_std)[:, 1]

            elif m_name == "Shallow MLP (1x32, ReLU)":
                # Deterministic minority oversampling strictly on training fold
                pos_indices = np.where(y_tr == 1)[0]
                neg_indices = np.where(y_tr == 0)[0]
                rng_bal = np.random.default_rng(42 + vid)
                oversampled_pos = rng_bal.choice(pos_indices, size=len(neg_indices), replace=True)
                bal_indices = np.concatenate([neg_indices, oversampled_pos])
                rng_bal.shuffle(bal_indices)
                X_tr_bal = X_tr_std[bal_indices]
                y_tr_bal = y_tr[bal_indices]

                clf = MLPClassifier(hidden_layer_sizes=(32,), activation="relu", alpha=0.05, max_iter=350, random_state=42)
                clf.fit(X_tr_bal, y_tr_bal)
                p_te = clf.predict_proba(X_te_std)[:, 1]

            elif m_name == "Gradient Boosted Trees (GBDT)":
                # Inverse-frequency sample weights on training fold
                pos_wt = len(y_tr) / (2.0 * max(1, np.sum(y_tr == 1)))
                neg_wt = len(y_tr) / (2.0 * max(1, np.sum(y_tr == 0)))
                sample_weights = np.where(y_tr == 1, pos_wt, neg_wt)

                clf = GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42)
                clf.fit(X_tr_std, y_tr, sample_weight=sample_weights)
                p_te = clf.predict_proba(X_te_std)[:, 1]

            elif m_name == "Random Forest (RF, depth=4)":
                clf = RandomForestClassifier(n_estimators=100, max_depth=4, class_weight="balanced", random_state=42)
                clf.fit(X_tr_std, y_tr)
                p_te = clf.predict_proba(X_te_std)[:, 1]

            lovo_probs[test_mask] = p_te
            evaluable_indices.extend(np.where(test_mask)[0].tolist())

            if n_pos_v > 0 and n_pos_v < n_v:
                auc_v = float(roc_auc_score(y_te, p_te))
                pr_v = float(average_precision_score(y_te, p_te))
            else:
                auc_v, pr_v = np.nan, np.nan

            fold_metrics[vid] = {
                "n_active": n_v,
                "n_pos": n_pos_v,
                "roc_auc": float(auc_v) if not np.isnan(auc_v) else None,
                "pr_auc": float(pr_v) if not np.isnan(pr_v) else None,
                "status": "EVALUATED"
            }

        eval_aucs = [fm["roc_auc"] for fm in fold_metrics.values() if fm["roc_auc"] is not None]
        eval_prs = [fm["pr_auc"] for fm in fold_metrics.values() if fm["pr_auc"] is not None]

        macro_mean_auc = float(np.mean(eval_aucs)) if eval_aucs else 0.5
        macro_med_auc = float(np.median(eval_aucs)) if eval_aucs else 0.5
        macro_mean_pr = float(np.mean(eval_prs)) if eval_prs else 0.0
        macro_med_pr = float(np.median(eval_prs)) if eval_prs else 0.0

        eval_idx_arr = np.array(evaluable_indices)
        if len(eval_idx_arr) > 0 and len(np.unique(y_act[eval_idx_arr])) > 1:
            pooled_auc = float(roc_auc_score(y_act[eval_idx_arr], lovo_probs[eval_idx_arr]))
            pooled_pr = float(average_precision_score(y_act[eval_idx_arr], lovo_probs[eval_idx_arr]))
            base_prev = float(np.mean(y_act[eval_idx_arr]))
        else:
            pooled_auc, pooled_pr, base_prev = 0.5, 0.0, 0.0

        benchmark_results[m_name] = {
            "pooled_roc_auc": pooled_auc,
            "pooled_pr_auc": pooled_pr,
            "macro_mean_roc_auc": macro_mean_auc,
            "macro_median_roc_auc": macro_med_auc,
            "macro_mean_pr_auc": macro_mean_pr,
            "macro_median_pr_auc": macro_med_pr,
            "base_prevalence": base_prev,
            "pr_auc_lift": float(pooled_pr / max(1e-6, base_prev)),
            "fold_metrics": fold_metrics
        }

    # P-1762: Comparative Summary Table (No arbitrary pass/fail cutoffs)
    print("\n" + "=" * 115)
    print("COMPARATIVE NONLINEAR BENCHMARK SUMMARY (10 DESIGN LOVO FOLDS, N = 1,915 ACTIVE QUERIES)")
    print("=" * 115)
    print(f"{'Model Architecture':<32} | {'Pooled AUC':<12} | {'Macro Mean AUC':<16} | {'Macro Med AUC':<16} | {'Pooled PR-AUC':<14} | {'PR Lift'}")
    print("-" * 105)
    for m_name in models_list:
        res = benchmark_results[m_name]
        print(f"{m_name:<32} | {res['pooled_roc_auc']:10.4f}   | {res['macro_mean_roc_auc']:14.4f}   | {res['macro_median_roc_auc']:14.4f}   | {res['pooled_pr_auc']:12.4f}   | {res['pr_auc_lift']:6.2f}x")

    # Detailed Per-Video Folds across Models
    print("\n" + "=" * 115)
    print("PER-VIDEO HELD-OUT TEST ROC-AUC COMPARISON ACROSS MODEL FAMILIES")
    print("=" * 115)
    print(f"{'Held-Out Video':<16} | {'Active (Pos)':<14} | {'Logistic Reg':<16} | {'Shallow MLP':<16} | {'GBDT (depth=3)':<16} | {'Random Forest'}")
    print("-" * 96)
    for vid in design_vids:
        n_act_v = benchmark_results["Logistic Regression (Linear)"]["fold_metrics"][vid]["n_active"]
        n_pos_v = benchmark_results["Logistic Regression (Linear)"]["fold_metrics"][vid]["n_pos"]
        row_str = f"video_{vid:<10d} | {n_act_v:4d} ({n_pos_v:2d})   "
        for m_name in models_list:
            auc_v = benchmark_results[m_name]["fold_metrics"][vid]["roc_auc"]
            row_str += f" | {f'{auc_v:.4f}' if auc_v is not None else 'N/A':<14}"
        print(row_str)

    results_payload = {
        "_meta": {
            "artifact_name": "pol3v_nonlinear_separability_results_v2",
            "driver_sha256": driver_sha256,
            "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
            "token_cache_sha256": EXPECTED_TOKEN_CACHE_SHA256,
            "epoch8_adapter_sha256": EXPECTED_EPOCH8_ADAPTER_SHA256,
            "design_videos": design_vids,
            "total_active_queries": len(active_queries)
        },
        "p1758_label_intersection_audit": {
            "all_queries": intersection_all,
            "active_queries": intersection_active
        },
        "p1759_nonlinear_separability_benchmark": benchmark_results
    }

    with open(NONLINEAR_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    res_hash = compute_sha256(NONLINEAR_RESULTS_JSON)
    print("\n" + "=" * 115)
    print(f"[+] Candidate 3V Nonlinear Separability Manifest Frozen: {NONLINEAR_RESULTS_JSON.resolve()}")
    print(f"    Manifest SHA-256:                                    {res_hash}")
    print("=" * 115)


if __name__ == "__main__":
    main()