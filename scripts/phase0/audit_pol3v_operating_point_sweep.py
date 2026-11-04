"""
D:/MARIVISION/H0_workbench/eval_tools/audit_pol3v_operating_point_sweep.py
Candidate 3V Pre-Flight: MLP Seed Stability Audit, Operational Gate Sweep & Verifier Freeze (v3)

Strict Compliance Checklist:
  - D-3692..D-3701: Restricted to 10 design videos; calibration (33, 46) & test (26, 40) UNTOUCHED.
  - P-1769: Gated artifact creation: freezes pol3v_candidate3v_seed41_verifier.pth iff delta_f1 > 0;
            otherwise logs REJECTED_DESIGN_GATE without producing a deployment artifact.
  - P-1770: Encodes explicit temporal prefilter contract (temporal_prefilter_score=0.20) in verifier artifact.
  - P-1766: Versioned frame-tensor cache with exact 1:1 (image_id, query_id) key alignment assertions.
  - P-1767: Deterministic equal-F1 tie-breaking favoring stronger suppression (higher tau).
  - P-1763: Multi-seed stability audit across seeds 41, 42, and 43 under LOVO cross-validation.
  - P-1764: Operational gate sweep over tau in [0.10, 0.95] rerunning exact Top-100 evaluation.
  - CHL-1449 / CHL-1453: Treats MLP output as a ranking score; applies hard query admission/suppression.
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
from sklearn.neural_network import MLPClassifier

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
FRAME_TENSORS_PTH = OUT_DIR / "pol3v_design_frames_tensors_v7.pth"
FINAL_VERIFIER_PTH = OUT_DIR / "pol3v_candidate3v_seed41_verifier.pth"
OPERATING_SWEEP_JSON = OUT_DIR / "pol3v_operating_point_sweep_results.json"

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
SEEDS = [41, 42, 43]
TAU_GRID = [0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

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
# 2. OPERATIONAL TOP-100 DETECTOR EVALUATOR
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
# 3. P-1766: FRAME TENSORS EXTRACTION WITH SAME-HISTORY & SCHEMA CERTIFICATION
# ------------------------------------------------------------------------------

def load_or_extract_frame_tensors(driver_sha256):
    if FRAME_TENSORS_PTH.exists():
        print(f"[+] Loading Memoized Frame Detection Tensors: {FRAME_TENSORS_PTH.resolve()}")
        cached_data = torch.load(FRAME_TENSORS_PTH, map_location="cpu")
        valid_schema = (
            isinstance(cached_data, dict)
            and cached_data.get("version") == "v7"
            and "frame_records" in cached_data
            and len(cached_data["frame_records"]) == 206
            and sum(fr["n_temp"] for fr in cached_data["frame_records"]) == 2045
        )
        if valid_schema:
            print("[+] Frame Tensors Cache Validated (Version: v7, 206 frames, 2,045 total temporal queries).")
            return cached_data["frame_records"]
        else:
            print("[!] Frame Tensors Cache failed schema or count assertions. Purging stale cache and re-extracting...")

    print("[*] Re-extracting frame tensors across 206 design target frames under strict v7 invariants...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = SeaClipsDetection(ZIP_PATH, TRAIN_JSON, is_train=False, random_flip=False)
    coco = dataset.coco
    id_to_idx = {iid: idx for idx, iid in enumerate(sorted(coco.getImgIds()))}

    if hasattr(dataset, "label_to_cat_id"):
        label_to_cat_id = dataset.label_to_cat_id
    else:
        cat_ids = sorted(coco.getCatIds())
        label_to_cat_id = {i: cat_id for i, cat_id in enumerate(cat_ids)}

    with open(SPLIT_MANIFEST_JSON, "r", encoding="utf-8") as f:
        design_vids = sorted(json.load(f)["_meta"]["train_videos"])

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

    frame_records = []

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

            gt_boxes_px = []
            gt_cats = []
            for g_idx in range(n_gt):
                gcx, gcy, gw, gh = gt_boxes_norm[g_idx].tolist()
                gx1 = (gcx - gw / 2.0) * img_w
                gy1 = (gcy - gh / 2.0) * img_h
                gt_boxes_px.append([gx1, gy1, gw * img_w, gh * img_h])
                gt_cats.append(label_to_cat_id[gt_labels_raw[g_idx].item()])

            out_d = model(samples, temporal_enabled=False)
            d_logits_cpu = out_d["pred_logits"][0].cpu()
            d_boxes_cpu = out_d["pred_boxes"][0].cpu()

            mb_corr = get_memory_bank_for_window(primary_w, hist_token_cache)
            if mb_corr:
                out_aug = model(samples, temporal_adapter=adapter, memory_bank=[mb_corr], temporal_enabled=True)
                temp_logits_cpu = out_aug["pred_logits"][0, 300:].cpu()
                temp_boxes_cpu = out_aug["pred_boxes"][0, 300:].cpu()
            else:
                temp_logits_cpu = torch.zeros((0, 7), dtype=torch.float32)
                temp_boxes_cpu = torch.zeros((0, 4), dtype=torch.float32)

            frame_records.append({
                "image_id": curr_iid,
                "video_id": vid,
                "img_w": img_w,
                "img_h": img_h,
                "gt_boxes_px": gt_boxes_px,
                "gt_cats": gt_cats,
                "d_logits": d_logits_cpu,
                "d_boxes": d_boxes_cpu,
                "temp_logits": temp_logits_cpu,
                "temp_boxes": temp_boxes_cpu,
                "n_temp": temp_logits_cpu.shape[0]
            })

            if count % 50 == 0 or count == len(windows_by_iid):
                print(f"    Processed {count}/{len(windows_by_iid)} frames...")

    torch.save({
        "version": "v7",
        "driver_sha256": driver_sha256,
        "n_frames": len(frame_records),
        "n_temp_total": sum(fr["n_temp"] for fr in frame_records),
        "frame_records": frame_records
    }, FRAME_TENSORS_PTH)
    print(f"[+] Saved Certified Frame Detection Tensors: {FRAME_TENSORS_PTH.resolve()}")
    return frame_records


# ------------------------------------------------------------------------------
# 4. MAIN BENCHMARK & FREEZE ENGINE (P-1763, P-1764, P-1769, P-1770)
# ------------------------------------------------------------------------------

def main():
    driver_sha256 = compute_sha256(Path(__file__))
    print("=" * 115)
    print("CANDIDATE 3V PRE-FLIGHT: MLP SEED STABILITY AUDIT, OPERATIONAL GATE SWEEP & VERIFIER FREEZE (v3)")
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

    assert QUERY_DATASET_PTH.exists(), f"[FATAL] Missing verified query dataset cache {QUERY_DATASET_PTH}"
    cached_dataset = torch.load(QUERY_DATASET_PTH, map_location="cpu")
    assert cached_dataset.get("version") == "v7", "[FATAL] Query dataset cache version must be v7!"
    all_queries = cached_dataset["extracted_queries"]
    print(f"[+] Loaded Certified Query Dataset: {len(all_queries)} queries.")

    frame_records = load_or_extract_frame_tensors(driver_sha256)
    print(f"[+] Loaded {len(frame_records)} Frame Detection Records across 10 Design Videos.")

    # P-1766: Assert exact alignment between frame tensors and query dataset keys
    frame_keys = []
    for fr in frame_records:
        for q_t in range(fr["n_temp"]):
            frame_keys.append((fr["image_id"], q_t))

    query_keys = [(q["image_id"], q["query_id"]) for q in all_queries]

    assert len(frame_records) == 206, f"[FATAL] Expected 206 frames, got {len(frame_records)}"
    assert sum(fr["n_temp"] for fr in frame_records) == 2045, f"[FATAL] Expected 2,045 queries in frame records, got {sum(fr['n_temp'] for fr in frame_records)}"
    assert frame_keys == query_keys, f"[FATAL] Query-key order mismatch between frame tensors and query dataset!"
    print("[+] Frame-tensor and query-dataset keys certified with 100% exact alignment (2,045 keys).")

    with open(SPLIT_MANIFEST_JSON, "r", encoding="utf-8") as f:
        design_vids = sorted(json.load(f)["_meta"]["train_videos"])

    dataset = SeaClipsDetection(ZIP_PATH, TRAIN_JSON, is_train=False, random_flip=False)
    if hasattr(dataset, "label_to_cat_id"):
        label_to_cat_id = dataset.label_to_cat_id
    else:
        coco = dataset.coco
        cat_ids = sorted(coco.getCatIds())
        label_to_cat_id = {i: cat_id for i, cat_id in enumerate(cat_ids)}

    active_indices = [idx for idx, q in enumerate(all_queries) if q["is_active"]]
    active_queries = [all_queries[idx] for idx in active_indices]

    X_act = np.array([q["features"] for q in active_queries])
    y_act = np.array([q["label_op_util"] for q in active_queries])
    v_act = np.array([q["video_id"] for q in active_queries])
    n_act = len(active_queries)
    n_pos = int(y_act.sum())
    base_prev = float(n_pos / max(1, n_act))

    # --------------------------------------------------------------------------
    # P-1763: MLP SEED STABILITY AUDIT (SEEDS 41, 42, 43)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("P-1763: MLP SEED STABILITY AUDIT (LOVO BENCHMARK OVER SEEDS 41, 42, 43)")
    print("=" * 115)

    seed_oof_scores = {}
    seed_audit_results = {}

    for s in SEEDS:
        print(f"[*] Training MLP LOVO Folds for Seed {s}...")
        lovo_probs = np.full(n_act, np.nan)
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

            mean_tr = np.mean(X_tr, axis=0)
            std_tr = np.std(X_tr, axis=0)
            std_tr[std_tr < 1e-6] = 1.0

            X_tr_std = (X_tr - mean_tr) / std_tr
            X_te_std = (X_te - mean_tr) / std_tr

            pos_indices = np.where(y_tr == 1)[0]
            neg_indices = np.where(y_tr == 0)[0]
            rng_bal = np.random.default_rng(s * 1000 + vid)
            oversampled_pos = rng_bal.choice(pos_indices, size=len(neg_indices), replace=True)
            bal_indices = np.concatenate([neg_indices, oversampled_pos])
            rng_bal.shuffle(bal_indices)
            X_tr_bal = X_tr_std[bal_indices]
            y_tr_bal = y_tr[bal_indices]

            clf = MLPClassifier(hidden_layer_sizes=(32,), activation="relu", alpha=0.05, max_iter=350, random_state=s)
            clf.fit(X_tr_bal, y_tr_bal)
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

        eval_idx_arr = np.array(evaluable_indices)
        pooled_auc = float(roc_auc_score(y_act[eval_idx_arr], lovo_probs[eval_idx_arr]))
        pooled_pr = float(average_precision_score(y_act[eval_idx_arr], lovo_probs[eval_idx_arr]))

        seed_oof_scores[s] = lovo_probs
        seed_audit_results[str(s)] = {
            "pooled_roc_auc": pooled_auc,
            "pooled_pr_auc": pooled_pr,
            "macro_mean_roc_auc": macro_mean_auc,
            "macro_median_roc_auc": macro_med_auc,
            "pr_auc_lift": float(pooled_pr / max(1e-6, base_prev)),
            "fold_metrics": fold_metrics
        }

    ensemble_oof = np.mean([seed_oof_scores[s] for s in SEEDS], axis=0)
    ens_pooled_auc = float(roc_auc_score(y_act[eval_idx_arr], ensemble_oof[eval_idx_arr]))
    ens_pooled_pr = float(average_precision_score(y_act[eval_idx_arr], ensemble_oof[eval_idx_arr]))

    print("\n" + "=" * 115)
    print("P-1763: MULTI-SEED STABILITY AUDIT COMPARISON (10 LOVO DESIGN FOLDS)")
    print("=" * 115)
    print(f"{'Configuration':<24} | {'Pooled ROC-AUC':<16} | {'Macro Mean AUC':<16} | {'Macro Med AUC':<16} | {'Pooled PR-AUC':<14} | {'PR Lift'}")
    print("-" * 105)
    for s in SEEDS:
        res = seed_audit_results[str(s)]
        print(f"Seed {s:<19} | {res['pooled_roc_auc']:12.4f}   | {res['macro_mean_roc_auc']:14.4f}   | {res['macro_median_roc_auc']:14.4f}   | {res['pooled_pr_auc']:12.4f}   | {res['pr_auc_lift']:6.2f}x")
    print(f"{'3-Seed Mean Ensemble':<24} | {ens_pooled_auc:12.4f}   | {'-':<16} | {'-':<16} | {ens_pooled_pr:12.4f}   | {ens_pooled_pr/max(1e-6, base_prev):6.2f}x")

    # --------------------------------------------------------------------------
    # P-1764 & P-1767: DESIGN OPERATIONAL GATE SWEEP WITH EXACT TIE-BREAKING
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("P-1764: OPERATIONAL GATE SWEEP (HARD THRESHOLDING ON DESIGN VIDEOS, 206 TARGET FRAMES)")
    print("=" * 115)

    base_tp, base_fp, base_fn = 0, 0, 0
    for fr in frame_records:
        tpd, fpd, fnd, _ = evaluate_operational_top100(
            fr["d_logits"], fr["d_boxes"], fr["gt_boxes_px"], fr["gt_cats"], label_to_cat_id, fr["img_w"], fr["img_h"]
        )
        base_tp += tpd
        base_fp += fpd
        base_fn += fnd

    base_prec = base_tp / max(1, base_tp + base_fp)
    base_rec = base_tp / max(1, base_tp + base_fn)
    base_f1 = 2 * base_prec * base_rec / max(1e-7, base_prec + base_rec)

    print(f"[*] Baseline M0 Detector Performance: TP = {base_tp}, FP = {base_fp}, FN = {base_fn} | Prec = {base_prec:.4f}, Rec = {base_rec:.4f}, F1 = {base_f1:.6f}")

    primary_oof_scores = seed_oof_scores[41] # Predeclared Seed 41 policy
    query_score_map = {}
    query_label_map = {}
    for idx_act, act_idx in enumerate(active_indices):
        q = all_queries[act_idx]
        key = (q["image_id"], q["query_id"])
        query_score_map[key] = float(primary_oof_scores[idx_act])
        query_label_map[key] = q["label_op_util"]

    print(f"\n{'Threshold (tau)':<16} | {'Admitted (Total)':<18} | {'Retained Oracle Util':<22} | {'TP':<6} | {'FP':<6} | {'FN':<6} | {'Precision':<11} | {'Recall':<10} | {'F1':<10} | {'Delta F1 vs D'}")
    print("-" * 136)

    sweep_results = {}
    best_tau = None
    best_f1 = -1.0
    best_delta_f1 = -999.0
    best_record = None

    for tau in TAU_GRID:
        tot_tp, tot_fp, tot_fn = 0, 0, 0
        admitted_count = 0
        retained_op_pos = 0

        for fr in frame_records:
            n_t = fr["n_temp"]
            if n_t > 0:
                joint_logits = torch.cat([fr["d_logits"], fr["temp_logits"]], dim=0).clone()
                joint_boxes = torch.cat([fr["d_boxes"], fr["temp_boxes"]], dim=0)

                # Hard query admission thresholding
                for q_t in range(n_t):
                    key = (fr["image_id"], q_t)
                    q_sc = query_score_map.get(key, 0.0)
                    if q_sc >= tau:
                        admitted_count += 1
                        if query_label_map.get(key, 0) == 1:
                            retained_op_pos += 1
                    else:
                        joint_logits[300 + q_t, :] = -1e9
            else:
                joint_logits = fr["d_logits"]
                joint_boxes = fr["d_boxes"]

            tpf, fpf, fnf, _ = evaluate_operational_top100(
                joint_logits, joint_boxes, fr["gt_boxes_px"], fr["gt_cats"], label_to_cat_id, fr["img_w"], fr["img_h"]
            )
            tot_tp += tpf
            tot_fp += fpf
            tot_fn += fnf

        prec = tot_tp / max(1, tot_tp + tot_fp)
        rec = tot_tp / max(1, tot_tp + tot_fn)
        f1 = 2 * prec * rec / max(1e-7, prec + rec)
        delta_f1 = f1 - base_f1

        sweep_results[str(tau)] = {
            "tau": tau,
            "admitted_total": admitted_count,
            "retained_oracle_utility_queries": retained_op_pos,
            "tp": tot_tp,
            "fp": tot_fp,
            "fn": tot_fn,
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "delta_f1": float(delta_f1)
        }

        print(f"tau = {tau:<10.2f} | {admitted_count:8d} / 2,045   | {retained_op_pos:2d} / 92 ({retained_op_pos/92*100:4.1f}%)         | {tot_tp:4d} | {tot_fp:4d} | {tot_fn:4d} | {prec:9.4f} | {rec:8.4f} | {f1:8.4f} | {delta_f1:+8.4f}")

        # P-1767: Deterministic tie-breaking favoring higher tau (stronger suppression)
        is_better = False
        if f1 > best_f1 + 1e-12:
            is_better = True
        elif abs(f1 - best_f1) <= 1e-12:
            if best_tau is None or tau > best_tau:
                is_better = True

        if is_better:
            best_f1 = f1
            best_tau = tau
            best_delta_f1 = delta_f1
            best_record = sweep_results[str(tau)]

    print("\n" + "=" * 115)
    print("P-1765 / P-1767: CANDIDATE 3V OPERATING POINT FREEZE (DESIGN COHORT)")
    print("=" * 115)

    # --------------------------------------------------------------------------
    # P-1769 & P-1770: CONDITIONAL ARTIFACT FREEZE & EXPLICIT PREFILTER CONTRACT
    # --------------------------------------------------------------------------
    verifier_hash = None
    gate_status = "REJECTED_DESIGN_GATE"

    if best_delta_f1 > 0:
        gate_status = "VERIFIED_DESIGN_GATE_ARTIFACT_FROZEN"
        print(f"[+] CANDIDATE 3V VERIFIER OPERATIONAL THRESHOLD FROZEN: tau* = {best_tau:.2f}")
        print(f"    - Baseline M0 F1:     {base_f1:.4f} (TP={base_tp}, FP={base_fp}, FN={base_fn})")
        print(f"    - Candidate 3V F1:    {best_f1:.4f} (TP={best_record['tp']}, FP={best_record['fp']}, FN={best_record['fn']})")
        print(f"    - Net Operational Gain: Delta F1 = {best_delta_f1:+.4f} ({base_f1:.4f} -> {best_f1:.4f})")
        print(f"    - Temporal Admission: Admitted {best_record['admitted_total']} / 2,045 queries ({best_record['retained_oracle_utility_queries']}/92 oracle utility queries retained)")

        print("\n" + "=" * 115)
        print("P-1768 / P-1770: FITTING & FREEZING FINAL FULL-DESIGN SEED-41 VERIFIER ARTIFACT")
        print("=" * 115)

        DEPLOYMENT_SEED = 41
        mean_full = np.mean(X_act, axis=0)
        std_full = np.std(X_act, axis=0)
        std_full[std_full < 1e-6] = 1.0

        X_act_std = (X_act - mean_full) / std_full

        pos_indices_all = np.where(y_act == 1)[0]
        neg_indices_all = np.where(y_act == 0)[0]
        rng_deploy = np.random.default_rng(DEPLOYMENT_SEED * 1000 + 777)
        oversampled_pos_all = rng_deploy.choice(pos_indices_all, size=len(neg_indices_all), replace=True)
        bal_indices_all = np.concatenate([neg_indices_all, oversampled_pos_all])
        rng_deploy.shuffle(bal_indices_all)
        X_deploy_bal = X_act_std[bal_indices_all]
        y_deploy_bal = y_act[bal_indices_all]

        deploy_clf = MLPClassifier(hidden_layer_sizes=(32,), activation="relu", alpha=0.05, max_iter=350, random_state=DEPLOYMENT_SEED)
        deploy_clf.fit(X_deploy_bal, y_deploy_bal)

        # P-1770: Explicitly record the temporal prefilter contract in the verifier artifact
        final_verifier_payload = {
            "version": "v1",
            "model_architecture": "MLPClassifier_15_32_1_ReLU",
            "n_params": 545,
            "seed": DEPLOYMENT_SEED,
            "temporal_prefilter_score": float(SCORE_THRESHOLD),
            "verifier_applies_to": "threshold_active_temporal_queries_only",
            "feature_names": FEATURE_NAMES,
            "mean": mean_full.tolist(),
            "std": std_full.tolist(),
            "tau_star": float(best_tau),
            "optimal_delta_f1": float(best_delta_f1),
            "clf_model": deploy_clf,
            "source_driver_sha256": driver_sha256,
            "query_dataset_sha256": compute_sha256(QUERY_DATASET_PTH),
            "frame_tensors_sha256": compute_sha256(FRAME_TENSORS_PTH),
            "epoch8_adapter_sha256": EXPECTED_EPOCH8_ADAPTER_SHA256,
        }

        torch.save(final_verifier_payload, FINAL_VERIFIER_PTH)
        verifier_hash = compute_sha256(FINAL_VERIFIER_PTH)
        print(f"[+] Final Candidate-3V Deployment Verifier Frozen: {FINAL_VERIFIER_PTH.resolve()}")
        print(f"    Verifier Artifact SHA-256:                    {verifier_hash}")
    else:
        # P-1769: If delta_f1 <= 0, do not create or preserve a deployment artifact
        print(f"[!] NO OPERATIONAL F1 IMPROVEMENT (Best Delta F1 = {best_delta_f1:+.4f}).")
        print("    Candidate 3V rejected at design gate. No deployment verifier artifact will be created (P-1769).")
        if FINAL_VERIFIER_PTH.exists():
            FINAL_VERIFIER_PTH.unlink()
            print("    [!] Removed stale verifier artifact from disk.")

    results_payload = {
        "_meta": {
            "artifact_name": "pol3v_operating_point_sweep_results_v3",
            "status": gate_status,
            "driver_sha256": driver_sha256,
            "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
            "token_cache_sha256": EXPECTED_TOKEN_CACHE_SHA256,
            "epoch8_adapter_sha256": EXPECTED_EPOCH8_ADAPTER_SHA256,
            "final_verifier_sha256": verifier_hash,
            "design_videos": design_vids,
            "total_design_frames": len(frame_records),
            "baseline_m0_f1": float(base_f1),
            "optimal_tau_star": float(best_tau) if best_tau is not None else None,
            "optimal_delta_f1": float(best_delta_f1)
        },
        "p1763_seed_stability_audit": seed_audit_results,
        "ensemble_metrics": {
            "pooled_roc_auc": float(ens_pooled_auc),
            "pooled_pr_auc": float(ens_pooled_pr)
        },
        "p1764_operational_gate_sweep": sweep_results
    }

    with open(OPERATING_SWEEP_JSON, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    res_hash = compute_sha256(OPERATING_SWEEP_JSON)
    print("\n" + "=" * 115)
    print(f"[+] Operating Point Sweep Manifest Frozen: {OPERATING_SWEEP_JSON.resolve()}")
    print(f"    Manifest SHA-256:                     {res_hash}\n")


if __name__ == "__main__":
    main()