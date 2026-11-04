"""
D:/MARIVISION/H0_workbench/eval_tools/run_marsen_micro_system.py
MARSEN-mu v1.2: Certified Maritime Sensory Evidence Network Micro-Proof & Ablation Engine

Strict Compliance Checklist:
  - D-3754..D-3763: MARSEN-mu architecture; 10 design videos ONLY; calibration/test UNTOUCHED.
  - P-1816: Decoder replay certification validating max|diff| < 5e-4 against certified frame_records.
  - P-1817: Immutable full-ledger cache provenance binding all upstream checkpoints, tokens, and manifests.
  - P-1818: Pre-training evidence health audit asserting 100% finiteness and reporting sensor validity rates.
  - P-1819: Full-system per-video operational decomposition at optimal tau* across all 10 design videos.
  - P-1812: Frame-pair affine ECC warp caching with query-independent residual motion computation.
  - P-1813: Horizon-relative environmental color/texture contrast prior (upper vs. lower scene zones).
  - P-1814: Rigorous boolean key_padding_mask in MHA and masked_fill(-1e9) in attention pooling.
  - Parameter Accounting: Exactly 88,386 trainable parameters certified.
  - Predeclared Decision Gate: F1_marsen > 0.439957 (beats Candidate 4M composite baseline).
"""

import os
import sys
import copy
import json
import hashlib
import zipfile
import io
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import defaultdict
from pycocotools.coco import COCO

from torchvision import transforms, models
from sklearn.metrics import average_precision_score, roc_auc_score

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
TRAIN_JSON = Path("D:/MARIVISION/seaclips-train.json")
SPLIT_MANIFEST_JSON = OUT_DIR / "pol2_candidate2m_micro_split.json"
QUERY_DATASET_PTH = OUT_DIR / "pol3v_design_queries_dataset_v7.pth"
FRAME_TENSORS_PTH = OUT_DIR / "pol3v_design_frames_tensors_v7.pth"
TOKEN_CACHE_PTH = OUT_DIR / "cache_m0_candidate2m_design_calib.pth"
CANDIDATES_JSON = OUT_DIR / "pol2_candidate_windows_v1.json"
CACHED_VISUAL_FEATURES_PTH = OUT_DIR / "pol4a_native_visual_features_1915.pth"
M0_CKPT_PATH = Path("D:/MARIVISION/checkpoints_r0/checkpoint_best.pth")
EPOCH8_MODEL_PTH = OUT_DIR / "pol2_candidate2m_seed41_epoch8.pth"
ZIP_PATH = Path("D:/MARIVISION/SeaClips.zip")

RESULTS_JSON = OUT_DIR / "marsen_micro_system_results.json"
CACHED_LATENTS_PTH = OUT_DIR / "marsen_decoder_latents_2045.pth"
CACHED_BRANCHES_PTH = OUT_DIR / "marsen_micro_authentic_branches_1915.pth"

EXPECTED_CANDIDATE_SHA256 = "f29e0b8a5cbfc46641222d84a6d370c3e9bc51ca4ca1124b7753078ca7c1e270"
EXPECTED_TRAIN_JSON_SHA256 = "edc32bdbd6b792eb83b6daf11e14708232f1f4a9e06153ebebd5bad1caef9b35"
EXPECTED_SPLIT_SHA256 = "7da471d2793fe3369508200dec50422bb76492860c67a1a61871645471856772"
EXPECTED_TOKEN_CACHE_SHA256 = "d1e6407722c59968f87e0ae9cfdead28cacf89924dcd765ca41cf8993da4dd4c"
EXPECTED_M0_HASH = "28e6288c3f6769c2d5275aa77ddb248735b55a09cd46a6b570e3f117890b268b"
EXPECTED_EPOCH8_ADAPTER_SHA256 = "aea024e2c1072fb5f46d1d376ca1b75c0cdab81a20681ac9958b5d6c0b5f00cf"

CROP_SIZE = 128
EXPANSION_FACTOR = 1.5
SCORE_THRESHOLD = 0.20
EVAL_IOU_THRESHOLD = 0.30
MEMORY_AGES = [2, 3, 4, 5]
CANDIDATE_4M_BENCHMARK = 0.439957
TAU_GRID = [0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


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
    return tp, fp, fn


# ------------------------------------------------------------------------------
# 1. CANDIDATE 2M MODULE ARCHITECTURE (FOR DECODER REPLAY & LATENT EXTRACTION)
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
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos), reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt = self.forward_ffn(tgt)
        return tgt
    DeformableTransformerDecoderLayer.forward = patched_layer_forward

    def patched_decoder_forward(self, tgt, reference_points, memory, spatial_shapes, level_start_index, valid_ratios, query_pos=None, src_padding_mask=None, attn_mask=None):
        output = tgt
        intermediate = []
        intermediate_reference_points = []
        for layer_id, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            else:
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]
            output = layer(output, query_pos, reference_points_input, memory, spatial_shapes, level_start_index, src_padding_mask, attn_mask=attn_mask)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)
        return output, reference_points
    DeformableTransformerDecoder.forward = patched_decoder_forward

    def patched_transformer_forward(self, srcs, masks, pos_embeds, query_embed=None, temporal_adapter=None, memory_bank=None, temporal_enabled=True):
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shapes.append((h, w))
            src_flatten.append(src.flatten(2).transpose(1, 2))
            mask_flatten.append(mask.flatten(1))
            lvl_pos_embed = pos_embed.flatten(2).transpose(1, 2) + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)

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
            tgt_all, ref_all, memory, spatial_shapes, level_start_index, valid_ratios, pos_all, mask_flatten, attn_mask=attn_mask
        )
        return hs, ref_all, inter_references, k_mem_count
    DeformableTransformer.forward = patched_transformer_forward

    def patched_detr_forward(self, samples: NestedTensor, temporal_adapter=None, memory_bank=None, temporal_enabled=True, return_decoder_tokens=False):
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
            srcs, masks, pos, query_embeds, temporal_adapter=temporal_adapter, memory_bank=memory_bank, temporal_enabled=temporal_enabled
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

        out = {'pred_logits': outputs_classes[-1], 'pred_boxes': outputs_coords[-1]}
        if return_decoder_tokens:
            return out, hs[-1]
        return out
    DeformableDETR.forward = patched_detr_forward


# ------------------------------------------------------------------------------
# 2. P-1816 & P-1817: REPLAY CERTIFICATION & DECODER LATENT EXTRACTION
# ------------------------------------------------------------------------------

def load_or_extract_decoder_latents(device, driver_sha256, frame_records, window_by_iid, hist_token_cache):
    provenance_bundle = {
        "driver_sha256": driver_sha256,
        "m0_sha256": EXPECTED_M0_HASH,
        "epoch8_sha256": EXPECTED_EPOCH8_ADAPTER_SHA256,
        "token_cache_sha256": EXPECTED_TOKEN_CACHE_SHA256,
        "candidates_sha256": EXPECTED_CANDIDATE_SHA256,
        "train_json_sha256": EXPECTED_TRAIN_JSON_SHA256,
        "frame_tensors_sha256": compute_sha256(FRAME_TENSORS_PTH)
    }

    if CACHED_LATENTS_PTH.exists():
        print(f"[+] Verifying Decoder Latents Cache Provenance: {CACHED_LATENTS_PTH.resolve()}")
        c_bundle = torch.load(CACHED_LATENTS_PTH, map_location="cpu")
        if (
            isinstance(c_bundle, dict)
            and c_bundle.get("version") == "v2"
            and c_bundle.get("n_latents") == 2045
            and all(c_bundle.get(k) == v for k, v in provenance_bundle.items())
            and "latents_dict" in c_bundle
        ):
            print("[+] P-1817: Immutable Decoder Latents Provenance Certified (2,045 keys).")
            return c_bundle["latents_dict"]
        print("[!] Latents cache failed provenance verification. Re-extracting with replay parity checks...")

    print("\n[*] P-1816: Executing Candidate-2M Replay & Decoder Latent Extraction with Exact Parity Checks...")
    apply_candidate2m_source_hooks()
    args = get_authoritative_r0_args()
    if hasattr(args, "conditioning_mode"):
        args.conditioning_mode = "none"

    model, _, _ = build(args)
    ckpt = torch.load(M0_CKPT_PATH, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    epoch8_ckpt = torch.load(EPOCH8_MODEL_PTH, map_location="cpu")
    adapter = HistoricalObjectTokenMemory(d_model=256, num_classes=7, max_age=6).to(device)
    adapter.load_state_dict(epoch8_ckpt["adapter_state"])
    adapter.gamma.data.fill_(10.0)
    adapter.eval()

    dataset = SeaClipsDetection(ZIP_PATH, TRAIN_JSON, is_train=False, random_flip=False)
    coco = dataset.coco
    id_to_idx = {iid: idx for idx, iid in enumerate(sorted(coco.getImgIds()))}

    latents_dict = {}
    max_logit_parity_diff = 0.0
    max_box_parity_diff = 0.0

    def get_mb_for_window(window, cache_dict):
        curr_fid = window["frame_ids"][-1]
        mb = []
        for f_idx in range(5):
            h_iid = window["image_ids"][f_idx]
            h_fid = window["frame_ids"][f_idx]
            age = curr_fid - h_fid
            if age in MEMORY_AGES and h_iid in cache_dict:
                for tok in cache_dict[h_iid]:
                    t_c = copy.deepcopy(tok)
                    t_c["age"] = age
                    mb.append(t_c)
        return mb

    with torch.no_grad():
        for count, fr in enumerate(frame_records, 1):
            curr_iid = fr["image_id"]
            n_t = fr["n_temp"]
            if n_t > 0:
                s_c = collate_fn([dataset[id_to_idx[curr_iid]]])
                samples = s_c[0].to(device)
                w = window_by_iid[curr_iid]
                mb = get_mb_for_window(w, hist_token_cache)
                assert len(mb) == n_t, f"Memory bank token count mismatch in frame {curr_iid}!"

                out_aug, hs_last = model(samples, temporal_adapter=adapter, memory_bank=[mb], temporal_enabled=True, return_decoder_tokens=True)
                replayed_logits = out_aug["pred_logits"][0, 300:].cpu()
                replayed_boxes = out_aug["pred_boxes"][0, 300:].cpu()
                temp_hs = hs_last[0, 300:].cpu().numpy().astype(np.float32)

                # P-1816: Rigorous numerical parity assertions against certified frame_records
                l_diff = (replayed_logits - fr["temp_logits"]).abs().max().item()
                b_diff = (replayed_boxes - fr["temp_boxes"]).abs().max().item()
                max_logit_parity_diff = max(max_logit_parity_diff, l_diff)
                max_box_parity_diff = max(max_box_parity_diff, b_diff)

                assert l_diff < 5e-4, f"[FATAL] Replay logit disparity in frame {curr_iid}: {l_diff} >= 5e-4"
                assert b_diff < 5e-4, f"[FATAL] Replay box disparity in frame {curr_iid}: {b_diff} >= 5e-4"

                for q_t in range(n_t):
                    latents_dict[(curr_iid, q_t)] = temp_hs[q_t]

            if count % 50 == 0 or count == len(frame_records):
                print(f"    Replay validated {count}/{len(frame_records)} frames (max logit diff: {max_logit_parity_diff:.2e}, max box diff: {max_box_parity_diff:.2e})...")

    assert len(latents_dict) == 2045, f"[FATAL] Expected 2,045 latents, got {len(latents_dict)}"
    print(f"[+] P-1816: Decoder Replay Parity CERTIFIED (max logit diff = {max_logit_parity_diff:.2e}, max box diff = {max_box_parity_diff:.2e}).")

    save_payload = {"version": "v2", "n_latents": len(latents_dict), "latents_dict": latents_dict}
    save_payload.update(provenance_bundle)
    torch.save(save_payload, CACHED_LATENTS_PTH)
    print(f"[+] P-1817: Saved Certified Decoder Latents Cache: {CACHED_LATENTS_PTH.resolve()}")
    return latents_dict


# ------------------------------------------------------------------------------
# 3. MARSEN-MU SPECIALIST ARCHITECTURE (EXACTLY 88,386 PARAMETERS)
# ------------------------------------------------------------------------------

class MARSENFusionSpecialist(nn.Module):
    def __init__(self, d_model=64, n_heads=4, d_ff=128):
        super().__init__()
        # Input Projectors for the 8 Authentic Evidence Streams (Total: 54,784 params)
        self.proj_global     = nn.Linear(256, d_model)  # 256*64 + 64 = 16,448
        self.proj_hr         = nn.Linear(128, d_model)  # 128*64 + 64 =  8,256
        self.proj_memory     = nn.Linear(320, d_model)  # 320*64 + 64 = 20,544
        self.proj_horizon    = nn.Linear(16,  d_model)  #  16*64 + 64 =  1,088
        self.proj_scene      = nn.Linear(32,  d_model)  #  32*64 + 64 =  2,112
        self.proj_motion     = nn.Linear(32,  d_model)  #  32*64 + 64 =  2,112
        self.proj_reflection = nn.Linear(32,  d_model)  #  32*64 + 64 =  2,112
        self.proj_metadata   = nn.Linear(32,  d_model)  #  32*64 + 64 =  2,112

        # Cross-Evidence Transformer Layer (Total: 33,472 params)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.pool_weights = nn.Linear(d_model, 1)  # 64*1 + 1 = 65

        # Pure Admission Head (Total: 65 params)
        self.head_adm = nn.Linear(d_model, 1)      # 64*1 + 1 = 65

    def forward(self, branch_dict, active_mask=None):
        e_g = self.proj_global(branch_dict["global"])
        e_hr = self.proj_hr(branch_dict["hr"])
        e_mem = self.proj_memory(branch_dict["memory"])
        e_h = self.proj_horizon(branch_dict["horizon"])
        e_sc = self.proj_scene(branch_dict["scene"])
        e_mot = self.proj_motion(branch_dict["motion"])
        e_ref = self.proj_reflection(branch_dict["reflection"])
        e_meta = self.proj_metadata(branch_dict["metadata"])

        tokens = [e_g, e_hr, e_mem, e_h, e_sc, e_mot, e_ref, e_meta]
        E = torch.stack(tokens, dim=1)  # [Batch, 8, 64]

        # P-1814: Rigorous boolean key_padding_mask for Multi-Head Attention
        key_padding_mask = None
        if active_mask is not None:
            drop_mask = (~active_mask.to(E.device)).unsqueeze(0).expand(E.shape[0], -1)  # [Batch, 8]
            key_padding_mask = drop_mask

        attn_out, _ = self.attn(E, E, E, key_padding_mask=key_padding_mask)
        E = self.norm1(E + attn_out)
        ffn_out = self.ffn(E)
        E = self.norm2(E + ffn_out)

        # P-1814: Rigorous attention pooling with -1e9 mask for ablated branches
        raw_pool = self.pool_weights(E)  # [Batch, 8, 1]
        if active_mask is not None:
            drop_3d = (~active_mask.to(E.device)).view(1, 8, 1)
            raw_pool = raw_pool.masked_fill(drop_3d, -1e9)

        weights = F.softmax(raw_pool, dim=1)
        e_fused = (weights * E).sum(dim=1)
        logit_adm = self.head_adm(e_fused)
        return logit_adm


# ------------------------------------------------------------------------------
# 4. AUTONOMOUS GEOMETRIC, SCENE, AND MOTION PRIMITIVES
# ------------------------------------------------------------------------------

def estimate_horizon_with_confidence(img_gray):
    blurred = cv2.GaussianBlur(img_gray, (9, 9), 2.0)
    edges = cv2.Canny(blurred, 30, 80)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=120)
    h, w = img_gray.shape
    if lines is not None and len(lines) > 0:
        rho, theta = lines[0][0]
        if 0.25 * np.pi <= theta <= 0.75 * np.pi:
            return float(theta), float(rho), 1.0
    return float(np.pi / 2.0), float(0.45 * h), 0.0


def compute_ecc_affine_matrix(img_prev_gray, img_curr_gray):
    h, w = img_curr_gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:int(0.60 * h), :] = 255  # Upper static sky/horizon band
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    success = 1.0
    try:
        _, warp_matrix = cv2.findTransformECC(img_prev_gray, img_curr_gray, warp_matrix, cv2.MOTION_AFFINE, criteria, mask, 5)
    except cv2.error:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        success = 0.0
    return warp_matrix, success


def compute_scene_prior_features(img_bgr, cx, cy, w_box, h_box, theta_h, rho_h, h_conf):
    H, W, _ = img_bgr.shape
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    sin_t = np.sin(theta_h)
    cos_t = np.cos(theta_h)
    x_samples = np.linspace(0, W - 1, 5)
    if abs(sin_t) > 1e-4:
        y_horiz_samples = (rho_h - x_samples * cos_t) / sin_t
        y_horiz_avg = float(np.clip(np.mean(y_horiz_samples), 0.1 * H, 0.9 * H))
    else:
        y_horiz_avg = 0.45 * H

    y_water_start = min(H - 10, int(y_horiz_avg + 0.05 * H))
    water_zone_hsv = hsv[y_water_start:, :]
    y_sky_end = max(10, int(y_horiz_avg - 0.05 * H))
    sky_zone_hsv = hsv[:y_sky_end, :]

    mu_water = water_zone_hsv.mean(axis=(0, 1)) / [180.0, 255.0, 255.0] if water_zone_hsv.size > 0 else np.array([0.5, 0.5, 0.5])
    mu_sky = sky_zone_hsv.mean(axis=(0, 1)) / [180.0, 255.0, 255.0] if sky_zone_hsv.size > 0 else np.array([0.5, 0.5, 0.5])

    bx1 = max(0, int((cx - w_box / 2.0) * W))
    by1 = max(0, int((cy - h_box / 2.0) * H))
    bx2 = min(W, max(bx1 + 1, int((cx + w_box / 2.0) * W)))
    by2 = min(H, max(by1 + 1, int((cy + h_box / 2.0) * H)))
    patch_hsv = hsv[by1:by2, bx1:bx2]
    patch_gray = gray[by1:by2, bx1:bx2]

    mu_patch = patch_hsv.mean(axis=(0, 1)) / [180.0, 255.0, 255.0] if patch_hsv.size > 0 else np.zeros(3)
    std_patch = patch_hsv.std(axis=(0, 1)) / [180.0, 255.0, 255.0] if patch_hsv.size > 0 else np.zeros(3)

    tex_patch = float(cv2.Laplacian(patch_gray, cv2.CV_32F).var()) if patch_gray.size > 4 else 0.0
    tex_water = float(cv2.Laplacian(gray[y_water_start:, :], cv2.CV_32F).var()) if water_zone_hsv.size > 4 else 1.0

    d_water_hsv = mu_patch - mu_water
    d_sky_hsv = mu_patch - mu_sky
    tex_ratio = tex_patch / (tex_water + 1e-4)

    d_horiz_val = (cx * W * cos_t + cy * H * sin_t - rho_h) / H
    frac_below_h = 1.0 if cy * H > y_horiz_avg else 0.0

    sc_feat = [
        d_water_hsv[0], d_water_hsv[1], d_water_hsv[2],
        d_sky_hsv[0], d_sky_hsv[1], d_sky_hsv[2],
        tex_ratio, np.log1p(tex_patch),
        mu_patch[0], mu_patch[1], mu_patch[2],
        std_patch[0], std_patch[1], std_patch[2],
        float(cy), float(d_horiz_val), h_conf, frac_below_h
    ]
    return np.pad(sc_feat, (0, 32 - len(sc_feat)), mode='constant')


def compute_reflection_features(img_gray, img_bgr, cx, cy, w_box, h_box, d_horiz, h_conf):
    H, W = img_gray.shape
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    bx1 = max(0, int((cx - w_box / 2.0) * W))
    by1 = max(0, int((cy - h_box / 2.0) * H))
    bx2 = min(W, max(bx1 + 1, int((cx + w_box / 2.0) * W)))
    by2 = min(H, max(by1 + 1, int((cy + h_box / 2.0) * H)))
    cand_patch = img_gray[by1:by2, bx1:bx2]
    cand_hsv = hsv[by1:by2, bx1:bx2]

    refl_y1 = min(H - 1, int((cy + h_box / 2.0) * H))
    refl_y2 = min(H, int(refl_y1 + h_box * H))
    refl_patch = img_gray[refl_y1:refl_y2, bx1:bx2] if refl_y2 > refl_y1 else np.zeros((1, 1), dtype=np.uint8)
    refl_hsv = hsv[refl_y1:refl_y2, bx1:bx2] if refl_y2 > refl_y1 else np.zeros((1, 1, 3), dtype=np.uint8)

    valid_refl = 1.0 if (cand_patch.size > 4 and refl_patch.size > 4 and refl_y2 > refl_y1) else 0.0

    if valid_refl > 0:
        refl_resized = cv2.resize(refl_patch, (cand_patch.shape[1], cand_patch.shape[0]))
        ncc = float(cv2.matchTemplate(cand_patch, refl_resized, cv2.TM_CCOEFF_NORMED)[0, 0])
        grad_cand_y = cv2.Sobel(cand_patch, cv2.CV_32F, 0, 1)
        grad_refl_y = cv2.Sobel(refl_resized, cv2.CV_32F, 0, 1)
        grad_sim = float(cv2.matchTemplate(grad_cand_y, grad_refl_y, cv2.TM_CCOEFF_NORMED)[0, 0])
    else:
        ncc = 0.0
        grad_sim = 0.0

    cand_val = cand_hsv[:, :, 2] / 255.0 if cand_hsv.size > 0 else np.zeros((1, 1))
    refl_val = refl_hsv[:, :, 2] / 255.0 if refl_hsv.size > 0 else np.zeros((1, 1))
    cand_sat = cand_hsv[:, :, 1] / 255.0 if cand_hsv.size > 0 else np.zeros((1, 1))
    refl_sat = refl_hsv[:, :, 1] / 255.0 if refl_hsv.size > 0 else np.zeros((1, 1))

    glint_frac_cand = float((cand_val > 0.85).mean())
    glint_frac_refl = float((refl_val > 0.85).mean())
    mean_val_cand = float(cand_val.mean())
    mean_val_refl = float(refl_val.mean())
    val_ratio = mean_val_cand / (mean_val_refl + 1e-4)

    refl_feat = [
        ncc, abs(ncc), grad_sim, abs(grad_sim),
        valid_refl, glint_frac_cand, glint_frac_refl,
        mean_val_cand, mean_val_refl, val_ratio,
        float(cand_sat.mean()), float(refl_sat.mean()),
        float(refl_y2 - refl_y1) / float(H), float(d_horiz), h_conf, 1.0 if ncc > 0.4 else 0.0
    ]
    return np.pad(refl_feat, (0, 32 - len(refl_feat)), mode='constant')


# ------------------------------------------------------------------------------
# 5. MAIN BENCHMARK ENGINE
# ------------------------------------------------------------------------------

def main():
    driver_sha256 = compute_sha256(Path(__file__))
    print("=" * 115)
    print("MARSEN-MU v1.2: CERTIFIED MARITIME SENSORY EVIDENCE NETWORK BENCHMARK (P-1816..P-1819)")
    print(f"Driver Script SHA-256: {driver_sha256}")
    print("=" * 115)

    assert compute_sha256(CANDIDATES_JSON) == EXPECTED_CANDIDATE_SHA256
    assert compute_sha256(TRAIN_JSON) == EXPECTED_TRAIN_JSON_SHA256
    assert compute_sha256(SPLIT_MANIFEST_JSON) == EXPECTED_SPLIT_SHA256
    assert compute_sha256(TOKEN_CACHE_PTH) == EXPECTED_TOKEN_CACHE_SHA256
    assert compute_sha256(M0_CKPT_PATH) == EXPECTED_M0_HASH
    assert compute_sha256(EPOCH8_MODEL_PTH) == EXPECTED_EPOCH8_ADAPTER_SHA256

    print("[+] All Upstream Data Manifests, Token Cache, M0, and E8 Adapter Certified by Immutable SHA-256.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Target: {device} ({torch.cuda.get_device_name(0)})")

    # Accurate parameter accounting assertion: exactly 88,386 parameters
    dummy_model = MARSENFusionSpecialist()
    total_params = sum(p.numel() for p in dummy_model.parameters() if p.requires_grad)
    assert total_params == 88386, f"[FATAL] Parameter count mismatch: {total_params} != 88,386"
    print(f"[+] MARSEN-mu Specialist Initialized: Exactly {total_params:,} Trainable Parameters Certified.")

    with open(SPLIT_MANIFEST_JSON, "r", encoding="utf-8") as f:
        design_vids = sorted(json.load(f)["_meta"]["train_videos"])

    cached_queries = torch.load(QUERY_DATASET_PTH, map_location="cpu")["extracted_queries"]
    frame_bundle = torch.load(FRAME_TENSORS_PTH, map_location="cpu")
    frame_records = frame_bundle["frame_records"]

    active_indices = [idx for idx, q in enumerate(cached_queries) if q["is_active"]]
    active_queries = [cached_queries[idx] for idx in active_indices]
    n_active = len(active_queries)
    y_act = np.array([q["label_op_util"] for q in active_queries], dtype=np.int32)
    v_act = np.array([q["video_id"] for q in active_queries], dtype=np.int32)
    base_prev = float(np.mean(y_act))

    frame_dict = {fr["image_id"]: fr for fr in frame_records}
    for q in cached_queries:
        fr = frame_dict[q["image_id"]]
        cx, cy, w, h = fr["temp_boxes"][q["query_id"]].tolist()
        q["pred_box_norm"] = (cx, cy, w, h)
        q["img_w"] = fr["img_w"]
        q["img_h"] = fr["img_h"]

    with open(CANDIDATES_JSON, "r", encoding="utf-8") as f:
        cand_windows = json.load(f)["windows"]
    window_by_iid = {w["image_ids"][-1]: w for w in cand_windows if w["video_id"] in design_vids and w.get("n_mem_incr_s20", w.get("n_memincr_s20", 0)) >= 1}
    hist_token_cache = torch.load(TOKEN_CACHE_PTH, map_location="cpu")

    print(f"[+] Loaded {n_active:,} Threshold-Active Queries (92 Positives, Natural Prevalence = {base_prev*100:.2f}%).")

    # --------------------------------------------------------------------------
    # P-1816: EXTRACT / LOAD REPLAY-CERTIFIED DECODER LATENTS (256-D)
    # --------------------------------------------------------------------------
    latents_dict = load_or_extract_decoder_latents(device, driver_sha256, frame_records, window_by_iid, hist_token_cache)
    assert len(latents_dict) == 2045

    # --------------------------------------------------------------------------
    # STEP 1: P-1817 AUTHENTIC 8-BRANCH EVIDENCE INGESTION
    # --------------------------------------------------------------------------
    cache_valid = False
    if CACHED_BRANCHES_PTH.exists():
        print(f"\n[+] Verifying Evidence Branches Cache Provenance: {CACHED_BRANCHES_PTH.resolve()}")
        c_bundle = torch.load(CACHED_BRANCHES_PTH, map_location="cpu")
        if (
            isinstance(c_bundle, dict)
            and c_bundle.get("driver_sha256") == driver_sha256
            and c_bundle.get("n_active") == n_active
            and c_bundle.get("query_dataset_sha256") == compute_sha256(QUERY_DATASET_PTH)
            and c_bundle.get("frame_tensors_sha256") == compute_sha256(FRAME_TENSORS_PTH)
            and c_bundle.get("token_cache_sha256") == EXPECTED_TOKEN_CACHE_SHA256
            and c_bundle.get("candidates_sha256") == EXPECTED_CANDIDATE_SHA256
            and c_bundle.get("train_json_sha256") == EXPECTED_TRAIN_JSON_SHA256
            and c_bundle.get("split_manifest_sha256") == EXPECTED_SPLIT_SHA256
            and c_bundle.get("latents_cache_sha256") == compute_sha256(CACHED_LATENTS_PTH)
            and c_bundle.get("visual_features_sha256") == compute_sha256(CACHED_VISUAL_FEATURES_PTH)
            and "branches" in c_bundle
            and all(k in c_bundle["branches"] for k in ["global", "hr", "memory", "horizon", "scene", "motion", "reflection", "metadata"])
        ):
            branches = c_bundle["branches"]
            cache_valid = True
            print("[+] P-1817: Complete Sealed Upstream Provenance Chain Verified. Evidence Ingested.")

    if not cache_valid:
        print("\n[*] Assembling Authentic 8-Branch Evidence Streams...")

        # Branch 1: P-1816 Authentic Global Decoder Context (256-D)
        feat_global_list = [latents_dict[(q["image_id"], q["query_id"])] for q in active_queries]
        feat_global = np.array(feat_global_list, dtype=np.float32)
        assert feat_global.shape == (n_active, 256)

        # Branch 2: Native HR Visual Embedding (128-D) from Certified Cache
        visual_bundle = torch.load(CACHED_VISUAL_FEATURES_PTH, map_location="cpu")
        feat_hr = visual_bundle["feat_visual"].astype(np.float32)
        assert feat_hr.shape == (n_active, 128)

        # Branch 3: Authentic Historical Memory Tokens (z_hist ∈ R^256) + Age (64-D) -> 320-D
        feat_memory_list = []
        for q in active_queries:
            iid = q["image_id"]
            w = window_by_iid[iid]
            curr_fid = w["frame_ids"][-1]
            q_id = q["query_id"]

            tok_count = 0
            target_tok = None
            for f_idx in range(5):
                h_iid = w["image_ids"][f_idx]
                h_fid = w["frame_ids"][f_idx]
                age = curr_fid - h_fid
                if age in MEMORY_AGES and h_iid in hist_token_cache:
                    for tok in hist_token_cache[h_iid]:
                        if tok_count == q_id:
                            target_tok = tok
                            target_tok["age"] = age
                            target_tok["hist_iid"] = h_iid
                            break
                        tok_count += 1
                    if target_tok is not None:
                        break

            assert target_tok is not None
            z_hist = target_tok["z"].numpy().astype(np.float32)
            age_vec = np.zeros(64, dtype=np.float32)
            age_vec[min(63, int(target_tok["age"]))] = 1.0
            feat_memory_list.append(np.concatenate([z_hist, age_vec]))
            q["hist_iid"] = target_tok["hist_iid"]
            q["hist_bbox_norm"] = target_tok["bbox"]
            q["hist_age"] = target_tok["age"]

        feat_memory = np.array(feat_memory_list, dtype=np.float32)
        assert feat_memory.shape == (n_active, 320)

        # Branch 8: Certified Proposal Metadata (32-D)
        feat_metadata = np.pad(np.array([q["features"] for q in active_queries], dtype=np.float32), ((0, 0), (0, 32 - 15)), mode='constant')

        # Autonomous Extraction of Branches 4 (Horizon), 5 (Scene), 6 (Motion), 7 (Reflection)
        with open(TRAIN_JSON, "r", encoding="utf-8") as f:
            img_meta = {img["id"]: img for img in json.load(f)["images"]}

        unique_iids = sorted(list(set(q["image_id"] for q in active_queries) | set(q["hist_iid"] for q in active_queries)))
        print(f"[*] Preloading {len(unique_iids)} unique frames for geometric, scene, and motion extraction...")

        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zip_index = {name.replace("\\", "/").strip("/"): name for name in zf.namelist()}
            for name in zf.namelist():
                norm = name.replace("\\", "/").strip("/")
                parts = norm.split("/")
                if len(parts) >= 2:
                    zip_index["/".join(parts[-2:])] = name

            img_cache_cv = {}
            for iid in unique_iids:
                rel = img_meta[iid]["file_name"].replace("\\", "/").strip("/")
                t_key = "/".join(rel.split("/")[-2:]) if len(rel.split("/")) >= 2 else rel
                entry = zip_index.get(rel) or zip_index.get(t_key)
                with zf.open(entry) as f:
                    img_cache_cv[iid] = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)

        print("[*] Computing P-1812 affine ECC warps, P-1813 environmental scene priors, and reflection features...")
        ecc_warp_cache = {}
        horizon_cache = {}

        feat_horizon_list = []
        feat_scene_list = []
        feat_motion_list = []
        feat_refl_list = []

        for idx, q in enumerate(active_queries, 1):
            iid = q["image_id"]
            h_iid = q["hist_iid"]
            img_curr = img_cache_cv[iid]
            img_hist = img_cache_cv[h_iid]

            H, W, _ = img_curr.shape
            curr_gray = cv2.cvtColor(img_curr, cv2.COLOR_BGR2GRAY)
            hist_gray = cv2.cvtColor(img_hist, cv2.COLOR_BGR2GRAY)
            cx, cy, w_box, h_box = q["pred_box_norm"]

            # 4. Horizon Geometry with Confidence
            if iid not in horizon_cache:
                horizon_cache[iid] = estimate_horizon_with_confidence(curr_gray)
            theta_h, rho_h, h_conf = horizon_cache[iid]
            d_horiz = (cx * W * np.cos(theta_h) + cy * H * np.sin(theta_h) - rho_h) / H
            h_feat = [
                d_horiz, abs(d_horiz), np.cos(theta_h), np.sin(theta_h),
                h_conf, 1.0 if d_horiz > 0 else 0.0, float(h_box) / (abs(d_horiz) + 1e-4),
                float(w_box) / (float(h_box) + 1e-4)
            ]
            feat_horizon_list.append(np.pad(h_feat, (0, 16 - len(h_feat)), mode='constant'))

            # 5. P-1813: Grounded Water / Sky Scene Prior
            sc_feat = compute_scene_prior_features(img_curr, cx, cy, w_box, h_box, theta_h, rho_h, h_conf)
            feat_scene_list.append(sc_feat)

            # 6. P-1812: Frame-Pair ECC Warp + Query-Specific Residual Motion
            pair_key = (h_iid, iid)
            if pair_key not in ecc_warp_cache:
                ecc_warp_cache[pair_key] = compute_ecc_affine_matrix(hist_gray, curr_gray)
            warp_matrix, ecc_ok = ecc_warp_cache[pair_key]

            cx_h, cy_h = q["hist_bbox_norm"][0], q["hist_bbox_norm"][1]
            pt_h_px = np.array([cx_h * W, cy_h * H, 1.0], dtype=np.float32)
            p_proj_px = warp_matrix @ pt_h_px
            cx_c_px, cy_c_px = cx * W, cy * H
            res_x = (cx_c_px - p_proj_px[0]) / float(W)
            res_y = (cy_c_px - p_proj_px[1]) / float(H)
            r_norm = float(np.sqrt(res_x ** 2 + res_y ** 2))
            scale_w = float(warp_matrix[0, 0])
            scale_h = float(warp_matrix[1, 1])

            mot_feat = [res_x, res_y, r_norm, float(np.log1p(r_norm)), ecc_ok, float(q["hist_age"]), scale_w, scale_h]
            feat_motion_list.append(np.pad(mot_feat, (0, 32 - len(mot_feat)), mode='constant'))

            # 7. Physical Reflection Reasoning
            refl_feat = compute_reflection_features(curr_gray, img_curr, cx, cy, w_box, h_box, d_horiz, h_conf)
            feat_refl_list.append(refl_feat)

            if idx % 250 == 0 or idx == n_active:
                print(f"    Processed authentic evidence streams for {idx}/{n_active} queries...")

        branches = {
            "global": feat_global,
            "hr": feat_hr,
            "memory": feat_memory,
            "horizon": np.array(feat_horizon_list, dtype=np.float32),
            "scene": np.array(feat_scene_list, dtype=np.float32),
            "motion": np.array(feat_motion_list, dtype=np.float32),
            "reflection": np.array(feat_refl_list, dtype=np.float32),
            "metadata": feat_metadata
        }

        save_branches_payload = {
            "driver_sha256": driver_sha256,
            "n_active": n_active,
            "query_dataset_sha256": compute_sha256(QUERY_DATASET_PTH),
            "frame_tensors_sha256": compute_sha256(FRAME_TENSORS_PTH),
            "token_cache_sha256": EXPECTED_TOKEN_CACHE_SHA256,
            "candidates_sha256": EXPECTED_CANDIDATE_SHA256,
            "train_json_sha256": EXPECTED_TRAIN_JSON_SHA256,
            "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
            "latents_cache_sha256": compute_sha256(CACHED_LATENTS_PTH),
            "visual_features_sha256": compute_sha256(CACHED_VISUAL_FEATURES_PTH),
            "branches": branches
        }
        torch.save(save_branches_payload, CACHED_BRANCHES_PTH)
        print(f"[+] All 8 Authentic Evidence Branches Cached: {CACHED_BRANCHES_PTH.resolve()}")

    # --------------------------------------------------------------------------
    # P-1818: EVIDENCE HEALTH & BRANCH FIDELITY AUDIT
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("P-1818: EVIDENCE HEALTH & BRANCH FIDELITY AUDIT")
    print("=" * 115)
    expected_shapes = {
        "global": 256, "hr": 128, "memory": 320, "horizon": 16,
        "scene": 32, "motion": 32, "reflection": 32, "metadata": 32
    }
    for b_name, b_dim in expected_shapes.items():
        arr = branches[b_name]
        assert arr.shape == (n_active, b_dim), f"[FATAL] Branch {b_name} shape mismatch: {arr.shape} != ({n_active}, {b_dim})"
        assert np.isfinite(arr).all(), f"[FATAL] Branch {b_name} contains non-finite/NaN values!"
        print(f"  - Branch {b_name:<12} : shape = {str(arr.shape):<14} | finite = 100% | range = [{arr.min():+7.2f}, {arr.max():+7.2f}]")

    # Sensor Operational Diagnostics
    horizon_valid = float((branches["horizon"][:, 4] == 1.0).mean()) * 100.0
    ecc_success = float((branches["motion"][:, 4] == 1.0).mean()) * 100.0
    refl_valid = float((branches["reflection"][:, 4] == 1.0).mean()) * 100.0
    motion_r = branches["motion"][:, 2]

    print(f"\n[*] Autonomous Maritime Sensor Health Summary:")
    print(f"  - Horizon Line Detection Success:     {horizon_valid:5.1f}% of active queries")
    print(f"  - ECC Background Homography Success:   {ecc_success:5.1f}% of active queries")
    print(f"  - Reflection Symmetry Valid Patches:   {refl_valid:5.1f}% of active queries")
    print(f"  - Residual Target Motion Magnitude:    min = {motion_r.min():.4f}, mean = {motion_r.mean():.4f}, max = {motion_r.max():.4f}")

    # --------------------------------------------------------------------------
    # STEP 2: P-1809 PERSISTENT LOVO MICRO-TRAINING (10 DESIGN FOLDS)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("PHASE A: LOVO CROSS-VALIDATION MICRO-TRAINING (MARSEN-MU SPECIALIST)")
    print("=" * 115)

    oof_scores = np.full(n_active, np.nan)
    models_by_fold = {}
    scalers_by_fold = {}

    for vid in design_vids:
        test_mask = (v_act == vid)
        train_mask = ~test_mask

        if test_mask.sum() == 0 or len(np.unique(y_act[train_mask])) < 2:
            continue

        b_tr = {}
        b_te = {}
        scalers_by_fold[vid] = {}

        for k in branches.keys():
            m_k = branches[k][train_mask].mean(axis=0)
            s_k = branches[k][train_mask].std(axis=0)
            s_k[s_k < 1e-6] = 1.0
            scalers_by_fold[vid][k] = (m_k, s_k)
            b_tr[k] = torch.tensor((branches[k][train_mask] - m_k) / s_k, dtype=torch.float32, device=device)
            b_te[k] = torch.tensor((branches[k][test_mask] - m_k) / s_k, dtype=torch.float32, device=device)

        y_tr_t = torch.tensor(y_act[train_mask], dtype=torch.float32, device=device).unsqueeze(1)

        torch.manual_seed(42 * 1000 + vid)
        fold_model = MARSENFusionSpecialist(d_model=64, n_heads=4, d_ff=128).to(device)
        optimizer = torch.optim.AdamW(fold_model.parameters(), lr=1e-3, weight_decay=1e-4)

        pos_weight = torch.tensor([(len(y_tr_t) - y_tr_t.sum()) / max(1.0, y_tr_t.sum())], device=device)

        fold_model.train()
        for epoch in range(40):
            optimizer.zero_grad()
            logit_adm = fold_model(b_tr)
            loss = F.binary_cross_entropy_with_logits(logit_adm, y_tr_t, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()

        fold_model.eval()
        with torch.no_grad():
            logit_te = fold_model(b_te)
            oof_scores[test_mask] = torch.sigmoid(logit_te).cpu().numpy().flatten()

        models_by_fold[vid] = fold_model

    assert np.isfinite(oof_scores).all(), "[FATAL] Non-finite OOF scores generated!"
    pooled_ap = float(average_precision_score(y_act, oof_scores))
    pooled_auc = float(roc_auc_score(y_act, oof_scores))
    print(f"[+] MARSEN-mu OOF Ranking Performance: Pooled AP = {pooled_ap:.4f} (Lift: {pooled_ap/base_prev:.2f}x) | ROC-AUC = {pooled_auc:.4f}")

    # --------------------------------------------------------------------------
    # STEP 3: OPERATIONAL GATE SWEEP (WHOLE-DETECTOR TOP-100 RERUN OVER 206 FRAMES)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("PHASE B: SYSTEM-LEVEL OPERATIONAL GATE SWEEP (206 FRAMES, BASELINE F1 = 0.431718)")
    print("=" * 115)

    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        coco_full = json.load(f)
    cat_ids = sorted(c["id"] for c in coco_full["categories"])
    label_to_cat_id = {i: cat_id for i, cat_id in enumerate(cat_ids)}

    base_tp, base_fp, base_fn = 196, 299, 217
    base_f1 = 0.431718

    active_key_set = set((q["image_id"], q["query_id"]) for q in active_queries)
    positive_key_set = set((q["image_id"], q["query_id"]) for q in active_queries if q["label_op_util"] == 1)
    score_map = { (q["image_id"], q["query_id"]): float(oof_scores[i]) for i, q in enumerate(active_queries) }

    best_tau = None
    best_f1 = -1.0
    best_delta_f1 = -999.0
    best_record = None

    print(f"{'Threshold (tau)':<16} | {'Admitted (Active / 1,915)':<26} | {'Retained Op Pos':<18} | {'TP':<6} | {'FP':<6} | {'FN':<6} | {'Precision':<11} | {'Recall':<10} | {'F1':<10} | {'Delta F1 vs D'}")
    print("-" * 140)

    for tau in TAU_GRID:
        tot_tp, tot_fp, tot_fn = 0, 0, 0
        admitted_count = 0
        retained_pos = 0

        for fr in frame_records:
            n_t = fr["n_temp"]
            if n_t > 0:
                joint_logits = torch.cat([fr["d_logits"], fr["temp_logits"]], dim=0).clone()
                joint_boxes = torch.cat([fr["d_boxes"], fr["temp_boxes"]], dim=0)

                for q_t in range(n_t):
                    key = (fr["image_id"], q_t)
                    if key in active_key_set:
                        sc = score_map[key]
                        if sc >= tau:
                            admitted_count += 1
                            if key in positive_key_set:
                                retained_pos += 1
                        else:
                            joint_logits[300 + q_t, :] = -1e9
                    else:
                        joint_logits[300 + q_t, :] = -1e9
            else:
                joint_logits = fr["d_logits"]
                joint_boxes = fr["d_boxes"]

            tpf, fpf, fnf = evaluate_operational_top100(
                joint_logits, joint_boxes, fr["gt_boxes_px"], fr["gt_cats"], label_to_cat_id, fr["img_w"], fr["img_h"]
            )
            tot_tp += tpf
            tot_fp += fpf
            tot_fn += fnf

        prec = tot_tp / max(1, tot_tp + tot_fp)
        rec = tot_tp / max(1, tot_tp + tot_fn)
        f1 = 2 * prec * rec / max(1e-7, prec + rec)
        delta_f1 = f1 - base_f1

        print(f"tau = {tau:<10.2f} | {admitted_count:6d} / 1,915 active (2,045 tot) | {retained_pos:2d} / 92 ({retained_pos/92*100:4.1f}%) | {tot_tp:4d} | {tot_fp:4d} | {tot_fn:4d} | {prec:9.4f} | {rec:8.4f} | {f1:8.4f} | {delta_f1:+8.4f}")

        if f1 > best_f1 + 1e-12 or (abs(f1 - best_f1) <= 1e-12 and (best_tau is None or tau > best_tau)):
            best_f1 = f1
            best_tau = tau
            best_delta_f1 = delta_f1
            best_record = {"tp": tot_tp, "fp": tot_fp, "fn": tot_fn, "prec": prec, "rec": rec, "f1": f1, "admitted": admitted_count, "pos": retained_pos}

    print("\n" + "=" * 115)
    print("PHASE C: PREDECLARED SYSTEM-LEVEL OPERATIONAL VERDICT (P-1804)")
    print("=" * 115)
    system_gate_passed = best_f1 > CANDIDATE_4M_BENCHMARK
    strong_proof_passed = best_delta_f1 >= 0.0100

    print(f"[*] MARSEN-mu Optimal Operating Point: tau* = {best_tau:.2f}")
    print(f"    - Baseline M0 F1:              {base_f1:.6f}")
    print(f"    - Candidate 4M Composite F1:   {CANDIDATE_4M_BENCHMARK:.6f} (Delta F1 = +0.0082)")
    print(f"    - MARSEN-mu Integrated F1:     {best_f1:.6f} (Delta F1 = {best_delta_f1:+.4f})")
    print(f"    - System-Level Proof Status:   {'PASSED (Outperformed Candidate 4M)' if system_gate_passed else 'FAILED (Did not beat 4M baseline)'}")
    print(f"    - Strong Proof Benchmark:      {'ACHIEVED (Delta F1 >= +0.0100)' if strong_proof_passed else 'NOT REACHED (< +0.0100)'}")

    # --------------------------------------------------------------------------
    # P-1819: FULL-SYSTEM PER-VIDEO OPERATIONAL DECOMPOSITION (AT tau*)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print(f"P-1819: FULL-SYSTEM PER-VIDEO OPERATIONAL DECOMPOSITION (AT tau* = {best_tau:.2f})")
    print("=" * 115)
    print(f"{'Video ID':<12} | {'Frames':<8} | {'Active (Pos)':<16} | {'Admitted (Op Pos)':<20} | {'Baseline (TP/FP/FN/F1)':<24} | {'MARSEN (TP/FP/FN/F1)':<24} | {'Delta F1'}")
    print("-" * 126)

    per_video_decomp = {}
    for vid in design_vids:
        vid_frames = [fr for fr in frame_records if fr["video_id"] == vid]
        b_tp_v, b_fp_v, b_fn_v = 0, 0, 0
        m_tp_v, m_fp_v, m_fn_v = 0, 0, 0
        adm_v, adm_pos_v, act_v, pos_v = 0, 0, 0, 0

        for fr in vid_frames:
            tpb, fpb, fnb = evaluate_operational_top100(
                fr["d_logits"], fr["d_boxes"], fr["gt_boxes_px"], fr["gt_cats"], label_to_cat_id, fr["img_w"], fr["img_h"]
            )
            b_tp_v += tpb
            b_fp_v += fpb
            b_fn_v += fnb

            n_t = fr["n_temp"]
            if n_t > 0:
                j_logits = torch.cat([fr["d_logits"], fr["temp_logits"]], dim=0).clone()
                j_boxes = torch.cat([fr["d_boxes"], fr["temp_boxes"]], dim=0)

                for q_t in range(n_t):
                    key = (fr["image_id"], q_t)
                    if key in active_key_set:
                        act_v += 1
                        if key in positive_key_set:
                            pos_v += 1
                        if score_map[key] >= best_tau:
                            adm_v += 1
                            if key in positive_key_set:
                                adm_pos_v += 1
                        else:
                            j_logits[300 + q_t, :] = -1e9
                    else:
                        j_logits[300 + q_t, :] = -1e9
            else:
                j_logits = fr["d_logits"]
                j_boxes = fr["d_boxes"]

            tpm, fpm, fnm = evaluate_operational_top100(
                j_logits, j_boxes, fr["gt_boxes_px"], fr["gt_cats"], label_to_cat_id, fr["img_w"], fr["img_h"]
            )
            m_tp_v += tpm
            m_fp_v += fpm
            m_fn_v += fnm

        f1_b_v = 2 * b_tp_v / max(1e-7, 2 * b_tp_v + b_fp_v + b_fn_v)
        f1_m_v = 2 * m_tp_v / max(1e-7, 2 * m_tp_v + m_fp_v + m_fn_v)
        d_f1_v = f1_m_v - f1_b_v

        per_video_decomp[str(vid)] = {
            "n_frames": len(vid_frames),
            "n_active": act_v,
            "n_pos": pos_v,
            "n_admitted": adm_v,
            "n_admitted_op_pos": adm_pos_v,
            "delta_tp": m_tp_v - b_tp_v,
            "delta_fp": m_fp_v - b_fp_v,
            "delta_fn": m_fn_v - b_fn_v,
            "f1_baseline": float(f1_b_v),
            "f1_model": float(f1_m_v),
            "delta_f1": float(d_f1_v)
        }

        b_str = f"{b_tp_v:2d}/{b_fp_v:2d}/{b_fn_v:2d} ({f1_b_v:.4f})"
        m_str = f"{m_tp_v:2d}/{m_fp_v:2d}/{m_fn_v:2d} ({f1_m_v:.4f})"
        adm_pos_str = f"{adm_v:4d} ({adm_pos_v:2d})"
        print(f"video_{vid:<6d} | {len(vid_frames):<8d} | {act_v:4d} ({pos_v:2d})       | {adm_pos_str:<20} | {b_str:<24} | {m_str:<24} | {d_f1_v:+8.4f}")

    # --------------------------------------------------------------------------
    # STEP 4: P-1814 RIGOROUS FOLD-PERSISTENT INFERENCE BRANCH ABLATION MATRIX
    # --------------------------------------------------------------------------
    ablation_results = {}
    if system_gate_passed:
        print("\n" + "=" * 115)
        print(f"PHASE D: P-1814 INFERENCE BRANCH ABLATION MATRIX (EVALUATED AT tau* = {best_tau:.2f})")
        print("=" * 115)
        print(f"{'Ablated Model Variant':<36} | {'Ablated Branch':<24} | {'TP':<5} | {'FP':<5} | {'FN':<5} | {'Ablated F1':<12} | {'F1 Drop vs Full'}")
        print("-" * 110)

        branch_names = [
            "Global Context", "Native HR Visual", "Temporal Memory", "Horizon Geometry",
            "Scene Prior", "Ego-Motion", "Reflection Reasoning", "Metadata Context"
        ]
        print(f"{'MARSEN-mu (Full Specialist)':<36} | {'None (All 8 Active)':<24} | {best_record['tp']:3d}   | {best_record['fp']:3d}   | {best_record['fn']:3d}   | {best_f1:10.6f}   | {'REF (0.0000)'}")

        for drop_idx, b_name in enumerate(branch_names):
            active_mask = torch.ones(8, dtype=torch.bool)
            active_mask[drop_idx] = False

            abl_scores = np.full(n_active, np.nan)

            for vid in design_vids:
                test_mask = (v_act == vid)
                if test_mask.sum() == 0:
                    continue

                fold_model = models_by_fold[vid]
                b_te_abl = {}
                for k in branches.keys():
                    m_k, s_k = scalers_by_fold[vid][k]
                    b_te_abl[k] = torch.tensor((branches[k][test_mask] - m_k) / s_k, dtype=torch.float32, device=device)

                with torch.no_grad():
                    logit_abl = fold_model(b_te_abl, active_mask=active_mask)
                    abl_scores[test_mask] = torch.sigmoid(logit_abl).cpu().numpy().flatten()

            abl_score_map = { (q["image_id"], q["query_id"]): float(abl_scores[i]) for i, q in enumerate(active_queries) }

            tot_tp_a, tot_fp_a, tot_fn_a = 0, 0, 0
            for fr in frame_records:
                n_t = fr["n_temp"]
                if n_t > 0:
                    j_logits = torch.cat([fr["d_logits"], fr["temp_logits"]], dim=0).clone()
                    j_boxes = torch.cat([fr["d_boxes"], fr["temp_boxes"]], dim=0)

                    for q_t in range(n_t):
                        key = (fr["image_id"], q_t)
                        if key in active_key_set and abl_score_map[key] >= best_tau:
                            pass
                        else:
                            j_logits[300 + q_t, :] = -1e9
                else:
                    j_logits = fr["d_logits"]
                    j_boxes = fr["d_boxes"]

                tpa, fpa, fna = evaluate_operational_top100(j_logits, j_boxes, fr["gt_boxes_px"], fr["gt_cats"], label_to_cat_id, fr["img_w"], fr["img_h"])
                tot_tp_a += tpa
                tot_fp_a += fpa
                tot_fn_a += fna

            f1_a = 2 * tot_tp_a / max(1e-7, 2 * tot_tp_a + tot_fp_a + tot_fn_a)
            drop_f1 = f1_a - best_f1
            ablation_results[b_name] = {"f1": float(f1_a), "drop_f1": float(drop_f1), "tp": tot_tp_a, "fp": tot_fp_a, "fn": tot_fn_a}
            print(f"{f'MARSEN-mu (- {b_name})':<36} | {b_name:<24} | {tot_tp_a:3d}   | {tot_fp_a:3d}   | {tot_fn_a:3d}   | {f1_a:10.6f}   | {drop_f1:+10.4f}")

    results_payload = {
        "_meta": {
            "artifact_name": "marsen_micro_system_results_v1_2",
            "driver_sha256": driver_sha256,
            "system_gate_passed": system_gate_passed,
            "strong_proof_passed": strong_proof_passed,
            "optimal_tau_star": float(best_tau),
            "baseline_m0_f1": float(base_f1),
            "candidate_4m_f1": float(CANDIDATE_4M_BENCHMARK),
            "marsen_micro_f1": float(best_f1),
            "delta_f1": float(best_delta_f1),
            "trainable_parameters": total_params,
            "sensor_health": {
                "horizon_valid_pct": float(horizon_valid),
                "ecc_success_pct": float(ecc_success),
                "refl_valid_pct": float(refl_valid),
                "motion_r_mean": float(motion_r.mean())
            }
        },
        "operational_best_record": best_record,
        "per_video_decomposition": per_video_decomp,
        "ablation_results": ablation_results if system_gate_passed else {}
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    res_hash = compute_sha256(RESULTS_JSON)
    print("\n" + "=" * 115)
    print(f"[+] MARSEN-mu System Results Frozen: {RESULTS_JSON.resolve()}")
    print(f"    Manifest SHA-256:                 {res_hash}")
    print("=" * 115)


if __name__ == "__main__":
    main()