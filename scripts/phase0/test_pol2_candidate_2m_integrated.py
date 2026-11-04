"""
D:/MARIVISION/H0_workbench/eval_tools/test_pol2_candidate_2m_integrated.py
Candidate 2M Real-Detector Structural Integration & Parity Verification Harness

Strict Compliance Checklist:
  - D-3588: Multi-age temporal memory scope locked to k in {2, 3, 4, 5}.
  - D-3589: Memory bank capacity locked to s >= 0.20, K_max = 8 tokens per age (max 32 tokens).
  - P-1682: Evaluates Contract 1 against a pristine unmodified M0 baseline before applying hooks.
  - P-1687 / D-3596: Direct live-vs-frozen cache equivalence check on all historical FIFO frames.
  - P-1688 / D-3597: Target-specific Hungarian matcher audit reporting MI target query assignments.
  - P-1689 / D-3598: Top-100 identity parity check verifying (query_idx, class_idx) set equality.
  - P-1684: Shape-matched sensitivity contract comparing equal-cardinality correct vs perturbed memory banks.
  - P-1686: Complete gradient flow assertions across all adapter submodules (token_proj, pos_proj, cls_proj, age_embed, norm, gamma).
  - D-3582: Asymmetric block-triangular self-attention masking isolating canonical queries from temporal queries.
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
from collections import deque
from pycocotools.coco import COCO

# Add Deformable-DETR to sys.path
DETR_PATH = Path("D:/MARIVISION/Deformable-DETR")
if str(DETR_PATH) not in sys.path:
    sys.path.insert(0, str(DETR_PATH))

if "R0_MSDA_BACKEND" not in os.environ:
    os.environ["R0_MSDA_BACKEND"] = "pytorch"

from models.deformable_detr import build, DeformableDETR
from models.deformable_transformer import DeformableTransformer, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from r0_preflight import get_authoritative_r0_args
from datasets.seaclips_detection import SeaClipsDetection
from util.misc import collate_fn, NestedTensor, inverse_sigmoid, nested_tensor_from_tensor_list

OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
CANDIDATES_JSON = OUT_DIR / "pol2_candidate_windows_v1.json"
TRAIN_JSON = Path("D:/MARIVISION/seaclips-train.json")
CACHE_FILE = OUT_DIR / "cache_m0_train_raw_top100.json"
DEV_POOL_JSON = OUT_DIR / "pol2_dev_pool_manifest.json"
SPLIT_MANIFEST_JSON = OUT_DIR / "pol2_candidate2m_micro_split.json"
AGE_AUDIT_JSON = OUT_DIR / "pol2_candidate2m_age_subsets.json"
M0_CKPT_PATH = Path("D:/MARIVISION/checkpoints_r0/checkpoint_best.pth")
ZIP_PATH = Path("D:/MARIVISION/SeaClips.zip")

EXPECTED_CANDIDATE_SHA256 = "f29e0b8a5cbfc46641222d84a6d370c3e9bc51ca4ca1124b7753078ca7c1e270"
EXPECTED_TRAIN_JSON_SHA256 = "edc32bdbd6b792eb83b6daf11e14708232f1f4a9e06153ebebd5bad1caef9b35"
EXPECTED_CACHE_SHA256 = "d9816083760d698935b44e9165522e709b490c3af29e710d75b8b0d2d90210e0"
EXPECTED_DEV_POOL_SHA256 = "533567e1d74544db7087a95253e640ef0959b998e5d503f57b0eb8074d8fe970"
EXPECTED_SPLIT_SHA256 = "7da471d2793fe3369508200dec50422bb76492860c67a1a61871645471856772"
EXPECTED_AGE_AUDIT_SHA256 = "84bbced4d6dbedd451f7fabacf11b721f1f1a8539d6a85e2f7ea6baa32613b39"
EXPECTED_M0_HASH = "28e6288c3f6769c2d5275aa77ddb248735b55a09cd46a6b570e3f117890b268b"

MEMORY_AGES = [2, 3, 4, 5]
MAX_TOKENS_PER_AGE = 8
SCORE_THRESHOLD = 0.20


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
# 1. CANDIDATE 2M MODULE ARCHITECTURE (SEAM 2: MEMORY-SEEDED QUERIES)
# ------------------------------------------------------------------------------

class HistoricalObjectTokenMemory(nn.Module):
    def __init__(self, d_model=256, num_classes=7, max_age=6):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes

        self.cls_proj = nn.Linear(num_classes, 64)
        self.age_embed = nn.Embedding(max_age + 1, 64)
        self.token_proj = nn.Linear(d_model + 64 + 1 + 64, d_model)
        self.pos_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

        self.gamma = nn.Parameter(torch.tensor(-4.0, dtype=torch.float32))

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
        tgt_mem = self.norm(self.token_proj(concat_feat))

        inv_ref = inverse_sigmoid(ref_coords)
        pos_proj_in = torch.cat([inv_ref, inv_ref.new_zeros((K, self.d_model - 2))], dim=-1)
        pos_mem = self.pos_proj(pos_proj_in)

        return tgt_mem, pos_mem, ref_coords


# ------------------------------------------------------------------------------
# 2. SOURCE HOOKS & ASYMMETRIC ATTENTION MASKING (P-1675, D-3582)
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
# 3. VERIFICATION HARNESS (P-1682 THROUGH P-1689)
# ------------------------------------------------------------------------------

def main():
    print("=" * 115)
    print("CANDIDATE 2M REAL-DETECTOR STRUCTURAL INTEGRATION & PARITY HARNESS")
    print("Locked Temporal Scope: k in {2, 3, 4, 5} | Seam 2: Memory-Seeded Temporal Queries")
    print("=" * 115)

    # 1. Full Live Provenance Verification
    assert CANDIDATES_JSON.exists(), f"[FATAL] Missing {CANDIDATES_JSON}"
    assert TRAIN_JSON.exists(), f"[FATAL] Missing {TRAIN_JSON}"
    assert CACHE_FILE.exists(), f"[FATAL] Missing {CACHE_FILE}"
    assert DEV_POOL_JSON.exists(), f"[FATAL] Missing {DEV_POOL_JSON}"
    assert SPLIT_MANIFEST_JSON.exists(), f"[FATAL] Missing {SPLIT_MANIFEST_JSON}"
    assert AGE_AUDIT_JSON.exists(), f"[FATAL] Missing {AGE_AUDIT_JSON}"
    assert M0_CKPT_PATH.exists(), f"[FATAL] Missing {M0_CKPT_PATH}"

    assert compute_sha256(CANDIDATES_JSON) == EXPECTED_CANDIDATE_SHA256, "[FATAL] Candidate JSON SHA mismatch!"
    assert compute_sha256(TRAIN_JSON) == EXPECTED_TRAIN_JSON_SHA256, "[FATAL] Train JSON SHA mismatch!"
    assert compute_sha256(CACHE_FILE) == EXPECTED_CACHE_SHA256, "[FATAL] Prediction cache SHA mismatch!"
    assert compute_sha256(DEV_POOL_JSON) == EXPECTED_DEV_POOL_SHA256, "[FATAL] Dev pool manifest SHA mismatch!"
    assert compute_sha256(SPLIT_MANIFEST_JSON) == EXPECTED_SPLIT_SHA256, "[FATAL] Split manifest SHA mismatch!"
    assert compute_sha256(AGE_AUDIT_JSON) == EXPECTED_AGE_AUDIT_SHA256, "[FATAL] Age audit manifest SHA mismatch!"
    assert compute_sha256(M0_CKPT_PATH) == EXPECTED_M0_HASH, "[FATAL] M0 checkpoint SHA mismatch!"

    print("[+] All 7 Upstream Source Manifests & M0 Checkpoint Certified by Immutable SHA-256.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device} ({torch.cuda.get_device_name(0)})")

    # Load dataset & COCO structures
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
    cat_to_label = {v: k for k, v in label_to_cat_id.items()}

    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache_data = json.load(f)
    raw_detections = {int(k): v for k, v in cache_data["detections"].items()}

    # Locate genuine Memory-Incremental window from video_4 (P-1685)
    with open(CANDIDATES_JSON, "r", encoding="utf-8") as f:
        cand_windows = json.load(f)["windows"]

    v4_mi_windows = [
        w for w in cand_windows
        if w["video_id"] == 4 and w.get("n_mem_incr_s20", w.get("n_memincr_s20", 0)) >= 1
    ]
    assert len(v4_mi_windows) >= 1, "[FATAL] No Memory-Incremental windows found in video_4!"
    target_window = v4_mi_windows[0]

    curr_target_img_id = target_window["image_ids"][-1]
    s_curr = collate_fn([dataset[id_to_idx[curr_target_img_id]]])
    samples = s_curr[0].to(device)

    # --------------------------------------------------------------------------
    # P-1682: PRISTINE M0 BASELINE RUN (BEFORE APPLYING ANY CLASS MONKEY-PATCHES)
    # --------------------------------------------------------------------------
    print("\n--- PRISTINE M0 REFERENCE GENERATION (BEFORE APPLYING 2M HOOKS) (P-1682) ---")
    args = get_authoritative_r0_args()
    if hasattr(args, "conditioning_mode"):
        args.conditioning_mode = "none"

    model_pristine, _, _ = build(args)
    ckpt = torch.load(M0_CKPT_PATH, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model_pristine.load_state_dict(state_dict, strict=True)
    model_pristine.to(device)
    for p in model_pristine.parameters():
        p.requires_grad = False
    model_pristine.eval()

    with torch.no_grad():
        out_pristine = model_pristine(samples)

    del model_pristine
    torch.cuda.empty_cache()
    print("[+] Pristine M0 reference output computed on genuine target frame and stored.")

    # --------------------------------------------------------------------------
    # APPLY 2M SOURCE HOOKS & BUILD CANDIDATE 2M MODEL
    # --------------------------------------------------------------------------
    apply_candidate2m_source_hooks()

    model, criterion, postprocessors = build(args)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    criterion.to(device)

    adapter = HistoricalObjectTokenMemory(d_model=256, num_classes=7, max_age=6).to(device)
    n_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    print(f"[+] Candidate 2M Model Loaded with Source Hooks: Adapter Parameters = {n_params:,}")

    # --------------------------------------------------------------------------
    # CONTRACT 1: EXACT PRISTINE M0 TEMPORAL-OFF BYPASS (P-1682)
    # --------------------------------------------------------------------------
    print("\n--- CONTRACT 1: EXACT PRISTINE M0 TEMPORAL-OFF BYPASS (P-1682) ---")
    with torch.no_grad():
        out_bypass = model(samples, temporal_adapter=adapter, memory_bank=None, temporal_enabled=False)

    logits_diff = (out_pristine["pred_logits"] - out_bypass["pred_logits"]).abs().max().item()
    boxes_diff = (out_pristine["pred_boxes"] - out_bypass["pred_boxes"]).abs().max().item()

    print(f"    - Max Absolute Logits Delta vs Pristine M0: {logits_diff:.10e}")
    print(f"    - Max Absolute Boxes Delta  vs Pristine M0: {boxes_diff:.10e}")
    assert logits_diff == 0.0 and boxes_diff == 0.0, (
        f"[FATAL] Contract 1 Failed: Patched temporal-off Candidate 2M diverged from pristine unmodified M0!"
    )
    print("[+] Contract 1 PASSED: Bitwise exact identity against pristine unmodified M0 verified.")

    # --------------------------------------------------------------------------
    # CONTRACT 4: DYNAMIC FIFO EXTRACTION & LIVE-VS-CACHE EQUIVALENCE (P-1683, P-1687)
    # --------------------------------------------------------------------------
    print("\n--- CONTRACT 4: STREAMING FIFO EXTRACTION & CACHE EQUIVALENCE (P-1683, P-1687) ---")
    window_frame_ids = target_window["frame_ids"]
    window_image_ids = target_window["image_ids"]
    assert len(window_frame_ids) == 6

    # Verify consecutive stream
    for idx in range(5):
        assert window_frame_ids[idx + 1] - window_frame_ids[idx] == 1, (
            f"[FATAL] Non-consecutive frames in window: {window_frame_ids[idx]} -> {window_frame_ids[idx+1]}"
        )

    fifo_buffer = deque(maxlen=5)

    with torch.no_grad():
        for f_idx in range(5):
            iid_f = window_image_ids[f_idx]
            fid_f = window_frame_ids[f_idx]
            s_f = collate_fn([dataset[id_to_idx[iid_f]]])
            out_f, dec_tokens = model(s_f[0].to(device), temporal_enabled=False, return_decoder_tokens=True)

            # P-1683: Canonical Top-100 DETR ranking
            prob_f = out_f["pred_logits"][0].sigmoid()
            num_classes = prob_f.shape[-1]
            topk_scores, topk_idx = torch.topk(prob_f.flatten(), 100)
            query_indices = torch.div(topk_idx, num_classes, rounding_mode="floor")
            class_labels = topk_idx % num_classes
            boxes_ranked = out_f["pred_boxes"][0][query_indices]

            # Unnormalize coordinates to raw pixels for cache comparison
            img_h, img_w = s_f[1][0]["orig_size"][0].item(), s_f[1][0]["orig_size"][1].item()
            live_cx = boxes_ranked[:, 0] * img_w
            live_cy = boxes_ranked[:, 1] * img_h
            live_rw = boxes_ranked[:, 2] * img_w
            live_rh = boxes_ranked[:, 3] * img_h
            live_x1 = live_cx - live_rw / 2.0
            live_y1 = live_cy - live_rh / 2.0

            # P-1687: Assert equivalence against frozen cache on each historical frame
            cached_preds = raw_detections.get(iid_f, [])
            n_check = min(8, len(cached_preds))
            for p_rank in range(n_check):
                live_sc = topk_scores[p_rank].item()
                cached_sc = cached_preds[p_rank]["score"]
                assert abs(live_sc - cached_sc) < 1e-4, f"[FATAL] Score mismatch on frame {fid_f} rank {p_rank}: {live_sc} vs {cached_sc}"

                live_cat = label_to_cat_id[class_labels[p_rank].item()]
                cached_cat = cached_preds[p_rank]["category_id"]
                assert live_cat == cached_cat, f"[FATAL] Category mismatch on frame {fid_f} rank {p_rank}: {live_cat} vs {cached_cat}"

                live_box_raw = [live_x1[p_rank].item(), live_y1[p_rank].item(), live_rw[p_rank].item(), live_rh[p_rank].item()]
                cached_box_raw = cached_preds[p_rank]["bbox"]
                iou_c = box_iou(live_box_raw, cached_box_raw)
                assert iou_c > 0.99, f"[FATAL] Bbox mismatch on frame {fid_f} rank {p_rank}: IoU={iou_c:.6f}"

            frame_tokens = []
            for p_rank in range(100):
                sc = topk_scores[p_rank].item()
                if sc >= SCORE_THRESHOLD:
                    q_idx = query_indices[p_rank].item()
                    frame_tokens.append({
                        "z": dec_tokens[0, q_idx].cpu(),
                        "bbox": boxes_ranked[p_rank].cpu().numpy().tolist(),
                        "logits": out_f["pred_logits"][0, q_idx].cpu(),
                        "score": sc,
                        "category_idx": class_labels[p_rank].item(),
                        "query_idx": q_idx,
                        "frame_id": fid_f
                    })
                    if len(frame_tokens) >= MAX_TOKENS_PER_AGE:
                        break

            fifo_buffer.append((fid_f, frame_tokens))

    print(f"[+] Live-vs-Frozen Cache Equivalence Certified on all 5 historical frames (P-1687).")
    print(f"    Every live Top-8 prediction matches cache score (<1e-4), category, and raw box (IoU > 0.99).")

    curr_fid = window_frame_ids[-1]
    memory_bank_extracted = []
    extracted_slot_ages = []

    for hist_fid, f_tokens in fifo_buffer:
        age = curr_fid - hist_fid
        assert 1 <= age <= 5, f"[FATAL] Invalid age {age} encountered in FIFO buffer!"
        if age in MEMORY_AGES:
            extracted_slot_ages.append(age)
            for tok in f_tokens:
                t_c = copy.deepcopy(tok)
                t_c["age"] = age
                memory_bank_extracted.append(t_c)

    unique_extracted_ages = sorted(list(set(extracted_slot_ages)))
    assert unique_extracted_ages == MEMORY_AGES, f"[FATAL] FIFO failed to certify slots {MEMORY_AGES}, got {unique_extracted_ages}"
    print(f"    - Extracted Target Slots: Certified ages {unique_extracted_ages} (k in {{2, 3, 4, 5}})")
    print(f"    - Extracted Memory Tokens: {len(memory_bank_extracted)} tokens (s >= 0.20, Top-8 per age)")
    print("[+] Contract 4 PASSED: Dynamic cache-compatible FIFO extraction across k in {2..5} verified.")

    # --------------------------------------------------------------------------
    # CONTRACT 2: ASYMMETRIC DECODER ATTENTION ISOLATION (D-3582)
    # --------------------------------------------------------------------------
    print("\n--- CONTRACT 2: ASYMMETRIC DECODER ATTENTION ISOLATION (D-3582) ---")
    with torch.no_grad():
        out_augmented = model(samples, temporal_adapter=adapter, memory_bank=[memory_bank_extracted], temporal_enabled=True)

    canon_logits_delta = (out_pristine["pred_logits"][:, :300, :] - out_augmented["pred_logits"][:, :300, :]).abs().max().item()
    canon_boxes_delta = (out_pristine["pred_boxes"][:, :300, :] - out_augmented["pred_boxes"][:, :300, :]).abs().max().item()

    print(f"    - Canonical Logits Delta under {len(memory_bank_extracted)} Temporal Queries: {canon_logits_delta:.10e}")
    print(f"    - Canonical Boxes Delta  under {len(memory_bank_extracted)} Temporal Queries: {canon_boxes_delta:.10e}")
    assert canon_logits_delta < 1e-4 and canon_boxes_delta < 1e-4, (
        f"[FATAL] Contract 2 Failed: Asymmetric mask leaked temporal information into canonical queries!"
    )
    print("[+] Contract 2 PASSED: Block-triangular self-attention mask mathematically isolates canonical queries.")

    # --------------------------------------------------------------------------
    # CONTRACT 3: ZERO-GATE LOGIT SUPPRESSION PARITY & IDENTITY SET MATCH (P-1689)
    # --------------------------------------------------------------------------
    print("\n--- CONTRACT 3: ZERO-GATE LOGIT SUPPRESSION PARITY & IDENTITY MATCH (P-1689) ---")
    adapter.gamma.data.fill_(-10.0)
    adapter.eval()

    with torch.no_grad():
        out_gated = model(samples, temporal_adapter=adapter, memory_bank=[memory_bank_extracted], temporal_enabled=True)

    prob_base = out_pristine["pred_logits"][0].sigmoid().flatten()
    topk_base_scores, topk_base_idx = torch.topk(prob_base, 100)
    base_q_idx = torch.div(topk_base_idx, 7, rounding_mode="floor")
    base_c_idx = topk_base_idx % 7

    prob_gated = out_gated["pred_logits"][0].sigmoid().flatten()
    topk_gated_scores, topk_gated_idx = torch.topk(prob_gated, 100)
    gated_q_idx = torch.div(topk_gated_idx, 7, rounding_mode="floor")
    gated_c_idx = topk_gated_idx % 7

    temporal_queries_in_top100 = (gated_q_idx >= 300).sum().item()
    max_score_discrepancy = (topk_base_scores - topk_gated_scores).abs().max().item()

    # Compare (query_idx, class_idx) detection identity sets (P-1689)
    base_detection_pairs = set(zip(base_q_idx.tolist(), base_c_idx.tolist()))
    gated_detection_pairs = set(zip(gated_q_idx.tolist(), gated_c_idx.tolist()))
    identity_diff = len(base_detection_pairs.symmetric_difference(gated_detection_pairs))

    print(f"    - Suppressed Temporal Queries entering Top-100: {temporal_queries_in_top100} / 100")
    print(f"    - Top-100 Score Discrepancy against Pristine M0: {max_score_discrepancy:.10e}")
    print(f"    - Top-100 Detection Pair Identity Discrepancies: {identity_diff} / 100")
    assert temporal_queries_in_top100 == 0, "[FATAL] Suppressed temporal queries leaked into Top-100 detections!"
    assert max_score_discrepancy < 1e-4, f"[FATAL] Top-100 detection scores diverged: {max_score_discrepancy}"
    assert identity_diff == 0, f"[FATAL] Top-100 detection identities diverged under zero gate: {identity_diff} mismatches!"
    print("[+] Contract 3 PASSED: Zero-gate confidence modulation guarantees exact Top-100 detection identity match.")

    # --------------------------------------------------------------------------
    # CONTRACT 5: SHAPE-MATCHED STRUCTURAL SENSITIVITY (P-1684)
    # --------------------------------------------------------------------------
    print("\n--- CONTRACT 5: SHAPE-MATCHED STRUCTURAL SENSITIVITY (P-1684) ---")
    adapter.gamma.data.fill_(0.0)

    # Branch absence: verify empty memory bank produces exactly 300 queries
    with torch.no_grad():
        out_null = model(samples, temporal_adapter=adapter, memory_bank=[], temporal_enabled=True)
    assert out_null["pred_logits"].shape[1] == 300, f"[FATAL] Empty memory bank failed to yield 300 queries!"
    print(f"    - Null Memory Branch: Verified output query count = 300 canonical queries")

    # Shape-matched sensitivity: compare against equal-cardinality synthetically perturbed bank
    perturbed_memory_bank = []
    for tok in memory_bank_extracted:
        t_p = copy.deepcopy(tok)
        t_p["bbox"] = [(t_p["bbox"][0] + 0.2) % 1.0, (t_p["bbox"][1] + 0.2) % 1.0, t_p["bbox"][2], t_p["bbox"][3]]
        perturbed_memory_bank.append(t_p)

    with torch.no_grad():
        out_corr = model(samples, temporal_adapter=adapter, memory_bank=[memory_bank_extracted], temporal_enabled=True)
        out_pert = model(samples, temporal_adapter=adapter, memory_bank=[perturbed_memory_bank], temporal_enabled=True)

    K_mem = len(memory_bank_extracted)
    assert out_corr["pred_logits"].shape[1] == 300 + K_mem
    assert out_pert["pred_logits"].shape[1] == 300 + K_mem

    diff_boxes = (out_corr["pred_boxes"][:, 300:, :] - out_pert["pred_boxes"][:, 300:, :]).abs().sum().item()
    diff_logits = (out_corr["pred_logits"][:, 300:, :] - out_pert["pred_logits"][:, 300:, :]).abs().sum().item()

    print(f"    - Augmented Output Shape: {out_corr['pred_logits'].shape} (300 Canonical + {K_mem} Temporal)")
    print(f"    - Sensitivity to Spatial Perturbation ||M_corr - M_pert||_boxes:  {diff_boxes:.4f}")
    print(f"    - Sensitivity to Spatial Perturbation ||M_corr - M_pert||_logits: {diff_logits:.4f}")
    assert diff_boxes > 0.01, "[FATAL] Temporal branch is invariant to memory coordinate perturbation!"
    print("[+] Contract 5 PASSED: Model exhibits verified sensitivity to historical token geometry.")

    # --------------------------------------------------------------------------
    # CONTRACT 6: TARGET-SPECIFIC HUNGARIAN AUDIT & GRADIENT FLOW (P-1685, P-1686)
    # --------------------------------------------------------------------------
    print("\n--- CONTRACT 6: MI-TARGET HUNGARIAN AUDIT & GRADIENT FLOW (P-1685, P-1686) ---")
    adapter.train()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4)
    optimizer.zero_grad()

    out_train = model(samples, temporal_adapter=adapter, memory_bank=[memory_bank_extracted], temporal_enabled=True)

    tgt_dict = {
        "boxes": s_curr[1][0]["boxes"].to(device),
        "labels": s_curr[1][0]["labels"].to(device)
    }

    # P-1685: Map actual Memory-Incremental target(s) from target_window to GT index
    mi_targets = [t for t in target_window["targets"] if t.get("mem_incr_s20", t.get("memincr_s20", False))]
    assert len(mi_targets) >= 1, "[FATAL] No MI targets found in target_window!"

    img_h, img_w = s_curr[1][0]["orig_size"][0].item(), s_curr[1][0]["orig_size"][1].item()
    gt_boxes_norm = tgt_dict["boxes"]

    mi_gt_indices = {}
    for t_mi in mi_targets:
        aid = t_mi["ann_id"]
        ann = ann_lookup[aid]
        bx, by, bw, bh = ann["bbox"]
        cx_norm = (bx + bw / 2.0) / img_w
        cy_norm = (by + bh / 2.0) / img_h
        w_norm = bw / img_w
        h_norm = bh / img_h
        t_box_norm = torch.tensor([cx_norm, cy_norm, w_norm, h_norm], device=gt_boxes_norm.device)
        diffs = (gt_boxes_norm - t_box_norm).abs().sum(dim=-1)
        gt_idx = diffs.argmin().item()
        assert diffs[gt_idx].item() < 1e-3, f"[FATAL] Failed to match MI target {aid} to ground truth box!"
        mi_gt_indices[aid] = gt_idx

    # Inspect Hungarian matcher assignments
    with torch.no_grad():
        matches = criterion.matcher(out_train, [tgt_dict])
        src_indices, tgt_indices = matches[0]
        n_gt_objects = len(tgt_indices)
        matched_canon = (src_indices < 300).sum().item()
        matched_temporal = (src_indices >= 300).sum().item()

    print(f"    [*] Hungarian Matcher Assignment Audit on video_4 Frame {curr_fid}:")
    print(f"        - Total Ground Truth Objects: {n_gt_objects}")
    print(f"        - Matched to Canonical Queries (0..299): {matched_canon} / {n_gt_objects}")
    print(f"        - Matched to Temporal Queries  (300..{300+K_mem-1}): {matched_temporal} / {n_gt_objects}")

    tgt_indices_list = tgt_indices.tolist()
    src_indices_list = src_indices.tolist()
    mi_target_matched_temporal = False

    print(f"\n    [*] Memory-Incremental Target Assignment Tracking (P-1685):")
    for aid, gt_idx in mi_gt_indices.items():
        ann = ann_lookup[aid]
        if gt_idx in tgt_indices_list:
            pos_m = tgt_indices_list.index(gt_idx)
            q_matched = src_indices_list[pos_m]
            is_temp = (q_matched >= 300)
            if is_temp:
                mi_target_matched_temporal = True
            branch_lbl = f"TEMPORAL QUERY {q_matched} (memory slot {q_matched - 300})" if is_temp else f"CANONICAL QUERY {q_matched}"
            print(f"        • MI Target {aid} (GT idx {gt_idx}, category {cats[ann['category_id']]}):")
            print(f"          --> Assigned to {branch_lbl}")
        else:
            print(f"        • MI Target {aid} (GT idx {gt_idx}): UNMATCHED")

    # Evaluate predeclared structural evidence tier (Q-1829)
    if mi_target_matched_temporal:
        print("        [+] STRUCTURAL VERDICT: STRONG EVIDENCE (>= 1 genuine MI target matched to a temporal query at initialization).")
    elif matched_temporal > 0:
        print("        [*] STRUCTURAL VERDICT: WEAK EVIDENCE (Temporal queries received positive assignments, but MI target was assigned to canonical query).")
    else:
        print("        [!] STRUCTURAL VERDICT: WARNING (Zero temporal queries received positive assignments at initialization).")

    loss_dict = criterion(out_train, [tgt_dict])
    total_loss = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict if k in criterion.weight_dict)
    total_loss.backward()

    # P-1686: Assert complete finite gradient flow
    g_token_proj = adapter.token_proj.weight.grad
    g_pos_proj = adapter.pos_proj.weight.grad
    g_cls_proj = adapter.cls_proj.weight.grad
    g_age_embed = adapter.age_embed.weight.grad
    g_norm_w = adapter.norm.weight.grad
    g_gamma = adapter.gamma.grad

    print(f"\n    [*] Adapter Gradient Magnitudes (Complete Flow Verification):")
    print(f"        - Task Loss:                  {total_loss.item():.4f}")
    print(f"        - ||dL/d_token_proj||:        {g_token_proj.norm().item():.6f}")
    print(f"        - ||dL/d_pos_proj||:          {g_pos_proj.norm().item():.6f}")
    print(f"        - ||dL/d_cls_proj||:          {g_cls_proj.norm().item():.6f}")
    print(f"        - ||dL/d_age_embed||:         {g_age_embed.norm().item():.6f}")
    print(f"        - ||dL/d_norm_w||:            {g_norm_w.norm().item():.6f}")
    print(f"        - dL/d_gamma:                 {g_gamma.item():+8.6f}")

    assert g_token_proj is not None and torch.isfinite(g_token_proj).all() and g_token_proj.norm().item() > 0.0, "[FATAL] Invalid gradient on token_proj!"
    assert g_pos_proj is not None and torch.isfinite(g_pos_proj).all() and g_pos_proj.norm().item() > 0.0, "[FATAL] Invalid gradient on pos_proj!"
    assert g_cls_proj is not None and torch.isfinite(g_cls_proj).all() and g_cls_proj.norm().item() > 0.0, "[FATAL] Invalid gradient on cls_proj!"
    assert g_age_embed is not None and torch.isfinite(g_age_embed).all() and g_age_embed.norm().item() > 0.0, "[FATAL] Invalid gradient on age_embed!"
    assert g_norm_w is not None and torch.isfinite(g_norm_w).all() and g_norm_w.norm().item() > 0.0, "[FATAL] Invalid gradient on norm!"
    assert g_gamma is not None and torch.isfinite(g_gamma).all() and abs(g_gamma.item()) > 0.0, "[FATAL] Invalid gradient on gamma!"

    # Assert baseline M0 parameters received zero gradients
    m0_grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(m0_grads) == 0, "[FATAL] Baseline M0 parameters received gradients!"
    print("[+] Contract 6 PASSED: Hungarian loss propagates through all adapter modules while M0 remains strictly frozen.")

    print("\n" + "=" * 115)
    print("ALL 6 STRUCTURAL INTEGRATION CONTRACTS PASSED CLEANLY (CANDIDATE 2M STRUCTURAL INTEGRATION CONTRACTS PASSED)")
    print("=" * 115)


if __name__ == "__main__":
    main()