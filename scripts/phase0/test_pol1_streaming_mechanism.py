"""
D:/MARIVISION/H0_workbench/eval_tools/test_pol1_streaming_mechanism.py
Phase-II Proof-of-Launch Stage 1 (PoL-1): Real Structural Causal-Memory Verification

Strict Compliance Checklist:
  - D-3308 / D-3309: Controlled repeatability diagnostics (canonical repeat, patched-off repeat, direct tgt delta).
  - D-3310: Explicit adapter.eval() before static-parity tests (no training-mode dropout).
  - D-3311: temporal_enabled=False as exact bitwise static baseline contract.
  - D-3312 / P-1545: Real sequential smoke test uses active gamma=0.5 in eval mode to ensure state consumption.
  - D-3291 / D-3292: Independent verification of F_ONLY, M_ONLY, and F_PLUS_M with real criterion loss gradients.
  - D-3283: Deterministic token-bank retention, aging, sentinel expiry, and boundary reset.
  - D-3284 / D-3295: Clean single-call criterion profiling for BS=1 and authoritative BS=2.
"""

import os
import sys
import copy
import json
from PIL import Image
import hashlib
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

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

ZIP_PATH = Path("D:/MARIVISION/SeaClips.zip")
TRAIN_JSON = Path("D:/MARIVISION/seaclips-train.json")
M0_CKPT_PATH = Path("D:/MARIVISION/checkpoints_r0/checkpoint_best.pth")
EXPECTED_M0_HASH = "28e6288c3f6769c2d5275aa77ddb248735b55a09cd46a6b570e3f117890b268b"


def compute_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------
# 1. RECURSIVE TEMPORAL STATE & ADAPTER
# ------------------------------------------------------------------------------

class RecursiveTemporalState:
    """Bounded, recursive multi-frame temporal state container (D-3272, D-3273)."""
    def __init__(self, batch_size=1, capacity_k=50, d_model=256, num_classes=7, max_age=4, device="cpu"):
        self.B = batch_size
        self.K = capacity_k
        self.D = d_model
        self.C = num_classes
        self.max_age = max_age
        self.device = device
        self.reset()

    def reset(self):
        self.feat_cache = None
        self.mem_tokens = torch.zeros((self.B, self.K, self.D), dtype=torch.float32, device=self.device)
        self.mem_boxes = torch.zeros((self.B, self.K, 4), dtype=torch.float32, device=self.device)
        self.mem_logits = torch.zeros((self.B, self.K, self.C), dtype=torch.float32, device=self.device)
        self.mem_scores = torch.zeros((self.B, self.K), dtype=torch.float32, device=self.device)
        self.mem_age = torch.zeros((self.B, self.K), dtype=torch.long, device=self.device)
        self.mem_mask = torch.zeros((self.B, self.K), dtype=torch.bool, device=self.device)

    def recursive_update(self, new_tokens, new_boxes, new_logits, new_scores):
        B, N_new, D = new_tokens.shape
        self.mem_age = torch.where(self.mem_mask, self.mem_age + 1, self.mem_age)
        self.mem_mask = self.mem_mask & (self.mem_age <= self.max_age)

        new_tok_d = new_tokens.detach()
        new_box_d = new_boxes.detach()
        new_log_d = new_logits.detach()
        new_sco_d = new_scores.detach()

        comb_tokens = torch.cat([self.mem_tokens, new_tok_d], dim=1)
        comb_boxes = torch.cat([self.mem_boxes, new_box_d], dim=1)
        comb_logits = torch.cat([self.mem_logits, new_log_d], dim=1)
        comb_scores = torch.cat([self.mem_scores, new_sco_d], dim=1)
        comb_age = torch.cat([self.mem_age, torch.zeros((B, N_new), dtype=torch.long, device=self.device)], dim=1)
        comb_mask = torch.cat([self.mem_mask, torch.ones((B, N_new), dtype=torch.bool, device=self.device)], dim=1)

        masked_scores = torch.where(comb_mask, comb_scores, torch.full_like(comb_scores, -1e4))
        _, topk_idxs = torch.topk(masked_scores, k=min(self.K, comb_scores.shape[1]), dim=1)

        b_idx = torch.arange(B, device=self.device).unsqueeze(-1)
        self.mem_tokens = comb_tokens[b_idx, topk_idxs]
        self.mem_boxes = comb_boxes[b_idx, topk_idxs]
        self.mem_logits = comb_logits[b_idx, topk_idxs]
        self.mem_scores = comb_scores[b_idx, topk_idxs]
        self.mem_age = comb_age[b_idx, topk_idxs]
        self.mem_mask = comb_mask[b_idx, topk_idxs] & (self.mem_scores > -1e3)


class TemporalQueryAdapter(nn.Module):
    def __init__(self, d_model=256, nheads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Branch A: Memory tokens M_{t-1}
        self.mem_cross_attn = nn.MultiheadAttention(d_model, nheads, dropout=dropout, batch_first=True)
        self.norm_mem = nn.LayerNorm(d_model)
        
        # Branch B: Cached multiscale features F_{t-1}^cache
        self.spatial_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.feat_proj = nn.Linear(d_model, d_model)
        self.feat_cross_attn = nn.MultiheadAttention(d_model, nheads, dropout=dropout, batch_first=True)
        self.norm_feat = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm_out = nn.LayerNorm(d_model)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # Diagnostic tracker for P-1544
        self.last_direct_tgt_delta = 0.0

    def forward(self, tgt, query_pos, temporal_state, temporal_enabled=True):
        if not temporal_enabled or temporal_state is None:
            return tgt

        delta_q = torch.zeros_like(tgt)
        has_signal = False

        if torch.any(temporal_state.mem_mask):
            key_pad_mask = ~temporal_state.mem_mask
            if not torch.all(key_pad_mask):
                mem_out, _ = self.mem_cross_attn(
                    query=tgt, key=temporal_state.mem_tokens, value=temporal_state.mem_tokens,
                    key_padding_mask=key_pad_mask
                )
                delta_q = delta_q + self.norm_mem(mem_out)
                has_signal = True

        if temporal_state.feat_cache is not None and len(temporal_state.feat_cache) > 0:
            pooled_tokens = []
            for fmap in temporal_state.feat_cache:
                p = self.spatial_pool(fmap).flatten(2).permute(0, 2, 1)
                pooled_tokens.append(p)
            feat_tokens = self.feat_proj(torch.cat(pooled_tokens, dim=1))
            feat_out, _ = self.feat_cross_attn(query=tgt, key=feat_tokens, value=feat_tokens)
            delta_q = delta_q + self.norm_feat(feat_out)
            has_signal = True

        if not has_signal:
            return tgt

        fused = self.norm_out(delta_q + self.mlp(delta_q))
        gate = torch.tanh(self.gamma)
        
        # Record direct target perturbation before decoder (P-1544)
        output = tgt + gate * fused
        self.last_direct_tgt_delta = (output - tgt).abs().max().item()
        return output


# ------------------------------------------------------------------------------
# 2. SOURCE MONKEY-PATCHING
# ------------------------------------------------------------------------------

def apply_clean_source_hooks():
    def patched_transformer_forward(self, srcs, masks, pos_embeds, query_embed=None,
                                    temporal_adapter=None, temporal_state=None, temporal_enabled=True):
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
            tgt = temporal_adapter(tgt=tgt, query_pos=query_embed, temporal_state=temporal_state, temporal_enabled=temporal_enabled)

        # Native call with NO transposition (preserves B, num_queries, D contract)
        hs, inter_references = self.decoder(
            tgt, reference_points, memory,
            spatial_shapes, level_start_index, valid_ratios,
            query_embed, mask_flatten
        )
        return hs, init_reference_out, inter_references, None, None

    DeformableTransformer.forward = patched_transformer_forward

    def patched_detr_forward(self, samples: NestedTensor,
                             temporal_adapter=None, temporal_state=None, temporal_enabled=True,
                             return_streaming_state=False):
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
            temporal_state=temporal_state,
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

        if return_streaming_state:
            feat_cache = [s.detach() for s in srcs]
            decoder_tokens = hs[-1].detach()
            return out, feat_cache, decoder_tokens

        return out

    DeformableDETR.forward = patched_detr_forward


# ------------------------------------------------------------------------------
# 3. VERIFICATION PROTOCOL
# ------------------------------------------------------------------------------

def main():
    print("=" * 115)
    print("PROOF-OF-LAUNCH STAGE 1 (PoL-1): REAL STRUCTURAL CAUSAL-MEMORY VERIFICATION")
    print(f"Target Checkpoint: {M0_CKPT_PATH.resolve()}")
    print("=" * 115)

    assert M0_CKPT_PATH.exists(), f"[FATAL] Missing M0 checkpoint: {M0_CKPT_PATH}"
    actual_hash = compute_sha256(M0_CKPT_PATH)
    assert actual_hash == EXPECTED_M0_HASH, f"[FATAL] Checkpoint SHA mismatch! {actual_hash} != {EXPECTED_M0_HASH}"
    print(f"[+] Certified M0 Checkpoint Verified: SHA-256 = {actual_hash}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Target: {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[*] Physical Hardware: {gpu_name} (Installed VRAM: {vram_gb:.2f} GB)")

    # Build UNTOUCHED model and load certified M0 checkpoint (D-3288, D-3289)
    args = get_authoritative_r0_args()
    if hasattr(args, "conditioning_mode"):
        args.conditioning_mode = "none"

    model, criterion, postprocessors = build(args)
    ckpt = torch.load(M0_CKPT_PATH, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    criterion.to(device)
    print("[+] Model built and certified M0 weights loaded into memory.")

    # --------------------------------------------------------------------------
    # STEP 1: LOAD ADJACENT REAL TRAIN PAIR & CAPTURE REPEATABLE CANONICAL REFERENCE
    # --------------------------------------------------------------------------
    print("\n--- STEP 1: LOADING ADJACENT TRAIN PAIR & MEASURING CANONICAL REPEATABILITY ---")
    dataset = SeaClipsDetection(ZIP_PATH, TRAIN_JSON, is_train=False, random_flip=False)
    coco = dataset.coco

    img_ids = sorted(coco.getImgIds())
    pair_found = False
    idx_t1, idx_t = None, None

    for i in range(len(img_ids) - 1):
        id_a, id_b = img_ids[i], img_ids[i + 1]
        info_a, info_b = coco.loadImgs([id_a])[0], coco.loadImgs([id_b])[0]

        if info_a["video_id"] == info_b["video_id"] and info_a["frame_id"] + 1 == info_b["frame_id"]:
            anns_a = coco.loadAnns(coco.getAnnIds(imgIds=[id_a]))
            anns_b = coco.loadAnns(coco.getAnnIds(imgIds=[id_b]))
            if len(anns_a) > 0 and len(anns_b) > 0:
                idx_t1, idx_t = i, i + 1
                pair_found = True
                break

    assert pair_found, "[FATAL] Could not find consecutive annotated frames in train split!"

    batch_t1 = collate_fn([dataset[idx_t1]])
    batch_t = collate_fn([dataset[idx_t]])

    samples_t1, targets_t1 = batch_t1[0].to(device), [{k: v.to(device) for k, v in t.items()} for t in batch_t1[1]]
    samples_t, targets_t = batch_t[0].to(device), [{k: v.to(device) for k, v in t.items()} for t in batch_t[1]]

    info_t1 = coco.loadImgs([img_ids[idx_t1]])[0]
    info_t = coco.loadImgs([img_ids[idx_t]])[0]
    print(f"[+] Loaded Adjacent Pair: Video {info_t['video_id']} | Frame {info_t1['frame_id']} -> Frame {info_t['frame_id']}")

    # Measure Canonical Repeatability (P-1543): run canonical model twice on identical input
    with torch.no_grad():
        canonical_run1 = model(samples_t)
        canonical_run2 = model(samples_t)

        delta_canon_rep_logits = (canonical_run1["pred_logits"] - canonical_run2["pred_logits"]).abs().max().item()
        delta_canon_rep_boxes = (canonical_run1["pred_boxes"] - canonical_run2["pred_boxes"]).abs().max().item()

        canonical_golden = {
            "pred_logits": canonical_run1["pred_logits"].detach().clone(),
            "pred_boxes": canonical_run1["pred_boxes"].detach().clone(),
            "aux_outputs": [
                {
                    "pred_logits": aux["pred_logits"].detach().clone(),
                    "pred_boxes": aux["pred_boxes"].detach().clone()
                }
                for aux in canonical_run1["aux_outputs"]
            ]
        }
    print(f"[+] Canonical Repeatability (Run 1 vs Run 2):")
    print(f"    - Delta Logits: {delta_canon_rep_logits:.10f}")
    print(f"    - Delta Boxes:  {delta_canon_rep_boxes:.10f}")

    # Apply surgical hooks now that canonical reference is frozen
    apply_clean_source_hooks()
    print("[+] Clean source hooks installed into DeformableTransformer and DeformableDETR.")

    adapter = TemporalQueryAdapter(d_model=args.hidden_dim, nheads=args.nheads).to(device)
    # D-3310: Explicitly set adapter to eval mode!
    adapter.eval()
    state = RecursiveTemporalState(batch_size=1, capacity_k=50, d_model=args.hidden_dim, device=device)
    print(f"[+] TemporalQueryAdapter initialized and set to EVAL mode: adapter.training = {adapter.training}")

    # --------------------------------------------------------------------------
    # STEP 2: CONTRACT 1 - PRE-PATCH VS. PATCHED STATIC PARITY (D-3281, D-3289, D-3309)
    # --------------------------------------------------------------------------
    print("\n--- STEP 2: VERIFYING CONTRACT 1 - CONTROLLED STATIC PARITY DIAGNOSTICS ---")
    with torch.no_grad():
        # Test 1A: Patched model with temporal_enabled=False run twice (repeatability)
        out_temp_off_1 = model(samples_t, temporal_adapter=adapter, temporal_state=state, temporal_enabled=False)
        out_temp_off_2 = model(samples_t, temporal_adapter=adapter, temporal_state=state, temporal_enabled=False)

        delta_off_rep_logits = (out_temp_off_1["pred_logits"] - out_temp_off_2["pred_logits"]).abs().max().item()
        delta_off_rep_boxes = (out_temp_off_1["pred_boxes"] - out_temp_off_2["pred_boxes"]).abs().max().item()
        delta_logits_off = (canonical_golden["pred_logits"] - out_temp_off_1["pred_logits"]).abs().max().item()
        delta_boxes_off = (canonical_golden["pred_boxes"] - out_temp_off_1["pred_boxes"]).abs().max().item()

        # Test 1B: Patched model with adapter in EVAL mode, active gate gamma = 0.0
        adapter.eval()
        adapter.gamma.data.fill_(0.0)
        state.mem_tokens = torch.randn(1, 50, args.hidden_dim, device=device)
        state.mem_mask = torch.ones(1, 50, dtype=torch.bool, device=device)
        
        out_temp_zero = model(samples_t, temporal_adapter=adapter, temporal_state=state, temporal_enabled=True)
        
        # P-1544: Direct adapter query delta before decoder layers
        direct_tgt_delta = adapter.last_direct_tgt_delta

        delta_logits_zero = (canonical_golden["pred_logits"] - out_temp_zero["pred_logits"]).abs().max().item()
        delta_boxes_zero = (canonical_golden["pred_boxes"] - out_temp_zero["pred_boxes"]).abs().max().item()

        aux_logits_max = 0.0
        aux_boxes_max = 0.0
        for lvl in range(len(canonical_golden["aux_outputs"])):
            diff_l = (canonical_golden["aux_outputs"][lvl]["pred_logits"] - out_temp_zero["aux_outputs"][lvl]["pred_logits"]).abs().max().item()
            diff_b = (canonical_golden["aux_outputs"][lvl]["pred_boxes"] - out_temp_zero["aux_outputs"][lvl]["pred_boxes"]).abs().max().item()
            aux_logits_max = max(aux_logits_max, diff_l)
            aux_boxes_max = max(aux_boxes_max, diff_b)

    print(f"[*] Controlled Numerical Diagnostic Results (P-1543, P-1544):")
    print(f"    1. Patched Temporal-Off Repeatability: Delta Logits={delta_off_rep_logits:.10f} | Delta Boxes={delta_off_rep_boxes:.10f}")
    print(f"    2. Patched Temporal-Off vs Canonical:  Delta Logits={delta_logits_off:.10f} | Delta Boxes={delta_boxes_off:.10f}")
    print(f"    3. Direct Adapter TGT Delta (gamma=0): Max |tgt_adapted - tgt| = {direct_tgt_delta:.10f}")
    print(f"    4. Final Head Zero-Gate vs Canonical:  Delta Logits={delta_logits_zero:.10f} | Delta Boxes={delta_boxes_zero:.10f}")
    print(f"    5. Aux Heads Max Zero-Gate Delta:     Delta Logits={aux_logits_max:.10f} | Delta Boxes={aux_boxes_max:.10f}")

    # Assert exact bitwise zero on the protected static bypass (D-3307, D-3311)
    assert delta_logits_off == 0.0 and delta_boxes_off == 0.0, (
        f"[FATAL] Discrepancy against pre-patch canonical when temporal disabled! "
        f"Logits={delta_logits_off}, Boxes={delta_boxes_off}"
    )
    print(f"[+] Contract 1A Certified: temporal_enabled=False is 100% bit-for-bit identical to canonical R0.")

    # Diagnostic assertion on gamma=0 path
    if delta_logits_zero == 0.0 and delta_boxes_zero == 0.0:
        print(f"[+] Contract 1B Certified: Zero-gate path is bit-for-bit identical to canonical R0.")
    else:
        print(f"[!] Contract 1B Numerical Observation: Zero-gate path exhibits delta: Logits={delta_logits_zero:.10e}, Boxes={delta_boxes_zero:.10e}")
        assert delta_logits_zero < 1e-4 and delta_boxes_zero < 1e-4, f"[FATAL] Zero-gate divergence exceeds safe threshold! Logits={delta_logits_zero}"

    # --------------------------------------------------------------------------
    # STEP 3: CONTRACT 2 - SINGLE-PASS CACHE CAPTURE (ZERO RE-ENCODING)
    # --------------------------------------------------------------------------
    print("\n--- STEP 3: VERIFYING CONTRACT 2 - SINGLE-PASS CACHE REUSE ---")
    backbone_calls = 0
    orig_bb = model.backbone.forward

    def hooked_bb(tensor_list):
        nonlocal backbone_calls
        backbone_calls += 1
        return orig_bb(tensor_list)

    model.backbone.forward = hooked_bb

    with torch.no_grad():
        out_t1, feat_cache_t1, tokens_t1 = model(samples_t1, return_streaming_state=True)

    assert backbone_calls == 1, f"[FATAL] Expected exactly 1 backbone call, got {backbone_calls}"
    assert len(feat_cache_t1) == args.num_feature_levels
    state.feat_cache = feat_cache_t1
    print(f"[+] Frame t-1 forward complete. Captured {len(state.feat_cache)} multiscale levels in a single pass.")

    with torch.no_grad():
        out_t, feat_cache_t, tokens_t = model(samples_t, temporal_adapter=adapter, temporal_state=state, temporal_enabled=True, return_streaming_state=True)

    assert backbone_calls == 2, f"[FATAL] Backbone called more than once per frame! Calls={backbone_calls}"
    print(f"[+] Contract 2 Certified: Previous frame features reused from cache with zero additional backbone calls.")
    model.backbone.forward = orig_bb

    # --------------------------------------------------------------------------
    # STEP 4: REAL CAUSAL SEQUENTIAL PLUMBING SMOKE TEST (D-3294, D-3312, P-1545)
    # --------------------------------------------------------------------------
    print("\n--- STEP 4: REAL CAUSAL SEQUENTIAL STATE PLUMBING SMOKE TEST (ACTIVE GATE) ---")
    # Populate memory bank with actual detector outputs from frame t-1
    scores_t1 = out_t1["pred_logits"][0].sigmoid().max(-1).values.unsqueeze(0)
    real_stream_state = RecursiveTemporalState(batch_size=1, capacity_k=50, d_model=args.hidden_dim, device=device)
    real_stream_state.recursive_update(
        new_tokens=tokens_t1,
        new_boxes=out_t1["pred_boxes"],
        new_logits=out_t1["pred_logits"],
        new_scores=scores_t1
    )
    real_stream_state.feat_cache = feat_cache_t1

    # D-3312: Explicitly set adapter to eval mode and activate gamma=0.5 to prove real state is consumed!
    adapter.eval()
    with torch.no_grad():
        adapter.gamma.data.fill_(0.5)
        out_real_stream = model(samples_t, temporal_adapter=adapter, temporal_state=real_stream_state, temporal_enabled=True)

    assert torch.isfinite(out_real_stream["pred_logits"]).all()
    assert torch.isfinite(out_real_stream["pred_boxes"]).all()
    delta_real_boxes = (out_real_stream["pred_boxes"] - canonical_golden["pred_boxes"]).abs().max().item()
    assert delta_real_boxes > 0.0, "[FATAL] Real state with active gamma failed to influence predictions!"
    print(f"[+] Real Causal Sequential State Plumbing Certified (D-3312):")
    print(f"    - Frame t-1 predictions committed to M_{{t-1}}: {real_stream_state.mem_mask.sum().item()} active tokens.")
    print(f"    - Frame t consumed real t-1 features + memory tokens under gamma=0.5 (Max Box Delta = {delta_real_boxes:.6f}).")

    # --------------------------------------------------------------------------
    # STEP 5: CONTRACTS 3 & 4 - INDEPENDENT F-ONLY, M-ONLY, AND F+M PATHS (D-3291, D-3292)
    # --------------------------------------------------------------------------
    print("\n--- STEP 5: VERIFYING CONTRACTS 3 & 4 - INDEPENDENT F-ONLY, M-ONLY, AND F+M PATHS ---")
    adapter.train()
    adapter.gamma.data.fill_(0.5)

    # Sub-test A: F_ONLY Path
    state_f_only = RecursiveTemporalState(batch_size=1, capacity_k=50, d_model=args.hidden_dim, device=device)
    state_f_only.feat_cache = [f.clone() for f in feat_cache_t1]
    state_f_only.mem_mask.fill_(False)

    model.zero_grad()
    adapter.zero_grad()
    out_f_only = model(samples_t, temporal_adapter=adapter, temporal_state=state_f_only, temporal_enabled=True)
    assert torch.isfinite(out_f_only["pred_logits"]).all() and torch.isfinite(out_f_only["pred_boxes"]).all()
    assert not torch.equal(out_f_only["pred_boxes"], canonical_golden["pred_boxes"]), "[FATAL] F_ONLY failed to alter predictions!"

    delta_boxes_f = (out_f_only["pred_boxes"] - canonical_golden["pred_boxes"]).abs().max().item()
    loss_dict_f = criterion(out_f_only, targets_t)
    loss_f = sum(loss_dict_f[k] * criterion.weight_dict[k] for k in loss_dict_f if k in criterion.weight_dict)
    loss_f.backward()

    feat_grad = adapter.feat_cross_attn.in_proj_weight.grad.norm().item()
    assert feat_grad > 0.0, "[FATAL] Zero gradient in feature cross-attention weights!"
    print(f"[+] Path 1 (F_ONLY) Certified:")
    print(f"    - Output Alteration: Max Box Delta = {delta_boxes_f:.6f}")
    print(f"    - Feature Cross-Attention Gradient Norm: {feat_grad:.6f}")

    # Sub-test B: M_ONLY Path
    state_m_only = RecursiveTemporalState(batch_size=1, capacity_k=50, d_model=args.hidden_dim, device=device)
    mem_tok_req = torch.randn(1, 50, args.hidden_dim, device=device, requires_grad=True)
    state_m_only.mem_tokens = mem_tok_req
    state_m_only.mem_mask.fill_(True)
    state_m_only.feat_cache = None

    model.zero_grad()
    adapter.zero_grad()
    out_m_only = model(samples_t, temporal_adapter=adapter, temporal_state=state_m_only, temporal_enabled=True)
    assert torch.isfinite(out_m_only["pred_logits"]).all() and torch.isfinite(out_m_only["pred_boxes"]).all()
    assert not torch.equal(out_m_only["pred_boxes"], canonical_golden["pred_boxes"]), "[FATAL] M_ONLY failed to alter predictions!"

    delta_boxes_m = (out_m_only["pred_boxes"] - canonical_golden["pred_boxes"]).abs().max().item()
    loss_dict_m = criterion(out_m_only, targets_t)
    loss_m = sum(loss_dict_m[k] * criterion.weight_dict[k] for k in loss_dict_m if k in criterion.weight_dict)
    loss_m.backward()

    mem_tok_grad = mem_tok_req.grad.norm().item()
    mem_attn_grad = adapter.mem_cross_attn.in_proj_weight.grad.norm().item()
    gamma_grad = adapter.gamma.grad.item()

    assert mem_tok_grad > 0.0, "[FATAL] Zero gradient flowing into memory tokens!"
    assert mem_attn_grad > 0.0, "[FATAL] Zero gradient in memory cross-attention weights!"
    assert abs(gamma_grad) > 0.0, "[FATAL] Zero gradient in gating parameter gamma!"
    print(f"[+] Path 2 (M_ONLY) Certified:")
    print(f"    - Output Alteration: Max Box Delta = {delta_boxes_m:.6f}")
    print(f"    - Memory Token Gradient Norm:          {mem_tok_grad:.6f}")
    print(f"    - Memory Cross-Attention Grad Norm:     {mem_attn_grad:.6f}")
    print(f"    - Gating Parameter (gamma) Grad:        {gamma_grad:.6f}")

    # Sub-test C: F_PLUS_M Path (Both active) (D-3291)
    state_fm = RecursiveTemporalState(batch_size=1, capacity_k=50, d_model=args.hidden_dim, device=device)
    state_fm.feat_cache = [f.clone() for f in feat_cache_t1]
    state_fm.mem_tokens = torch.randn(1, 50, args.hidden_dim, device=device, requires_grad=True)
    state_fm.mem_mask.fill_(True)

    model.zero_grad()
    adapter.zero_grad()
    out_fm = model(samples_t, temporal_adapter=adapter, temporal_state=state_fm, temporal_enabled=True)
    assert torch.isfinite(out_fm["pred_logits"]).all() and torch.isfinite(out_fm["pred_boxes"]).all()
    assert not torch.equal(out_fm["pred_boxes"], canonical_golden["pred_boxes"]), "[FATAL] F_PLUS_M failed to alter predictions!"

    delta_boxes_fm = (out_fm["pred_boxes"] - canonical_golden["pred_boxes"]).abs().max().item()
    loss_dict_fm = criterion(out_fm, targets_t)
    loss_fm = sum(loss_dict_fm[k] * criterion.weight_dict[k] for k in loss_dict_fm if k in criterion.weight_dict)
    loss_fm.backward()
    print(f"[+] Path 3 (F_PLUS_M) Certified:")
    print(f"    - Output Alteration: Max Box Delta = {delta_boxes_fm:.6f}")
    print(f"    - Combined Loss Backward Executed Cleanly (Loss = {loss_fm.item():.4f}).")

    # --------------------------------------------------------------------------
    # STEP 6: CONTRACT 5 - DETERMINISTIC TOKEN-BANK TEST WITH SENTINEL EXPIRY (D-3283)
    # --------------------------------------------------------------------------
    print("\n--- STEP 6: VERIFYING CONTRACT 5 - DETERMINISTIC RETENTION & SENTINEL EXPIRY ---")
    rec_state = RecursiveTemporalState(batch_size=1, capacity_k=10, d_model=256, max_age=3, device=device)

    # Step 0: Insert 5 tokens including a unique sentinel token at index 0
    toks_0 = torch.zeros(1, 5, 256, device=device)
    toks_0[0, 0, :] = 777.0  # Sentinel signature
    boxes_0 = torch.rand(1, 5, 4, device=device)
    logits_0 = torch.randn(1, 5, 7, device=device)
    scores_0 = torch.tensor([[0.999, 0.90, 0.85, 0.80, 0.75]], device=device)
    rec_state.recursive_update(toks_0, boxes_0, logits_0, scores_0)

    assert (rec_state.mem_tokens[0, 0, :] == 777.0).all()
    assert rec_state.mem_age[0, 0].item() == 0

    # Steps 1 to 3: Advance cycles with lower-scoring new tokens
    for cycle in range(1, 4):
        scores_c = torch.tensor([[0.70 - 0.05 * cycle] * 5], device=device)
        rec_state.recursive_update(torch.zeros(1, 5, 256, device=device), boxes_0, logits_0, scores_c)
        sentinel_idx = (rec_state.mem_tokens[0, :, 0] == 777.0).nonzero(as_tuple=True)[0]
        assert len(sentinel_idx) == 1
        assert rec_state.mem_age[0, sentinel_idx].item() == cycle

    # Step 4: Advance cycle -> sentinel age becomes 4 > max_age (3) -> EVICTED
    scores_4 = torch.tensor([[0.50, 0.50, 0.50, 0.50, 0.50]], device=device)
    rec_state.recursive_update(torch.zeros(1, 5, 256, device=device), boxes_0, logits_0, scores_4)
    sentinel_idx_post = (rec_state.mem_tokens[0, :, 0] == 777.0).nonzero(as_tuple=True)[0]
    assert len(sentinel_idx_post) == 0, "[FATAL] Sentinel token was not evicted after exceeding max_age=3!"
    print(f"[+] Contract 5 Certified: Sentinel token (777.0) survived across ages 0->1->2->3 and was evicted at age 4.")

    rec_state.reset()
    assert rec_state.feat_cache is None
    assert rec_state.mem_mask.sum().item() == 0
    print("[+] Boundary reset certified: All state tensors zeroed and cache cleared.")

    # --------------------------------------------------------------------------
    # STEP 7: CONTRACT 6 - CLEAN ISOLATED VRAM & LATENCY PROFILING (BS=1 AND BS=2)
    # --------------------------------------------------------------------------
    print("\n--- STEP 7: PROFILING BASELINE VS. TEMPORAL VRAM & LATENCY (BS=1 & BS=2) ---")
    if not torch.cuda.is_available():
        print("[!] CUDA unavailable; skipping hardware profile.")
        return

    batch_bs2 = collate_fn([dataset[idx_t1], dataset[idx_t]])
    samples_bs2 = batch_bs2[0].to(device)
    targets_bs2 = [{k: v.to(device) for k, v in t.items()} for t in batch_bs2[1]]

    def run_clean_profile(bs_label, samples_in, targets_in):
        print(f"\n[*] Evaluating Hardware Envelope for {bs_label}:")
        print(f"  {'Configuration':<22} | {'Peak VRAM (MB)':<18} | {'Incremental VRAM':<20} | {'Latency (ms)':<15}")
        print("  " + "-" * 82)

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        base_model = copy.deepcopy(model).train()
        opt_b = torch.optim.AdamW(base_model.parameters(), lr=1e-4)

        for _ in range(3):
            opt_b.zero_grad()
            out_b = base_model(samples_in)
            loss_dict_b = criterion(out_b, targets_in)
            loss_b = sum(loss_dict_b[k] * criterion.weight_dict[k] for k in loss_dict_b if k in criterion.weight_dict)
            loss_b.backward()
            opt_b.step()

        torch.cuda.synchronize()
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)

        start_ev.record()
        for _ in range(10):
            opt_b.zero_grad()
            out_b = base_model(samples_in)
            loss_dict_b = criterion(out_b, targets_in)
            loss_b = sum(loss_dict_b[k] * criterion.weight_dict[k] for k in loss_dict_b if k in criterion.weight_dict)
            loss_b.backward()
            opt_b.step()
        end_ev.record()
        torch.cuda.synchronize()

        base_latency = start_ev.elapsed_time(end_ev) / 10.0
        base_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"  {'Baseline M0':<22} | {base_vram:8.2f} MB       | {'REFERENCE':<20} | {base_latency:6.2f} ms")

        del base_model, opt_b, out_b, loss_b
        gc.collect()
        torch.cuda.empty_cache()

        curr_b = samples_in.tensors.shape[0]
        for cap in [20, 50, 100]:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            str_model = copy.deepcopy(model).train()
            str_adapter = TemporalQueryAdapter(d_model=args.hidden_dim, nheads=args.nheads).to(device).train()
            str_adapter.gamma.data.fill_(0.5)

            str_state = RecursiveTemporalState(batch_size=curr_b, capacity_k=cap, d_model=args.hidden_dim, device=device)
            str_state.feat_cache = [s.detach().repeat(curr_b, 1, 1, 1) for s in feat_cache_t1]
            str_state.mem_tokens = torch.randn(curr_b, cap, args.hidden_dim, device=device)
            str_state.mem_mask = torch.ones(curr_b, cap, dtype=torch.bool, device=device)

            opt_str = torch.optim.AdamW(list(str_model.parameters()) + list(str_adapter.parameters()), lr=1e-4)

            for _ in range(3):
                opt_str.zero_grad()
                out_s = str_model(samples_in, temporal_adapter=str_adapter, temporal_state=str_state, temporal_enabled=True)
                loss_dict_s = criterion(out_s, targets_in)
                loss_s = sum(loss_dict_s[k] * criterion.weight_dict[k] for k in loss_dict_s if k in criterion.weight_dict)
                loss_s.backward()
                opt_str.step()

            torch.cuda.synchronize()
            start_s = torch.cuda.Event(enable_timing=True)
            end_s = torch.cuda.Event(enable_timing=True)

            start_s.record()
            for _ in range(10):
                opt_str.zero_grad()
                out_s = str_model(samples_in, temporal_adapter=str_adapter, temporal_state=str_state, temporal_enabled=True)
                loss_dict_s = criterion(out_s, targets_in)
                loss_s = sum(loss_dict_s[k] * criterion.weight_dict[k] for k in loss_dict_s if k in criterion.weight_dict)
                loss_s.backward()
                opt_str.step()
            end_s.record()
            torch.cuda.synchronize()

            str_lat = start_s.elapsed_time(end_s) / 10.0
            str_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
            vram_delta = str_vram - base_vram

            print(f"  {f'Streaming (K={cap})':<22} | {str_vram:8.2f} MB       | {vram_delta:+8.2f} MB         | {str_lat:6.2f} ms")

            del str_model, str_adapter, str_state, opt_str, out_s, loss_s
            gc.collect()
            torch.cuda.empty_cache()

    run_clean_profile("Batch Size 1", samples_t, targets_t)
    run_clean_profile("Batch Size 2 (Authoritative Training Regime)", samples_bs2, targets_bs2)

    print("\n" + "=" * 115)
    print("POL-1 STRUCTURAL VERIFICATION COMPLETE: ALL 6 REAL INTEGRATION CONTRACTS CERTIFIED.")
    print("=" * 115)


if __name__ == "__main__":
    main()