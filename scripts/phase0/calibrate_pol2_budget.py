"""
D:/MARIVISION/H0_workbench/eval_tools/calibrate_pol2_budget.py
Phase-II Proof-of-Launch Stage 2 (PoL-2): Tri-Seed Non-Comparative Step-Budget Calibration

Authoritative Compliance Checklist:
  - D-3400 / D-3401: Calibration budget permanently fixed over E in {0..25};
                     earliest argmin of mean tri-seed held-out loss.
  - D-3402: If E* = 0, records that positive training was not earned; does not coerce positive budget.
  - D-3403: Memory formally scoped as bounded recursively selected historical decoder-token bank.
  - D-3391: Precomputes frozen-M0 states strictly for frames 40..44 (indices 0..4);
            held-out frame 45 (index 5) is strictly isolated to no-grad evaluation.
  - D-3392 / D-3393: Uniform inherited AdamW lr = 2e-4, weight_decay = 1e-4 across all adapter parameters.
  - D-3394: Attention dropout locked to 0.0 to eliminate branch-dependent stochastic variance.
  - D-3395 / D-3397 / D-3399: Evaluates across fixed seeds {41, 42, 43}; exports detached CPU initializations
                              to pol2_adapter_initializations.pth for exact pairing in A/B/C.
  - P-1574: Hard-asserts live hashes for checkpoint_best.pth and seaclips-train.json.
  - P-1576: Finite-loss assertions on all training and evaluation endpoints.
  - P-1577: Reports exact epoch-0 held-out loss spread across seeds.
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
COHORT_S_JSON = OUT_DIR / "pol2_primary_same_sensor_clips.json"
ZIP_PATH = Path("D:/MARIVISION/SeaClips.zip")
TRAIN_JSON = Path("D:/MARIVISION/seaclips-train.json")
M0_CKPT_PATH = Path("D:/MARIVISION/checkpoints_r0/checkpoint_best.pth")

CALIB_RECORD_JSON = OUT_DIR / "pol2_budget_calibration.json"
ADAPTER_INITS_PTH = OUT_DIR / "pol2_adapter_initializations.pth"

EXPECTED_COHORT_S_SHA256 = "b157b11e0b1bb8f9a77729a4999d48693f5d1e0cb052d8423b5a8dc903f029dc"
EXPECTED_M0_HASH = "28e6288c3f6769c2d5275aa77ddb248735b55a09cd46a6b570e3f117890b268b"
EXPECTED_TRAIN_JSON_SHA256 = "edc32bdbd6b792eb83b6daf11e14708232f1f4a9e06153ebebd5bad1caef9b35"

CALIB_SEEDS = [41, 42, 43]
CALIB_MAX_EPOCHS = 25


def compute_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------------------
# 1. RECURSIVE TEMPORAL STATE & ADAPTER (DROPOUT = 0.0)
# ------------------------------------------------------------------------------

class RecursiveTemporalState:
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
    def __init__(self, d_model=256, nheads=8, dropout=0.0):
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
        return tgt + gate * fused


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
# 3. TRI-SEED STEP-BUDGET CALIBRATION ENGINE
# ------------------------------------------------------------------------------

def main():
    print("=" * 115)
    print("PROOF-OF-LAUNCH STAGE 2 (PoL-2): TRI-SEED STEP-BUDGET CALIBRATION (SEEDS 41, 42, 43)")
    print("=" * 115)

    # Hard-assert live artifact hashes (P-1574)
    assert COHORT_S_JSON.exists(), f"[FATAL] Missing {COHORT_S_JSON}"
    assert M0_CKPT_PATH.exists(), f"[FATAL] Missing {M0_CKPT_PATH}"
    assert TRAIN_JSON.exists(), f"[FATAL] Missing {TRAIN_JSON}"

    actual_cohort_s_hash = compute_sha256(COHORT_S_JSON)
    actual_m0_hash = compute_sha256(M0_CKPT_PATH)
    actual_train_hash = compute_sha256(TRAIN_JSON)

    assert actual_cohort_s_hash == EXPECTED_COHORT_S_SHA256, f"[FATAL] Cohort S hash mismatch! {actual_cohort_s_hash}"
    assert actual_m0_hash == EXPECTED_M0_HASH, f"[FATAL] M0 checkpoint hash mismatch! {actual_m0_hash}"
    assert actual_train_hash == EXPECTED_TRAIN_JSON_SHA256, f"[FATAL] Train JSON hash mismatch! {actual_train_hash}"

    with open(COHORT_S_JSON, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    manifest_meta = manifest_data.get("_meta", {})
    assert manifest_meta.get("m0_checkpoint_sha256") == actual_m0_hash, "[FATAL] Manifest M0 hash mismatch!"
    assert manifest_meta.get("train_json_sha256") == actual_train_hash, "[FATAL] Manifest train JSON hash mismatch!"

    print(f"[+] All Live Provenance Invariants Certified (P-1574):")
    print(f"    - Cohort S Manifest SHA-256: {actual_cohort_s_hash}")
    print(f"    - Certified M0 SHA-256:       {actual_m0_hash}")
    print(f"    - Live Train JSON SHA-256:   {actual_train_hash}")

    calib_clip_meta = manifest_data["calibration_clip"]
    print(f"[+] Calibration Target: Video {calib_clip_meta['video_id']} | Frames {calib_clip_meta['frame_start']} -> {calib_clip_meta['frame_end']} ({calib_clip_meta['sensor']})")
    print(f"    - Frame t-5 (Frame {calib_clip_meta['frame_ids'][0]}): State Initialization ONLY (NO backward)")
    print(f"    - Frames t-4..t-1 (Frames {calib_clip_meta['frame_ids'][1:5]}): 4 Trainable History Frames")
    print(f"    - Frame t (Frame {calib_clip_meta['frame_ids'][5]}): Strictly Held-Out Evaluation Frame")

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

    # Freeze all baseline M0 parameters completely
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    criterion.to(device)
    print("[+] Model built; all baseline M0 parameters strictly frozen.")

    # Load 6 calibration frames
    dataset = SeaClipsDetection(ZIP_PATH, TRAIN_JSON, is_train=False, random_flip=False)
    coco = dataset.coco
    img_ids = sorted(coco.getImgIds())
    id_to_idx = {iid: idx for idx, iid in enumerate(img_ids)}

    calib_batches = []
    for iid in calib_clip_meta["image_ids"]:
        idx = id_to_idx[iid]
        sample = dataset[idx]
        b = collate_fn([sample])
        samples = b[0].to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in b[1]]
        calib_batches.append((samples, targets))

    assert len(calib_batches) == 6, f"[FATAL] Expected exactly 6 frames, got {len(calib_batches)}"

    # --------------------------------------------------------------------------
    # STEP 1: PRECOMPUTE FROZEN-M0 STATES STRICTLY FOR FRAMES 0..4 (D-3391)
    # --------------------------------------------------------------------------
    print("\n--- STEP 1: PRECOMPUTING FROZEN-M0 STATES FOR FRAMES 0..4 ONLY (D-3391) ---")
    frozen_states = []
    state_tracker = RecursiveTemporalState(batch_size=1, capacity_k=50, d_model=args.hidden_dim, device=device)

    with torch.no_grad():
        for f_idx in range(5):  # ONLY t-5..t-1 (indices 0..4); frame 45 is NEVER processed here!
            samples, targets = calib_batches[f_idx]
            out, feat_cache, decoder_tokens = model(samples, temporal_enabled=False, return_streaming_state=True)
            scores = out["pred_logits"][0].sigmoid().max(-1).values.unsqueeze(0)
            state_tracker.recursive_update(
                new_tokens=decoder_tokens,
                new_boxes=out["pred_boxes"],
                new_logits=out["pred_logits"],
                new_scores=scores
            )
            state_tracker.feat_cache = [f.detach().clone() for f in feat_cache]
            frozen_states.append(copy.deepcopy(state_tracker))

    assert len(frozen_states) == 5, f"[FATAL] Expected exactly 5 precomputed states, got {len(frozen_states)}"
    print(f"[+] Exactly 5 historical states precomputed (frames 40..44). Held-out frame 45 is strictly isolated.")

    # --------------------------------------------------------------------------
    # STEP 2: TRI-SEED CALIBRATION SCAN INCLUDING EPOCH 0 (D-3401, P-1573, P-1576)
    # --------------------------------------------------------------------------
    print(f"\n--- STEP 2: RUNNING CALIBRATION ACROSS SEEDS {CALIB_SEEDS} (EPOCHS 0..{CALIB_MAX_EPOCHS}) ---")
    per_seed_train_losses = {}
    per_seed_heldout_losses = {}
    adapter_initial_weights = {}

    samples_5, targets_5 = calib_batches[5]
    state_5_in = frozen_states[4]  # Full history from frames 0..4

    for seed in CALIB_SEEDS:
        print(f"\n[*] Initializing & Evaluating Seed {seed}...")
        set_seed(seed)

        # Instantiate fresh adapter with dropout = 0.0 (D-3394)
        adapter = TemporalQueryAdapter(d_model=args.hidden_dim, nheads=args.nheads, dropout=0.0).to(device)
        adapter.gamma.data.fill_(0.0)

        # Save initial adapter weights as detached CPU clones (D-3397, D-3399, P-1575)
        adapter_initial_weights[seed] = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}

        # Epoch 0: Evaluate held-out frame 45 BEFORE any gradient updates (P-1573)
        adapter.eval()
        with torch.no_grad():
            out_eval_0 = model(
                samples_5,
                temporal_adapter=adapter,
                temporal_state=state_5_in,
                temporal_enabled=True
            )
            loss_dict_0 = criterion(out_eval_0, targets_5)
            heldout_loss_0 = sum(loss_dict_0[term] * criterion.weight_dict[term] for term in loss_dict_0 if term in criterion.weight_dict).item()

        # P-1576: Finite-loss assertion
        assert np.isfinite(heldout_loss_0), f"[FATAL] Non-finite epoch-0 held-out loss for seed {seed}: {heldout_loss_0}"

        seed_train_curve = [None]  # No training at epoch 0
        seed_heldout_curve = [heldout_loss_0]

        print(f"  Epoch 0 (Untrained Base) | Held-Out Loss: {heldout_loss_0:.4f} | Gate: {adapter.gamma.item():.4f}")

        # Unified AdamW optimizer (D-3392, D-3393): lr = 2e-4, weight_decay = 1e-4 across all parameters
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=2e-4, weight_decay=1e-4)

        for epoch in range(1, CALIB_MAX_EPOCHS + 1):
            # Train on history frames 41..44 (indices 1..4)
            adapter.train()
            step_losses = []

            for k in range(1, 5):
                samples_k, targets_k = calib_batches[k]
                state_in = frozen_states[k - 1]

                optimizer.zero_grad()
                out = model(
                    samples_k,
                    temporal_adapter=adapter,
                    temporal_state=state_in,
                    temporal_enabled=True
                )

                loss_dict = criterion(out, targets_k)
                loss = sum(loss_dict[term] * criterion.weight_dict[term] for term in loss_dict if term in criterion.weight_dict)
                loss.backward()

                # Guarded gradient check
                grad_norm = sum(p.grad.norm().item() for p in adapter.parameters() if p.grad is not None)
                assert grad_norm > 0.0, f"[FATAL] Zero gradient in trainable adapter at epoch {epoch}, frame index {k}!"
                optimizer.step()
                step_losses.append(loss.item())

            mean_train_loss = float(np.mean(step_losses))
            # P-1576: Finite-loss assertion
            assert np.isfinite(mean_train_loss), f"[FATAL] Non-finite train loss: seed={seed}, epoch={epoch}, loss={mean_train_loss}"
            seed_train_curve.append(mean_train_loss)

            # Evaluate held-out frame 45 (index 5) in eval mode (D-3386)
            adapter.eval()
            with torch.no_grad():
                out_eval = model(
                    samples_5,
                    temporal_adapter=adapter,
                    temporal_state=state_5_in,
                    temporal_enabled=True
                )
                loss_dict_eval = criterion(out_eval, targets_5)
                heldout_loss = sum(loss_dict_eval[term] * criterion.weight_dict[term] for term in loss_dict_eval if term in criterion.weight_dict).item()

            # P-1576: Finite-loss assertion
            assert np.isfinite(heldout_loss), f"[FATAL] Non-finite held-out loss: seed={seed}, epoch={epoch}, loss={heldout_loss}"
            seed_heldout_curve.append(heldout_loss)

            gamma_val = adapter.gamma.item()
            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:<3d} (Trained)        | Train Loss: {mean_train_loss:<10.4f} | Held-Out Loss: {heldout_loss:<10.4f} | Gate: {gamma_val:<8.4f}")

        per_seed_train_losses[seed] = seed_train_curve
        per_seed_heldout_losses[seed] = seed_heldout_curve

    # --------------------------------------------------------------------------
    # STEP 3: EPOCH-0 SPREAD DIAGNOSTIC & THRESHOLD-FREE E* SELECTION (P-1577, D-3401)
    # --------------------------------------------------------------------------
    epoch_0_losses = [per_seed_heldout_losses[s][0] for s in CALIB_SEEDS]
    epoch_0_spread = max(epoch_0_losses) - min(epoch_0_losses)
    print("\n" + "=" * 115)
    print("EPOCH-0 GATE SUPPRESSION DIAGNOSTIC (P-1577)")
    print("=" * 115)
    print(f"[*] Seed 41 Epoch-0 Loss: {epoch_0_losses[0]:.10f}")
    print(f"[*] Seed 42 Epoch-0 Loss: {epoch_0_losses[1]:.10f}")
    print(f"[*] Seed 43 Epoch-0 Loss: {epoch_0_losses[2]:.10f}")
    print(f"[*] Epoch-0 Held-Out Spread: {epoch_0_spread:.10e} (Max - Min across Seeds)")

    mean_heldout_losses = []
    for e_idx in range(CALIB_MAX_EPOCHS + 1):  # Indices 0..25 (26 values)
        avg_l = float(np.mean([per_seed_heldout_losses[s][e_idx] for s in CALIB_SEEDS]))
        mean_heldout_losses.append(avg_l)

    min_mean_loss = min(mean_heldout_losses)
    # Earliest argmin tie-breaker across epochs 0..25 (D-3401)
    optimal_epoch = mean_heldout_losses.index(min_mean_loss)

    print("\n" + "=" * 115)
    print("TRI-SEED CALIBRATION SCAN RESULTS (EPOCHS 0..25)")
    print("=" * 115)
    print(f"{'Epoch':<8} | {'Seed 41 Held-Out':<20} | {'Seed 42 Held-Out':<20} | {'Seed 43 Held-Out':<20} | {'Mean Held-Out Loss':<22}")
    print("-" * 95)
    for e in range(CALIB_MAX_EPOCHS + 1):
        l41 = per_seed_heldout_losses[41][e]
        l42 = per_seed_heldout_losses[42][e]
        l43 = per_seed_heldout_losses[43][e]
        m_l = mean_heldout_losses[e]
        marker = " <--- OPTIMAL (E*)" if e == optimal_epoch else ""
        label = f"Epoch {e:<3d}" if e > 0 else "Base (0)"
        print(f"{label:<8} | {l41:<20.4f} | {l42:<20.4f} | {l43:<20.4f} | {m_l:<22.4f}{marker}")

    print("\n" + "=" * 115)
    if optimal_epoch == 0:
        print("[!] CALIBRATION RESULT: ZERO-TRAINING BASELINE OPTIMAL (E* = 0) (D-3402).")
        print("    No positive epoch achieved a lower mean tri-seed held-out loss than epoch 0.")
        print("    Calibration did not earn positive-gradient training under the frozen PoL-2 contract.")
        total_sync_steps = 0
    else:
        total_sync_steps = optimal_epoch * 4
        print(f"[+] Optimal Epoch Budget Locked (E*):       {optimal_epoch} Epochs")
        print(f"    Minimum Mean Held-Out Loss:            {min_mean_loss:.4f} (vs. Epoch 0 Base = {mean_heldout_losses[0]:.4f})")
        print(f"    Synchronized Updates per Epoch:        4 Updates (1 per relative history frame)")
        print(f"    Authoritative Synchronized Steps (A/B/C): {total_sync_steps} Total Gradient Updates")
    print("=" * 115)

    # --------------------------------------------------------------------------
    # STEP 4: EXPORT FROZEN ARTIFACTS (D-3397, D-3399)
    # --------------------------------------------------------------------------
    torch.save(adapter_initial_weights, ADAPTER_INITS_PTH)
    inits_hash = compute_sha256(ADAPTER_INITS_PTH)
    print(f"[+] Adapter Initial Weights Frozen (CPU): {ADAPTER_INITS_PTH.resolve()}")
    print(f"    Initial Weights SHA-256:             {inits_hash}")

    payload = {
        "_meta": {
            "artifact_name": "pol2_budget_calibration_frozen_v2",
            "cohort_s_sha256": actual_cohort_s_hash,
            "calibration_video_id": calib_clip_meta["video_id"],
            "calibration_frames": f"{calib_clip_meta['frame_start']} -> {calib_clip_meta['frame_end']}",
            "calib_sensor": calib_clip_meta["sensor"],
            "temporal_memory_type": "bounded_recursively_selected_historical_decoder_token_bank",
            "held_out_frame_id": calib_clip_meta["frame_ids"][5],
            "train_frame_ids": calib_clip_meta["frame_ids"][1:5],
            "init_frame_id": calib_clip_meta["frame_ids"][0],
            "selection_rule": "earliest_argmin_mean_heldout_frame_loss_tri_seed_with_epoch_0",
            "seeds": CALIB_SEEDS,
            "epoch_0_spread": float(epoch_0_spread),
            "max_scan_epochs": CALIB_MAX_EPOCHS,
            "optimizer": "AdamW",
            "lr": 2e-4,
            "weight_decay": 1e-4,
            "dropout": 0.0,
            "adapter_inits_sha256": inits_hash
        },
        "per_seed_train_losses": {str(k): v for k, v in per_seed_train_losses.items()},
        "per_seed_heldout_losses": {str(k): v for k, v in per_seed_heldout_losses.items()},
        "mean_heldout_losses": mean_heldout_losses,
        "optimal_epoch": optimal_epoch,
        "min_mean_heldout_loss": min_mean_loss,
        "steps_per_epoch": 4,
        "total_synchronized_steps": total_sync_steps
    }

    with open(CALIB_RECORD_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    calib_hash = compute_sha256(CALIB_RECORD_JSON)
    print(f"[+] Calibration Manifest Frozen:           {CALIB_RECORD_JSON.resolve()}")
    print(f"    Calibration Record SHA-256:            {calib_hash}\n")


if __name__ == "__main__":
    main()