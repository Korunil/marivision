"""
D:/MARIVISION/H0_workbench/eval_tools/audit_pol2_candidate2m_assignment_diagnostic.py
PoL-2M Stage 0C: Hungarian Matcher Assignment & Identity-Preserving Initialization Diagnostic

Strict Compliance Checklist:
  - D-3556: Evaluates strictly on the 10 frozen design videos (Calibration & Test UNTOUCHED).
  - P-1694 / CLM-2841: Exact canonical fundamentalvision focal matching cost formula (alpha=0.25, gamma=2.0).
  - P-1695 / CLM-2842: Open gate (gamma=+10.0) during matching diagnosis; gate gradient measured at gamma=0.0.
  - P-1696 / CLM-2843: Exact latent-preserving residual adapter (q_i = z_i + delta, delta=0 at init Condition B).
  - P-1697 / CLM-2844: Deterministic shared initialization (seed 41) outside video loop; identical models for all 10 videos.
  - D-3609 / CLM-2845: M0 held strictly in eval() mode during backward passes; only adapter enters train mode.
  - Complete Provenance: Verifies live hashes for all 7 upstream manifests and M0 checkpoint.
  - Idempotent Output: Normalizes JSON payload before serialization to pol2_candidate2m_assignment_diagnostic.json.
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
from scipy.optimize import linear_sum_assignment
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
AGE_AUDIT_JSON = OUT_DIR / "pol2_candidate2m_age_subsets.json"
M0_CKPT_PATH = Path("D:/MARIVISION/checkpoints_r0/checkpoint_best.pth")
ZIP_PATH = Path("D:/MARIVISION/SeaClips.zip")

DIAGNOSTIC_JSON = OUT_DIR / "pol2_candidate2m_assignment_diagnostic.json"

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


def save_or_verify_json(file_path: Path, payload: dict, artifact_name: str) -> str:
    normalized_payload = json.loads(json.dumps(payload))
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
        assert existing_data == normalized_payload, (
            f"[FATAL] Existing {artifact_name} content mismatch with newly generated payload! Aborting to prevent mutation."
        )
        sha = compute_sha256(file_path)
        print(f"[+] Reusing verified existing frozen {artifact_name}: {file_path.resolve()}")
        print(f"    Existing Manifest SHA-256: {sha}")
        return sha
    else:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(normalized_payload, f, indent=2)
        sha = compute_sha256(file_path)
        print(f"[+] Created and frozen new {artifact_name}: {file_path.resolve()}")
        print(f"    Immutable Manifest SHA-256: {sha}")
        return sha


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

        # Confidence gate initialized to 10.0 (open gate for diagnosis)
        self.gamma = nn.Parameter(torch.tensor(10.0, dtype=torch.float32))

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

        # P-1696: Residual formulation: q_i = z_i + delta
        delta_tgt = self.token_proj(self.norm(concat_feat))
        tgt_mem = z_stack + delta_tgt

        inv_ref = inverse_sigmoid(ref_coords)
        pos_proj_in = torch.cat([inv_ref, inv_ref.new_zeros((K, self.d_model - 2))], dim=-1)
        pos_mem = self.pos_proj(pos_proj_in)

        return tgt_mem, pos_mem, ref_coords


def init_identity_preserving(adapter: HistoricalObjectTokenMemory):
    """
    P-1696: Sets residual branch to exactly zero so q_i = z_i + 0 = z_i identically.
    Sets pos_proj to zero so temporal queries start from unbiased reference coordinates.
    Sets gamma = 10.0 for open gate.
    """
    with torch.no_grad():
        adapter.token_proj.weight.zero_()
        adapter.token_proj.bias.zero_()
        adapter.norm.weight.fill_(1.0)
        adapter.norm.bias.zero_()
        adapter.cls_proj.weight.zero_()
        adapter.cls_proj.bias.zero_()
        adapter.age_embed.weight.zero_()
        adapter.pos_proj.weight.zero_()
        adapter.pos_proj.bias.zero_()
        adapter.gamma.data.fill_(10.0)


# ------------------------------------------------------------------------------
# 2. SOURCE HOOKS & ASYMMETRIC ATTENTION MASKING (D-3582)
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
# 3. EXACT BIPARTITE COST DECOMPOSITION (P-1694, CLM-2841)
# ------------------------------------------------------------------------------

def compute_decomposed_matching_costs(matcher, out, tgt_dict):
    """
    Computes pairwise cost matrix and decomposes it into C_class, C_bbox, C_giou
    using the exact canonical fundamentalvision focal matching formula (alpha=0.25, gamma=2.0).
    Verifies that linear_sum_assignment(C_total) matches official matcher output identically.
    """
    out_prob = out["pred_logits"][0].sigmoid() # (Q, num_classes)
    out_bbox = out["pred_boxes"][0]            # (Q, 4) in cxcywh

    tgt_labels = tgt_dict["labels"]            # (N_gt,)
    tgt_bbox = tgt_dict["boxes"]               # (N_gt, 4) in cxcywh

    # Exact canonical fundamentalvision focal matching formula (P-1694)
    alpha = 0.25
    gamma = 2.0
    neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
    pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
    cost_class = pos_cost_class[:, tgt_labels] - neg_cost_class[:, tgt_labels]

    cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
    cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

    C_total = matcher.cost_bbox * cost_bbox + matcher.cost_class * cost_class + matcher.cost_giou * cost_giou

    with torch.no_grad():
        official_matches = matcher(out, [tgt_dict])[0]
        row_ind, col_ind = linear_sum_assignment(C_total.cpu().numpy())
        assert np.array_equal(row_ind, official_matches[0].cpu().numpy()), "[FATAL] Matcher cost matrix diverged from official matcher row indices!"
        assert np.array_equal(col_ind, official_matches[1].cpu().numpy()), "[FATAL] Matcher assignment diverged from official matcher col indices!"

    return C_total.cpu(), cost_class.cpu(), cost_bbox.cpu(), cost_giou.cpu(), official_matches


# ------------------------------------------------------------------------------
# 4. MAIN DIAGNOSTIC ENGINE (P-1694 THROUGH P-1697)
# ------------------------------------------------------------------------------

def main():
    print("=" * 115)
    print("STAGE 0C: CANDIDATE-2M BIPARTITE MATCHER ASSIGNMENT & INITIALIZATION DIAGNOSTIC")
    print("Auditing 10 Design Videos: Fixed Random (Condition A) vs Exact Latent-Preserving (Condition B) (P-1696)")
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

    # Load split manifest to identify the 10 design videos (D-3556)
    with open(SPLIT_MANIFEST_JSON, "r", encoding="utf-8") as f:
        split_meta = json.load(f)["_meta"]
    design_vids = sorted(split_meta["train_videos"])
    assert len(design_vids) == 10, f"[FATAL] Expected 10 design videos, got {len(design_vids)}"
    print(f"[+] Loaded 10 Design Videos: {design_vids} (Calibration & Test Strictly Untouched)")

    # Apply 2M source hooks
    apply_candidate2m_source_hooks()

    args = get_authoritative_r0_args()
    if hasattr(args, "conditioning_mode"):
        args.conditioning_mode = "none"

    model, criterion, postprocessors = build(args)
    ckpt = torch.load(M0_CKPT_PATH, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    # D-3609: M0 parameters strictly frozen and kept in eval() mode permanently
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    criterion.to(device)

    # P-1697: Instantiate shared Condition A and Condition B models outside video loop
    torch.manual_seed(41)
    torch.cuda.manual_seed_all(41)
    adapter_A = HistoricalObjectTokenMemory(d_model=256, num_classes=7, max_age=6).to(device)
    adapter_A.gamma.data.fill_(10.0) # Open gate for unhindered matching diagnosis (P-1695)

    adapter_B = copy.deepcopy(adapter_A)
    init_identity_preserving(adapter_B) # Exact latent preservation: delta = 0 (P-1696)

    print("[+] Instantiated Deterministic Shared Adapters (Seed 41) for All 10 Design Videos (P-1697):")
    print(f"    - Condition A (Fixed Random):           Trainable parameters = {sum(p.numel() for p in adapter_A.parameters()):,}")
    print(f"    - Condition B (Exact Latent-Preserving): token_proj = 0, delta = 0, q_i^mem = z_i exactly (P-1696)")
    print(f"    - Temporal Gate for Matching Diagnosis: gamma = +10.0 (open gate, sigma ~ 1.0, logit offset ~ 0.0) (P-1695)")

    # Extract 1 deterministic genuine MI window per design video (earliest eligible window)
    with open(CANDIDATES_JSON, "r", encoding="utf-8") as f:
        cand_windows = json.load(f)["windows"]

    selected_windows = {}
    for vid in design_vids:
        v_wins = [
            w for w in cand_windows
            if w["video_id"] == vid and w.get("n_mem_incr_s20", w.get("n_memincr_s20", 0)) >= 1
        ]
        assert len(v_wins) >= 1, f"[FATAL] No MI windows found for design video {vid}!"
        v_wins.sort(key=lambda w: w["frame_start"])
        selected_windows[vid] = v_wins[0]

    diagnostic_records = []
    video_summary = defaultdict(lambda: {
        "n_mi_targets": 0,
        "matched_temporal_A": 0,
        "matched_temporal_B": 0,
        "margins_A": [],
        "margins_B": []
    })

    # Evaluate across 10 design videos
    for vid in design_vids:
        w = selected_windows[vid]
        curr_target_img_id = w["image_ids"][-1]
        curr_fid = w["frame_ids"][-1]

        s_curr = collate_fn([dataset[id_to_idx[curr_target_img_id]]])
        samples = s_curr[0].to(device)
        tgt_dict = {
            "boxes": s_curr[1][0]["boxes"].to(device),
            "labels": s_curr[1][0]["labels"].to(device)
        }
        img_h, img_w = s_curr[1][0]["orig_size"][0].item(), s_curr[1][0]["orig_size"][1].item()

        # Build dynamic FIFO memory bank from 5 historical frames using cache-compatible Top-8
        fifo_buffer = deque(maxlen=5)
        with torch.no_grad():
            for f_idx in range(5):
                iid_f = w["image_ids"][f_idx]
                fid_f = w["frame_ids"][f_idx]
                s_f = collate_fn([dataset[id_to_idx[iid_f]]])
                out_f, dec_tokens = model(s_f[0].to(device), temporal_enabled=False, return_decoder_tokens=True)

                prob_f = out_f["pred_logits"][0].sigmoid()
                num_classes = prob_f.shape[-1]
                topk_scores, topk_idx = torch.topk(prob_f.flatten(), 100)
                query_indices = torch.div(topk_idx, num_classes, rounding_mode="floor")
                class_labels = topk_idx % num_classes
                boxes_ranked = out_f["pred_boxes"][0][query_indices]

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

        memory_bank = []
        for hist_fid, f_tokens in fifo_buffer:
            age = curr_fid - hist_fid
            if age in MEMORY_AGES:
                for tok in f_tokens:
                    t_c = copy.deepcopy(tok)
                    t_c["age"] = age
                    memory_bank.append(t_c)

        K_mem = len(memory_bank)

        # Map genuine MI targets to GT index in tgt_dict
        mi_targets = [t for t in w["targets"] if t.get("mem_incr_s20", t.get("memincr_s20", False))]
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
            assert diffs[gt_idx].item() < 1e-3, f"[FATAL] Failed to map MI target {aid} to GT box in video {vid}!"
            mi_gt_indices[aid] = gt_idx

        # ----------------------------------------------------------------------
        # TEST CONDITION A: Fixed Random Adapter (Shared Seed 41)
        # ----------------------------------------------------------------------
        adapter_A.eval()
        adapter_A.gamma.data.fill_(10.0) # Open gate for matching
        with torch.no_grad():
            out_A = model(samples, temporal_adapter=adapter_A, memory_bank=[memory_bank], temporal_enabled=True)
            C_tot_A, C_cls_A, C_bb_A, C_gi_A, matches_A = compute_decomposed_matching_costs(criterion.matcher, out_A, tgt_dict)

        # Backward diagnostic at candidate initialization gamma = 0.0 (D-3609: M0 stays in eval)
        adapter_A.train()
        adapter_A.gamma.data.fill_(0.0)
        out_A_train = model(samples, temporal_adapter=adapter_A, memory_bank=[memory_bank], temporal_enabled=True)
        loss_dict_A = criterion(out_A_train, [tgt_dict])
        loss_A = sum(loss_dict_A[k] * criterion.weight_dict[k] for k in loss_dict_A if k in criterion.weight_dict)
        loss_A.backward()
        d_gamma_A = float(adapter_A.gamma.grad.item())
        adapter_A.zero_grad()
        adapter_A.gamma.data.fill_(10.0)
        adapter_A.eval()

        # ----------------------------------------------------------------------
        # TEST CONDITION B: Exact Latent-Preserving (W_token = [I, 0], delta = 0)
        # ----------------------------------------------------------------------
        adapter_B.eval()
        adapter_B.gamma.data.fill_(10.0) # Open gate for matching
        with torch.no_grad():
            out_B = model(samples, temporal_adapter=adapter_B, memory_bank=[memory_bank], temporal_enabled=True)
            C_tot_B, C_cls_B, C_bb_B, C_gi_B, matches_B = compute_decomposed_matching_costs(criterion.matcher, out_B, tgt_dict)

        # Backward diagnostic at candidate initialization gamma = 0.0 (D-3609: M0 stays in eval)
        adapter_B.train()
        adapter_B.gamma.data.fill_(0.0)
        out_B_train = model(samples, temporal_adapter=adapter_B, memory_bank=[memory_bank], temporal_enabled=True)
        loss_dict_B = criterion(out_B_train, [tgt_dict])
        loss_B = sum(loss_dict_B[k] * criterion.weight_dict[k] for k in loss_dict_B if k in criterion.weight_dict)
        loss_B.backward()
        d_gamma_B = float(adapter_B.gamma.grad.item())
        adapter_B.zero_grad()
        adapter_B.gamma.data.fill_(10.0)
        adapter_B.eval()

        # Process matching results for each MI target
        src_A_list = matches_A[0].tolist()
        tgt_A_list = matches_A[1].tolist()
        src_B_list = matches_B[0].tolist()
        tgt_B_list = matches_B[1].tolist()

        for aid, gt_idx in mi_gt_indices.items():
            ann = ann_lookup[aid]
            video_summary[vid]["n_mi_targets"] += 1

            # Condition A
            pos_m_A = tgt_A_list.index(gt_idx) if gt_idx in tgt_A_list else None
            q_win_A = src_A_list[pos_m_A] if pos_m_A is not None else None
            branch_A = "TEMPORAL" if (q_win_A is not None and q_win_A >= 300) else ("CANONICAL" if q_win_A is not None else "UNMATCHED")
            if branch_A == "TEMPORAL":
                video_summary[vid]["matched_temporal_A"] += 1

            q_best_canon_A = C_tot_A[:300, gt_idx].argmin().item()
            q_best_temp_A = (300 + C_tot_A[300:, gt_idx].argmin().item()) if K_mem > 0 else 300
            cost_canon_A = C_tot_A[q_best_canon_A, gt_idx].item()
            cost_temp_A = C_tot_A[q_best_temp_A, gt_idx].item() if K_mem > 0 else float('inf')
            margin_A = cost_temp_A - cost_canon_A
            video_summary[vid]["margins_A"].append(margin_A)

            # Condition B
            pos_m_B = tgt_B_list.index(gt_idx) if gt_idx in tgt_B_list else None
            q_win_B = src_B_list[pos_m_B] if pos_m_B is not None else None
            branch_B = "TEMPORAL" if (q_win_B is not None and q_win_B >= 300) else ("CANONICAL" if q_win_B is not None else "UNMATCHED")
            if branch_B == "TEMPORAL":
                video_summary[vid]["matched_temporal_B"] += 1

            q_best_canon_B = C_tot_B[:300, gt_idx].argmin().item()
            q_best_temp_B = (300 + C_tot_B[300:, gt_idx].argmin().item()) if K_mem > 0 else 300
            cost_canon_B = C_tot_B[q_best_canon_B, gt_idx].item()
            cost_temp_B = C_tot_B[q_best_temp_B, gt_idx].item() if K_mem > 0 else float('inf')
            margin_B = cost_temp_B - cost_canon_B
            video_summary[vid]["margins_B"].append(margin_B)

            best_temp_age_B = memory_bank[q_best_temp_B - 300]["age"] if K_mem > 0 else None

            cost_comps_B = {
                "temp_class": float(criterion.matcher.cost_class * C_cls_B[q_best_temp_B, gt_idx].item()) if K_mem > 0 else None,
                "temp_bbox":  float(criterion.matcher.cost_bbox * C_bb_B[q_best_temp_B, gt_idx].item()) if K_mem > 0 else None,
                "temp_giou":  float(criterion.matcher.cost_giou * C_gi_B[q_best_temp_B, gt_idx].item()) if K_mem > 0 else None,
                "canon_class": float(criterion.matcher.cost_class * C_cls_B[q_best_canon_B, gt_idx].item()),
                "canon_bbox":  float(criterion.matcher.cost_bbox * C_bb_B[q_best_canon_B, gt_idx].item()),
                "canon_giou":  float(criterion.matcher.cost_giou * C_gi_B[q_best_canon_B, gt_idx].item()),
            }

            rec = {
                "video_id": vid,
                "frame_id": curr_fid,
                "ann_id": aid,
                "category": cats[ann["category_id"]],
                "gt_index": gt_idx,
                "k_mem_tokens": K_mem,
                "condition_A_random": {
                    "matched_branch": branch_A,
                    "winning_query": q_win_A,
                    "cost_canon_best": cost_canon_A,
                    "cost_temp_best": cost_temp_A,
                    "margin_temp_minus_canon": margin_A,
                    "d_gamma": d_gamma_A
                },
                "condition_B_identity": {
                    "matched_branch": branch_B,
                    "winning_query": q_win_B,
                    "cost_canon_best": cost_canon_B,
                    "cost_temp_best": cost_temp_B,
                    "margin_temp_minus_canon": margin_B,
                    "best_temp_age": best_temp_age_B,
                    "cost_components": cost_comps_B,
                    "d_gamma": d_gamma_B
                }
            }
            diagnostic_records.append(rec)

    # --------------------------------------------------------------------------
    # PRINT COMPREHENSIVE DIAGNOSTIC AUDIT REPORT
    # --------------------------------------------------------------------------
    total_targets = len(diagnostic_records)
    total_temp_A = sum(1 for r in diagnostic_records if r["condition_A_random"]["matched_branch"] == "TEMPORAL")
    total_temp_B = sum(1 for r in diagnostic_records if r["condition_B_identity"]["matched_branch"] == "TEMPORAL")

    print("\n" + "=" * 115)
    print("DETAILED PER-TARGET MATCHING DIAGNOSTIC: CONDITION A (RANDOM) VS CONDITION B (EXACT LATENT-PRESERVING)")
    print("Open Gate Diagnostic (gamma = +10.0) | Target Frame MI Assignments")
    print("=" * 115)
    print(f"{'Target ID':<10} | {'Video':<7} | {'Category':<14} | {'Branch A':<10} | {'Margin A':<10} | {'Branch B':<10} | {'Margin B':<10} | {'Age':<5} | {'Delta Margin (B - A)'}")
    print("-" * 115)

    for r in diagnostic_records:
        tid = str(r["ann_id"])
        vid = f"v{r['video_id']}"
        cat = r["category"]
        br_A = r["condition_A_random"]["matched_branch"]
        mg_A = r["condition_A_random"]["margin_temp_minus_canon"]
        br_B = r["condition_B_identity"]["matched_branch"]
        mg_B = r["condition_B_identity"]["margin_temp_minus_canon"]
        age_B = str(r["condition_B_identity"]["best_temp_age"])
        d_mg = mg_B - mg_A

        marker = " <--- TEMPORAL MATCH GAINED!" if br_A != "TEMPORAL" and br_B == "TEMPORAL" else ""
        print(f"{tid:<10} | {vid:<7} | {cat:<14} | {br_A:<10} | {mg_A:+9.3f} | {br_B:<10} | {mg_B:+9.3f} | {age_B:<5} | {d_mg:+9.3f}{marker}")

    # Cost Component Decomposition for Condition B
    print("\n" + "=" * 115)
    print("CONDITION B (EXACT LATENT-PRESERVING) COST DECOMPOSITION (BEST TEMPORAL VS BEST CANONICAL)")
    print("Components: [Temporal Cost] vs [Canonical Cost] (Negative Delta = Temporal Query Wins Component)")
    print("=" * 115)
    print(f"{'Target ID':<10} | {'Class Delta [T - C]':<22} | {'Bbox L1 Delta [T - C]':<24} | {'GIoU Delta [T - C]':<22} | {'Net Margin [T - C]'}")
    print("-" * 96)

    mean_d_cls = []
    mean_d_bb = []
    mean_d_giou = []
    mean_d_net = []

    for r in diagnostic_records:
        tid = str(r["ann_id"])
        c = r["condition_B_identity"]["cost_components"]
        d_c = c["temp_class"] - c["canon_class"]
        d_b = c["temp_bbox"] - c["canon_bbox"]
        d_g = c["temp_giou"] - c["canon_giou"]
        net = r["condition_B_identity"]["margin_temp_minus_canon"]

        mean_d_cls.append(d_c)
        mean_d_bb.append(d_b)
        mean_d_giou.append(d_g)
        mean_d_net.append(net)

        print(f"{tid:<10} | {d_c:+10.4f} (temp {c['temp_class']:4.2f}) | {d_b:+10.4f} (temp {c['temp_bbox']:4.2f})  | {d_g:+10.4f} (temp {c['temp_giou']:4.2f}) | {net:+10.4f}")

    print("-" * 96)
    print(f"{'MEAN':<10} | {np.mean(mean_d_cls):+10.4f}             | {np.mean(mean_d_bb):+10.4f}              | {np.mean(mean_d_giou):+10.4f}             | {np.mean(mean_d_net):+10.4f}")

    # Per-Video Robustness Summary
    print("\n" + "=" * 115)
    print("PER-VIDEO DESIGN SUMMARY: TEMPORAL MATCHING RATES ACROSS 10 DESIGN VIDEOS")
    print("=" * 115)
    print(f"{'Video ID':<10} | {'MI Targets':<12} | {'Temporal Matches A':<20} | {'Temporal Matches B':<20} | {'Mean Margin A':<16} | {'Mean Margin B'}")
    print("-" * 96)

    for vid in design_vids:
        s = video_summary[vid]
        n_t = s["n_mi_targets"]
        t_A = s["matched_temporal_A"]
        t_B = s["matched_temporal_B"]
        m_A = np.mean(s["margins_A"])
        m_B = np.mean(s["margins_B"])
        print(f"video_{vid:<4d} | {n_t:<12d} | {t_A:2d} / {n_t:2d} ({t_A/n_t*100:5.1f}%)        | {t_B:2d} / {n_t:2d} ({t_B/n_t*100:5.1f}%)        | {m_A:+10.3f}      | {m_B:+10.3f}")

    # Aggregate Evaluation & Structural Decision Verdict
    print("\n" + "=" * 115)
    print("AGGREGATE INITIALIZATION & SUPERVISION VERDICT (Q-1835..Q-1838)")
    print("=" * 115)
    pct_A = (total_temp_A / total_targets) * 100.0
    pct_B = (total_temp_B / total_targets) * 100.0

    print(f"[*] Overall MI Target Assignment to Temporal Queries (Open Gate gamma = +10.0):")
    print(f"    - Condition A (Fixed Random):           {total_temp_A:2d} / {total_targets:2d} ({pct_A:5.1f}%)")
    print(f"    - Condition B (Exact Latent-Preserving):{total_temp_B:2d} / {total_targets:2d} ({pct_B:5.1f}%)")
    print(f"    - Net Temporal Assignment Gain:         {total_temp_B - total_temp_A:+2d} targets ({pct_B - pct_A:+5.1f}%)")

    mean_gamma_A = np.mean([r["condition_A_random"]["d_gamma"] for r in diagnostic_records])
    mean_gamma_B = np.mean([r["condition_B_identity"]["d_gamma"] for r in diagnostic_records])
    print(f"\n[*] Global Confidence Gate Gradient (dL/d_gamma evaluated at candidate init gamma = 0.0):")
    print(f"    - Condition A Mean dL/d_gamma: {mean_gamma_A:+8.4f} (Positive = Suppressive Pressure)")
    print(f"    - Condition B Mean dL/d_gamma: {mean_gamma_B:+8.4f}")

    # Serialize diagnostic results (P-1665, P-1667)
    diagnostic_payload = {
        "_meta": {
            "artifact_name": "pol2_candidate2m_assignment_diagnostic_v1",
            "role": "POL2_CANDIDATE_2M_INITIALIZATION_AND_SUPERVISION_DIAGNOSTIC",
            "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
            "m0_checkpoint_sha256": EXPECTED_M0_HASH,
            "total_design_videos_audited": 10,
            "total_mi_targets_audited": total_targets,
            "pct_temporal_matches_condition_A": pct_A,
            "pct_temporal_matches_condition_B": pct_B
        },
        "aggregate_metrics": {
            "condition_A_random": {
                "total_matched_temporal": total_temp_A,
                "pct_matched_temporal": pct_A,
                "mean_margin": float(np.mean([r["condition_A_random"]["margin_temp_minus_canon"] for r in diagnostic_records])),
                "mean_d_gamma": float(mean_gamma_A)
            },
            "condition_B_identity": {
                "total_matched_temporal": total_temp_B,
                "pct_matched_temporal": pct_B,
                "mean_margin": float(np.mean([r["condition_B_identity"]["margin_temp_minus_canon"] for r in diagnostic_records])),
                "mean_d_gamma": float(mean_gamma_B),
                "mean_cost_deltas": {
                    "delta_class": float(np.mean(mean_d_cls)),
                    "delta_bbox": float(np.mean(mean_d_bb)),
                    "delta_giou": float(np.mean(mean_d_giou)),
                    "delta_net": float(np.mean(mean_d_net))
                }
            }
        },
        "per_target_records": diagnostic_records,
        "per_video_summary": {str(k): v for k, v in video_summary.items()}
    }

    diag_hash = save_or_verify_json(DIAGNOSTIC_JSON, diagnostic_payload, "Candidate 2M Assignment Diagnostic Manifest")
    print("=" * 115)


if __name__ == "__main__":
    main()