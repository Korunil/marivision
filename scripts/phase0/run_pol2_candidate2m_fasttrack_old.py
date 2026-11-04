"""
D:/MARIVISION/H0_workbench/eval_tools/run_pol2_candidate2m_fasttrack.py
Candidate 2M Fast-Track Driver: Genuine Stage 0D -> Token Cache -> Seed 41 Screen -> Tri-Seed Test

Strict Compliance Checklist:
  - D-3626 / D-3631: Fast-track mode enabled; automated gating transitions without intermediate review pauses.
  - P-1708 / D-3633: Deterministic spatial-shift counterfactual M_shifted (dx = dy = +0.25 mod 1.0).
  - P-1709: Stage 0D smoke test uses genuine cache-compatible memory from video_4 (P-1704 two-step contract).
  - P-1710 / D-3634: Eligible checkpoint selection E* in {1, 2, 4, 8, 12}; persists and reuses Seed-41 weights directly.
  - P-1711 / D-3635: Operational quality guardrail on whole augmented predictions: requires F1(M_corr) >= F1(D).
  - P-1705: Precomputes and freezes historical M0 token cache for design and calibration pools.
  - Fail-Fast Gate: If E* == 0, terminates immediately, leaving protected test videos 26 and 40 unspent.
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
from r0_preflight import get_authoritative_r0_args
from datasets.seaclips_detection import SeaClipsDetection
from util.misc import collate_fn, NestedTensor, inverse_sigmoid, nested_tensor_from_tensor_list

OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
CANDIDATES_JSON = OUT_DIR / "pol2_candidate_windows_v1.json"
TRAIN_JSON = Path("D:/MARIVISION/seaclips-train.json")
CACHE_FILE = OUT_DIR / "cache_m0_train_raw_top100.json"
DEV_POOL_JSON = OUT_DIR / "pol2_dev_pool_manifest.json"
SPLIT_MANIFEST_JSON = OUT_DIR / "pol2_candidate2m_micro_split.json"
M0_CKPT_PATH = Path("D:/MARIVISION/checkpoints_r0/checkpoint_best.pth")
ZIP_PATH = Path("D:/MARIVISION/SeaClips.zip")

TOKEN_CACHE_PTH = OUT_DIR / "cache_m0_candidate2m_design_calib.pth"
FINAL_MODELS_PTH = OUT_DIR / "pol2_candidate2m_final_models.pth"
RESULTS_JSON = OUT_DIR / "pol2_candidate2m_fasttrack_results.json"

EXPECTED_CANDIDATE_SHA256 = "f29e0b8a5cbfc46641222d84a6d370c3e9bc51ca4ca1124b7753078ca7c1e270"
EXPECTED_TRAIN_JSON_SHA256 = "edc32bdbd6b792eb83b6daf11e14708232f1f4a9e06153ebebd5bad1caef9b35"
EXPECTED_CACHE_SHA256 = "d9816083760d698935b44e9165522e709b490c3af29e710d75b8b0d2d90210e0"
EXPECTED_DEV_POOL_SHA256 = "533567e1d74544db7087a95253e640ef0959b998e5d503f57b0eb8074d8fe970"
EXPECTED_SPLIT_SHA256 = "7da471d2793fe3369508200dec50422bb76492860c67a1a61871645471856772"
EXPECTED_M0_HASH = "28e6288c3f6769c2d5275aa77ddb248735b55a09cd46a6b570e3f117890b268b"

MEMORY_AGES = [2, 3, 4, 5]
MAX_TOKENS_PER_AGE = 8
SCORE_THRESHOLD = 0.20
EVAL_IOU_THRESHOLD = 0.30
LAMBDA_MI = 1.0
CALIB_CHECKPOINTS = [1, 2, 4, 8, 12]
MAX_SCREEN_EPOCHS = 12


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
    """
    P-1708 / D-3633: Deterministic spatial-shift counterfactual M_shifted.
    Shifts center coordinates by dx = dy = +0.25 (mod 1.0), preserving appearance z,
    class logits, confidence score, age, width, height, and cardinality.
    """
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

        # Frozen-open confidence gate (gamma = +10.0, requires_grad = False) (D-3620)
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
    """
    P-1696 / D-3631: Zeroes token_proj and pos_proj so q_i = z_i + 0 = z_i identically.
    Leaves cls_proj and age_embed with valid non-zero parameters and positive bias
    to guarantee active ReLU derivatives (relu' == 1.0) when token_proj opens up.
    """
    with torch.no_grad():
        adapter.token_proj.weight.zero_()
        adapter.token_proj.bias.zero_()
        adapter.pos_proj.weight.zero_()
        adapter.pos_proj.bias.zero_()
        adapter.norm.weight.fill_(1.0)
        adapter.norm.bias.zero_()
        # Initialize cls_proj with positive bias so pre-activations lie in active linear region
        nn.init.kaiming_uniform_(adapter.cls_proj.weight, a=1.0)
        adapter.cls_proj.bias.fill_(0.01)
        nn.init.normal_(adapter.age_embed.weight, std=0.02)
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
# 3. OPERATIONAL EVALUATION HELPER (P-1711: RESCUE + WHOLE-DETECTOR F1)
# ------------------------------------------------------------------------------

def evaluate_frame_operational(model, criterion, samples, targets_dict, eval_targets, label_to_cat_id, memory_bank, adapter, temporal_enabled=True):
    """
    Evaluates:
      1. Canonical monitor loss L_canon(Y_all) over 300 canonical queries.
      2. Memory-Incremental target rescue mapping.
      3. Whole-detector operational TP, FP, FN at s >= 0.20, IoU >= 0.30 over ALL queries.
    """
    out = model(samples, temporal_adapter=adapter, memory_bank=memory_bank, temporal_enabled=temporal_enabled)

    # Monitor loss over 300 canonical queries against Y_all (P-1701)
    out_canon = {
        "pred_logits": out["pred_logits"][:, :300, :],
        "pred_boxes":  out["pred_boxes"][:, :300, :]
    }
    if "aux_outputs" in out:
        out_canon["aux_outputs"] = [
            {"pred_logits": aux["pred_logits"][:, :300, :], "pred_boxes": aux["pred_boxes"][:, :300, :]}
            for aux in out["aux_outputs"]
        ]
    loss_dict_canon = criterion(out_canon, [targets_dict])
    loss_canon = sum(loss_dict_canon[k] * criterion.weight_dict[k] for k in loss_dict_canon if k in criterion.weight_dict).item()

    # Exact Top-100 DETR ranking over ALL active queries (up to 300 + K_mem)
    prob = out["pred_logits"][0].sigmoid()
    num_classes = prob.shape[-1]
    topk_scores, topk_idx = torch.topk(prob.flatten(), 100)
    query_indices = torch.div(topk_idx, num_classes, rounding_mode="floor")
    class_labels = topk_idx % num_classes
    boxes_ranked = out["pred_boxes"][0][query_indices]

    pred_cat_ids = np.array([label_to_cat_id[int(lbl.item())] for lbl in class_labels])

    img_h, img_w = targets_dict["orig_size"][0].item(), targets_dict["orig_size"][1].item()
    pred_x1 = (boxes_ranked[:, 0] - boxes_ranked[:, 2] / 2.0) * img_w
    pred_y1 = (boxes_ranked[:, 1] - boxes_ranked[:, 3] / 2.0) * img_h
    pred_w = boxes_ranked[:, 2] * img_w
    pred_h = boxes_ranked[:, 3] * img_h
    pred_boxes_xywh = torch.stack([pred_x1, pred_y1, pred_w, pred_h], dim=-1).cpu().numpy()
    pred_scores_np = topk_scores.cpu().numpy()

    # 1. MI Target Rescue
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

    # 2. Whole-Detector Operational Metrics (P-1711)
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
    frame_tp = 0
    frame_fp = 0

    for p_idx in range(100):
        if pred_scores_np[p_idx] >= SCORE_THRESHOLD:
            p_box = pred_boxes_xywh[p_idx]
            p_cat = pred_cat_ids[p_idx]

            best_iou = -1.0
            best_g = -1
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

    return loss_canon, rescued_map, frame_tp, frame_fp, frame_fn


# ------------------------------------------------------------------------------
# 4. MAIN FAST-TRACK PIPELINE ENGINE
# ------------------------------------------------------------------------------

def main():
    print("=" * 115)
    print("CANDIDATE 2M FAST-TRACK EXECUTION PIPELINE (D-3626, D-3631)")
    print("Locked Temporal Scope: k in {2, 3, 4, 5} | Auxiliary MI Loss | Screening Seed 41")
    print("=" * 115)

    # Provenance assertions
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

    with open(SPLIT_MANIFEST_JSON, "r", encoding="utf-8") as f:
        split_meta = json.load(f)["_meta"]
    design_vids = sorted(split_meta["train_videos"])
    calib_vids = sorted(split_meta["calib_videos"])
    test_vids = sorted(split_meta["test_videos"])

    print(f"[+] Loaded Clean 10/2/2 Partition ({split_meta['artifact_name']}):")
    print(f"    - Design Videos (10):      {design_vids} ({split_meta['n_train_targets']} targets)")
    print(f"    - Calibration Videos (2):  {calib_vids} ({split_meta['n_calib_targets']} targets)")
    print(f"    - Protected Test (2):      {test_vids} ({split_meta['n_test_targets']} targets) [HELD UNTOUCHED]")

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

    # --------------------------------------------------------------------------
    # STEP 1: STAGE 0D SMOKE TEST ON GENUINE MI MEMORY (P-1709, P-1704)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("STEP 1: STAGE 0D SMOKE TEST ON GENUINE MI MEMORY (TWO-STEP GRADIENT CONTRACT P-1704, P-1709)")
    print("=" * 115)

    with open(CANDIDATES_JSON, "r", encoding="utf-8") as f:
        cand_windows = json.load(f)["windows"]

    v4_mi_windows = [w for w in cand_windows if w["video_id"] == 4 and w.get("n_mem_incr_s20", w.get("n_memincr_s20", 0)) >= 1]
    smoke_window = v4_mi_windows[0]
    smoke_img_id = smoke_window["image_ids"][-1]
    s_smoke = collate_fn([dataset[id_to_idx[smoke_img_id]]])
    samples_smoke = s_smoke[0].to(device)

    # P-1709: Extract genuine cache-compatible memory for smoke test
    smoke_mb = []
    curr_smoke_fid = smoke_window["frame_ids"][-1]
    with torch.no_grad():
        for f_idx in range(5):
            iid_f = smoke_window["image_ids"][f_idx]
            fid_f = smoke_window["frame_ids"][f_idx]
            age = curr_smoke_fid - fid_f
            if age in MEMORY_AGES:
                s_f = collate_fn([dataset[id_to_idx[iid_f]]])
                out_f, dec_tokens = model(s_f[0].to(device), temporal_enabled=False, return_decoder_tokens=True)
                prob_f = out_f["pred_logits"][0].sigmoid()
                topk_scores, topk_idx = torch.topk(prob_f.flatten(), 100)
                query_indices = torch.div(topk_idx, 7, rounding_mode="floor")
                class_labels = topk_idx % 7
                boxes_ranked = out_f["pred_boxes"][0][query_indices]
                for p_rank in range(100):
                    sc = topk_scores[p_rank].item()
                    if sc >= SCORE_THRESHOLD:
                        q_idx = query_indices[p_rank].item()
                        smoke_mb.append({
                            "z": dec_tokens[0, q_idx].cpu(),
                            "bbox": boxes_ranked[p_rank].cpu().numpy().tolist(),
                            "logits": out_f["pred_logits"][0, q_idx].cpu(),
                            "score": sc,
                            "category_idx": class_labels[p_rank].item(),
                            "query_idx": q_idx,
                            "age": age
                        })
                        if sum(1 for tok in smoke_mb if tok["age"] == age) >= MAX_TOKENS_PER_AGE:
                            break

    assert len(smoke_mb) > 0, "[FATAL] Failed to extract genuine memory tokens for Stage 0D smoke test!"
    print(f"[+] Extracted Genuine Memory for Smoke Test: {len(smoke_mb)} tokens spanning ages {sorted(list({tok['age'] for tok in smoke_mb}))} (P-1709).")

    smoke_adapter = HistoricalObjectTokenMemory(d_model=256, num_classes=7, max_age=6).to(device)
    init_identity_preserving(smoke_adapter)
    smoke_adapter.train()

    tgt_smoke_all = {"boxes": s_smoke[1][0]["boxes"].to(device), "labels": s_smoke[1][0]["labels"].to(device)}
    smoke_mi_target = [t for t in smoke_window["targets"] if t.get("mem_incr_s20", t.get("memincr_s20", False))][0]
    ann_s = ann_lookup[smoke_mi_target["ann_id"]]
    img_h, img_w = s_smoke[1][0]["orig_size"][0].item(), s_smoke[1][0]["orig_size"][1].item()
    t_box_norm = torch.tensor([(ann_s["bbox"][0] + ann_s["bbox"][2]/2.0)/img_w, (ann_s["bbox"][1] + ann_s["bbox"][3]/2.0)/img_h, ann_s["bbox"][2]/img_w, ann_s["bbox"][3]/img_h], device=device)
    diffs = (tgt_smoke_all["boxes"] - t_box_norm).abs().sum(dim=-1)
    mi_gt_idx = diffs.argmin().item()

    tgt_smoke_mi = {"boxes": tgt_smoke_all["boxes"][[mi_gt_idx]], "labels": tgt_smoke_all["labels"][[mi_gt_idx]]}

    # Step 0 check (P-1704)
    opt_smoke = torch.optim.AdamW([p for p in smoke_adapter.parameters() if p.requires_grad], lr=1e-4)
    opt_smoke.zero_grad()
    out_smoke_0 = model(samples_smoke, temporal_adapter=smoke_adapter, memory_bank=[smoke_mb], temporal_enabled=True)
    out_temp_0 = {"pred_logits": out_smoke_0["pred_logits"][:, 300:, :], "pred_boxes": out_smoke_0["pred_boxes"][:, 300:, :]}
    if "aux_outputs" in out_smoke_0:
        out_temp_0["aux_outputs"] = [{"pred_logits": a["pred_logits"][:, 300:, :], "pred_boxes": a["pred_boxes"][:, 300:, :]} for a in out_smoke_0["aux_outputs"]]

    loss_dict_s0 = criterion(out_temp_0, [tgt_smoke_mi])
    loss_s0 = LAMBDA_MI * sum(loss_dict_s0[k] * criterion.weight_dict[k] for k in loss_dict_s0 if k in criterion.weight_dict)
    loss_s0.backward()

    assert smoke_adapter.token_proj.weight.grad.norm().item() > 0.0, "[FATAL] token_proj grad == 0 at Step 0!"
    assert smoke_adapter.pos_proj.weight.grad.norm().item() > 0.0, "[FATAL] pos_proj grad == 0 at Step 0!"
    assert smoke_adapter.cls_proj.weight.grad.norm().item() == 0.0, "[FATAL] cls_proj grad != 0 at Step 0!"
    assert smoke_adapter.age_embed.weight.grad.norm().item() == 0.0, "[FATAL] age_embed grad != 0 at Step 0!"
    assert smoke_adapter.norm.weight.grad.norm().item() == 0.0, "[FATAL] norm grad != 0 at Step 0!"
    assert smoke_adapter.gamma.grad is None, "[FATAL] gamma received gradient!"
    print("    [+] Step 0 Invariant PASSED: token_proj & pos_proj > 0; upstream residual features == 0.0.")

    # Step 1 check (P-1704)
    opt_smoke.step()
    opt_smoke.zero_grad()
    out_smoke_1 = model(samples_smoke, temporal_adapter=smoke_adapter, memory_bank=[smoke_mb], temporal_enabled=True)
    out_temp_1 = {"pred_logits": out_smoke_1["pred_logits"][:, 300:, :], "pred_boxes": out_smoke_1["pred_boxes"][:, 300:, :]}
    if "aux_outputs" in out_smoke_1:
        out_temp_1["aux_outputs"] = [{"pred_logits": a["pred_logits"][:, 300:, :], "pred_boxes": a["pred_boxes"][:, 300:, :]} for a in out_smoke_1["aux_outputs"]]

    loss_dict_s1 = criterion(out_temp_1, [tgt_smoke_mi])
    loss_s1 = LAMBDA_MI * sum(loss_dict_s1[k] * criterion.weight_dict[k] for k in loss_dict_s1 if k in criterion.weight_dict)
    loss_s1.backward()

    assert smoke_adapter.token_proj.weight.grad.norm().item() > 0.0, "[FATAL] token_proj grad == 0 at Step 1!"
    assert smoke_adapter.pos_proj.weight.grad.norm().item() > 0.0, "[FATAL] pos_proj grad == 0 at Step 1!"
    assert smoke_adapter.cls_proj.weight.grad.norm().item() > 0.0, "[FATAL] cls_proj grad == 0 at Step 1!"
    assert smoke_adapter.age_embed.weight.grad.norm().item() > 0.0, "[FATAL] age_embed grad == 0 at Step 1!"
    assert smoke_adapter.norm.weight.grad.norm().item() > 0.0, "[FATAL] norm grad == 0 at Step 1!"
    print("    [+] Step 1 Invariant PASSED: cls_proj, age_embed & norm receive non-zero gradients once W_token != 0.")
    print("[+] STAGE 0D SMOKE TEST PASSED (D-3627: Fast-Track Micro-Training Authorized).")

    # --------------------------------------------------------------------------
    # STEP 2: PRECOMPUTE / VERIFY FROZEN HISTORICAL TOKEN CACHE (P-1705)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("STEP 2: PRECOMPUTING / LOADING FROZEN HISTORICAL TOKEN CACHE (P-1705)")
    print("=" * 115)

    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache_data = json.load(f)
    raw_detections = {int(k): v for k, v in cache_data["detections"].items()}

    cache_target_vids = set(design_vids + calib_vids)
    all_needed_iids = set()
    windows_by_vid = defaultdict(list)

    for w in cand_windows:
        v_id = w["video_id"]
        if v_id in cache_target_vids and w.get("n_mem_incr_s20", w.get("n_memincr_s20", 0)) >= 1:
            windows_by_vid[v_id].append(w)
            for iid in w["image_ids"]:
                all_needed_iids.add(iid)

    print(f"[*] Total Windows in Scope: {sum(len(w) for w in windows_by_vid.values())} across {len(cache_target_vids)} videos.")
    print(f"[*] Total Unique Historical Frames to Cache: {len(all_needed_iids)}")

    if TOKEN_CACHE_PTH.exists():
        print(f"[+] Loading existing precomputed token cache: {TOKEN_CACHE_PTH.resolve()}")
        hist_token_cache = torch.load(TOKEN_CACHE_PTH, map_location="cpu")
        print(f"    Loaded {len(hist_token_cache)} frames from cache.")
    else:
        print(f"[*] Precomputing Top-8 tokens (z, bbox, logits, score) with frozen M0...")
        hist_token_cache = {}
        with torch.no_grad():
            for count, iid in enumerate(sorted(list(all_needed_iids)), 1):
                s_frame = collate_fn([dataset[id_to_idx[iid]]])
                out_f, dec_tokens = model(s_frame[0].to(device), temporal_enabled=False, return_decoder_tokens=True)

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
                            "query_idx": q_idx
                        })
                        if len(frame_tokens) >= MAX_TOKENS_PER_AGE:
                            break

                hist_token_cache[iid] = frame_tokens
                if count % 100 == 0 or count == len(all_needed_iids):
                    print(f"    Cached {count}/{len(all_needed_iids)} frames...")

        torch.save(hist_token_cache, TOKEN_CACHE_PTH)
        print(f"[+] Token Cache Saved & Frozen: {TOKEN_CACHE_PTH.resolve()}")

    cache_hash = compute_sha256(TOKEN_CACHE_PTH)
    print(f"    Token Cache SHA-256: {cache_hash}")

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

    # --------------------------------------------------------------------------
    # STEP 3: SEED 41 SCREENING RUN (12 EPOCHS) (D-3628, P-1710, P-1711)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("STEP 3: SEED 41 SCREENING RUN (12 EPOCHS, SPARSE CALIBRATION CHECKPOINTS)")
    print("=" * 115)

    SCREEN_SEED = 41
    set_seed(SCREEN_SEED)

    adapter = HistoricalObjectTokenMemory(d_model=256, num_classes=7, max_age=6).to(device)
    init_identity_preserving(adapter)
    optimizer = torch.optim.AdamW([p for p in adapter.parameters() if p.requires_grad], lr=2e-4, weight_decay=1e-4)

    calib_windows_v33 = windows_by_vid[33]
    calib_windows_v46 = windows_by_vid[46]

    def build_eval_stream(video_windows):
        eval_items = []
        seen_targets = set()
        for w in video_windows:
            curr_iid = w["image_ids"][-1]
            mi_tgts = [t for t in w["targets"] if t.get("mem_incr_s20", t.get("memincr_s20", False))]
            unseen = [t for t in mi_tgts if t["ann_id"] not in seen_targets]
            if unseen:
                for t in unseen: seen_targets.add(t["ann_id"])
                eval_items.append({
                    "window": w,
                    "image_id": curr_iid,
                    "targets": unseen,
                    "eval_targets": [{"ann_id": t["ann_id"], "category_id": ann_lookup[t["ann_id"]]["category_id"], "bbox": ann_lookup[t["ann_id"]]["bbox"]} for t in unseen]
                })
        return eval_items

    stream_calib_33 = build_eval_stream(calib_windows_v33)
    stream_calib_46 = build_eval_stream(calib_windows_v46)
    n_calib_33 = sum(len(x["targets"]) for x in stream_calib_33)
    n_calib_46 = sum(len(x["targets"]) for x in stream_calib_46)
    assert n_calib_33 + n_calib_46 == 44, f"[FATAL] Calibration targets count mismatch: {n_calib_33 + n_calib_46} != 44"

    def run_calibration_evaluation(current_adapter):
        current_adapter.eval()
        rescued_D = 0
        rescued_corr = 0
        rescued_shifted = 0
        canon_losses = []

        total_tp_D = 0; total_fp_D = 0; total_fn_D = 0
        total_tp_corr = 0; total_fp_corr = 0; total_fn_corr = 0

        with torch.no_grad():
            for v_curr, stream_curr in [(33, stream_calib_33), (46, stream_calib_46)]:
                for idx, item in enumerate(stream_curr):
                    s_c = collate_fn([dataset[id_to_idx[item["image_id"]]]])
                    tgt_dict = {"boxes": s_c[1][0]["boxes"].to(device), "labels": s_c[1][0]["labels"].to(device), "orig_size": s_c[1][0]["orig_size"]}

                    mb_corr = get_memory_bank_for_window(item["window"], hist_token_cache)
                    # P-1708: Deterministic spatial-shift counterfactual M_shifted
                    mb_shifted = make_shifted_memory_bank(mb_corr)

                    l_d, r_d, tp_d, fp_d, fn_d = evaluate_frame_operational(model, criterion, s_c[0].to(device), tgt_dict, item["eval_targets"], label_to_cat_id, None, None, False)
                    l_c, r_c, tp_c, fp_c, fn_c = evaluate_frame_operational(model, criterion, s_c[0].to(device), tgt_dict, item["eval_targets"], label_to_cat_id, [mb_corr], current_adapter, True)
                    _,   r_s, _,    _,    _    = evaluate_frame_operational(model, criterion, s_c[0].to(device), tgt_dict, item["eval_targets"], label_to_cat_id, [mb_shifted], current_adapter, True)

                    canon_losses.append(l_c)
                    total_tp_D += tp_d; total_fp_D += fp_d; total_fn_D += fn_d
                    total_tp_corr += tp_c; total_fp_corr += fp_c; total_fn_corr += fn_c

                    for t in item["eval_targets"]:
                        aid = t["ann_id"]
                        if r_d.get(aid, False): rescued_D += 1
                        if r_c.get(aid, False): rescued_corr += 1
                        if r_s.get(aid, False): rescued_shifted += 1

        prec_D = total_tp_D / max(1, total_tp_D + total_fp_D)
        rec_D = total_tp_D / max(1, total_tp_D + total_fn_D)
        f1_D = 2 * prec_D * rec_D / max(1e-7, prec_D + rec_D)

        prec_corr = total_tp_corr / max(1, total_tp_corr + total_fp_corr)
        rec_corr = total_tp_corr / max(1, total_tp_corr + total_fn_corr)
        f1_corr = 2 * prec_corr * rec_corr / max(1e-7, prec_corr + rec_corr)

        current_adapter.train()
        return rescued_D, rescued_corr, rescued_shifted, float(np.mean(canon_losses)), f1_D, f1_corr

    # Evaluate Epoch 0 (Untrained Baseline)
    calib_history = {}
    r0_D, r0_corr, r0_shifted, l0_canon, f1_0_D, f1_0_corr = run_calibration_evaluation(adapter)
    delta_r0 = r0_corr - r0_shifted
    calib_history[0] = {"rescued_D": r0_D, "rescued_corr": r0_corr, "rescued_shifted": r0_shifted, "delta_r": delta_r0, "loss_canon": l0_canon, "f1_D": f1_0_D, "f1_corr": f1_0_corr}
    print(f"  Epoch 0 (Baseline) | Rescues: corr={r0_corr}/44, shifted={r0_shifted}/44, D={r0_D}/44 | Delta_R = {delta_r0:+2d} | F1_corr={f1_0_corr:.4f} (F1_D={f1_0_D:.4f})")

    rng_train = np.random.default_rng(SCREEN_SEED * 1000 + 42)
    best_E_star = 0
    best_delta_r = -999
    best_seed41_state = None

    for epoch in range(1, MAX_SCREEN_EPOCHS + 1):
        adapter.train()
        for vid in design_vids:
            w_list = windows_by_vid[vid]
            chosen_w = w_list[int(rng_train.integers(0, len(w_list)))]

            target_iid = chosen_w["image_ids"][-1]
            s_t = collate_fn([dataset[id_to_idx[target_iid]]])
            samples_t = s_t[0].to(device)

            tgt_all_t = {"boxes": s_t[1][0]["boxes"].to(device), "labels": s_t[1][0]["labels"].to(device)}

            mi_tgts_t = [t for t in chosen_w["targets"] if t.get("mem_incr_s20", t.get("memincr_s20", False))]
            if not mi_tgts_t:
                continue

            img_h, img_w = s_t[1][0]["orig_size"][0].item(), s_t[1][0]["orig_size"][1].item()
            mi_indices = []
            for t_mi in mi_tgts_t:
                ann = ann_lookup[t_mi["ann_id"]]
                bx, by, bw, bh = ann["bbox"]
                t_box = torch.tensor([(bx+bw/2.0)/img_w, (by+bh/2.0)/img_h, bw/img_w, bh/img_h], device=device)
                mi_indices.append((tgt_all_t["boxes"] - t_box).abs().sum(-1).argmin().item())

            tgt_mi_t = {"boxes": tgt_all_t["boxes"][mi_indices], "labels": tgt_all_t["labels"][mi_indices]}
            mb_train = get_memory_bank_for_window(chosen_w, hist_token_cache)
            if not mb_train:
                continue

            optimizer.zero_grad()
            out_tr = model(samples_t, temporal_adapter=adapter, memory_bank=[mb_train], temporal_enabled=True)

            out_temp_tr = {"pred_logits": out_tr["pred_logits"][:, 300:, :], "pred_boxes": out_tr["pred_boxes"][:, 300:, :]}
            if "aux_outputs" in out_tr:
                out_temp_tr["aux_outputs"] = [{"pred_logits": a["pred_logits"][:, 300:, :], "pred_boxes": a["pred_boxes"][:, 300:, :]} for a in out_tr["aux_outputs"]]

            loss_dict_aux = criterion(out_temp_tr, [tgt_mi_t])
            loss_aux = LAMBDA_MI * sum(loss_dict_aux[k] * criterion.weight_dict[k] for k in loss_dict_aux if k in criterion.weight_dict)
            loss_aux.backward()
            optimizer.step()

        if epoch in CALIB_CHECKPOINTS:
            r_D, r_corr, r_shifted, l_canon, f1_D, f1_corr = run_calibration_evaluation(adapter)
            d_r = r_corr - r_shifted
            calib_history[epoch] = {"rescued_D": r_D, "rescued_corr": r_corr, "rescued_shifted": r_shifted, "delta_r": d_r, "loss_canon": l_canon, "f1_D": f1_D, "f1_corr": f1_corr}
            print(f"  Epoch {epoch:<2d} (Checkpointed) | Rescues: corr={r_corr}/44, shifted={r_shifted}/44, D={r_D}/44 | Delta_R = {d_r:+2d} | F1_corr={f1_corr:.4f} (F1_D={f1_D:.4f})")

            # P-1710 / P-1711: Eligible checkpoint selection
            if d_r > 0 and r_corr > r_D and f1_corr >= f1_D:
                if d_r > best_delta_r:
                    best_delta_r = d_r
                    best_E_star = epoch
                    best_seed41_state = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}

    print("\n" + "=" * 115)
    print("STEP 4: CALIBRATION DECISION GATE VERDICT (P-1710, P-1711)")
    print("=" * 115)
    print(f"{'Epoch':<8} | {'Rescued Correct':<18} | {'Rescued Shifted':<18} | {'Rescued D':<14} | {'Delta_R':<12} | {'F1_corr':<10} | {'F1_D':<10}")
    print("-" * 96)
    for e in [0] + CALIB_CHECKPOINTS:
        rec = calib_history[e]
        print(f"Epoch {e:<2d} | {rec['rescued_corr']:2d} / 44 ({rec['rescued_corr']/44*100:4.1f}%)    | {rec['rescued_shifted']:2d} / 44 ({rec['rescued_shifted']/44*100:4.1f}%)    | {rec['rescued_D']:2d} / 44       | {rec['delta_r']:+3d}        | {rec['f1_corr']:.4f}    | {rec['f1_D']:.4f}")

    print(f"\n[*] Calibration Gate Result: E* = {best_E_star} Epochs (Delta_R = {best_delta_r:+d})")

    # --------------------------------------------------------------------------
    # STEP 4B: HARD GATE ENFORCEMENT
    # --------------------------------------------------------------------------
    if best_E_star == 0 or best_delta_r <= 0:
        print("\n" + "=" * 115)
        print("[!] CALIBRATION MECHANISM GATE: FAILED (E* = 0).")
        print("    Candidate 2M failed to earn positive correct-vs-shifted target rescue margin with F1 non-degradation.")
        print("    PROTECTED TEST SET WILL NOT BE TOUCHED. Videos 26 and 40 remain strictly unspent.")
        print("=" * 115)

        results_payload = {
            "_meta": {
                "artifact_name": "pol2_candidate2m_fasttrack_results_v1",
                "status": "CALIBRATION_FAILED_NO_POSITIVE_RESCUE",
                "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
                "token_cache_sha256": cache_hash,
                "optimal_budget_E_star": 0,
                "best_delta_r": best_delta_r
            },
            "calibration_history": calib_history,
            "test_results": {}
        }
        with open(RESULTS_JSON, "w", encoding="utf-8") as f:
            json.dump(results_payload, f, indent=2)
        res_hash = compute_sha256(RESULTS_JSON)
        print(f"[+] Calibration Failure Record Frozen: {RESULTS_JSON.resolve()}")
        print(f"    Record SHA-256:                    {res_hash}\n")
        return

    # --------------------------------------------------------------------------
    # STEP 5: CONDITIONAL REPLICATION & PROTECTED TEST ACCESS (P-1707, D-3634)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print(f"[+] CALIBRATION MECHANISM GATE: PASSED (E* = {best_E_star} Epochs).")
    print("    Authorizing Tri-Seed Replication & Protected Test Evaluation on videos 26 and 40.")
    print("=" * 115)

    tri_seed_models = {}
    tri_seed_models["41"] = best_seed41_state
    print(f"[+] Reusing Frozen Seed 41 State directly at E* = {best_E_star} (No retraining of Seed 41) (D-3634).")

    REPLICATION_SEEDS = [42, 43]
    for s in REPLICATION_SEEDS:
        print(f"\n[*] Training Candidate 2M for Seed {s} for exactly E* = {best_E_star} Epochs...")
        set_seed(s)
        ad_s = HistoricalObjectTokenMemory(d_model=256, num_classes=7, max_age=6).to(device)
        init_identity_preserving(ad_s)
        opt_s = torch.optim.AdamW([p for p in ad_s.parameters() if p.requires_grad], lr=2e-4, weight_decay=1e-4)

        rng_s = np.random.default_rng(s * 1000 + 42)
        for ep in range(1, best_E_star + 1):
            ad_s.train()
            for vid in design_vids:
                w_list = windows_by_vid[vid]
                chosen_w = w_list[int(rng_s.integers(0, len(w_list)))]
                target_iid = chosen_w["image_ids"][-1]
                s_t = collate_fn([dataset[id_to_idx[target_iid]]])
                tgt_all_t = {"boxes": s_t[1][0]["boxes"].to(device), "labels": s_t[1][0]["labels"].to(device)}

                mi_tgts_t = [t for t in chosen_w["targets"] if t.get("mem_incr_s20", t.get("memincr_s20", False))]
                if not mi_tgts_t: continue

                img_h, img_w = s_t[1][0]["orig_size"][0].item(), s_t[1][0]["orig_size"][1].item()
                mi_indices = []
                for t_mi in mi_tgts_t:
                    ann = ann_lookup[t_mi["ann_id"]]
                    bx, by, bw, bh = ann["bbox"]
                    t_box = torch.tensor([(bx+bw/2.0)/img_w, (by+bh/2.0)/img_h, bw/img_w, bh/img_h], device=device)
                    mi_indices.append((tgt_all_t["boxes"] - t_box).abs().sum(-1).argmin().item())

                tgt_mi_t = {"boxes": tgt_all_t["boxes"][mi_indices], "labels": tgt_all_t["labels"][mi_indices]}
                mb_train = get_memory_bank_for_window(chosen_w, hist_token_cache)
                if not mb_train: continue

                opt_s.zero_grad()
                out_tr = model(s_t[0].to(device), temporal_adapter=ad_s, memory_bank=[mb_train], temporal_enabled=True)
                out_temp_tr = {"pred_logits": out_tr["pred_logits"][:, 300:, :], "pred_boxes": out_tr["pred_boxes"][:, 300:, :]}
                if "aux_outputs" in out_tr:
                    out_temp_tr["aux_outputs"] = [{"pred_logits": a["pred_logits"][:, 300:, :], "pred_boxes": a["pred_boxes"][:, 300:, :]} for a in out_tr["aux_outputs"]]

                loss_dict_aux = criterion(out_temp_tr, [tgt_mi_t])
                loss_aux = LAMBDA_MI * sum(loss_dict_aux[k] * criterion.weight_dict[k] for k in loss_dict_aux if k in criterion.weight_dict)
                loss_aux.backward()
                opt_s.step()

        tri_seed_models[str(s)] = {k: v.detach().cpu().clone() for k, v in ad_s.state_dict().items()}
        print(f"[+] Final Training Complete for Seed {s} ({best_E_star} Epochs).")

    torch.save(tri_seed_models, FINAL_MODELS_PTH)
    models_hash = compute_sha256(FINAL_MODELS_PTH)
    print(f"\n[+] All 3 Final Candidate 2M Models Frozen (CPU): {FINAL_MODELS_PTH.resolve()}")
    print(f"    Final Models SHA-256:                         {models_hash}")

    # Build and cache test stream memory tokens for video_26 and video_40
    print("\n[*] Caching test stream memory tokens for protected videos 26 and 40...")
    test_windows_v26 = [w for w in cand_windows if w["video_id"] == 26 and w.get("n_mem_incr_s20", w.get("n_memincr_s20", 0)) >= 1]
    test_windows_v40 = [w for w in cand_windows if w["video_id"] == 40 and w.get("n_mem_incr_s20", w.get("n_memincr_s20", 0)) >= 1]

    with torch.no_grad():
        for w_list in [test_windows_v26, test_windows_v40]:
            for w in w_list:
                for iid in w["image_ids"]:
                    if iid not in hist_token_cache:
                        s_f = collate_fn([dataset[id_to_idx[iid]]])
                        out_f, dec_tokens = model(s_f[0].to(device), temporal_enabled=False, return_decoder_tokens=True)
                        prob_f = out_f["pred_logits"][0].sigmoid()
                        num_classes = prob_f.shape[-1]
                        topk_scores, topk_idx = torch.topk(prob_f.flatten(), 100)
                        query_indices = torch.div(topk_idx, num_classes, rounding_mode="floor")
                        class_labels = topk_idx % num_classes
                        boxes_ranked = out_f["pred_boxes"][0][query_indices]

                        f_tokens = []
                        for p_rank in range(100):
                            sc = topk_scores[p_rank].item()
                            if sc >= SCORE_THRESHOLD:
                                q_idx = query_indices[p_rank].item()
                                f_tokens.append({
                                    "z": dec_tokens[0, q_idx].cpu(),
                                    "bbox": boxes_ranked[p_rank].cpu().numpy().tolist(),
                                    "logits": out_f["pred_logits"][0, q_idx].cpu(),
                                    "score": sc,
                                    "category_idx": class_labels[p_rank].item(),
                                    "query_idx": q_idx
                                })
                                if len(f_tokens) >= MAX_TOKENS_PER_AGE: break
                        hist_token_cache[iid] = f_tokens

    stream_test_26 = build_eval_stream(test_windows_v26)
    stream_test_40 = build_eval_stream(test_windows_v40)
    n_test_26 = sum(len(x["targets"]) for x in stream_test_26)
    n_test_40 = sum(len(x["targets"]) for x in stream_test_40)
    assert n_test_26 + n_test_40 == 61, f"[FATAL] Test targets count mismatch: {n_test_26 + n_test_40} != 61"

    print(f"[+] Loaded Protected Test Streams: video_26 ({n_test_26} targets), video_40 ({n_test_40} targets)")

    test_results_per_seed = {}
    seed_rescued_corr = defaultdict(set)
    seed_rescued_shifted = defaultdict(set)

    ALL_SEEDS = [41, 42, 43]
    for s in ALL_SEEDS:
        ad_eval = HistoricalObjectTokenMemory(d_model=256, num_classes=7, max_age=6).to(device)
        ad_eval.load_state_dict(tri_seed_models[str(s)])
        ad_eval.eval()

        seed_res = {}
        with torch.no_grad():
            for v_curr, stream_curr in [(26, stream_test_26), (40, stream_test_40)]:
                r_D, r_corr, r_shifted = 0, 0, 0
                tp_d = 0; fp_d = 0; fn_d = 0
                tp_c = 0; fp_c = 0; fn_c = 0

                for idx, item in enumerate(stream_curr):
                    s_c = collate_fn([dataset[id_to_idx[item["image_id"]]]])
                    tgt_dict = {"boxes": s_c[1][0]["boxes"].to(device), "labels": s_c[1][0]["labels"].to(device), "orig_size": s_c[1][0]["orig_size"]}

                    mb_corr = get_memory_bank_for_window(item["window"], hist_token_cache)
                    mb_shifted = make_shifted_memory_bank(mb_corr)

                    _, res_d, tpd, fpd, fnd = evaluate_frame_operational(model, criterion, s_c[0].to(device), tgt_dict, item["eval_targets"], label_to_cat_id, None, None, False)
                    _, res_c, tpc, fpc, fnc = evaluate_frame_operational(model, criterion, s_c[0].to(device), tgt_dict, item["eval_targets"], label_to_cat_id, [mb_corr], ad_eval, True)
                    _, res_s, _,   _,   _   = evaluate_frame_operational(model, criterion, s_c[0].to(device), tgt_dict, item["eval_targets"], label_to_cat_id, [mb_shifted], ad_eval, True)

                    tp_d += tpd; fp_d += fpd; fn_d += fnd
                    tp_c += tpc; fp_c += fpc; fn_c += fnc

                    for t in item["eval_targets"]:
                        aid = t["ann_id"]
                        if res_d.get(aid, False): r_D += 1
                        if res_c.get(aid, False):
                            r_corr += 1
                            seed_rescued_corr[s].add(aid)
                        if res_s.get(aid, False):
                            r_shifted += 1
                            seed_rescued_shifted[s].add(aid)

                p_d = tp_d / max(1, tp_d + fp_d); rec_d = tp_d / max(1, tp_d + fn_d)
                f1_d = 2 * p_d * rec_d / max(1e-7, p_d + rec_d)
                p_c = tp_c / max(1, tp_c + fp_c); rec_c = tp_c / max(1, tp_c + fn_c)
                f1_c = 2 * p_c * rec_c / max(1e-7, p_c + rec_c)

                seed_res[v_curr] = {
                    "rescued_D": r_D, "rescued_corr": r_corr, "rescued_shifted": r_shifted,
                    "f1_D": f1_d, "f1_corr": f1_c
                }

        test_results_per_seed[s] = seed_res
        print(f"  Seed {s} Protected Test | v26: corr={seed_res[26]['rescued_corr']}/{n_test_26}, shifted={seed_res[26]['rescued_shifted']}/{n_test_26} | v40: corr={seed_res[40]['rescued_corr']}/{n_test_40}, shifted={seed_res[40]['rescued_shifted']}/{n_test_40}")

    mean_corr = np.mean([len(seed_rescued_corr[s]) for s in ALL_SEEDS])
    mean_shifted = np.mean([len(seed_rescued_shifted[s]) for s in ALL_SEEDS])
    all_test_aids = {t["ann_id"] for item in stream_test_26 + stream_test_40 for t in item["targets"]}
    corr_freq = {aid: sum(1 for s in ALL_SEEDS if aid in seed_rescued_corr[s]) for aid in all_test_aids}

    print("\n" + "=" * 115)
    print("FINAL CANDIDATE 2M PROTECTED MECHANISM TEST REPORT (N = 61 TARGETS)")
    print("=" * 115)
    print(f"Tri-Seed Rescue Means: Correct = {mean_corr:.2f} / 61 ({mean_corr/61*100:.1f}%) | Shifted = {mean_shifted:.2f} / 61 ({mean_shifted/61*100:.1f}%) | Delta_R = {mean_corr - mean_shifted:+.2f}")
    print(f"Annotation Consistency across Seeds: In >=1 seed = {sum(1 for c in corr_freq.values() if c >= 1)} | In 2/3 seeds = {sum(1 for c in corr_freq.values() if c >= 2)} | In 3/3 seeds = {sum(1 for c in corr_freq.values() if c == 3)}")

    results_payload = {
        "_meta": {
            "artifact_name": "pol2_candidate2m_fasttrack_results_v1",
            "status": "COMPLETED",
            "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
            "token_cache_sha256": cache_hash,
            "final_models_sha256": models_hash,
            "optimal_budget_E_star": best_E_star,
            "mean_rescued_correct": float(mean_corr),
            "mean_rescued_shifted": float(mean_shifted),
            "delta_rescue_mean": float(mean_corr - mean_shifted)
        },
        "calibration_history": calib_history,
        "test_results_per_seed": test_results_per_seed,
        "test_target_consistency": {
            "in_any_seed": sum(1 for c in corr_freq.values() if c >= 1),
            "in_2_of_3": sum(1 for c in corr_freq.values() if c >= 2),
            "in_all_3": sum(1 for c in corr_freq.values() if c == 3)
        }
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    res_hash = compute_sha256(RESULTS_JSON)
    print("\n" + "=" * 115)
    print(f"[+] Candidate 2M Final Fast-Track Results Frozen: {RESULTS_JSON.resolve()}")
    print(f"    Results Manifest SHA-256:                    {res_hash}")
    print("=" * 115)


if __name__ == "__main__":
    main()