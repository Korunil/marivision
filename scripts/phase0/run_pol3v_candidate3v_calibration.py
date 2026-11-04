"""
D:/MARIVISION/H0_workbench/eval_tools/run_pol3v_candidate3v_calibration.py
Candidate 3V Formal Calibration Driver: Reused Development Videos 33 & 46 (P-1771, P-1772, P-1774)

Strict Compliance Checklist:
  - D-3702..D-3707: Evaluates exact frozen Seed-41 verifier artifact at tau* = 0.80 on videos 33 & 46.
  - Chronological Alignment: Explicitly sorts both calibration streams by current frame ID before modulo pairing.
  - P-1772: Formal correspondence-control branch is the pre-established cross-video history swap (M_wrong).
  - P-1773: Coordinate shift (+0.25 mod 1.0) retained as secondary descriptive diagnostic only.
  - P-1774: Asserts exact 15-feature schema/order match and finite scaler parameters from verifier artifact.
  - Formal Gate Criteria: F1_corr > F1_D (0.469388), R_corr > R_wrong, R_corr > R_D (0).
  - D-3705: Gate evaluated on pooled 33+46; per-video results reported as mandatory descriptive diagnostics.
  - Protected Test Protection: Videos 26 and 40 remain untouched and unspent.
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
VERIFIER_PTH = OUT_DIR / "pol3v_candidate3v_seed41_verifier.pth"
M0_CKPT_PATH = Path("D:/MARIVISION/checkpoints_r0/checkpoint_best.pth")
ZIP_PATH = Path("D:/MARIVISION/SeaClips.zip")

CALIB_RESULTS_JSON = OUT_DIR / "pol3v_candidate3v_calibration_results.json"

EXPECTED_CANDIDATE_SHA256 = "f29e0b8a5cbfc46641222d84a6d370c3e9bc51ca4ca1124b7753078ca7c1e270"
EXPECTED_TRAIN_JSON_SHA256 = "edc32bdbd6b792eb83b6daf11e14708232f1f4a9e06153ebebd5bad1caef9b35"
EXPECTED_CACHE_SHA256 = "d9816083760d698935b44e9165522e709b490c3af29e710d75b8b0d2d90210e0"
EXPECTED_DEV_POOL_SHA256 = "533567e1d74544db7087a95253e640ef0959b998e5d503f57b0eb8074d8fe970"
EXPECTED_SPLIT_SHA256 = "7da471d2793fe3369508200dec50422bb76492860c67a1a61871645471856772"
EXPECTED_TOKEN_CACHE_SHA256 = "d1e6407722c59968f87e0ae9cfdead28cacf89924dcd765ca41cf8993da4dd4c"
EXPECTED_EPOCH8_ADAPTER_SHA256 = "aea024e2c1072fb5f46d1d376ca1b75c0cdab81a20681ac9958b5d6c0b5f00cf"
EXPECTED_VERIFIER_SHA256 = "2103e6eb6704e2880ea68e9f2cf835e1ca19ebc7faba16299db472837a0a3654"
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


def make_shifted_memory_bank(mb):
    shifted = []
    for tok in mb:
        t_s = copy.deepcopy(tok)
        cx, cy, w, h = t_s["bbox"]
        t_s["bbox"] = [(cx + 0.25) % 1.0, (cy + 0.25) % 1.0, w, h]
        shifted.append(t_s)
    return shifted


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
    return tp, fp, fn, pred_scores, pred_cat_ids, pred_boxes_px


# ------------------------------------------------------------------------------
# 3. MAIN CANDIDATE 3V CALIBRATION ENGINE (P-1771, P-1772, P-1774)
# ------------------------------------------------------------------------------

def main():
    driver_sha256 = compute_sha256(Path(__file__))
    print("=" * 115)
    print("CANDIDATE 3V FORMAL CALIBRATION EXECUTION: VIDEOS 33 & 46 (P-1771, P-1772)")
    print(f"Driver Script SHA-256: {driver_sha256}")
    print("=" * 115)

    for p_path in [CANDIDATES_JSON, TRAIN_JSON, CACHE_FILE, DEV_POOL_JSON, SPLIT_MANIFEST_JSON, TOKEN_CACHE_PTH, EPOCH8_MODEL_PTH, VERIFIER_PTH, M0_CKPT_PATH]:
        assert p_path.exists(), f"[FATAL] Missing {p_path}"

    assert compute_sha256(CANDIDATES_JSON) == EXPECTED_CANDIDATE_SHA256
    assert compute_sha256(TRAIN_JSON) == EXPECTED_TRAIN_JSON_SHA256
    assert compute_sha256(CACHE_FILE) == EXPECTED_CACHE_SHA256
    assert compute_sha256(DEV_POOL_JSON) == EXPECTED_DEV_POOL_SHA256
    assert compute_sha256(SPLIT_MANIFEST_JSON) == EXPECTED_SPLIT_SHA256
    assert compute_sha256(TOKEN_CACHE_PTH) == EXPECTED_TOKEN_CACHE_SHA256
    assert compute_sha256(EPOCH8_MODEL_PTH) == EXPECTED_EPOCH8_ADAPTER_SHA256
    assert compute_sha256(VERIFIER_PTH) == EXPECTED_VERIFIER_SHA256
    assert compute_sha256(M0_CKPT_PATH) == EXPECTED_M0_HASH

    print("[+] All Upstream Manifests, Token Cache, Epoch-8 Adapter, and Deployment Verifier Certified.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Target: {device} ({torch.cuda.get_device_name(0)})")

    # Load verifier payload and assert deployment contract (P-1774)
    verifier_payload = torch.load(VERIFIER_PTH, map_location="cpu")
    assert verifier_payload.get("seed") == 41, "[FATAL] Verifier seed mismatch!"
    assert verifier_payload.get("tau_star") == 0.80, "[FATAL] Verifier tau_star mismatch!"
    assert verifier_payload.get("temporal_prefilter_score") == 0.20, "[FATAL] Prefilter contract mismatch!"
    assert verifier_payload.get("verifier_applies_to") == "threshold_active_temporal_queries_only", "[FATAL] Contract scope mismatch!"

    # P-1774: Explicit verifier feature-schema and scaler assertions
    assert verifier_payload["feature_names"] == FEATURE_NAMES, "[FATAL] Feature names/order mismatch!"
    v_mean = np.array(verifier_payload["mean"], dtype=np.float32)
    v_std = np.array(verifier_payload["std"], dtype=np.float32)
    assert len(v_mean) == len(v_std) == len(FEATURE_NAMES) == 15, "[FATAL] Scaler length mismatch!"
    assert np.isfinite(v_mean).all() and np.isfinite(v_std).all(), "[FATAL] Non-finite scaler parameters!"

    tau_star = verifier_payload["tau_star"]
    prefilter_threshold = verifier_payload["temporal_prefilter_score"]
    clf_verifier = verifier_payload["clf_model"]

    print(f"[+] Loaded Frozen Candidate-3V Verifier Artifact (SHA: {EXPECTED_VERIFIER_SHA256[:16]}...):")
    print(f"    - Hard Operating Point: tau* = {tau_star:.2f}")
    print(f"    - Temporal Prefilter:   max_score >= {prefilter_threshold:.2f}")
    print(f"    - Feature Schema:       15 features certified matching FEATURE_NAMES exactly.")

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
    calib_vids = sorted(split_meta["calib_videos"])
    assert calib_vids == [33, 46], f"[FATAL] Calibration videos mismatch: {calib_vids} != [33, 46]"

    print(f"[+] Loaded Reused Development Calibration Partition: {calib_vids} ({split_meta['n_calib_targets']} targets)")
    print(f"    (Protected Test Videos 26 & 40 STRICTLY HELD UNTOUCHED AND UNSPENT)")

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
                assert w["image_ids"] == primary_w["image_ids"], "[FATAL] History image_ids mismatch across merged windows!"
                assert w["frame_ids"] == primary_w["frame_ids"], "[FATAL] History frame_ids mismatch across merged windows!"
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

    # Explicit chronological sorting by current target frame ID before pairing
    stream_calib_33.sort(key=lambda x: x["window"]["frame_ids"][-1])
    stream_calib_46.sort(key=lambda x: x["window"]["frame_ids"][-1])

    assert len(stream_calib_33) == 2, f"Expected 2 unique frames in v33, got {len(stream_calib_33)}"
    assert len(stream_calib_46) == 42, f"Expected 42 unique frames in v46, got {len(stream_calib_46)}"
    assert sum(len(x["eval_targets"]) for x in stream_calib_33) + sum(len(x["eval_targets"]) for x in stream_calib_46) == 44

    # P-1772: Pre-established cross-video history swap pairing between videos 33 and 46
    paired_stream_map = {
        33: stream_calib_46,
        46: stream_calib_33
    }

    print(f"\n[*] Evaluating 44 Chronologically Ordered Unique Calibration Frames:")
    print(f"    - video_33:  2 frames (Wrong-History paired with video_46 [0..1 % 42])")
    print(f"    - video_46: 42 frames (Wrong-History paired with video_33 [0..41 % 2])")

    results_by_vid = {
        33: {
            "D": {"tp": 0, "fp": 0, "fn": 0, "rescued": 0},
            "corr": {"tp": 0, "fp": 0, "fn": 0, "rescued": 0, "admitted": 0, "total_temp": 0},
            "wrong": {"tp": 0, "fp": 0, "fn": 0, "rescued": 0, "admitted": 0, "total_temp": 0},
            "shifted": {"tp": 0, "fp": 0, "fn": 0, "rescued": 0, "admitted": 0, "total_temp": 0}
        },
        46: {
            "D": {"tp": 0, "fp": 0, "fn": 0, "rescued": 0},
            "corr": {"tp": 0, "fp": 0, "fn": 0, "rescued": 0, "admitted": 0, "total_temp": 0},
            "wrong": {"tp": 0, "fp": 0, "fn": 0, "rescued": 0, "admitted": 0, "total_temp": 0},
            "shifted": {"tp": 0, "fp": 0, "fn": 0, "rescued": 0, "admitted": 0, "total_temp": 0}
        },
    }

    with torch.no_grad():
        for vid, stream_curr in [(33, stream_calib_33), (46, stream_calib_46)]:
            paired_stream = paired_stream_map[vid]

            for item_idx, item in enumerate(stream_curr):
                curr_iid = item["image_id"]
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

                # --------------------------------------------------------------
                # Branch 1: Baseline D (300 Canonical Queries)
                # --------------------------------------------------------------
                out_d = model(samples, temporal_enabled=False)
                tpd, fpd, fnd, scores_d, cats_d, boxes_d = evaluate_operational_top100(
                    out_d["pred_logits"][0], out_d["pred_boxes"][0], gt_boxes_px, gt_cats, label_to_cat_id, img_w, img_h
                )
                results_by_vid[vid]["D"]["tp"] += tpd
                results_by_vid[vid]["D"]["fp"] += fpd
                results_by_vid[vid]["D"]["fn"] += fnd

                for t in item["eval_targets"]:
                    aid = t["ann_id"]
                    gt_box = t["bbox"]
                    gt_cat = t["category_id"]
                    is_resc = False
                    for p_i in range(len(scores_d)):
                        if scores_d[p_i] >= SCORE_THRESHOLD and cats_d[p_i] == gt_cat:
                            if box_iou(gt_box, boxes_d[p_i]) >= EVAL_IOU_THRESHOLD:
                                is_resc = True
                                break
                    if is_resc:
                        results_by_vid[vid]["D"]["rescued"] += 1

                # Pure baseline canonical outputs for canonical_support
                d_boxes_norm = out_d["pred_boxes"][0]
                base_d_boxes_px = []
                for b in d_boxes_norm:
                    cx, cy, w, h = b.tolist()
                    base_d_boxes_px.append([(cx - w / 2.0) * img_w, (cy - h / 2.0) * img_h, w * img_w, h * img_h])
                base_d_logits = out_d["pred_logits"][0]

                # Helper to evaluate temporal memory bank under frozen verifier
                def evaluate_temporal_memory_bank(mb_input):
                    K_mem = len(mb_input)
                    if K_mem == 0:
                        joint_logits = out_d["pred_logits"][0]
                        joint_boxes = out_d["pred_boxes"][0]
                        tp_b, fp_b, fn_b, scores_b, cats_b, boxes_b = evaluate_operational_top100(
                            joint_logits, joint_boxes, gt_boxes_px, gt_cats, label_to_cat_id, img_w, img_h
                        )
                        return tp_b, fp_b, fn_b, scores_b, cats_b, boxes_b, 0, 0

                    out_aug, hs_last = model(samples, temporal_adapter=adapter, memory_bank=[mb_input], temporal_enabled=True, return_decoder_tokens=True)
                    temp_logits = out_aug["pred_logits"][0, 300:]
                    temp_boxes = out_aug["pred_boxes"][0, 300:]
                    temp_hs = hs_last[0, 300:]

                    gated_temp_logits = temp_logits.clone()
                    admitted_count = 0

                    for q_t in range(K_mem):
                        tok = mb_input[q_t]
                        h_i_t = temp_hs[q_t]
                        b_i_t = temp_boxes[q_t]
                        l_i_t = temp_logits[q_t]

                        feat_vec, box_px, pred_class, max_s = extract_query_evidence_features(
                            tok, h_i_t, b_i_t, l_i_t, base_d_boxes_px, base_d_logits, img_w, img_h
                        )

                        # Enforce temporal prefilter contract
                        if max_s < prefilter_threshold:
                            gated_temp_logits[q_t, :] = -1e9
                        else:
                            feat_std = (np.array(feat_vec, dtype=np.float32) - v_mean) / v_std
                            p_score = float(clf_verifier.predict_proba(feat_std.reshape(1, -1))[0, 1])
                            if p_score >= tau_star:
                                admitted_count += 1
                            else:
                                gated_temp_logits[q_t, :] = -1e9

                    joint_logits = torch.cat([out_d["pred_logits"][0], gated_temp_logits], dim=0)
                    joint_boxes = torch.cat([out_d["pred_boxes"][0], temp_boxes], dim=0)

                    tp_b, fp_b, fn_b, scores_b, cats_b, boxes_b = evaluate_operational_top100(
                        joint_logits, joint_boxes, gt_boxes_px, gt_cats, label_to_cat_id, img_w, img_h
                    )
                    return tp_b, fp_b, fn_b, scores_b, cats_b, boxes_b, admitted_count, K_mem

                # --------------------------------------------------------------
                # Branch 2: Candidate 3V Correct Memory (M_corr)
                # --------------------------------------------------------------
                mb_corr = get_memory_bank_for_window(item["window"], hist_token_cache)
                tpc, fpc, fnc, sc_c, cats_c, boxes_c, adm_c, tot_c = evaluate_temporal_memory_bank(mb_corr)

                results_by_vid[vid]["corr"]["tp"] += tpc
                results_by_vid[vid]["corr"]["fp"] += fpc
                results_by_vid[vid]["corr"]["fn"] += fnc
                results_by_vid[vid]["corr"]["admitted"] += adm_c
                results_by_vid[vid]["corr"]["total_temp"] += tot_c

                for t in item["eval_targets"]:
                    aid = t["ann_id"]
                    gt_box = t["bbox"]
                    gt_cat = t["category_id"]
                    is_resc = False
                    for p_i in range(len(sc_c)):
                        if sc_c[p_i] >= SCORE_THRESHOLD and cats_c[p_i] == gt_cat:
                            if box_iou(gt_box, boxes_c[p_i]) >= EVAL_IOU_THRESHOLD:
                                is_resc = True
                                break
                    if is_resc:
                        results_by_vid[vid]["corr"]["rescued"] += 1

                # --------------------------------------------------------------
                # Branch 3: Pre-Established Cross-Video History Swap (M_wrong) [FORMAL GATE]
                # --------------------------------------------------------------
                paired_item = paired_stream[item_idx % len(paired_stream)]
                mb_wrong = get_memory_bank_for_window(paired_item["window"], hist_token_cache)
                tpw, fpw, fnw, sc_w, cats_w, boxes_w, adm_w, tot_w = evaluate_temporal_memory_bank(mb_wrong)

                results_by_vid[vid]["wrong"]["tp"] += tpw
                results_by_vid[vid]["wrong"]["fp"] += fpw
                results_by_vid[vid]["wrong"]["fn"] += fnw
                results_by_vid[vid]["wrong"]["admitted"] += adm_w
                results_by_vid[vid]["wrong"]["total_temp"] += tot_w

                for t in item["eval_targets"]:
                    aid = t["ann_id"]
                    gt_box = t["bbox"]
                    gt_cat = t["category_id"]
                    is_resc = False
                    for p_i in range(len(sc_w)):
                        if sc_w[p_i] >= SCORE_THRESHOLD and cats_w[p_i] == gt_cat:
                            if box_iou(gt_box, boxes_w[p_i]) >= EVAL_IOU_THRESHOLD:
                                is_resc = True
                                break
                    if is_resc:
                        results_by_vid[vid]["wrong"]["rescued"] += 1

                # --------------------------------------------------------------
                # Branch 4: Spatial-Shift Counterfactual (M_shifted) [DESCRIPTIVE]
                # --------------------------------------------------------------
                mb_shifted = make_shifted_memory_bank(mb_corr)
                tps, fps, fns, sc_s, cats_s, boxes_s, adm_s, tot_s = evaluate_temporal_memory_bank(mb_shifted)

                results_by_vid[vid]["shifted"]["tp"] += tps
                results_by_vid[vid]["shifted"]["fp"] += fps
                results_by_vid[vid]["shifted"]["fn"] += fns
                results_by_vid[vid]["shifted"]["admitted"] += adm_s
                results_by_vid[vid]["shifted"]["total_temp"] += tot_s

                for t in item["eval_targets"]:
                    aid = t["ann_id"]
                    gt_box = t["bbox"]
                    gt_cat = t["category_id"]
                    is_resc = False
                    for p_i in range(len(sc_s)):
                        if sc_s[p_i] >= SCORE_THRESHOLD and cats_s[p_i] == gt_cat:
                            if box_iou(gt_box, boxes_s[p_i]) >= EVAL_IOU_THRESHOLD:
                                is_resc = True
                                break
                    if is_resc:
                        results_by_vid[vid]["shifted"]["rescued"] += 1

    # --------------------------------------------------------------------------
    # 4. RESULTS AGGREGATION & FORMAL GATE VERDICT
    # --------------------------------------------------------------------------
    def calc_f1(tp, fp, fn):
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        return p, r, 2 * p * r / max(1e-7, p + r)

    pooled_D = {k: results_by_vid[33]["D"][k] + results_by_vid[46]["D"][k] for k in ["tp", "fp", "fn", "rescued"]}
    pooled_corr = {k: results_by_vid[33]["corr"][k] + results_by_vid[46]["corr"][k] for k in ["tp", "fp", "fn", "rescued", "admitted", "total_temp"]}
    pooled_wrong = {k: results_by_vid[33]["wrong"][k] + results_by_vid[46]["wrong"][k] for k in ["tp", "fp", "fn", "rescued", "admitted", "total_temp"]}
    pooled_shifted = {k: results_by_vid[33]["shifted"][k] + results_by_vid[46]["shifted"][k] for k in ["tp", "fp", "fn", "rescued", "admitted", "total_temp"]}

    p_d, r_d, f1_d = calc_f1(pooled_D["tp"], pooled_D["fp"], pooled_D["fn"])
    p_c, r_c, f1_c = calc_f1(pooled_corr["tp"], pooled_corr["fp"], pooled_corr["fn"])
    p_w, r_w, f1_w = calc_f1(pooled_wrong["tp"], pooled_wrong["fp"], pooled_wrong["fn"])
    p_s, r_s, f1_s = calc_f1(pooled_shifted["tp"], pooled_shifted["fp"], pooled_shifted["fn"])

    # Strict assertion that baseline D reproduces exact established calibration baseline
    assert pooled_D["tp"] == 46 and pooled_D["fp"] == 58 and pooled_D["fn"] == 46, f"[FATAL] Calibration baseline mismatch: TP={pooled_D['tp']}, FP={pooled_D['fp']}, FN={pooled_D['fn']}"
    assert abs(f1_d - 0.469388) < 1e-5, f"[FATAL] Calibration baseline F1 mismatch: {f1_d:.6f} != 0.469388"

    # Predeclared Calibration Gate Evaluation against M_wrong (P-1772, D-3705)
    f1_gate_passed = f1_c > f1_d
    corr_margin_passed = pooled_corr["rescued"] > pooled_wrong["rescued"]
    rescue_over_D_passed = pooled_corr["rescued"] > pooled_D["rescued"]
    calib_verdict_passed = f1_gate_passed and corr_margin_passed and rescue_over_D_passed

    failure_reasons = []
    if not f1_gate_passed: failure_reasons.append("FAILED_F1_GUARDRAIL")
    if not corr_margin_passed: failure_reasons.append("NO_CORRESPONDENCE_MARGIN_VS_WRONG")
    if not rescue_over_D_passed: failure_reasons.append("NO_RESCUE_OVER_BASELINE")
    final_status = "CALIBRATION_PASSED" if calib_verdict_passed else f"CALIBRATION_FAILED_{'&'.join(failure_reasons)}"

    print("\n" + "=" * 115)
    print("CANDIDATE 3V FORMAL CALIBRATION VERDICT (POOLED 33 & 46, N = 44 TARGETS)")
    print("=" * 115)
    print(f"{'Evaluation Branch':<34} | {'Admitted / Total Temp':<24} | {'Rescued MI Targets':<20} | {'TP':<5} | {'FP':<5} | {'FN':<5} | {'Precision':<10} | {'Recall':<10} | {'F1 Score'}")
    print("-" * 140)
    print(f"{'Baseline M0 (D)':<34} | {'-':<24} | {pooled_D['rescued']:2d} / 44 ( 0.0%)          | {pooled_D['tp']:3d} | {pooled_D['fp']:3d} | {pooled_D['fn']:3d} | {p_d:9.4f}  | {r_d:9.4f}  | {f1_d:8.6f}")
    print(f"{'Candidate 3V (M_corr)':<34} | {pooled_corr['admitted']:3d} / {pooled_corr['total_temp']:<18d} | {pooled_corr['rescued']:2d} / 44 ({pooled_corr['rescued']/44*100:4.1f}%)          | {pooled_corr['tp']:3d} | {pooled_corr['fp']:3d} | {pooled_corr['fn']:3d} | {p_c:9.4f}  | {r_c:9.4f}  | {f1_c:8.6f}")
    print(f"{'Cross-Video Control (M_wrong) [GATE]':<34} | {pooled_wrong['admitted']:3d} / {pooled_wrong['total_temp']:<18d} | {pooled_wrong['rescued']:2d} / 44 ({pooled_wrong['rescued']/44*100:4.1f}%)          | {pooled_wrong['tp']:3d} | {pooled_wrong['fp']:3d} | {pooled_wrong['fn']:3d} | {p_w:9.4f}  | {r_w:9.4f}  | {f1_w:8.6f}")
    print(f"{'Spatial Shift Control (M_shifted)':<34} | {pooled_shifted['admitted']:3d} / {pooled_shifted['total_temp']:<18d} | {pooled_shifted['rescued']:2d} / 44 ({pooled_shifted['rescued']/44*100:4.1f}%)          | {pooled_shifted['tp']:3d} | {pooled_shifted['fp']:3d} | {pooled_shifted['fn']:3d} | {p_s:9.4f}  | {r_s:9.4f}  | {f1_s:8.6f}")

    print("\n[*] Mandatory Predeclared Calibration Gate Evaluation:")
    print(f"    1. Operational Quality Gate: F1_corr ({f1_c:.6f}) > F1_D ({f1_d:.6f})             ──> {'PASSED' if f1_gate_passed else 'FAILED'}")
    print(f"    2. Spatial Correspondence:   R_corr ({pooled_corr['rescued']}) > R_wrong ({pooled_wrong['rescued']})                 ──> {'PASSED' if corr_margin_passed else 'FAILED'}")
    print(f"    3. Target Rescue over M0:    R_corr ({pooled_corr['rescued']}) > R_D ({pooled_D['rescued']})                       ──> {'PASSED' if rescue_over_D_passed else 'FAILED'}")
    print(f"\n[+] FINAL CALIBRATION VERDICT: {final_status}")

    # --------------------------------------------------------------------------
    # 5. DESCRIPTIVE PER-VIDEO BREAKDOWN (VIDEOS 33 & 46)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("MANDATORY DESCRIPTIVE PER-VIDEO DIAGNOSTICS (D-3705)")
    print("=" * 115)
    print(f"{'Video ID':<10} | {'Frames':<8} | {'Branch':<12} | {'Admitted / Total':<18} | {'Rescues':<14} | {'TP':<5} | {'FP':<5} | {'FN':<5} | {'Precision':<10} | {'Recall':<10} | {'F1 Score'}")
    print("-" * 120)

    per_video_diagnostic = {}
    for vid, n_fr in [(33, 2), (46, 42)]:
        p_d_v, r_d_v, f1_d_v = calc_f1(results_by_vid[vid]["D"]["tp"], results_by_vid[vid]["D"]["fp"], results_by_vid[vid]["D"]["fn"])
        p_c_v, r_c_v, f1_c_v = calc_f1(results_by_vid[vid]["corr"]["tp"], results_by_vid[vid]["corr"]["fp"], results_by_vid[vid]["corr"]["fn"])
        p_w_v, r_w_v, f1_w_v = calc_f1(results_by_vid[vid]["wrong"]["tp"], results_by_vid[vid]["wrong"]["fp"], results_by_vid[vid]["wrong"]["fn"])
        p_s_v, r_s_v, f1_s_v = calc_f1(results_by_vid[vid]["shifted"]["tp"], results_by_vid[vid]["shifted"]["fp"], results_by_vid[vid]["shifted"]["fn"])
        n_mi = 2 if vid == 33 else 42

        print(f"video_{vid:<4d} | {n_fr:<8d} | {'D':<12} | {'-':<18} | {results_by_vid[vid]['D']['rescued']:2d} / {n_mi:2d}       | {results_by_vid[vid]['D']['tp']:3d} | {results_by_vid[vid]['D']['fp']:3d} | {results_by_vid[vid]['D']['fn']:3d} | {p_d_v:9.4f}  | {r_d_v:9.4f}  | {f1_d_v:8.6f}")
        print(f"video_{vid:<4d} | {n_fr:<8d} | {'M_corr':<12} | {results_by_vid[vid]['corr']['admitted']:3d} / {results_by_vid[vid]['corr']['total_temp']:<12d} | {results_by_vid[vid]['corr']['rescued']:2d} / {n_mi:2d}       | {results_by_vid[vid]['corr']['tp']:3d} | {results_by_vid[vid]['corr']['fp']:3d} | {results_by_vid[vid]['corr']['fn']:3d} | {p_c_v:9.4f}  | {r_c_v:9.4f}  | {f1_c_v:8.6f}")
        print(f"video_{vid:<4d} | {n_fr:<8d} | {'M_wrong':<12} | {results_by_vid[vid]['wrong']['admitted']:3d} / {results_by_vid[vid]['wrong']['total_temp']:<12d} | {results_by_vid[vid]['wrong']['rescued']:2d} / {n_mi:2d}       | {results_by_vid[vid]['wrong']['tp']:3d} | {results_by_vid[vid]['wrong']['fp']:3d} | {results_by_vid[vid]['wrong']['fn']:3d} | {p_w_v:9.4f}  | {r_w_v:9.4f}  | {f1_w_v:8.6f}")
        print(f"video_{vid:<4d} | {n_fr:<8d} | {'M_shifted':<12} | {results_by_vid[vid]['shifted']['admitted']:3d} / {results_by_vid[vid]['shifted']['total_temp']:<12d} | {results_by_vid[vid]['shifted']['rescued']:2d} / {n_mi:2d}       | {results_by_vid[vid]['shifted']['tp']:3d} | {results_by_vid[vid]['shifted']['fp']:3d} | {results_by_vid[vid]['shifted']['fn']:3d} | {p_s_v:9.4f}  | {r_s_v:9.4f}  | {f1_s_v:8.6f}")
        print("-" * 120)

        per_video_diagnostic[vid] = {
            "n_frames": n_fr,
            "n_targets": n_mi,
            "D": {"tp": results_by_vid[vid]["D"]["tp"], "fp": results_by_vid[vid]["D"]["fp"], "fn": results_by_vid[vid]["D"]["fn"], "rescued": results_by_vid[vid]["D"]["rescued"], "f1": float(f1_d_v)},
            "corr": {"tp": results_by_vid[vid]["corr"]["tp"], "fp": results_by_vid[vid]["corr"]["fp"], "fn": results_by_vid[vid]["corr"]["fn"], "rescued": results_by_vid[vid]["corr"]["rescued"], "admitted": results_by_vid[vid]["corr"]["admitted"], "f1": float(f1_c_v)},
            "wrong": {"tp": results_by_vid[vid]["wrong"]["tp"], "fp": results_by_vid[vid]["wrong"]["fp"], "fn": results_by_vid[vid]["wrong"]["fn"], "rescued": results_by_vid[vid]["wrong"]["rescued"], "admitted": results_by_vid[vid]["wrong"]["admitted"], "f1": float(f1_w_v)},
            "shifted": {"tp": results_by_vid[vid]["shifted"]["tp"], "fp": results_by_vid[vid]["shifted"]["fp"], "fn": results_by_vid[vid]["shifted"]["fn"], "rescued": results_by_vid[vid]["shifted"]["rescued"], "admitted": results_by_vid[vid]["shifted"]["admitted"], "f1": float(f1_s_v)},
        }

    results_payload = {
        "_meta": {
            "artifact_name": "pol3v_candidate3v_calibration_results_v2",
            "driver_sha256": driver_sha256,
            "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
            "token_cache_sha256": EXPECTED_TOKEN_CACHE_SHA256,
            "epoch8_adapter_sha256": EXPECTED_EPOCH8_ADAPTER_SHA256,
            "verifier_sha256": EXPECTED_VERIFIER_SHA256,
            "tau_star": float(tau_star),
            "temporal_prefilter_score": float(prefilter_threshold),
            "calib_status": final_status,
            "calib_passed": calib_verdict_passed
        },
        "pooled_results": {
            "D": {"tp": pooled_D["tp"], "fp": pooled_D["fp"], "fn": pooled_D["fn"], "rescued": pooled_D["rescued"], "f1": float(f1_d)},
            "corr": {"tp": pooled_corr["tp"], "fp": pooled_corr["fp"], "fn": pooled_corr["fn"], "rescued": pooled_corr["rescued"], "admitted": pooled_corr["admitted"], "f1": float(f1_c)},
            "wrong": {"tp": pooled_wrong["tp"], "fp": pooled_wrong["fp"], "fn": pooled_wrong["fn"], "rescued": pooled_wrong["rescued"], "admitted": pooled_wrong["admitted"], "f1": float(f1_w)},
            "shifted": {"tp": pooled_shifted["tp"], "fp": pooled_shifted["fp"], "fn": pooled_shifted["fn"], "rescued": pooled_shifted["rescued"], "admitted": pooled_shifted["admitted"], "f1": float(f1_s)}
        },
        "gate_evaluation": {
            "f1_gate_passed": f1_gate_passed,
            "corr_margin_passed": corr_margin_passed,
            "rescue_over_D_passed": rescue_over_D_passed,
            "delta_f1_vs_D": float(f1_c - f1_d),
            "rescue_margin_corr_vs_wrong": int(pooled_corr["rescued"] - pooled_wrong["rescued"]),
            "rescue_margin_corr_vs_shifted": int(pooled_corr["rescued"] - pooled_shifted["rescued"])
        },
        "per_video_diagnostic": per_video_diagnostic
    }

    with open(CALIB_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    res_hash = compute_sha256(CALIB_RESULTS_JSON)
    print(f"\n[+] Candidate 3V Calibration Results Frozen: {CALIB_RESULTS_JSON.resolve()}")
    print(f"    Manifest SHA-256:                      {res_hash}")
    print("=" * 115)


if __name__ == "__main__":
    main()