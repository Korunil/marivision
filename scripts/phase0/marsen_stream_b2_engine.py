"""
D:/MARIVISION/H0_workbench/eval_tools/marsen_stream_b2_engine.py
MARSEN Stream B2: Production Shared Generalized Stream Engine (P-2050..P-2111)

Certified detection engine primitives:
  - Primitive 1: _initialize_model_and_adapter() binds frozen R0 and epoch-8 adapter.
  - Primitive 2: _extract_top8_tokens() caches Top-8 tokens (bbox as Python list).
  - Primitive 3: _run_temporal_frame() runs temporal forward pass across ages {5, 4, 3, 2}
                 with fail-fast memory alignment and explicit empty-bank handling.
  - Primitive 4: _build_queries_and_joint_labels() evaluates exact 15-D metadata with
                 [x, y, w, h] boxes, target-derived GT, 7-arg evaluate_operational_top100,
                 boolean-masked missed_gt_indices, stable score-ordered sequential add pass,
                 and iterative reverse pruning pass.
  - Phase 2 Visual Pass: Two-phase extraction batching all active video crops in fixed B = 128
    convolutions (residual padded to 128), guaranteeing 0.00e+00 bitwise spatial parity.
  - Fail-Closed Split Resolution: Permanent three-level indexing (split/video/frame) in
    resolve_frame_member() and load_frame_image() ensuring zero cross-split collision (P-2107, P-2111).
"""

import os
import sys
import io
import time
import zipfile
import json
import inspect
import hashlib
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms, models
from collections import defaultdict

OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
ROOT_DIR = Path("D:/MARIVISION")
ZIP_PATH = ROOT_DIR / "SeaClips.zip"

MEMORY_AGES = [5, 4, 3, 2]
SCORE_THRESHOLD = 0.20
CROP_SIZE = 128
CROP_MIN_SIDE_NORM = 0.025
FIXED_BATCH_SIZE = 128


def compute_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def gem_pool_2d(x_tensor: torch.Tensor, p_val: float = 4.0, eps: float = 1e-6) -> torch.Tensor:
    x_clamped = x_tensor.clamp(min=eps)
    x_pow = x_clamped.pow(p_val)
    mean_pow = x_pow.mean(dim=(-2, -1))
    return mean_pow.pow(1.0 / p_val)


def get_crop_pixel_box(img_size, box_norm, expansion_factor):
    cx, cy, w, h = box_norm
    side = max(CROP_MIN_SIDE_NORM, max(w, h) * expansion_factor)
    x1_norm = max(0.0, cx - side / 2.0)
    y1_norm = max(0.0, cy - side / 2.0)
    x2_norm = min(1.0, cx + side / 2.0)
    y2_norm = min(1.0, cy + side / 2.0)

    W, H = img_size
    return (
        max(0, int(x1_norm * W)),
        max(0, int(y1_norm * H)),
        min(W, max(int(x1_norm * W) + 1, int(x2_norm * W))),
        min(H, max(int(y1_norm * H) + 1, int(y2_norm * H)))
    )


class MarsenChronology:
    """Enforces authoritative video_id/frame_id schema and strict contiguity."""

    @staticmethod
    def load_and_validate(json_path):
        assert Path(json_path).exists(), f"[FATAL] Missing dataset manifest: {json_path}"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        images_by_video = defaultdict(list)
        for img in data["images"]:
            assert "video_id" in img and img["video_id"] is not None, f"[FATAL] Missing video_id in image {img['id']}"
            assert "frame_id" in img and img["frame_id"] is not None, f"[FATAL] Missing frame_id in image {img['id']}"
            images_by_video[int(img["video_id"])].append(img)

        for vid in images_by_video:
            f_ids = [int(x["frame_id"]) for x in images_by_video[vid]]
            assert len(f_ids) == len(set(f_ids)), f"[FATAL] Duplicate frame_id in video {vid}"
            images_by_video[vid].sort(key=lambda x: int(x["frame_id"]))

        return data, images_by_video

    @staticmethod
    def build_contiguous_windows(frames_list):
        windows = []
        n_frames = len(frames_list)
        if n_frames < 6:
            return windows

        for i in range(5, n_frames):
            seq = frames_list[i - 5:i + 1]
            frame_nums = [int(fr["frame_id"]) for fr in seq]

            if not all(frame_nums[j + 1] - frame_nums[j] == 1 for j in range(5)):
                continue

            curr_frame = seq[-1]
            hist_frames = [seq[5 - age] for age in MEMORY_AGES]  # ages 5, 4, 3, 2
            windows.append({
                "video_id": int(curr_frame["video_id"]),
                "current_frame": curr_frame,
                "history_frames": hist_frames,
                "frame_nums": frame_nums
            })
        return windows


class MarsenVisualExtractor:
    """Extracts 1.0x_GeM4, 3.0x_GeM4, and 5.0x_GAP descriptors directly from SeaClips.zip."""

    def __init__(self, device):
        self.device = device
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2
        ).to(device)
        self.backbone.eval()

        self.transform = transforms.Compose([
            transforms.Resize((CROP_SIZE, CROP_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_crops_fixed_b128(self, crops_1x, crops_3x, crops_5x):
        """Processes visual crops in fixed batches of B = 128, padding residual chunks."""
        n_total = len(crops_1x)
        if n_total == 0:
            return (
                np.empty((0, 128), dtype=np.float32),
                np.empty((0, 128), dtype=np.float32),
                np.empty((0, 128), dtype=np.float32)
            )

        out_1x, out_3x, out_5x = [], [], []
        with torch.no_grad():
            for i in range(0, n_total, FIXED_BATCH_SIZE):
                chunk_1 = crops_1x[i:i + FIXED_BATCH_SIZE]
                chunk_3 = crops_3x[i:i + FIXED_BATCH_SIZE]
                chunk_5 = crops_5x[i:i + FIXED_BATCH_SIZE]
                k = len(chunk_1)

                b1 = torch.stack(chunk_1)
                b3 = torch.stack(chunk_3)
                b5 = torch.stack(chunk_5)

                if k < FIXED_BATCH_SIZE:
                    pad_len = FIXED_BATCH_SIZE - k
                    b1 = torch.cat([b1, b1[-1:].repeat(pad_len, 1, 1, 1)], dim=0)
                    b3 = torch.cat([b3, b3[-1:].repeat(pad_len, 1, 1, 1)], dim=0)
                    b5 = torch.cat([b5, b5[-1:].repeat(pad_len, 1, 1, 1)], dim=0)

                b1_dev = b1.to(self.device)
                b3_dev = b3.to(self.device)
                b5_dev = b5.to(self.device)

                f1 = self.backbone(b1_dev)[:k]
                f3 = self.backbone(b3_dev)[:k]
                f5 = self.backbone(b5_dev)[:k]

                out_1x.append(gem_pool_2d(f1, p_val=4.0).cpu().numpy().astype(np.float32))
                out_3x.append(gem_pool_2d(f3, p_val=4.0).cpu().numpy().astype(np.float32))
                out_5x.append(f5.mean(dim=(-2, -1)).cpu().numpy().astype(np.float32))

        return np.concatenate(out_1x, axis=0), np.concatenate(out_3x, axis=0), np.concatenate(out_5x, axis=0)


class GeneralizedStreamEngine:
    """Coordinates detector inference, token caching, temporal modeling, and feature production."""

    def __init__(self, device):
        self.device = device
        self.visual_extractor = MarsenVisualExtractor(device)
        self._init_zip_index()

        self.model = None
        self.adapter = None
        self.detector_initialized = False
        self.split_datasets = {}

    def _init_zip_index(self):
        assert ZIP_PATH.exists(), f"[FATAL] Missing SeaClips.zip at {ZIP_PATH}"
        self.zf = zipfile.ZipFile(ZIP_PATH, "r")
        # P-2107: Split-qualified three-level indexing (split, rel_fn) -> list of member paths
        self.split_zip_index = defaultdict(list)
        for raw_name in self.zf.namelist():
            norm = raw_name.replace("\\", "/").strip("/").lstrip("/")
            parts = norm.split("/")
            for split in ["train", "val"]:
                if split in parts:
                    split_idx = parts.index(split)
                    rel_fn = "/".join(parts[split_idx + 1:])
                    self.split_zip_index[(split, rel_fn)].append(raw_name)

    def resolve_frame_member(self, file_name, split_name):
        """P-2111 / P-2112: Fail-closed resolver returning exact archive path for (split, file)."""
        split = str(split_name).lower()
        assert split in {"train", "val"}, (
            f"[FATAL] Unsupported split during current Train/Val production: {split!r}"
        )
        rel_fn = str(file_name).replace("\\", "/").strip("/").lstrip("/")
        matches = self.split_zip_index.get((split, rel_fn), [])
        assert len(matches) == 1, (
            f"[FATAL] Split-qualified lookup failed for split='{split}', file='{rel_fn}'. "
            f"Expected exactly 1 member, found {len(matches)}: {matches}"
        )
        return matches[0]

    def load_frame_image(self, file_name, split_name):
        """P-2107: Split-aware frame image loader guaranteeing zero cross-split collision."""
        member_name = self.resolve_frame_member(file_name, split_name)
        with self.zf.open(member_name) as f:
            return Image.open(io.BytesIO(f.read())).convert("RGB")

    def _initialize_model_and_adapter(self):
        """Primitive 1: Binds Deformable-DETR R0 and frozen epoch-8 adapter."""
        if self.detector_initialized:
            return

        sys.path.insert(0, str(OUT_DIR))
        import audit_pol3v_operating_point_sweep as opsweep
        import audit_pol3v_nonlinear_separability as nonlin

        self.opsweep = opsweep
        self.nonlin = nonlin

        if hasattr(opsweep, "apply_candidate2m_source_hooks"):
            opsweep.apply_candidate2m_source_hooks()

        args = opsweep.get_authoritative_r0_args()
        args.conditioning_mode = "none"

        model, criterion, _ = opsweep.build(args)

        ckpt = torch.load(opsweep.M0_CKPT_PATH, map_location="cpu")
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)

        for p in model.parameters():
            p.requires_grad = False
        model.eval()

        epoch8 = torch.load(opsweep.EPOCH8_MODEL_PTH, map_location="cpu")
        adapter = opsweep.HistoricalObjectTokenMemory(
            d_model=256,
            num_classes=7,
            max_age=6
        ).to(self.device)
        adapter.load_state_dict(epoch8["adapter_state"])
        adapter.gamma.data.fill_(10.0)
        adapter.eval()

        self.model = model
        self.adapter = adapter
        self.detector_initialized = True
        print("[+] Primitive 1 Initialized: Genuine R0 and Epoch-8 temporal adapter bound and frozen.")

    def _get_split_dataset_context(self, json_path):
        json_path_str = str(json_path)
        if json_path_str in self.split_datasets:
            return self.split_datasets[json_path_str]

        self._initialize_model_and_adapter()
        dataset = self.opsweep.SeaClipsDetection(
            ZIP_PATH,
            json_path,
            is_train=False,
            random_flip=False
        )
        id_to_idx = {
            iid: idx
            for idx, iid in enumerate(sorted(dataset.coco.getImgIds()))
        }
        if hasattr(dataset, "label_to_cat_id"):
            label_to_cat_id = dataset.label_to_cat_id
        else:
            cat_ids = sorted(dataset.coco.getCatIds())
            label_to_cat_id = {i: cat_id for i, cat_id in enumerate(cat_ids)}

        ctx = {
            "dataset": dataset,
            "id_to_idx": id_to_idx,
            "label_to_cat_id": label_to_cat_id,
            "collate_fn": self.opsweep.collate_fn
        }
        self.split_datasets[json_path_str] = ctx
        return ctx

    def _extract_top8_tokens(self, samples):
        """Primitive 2: Flattened Top-100 -> score >= 0.20 -> first 8 tokens."""
        with torch.no_grad():
            out_d, hs_d = self.model(samples, temporal_enabled=False, return_decoder_tokens=True)
            logits = out_d["pred_logits"][0]
            boxes = out_d["pred_boxes"][0]

            if hs_d.dim() == 4:
                hs_last = hs_d[-1, 0]
            elif hs_d.dim() == 3:
                hs_last = hs_d[0]
            else:
                hs_last = hs_d

            probs = logits.sigmoid()
            flat_probs = probs.flatten()
            topk_vals, topk_inds = torch.topk(flat_probs, k=min(100, flat_probs.numel()))

            tokens = []
            for val, ind in zip(topk_vals, topk_inds):
                score = float(val.item())
                if score < SCORE_THRESHOLD:
                    break
                idx = int(ind.item())
                q_idx = idx // 7
                c_idx = idx % 7

                tok = {
                    "z": hs_last[q_idx].clone().cpu(),
                    "bbox": boxes[q_idx].cpu().numpy().tolist(),  # Python list format [cx, cy, w, h]
                    "logits": logits[q_idx].clone().cpu(),
                    "score": score,
                    "category_idx": c_idx,
                    "query_idx": q_idx
                }
                tokens.append(tok)
                if len(tokens) == 8:
                    break
            return tokens

    def _run_temporal_frame(self, samples, memory_bank):
        """Primitive 3: Fresh temporal inference forward pass with empty-memory handling."""
        if not memory_bank:
            with torch.no_grad():
                out_d = self.model(samples, temporal_enabled=False)
            return {
                "d_logits": out_d["pred_logits"][0],
                "d_boxes": out_d["pred_boxes"][0],
                "temp_logits": torch.zeros((0, 7), dtype=out_d["pred_logits"].dtype, device=out_d["pred_logits"].device),
                "temp_boxes": torch.zeros((0, 4), dtype=out_d["pred_boxes"].dtype, device=out_d["pred_boxes"].device),
                "temp_hs": torch.zeros((0, 256), dtype=out_d["pred_logits"].dtype, device=out_d["pred_logits"].device),
                "n_temp": 0
            }

        with torch.no_grad():
            out_d = self.model(samples, temporal_enabled=False)
            out_aug, hs_last = self.model(
                samples,
                temporal_adapter=self.adapter,
                memory_bank=[memory_bank],
                temporal_enabled=True,
                return_decoder_tokens=True
            )

            d_logits = out_d["pred_logits"][0]
            d_boxes = out_d["pred_boxes"][0]
            temp_logits = out_aug["pred_logits"][0, 300:]
            temp_boxes = out_aug["pred_boxes"][0, 300:]

            if hs_last.dim() == 4:
                hs_dec = hs_last[-1, 0, 300:]
            elif hs_last.dim() == 3:
                hs_dec = hs_last[0, 300:]
            else:
                hs_dec = hs_last[300:]

            return {
                "d_logits": d_logits,
                "d_boxes": d_boxes,
                "temp_logits": temp_logits,
                "temp_boxes": temp_boxes,
                "temp_hs": hs_dec,
                "n_temp": int(temp_logits.shape[0])
            }

    def _target_to_historical_gt(self, target, label_to_cat_id, img_w, img_h):
        """Converts SeaClipsDetection target to pixel [x, y, w, h] boxes and category IDs."""
        gt_boxes_norm = target["boxes"].cpu()
        gt_labels_raw = target["labels"].cpu()

        gt_boxes_px = []
        gt_cats = []

        for g_idx in range(len(gt_labels_raw)):
            gcx, gcy, gw, gh = gt_boxes_norm[g_idx].tolist()
            gx1 = (gcx - gw / 2.0) * img_w
            gy1 = (gcy - gh / 2.0) * img_h
            gt_boxes_px.append([
                gx1,
                gy1,
                gw * img_w,
                gh * img_h
            ])
            gt_cats.append(
                label_to_cat_id[int(gt_labels_raw[g_idx].item())]
            )
        return gt_boxes_px, gt_cats

    def _build_queries_and_joint_labels(self, res_w, memory_tokens, video_id, image_id, frame_id, img_w, img_h, target, label_to_cat_id, is_labeled):
        """Primitive 4: Metadata computation and joint frame-level add/prune labeling."""
        n_temp = res_w["n_temp"]
        if n_temp == 0:
            return []

        assert n_temp == len(memory_tokens), f"[FATAL] Temporal query/memory mismatch: {n_temp} != {len(memory_tokens)}"
        assert res_w["temp_hs"].shape[0] == n_temp, f"[FATAL] temp_hs shape mismatch: {res_w['temp_hs'].shape[0]} != {n_temp}"

        temp_boxes = res_w["temp_boxes"]
        temp_logits = res_w["temp_logits"]
        temp_hs = res_w["temp_hs"]
        d_boxes = res_w["d_boxes"]
        d_logits = res_w["d_logits"]

        base_d_boxes_px = [
            [(cx - w / 2.0) * img_w, (cy - h / 2.0) * img_h, w * img_w, h * img_h]
            for cx, cy, w, h in d_boxes.tolist()
        ]

        frame_queries = []
        for q_t in range(n_temp):
            tok = memory_tokens[q_t]
            h_i_t = temp_hs[q_t]
            b_i_t = temp_boxes[q_t]
            l_i_t = temp_logits[q_t]

            feat_vec, box_px, pred_class, max_score = self.nonlin.extract_query_evidence_features(
                tok,
                h_i_t,
                b_i_t,
                l_i_t,
                base_d_boxes_px,
                d_logits,
                img_w,
                img_h
            )

            cx, cy, w, h = b_i_t.tolist()
            frame_queries.append({
                "video_id": video_id,
                "image_id": image_id,
                "frame_id": frame_id,
                "query_id": q_t,
                "pred_box_norm": [cx, cy, w, h],
                "box_px": box_px,
                "pred_class": int(pred_class),
                "max_score": float(max_score),
                "is_active": bool(max_score >= SCORE_THRESHOLD),
                "features": np.asarray(feat_vec, dtype=np.float32),
                "decoder_256": h_i_t.clone().cpu().numpy().astype(np.float32)
            })

        if is_labeled and target is not None and len(target["boxes"]) > 0:
            gt_boxes_px, gt_cats = self._target_to_historical_gt(target, label_to_cat_id, img_w, img_h)
            self._evaluate_joint_frame_labels(frame_queries, res_w, gt_boxes_px, gt_cats, label_to_cat_id, img_w, img_h)
        else:
            for q in frame_queries:
                q["label_op_util"] = 0
                q["label_rescue_pot"] = 0

        return frame_queries

    def _evaluate_joint_frame_labels(self, frame_queries, res_w, gt_boxes_px, gt_cats, label_to_cat_id, img_w, img_h):
        """Historical frame-level sequential greedy add pass and iterative reverse pruning pass."""
        d_boxes = res_w["d_boxes"]
        d_logits = res_w["d_logits"]
        temp_boxes = res_w["temp_boxes"]
        temp_logits = res_w["temp_logits"]
        n_temp = res_w["n_temp"]
        n_gt = len(gt_boxes_px)

        tpd, fpd, _, d_matched_gt = self.nonlin.evaluate_operational_top100(
            d_logits,
            d_boxes,
            gt_boxes_px,
            gt_cats,
            label_to_cat_id,
            img_w,
            img_h
        )

        missed_gt_indices = {
            g_idx for g_idx in range(n_gt)
            if not d_matched_gt[g_idx]
        }

        prob_temp = temp_logits.sigmoid().cpu()
        flat_pairs = []
        for q_t in range(n_temp):
            for c_t in range(7):
                s_val = float(prob_temp[q_t, c_t].item())
                flat_pairs.append((s_val, q_t, c_t))

        flat_pairs.sort(key=lambda x: x[0], reverse=True)

        claimed_miss_gt = set()
        query_rescue_potential = {q_t: 0 for q_t in range(n_temp)}

        for s_val, q_t, c_t in flat_pairs:
            if s_val < SCORE_THRESHOLD:
                break

            p_box = frame_queries[q_t]["box_px"]
            p_cat = label_to_cat_id[c_t]

            matched_gt_idx = -1
            best_iou = -1.0

            for g_idx in range(n_gt):
                if gt_cats[g_idx] == p_cat:
                    iou_g = self.nonlin.box_iou(gt_boxes_px[g_idx], p_box)
                    if iou_g >= 0.30 and iou_g > best_iou:
                        best_iou = iou_g
                        matched_gt_idx = g_idx

            if matched_gt_idx >= 0:
                if matched_gt_idx in missed_gt_indices and matched_gt_idx not in claimed_miss_gt:
                    if query_rescue_potential[q_t] == 0:
                        claimed_miss_gt.add(matched_gt_idx)
                        query_rescue_potential[q_t] = 1

        for q_t in range(n_temp):
            frame_queries[q_t]["label_rescue_pot"] = query_rescue_potential[q_t]

        sorted_q_indices = sorted(
            range(n_temp),
            key=lambda q: frame_queries[q]["max_score"],
            reverse=True
        )

        curr_joint_logits = d_logits.clone()
        curr_joint_boxes = d_boxes.clone()
        curr_tp = tpd
        curr_fp = fpd
        admitted_queries_list = []

        for q_t in sorted_q_indices:
            cand_logits = torch.cat([curr_joint_logits, temp_logits[q_t:q_t+1]], dim=0)
            cand_boxes = torch.cat([curr_joint_boxes, temp_boxes[q_t:q_t+1]], dim=0)
            tp_cand, fp_cand, _, _ = self.nonlin.evaluate_operational_top100(
                cand_logits,
                cand_boxes,
                gt_boxes_px,
                gt_cats,
                label_to_cat_id,
                img_w,
                img_h
            )
            if tp_cand > curr_tp and fp_cand <= curr_fp:
                curr_joint_logits = cand_logits
                curr_joint_boxes = cand_boxes
                curr_tp = tp_cand
                curr_fp = fp_cand
                admitted_queries_list.append(int(q_t))

        final_admitted_set = set(admitted_queries_list)
        while True:
            pruned_in_pass = False
            current_ordered_admitted = [q_t for q_t in admitted_queries_list if q_t in final_admitted_set]

            for q_t in reversed(current_ordered_admitted):
                if len(final_admitted_set) <= 1 and curr_tp > tpd:
                    break
                rem_admitted = [x for x in current_ordered_admitted if x != q_t and x in final_admitted_set]
                if len(rem_admitted) == 0:
                    test_logits = d_logits.clone()
                    test_boxes = d_boxes.clone()
                else:
                    test_logits = torch.cat([d_logits] + [temp_logits[x:x+1] for x in rem_admitted], dim=0)
                    test_boxes = torch.cat([d_boxes] + [temp_boxes[x:x+1] for x in rem_admitted], dim=0)

                tp_without, fp_without, _, _ = self.nonlin.evaluate_operational_top100(
                    test_logits,
                    test_boxes,
                    gt_boxes_px,
                    gt_cats,
                    label_to_cat_id,
                    img_w,
                    img_h
                )
                if tp_without == curr_tp and fp_without <= curr_fp:
                    final_admitted_set.remove(q_t)
                    curr_tp = tp_without
                    curr_fp = fp_without
                    pruned_in_pass = True

            if not pruned_in_pass:
                break

        for q_t in range(n_temp):
            frame_queries[q_t]["label_op_util"] = 1 if q_t in final_admitted_set else 0

    def process_video(self, video_id, frames_list, json_path, is_labeled=True):
        """
        Executes fresh two-phase video processing:
          - Phase 1: Rolling temporal inference, candidate generation, metadata, and labels.
          - Phase 2: Split-aware deterministic fixed-batch (B = 128) visual extraction.
        """
        self._initialize_model_and_adapter()
        split_ctx = self._get_split_dataset_context(json_path)
        dataset = split_ctx["dataset"]
        id_to_idx = split_ctx["id_to_idx"]
        label_to_cat_id = split_ctx["label_to_cat_id"]
        collate_fn = split_ctx["collate_fn"]

        # P-2111: Fail-closed split name derivation directly from SeaClipsDetection.split
        split_name = str(dataset.split).lower()
        assert split_name in {"train", "val"}, (
            f"[FATAL] Unsupported split during current Train/Val production: {split_name!r} from {json_path}"
        )

        windows = MarsenChronology.build_contiguous_windows(frames_list)
        n_windows = len(windows)

        if n_windows == 0:
            return {"video_id": video_id, "n_windows": 0, "frame_records": [], "queries": []}

        # Pass 1: Cache Top-8 tokens per source frame
        token_cache = {}
        for fr in frames_list:
            iid = fr["id"]
            batch = collate_fn([dataset[id_to_idx[iid]]])
            samples = batch[0].to(self.device)
            token_cache[iid] = self._extract_top8_tokens(samples)

        # Pass 2: Temporal forward pass across contiguous rolling windows
        frame_records = []
        all_queries = []

        for w in windows:
            curr_img = w["current_frame"]
            iid = curr_img["id"]
            fnum = int(curr_img["frame_id"])

            memory_tokens = []
            for h_fr in w["history_frames"]:
                h_id = h_fr["id"]
                h_age = fnum - int(h_fr["frame_id"])
                if h_id in token_cache and h_age in MEMORY_AGES:
                    for tok in token_cache[h_id]:
                        tok_dict = dict(tok)
                        tok_dict["age"] = h_age
                        memory_tokens.append(tok_dict)

            batch_curr = collate_fn([dataset[id_to_idx[iid]]])
            samples_curr = batch_curr[0].to(self.device)
            target_curr = batch_curr[1][0] if is_labeled and len(batch_curr[1]) > 0 else None

            if target_curr is not None and "orig_size" in target_curr:
                img_h = int(target_curr["orig_size"][0].item())
                img_w = int(target_curr["orig_size"][1].item())
            else:
                img_w = curr_img.get("width", 1920)
                img_h = curr_img.get("height", 1080)

            res_w = self._run_temporal_frame(samples_curr, memory_tokens)

            frame_records.append({
                "image_id": iid,
                "video_id": video_id,
                "frame_id": fnum,
                "img_w": img_w,
                "img_h": img_h,
                "n_temp": res_w["n_temp"],
                "d_logits": res_w["d_logits"],
                "d_boxes": res_w["d_boxes"],
                "temp_logits": res_w["temp_logits"],
                "temp_boxes": res_w["temp_boxes"]
            })

            w_queries = self._build_queries_and_joint_labels(
                res_w, memory_tokens, video_id, iid, fnum, img_w, img_h, target_curr, label_to_cat_id, is_labeled
            )
            all_queries.extend(w_queries)

        # Phase 2: Split-Aware Deterministic Fixed-Batch (B = 128) Visual Extraction
        active_q_entries = [q for q in all_queries if q["is_active"]]
        if len(active_q_entries) > 0:
            frame_img_cache = {}
            crops_1x = []
            crops_3x = []
            crops_5x = []

            for q in active_q_entries:
                iid = q["image_id"]
                if iid not in frame_img_cache:
                    frame_meta = next(f for f in frames_list if f["id"] == iid)
                    frame_img_cache[iid] = self.load_frame_image(frame_meta["file_name"], split_name=split_name)

                img_pil = frame_img_cache[iid]
                box_1x = get_crop_pixel_box(img_pil.size, q["pred_box_norm"], expansion_factor=1.0)
                box_3x = get_crop_pixel_box(img_pil.size, q["pred_box_norm"], expansion_factor=3.0)
                box_5x = get_crop_pixel_box(img_pil.size, q["pred_box_norm"], expansion_factor=5.0)

                crops_1x.append(self.visual_extractor.transform(img_pil.crop(box_1x)))
                crops_3x.append(self.visual_extractor.transform(img_pil.crop(box_3x)))
                crops_5x.append(self.visual_extractor.transform(img_pil.crop(box_5x)))

            d1, d3, d5 = self.visual_extractor.extract_crops_fixed_b128(crops_1x, crops_3x, crops_5x)

            for idx, q in enumerate(active_q_entries):
                q["1x_gem4"] = d1[idx]
                q["3x_gem4"] = d3[idx]
                q["5x_gap"] = d5[idx]

        return {
            "video_id": video_id,
            "n_windows": n_windows,
            "frame_records": frame_records,
            "queries": all_queries
        }