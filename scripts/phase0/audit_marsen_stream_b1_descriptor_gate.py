"""
D:/MARIVISION/H0_workbench/eval_tools/audit_marsen_stream_b1_descriptor_gate.py
MARSEN Stream B1: Visual Crop & Descriptor Reproduction Gate (v2, P-1992, P-2002, P-2005)

Strict Compliance Checklist:
  - P-2005: Recovers historical 2,045 temporal-query ledger before filtering to 1,915 active queries.
  - Reconstructs proposal geometries directly from frame tensors matching original extraction code.
  - Re-extracts fresh native crops from SeaClips.zip via ResNet-18 layer2 with GeM4 and GAP pooling.
  - Asserts numerical descriptor parity against frozen historical caches within 1e-5 tolerance.
"""

import os
import sys
import io
import time
import zipfile
import json
import hashlib
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms, models

OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
TRAIN_JSON = Path("D:/MARIVISION/seaclips-train.json")
ZIP_PATH = Path("D:/MARIVISION/SeaClips.zip")
SPLIT_MANIFEST_JSON = OUT_DIR / "pol2_candidate2m_micro_split.json"
QUERY_DATASET_PTH = OUT_DIR / "pol3v_design_queries_dataset_v7.pth"
FRAME_TENSORS_PTH = OUT_DIR / "pol3v_design_frames_tensors_v7.pth"
CACHED_POOLED_DESCRIPTORS_PTH = OUT_DIR / "marsen_pooled_descriptors_1915.pth"

EXPECTED_DESIGN_VIDS = [4, 7, 16, 19, 25, 29, 30, 32, 35, 47]
CROP_SIZE = 128
CROP_MIN_SIDE_NORM = 0.025
PARITY_TOLERANCE = 1e-5


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


def main():
    t0 = time.time()
    driver_sha256 = compute_sha256(Path(__file__))
    print("=" * 115)
    print("MARSEN STREAM B1: VISUAL CROP & DESCRIPTOR REPRODUCTION GATE (v2, P-1992, P-2002, P-2005)")
    print(f"Driver Script SHA-256: {driver_sha256}")
    print("=" * 115)

    assert SPLIT_MANIFEST_JSON.exists(), "[FATAL] Missing split manifest!"
    assert QUERY_DATASET_PTH.exists(), "[FATAL] Missing design queries dataset!"
    assert FRAME_TENSORS_PTH.exists(), "[FATAL] Missing frame tensors bundle!"
    assert CACHED_POOLED_DESCRIPTORS_PTH.exists(), "[FATAL] Missing cached pooled descriptors!"
    assert ZIP_PATH.exists(), f"[FATAL] Missing SeaClips.zip at {ZIP_PATH}!"

    with open(SPLIT_MANIFEST_JSON, "r", encoding="utf-8") as f:
        design_vids = sorted(json.load(f)["_meta"]["train_videos"])
    assert design_vids == EXPECTED_DESIGN_VIDS

    cached_queries = torch.load(QUERY_DATASET_PTH, map_location="cpu")["extracted_queries"]
    cached_descriptors = torch.load(CACHED_POOLED_DESCRIPTORS_PTH, map_location="cpu")["descriptors"]
    v1_ref = cached_descriptors["1.0x_gem_p4"]
    c5_ref = cached_descriptors["5.0x_gap"]

    # 1. P-2005: Recover Exact Historical Query Ledger
    print("[*] Gate 1: Reconstructing Historical Query Ledger & Geometries...")
    frame_bundle = torch.load(FRAME_TENSORS_PTH, map_location="cpu")
    frame_records_list = frame_bundle["frame_records"]
    frame_dict = {fr["image_id"]: fr for fr in frame_records_list}

    frame_keys = []
    for fr in frame_records_list:
        for q_t in range(fr["n_temp"]):
            frame_keys.append((fr["image_id"], q_t))

    query_keys = [(q["image_id"], q["query_id"]) for q in cached_queries]

    assert len(frame_keys) == 2045, f"[FATAL] Expected 2,045 temporal queries, got {len(frame_keys)}"
    assert frame_keys == query_keys, "[FATAL] Query-key order mismatch between frame tensors and query dataset!"

    # Reconstruct proposal geometry
    for q in cached_queries:
        fr = frame_dict[q["image_id"]]
        cx, cy, w, h = fr["temp_boxes"][q["query_id"]].tolist()
        q["pred_box_norm"] = (cx, cy, w, h)
        q["img_w"] = fr["img_w"]
        q["img_h"] = fr["img_h"]

    active_queries = [q for q in cached_queries if q["is_active"]]
    n_active = len(active_queries)
    n_pos = sum(1 for q in active_queries if q["label_op_util"] == 1)

    assert n_active == 1915, f"[FATAL] Active query count mismatch: {n_active} != 1915"
    assert n_pos == 92, f"[FATAL] Positive query count mismatch: {n_pos} != 92"
    print(f"[+] Gate 1 Passed: Certified 2,045 temporal query ledger -> exactly {n_active} active proposals and {n_pos} positives.")

    # 2. Re-extract Native Image Crops and Descriptors from Source ZIP
    print("\n[*] Gate 2: Re-extracting 1.0x and 5.0x Native Crops via ResNet-18 layer2...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"    Compute Target: {device}")

    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    backbone = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        resnet.layer1, resnet.layer2
    ).to(device)
    backbone.eval()

    img_transform = transforms.Compose([
        transforms.Resize((CROP_SIZE, CROP_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        img_meta_by_id = {img["id"]: img for img in json.load(f)["images"]}

    unique_iids = sorted(list(set(q["image_id"] for q in active_queries)))
    img_cache = {}

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zip_index = {name.replace("\\", "/").strip("/"): name for name in zf.namelist()}
        for name in zf.namelist():
            norm = name.replace("\\", "/").strip("/")
            parts = norm.split("/")
            if len(parts) >= 2:
                zip_index["/".join(parts[-2:])] = name

        for iid in unique_iids:
            rel_path = img_meta_by_id[iid]["file_name"].replace("\\", "/").strip("/")
            target_key = "/".join(rel_path.split("/")[-2:]) if len(rel_path.split("/")) >= 2 else rel_path
            zip_entry = zip_index.get(rel_path) or zip_index.get(target_key)
            if zip_entry is None:
                for name in zf.namelist():
                    if name.replace("\\", "/").endswith(rel_path):
                        zip_entry = name
                        break
            assert zip_entry is not None, f"[FATAL] Frame {rel_path} not found in zip!"
            with zf.open(zip_entry) as f:
                img_cache[iid] = Image.open(io.BytesIO(f.read())).convert("RGB")

    print(f"[+] Loaded {len(img_cache)} unique source frames from ZIP into memory.")

    crops_1x, crops_5x = [], []
    for q in active_queries:
        img_nat = img_cache[q["image_id"]]
        box_1x = get_crop_pixel_box(img_nat.size, q["pred_box_norm"], expansion_factor=1.0)
        box_5x = get_crop_pixel_box(img_nat.size, q["pred_box_norm"], expansion_factor=5.0)

        crops_1x.append(img_transform(img_nat.crop(box_1x)))
        crops_5x.append(img_transform(img_nat.crop(box_5x)))

    batch_size = 128
    desc_1x_list, desc_5x_list = [], []

    with torch.no_grad():
        for i in range(0, n_active, batch_size):
            b1 = torch.stack(crops_1x[i:i + batch_size]).to(device)
            b5 = torch.stack(crops_5x[i:i + batch_size]).to(device)

            f1 = backbone(b1)
            f5 = backbone(b5)

            d1 = gem_pool_2d(f1, p_val=4.0).cpu().numpy().astype(np.float32)
            d5 = f5.mean(dim=(-2, -1)).cpu().numpy().astype(np.float32)

            desc_1x_list.append(d1)
            desc_5x_list.append(d5)

    v1_reproduced = np.concatenate(desc_1x_list, axis=0)
    c5_reproduced = np.concatenate(desc_5x_list, axis=0)

    diff_v1 = float(np.max(np.abs(v1_reproduced - v1_ref)))
    diff_c5 = float(np.max(np.abs(c5_reproduced - c5_ref)))

    print("\n[*] Descriptor Parity Gate Results:")
    print(f"  - 1.0x_GeM4 Max Absolute Difference: {diff_v1:.2e} (Tolerance: {PARITY_TOLERANCE:.1e})")
    print(f"  - 5.0x_GAP  Max Absolute Difference: {diff_c5:.2e} (Tolerance: {PARITY_TOLERANCE:.1e})")

    assert diff_v1 < PARITY_TOLERANCE, f"[FATAL] 1.0x_GeM4 descriptor failed parity: {diff_v1:.2e} >= {PARITY_TOLERANCE:.1e}"
    assert diff_c5 < PARITY_TOLERANCE, f"[FATAL] 5.0x_GAP descriptor failed parity: {diff_c5:.2e} >= {PARITY_TOLERANCE:.1e}"
    print("[+] Numerical descriptor parity confirmed within 1e-5.")

    print("\n" + "=" * 115)
    print("STREAM B1 DESCRIPTOR REPRODUCTION GATE PASSED")
    print("Native crop and descriptor extraction contract successfully reproduced on design cohort.")
    print(f"Elapsed Time: {time.time() - t0:.2f}s")
    print("=" * 115)


if __name__ == "__main__":
    main()