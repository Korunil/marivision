"""
D:/MARIVISION/H0_workbench/eval_tools/audit_marsen_7x_sentinel_extract.py
MARSEN Terminal Boundary Sentinel: Label-Free 7.0x Extraction & Descriptors (v2, P-1936, P-1940, P-1941)
"""

import os
import sys
import time
import zipfile
import io
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
SPLIT_MANIFEST_JSON = OUT_DIR / "pol2_candidate2m_micro_split.json"
QUERY_DATASET_PTH = OUT_DIR / "pol3v_design_queries_dataset_v7.pth"
FRAME_TENSORS_PTH = OUT_DIR / "pol3v_design_frames_tensors_v7.pth"
ZIP_PATH = Path("D:/MARIVISION/SeaClips.zip")

# P-1940: Consistent artifact filenames
TARGET_7X_PREPOOL_PTH = OUT_DIR / "marsen_7x_prepool_spatial_maps_1915.pth"
TARGET_7X_DESCRIPTORS_PTH = OUT_DIR / "marsen_7x_pooled_descriptors_1915.pth"

EXPECTED_DESIGN_VIDS = [4, 7, 16, 19, 25, 29, 30, 32, 35, 47]
CROP_SIZE = 128
CROP_MIN_SIDE_NORM = 0.025
FOV_SCALE_7X = 7.0
FOV_SCALE_5X = 5.0


def compute_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def gem_pool_cuda(x_tensor, p_val=3.0, eps=1e-6):
    x_clamped = x_tensor.clamp(min=eps)
    x_pow = x_clamped.pow(p_val)
    mean_pow = x_pow.mean(dim=(-2, -1))
    return mean_pow.pow(1.0 / p_val)


def main():
    t0 = time.time()
    driver_sha256 = compute_sha256(Path(__file__))
    print("=" * 115)
    print("MARSEN 7.0x TERMINAL BOUNDARY SENTINEL: EXTRACTION & PRE-POOLING (v2, P-1936, P-1940, P-1941)")
    print(f"Driver Script SHA-256: {driver_sha256}")
    print("=" * 115)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Target: {device} ({torch.cuda.get_device_name(0)})")

    with open(SPLIT_MANIFEST_JSON, "r", encoding="utf-8") as f:
        design_vids = sorted(json.load(f)["_meta"]["train_videos"])
    assert design_vids == EXPECTED_DESIGN_VIDS, f"[FATAL] Cohort mismatch: {design_vids}"

    cached_queries = torch.load(QUERY_DATASET_PTH, map_location="cpu")["extracted_queries"]
    frame_bundle = torch.load(FRAME_TENSORS_PTH, map_location="cpu")
    frame_records = frame_bundle["frame_records"]

    active_queries = [q for q in cached_queries if q["is_active"]]
    n_active = len(active_queries)
    assert n_active == 1915

    frame_dict = {fr["image_id"]: fr for fr in frame_records}
    for q in cached_queries:
        fr = frame_dict[q["image_id"]]
        cx, cy, w, h = fr["temp_boxes"][q["query_id"]].tolist()
        q["pred_box_norm"] = (cx, cy, w, h)

    print("[*] Initializing Frozen ResNet-18 Early Stem...")
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

    def get_crop_pixel_box(img_size, box_norm, expansion_factor):
        cx, cy, w, h = box_norm
        unclipped_side = max(w, h) * expansion_factor
        floor_bound = (unclipped_side < CROP_MIN_SIDE_NORM)
        side = max(CROP_MIN_SIDE_NORM, unclipped_side)

        x1_norm = max(0.0, cx - side / 2.0)
        y1_norm = max(0.0, cy - side / 2.0)
        x2_norm = min(1.0, cx + side / 2.0)
        y2_norm = min(1.0, cy + side / 2.0)

        W, H = img_size
        px_box = (
            max(0, int(x1_norm * W)),
            max(0, int(y1_norm * H)),
            min(W, max(int(x1_norm * W) + 1, int(x2_norm * W))),
            min(H, max(int(y1_norm * H) + 1, int(y2_norm * H)))
        )
        return px_box, floor_bound

    unique_iids = sorted(list(set(q["image_id"] for q in active_queries)))
    img_cache = {}
    print(f"[*] Preloading {len(unique_iids)} Source Frames...")

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
            assert zip_entry is not None, f"[FATAL] Cannot find {rel_path} in SeaClips.zip"
            with zf.open(zip_entry) as f:
                img_cache[iid] = Image.open(io.BytesIO(f.read())).convert("RGB")

    print("[*] Extracting Native Crops for 7.0x and Evaluating P-1941 Diagnostics...")
    crop_tensors_7x = []
    boxes_7x = []
    boxes_5x = []
    floor_hits_7x = 0

    for q in active_queries:
        img_nat = img_cache[q["image_id"]]
        
        px_box_7x, is_floor_7x = get_crop_pixel_box(img_nat.size, q["pred_box_norm"], FOV_SCALE_7X)
        px_box_5x, _ = get_crop_pixel_box(img_nat.size, q["pred_box_norm"], FOV_SCALE_5X)
        
        if is_floor_7x:
            floor_hits_7x += 1
            
        boxes_7x.append(px_box_7x)
        boxes_5x.append(px_box_5x)
        crop_tensors_7x.append(img_transform(img_nat.crop(px_box_7x)))

    floor_rate_7x = floor_hits_7x / n_active * 100.0
    same_5x_7x_rate = sum(1 for i in range(n_active) if boxes_5x[i] == boxes_7x[i]) / n_active * 100.0

    print(f"  - Crop Floor Binding Rate at 7.0x: {floor_rate_7x:.1f}%")
    print(f"  - P-1941 Pairwise Identity Rate (5.0x vs. 7.0x): {same_5x_7x_rate:.1f}% of queries produce 100% IDENTICAL crops")

    spatial_maps = []
    batch_size = 128
    with torch.no_grad():
        for i in range(0, len(crop_tensors_7x), batch_size):
            batch = torch.stack(crop_tensors_7x[i:i + batch_size]).to(device)
            spatial_maps.append(backbone(batch))

    maps_tensor = torch.cat(spatial_maps, dim=0)  # [1915, 128, 16, 16] on CUDA
    print(f"[+] Spatial Map Tensor Computed: {maps_tensor.shape}")

    # P-1940: Save the raw spatial maps tensor (~251 MB)
    maps_np = maps_tensor.cpu().numpy().astype(np.float32)
    torch.save({
        "driver_sha256": driver_sha256,
        "fov_scale": FOV_SCALE_7X,
        "maps": maps_np
    }, TARGET_7X_PREPOOL_PTH)
    print(f"[+] P-1940 Certified: Raw Pre-Pooling Spatial Maps Frozen: {TARGET_7X_PREPOOL_PTH.resolve()} (SHA-256: {compute_sha256(TARGET_7X_PREPOOL_PTH)})")

    # Compute GAP and GeM descriptors
    descriptors = {
        "7.0x_gap": maps_tensor.mean(dim=(-2, -1)).cpu().numpy().astype(np.float32),
        "7.0x_gem_p2": gem_pool_cuda(maps_tensor, p_val=2.0).cpu().numpy().astype(np.float32),
        "7.0x_gem_p3": gem_pool_cuda(maps_tensor, p_val=3.0).cpu().numpy().astype(np.float32),
        "7.0x_gem_p4": gem_pool_cuda(maps_tensor, p_val=4.0).cpu().numpy().astype(np.float32),
    }

    # Save pooled descriptors
    save_payload = {
        "driver_sha256": driver_sha256,
        "fov_scale": FOV_SCALE_7X,
        "floor_binding_rate": floor_rate_7x,
        "pairwise_identity_vs_5x": same_5x_7x_rate,
        "descriptors": descriptors
    }
    torch.save(save_payload, TARGET_7X_DESCRIPTORS_PTH)

    print(f"[+] P-1940 Certified: 7.0x Descriptors Frozen: {TARGET_7X_DESCRIPTORS_PTH.resolve()} (SHA-256: {compute_sha256(TARGET_7X_DESCRIPTORS_PTH)})")
    print(f"    Elapsed Time: {time.time() - t0:.2f}s")
    print("=" * 115)


if __name__ == "__main__":
    main()