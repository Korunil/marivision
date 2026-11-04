"""
D:/MARIVISION/H0_workbench/eval_tools/audit_marsen_generate_gem_descriptors.py
MARSEN Phase B2.5 Staged Preprocessing: Deterministic GAP & GeM Feature Generation (P-1907)

Strict Compliance Checklist:
  - D-3867: Deterministic spatial pooling generation (zero labels used; zero design leakage).
  - P-1900 / P-1907: Operates directly on frozen marsen_prepool_spatial_maps_1915.pth on CUDA.
  - Generates GAP and GeM (p in {2, 3, 4}) across ALL 6 FOV scales (1.0x to 5.0x).
  - Output: Saves 24 pooled 128-D descriptors (6 scales x 4 operators) to marsen_pooled_descriptors_1915.pth.
"""

import os
import sys
import time
import hashlib
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
CACHED_PREPOOL_MAPS_PTH = OUT_DIR / "marsen_prepool_spatial_maps_1915.pth"
CACHED_POOLED_DESCRIPTORS_PTH = OUT_DIR / "marsen_pooled_descriptors_1915.pth"
ALL_FOV_SCALES = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]


def compute_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def gem_pool_cuda(x_tensor, p_val=3.0, eps=1e-6):
    """
    x_tensor: [N, C, H, W] on CUDA
    GeM = ( 1/(H*W) sum( x^p ) ) ^ (1/p)
    """
    x_clamped = x_tensor.clamp(min=eps)
    x_pow = x_clamped.pow(p_val)
    mean_pow = x_pow.mean(dim=(-2, -1))
    return mean_pow.pow(1.0 / p_val)


def main():
    t_start = time.time()
    driver_sha256 = compute_sha256(Path(__file__))
    print("=" * 115)
    print("MARSEN PHASE B2.5 PREPROCESSING: DETERMINISTIC GAP & GeM FEATURE GENERATION (P-1907)")
    print(f"Driver Script SHA-256: {driver_sha256}")
    print("=" * 115)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Target: {device} ({torch.cuda.get_device_name(0)})")

    if not CACHED_PREPOOL_MAPS_PTH.exists():
        print(f"[!] Target spatial map cache not found: {CACHED_PREPOOL_MAPS_PTH.resolve()}")
        print("    Waiting for Terminal 1 to finish feature extraction...")
        sys.exit(1)

    print(f"[*] Ingesting pre-pooling spatial maps: {CACHED_PREPOOL_MAPS_PTH.resolve()}")
    maps_bundle = torch.load(CACHED_PREPOOL_MAPS_PTH, map_location="cpu")
    maps_dict = maps_bundle["maps"]

    assert all(s in maps_dict for s in ALL_FOV_SCALES), "[FATAL] Incomplete spatial maps!"
    print(f"[+] Loaded spatial maps for all 6 FOVs: {ALL_FOV_SCALES}")

    descriptors = {}
    p_values = [2.0, 3.0, 4.0]

    with torch.no_grad():
        for scale in ALL_FOV_SCALES:
            arr_np = maps_dict[scale]  # shape: (1915, 128, 16, 16)
            tensor_cuda = torch.from_numpy(arr_np).to(device)
            N, C, H, W = tensor_cuda.shape
            print(f"[*] Processing FOV {scale:.1f}x (tensor: {N}x{C}x{H}x{W})...")

            # 1. Standard GAP
            gap_feat = tensor_cuda.mean(dim=(-2, -1)).cpu().numpy().astype(np.float32)
            descriptors[f"{scale}x_gap"] = gap_feat

            # 2. GeM with p in {2, 3, 4}
            for p in p_values:
                gem_feat = gem_pool_cuda(tensor_cuda, p_val=p).cpu().numpy().astype(np.float32)
                descriptors[f"{scale}x_gem_p{int(p)}"] = gem_feat

    save_payload = {
        "driver_sha256": driver_sha256,
        "n_samples": 1915,
        "n_channels": 128,
        "descriptors": descriptors,
        "available_keys": list(descriptors.keys())
    }
    torch.save(save_payload, CACHED_POOLED_DESCRIPTORS_PTH)

    t_elapsed = time.time() - t_start
    print("\n" + "=" * 115)
    print(f"[+] All 24 Spatial Pooling Descriptors Successfully Generated & Frozen in {t_elapsed:.2f}s:")
    print(f"    - Target Cache:  {CACHED_POOLED_DESCRIPTORS_PTH.resolve()}")
    print(f"    - Manifest Hash: {compute_sha256(CACHED_POOLED_DESCRIPTORS_PTH)}")
    print(f"    - Total Keys:    {len(descriptors)} ({[k for k in descriptors.keys()][:8]}...)")
    print("=" * 115)


if __name__ == "__main__":
    main()