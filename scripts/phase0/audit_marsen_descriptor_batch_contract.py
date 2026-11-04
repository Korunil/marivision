"""
D:/MARIVISION/H0_workbench/eval_tools/audit_marsen_descriptor_batch_contract.py
MARSEN Spatial Verifier: Descriptor Batch-Contract Audit (P-2080, P-2081, P-2084, P-2085)

Tests the batch-shape hypothesis on the 1,915 active design proposals:
  - Contract A: Global fixed batch B = 128 (Historical / B1 baseline).
  - Contract B: Per-frame variable batches N (Mode-2A streaming).
  - Contract C: Fixed-shape padded batches B = 128.
Reports active-query batch-size distribution, direct A<->B / A<->C deltas,
and result-dependent status verdicts. Runtime: ~10s.
"""

import os
import sys
import io
import time
import zipfile
import json
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
TRAIN_JSON = ROOT_DIR / "seaclips-train.json"

CERT_QUERIES_PTH = OUT_DIR / "stream_b2_reproduced_queries.pth"
CERT_FRAMES_PTH = OUT_DIR / "stream_b2_reproduced_frames.pth"
FROZEN_DESCRIPTORS_PTH = OUT_DIR / "marsen_pooled_descriptors_1915.pth"
RESULTS_JSON = OUT_DIR / "marsen_descriptor_batch_contract_results.json"

CROP_SIZE = 128
CROP_MIN_SIDE_NORM = 0.025
STRICT_TOL = 1e-5


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
    print("MARSEN SPATIAL VERIFIER: DESCRIPTOR BATCH-CONTRACT AUDIT (P-2080, P-2081, P-2084, P-2085)")
    print(f"Driver Script SHA-256: {driver_sha256}")
    print("=" * 115)

    assert TRAIN_JSON.exists(), f"[FATAL] Missing {TRAIN_JSON}"
    assert ZIP_PATH.exists(), f"[FATAL] Missing {ZIP_PATH}"
    assert CERT_QUERIES_PTH.exists(), f"[FATAL] Missing {CERT_QUERIES_PTH}"
    assert CERT_FRAMES_PTH.exists(), f"[FATAL] Missing {CERT_FRAMES_PTH}"
    assert FROZEN_DESCRIPTORS_PTH.exists(), f"[FATAL] Missing {FROZEN_DESCRIPTORS_PTH}"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Target: {device}")

    # P-2081: Record Backend Flags
    backend_info = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }
    print("[*] Backend Environment Ledger:")
    for k, v in backend_info.items():
        print(f"    - {k:<25}: {v}")

    # Visual Extractor
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

    # Ingest Queries & Target Reference Descriptors
    ref_desc = torch.load(FROZEN_DESCRIPTORS_PTH, map_location="cpu")["descriptors"]
    ref_d1 = ref_desc["1.0x_gem_p4"]
    ref_d3 = ref_desc["3.0x_gem_p4"]
    ref_d5 = ref_desc["5.0x_gap"]

    all_queries = torch.load(CERT_QUERIES_PTH, map_location="cpu")["extracted_queries"]
    active_queries = [q for q in all_queries if q["is_active"]]
    n_active = len(active_queries)
    assert n_active == 1915

    # Retrieve Boxes from Frame Tensors
    frame_records = torch.load(CERT_FRAMES_PTH, map_location="cpu")["frame_records"]
    frame_dict = {fr["image_id"]: fr for fr in frame_records}

    for q in active_queries:
        fr = frame_dict[q["image_id"]]
        cx, cy, w, h = fr["temp_boxes"][q["query_id"]].tolist()
        q["pred_box_norm"] = (cx, cy, w, h)

    # Ingest Images into Memory
    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        img_meta = {img["id"]: img for img in json.load(f)["images"]}

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
            rel_path = img_meta[iid]["file_name"].replace("\\", "/").strip("/")
            target_key = "/".join(rel_path.split("/")[-2:]) if len(rel_path.split("/")) >= 2 else rel_path
            zip_entry = zip_index.get(rel_path) or zip_index.get(target_key)
            if zip_entry is None:
                for name in zf.namelist():
                    if name.replace("\\", "/").endswith(rel_path):
                        zip_entry = name
                        break
            with zf.open(zip_entry) as f:
                img_cache[iid] = Image.open(io.BytesIO(f.read())).convert("RGB")

    print(f"[+] Ingested {len(img_cache)} unique images into RAM.")

    # Prepare Preprocessed Tensor Crops (Shared across all contracts)
    crops_1x = []
    crops_3x = []
    crops_5x = []

    for q in active_queries:
        img_pil = img_cache[q["image_id"]]
        box_1x = get_crop_pixel_box(img_pil.size, q["pred_box_norm"], expansion_factor=1.0)
        box_3x = get_crop_pixel_box(img_pil.size, q["pred_box_norm"], expansion_factor=3.0)
        box_5x = get_crop_pixel_box(img_pil.size, q["pred_box_norm"], expansion_factor=5.0)

        crops_1x.append(img_transform(img_pil.crop(box_1x)))
        crops_3x.append(img_transform(img_pil.crop(box_3x)))
        crops_5x.append(img_transform(img_pil.crop(box_5x)))

    # P-2085: Streaming Batch Census
    queries_by_img = defaultdict(list)
    for idx, q in enumerate(active_queries):
        queries_by_img[q["image_id"]].append(idx)

    group_sizes = np.array([len(indices) for indices in queries_by_img.values()], dtype=int)
    print(
        f"[*] Streaming active-query batch sizes: frames={len(group_sizes)}, "
        f"min={group_sizes.min()}, median={np.median(group_sizes):.1f}, "
        f"mean={group_sizes.mean():.2f}, max={group_sizes.max()}"
    )

    # =========================================================================
    # CONTRACT A: Global Fixed Batches (B = 128, Historical B1 Order)
    # =========================================================================
    print("\n[*] Evaluating Contract A (Historical / B1 Global Batches: B = 128)...")
    t_a0 = time.time()
    out_1x_a, out_3x_a, out_5x_a = [], [], []

    with torch.no_grad():
        for i in range(0, n_active, 128):
            b1 = torch.stack(crops_1x[i:i + 128]).to(device)
            b3 = torch.stack(crops_3x[i:i + 128]).to(device)
            b5 = torch.stack(crops_5x[i:i + 128]).to(device)

            f1 = backbone(b1)
            f3 = backbone(b3)
            f5 = backbone(b5)

            out_1x_a.append(gem_pool_2d(f1, p_val=4.0).cpu().numpy().astype(np.float32))
            out_3x_a.append(gem_pool_2d(f3, p_val=4.0).cpu().numpy().astype(np.float32))
            out_5x_a.append(f5.mean(dim=(-2, -1)).cpu().numpy().astype(np.float32))

    res_1x_a = np.concatenate(out_1x_a, axis=0)
    res_3x_a = np.concatenate(out_3x_a, axis=0)
    res_5x_a = np.concatenate(out_5x_a, axis=0)

    diff_1x_a = float(np.max(np.abs(res_1x_a - ref_d1)))
    diff_3x_a = float(np.max(np.abs(res_3x_a - ref_d3)))
    diff_5x_a = float(np.max(np.abs(res_5x_a - ref_d5)))
    print(f"    - Max Delta 1.0x_GeM4: {diff_1x_a:.2e} (Elapsed: {time.time() - t_a0:.2f}s)")
    print(f"    - Max Delta 3.0x_GeM4: {diff_3x_a:.2e}")
    print(f"    - Max Delta 5.0x_GAP:  {diff_5x_a:.2e}")

    # Mandatory assertion on Contract A
    assert max(diff_1x_a, diff_3x_a, diff_5x_a) < STRICT_TOL, (
        f"[FATAL] Contract A failed historical descriptor parity: "
        f"max delta = {max(diff_1x_a, diff_3x_a, diff_5x_a):.2e} >= {STRICT_TOL:.1e}"
    )

    # =========================================================================
    # CONTRACT B: Mode-2A Streaming Batches (Per-Frame Variable Mini-Batches)
    # =========================================================================
    print("\n[*] Evaluating Contract B (Mode-2A Streaming: Per-Frame Variable Mini-Batches)...")
    t_b0 = time.time()
    res_1x_b = np.zeros_like(ref_d1)
    res_3x_b = np.zeros_like(ref_d3)
    res_5x_b = np.zeros_like(ref_d5)

    with torch.no_grad():
        for iid, indices in queries_by_img.items():
            b1 = torch.stack([crops_1x[i] for i in indices]).to(device)
            b3 = torch.stack([crops_3x[i] for i in indices]).to(device)
            b5 = torch.stack([crops_5x[i] for i in indices]).to(device)

            f1 = backbone(b1)
            f3 = backbone(b3)
            f5 = backbone(b5)

            d1 = gem_pool_2d(f1, p_val=4.0).cpu().numpy().astype(np.float32)
            d3 = gem_pool_2d(f3, p_val=4.0).cpu().numpy().astype(np.float32)
            d5 = f5.mean(dim=(-2, -1)).cpu().numpy().astype(np.float32)

            for local_i, global_i in enumerate(indices):
                res_1x_b[global_i] = d1[local_i]
                res_3x_b[global_i] = d3[local_i]
                res_5x_b[global_i] = d5[local_i]

    diff_1x_b = float(np.max(np.abs(res_1x_b - ref_d1)))
    diff_3x_b = float(np.max(np.abs(res_3x_b - ref_d3)))
    diff_5x_b = float(np.max(np.abs(res_5x_b - ref_d5)))
    print(f"    - Max Delta 1.0x_GeM4: {diff_1x_b:.2e} (Elapsed: {time.time() - t_b0:.2f}s)")
    print(f"    - Max Delta 3.0x_GeM4: {diff_3x_b:.2e}")
    print(f"    - Max Delta 5.0x_GAP:  {diff_5x_b:.2e}")

    # =========================================================================
    # CONTRACT C: Fixed-Shape Padded Batches (Frame-Grouped, Padded to B = 128)
    # =========================================================================
    print("\n[*] Evaluating Contract C (Frame-Grouped, Padded to Fixed B = 128)...")
    t_c0 = time.time()
    res_1x_c = np.zeros_like(ref_d1)
    res_3x_c = np.zeros_like(ref_d3)
    res_5x_c = np.zeros_like(ref_d5)

    with torch.no_grad():
        for iid, indices in queries_by_img.items():
            k = len(indices)
            actual_b1 = torch.stack([crops_1x[i] for i in indices])
            actual_b3 = torch.stack([crops_3x[i] for i in indices])
            actual_b5 = torch.stack([crops_5x[i] for i in indices])

            pad_len = 128 - k
            pad_1 = actual_b1[-1:].repeat(pad_len, 1, 1, 1)
            pad_3 = actual_b3[-1:].repeat(pad_len, 1, 1, 1)
            pad_5 = actual_b5[-1:].repeat(pad_len, 1, 1, 1)

            b1_padded = torch.cat([actual_b1, pad_1], dim=0).to(device)
            b3_padded = torch.cat([actual_b3, pad_3], dim=0).to(device)
            b5_padded = torch.cat([actual_b5, pad_5], dim=0).to(device)

            f1 = backbone(b1_padded)[:k]
            f3 = backbone(b3_padded)[:k]
            f5 = backbone(b5_padded)[:k]

            d1 = gem_pool_2d(f1, p_val=4.0).cpu().numpy().astype(np.float32)
            d3 = gem_pool_2d(f3, p_val=4.0).cpu().numpy().astype(np.float32)
            d5 = f5.mean(dim=(-2, -1)).cpu().numpy().astype(np.float32)

            for local_i, global_i in enumerate(indices):
                res_1x_c[global_i] = d1[local_i]
                res_3x_c[global_i] = d3[local_i]
                res_5x_c[global_i] = d5[local_i]

    diff_1x_c = float(np.max(np.abs(res_1x_c - ref_d1)))
    diff_3x_c = float(np.max(np.abs(res_3x_c - ref_d3)))
    diff_5x_c = float(np.max(np.abs(res_5x_c - ref_d5)))
    print(f"    - Max Delta 1.0x_GeM4: {diff_1x_c:.2e} (Elapsed: {time.time() - t_c0:.2f}s)")
    print(f"    - Max Delta 3.0x_GeM4: {diff_3x_c:.2e}")
    print(f"    - Max Delta 5.0x_GAP:  {diff_5x_c:.2e}")

    # =========================================================================
    # DIRECT PAIRWISE COMPARISONS (A <-> B, A <-> C)
    # =========================================================================
    ab_1 = float(np.max(np.abs(res_1x_a - res_1x_b)))
    ab_3 = float(np.max(np.abs(res_3x_a - res_3x_b)))
    ab_5 = float(np.max(np.abs(res_5x_a - res_5x_b)))

    ac_1 = float(np.max(np.abs(res_1x_a - res_1x_c)))
    ac_3 = float(np.max(np.abs(res_3x_a - res_3x_c)))
    ac_5 = float(np.max(np.abs(res_5x_a - res_5x_c)))

    print("\n[*] Direct Pairwise Contract Deltas:")
    print(f"    - A <-> B (Effect of Changing Batch Grouping): 1x={ab_1:.2e}, 3x={ab_3:.2e}, 5x={ab_5:.2e}")
    print(f"    - A <-> C (Restoring B=128 on Grouped Crops):   1x={ac_1:.2e}, 3x={ac_3:.2e}, 5x={ac_5:.2e}")

    # =========================================================================
    # RESULT-DEPENDENT SYNTHESIS TABLE (P-2084)
    # =========================================================================
    status_a = "PASS: Historical parity" if max(diff_1x_a, diff_3x_a, diff_5x_a) < STRICT_TOL else "FAIL"

    mode2a_target = np.array([9.79e-4, 9.79e-4, 4.34e-4])
    mode2a_obs = np.array([diff_1x_b, diff_3x_b, diff_5x_b])
    status_b = "Matches Mode-2A scale" if np.allclose(mode2a_obs, mode2a_target, rtol=0.15, atol=5e-5) else "Different from Mode-2A"

    status_c = "PASS: Fixed-shape restores parity" if max(diff_1x_c, diff_3x_c, diff_5x_c) < STRICT_TOL else "FAIL: parity not restored"

    print("\n" + "=" * 115)
    print("DESCRIPTOR BATCH-CONTRACT SYNTHESIS TABLE (VS. FROZEN REFERENCE 1,915 DESCRIPTORS)")
    print("=" * 115)
    print(f"{'Execution Contract':<35} | {'1.0x_GeM4 Delta':<18} | {'3.0x_GeM4 Delta':<18} | {'5.0x_GAP Delta':<18} | {'Status'}")
    print("-" * 115)
    print(f"{'Contract A (Global B = 128)':<35} | {diff_1x_a:<18.2e} | {diff_3x_a:<18.2e} | {diff_5x_a:<18.2e} | {status_a}")
    print(f"{'Contract B (Streaming Variable N)':<35} | {diff_1x_b:<18.2e} | {diff_3x_b:<18.2e} | {diff_5x_b:<18.2e} | {status_b}")
    print(f"{'Contract C (Padded Fixed B = 128)':<35} | {diff_1x_c:<18.2e} | {diff_3x_c:<18.2e} | {diff_5x_c:<18.2e} | {status_c}")
    print("=" * 115)

    save_payload = {
        "_meta": {
            "driver_sha256": driver_sha256,
            "backend": backend_info,
            "group_size_stats": {
                "n_frames": int(len(group_sizes)),
                "min": int(group_sizes.min()),
                "median": float(np.median(group_sizes)),
                "mean": float(group_sizes.mean()),
                "max": int(group_sizes.max())
            },
            "elapsed_seconds": float(time.time() - t0)
        },
        "contract_a_global_128": {"diff_1x": diff_1x_a, "diff_3x": diff_3x_a, "diff_5x": diff_5x_a, "status": status_a},
        "contract_b_streaming_var": {"diff_1x": diff_1x_b, "diff_3x": diff_3x_b, "diff_5x": diff_5x_b, "status": status_b},
        "contract_c_padded_128": {"diff_1x": diff_1x_c, "diff_3x": diff_3x_c, "diff_5x": diff_5x_c, "status": status_c},
        "pairwise_deltas": {
            "ab": {"1x": ab_1, "3x": ab_3, "5x": ab_5},
            "ac": {"1x": ac_1, "3x": ac_3, "5x": ac_5}
        }
    }
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(save_payload, f, indent=2)

    print(f"[+] Audit Results Frozen: {RESULTS_JSON.resolve()}")
    print("=" * 115)


if __name__ == "__main__":
    main()