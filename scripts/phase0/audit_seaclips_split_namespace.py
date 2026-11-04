"""
D:/MARIVISION/H0_workbench/eval_tools/audit_seaclips_split_namespace.py
MARSEN Stream B2: Train/Val Split-Namespace & Archive Audit (P-2104..P-2107)

Audits split-qualified archive resolution across official Train (47) and Val (9) splits:
  1. Fail-Closed Split Index: Confirms exact 1:1 mapping to train/... and val/...
  2. Decoded-Pixel Disjointness: Computes raw and decoded RGB SHA-256 across all 3,484 collisions.
  3. Exhaustive Loader Fidelity: Tests both SeaClipsDetection (R0) and GeneralizedStreamEngine (Spatial)
     across all 20,310 Train and 4,266 Val frames.
  4. Mode-2A Retrospective: Checks the 10 design videos (including the 206 historical frames).
Test split remains completely unopened per D-4142. Runtime: ~20 seconds.
"""

import io
import sys
import time
import json
import zipfile
import hashlib
from pathlib import Path
from PIL import Image
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)

OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
ROOT_DIR = Path("D:/MARIVISION")
ZIP_PATH = ROOT_DIR / "SeaClips.zip"
TRAIN_JSON = ROOT_DIR / "seaclips-train.json"
VAL_JSON = ROOT_DIR / "seaclips-val.json"

EXPECTED_COUNTS = {"train": 20310, "val": 4266}
DESIGN_VIDEOS = [4, 7, 16, 19, 25, 29, 30, 32, 35, 47]


def compute_bytes_sha256(data_bytes):
    return hashlib.sha256(data_bytes).hexdigest()


def compute_pixel_sha256(img_pil):
    rgb_img = img_pil.convert("RGB")
    payload = str(rgb_img.size).encode() + rgb_img.mode.encode() + rgb_img.tobytes()
    return hashlib.sha256(payload).hexdigest()


def normalize_zip_path(path_str):
    return str(path_str).replace("\\", "/").strip("/").lstrip("/")


def main():
    t_start = time.time()
    print("=" * 115, flush=True)
    print("MARSEN STREAM B2: TRAIN/VAL SPLIT-NAMESPACE & ARCHIVE AUDIT (P-2104..P-2107)", flush=True)
    print("=" * 115, flush=True)

    assert ZIP_PATH.exists(), f"[FATAL] Missing {ZIP_PATH}"
    assert TRAIN_JSON.exists(), f"[FATAL] Missing {TRAIN_JSON}"
    assert VAL_JSON.exists(), f"[FATAL] Missing {VAL_JSON}"

    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        data_train = json.load(f)
    with open(VAL_JSON, "r", encoding="utf-8") as f:
        data_val = json.load(f)

    zf = zipfile.ZipFile(ZIP_PATH, "r")
    all_members = [normalize_zip_path(m) for m in zf.namelist()]
    print(f"[*] Loaded SeaClips.zip: {len(all_members)} total archive entries.", flush=True)

    # =========================================================================
    # PART 1: Fail-Closed Split Index (P-2104)
    # =========================================================================
    print("\n" + "=" * 115, flush=True)
    print("PART 1: FAIL-CLOSED SPLIT-QUALIFIED ARCHIVE RESOLUTION AUDIT (P-2104)", flush=True)
    print("=" * 115, flush=True)

    split_member_index = defaultdict(list)
    for raw_name in zf.namelist():
        norm = normalize_zip_path(raw_name)
        parts = norm.split("/")
        for split in ["train", "val"]:
            if split in parts:
                split_idx = parts.index(split)
                rel_fn = "/".join(parts[split_idx + 1:])
                split_member_index[(split, rel_fn)].append(raw_name)

    resolved_train = {}
    missing_train = []
    ambiguous_train = []

    for img in data_train["images"]:
        fn = normalize_zip_path(img["file_name"])
        matches = split_member_index[("train", fn)]
        if len(matches) == 1:
            resolved_train[img["id"]] = (fn, matches[0])
        elif len(matches) == 0:
            missing_train.append((img["id"], fn))
        else:
            ambiguous_train.append((img["id"], fn, matches))

    resolved_val = {}
    missing_val = []
    ambiguous_val = []

    for img in data_val["images"]:
        fn = normalize_zip_path(img["file_name"])
        matches = split_member_index[("val", fn)]
        if len(matches) == 1:
            resolved_val[img["id"]] = (fn, matches[0])
        elif len(matches) == 0:
            missing_val.append((img["id"], fn))
        else:
            ambiguous_val.append((img["id"], fn, matches))

    print(f"[*] Train Split Resolution: {len(resolved_train)} / {EXPECTED_COUNTS['train']} uniquely resolved", flush=True)
    print(f"    - Missing: {len(missing_train)} | Ambiguous: {len(ambiguous_train)}", flush=True)
    print(f"[*] Val Split Resolution:   {len(resolved_val)} / {EXPECTED_COUNTS['val']} uniquely resolved", flush=True)
    print(f"    - Missing: {len(missing_val)} | Ambiguous: {len(ambiguous_val)}", flush=True)

    assert len(missing_train) == 0 and len(ambiguous_train) == 0, "[FATAL] Train split failed 1:1 resolution!"
    assert len(missing_val) == 0 and len(ambiguous_val) == 0, "[FATAL] Val split failed 1:1 resolution!"
    print("[+] P-2104 Passed: 100% of Train and Val images map uniquely to exactly one physical member.", flush=True)

    # =========================================================================
    # PART 2: Decoded-Pixel Identity Audit (P-2105)
    # =========================================================================
    print("\n" + "=" * 115, flush=True)
    print("PART 2: DECODED-PIXEL IDENTITY AUDIT ACROSS 3,484 OVERLAPPING NAMES (P-2105)", flush=True)
    print("=" * 115, flush=True)

    train_fn_set = {normalize_zip_path(img["file_name"]) for img in data_train["images"]}
    val_fn_set = {normalize_zip_path(img["file_name"]) for img in data_val["images"]}
    overlap_fns = sorted(list(train_fn_set.intersection(val_fn_set)))

    print(f"[*] Shared Relative Path Strings: {len(overlap_fns)} paths.", flush=True)
    print("[*] Auditing raw file bytes and decoded RGB pixels...", flush=True)

    exact_file_dups = 0
    exact_pixel_dups = 0
    truly_distinct = 0
    sample_pairs = []

    for idx, fn in enumerate(overlap_fns):
        m_train = split_member_index[("train", fn)][0]
        m_val = split_member_index[("val", fn)][0]

        with zf.open(m_train) as f_t, zf.open(m_val) as f_v:
            bytes_t = f_t.read()
            bytes_v = f_v.read()

        raw_sha_t = compute_bytes_sha256(bytes_t)
        raw_sha_v = compute_bytes_sha256(bytes_v)

        raw_identical = (raw_sha_t == raw_sha_v)
        if raw_identical:
            exact_file_dups += 1

        img_t = Image.open(io.BytesIO(bytes_t))
        img_v = Image.open(io.BytesIO(bytes_v))

        pix_sha_t = compute_pixel_sha256(img_t)
        pix_sha_v = compute_pixel_sha256(img_v)

        is_pix_same = (pix_sha_t == pix_sha_v)
        if is_pix_same:
            exact_pixel_dups += 1
        else:
            truly_distinct += 1

        if len(sample_pairs) < 3:
            sample_pairs.append((fn, m_train, raw_sha_t[:8], pix_sha_t[:8], m_val, raw_sha_v[:8], pix_sha_v[:8], is_pix_same))

        if (idx + 1) % 1000 == 0 or (idx + 1) == len(overlap_fns):
            print(f"    - Audited {idx + 1:>4} / {len(overlap_fns)} collision pairs...", flush=True)

    print("\n[*] Physical Content Audit Findings:", flush=True)
    print(f"    - Exact Raw File Duplicates:         {exact_file_dups:>5} / {len(overlap_fns)}", flush=True)
    print(f"    - Total Decoded RGB Pixel Duplicates:{exact_pixel_dups:>5} / {len(overlap_fns)}", flush=True)
    print(f"    - Physically Distinct Image Content: {truly_distinct:>5} / {len(overlap_fns)} ({(truly_distinct/len(overlap_fns))*100:.2f}%)", flush=True)

    print("\n[*] Sample Pair Inspections:", flush=True)
    for fn, mt, rst, pst, mv, rsv, psv, is_same in sample_pairs:
        print(f"    - Relative Path: {fn}", flush=True)
        print(f"      Train: Member={mt} | Raw={rst}... | Pixels={pst}...", flush=True)
        print(f"      Val:   Member={mv} | Raw={rsv}... | Pixels={psv}...", flush=True)
        print(f"      Visual Duplicate: {is_same}", flush=True)

    # =========================================================================
    # PART 3: Full Loader Fidelity & Cross-Split Misdirection Audit (P-2106)
    # =========================================================================
    print("\n" + "=" * 115, flush=True)
    print("PART 3: EXHAUSTIVE LOADER FIDELITY AUDIT (P-2106)", flush=True)
    print("=" * 115, flush=True)

    sys.path.insert(0, str(OUT_DIR))
    import audit_pol3v_operating_point_sweep as opsweep

    # 1. Audit SeaClipsDetection upstream resolver
    print("[*] Auditing SeaClipsDetection (R0 / Temporal Upstream Loader)...", flush=True)
    ds_train = opsweep.SeaClipsDetection(ZIP_PATH, TRAIN_JSON, is_train=False, random_flip=False)
    ds_val = opsweep.SeaClipsDetection(ZIP_PATH, VAL_JSON, is_train=False, random_flip=False)

    def resolve_ds(ds, fn):
        res = ds._resolve_zip_member(fn)
        if hasattr(res, "filename"):
            return normalize_zip_path(res.filename)
        return normalize_zip_path(str(res))

    r0_train_wrong = 0
    for img_id in ds_train.img_ids:
        img_info = ds_train.coco.loadImgs(img_id)[0]
        fn = normalize_zip_path(img_info["file_name"])
        expected = normalize_zip_path(split_member_index[("train", fn)][0])
        actual = normalize_zip_path(resolve_ds(ds_train, fn))
        if actual != expected:
            r0_train_wrong += 1

    r0_val_wrong = 0
    for img_id in ds_val.img_ids:
        img_info = ds_val.coco.loadImgs(img_id)[0]
        fn = normalize_zip_path(img_info["file_name"])
        expected = normalize_zip_path(split_member_index[("val", fn)][0])
        actual = normalize_zip_path(resolve_ds(ds_val, fn))
        if actual != expected:
            r0_val_wrong += 1

    print(f"    - SeaClipsDetection Train Resolution: {len(ds_train.img_ids) - r0_train_wrong} / {len(ds_train.img_ids)} correct (Wrong: {r0_train_wrong})", flush=True)
    print(f"    - SeaClipsDetection Val Resolution:   {len(ds_val.img_ids) - r0_val_wrong} / {len(ds_val.img_ids)} correct (Wrong: {r0_val_wrong})", flush=True)

    # 2. Audit GeneralizedStreamEngine native crop loader
    print("\n[*] Auditing GeneralizedStreamEngine.load_frame_image() (Spatial Crop Loader)...", flush=True)
    engine_zip_index = {normalize_zip_path(name): name for name in zf.namelist()}
    for name in zf.namelist():
        norm = normalize_zip_path(name)
        parts = norm.split("/")
        if len(parts) >= 2:
            engine_zip_index["/".join(parts[-2:])] = name

    def engine_resolve(fn):
        rel_path = normalize_zip_path(fn)
        target_key = "/".join(rel_path.split("/")[-2:]) if len(rel_path.split("/")) >= 2 else rel_path
        res = engine_zip_index.get(rel_path) or engine_zip_index.get(target_key)
        return normalize_zip_path(res) if res else None

    spatial_train_wrong = 0
    for img in data_train["images"]:
        fn = normalize_zip_path(img["file_name"])
        expected = normalize_zip_path(split_member_index[("train", fn)][0])
        actual = engine_resolve(fn)
        if actual != expected:
            spatial_train_wrong += 1

    spatial_val_wrong = 0
    for img in data_val["images"]:
        fn = normalize_zip_path(img["file_name"])
        expected = normalize_zip_path(split_member_index[("val", fn)][0])
        actual = engine_resolve(fn)
        if actual != expected:
            spatial_val_wrong += 1

    print(f"    - MARSEN Native Loader Train Resolution: {len(data_train['images']) - spatial_train_wrong} / {len(data_train['images'])} correct (Wrong: {spatial_train_wrong})", flush=True)
    print(f"    - MARSEN Native Loader Val Resolution:   {len(data_val['images']) - spatial_val_wrong} / {len(data_val['images'])} correct (Wrong: {spatial_val_wrong})", flush=True)

    # =========================================================================
    # PART 4: Mode-2A Design Cohort Retrospective Check
    # =========================================================================
    print("\n" + "=" * 115, flush=True)
    print("PART 4: MODE-2A DESIGN COHORT RETROSPECTIVE CHECK (10 DESIGN VIDEOS)", flush=True)
    print("=" * 115, flush=True)

    design_imgs = [img for img in data_train["images"] if int(img["video_id"]) in DESIGN_VIDEOS]
    print(f"[*] Total Design Cohort Frames: {len(design_imgs)} frames across 10 videos.", flush=True)

    design_r0_wrong = 0
    design_spatial_wrong = 0

    for img in design_imgs:
        fn = normalize_zip_path(img["file_name"])
        expected = normalize_zip_path(split_member_index[("train", fn)][0])

        actual_r0 = normalize_zip_path(resolve_ds(ds_train, fn))
        if actual_r0 != expected:
            design_r0_wrong += 1

        actual_spatial = engine_resolve(fn)
        if actual_spatial != expected:
            design_spatial_wrong += 1

    print(f"[*] Mode-2A Design Cohort Retrospective:", flush=True)
    print(f"    - R0 Upstream Misdirected Frames:  {design_r0_wrong:>3} / {len(design_imgs)}", flush=True)
    print(f"    - Spatial Crop Misdirected Frames: {design_spatial_wrong:>3} / {len(design_imgs)}", flush=True)

    if design_r0_wrong == 0 and design_spatial_wrong == 0:
        print("[+] VERDICT: Certified. Mode-2A executed strictly on authentic train/ assets with zero cross-split contamination.", flush=True)
    else:
        print(f"[!] VERDICT: Impact detected. Cross-split contamination occurred on {design_spatial_wrong} design frames.", flush=True)

    zf.close()
    print("\n" + "=" * 115, flush=True)
    print(f"AUDIT COMPLETE (Elapsed: {time.time() - t_start:.2f}s)", flush=True)
    print("=" * 115, flush=True)


if __name__ == "__main__":
    main()