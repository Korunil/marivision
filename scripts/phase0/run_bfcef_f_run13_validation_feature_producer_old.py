from __future__ import annotations
"""RUN13 official Validation feature producer (READ ONLY against source/checkpoints).

This generator is deliberately separate from RUN13 evaluation. It reproduces the
frozen E0 canonical R0 taps, F spatial descriptors, E0 temporal association,
and F Kmax7 causal states on the official nine Validation videos.
No Train cache substitutes, training, Test reads, threshold changes or scores.
"""
import argparse
import gc
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image  # preserve Windows import ordering ahead of torch

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "1")
os.environ.setdefault("R0_MSDA_BACKEND", "pytorch")

ROOT = Path("D:/MARIVISION")
EVAL = ROOT / "H0_workbench" / "eval_tools"
VAL_JSON = ROOT / "seaclips-val.json"
ZIP = ROOT / "SeaClips.zip"
R0 = ROOT / "checkpoints_r0" / "checkpoint_best.pth"
OUTPUT = EVAL / "bfcef_f_val" / "run13_val_features_v1"
MANIFEST = OUTPUT / "run13_validation_feature_manifest.json"
EXPECTED_R0 = "28e6288c3f6769c2d5275aa77ddb248735b55a09cd46a6b570e3f117890b268b"
EXPECTED_SOURCES = {
    "run_bfcef_e0_substrate.py": "11a4524c69692bf9d9f177991dc52bfa1bb92278b18399bc83d07844aba84eaa",
    "build_bfcef_e0_canonical_cache.py": "8fd75ebc84f2418a5cc36f2bb160e140a8f940299d1cff35533d154db478f4d7",
    "build_bfcef_e0_temporal_association_cache.py": "4bdb2dfdd2daf1cd786093e59795ab4cd1e6d441bd4369cf7bbcc151c82e519f",
    "build_bfcef_f_spatial_evidence_cache.py": "c825c5379a5a976f5264c0e8165eb8a721bf01b9996cfbbaacb65c2ac9c201ba",
    "build_bfcef_f_temporal_state_cache.py": "dd4bf0bc13e175ca9e28ded4e0d8978f666480e4445a7807b3b7642d94b579b5",
}
EXPECTED_VIDEOS = 9
EXPECTED_FRAMES = 4266
EXPECTED_BOXES = 13699
EXPECTED_NAMES = ("ANIMAL", "BOAT", "MARINE_MARKER", "OBJECT", "LEISURE_VEHICLE", "SAILING_VESSEL", "SHIP")


def require(ok, why):
    if not ok:
        raise AssertionError("[FATAL] " + why)


def sha(path):
    p = Path(path)
    require(p.is_file(), f"Missing source: {p}")
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def find_repo():
    for repo in (ROOT / "Deformable-DETR", ROOT / "deformable_detr"):
        if (repo / "datasets" / "seaclips_detection.py").is_file() and (repo / "models" / "deformable_detr.py").is_file():
            return repo
    raise AssertionError("[FATAL] Authoritative Deformable-DETR repo not found")


def source_preflight():
    require(VAL_JSON.is_file(), f"Official Validation annotation missing: {VAL_JSON}")
    require(ZIP.is_file(), f"Source image archive missing: {ZIP}")
    require(sha(R0) == EXPECTED_R0, "Frozen R0 checkpoint SHA mismatch")
    for filename, digest in EXPECTED_SOURCES.items():
        observed = sha(EVAL / filename)
        require(observed == digest, f"{filename} source drift: {observed} != {digest}")
    repo = find_repo()
    # Verify the original Train-side cache roots separately; do not load Train records.
    print("[PASS] R0 and all five frozen feature-source SHA-256 values verified")
    return repo


def order_video_frames(frames):
    """Require an explicit chronological key, never silently sort by image ID."""
    require(frames, "Empty Validation video")
    candidates = ("frame_id", "frame_index", "frame_number", "seq_idx")
    for key in candidates:
        if all(key in x and isinstance(x[key], (int, float)) for x in frames):
            values = [float(x[key]) for x in frames]
            require(len(values) == len(set(values)), f"Duplicate {key} in Validation video")
            return sorted(frames, key=lambda x: float(x[key])), key
    # If file stems are numeric (e.g. frame_000012.jpg), use that numeric index.
    names = [Path(str(x.get("file_name", ""))).stem for x in frames]
    parsed = [re.search(r"(?:^|[_\-])(\d+)$", stem) for stem in names]
    if all(parsed):
        keys = [int(m.group(1)) for m in parsed]
        require(len(keys) == len(set(keys)), "Duplicate numeric filename frame index")
        return [x for _, x in sorted(zip(keys, frames), key=lambda y: y[0])], "file_name_numeric_suffix"
    raise AssertionError("[FATAL] No unambiguous chronological order for Validation video. "
                         "Need frame_id/frame_index/frame_number/seq_idx or numeric filename suffix; "
                         "do NOT substitute image_id ordering")


def inventory():
    meta = json.loads(VAL_JSON.read_text(encoding="utf-8"))
    for key in ("images", "annotations", "videos", "categories"):
        require(isinstance(meta.get(key), list), f"Validation JSON missing {key} list")
    images, anns, videos = meta["images"], meta["annotations"], meta["videos"]
    require((len(images), len(anns), len(videos)) == (EXPECTED_FRAMES, EXPECTED_BOXES, EXPECTED_VIDEOS),
            f"Val census unexpected: images={len(images)} annotations={len(anns)} videos={len(videos)}")
    cats = {int(c["id"]): str(c["name"]).upper().replace(" ", "_").replace("-", "_") for c in meta["categories"]}
    require(tuple(sorted(cats)) == tuple(range(1, 8)) and tuple(cats[i] for i in range(1, 8)) == EXPECTED_NAMES,
            f"Unexpected categories: {cats}")
    vid_ids = [int(v["id"]) for v in videos]
    require(len(set(vid_ids)) == EXPECTED_VIDEOS, "Duplicate Val video IDs")
    iid = [int(x["id"]) for x in images]
    require(len(set(iid)) == EXPECTED_FRAMES, "Duplicate Validation image IDs")
    groups = defaultdict(list)
    for im in images:
        vid = int(im["video_id"])
        require(vid in vid_ids and int(im["width"]) > 0 and int(im["height"]) > 0, "Invalid video ID or dimensions")
        groups[vid].append(im)
    image_map = {int(i["id"]): i for i in images}
    by_image = defaultdict(list)
    for a in anns:
        ii = int(a["image_id"])
        require(ii in image_map and int(a["category_id"]) in cats, "Invalid GT image or category reference")
        bbox = [float(x) for x in a["bbox"]]
        require(len(bbox) == 4 and all(__import__('math').isfinite(x) for x in bbox), "Invalid GT bbox")
        require(bbox[2] > 0 and bbox[3] > 0, "Invalid GT width/height; reject silent GT filtering")
        by_image[ii].append(a)
    ordered, schemes = {}, {}
    for vid in sorted(vid_ids):
        ordered[vid], schemes[vid] = order_video_frames(groups[vid])
        require(len(ordered[vid]) > 0, "Empty validation video")
    print(f"[PASS] Official Validation census: {len(videos)} videos, {len(images)} frames, {len(anns)} GT; categories exact")
    print("[+] Chronological fields:", schemes)
    return meta, ordered, by_image, schemes


def load_algorithms(repo):
    # Mimic the canonical E0 import precedence.
    for p in reversed((EVAL, repo, ROOT)):
        if str(p) in sys.path:
            sys.path.remove(str(p))
        sys.path.insert(0, str(p))
    import torch
    from util.misc import collate_fn
    from datasets.seaclips_detection import SeaClipsDetection
    from models.deformable_detr import build as build_deformable_detr
    import run_bfcef_e0_substrate as substrate
    import build_bfcef_e0_canonical_cache as e0
    import build_bfcef_e0_temporal_association_cache as assoc
    import build_bfcef_f_spatial_evidence_cache as spatial
    import build_bfcef_f_temporal_state_cache as temporal
    assert assoc.ASSOCIATION_VERSION == temporal.FROZEN_ASSOCIATION_VERSION
    assert spatial.SPATIAL_SCHEMA_VERSION == "bfcef_f_spatial_v1_s21_l24"
    return torch, collate_fn, SeaClipsDetection, build_deformable_detr, substrate, e0, assoc, spatial, temporal


def build_all():
    repo = source_preflight()
    meta, ordered, by_image, schemes = inventory()
    require(not OUTPUT.exists(), f"Validation output directory already exists: {OUTPUT}. Refuse overwrite/re-run")
    (torch, collate_fn, SeaClipsDetection, build_deformable_detr,
     substrate, e0, assoc, spatial, temporal) = load_algorithms(repo)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = substrate.load_authoritative_r0_args()
    args.device = str(device)
    model, _, _ = build_deformable_detr(args)
    weights = torch.load(R0, map_location="cpu", weights_only=False)
    raw = weights.get("model", weights)
    cleaned = {k[7:] if k.startswith("module.") else k:v for k,v in raw.items()}
    model.load_state_dict(cleaned, strict=True)
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    dataset = SeaClipsDetection(ZIP, VAL_JSON, is_train=False, random_flip=False)
    idx_list = getattr(dataset, "ids", getattr(dataset, "img_ids", None))
    require(idx_list is not None, "Dataset loader lacks authoritative image ID inventory")
    iid_to_idx = {int(i):j for j,i in enumerate(idx_list)}
    all_ids = {int(i["id"]) for i in meta["images"]}
    require(len(iid_to_idx) == EXPECTED_FRAMES and set(iid_to_idx) == all_ids,
            "SeaClipsDetection filtered/reordered Validation images unexpectedly")

    taps = {}
    def hook(module, input_, output):
        srcs=input_[0]; hs, init_ref, inter_refs=output[0],output[1],output[2]
        last=hs.shape[0]-1
        taps.clear()
        taps.update(p3=srcs[0].detach(),p4=srcs[1].detach(),p5=srcs[2].detach(),
                    q_native=hs[last].detach(),ref_pts=(init_ref if last == 0 else inter_refs[last-1]).detach())
    handle=model.transformer.register_forward_hook(hook)
    OUTPUT.mkdir(parents=True,exist_ok=False)
    ledger=[]; total_frames=0; total_gt=0
    try:
        for vid, frames in sorted(ordered.items()):
            canonical=[]; spatial_recs=[]
            service=assoc.CanonicalTemporalAssociationService()
            service.reset(vid)
            first_order=[]
            with torch.no_grad():
                for seq_idx, im in enumerate(frames):
                    iid=int(im["id"])
                    img,target=dataset[iid_to_idx[iid]]
                    require(tuple(img.shape[-2:]) == (640,640),
                            f"R0 image preprocessing not 640x640 on image {iid}: {tuple(img.shape)}")
                    samples,_=collate_fn([(img,target)])
                    samples=samples.to(device)
                    output=model(samples)
                    dl=output["pred_logits"][0].detach()
                    db=output["pred_boxes"][0].detach()
                    q=taps["q_native"][0].detach()
                    refs=taps["ref_pts"][0].detach()
                    p3,p4,p5=taps["p3"],taps["p4"],taps["p5"]
                    require(tuple(dl.shape)==(300,7) and tuple(db.shape)==(300,4) and tuple(q.shape)==(300,256),
                            f"R0 shapes wrong on image {iid}")
                    gt=target["boxes"].detach().float().cpu()
                    gl=target["labels"].detach().long().cpu()
                    ann=by_image[iid]
                    require(len(ann)==len(gt)==len(gl), f"Dataset transformed/dropped GT on image {iid}")
                    wh=float(im["width"]);hh=float(im["height"])
                    expected=sorted((int(a["category_id"])-1,
                                     (float(a["bbox"][0])+float(a["bbox"][2])/2)/wh,
                                     (float(a["bbox"][1])+float(a["bbox"][3])/2)/hh,
                                     float(a["bbox"][2])/wh,float(a["bbox"][3])/hh) for a in ann)
                    seen=sorted((int(c), *map(float,b)) for c,b in zip(gl.tolist(),gt.tolist()))
                    require(len(seen)==len(expected) and all(x[0]==y[0] and max(abs(a-b) for a,b in zip(x[1:],y[1:]))<2e-3 for x,y in zip(seen,expected)),
                            f"Val GT transformed unexpectedly on image {iid}")
                    cues=e0.extract_compact_raw_cues(db,dl,refs,p3,p4,p5,img_w=640.,img_h=640.)
                    tp=e0.compute_canonical_baseline_tp_mask(dl,db,gt.to(device),gl.to(device))
                    fr={"image_id":iid,"video_id":vid,"seq_idx":seq_idx,"img_w":640.,"img_h":640.,
                        "d_logits":dl.float().cpu(),"d_boxes":db.float().cpu(),"q_native":q.float().cpu(),"ref_pts":refs.float().cpu(),
                        **cues,"gt_boxes_norm":gt,"gt_labels":gl,"canonical_tp_mask":tp.cpu()}
                    ss=spatial.compute_scale_spatial_descriptor(db,{"P3":p3,"P4":p4,"P5":p5})
                    ll=spatial.compute_localization_spatial_descriptor(db,{"P3":p3,"P4":p4,"P5":p5})
                    require(tuple(ss.shape)==(300,21) and tuple(ll.shape)==(300,24), f"Spatial tensor drift image {iid}")
                    require(torch.isfinite(dl).all() and torch.isfinite(db).all() and torch.isfinite(ss).all() and torch.isfinite(ll).all(),
                            f"Nonfinite R0 or spatial evidence on image {iid}")
                    spatial_recs.append({"image_id":iid,"video_id":vid,"seq_idx":seq_idx,
                                         "scale_spatial":ss.detach().float().cpu(),"loc_spatial":ll.detach().float().cpu()})
                    first=service.associate(fr["d_boxes"].to(device),fr["appearance_desc"].to(device),fr["d_logits"].to(device).sigmoid())
                    first_order.append({"image_id":iid,"video_id":vid,"seq_idx":seq_idx,**{k:v.cpu() for k,v in first.items()}})
                    canonical.append(fr)
                    total_gt+=len(gl)
                    if seq_idx % 100==0:
                        print(f"    Val video={vid} frames={seq_idx+1}/{len(frames)}",flush=True)
                    del img,target,samples,output,dl,db,q,refs,p3,p4,p5,ss,ll
            temporal.verify_frame_alignment(canonical,first_order,vid)
            temporal_recs=[]
            for j,fr in enumerate(canonical):
                state=temporal.build_frame_kstep_state(canonical,first_order,j,7)
                temporal_recs.append({"image_id":fr["image_id"],"video_id":vid,"seq_idx":j,**state})
            record={"video_id":vid,"frames":len(canonical)}
            for label,values in (("canonical",canonical),("spatial",spatial_recs),("temporal",temporal_recs)):
                fname=f"val_vid{vid}_{label}.pth"
                outfile=OUTPUT/fname
                torch.save(values,outfile)
                record[label]={"path":fname,"sha256":sha(outfile)}
            ledger.append(record)
            total_frames+=len(canonical)
            print(f"[PASS] Val video {vid} {len(canonical)} frames, GT {sum(len(x['gt_labels']) for x in canonical)}; three shards SHA certified",flush=True)
            del canonical,spatial_recs,first_order,temporal_recs
            gc.collect()
            if device.type=="cuda":torch.cuda.empty_cache()
    finally:
        handle.remove()
    require(total_frames==EXPECTED_FRAMES and total_gt==EXPECTED_BOXES and len(ledger)==EXPECTED_VIDEOS,
            f"Final Val cache census failed: {total_frames} frames, {total_gt} GT, {len(ledger)} videos")
    manifest={"schema":"bfcef_f_run13_val_features_v1","split":"validation",
              "annotation_sha256":sha(VAL_JSON),"r0_checkpoint_sha256":EXPECTED_R0,
              "image_preprocessing":"R0_640x640_nonaspect_warp_frozen",
              "temporal_recipe":"RUN07_S0A_K3_from_validated_Kmax7",
              "spatial_recipe":"RUN07_S0A_scale21_loc24",
              "uses_train_cache_as_validation":False,"test_accessed":False,
              "num_frames":total_frames,"num_gt":total_gt,
              "producer":{"path":Path(__file__).name,"sha256":sha(Path(__file__))},
              "source_hashes":EXPECTED_SOURCES,"chronological_order_fields":schemes,
              "videos":ledger}
    MANIFEST.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print("[PASS] RUN13 Validation feature generation finished; all 27 shards + manifest certified")
    print("Manifest:",MANIFEST,"SHA:",sha(MANIFEST))
    print("STOP: DO NOT OPEN TEST. RUN13 input-preflight for independent audit.")


def selftest():
    def sample(vid,fn):return {"id":vid,"frame_id":fn,"file_name":f"frame_{fn:06}.jpg"}
    data=[sample(10,5),sample(12,1),sample(13,3)]
    ordered,how=order_video_frames(data)
    require([x["frame_id"] for x in ordered]==[1,3,5] and how=="frame_id", "Chronology test failed")
    no_frame=[{"file_name":"abc.jpg"}]
    try:order_video_frames(no_frame)
    except AssertionError:pass
    else:raise AssertionError("Chronology fail-close test failed")
    print("[PASS] RUN13 Validation producer self-tests; no dataset opened")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--phase",choices=("selftest","inventory","build"),required=True)
    a=p.parse_args()
    selftest()
    if a.phase=="selftest":return
    repo=source_preflight()
    inventory()
    if a.phase=="inventory":
        require(not OUTPUT.exists(),f"Output already exists: {OUTPUT}")
        print("[PASS] Val input inventory only; no R0 inference and no model tuning")
        return
    build_all()


if __name__=="__main__":
    main()
