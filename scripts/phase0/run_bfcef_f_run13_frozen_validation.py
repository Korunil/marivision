from __future__ import annotations
"""RUN_13: single-use frozen official Validation evaluation.

IMPORTANT: RUN_07--RUN_12 caches are TRAIN ONLY. This runner deliberately
requires a separately certified 9-video Validation canonical/spatial/temporal
cache manifest; it DOES NOT create or infer Validation features from Train.
No training, no threshold adaptation, no Test access, no checkpoint writes.
"""
import argparse
import contextlib
import copy
import csv
import gc
import hashlib
import io
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image  # Windows PIL DLL / import ordering compatibility

for key, val in {"OMP_NUM_THREADS":"1", "MKL_NUM_THREADS":"1", "OPENBLAS_NUM_THREADS":"1", "R0_MSDA_BACKEND":"pytorch"}.items():
    os.environ.setdefault(key, val)

ROOT = Path("D:/MARIVISION")
EVAL = ROOT / "H0_workbench" / "eval_tools"
RUN12 = EVAL / "bfcef_f_s0" / "run12_final_train_v1"
F_CKPT = RUN12 / "run12_frozen_final_candidate.pth"
F_CERT = RUN12 / "run12_frozen_final_candidate_certificate.json"
E_CERT = RUN12 / "run12_fulltrain_experts_certificate.json"
D_CERT = RUN12 / "run12_fulltrain_downstream_certificate.json"
P_PREFLIGHT = RUN12 / "run12_validation_preflight.json"
VAL_JSON = ROOT / "seaclips-val.json"
VAL_ROOT = EVAL / "bfcef_f_val" / "run13_val_features_v1"
VAL_MANIFEST = VAL_ROOT / "run13_validation_feature_manifest.json"
OUT = EVAL / "bfcef_f_s0" / "run13_frozen_independent_validation_v1"

SHA_FINAL = "1211292b180eaf00862fc8098ad5cda6a2f64701c0954b5f8f752f090d2ce84c"
SHA_EXPERT = "9a7102d2aa0035b4d1f76cd416fae56d43c0b199e1614f6143f6466f58f91e1d"
SHA_DOWNSTREAM = "37264115931693c6da14fff0e2c403c705dbbb072c6466b2764e1b0b11f654c3"
SHA_PREFLIGHT = "5a1c307745ece853a1dcfd64716f816302b59660e93428799959a5c0df83a589"
SHA_R0 = "28e6288c3f6769c2d5275aa77ddb248735b55a09cd46a6b570e3f117890b268b"
SHA_R12_SOURCE = "f4f778f5c9132c80afbd121dda6f6b12fbfb5c6bd87cc2507e484e622a662b8a"
SHA_R09D_SOURCE = "5e15f4c02dd99e450de22b54db085277f1c2cfcd43c6482e17ab93f0f687d812"

EXPECTED_FRAMES = 4266
EXPECTED_GT = 13699
EXPECTED_VIDEOS = 9
EXPECTED_CLASSES = (1, 2, 3, 4, 5, 6, 7)
EXPECTED_CLASS_NAMES = (
    "ANIMAL", "BOAT", "MARINE_MARKER", "OBJECT",
    "LEISURE_VEHICLE", "SAILING_VESSEL", "SHIP"
)
SENSORS = ("e-CAM", "FLIR", "Axis")
EXPERTS = ("S", "L", "C", "T")


def require(ok, reason):
    if not ok:
        raise AssertionError("[FATAL] " + reason)


def sha(path):
    path = Path(path)
    require(path.is_file(), f"Missing file: {path}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def exact_sha(path, expected, label):
    actual = sha(path)
    require(actual == expected, f"{label} SHA mismatch: expected {expected}, observed {actual}")


def jload(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_path(base, rel):
    require(isinstance(rel, str) and rel and not Path(rel).is_absolute(), "Invalid relative shard path")
    p = (base / rel).resolve()
    require(p.is_relative_to(base.resolve()), "Shard escaped Validation cache directory")
    return p


def metrics(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"tp":int(tp),"fp":int(fp),"fn":int(fn),"precision":p,"recall":r,"f1":f}


def gate_decision(base, full):
    reasons = []
    if full["map30"] < base["map30"]: reasons.append("mAP@0.30 regressed")
    if full["f1"] <= base["f1"]: reasons.append("F1 did not improve")
    if full["fp"] >= base["fp"]: reasons.append("False positives did not decrease")
    if full["recall"] < base["recall"] - 0.01: reasons.append("Recall declined more than 1 pp")
    return {"validation_gate": "GO_FOR_TEST_PREPARATION_REVIEW" if not reasons else "HOLD", "reasons":reasons}


def selftest():
    assert metrics(1,1,0)["f1"] == 2/3
    assert gate_decision({"map30":0.2,"f1":0.5,"fp":10,"recall":0.8},
                         {"map30":0.21,"f1":0.55,"fp":8,"recall":0.795})["validation_gate"].startswith("GO")
    assert gate_decision({"map30":0.2,"f1":0.5,"fp":10,"recall":0.8},
                         {"map30":0.19,"f1":0.55,"fp":8,"recall":0.795})["validation_gate"] == "HOLD"
    assert EXPECTED_CLASSES == tuple(range(1,8)) and EXPECTED_VIDEOS == 9
    print("[PASS] RUN13 policy/math selftests; no Validation/Test files opened")


def source_preflight():
    exact_sha(F_CKPT, SHA_FINAL, "Frozen final candidate")
    exact_sha(P_PREFLIGHT, SHA_PREFLIGHT, "RUN12 numerical preflight")
    exact_sha(EVAL / "run_bfcef_f_run12_final_train.py", SHA_R12_SOURCE, "RUN12 frozen driver")
    exact_sha(EVAL / "audit_bfcef_f_run09d_calibrated_qualification.py", SHA_R09D_SOURCE, "RUN09D evaluator")
    f = jload(F_CERT)
    e = jload(E_CERT)
    d = jload(D_CERT)
    p = jload(P_PREFLIGHT)
    require(f["checkpoint_sha256"] == SHA_FINAL, "Final certificate drift")
    require(e["checkpoint_sha256"] == SHA_EXPERT and d["checkpoint_sha256"] == SHA_DOWNSTREAM, "Component SHA drift")
    require(f["full_expert_checkpoint_sha256"] == SHA_EXPERT and
            f["downstream_raw_checkpoint_sha256"] == SHA_DOWNSTREAM, "Final lineage drift")
    require(p["checkpoint_sha256"] == SHA_FINAL and p["expert_sha256"] == SHA_EXPERT and
            p["downstream_sha256"] == SHA_DOWNSTREAM, "RUN12 bridge lineage drift")
    require(p["all_direct_production_parity_pass"] is True and p["neutral_baseline_bitwise_pass"] is True,
            "RUN12 production integrity drift")
    require(p["validation_status"] == "SEALED_NOT_ACCESSED" and p["test_status"] == "SEALED_NOT_ACCESSED",
            "RUN12 embargo drift")
    require(f["official_validation_accessed"] is False and f["official_test_accessed"] is False,
            "RUN12 certification indicates embargo violation")
    require(f["train_videos"] == list(range(1,48)), "RUN12 final Train inventory drift")
    require(f["run12_driver_sha256"] == SHA_R12_SOURCE, "Final source lineage drift")
    require(f["calibration_frames"] == 20075 and f["calibration_uses_oof_experts"] is True, "Calibration basis drift")
    require(f["calibration_uses_heldout_validation"] is False, "Validation used in calibration")
    frozen = {task:{x:float(f["calibration"][task][x]["threshold"]) for x in EXPERTS}
              for task in ("cls","box")}
    for task in frozen:
        for x in EXPERTS:
            require(0 <= frozen[task][x] <= 1, "Calibrated threshold range drift")
    print("[PASS] RUN12 final checkpoint, source, calibration, lineage and embargo hashes verified")
    return frozen


def manifest_preflight(*, read_validation=False):
    """Without --read_validation, require only manifest/root metadata; do NOT open Val labels."""
    require(VAL_JSON.is_file(), f"Missing official Val annotation file: {VAL_JSON}")
    require(VAL_MANIFEST.is_file(),
            "No independent Validation feature manifest. RUN07-12 Train caches are prohibited. "
            f"Expected: {VAL_MANIFEST}")
    m = jload(VAL_MANIFEST)
    require(m.get("schema") == "bfcef_f_run13_val_features_v1", "Validation feature manifest schema drift")
    require(m.get("split") == "validation", "Feature manifest does not explicitly identify Validation")
    require(m.get("r0_checkpoint_sha256") == SHA_R0, "Validation features have wrong R0 checkpoint")
    require(m.get("annotation_sha256") == sha(VAL_JSON), "Validation feature/annotation hash drift")
    require(m.get("image_preprocessing") == "R0_640x640_nonaspect_warp_frozen", "Uncertified R0 preprocessing")
    require(m.get("temporal_recipe") == "RUN07_S0A_K3_from_validated_Kmax7", "Uncertified temporal recipe")
    require(m.get("spatial_recipe") == "RUN07_S0A_scale21_loc24", "Uncertified spatial recipe")
    require(m.get("uses_train_cache_as_validation") is False, "Validation manifest references Train caches")
    require(m.get("test_accessed") is False, "Validation producer reports Test access")
    require(m.get("num_frames") == EXPECTED_FRAMES and m.get("num_gt") == EXPECTED_GT, "Validation feature census drift")
    require(isinstance(m.get("producer"), dict) and m["producer"].get("path") and m["producer"].get("sha256"),
            "Validation feature producer source+SHA required; cannot certify Val creation without code")
    producer = safe_path(EVAL, m["producer"]["path"])
    exact_sha(producer, m["producer"]["sha256"], "Validation producer source")
    shards = m.get("videos")
    require(isinstance(shards,list) and len(shards)==EXPECTED_VIDEOS, "Exactly nine Val video shard triplets required")
    ids=[]; total=0
    for row in shards:
        vid=row.get("video_id")
        require(isinstance(vid,int) and vid not in ids,"Duplicate/bad Validation video_id")
        ids.append(vid)
        n=row.get("frames")
        require(isinstance(n,int) and n>0,"Invalid Validation frame count")
        total+=n
        for section in ("canonical","spatial","temporal"):
            desc=row.get(section)
            require(isinstance(desc,dict) and desc.get("path") and desc.get("sha256"), f"{section} shard requires path/SHA")
            p=safe_path(VAL_ROOT, desc["path"])
            require("train" not in p.name.lower() and "test" not in p.name.lower(),"Forbidden split in shard filename")
            exact_sha(p,desc["sha256"],f"Validation {vid} {section}")
    require(total==EXPECTED_FRAMES,"Validation shards do not cover 4266 images")
    print(f"[PASS] Independent Validation feature manifest and nine triplet shard SHAs verified; producer SHA={m['producer']['sha256'][:16]}")
    if not read_validation: return m,None
    data=jload(VAL_JSON)
    require(len(data["images"])==EXPECTED_FRAMES and len(data["annotations"])==EXPECTED_GT,
            "Official Validation JSON unexpected image/GT census")
    anns_vid={int(v.get("id",v.get("video_id"))):v for v in data["videos"]}
    require(set(anns_vid)==set(ids),"Val JSON and feature manifest video IDs mismatch")
    ann_images={int(im["id"]):im for im in data["images"]}
    require(len(ann_images)==EXPECTED_FRAMES,"Duplicate Validation image IDs")
    require(set(int(im["video_id"]) for im in data["images"]) == set(ids),"Unexpected Validation video IDs")
    cat={int(c["id"]):str(c["name"]).upper().replace(" ","_").replace("-","_") for c in data["categories"]}
    require(tuple(sorted(cat)) == EXPECTED_CLASSES,"Category ID order drift")
    require(tuple(cat[x] for x in EXPECTED_CLASSES)==EXPECTED_CLASS_NAMES,"Category name mapping drift")
    for row in data["annotations"]:
        require(int(row["image_id"]) in ann_images and int(row["category_id"]) in cat,
                "Malformed Validation annotation image/class reference")
    for row in shards:
        vid=row["video_id"]
        require(sum(int(im["video_id"])==vid for im in data["images"])==row["frames"],
                f"Video {vid} frame census drift")
    return m,data


def check_records(vid,row,can,spa,tmp,images,ann_by_image):
    import torch
    n=row["frames"]
    require(len(can)==len(spa)==len(tmp)==n, f"Val video {vid} cache lengths drift")
    seen=[]
    # GT alignment checks use native normalized cxcywh as in RUN09D;
    # reject any silent removal/alteration of official Val annotations.
    for idx,(fr,s,t) in enumerate(zip(can,spa,tmp)):
        iid=int(fr["image_id"])
        require(iid in images and int(images[iid]["video_id"])==vid,"Val canonical image out of split")
        require(iid not in seen,"Repeated frame/image in Val cache")
        seen.append(iid)
        for name,rec in (("spatial",s),("temporal",t)):
            require(int(rec["image_id"])==iid and int(rec["video_id"])==vid and
                    int(rec["seq_idx"])==idx, f"Val {vid} {name} alignment mismatch")
        require(tuple(fr["d_logits"].shape)==(300,7) and tuple(fr["d_boxes"].shape)==(300,4)
                and tuple(fr["q_native"].shape)==(300,256),f"Val {vid} R0 tensor shape drift")
        require(tuple(s["scale_spatial"].shape)==(300,21) and
                tuple(s["loc_spatial"].shape)==(300,24), "Val spatial evidence shape drift")
        require(torch.isfinite(fr["d_logits"]).all() and torch.isfinite(fr["d_boxes"]).all(),"Nonfinite R0 cache")
        gt=fr["gt_boxes_norm"].detach().cpu().float()
        gl=fr["gt_labels"].detach().cpu().long()
        expect=[]
        w,h=float(images[iid]["width"]),float(images[iid]["height"])
        for a in ann_by_image.get(iid,[]):
            x,y,bw,bh=map(float,a["bbox"])
            expect.append((int(a["category_id"])-1, (x+bw/2)/w, (y+bh/2)/h, bw/w, bh/h))
        observed=sorted([(int(c),*map(float,b)) for c,b in zip(gl.tolist(),gt.tolist())])
        expect=sorted(expect)
        require(len(observed)==len(expect),f"Val GT loss in cached frame {iid}")
        for a,b in zip(observed,expect):
            require(a[0]==b[0] and max(abs(x-y) for x,y in zip(a[1:],b[1:]))<2e-3,
                    f"Val GT class/coordinate disagreement image={iid}; do not silently drop boxes")
    require(set(seen)=={int(iid) for iid,im in images.items() if int(im["video_id"])==vid},
            f"Val {vid} image inventory mismatch")
    return seen


def run_evaluation():
    # Fail before any Val annotation access if final artifact hashes are wrong.
    frozen=source_preflight()
    manifest,_=manifest_preflight(read_validation=False)
    require(not OUT.exists(), f"RUN13 output directory already exists: {OUT}; do not overwrite/re-evaluate")
    # Only now open the official Val annotation file.
    manifest,data=manifest_preflight(read_validation=True)
    if str(EVAL) not in sys.path: sys.path.insert(0,str(EVAL))
    import torch
    import train_bfcef_f_run09_joint_uh_gate_router as r9
    import run_bfcef_f_run12_final_train as r12
    import run_bfcef_f_run08_oof_counterfactual_audit as r8
    import audit_bfcef_f_run09d_calibrated_qualification as d
    from pycocotools.coco import COCO
    exact_sha(d.__file__, SHA_R09D_SOURCE, "Frozen RUN09D evaluator")
    require(r8.TOPK==100 and r8.SCORE_THRESHOLD==0.2 and r8.IOU_THRESHOLD==0.3 and
            r8.NUM_QUERIES==300 and r8.NUM_CLASSES==7,"R0/Full evaluator semantics drift")
    require(tuple(d.EXPERTS)==EXPERTS,"Expert roster drift")
    model,_,CO,FX=r9.initialize_runtime()
    ckpt=torch.load(F_CKPT,map_location="cpu",weights_only=False)
    require(ckpt["_meta"]["schema"]=="run12_frozen_deployment_candidate_v1", "Deployment checkpoint schema drift")
    r12.load_experts(model,ckpt)
    r12.load_downstream(model,ckpt)
    model.eval()
    for par in model.parameters():par.requires_grad_(False)
    for task in ("cls","box"):
        for expert in EXPERTS:
            measured=float(getattr(model,"gate_"+task)[expert].mask_thresh.item())
            require(abs(measured-frozen[task][expert])<1e-6,"Loaded gate threshold drift")

    # Use ONLY the official validation ground truth, never RUN08's Train COCO.
    coco=COCO(str(VAL_JSON))
    coco_ca=copy.deepcopy(coco)
    coco_ca.dataset["categories"]=[{"id":1,"name":"object"}]
    for ann in coco_ca.dataset["annotations"]: ann["category_id"]=1
    with contextlib.redirect_stdout(io.StringIO()): coco_ca.createIndex()
    evaluators={v:r8.make_evaluators(coco,coco_ca) for v in ("Base","Full")}
    class_names=list(EXPECTED_CLASS_NAMES)
    cat_ids=torch.tensor(EXPECTED_CLASSES,dtype=torch.long)
    images={int(im["id"]):im for im in data["images"]}
    video_camera={int(v.get("id",v.get("video_id"))):str(v["camera_sensor"]).strip() for v in data["videos"]}
    require(set(video_camera.values()).issubset(SENSORS),"Unknown official Val camera")
    ann_by_image=defaultdict(list)
    for ann in data["annotations"]:ann_by_image[int(ann["image_id"])].append(ann)
    counts={v:Counter() for v in ("Base","Full")}
    by_video={v:{} for v in ("Base","Full")}
    by_camera={v:defaultdict(Counter) for v in ("Base","Full")}
    transitions=d.new_transition()
    diag=d.FullDiagnostics()
    seen=set()
    with torch.no_grad():
        for row in manifest["videos"]:
            vid=int(row["video_id"])
            paths={k:safe_path(VAL_ROOT,row[k]["path"]) for k in ("canonical","spatial","temporal")}
            can=torch.load(paths["canonical"],map_location="cpu",weights_only=False)
            spa=torch.load(paths["spatial"],map_location="cpu",weights_only=False)
            tmp=torch.load(paths["temporal"],map_location="cpu",weights_only=False)
            image_ids=check_records(vid,row,can,spa,tmp,images,ann_by_image)
            require(not (seen & set(image_ids)),"Duplicate images across Val shards")
            seen.update(image_ids)
            preds={v:{} for v in ("Base","Full")}
            preds_ca={v:{} for v in ("Base","Full")}
            local={v:Counter() for v in ("Base","Full")}
            for i in range(len(can)):
                fr=can[i]
                O,X=r9.s0a.make_O_X(fr,spa[i],tmp[i],CO,FX)
                z,b,info=d.production_forward(model,O,X)
                require(torch.isfinite(z).all() and torch.isfinite(b).all(),"Nonfinite final model output")
                outputs={"Base":(O.z,O.b),"Full":(z,b)}
                gt_b=fr["gt_boxes_norm"].float();gt_l=fr["gt_labels"].long()
                iid=int(fr["image_id"])
                matches={};fpreds={}
                for variant,(zv,bv) in outputs.items():
                    pred=r8.extract_top100_predictions(zv.detach().float().cpu(),bv.detach().float().cpu())
                    fpreds[variant]=pred
                    r8.add_coco_prediction(preds[variant],preds_ca[variant],iid,pred,
                                           float(images[iid]["width"]),float(images[iid]["height"]),cat_ids)
                    m=r8.match_operational_detections_to_gt(pred["op_scores"],pred["op_c_idx"],
                                              pred["op_boxes"],pred["op_q_idx"],gt_b,gt_l)
                    matches[variant]=m
                    for field in ("tp","fp","fn"):
                        local[variant][field]+=int(m[field]);counts[variant][field]+=int(m[field]);
                        by_camera[variant][video_camera[vid]][field]+=int(m[field])
                base_m,full_m=matches["Base"],matches["Full"]
                bg,fg=base_m["matched_gt_indices"],full_m["matched_gt_indices"]
                transitions["rescued"]+=len(fg-bg);transitions["lost"]+=len(bg-fg)
                transitions["retained"]+=len(bg&fg)
                bf,ff=set(base_m["fp_dict"]),set(full_m["fp_dict"])
                transitions["fp_added"]+=len(ff-bf);transitions["fp_removed"]+=len(bf-ff)
                diag.update(O,z,b,info,fpreds["Base"],fpreds["Full"])
                del O,X,z,b,info
            for variant in ("Base","Full"):
                ev,evca=evaluators[variant]
                ev.update(preds[variant]);evca.update(preds_ca[variant])
                by_video[variant][str(vid)]=dict(local[variant])
            print(f"    [+] Official Validation video={vid}: {len(can)} frames, paired Base/Full complete",flush=True)
            del can,spa,tmp,preds,preds_ca
            gc.collect()
            if r9.DEVICE.type=="cuda":torch.cuda.empty_cache()

    require(len(seen)==EXPECTED_FRAMES,"Final Validation image census drift")
    require(all(counts[v]["tp"]+counts[v]["fn"]==EXPECTED_GT for v in ("Base","Full")),
            "Operational Val GT census drift")
    require(transitions["retained"]+transitions["lost"]==counts["Base"]["tp"] and
            transitions["retained"]+transitions["rescued"]==counts["Full"]["tp"],"GT transition conservation failed")
    require(transitions["fp_added"]-transitions["fp_removed"]==counts["Full"]["fp"]-counts["Base"]["fp"],
            "FP transition conservation failed")
    results={}
    for v in ("Base","Full"):
        ev,evca=evaluators[v]
        results[v]={**d.metrics_from_counts(**{k:counts[v][k] for k in ("tp","fp","fn")}),
                    **d.finalize_variant_evaluator(ev,evca,class_names),
                    "by_video_operational":by_video[v]}
    # Original RUN08 returns numerical zero for absent classes. Preserve
    # undefined AP as null when official Val contains no GT for that class.
    support=Counter(int(a["category_id"]) for a in data["annotations"])
    for v in ("Base","Full"):
        for cid,name in zip(EXPECTED_CLASSES,EXPECTED_CLASS_NAMES):
            if support[cid]==0:
                results[v]["per_class_ap30"][name]=None
    camera={}
    for sensor in SENSORS:
        if sensor not in set(video_camera.values()):continue
        camera[sensor]={v:metrics(**{k:by_camera[v][sensor][k] for k in ("tp","fp","fn")}) for v in ("Base","Full")}
        camera[sensor]["delta_recall"]=camera[sensor]["Full"]["recall"]-camera[sensor]["Base"]["recall"]
    base,full=results["Base"],results["Full"]
    decision=gate_decision(base,full)
    diagnostics=diag.finalize()
    require(diagnostics["frames"]==EXPECTED_FRAMES,"Routing diagnostics frame drift")
    report={"_meta":{"schema":"bfcef_f_run13_official_validation_v1","read_only":True,
                     "single_candidate":SHA_FINAL,"r0_baseline":SHA_R0,
                     "run12_driver_sha256":SHA_R12_SOURCE,"run13_driver_sha256":sha(Path(__file__)),
                     "val_annotation_sha256":sha(VAL_JSON),"val_feature_manifest_sha256":sha(VAL_MANIFEST),
                     "test_accessed":False,"training_or_calibration_performed":False,
                     "source_label":"official 9-video Validation",
                     "evaluator_contract":{"topk":100,"score_threshold":0.2,"iou_threshold":0.3,"matching":"same-class greedy"}},
            "validation_video_ids":sorted(video_camera),"frame_census":len(seen),"gt_census":EXPECTED_GT,
            "variants":results,"camera_operational":camera,"transition":dict(transitions),
            "full_routing_diagnostics":diagnostics,"prospective_decision":decision,
            "caveats":["One final candidate; no selection/tuning using Validation",
                       "Val is not official Test; paper benchmark claims require untouched Test",
                       "GO is only a review gate for Test preparation, not permission to open Test"]}
    OUT.mkdir(parents=True,exist_ok=False)
    output=OUT/"run13_official_validation_report.json"
    output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    with (OUT/"run13_video_metrics.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f);writer.writerow(["video","camera","base_tp","base_fp","base_fn","full_tp","full_fp","full_fn","delta_f1"])
        for vid in sorted(video_camera):
            b=by_video["Base"][str(vid)];g=by_video["Full"][str(vid)]
            writer.writerow([vid,video_camera[vid],b["tp"],b["fp"],b["fn"],g["tp"],g["fp"],g["fn"],
                             metrics(**g)["f1"]-metrics(**b)["f1"]])
    print("="*95)
    print("[PASS] RUN13 ONE OFFICIAL VALIDATION EVALUATION FINISHED; OFFICIAL TEST SEALED")
    for v in ("Base","Full"):
        m=results[v]
        print(f"{v:5} mAP30={m['map30']:.6f} mAPca={m['mapca30']:.6f} TP={m['tp']} FP={m['fp']} FN={m['fn']} F1={m['f1']:.6f}")
    print("Prospective status:",decision["validation_gate"],decision["reasons"])
    print("Report:",output,"SHA-256:",sha(output))
    print("STOP: REVIEW REPORT. DO NOT OPEN OFFICIAL TEST.")
    print("="*95)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--phase",choices=("selftest","model-preflight","input-preflight","evaluate"),required=True)
    a=p.parse_args()
    selftest()
    if a.phase=="selftest":return
    source_preflight()
    if a.phase=="model-preflight":
        print("[PASS] Frozen model preflight; official Validation/Test not opened")
    elif a.phase=="input-preflight":
        manifest_preflight(read_validation=False)
        print("[PASS] Validation cache/producer provenance checked WITHOUT reading Validation labels")
    else:run_evaluation()

if __name__=="__main__":main()
