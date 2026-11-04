from __future__ import annotations
"""B-FCEF-F RUN_12: frozen, Train-only final fit and Validation preflight.

Stages: inventory -> experts -> downstream -> calibrate -> preflight.
Authoritative objective implementations are imported from unchanged RUN07/RUN09B/RUN09C.
Final downstream supervision uses honest OOF expert outputs/consequences; the
single full-Train expert stack is used at deployment. That distribution bridge
is measured and logged before independent Validation. NO Val/Test access here.
"""
import argparse
import gc
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from collections import Counter
from PIL import Image  # Windows/Pillow import-order compatibility; never used for evaluation

os.environ.setdefault("R0_MSDA_BACKEND", "pytorch")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = Path("D:/MARIVISION")
EVAL = ROOT / "H0_workbench" / "eval_tools"
OUT = EVAL / "bfcef_f_s0" / "run12_final_train_v1"
E_CKPT = OUT / "run12_fulltrain_experts.pth"
E_CERT = OUT / "run12_fulltrain_experts_certificate.json"
D_CKPT = OUT / "run12_fulltrain_downstream_raw.pth"
D_CERT = OUT / "run12_fulltrain_downstream_certificate.json"
F_CKPT = OUT / "run12_frozen_final_candidate.pth"
F_CERT = OUT / "run12_frozen_final_candidate_certificate.json"
P_REPORT = OUT / "run12_validation_preflight.json"
TRAIN = list(range(1, 48))
EXPERTS = ("S", "L", "C", "T")
EXPECTED = {
    "s0a": "a2455d2aa37f91f3a7c01c9fa307bf06026741115d46a0cd9e391642bbed5009",
    "r9": "d51399ed4eaea068dbcfd0a330ad87b01fe277ce907df9e3121af460a514976e",
    "c": "40f7f66dfeaa2e0841da911324bbfdd3de6391293ea5359d9c9d61013bbdccb3",
    "oof": "1c2569f1871b475e4aa71d20e505c8bb738d9b3ed7d4df185787c1c5546c6001",
    "consequences": "f6dcf39a073d7c9fe27201dbe8c38d658cb1b0093f5ba1797a9740314dc70b72",
    "run11": "8c7f9f24a6535610ff65ff1a28e9c9562b46a4ca9293e6ad7bc435682d7e11c1",
}
RUN11 = EVAL / "bfcef_f_s0" / "run11_five_fold_scientific_audit_v1" / "run11_five_fold_audit.json"


def require(ok, reason):
    if not ok:
        raise AssertionError("[FATAL] " + reason)


def sha(path):
    path = Path(path)
    require(path.is_file(), f"File absent: {path}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def jload(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jsave(path, obj):
    require(not Path(path).exists(), f"Refusing to overwrite: {path}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def save_ckpt(torch, path, obj):
    require(not path.exists(), f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)
    return sha(path)


def verify_pair(checkpoint, certificate, kind):
    if not checkpoint.exists() and not certificate.exists():
        return False
    require(checkpoint.is_file() and certificate.is_file(), f"Incomplete {kind} checkpoint/certificate pair")
    info = jload(certificate)
    require(info["checkpoint_sha256"] == sha(checkpoint), f"{kind} binary/certificate SHA mismatch")
    require(info["train_videos"] == TRAIN and info["kind"] == kind, f"{kind} inventory or schema mismatch")
    require(info["source_sha256"] == EXPECTED, f"{kind} upstream root mismatch")
    require(info.get("run12_driver_sha256") == sha(Path(__file__)),
            f"{kind} driver edited after artifact certification")
    return True


def load_libs():
    if str(EVAL) not in sys.path:
        sys.path.insert(0, str(EVAL))
    import torch
    import train_bfcef_f_s0_final_experts as s0a
    import train_bfcef_f_run09_joint_uh_gate_router as r9
    import audit_bfcef_f_run09c_threshold_calibration as c
    for label, mod in (("s0a",s0a),("r9",r9),("c",c)):
        require(sha(mod.__file__) == EXPECTED[label], f"{label} source SHA drift. Audit before altering allowlist")
    require(sha(r9.OOF_MANIFEST) == EXPECTED["oof"], "RUN07 OOF manifest mismatch")
    require(sha(r9.CONSEQUENCE_MANIFEST) == EXPECTED["consequences"], "RUN09 consequence manifest mismatch")
    # RUN11 audited input is metadata only, not opened to choose hyperparameters.
    if RUN11.is_file():
        report = jload(RUN11)
        require(report.get("_meta",{}).get("source_sha256") == EXPECTED["run11"], "RUN11 source audit root drift")
        require(report.get("_meta",{}).get("no_validation_or_test_access") is True, "RUN11 embargo provenance invalid")
    require(s0a.EPOCHS == 2 and s0a.LR == 2e-4 and s0a.SEED == 20261007, "RUN07 training constants drift")
    require(r9.WARMUP_EPOCHS == 1 and r9.JOINT_EPOCHS == 2 and r9.SEED == 20261009, "RUN09 training constants drift")
    require(c.BINS == 10000 and c.EXPERTS == EXPERTS, "RUN09C calibration drift")
    require(tuple(r9.EXPERTS) == EXPERTS, "Expert roster drift")
    folds = r9.parse_folds()
    require(sorted(v for group in folds.values() for v in group) == TRAIN, "47-video fold inventory mismatch")
    return torch, s0a, r9, c, folds


def contract_roots(s0a, r9):
    s0a.verify_source_and_roots()
    require(s0a.verify_match_cache(), "Certified 47-video fixed-Hungarian-match cache invalid or absent; DO NOT rebuild it in RUN12")
    require(sha(s0a.R0_CHECKPOINT) == s0a.FROZEN_R0_SHA, "R0 baseline SHA changed")
    require(s0a.EXPECTED_TOTAL_FRAMES == 20075, "Cached Train frame count drift")
    # OOF manifest is certified independently by immutable root.
    print("[PASS] Certified R0, 47-video Train cache, OOF/consequence roots and fixed assignments")


def counts_sealed(payload):
    return payload.get("train_videos") == TRAIN and payload.get("official_validation_accessed") is False and payload.get("official_test_accessed") is False


def expert_module_state(s0a, model):
    return {x: s0a.cpu_state_dict(getattr(model,x)) for x in ("expert_s","expert_l","expert_c","expert_t","context_bus")}


def downstream_module_state(s0a, model):
    obj = {x:{e:s0a.cpu_state_dict(getattr(model,x)[e]) for e in EXPERTS}
           for x in ("self_assessment","gate_cls","gate_box")}
    obj.update({x:s0a.cpu_state_dict(getattr(model,x)) for x in ("coord_cls","coord_box")})
    return obj


def load_experts(model, state):
    for x in ("expert_s","expert_l","expert_c","expert_t","context_bus"):
        getattr(model,x).load_state_dict(state[x], strict=True)


def load_downstream(model, state):
    for x in ("self_assessment","gate_cls","gate_box"):
        for e in EXPERTS:
            getattr(model,x)[e].load_state_dict(state[x][e],strict=True)
    for x in ("coord_cls","coord_box"):
        getattr(model,x).load_state_dict(state[x],strict=True)


def phase_inventory():
    torch,s0a,r9,c,folds = load_libs()
    contract_roots(s0a,r9)
    for kind,p,q in (("EXPERT",E_CKPT,E_CERT),("DOWNSTREAM",D_CKPT,D_CERT),("FINAL",F_CKPT,F_CERT)):
        print(f"[+] {kind}: {'CERTIFIED' if verify_pair(p,q,kind) else 'MISSING: will be trained/assembled'}")
    print("[PASS] RUN12 inventory; Validation/Test not accessed")


def phase_experts():
    torch,s0a,r9,c,folds=load_libs()
    contract_roots(s0a,r9)
    if verify_pair(E_CKPT,E_CERT,"EXPERT"):
        print("[SKIP] Certified full-Train expert stack already present")
        return
    require(not E_CKPT.exists() and not E_CERT.exists(), "Partial expert artifact")
    (Cfg, CO, BF, FX, argsfn, build, ops) = s0a.prepare_imports()
    config=Cfg.from_json(s0a.SCREENING_CONFIG)
    native_cls,native_box,criterion=s0a.build_native_templates_and_criterion(argsfn,build)
    s0a.set_seed(s0a.SEED)
    model=s0a.construct_model(native_cls,native_box,config,BF)
    weights=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(weights,lr=s0a.LR,weight_decay=s0a.WEIGHT_DECAY)
    frozen_modules=(model.adapter,model.self_assessment,model.gate_cls,model.gate_box,model.coord_cls,model.coord_box)
    require(all(not p.requires_grad for m in frozen_modules for p in m.parameters()), "Nonexpert parameters trainable")
    history=[]
    for epoch in range(s0a.EPOCHS):
        model.train()
        rng=random.Random(s0a.SEED + epoch) # exact RUN07 video/frame shuffling recipe with final-fold offset 0
        videos=TRAIN.copy(); rng.shuffle(videos)
        sums=Counter(); frames=0
        for vid in videos:
            can,spa,tmp=s0a.load_video_triplet(vid)
            matching=torch.load(s0a.match_shard_path(vid),map_location="cpu",weights_only=False)
            require(len(can)==len(spa)==len(tmp)==len(matching),f"Expert fixed-match alignment vid{vid}")
            idxs=list(range(len(can))); rng.shuffle(idxs)
            for i in idxs:
                O,X=s0a.make_O_X(can[i],spa[i],tmp[i],CO,FX)
                optimizer.zero_grad(set_to_none=True)
                dz,db,Li,_=s0a.final_expert_forward(model,O,X)
                total,Lc,Lb,Laux=s0a.task_decoupled_loss(model,criterion,ops,O,can[i],dz,db,Li,matching[i])
                require(bool(torch.isfinite(total)),f"Nonfinite Final Expert loss vid{vid} frame{i}")
                total.backward(); torch.nn.utils.clip_grad_norm_(weights,s0a.GRAD_CLIP)
                optimizer.step()
                for key,v in (("total",total),("cls",Lc),("box",Lb),("liou",Laux)):
                    sums[key]+=float(v.detach())
                frames+=1
                del O,X,dz,db,Li,total,Lc,Lb,Laux
            print(f"    Expert epoch {epoch+1}/2 video {vid:02d} complete",flush=True)
            del can,spa,tmp,matching
            gc.collect()
            if s0a.DEVICE.type=="cuda": torch.cuda.empty_cache()
        require(frames==s0a.EXPECTED_TOTAL_FRAMES, "Full-expert training frame census mismatch")
        row={"epoch":epoch+1,"frames":frames,**{f"mean_{k}":v/frames for k,v in sums.items()}}
        require(all(math.isfinite(v) for k,v in row.items() if isinstance(v,float)),"Nonfinite aggregate expert losses")
        history.append(row)
        print("[+] Full-expert epoch complete",row,flush=True)
    payload={"_meta":{"schema":"run12_experts_full47_v1","train_videos":TRAIN,"epochs":s0a.EPOCHS,
       "seed":s0a.SEED,"objective":s0a.OBJECTIVE_VERSION,"fixed_match_sha256":sha(s0a.match_manifest_path()),
       "r0_sha256":s0a.FROZEN_R0_SHA,"train_history":history,"source_sha256":EXPECTED,
       "source_script_sha256":sha(Path(__file__)),"uses_validation_or_test":False},**expert_module_state(s0a,model)}
    digest=save_ckpt(torch,E_CKPT,payload)
    jsave(E_CERT,{"kind":"EXPERT","train_videos":TRAIN,"source_sha256":EXPECTED,
       "checkpoint_sha256":digest,"run12_driver_sha256":sha(Path(__file__)),
       "fixed_match_sha256":sha(s0a.match_manifest_path()),
       "training_history":history,"official_validation_accessed":False,"official_test_accessed":False})
    print("[PASS] Full47 experts trained and binary certified",digest)


def phase_downstream():
    torch,s0a,r9,c,folds=load_libs()
    contract_roots(s0a,r9)
    require(verify_pair(E_CKPT,E_CERT,"EXPERT"), "Final expert checkpoint not certified")
    if verify_pair(D_CKPT,D_CERT,"DOWNSTREAM"):
        print("[SKIP] Certified downstream final model present"); return
    require(not D_CKPT.exists() and not D_CERT.exists(), "Partial downstream artifacts")
    manifest=r9.load_json(r9.CONSEQUENCE_MANIFEST)
    v2f=r9.video_fold_map(folds)
    # Mechanism initialization matches RUN09B pilot, then each video is trained
    # with its honest RUN07 source-fold expert stack, NOT in-sample experts.
    model,criterion,CO,FX=r9.initialize_runtime()
    r9.set_seed(r9.SEED)
    r9.META_FOLD=0
    r9.exact_mechanism_smoke(model,criterion,CO,FX,folds,v2f,manifest)
    print("[PASS] Original RUN09B mechanism smoke before final downstream fitting")
    del model,criterion
    gc.collect()
    if r9.DEVICE.type=="cuda":torch.cuda.empty_cache()
    model,criterion,CO,FX=r9.initialize_runtime()
    r9.set_seed(r9.SEED)
    weights=r9.aggregate_training_counts(manifest,TRAIN)
    history=[]
    r9.set_training_modes(model,joint=False)
    optimizer=r9.make_optimizer(model,joint=False)
    for epoch in range(r9.WARMUP_EPOCHS):
        h=r9.run_epoch(epoch_name=f"FINAL WARMUP {epoch+1}/1",model=model,criterion=criterion,
            CanonicalObjectState=CO,frame_inputs_from_records=FX,train_videos=TRAIN,
            v2f=v2f,weights=weights,joint=False,seed=r9.SEED+epoch,optimizer=optimizer)
        require(h["frames"]==20075,"Downstream warmup frame census mismatch")
        history.append(h); print("[+]",h,flush=True)
    r9.set_training_modes(model,joint=True)
    optimizer=r9.make_optimizer(model,joint=True)
    for epoch in range(r9.JOINT_EPOCHS):
        h=r9.run_epoch(epoch_name=f"FINAL JOINT {epoch+1}/2",model=model,criterion=criterion,
            CanonicalObjectState=CO,frame_inputs_from_records=FX,train_videos=TRAIN,
            v2f=v2f,weights=weights,joint=True,seed=r9.SEED+100+epoch,optimizer=optimizer)
        require(h["frames"]==20075,"Downstream joint frame census mismatch")
        history.append(h); print("[+]",h,flush=True)
    payload={"_meta":{"schema":"run12_downstream_full47_oof_supervision_v1","train_videos":TRAIN,
       "warmup_epochs":r9.WARMUP_EPOCHS,"joint_epochs":r9.JOINT_EPOCHS,"seed":r9.SEED,
       "expert_training_source":"RUN07 honest OOF experts; downstream local targets from RUN09A",
       "deployment_experts_checkpoint_sha256":sha(E_CKPT),
       "oof_manifest_sha256":EXPECTED["oof"],"consequence_manifest_sha256":EXPECTED["consequences"],
       "training_history":history,"source_sha256":EXPECTED,"no_val_test":True},
       **downstream_module_state(s0a,model)}
    digest=save_ckpt(torch,D_CKPT,payload)
    jsave(D_CERT,{"kind":"DOWNSTREAM","train_videos":TRAIN,"source_sha256":EXPECTED,
       "checkpoint_sha256":digest,"run12_driver_sha256":sha(Path(__file__)),
       "training_history":history,"uses_oof_supervision":True,
       "expert_checkpoint_sha256":sha(E_CKPT),
       "official_validation_accessed":False,"official_test_accessed":False})
    print("[PASS] Full47 downstream trained under OOF supervision and certified",digest)


def phase_calibrate():
    torch,s0a,r9,c,folds=load_libs()
    contract_roots(s0a,r9)
    require(verify_pair(E_CKPT,E_CERT,"EXPERT") and verify_pair(D_CKPT,D_CERT,"DOWNSTREAM"),"Prerequisite checkpoint missing/drift")
    if verify_pair(F_CKPT,F_CERT,"FINAL"):
        print("[SKIP] Frozen deployment candidate already certified"); return
    require(not F_CKPT.exists() and not F_CERT.exists(),"Partial final artifacts")
    e_state=torch.load(E_CKPT,map_location="cpu",weights_only=False)
    d_state=torch.load(D_CKPT,map_location="cpu",weights_only=False)
    model,criterion,CO,FX=r9.initialize_runtime()
    load_downstream(model,d_state)
    model.eval()
    stats={task:{e:c.empty_stat() for e in EXPERTS} for task in ("cls","box")}
    v2f=r9.video_fold_map(folds)
    sample_census=0
    with torch.no_grad():
        for source_fold in sorted(folds):
            r9.load_source_fold(model,source_fold)
            for video in sorted(folds[source_fold]):
                can,spa,tmp,labels=r9.load_video_all(video,source_fold)
                for i in range(len(can)):
                    O,X=r9.s0a.make_O_X(can[i],spa[i],tmp[i],CO,FX)
                    out=r9.mechanism_forward(model,O,X,r9.expert_bundle(model,O,X),hard=False)
                    q=labels[i]["candidate_queries"].long().to(r9.DEVICE)
                    for k,e in enumerate(EXPERTS):
                        ex=labels[i]["experts"][e]
                        for task,add,loss in (("cls","u_cls_target","h_cls_target"),("box","u_box_target","h_box_target")):
                            net=(ex[add].float()-ex[loss].float()).to(r9.DEVICE)
                            c.update_stat(stats[task][e],out["g_"+task][q,k],net)
                    sample_census+=1
                    del O,X,out
                print(f"    Calibration OOF video={video:02d}, source={source_fold}, frames={len(can)}",flush=True)
                del can,spa,tmp,labels
                gc.collect()
                if r9.DEVICE.type=="cuda":torch.cuda.empty_cache()
    require(sample_census==20075,"Calibration full Train frame census mismatch")
    calibrated={"cls":{},"box":{}}
    for task in ("cls","box"):
        for e in EXPERTS:
            cal=c.calibrate_stat(stats[task][e])
            calibrated[task][e]=cal
            getattr(model,"gate_"+task)[e].mask_thresh.data.fill_(float(cal["threshold"]))
            print(f"[+] {task.upper()} {e}: tau={cal['threshold']:.4f} status={cal['status']}",flush=True)
    state=downstream_module_state(s0a,model)
    for part in state:
        if part in ("gate_cls","gate_box"):
            for e in EXPERTS:
                for k,v in state[part][e].items():
                    if k!="mask_thresh":
                        require(torch.equal(v,d_state[part][e][k]),f"Unauthorized calibration parameter motion {part}.{e}.{k}")
        elif part=="self_assessment":
            for e in EXPERTS:
                for k,v in state[part][e].items():
                    require(torch.equal(v,d_state[part][e][k]),f"Unauthorized SA parameter motion {e}.{k}")
        else:
            for k,v in state[part].items():
                require(torch.equal(v,d_state[part][k]),f"Unauthorized coordinator motion {part}.{k}")
    # Combined self-contained downstream/expert state, with R0 native heads and
    # caches referenced immutably. Final experts load for deployment only.
    payload={"_meta":{"schema":"run12_frozen_deployment_candidate_v1","train_videos":TRAIN,
       "full_experts_sha256":sha(E_CKPT),"raw_downstream_sha256":sha(D_CKPT),
       "r0_sha256":s0a.FROZEN_R0_SHA,"training_feature_basis":"RUN07 OOF",
       "deployment_feature_basis":"RUN12 full-Train expert stack",
       "calibration_feature_basis":"RUN07 OOF","calibration_rule":"original RUN09C exact histogram signed U-H (10k bins)",
       "calibration_frames":sample_census,"no_validation_or_test_access":True,"source_sha256":EXPECTED},
       "calibration":calibrated,**state,**{key:e_state[key] for key in ("expert_s","expert_l","expert_c","expert_t","context_bus")}}
    digest=save_ckpt(torch,F_CKPT,payload)
    jsave(F_CERT,{"kind":"FINAL","train_videos":TRAIN,"source_sha256":EXPECTED,
       "checkpoint_sha256":digest,"run12_driver_sha256":sha(Path(__file__)),
       "full_expert_checkpoint_sha256":sha(E_CKPT),
       "downstream_raw_checkpoint_sha256":sha(D_CKPT),
       "calibration":calibrated,"calibration_frames":sample_census,
       "calibration_uses_oof_experts":True,"calibration_uses_heldout_validation":False,
       "official_validation_accessed":False,"official_test_accessed":False})
    print("[PASS] Full47 downstream calibrated; only eight threshold buffers modified; frozen deployment candidate SHA",digest)


def phase_preflight():
    torch,s0a,r9,c,folds=load_libs()
    contract_roots(s0a,r9)
    require(verify_pair(E_CKPT,E_CERT,"EXPERT") and verify_pair(D_CKPT,D_CERT,"DOWNSTREAM") and verify_pair(F_CKPT,F_CERT,"FINAL"),"Missing or altered checkpoint")
    require(not P_REPORT.exists(),"Existing preflight report; refuse overwrite")
    state=torch.load(F_CKPT,map_location="cpu",weights_only=False)
    model,criterion,CO,FX=r9.initialize_runtime()
    load_experts(model,state)
    load_downstream(model,state)
    model.eval()
    v2f=r9.video_fold_map(folds)
    bridge=[]
    def outputs(O,X):
        a=r9.mechanism_forward(model,O,X,r9.expert_bundle(model,O,X),hard=True)
        z,b,info=model(O,X,m_ext=None,m_hat_vis=None,m_avail=None,context_mode="auto")
        require(torch.isfinite(z).all() and torch.isfinite(b).all(),"Nonfinite deployment outputs")
        require((z-a["z"]).abs().max().item()<1e-6 and (b-a["b"]).abs().max().item()<1e-6,"RUN09 direct-production parity failed")
        return z.detach().cpu(),b.detach().cpu()
    with torch.no_grad():
        for source_fold in sorted(folds):
            for video in sorted(folds[source_fold]):
                can,spa,tmp=s0a.load_video_triplet(video)
                i=len(can)//2  # predetermined, one Train frame per video; NOT selected by outcomes
                O,X=r9.s0a.make_O_X(can[i],spa[i],tmp[i],CO,FX)
                r9.load_source_fold(model,source_fold)
                model.eval()
                oz,ob=outputs(O,X)
                load_experts(model,state)
                model.eval()
                fz,fb=outputs(O,X)
                dz=(fz-oz).abs(); db=(fb-ob).abs()
                bridge.append({"video":video,"source_fold":source_fold,"frame_index":i,
                   "max_logit_difference":float(dz.max()),"mean_logit_difference":float(dz.mean()),
                   "max_box_difference":float(db.max()),"mean_box_difference":float(db.mean()),
                   "class_predictions_differ_gt_1e-5":bool((dz>1e-5).any()),
                   "box_predictions_differ_gt_1e-5":bool((db>1e-5).any())})
                print(f"    Final-stack/OOF Train bridge video={video:02d}: max_dz={float(dz.max()):.5f} max_db={float(db.max()):.5f}",flush=True)
                del can,spa,tmp,O,X,oz,ob,fz,fb,dz,db
                gc.collect()
                if r9.DEVICE.type=="cuda":torch.cuda.empty_cache()
        # Genuine zero-influence control: closing all eight masks must restore
        # original R0 outputs bit-for-bit, even under the full expert stack.
        assert len(bridge)==47
        can,spa,tmp=s0a.load_video_triplet(1)
        O,X=r9.s0a.make_O_X(can[len(can)//2],spa[len(can)//2],tmp[len(can)//2],CO,FX)
        load_experts(model,state);model.eval()
        stored={(task,e):getattr(model,"gate_"+task)[e].mask_thresh.detach().clone() for task in ("cls","box") for e in EXPERTS}
        try:
            for task,e in stored: getattr(model,"gate_"+task)[e].mask_thresh.data.fill_(1.1)
            zero=r9.mechanism_forward(model,O,X,r9.expert_bundle(model,O,X),hard=True)
            require(torch.equal(zero["z"],O.z) and torch.equal(zero["b"],O.b),"Final-stack exact neutral fallback not bitwise")
        finally:
            for (task,e),value in stored.items():getattr(model,"gate_"+task)[e].mask_thresh.data.copy_(value)
    report={"schema":"run12_final_val_preflight_v1","status":"REVIEW_REQUIRED_BEFORE_VALIDATION",
       "train_videos":TRAIN,"bridge_sample_rule":"one predetermined median cached Train frame per video",
       "bridge_fulltrain_vs_oof":bridge,
       "mean_abs_logit_shift_mean_over_videos":sum(x["mean_logit_difference"] for x in bridge)/47,
       "mean_abs_box_shift_mean_over_videos":sum(x["mean_box_difference"] for x in bridge)/47,
       "all_direct_production_parity_pass":True,"neutral_baseline_bitwise_pass":True,
       "checkpoint_sha256":sha(F_CKPT),"expert_sha256":sha(E_CKPT),"downstream_sha256":sha(D_CKPT),
       "label_shift_limitation":"U-H and gate calibration derived from honest RUN07 OOF experts; final deployed experts are a new full47 fit. Shift recorded; no Val/Test used.",
       "validation_status":"SEALED_NOT_ACCESSED","test_status":"SEALED_NOT_ACCESSED",
       "predeclared_validation_metrics":["COCO mAP@0.30","mAPca@0.30","AP_S/M/L","per-class AP@0.30","TP/FP/FN","precision/recall/F1","camera recall","R0 vs Full paired"],
       "no_evaluation_or_selection_on_Train_bridge":True,"source_sha256":EXPECTED}
    jsave(P_REPORT,report)
    print("[PASS] RUN12 Train-only final-stack numerical preflight; artifact review required BEFORE independent Validation")
    print("[+] Final candidate", F_CKPT,sha(F_CKPT));print("[+] Preflight",P_REPORT,sha(P_REPORT))


def selftest():
    assert len(TRAIN)==47 and len(set(TRAIN))==47
    assert EXPERTS==("S","L","C","T")
    assert EXPECTED["r9"]!=EXPECTED["oof"]
    assert str(F_CKPT).lower().endswith(".pth")
    assert len({E_CKPT,D_CKPT,F_CKPT,P_REPORT})==4
    print("[PASS] RUN12 static contract tests")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--phase",choices=("selftest","inventory","experts","downstream","calibrate","preflight"),required=True)
    x=p.parse_args()
    selftest()
    if x.phase=="selftest":return
    {"inventory":phase_inventory,"experts":phase_experts,"downstream":phase_downstream,
     "calibrate":phase_calibrate,"preflight":phase_preflight}[x.phase]()


if __name__=="__main__":
    main()
