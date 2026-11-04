from __future__ import annotations
"""RUN15A continuation: NEW internal downstream fit and single-use Train-only qualification.

This does not change original RUN12 files. Fixed full-47 expert weights are used
for FIT/CAL/QUAL so the *new downstream* is representation-consistent. They and
R0 previously saw all Train videos, so this is NOT an independent detector test.
No Val or Test paths are opened. Source/contract failures HOLD, never auto-repair.
"""
import argparse, collections, contextlib, copy, gc, hashlib, io, json, math, os, random, sys
from pathlib import Path
from PIL import Image  # consistent Windows native/CUDA import sequence
for k,v in {'R0_MSDA_BACKEND':'pytorch','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
            'OPENBLAS_NUM_THREADS':'1'}.items():os.environ.setdefault(k,v)
import torch
import torch.nn.functional as F
from run15a_internal_mechanism import mechanism_forward as mechanism_v2, LEGACY_SHA256

EVAL=Path('D:/MARIVISION/H0_workbench/eval_tools')
OUT=EVAL/'bfcef_f_s0'/'run15a_internal_rebuild_v1'
TRAIN_JSON=Path('D:/MARIVISION/seaclips-train.json')
ORIGINAL_R9='d51399ed4eaea068dbcfd0a330ad87b01fe277ce907df9e3121af460a514976e'
ORIGINAL_R12='f4f778f5c9132c80afbd121dda6f6b12fbfb5c6bd87cc2507e484e622a662b8a'
PARENT_SHA='1211292b180eaf00862fc8098ad5cda6a2f64701c0954b5f8f752f090d2ce84c'
FIT=(2,3,4,5,6,7,8,13,14,16,17,18,19,20,24,25,29,31,35,36,37,38,40,41,42,43,44,45,47)
CAL=(11,12,15,21,23,30,33,39,46)
QUAL=(1,9,10,22,26,27,28,32,34)
CLASSES=('ANIMAL','BOAT','MARINE_MARKER','OBJECT','LEISURE_VEHICLE','SAILING_VESSEL','SHIP')
FIXED_GATE_THRESHOLD=0.020  # > sigmoid(-4)=0.017986: neutral initial hard gates CLOSED; fixed before FIT
EPOCHS=2
LR=1.0e-4
LOSS_WEIGHTS={'det':0.25,'tp_class':8.0,'tp_box':3.0,'fp_class':0.20,'regularize':0.005,'joint_marginal_uh':0.10}
SEED=20261010

def need(b,s):
    if not b:raise AssertionError('[RUN15A INTERNAL HOLD] '+str(s))

def sha(p):
    p=Path(p);need(p.is_file(),f'Missing input: {p}')
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(1<<20),b''):h.update(x)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def write_new(p,obj):
    p=Path(p);need(not p.exists(),f'Refuse overwrite: {p}')
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(str(p)+'.tmp');need(not tmp.exists(),f'Unfinished output exists: {tmp}')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
    tmp.rename(p)

def checkpoint_new(p,obj):
    p=Path(p);need(not p.exists(),f'Refuse checkpoint overwrite: {p}')
    p.parent.mkdir(parents=True,exist_ok=True)
    torch.save(obj,p)
    return sha(p)

def metrics(d):
    tp,fp,fn=[int(d.get(k,0)) for k in ('tp','fp','fn')]
    return dict(tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else 0.,
                recall=tp/(tp+fn) if tp+fn else 0.,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)

def validate_splits():
    allv=FIT+CAL+QUAL
    need(len(allv)==47 and set(allv)==set(range(1,48)),'Split missing/duplicated Train videos')
    need(not (set(FIT)&set(CAL) or set(FIT)&set(QUAL) or set(CAL)&set(QUAL)),
         'FIT/CAL/QUAL video leakage')

def imports_and_verify():
    validate_splits()
    need(LEGACY_SHA256==ORIGINAL_R9,'Mechanism clone not derived from certified RUN09B source')
    need(sha(EVAL/'train_bfcef_f_run09_joint_uh_gate_router.py')==ORIGINAL_R9,'RUN09 source drift')
    need(sha(EVAL/'run_bfcef_f_run12_final_train.py')==ORIGINAL_R12,'RUN12 source drift')
    import train_bfcef_f_run09_joint_uh_gate_router as r9
    import run_bfcef_f_run12_final_train as r12
    import audit_bfcef_f_run09d_calibrated_qualification as d
    r8=d.r8
    import train_bfcef_f_s0_final_experts as s0a
    s0a.verify_source_and_roots()
    need(sha(r12.F_CKPT)==PARENT_SHA,'RUN12 frozen parent drift')
    need(r12.verify_pair(r12.F_CKPT,r12.F_CERT,'FINAL'),'RUN12 parent certificate invalid')
    need(sha(TRAIN_JSON)== 'edc32bdbd6b792eb83b6daf11e14708232f1f4a9e06153ebebd5bad1caef9b35',
         'Authoritative Train annotation drift')
    raw=read(TRAIN_JSON)
    need({int(c['id']):c['name'].upper() for c in raw['categories']}==
         {i+1:c for i,c in enumerate(CLASSES)},'Category vs logit index drift')
    print('[PASS] Immutable RUN09/RUN12, Train-only cache/labels, video partitions, seven-class mapping')
    return r9,r12,r8,d,s0a,raw

def make_model(r9,r12):
    model,criterion,CO,FX=r9.initialize_runtime()
    frozen=torch.load(r12.F_CKPT,map_location='cpu',weights_only=False)
    r12.load_experts(model,frozen)
    # Deliberately DO NOT load original downstream, whose calibration saw all Train.
    # Newly constructed SA/gates/coordinators are FIT-only and use exactly these experts.
    for task in ('cls','box'):
        for e in ('S','L','C','T'):
            getattr(model,'gate_'+task)[e].mask_thresh.data.fill_(FIXED_GATE_THRESHOLD)
    for mod in (model.expert_s,model.expert_l,model.expert_c,model.expert_t,
                model.context_bus,model.adapter):
        for p in mod.parameters():p.requires_grad_(False)
    return model,criterion,CO,FX

def state_downstream(model):
    return {'self_assessment':{k:copy.deepcopy(v.state_dict()) for k,v in model.self_assessment.items()},
            'gate_cls':{k:copy.deepcopy(v.state_dict()) for k,v in model.gate_cls.items()},
            'gate_box':{k:copy.deepcopy(v.state_dict()) for k,v in model.gate_box.items()},
            'coord_cls':copy.deepcopy(model.coord_cls.state_dict()),
            'coord_box':copy.deepcopy(model.coord_box.state_dict())}

def restore_downstream(model,state):
    for part in ('self_assessment','gate_cls','gate_box'):
        for e,v in state[part].items():getattr(model,part)[e].load_state_dict(v,strict=True)
    for part in ('coord_cls','coord_box'):getattr(model,part).load_state_dict(state[part],strict=True)
    for part in ('gate_cls','gate_box'):
        for e in ('S','L','C','T'):
            t=float(getattr(model,part)[e].mask_thresh.item())
            need(abs(t-FIXED_GATE_THRESHOLD)<1e-7,'Fixed Train/Deploy hard threshold changed')

def capture_match(z,b,gt_boxes,gt_labels):
    """Exact operational order: global sigmoid Top100, >=0.20, greedy same-class IoU>=.30."""
    with torch.no_grad():
        z=z.detach().float().cpu();b=b.detach().float().cpu()
        gt_boxes=gt_boxes.detach().float().cpu();gt_labels=gt_labels.detach().long().cpu()
        prob=z.sigmoid().flatten();score,ix=prob.topk(min(100,prob.numel()))
        q=(ix//len(CLASSES)).tolist();c=(ix%len(CLASSES)).tolist()
        pairs=[];used=set(); matched=[]; fps=[]
        for rank,(s,qi,ci) in enumerate(zip(score.tolist(),q,c)):
            if s<0.2:continue
            best=None;biou=0.3
            for g in range(len(gt_labels)):
                if g in used or int(gt_labels[g])!=ci:continue
                v=float(single_iou(b[qi],gt_boxes[g]).item())
                if v>=biou and (best is None or v>biou):best=g;biou=v
            if best is None:fps.append((qi,ci))
            else:used.add(best);matched.append((qi,ci,best))
            pairs.append((qi,ci,rank))
        return matched,fps,{'tp':len(matched),'fp':len(fps),'fn':len(gt_labels)-len(matched)}

def single_iou(x,y):
    # normalized cxcywh, differentiable w.r.t. x
    x0=x[:2]-x[2:]/2;x1=x[:2]+x[2:]/2
    y0=y[:2]-y[2:]/2;y1=y[:2]+y[2:]/2
    wh=(torch.minimum(x1,y1)-torch.maximum(x0,y0)).clamp_min(0)
    inter=wh.prod();union=x[2:].clamp_min(0).prod()+y[2:].clamp_min(0).prod()-inter
    return inter/union.clamp_min(1e-9)

def joint_marginal_uh_loss(model,O,out,matched,fps,gt):
    """Actual joint-context routed marginal utility/harm supervision.

    For each expert, measure the native-head effect of subtracting its CURRENT
    routed residual while holding other routes fixed. This is a contextual
    conditional marginal, NOT a proof of causal attribution after re-optimizing
    coordinator/router. All GT is FIT-only and targets are detached.
    """
    pi_c=out['pi_cls'];pi_b=out['pi_box']
    dc=out['delta_cls'];db=out['delta_box']
    hc=O.h+(pi_c[:,1:].unsqueeze(-1)*dc).sum(dim=1)
    hb=O.h+(pi_b[:,1:].unsqueeze(-1)*db).sum(dim=1)
    gate_uh=[]
    with torch.no_grad():
        for k,e in enumerate(('S','L','C','T')):
            z_minus=model.adapter.decode_cls(hc-pi_c[:,k+1:k+2]*dc[:,k])
            b_minus=model.adapter.decode_box(hb-pi_b[:,k+1:k+2]*db[:,k],O.r)
            # Classification: a valid GT's score should rise; an unmatched FP
            # hypothesis should fall. Target magnitudes are clipped to [0,1].
            qk=[];ck=[];direction=[]
            for q,c,g in matched:qk.append(q);ck.append(c);direction.append(1.0)
            for q,c in fps:
                if c==4:continue  # do not teach rare-class suppression from absent GT
                qk.append(q);ck.append(c);direction.append(-1.0)
            if qk:
                qi=torch.tensor(qk,dtype=torch.long,device=O.h.device)
                ci=torch.tensor(ck,dtype=torch.long,device=O.h.device)
                sign=torch.tensor(direction,dtype=O.h.dtype,device=O.h.device)
                pnew=out['z'][qi,ci].sigmoid();pold=z_minus[qi,ci].sigmoid()
                signed=(pnew-pold)*sign
                benefit=(signed/0.2).clamp(0,1)
                harm=(-signed/0.2).clamp(0,1)
            else:qi=ci=benefit=harm=None
            if matched:
                qs=torch.tensor([q for q,c,g in matched],dtype=torch.long,device=O.h.device)
                idx=[g for q,c,g in matched]
                full_iou=torch.stack([single_iou(out['b'][q],gt[g]) for q,c,g in matched])
                minus_iou=torch.stack([single_iou(b_minus[q],gt[g]) for q,c,g in matched])
                box_delta=(full_iou-minus_iou)
                ub=(box_delta/0.2).clamp(0,1);hb_t=(-box_delta/0.2).clamp(0,1)
            else:qs=ub=hb_t=None
            gate_uh.append((e,qi,benefit,harm,qs,ub,hb_t))
    losses=[]
    for e,qi,benefit,harm,qs,ub,hb_t in gate_uh:
        if qi is not None:
            cc=out['cls_channels'][e]
            losses.append(F.mse_loss(cc['U'][qi,0],benefit))
            losses.append(F.mse_loss(cc['H'][qi,0],harm))
        if qs is not None:
            bc=out['box_channels'][e]
            losses.append(F.mse_loss(bc['U'][qs,0],ub))
            losses.append(F.mse_loss(bc['H'][qs,0],hb_t))
    return torch.stack(losses).mean() if losses else out['z'].sum()*0

def preservation_objective(model,O,out,can,criterion,r9):
    matched,fps,bmetrics=capture_match(O.z,O.b,can['gt_boxes_norm'],can['gt_labels'])
    z=out['z'];b=out['b']; device=z.device
    gt=can['gt_boxes_norm'].to(device).float()
    Ldet=r9.detection_loss(criterion,z,b,can)
    # Baseline GT *identities* are protected by q/c anchors, not by TP totals.
    if matched:
        q=torch.tensor([x[0] for x in matched],dtype=torch.long,device=device)
        c=torch.tensor([x[1] for x in matched],dtype=torch.long,device=device)
        desired=O.z[q,c].detach()-0.04
        Lcls=F.relu(desired-z[q,c]).square().mean()
        curr=torch.stack([single_iou(b[qi],gt[gi]) for qi,ci,gi in matched])
        old=torch.stack([single_iou(O.b[qi].detach(),gt[gi]) for qi,ci,gi in matched]).detach()
        Lbox=F.relu(torch.maximum(old-0.015,torch.full_like(old,0.34))-curr).square().mean()
    else:
        Lcls=z.sum()*0;Lbox=b.sum()*0
    # Supervise baseline-unmatched query/class pairs conservatively. Skip LV (rare-class danger).
    negatives=[(q,c) for q,c in fps if c!=4]
    if negatives:
        qs=torch.tensor([v[0] for v in negatives],dtype=torch.long,device=device)
        cs=torch.tensor([v[1] for v in negatives],dtype=torch.long,device=device)
        Lfp=F.softplus(z[qs,cs]).mean()
    else:Lfp=z.sum()*0
    route=(out['pi_cls'][:,1:].square().mean()+out['pi_box'][:,1:].square().mean())
    Lj=joint_marginal_uh_loss(model,O,out,matched,fps,gt)
    total=(LOSS_WEIGHTS['det']*Ldet+LOSS_WEIGHTS['tp_class']*Lcls+
           LOSS_WEIGHTS['tp_box']*Lbox+LOSS_WEIGHTS['fp_class']*Lfp+
           LOSS_WEIGHTS['regularize']*route+LOSS_WEIGHTS['joint_marginal_uh']*Lj)
    need(bool(torch.isfinite(total)),'Nonfinite corrected joint detector loss')
    return total,dict(det=float(Ldet.detach()),tp_class=float(Lcls.detach()),
                      tp_box=float(Lbox.detach()),fp_class=float(Lfp.detach()),
                      n_protected=len(matched),n_baseline_fp=len(fps),joint_marginal_uh=float(Lj.detach())),bmetrics

def static_tests():
    validate_splits()
    a=torch.tensor([.5,.5,.2,.2]);b=torch.tensor([.5,.5,.2,.2])
    need(abs(float(single_iou(a,b))-1.0)<1e-6,'Box identity IoU broken')
    z=torch.full((300,7),-12.);boxes=torch.full((300,4),.1)
    z[3,1]=5.;z[9,4]=4.; boxes[3]=torch.tensor([.5,.5,.2,.2]);boxes[9]=torch.tensor([.2,.2,.1,.1])
    m,fp,counts=capture_match(z,boxes,boxes[3:4],torch.tensor([1]))
    need(m==[(3,1,0)] and counts['tp']==1,'GT query/class matcher broken')
    # deterministic capability mask and STE property tested independently in companion suite
    print('[PASS] Static split, identity and greedy matching fixtures; no dataset opened')

def dynamic_preflight():
    """Actual authoritative Train-frame parity, availability and abstention checks.

    Evaluates only FIT video 2 (deterministic first cached frame). Does not inspect
    calibration or qualification labels, and does not train or write checkpoints.
    """
    r9,r12,r8,d,s0a,raw=imports_and_verify()
    torch.manual_seed(SEED)
    model,criterion,CO,FX=make_model(r9,r12)
    model.eval()
    vid=FIT[0]
    can,spa,tmp=s0a.load_video_triplet(vid)
    need(len(can)>0,'Preflight FIT sample empty')
    O,X=s0a.make_O_X(can[0],spa[0],tmp[0],CO,FX)
    with torch.no_grad():
        bundle=r9.expert_bundle(model,O,X)
        # The new mechanism's HARD evaluation must agree with deployed model
        # while all capabilities are available, without relying on RUN09 train path.
        own=mechanism_v2(model,O,X,bundle,hard=True)
        train_value=mechanism_v2(model,O,X,bundle,hard=False)
        need(float((train_value['z']-own['z']).abs().max())<1e-6 and
             float((train_value['b']-own['b']).abs().max())<1e-6,
             'Hard-forward training and deployment diverge on real FIT frame')
        zp,bp,info=model(O,X,m_ext=None,m_hat_vis=None,m_avail=None,context_mode='auto')
        need(torch.equal(zp,O.z) and torch.equal(bp,O.b),
             'New neutral model does not begin as exact R0 fallback')
        dz=float((own['z']-zp).abs().max());db=float((own['b']-bp).abs().max())
        need(dz<1e-6 and db<1e-6,f'New hard mechanism vs production mismatch: z={dz} b={db}')
        for task in ('cls','box'):
            for e in ('S','L','C','T'):
                getattr(model,'gate_'+task)[e].mask_thresh.data.fill_(1.01)
        zeroz,zerob,info=model(O,X,m_ext=None,m_hat_vis=None,m_avail=None,context_mode='auto')
        need(torch.equal(zeroz,O.z) and torch.equal(zerob,O.b),
             'Real-frame forced-all-closed gates did not return bitwise R0')
    # Differentiability on real weights: initial hard mask closed, STE still propagates.
    for task in ('cls','box'):
        for e in ('S','L','C','T'):
            getattr(model,'gate_'+task)[e].mask_thresh.data.fill_(FIXED_GATE_THRESHOLD)
    model.train()
    for frozen in (model.expert_s,model.expert_l,model.expert_c,model.expert_t,model.context_bus):
        frozen.eval()
    model.zero_grad(set_to_none=True)
    out=mechanism_v2(model,O,X,bundle,hard=False)
    (out['z'].square().mean()+out['b'].square().mean()).backward()
    gate_grad=float(sum(p.grad.abs().sum() for p in model.gate_cls.parameters() if p.grad is not None))
    need(gate_grad>0, 'Real trained TSK gate path is gradient dead at neutral startup')
    # Reinstantiate, so no in-memory threshold contamination flows into FIT.
    print('[PASS] Real FIT-frame hard-production parity, exact no-action R0; '
          f'dz={dz:.3g} db={db:.3g}; original frozen sources untouched',flush=True)
    return True

def fit():
    dynamic_preflight()
    r9,r12,r8,d,s0a,raw=imports_and_verify()
    need(not (OUT/'downstream_internal_fitted.pth').exists(),'Internal FIT already completed')
    torch.manual_seed(SEED);random.seed(SEED)
    model,criterion,CO,FX=make_model(r9,r12)
    r9.set_training_modes(model,joint=True)
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=LR,weight_decay=1e-4)
    required_frames={int(k):None for k in FIT}
    history=[]
    for ep in range(EPOCHS):
        rng=random.Random(SEED+ep);vids=list(FIT);rng.shuffle(vids)
        loss_sum=0.;n=0;ggrad=0.
        for vid in vids:
            can,spa,tmp=s0a.load_video_triplet(vid)
            ids=list(range(len(can)));rng.shuffle(ids)
            for j in ids:
                O,X=s0a.make_O_X(can[j],spa[j],tmp[j],CO,FX)
                bundle=r9.expert_bundle(model,O,X)
                optimizer.zero_grad(set_to_none=True)
                out=mechanism_v2(model,O,X,bundle,hard=False)
                total,detail,base=preservation_objective(model,O,out,can[j],criterion,r9)
                total.backward()
                torch.nn.utils.clip_grad_norm_(params,0.1)
                ggrad+=float(sum(p.grad.abs().sum() for p in model.gate_cls.parameters() if p.grad is not None))
                optimizer.step();loss_sum+=float(total.detach());n+=1
                del O,X,bundle,out,total
            if ep==0:required_frames[vid]=len(can)
            print(f'    [FIT] epoch={ep+1}/{EPOCHS} video={vid:02d} frames={len(can)}',flush=True)
            del can,spa,tmp;gc.collect()
            if r9.DEVICE.type=='cuda':torch.cuda.empty_cache()
        need(n>0 and ggrad>0,'No data or no gradient reached cls gates')
        history.append(dict(epoch=ep+1,frames=n,mean_loss=loss_sum/n,gate_gradient_sum=ggrad))
        print('[FIT] summary',history[-1],flush=True)
    checkpoint={'schema':'run15a_internal_downstream_fitted_v1','parent_sha256':PARENT_SHA,
        'source_sha256':sha(Path(__file__)),'mechanism_sha256':sha(Path(__file__).with_name('run15a_internal_mechanism.py')),
        'fixed_gate_threshold':FIXED_GATE_THRESHOLD,'train_videos':list(FIT),
        'architecture':'same R12 full expert stack, FRESH downstream, hard-forward STE training',
        'downstream_state':state_downstream(model),'history':history}
    digest=checkpoint_new(OUT/'downstream_internal_fitted.pth',checkpoint)
    write_new(OUT/'internal_fit_certificate.json',{'checkpoint_sha256':digest,'parent_sha256':PARENT_SHA,
               'source_sha256':sha(Path(__file__)),'train_videos':list(FIT),'history':history,
               'no_val_test':True,'frames_by_video':required_frames})
    print('[PASS] NEW internal downstream FIT saved:',digest)

def load_fitted(r9,r12):
    cert=read(OUT/'internal_fit_certificate.json')
    need(sha(OUT/'downstream_internal_fitted.pth')==cert['checkpoint_sha256'],'Fitted downstream drift')
    need(cert['source_sha256']==sha(Path(__file__)),'Internal driver modified after FIT')
    payload=torch.load(OUT/'downstream_internal_fitted.pth',map_location='cpu',weights_only=False)
    need(payload['parent_sha256']==PARENT_SHA and payload['train_videos']==list(FIT),'Fitted Train provenance mismatch')
    need(payload['mechanism_sha256']==sha(Path(__file__).with_name('run15a_internal_mechanism.py')),
         'FIT mechanism source modified after training')
    model,criterion,CO,FX=make_model(r9,r12)
    restore_downstream(model,payload['downstream_state'])
    model.eval()
    return model,criterion,CO,FX

def evaluate_group(videos,with_coco=False):
    r9,r12,r8,d,s0a,raw=imports_and_verify()
    model,criterion,CO,FX=load_fitted(r9,r12)
    report={'schema':'run15a_internal_evaluation_v1','videos':list(videos),
            'gate_threshold':FIXED_GATE_THRESHOLD,'strict_train_only':True,
            'upstream_seen_all_47_train':True,'metrics':{},'per_video':{},'per_class_recall':{},
            'frames':0,'gt':0,'baseline_matched_gt_lost':0,'newly_rescued_gt':0,
            'gate_active_queries':{'cls':0,'box':0},
            'per_camera':{},'per_scale_recall':{},
            'size_protocol':'cached640_normalized_area_proxy; not native-image COCO area'}
    total={k:collections.Counter() for k in ('Base','Internal')}
    byclass={k:collections.defaultdict(lambda:collections.Counter()) for k in ('Base','Internal')}
    camera_by_variant={c:{k:collections.Counter() for k in ('Base','Internal')} for c in ('e-CAM','FLIR','Axis')}
    scale_support=collections.Counter();scale_tp={k:collections.Counter() for k in ('Base','Internal')}
    pred_store={k:{} for k in ('Base','Internal')}
    ca_store={k:{} for k in ('Base','Internal')}
    frame_meta={};cached_ids=set();pred_report={}
    gts_by_image=collections.defaultdict(list)
    for a in raw['annotations']:gts_by_image[int(a['image_id'])].append(a)
    official_img={int(im['id']):im for im in raw['images']}
    camera_lookup={int(v['id']):str(v['camera_sensor']) for v in raw['videos']}
    need(set(camera_lookup.values()).issubset({'e-CAM','FLIR','Axis'}),'Unknown camera group')
    with torch.no_grad():
        for vid in sorted(videos):
            can,spa,tmp=s0a.load_video_triplet(vid)
            vid_counts={k:collections.Counter() for k in ('Base','Internal')}
            for i,fr in enumerate(can):
                iid=int(fr['image_id']);need(iid not in cached_ids,'Duplicate Train cached image ID')
                cached_ids.add(iid);meta=official_img[iid]
                need(int(meta['video_id'])==vid,'Train image->video mismatch')
                need(len(gts_by_image[iid])==len(fr['gt_labels']),'Canonical/raw GT mismatch')
                frame_meta[iid]=meta
                O,X=s0a.make_O_X(fr,spa[i],tmp[i],CO,FX)
                z,b,info=model(O,X,m_ext=None,m_hat_vis=None,m_avail=None,context_mode='auto')
                # exact no-gate fallback must be tested even on active frames via forced buffers
                frame_matched={}
                for name,(zz,bb) in {'Base':(O.z,O.b),'Internal':(z,b)}.items():
                    m,fp,d=capture_match(zz,bb,fr['gt_boxes_norm'],fr['gt_labels'])
                    total[name].update(d);vid_counts[name].update(d)
                    camera_by_variant[camera_lookup[vid]][name].update(d)
                    ids={ix for q,c,ix in m};frame_matched[name]=ids
                    gl=fr['gt_labels'].detach().cpu().long().tolist()
                    for ix,ci in enumerate(gl):
                        byclass[name][ci]['support']+=1
                        box_gt=fr['gt_boxes_norm'][ix]
                        area=float(box_gt[2]*box_gt[3])
                        label='small' if area <= 32**2/640**2 else ('medium' if area <= 96**2/640**2 else 'large')
                        if name=='Base':scale_support[label]+=1
                        if ix in ids:
                            byclass[name][ci]['tp']+=1
                            scale_tp[name][label]+=1
                    if with_coco:
                        pr=r8.extract_top100_predictions(zz.detach().float().cpu(),bb.detach().float().cpu())
                        r8.add_coco_prediction(pred_store[name],ca_store[name],iid,pr,
                                               float(meta['width']),float(meta['height']),
                                               torch.arange(1,8,dtype=torch.long))
                report['baseline_matched_gt_lost']+=len(frame_matched['Base']-frame_matched['Internal'])
                report['newly_rescued_gt']+=len(frame_matched['Internal']-frame_matched['Base'])
                report['frames']+=1;report['gt']+=len(fr['gt_labels'])
                # Under original production no bypass info has m_cls/m_box; both cases handled
                report['gate_active_queries']['cls']+=int((info['m_cls'].sum(dim=-1)>0).sum())
                report['gate_active_queries']['box']+=int((info['m_box'].sum(dim=-1)>0).sum())
            report['per_video'][str(vid)]={k:metrics(v) for k,v in vid_counts.items()}
            print(f'    [EVAL] Train video={vid:02d} frames={len(can)}',flush=True)
            del can,spa,tmp;gc.collect()
            if r9.DEVICE.type=='cuda':torch.cuda.empty_cache()
    report['metrics']={k:metrics(total[k]) for k in total}
    raw_images_in_these_videos={int(im['id']) for im in raw['images']
        if int(im['video_id']) in set(videos)}
    report['coco_image_universe']={'raw_images':len(raw_images_in_these_videos),
        'cached_images':len(cached_ids),'omitted_images':len(raw_images_in_these_videos-cached_ids),
        'official_val_test_opened':False}
    report['per_camera']={camera:{k:metrics(value[k]) for k in ('Base','Internal')} for camera,value in camera_by_variant.items()}
    report['per_scale_recall']={name:{'support':int(total),
        'base':scale_tp['Base'][name]/total if total else None,
        'internal':scale_tp['Internal'][name]/total if total else None}
        for name,total in sorted(scale_support.items())}
    report['per_class_recall']={name:{'support':int(byclass['Base'][i]['support']),
        'base':byclass['Base'][i]['tp']/byclass['Base'][i]['support'] if byclass['Base'][i]['support'] else None,
        'internal':byclass['Internal'][i]['tp']/byclass['Internal'][i]['support'] if byclass['Internal'][i]['support'] else None}
        for i,name in enumerate(CLASSES)}
    if with_coco:
        from pycocotools.coco import COCO
        subset=copy.deepcopy(raw)
        subset['images']=[frame_meta[i] for i in sorted(cached_ids)]
        subset['annotations']=[a for a in raw['annotations'] if int(a['image_id']) in cached_ids]
        subset['videos']=[v for v in raw['videos'] if int(v['id']) in set(videos)]
        need(len(subset['annotations'])==report['gt'],'COCO/cache GT census mismatch')
        coco=COCO();coco.dataset=subset
        with contextlib.redirect_stdout(io.StringIO()):coco.createIndex()
        ca=copy.deepcopy(coco);ca.dataset['categories']=[{'id':1,'name':'object'}]
        for ann in ca.dataset['annotations']:ann['category_id']=1
        with contextlib.redirect_stdout(io.StringIO()):ca.createIndex()
        res={}
        for name in ('Base','Internal'):
            ev,ec=r8.make_evaluators(coco,ca)
            ev.update(pred_store[name]);ec.update(ca_store[name])
            res[name]=d.finalize_variant_evaluator(ev,ec,list(CLASSES))
        report['coco']=res
        # Invariant: unsupported class AP must be null; raw evaluator may use 0 for display.
        for name in ('Base','Internal'):
            for cls,record in report['per_class_recall'].items():
                if record['support']==0:
                    res[name]['per_class_ap30'][cls]=None
        b=report['metrics']['Base'];f=report['metrics']['Internal']
        failures=[]
        if report['baseline_matched_gt_lost']>0:
            failures.append('Baseline GT identity loss >0 (strict preservation)')
        if f['f1']<=b['f1']:failures.append('F1 non-improvement')
        if f['fp']>=b['fp']:failures.append('FP non-reduction')
        if f['recall']<b['recall']-0.01:failures.append('Recall drop over 1 pp')
        if res['Internal']['map30']<res['Base']['map30']:failures.append('COCO mAP regression')
        for cls,record in report['per_class_recall'].items():
            if record['support']>=30 and record['internal']<record['base']-.01:
                failures.append(f'Class {cls} recall drop >1pp')
        for sz,record in report['per_scale_recall'].items():
            if record['support']>=30 and record['internal']<record['base']-.01:
                failures.append(f'Scale {sz} recall drop >1pp')
        for cam,record in report['per_camera'].items():
            if record['Base']['tp']+record['Base']['fn']>=30 and \
               record['Internal']['recall']<record['Base']['recall']-.01:
                failures.append(f'Camera {cam} recall drop >1pp')
        report['decision']={'status':'PASS_PROVISIONAL_ONLY' if not failures else 'HOLD',
                            'failures':failures,'no_official_val_test_authorization':True}
    return report

def calibrate():
    need(not (OUT/'internal_calibration.json').exists(),'Calibration already executed')
    report=evaluate_group(CAL,with_coco=False)
    b=report['metrics']['Base'];f=report['metrics']['Internal']
    report['screen']={'status':'PASS' if f['f1']>b['f1'] and f['fp']<b['fp']
                      and report['baseline_matched_gt_lost']==0
                      and f['recall']>=b['recall']-.01 else 'HOLD',
                      'threshold_was_fixed_before_training':True,
                      'threshold_was_not_chosen_from_CAL':True}
    write_new(OUT/'internal_calibration.json',report)
    print('[CAL]',report['screen']['status'],b,f)
    return report['screen']['status']

def qualify():
    cal=read(OUT/'internal_calibration.json')
    need(cal['screen']['status']=='PASS','Calibration was HOLD: qualification embargo')
    need(not (OUT/'internal_qualification.json').exists(),'One-time qualification already executed')
    report=evaluate_group(QUAL,with_coco=True)
    write_new(OUT/'internal_qualification.json',report)
    print('[QUAL]',report['decision']['status'],report['metrics'],
          'mAP', {k:v['map30'] for k,v in report['coco'].items()})
    return report['decision']['status']

def run_one():
    fit()
    if calibrate()!='PASS':return 2
    return 0 if qualify()=='PASS_PROVISIONAL_ONLY' else 2

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',required=True,choices=('selftest','preflight','fit','calibrate','qualify','one-run'))
    p=parser.parse_args()
    static_tests()
    if p.phase=='selftest':return
    if p.phase=='preflight':dynamic_preflight();return
    if p.phase=='fit':fit();return
    if p.phase=='calibrate':need(calibrate()=='PASS','Calibration HOLD');return
    if p.phase=='qualify':need(qualify()=='PASS_PROVISIONAL_ONLY','Qualification HOLD');return
    if p.phase=='one-run':sys.exit(run_one())

if __name__=='__main__':main()
