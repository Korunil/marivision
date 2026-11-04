from __future__ import annotations
"""RUN15A: SINGLE integrated Train-only forensic conformance -> policy fit -> calibration -> qualification.

Changes a REAL deployment policy without altering R0, expert or RUN12 binaries.
The learned action selector chooses among Baseline, ClsOnly, BoxOnly, Full.
This is a NEW, explicitly scoped risk-guard candidate; it is NOT a claim that
all internal B-FCEF-F conceptual gaps have been repaired. Status is fail-closed.
No official Validation/Test file is imported, opened or accessed.

Execution contract:
  --phase selftest  pure numerical counterexamples, no dataset/weights.
  --phase run       complete immutable Train-only pipeline, outputs only in RUN15A dir.

Inputs must exist in original D:/MARIVISION certified RUN07/RUN12 workspace.
Previously frozen RUN12 candidate must hash match.
"""
import argparse
import collections
import copy
import csv
import gc
import hashlib
import io
import json
import math
import os
import random
import sys
from pathlib import Path
from PIL import Image  # Same import order as certified Windows environment.
for k,v in {'R0_MSDA_BACKEND':'pytorch','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}.items():
    os.environ.setdefault(k,v)
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT=Path('D:/MARIVISION')
EVAL=ROOT/'H0_workbench'/'eval_tools'
TRAIN_JSON=ROOT/'seaclips-train.json'
OUT=EVAL/'bfcef_f_s0'/'run15a_integrated_trainonly_v1_patch01'
SHARDS=OUT/'paired_train_frames'
FINAL=EVAL/'bfcef_f_s0'/'run12_final_train_v1'/'run12_frozen_final_candidate.pth'
CERT=EVAL/'bfcef_f_s0'/'run12_final_train_v1'/'run12_frozen_final_candidate_certificate.json'
R12_SHA='1211292b180eaf00862fc8098ad5cda6a2f64701c0954b5f8f752f090d2ce84c'
ACTIONS=('Base','ClsOnly','BoxOnly','Full')
CLASSES=('ANIMAL','BOAT','MARINE_MARKER','OBJECT','LEISURE_VEHICLE','SAILING_VESSEL','SHIP')
FEATURES=('n_base_op','n_full_op','base_op_score_mean','full_op_score_mean',
          'base_top_score_mean','full_top_score_mean','top_score_delta_abs_mean',
          'baseline_pair_down_frac','baseline_pair_down_mean','baseline_pair_op_lost_frac',
          'box_mean_l1','box_max_l1','class_argmax_changed_frac','top100_pair_change_frac',
          'base_lv_top100_frac','full_lv_top100_frac','base_lv_op','full_lv_op',
          'base_score_p90','full_score_p90')
SEED=20261010
FIT_FOLDS=(0,1,2)
CAL_FOLDS=(3,)
QUAL_FOLDS=(4,)
GT_TP_WEIGHT=8.0
HIDDEN=48
EPOCHS=18
LEARNING_RATE=1e-3
CAL_THRESHOLDS=(0.50,0.60,0.70,0.80,0.85,0.90,0.95,0.98,0.995,1.01)


def require(ok,msg):
    if not ok:raise AssertionError('[RUN15A FATAL] '+msg)

def sha(path):
    p=Path(path);require(p.is_file(),f'Missing input: {p}')
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def jsave(path,obj):
    path=Path(path);require(not path.exists(),f'Refuse overwrite: {path}')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')

def jload(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def counts_metric(tp,fp,fn):
    return {'tp':int(tp),'fp':int(fp),'fn':int(fn),
            'precision':tp/(tp+fp) if tp+fp else 0.,
            'recall':tp/(tp+fn) if tp+fn else 0.,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}


def choose_from_probability(prob, threshold):
    """Strong abstention: only a nonbaseline action with sufficient margin is accepted."""
    x=np.asarray(prob,dtype=np.float64)
    require(x.ndim==2 and x.shape[1]==len(ACTIONS),'Predictor action shape')
    require(np.isfinite(x).all(),'Nonfinite predicted probabilities')
    arg=x.argmax(axis=1)
    rank=np.sort(x,axis=1)
    accept=(arg!=0)&(x[np.arange(len(arg)),arg]>=threshold)&(rank[:,-1]-rank[:,-2]>=0.10)
    return np.where(accept,arg,0).astype(np.int64)


def safe_frame_label(per_action, matched_gt_ids):
    """Joint Train-only supervision with *identity-level* baseline TP preservation.

    TP counts alone are insufficient: an action that loses GT #7 and rescues GT #9
    must be treated as unsafe even when the aggregate TP count is unchanged.
    The input GT identities are used ONLY for target construction in Train, never
    for inference features or the deployed model. This is not a guarantee on unseen GT.
    """
    require(set(per_action)==set(ACTIONS) and set(matched_gt_ids)==set(ACTIONS),
            'Incomplete four-action training supervision')
    matches={a:set(matched_gt_ids[a]) for a in ACTIONS}
    for a in ACTIONS:
        require(len(matches[a])==per_action[a]['tp'],
                f'Matched GT identity/count inconsistency for {a}')
        require(all(isinstance(g,(int,np.integer)) and g>=0 for g in matches[a]),
                f'Invalid GT identity for {a}')
    base=per_action['Base'];base_ids=matches['Base']
    valid=[]
    for a in ACTIONS[1:]:
        d_tp=per_action[a]['tp']-base['tp']
        d_fp=per_action[a]['fp']-base['fp']
        preserves_identities=base_ids.issubset(matches[a])
        if preserves_identities and d_tp>=0 and d_fp<=0 and (d_tp>0 or d_fp<0):
            benefit=GT_TP_WEIGHT*d_tp-d_fp
            valid.append((benefit,-ACTIONS.index(a),ACTIONS.index(a)))
    return max(valid)[2] if valid else 0


def paired_features(z0,b0,z1,b1,r8):
    """Input features derive ONLY from model outputs, never GT or Train labels."""
    with torch.no_grad():
        a=r8.extract_top100_predictions(z0.detach().float().cpu(),b0.detach().float().cpu())
        q=r8.extract_top100_predictions(z1.detach().float().cpu(),b1.detach().float().cpu())
        p0=z0.detach().float().cpu().sigmoid();p1=z1.detach().float().cpu().sigmoid()
        s0=a['all_scores'];s1=q['all_scores'];ids0=a['all_q_idx']*7+a['all_c_idx'];ids1=q['all_q_idx']*7+q['all_c_idx']
        base_scores_new=p1.flatten()[ids0]
        top_change=1-len(set(ids0.tolist())&set(ids1.tolist()))/max(1,len(ids0))
        bl=b0.detach().float().cpu();fl=b1.detach().float().cpu(); bd=(bl-fl).abs().mean(dim=-1)
        score_reduction=(s0-base_scores_new).clamp_min(0)
        w=[len(a['op_scores']),len(q['op_scores']),
           float(a['op_scores'].mean()) if len(a['op_scores']) else 0.,
           float(q['op_scores'].mean()) if len(q['op_scores']) else 0.,
           float(s0.mean()),float(s1.mean()),float((s0-s1).abs().mean()),
           float((base_scores_new<s0).float().mean()),float(score_reduction.mean()),
           float(((s0>=.2)&(base_scores_new<.2)).float().mean()),
           float(bd.mean()),float(bd.max()),
           float((p0.argmax(-1)!=p1.argmax(-1)).float().mean()),top_change,
           float((a['all_c_idx']==4).float().mean()),float((q['all_c_idx']==4).float().mean()),
           int((a['op_c_idx']==4).sum()),int((q['op_c_idx']==4).sum()),
           float(torch.quantile(s0,.9)),float(torch.quantile(s1,.9))]
        require(len(w)==len(FEATURES) and all(math.isfinite(v) for v in w),'Unsafe feature vector')
        return w


def variants(O,z,b):
    # Output branches of the SAME frozen network; no architecture or parameter changes.
    return {'Base':(O.z,O.b),'ClsOnly':(z,O.b),'BoxOnly':(O.z,b),'Full':(z,b)}


class FrameActionGuard(nn.Module):
    def __init__(self,dim=len(FEATURES)):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dim,HIDDEN),nn.GELU(),nn.Linear(HIDDEN,HIDDEN),
                               nn.GELU(),nn.Linear(HIDDEN,len(ACTIONS)))
    def forward(self,x):return self.net(x)


def deploy_guard(model,mu,sigma,threshold,O,full_logits,full_boxes,r8):
    """Public inference policy. Never receives ground truth; closed selection is EXACT R0.
    The output is one complete, coherent native R0/ClsOnly/BoxOnly/Full pair.
    """
    x=paired_features(O.z,O.b,full_logits,full_boxes,r8)
    model.eval()
    with torch.no_grad():
        xnorm=(np.asarray(x,dtype=np.float32)-mu)/sigma
        p=model(torch.from_numpy(xnorm[None,:])).softmax(-1).numpy()
    ix=int(choose_from_probability(p,float(threshold))[0])
    zz,bb=variants(O,full_logits,full_boxes)[ACTIONS[ix]]
    if ix==0:require(torch.equal(zz,O.z) and torch.equal(bb,O.b),'Exact fallback violation')
    return zz,bb,{'action':ACTIONS[ix],'posterior':p[0].tolist()}


def selftest():
    fixture={'Base':{'tp':1,'fp':3},'ClsOnly':{'tp':0,'fp':0},
             'BoxOnly':{'tp':1,'fp':2},'Full':{'tp':1,'fp':4}}
    require(safe_frame_label(fixture,{'Base':{0},'ClsOnly':set(),
            'BoxOnly':{0},'Full':{0}})==2,
            'Asymmetric GT-identity-preservation supervision failed')
    require(safe_frame_label({a:{'tp':1,'fp':3} for a in ACTIONS},
            {a:{0} for a in ACTIONS})==0,
            'No-benefit fallback failed')
    require(choose_from_probability([[1,0,0,0],[.1,.7,.1,.1],[.1,.32,.3,.28]],.60).tolist()==[0,1,0],
            'Conservative calibrated abstention failed')
    z=torch.linspace(-4.,4.,2100).reshape(300,7)
    b=torch.rand(300,4)
    q=z.clone();r=b.clone()
    require(torch.equal(z,q) and torch.equal(b,r),'Native identity fixture failed')
    # Example: pruning two FPs while losing one TP MUST NOT be an acceptable target.
    require(safe_frame_label({'Base':{'tp':1,'fp':2},'ClsOnly':{'tp':0,'fp':0},
                             'BoxOnly':{'tp':1,'fp':2},'Full':{'tp':1,'fp':2}},
                             {'Base':{0},'ClsOnly':set(),'BoxOnly':{0},'Full':{0}})==0,
            'Local FP pruning improperly overrode TP preservation')
    # Equal aggregate TP count must NOT conceal loss of a specific baseline match.
    swapped={'Base':{'tp':1,'fp':2},'ClsOnly':{'tp':1,'fp':0},
             'BoxOnly':{'tp':1,'fp':2},'Full':{'tp':1,'fp':2}}
    require(safe_frame_label(swapped,{'Base':{0},'ClsOnly':{1},
                             'BoxOnly':{0},'Full':{0}})==0,
            'Identity-swap counterexample: unsafe correction incorrectly promoted')
    torch.manual_seed(23)
    model=FrameActionGuard();xx=torch.randn(6,len(FEATURES)); y=torch.tensor([0,1,2,3,1,0]);
    loss=F.cross_entropy(model(xx),y);loss.backward()
    require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()),
            'Guard gradient contract failed')
    # No-GT deployment entrypoint, forced no-action on a nontrivial changed proposal.
    from types import SimpleNamespace
    class FakeDetectorRanker:
        @staticmethod
        def extract_top100_predictions(zz,bb):
            scores,indices=zz.sigmoid().flatten().topk(100)
            qs=indices//7;cs=indices%7;active=scores>=.2
            return {'all_scores':scores,'all_q_idx':qs,'all_c_idx':cs,'all_boxes':bb[qs],
                    'op_scores':scores[active],'op_q_idx':qs[active],'op_c_idx':cs[active],
                    'op_boxes':bb[qs][active]}
    mock=FakeDetectorRanker()
    original=torch.randn(300,7); bb=torch.rand(300,4)
    obj=SimpleNamespace(z=original,b=bb)
    revised=original+0.5; newboxes=bb+0.02
    features=paired_features(original,bb,revised,newboxes,mock)
    require(len(features)==len(FEATURES),'Feature vector differs from declared schema')
    abstainer=FrameActionGuard()
    with torch.no_grad():
        for par in abstainer.parameters():par.zero_()
        abstainer.net[-1].bias[0]=20
    outz,outb,meta=deploy_guard(abstainer,np.zeros(len(FEATURES),dtype=np.float32),
                                np.ones(len(FEATURES),dtype=np.float32),.5,
                                obj,revised,newboxes,mock)
    require(meta['action']=='Base' and torch.equal(outz,original) and torch.equal(outb,bb),
            'RUN15A deployment abstention does not preserve exact original tensors')
    print('[PASS] RUN15A numerical fixtures, GT-protection target, conservative fallback, gradients; no datasets opened')


def phase_preflight():
    require(sha(FINAL)==R12_SHA,'RUN12 candidate checksum changed')
    cert=jload(CERT)
    require(cert['checkpoint_sha256']==R12_SHA and cert['official_validation_accessed'] is False
            and cert['official_test_accessed'] is False,'RUN12 provenance inconsistency')
    require(TRAIN_JSON.exists(),'Official Train annotation missing')
    if str(EVAL) not in sys.path:sys.path.insert(0,str(EVAL))
    import train_bfcef_f_run09_joint_uh_gate_router as r9
    import run_bfcef_f_run12_final_train as r12
    import run_bfcef_f_run08_oof_counterfactual_audit as r8
    import audit_bfcef_f_run09d_calibrated_qualification as d
    require((r8.SCORE_THRESHOLD,r8.IOU_THRESHOLD,r8.TOPK,r8.NUM_CLASSES)==(.2,.3,100,7),
            'Evaluation policy differs from RUN13')
    _,s0a,rr9,c,folds=r12.load_libs()
    require(rr9 is r9,'Imported downstream module identity drift')
    r12.contract_roots(s0a,r9)
    folds={int(k):[int(x) for x in v] for k,v in folds.items()}
    require(set(folds)==set(range(5)),'Expected 5 source meta-folds')
    require(sorted(v for vv in folds.values() for v in vv)==list(range(1,48)),
            '47 video source-fold partition invalid')
    train={name:sorted(v for f in ids for v in folds[f]) for name,ids in
           [('fit',FIT_FOLDS),('calibration',CAL_FOLDS),('qualification',QUAL_FOLDS)]}
    require(all(set(train[k]).isdisjoint(train[j]) for k in train for j in train if k!=j),
            'Guard split overlap')
    data=jload(TRAIN_JSON)
    require('videos' in data and 'categories' in data and 'annotations' in data,'Train COCO malformed')
    actual={int(c['id']):c['name'].upper() for c in data['categories']}
    require(actual=={i+1:name for i,name in enumerate(CLASSES)},
            f'Train category/logit schema mismatch: {actual}')
    # Warn explicitly: all 47 videos already fitted R0 and RUN12 model.
    print('[PASS] R0/final source roots, Train caches, 5 source-fold inventory and category/column contract')
    return r9,r12,r8,d,s0a,folds,train,data


def audit_legacy_conformance(r9,r12,data):
    """Adversarially register legacy defects, not a false PASS list.
    This does not claim every problem is remediated by the external safety guard.
    """
    import inspect
    r9_src=inspect.getsource(r9.mechanism_forward)
    cal_src=inspect.getsource(r12.phase_calibrate)
    experts_src=inspect.getsource(r12.phase_experts)
    downstream_src=inspect.getsource(r12.phase_downstream)
    require(r12.verify_pair(r12.E_CKPT,r12.E_CERT,'EXPERT') and
            r12.verify_pair(r12.D_CKPT,r12.D_CERT,'DOWNSTREAM') and
            r12.verify_pair(r12.F_CKPT,r12.F_CERT,'FINAL'),
            'RUN12 source/weight/certificate verification failed')
    # The metadata was serialized BEFORE RUN15A and independently pinned to R12_SHA.
    final_state=torch.load(FINAL,map_location='cpu',weights_only=False)
    frozen_meta=final_state.get('_meta',{})
    g06_certified=(
        frozen_meta.get('training_feature_basis')=='RUN07 OOF' and
        frozen_meta.get('calibration_feature_basis')=='RUN07 OOF' and
        frozen_meta.get('deployment_feature_basis')=='RUN12 full-Train expert stack' and
        frozen_meta.get('full_experts_sha256')==sha(r12.E_CKPT) and
        'r9.load_source_fold' in cal_src and
        'r9.run_epoch' in downstream_src and
        'train_videos=TRAIN' in downstream_src and
        'videos=TRAIN.copy()' in experts_src and
        'load_experts(model, state)' in inspect.getsource(r12.load_experts)
    )
    require(g06_certified,'G06 Train OOF vs final deployed full-Train expert provenance not verified')
    del final_state
    legacy={
        'G02_soft_vs_hard':{'legacy_status':'DEVIATION',
            'evidence':'RUN09 uses g_soft when hard=False and g_hard when hard=True',
            'detected':('g_soft' in r9_src and 'g_hard' in r9_src)},
        'G03_isolated_vs_joint_utility':{'legacy_status':'DEVIATION',
            'evidence':'RUN09 uses isolated expert U/H; RUN15A guard targets whole integrated output choices',
            'detected':('pi_cls' in r9_src and 'cls_channels' in r9_src)},
        'G05_in_sample_calibration':{'legacy_status':'DEVIATION',
            'evidence':'RUN12 calibrates its fitted downstream using all 47 OOF expert records',
            'detected':('for video in sorted(folds[source_fold])' in cal_src)},
        'G06_expert_representation':{'legacy_status':'DEVIATION',
            'evidence':'RUN12 final uses newly full47-trained experts, not OOF ones',
            'detected':g06_certified},
        'G07_physical_context':{'legacy_status':'UNVERIFIED_PHYSICAL_SEMANTICS',
            'evidence':'No sensor-ground-truth validation of visually inferred environment',
            'detected':True},
        'G08_temporal_motion':{'legacy_status':'UNVERIFIED_FULL_MOTION_COMPENSATION',
            'evidence':'Temporal median displacement is a proxy, not certified camera 6DOF',
            'detected':True},
        'G09_rare_scale':{'legacy_status':'NOT_QUALIFIED',
            'evidence':'Small LEISURE_VEHICLE generalization not directly tested',
            'detected':True},
    }
    # Fail loudly if source changed so significantly that existing gap statements need re-auditing.
    for name,item in legacy.items():require(item['detected'],f'Legacy gap audit source guard failed: {name}')
    output={'schema':'run15a_adversarial_legacy_gap_register_v1',
            'legacy_gaps':legacy,'guard_remediation_scope':[
                'Joint instead of isolated per-frame output consequence supervision',
                'Exact baseline-or-four-action selection',
                'Video-disjoint fitting / calibration / qualification for the new guard',
                'No alteration to original R0/Full core checkpoint or weights'],
            'not_remediated_by_guard':[
                'Internal RUN09 soft training vs hard deployment semantics',
                'Internal U/H isolated utility targets',
                'Original RUN12 in-sample downstream threshold calibration',
                'Original OOF-to-Full expert transfer shift',
                'Certified external physical context and 6DOF stabilization',
                'Generalization to small LEISURE_VEHICLE under unseen Test'],
            'overall_architecture_conformance':'NOT_CERTIFIED'}
    p=OUT/'run15a_concept_gap_register.json'
    if p.exists():require(jload(p)==output,'Concept gap register drift')
    else:jsave(p,output)
    print('[AUDIT] Legacy source-to-concept gaps explicitly registered; not misreported as fixed')
    return output


def shard_paths(vid):
    return SHARDS/f'run15a_train_vid{vid}.pth',SHARDS/f'run15a_train_vid{vid}.json'


def build_train_shards(r9,r12,r8,d,s0a,folds,train,data):
    model,_,CO,FX=r9.initialize_runtime()
    ckpt=torch.load(FINAL,map_location='cpu',weights_only=False)
    r12.load_experts(model,ckpt);r12.load_downstream(model,ckpt)
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    sensor={int(v['id']):str(v['camera_sensor']) for v in data['videos']}
    image_dims={int(im['id']):(float(im['width']),float(im['height'])) for im in data['images']}
    have=set()
    for vid in range(1,48):
        p,c=shard_paths(vid)
        if p.exists() or c.exists():
            require(p.exists() and c.exists(),f'Partial video shard {vid}')
            info=jload(c)
            require(info['sha256']==sha(p) and info['frozen_sha256']==R12_SHA
                    and info['video_id']==vid and info['schema']=='run15a_train_shard_v1',
                    f'Stale or altered Train output vid{vid}')
            print(f'    [PASS/SKIP] Certified Train video {vid:02d}: {info["frames"]} frames',flush=True)
            have.add(vid);continue
        can,spa,tmp=s0a.load_video_triplet(vid)
        require(len(can)==len(spa)==len(tmp),'Train triplet alignment missing')
        rows=[]
        with torch.no_grad():
            for i in range(len(can)):
                O,X=s0a.make_O_X(can[i],spa[i],tmp[i],CO,FX)
                z,b,info=d.production_forward(model,O,X)
                require(torch.isfinite(z).all() and torch.isfinite(b).all(),'Nonfinite Full output')
                if i==0:
                    # Real-source native head parity: crucial when active corrections are decoded.
                    native_z=model.adapter.decode_cls(O.h)
                    native_b=model.adapter.decode_box(O.h,O.r)
                    require((native_z-O.z).abs().max().item()<1e-4 and
                            (native_b-O.b).abs().max().item()<1e-4,
                            f'Native R0 decoder reconstruction drift Train video={vid}')
                    stored={(task,e):getattr(model,'gate_'+task)[e].mask_thresh.detach().clone()
                            for task in ('cls','box') for e in ('S','L','C','T')}
                    try:
                        for (task,e) in stored:
                            getattr(model,'gate_'+task)[e].mask_thresh.data.fill_(1.1)
                        neutral_z,neutral_b,_=d.production_forward(model,O,X)
                        require(torch.equal(neutral_z,O.z) and torch.equal(neutral_b,O.b),
                                f'Exact production closed-gate R0 identity failed Train video={vid}')
                    finally:
                        for (task,e),value in stored.items():
                            getattr(model,'gate_'+task)[e].mask_thresh.data.copy_(value)
                if i==0:
                    from dataclasses import replace
                    # Explicit missing-expert capability: production MUST close T task channels.
                    no_temporal=replace(O,capabilities={**O.capabilities,'temporal':False})
                    ones=torch.ones(O.h.shape[0],1,device=O.h.device,dtype=O.h.dtype)
                    for task in ('cls','box'):
                        avail=model._availability(no_temporal,ones,task)
                        require(torch.equal(avail[:,3],torch.zeros_like(avail[:,3])),
                                f'Production temporal capability ignored: video {vid}/{task}')
                        require(avail.shape==(O.h.shape[0],4),
                                f'Production availability dimensions shifted: video {vid}/{task}')
                if i==0:
                    # Use frozen production/RUN09 helper parity on deterministic first frame.
                    from_r9=r9.mechanism_forward(model,O,X,r9.expert_bundle(model,O,X),hard=True)
                    require(torch.allclose(z,from_r9['z'],atol=1e-6,rtol=0) and
                            torch.allclose(b,from_r9['b'],atol=1e-6,rtol=0),
                            f'RUN15A production/helper parity vid{vid}')
                conf=variants(O,z,b)
                labels=can[i]['gt_labels'].detach().float().cpu().long()
                gtboxes=can[i]['gt_boxes_norm'].detach().float().cpu()
                counts={}
                matched_class={};matched_scale={};matched_gt_ids={}
                w,h=image_dims[int(can[i]['image_id'])]
                sizes=['small' if area*w*h<32**2 else ('medium' if area*w*h<96**2 else 'large')
                       for area in (gtboxes[:,2]*gtboxes[:,3]).tolist()]
                for name,(zz,bb) in conf.items():
                    pred=r8.extract_top100_predictions(zz.detach().float().cpu(),bb.detach().float().cpu())
                    m=r8.match_operational_detections_to_gt(pred['op_scores'],pred['op_c_idx'],pred['op_boxes'],
                                                             pred['op_q_idx'],gtboxes,labels)
                    counts[name]={k:int(m[k]) for k in ('tp','fp','fn')}
                    matched_gt_ids[name]=sorted(int(j) for j in m['matched_gt_indices'])
                    matched_class[name]=dict(collections.Counter(int(labels[j]) for j in m['matched_gt_indices']))
                    matched_scale[name]=dict(collections.Counter(sizes[j] for j in m['matched_gt_indices']))
                    require(counts[name]['tp']+counts[name]['fn']==len(labels),'GT conservation failed')
                row={'video':vid,'frame':i,'image_id':int(can[i]['image_id']),
                     'camera':sensor[vid],'features':paired_features(O.z,O.b,z,b,r8),
                     'target':safe_frame_label(counts,matched_gt_ids),'counts':counts,
                     'matched_gt_ids':matched_gt_ids,
                     'matched_class':matched_class,'matched_scale':matched_scale,'gt_scale':sizes,
                     'class_gt':[int(v) for v in labels.tolist()],
                     'class_gt_area':[float(v) for v in (gtboxes[:,2]*gtboxes[:,3]).tolist()]}
                rows.append(row)
                if (i+1)%500==0:print(f'        Train {vid:02d} frames={i+1}/{len(can)}',flush=True)
                del O,X,z,b,info
        require(len(rows)==len(can) and len({r['image_id'] for r in rows})==len(rows),
                f'Train video {vid} frame counts or IDs invalid')
        p.parent.mkdir(parents=True,exist_ok=True)
        torch.save(rows,p)
        jsave(c,{'schema':'run15a_train_shard_v1','video_id':vid,'frames':len(rows),
                 'sha256':sha(p),'frozen_sha256':R12_SHA,'feature_schema':list(FEATURES),
                 'train_only':True,'annotations_sha256':sha(TRAIN_JSON)})
        have.add(vid)
        print(f'    [PASS] Train video {vid:02d}: {len(rows)} paired frames and counterfactual targets',flush=True)
        del can,spa,tmp,rows
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    require(have==set(range(1,48)),'Incomplete Train shard census')


def read_guard_data():
    allrows=[];ids=set()
    for vid in range(1,48):
        p,c=shard_paths(vid);require(p.is_file() and c.is_file(),'Train shard missing')
        info=jload(c)
        require(sha(p)==info['sha256'] and info['annotations_sha256']==sha(TRAIN_JSON)
                and info['feature_schema']==list(FEATURES),'Certified shard changed')
        rs=torch.load(p,map_location='cpu',weights_only=False)
        require(len(rs)==info['frames'] and all(int(r['video'])==vid for r in rs),
                'Shard row/video schema mismatch')
        for r in rs:
            require(r['image_id'] not in ids,'Duplicate Train image ID');ids.add(r['image_id'])
            require(len(r['features'])==len(FEATURES),'Feature dimension shift')
            require(0<=r['target']<4,'Target malformed')
            require('matched_gt_ids' in r,'Missing per-action GT identity contract')
            require(all(len(r['matched_gt_ids'][a])==r['counts'][a]['tp'] and
                        len(set(r['matched_gt_ids'][a]))==len(r['matched_gt_ids'][a]) and
                        all(0<=j<len(r['class_gt']) for j in r['matched_gt_ids'][a])
                        for a in ACTIONS),'GT match identity/count or range mismatch')
            require(r['target']==safe_frame_label(r['counts'],r['matched_gt_ids']),
                    'Training label not reproducible from certified GT identity ledger')
            require(sum(r['counts'][a]['tp']+r['counts'][a]['fn'] for a in ACTIONS)==
                    4*len(r['class_gt']),'GT count not conserved')
        allrows.extend(rs)
    require(len(allrows)==20075,f'Train frame census {len(allrows)} != 20075')
    print(f'[PASS] 47 certified paired Train video shards: {len(allrows)} frames')
    return allrows


def x_normalizer(rows):
    x=np.asarray([r['features'] for r in rows],dtype=np.float32)
    require(x.ndim==2 and np.isfinite(x).all(),'Fit feature matrix malformed')
    mu=x.mean(0);sigma=x.std(0);sigma=np.maximum(sigma,1e-5)
    return mu,sigma


def predict_proba(model,rows,mu,sigma):
    model.eval()
    x=np.asarray([r['features'] for r in rows],dtype=np.float32)
    with torch.no_grad():
        z=(x-mu)/sigma
        t=torch.from_numpy(z.astype(np.float32))
        out=model(t).softmax(-1).numpy()
    require(np.isfinite(out).all(),'Nonfinite probabilities')
    return out


def train_guard(fit,cal,qual):
    ck=OUT/'run15a_preservation_guard.pth';cc=OUT/'run15a_guard_training_certificate.json'
    if ck.exists() or cc.exists():
        require(ck.exists() and cc.exists(),'Partial guard checkpoint pair')
        meta=jload(cc)
        require(meta['sha256']==sha(ck) and meta['schema']=='run15a_guard_train_v1'
                and meta['fit_videos']==sorted({r['video'] for r in fit}),
                'Guard checkpoint edited or split changed')
        z=torch.load(ck,map_location='cpu',weights_only=False)
        model=FrameActionGuard();model.load_state_dict(z['state'],strict=True)
        return model,np.asarray(z['mu'],dtype=np.float32),np.asarray(z['sigma'],dtype=np.float32)
    torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
    mu,sigma=x_normalizer(fit)
    tx=torch.tensor((np.asarray([r['features'] for r in fit],dtype=np.float32)-mu)/sigma)
    ty=torch.tensor([r['target'] for r in fit],dtype=torch.long)
    require(len(set(ty.tolist()))>=2,'Train guard has no actionable correction target diversity')
    model=FrameActionGuard()
    optimizer=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=1e-4)
    gen=torch.Generator().manual_seed(SEED)
    history=[]
    for epoch in range(EPOCHS):
        model.train();order=torch.randperm(len(tx),generator=gen)
        losses=[]
        for ids in order.split(256):
            optimizer.zero_grad(set_to_none=True)
            logits=model(tx[ids]);loss=F.cross_entropy(logits,ty[ids])
            require(bool(torch.isfinite(loss)),'Guard loss nonfinite')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step();losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
        print(f'    [GUARD] Train epoch {epoch+1:02d}/{EPOCHS} loss={history[-1]:.5f}',flush=True)
    # Freeze after fitting. Holdout calibration/qualification GT NEVER used in gradients.
    for p in model.parameters():p.requires_grad_(False)
    torch.save({'schema':'run15a_guard_v1','state':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                'mu':mu,'sigma':sigma,'features':FEATURES,'actions':ACTIONS,
                'fit_videos':sorted({r['video'] for r in fit}),
                'frozen_parent_sha256':R12_SHA},ck)
    jsave(cc,{'schema':'run15a_guard_train_v1','sha256':sha(ck),'train_only':True,
              'fit_videos':sorted({r['video'] for r in fit}),
              'calibration_videos':sorted({r['video'] for r in cal}),
              'qualification_videos':sorted({r['video'] for r in qual}),
              'epochs':EPOCHS,'losses':history,'source_sha256':sha(Path(__file__))})
    return model,mu,sigma


def evaluate_choices(rows,choice):
    require(len(rows)==len(choice),'Bad evaluation choice length')
    totals={name:collections.Counter() for name in ('Base','Selected')}
    byvideo=collections.defaultdict(lambda:{'Base':collections.Counter(),'Selected':collections.Counter()})
    camera=collections.defaultdict(lambda:{'Base':collections.Counter(),'Selected':collections.Counter()})
    action_count=collections.Counter()
    byclass=collections.Counter(); byscale=collections.Counter()
    class_tp={name:collections.Counter() for name in ('Base','Selected')}
    scale_tp={name:collections.Counter() for name in ('Base','Selected')}
    for r,chosen in zip(rows,choice):
        a=ACTIONS[int(chosen)];action_count[a]+=1
        for variant,key in (('Base','Base'),('Selected',a)):
            totals[variant].update(r['counts'][key]);byvideo[int(r['video'])][variant].update(r['counts'][key]);
            camera[r['camera']][variant].update(r['counts'][key])
        byclass.update(r['class_gt']);byscale.update(r['gt_scale'])
        for scope,a2 in (('Base','Base'),('Selected',a)):
            class_tp[scope].update({int(k):int(v) for k,v in r['matched_class'][a2].items()})
            scale_tp[scope].update(r['matched_scale'][a2])
    stats={k:counts_metric(**v) for k,v in totals.items()}
    per_class={CLASSES[c]:{'support':int(n),'base_recall':class_tp['Base'][c]/n,
                           'selected_recall':class_tp['Selected'][c]/n} for c,n in sorted(byclass.items())}
    per_scale={key:{'support':int(n),'base_recall':scale_tp['Base'][key]/n,
                     'selected_recall':scale_tp['Selected'][key]/n} for key,n in sorted(byscale.items())}
    return {'metrics':stats,'per_class_recall':per_class,'per_scale_recall':per_scale,
            'selected_action_counts':dict(action_count),
            'video_operational':{str(v):{k:counts_metric(**d[k]) for k in d} for v,d in byvideo.items()},
            'camera_operational':{v:{k:counts_metric(**d[k]) for k in d} for v,d in camera.items()},
            'gt_class_support':{CLASSES[c]:int(n) for c,n in sorted(byclass.items())}}


def qualification_coco_ap(rows,choice,r9,r12,r8,d,s0a,source_data):
    """Exact COCO mAP@.30 of the locked Train GUARD-heldout video set.

    Never executed during calibration; never uses official Val/Test.
    Batches predictions by video to bound evaluator peak memory.
    """
    from pycocotools.coco import COCO
    videos=set(r['video'] for r in rows)
    image_data={int(i['id']):i for i in source_data['images']
                if int(i['video_id']) in videos}
    image_ids=set(image_data)
    gt=[a for a in source_data['annotations'] if int(a['image_id']) in image_ids]
    require(len(gt)>0 and image_ids,'Qualification COCO Train GT/images empty')
    sub={k:copy.deepcopy(source_data[k]) for k in source_data if k not in ('images','annotations','videos')}
    sub['images']=list(image_data.values());sub['annotations']=gt
    sub['videos']=[v for v in source_data['videos'] if int(v['id']) in videos]
    coco=COCO();coco.dataset=sub
    with io.StringIO() as capture:
        # COCO prints, but output is kept out of the Windows console.
        import contextlib
        with contextlib.redirect_stdout(capture):coco.createIndex()
    coco_ca=copy.deepcopy(coco)
    coco_ca.dataset['categories']=[{'id':1,'name':'object'}]
    for a in coco_ca.dataset['annotations']:a['category_id']=1
    with io.StringIO() as capture:
        import contextlib
        with contextlib.redirect_stdout(capture):coco_ca.createIndex()
    evals={v:r8.make_evaluators(coco,coco_ca) for v in ('Base','Selected')}
    row_lookup={int(r['image_id']):(r,int(a)) for r,a in zip(rows,choice)}
    require(set(row_lookup)==image_ids,'Qualification COCO image identity coverage mismatch')
    model,_,CO,FX=r9.initialize_runtime()
    ckpt=torch.load(FINAL,map_location='cpu',weights_only=False)
    r12.load_experts(model,ckpt);r12.load_downstream(model,ckpt);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    cats=torch.arange(1,8,dtype=torch.long)
    with torch.no_grad():
        for vid in sorted(videos):
            can,spa,tmp=s0a.load_video_triplet(vid)
            pred={v:{} for v in evals};pred_ca={v:{} for v in evals}
            for i,fr in enumerate(can):
                iid=int(fr['image_id'])
                require(iid in row_lookup and row_lookup[iid][0]['video']==vid,
                        'Qualification image ID mismatch')
                O,X=s0a.make_O_X(fr,spa[i],tmp[i],CO,FX)
                z,b,_=d.production_forward(model,O,X)
                outs=variants(O,z,b)
                name=ACTIONS[row_lookup[iid][1]]
                for variant,action in (('Base','Base'),('Selected',name)):
                    zz,bb=outs[action]
                    pp=r8.extract_top100_predictions(zz.detach().float().cpu(),bb.detach().float().cpu())
                    r8.add_coco_prediction(pred[variant],pred_ca[variant],iid,pp,
                                           float(image_data[iid]['width']),float(image_data[iid]['height']),cats)
            for variant in evals:
                ev,ec=evals[variant]
                ev.update(pred[variant]);ec.update(pred_ca[variant])
            print(f'    [COCO] Train-qualification video={vid} frames={len(can)}',flush=True)
            del can,spa,tmp,pred,pred_ca
            gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
    results={name:d.finalize_variant_evaluator(*evals[name],list(CLASSES)) for name in evals}
    require(all(math.isfinite(results[v]['map30']) and 0<=results[v]['map30']<=1 for v in results),
            'COCO qualification AP invalid')
    return results


def prospective_gate(report,min_active=1):
    b=report['metrics']['Base'];n=report['metrics']['Selected']; active=sum(v for k,v in report['selected_action_counts'].items() if k!='Base')
    fail=[]
    if active<min_active:fail.append('No accepted corrective actions')
    if n['recall']<b['recall']-0.01:fail.append('Recall loss more than 1pp')
    if n['f1']<=b['f1']:fail.append('F1 not improved')
    if n['fp']>=b['fp']:fail.append('False positives not reduced')
    if 'coco' in report and report['coco']['Selected']['map30'] < report['coco']['Base']['map30']:
        fail.append('COCO mAP@0.30 regressed')
    for camera,vals in report['camera_operational'].items():
        if vals['Selected']['recall'] < vals['Base']['recall']-0.01:
            fail.append(f'Camera {camera} recall fell >1pp')
    for cls,vals in report['per_class_recall'].items():
        if vals['support']>=30 and vals['selected_recall'] < vals['base_recall']-0.01:
            fail.append(f'Class {cls} recall fell >1pp')
    for sz,vals in report['per_scale_recall'].items():
        if vals['support']>=30 and vals['selected_recall'] < vals['base_recall']-0.01:
            fail.append(f'Scale {sz} recall fell >1pp')
    return {'status':'PASS' if not fail else 'HOLD','failures':fail,'accepted_frames':active}


def calibrate(model,cal,mu,sigma):
    p=OUT/'run15a_guard_calibration.json'
    if p.exists():
        z=jload(p);require(z['guard_checkpoint_sha256']==sha(OUT/'run15a_preservation_guard.pth'),
                           'Calibration references different learned guard')
        return z
    probs=predict_proba(model,cal,mu,sigma)
    rows=[]
    for threshold in CAL_THRESHOLDS:
        chosen=choose_from_probability(probs,threshold)
        report=evaluate_choices(cal,chosen)
        decision=prospective_gate(report)
        rows.append({'threshold':threshold,'gate':decision,'base':report['metrics']['Base'],
                     'selected':report['metrics']['Selected'],
                     'selected_action_counts':report['selected_action_counts']})
    eligible=[r for r in rows if r['gate']['status']=='PASS']
    # Only calibration split is used for threshold selection; never qualification split.
    # Utility is precision-biased WITH hard recall/F1/FP constraints, then conservative threshold.
    selected=max(eligible,key=lambda r:(r['base']['fp']-r['selected']['fp'],r['selected']['f1'],r['threshold'])) if eligible else None
    obj={'schema':'run15a_guard_calibration_v1','train_only':True,
         'guard_checkpoint_sha256':sha(OUT/'run15a_preservation_guard.pth'),
         'status':'CALIBRATED' if selected else 'HOLD_NO_SAFE_TRAIN_CALIBRATION',
         'selected_threshold':selected['threshold'] if selected else None,
         'grid':rows,'calibration_video_count':len(set(r['video'] for r in cal)),
         'no_qualification_labels_used_in_selection':True}
    jsave(p,obj)
    return obj


def qualify(model,qual,mu,sigma,cal,r9,r12,r8,d,s0a,data):
    p=OUT/'run15a_trainonly_qualification.json'
    require(not p.exists(),'Qualification exists; prohibit a second selection/evaluation')
    require(cal['status']=='CALIBRATED','Calibration HOLD; qualification must not run')
    proba=predict_proba(model,qual,mu,sigma)
    choices=choose_from_probability(proba,cal['selected_threshold'])
    results=evaluate_choices(qual,choices)
    results['coco']=qualification_coco_ap(qual,choices,r9,r12,r8,d,s0a,data)
    results.update({'schema':'run15a_guard_qualification_v1','partition':'TRAIN-video-heldout-for-guard-only',
                    'train_only':True,'candidate_sha256':sha(FINAL),
                    'guard_sha256':sha(OUT/'run15a_preservation_guard.pth'),
                    'calibration_sha256':sha(OUT/'run15a_guard_calibration.json'),
                    'fixed_threshold':cal['selected_threshold'],
                    'qualification_videos':sorted(set(r['video'] for r in qual)),
                    'decision':prospective_gate(results),
                    'critical_limitations':[
                       'Full RUN12 expert/downstream core already saw all 47 Train videos; this holdout is guard-only.',
                       'Existing internal RUN09 local-UH, OOF to full expert shift and hard-mask mismatch are not retroactively repaired.',
                       'This does not establish official independent Validation or Test superiority.',
                       'Rare small LEISURE_VEHICLE positive class has no certified Train support; cannot guarantee Test performance.']})
    jsave(p,results)
    return results


def run():
    print('========== RUN15A SINGLE INTEGRATED TRAIN-ONLY RESEARCH RUN ==========',flush=True)
    print('A: adversarial preflight -> B: actual guard training -> C: disjoint Train calibration/qualification',flush=True)
    require(not (OUT/'run15a_trainonly_qualification.json').exists(),
            'Final qualification is single-use; refuse repeated evaluation')
    r9,r12,r8,d,s0a,folds,parts,data=phase_preflight()
    meta=OUT/'run15a_contract.json'
    if not meta.exists():
        jsave(meta,{'schema':'run15a_contract_v1','source_sha256':sha(Path(__file__)),
                   'parent_checkpoint_sha256':R12_SHA,'train_json_sha256':sha(TRAIN_JSON),
                   'partitions':parts,'actions':ACTIONS,'features':FEATURES,
                   'no_validation_or_test_access':True,
                   'research_limits':{'exact_fallback_when_abstaining':True,
                        'never_worse_on_unseen_images_guaranteed':False,
                        'internal_RUN09_soft_hard_and_target_mismatches_corrected':False,
                        'upstream_full_train_core_independent_of_guard_qualification':False}})
    else:
        saved=jload(meta)
        require(saved['source_sha256']==sha(Path(__file__)) and saved['partitions']==parts
                and saved['parent_checkpoint_sha256']==R12_SHA,'Source, split or anchor changed after RUN15A began')
    audit_legacy_conformance(r9,r12,data)
    print('[PHASE A] Generating original joint-action Train counterfactual ledger')
    build_train_shards(r9,r12,r8,d,s0a,folds,parts,data)
    allrows=read_guard_data()
    groups={k:[r for r in allrows if r['video'] in set(vids)] for k,vids in parts.items()}
    require(all(groups.values()),'Empty Train partition')
    require(sum(len(a) for a in groups.values())==20075,'Incomplete Train split')
    print('[PHASE B] Training preservation-aware learned selector using FIT videos only')
    model,mu,sigma=train_guard(groups['fit'],groups['calibration'],groups['qualification'])
    print('[PHASE C1] Selecting a threshold using CALIBRATION videos only')
    cal=calibrate(model,groups['calibration'],mu,sigma)
    if cal['status']!='CALIBRATED':
        print('[HOLD] No calibration threshold met TP/FP/recall/F1 constraints. No qualifier opened.')
        return 2
    print('[PHASE C2] One-time qualification on independent GUARD-heldout Train videos')
    report=qualify(model,groups['qualification'],mu,sigma,cal,r9,r12,r8,d,s0a,data)
    print('[RESULT] Qualification',report['decision']['status'],'Base:',report['metrics']['Base'],
          'Selected:',report['metrics']['Selected'],flush=True)
    if report['decision']['status']!='PASS':
        print('[HOLD] Train-only qualification failed. No final qualified deployment artifact.')
        return 2
    deployment=OUT/'run15a_provisional_deployment_package.pth'
    require(not deployment.exists(),'Candidate deployment already exists')
    torch.save({'schema':'run15a_guarded_bfcef_f_provisional_v1',
                'parent_R12_sha256':R12_SHA,
                'guard_checkpoint_sha256':sha(OUT/'run15a_preservation_guard.pth'),
                'guard_state':model.state_dict(),'feature_names':FEATURES,'actions':ACTIONS,
                'normalizer_mean':mu,'normalizer_std':sigma,
                'threshold':float(cal['selected_threshold']),
                'scope':'TRAIN-ONLY provisional, NOT official Test qualified',
                'qualification_sha256':sha(OUT/'run15a_trainonly_qualification.json')},deployment)
    jsave(OUT/'run15a_provisional_deployment_certificate.json',
          {'schema':'run15a_provisional_certificate_v1','sha256':sha(deployment),
           'parent_sha256':R12_SHA,'qualification_sha256':sha(OUT/'run15a_trainonly_qualification.json'),
           'certification_level':'TRAIN_GUARD_ONLY_NOT_INDEPENDENT_FULL_MODEL',
           'official_test_accessed':False})
    print('[PASS] Train-guard qualification; PROVISIONAL package saved, NOT permission to open Test')
    return 0


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=('selftest','run'),required=True)
    args=p.parse_args();selftest()
    if args.phase=='selftest':return
    sys.exit(run())

if __name__=='__main__':main()
