"""RUN15A read-only FIT-only risk/action target census; no optimizer/checkpoint writes.

Quantifies isolated full-amplitude expert action targets and the risk scores
of an existing *frozen* trained pilot checkpoint. Does NOT evaluate mAP or
certify correction safety. Never searches outside the predeclared FIT split.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
from PIL import Image  # Stable Windows import ordering previously used
import torch

from run15a_final_risk import RiskPolicy
from run15a_final_mechanism import mechanism_forward
from run15a_final_supervision import isolated_native_targets
from run15a_pilot32_admission_postcheck import (
    EXPECTED_SOURCES, EXPECTED_CHECKPOINT, EXPERTS, risk_for_task,
    percentile,
)
from run15a_final_windows import FIT, load_runtime, require, sha, write_new

EXPECTED_POSTCHECK_SHA = '8ffe141883aa9d7768f0e5bdbcd90a00cc2bf5386e614f3178ebcbab6fa411e9'


def cls_action_labels(t, available):
    """Isolated global-action class transition labels, NOT joint guarantees."""
    state=t['transition']
    relevant=t['relevant']
    if state.ndim != 2 or state.shape != relevant.shape or state.shape[0] != available.shape[0]:
        raise ValueError('Classification transition shape mismatch')
    useful=((state==3)|(state==5)|(state==6)) & relevant
    harmful=((state==1)|(state==2)|(state==7)) & relevant
    valid=relevant.any(-1) & available
    return useful.any(-1) & valid, harmful.any(-1) & valid, valid


def box_action_labels(t, available):
    """GT match rescue/loss only. FP-pruning not encoded in original box targets."""
    rescue=t['rescue'].reshape(-1)>0.5
    lost=t['loss'].reshape(-1)>0.5
    valid=t['relevant'].reshape(-1).bool() & available
    if rescue.shape != valid.shape or lost.shape != valid.shape:
        raise ValueError('Box rescue/loss shape mismatch')
    return rescue & valid, lost & valid, valid


def group_counts(useful, harm, valid, accepted):
    if not(useful.shape==harm.shape==valid.shape==accepted.shape):
        raise ValueError('Invalid shape in census')
    good=useful & ~harm & valid
    bad=harm & valid
    mixed=useful & harm & valid
    neutral=~useful & ~harm & valid
    return {
        'eligible_action_examples':int(valid.sum().item()),
        'useful_any':int((useful & valid).sum().item()),
        'harm_any':int(bad.sum().item()),
        'safe_useful':int(good.sum().item()),
        'mixed_useful_harmful':int(mixed.sum().item()),
        'neutral':int(neutral.sum().item()),
        'accepted':int((accepted & valid).sum().item()),
        'safe_useful_accepted':int((good & accepted).sum().item()),
        'harmful_accepted':int((bad & accepted).sum().item()),
        'safe_useful_rejected':int((good & ~accepted).sum().item()),
    }


def run(project, videos, frames):
    project=Path(project).resolve()
    require(1<=len(videos)<=4 and len(set(videos))==len(videos) and all(v in FIT for v in videos),
            'Unique permitted FIT videos only')
    require(1<=frames<=64,'Invalid frame budget')
    for f,digest in EXPECTED_SOURCES.items():
        require(sha(project/f)==digest,'Production source drift: '+f)
    ckpt=project/'bfcef_f_s0'/'run15a_final_candidate_sandbox'/'pilot_fit'/'pilot_videos_2_3_frames16.pth'
    receipt=ckpt.with_suffix('.receipt.json')
    require(ckpt.exists() and receipt.exists(), 'Pilot checkpoint missing')
    require(sha(ckpt)==EXPECTED_CHECKPOINT==EXPECTED_POSTCHECK_SHA,'Checkpoint SHA drift')
    rc=json.loads(receipt.read_text(encoding='utf8'))
    require(rc['checkpoint_sha256']==EXPECTED_CHECKPOINT and rc['frames']==32,'Pilot receipt mismatch')

    old,r9,r8,d,s0a,raw,_,_,_=load_runtime(project,downstream='none')
    r12=old.imports_and_verify()[1]
    model,criterion,CO,FX=old.make_model(r9,r12)
    payload=torch.load(ckpt,map_location='cpu',weights_only=False)
    require(payload.get('checkpoint_type')=='PILOT_ONLY_NOT_CALIBRATED','Unexpected checkpoint')
    require(tuple(payload['video_ids'])==(2,3) and payload['max_frames_per_video']==16,'Pilot identity drift')
    old.restore_downstream(model,payload['downstream_state'])
    old.lock_gate_thresholds(model)
    old.verify_gate_thresholds(model,'risk-action census')
    model.eval()
    policy=RiskPolicy()

    counts={f'{task}_{e}':collections.Counter() for task in ('cls','box') for e in EXPERTS}
    margins={k:{name:[] for name in ('safe_useful','harmful','neutral','mixed')} for k in counts}
    video_summaries={}
    nframe=0
    with torch.no_grad():
        for vid in videos:
            can,spa,tmp=s0a.load_video_triplet(vid)
            for j in range(min(frames,len(can))):
                fr=can[j]
                O,X=s0a.make_O_X(fr,spa[j],tmp[j],CO,FX)
                bundle=r9.expert_bundle(model,O,X)
                out=mechanism_forward(model,O,X,bundle,hard=True,policy=policy)
                target=isolated_native_targets(model,O,bundle,fr['gt_boxes_norm'],fr['gt_labels'],old.capture_match)
                for task in ('cls','box'):
                    availability=out['A_'+task].bool()
                    for k,e in enumerate(EXPERTS):
                        eligible=availability[:,k]
                        if task=='cls':
                            good,bad,valid=cls_action_labels(target[e]['cls'],eligible)
                        else:
                            good,bad,valid=box_action_labels(target[e]['box'],eligible)
                        risk=risk_for_task(out,O,e,task,policy)
                        margin=risk['score'].reshape(-1)
                        accepted=risk['mask_hard'].reshape(-1).bool()
                        key=f'{task}_{e}'
                        counts[key].update(group_counts(good,bad,valid,accepted))
                        sub={
                            'safe_useful':good & ~bad & valid,
                            'harmful':bad & valid,
                            'neutral':~good & ~bad & valid,
                            'mixed':good & bad & valid,
                        }
                        for name,mask in sub.items():
                            margins[key][name].extend(margin[mask].detach().cpu().tolist())
                nframe+=1
            video_summaries[str(vid)]={'frames':min(frames,len(can))}
            print(f'[TARGET CENSUS] video={vid}, frames={video_summaries[str(vid)]["frames"]}',flush=True)
    summary={}
    for key,cnt in sorted(counts.items()):
        n=cnt['eligible_action_examples']; positives=cnt['safe_useful']
        summary[key]={**dict(cnt),
            'safe_useful_prevalence':positives/n if n else None,
            'risk_recall_safe_useful':cnt['safe_useful_accepted']/positives if positives else None,
            'risk_margin_by_outcome_p05_p50_p95':{
                kind:[percentile(v,p) for p in (.05,.5,.95)]
                for kind,v in margins[key].items()
            }}
    return {'schema':'RUN15A_RISK_ACTION_CENSUS_v1',
            'status':'DIAGNOSTIC_NOT_TRAINED_OR_QUALIFIED',
            'videos':list(videos),'frames':nframe,'checkpoint_sha256':EXPECTED_CHECKPOINT,
            'source_sha256':{f:sha(project/f) for f in EXPECTED_SOURCES},
            'per_video':video_summaries,'per_expert_task':summary,
            'no_training_or_optimizer':True,'CAL_QUAL_Val_Test_unopened':True,
            'limitations':[
                'Outcome labels use isolated full-amplitude expert counterfactuals on FIT; do not establish joint-action safety',
                'Box usefulness labels use GT rescue only: FP pruning and AP ranking changes are omitted',
                'Classwise action targets may include competing class transitions; mixed actions flagged separately',
                'No mAP is calculated; probability scores are not calibrated',
                'All FIT videos have previously been used by upstream and earlier downstream models',
            ]}


def selftest():
    t={'transition':torch.tensor([[3,0],[5,8],[1,6],[8,8],[6,7]]),
       'relevant':torch.tensor([[1,1],[1,1],[1,1],[1,1],[1,1]],dtype=torch.bool)}
    u,h,v=cls_action_labels(t,torch.ones(5,dtype=torch.bool))
    assert u.tolist()==[True,True,True,False,True]
    assert h.tolist()==[False,False,True,False,True]
    c=group_counts(u,h,v,torch.tensor([0,1,0,0,0]).bool())
    assert c['safe_useful']==2 and c['mixed_useful_harmful']==2 and c['neutral']==1
    bt={'rescue':torch.tensor([[1],[0],[1]],dtype=torch.float32),
        'loss':torch.tensor([[0],[1],[1]],dtype=torch.float32),
        'relevant':torch.ones(3,1,dtype=torch.bool)}
    u,h,v=box_action_labels(bt,torch.ones(3,dtype=torch.bool))
    assert (u.tolist(),h.tolist())==([True,False,True],[False,True,True])
    print('[PASS] risk-action census class/box label and mixed-harm selftests')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project',default=r'D:\MARIVISION\H0_workbench\eval_tools')
    ap.add_argument('--videos',default='2,3,4')
    ap.add_argument('--frames',type=int,default=16)
    ap.add_argument('--selftest',action='store_true')
    a=ap.parse_args()
    if a.selftest:selftest();return
    vids=[int(s) for s in a.videos.split(',')]
    report=run(a.project,vids,a.frames)
    out=Path(a.project)/'bfcef_f_s0'/'run15a_final_candidate_sandbox'/'pilot_postcheck'/f'risk_action_census_videos_{"_".join(map(str,vids))}_frames{a.frames}.json'
    write_new(out,report)
    print('[RISK TARGET CENSUS COMPLETE]',out)
    for key,v in sorted(report['per_expert_task'].items()):
        print(f'  {key}: safe_useful={v.get("safe_useful",0)} / {v.get("eligible_action_examples",0)}; harm={v.get("harm_any",0)}; admitted_good={v.get("safe_useful_accepted",0)}; median_good_margin={v["risk_margin_by_outcome_p05_p50_p95"]["safe_useful"][1]}')
    print('[HOLD] Diagnostic only. No training, CAL/QUAL/Validation/Test, or AP claim')

if __name__=='__main__':main()
