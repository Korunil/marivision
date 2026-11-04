#!/usr/bin/env python3
"""RUN_15A FIT-only feasibility evidence, NON-mutating to existing research artifacts.

Three independent phases:
  aggregate: certified existing stride-1 FIT report -> *video-level* conservative
             GT-informed intervention envelope. Does NOT infer frame-level labels.
  export:    source-pinned replay on *FIT-only* cached frames; new JSONL evidence,
             operational TP/FP/GT identity, observable proposal features, BCI coverage.
  probe:     evaluate frame-level GT-informed selection and video-grouped lightweight
             predictor with a frozen 0.5/0.10 decision rule. No trained weights saved.

No experiment changes; no CAL/QUAL/official Validation/Test; not a Swin-B evaluator.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

FIT_VIDEOS=(2,3,4,5,6,7,8,13,14,16,17,18,19,20,24,25,29,31,35,36,37,38,40,41,42,43,44,45,47)
CAL_VIDEOS=(11,12,15,21,23,30,33,39,46)
QUAL_VIDEOS=(1,9,10,22,26,27,28,32,34)
ACTIONS=('Base','ClsOnly','BoxOnly','Full')
CHANGES=('ClsOnly','BoxOnly','Full')
REPORT_SCHEMA='run15a_v2_fit_mechanism_audit_v1'
EXPECTED_DRIVER='6fa33f1128cbc22266c580e7be85a4b69fd1bac2651ed358b64140236cd733e2'
EXPECTED_MECHANISM='42404d765097c65b23ec318ed51b8c4fc7b2acdd9eb34a331a76c09313bcb637'
EXPECTED_CKPT='b265aa6a12987de61afaa0010071717b853b01ea0844d1502cc8321029cb3f8d'
EXPECTED_CERT='d44191b7c8894854b8cc8770fe12e6e3c6ba18b0aa0fe8aba28812283329b1a4'
EXPECTED_REPORT_SHA='d045d82a50943467d70d63af1f53fd8f42ae5e8ec7358f3333677e7c28841504'
EXPECTED_AUDIT_SOURCE_SHA='7dc6f36d0189f6af149c6aa11e4efc5f6a1bd914df29e13c7a045b6de4b7c043'
PROOF_SCHEMA='run15a_fit_feasibility_v1'


def require(x, description):
    if not x: raise RuntimeError('[RUN15A FEASIBILITY HOLD] '+str(description))


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def load(path):
    with open(path,encoding='utf-8') as f:return json.load(f)


def write_new(path,data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(data,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print('[NEW]',path)


def metric(tp,fp,fn):
    p=tp/(tp+fp) if tp+fp else 0
    r=tp/(tp+fn) if tp+fn else 0
    f=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0
    return dict(tp=int(tp),fp=int(fp),fn=int(fn),precision=p,recall=r,f1=f)


def validate_report(path):
    require(sha256(path)==EXPECTED_REPORT_SHA,'Archived FIT report SHA-256 mismatch')
    d=load(path)
    require(d.get('schema')==REPORT_SCHEMA,'Audit schema mismatch')
    require(d.get('stage')=='FIT_ONLY_POSTHOC_REPLAY','Not FIT-only report')
    require(d.get('stride')==1,'Only full FIT audit report may anchor this proof')
    require(tuple(d['FIT_videos'])==FIT_VIDEOS,'FIT manifest mismatch')
    require(set(d['FIT_videos']).isdisjoint(CAL_VIDEOS+QUAL_VIDEOS),'Population overlap')
    pins=d['source_pins']
    for key,digest in [('driver',EXPECTED_DRIVER),('mechanism',EXPECTED_MECHANISM),
                       ('fit_checkpoint',EXPECTED_CKPT),('fit_certificate',EXPECTED_CERT)]:
        require(pins[key]==digest,key+' immutable SHA mismatch')
    v=d['per_video']
    require(set(v)==set(map(str,FIT_VIDEOS)),'Per-video coverage mismatch')
    require(d['FIT_video_count']==29 and d['FIT_cached_frames_sampled']==11993 and d['FIT_gt_sampled']==41284,
            'Unexpected FIT sample census')
    require(sum(t['frames'] for t in v.values())==11993,'Frame conservation')
    require(sum(t['gt'] for t in v.values())==41284,'GT conservation')
    for vid,t in v.items():
        for a in ACTIONS:
            m=t['variants'][a]
            require(m['tp']+m['fn']==t['gt'],f'GT conservation video={vid},action={a}')
            require(t['rescued_gt_against_base'][a]-t['lost_gt_against_base'][a]==
                    m['tp']-t['variants']['Base']['tp'],f'Identity conservation video={vid},action={a}')
        require(t['lost_gt_against_base']['Base']==0 and t['rescued_gt_against_base']['Base']==0,
                'Baseline identity self-contrast not zero')
    for a in ACTIONS:
        for k in ('tp','fp','fn'):
            require(sum(t['variants'][a][k] for t in v.values())==d['metrics'][a][k],
                    f'Global mismatch variant={a} key={k}')
        for key,global_key in [('lost_gt_against_base','lost_gt_against_base'),
                                ('rescued_gt_against_base','rescued_gt_against_base')]:
            require(sum(t[key][a] for t in v.values())==d[global_key][a],f'{key}/{a} mismatch')
    require(d['lost_gt_against_base']['Full']==sum(d['lost_gt_stage_counts'].values()),
            'Loss-stage conservation')
    return d


def choose_safe(action_records):
    """Diagnostic GT-informed choice: frame/video whole-action, ZERO baseline GT identities lost;
    restrict to candidates with no worse FP and nondecreasing pooled TP. Maximizes *local*
    F1 then FP reduction and TP gain; not globally F1 optimal, not deployable.
    """
    base=action_records['Base']
    eligible=[]
    for a in ACTIONS:
        x=action_records[a]
        if x['lost']==0 and x['fp']<=base['fp'] and x['tp']>=base['tp']:
            eligible.append(a)
    require('Base' in eligible,'Baseline not available as safe fallback')
    return max(eligible,key=lambda a:(metric(action_records[a]['tp'],action_records[a]['fp'],
                                             action_records[a]['fn'])['f1'],
                                     base['fp']-action_records[a]['fp'],
                                     action_records[a]['tp']-base['tp'],
                                     -ACTIONS.index(a)))


def action_from_summary(summary,a):
    m=summary['variants'][a]
    return {'tp':m['tp'],'fp':m['fp'],'fn':m['fn'],
            'lost':summary['lost_gt_against_base'][a],
            'rescued':summary['rescued_gt_against_base'][a]}


def aggregate(fit_report,out):
    d=validate_report(fit_report)
    selected=[];bases=[];choices=collections.Counter();per_video={}
    for vid in FIT_VIDEOS:
        s=d['per_video'][str(vid)]
        decisions={a:action_from_summary(s,a) for a in ACTIONS}
        pick=choose_safe(decisions)
        selected.append(decisions[pick]);bases.append(decisions['Base']);choices[pick]+=1
        per_video[str(vid)]={'camera':s['camera'],'choice':pick,'gt':s['gt'],
                             'lost':decisions[pick]['lost'],'rescued':decisions[pick]['rescued'],
                             'delta_fp':decisions[pick]['fp']-decisions['Base']['fp'],
                             'delta_tp':decisions[pick]['tp']-decisions['Base']['tp']}
    def pool(rows):return metric(*(sum(r[k] for r in rows) for k in ('tp','fp','fn')))
    baseline=pool(bases); chosen=pool(selected)
    require(sum(r['lost'] for r in selected)==0,'Oracle violates no-baseline-identity-loss contract')
    assert chosen['tp']>=baseline['tp'] and chosen['fp']<=baseline['fp']
    result={'schema':PROOF_SCHEMA,'phase':'aggregate','interpretation':
            'GT-informed conservative per-video action chooser; not framewise oracle, not deployable',
            'input_report_sha256':sha256(fit_report),'fit_videos':list(FIT_VIDEOS),
            'baseline_pooled':baseline,'video_oracle_pooled':chosen,
            'global_delta_tp':chosen['tp']-baseline['tp'],
            'global_delta_fp':chosen['fp']-baseline['fp'],
            'choices':dict(choices),'per_video':per_video,
            'unsupported':{'per_frame_oracle':'REQUIRES PER-FRAME COUNTERFACTUALS',
                           'observable_risk_predictability':'REQUIRES FRAME-LEVEL OBSERVABLE FEATURE/OUTCOME PAIRS',
                           'bci_gt_coverage':'REQUIRES QUERY-LEVEL EVIDENCE',
                           'mAP':'CANNOT INFER FROM TP/FP/FN COUNTS',
                           'Swin-B':'SEPARATE ANCHOR, NOT EVALUATED'}}
    write_new(out,result)
    print('[AGGREGATE COMPLETE] Base:',baseline,'\nVideo-GT-informed:',chosen,'\nChoice census',dict(choices))
    return result


def tensor_features(original,out,action,cam):
    """All features measured before GT inspection, from frozen inference outputs only."""
    import torch
    with torch.no_grad():
        z=original.z.detach().float();b=original.b.detach().float()
        z2,b2=action
        z2=z2.detach().float();b2=b2.detach().float()
        cand=((out['A_cls']>0).any(1)|(out['A_box']>0).any(1))
        require(int(cand.sum())>0,'BCI has no eligible queries')
        probs=z.sigmoid();top=probs.flatten().topk(min(100,probs.numel())).values
        small=(b[:,2].clamp_min(0)*b[:,3].clamp_min(0)<=32**2/640**2)
        cl=out['hard_cls'];bx=out['hard_box']
        keys={'camera_Axis':float(cam=='Axis'),'camera_FLIR':float(cam=='FLIR'),
              'camera_eCAM':float(cam=='e-CAM'),
              'n_candidate':float(cand.sum()),
              'base_top100_score_mean':float(top.mean()),
              'base_top100_score_max':float(top[0]),
              'base_top100_score_at_rank100':float(top[-1]),
              'base_operating_scores':float((top>=.2).sum()),
              'base_candidate_score_mean':float(probs[cand].max(1).values.mean()),
              'base_candidate_box_area_mean':float((b[cand,2].clamp_min(0)*b[cand,3].clamp_min(0)).mean()),
              'base_candidate_cached640_small_fraction':float(small[cand].float().mean()),
              'cls_route_sum':float(out['pi_cls'][cand,1:].sum(1).mean()),
              'box_route_sum':float(out['pi_box'][cand,1:].sum(1).mean()),
              'cls_hard_fraction':float((cl[cand].sum(1)>0).float().mean()),
              'box_hard_fraction':float((bx[cand].sum(1)>0).float().mean()),
              'cls_delta_abs_mean':float((z2[cand]-z[cand]).abs().mean()),
              'box_delta_abs_mean':float((b2[cand]-b[cand]).abs().mean()),
              'score_delta_abs_mean':float((z2[cand].sigmoid()-probs[cand]).abs().mean()),
              'corrected_candidate_score_mean':float(z2[cand].sigmoid().max(1).values.mean())}
        return keys,cand


def coverage_for_missing_gt(original,match0,gt,cand,iou_threshold=0.30):
    """Overlap coverage by EXISTING original R0 boxes. Not a strict bound on box correction."""
    import torch
    with torch.no_grad():
        matched={int(j) for _,_,j in match0}
        missing=[j for j in range(gt.shape[0]) if j not in matched]
        yes_cand=yes_all=0
        boxes=original.b.detach().float()
        for j in missing:
            target=gt[j].to(boxes.device).float()
            # Vectorized cxcywh IoU; only baseline box overlap, class-agnostic.
            p0=boxes[:,:2]-boxes[:,2:]/2;p1=boxes[:,:2]+boxes[:,2:]/2
            t0=target[:2]-target[2:]/2;t1=target[:2]+target[2:]/2
            wh=(torch.minimum(p1,t1)-torch.maximum(p0,t0)).clamp_min(0)
            inter=wh[:,0]*wh[:,1]
            area=boxes[:,2].clamp_min(0)*boxes[:,3].clamp_min(0)
            ta=target[2].clamp_min(0)*target[3].clamp_min(0)
            iou=inter/(area+ta-inter).clamp_min(1e-9)
            yes_cand+=int(bool((iou[cand]>=iou_threshold).any().item()))
            yes_all+=int(bool((iou>=iou_threshold).any().item()))
        require(yes_cand<=yes_all<=len(missing),'Candidate coverage monotonicity violated')
        return {'missing_gt':len(missing),'selected_bci_iou30':yes_cand,
                'all_R0_query_iou30':yes_all}


def export_fit(project,fit_report,out_dir,stride):
    require(int(stride)>=1,'Stride must be >=1')
    project=Path(project).resolve();out_dir=Path(out_dir).resolve()
    original_sha=sha256(fit_report);d=validate_report(fit_report)
    require(sha256(project/'run15a_v2_fit_mechanism_audit.py')==EXPECTED_AUDIT_SOURCE_SHA,
            'Original certified FIT audit Python source modified')
    sys.path.insert(0,str(project))
    import run15a_v2_fit_mechanism_audit as audit
    original,mech,pins,_=audit.verify_inputs(project)
    require(pins==d['source_pins'],'Current FIT source differs from report')
    r9,r12,r8,dev,s0a,raw=original.imports_and_verify()
    model,criterion,co,fx=original.load_fitted(r9,r12)
    model.eval()
    before=audit.downstream_state_digest(original,model)
    # Contract created before any FIT cache access, never overwritten.
    contract={'schema':PROOF_SCHEMA,'phase':'export','FIT_videos':list(FIT_VIDEOS),
              'stride':stride,'fit_report_sha256':original_sha,
              'source_pins':pins,'all_existing_artifacts_read_only':True,
              'CAL_QUAL_official_val_test_unopened':True}
    cp=out_dir/'export_contract.json';out_dir.mkdir(parents=True,exist_ok=True)
    if cp.exists():require(load(cp)==contract,'Export resume contract mismatch')
    else:write_new(cp,contract)
    by_id={int(im['id']):im for im in raw['images']}
    gt_count=collections.Counter(int(a['image_id']) for a in raw['annotations'])
    cams={int(v['id']):str(v['camera_sensor']) for v in raw['videos']}
    gt_classes=collections.defaultdict(collections.Counter)
    for a in raw['annotations']:
        gt_classes[int(a['image_id'])][int(a['category_id'])-1]+=1
    total=0
    import torch
    with torch.no_grad():
        for vid in FIT_VIDEOS:
            dst=out_dir/f'FIT_video_{vid:02d}.jsonl'
            receipt=dst.with_suffix('.receipt.json')
            if dst.exists():
                require(receipt.is_file(),'Existing data without receipt: '+str(dst))
                meta=load(receipt)
                require(meta['video']==vid and meta['stride']==stride and
                        meta['file_sha256']==sha256(dst) and
                        meta['contract_sha256']==sha256(cp),
                        'Resume input SHA/contract failure: '+str(dst))
                print('[RESUME] SHA/contract-verified:',dst)
                continue
            require(not receipt.exists(),'Orphaned receipt: '+str(receipt))
            can,spa,tmp=s0a.load_video_triplet(vid)
            require(len(can)>0 and len(can)==len(spa)==len(tmp),f'FIT triplet length mismatch {vid}')
            video_cam=cams[vid]
            require(video_cam in ('Axis','FLIR','e-CAM'),'Camera unknown')
            # New evidence only; file written to .partial then committed on success.
            partial=dst.with_suffix('.jsonl.partial')
            require(not partial.exists(),'Incomplete transaction: '+str(partial))
            rows=0
            try:
                with partial.open('x',encoding='utf-8',newline='\n') as f:
                    for i in range(0,len(can),stride):
                        fr=can[i];iid=int(fr['image_id']);gt=fr['gt_boxes_norm']
                        require(iid in by_id and int(by_id[iid]['video_id'])==vid,'FIT frame population leak')
                        require(gt_count[iid]==len(fr['gt_labels']),'GT image count mismatch')
                        require(gt_classes[iid]==collections.Counter(map(int,fr['gt_labels'].tolist())),
                                'GT category mismatch')
                        O,X=s0a.make_O_X(fr,spa[i],tmp[i],co,fx)
                        bundle=r9.expert_bundle(model,O,X)
                        result=mech.mechanism_forward(model,O,X,bundle,hard=True)
                        audit.assert_model_shapes(result,O)
                        outputs={'Base':(O.z,O.b),'ClsOnly':(result['z'],O.b),
                                 'BoxOnly':(O.z,result['b']),'Full':(result['z'],result['b'])}
                        matches={};counts={};feat={};match_tuples={}
                        for a,(z,b) in outputs.items():
                            m,fp,c=original.capture_match(z,b,gt,fr['gt_labels'])
                            match_tuples[a]=m
                            matches[a]={int(j) for _,_,j in m}
                            counts[a]=c
                            if a!='Base':feat[a],cand=tensor_features(O,result,(z,b),video_cam)
                            else:_,cand=tensor_features(O,result,(z,b),video_cam)
                        base=matches['Base']
                        cov=coverage_for_missing_gt(O,match_tuples['Base'],gt,cand)
                        frame={'schema':PROOF_SCHEMA,'video':vid,'image_id':iid,'camera':video_cam,
                               'stride':stride,'frame_cache_idx':i,'gt_count':len(fr['gt_labels']),
                               'coverage':cov,'actions':{}}
                        for a in ACTIONS:
                            c=counts[a]
                            frame['actions'][a]={'tp':c['tp'],'fp':c['fp'],'fn':c['fn'],
                                'lost':len(base-matches[a]),'rescued':len(matches[a]-base)}
                            if a!='Base':frame['actions'][a]['features']=feat[a]
                            require(c['tp']-counts['Base']['tp']==
                                    frame['actions'][a]['rescued']-frame['actions'][a]['lost'],
                                    'Frame GT identity accounting failed')
                        f.write(json.dumps(frame,allow_nan=False,sort_keys=True)+'\n');rows+=1
                partial.rename(dst)
            except BaseException:
                print('[HOLD] Partial transaction retained:',partial)
                raise
            require(rows==len(range(0,len(can),stride)),'Sampled frame census mismatch')
            write_new(receipt,{'schema':PROOF_SCHEMA,'video':vid,'stride':stride,
                 'rows':rows,'contract_sha256':sha256(cp),'file_sha256':sha256(dst)})
            print(f'[FIT evidence] video={vid}, sampled frames={rows}, file={dst}',flush=True)
            total+=rows
    require(before==audit.downstream_state_digest(original,model),'Fitted model mutated in RAM')
    require(sha256(project/'bfcef_f_s0'/'run15a_internal_rebuild_v2_fixed_gate'/'downstream_internal_fitted.pth')==EXPECTED_CKPT,
            'Fitted checkpoint changed')
    require(sha256(fit_report)==original_sha,'Original audit report changed')
    print('[EXPORT COMPLETE] Newly exported frames:',total,'; stride:',stride,
          '\n[Next] --phase probe --evidence',out_dir)


def all_evidence(evidence):
    p=Path(evidence).resolve();contract=load(p/'export_contract.json')
    require(contract['schema']==PROOF_SCHEMA and contract['phase']=='export','Export contract invalid')
    require(tuple(contract['FIT_videos'])==FIT_VIDEOS,'Export FIT population mismatch')
    rows=[];seen=set()
    for vid in FIT_VIDEOS:
        fn=p/f'FIT_video_{vid:02d}.jsonl'
        receipt=fn.with_suffix('.receipt.json')
        require(fn.is_file() and receipt.is_file(),'Incomplete FIT-video evidence: '+str(fn))
        meta=load(receipt)
        require(meta['schema']==PROOF_SCHEMA and meta['video']==vid and
                meta['stride']==contract['stride'] and
                meta['contract_sha256']==sha256(p/'export_contract.json') and
                meta['file_sha256']==sha256(fn), 'Evidence receipt SHA or contract mismatch')
        count=0
        with fn.open(encoding='utf-8') as f:
            for line in f:
                row=json.loads(line)
                require(row['schema']==PROOF_SCHEMA and row['video']==vid,'Unexpected file row')
                require(row['stride']==contract['stride'],'Mixed sampling stride')
                require(set(row['actions'])==set(ACTIONS),'Action count mismatch')
                key=(vid,row['image_id']);require(key not in seen,'Repeated image')
                seen.add(key);rows.append(row);count+=1
        require(count==meta['rows'],'Evidence row count vs receipt mismatch')
    require(rows,'Empty evidence')
    if contract['stride']==1:
        require(len(rows)==11993,'Full stride-1 FIT count mismatch')
        require(sum(r['gt_count'] for r in rows)==41284,'Full FIT GT count mismatch')
    return contract,rows


def oracle_frame(rows):
    picks=[];selected=[];bases=[];count=collections.Counter()
    for r in rows:
        pick=choose_safe(r['actions']);picks.append(pick);count[pick]+=1
        selected.append(r['actions'][pick]);bases.append(r['actions']['Base'])
    def pool(xs):return metric(*(sum(r[k] for r in xs) for k in ('tp','fp','fn')))
    base=pool(bases);oracle=pool(selected)
    require(sum(x['lost'] for x in selected)==0,'Oracle GT-identity breach')
    return dict(baseline=base,selected=oracle,choices=dict(count),
                gt_informed=True,decision_scale='per_frame',
                f1_non_global_optimum=True),picks


def cross_video_probe(rows):
    """Diagnostic selector only. No saved fitted weights; strict video-grouped folds.

    Binary heads predict P(useful) and P(harm) for each action vs R0. Fixed
    policy: choose top P(useful) >= 0.5, P(harm) <= 0.10, else baseline.
    """
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        return {'status':'NOT_RUN_MISSING_SKLEARN','message':str(exc)},None
    names=sorted(rows[0]['actions']['ClsOnly']['features'])
    sample=[];groups=[];useful=[];harm=[];action_idx=[]
    for i,r in enumerate(rows):
        base=r['actions']['Base']
        for a_idx,a in enumerate(CHANGES):
            act=r['actions'][a]
            require(set(act['features'])==set(names),'Feature schema changed')
            sample.append([act['features'][k] for k in names]+[float(a_idx==j) for j in range(3)])
            groups.append(r['video']);action_idx.append((i,a))
            useful.append(int(act['lost']==0 and act['fp']<=base['fp'] and act['tp']>=base['tp'] and
                              (act['fp']<base['fp'] or act['rescued']>0)))
            harm.append(int(act['lost']>0))
    X=np.asarray(sample,dtype=np.float64);Yuse=np.asarray(useful,dtype=int)
    Yharm=np.asarray(harm,dtype=int);g=np.asarray(groups)
    require(bool(np.isfinite(X).all()),'Features nonfinite')
    p_good=np.zeros(X.shape[0]);p_harm=np.ones(X.shape[0]);holdout=[]
    cv=GroupKFold(n_splits=5)
    for tr,te in cv.split(X,Yuse,g):
        require(set(g[tr]).isdisjoint(set(g[te])),'Cross-video data leakage')
        for targets,pred,mode in [(Yuse,p_good,'useful'),(Yharm,p_harm,'harm')]:
            if len(set(targets[tr].tolist()))<2:
                pred[te]=float(targets[tr][0])
            else:
                model=make_pipeline(StandardScaler(),LogisticRegression(
                    C=0.1,max_iter=250,random_state=0))
                model.fit(X[tr],targets[tr]);pred[te]=model.predict_proba(X[te])[:,1]
        holdout.append({'videos':sorted(set(g[te].tolist())),'rows':len(te),
                        'useful_prevalence':float(Yuse[te].mean()),
                        'harm_prevalence':float(Yharm[te].mean())})
    candidate={}
    for k,(i,a) in enumerate(action_idx):candidate.setdefault(i,[]).append((a,p_good[k],p_harm[k]))
    results=[];picked=collections.Counter();harm_count=0;selected_gain=0
    for i,r in enumerate(rows):
        eligible=[(a,pg,ph) for a,pg,ph in candidate[i] if pg>=0.5 and ph<=0.10]
        if eligible:
            action=max(eligible,key=lambda e:(e[1]-e[2],e[1],-CHANGES.index(e[0])))[0]
        else:action='Base'
        picked[action]+=1;res=r['actions'][action];results.append(res)
        harm_count+=res['lost'];selected_gain+=res['rescued']
    pooled=metric(*(sum(rr[k] for rr in results) for k in ('tp','fp','fn')))
    def auroc(y,p):
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y,p)) if len(set(y.tolist()))==2 else None
    return {'status':'COMPLETED_DIAGNOSTIC_ONLY',
            'validation':'5-fold grouped-by-FIT-video, no CAL/QUAL, upstream R0/experts and V2 saw FIT',
            'fixed_rule':'P(useful)>=0.50 and P(harm)<=0.10; max P(useful)-P(harm)',
            'no_threshold_tuning':True,'n_FIT_videos':len(set(groups)),
            'n_frames':len(rows),'n_action_examples':len(sample),
            'feature_keys':names,'fitted_weights_saved':False,
            'useful_prevalence':float(Yuse.mean()),'harm_prevalence':float(Yharm.mean()),
            'useful_auroc':auroc(Yuse,p_good),'harm_auroc':auroc(Yharm,p_harm),
            'selected_GT_losses':harm_count,'selected_rescues':selected_gain,
            'pooled_policy':pooled,'selected_census':dict(picked),
            'folds':holdout},results


def probe(evidence,out):
    contract,rows=all_evidence(evidence)
    oracle,picks=oracle_frame(rows)
    prediction,_=cross_video_probe(rows)
    missing=sum(r['coverage']['missing_gt'] for r in rows)
    selected=sum(r['coverage']['selected_bci_iou30'] for r in rows)
    allq=sum(r['coverage']['all_R0_query_iou30'] for r in rows)
    require(selected<=allq<=missing,'Aggregate query coverage monotonicity')
    result={'schema':PROOF_SCHEMA,'phase':'probe','export_contract_sha256':sha256(Path(evidence)/'export_contract.json'),
            'stride':contract['stride'],'sampled_frames':len(rows),'video_count':len(set(r['video'] for r in rows)),
            'frame_oracle':oracle,'grouped_predictability':prediction,
            'BCI_existing_box_overlap':{'original_R0_missed_gt':missing,
               'selected_candidate_original_box_iou30':selected,
               'all_300_query_original_box_iou30':allq,
               'selected_over_missed':selected/missing if missing else None,
               'all300_over_missed':allq/missing if missing else None,
               'interpretation':'Existing R0 box overlap only; NOT a hard upper bound on correctable boxes'},
            'limitations':['All FIT observations exposed to R0/full expert fitting and V2 downstream FIT',
              'GT-informed oracle is non-deployable and constrained to whole-frame actions',
              'Framewise F1 greedy is not globally optimal pooled F1',
              'Predictability CV holds out FIT videos only for newly trained linear action probe',
              'Probe labels/metrics use operational Top100, threshold .20, same-class IoU .30',
              'Predicted-action ranking evaluated with F1; COCO mAP not computed',
              'Sampled stride evidence cannot certify full 11,993-frame metrics',
              'Swin-B remains a separately certified alternative']}
    write_new(out,result)
    print('[PROBE COMPLETE] sampled frames',len(rows),'oracle',oracle['selected'],
          'selector',prediction.get('pooled_policy'),'BCI',result['BCI_existing_box_overlap'])
    return result


def selftest():
    # Genuine identity swaps cannot pass using equal net TP alone.
    x={'Base':{'tp':1,'fp':1,'fn':1,'lost':0,'rescued':0},
       'ClsOnly':{'tp':1,'fp':0,'fn':1,'lost':1,'rescued':1},
       'BoxOnly':{'tp':2,'fp':0,'fn':0,'lost':0,'rescued':1},
       'Full':{'tp':0,'fp':0,'fn':2,'lost':1,'rescued':0}}
    require(choose_safe(x)=='BoxOnly','Oracle preference / swap regression')
    y=dict(x);y['BoxOnly']=dict(x['Base'])
    require(choose_safe(y)=='Base','Identity swap mistakenly considered safe')
    for tp,fp,fn in ((0,0,1),(1,3,0),(3,0,4),(0,0,0)):
        m=metric(tp,fp,fn);require(0<=m['f1']<=1,'F1 bounds failure')
    print('[PASS] selftest: swap, rescue, fallback, F1 edge cases')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',required=True,choices=('selftest','aggregate','export','probe'))
    p.add_argument('--project',default='.',help='Windows project root for export only')
    p.add_argument('--fit-report',help='existing immutable FIT stride-1 mechanism report')
    p.add_argument('--out',help='new JSON output for aggregate/probe; new directory for export')
    p.add_argument('--evidence',help='new FIT JSONL export directory for probe')
    p.add_argument('--stride',type=int,default=10,help='export sampling stride; default 10, FIT only')
    a=p.parse_args()
    if a.phase=='selftest':selftest();return
    require(a.out,'--out is mandatory')
    if a.phase in ('aggregate','export'):
        require(a.fit_report,'--fit-report mandatory')
    if a.phase=='aggregate':aggregate(a.fit_report,a.out)
    elif a.phase=='export':export_fit(a.project,a.fit_report,a.out,a.stride)
    elif a.phase=='probe':require(a.evidence,'--evidence mandatory');probe(a.evidence,a.out)


if __name__=='__main__':main()
