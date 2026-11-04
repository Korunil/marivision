"""One-shot FIT TRAIN/TUNE preservation-first experiment (NOT production B-FCEF-F).

Frozen R50 and expert native outputs; no CAL/QUAL/Val/Test; opened OCT12 EVAL
videos are never loaded. Train a cost-sensitive TP-protection veto on TRAIN only.
Select the veto cutoff exclusively with video-held-out TRAIN predictions, then
check the selected policy ONCE on the four TUNE videos. Neither split can prove
future/official-test safety. No parameters of upstream networks are updated.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, math
from pathlib import Path
import run15a_oct12_fp_recovery_ablation as a
import numpy as np
import torch

SOURCE_SHA='45ca917eec61a8571660ffbec1d0a6d8063f2c2d1b96fcdeaee4c4a422821b70'
RECEIPT_SHA='4e827507e1f5997bcaf77358c1f3a67f5226ef22a52cdd1e1fb8e3cfad746329'
NAME='oct12_preservation_first_train_oof_tune_only_v2_reference_audit'
ORIGINAL_THRESHOLD=0.75
TP_COST=8.0
CUTS=(0.05,0.10,0.20,0.35,0.50,0.70,0.90)
# Video-level folds fixed BEFORE running, covers all 17 FIT TRAIN videos.
FOLDS=((2,7,16,20,35),(3,8,17,24),(4,13,18,25),(5,14,19,31))


def check(cond,message):
    if not cond:raise AssertionError('[PRESERVATION FIRST HOLD] '+message)


def verify_scope(project):
    project=Path(project)
    check(a.sha(Path(a.__file__))==SOURCE_SHA,'Original executable source drift')
    receipt=project/'bfcef_f_s0'/'run15a_final_candidate_sandbox'/'oct12_recovery_ablation'/'trained_classifier_and_tuning.json'
    check(receipt.is_file() and a.sha(receipt)==RECEIPT_SHA,'Original TRAIN/TUNE receipt drift')
    rec=json.loads(receipt.read_text(encoding='utf8'))
    check(rec['status']=='TRAINED_TUNED' and rec['new_script_sha256']==SOURCE_SHA,'Original receipt incompatible')
    check(rec['train_videos']==list(a.TRAIN) and rec['tune_videos']==list(a.TUNE),'Original partitions drift')
    check(set(a.TRAIN).isdisjoint(a.TUNE) and set(a.TRAIN).isdisjoint(a.EVAL) and set(a.TUNE).isdisjoint(a.EVAL),'Video leakage')
    check(set(v for fold in FOLDS for v in fold)==set(a.TRAIN),'OOF folds do not cover TRAIN')
    check(sum(len(f) for f in FOLDS)==len(a.TRAIN),'Repeated video in TRAIN OOF folds')
    a.verification(project)  # Pins frozen networks and runtime without opening EVAL.
    dest=project/'bfcef_f_s0'/'run15a_final_candidate_sandbox'/'oct12_preservation_first'/'train_oof_tune_receipt.json'
    check(not dest.exists(),'The one-shot TRAIN/TUNE report already exists; refuse overwrite')
    return rec,dest


def to_training_matrix(frames):
    Xs=[];ys=[];gs=[]
    for fr in frames:
        n=len(fr['label'])
        if not n:continue
        Xs.append(fr['expert_x']);ys.append(1.-fr['label']);gs.extend([fr['video']]*n)
    check(bool(Xs),'No training pairs')
    X=np.concatenate(Xs,axis=0);y=np.concatenate(ys).astype(np.float64)
    groups=np.asarray(gs,dtype=np.int64)
    check(y.shape==(len(X),) and (y==0).sum()>=20 and (y==1).sum()>=20,'Insufficient TP/FP guard classes')
    return X,y,groups


def fit_tp_guard(X,y_tp,groups,cost=TP_COST,epochs=400,learning_rate=.075,l2=.025):
    """Cost-sensitive TP-protection *score*, not a calibrated TP probability.

    An error on baseline TP is weighted cost times an error on baseline FP.
    Video groups receive equal total weight before class reweighting.
    """
    X=np.asarray(X,dtype=np.float64);y=np.asarray(y_tp,dtype=np.float64)
    groups=np.asarray(groups)
    check(X.ndim==2 and y.shape==(len(X),),'Guard training shapes')
    check(len(np.unique(groups))>=4,'At least four training videos required')
    check((y==1).sum()>=20 and (y==0).sum()>=20,'Guard classes degenerate')
    mu=X.mean(0);std=X.std(0);std[std<1e-6]=1.
    xn=np.clip((X-mu)/std,-8,8)
    # Exactly one video-normalization, then asymmetric TP risk weighting.
    weights=np.asarray([1./max(1,(groups==g).sum()) for g in groups],dtype=np.float64)
    weights*=np.where(y==1.,float(cost),1.0)
    weights/=weights.sum()
    w=np.zeros(X.shape[1],dtype=np.float64)
    bias=float(np.log((y.mean()+1e-8)/(1.-y.mean()+1e-8)))
    mw=np.zeros_like(w);vw=np.zeros_like(w);mb=0.;vb=0.
    for t in range(1,epochs+1):
        p=a.sigmoid(np.sum(xn*w[None,:],axis=1)+bias)
        g=(p-y)*weights
        gw=np.sum(xn*g[:,None],axis=0)+l2*w;gb=float(g.sum())
        mw=.9*mw+.1*gw;vw=.999*vw+.001*gw*gw
        mb=.9*mb+.1*gb;vb=.999*vb+.001*gb*gb
        w-=learning_rate*(mw/(1.-.9**t))/(np.sqrt(vw/(1.-.999**t))+1e-8)
        bias-=learning_rate*(mb/(1.-.9**t))/(math.sqrt(vb/(1.-.999**t))+1e-8)
    check(np.isfinite(w).all() and math.isfinite(bias),'Guard parameters nonfinite')
    return {'mean':mu.tolist(),'std':std.tolist(),'w':w.tolist(),'bias':bias,
            'feature_dim':int(len(w)),'tp_cost':float(cost),
            'train_n':int(len(y)),'train_tp_count':int(y.sum()),
            'interpretation':'Uncalibrated cost-sensitive TP-veto score, not TP posterior'}


def apply_guard(frame,fp_model,guard_model,cutoff,threshold=ORIGINAL_THRESHOLD):
    """GT-free forward; same native Expert corrections as original ablation.

    Only added operation: reject a chosen candidate if protection score exceeds
    cutoff. No artificial TP lookup, GT, matching or access to other videos.
    """
    out=frame['z'].clone()
    pf=a.predict(fp_model,frame['expert_x'])
    risk=a.predict(guard_model,frame['expert_x'])
    props=[]
    for i,(q,c) in enumerate(frame['pairs']):
        if pf[i] < threshold or risk[i] > cutoff:continue
        dz=(frame['expert_z'][:,q,c]-frame['z'][q,c]).detach().cpu().numpy()
        k=int(np.argmin(dz))
        if dz[k]>=-1e-4:continue
        props.append((float(pf[i]),int(q),int(c),int(k)))
    props.sort(reverse=True)
    used=set()
    for p,q,c,k in props:
        if q in used:continue
        used.add(q)
        out[q]=frame['expert_z'][k,q]
    check(bool(torch.isfinite(out).all()),'Nonfinite guarded logits')
    return out,len(used)


def eval_frames(frames,fp_model,guard_model,old,cutoff):
    from run15a_final_preservation import identity_ledger
    base=collections.Counter();cand=collections.Counter();actions=lost=rescued=0
    pv=collections.defaultdict(lambda:collections.Counter())
    for fr in frames:
        z,acted=apply_guard(fr,fp_model,guard_model,cutoff)
        matched,fp,m=old.capture_match(z,fr['b'],fr['gt_boxes'],fr['gt_labels'])
        led=identity_ledger(fr['orig_matches'],matched)
        nloss=len(led['lost']);nres=len(led['rescued'])
        lost+=nloss;rescued+=nres;actions+=acted
        base.update(fr['counts']);cand.update(m)
        pv[int(fr['video'])].update({'actions':acted,'lost':nloss,'rescued':nres,
                                      'fp_reduction':fr['counts']['fp']-m['fp'],
                                      'tp_delta':m['tp']-fr['counts']['tp']})
    def summary(t):
        tp,fp,fn=(int(t.get(key,0)) for key in ('tp','fp','fn'))
        return {'tp':tp,'fp':fp,'fn':fn,'f1':(2.*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.}
    b=summary(base);c=summary(cand)
    videos={str(v):dict(rec) for v,rec in sorted(pv.items())}
    active=sum(int(rec.get('actions',0)>0) for rec in videos.values())
    max_share=max((rec.get('actions',0) for rec in videos.values()),default=0)/max(1,actions)
    return {'base':b,'guarded':c,'delta_f1':c['f1']-b['f1'],
            'fp_reduction':b['fp']-c['fp'],'lost':int(lost),'rescued':int(rescued),
            'actions':int(actions),'active_videos':int(active),
            'max_video_action_fraction':float(max_share),'per_video':videos}


def merge_fold_reports(reports):
    # Pool absolute TP/FP/FN first; do NOT average per-fold F1.
    b=collections.Counter();c=collections.Counter();lost=rescued=actions=0;pv={}
    for r in reports:
        b.update({k:r['base'][k] for k in ('tp','fp','fn')})
        c.update({k:r['guarded'][k] for k in ('tp','fp','fn')})
        lost+=r['lost'];rescued+=r['rescued'];actions+=r['actions']
        for v,vr in r['per_video'].items():
            check(v not in pv,'Duplicate OOF video')
            pv[v]=vr
    def score(rec):
        tp,fp,fn=(rec[k] for k in ('tp','fp','fn'))
        return {'tp':int(tp),'fp':int(fp),'fn':int(fn),
                'f1':float(2*tp/(2*tp+fp+fn)) if (2*tp+fp+fn) else 0.}
    active=sum(int(v.get('actions',0)>0) for v in pv.values())
    mx=max((v.get('actions',0) for v in pv.values()),default=0)/max(1,actions)
    sb=score(b);sc=score(c)
    return {'base':sb,'guarded':sc,'delta_f1':sc['f1']-sb['f1'],
            'fp_reduction':sb['fp']-sc['fp'],
            'lost':int(lost),'rescued':int(rescued),'actions':int(actions),
            'active_videos':int(active),'max_video_action_fraction':float(mx),
            'per_video':dict(sorted(pv.items(),key=lambda kv:int(kv[0])))}


def oof_eligible(m):
    """Predeclared strictly preservation-first admission to TUNE."""
    return (m['lost']==0 and m['delta_f1']>0 and m['fp_reduction']>=5
            and m['actions']>=5 and m['active_videos']>=3
            and m['max_video_action_fraction']<=.80
            and all(v.get('lost',0)==0 for v in m['per_video'].values()))


def tune_passes(m):
    return (m['lost']==0 and m['actions']>=2 and m['delta_f1']>0
            and m['fp_reduction']>0 and m['rescued']>=0)


def preservation_evidence(no_veto, guarded):
    """Describe what these *same-frame* outcomes can and cannot establish.

    A zero-loss guarded result is NOT evidence of risk prevention if its
    no-veto reference already has zero loss. The comparison is descriptive,
    in-sample for upstream networks, and is NOT a formal causal guarantee.
    """
    if guarded['lost'] > 0:
        return 'GUARDED_LOSSES_OBSERVED'
    if no_veto['lost'] == 0:
        return 'PRESERVATION_UNTESTED_REFERENCE_LOSS_FREE'
    if guarded['lost'] < no_veto['lost']:
        return 'OBSERVED_OOF_LOSS_REDUCTION_NOT_GENERALIZATION_PROOF'
    return 'NO_OBSERVED_PRESERVATION_IMPROVEMENT'


def outcome_status(no_veto_oof, guarded_oof, guarded_tune):
    """Descriptive development status; intentionally never labels a qualification."""
    if not tune_passes(guarded_tune):
        return 'HOLD_TUNE_PRESERVATION_OR_UTILITY'
    if guarded_oof['lost'] != 0:
        return 'HOLD_OOF_GUARD_LOSSES'
    if no_veto_oof['lost'] == 0:
        return 'DEV_UTILITY_ONLY_PRESERVATION_UNTESTED'
    return 'DEV_OOF_LOSS_REDUCTION_OBSERVED_TUNE_PASS_NOT_QUALIFIED'


def run(project):
    rec,dest=verify_scope(project)
    rt=a.setup_native(project)
    old=rt[0]
    train=list(a.native_frames(rt,a.TRAIN))
    check(set(int(f['video']) for f in train)==set(a.TRAIN),'Training frames missing')
    folds=[]
    for i,valid_videos in enumerate(FOLDS):
        hold=set(valid_videos)
        ftrain=[fr for fr in train if fr['video'] not in hold]
        fvalid=[fr for fr in train if fr['video'] in hold]
        check(set(fr['video'] for fr in ftrain).isdisjoint(hold),'OOF contamination')
        X,YTP,G=to_training_matrix(ftrain)
        guard=fit_tp_guard(X,YTP,G)
        # Refit FP model EXCLUDING held-out videos: otherwise this would be
        # an in-sample policy, invalidating the video-level safety check.
        fp=a.fit_linear(X,1.-YTP,G)
        folds.append((i,valid_videos,fvalid,fp,guard))
        print(f'[OOF FIT] fold={i+1} train_videos={len(set(G))} held_out={valid_videos} positives_tp={int(YTP.sum())}',flush=True)
    # Cutoff 1.0 rejects nothing because the sigmoid risk lies in [0,1].
    # Use *the same fold-specific FP and guard models* as all selectable rows.
    # This reference MUST NOT compete for policy selection.
    no_veto_oof=merge_fold_reports([
        eval_frames(fvalid,fp,guard,old,1.0) for _,_,fvalid,fp,guard in folds
    ])
    print(f"[OOF NONSELECTABLE NO-VETO] actions={no_veto_oof['actions']} lost={no_veto_oof['lost']} FP_reduction={no_veto_oof['fp_reduction']} delta_F1={no_veto_oof['delta_f1']:+.6f}",flush=True)
    ledger=[]
    for cut in CUTS:
        reports=[eval_frames(fvalid,fp,guard,old,cut) for _,_,fvalid,fp,guard in folds]
        pooled=merge_fold_reports(reports)
        evidence=preservation_evidence(no_veto_oof,pooled)
        ledger.append({'cutoff':cut,'eligible':oof_eligible(pooled),
                       'nonselectable':False,'preservation_evidence':evidence,
                       'loss_reduction_vs_no_veto':int(no_veto_oof['lost']-pooled['lost']),
                       'oof':pooled})
        print(f"[OOF] cutoff={cut:.2f} actions={pooled['actions']} active_videos={pooled['active_videos']} lost={pooled['lost']} FP_reduction={pooled['fp_reduction']} delta_F1={pooled['delta_f1']:+.6f} eligible={oof_eligible(pooled)} evidence={evidence}",flush=True)
    qualified=[r for r in ledger if r['eligible']]
    selected=max(qualified,key=lambda r:(r['oof']['delta_f1'],-r['oof']['actions'],-r['cutoff'])) if qualified else None
    report={'schema':NAME,'status':'HOLD_TRAIN_OOF_NO_SAFE_IMPROVEMENT',
            'base_source_sha256':SOURCE_SHA,'base_train_receipt_sha256':RECEIPT_SHA,
            'this_source_sha256':a.sha(Path(__file__)),'train_videos':list(a.TRAIN),
            'tune_videos':list(a.TUNE),'excluded_previous_eval_videos':list(a.EVAL),
            'original_expert_policy_threshold':ORIGINAL_THRESHOLD,
            'expert_residual_amplitude':a.ALPHA,'guard_cost_baseline_tp':TP_COST,
            'cutoff_grid':list(CUTS),'train_oof_folds':[list(f) for f in FOLDS],
            'train_oof_no_veto_nonselectable':{
                'cutoff':1.0,'nonselectable':True,
                'scope':'Same OOF folds, fold-refitted FP and guard; not original globally trained Expert FP model',
                'oof':no_veto_oof},
            'train_oof_grid':ledger,'selected_guard_cutoff':selected['cutoff'] if selected else None,
            'selected_oof_preservation_evidence':(preservation_evidence(no_veto_oof,selected['oof']) if selected else None),
            'limitations':['All upstream models trained on official Train, including FIT','OOF only guards against this correction fitting on its held-out TRAIN video; no external generalization guarantee',
                           'The original FIT EVAL has been opened and is never revisited',
                           'TUNE previously selected the original FP-policy cutoff; its check is NOT independent of the entire pipeline','Guard scores are cost-weighted, NOT calibrated TP probabilities',
                           'Video correlation; do not infer statistical significance from action counts',
                           'Even an observed OOF reduction from no-veto losses does not prove downstream/general safety',
                           'The TP-veto target uses the baseline pair label, not the actual multi-class identity loss mechanism',
                           'The guard shares features/model family with the FP classifier and is not independent TP evidence',
                           'Guard cost/grid were designed after opened EVAL was observed; no EVAL retuning or fresh qualification',
                           'This policy is NOT the complete production B-FCEF-F']}
    if selected is not None:
        # TUNE is loaded once, and only the OOF-selected cutoff is evaluated.
        X,YTP,G=to_training_matrix(train)
        final_guard=fit_tp_guard(X,YTP,G)
        # Freeze ORIGINAL source-pinned Expert model; no EVAL-derived refit.
        fp_model=rec['models']['Expert']
        tune=list(a.native_frames(rt,a.TUNE))
        check(set(int(f['video']) for f in tune)==set(a.TUNE),'TUNE missing videos')
        original_tune=a.measure(tune,{'Expert':fp_model},'Expert',ORIGINAL_THRESHOLD,1.0,old,include_per_video=True)
        expected=rec['choices']['Expert']['tune']
        for k in ('lost','rescued','actions'):
            check(original_tune[k]==expected[k],'Original Expert TUNE parity drift: '+k)
        for k in ('tp','fp','fn'):
            check(original_tune['Candidate'][k]==expected['Candidate'][k],'Original Expert TUNE metric drift: '+k)
        # Explicit no-veto reference for TUNE. Check the direct native replay
        # against the original frozen-policy receipt before interpreting guard.
        tune_no_veto=eval_frames(tune,fp_model,final_guard,old,1.0)
        for k in ('lost','rescued','actions'):
            check(tune_no_veto[k]==original_tune[k],
                  'No-veto TUNE parity drift: '+k)
        for k in ('tp','fp','fn'):
            check(tune_no_veto['guarded'][k]==original_tune['Candidate'][k],
                  'No-veto TUNE matching/count drift: '+k)
        result=eval_frames(tune,fp_model,final_guard,old,selected['cutoff'])
        report['unchanged_original_expert_tune']={k:original_tune[k] for k in ('Base','Candidate','lost','rescued','actions','delta_f1','per_video')}
        report['tune_no_veto_reference']={
            'cutoff':1.0,'nonselectable':True,
            'scope':'Original frozen Expert policy; source/receipt parity verified',
            'result':tune_no_veto}
        report['tune_check']=result
        report['tune_preservation_evidence']=preservation_evidence(tune_no_veto,result)
        report['status']=outcome_status(no_veto_oof,selected['oof'],result)
        # An experimental guard may be retained for development, never as a
        # qualified preservation guarantee. Status distinguishes untested safety.
        if tune_passes(result):
            report['final_guard_model']=final_guard
            report['guard_model_status']='EXPERIMENTAL_TRAIN_TUNE_ONLY_NOT_QUALIFIED'
        print(f"[TUNE] no_veto_lost={tune_no_veto['lost']} guarded_lost={result['lost']} actions={result['actions']} FP_reduction={result['fp_reduction']} delta_F1={result['delta_f1']:+.6f} evidence={report['tune_preservation_evidence']} status={report['status']}",flush=True)
    else:
        print('[HOLD] No risk cutoff passed TRAIN video-held-out preservation and utility gates. TUNE not loaded.',flush=True)
    a.write_new(dest,report)
    print('[PRESERVATION REPORT]',dest,'status=',report['status'],flush=True)
    return report


def selftest():
    # 1. Strict folds, 2. cost asymmetry 3. single-query restriction 4. no GT at inference
    assert set(v for f in FOLDS for v in f)==set(a.TRAIN)
    assert sum(map(len,FOLDS))==len(a.TRAIN)
    rng=np.random.default_rng(12)
    x=rng.normal(size=(240,18));y=np.tile([0.,1.],120);v=np.repeat(np.arange(6),40)
    cost=fit_tp_guard(x,y,v,epochs=30);base=fit_tp_guard(x,y,v,cost=1.,epochs=30)
    assert cost['tp_cost']==8. and base['tp_cost']==1.
    assert len(cost['w'])==18
    f={'z':torch.zeros((3,7)), 'expert_z':torch.zeros((4,3,7)),
       'pairs':[(0,2)],'expert_x':np.zeros((1,18))}
    f['expert_z'][0,0,2]=-1.
    always={'mean':[0.]*18,'std':[1.]*18,'w':[0.]*18,'bias':20.}
    risklow={'mean':[0.]*18,'std':[1.]*18,'w':[0.]*18,'bias':-20.}
    riskhigh={'mean':[0.]*18,'std':[1.]*18,'w':[0.]*18,'bias':20.}
    o,n=apply_guard(f,always,risklow,.50);assert n==1 and o[0,2]<0
    q,k=apply_guard(f,always,riskhigh,.50);assert k==0 and torch.equal(q,f['z'])
    assert o[1:].equal(f['z'][1:])
    assert not oof_eligible({'lost':1,'delta_f1':.05,'fp_reduction':10,'actions':20,'active_videos':5,'max_video_action_fraction':.1,'per_video':{}})
    assert not tune_passes({'lost':1,'actions':10,'delta_f1':.2,'fp_reduction':20,'rescued':0})
    assert preservation_evidence({'lost':0},{'lost':0})=='PRESERVATION_UNTESTED_REFERENCE_LOSS_FREE'
    assert preservation_evidence({'lost':2},{'lost':0})=='OBSERVED_OOF_LOSS_REDUCTION_NOT_GENERALIZATION_PROOF'
    assert outcome_status({'lost':0},{'lost':0},
            {'lost':0,'actions':2,'delta_f1':.1,'fp_reduction':2,'rescued':0})=='DEV_UTILITY_ONLY_PRESERVATION_UNTESTED'
    print('[PASS] Preservation-first selftest: folds, TP-cost, GT-free veto, fallback, reference evidence, safety gates')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=('selftest','verify','train-tune'),required=True)
    args=parser.parse_args()
    if args.phase=='selftest':selftest()
    elif args.phase=='verify':
        rec,dest=verify_scope(a.BASE)
        print('[PASS] Source + TRAIN receipt + hardcoded TRAIN/TUNE/embargo verified; no native frames opened')
        print('[NEW OUTPUT]',dest)
    else:run(a.BASE)

if __name__=='__main__':main()
