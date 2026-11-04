"""OCT13: TRAIN-only, video-held-out incremental frozen-expert information audit.

Models: A = frozen R50 confidence; B = 14 R50-derived Simple features;
C = same 14 + four frozen S/L/C/T logit-response features. B/C use the
identical deterministic logistic fit; no predictor is deployed, no AP claim.

GT is used ONLY for labels and offline evaluation. The previously opened EVAL,
TUNE, CAL, QUAL, official Val and Test data are never loaded. The historical
OCT12 forensic receipt is *verified*, not reopened for selection.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, math
from pathlib import Path
import numpy as np
import torch
import run15a_oct12_fp_recovery_ablation as a
import run15a_oct12_preservation_first as p
import run15a_oct12_harm_forensic as f

SCHEMA = 'oct13_expert_incremental_information_train_oof_v1'
SOURCE_A = '45ca917eec61a8571660ffbec1d0a6d8063f2c2d1b96fcdeaee4c4a422821b70'
SOURCE_P = '5b339d611d8653c86def5c9f6c3ac1c955dda0480f79044a71e5d1d628a3a38c'
SOURCE_F = 'cefa4a80e8a68d2b15e8a73ac848138c01b056e4cb5f6f16d3a4160770e80e66'
RECEIPT_OOF = 'a4d191c39c0c6a04224f1d78998dae10d6cddb5d842db1df63e2827faf1d5cf1'
RECEIPT_FORENSIC = '793cd7d4c1c6cd4cec35edcf9f40420cba6d0a898de5adefbe34c91363927fbe'
FOLDS = ((2,7,16,20,35),(3,8,17,24),(4,13,18,25),(5,14,19,31))
BOOT_REPLICATES = 3000
SEED = 20261013
EDGE_LOW, EDGE_HIGH = 0.20, 0.27  # explicitly post-hoc / exploratory
CUTOFFS = (0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75, 0.90)  # descriptive null ONLY


def check(cond, message):
    if not cond:
        raise AssertionError('[OCT13 INFORMATION HOLD] ' + str(message))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as r:
        for b in iter(lambda:r.read(1 << 20),b''):
            h.update(b)
    return h.hexdigest()


def scope(project, require_available=True):
    project=Path(project)
    for script,digest in ((a.__file__,SOURCE_A),(p.__file__,SOURCE_P),(f.__file__,SOURCE_F)):
        check(sha(script)==digest,'Immutable script hash drift: '+str(script))
    check(tuple(a.TRAIN)==(2,3,4,5,7,8,13,14,16,17,18,19,20,24,25,31,35),
          'TRAIN partition drift')
    check(tuple(map(tuple,p.FOLDS))==FOLDS,'OOF folds drift')
    check(set(a.TRAIN)==set().union(*(set(row) for row in FOLDS)), 'OOF TRAIN coverage')
    check(sum(map(len,FOLDS))==len(a.TRAIN),'Repeated OOF TRAIN video')
    check(all(set(a.TRAIN).isdisjoint(s) for s in (a.TUNE,a.EVAL)),'TRAIN embargo violation')
    prior=project/'bfcef_f_s0'/'run15a_final_candidate_sandbox'/'oct12_preservation_first'
    old=prior/'train_oof_tune_receipt.json';forensic=prior/'oct12_harm_forensic_train_oof.json'
    check(old.is_file() and sha(old)==RECEIPT_OOF,'Historical OOF receipt drift')
    check(forensic.is_file() and sha(forensic)==RECEIPT_FORENSIC,'Historical forensic receipt drift')
    x=json.loads(forensic.read_text(encoding='utf8'))
    check(x.get('status')=='STOP_DIRECT_HARM_DOMINANT','Unexpected historical forensic status')
    check(x['gate0']['status']=='EXACT_PARITY' and (x['gate0']['actions'],x['gate0']['fp_reduction'],x['gate0']['lost_gt_identities'])==(701,378,13),
          'Forensic parity certification drift')
    check(x['frozen_contract']['train_videos']==list(a.TRAIN) and x['frozen_contract']['folds']==[list(i) for i in FOLDS],
          'Forensic partition mismatch')
    a.verification(project)  # Frozen network/checkpoint pins; no frame extraction.
    dest=project/'bfcef_f_s0'/'run15a_final_candidate_sandbox'/'oct13_information_test'/'expert_information_train_oof.json'
    if require_available:
        check(not dest.exists() and not Path(str(dest)+'.tmp').exists(),'Result exists; never overwrite')
    return dest,x


def roc_auc(y,score):
    """Tie-correct, deterministic rank AUC; no sklearn or BLAS required."""
    y=np.asarray(y,dtype=np.int8); s=np.asarray(score,dtype=np.float64)
    check(y.ndim==s.ndim==1 and len(y)==len(s),'AUC shapes')
    check(np.isfinite(s).all() and np.isin(y,(0,1)).all(),'Invalid score/label')
    pos=int(y.sum()); neg=len(y)-pos
    if not pos or not neg:
        return None
    order=np.argsort(s,kind='mergesort')
    ys=y[order]; ss=s[order]
    accum_neg=0;wins=0.
    boundaries=np.flatnonzero(np.r_[ss[1:]!=ss[:-1],True])+1
    lo=0
    for hi in boundaries:
        npos=int(ys[lo:hi].sum());nneg=(hi-lo)-npos
        wins+=npos*(accum_neg+0.5*nneg)
        accum_neg+=nneg;lo=int(hi)
    return float(wins/(pos*neg))


def counts_rows(rows):
    if not rows:return {'n':0,'TP':0,'FP':0}
    y=np.concatenate([r['tp_y'] for r in rows]);return {'n':len(y),'TP':int(y.sum()),'FP':int(len(y)-y.sum())}


def merged(rows):
    check(bool(rows),'No scored hypotheses')
    return (np.concatenate([r['tp_y'] for r in rows]),
            *(np.concatenate([r['scores'][name] for r in rows]) for name in ('A_R50','B_Simple','C_Expert')))


def metrics(rows):
    y,sa,sb,sc=merged(rows)
    A=roc_auc(y,sa);B=roc_auc(y,sb);C=roc_auc(y,sc)
    check(all(v is not None for v in (A,B,C)),'Class-degenerate pooled AUC')
    return {'A_R50':A,'B_Simple':B,'C_Expert':C,'delta_C_minus_B':C-B,
            'delta_B_minus_A':B-A,**counts_rows(rows)}


def paired_video_bootstrap(rows,reps=BOOT_REPLICATES,seed=SEED):
    """Resample whole video blocks, paired scorers; fit models NOT retrained."""
    byvid={r['video']:r for r in rows}
    videos=sorted(byvid)
    check(len(videos)>=10,'Insufficient contributing video clusters')
    rng=np.random.default_rng(seed);draws=[];skipped=0
    for _ in range(reps):
        vids=rng.choice(videos,size=len(videos),replace=True)
        select=[byvid[int(v)] for v in vids]
        try: draw=metrics(select)
        except AssertionError:skipped+=1;continue
        draws.append(draw['delta_C_minus_B'])
    check(len(draws)>=int(reps*.95),'Too many degenerate video bootstrap draws')
    lo,hi=np.quantile(draws,[.025,.975]);
    return {'replicates_requested':reps,'replicates_valid':len(draws),'skipped':skipped,
            'seed':seed,'unit':'whole TRAIN video; no model refits; conditional on OOF fits',
            'delta_C_minus_B_ci95':[float(lo),float(hi)],
            'bootstrap_mean':float(np.mean(draws))}


def decide(paired,bootstrap):
    d=paired['delta_C_minus_B'];lo,hi=bootstrap['delta_C_minus_B_ci95']
    if lo>0.02:return 'INCREMENTAL_EXPERT_INFORMATION_PRESENT_TRAIN_ONLY'
    if d<0.01 and hi<0.03:return 'NO_PERSUASIVE_INCREMENTAL_EXPERT_INFORMATION'
    return 'INCONCLUSIVE_DO_NOT_CONTINUE_SELECTOR_TRAINING'


def scores_by_video(contexts):
    # `contexts` uses native frozen outputs and four genuine held-out folds.
    video_fr=collections.defaultdict(list)
    for (vid,j),(fr,fp,guard,fold) in sorted(contexts.items()):
        video_fr[vid].append((fr,fp,fold))
    check(set(video_fr)==set(a.TRAIN),'OOF frame coverage drift')
    predictions=[];fold_fit=[]
    for fold, held in enumerate(FOLDS,1):
        ftrain=[fr for (vid,j),(fr,fp,guard,k) in contexts.items() if k!=fold]
        check(set(fr['video'] for fr in ftrain)==set(a.TRAIN)-set(held),'Fold TRAIN mismatch')
        X,YTP,G=p.to_training_matrix(ftrain)  # 18 expert features; train labels used here only
        check(X.shape[1]==18,'C feature dimensionality drift')
        simple=a.fit_linear(X[:,:14],1.-YTP,G)
        ftest=[(fr,fp) for (vid,j),(fr,fp,guard,k) in sorted(contexts.items()) if k==fold]
        check(set(fr['video'] for fr,fp in ftest)==set(held),'Fold held-out videos mismatch')
        fold_fit.append({'fold':fold,'train_videos':sorted(set(G.tolist())),
                         'held_out_videos':list(held),'fit_n':len(X),
                         'B_train_bce':float(simple['train_bce']),
                         'C_train_bce':float(fp['train_bce'])})
        for fr,fp in ftest:
            bf,ef=fr['base_x'],fr['expert_x'];n=len(fr['label'])
            check(np.array_equal(bf,ef[:,:14]),'Base/Expert feature alignment drift')
            check(n==len(bf)==len(ef)==len(fr['pairs']),'Pairs/labels/features misaligned')
            if not n:continue
            sA=bf[:,1].astype(np.float64,copy=True)
            sB=1.-a.predict(simple,bf)
            sC=1.-a.predict(fp,ef)
            check(all(len(v)==n and np.isfinite(v).all() for v in (sA,sB,sC)),'OOF predictions nonfinite')
            predictions.append({'video':int(fr['video']),'frame':int(fr['frame']),
                    'image_id':int(fr['image_id']),'pairs':[(int(q),int(c)) for q,c in fr['pairs']],
                    'fold':fold,'tp_y':(1.-fr['label']).astype(np.int8),
                    'r50_score':sA, 'scores':{'A_R50':sA,'B_Simple':sB,'C_Expert':sC}})
    check(len(predictions)>0,'Empty OOF predictions')
    check(len({(x['video'],x['frame']) for x in predictions})==len(predictions),'OOF duplicate frame')
    return predictions,fold_fit


def combine_per_video(frames):
    block={}
    for r in frames:
        vid=r['video']
        if vid not in block:
            block[vid]={'video':vid,'tp_y':[],'r50_score':[],
                        'scores':{k:[] for k in ('A_R50','B_Simple','C_Expert')}}
        x=block[vid];x['tp_y'].append(r['tp_y']);x['r50_score'].append(r['r50_score'])
        for k in x['scores']: x['scores'][k].append(r['scores'][k])
    for vid,r in block.items():
        r['tp_y']=np.concatenate(r['tp_y']);r['r50_score']=np.concatenate(r['r50_score'])
        for k in r['scores']:r['scores'][k]=np.concatenate(r['scores'][k])
    return [block[v] for v in sorted(block)]


def secondary_video_metrics(rows):
    results=[]
    for r in rows:
        auc=metrics([r]) if 0<r['tp_y'].sum()<len(r['tp_y']) else None
        results.append({'video':r['video'],**counts_rows([r]),
                        'auc':auc,'single_class_undefined':auc is None})
    usable=[r['auc'] for r in results if r['auc'] is not None]
    leave=[]
    for omit in rows:
        m=metrics([r for r in rows if r['video']!=omit['video']])
        leave.append({'omitted_video':omit['video'], 'delta_C_minus_B':m['delta_C_minus_B'],
                      'n':m['n']})
    return {'per_video':results,'leave_one_video_out':leave,
            'valid_per_video_auc_count':len(usable),
            'equal_video_mean_delta_C_minus_B':(float(np.mean([x['delta_C_minus_B'] for x in usable])) if usable else None),
            'any_loo_sign_reversal':any(x['delta_C_minus_B']<0 for x in leave)}


def edge_band(rows):
    filtered=[]
    for r in rows:
        mask=(r['r50_score']>=EDGE_LOW)&(r['r50_score']<=EDGE_HIGH)
        if int(mask.sum())==0:continue
        filtered.append({**r,'tp_y':r['tp_y'][mask],
            'r50_score':r['r50_score'][mask],
            'scores':{k:v[mask] for k,v in r['scores'].items()}})
    if not filtered:return {'status':'NO_EDGE_HYPOTHESES'}
    y=np.concatenate([r['tp_y'] for r in filtered]);
    if y.sum()==0 or y.sum()==len(y):return {'status':'CLASS_DEGENERATE',**counts_rows(filtered)}
    return {'status':'EXPLORATORY_POST_HOC_NOT_FOR_SELECTION',
            'r50_score_band':[EDGE_LOW,EDGE_HIGH],**metrics(filtered)}


def r50_cutoff_null(contexts,old):
    # Descriptive, fixed grid, native matching and GT assignment, NO cutoff selected.
    totals={str(t):collections.Counter() for t in CUTOFFS}
    for (vid,j),(fr,fp,guard,fold) in sorted(contexts.items()):
        z=fr['z'];b=fr['b']
        for t in CUTOFFS:
            zz=z.clone()
            zz[z.sigmoid()<t]=-40.0
            _,_,counts=old.capture_match(zz,b,fr['gt_boxes'],fr['gt_labels'])
            totals[str(t)].update(counts)
        for k in ('tp','fp','fn'):
            check(totals[str(CUTOFFS[0])][k]>=0,'Native null invalid')
    expected={'tp':754,'fp':1013,'fn':271}
    for k,v in expected.items():
        check(totals[str(CUTOFFS[0])][k]==v,'Native 0.20 baseline cutoff parity: '+k)
    report=[]
    for t in CUTOFFS:
        c=totals[str(t)];tp,fp,fn=[int(c[k]) for k in ('tp','fp','fn')]
        f1=2*tp/max(1,(2*tp+fp+fn))
        report.append({'threshold':float(t),'tp':tp,'fp':fp,'fn':fn,'f1':float(f1)})
    return {'status':'DESCRIPTIVE_NONSELECTABLE_CUTOFF_NULL',
            'note':'Every cutoff evaluated on same TRAIN frames with native matching; no selection or AP implication',
            'grid':report}


def run(project):
    dest,history=scope(project)
    print('[PASS] OCT13 source/receipt pins, TRAIN folds, opened-EVAL/TUNE embargo',flush=True)
    rt=a.setup_native(project);old=rt[0]
    contexts=f.extract_oof(rt)
    parity_obs=f.collect_parity(contexts,old)
    certificate=f.parity(json.loads((Path(project)/'bfcef_f_s0'/'run15a_final_candidate_sandbox'/'oct12_preservation_first'/'train_oof_tune_receipt.json').read_text('utf8')),
                         parity_obs['observed'])
    check(certificate['status']=='EXACT_PARITY','Native forensic Gate 0 parity')
    print('[GATE 0 PASS] 701 actions / 378 FP removed / 13 GT lost; per-video exact',flush=True)
    rows,fold_fit=scores_by_video(contexts)
    videos=combine_per_video(rows)
    pooled=metrics(videos)
    check((pooled['n'],pooled['TP'],pooled['FP'])==(1767,754,1013),'Locked hypothesis universe 1767/754/1013')
    print('[AUC] R50={:.6f} Simple={:.6f} Expert={:.6f} delta(C-B)={:+.6f}'.format(pooled['A_R50'],pooled['B_Simple'],pooled['C_Expert'],pooled['delta_C_minus_B']),flush=True)
    boot=paired_video_bootstrap(videos)
    secondary=secondary_video_metrics(videos)
    edge=edge_band(videos)
    recal=r50_cutoff_null(contexts,old)
    status=decide(pooled,boot)
    paired=[]
    for frame in rows:
        for i,(q,c) in enumerate(frame['pairs']):
            paired.append({'video':frame['video'],'frame':frame['frame'],'image_id':frame['image_id'],
                           'fold':frame['fold'],'q':q,'c':c,'is_tp':int(frame['tp_y'][i]),
                           'r50_score':float(frame['scores']['A_R50'][i]),
                           'simple_tp_score':float(frame['scores']['B_Simple'][i]),
                           'expert_tp_score':float(frame['scores']['C_Expert'][i])})
    check(len(paired)==1767,'Paired hypothesis serialization alignment')
    report={'schema':SCHEMA,'status':status,
            'this_source_sha256':sha(__file__),'pins':{'ablation':SOURCE_A,'preservation':SOURCE_P,
                'forensic_script':SOURCE_F,'OOF_receipt':RECEIPT_OOF,'forensic_receipt':RECEIPT_FORENSIC},
            'gate0':certificate,'train_videos':list(a.TRAIN),'folds':[list(x) for x in FOLDS],
            'candidate_universe':'R50 Top100 at >=0.20, eligible queries only; native GT matching',
            'labels':'TP=1 FP=0; FP-classifier scores inverted for TP AUC',
            'fold_fit':fold_fit,'pooled_oof':pooled,'cluster_bootstrap':boot,
            'paired_oof_hypotheses':paired,
            'video_robustness':secondary,'edge_band_exploratory':edge,
            'r50_global_cutoff_null':recal,
            'decision_rules':{'positive':'lower bootstrap 95% bound delta(C-B)>0.02',
                'negative':'point delta<0.01 AND upper bootstrap 95% bound<0.03',
                'otherwise':'INCONCLUSIVE, no additional selector training'},
            'limitations':['All upstream detectors saw official TRAIN videos; positive result is not external generalization',
                'Paired video bootstrap conditions on fixed four-fold models; does not refit in bootstrap',
                'The four expert features are tested only through one specified linear classifier family',
                'No AP metric, threshold intervention or S/L/C/T specialist-level contribution is estimated',
                '0.20-0.27 band was discovered after OCT12 harms: exploratory only',
                'TRAIN has strong video imbalance, notably video 18; per-video effects may be undefined with one class',
                'TUNE, opened EVAL, CAL, QUAL, validation and official Test deliberately excluded',
                'Baseline cutoff grid is descriptive; must not select a new threshold on TRAIN retrospectively']}
    a.write_new(dest,report)
    print('[OCT13 REPORT]',dest,'status',status,flush=True)
    return report


def selftest():
    check(set().union(*(set(row) for row in FOLDS))==set(a.TRAIN),'Fold coverage')
    check(len(set(a.TRAIN))==17 and set(a.TRAIN).isdisjoint(a.EVAL),'TRAIN partition')
    check(abs(roc_auc([1,0,1,0],[.9,.1,.9,.1])-1.)<1e-12,'Perfect AUC')
    check(abs(roc_auc([1,0,1,0],[.5,.5,.5,.5])-.5)<1e-12,'Tied AUC')
    check(roc_auc([0,0],[.2,.3]) is None,'Single class AUC')
    print('[PASS] OCT13 TRAIN split, paired video-cluster AUC and no-fit audit contracts')


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument('--phase',required=True,choices=('selftest','verify','audit'))
    cli.add_argument('--project',default=str(a.BASE))
    args=cli.parse_args()
    if args.phase=='selftest':selftest()
    elif args.phase=='verify':
        dest,_=scope(args.project)
        print('[PASS] OCT13 frozen sources/receipts, TRAIN split, embargo; no native frames opened')
        print('[NEW OUTPUT]',dest)
    else: run(args.project)

if __name__=='__main__':main()
