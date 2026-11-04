"""FIT-only, non-production classification recovery ABLATION, no upstream training.

Compare a calibrated base-score suppression model with a policy using genuine
frozen expert native classification responses at alpha=0.25. No GT is allowed
in features or forward decisions. Video groups are hardcoded to prevent leakage.
This is NOT a qualified B-FCEF-F or a replacement for its trained U/H router.
"""
from __future__ import annotations
import argparse, collections, contextlib, copy, hashlib, io, json, math, os
from pathlib import Path
from PIL import Image  # Windows/PyTorch legacy import order
for _k,_v in {'R0_MSDA_BACKEND':'pytorch','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}.items():
    os.environ.setdefault(_k,_v)
import numpy as np
import torch

BASE=Path(r'D:\MARIVISION\H0_workbench\eval_tools')
TRAIN=(2,3,4,5,7,8,13,14,16,17,18,19,20,24,25,31,35)
TUNE=(6,29,36,37)
EVAL=(38,40,41,42,43,44,45,47)
FIT=TRAIN+TUNE+EVAL
EXPERTS=('S','L','C','T')
PINS={
 'run15a_final_mechanism.py':'fa35e32419c37af92b323c48273ff2154e7913358c5f0567b6498f811bad7a3b',
 'run15a_final_risk.py':'6f72ff8a995e181f84d1ba55ee59994930834e36d4f19ee4684750333c48e2d3',
 'run15a_final_supervision.py':'fb854a002c4675017cb548211e8167ad99a264946869924bba5a5bf80b70a3d6',
 'run15a_final_windows.py':'cbf847e2ef594a51b6bf6e70883fb105da0ac17aa9b61d883546f0969891667f',
 'run15a_final_preservation.py':'325e27481a7cf40b8b12b77bfdfb8187914124f0c595d11eb81a48bfa52b3ae3',
 'run15a_final_pilot_fit.py':'edd5086f147dd23d5c6b0f2bfe012bd1b39559676ce35046c287095df0a74182',
}
PILOT_SHA='8ffe141883aa9d7768f0e5bdbcd90a00cc2bf5386e614f3178ebcbab6fa411e9'
SCHEMA='oct12_class_fp_recovery_ablation_v2_reach_audited'
GRID_THRESH=(0.60,0.75,0.85,0.92,0.97)
GRID_GAIN=(0.5,2.0,8.0)  # Same 3 gain choices; extend Simple reach for high-confidence FPs.
STRIDE=10
NFRAMES=16
ALPHA=0.25


def check(x,msg):
    if not x:raise AssertionError('[OCT12 ABLATION HOLD] '+msg)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda:f.read(1<<20),b''):h.update(part)
    return h.hexdigest()


def write_new(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    check(not p.exists(),'Refuse overwrite: '+str(p))
    temp=Path(str(p)+'.tmp');check(not temp.exists(),'Stale tmp: '+str(temp))
    temp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf8');temp.rename(p)


def verification(project):
    project=Path(project)
    from run15a_final_windows import FIT as _FIT,load_runtime
    check(set(FIT)==set(_FIT),'Development FIT universe drift')
    check(len(FIT)==29 and len(set(FIT))==29 and all(set(a).isdisjoint(b) for a,b in ((TRAIN,TUNE),(TRAIN,EVAL),(TUNE,EVAL))),
          'Video-group leakage')
    for name,digest in PINS.items():check(sha(project/name)==digest,'Source drift: '+name)
    cp=project/'bfcef_f_s0'/'run15a_final_candidate_sandbox'/'pilot_fit'/'pilot_videos_2_3_frames16.pth'
    check(cp.exists() and sha(cp)==PILOT_SHA,'Frozen pilot checkpoint drift')
    return cp


def setup_native(project):
    from run15a_final_windows import load_runtime
    cp=verification(project)
    old,r9,r8,d,s0a,raw,_,_,_=load_runtime(project,downstream='none')
    r12=old.imports_and_verify()[1]
    model,criterion,CO,FX=old.make_model(r9,r12)
    state=torch.load(cp,map_location='cpu',weights_only=False)
    check(state.get('checkpoint_type')=='PILOT_ONLY_NOT_CALIBRATED','Wrong checkpoint type')
    old.restore_downstream(model,state['downstream_state'])
    old.lock_gate_thresholds(model);old.verify_gate_thresholds(model,'OCT12 recovery ablation')
    model.eval()
    return old,r9,r8,d,s0a,raw,model,CO,FX


def sigmoid(x):
    x=np.asarray(x,dtype=np.float64)
    return 1/(1+np.exp(-np.clip(x,-40,40)))


def feat_arrays(z,b,expert_z,pairs,global_rank):
    """GT-free per hypothesis. Expert features are ACTUAL frozen native responses."""
    z=np.asarray(z);b=np.asarray(b);expert_z=np.asarray(expert_z)
    probs=sigmoid(z)
    order=np.sort(probs,axis=1)
    base=[];enriched=[]
    for q,c in pairs:
        q=int(q);c=int(c)
        area=max(1e-8,float(b[q,2]*b[q,3]));aspect=max(1e-5,float(b[q,2]/max(1e-5,b[q,3])))
        onehot=[1. if i==c else 0. for i in range(7)]
        p=float(probs[q,c]);top=float(order[q,-1]);gap=float(order[q,-1]-order[q,-2])
        bfeat=[float(np.clip(z[q,c],-12,12)),p,top,gap,
               float(np.clip(np.log(area),-12,0)),float(np.clip(np.log(aspect),-5,5)),
               float(global_rank[(q,c)])/100.]+onehot
        change=np.clip(expert_z[:,q,c]-z[q,c],-12,12).tolist()
        base.append(bfeat);enriched.append(bfeat+change)
    return np.asarray(base,dtype=np.float64).reshape((-1,14)),np.asarray(enriched,dtype=np.float64).reshape((-1,18))


def native_frames(rt,videos):
    old,r9,r8,d,s0a,raw,model,CO,FX=rt
    with torch.no_grad():
        for vid in videos:
            can,spa,tmp=s0a.load_video_triplet(vid)
            selected=list(range(0,len(can),STRIDE))[:NFRAMES]
            print(f'[EXTRACT] FIT video={vid}: {len(selected)} cached frames',flush=True)
            for j in selected:
                fr=can[j];O,X=s0a.make_O_X(fr,spa[j],tmp[j],CO,FX)
                bundle=r9.expert_bundle(model,O,X)
                available=bundle['candidate_mask'].reshape(-1).bool()
                qids=available.nonzero().flatten()
                z=O.z.detach().float().cpu();b=O.b.detach().float().cpu()
                ez=z.unsqueeze(0).expand(4,-1,-1).clone()
                for k in range(4):
                    if len(qids):
                        # Authoritative same-sized candidate-batch decoder, native logits.
                        native=model.adapter.decode_cls(O.h[qids]+ALPHA*bundle['delta_cls'][qids,k])
                        check(tuple(native.shape)==(len(qids),7),'Native head shape drift')
                        ez[k,qids.cpu()]=native.detach().float().cpu()
                vals,ii=z.sigmoid().flatten().topk(min(100,z.numel()))
                ranks={(int(ix//7),int(ix%7)):rank for rank,ix in enumerate(ii.tolist()) if vals[rank]>=.20}
                # All training units belong to operational Base Top100, threshold >=.20.
                chosen=[(q,c) for (q,c) in ranks if bool(available[q])]
                bf,ef=feat_arrays(z.numpy(),b.numpy(),ez.numpy(),chosen,ranks)
                base_match,base_fp,base_counts=old.capture_match(z,b,fr['gt_boxes_norm'],fr['gt_labels'])
                fp=set((int(q),int(c)) for q,c in base_fp)
                tp=set((int(q),int(c)) for q,c,g in base_match)
                check(not fp.intersection(tp),'Duplicate TP/FP hypothesis')
                check(fp.union(tp)==set(ranks.keys()),'Matcher/ranked >=.20 hypothesis mismatch')
                yy=np.array([int(item in fp) for item in chosen],dtype=np.float64)
                yield {'video':vid,'frame':j,'image_id':int(fr['image_id']),
                       'z':z,'b':b,'expert_z':ez,'pairs':chosen,'base_x':bf,'expert_x':ef,
                       'label':yy,'gt_boxes':fr['gt_boxes_norm'].detach().cpu(),
                       'gt_labels':fr['gt_labels'].detach().cpu(),
                       'orig_matches':base_match,'orig_fp':base_fp,
                       'counts':base_counts,'n_candidates':int(available.sum())}
            del can,spa,tmp


def fit_linear(X,y,groups,epochs=400,learning_rate=.075,l2=.025):
    """Deterministic standardized unweighted BCE, with per-video mass balance."""
    X=np.asarray(X,dtype=np.float64);y=np.asarray(y,dtype=np.float64);groups=np.asarray(groups)
    check(X.ndim==2 and y.shape==(len(X),),'Training matrix invalid')
    check(len(X)>=100 and y.sum()>=20 and (1-y).sum()>=20,'Insufficient TP/FP labeled data')
    check(len(set(groups.tolist()))>=4,'Insufficient videos for group-aware fit')
    mean=X.mean(0);std=X.std(0);std[std<1e-6]=1
    xn=np.clip((X-mean)/std,-8,8)
    weight=np.array([1/max(1,int((groups==g).sum())) for g in groups],dtype=np.float64)
    weight/=weight.sum()
    w=np.zeros(X.shape[1],dtype=np.float64);bias=float(np.log(y.mean()/(1-y.mean())))
    mw=np.zeros_like(w);vw=np.zeros_like(w);mb=vb=0.
    for i in range(1,epochs+1):
        pred=sigmoid(np.sum(xn*w[None,:],axis=1)+bias)
        g=(pred-y)*weight
        gw=np.sum(xn*g[:,None],axis=0)+l2*w;gb=float(g.sum())
        mw=.9*mw+.1*gw;vw=.999*vw+.001*gw*gw
        mb=.9*mb+.1*gb;vb=.999*vb+.001*gb*gb
        w-=learning_rate*(mw/(1-.9**i))/(np.sqrt(vw/(1-.999**i))+1e-8)
        bias-=learning_rate*(mb/(1-.9**i))/(np.sqrt(vb/(1-.999**i))+1e-8)
    pr=sigmoid(np.sum(xn*w[None,:],axis=1)+bias)
    check(np.isfinite(w).all() and math.isfinite(bias),'Bad logistic model')
    return {'mean':mean.tolist(),'std':std.tolist(),'w':w.tolist(),'bias':bias,
            'train_n':len(y),'train_fp_rate':float(y.mean()),
            'train_bce':float(np.mean(-y*np.log(pr+1e-9)-(1-y)*np.log(1-pr+1e-9)))}


def predict(model,X):
    X=np.asarray(X,dtype=np.float64)
    if len(X)==0:return np.zeros((0,),dtype=np.float64)
    mu=np.asarray(model['mean']);sd=np.asarray(model['std']);w=np.asarray(model['w'])
    check(X.shape[1]==len(w),'Feature/schema dimension drift')
    return sigmoid(np.sum(np.clip((X-mu)/sd,-8,8)*w[None,:],axis=1)+float(model['bias']))


def summarize_action_strength(events):
    """Inference-only, per-actual-action diagnostic, not GT or AP attribution.

    Simple action unit = (query,class); Expert unit = one full 7-class query row.
    Never pool the n counts as if the intervention footprints were equivalent.
    """
    def stats(values):
        if not values:return None
        a=np.asarray(values,dtype=np.float64)
        return {'min':float(a.min()),'p25':float(np.quantile(a,.25)),
                'median':float(np.median(a)),'p75':float(np.quantile(a,.75)),
                'p95':float(np.quantile(a,.95)),'max':float(a.max()),
                'mean':float(a.mean())}
    n=len(events)
    if not n:return {'action_units':0,'selected_class_logit_drop':None,
                     'selected_class_crossed_below_0p20':0,'fraction_crossed_below_0p20':None,
                     'other_class_max_abs_logit_change':None,
                     'n_original_score_at_least_0p65':0,
                     'action_footprint_note':'Simple acts on one (query,class); Expert replaces seven logits of one query'}
    crossed=sum(int(e['crossed_below_0p20']) for e in events)
    return {'action_units':n,
            'selected_class_logit_drop':stats([e['selected_class_logit_drop'] for e in events]),
            'required_drop_to_0p20':stats([e['required_drop_to_0p20'] for e in events]),
            'selected_class_crossed_below_0p20':crossed,
            'fraction_crossed_below_0p20':float(crossed/n),
            'other_class_max_abs_logit_change':stats([e['other_class_max_abs_logit_change'] for e in events]),
            'n_original_score_at_least_0p65':sum(e['original_score_at_least_0p65'] for e in events),
            'action_footprint_note':'Simple acts on one (query,class); Expert replaces seven logits of one query'}


def record_action_strength(audit,raw,out,q,c):
    """Read only logits; store no GT and no large per-action event ledger."""
    if audit is None:return
    pre=float(raw[q,c]);post=float(out[q,c])
    other=(out[q]-raw[q]).detach().abs().clone()
    other[int(c)]=0
    audit.append({'selected_class_logit_drop':pre-post,
                  'required_drop_to_0p20':max(0.,pre-math.log(.2/.8)),
                  'crossed_below_0p20':bool(pre>=math.log(.2/.8) and post<math.log(.2/.8)),
                  'other_class_max_abs_logit_change':float(other.max().item()),
                  'original_score_at_least_0p65':bool(pre>=math.log(.65/.35))})


def intervene(frame,model,kind,threshold,gain=1.0,audit=None):
    """Inference-time policy gets BASE and frozen expert responses only, never GT."""
    check(kind in ('Simple','Expert'),'Unknown policy')
    raw=frame['z'];out=raw.clone()
    p=predict(model,frame['base_x'] if kind=='Simple' else frame['expert_x'])
    candidates=[];items=0
    for idx,(q,c) in enumerate(frame['pairs']):
        conf=float(p[idx]);
        if conf <threshold:continue
        items+=1
        if kind=='Simple':
            attenuation=max(0.,(conf-threshold)/max(1e-9,1-threshold))
            out[q,c]=raw[q,c]-float(gain)*attenuation
            record_action_strength(audit,raw,out,q,c)
        else:
            dz=(frame['expert_z'][:,q,c]-raw[q,c]).numpy()
            k=int(np.argmin(dz))
            if dz[k] >= -1e-4:continue
            candidates.append((conf,int(q),int(c),k))
    if kind=='Expert':
        # One expert per query; never allow accidental last-write-wins.
        candidates.sort(reverse=True)
        chosen=set()
        for conf,q,c,k in candidates:
            if q in chosen:continue
            chosen.add(q)
            out[q]=frame['expert_z'][k,q]
            record_action_strength(audit,raw,out,q,c)
        acted=len(chosen)
    else:acted=items
    check(bool(torch.isfinite(out).all()),'Nonfinite candidate logits')
    return out,int(acted)


def measure(frames,models,kind,threshold,gain,old,include_per_video=False):
    from run15a_final_preservation import identity_ledger
    base=collections.Counter();cand=collections.Counter();lost=rescued=actions=0
    per_vid=collections.defaultdict(lambda:collections.Counter())
    action_strength_events=[]
    for fr in frames:
        z,acted=intervene(fr,models[kind],kind,threshold,gain,audit=action_strength_events)
        matched,fp,m=old.capture_match(z,fr['b'],fr['gt_boxes'],fr['gt_labels'])
        from_l=identity_ledger(fr['orig_matches'],matched)
        base.update(fr['counts']);cand.update(m)
        actions+=acted;lost+=len(from_l['lost']);rescued+=len(from_l['rescued'])
        V=per_vid[fr['video']];V['lost']+=len(from_l['lost']);V['rescued']+=len(from_l['rescued'])
        V['actions']+=acted;V['tp_delta']+=m['tp']-fr['counts']['tp'];V['fp_delta']+=m['fp']-fr['counts']['fp']
    def score(x):
        tp,fp,fn=(x[k] for k in ('tp','fp','fn'))
        return {'tp':int(tp),'fp':int(fp),'fn':int(fn),
                'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}
    check(len(action_strength_events)==actions,'Action-strength count inconsistent with executed actions')
    result={'Base':score(base),'Candidate':score(cand),'lost':lost,'rescued':rescued,'actions':actions,
            'delta_f1':score(cand)['f1']-score(base)['f1'],
            'action_strength':summarize_action_strength(action_strength_events)}
    if include_per_video:result['per_video']={str(v):dict(c) for v,c in per_vid.items()}
    return result


def choose(frames,models,old):
    selections={};ledger=[]
    for kind in ('Simple','Expert'):
        best=None
        for threshold in GRID_THRESH:
            for gain in (GRID_GAIN if kind=='Simple' else (1.0,)):
                m=measure(frames,models,kind,threshold,gain,old)
                ledger.append({'method':kind,'threshold':threshold,'gain':gain,**m})
                # Conservative no-loss FIT-tune filter: not a real unseen guarantee.
                if m['lost']!=0 or m['delta_f1']<=0 or m['actions']<1:continue
                # Choose F1, tie: fewer actions; then higher threshold.
                key=(m['delta_f1'],-m['actions'],threshold)
                if best is None or key>best[0]:best=(key,{'threshold':threshold,'gain':gain,'tune':m})
        selections[kind]=best[1] if best else None
    return selections,ledger


def paired_coco(rt,frames,models,choices):
    """All paired COCO predictions evaluated over EXACT same FIT image universe."""
    from run15a_final_preservation import identity_ledger
    from pycocotools.coco import COCO
    old,r9,r8,d,s0a,raw,model,CO,FX=rt
    meta={int(x['id']):x for x in raw['images']}
    image_ids=set(); pred={k:{} for k in ('Base','Simple','Expert')};ca={k:{} for k in pred}
    ops={k:collections.Counter() for k in pred};per_video=collections.defaultdict(lambda:collections.defaultdict(collections.Counter))
    action_strength_events={k:[] for k in pred}
    for fr in frames:
        iid=fr['image_id'];check(iid not in image_ids,'Duplicate image_id');image_ids.add(iid)
        ann=meta[iid]
        check(int(ann['video_id'])==fr['video'],'GT image to video mismatch')
        decisions={'Base':(fr['z'],0)}
        for kind in ('Simple','Expert'):
            selected=choices.get(kind)
            if selected is None:decisions[kind]=(fr['z'],0)
            else:decisions[kind]=intervene(fr,models[kind],kind,selected['threshold'],selected['gain'],
                                             audit=action_strength_events[kind])
        for kind,(z,actions) in decisions.items():
            matched,fps,m=old.capture_match(z,fr['b'],fr['gt_boxes'],fr['gt_labels'])
            ops[kind].update(m);ops[kind]['actions']+=actions
            pr=r8.extract_top100_predictions(z.detach().float().cpu(),fr['b'].detach().float().cpu())
            r8.add_coco_prediction(pred[kind],ca[kind],iid,pr,float(ann['width']),float(ann['height']),torch.arange(1,8,dtype=torch.long))
            ledger=identity_ledger(fr['orig_matches'],matched)
            ops[kind]['lost']+=len(ledger['lost']);ops[kind]['rescued']+=len(ledger['rescued'])
            per_video[str(fr['video'])][kind].update({'lost':len(ledger['lost']),'rescued':len(ledger['rescued']),
                                    'actions':actions,'tp_delta':m['tp']-fr['counts']['tp'],'fp_delta':m['fp']-fr['counts']['fp']})
    gt_universe=[a for a in raw['annotations'] if int(a['image_id']) in image_ids]
    subset=copy.deepcopy(raw);subset['images']=[meta[i] for i in sorted(image_ids)]
    subset['annotations']=gt_universe
    subset['videos']=[v for v in raw['videos'] if int(v['id']) in EVAL]
    check(len(gt_universe)==sum(int(x['tp']+x['fn']) for x in [ops['Base']]),'COCO cached GT mismatch')
    coco=COCO();coco.dataset=subset
    with contextlib.redirect_stdout(io.StringIO()):coco.createIndex()
    coco_ca=copy.deepcopy(coco);coco_ca.dataset['categories']=[{'id':1,'name':'object'}]
    for an in coco_ca.dataset['annotations']:an['category_id']=1
    with contextlib.redirect_stdout(io.StringIO()):coco_ca.createIndex()
    results={}
    for name in pred:
        ev,ec=r8.make_evaluators(coco,coco_ca)
        ev.update(pred[name]);ec.update(ca[name])
        results[name]=d.finalize_variant_evaluator(ev,ec,list(old.CLASSES))
    def record(v):
        tp,fp,fn=(int(v.get(k,0)) for k in ('tp','fp','fn'))
        return {'tp':tp,'fp':fp,'fn':fn,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
                'lost':int(v['lost']),'rescued':int(v['rescued']),'actions':int(v['actions'])}
    for kind in ('Simple','Expert'):
        check(len(action_strength_events[kind])==ops[kind]['actions'],
              'EVAL action-strength count inconsistent with executed actions: '+kind)
    return {'coco_metrics':results,'counts':{k:record(v) for k,v in ops.items()},
            'action_strength':{k:summarize_action_strength(v) for k,v in action_strength_events.items()},
            'per_video':{v:{k:dict(c) for k,c in methods.items()} for v,methods in per_video.items()},
            'images':len(image_ids),'gt':len(gt_universe)}


def paths(project):
    root=Path(project)/'bfcef_f_s0'/'run15a_final_candidate_sandbox'/'oct12_recovery_ablation'
    return root/'trained_classifier_and_tuning.json',root/'paired_eval.json'


def run_train(project):
    cp=verification(project)
    ck,ev=paths(project);check(not ck.exists() and not ev.exists(),'Recovery experiment already exists')
    rt=setup_native(project);old=rt[0]
    train=list(native_frames(rt,TRAIN));tune=list(native_frames(rt,TUNE))
    # Feature selection exclusively from TRAIN; TUNE used only for policy thresholds.
    xx={};y=[];vv=[]
    for fr in train:
        y.extend(fr['label'].tolist());vv.extend([fr['video']]*len(fr['label']))
    yy=np.asarray(y,dtype=np.float64);groups=np.asarray(vv)
    check(yy.size>100,'Insufficient labeled hypotheses')
    train_distrib={str(int(g)):{'pairs':int((groups==g).sum()),'FP':int(yy[groups==g].sum()),
                           'TP':int((groups==g).sum()-yy[groups==g].sum())} for g in TRAIN}
    print('[TRAIN TARGETS]',json.dumps({'n':int(len(yy)),'FP':int(yy.sum()),'TP':int(len(yy)-yy.sum())}),flush=True)
    models={}
    for name,key in [('Simple','base_x'),('Expert','expert_x')]:
        X=np.concatenate([fr[key] for fr in train if len(fr['label'])],axis=0)
        models[name]=fit_linear(X,yy,groups)
    choices,ledger=choose(tune,models,old)
    status='TRAINED_TUNED' if any(choices.values()) else 'HOLD_NO_SAFE_BENEFICIAL_TUNE_POLICY'
    report={'schema':SCHEMA,'status':status,'pilot32_checkpoint_sha256':PILOT_SHA,
            'source_pins':PINS,'new_script_sha256':sha(Path(__file__)),
            'train_videos':list(TRAIN),'tune_videos':list(TUNE),'sealed_eval_videos':list(EVAL),
            'stride':STRIDE,'max_frames_per_video':NFRAMES,'expert_amplitude':ALPHA,
            'train_distribution':train_distrib,'models':models,'choices':choices,'tuning_grid':ledger,
            'FIT_only':True,'NO_upstream_model_changes':True,'NO_official_CAL_QUAL_Validation_Test':True,
            'limitations':['This is a classification-recovery ABLATION, not production B-FCEF-F U/H admission',
                           'GT labels supervise FP-vs-TP discrimination on upstream-seen FIT videos',
                           'Thresholds tuned only on distinct FIT videos; cannot certify unseen safety',
                           'The classifier compares native expert-response features to simple score calibration',
                           'The final EVAL set must not be used for further tuning',
                           'Simple and Expert modify different numbers of class logits per action; this is a heuristic intervention comparison, not isolated feature attribution',
                           'Simple correction is gain times model-confidence attenuation; finite gain does not guarantee removal of every high-confidence FP',
                           'Per-action logit-drop summaries are inference-only and cannot substitute for AP or GT identity metrics']}
    write_new(ck,report)
    print('[TRAIN RECEIPT]',ck,'status=',status,'choices=',json.dumps(choices),flush=True)
    return report


def run_eval(project):
    ck,ev=paths(project);check(ck.exists() and not ev.exists(),'Missing receipt or evaluation already written')
    rec=json.loads(ck.read_text(encoding='utf8'))
    check(rec.get('schema')==SCHEMA and rec.get('source_pins')==PINS and rec.get('new_script_sha256')==sha(Path(__file__)),
          'Training receipt does not describe this exact code')
    check(rec['train_videos']==list(TRAIN) and rec['tune_videos']==list(TUNE) and rec['sealed_eval_videos']==list(EVAL),
          'Training/development/held-out video groups changed')
    check(rec['status']=='TRAINED_TUNED','No beneficial safe tuning result: evaluation withheld')
    rt=setup_native(project)
    frames=list(native_frames(rt,EVAL))
    result=paired_coco(rt,frames,rec['models'],rec['choices'])
    result.update({'schema':SCHEMA+'_evaluation','status':'FIT_PAIRED_EVAL_NOT_QUALIFIED',
                   'checkpoint_receipt_sha256':sha(ck),'source_sha256':sha(Path(__file__)),
                   'train_videos':list(TRAIN),'tune_videos':list(TUNE),'eval_videos':list(EVAL),
                   'FIT_only':True,'CAL_QUAL_official_val_test_unopened':True,
                   'limitations':['Do not compare absolute FIT AP with official SeaClips Test AP',
                                  'Upstream R50 and experts saw these Train videos',
                                  'One FIT evaluation only; do not tune against its results',
                                  'Expert arm is a recovery ablation, NOT the production B-FCEF-F router',
                                  'Simple modifies a selected (query,class), Expert overwrites seven logits for one query',
                                  'The prior 29-video FIT oracle included these upstream-seen videos; video-group EVAL is not wholly untouched research-wise']})
    write_new(ev,result)
    print('[PAIRED EVAL REPORT]',ev)
    for name,metrics in result['coco_metrics'].items():
        print(name,'mAP30',metrics['map30'],'mAPca30',metrics['mapca30'],
              'F1',result['counts'][name]['f1'],'lost',result['counts'][name]['lost'],
              'rescued',result['counts'][name]['rescued'],'actions',result['counts'][name]['actions'])
    return result


def selftest():
    # Split contract, feature isolation, cross-query control, and train/score logic.
    assert set(TRAIN).isdisjoint(TUNE) and set(TRAIN).isdisjoint(EVAL) and set(TUNE).isdisjoint(EVAL)
    z=np.zeros((3,7));z[0,0]=2.;b=np.ones((3,4))*.25
    ez=np.tile(z,(4,1,1));ez[0,0,0]=1.;ez[1,0,0]=3.
    rank={(0,0):0,(1,1):3}
    bf,ef=feat_arrays(z,b,ez,[(0,0),(1,1)],rank)
    assert bf.shape==(2,14) and ef.shape==(2,18)
    assert ef[0,-4]==-1 and ef[0,-3]==1
    rng=np.random.default_rng(123)
    x=rng.normal(size=(150,14));y=np.zeros(150);y[:45]=1;v=np.repeat(np.arange(5),30)
    model=fit_linear(x,y,v,epochs=20)
    p=predict(model,x)
    assert p.shape==(150,) and bool(np.isfinite(p).all())
    import types
    f={'z':torch.tensor(z,dtype=torch.float32),'b':torch.tensor(b,dtype=torch.float32),
       'expert_z':torch.tensor(ez,dtype=torch.float32),'pairs':[(0,0),(1,1)],
       'base_x':bf,'expert_x':ef}
    fake={'mean':[0]*18,'std':[1]*18,'w':[0]*18,'bias':10}
    o,n=intervene(f,fake,'Expert',0.9)
    assert n==1 and torch.equal(o[0],f['expert_z'][0,0]) and torch.equal(o[1],f['z'][1])
    assert torch.equal(f['z'],torch.tensor(z,dtype=torch.float32)), 'Candidate intervention mutated baseline'
    fakeb={'mean':[0]*14,'std':[1]*14,'w':[0]*14,'bias':10}
    s,k=intervene(f,fakeb,'Simple',0.9,1.)
    assert k==2 and s[0,0]<f['z'][0,0] and torch.equal(s[0,1:],f['z'][0,1:])
    print('[PASS] OCT12 recovery ablation: split, decoder feature shapes, baseline immutability, logistic training, intervention contract')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--phase',choices=('selftest','train','eval'),required=True)
    ap.add_argument('--project',default=str(BASE))
    args=ap.parse_args()
    if args.phase=='selftest':selftest();return
    project=Path(args.project).resolve()
    if args.phase=='train':run_train(project)
    else:run_eval(project)

if __name__=='__main__':main()
