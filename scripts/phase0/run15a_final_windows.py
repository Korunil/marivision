"""Read-only FIT-native integration and metrics for the B-FCEF-F final candidate.

The certified V2 checkpoint is reused ONLY as a diagnostic warm start.
This DOES NOT train new U/H heads or claim V3 qualification.
Never opens CAL, QUAL, official Validation or Test video caches.
"""
from __future__ import annotations
import argparse, collections, contextlib, copy, gc, hashlib, io, json, os, sys
from pathlib import Path
from PIL import Image  # retain legacy Windows import context
for key, val in {'R0_MSDA_BACKEND':'pytorch','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}.items():
    os.environ.setdefault(key,val)
import torch
from run15a_final_risk import RiskPolicy
from run15a_final_mechanism import mechanism_forward
from run15a_final_preservation import identity_ledger,protected_gt_loss

PINNED_DRIVER='6fa33f1128cbc22266c580e7be85a4b69fd1bac2651ed358b64140236cd733e2'
PINNED_V2_MECHANISM='42404d765097c65b23ec318ed51b8c4fc7b2acdd9eb34a331a76c09313bcb637'
FIT=(2,3,4,5,6,7,8,13,14,16,17,18,19,20,24,25,29,31,35,36,37,38,40,41,42,43,44,45,47)


def require(condition,msg):
    if not condition:raise AssertionError('[FINAL CANDIDATE HOLD] '+msg)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def write_new(p, obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    require(not p.exists(),f'Refuse overwrite existing report: {p}')
    tmp=p.with_suffix(p.suffix+'.tmp');require(not tmp.exists(),f'Incomplete temp report: {tmp}')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8');tmp.rename(p)

def load_runtime(project, downstream="v2"):
    project=Path(project).resolve()
    require(sha(project/'run15a_internal_rebuild_PATCH03.py')==PINNED_DRIVER,'Certified V2 driver source drift')
    require(sha(project/'run15a_internal_mechanism.py')==PINNED_V2_MECHANISM,'Certified V2 mechanism source drift')
    sys.path.insert(0,str(project))
    import run15a_internal_rebuild_PATCH03 as old
    require(tuple(old.FIT)==FIT and set(FIT).isdisjoint(old.CAL) and set(FIT).isdisjoint(old.QUAL),'Split drift')
    r9,r12,r8,d,s0a,raw=old.imports_and_verify()
    # For fresh pilot, do NOT load a duplicate V2 GPU model/checkpoint.
    if downstream=='v2':
        model,criterion,CO,FX=old.load_fitted(r9,r12)
        model.eval();old.verify_gate_thresholds(model,'RUN15A final readonly preflight')
    elif downstream=='none':
        model=CO=FX=None
    else:raise ValueError('Unsupported downstream mode')
    return old,r9,r8,d,s0a,raw,model,CO,FX

def sample_frame(runtime,vid,idx):
    old,r9,r8,d,s0a,raw,model,CO,FX=runtime
    require(vid in FIT,'Cannot open non-FIT video')
    can,spa,tmp=s0a.load_video_triplet(vid)
    require(0<=idx<len(can),'Frame index out of range')
    frame=can[idx]; O,X=s0a.make_O_X(frame,spa[idx],tmp[idx],CO,FX)
    bundle=r9.expert_bundle(model,O,X)
    return O,X,bundle,frame

def preflight(runtime):
    old,r9,r8,d,s0a,raw,model,CO,FX=runtime
    # The immutable model is loaded and exercised with a REAL cached FIT frame.
    # Policy overrides are local arguments; no module parameters or state mutated.
    O,X,bundle,fr=sample_frame(runtime,2,0)
    with torch.no_grad():
        closed=mechanism_forward(model,O,X,bundle,hard=True,policy=RiskPolicy(risk_margin_cls=3.,risk_margin_box=3.))
        require(torch.equal(O.z,closed['z']) and torch.equal(O.b,closed['b']),'Exact no-action R0 identity violated')
        require(torch.count_nonzero(closed['pi_cls'][:,1:])==0 and torch.count_nonzero(closed['pi_box'][:,1:])==0,'Closed route reopened')
        box_only=mechanism_forward(model,O,X,bundle,hard=True,policy=RiskPolicy(risk_margin_cls=3.,risk_margin_box=-3.))
        require(torch.equal(O.z,box_only['z']),'Classification changed with class gate closed')
        forced=RiskPolicy(risk_margin_cls=-3.,risk_margin_box=-3.)
        hard=mechanism_forward(model,O,X,bundle,hard=True,policy=forced)
        # All risk permissions forced open (TSK decision unchanged): outputs
        # MUST equal the immutable V2 implementation for actual FIT evidence.
        from run15a_internal_mechanism import mechanism_forward as old_native
        legacy_hard=old_native(model,O,X,bundle,hard=True)
        require(torch.equal(hard['z'],legacy_hard['z']) and torch.equal(hard['b'],legacy_hard['b']),
                'Forced-open risk cannot reproduce source-pinned V2 native outputs')
        ste=mechanism_forward(model,O,X,bundle,hard=False,policy=forced)
        require(torch.equal(hard['z'],ste['z']) and torch.equal(hard['b'],ste['b']),'Training and inference hard forward not bitwise equal')
        for name,x in [('z',hard['z']),('b',hard['b']),('pi_cls',hard['pi_cls']),('pi_box',hard['pi_box'])]:require(bool(torch.isfinite(x).all()),'Nonfinite '+name)
    matched,_,_=old.capture_match(O.z,O.b,fr['gt_boxes_norm'],fr['gt_labels'])
    if matched:
        x=protected_gt_loss(O.z,O.b,hard['z'],hard['b'],fr['gt_boxes_norm'],matched)
        require(bool(torch.isfinite(x['total'])),'Preservation surrogate nonfinite')
    return {'status':'PASS_NATIVE_FIT_PREFLIGHT','fit_video':2,'cached_image_id':int(fr['image_id']),
            'queries':len(O.h),'baseline_matched_gt':len(matched),
            'forced_open_cls_queries':int((hard['hard_cls'].sum(-1)>0).sum()),
            'forced_open_box_queries':int((hard['hard_box'].sum(-1)>0).sum()),
            'source_pinned_v2_native_parity':True,
            'V2_warm_start_only':True,'trained_or_calibrated_final_candidate':False}


def fit_eval(runtime,videos,stride=10):
    """Paired cached-FIT COCO AP30 / class-agnostic AP30 / operational F1.

    Input checkpoint is V2 warm-start; no new model training is implied.
    Uses exact repository evaluator to avoid divergent metrics.
    """
    old,r9,r8,d,s0a,raw,model,CO,FX=runtime
    videos=tuple(sorted(set(videos)))
    require(isinstance(stride,int) and stride>=1 and stride<=100,'Invalid FIT-eval stride')
    require(videos and all(x in FIT for x in videos),'FIT-only evaluation required')
    meta={int(im['id']):im for im in raw['images']}
    anns=collections.defaultdict(list)
    for a in raw['annotations']:anns[int(a['image_id'])].append(a)
    pred={k:{} for k in ('Base','Candidate')};ca_pred={k:{} for k in pred}
    total={k:collections.Counter() for k in pred};lost=rescued=frames=gt=0; image_ids=set()
    by_video={};gate_counts=collections.Counter()
    policy=RiskPolicy()
    with torch.no_grad():
        for vid in videos:
            can,spa,tmp=s0a.load_video_triplet(vid)
            local=collections.Counter()
            for j in range(0,len(can),stride):
                fr=can[j]
                iid=int(fr['image_id']);require(iid not in image_ids,'Duplicate cached image ID');image_ids.add(iid)
                require(int(meta[iid]['video_id'])==vid,'Image/video map mismatch')
                require(len(anns[iid])==len(fr['gt_labels']),'COCO GT/cached GT mismatch')
                O,X=s0a.make_O_X(fr,spa[j],tmp[j],CO,FX)
                bundle=r9.expert_bundle(model,O,X)
                out=mechanism_forward(model,O,X,bundle,hard=True,policy=policy)
                gate_counts['candidate_queries']+=int(bundle['candidate_mask'].sum())
                for task in ('cls','box'):
                    gate_counts[f'active_{task}']+=int((out['hard_'+task].sum(-1)>0).sum())
                    for k,e in enumerate(('S','L','C','T')):
                        gate_counts[f'open_{task}_{e}']+=int(out['hard_'+task][:,k].sum())
                identities={}
                for name,zz,bb in [('Base',O.z,O.b),('Candidate',out['z'],out['b'])]:
                    mm,fp,m=old.capture_match(zz,bb,fr['gt_boxes_norm'],fr['gt_labels'])
                    total[name].update(m);identities[name]=mm
                    pr=r8.extract_top100_predictions(zz.detach().float().cpu(),bb.detach().float().cpu())
                    r8.add_coco_prediction(pred[name],ca_pred[name],iid,pr,
                        float(meta[iid]['width']),float(meta[iid]['height']),
                        torch.arange(1,8,dtype=torch.long))
                ledger=identity_ledger(identities['Base'],identities['Candidate'])
                lost+=len(ledger['lost']);rescued+=len(ledger['rescued']);gt+=len(fr['gt_labels']);frames+=1
                local['lost']+=len(ledger['lost']);local['rescued']+=len(ledger['rescued']);local['frames']+=1
            by_video[str(vid)]=dict(local)
            del can,spa,tmp;gc.collect()
    # Build COCO evaluator from *cached images only*, not the raw Train universe.
    from pycocotools.coco import COCO
    subset=copy.deepcopy(raw)
    subset['images']=[meta[i] for i in sorted(image_ids)]
    subset['annotations']=[a for a in raw['annotations'] if int(a['image_id']) in image_ids]
    subset['videos']=[v for v in raw['videos'] if int(v['id']) in videos]
    require(len(subset['annotations'])==gt,'Cached GT universe mismatch')
    coco=COCO();coco.dataset=subset
    with contextlib.redirect_stdout(io.StringIO()):coco.createIndex()
    coco_ca=copy.deepcopy(coco);coco_ca.dataset['categories']=[{'id':1,'name':'object'}]
    for a in coco_ca.dataset['annotations']:a['category_id']=1
    with contextlib.redirect_stdout(io.StringIO()):coco_ca.createIndex()
    results={}
    for name in pred:
        ev,ec=r8.make_evaluators(coco,coco_ca)
        ev.update(pred[name]);ec.update(ca_pred[name])
        results[name]=d.finalize_variant_evaluator(ev,ec,list(old.CLASSES))
    require({int(x['id']) for x in subset['images']}==image_ids,'COCO image universe drift')
    return {'status':'FIT_DIAGNOSTIC_ONLY_NOT_QUALIFIED','videos':videos,'stride':stride,
        'cached_frames':frames,'gt':gt,'per_video':by_video,'gate_counts':dict(gate_counts),
        'counts':{k:old.metrics(v) for k,v in total.items()},
        'baseline_matched_gt_lost':lost,'previously_missed_gt_rescued':rescued,
        'coco_metrics':results,'existing_V2_weights_warm_start':True,
        'NO_new_training':True,'CAL_QUAL_official_val_test_unopened':True,
        'limitations':['This evaluates the new admission mechanism with OLD V2 downstream weights',
                       'mAP@0.3/mAPca@0.3 are computed on FIT cached images; cannot be compared to published official Test absolute values',
                       'No GT-independent unseen improvement claim is possible']}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project',default=r'D:\MARIVISION\H0_workbench\eval_tools')
    ap.add_argument('--phase',choices=('preflight','fit-eval'),required=True)
    ap.add_argument('--videos',default='2,3')
    ap.add_argument('--stride',type=int,default=10)
    args=ap.parse_args()
    runtime=load_runtime(args.project)
    if args.phase=='preflight':report=preflight(runtime)
    else:report=fit_eval(runtime,[int(v.strip()) for v in args.videos.split(',')],stride=args.stride)
    root=Path(args.project)/'bfcef_f_s0'/'run15a_final_candidate_sandbox'
    filename='native_preflight.json' if args.phase=='preflight' else 'fit_eval_videos_'+args.videos.replace(',','_')+f'_stride{args.stride}.json'
    report.update({'source_shas':{f:sha(Path(__file__).with_name(f)) for f in ('run15a_final_mechanism.py','run15a_final_risk.py','run15a_final_preservation.py','run15a_final_windows.py')},
                   'pinned_original_driver_sha256':PINNED_DRIVER,'pinned_original_mechanism_sha256':PINNED_V2_MECHANISM})
    p=root/filename
    if p.exists():
        require(json.loads(p.read_text(encoding='utf-8'))==report,'Existing result differs: HOLD; preserve both')
        print('[RESUME VERIFIED]',p)
    else:write_new(p,report);print('[NEW]',p)
    print('[RESULT]',json.dumps({k:report[k] for k in report if k in ('status','coco_metrics','counts','baseline_matched_gt_lost','previously_missed_gt_rescued')},default=str))

if __name__=='__main__':main()
