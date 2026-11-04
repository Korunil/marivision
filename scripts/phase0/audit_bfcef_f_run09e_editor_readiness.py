from __future__ import annotations
"""RUN_09E -- Train-only, read-only, expert/reviewer/Editor readiness audit.

No learning, editing, threshold calibration, Val or Test access.
The run examines whether observable evidence from a different expert might
help review another expert's proposals, and measures upper-bound opportunities
without mistaking retrospective ground-truth outcomes for inference features.

Prerequisites: certified RUN_07 OOF, RUN_09A local consequences and RUN_09D
report plus original Train COCO annotations. Uses the certified RUN_08
operational matcher. Outputs JSON and reviewer summary CSV.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import torch

ROOT = Path('D:/MARIVISION')
EVAL = ROOT / 'H0_workbench' / 'eval_tools'
if str(EVAL) not in sys.path:
    sys.path.insert(0, str(EVAL))
import run_bfcef_f_run08_oof_counterfactual_audit as r8

TRAIN_JSON = ROOT / 'seaclips-train.json'
LOCAL_ROOT = EVAL / 'bfcef_f_s0' / 'run09_final_local_consequences_v1'
LOCAL_MANIFEST = LOCAL_ROOT / 'run09_final_consequence_manifest.json'
OOF_ROOT = EVAL / 'bfcef_f_s0' / 'final_expert_crossfit_v1'
OOF_MANIFEST = OOF_ROOT / 'final_expert_oof_manifest.json'
RUN09D_REPORT = EVAL / 'bfcef_f_s0' / 'run09d_calibrated_qualification_v1' / 'run09d_calibrated_integrated_qualification.json'
OUT = EVAL / 'bfcef_f_s0' / 'run09e_editor_readiness_v1'

EXPECTED_LOCAL_SHA = 'f6dcf39a073d7c9fe27201dbe8c38d658cb1b0093f5ba1797a9740314dc70b72'
EXPECTED_OOF_SHA = '1c2569f1871b475e4aa71d20e505c8bb738d9b3ed7d4df185787c1c5546c6001'
EXPECTED_RUN09D_SHA = '629365fa4e4cf45312dfad714146aafb46a2808c9f259a298eab6b040d2cdd5f'
EXPERTS = ('S','L','C','T')
SAMPLE_VIDS = (6, 3, 2, 11, 1)  # Exactly one video per immutable source fold.
EPS = 1e-6


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(2**20),b''):
            h.update(data)
    return h.hexdigest()


def assert_sha(path, expected):
    if not path.exists():
        raise AssertionError(f'Missing: {path}')
    actual=sha(path)
    if actual != expected:
        raise AssertionError(f'SHA mismatch for {path}: expected {expected}; got {actual}')


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def to_int(x):
    return int(x.item()) if hasattr(x,'item') else int(x)


def iou_matrix(a, b):
    # Shapes [N,4], [M,4], cxcywh normalized; vectorized and deterministic.
    a=np.asarray(a, dtype=np.float64).reshape(-1,4)
    b=np.asarray(b, dtype=np.float64).reshape(-1,4)
    if len(a)==0 or len(b)==0:
        return np.zeros((len(a),len(b)),dtype=np.float64)
    aa1=a[:,:2]-a[:,2:]/2; aa2=a[:,:2]+a[:,2:]/2
    bb1=b[:,:2]-b[:,2:]/2; bb2=b[:,:2]+b[:,2:]/2
    xy1=np.maximum(aa1[:,None,:],bb1[None,:,:])
    xy2=np.minimum(aa2[:,None,:],bb2[None,:,:])
    wh=np.clip(xy2-xy1,0,None)
    inter=wh[:,:,0]*wh[:,:,1]
    area_a=(a[:,2]*a[:,3])[:,None]
    area_b=(b[:,2]*b[:,3])[None,:]
    return inter/np.maximum(area_a+area_b-inter,1e-12)


def gt_scale(box, width, height):
    pixels=float(box[2])*width*float(box[3])*height
    return 'small' if pixels < 32**2 else ('medium' if pixels < 96**2 else 'large')


def local_box_path(fold, vid):
    return LOCAL_ROOT / f'fold{fold}' / f'bfcef_f_run09_consequence_vid{vid}.pth'


def oof_path(fold, vid):
    return OOF_ROOT / f'fold{fold}' / f'bfcef_f_s0a_oof_vid{vid}.pth'


def stats_from_binary(y):
    v=np.asarray(y,dtype=np.float64)
    return {'n':int(len(v)), 'mean':float(v.mean()) if len(v) else None}


def binary_state_name(s):
    return ('TP','FP','NONOP')[int(s)]


def transition_delta_tp(s):
    return (1 if s in (3,6) else (-1 if s in (1,2) else 0))


def assert_consequence_shape(record, source_oof):
    q=record['candidate_queries'].long()
    if not torch.equal(q,source_oof['candidate_queries'].long()):
        raise AssertionError('RUN_09A candidate set not equal to RUN_07 OOF candidate set')
    n=len(q)
    for ex in EXPERTS:
        d=record['experts'][ex]
        assert tuple(d['cls_transition'].shape)==(n,7)
        assert tuple(d['box_rescue_target'].shape)==(n,1)
        assert tuple(d['box_loss_target'].shape)==(n,1)
        assert tuple(d['box_retain_target'].shape)==(n,1)
    return q


def exact_match(z,b,gtb,gtl):
    p=r8.extract_top100_predictions(z.float(),b.float())
    m=r8.match_operational_detections_to_gt(p['op_scores'],p['op_c_idx'],p['op_boxes'],p['op_q_idx'],gtb.float(),gtl.long())
    return p,m


def identity_state(pair, result):
    q,c=pair
    if any(d[3]==q and d[4]==c for d in result['gt_to_det'].values()):
        return 0
    return 1 if (q,c) in result['fp_dict'] else 2


def setup():
    for path, expected in ((LOCAL_MANIFEST,EXPECTED_LOCAL_SHA),(OOF_MANIFEST,EXPECTED_OOF_SHA),(RUN09D_REPORT,EXPECTED_RUN09D_SHA)):
        assert_sha(path,expected)
    local=load_json(LOCAL_MANIFEST)
    oof=load_json(OOF_MANIFEST)
    run09d=load_json(RUN09D_REPORT)
    if local['_meta']['total_frames'] != 20075:
        raise AssertionError('RUN09A full frame inventory drift')
    if oof['_meta']['total_oof_frames'] != 20075:
        raise AssertionError('RUN07 full frame inventory drift')
    if list(oof['_meta']['expert_order']) != list(EXPERTS):
        raise AssertionError('RUN07 expert order drift')
    folds=r8.parse_folds(); v2f={v:f for f,vids in folds.items() for v in vids}
    if sorted(v2f)!=list(range(1,48)) or len(v2f)!=47:
        raise AssertionError('Fold inventory is not exactly 47 Train videos')
    if len(local['shard_sha256_ledger'])!=47:
        raise AssertionError('RUN09A should have exactly 47 consequence shards')
    if len(oof['oof_shard_sha256_ledger'])!=47:
        raise AssertionError('RUN07 should have exactly 47 OOF shards')
    train=load_json(TRAIN_JSON)
    cats=sorted(train['categories'],key=lambda x:int(x['id']))
    if len(cats)!=7:
        raise AssertionError('Expected exactly seven Train categories')
    cat_ids=[int(c['id']) for c in cats]
    class_names=[str(c['name']) for c in cats]
    cat2idx={cid:i for i,cid in enumerate(cat_ids)}
    ann_by_img=collections.defaultdict(collections.Counter)
    raw_counts=collections.Counter()
    for ann in train['annotations']:
        cid=int(ann['category_id']); image_id=int(ann['image_id'])
        if cid not in cat2idx:
            raise AssertionError(f'Unknown original Train category ID {cid}')
        ann_by_img[image_id][cat2idx[cid]]+=1
        raw_counts[cat2idx[cid]]+=1
    sizes={int(x['id']):(int(x['width']),int(x['height'])) for x in train['images']}
    image_to_vid={int(x['id']):int(x['video_id']) for x in train['images']}
    cameras={int(v.get('id',v.get('video_id'))):str(v.get('camera_sensor','unknown')) for v in train['videos']}
    print('[PASS] immutable RUN07/RUN09A/RUN09D manifest SHA roots')
    print('[+] Raw Train class mapping:')
    for i,c in enumerate(cats):
        print(f'    index={i} category_id={cat_ids[i]} name={class_names[i]} raw_annotations={raw_counts[i]}')
    return local,oof,run09d,v2f,train,cat_ids,class_names,ann_by_img,raw_counts,sizes,image_to_vid,cameras


def metric_row():
    return collections.Counter()


def reviewer_row():
    return collections.Counter()


def evaluate(args):
    t0=time.time()
    OUT.mkdir(parents=True,exist_ok=True)
    local,oof,prior,v2f,train,cat_ids,class_names,ann_by_img,raw_counts,sizes,image_to_vid,cameras=setup()
    # Local-label TP loss versus FP pruning for a proposer. Reviewer signals are
    # ONLY signed pre-intervention changes in the reviewer's own OOF logits.
    # Ground-truth labels are used for retrospective scoring, never as reviewer inputs.
    reviewer=collections.defaultdict(reviewer_row)
    reviewer_by_fold=collections.defaultdict(reviewer_row)
    reviewer_by_class=collections.defaultdict(reviewer_row)
    oracle_box=collections.defaultdict(metric_row)
    class_support=collections.Counter()
    video_support=collections.defaultdict(collections.Counter)
    camera_support=collections.defaultdict(collections.Counter)
    scale_support=collections.defaultdict(collections.Counter)
    cached_counters=collections.defaultdict(collections.Counter)
    candidate_ceiling=collections.defaultdict(collections.Counter)
    temporal=collections.defaultdict(collections.Counter)
    missing_raw=collections.Counter()
    suspicious_extra=[]
    total_frames=0; total_candidates=0; n_canonical_gt=0
    n_baseline_fn=0
    snapshot_rows=[]

    for vid in range(1,48):
        f=v2f[vid]
        can_path=r8.canonical_path(vid)
        o_path=oof_path(f,vid)
        l_path=local_box_path(f,vid)
        rel_o=f'fold{f}/{o_path.name}'
        rel_l=f'fold{f}/{l_path.name}'
        for path, expected in ((o_path,oof['oof_shard_sha256_ledger'][rel_o]),
                               (l_path,local['shard_sha256_ledger'][rel_l])):
            assert_sha(path,expected)
        can=torch.load(can_path,map_location='cpu',weights_only=False)
        oo=torch.load(o_path,map_location='cpu',weights_only=False)
        ll=torch.load(l_path,map_location='cpu',weights_only=False)['records']
        if not(len(can)==len(oo)==len(ll)):
            raise AssertionError(f'Video {vid}: shard length mismatch')
        total_frames+=len(can)
        camera=cameras.get(vid,'unknown')
        for idx,(fr,of,lf) in enumerate(zip(can,oo,ll)):
            for k in ('image_id','video_id','seq_idx'):
                if to_int(fr[k])!=to_int(of[k]) or to_int(fr[k])!=to_int(lf[k]):
                    raise AssertionError(f'Video {vid}, frame {idx}: {k} drift')
            q=assert_consequence_shape(lf,of)
            n=len(q); total_candidates+=n
            image_id=to_int(fr['image_id'])
            if image_to_vid.get(image_id)!=vid:
                raise AssertionError(f'Video ownership mismatch image_id={image_id}')
            gtl=fr['gt_labels'].long().tolist()
            gtb=fr['gt_boxes_norm'].float()
            n_canonical_gt+=len(gtl)
            counter=collections.Counter(gtl)
            available=ann_by_img[image_id]
            for c,num in counter.items():
                if c not in range(7) or num>available[c]:
                    suspicious_extra.append({'video':vid,'image_id':image_id,'idx':c,'canonical':num,'raw':available[c]})
                class_support[c]+=num
                video_support[str(vid)][c]+=num
                camera_support[camera][c]+=num
                cached_counters[c]['seen']+=num
            for c,num in available.items():
                missing_raw[c]+=max(0,num-counter[c])
            width,height=sizes[image_id]
            for cls,box in zip(gtl,gtb.tolist()):
                scale_support[gt_scale(box,width,height)][cls]+=1

            scores=fr['d_logits'].float().sigmoid()
            z_ex=of['expert_cls_logits'].float().sigmoid()
            dprob=(z_ex-scores[None,:,:])[:,q,:].numpy()  # [4,n,7]
            trans=np.stack([lf['experts'][k]['cls_transition'].numpy() for k in EXPERTS],axis=0)
            for k,proposer in enumerate(EXPERTS):
                # Restrict to actual confidence-decreasing proposals; the scored
                # target reflects exact local operational TP->NONOP or FP->NONOP.
                neg=dprob[k] < -EPS
                tp_lost=neg & (trans[k]==2)
                fp_pruned=neg & (trans[k]==5)
                for j,review_name in enumerate(EXPERTS):
                    if j==k: continue
                    # Independent *source* signal, but not guaranteed statistically
                    # independent because expert backbones share R0 features.
                    reviewer_opposes=dprob[j] > EPS
                    key=(proposer,review_name)
                    row=reviewer[key]
                    for label,mask in (('TP_LOSS',tp_lost),('FP_PRUNE',fp_pruned)):
                        row[f'{label}_count']+=int(mask.sum())
                        row[f'{label}_opposed']+=int((mask & reviewer_opposes).sum())
                        for c in range(7):
                            n_case=int(mask[:,c].sum())
                            n_opposed=int((mask[:,c] & reviewer_opposes[:,c]).sum())
                            reviewer_by_fold[(f,proposer,review_name)][f'{label}_count']+=n_case
                            reviewer_by_fold[(f,proposer,review_name)][f'{label}_opposed']+=n_opposed
                            reviewer_by_class[(c,proposer,review_name)][f'{label}_count']+=n_case
                            reviewer_by_class[(c,proposer,review_name)][f'{label}_opposed']+=n_opposed
                    # Potentially harmful positive promotion reviewer disagreement.
                    fp_created=(dprob[k]>EPS) & (trans[k]==7)
                    reviewer_disagree=dprob[j] < -EPS
                    row['FP_CREATE_count']+=int(fp_created.sum())
                    row['FP_CREATE_disagree']+=int((fp_created & reviewer_disagree).sum())

                bx=lf['experts'][proposer]
                lose=(bx['box_loss_target'].reshape(-1)>0).numpy()
                rescue=(bx['box_rescue_target'].reshape(-1)>0).numpy()
                oracle_box[proposer]['box_loss_queries']+=int(lose.sum())
                oracle_box[proposer]['box_rescue_queries']+=int(rescue.sum())
                for j,review_name in enumerate(EXPERTS):
                    if j==k:continue
                    other=lf['experts'][review_name]
                    other_rescue=(other['box_rescue_target'].reshape(-1)>0).numpy()
                    other_loss=(other['box_loss_target'].reshape(-1)>0).numpy()
                    row=oracle_box[f'{proposer}<-{review_name}']
                    # ORACLE upper bounds only. These values may not be fed to reviewers.
                    row['proposer_loss_with_reviewer_rescue']+=int((lose & other_rescue).sum())
                    row['proposer_rescue_with_reviewer_loss']+=int((rescue & other_loss).sum())
                    row['proposer_loss']+=int(lose.sum())
                    row['proposer_rescue']+=int(rescue.sum())

            # Baseline candidate reachability: direct proposal overlap, NOT an
            # architectural upper bound on possible future geometric correction.
            base_pred,base_match=exact_match(fr['d_logits'],fr['d_boxes'],gtb,fr['gt_labels'])
            n_baseline_fn+=base_match['fn']
            if base_match['fn']:
                all_g=np.arange(len(gtl))
                missed=np.asarray(sorted(set(all_g)-set(base_match['matched_gt_indices'])),dtype=np.int64)
                base_boxes=fr['d_boxes'][q].numpy()
                all_ex=of['expert_box_boxes'][:,q,:].numpy()
                base_iou=iou_matrix(base_boxes,gtb.numpy())
                expert_iou=np.stack([iou_matrix(all_ex[k],gtb.numpy()) for k in range(4)])
                for gi in missed:
                    c=gtl[int(gi)]
                    row=candidate_ceiling[class_names[c]]
                    row['baseline_fn']+=1
                    best_base=(base_iou[:,gi].max() if len(q) else 0.0)
                    best_ex=(expert_iou[:,:,gi].max() if len(q) else 0.0)
                    row['native_candidate_overlap_ge_030']+=int(best_base>=0.30)
                    row['any_final_expert_overlap_ge_030']+=int(best_ex>=0.30)
                    row['neither_overlap_ge_030']+=int(max(best_base,best_ex)<0.30)
            if idx<3:
                for k,name in enumerate(EXPERTS):
                    if name=='T':
                        tt=temporal['first_three_frames']
                        d=lf['experts'][name]
                        tt['box_rescue']+=int((d['box_rescue_target']>0).sum())
                        tt['box_loss']+=int((d['box_loss_target']>0).sum())
                        tt['candidate_queries']+=n
            # Collect deterministic sample with first/center/last frame per fold.
            if vid in SAMPLE_VIDS and idx in {0,len(can)//2,len(can)-1}:
                qs=q.tolist()
                chosen=sorted(set(qs[i] for i in (0,len(qs)//2,len(qs)-1))) if qs else []
                snapshot_rows.append((vid,idx,fr,of,lf,chosen))
        print(f'[+] audited Train video {vid:02d}: {len(can)} frames, source fold {f}')
        del can,oo,ll
    if total_frames!=20075 or n_canonical_gt != 64670 or total_candidates != 960149 or n_baseline_fn != 13929:
        raise AssertionError(f'Frozen Train inventory drift frames={total_frames}, candidates={total_candidates}, canonical_GT={n_canonical_gt}, baseline_FN={n_baseline_fn}')
    if suspicious_extra:
        raise AssertionError(f'Canonical labels exceed original Train image-level category support: sample {suspicious_extra[:5]}')
    print(f'[PASS] full scope: {total_frames} frames, {total_candidates} candidates, {n_canonical_gt} cached GT')

    # Strict, deterministic exact FULL-global matcher comparison sample. We
    # compare the producer's q-local TP label signal to full-system TP changes,
    # which may differ if other query hypotheses are displaced.
    replay=collections.defaultdict(collections.Counter)
    for vid,idx,fr,of,lf,queries in snapshot_rows:
        z=fr['d_logits'].float();b=fr['d_boxes'].float();gtb=fr['gt_boxes_norm'].float();gtl=fr['gt_labels'].long()
        base_p,base_m=exact_match(z,b,gtb,gtl)
        q_ids=lf['candidate_queries'].long().tolist()
        for q in queries:
            qi=q_ids.index(q)
            for ki,k in enumerate(EXPERTS):
                # all seven class logits replaced, as in RUN09A q-local cls
                z_cf=z.clone();z_cf[q]=of['expert_cls_logits'][ki,q].float()
                p,m=exact_match(z_cf,b,gtb,gtl)
                states=lf['experts'][k]['cls_transition'][qi].long().tolist()
                for c,state in enumerate(states):
                    observed=identity_state((q,c),m)
                    if observed != state%3:
                        raise AssertionError(f'RUN09A q-local cls mismatch vid={vid} idx={idx} expert={k} q={q} c={c}: ledger={state%3} observed={observed}')
                local_net=sum(transition_delta_tp(s) for s in states)
                global_net=m['tp']-base_m['tp']
                row=replay[f'{k}_cls'];row['checked']+=1
                row['local_net_tp']+=int(local_net);row['global_net_tp']+=int(global_net)
                row['external_net_tp_not_equal']+=int(local_net!=global_net)
                row['global_fp_changed']+=int(m['fp']!=base_m['fp'])
                row['top100_changed']+=int(set(zip(p['all_q_idx'].tolist(),p['all_c_idx'].tolist()))!=set(zip(base_p['all_q_idx'].tolist(),base_p['all_c_idx'].tolist())))
                # Box: global matching identities at q must equal ledger; other
                # query assignments may still be affected.
                b_cf=b.clone();b_cf[q]=of['expert_box_boxes'][ki,q].float()
                _,m=exact_match(z,b_cf,gtb,gtl)
                for c in range(7):
                    actual=m['gt_to_det']
                    matched_ids=[g for g,v in actual.items() if v[3]==q and v[4]==c]
                    got=matched_ids[0] if matched_ids else -1
                    ledger=int(lf['experts'][k]['box_after_gt_idx'][qi,c].item())
                    if got!=ledger:
                        raise AssertionError(f'RUN09A q-local box identity mismatch vid={vid} idx={idx} exp={k} q={q} c={c}: expected={ledger}, got={got}')
                row=replay[f'{k}_box'];row['checked']+=1
                local_net=int(lf['experts'][k]['box_rescue_target'][qi].item())-int(lf['experts'][k]['box_loss_target'][qi].item())
                global_net=m['tp']-base_m['tp']
                row['local_net_tp']+=local_net;row['global_net_tp']+=global_net
                row['external_net_tp_not_equal']+=int(local_net!=global_net)
                row['global_fp_changed']+=int(m['fp']!=base_m['fp'])
    if not replay:
        raise AssertionError('No local-global replay examples generated')
    print('[PASS] RUN09A local state identity agrees with certified full global matcher on deterministic sample')

    reviewer_out={}
    for (prop,rev),row in sorted(reviewer.items()):
        tp_n=row['TP_LOSS_count'];fp_n=row['FP_PRUNE_count']
        tp_rate=row['TP_LOSS_opposed']/tp_n if tp_n else None
        fp_rate=row['FP_PRUNE_opposed']/fp_n if fp_n else None
        reviewer_out[f'{prop}<-{rev}']={**dict(row),
            'opposition_rate_given_TP_loss':tp_rate,
            'opposition_rate_given_correct_FP_prune':fp_rate,
            'protection_specificity_gap':(tp_rate-fp_rate if tp_rate is not None and fp_rate is not None else None),
            'note':'Observable reviewer disagreement is retrospective association, not a trained Editor nor independent statistical evidence.'}
    diagnosis={
        '_meta':{'schema_version':'bfcef_f_run09e_editor_readiness_v1','read_only':True,'data_scope':'Train only','test_embargo':True,
                 'run07_oof_manifest_sha256':EXPECTED_OOF_SHA,'run09a_consequence_manifest_sha256':EXPECTED_LOCAL_SHA,
                 'run09d_report_sha256':EXPECTED_RUN09D_SHA,'elapsed_minutes':(time.time()-t0)/60.0},
        'scope':{'videos':47,'frames':total_frames,'candidate_queries':total_candidates,'cached_gt':n_canonical_gt,'baseline_operational_fn':n_baseline_fn},
        'class_mapping':[{'model_idx':i,'train_category_id':cat_ids[i],'name':class_names[i],
                          'raw_train_annotations':int(raw_counts[i]),'canonical_gt':int(class_support[i]),
                          'excluded_from_cached_gt':int(missing_raw[i])} for i in range(7)],
        'train_class_video_counts':{v:{class_names[k]:int(n) for k,n in counts.items()} for v,counts in video_support.items()},
        'train_class_camera_counts':{cam:{class_names[k]:int(n) for k,n in counts.items()} for cam,counts in camera_support.items()},
        'train_class_scale_counts':{s:{class_names[k]:int(n) for k,n in counts.items()} for s,counts in scale_support.items()},
        'candidate_reachability_of_baseline_FNs':{k:dict(v) for k,v in candidate_ceiling.items()},
        'collaborative_cls_review_observable_opposition':reviewer_out,
        'collaborative_cls_review_by_source_fold':{f'{fold}:{prop}<-{rev}':dict(row) for (fold,prop,rev),row in reviewer_by_fold.items()},
        'collaborative_cls_review_by_class':{f'{class_names[c]}:{prop}<-{rev}':dict(row) for (c,prop,rev),row in reviewer_by_class.items()},
        'collaborative_box_oracle_only_headroom':{k:dict(v) for k,v in oracle_box.items()},
        'local_vs_global_replay_sample':{k:dict(v) for k,v in replay.items()},
        'early_temporal_proxy':{k:dict(v) for k,v in temporal.items()},
        'run09d_prior_integrated_diagnostics':{
            'report_sha256':EXPECTED_RUN09D_SHA,
            'full_mechanism_diagnostics':prior.get('full_mechanism_diagnostics',{}),
            'note':'Inherited certified pilot Fold0, not re-estimated on Test or Val.'},
        'limitations':[
            'Review opposition is a pre-specified proxy (other expert positive signed class score change); not proof of learned reviewer quality.',
            'Box rescue/loss co-occurrence uses ground truth retrospectively and is ORACLE-ONLY: forbidden as inference-time evidence.',
            'Cached training GT may exclude official annotation rows; exclusions are counted and must be explained before any class-specific claims.',
            'Candidate IoU reachability at 0.30 measures coverage by existing Base/Final boxes, not a theoretical upper bound for new proposal generation.',
            'The deterministic global replay is a sampled consistency audit; the full 20,075-frame local ledger remains separately certified by RUN09A.',
            'No trained Editor, no independent evaluation of reviewer ability and no unseen-video deployment claim yet.',
            'Fold0 and its 09D output are development evidence; baseline R50 trained on the full Train split.'
        ]}
    report=OUT/'run09e_editor_readiness_audit.json'
    report.write_text(json.dumps(diagnosis,indent=2),encoding='utf-8')
    csvp=OUT/'run09e_reviewer_opposition.csv'
    keys=['proposer','reviewer','TP_LOSS_count','TP_LOSS_opposed','opposition_rate_given_TP_loss','FP_PRUNE_count','FP_PRUNE_opposed','opposition_rate_given_correct_FP_prune','protection_specificity_gap']
    with csvp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for pair,rec in sorted(reviewer_out.items()):
            prop,rev=pair.split('<-')
            w.writerow({'proposer':prop,'reviewer':rev,**{key:rec.get(key) for key in keys if key not in ('proposer','reviewer')}})
    print('\n'+'='*108)
    print('RUN_09E CLASS / REVIEWER / EDITOR OPPORTUNITY SUMMARY')
    print('='*108)
    for x in diagnosis['class_mapping']:
        print(f"[+] class idx={x['model_idx']} id={x['train_category_id']} {x['name']:<20} raw={x['raw_train_annotations']:>6} cached={x['canonical_gt']:>6} omitted={x['excluded_from_cached_gt']:>5}")
    for pair,rec in reviewer_out.items():
        a=rec['opposition_rate_given_TP_loss'];b=rec['opposition_rate_given_correct_FP_prune']
        print(f"[+] {pair}: TP-loss review-opposition {a if a is not None else 'NA'}; correct FP-prune opposition {b if b is not None else 'NA'}; nTP={rec['TP_LOSS_count']}, nFP={rec['FP_PRUNE_count']}")
    for ex,rec in replay.items():
        print(f"[+] {ex}: replays={rec['checked']}, local/global TP externalities={rec['external_net_tp_not_equal']}")
    print(f'[+] Baseline FNs analyzed: {n_baseline_fn}')
    print(f'[+] Report: {report} SHA={sha(report)}')
    print(f'[+] Reviewer CSV: {csvp} SHA={sha(csvp)}')
    print('[PASS] RUN_09E TRAIN-ONLY EDITOR OPPORTUNITY / SCIENTIFIC READINESS AUDIT COMPLETE')
    print('DO NOT START NEW MODEL TRAINING OR TEST EVALUATION UNTIL GO/NO-GO REVIEW.')


def self_test():
    a=np.asarray([[.5,.5,.2,.2],[.1,.1,.05,.05]])
    b=np.asarray([[.5,.5,.2,.2]])
    iou=iou_matrix(a,b)
    assert iou.shape==(2,1) and abs(iou[0,0]-1.0)<1e-9 and abs(iou[1,0])<1e-9
    assert [transition_delta_tp(i) for i in range(9)]==[0,-1,-1,1,0,0,1,0,0]
    assert binary_state_name(0)=='TP' and binary_state_name(1)=='FP'
    print('[PASS] RUN_09E synthetic geometry / transition-unit self-test')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--self-test-only',action='store_true')
    args=parser.parse_args()
    self_test()
    if not args.self_test_only:
        evaluate(args)
