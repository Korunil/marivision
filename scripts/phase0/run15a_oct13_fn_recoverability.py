"""OCT13 TRAIN-only, frozen R50 missed-detection recoverability audit.

No fitting or threshold tuning. Oracle operations use GT offline and are not policies.
Native Top100 predictions are used for COCO AP; operational matching uses >=0.20.
Previous frame means frame_id exactly one less within same video, never assumed.
"""
from __future__ import annotations
import argparse, collections, contextlib, copy, hashlib, io, json, math
from pathlib import Path
import numpy as np
import torch
import run15a_oct12_fp_recovery_ablation as a

SOURCE_A='45ca917eec61a8571660ffbec1d0a6d8063f2c2d1b96fcdeaee4c4a422821b70'
FORENSIC_SHA='793cd7d4c1c6cd4cec35edcf9f40420cba6d0a898de5adefbe34c91363927fbe'
SCHEMA='oct13_train_fn_recoverability_gate_v2_review_corrected'
IOU=.30; PREV_IOU=.50; THRESH=.20; PROMOTED=.21
WEAK_OVERLAP_FLOOR=.05; MIN_ADJACENCY_COVERAGE=.95
RANDOM_REPS=32; RANDOM_SEED=20261013
REQUIRED_FN=40; REQUIRED_VIDEOS=4; MAX_VIDEO_SHARE=.40
REQUIRED_ORACLE_AP=.02; REQUIRED_PRECISION=.30
LOWER_CUTOFFS=(.05,.10,.15,.18,.19,.20)


def check(ok,msg):
    if not ok:raise AssertionError('[OCT13 RECOVERABILITY HOLD] '+str(msg))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def pathscope(project,unused_output=True):
    project=Path(project)
    check(sha(a.__file__)==SOURCE_A,'Pinned OCT12 base source drift')
    check(tuple(a.TRAIN)==(2,3,4,5,7,8,13,14,16,17,18,19,20,24,25,31,35),'TRAIN drift')
    check(set(a.TRAIN).isdisjoint(a.TUNE) and set(a.TRAIN).isdisjoint(a.EVAL),'Partition overlap')
    cert=project/'bfcef_f_s0'/'run15a_final_candidate_sandbox'/'oct12_preservation_first'/'oct12_harm_forensic_train_oof.json'
    check(cert.exists() and sha(cert)==FORENSIC_SHA,'Historical forensic receipt drift')
    r=json.loads(cert.read_text(encoding='utf-8'))
    check(r['gate0']=={'status':'EXACT_PARITY','actions':701,'lost_gt_identities':13,'fp_reduction':378,'videos':17},'Historical parity drift')
    a.verification(project)
    dest=project/'bfcef_f_s0'/'run15a_final_candidate_sandbox'/'oct13_recoverability_gate'/'train_fn_recoverability.json'
    if unused_output:check(not dest.exists() and not Path(str(dest)+'.tmp').exists(),'Existing one-shot audit output')
    return dest


def rank_list(z,n=100):
    prob=z.detach().cpu().float().sigmoid().flatten()
    val,idx=prob.topk(min(n,prob.numel()))
    return [(int(k//7),int(k%7),float(v)) for k,v in zip(idx.tolist(),val.tolist())]


def box_ious(boxes,gt):
    """Normalized cxcywh pairwise IoU, QxG. Geometry only."""
    b=np.asarray(boxes,dtype=np.float64).reshape(-1,4)
    g=np.asarray(gt,dtype=np.float64).reshape(-1,4)
    if not len(g):return np.zeros((len(b),0),dtype=np.float64)
    blo=b[:,:2]-b[:,2:]/2; bhi=b[:,:2]+b[:,2:]/2
    glo=g[:,:2]-g[:,2:]/2; ghi=g[:,:2]+g[:,2:]/2
    inter=np.maximum(0,np.minimum(bhi[:,None,:],ghi[None,:,:])-np.maximum(blo[:,None,:],glo[None,:,:]))
    area=np.maximum(0,b[:,2]).clip(min=0)*np.maximum(0,b[:,3]).clip(min=0)
    ga=np.maximum(0,g[:,2])*np.maximum(0,g[:,3])
    i=inter[:,:,0]*inter[:,:,1]
    return i/np.maximum(1e-9,area[:,None]+ga[None,:]-i)


def consecutive(meta,prev_iid,cur_iid,video):
    """Require authoritative frame_id metadata; image ID adjacency alone is NOT evidence."""
    if prev_iid is None or prev_iid not in meta or cur_iid not in meta:return False
    prev=meta[prev_iid];cur=meta[cur_iid]
    if int(prev.get('video_id',-1))!=video or int(cur.get('video_id',-1))!=video:return False
    for key in ('frame_id','frame_index','frame_number'):
        if key in prev and key in cur:
            try:return int(cur[key])-int(prev[key])==1
            except (TypeError,ValueError):return False
    return False


def require_native_iou(old):
    """Fail before loading TRAIN caches if the threshold-aware matcher cannot run."""
    check(callable(getattr(old,'single_iou',None)),
          'Pinned native matcher lacks callable single_iou')


def adjacency_coverage(checked,verified):
    return float(verified/checked) if checked else 0.0


def fn_anatomy(fr):
    z=fr['z'].detach().cpu().float().sigmoid().numpy();b=fr['b'].detach().cpu().float().numpy()
    gt=fr['gt_boxes'].detach().cpu().float().numpy();lab=fr['gt_labels'].detach().cpu().long().numpy()
    overlap=box_ious(b,gt);rank=rank_list(fr['z'])
    rank_ids={(q,c) for q,c,s in rank}
    # Restrict weak-overlap anatomy to hypotheses the evaluator can actually
    # inspect; raw 300-query geometry is NOT an L-expert opportunity.
    rank_supported={(q,c) for q,c,score in rank if score>=WEAK_OVERLAP_FLOOR}
    rank_usable_queries={q for q,c in rank_supported}
    matched={int(g) for q,c,g in fr['orig_matches']}
    found=[]
    for g in range(len(lab)):
        if g in matched:continue
        c=int(lab[g]);good=np.flatnonzero(overlap[:,g]>=IOU)
        ranked_under=[{'q':int(q),'c':c,'score':float(z[q,c]),'iou':float(overlap[q,g])}
                      for q in good if (int(q),c) in rank_ids and float(z[q,c])<THRESH]
        ranked_under.sort(key=lambda it:(-it['score'],-it['iou'],it['q']))
        if ranked_under:typ='SUBTHRESHOLD_TOP100_SAMECLASS'
        elif any((int(q),c) in rank_ids and float(z[q,c])>=THRESH for q in good):
            typ='ACCEPTED_SAMECLASS_COMPETITION'
        else:
            # For a 7-way decoder every query technically has a logit for every
            # class, so class confusion means a dominant WRONG class, not an
            # absent correct-class logit. Report it before generic weak-query.
            wrong=[int(q) for q in good if (int(q),int(np.argmax(z[q]))) in rank_ids
                   and int(np.argmax(z[q]))!=c and float(z[q].max())>=THRESH
                   and float(z[q,c])<THRESH]
            if wrong:typ='ACCEPTED_WRONG_CLASS_OVERLAP'
            elif len(good):typ='SAMECLASS_GEOMETRY_UNAVAILABLE_AT_SCORE_OR_TOP100'
            elif any(float(overlap[q,g])>=.10 for q in rank_usable_queries):
                typ='WEAK_OVERLAP_USABLE_TOP100_ANYCLASS'
            else:typ='NO_USABLE_OVERLAP_TOP100_ANYCLASS'
        weak_top100=max((float(overlap[q,g]) for q in rank_usable_queries),default=0.)
        found.append({'g':int(g),'class':c,'type':typ,'best_qualified':ranked_under[0] if ranked_under else None,
                      'max_box_iou_raw300':float(np.max(overlap[:,g])) if len(overlap) else 0,
                      'max_box_iou_usable_top100':weak_top100,
                      'weak_overlap_score_floor':WEAK_OVERLAP_FLOOR})
    return found


def oracle_action(fr,anatomy):
    """Optimistic GT-derived score promotion. Not a deployable action or a true AP upper bound."""
    z=fr['z'].clone();chosen=[];usedq=set()
    for entry in sorted(anatomy,key=lambda e:e['g']):
        cands=entry['best_qualified']
        if cands is None:continue
        q,c=cands['q'],cands['c']
        if q in usedq:continue
        usedq.add(q);chosen.append({'g':entry['g'],'q':q,'c':c})
    original_max=float(z.max().item())
    for i,x in enumerate(chosen):
        z[x['q'],x['c']]=max(8.,original_max+1.) + .01*(len(chosen)-i)
    return z,chosen


def corroborated_actions(fr,prev):
    """Fixed GT-free one-step same-class IoU >= .5 rule, no ego-motion model."""
    if prev is None:return []
    pp=rank_list(prev['z']);pp=[(q,c) for q,c,score in pp if score>=THRESH]
    if not pp:return []
    cur=rank_list(fr['z']);cur=[(q,c,score) for q,c,score in cur if score<THRESH]
    ious=box_ious(fr['b'].numpy(),prev['b'].numpy());out=[]
    for q,c,s in cur:
        best=max((float(ious[q,pq]) for pq,pc in pp if pc==c),default=0.)
        if best>=PREV_IOU:out.append({'q':q,'c':c,'score':s,'prev_iou':best})
    return out


def random_matched_actions(z,count,rng):
    """Uniformly promote exactly count native sub-threshold Top100 hypotheses."""
    pool=[{'q':q,'c':c} for q,c,score in rank_list(z) if score<THRESH]
    check(0<=count<=len(pool),'Random matched count exceeds sub-threshold pool')
    idx=rng.choice(len(pool),size=count,replace=False).tolist() if count else []
    return [pool[int(j)] for j in idx]


def promote(fr,acts,score=PROMOTED):
    z=fr['z'].clone();target=math.log(score/(1-score))
    for act in acts:z[act['q'],act['c']]=max(float(z[act['q'],act['c']]),target)
    return z


def classify_rule_passers(baseline_matched,post_matched,actions):
    """A promoted TP counts only if it matches a NEW GT identity.

    Promoted detections taking over an already-baseline-matched GT are swaps,
    not false-negative recoveries. Net TP/FP deltas are reported separately.
    """
    act={(int(x['q']),int(x['c'])) for x in actions}
    base_g={int(g) for q,c,g in baseline_matched}
    genuinely_new=0;swaps=0
    for q,c,g in post_matched:
        if (int(q),int(c)) not in act:continue
        if int(g) in base_g:swaps+=1
        else:genuinely_new+=1
    return {'new_gt_matched_by_passers':genuinely_new,
            'promoted_tp_swaps_of_existing_gt':swaps,
            'baseline_matched_gt_count':len(base_g),
            'post_matched_gt_count':len({int(g) for q,c,g in post_matched})}


def capture_threshold(old,z,b,gtb,gtl,cutoff):
    """R50 native Top100, with explicit alternative operational score threshold.

    old.capture_match hardcodes 0.20, so masking scores cannot simulate a
    LOWER cutoff. This is a deliberately separate score-threshold null.
    """
    with torch.no_grad():
        prob=z.detach().float().cpu().sigmoid().flatten()
        val,idx=prob.topk(min(100,prob.numel()))
        b=b.detach().float().cpu();gtb=gtb.detach().float().cpu();gtl=gtl.detach().long().cpu()
        used=set();matched=[];fps=[]
        for v,ix in zip(val.tolist(),idx.tolist()):
            if v<cutoff:continue
            q,c=divmod(int(ix),7);best=None;biou=IOU
            for g in range(len(gtl)):
                if g in used or int(gtl[g])!=c:continue
                ov=float(old.single_iou(b[q],gtb[g]).item())
                if ov>=biou and (best is None or ov>biou):best=g;biou=ov
            if best is None:fps.append((q,c))
            else:used.add(best);matched.append((q,c,best))
        return matched,fps,{'tp':len(matched),'fp':len(fps),'fn':len(gtl)-len(matched)}

def brief(m):
    tp,fp,fn=(int(m[k]) for k in ('tp','fp','fn'))
    return {'tp':tp,'fp':fp,'fn':fn,'f1':float(2*tp/max(1,2*tp+fp+fn))}


def evaluate_ap(rt,image_ids,z_per_variant,records):
    from pycocotools.coco import COCO
    old,r9,r8,d,s0a,raw,model,CO,FX=rt
    image_ids=set(image_ids);meta={int(im['id']):im for im in raw['images']}
    subset=copy.deepcopy(raw)
    subset['images']=[meta[x] for x in sorted(image_ids)]
    subset['annotations']=[ann for ann in raw['annotations'] if int(ann['image_id']) in image_ids]
    subset['videos']=[v for v in raw['videos'] if int(v['id']) in a.TRAIN]
    coco=COCO();coco.dataset=subset
    with contextlib.redirect_stdout(io.StringIO()):coco.createIndex()
    ca=copy.deepcopy(coco);ca.dataset['categories']=[{'id':1,'name':'object'}]
    for ann in ca.dataset['annotations']:ann['category_id']=1
    with contextlib.redirect_stdout(io.StringIO()):ca.createIndex()
    metrics={}
    for kind in z_per_variant:
        preds={};preds_ca={}
        for iid in sorted(image_ids):
            z=z_per_variant[kind][iid];b=records[iid]['b'];ann=meta[iid]
            rows=r8.extract_top100_predictions(z,b)
            r8.add_coco_prediction(preds,preds_ca,iid,rows,float(ann['width']),float(ann['height']),torch.arange(1,8,dtype=torch.long))
        ev,ec=r8.make_evaluators(coco,ca)
        ev.update(preds);ec.update(preds_ca)
        metrics[kind]=d.finalize_variant_evaluator(ev,ec,list(old.CLASSES))
    return metrics,len(subset['annotations'])


def decide(summary):
    q=summary['qualifying']
    first=q['count']>=REQUIRED_FN and q['active_videos']>=REQUIRED_VIDEOS and q['max_share']<=MAX_VIDEO_SHARE
    oracle=summary['oracle_delta']
    second=oracle['map30']>=REQUIRED_ORACLE_AP and oracle['mapca30']>=REQUIRED_ORACLE_AP
    rule=summary['temporal_rule']
    if rule['adjacent_coverage']<MIN_ADJACENCY_COVERAGE:
        return 'HOLD_INSUFFICIENT_AUTHORITATIVE_ADJACENCY_COVERAGE'
    third=rule['precision'] is not None and rule['precision']>=REQUIRED_PRECISION
    if not first:return 'STOP_INSUFFICIENT_FN_OPPORTUNITY'
    if not second:return 'STOP_INSUFFICIENT_ORACLE_AP_GAIN'
    if not third:return 'STOP_TEMPORAL_RULE_LOW_PRECISION'
    nulls=summary['null_assessment']
    if not nulls['count_matched_random_ap']['temporal_above_random_p95_both_ap']:
        return 'STOP_NO_TEMPORAL_AP_ADVANTAGE_OVER_COUNT_MATCHED_RANDOM'
    # The lower-cutoff null is a descriptive operating-point comparator, NOT AP evidence.
    # The T expert isolated-rescue null remains deferred: never issue a GO.
    return 'HOLD_NEEDS_ISOLATED_T_EXPERT_AND_OPERATIONAL_NULL_REVIEW'


def run(project):
    dest=pathscope(project)
    rt=a.setup_native(project);old,r9,r8,d,s0a,raw,model,CO,FX=rt
    require_native_iou(old)
    meta={int(x['id']):x for x in raw['images']}
    counter=collections.Counter();per_video=collections.defaultdict(collections.Counter)
    baseline=collections.Counter();ops={k:collections.Counter() for k in ('Oracle','Temporal')}
    cutoff={str(x):collections.Counter() for x in LOWER_CUTOFFS}
    rows={k:{} for k in ('Base','Oracle','Temporal')};record={};losses=[]
    total_pass=0;true_pass=0;swapped_pass=0;adjacent_checked=adjacent_valid=0;oracle_promotions=0
    with torch.no_grad():
        for vid in a.TRAIN:
            can,spa,tmp=s0a.load_video_triplet(vid)
            indexes=list(range(0,len(can),a.STRIDE))[:a.NFRAMES]
            print(f'[TRAIN CACHE] video={vid} sampled={len(indexes)}',flush=True)
            for j in indexes:
                frame=can[j];iid=int(frame['image_id']);check(iid in meta and int(meta[iid]['video_id'])==vid,'Image/video mismatch')
                O,X=s0a.make_O_X(frame,spa[j],tmp[j],CO,FX)
                if j==indexes[0]:
                    O2,X2=s0a.make_O_X(frame,spa[j],tmp[j],CO,FX)
                    check(torch.equal(O.z,O2.z) and torch.equal(O.b,O2.b),
                          'Cached frame extraction is not idempotent')
                    del O2,X2
                z=O.z.detach().float().cpu();b=O.b.detach().float().cpu()
                check(z.ndim==2 and z.shape[1]==7 and b.shape==(len(z),4),'R50 native shapes')
                # No expert forward pass: first gate only probes raw cached R50.
                gtb=frame['gt_boxes_norm'].detach().float().cpu();gtl=frame['gt_labels'].detach().long().cpu()
                matched,fps,c=old.capture_match(z,b,gtb,gtl)
                baseline.update(c)
                fr={'z':z,'b':b,'gt_boxes':gtb,'gt_labels':gtl,'orig_matches':matched}
                anat=fn_anatomy(fr);check(len(anat)==c['fn'],'FN anatomy does not cover missed GT')
                for x in anat:
                    counter[x['type']]+=1;per_video[vid][x['type']]+=1
                    losses.append({'video':vid,'frame':j,'image_id':iid,**x})
                zo,boost=oracle_action(fr,anat);oracle_promotions+=len(boost)
                prev=None
                if j>0:
                    adjacent_checked+=1
                    ip=int(can[j-1]['image_id'])
                    if consecutive(meta,ip,iid,vid):
                        P,XP=s0a.make_O_X(can[j-1],spa[j-1],tmp[j-1],CO,FX)
                        prev={'z':P.z.detach().float().cpu(),'b':P.b.detach().float().cpu()}
                        adjacent_valid+=1
                acts=corroborated_actions(fr,prev)
                zr=promote(fr,acts)
                for name,v in (('Oracle',zo),('Temporal',zr)):
                    m,fp,sc=old.capture_match(v,b,gtb,gtl)
                    ops[name].update(sc)
                    if name=='Temporal':
                        attribution=classify_rule_passers(matched,m,acts)
                        total_pass+=len(acts)
                        true_pass+=attribution['new_gt_matched_by_passers']
                        swapped_pass+=attribution['promoted_tp_swaps_of_existing_gt']
                    rows[name][iid]=v
                for t in LOWER_CUTOFFS:
                    mth,fth,sc=capture_threshold(old,z,b,gtb,gtl,t)
                    if t==THRESH:
                        check(mth==matched and fth==fps and sc==c,'Lower-cutoff null native 0.20 parity drift')
                    cutoff[str(t)].update(sc)
                rows['Base'][iid]=z;record[iid]={'b':b,'video':vid,'frame':j,'gt_n':len(gtl),'oracle_promotions':boost,
                                                'temporal_passers':acts,'prev_consecutive':prev is not None}
            del can,spa,tmp
    check((baseline['tp'],baseline['fp'],baseline['fn'])==(754,1013,271),'Source-pinned baseline 754/1013/271 failed')
    check(len(record)==17*16,'Expected 272 TRAIN sampled frames')
    check(sum(r['gt_n'] for r in record.values())==1025,'GT count mismatch')
    # Count-matched null: exactly the temporal passers' count per frame,
    # randomly promoted from that frame's entire sub-threshold Top100 pool.
    # No GT and no tuning used by this null. Fixed seed, fixed 32 replicates.
    random_gen=np.random.default_rng(RANDOM_SEED)
    random_names=[]
    for i in range(RANDOM_REPS):
        name=f'RandomPromotion_{i:02d}';random_names.append(name)
        rows[name]={}
        for iid,rec in record.items():
            z=rows['Base'][iid]
            count=len(rec['temporal_passers'])
            picks=random_matched_actions(z,count,random_gen)
            rows[name][iid]=promote({'z':z},picks)
    metric,gt_count=evaluate_ap(rt,record.keys(),rows,record)
    check(gt_count==1025,'COCO GT mismatch')
    eligible=counter['SUBTHRESHOLD_TOP100_SAMECLASS']
    qvids={v:int(x['SUBTHRESHOLD_TOP100_SAMECLASS']) for v,x in per_video.items() if x['SUBTHRESHOLD_TOP100_SAMECLASS']>0}
    main={'count':eligible,'active_videos':len(qvids),'per_video':{str(k):v for k,v in sorted(qvids.items())},
          'max_share':(max(qvids.values())/eligible if eligible else 0)}
    deltas={k:float(metric['Oracle'][k]-metric['Base'][k]) for k in ('map30','mapca30','ap_s30')}
    precision=true_pass/total_pass if total_pass else None
    fp_added=ops['Temporal']['fp']-baseline['fp'];tp_added=ops['Temporal']['tp']-baseline['tp']
    temporal_delta={key:float(metric['Temporal'][key]-metric['Base'][key])
                    for key in ('map30','mapca30','ap_s30')}
    random_deltas={key:[float(metric[name][key]-metric['Base'][key]) for name in random_names]
                   for key in ('map30','mapca30','ap_s30')}
    random_summary={key:{'mean':float(np.mean(vals)),'p95':float(np.percentile(vals,95)),
                         'min':float(min(vals)),'max':float(max(vals))}
                    for key,vals in random_deltas.items()}
    beats_random=all(temporal_delta[key]>random_summary[key]['p95']
                     for key in ('map30','mapca30'))
    # The selected lower-cutoff control is descriptive: closest nonnegative FP
    # difference; not an optimization target for the temporal mechanism.
    controls=[]
    for t in LOWER_CUTOFFS:
        x=cutoff[str(t)];controls.append({'threshold':t,**brief(x),'extra_fp':int(x['fp']-baseline['fp']),
                                          'extra_tp':int(x['tp']-baseline['tp'])})
    comparable=[x for x in controls if x['extra_fp']>=0]
    matched=min(comparable,key=lambda x:(abs(x['extra_fp']-fp_added),abs(x['threshold']-THRESH))) if comparable else None
    # Causal T-only full-response is a separate native null; isolated query T
    # rescue is not measured and must not be claimed.
    low_beat=bool(matched and tp_added>matched['extra_tp'] and fp_added<=matched['extra_fp'])
    summary={'fn_anatomy':dict(counter),'fn_per_video':{str(k):dict(v) for k,v in per_video.items()},
             'qualifying':main,'oracle_delta':deltas,
             'temporal_rule':{'promoted_actions':total_pass,
                              'new_gt_matched_by_passers':true_pass,
                              'promoted_tp_swaps_of_existing_gt':swapped_pass,
                              'precision':precision,
                              'net_tp_added':int(tp_added),'net_fp_added':int(fp_added),
                              'adjacent_checked':adjacent_checked,
                              'adjacent_verified':adjacent_valid,
                              'adjacent_coverage':adjacency_coverage(adjacent_checked,adjacent_valid),
                              'delta_ap':temporal_delta},
             'null_assessment':{'T_only_beaten':None,'T_isolated_null_deferred':True,
                                'lower_cutoff_operational_only':True,
                                'lower_cutoff_operational_advantage':bool(low_beat),
                                'lower_cutoff_comparator':matched,
                                'count_matched_random_ap':{'reps':RANDOM_REPS,'seed':RANDOM_SEED,
                                    'per_frame_count_exact':True,
                                    'summary':random_summary,
                                    'temporal_above_random_p95_both_ap':bool(beats_random)}},
             'fixed_threshold_nulls':controls,
             'operational_counts':{'Base':brief(baseline),**{k:brief(v) for k,v in ops.items()}},
             'coco_metrics':{k:v for k,v in metric.items() if k in ('Base','Oracle','Temporal')},
             'oracle_promotions':oracle_promotions}
    status=decide(summary)
    # The only null involving T here is full native T-response, not the proposed
    # isolated query transition-6 reference; even a tentative GO must be held.
    doc={'schema':SCHEMA,'status':status,'source_sha256':sha(__file__),'pinned_ablation_sha256':SOURCE_A,
         'frozen_forensic_sha256':FORENSIC_SHA,'TRAIN_only':True,'sample_stride':a.STRIDE,
         'no_training':True,'no_new_expert_inference':True,
         'thresholds_fixed':{'matching_iou':IOU,'prev_iou':PREV_IOU,'operational_score':THRESH,
                              'temporal_promotion_score':PROMOTED,'weak_overlap_floor':WEAK_OVERLAP_FLOOR,
                              'min_adjacency_coverage':MIN_ADJACENCY_COVERAGE,
                              'random_null_reps':RANDOM_REPS,'random_seed':RANDOM_SEED},
         'research_gates':{'eligible_fn_min':REQUIRED_FN,'eligible_videos_min':REQUIRED_VIDEOS,
            'max_video_fraction':MAX_VIDEO_SHARE,'oracle_map_and_ca_min':REQUIRED_ORACLE_AP,'passer_precision_min':REQUIRED_PRECISION,
            'temporal_ap_vs_random':'both metrics above fixed 32-replicate random promotion 95th percentiles'},
         'caveats':['Oracle promotes only pre-qualified GT-backed Top100 hypotheses; optimistic GT-informed scenario is NOT a mathematical upper bound',
            'Current selected frames are stride 10, previous frame used only where frame_id verifies adjacency',
            'Previously accepted may include false-positive clutter: causal corroboration is not assumed correct',
            'The frozen T-expert isolated rescue null is NOT evaluated in this cheap first gate; any otherwise positive result must HOLD',
            'Lower-cutoff comparison is an operational count null only: monotone cutoff does not establish AP advantage',
            'Random count-matched AP null is descriptive and uses a fixed 32-replicate seed; no significance claim',
            'Only novel GT identities matched by promoted hypotheses count as recovered; TP swaps are separate',
            'Weak-overlap categories require a Top100 hypothesis >= 0.05; raw 300-query geometry is only a diagnostic',
            'A minimum 95% verified consecutive-frame coverage is required; low coverage forces HOLD',
            'Deterministic first sampled-frame double extraction must yield identical raw logits/boxes',
            'AP is computed from full Top100 ranking, not from a 0.20-thresholded list',
            'No motion compensation, no model fitting, no TUNE/EVAL/CAL/QUAL/official Test access',
            'Upstream models have seen TRAIN, so positive evidence is insufficient for external claims'],
         **summary,'fn_events':losses}
    a.write_new(dest,doc)
    print(f'[PARITY PASS] TP={baseline["tp"]} FP={baseline["fp"]} FN={baseline["fn"]}; frames={len(record)}')
    print('[ANATOMY]',dict(counter));print('[ORACLE DELTA]',deltas)
    print('[TEMPORAL]',doc['temporal_rule'])
    print('[RESULT]',status,str(dest))
    return doc


def selftest():
    boxes=np.array([[.5,.5,.2,.2],[.25,.25,.1,.1]])
    assert np.allclose(np.diag(box_ious(boxes,boxes)),1.)
    assert box_ious(boxes,boxes).shape==(2,2)
    meta={10:{'video_id':2,'frame_id':4},11:{'video_id':2,'frame_id':5},12:{'video_id':3,'frame_id':6}}
    assert consecutive(meta,10,11,2)
    assert not consecutive(meta,11,12,2)
    assert not consecutive({10:{'video_id':2},11:{'video_id':2}},10,11,2)
    z=torch.full((2,7),-5.);z[0,2]=-1.5
    rank=rank_list(z);assert rank[0][:2]==(0,2)
    fr={'z':z,'b':torch.tensor(boxes,dtype=torch.float32)}
    prev={'z':z.clone(),'b':fr['b'].clone()};prev['z'][0,2]=-1
    a1=corroborated_actions(fr,prev)
    assert a1 and a1[0]['q']==0 and a1[0]['c']==2
    assert promote(fr,a1)[0,2].sigmoid().item()>=.209
    s={'qualifying':{'count':45,'active_videos':5,'max_share':.25},
       'oracle_delta':{'map30':.03,'mapca30':.04},
       'temporal_rule':{'precision':.5,'adjacent_verified':10,'adjacent_coverage':1.0},
       'null_assessment':{'count_matched_random_ap':{'temporal_above_random_p95_both_ap':True}}}
    assert decide(s)=='HOLD_NEEDS_ISOLATED_T_EXPERT_AND_OPERATIONAL_NULL_REVIEW'
    s['qualifying']['count']=30;assert decide(s).startswith('STOP_')
    return '[PASS] Geometry, frame continuity, GT-free temporal rule, fixed gates'


def metadata(project):
    """Inspect authoritative COCO frame indexing before any native frame extraction."""
    pathscope(project)
    from run15a_final_windows import load_runtime
    old,r9,r8,d,s0a,raw,model,CO,FX=load_runtime(project,downstream='none')
    relevant=[im for im in raw['images'] if int(im['video_id']) in a.TRAIN]
    keys=set().union(*(set(x) for x in relevant)) if relevant else set()
    frame_keys=[k for k in ('frame_id','frame_index','frame_number') if k in keys]
    print('[COCO TRAIN FRAME METADATA] n_images=',len(relevant),'possible_frame_keys=',frame_keys,
          'example_keys=',sorted(keys),flush=True)
    check(bool(relevant),'No TRAIN image metadata')
    if not frame_keys:
        raise AssertionError('[HOLD] No authoritative frame index; do not infer adjacency from image IDs')
    else:
        print('[PASS] Frame index field exists; full audit must still verify consecutive frame IDs per pair',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',required=True,choices=('selftest','verify','metadata','audit'))
    ap.add_argument('--project',default=r'D:\MARIVISION\H0_workbench\eval_tools');args=ap.parse_args()
    if args.phase=='selftest':print(selftest())
    elif args.phase=='verify':print('[PASS] Frozen source/forensic receipt and TRAIN embargo; output=',pathscope(args.project))
    elif args.phase=='metadata':metadata(args.project)
    else:run(args.project)

if __name__=='__main__':main()
