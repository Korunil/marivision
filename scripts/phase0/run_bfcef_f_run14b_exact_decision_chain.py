from __future__ import annotations
"""RUN14B: exact query/class survival and GT-level attribution; read-only retrospective.
No training, parameter changes, threshold search, model selection or Test access.
The only generated output is a new audit JSON/CSV in a RUN14-specific folder.
RUN13 remains the ONLY independent validation qualification.
"""
import argparse
import csv
import gc
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image  # Windows/Pillow import-order consistency

for k,v in {"R0_MSDA_BACKEND":"pytorch","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"}.items():
    os.environ.setdefault(k,v)

ROOT=Path('D:/MARIVISION')
EVAL=ROOT/'H0_workbench'/'eval_tools'
VAL_OUT=EVAL/'bfcef_f_s0'/'run13_frozen_independent_validation_v1'
RUN13_JSON=VAL_OUT/'run13_official_validation_report.json'
RUN13_SHA='163d70a92b7b4d3a31859db428302f775fe5165b560e6d380743fa2c9fd765c7'
RUN13_SOURCE='273c166e353183e9c8ad2a9aabe86834ee22e7696daf9aa9fe1d375ca5c689f4'
OUT=EVAL/'bfcef_f_s0'/'run14b_exact_decision_chain_v1'
EXPERTS=('S','L','C','T')

RUN14_JSON=EVAL/'bfcef_f_s0'/'run14_readonly_causal_attribution_v1'/'run14_readonly_causal_attribution.json'
RUN14_SHA='4acd0384574443e13e9f3ad2dada0f753b09dade643fc20a0a940aff23cb6bdb'
RUN14_SOURCE_SHA='4695789c4fee34f2e6b4dc909901d64547c8634bbe35187cee4b69c095ad10e2'
REFERENCE_GT_LOSSES=755



def require(ok,msg):
    if not ok:raise AssertionError('[FATAL] '+msg)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def decision_stage(exact_pair, threshold_survives, iou_survives):
    """Mutually exclusive observable decision-stage, NOT upstream causal proof."""
    if not exact_pair:
        return 'exact_pair_evicted_from_top100'
    if not threshold_survives:
        return 'same_pair_below_operating_score'
    if not iou_survives:
        return 'same_pair_localization_below_iou'
    return 'same_pair_survives_but_matching_changes'


def selftest():
    # Demonstrates documented RUN09C quantization/inference boundary mismatch.
    bins=10000;g=.00205;t=.0020
    require(int(g*bins)==int(t*bins) and g>t,'Bin semantic fixture changed')
    require(not(int(g*bins)>=int(t*bins)+1),'Histogram discrepancy fixture not reproduced')
    # Two-branch exact Shapley conservation.
    yb,yc,yl,yf=10,7,12,9
    sc=((yc-yb)+(yf-yl))/2;sb=((yl-yb)+(yf-yc))/2
    require(sc+sb==yf-yb,'Shapley sum mismatch')
    toy={(2,1),(9,3)}
    require((2,1) in toy and (2,2) not in toy,'Pair membership must be exact, not query-only')
    expected=['exact_pair_evicted_from_top100', 'same_pair_below_operating_score',
              'same_pair_localization_below_iou','same_pair_survives_but_matching_changes']
    actual=[decision_stage(False,False,False),decision_stage(True,False,True),
            decision_stage(True,True,False),decision_stage(True,True,True)]
    require(actual==expected,'Decision-stage precedence fixture failed')
    print('[PASS] RUN14B numerical/selftest; no dataset or checkpoint opened')


def compute_match(r8,z,b,gtb,gtl):
    pred=r8.extract_top100_predictions(z.detach().float().cpu(),b.detach().float().cpu())
    m=r8.match_operational_detections_to_gt(pred['op_scores'],pred['op_c_idx'],pred['op_boxes'],pred['op_q_idx'],gtb,gtl)
    return pred,m


def run():
    require(not OUT.exists(),'RUN14B output already exists; refuse overwrite')
    require(sha(EVAL/'run_bfcef_f_run14_readonly_causal_attribution.py')==RUN14_SOURCE_SHA,
            'The original RUN14 replay source was modified. Audit source difference before continuing.')
    require(RUN14_JSON.is_file() and sha(RUN14_JSON)==RUN14_SHA,
            'RUN14 forensic reference missing or digest changed. Do not rerun/overwrite RUN14.')
    reference14=json.loads(RUN14_JSON.read_text(encoding='utf8'))
    require(reference14.get('lost_gt_total')==REFERENCE_GT_LOSSES,'RUN14 baseline lost GT count changed')
    require(RUN13_JSON.is_file() and sha(RUN13_JSON)==RUN13_SHA,'RUN13 benchmark report hash changed')
    if str(EVAL) not in sys.path:sys.path.insert(0,str(EVAL))
    import run_bfcef_f_run13_frozen_validation as r13
    import run_bfcef_f_run12_final_train as r12
    import train_bfcef_f_run09_joint_uh_gate_router as r9
    import run_bfcef_f_run08_oof_counterfactual_audit as r8
    import audit_bfcef_f_run09d_calibrated_qualification as d
    import torch
    require(sha(Path(r13.__file__))==RUN13_SOURCE,'RUN13 source drift')
    require(r8.TOPK==100 and r8.IOU_THRESHOLD==.3 and r8.SCORE_THRESHOLD==.2,'Matching policy changed')
    frozen=r13.source_preflight()
    manifest,data=r13.manifest_preflight(read_validation=False)
    _,data=r13.manifest_preflight(read_validation=True)
    base_report=json.loads(RUN13_JSON.read_text(encoding='utf8'))
    require(base_report['prospective_decision']['validation_gate']=='HOLD','Frozen RUN13 outcome changed')
    require(base_report['_meta']['single_candidate']==r13.SHA_FINAL,'Model identity changed')
    model,_,CO,FX=r9.initialize_runtime()
    ck=torch.load(r13.F_CKPT,map_location='cpu',weights_only=False)
    r12.load_experts(model,ck);r12.load_downstream(model,ck)
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    for task in ('cls','box'):
        for e in EXPERTS:
            require(abs(float(getattr(model,'gate_'+task)[e].mask_thresh)-frozen[task][e])<1e-6,'Calibration changed')
    images={int(im['id']):im for im in data['images']}
    annotations=defaultdict(list)
    for ann in data['annotations']:annotations[int(ann['image_id'])].append(ann)
    category_names={int(c['id'])-1:c['name'] for c in data['categories']}
    video_cameras={int(v['id']):str(v['camera_sensor']) for v in data['videos']}
    counts={v:Counter() for v in ('Base','ClsOnly','BoxOnly','Full')}
    byvideo={v:defaultdict(Counter) for v in counts}
    by_class_lost=Counter();by_video_lost=Counter();by_camera_lost=Counter();by_size_lost=Counter()
    tags=Counter();tag_by_video=defaultdict(Counter)
    tp_shapley=Counter()
    gsoft={task:{e: {'n':0,'above':0,'min':float('inf'),'max':float('-inf'),'sum':0.0,'near_bin':0} for e in EXPERTS}
           for task in ('cls','box')}
    lv_op_counts=Counter();affected=[];nframes=0;seen=set()
    # Exact outputs: do not infer membership from score rank (especially on ties).
    exact_causes=Counter();cause_by_video=defaultdict(Counter);cause_by_class=defaultdict(Counter)
    lost_all=[];rescued_all=[];retained_switched=Counter()
    lv_top100=Counter();lv_by_video=defaultdict(Counter)
    cause_flag_count=Counter();branch_loss=Counter()
    margin_bands=Counter();score_shift_bins=Counter()
    def margin_band(score):
        d=abs(float(score)-0.20)
        if d<.005:return '<0.005'
        if d<.02:return '0.005-0.02'
        if d<.05:return '0.02-0.05'
        return '>=0.05'
    def score_bin(delta):
        if delta < -.1:return '<-0.1'
        if delta < -.02:return '[-0.1,-0.02)'
        if delta < -.005:return '[-0.02,-0.005)'
        if delta < 0:return '[-0.005,0)'
        if delta == 0:return '0'
        if delta <= .005:return '(0,0.005]'
        if delta <= .02:return '(0.005,0.02]'
        if delta <= .1:return '(0.02,0.1]'
        return '>0.1'

    with torch.no_grad():
        for row in manifest['videos']:
            vid=int(row['video_id'])
            paths={kind:r13.safe_path(r13.VAL_ROOT,row[kind]['path']) for kind in ('canonical','spatial','temporal')}
            can=torch.load(paths['canonical'],map_location='cpu',weights_only=False)
            spa=torch.load(paths['spatial'],map_location='cpu',weights_only=False)
            tmp=torch.load(paths['temporal'],map_location='cpu',weights_only=False)
            ids=r13.check_records(vid,row,can,spa,tmp,images,annotations)
            require(not(set(ids)&seen),'Duplicate Validation frame')
            seen.update(ids)
            for idx,fr in enumerate(can):
                iid=int(fr['image_id']);cam=video_cameras[vid]
                O,X=r9.s0a.make_O_X(fr,spa[idx],tmp[idx],CO,FX)
                zf,bf,info=d.production_forward(model,O,X)
                variants={'Base':(O.z,O.b),'ClsOnly':(zf,O.b),'BoxOnly':(O.z,bf),'Full':(zf,bf)}
                gtb=fr['gt_boxes_norm'].float();gtl=fr['gt_labels'].long()
                results={}
                for v,(z,b) in variants.items():
                    pr,mt=compute_match(r8,z,b,gtb,gtl)
                    results[v]=(pr,mt)
                    for metric in ('tp','fp','fn'):
                        counts[v][metric]+=int(mt[metric]);byvideo[v][vid][metric]+=int(mt[metric])
                nframes+=1
                prb,mb=results['Base'];prf,mf=results['Full']
                # Exact flatten-index set from the very same topk implementation used by evaluator.
                keyB={(int(q),int(c)) for q,c in zip(prb['all_q_idx'].tolist(),prb['all_c_idx'].tolist())}
                keyF={(int(q),int(c)) for q,c in zip(prf['all_q_idx'].tolist(),prf['all_c_idx'].tolist())}
                rankF={(int(q),int(c)):i+1 for i,(q,c) in enumerate(zip(prf['all_q_idx'].tolist(),prf['all_c_idx'].tolist()))}
                # LV has zero Val positives; its predicted pairs consume Top-100 slots.
                lv_index=next((i for i,name in category_names.items() if name.upper().replace(' ','_')=='LEISURE_VEHICLE'),None)
                require(lv_index is not None,'No LEISURE_VEHICLE category in Val categories')
                lv_op_counts['Base']+=int((prb['op_c_idx']==lv_index).sum())
                lv_op_counts['Full']+=int((prf['op_c_idx']==lv_index).sum())
                for variant,pred in (('Base',prb),('Full',prf)):
                    ct=int((pred['all_c_idx']==lv_index).sum())
                    lv_top100[variant]+=ct
                    lv_by_video[str(vid)][variant]+=ct

                lost=mb['matched_gt_indices']-mf['matched_gt_indices']
                cgood=results['ClsOnly'][1]['matched_gt_indices']
                bgood=results['BoxOnly'][1]['matched_gt_indices']
                for g in sorted(lost):
                    gcid=int(gtl[g]);label=category_names[gcid]
                    by_class_lost[label]+=1;by_video_lost[vid]+=1;by_camera_lost[cam]+=1
                    scale=r8.gt_scale_name(gtb[g],float(images[iid]['width']),float(images[iid]['height']))
                    by_size_lost[scale]+=1
                    same_cls=g in cgood;same_box=g in bgood
                    f={'classification_only_loses':not same_cls,'box_only_loses':not same_box,
                       'combined_only_loss':same_cls and same_box}
                    det=mb['gt_to_det'][g];q=int(det[3]);cc=int(det[4]);base_score=float(det[2]);
                    sf=float(zf[q,cc].sigmoid());rank=int((zf.sigmoid().flatten()>sf).sum())
                    f['baseline_pair_drops_below_score']=sf<.2
                    f['baseline_pair_below_top100_approx']=rank>=100
                    if (not f['baseline_pair_drops_below_score'] and not f['baseline_pair_below_top100_approx']):
                        iou=float(r8.box_iou_norm_cxcywh(bf[q].detach().cpu().tolist(),gtb[g].tolist()))
                        f['same_query_box_falls_below_iou']=iou<.3
                        f['greedy_or_competition_possible']=iou>=.3
                    # Primary attribution is a deterministic DECISION-CHAIN stage, not a
                    # mechanistic causal claim. Overlap flags are logged independently.
                    exact_pair=(q,cc) in keyF
                    threshold_survives=(sf>=.2)
                    iou_full=float(r8.box_iou_norm_cxcywh(bf[q].detach().float().cpu().tolist(),gtb[g].tolist()))
                    iou_base=float(det[1])
                    q_selected=any(qq==q for qq,_ in keyF)
                    all_class_scores=zf[q].detach().float().sigmoid().cpu()
                    best_comp_idx=int(all_class_scores.argmax().item())
                    best_comp_score=float(all_class_scores[best_comp_idx])
                    primary=decision_stage(exact_pair,threshold_survives,iou_full>=.3)
                    exact_causes[primary]+=1;cause_by_video[str(vid)][primary]+=1
                    cause_by_class[label][primary]+=1
                    cause_flag_count['score_below_.20']+=not threshold_survives
                    cause_flag_count['pair_outside_top100']+=not exact_pair
                    cause_flag_count['q_absent_from_top100']+=not q_selected
                    cause_flag_count['box_iou_below_.30']+=iou_full<.3
                    cause_flag_count['top_class_changed']+=best_comp_idx!=cc
                    branch_loss['cls_only_lost']+=not same_cls
                    branch_loss['box_only_lost']+=not same_box
                    branch_loss['both_single_branch_preserve']+=same_cls and same_box
                    margin_bands[margin_band(base_score)]+=1
                    score_shift_bins[score_bin(sf-base_score)]+=1
                    lost_all.append({'video':vid,'camera':cam,'image_id':iid,
                         'gt_index':g,'gt_class':label,'gt_scale':scale,
                         'primary_decision_stage':primary,
                         'baseline_query':q,'baseline_class_index':cc,
                         'baseline_score':base_score,'full_same_pair_score':sf,
                         'delta_score':sf-base_score,'full_exact_pair_top100':exact_pair,
                         'full_exact_pair_top100_rank_1indexed':rankF.get((q,cc)),
                         'full_query_appears_top100':q_selected,
                         'full_same_pair_score_passes_op':threshold_survives,
                         'baseline_iou':iou_base,'full_same_query_iou':iou_full,
                         'full_query_top_class':category_names.get(best_comp_idx,str(best_comp_idx)),
                         'full_query_top_class_score':best_comp_score,
                         'classification_only_gt_retained':same_cls,
                         'box_only_gt_retained':same_box})
                    for tag,yes in f.items():
                        if yes:tags[tag]+=1;tag_by_video[str(vid)][tag]+=1
                    # Example IDs strictly for forensic traceability, not model selection.
                    if len(affected)<200:
                        affected.append({'video':vid,'image_id':iid,'gt_index':g,'class':label,'scale':scale,
                                         'base_query':q,'base_class':cc,'base_score':base_score,
                                         'same_pair_score_full':sf,'same_pair_rank_full_approx':rank,'flags':f})
                # Rescue identities, and identities retained with a changed matching query.
                new_gt=mf['matched_gt_indices']-mb['matched_gt_indices']
                for g in sorted(new_gt):
                    det=mf['gt_to_det'][g]
                    rescued_all.append({'video':vid,'camera':cam,'image_id':iid,
                        'gt_index':g,'gt_class':category_names[int(gtl[g])],
                        'full_query':int(det[3]),'full_class_index':int(det[4]),
                        'full_score':float(det[2]),'full_iou':float(det[1]),
                        'rescue_cls_only':g in cgood,'rescue_box_only':g in bgood})
                for g in (mb['matched_gt_indices'] & mf['matched_gt_indices']):
                    d1=mb['gt_to_det'][g];d2=mf['gt_to_det'][g]
                    if (int(d1[3]),int(d1[4]))!=(int(d2[3]),int(d2[4])):
                        retained_switched['match_pair_changed']+=1
                # Read-only matched-soft gate debug; invariant with production masks.
                bundle=r9.expert_bundle(model,O,X)
                soft=r9.mechanism_forward(model,O,X,bundle,hard=False)
                for task in ('cls','box'):
                    amp=soft['g_'+task];hard=soft['hard_'+task];avail=soft['A_'+task]
                    real=info['m_'+task]
                    require(torch.equal(real,hard),'Production vs soft diagnostic hard-mask disagreement')
                    for j,e in enumerate(EXPERTS):
                        subset=amp[:,j][avail[:,j]>0].detach().cpu().float()
                        s=gsoft[task][e]
                        if not subset.numel():continue
                        t=float(frozen[task][e]);bin_index=int(t*10000+1e-8)
                        s['n']+=len(subset);s['above']+=int((subset>t).sum())
                        s['sum']+=float(subset.sum());s['min']=min(s['min'],float(subset.min()))
                        s['max']=max(s['max'],float(subset.max()))
                        s['near_bin']+=int(((subset>t)&(subset<(bin_index+1)/10000)).sum())
                del O,X,zf,bf,info,bundle,soft
            print(f'[+] RUN14B retrospective video {vid}/9 complete; frames={len(can)}',flush=True)
            del can,spa,tmp
            gc.collect()
            if r9.DEVICE.type=='cuda':torch.cuda.empty_cache()
    require(len(seen)==4266 and nframes==4266,'Validation replay frame census drift')
    for v in ('Base','Full'):
        original=base_report['variants'][v]
        for key in ('tp','fp','fn'):
            require(counts[v][key]==original[key],f'{v} {key} differs from frozen RUN13 benchmark')
        for vid in range(1,10):
            for key in ('tp','fp','fn'):
                require(byvideo[v][vid][key]==original['by_video_operational'][str(vid)][key],
                        f'{v} video={vid} {key} differs from frozen RUN13')
    for variant in ('Base','ClsOnly','BoxOnly','Full'):
        for metric in ('tp','fp','fn'):
            require(counts[variant][metric]==reference14['counts'][variant][metric],
                    f'RUN14 {variant}/{metric} differs from certified forensic reference')
    for name,observed,ref in (
        ('camera',by_camera_lost,reference14['lost_by_camera']),
        ('scale',by_size_lost,reference14['lost_by_size'])):
        require(dict(observed)=={str(k):int(v) for k,v in ref.items()},
                f'RUN14 loss-by-{name} evidence mismatch')
    # Forensic chain must reproduce the ORIGINAL RUN14 full loss and branch counts.
    require(len(lost_all)==REFERENCE_GT_LOSSES,'GT loss trace incomplete')
    require(len(rescued_all)==178,'GT rescue trace incomplete')
    require(sum(exact_causes.values())==REFERENCE_GT_LOSSES,'Nonexclusive primary causes or missing losses')
    require(dict(by_video_lost)=={int(k):int(v) for k,v in reference14['lost_by_video'].items()},
            'RUN14 per-video lost GT conservation failed')
    require(dict(by_class_lost)=={str(k):int(v) for k,v in reference14['lost_by_class'].items()},
            'RUN14 per-class lost GT conservation failed')
    require(dict(lv_op_counts)==reference14['leisure_vehicle_operational_predictions_no_gt'],
            'RUN14 LV operating prediction count changed')
    # Exact post-hoc 2-factor Shapley decomposition of TP at four output corners.
    Y=lambda v:counts[v]['tp']
    tp_shapley['classification']=(Y('ClsOnly')-Y('Base')+Y('Full')-Y('BoxOnly'))/2
    tp_shapley['boxes']=(Y('BoxOnly')-Y('Base')+Y('Full')-Y('ClsOnly'))/2
    interaction=Y('Full')-Y('ClsOnly')-Y('BoxOnly')+Y('Base')
    require(tp_shapley['classification']+tp_shapley['boxes']==Y('Full')-Y('Base'),
            'TP Shapley conservation failure')
    for task in gsoft:
        for e,s in gsoft[task].items():
            if s['n']:
                s['mean']=s['sum']/s['n'];s['active_fraction']=s['above']/s['n']
            else:
                s['min']=None;s['max']=None;s['mean']=None;s['active_fraction']=None
            s['threshold']=float(frozen[task][e]);del s['sum']
    for task in ('cls','box'):
        for ex in EXPERTS:
            baseline=reference14['g_soft_by_expert'][task][ex]
            for k in ('n','above'):
                require(gsoft[task][ex][k]==baseline[k],
                        f'RUN14 diagnostic {task}/{ex}/{k} count drift')
    report={'schema':'run14b_exact_decision_chain_v1',
            'forensic_only_not_independent_evaluation':True,'test_accessed':False,
            'frozen_final_sha256':r13.SHA_FINAL,'run13_report_sha256':RUN13_SHA,
            'run13_source_sha256':RUN13_SOURCE,'run14_source_sha256':sha(Path(__file__)),
            'counts':{k:dict(v) for k,v in counts.items()},
            'video_counts':{k:{str(i):dict(v) for i,v in val.items()} for k,val in byvideo.items()},
            'tp_shapley':dict(tp_shapley),'tp_two_factor_interaction':interaction,
            'lost_gt_total':len(mb['matched_gt_indices']) if False else sum(by_video_lost.values()),
            'lost_by_video':dict(by_video_lost),'lost_by_class':dict(by_class_lost),
            'lost_by_camera':dict(by_camera_lost),'lost_by_size':dict(by_size_lost),
            'lost_flags_nonexclusive':dict(tags),'lost_flag_by_video_nonexclusive':dict(tag_by_video),
            'g_soft_by_expert':gsoft,'leisure_vehicle_operational_predictions_no_gt':dict(lv_op_counts),
            'first_200_lost_gt_traces':affected,
            'limitations':['Flags are nonexclusive, and baseline-pair rank is approximate on ties',
                           'Output branch 2x2 interventions do not identify upstream individual expert effects',
                           'No checkpoint training, threshold selection or Test access',
                           'All Validation replay output is post-hoc forensic, never a second independent validation']}
    report['run14_reference_sha256']=RUN14_SHA
    report['forensic_upgrade']='RUN14B exact top100 (query,class) and exclusive decision-chain stages'
    report['exclusive_primary_decision_stage']=dict(exact_causes)
    report['primary_by_video']={k:dict(v) for k,v in cause_by_video.items()}
    report['primary_by_class']={k:dict(v) for k,v in cause_by_class.items()}
    report['overlapping_exact_flags']=dict(cause_flag_count)
    report['single_output_branch_loss_flags']=dict(branch_loss)
    report['baseline_score_margin_bands']=dict(margin_bands)
    report['full_minus_base_score_shift_bins']=dict(score_shift_bins)
    report['retained_gt_matching_pair_swaps']=dict(retained_switched)
    report['lv_top100_candidate_entries_no_gt']=dict(lv_top100)
    report['lv_top100_by_video']={k:dict(v) for k,v in lv_by_video.items()}
    report['all_lost_gt_traces_file']='run14b_all_755_lost_gt_traces.csv'
    report['all_rescued_gt_traces_file']='run14b_all_178_rescued_gt_traces.csv'
    report['causal_limits']=[
        'Primary reasons classify survival of the original baseline pair. Pair eviction can combine score and rank changes.',
        'GT match changes also depend on competing detections and greedy assignment; attribution is not an independent causal explanation.',
        'Four output-corner interventions identify cls/box effects, not individual expert or gate cause.',
        'Do NOT use any Validation diagnostic to select a new model, threshold, checkpoint, or architecture. Test sealed.'
    ]
    OUT.mkdir(parents=True,exist_ok=False)
    p=OUT/'run14_readonly_causal_attribution.json'
    p.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    # Complete GT ledgers (not the previous first-200 encounter-order sample).
    for filename,rows in (('run14b_all_755_lost_gt_traces.csv',lost_all),
                          ('run14b_all_178_rescued_gt_traces.csv',rescued_all)):
        require(rows,'Empty forensic GT-identity ledger')
        with (OUT/filename).open('w',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
            w.writeheader();w.writerows(rows)
    print('[PASS] RUN14B replay agrees with original RUN13 and RUN14 per-video/class GT outcomes')
    print('Exact decision-chain stage (not mechanistic root cause):',dict(exact_causes))
    print('All 755 loss identities + 178 rescues serialized; no Test opened')
    print('TP attribution:',dict(tp_shapley),'interaction',interaction)
    print('Lost GT:',sum(by_video_lost.values()),'by video',dict(by_video_lost))
    print('Gate S box:',gsoft['box']['S'])
    print('Audit:',p,'SHA:',sha(p))
    print('STOP: scientific review required. No further Validation model selection. TEST SEALED.')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',required=True,choices=('selftest','forensic'))
    a=parser.parse_args()
    selftest()
    if a.phase=='forensic':run()
