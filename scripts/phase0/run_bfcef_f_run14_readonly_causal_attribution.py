from __future__ import annotations
"""RUN14 Stage 1: frozen, read-only retrospective Validation attribution.
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
OUT=EVAL/'bfcef_f_s0'/'run14_readonly_causal_attribution_v1'
EXPERTS=('S','L','C','T')


def require(ok,msg):
    if not ok:raise AssertionError('[FATAL] '+msg)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def selftest():
    # Demonstrates documented RUN09C quantization/inference boundary mismatch.
    bins=10000;g=.00205;t=.0020
    require(int(g*bins)==int(t*bins) and g>t,'Bin semantic fixture changed')
    require(not(int(g*bins)>=int(t*bins)+1),'Histogram discrepancy fixture not reproduced')
    # Two-branch exact Shapley conservation.
    yb,yc,yl,yf=10,7,12,9
    sc=((yc-yb)+(yf-yl))/2;sb=((yl-yb)+(yf-yc))/2
    require(sc+sb==yf-yb,'Shapley sum mismatch')
    print('[PASS] RUN14 numerical selftests; no dataset or checkpoint opened')


def compute_match(r8,z,b,gtb,gtl):
    pred=r8.extract_top100_predictions(z.detach().float().cpu(),b.detach().float().cpu())
    m=r8.match_operational_detections_to_gt(pred['op_scores'],pred['op_c_idx'],pred['op_boxes'],pred['op_q_idx'],gtb,gtl)
    return pred,m


def run():
    require(not OUT.exists(),'RUN14 audit output already exists; refuse overwrite')
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
                # LV has zero Val positives; its predicted pairs consume Top-100 slots.
                lv_index=next((i for i,name in category_names.items() if name.upper().replace(' ','_')=='LEISURE_VEHICLE'),None)
                require(lv_index is not None,'No LEISURE_VEHICLE category in Val categories')
                lv_op_counts['Base']+=int((prb['op_c_idx']==lv_index).sum())
                lv_op_counts['Full']+=int((prf['op_c_idx']==lv_index).sum())
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
                    for tag,yes in f.items():
                        if yes:tags[tag]+=1;tag_by_video[str(vid)][tag]+=1
                    # Example IDs strictly for forensic traceability, not model selection.
                    if len(affected)<200:
                        affected.append({'video':vid,'image_id':iid,'gt_index':g,'class':label,'scale':scale,
                                         'base_query':q,'base_class':cc,'base_score':base_score,
                                         'same_pair_score_full':sf,'same_pair_rank_full_approx':rank,'flags':f})
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
            print(f'[+] RUN14 retrospective video {vid}/9 complete; frames={len(can)}',flush=True)
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
    report={'schema':'run14_readonly_attribution_v1',
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
    OUT.mkdir(parents=True,exist_ok=False)
    p=OUT/'run14_readonly_causal_attribution.json'
    p.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    print('[PASS] RUN14 post-hoc replay agrees EXACTLY with original RUN13 Base and Full video metrics')
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
