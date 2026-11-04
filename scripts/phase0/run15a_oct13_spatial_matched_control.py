"""OCT13 TRAIN-only matched-control spatial diagnostic.

RESEARCH ONLY. Cached decoder O.b/O.z, GT COCO, no new model forward,
no training, no inference-time policies, no access to EVAL/TUNE/CAL/QUAL/Test.
Decoded-box IoU and decoded-center contrasts are outcome-definition confounded, NOT
independent evidence for a query-placement mechanism. O.r may carry cached reference
coordinates; identify and describe them without asserting learned decoder semantics.
The output is descriptive and cannot authorize a new architecture.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, math
from pathlib import Path
import numpy as np
import torch
import run15a_oct13_fn_recoverability as f

FN_SCRIPT_SHA='9df5615a70a229dcdc7d7bf7af7846fb7ae6a005a012e07e7a8784fba14597fa'
FN_REPORT_SHA='358c8d277a3ef354477c0b7cea7afd3fffd6c4a6960281f5d8ad063aec67c5c3'
SCHEMA='oct13_train_spatial_matched_control_v3_claude_review'
TRAIN=(2,3,4,5,7,8,13,14,16,17,18,19,20,24,25,31,35)
MIN_GEOMETRY_IOU=.995
NEAR_LO=.25; NEAR_HI=.30
CLASS_NAMES=('ANIMAL','BOAT','MARINE_MARKER','OBJECT','LEISURE_VEHICLE','SAILING_VESSEL','SHIP')


def require(cond,why):
    if not cond: raise AssertionError('[SPATIAL DIAGNOSTIC HOLD] '+str(why))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for part in iter(lambda: stream.read(1<<20),b''): h.update(part)
    return h.hexdigest()


def paths(project,require_unused=True):
    p=Path(project)
    require(sha(f.__file__)==FN_SCRIPT_SHA,'FN diagnostic dependency/source drift')
    require(tuple(f.a.TRAIN)==TRAIN,'TRAIN partition drift')
    require(set(TRAIN).isdisjoint(f.a.TUNE) and set(TRAIN).isdisjoint(f.a.EVAL),'TRAIN partition overlap')
    src=p/'bfcef_f_s0'/'run15a_final_candidate_sandbox'/'oct13_recoverability_gate'/'train_fn_recoverability.json'
    require(src.exists() and sha(src)==FN_REPORT_SHA,'Immutable OCT13 FN receipt missing/drifted')
    d=json.loads(src.read_text(encoding='utf-8'))
    require(d['status']=='STOP_INSUFFICIENT_FN_OPPORTUNITY','Unexpected prior result')
    require(d['TRAIN_only'] and len(d['fn_events'])==271,'Historical FN count/scope drift')
    f.pathscope(p,unused_output=False)
    target=p/'bfcef_f_s0'/'run15a_final_candidate_sandbox'/'oct13_spatial_matched_control'
    if require_unused:
        for leaf in ('train_spatial_diagnostic.json','train_spatial_units.csv'):
            require(not (target/leaf).exists() and not (target/(leaf+'.tmp')).exists(),
                    f'Existing one-shot output {leaf}')
    return d,target


def coco_annotations_for_image(raw):
    by=collections.defaultdict(list)
    for an in raw['annotations']:
        by[int(an['image_id'])].append(an)
    return by


def xywh_to_norm(box,w,h):
    x,y,bw,bh=[float(t) for t in box]
    return [ (x+bw/2)/w, (y+bh/2)/h, bw/w, bh/h ]


def validate_coco_alignment(gt,labels,anns,meta):
    """One-to-one class/geometry multiset matching; do NOT assume COCO annotation order."""
    w,h=int(meta['width']),int(meta['height'])
    require(w>0 and h>0,'No original source-image dimensions')
    require(len(gt)==len(anns),'Canonical versus COCO GT count differs')
    candidates=[]
    for an in anns:
        c=int(an['category_id'])-1
        require(0<=c<7,'Unexpected COCO seven-class mapping')
        candidates.append((int(an['id']),c,xywh_to_norm(an['bbox'],w,h)))
    remaining=set(range(len(candidates)))
    result={}
    for g in range(len(gt)):
        same=[i for i in remaining if candidates[i][1]==int(labels[g])]
        require(bool(same),'COCO-versus-cache class mismatch')
        overlaps=f.box_ious(np.asarray([candidates[i][2] for i in same]),gt[g:g+1]).ravel()
        ix=int(np.argmax(overlaps));ci=same[ix]
        require(float(overlaps[ix])>=MIN_GEOMETRY_IOU,
                f'Canonical GT to COCO bbox failed: gt_idx={g}, max_iou={float(overlaps[ix]):.5f}')
        remaining.remove(ci)
        result[g]={'annotation_id':candidates[ci][0],'coco_label':candidates[ci][1]}
    require(not remaining,'COCO annotations remain unmatched')
    return result



def inspect_O_r(O, cached_frame):
    """Check O.r against the authoritative cached `ref_pts` field.

    Identity, shape, finiteness and [0,1] range are verified; this does NOT
    certify that coordinates are learned decoder references or feature priors.
    """
    details={'status':'O_R_UNVERIFIED','O_r':None,'O_z':None,'O_b':None,
             'cached_ref_pts':None,'max_abs_difference':None}
    def describe_tensor(x):
        if not isinstance(x,torch.Tensor):
            return {'type':type(x).__name__,'shape':list(x.shape) if hasattr(x,'shape') else None}
        arr=x.detach().float().cpu().numpy()
        return {'type':type(x).__name__, 'dtype':str(x.dtype), 'shape':list(x.shape),
                'finite':bool(np.isfinite(arr).all()),
                'min':float(np.min(arr)) if arr.size else None,
                'max':float(np.max(arr)) if arr.size else None}
    for key,att in (('O_r','r'),('O_z','z'),('O_b','b')):
        details[key]=describe_tensor(getattr(O,att,None))
    r=getattr(O,'r',None);cached=cached_frame.get('ref_pts')
    details['cached_ref_pts']=describe_tensor(cached)
    if not isinstance(r,torch.Tensor) or not isinstance(cached,torch.Tensor):
        details['status']='O_R_OR_CACHED_REF_PTS_ABSENT';return details,None
    ra=r.detach().float().cpu().numpy();ca=cached.detach().float().cpu().numpy()
    if ra.ndim!=2 or ra.shape!=(300,2) or ca.shape!=(300,2):
        details['status']='UNEXPECTED_REFERENCE_SHAPE';return details,None
    if not np.isfinite(ra).all() or not np.isfinite(ca).all():
        details['status']='NONFINITE_REFERENCE_COORDINATES';return details,None
    details['max_abs_difference']=float(np.max(np.abs(ra-ca)))
    if not np.array_equal(ra,ca):
        details['status']='O_R_DIFFERS_FROM_CACHED_REF_PTS';return details,None
    if np.any(ra<0) or np.any(ra>1):
        details['status']='REFERENCE_OUTSIDE_NORMALIZED_IMAGE';return details,None
    details['status']='O_R_EQUALS_CACHED_REF_PTS_COORDINATE_IDENTITY_ONLY'
    return details,ra


def native_unit(vid,j,iid,g,is_tp,label,gt,boxes,logits,ann_id,meta,density,refs=None):
    """GT-blind geometry features, with GT used only for offline diagnosis."""
    overlap=f.box_ious(boxes,gt[None,:])[:,0]
    q=int(np.argmax(overlap));iou=float(overlap[q]);px,py,pw,ph=[float(x) for x in boxes[q]]
    gx,gy,gw,gh=[float(x) for x in gt]
    require(gw>1e-8 and gh>1e-8,'Degenerate GT dimensions')
    sw,sh=int(meta['width']),int(meta['height'])
    score=1/(1+np.exp(-np.clip(logits,-40,40)))
    signed_dx=(px-gx)/gw;signed_dy=(py-gy)/gh
    # Bounding-box centers are NOT decoder learned/deformable reference points.
    normalized_distance=np.sqrt(((boxes[:,0]-gx)/gw)**2+((boxes[:,1]-gy)/gh)**2)
    center_density=int(np.sum(normalized_distance<=density))
    reference_density=None
    if refs is not None:
        require(refs.shape==(300,2), 'verified references must be 300x2')
        rdist=np.sqrt(((refs[:,0]-gx)/gw)**2+((refs[:,1]-gy)/gh)**2)
        reference_density=int(np.sum(rdist<=density))
    best_rank=int(np.sum(score.reshape(-1)>score[q].max()))+1
    return {'video':int(vid),'frame':int(j),'image_id':int(iid),'gt_index':int(g),
            'annotation_id':int(ann_id),'class':int(label),'class_name':CLASS_NAMES[int(label)],
            'is_tp':int(is_tp),'camera':str(meta.get('camera_sensor','UNKNOWN')),
            'sea_state':str(meta.get('sea_state','UNKNOWN')),
            'source_w':sw,'source_h':sh,
            'gt_w_source_px':float(gw*sw),'gt_h_source_px':float(gh*sh),
            'gt_area_source_px2':float(gw*sw*gh*sh),
            'gt_area_norm':float(gw*gh),
            'best_raw_iou':iou,'best_raw_query':q,
            'best_raw_query_class':int(np.argmax(score[q])),'best_raw_query_max_score':float(score[q].max()),
            'best_raw_query_score_for_gt_class':float(score[q,int(label)]),
            'best_query_anyclass_rank':best_rank,
            'signed_center_x_over_gt_w':float(signed_dx),'signed_center_y_over_gt_h':float(signed_dy),
            'log_pred_w_over_gt_w':float(math.log(max(pw,1e-12)/gw)),
            'log_pred_h_over_gt_h':float(math.log(max(ph,1e-12)/gh)),
            'decoded_box_center_density_1gt':center_density,
            'cached_refpoint_density_1gt':reference_density,
            'query_geometry_inference_status':'SELECTION_CONFOUNDED_NOT_INFERENTIAL',
            'gt_count_in_frame':int(len(meta.get('_gt_for_frame',[]))),
            'input_pixel_size_status':'UNVERIFIED_CANONICAL_TRANSFORM',
            'decoder_refpoint_density_status':('CACHED_REF_PTS_COORDINATE_IDENTITY_SEMANTICS_UNVERIFIED' if refs is not None else 'NOT_COMPUTED_O_R_UNVERIFIED')}


def compare_fn_report(actual,prior):
    """Full identity/anatomy/max-IoU parity against historical FN report."""
    key=lambda r:(int(r['video']),int(r['frame']),int(r['image_id']),int(r['g']))
    aa={key(r):r for r in actual};pp={key(r):r for r in prior}
    require(len(aa)==len(actual)==len(pp)==len(prior)==271,'FN identity coverage differs')
    for k,r in aa.items():
        require(k in pp,'New FN key absent in prior')
        p=pp[k]
        require(r['class']==p['class'] and r['type']==p['type'],'FN label/anatomy mismatch')
        require(abs(r['max_box_iou_raw300']-p['max_box_iou_raw300'])<=1e-5,
                'Decoder geometry parity mismatch')
    return True


def match_controls(rows,fn_per_unit=3):
    """Same-video, same-class GT area nearest-neighbor matching; reuse disclosed.

    Matched items are descriptive pairs, NOT independent observations. We report
    unique control counts and refrain from a pseudo-replicated confidence interval.
    """
    tps=[r for r in rows if r['is_tp']==1]; fns=[r for r in rows if r['is_tp']==0]
    by=collections.defaultdict(list)
    for r in tps:by[(r['video'],r['class'])].append(r)
    pairs=[];unmatched=[]
    for m in fns:
        options=by.get((m['video'],m['class']),[])
        if not options:
            unmatched.append((m['video'],m['frame'],m['gt_index']));continue
        options=sorted(options,key=lambda t:(abs(math.log(max(m['gt_area_source_px2'],1e-9)/max(t['gt_area_source_px2'],1e-9))),
                                            abs(m['frame']-t['frame']),t['image_id'],t['gt_index']))
        # Factor-of-two source-area caliper; otherwise genuinely unmatched.
        options=[t for t in options if abs(math.log(m['gt_area_source_px2']/t['gt_area_source_px2']))<=math.log(2)]
        if not options:
            unmatched.append((m['video'],m['frame'],m['gt_index']));continue
        for t in options[:fn_per_unit]:
            pairs.append({'fn_key':(m['video'],m['frame'],m['gt_index']),
                          'tp_key':(t['video'],t['frame'],t['gt_index']),
                          'log_area_ratio_fn_to_tp':float(math.log(m['gt_area_source_px2']/t['gt_area_source_px2'])),
                          'fn_area':m['gt_area_source_px2'],'tp_area':t['gt_area_source_px2'],
                          'fn_iou':m['best_raw_iou'],'tp_iou':t['best_raw_iou'],
                          'fn_density':m['decoded_box_center_density_1gt'],
                          'tp_density':t['decoded_box_center_density_1gt']})
    return pairs,unmatched


def paired_density_contrasts(pairs):
    """Average within each FN before video aggregation; no pseudo-independent pairs."""
    by_fn=collections.defaultdict(list)
    for x in pairs:by_fn[tuple(x['fn_key'])].append(x)
    by_video=collections.defaultdict(list)
    for fk,items in by_fn.items():
        density=float(items[0]['fn_density'])-float(np.mean([p['tp_density'] for p in items]))
        by_video[fk[0]].append(density)
    res={str(v):{'matched_fn_n':len(d), 'mean_density_difference_FN_minus_controls':float(np.mean(d)),
                 'median_density_difference':float(np.median(d))}
         for v,d in sorted(by_video.items())}
    without18=[float(np.mean(d)) for v,d in by_video.items() if v!=18 and len(d)>=3]
    # Neither inference from a selection-defined geometry statistic nor a
    # percentile interval from only ~5 videos is defensible. No bootstrap.
    return {'by_video':res,
            'non18_equal_video_mean':float(np.mean(without18)) if without18 else None,
            'non18_video_count':len(without18),
            'non18_video_bootstrap_interval':None,
            'interval_status':'INSUFFICIENT_CLUSTERS_AND_SELECTION_CONFOUNDED',
            'inference_status':'SELECTION_CONFOUNDED_NOT_INFERENTIAL',
            'video18_interval':'NOT_ESTIMABLE_AS_VIDEO_CLUSTER_BOOTSTRAP_N1'}

def descriptive_analysis(rows,pairs):
    fn=[x for x in rows if not x['is_tp']];tp=[x for x in rows if x['is_tp']]
    def stats(items,key):
        vals=[float(x[key]) for x in items if x.get(key) is not None]
        return {'n':len(vals),'median':float(np.median(vals)) if vals else None,
                'q10':float(np.quantile(vals,.1)) if vals else None,
                'q90':float(np.quantile(vals,.9)) if vals else None}
    groups={}
    for label,subset in [('ALL',rows),('VIDEO18',[x for x in rows if x['video']==18]),
                         ('EXCLUDING18',[x for x in rows if x['video']!=18])]:
        groups[label]={
            'fn_n':sum(not x['is_tp'] for x in subset),'tp_n':sum(x['is_tp'] for x in subset),
            'fn_size_source_px2':stats([x for x in subset if not x['is_tp']],'gt_area_source_px2'),
            'tp_size_source_px2':stats([x for x in subset if x['is_tp']],'gt_area_source_px2'),
            'fn_best_raw_iou':stats([x for x in subset if not x['is_tp']],'best_raw_iou'),
            'tp_best_raw_iou':stats([x for x in subset if x['is_tp']],'best_raw_iou'),
        }
    by=collections.defaultdict(lambda:{'FN':[],'TP':[]})
    for r in rows:by[(r['video'],r['class'])]['TP' if r['is_tp'] else 'FN'].append(r)
    class_video=[]
    for (video,c),data in sorted(by.items()):
        ff=data['FN'];tt=data['TP']
        if not ff and not tt:continue
        item={'video':video,'class':c,'class_name':CLASS_NAMES[c],
              'fn_n':len(ff),'tp_n':len(tt),'miss_rate':len(ff)/(len(ff)+len(tt)),
              'fn_size':stats(ff,'gt_area_source_px2'),
              'tp_size':stats(tt,'gt_area_source_px2'),
              'fn_query_center_density':stats(ff,'decoded_box_center_density_1gt'),
              'tp_query_center_density':stats(tt,'decoded_box_center_density_1gt'),
              'fn_best_raw_iou':stats(ff,'best_raw_iou'),
              'tp_best_raw_iou':stats(tt,'best_raw_iou'),
              'fn_near_misses_025_030':sum(NEAR_LO<=f['best_raw_iou']<NEAR_HI for f in ff),
              'query_geometry_inference_status':'SELECTION_CONFOUNDED_NOT_INFERENTIAL'}
        # Descriptive rule for H-S1 ONLY, not a GO; 3 independent videos required,
        # 2 outside video18, plus verification of effective input pixel geometry.
        item['source_pixel_fn_median_below_tp_p10']=(
          len(ff)>=3 and len(tt)>=3 and item['fn_size']['median']<item['tp_size']['q10'])
        class_video.append(item)
    matched_stats={'pairs':len(pairs),
                   'unique_fn':len(set(tuple(x['fn_key']) for x in pairs)),
                   'unique_tp':len(set(tuple(x['tp_key']) for x in pairs)),
                   'fn_without_same_class_video_controls':len(fn)-len(set(tuple(x['fn_key']) for x in pairs)),
                   'median_log_size_ratio':float(np.median([x['log_area_ratio_fn_to_tp'] for x in pairs])) if pairs else None}
    near=[r for r in fn if NEAR_LO<=r['best_raw_iou']<NEAR_HI]
    near_by=collections.Counter(r['video'] for r in near)
    return {'subsets':groups,'class_video':class_video,'matched_controls':matched_stats,
            'paired_query_box_center_density':paired_density_contrasts(pairs),
            'near_misses':{'n':len(near),'per_video':{str(k):v for k,v in sorted(near_by.items())},
                           'signed_x_center':stats(near,'signed_center_x_over_gt_w'),
                           'signed_y_center':stats(near,'signed_center_y_over_gt_h'),
                           'log_w_ratio':stats(near,'log_pred_w_over_gt_w'),
                           'log_h_ratio':stats(near,'log_pred_h_over_gt_h')},
            'query_center_density_note':'Box IoU and decoded-center density are outcome-definition confounded; never use FN/TP differences to claim query coverage.',
            'cached_refpoint_note':'O.r may equal cached ref_pts. Report the tensor identity separately; reference coordinate density is descriptive, semantics unverified and not a learned-query causal test.',
            'allowed_inferential_quantity':'GT_SOURCE_PIXEL_SIZE_ONLY_CONDITIONAL_ON_EFFECTIVE_INPUT_TRANSFORM',
            'geometry_comparison_inference_status':'SELECTION_CONFOUNDED_NOT_INFERENTIAL',
            'inference_status':'DESCRIPTIVE_ONLY_NO_CAUSAL_HYPOTHESIS_ACCEPTED'}


def run(project):
    historical,target=paths(project,True)
    rt=f.a.setup_native(project)
    old,r9,r8,d,s0a,raw,model,CO,FX=rt
    f.require_native_iou(old)
    by_ann=coco_annotations_for_image(raw);meta={int(x['id']):x for x in raw['images']}
    baseline=collections.Counter();out=[];events=[];img_count=0;ref_contract_counts=collections.Counter()
    with torch.no_grad():
        for vid in TRAIN:
            can,spa,tmp=s0a.load_video_triplet(vid)
            indices=list(range(0,len(can),f.a.STRIDE))[:f.a.NFRAMES]
            require(len(indices)==16,f'Expected 16 sampled frames in video {vid}')
            print(f'[SPATIAL CACHE] video={vid}: {len(indices)} frames',flush=True)
            for j in indices:
                src=can[j];iid=int(src['image_id']);info=meta.get(iid)
                require(info is not None and int(info['video_id'])==vid,'Missing source image metadata')
                O,X=s0a.make_O_X(src,spa[j],tmp[j],CO,FX)
                z=O.z.detach().float().cpu().numpy(); b=O.b.detach().float().cpu().numpy()
                gt=src['gt_boxes_norm'].detach().float().cpu().numpy()
                labels=src['gt_labels'].detach().long().cpu().numpy()
                require(z.shape==(300,7) and b.shape==(300,4),'Expected frozen R50 300x7 boxes/logits')
                ref_details,refs=inspect_O_r(O,src)
                ref_contract_counts[ref_details['status']]+=1
                alignment=validate_coco_alignment(gt,labels,by_ann.get(iid,[]),info)
                matches,fps,counts=old.capture_match(O.z,O.b,src['gt_boxes_norm'],src['gt_labels'])
                baseline.update(counts); img_count+=1
                original_matches={int(g) for q,c,g in matches}
                for g in range(len(gt)):
                    record=native_unit(vid,j,iid,g,g in original_matches,int(labels[g]),gt[g],b,z,
                                       alignment[g]['annotation_id'],{**info,'_gt_for_frame':gt},density=1.0,refs=refs)
                    out.append(record)
                fr={'z':O.z.detach().float().cpu(),'b':O.b.detach().float().cpu(),
                    'gt_boxes':src['gt_boxes_norm'].detach().float().cpu(),
                    'gt_labels':src['gt_labels'].detach().long().cpu(),
                    'orig_matches':matches}
                for e in f.fn_anatomy(fr):events.append({'video':vid,'frame':j,'image_id':iid,**e})
            del can,spa,tmp
    require(img_count==272,'Sampled image count mismatch')
    require((baseline['tp'],baseline['fp'],baseline['fn'])==(754,1013,271),
            'Native baseline must exactly reproduce 754/1013/271')
    require(len(out)==1025 and sum(x['is_tp'] for x in out)==754,'GT unit count or TP mismatch')
    compare_fn_report(events,historical['fn_events'])
    pairs,unmatched=match_controls(out)
    report=descriptive_analysis(out,pairs)
    doc={'schema':SCHEMA,'status':'HOLD_FOR_INPUT_RESOLUTION_AND_QUALITATIVE_REVIEW',
         'study':'TRAIN-only cached R50 spatial evidence matched-control measurement',
         'source_sha256':sha(__file__),'pinned_fn_source_sha256':FN_SCRIPT_SHA,
         'pinned_historical_fn_report_sha256':FN_REPORT_SHA,
         'training':False,'new_detector_forward':False,'TRAIN_only':True,
         'images':img_count,'GT_units':len(out),'baseline':dict(baseline),
         'fn_parity':'EXACT_271_AND_EACH_RAW_BOX_IOU',
         'effective_input_pixels':'UNVERIFIED; no numerical assumption from original source pixels',
         'decoder_reference_points':'O.r checked against cached ref_pts; coordinate identity alone does not establish learned decoder semantics',
         'reference_tensor_contract_by_frame':dict(ref_contract_counts),
         'selection_confound_warning':'Every FN/TP contrast in best raw IoU, decoded-center density, or cached reference coordinates is outcome-adjacent or selection-confounded. Not inferential.',
         'statistical_limitations':[
             'Only 17 video clusters, and video 18 is a single cluster; a within-video video bootstrap is invalid.',
             'Matched TP controls can be reused; pairs are not independent observations.',
             'Original detector was trained on official TRAIN, so findings are not independent validation.',
             'Missed detections are correlated through frames and GT tracks.',
             'No source images reviewed; annotation ambiguity cannot be excluded.',
             'The metric is IoU 0.30; tiny coordinate shifts can change FN assignment.',
             'Best raw IoU and decoded box-center density are selection-confounded; do not interpret FN/TP differences as query-coverage mechanism.',
             'O.r tensor identity with cache ref_pts does not independently establish learned decoder-reference semantics.',
             'No percentile bootstrap over ~five non18 videos is presented.',
             'Effective R50 input dimensions cannot be derived from raw image dimensions alone.'
         ],'descriptive_analysis':report,
         'unmatched_fn_keys':unmatched,
         'all_gt_units':out,
         'matched_control_pairs':pairs,
         'next_step':'Review effective detector input transform and 40 stratified raw-image cases before deciding H-S1/H-S2/H-S4.'}
    target.mkdir(parents=True,exist_ok=True)
    csvpath=target/'train_spatial_units.csv';jsonpath=target/'train_spatial_diagnostic.json'
    require(not jsonpath.exists() and not csvpath.exists(),'One-shot report already exists')
    # Two-file staging: ensure no partial reports on exception. If disk failure occurs,
    # tmp file remains and further execution fails until explicitly audited.
    csvtmp=Path(str(csvpath)+'.tmp');jsontmp=Path(str(jsonpath)+'.tmp')
    require(not csvtmp.exists() and not jsontmp.exists(),'Stale partial output exists')
    import csv
    with csvtmp.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(out[0]),extrasaction='ignore')
        writer.writeheader();writer.writerows(out)
    jsontmp.write_text(json.dumps(doc,indent=2,allow_nan=False),encoding='utf-8')
    csvtmp.rename(csvpath);jsontmp.rename(jsonpath)
    print(f'[PARITY PASS] 272 images / 1025 GT / 754 TP / 1013 FP / 271 FN, exact FN identities')
    print(f'[CONTROLS] {len(pairs)} matched comparisons; {report["matched_controls"]["unique_tp"]} unique TP controls')
    print(f'[REPORT] {doc["status"]}: {jsonpath}')
    return doc


def selftest():
    boxes=np.asarray([[.5,.5,.2,.2],[.2,.2,.05,.05]],dtype=float)
    logits=np.zeros((2,7),dtype=float)
    gt=np.array([.52,.5,.1,.1],dtype=float)
    x=native_unit(2,0,1,0,False,3,gt,boxes,logits,100,{'width':1000,'height':600},density=1.)
    assert x['best_raw_query']==0
    assert x['gt_w_source_px']>0
    assert x['decoder_refpoint_density_status']=='NOT_COMPUTED_O_R_UNVERIFIED'
    assert x['query_geometry_inference_status']=='SELECTION_CONFOUNDED_NOT_INFERENTIAL'
    aligned=validate_coco_alignment(np.array([[.5,.5,.1,.1]]),np.array([3]),
                      [{'id':1,'category_id':4,'bbox':[45,45,10,10]}],{'width':100,'height':100})
    assert aligned[0]['annotation_id']==1
    return '[PASS] COCO alignment, best raw IoU, pixel size, decoded box-center density'


def metadata(project):
    """Non-writing TRAIN-only capability inspection, not evidence of input pixel scale."""
    paths(project,True)
    rt=f.a.setup_native(project)
    old,r9,r8,d,s0a,raw,model,CO,FX=rt
    can,spa,tmp=s0a.load_video_triplet(TRAIN[0])
    item=can[0]
    def describe(obj):
        if isinstance(obj,dict):
            return {str(k):{'type':type(v).__name__,
                            'shape':list(v.shape) if hasattr(v,'shape') else None}
                    for k,v in obj.items()}
        return {'type':type(obj).__name__,'shape':list(obj.shape) if hasattr(obj,'shape') else None}
    im=next(x for x in raw['images'] if int(x['id'])==int(item['image_id']))
    print('[TRAIN CACHE FRAME FIELDS]',json.dumps(describe(item),sort_keys=True))
    print('[SPATIAL CACHE FIELDS]',json.dumps(describe(spa[0]),sort_keys=True))
    print('[TEMPORAL CACHE FIELDS]',json.dumps(describe(tmp[0]),sort_keys=True))
    O,X=s0a.make_O_X(item,spa[0],tmp[0],CO,FX)
    ref_details,ref_coords=inspect_O_r(O,item)
    print('[CACHED REFERENCE CONTRACT]',json.dumps(ref_details,sort_keys=True))
    print('[SOURCE IMAGE]',json.dumps({'id':im['id'],'width':im['width'],'height':im['height'],
        'camera_sensor':im.get('camera_sensor','UNKNOWN')},sort_keys=True))
    print('[UNKNOWN] True R50 input tensor dimensions and semantic origin of O.r require authoritative preprocessing/decoder trace; coordinate identity is not semantic proof')
    return True

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True,choices=('selftest','verify','metadata','audit'))
    p.add_argument('--project',default=r'D:\MARIVISION\H0_workbench\eval_tools')
    args=p.parse_args()
    if args.phase=='selftest':print(selftest())
    elif args.phase=='verify':
        _,dest=paths(args.project,True)
        print('[PASS] Source/receipt pins, TRAIN embargo, outputs unused; destination=',dest)
    elif args.phase=='metadata':metadata(args.project)
    else:run(args.project)

if __name__=='__main__':main()
