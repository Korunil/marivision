from __future__ import annotations
"""RUN15A PATCH02: Resume *unfinished* frozen guard qualification after COCO universe bug.

No fitting, no threshold selection, no original output overwrite, no Val/Test access.
The original PATCH01 Python source, guard checkpoint, Train data and calibration
are immutable inputs. A NEW sidecar output directory contains a single forensic
qualification and (on PASS only) a scoped/provisional deployment package.

Primary repair: COCO GT is restricted to the SAME cached Train image-ID set
that produced guard action rows and predictions. Any excluded official Train
images/annotations are counted and exposed, never represented as evaluated.
"""
import argparse
import collections
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import math
import os
import sys
from pathlib import Path
from PIL import Image  # Windows/Pillow DLL import-order compatibility
for k,v in {'R0_MSDA_BACKEND':'pytorch','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
            'OPENBLAS_NUM_THREADS':'1','PYTHONHASHSEED':'20261010'}.items():
    os.environ.setdefault(k,v)
import numpy as np
import torch

EVAL=Path('D:/MARIVISION/H0_workbench/eval_tools')
ORIGINAL=EVAL/'run_bfcef_f_run15a_integrated.py'
PINNED_PATCH01='1d7cea42f251bef0dc485f893a590ee7d924966aa1a3b1c99cedf5747d82304c'
OUT=EVAL/'bfcef_f_s0'/'run15a_patch02_qualification_recovery_v1'
ORIG_OUT=EVAL/'bfcef_f_s0'/'run15a_integrated_trainonly_v1_patch01'


def need(cond,msg):
    if not cond: raise AssertionError('[RUN15A PATCH02 FATAL] '+str(msg))


def sha(p):
    p=Path(p);need(p.is_file(),f'Missing file {p}')
    h=hashlib.sha256()
    with p.open('rb') as f:
        for buf in iter(lambda:f.read(1<<20),b''): h.update(buf)
    return h.hexdigest()


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json_new(path,data):
    path=Path(path);need(not path.exists(),f'Output already exists: {path}')
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    need(not tmp.exists(),f'Partial output exists: {tmp}')
    try:
        tmp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
        need(not path.exists(),f'Output appeared while preparing: {path}')
        tmp.rename(path)
    finally:
        if tmp.exists():tmp.unlink()


def load_original():
    need(sha(ORIGINAL)==PINNED_PATCH01,
         'Installed original RUN15A PATCH01 source differs; preserve originals and investigate')
    spec=importlib.util.spec_from_file_location('run15a_pinned_patch01_recovery',ORIGINAL)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    need(Path(m.OUT).resolve()==ORIG_OUT.resolve(),'Original output directory changed')
    need(str(m.R12_SHA)=='1211292b180eaf00862fc8098ad5cda6a2f64701c0954b5f8f752f090d2ce84c',
         'Frozen R12 parent drift')
    return m


def select_cached_universe(rows,source_data):
    """Pure function: fail closed for row/image/category inconsistency."""
    need(rows,'No qualification rows')
    target_videos={int(r['video']) for r in rows}
    cached_ids=[int(r['image_id']) for r in rows]
    need(len(cached_ids)==len(set(cached_ids)),'Duplicate cached image IDs')
    cached_set=set(cached_ids)
    official_imgs={int(im['id']):im for im in source_data['images']}
    need(len(official_imgs)==len(source_data['images']),'Duplicate official Train image IDs')
    official_selected={i for i,im in official_imgs.items() if int(im['video_id']) in target_videos}
    need(cached_set.issubset(official_selected),
         f'Cached qualification IDs outside selected official Train videos: {len(cached_set-official_selected)}')
    for r in rows:
        im=official_imgs[int(r['image_id'])]
        need(int(im['video_id'])==int(r['video']),'Cached row assigns image to wrong video')
        need(float(im['width'])>0 and float(im['height'])>0,'Invalid native image dimensions')
    selected_anns=[a for a in source_data['annotations'] if int(a['image_id']) in cached_set]
    anns_by_image=collections.defaultdict(list)
    for a in selected_anns:
        aid=int(a['image_id']);need(1<=int(a['category_id'])<=7,'Invalid COCO category')
        need('bbox' in a and len(a['bbox'])==4,'Malformed COCO bbox')
        anns_by_image[aid].append(a)
    for r in rows:
        iid=int(r['image_id']);expected=sorted(int(k)+1 for k in r['class_gt'])
        observed=sorted(int(a['category_id']) for a in anns_by_image[iid])
        need(expected==observed,
             f'Cached GT != official Train GT on image {iid}: cached {expected}, source {observed}')
        # Box-area invariant: canonical normalized cxcywh must represent raw COCO
        # pixel-bbox areas after non-aspect 640x640 warp. Only compare unordered
        # class+area multisets; original annotation order may differ.
        im=official_imgs[iid];wh=float(im['width'])*float(im['height'])
        expected_pairs=sorted((int(c)+1,float(area)) for c,area in zip(r['class_gt'],r['class_gt_area']))
        observed_pairs=sorted((int(a['category_id']),float(a['bbox'][2])*float(a['bbox'][3])/wh)
                              for a in anns_by_image[iid])
        need(len(expected_pairs)==len(observed_pairs),'GT area pair cardinality drift')
        # Reconcile per class without requiring annotation-list ordering.
        classes=sorted({c for c,_ in expected_pairs})
        for c in classes:
            exp=sorted(v for cc,v in expected_pairs if cc==c)
            obs=sorted(v for cc,v in observed_pairs if cc==c)
            need(len(exp)==len(obs),'GT class-specific area cardinality drift')
            for e,o in zip(exp,obs):
                need(abs(e-o)<=max(2e-7,0.05*max(e,o)),
                     f'GT bbox-area drift: image {iid}, class {c}, cache {e:.8g} source {o:.8g}')
    all_selected_anns=[a for a in source_data['annotations']
                       if int(a['image_id']) in official_selected]
    uncached_ids=official_selected-cached_set
    omitted_gt=sum(int(a['image_id']) in uncached_ids for a in all_selected_anns)
    require_metadata={'selected_videos':sorted(target_videos),
        'raw_train_video_image_count':len(official_selected),
        'certified_cached_image_count':len(cached_set),
        'uncached_raw_train_image_count':len(uncached_ids),
        'uncached_raw_train_gt_count':int(omitted_gt),
        'cached_raw_train_gt_count':len(selected_anns),
        'cached_gt_contract_count':sum(len(r['class_gt']) for r in rows),
        'coco_universe':'only certified cached Train images for selected qualification videos',
        'not_full_official_train_video_universe':bool(uncached_ids),
        'uncached_image_ids_sha256':hashlib.sha256(json.dumps(sorted(uncached_ids)).encode()).hexdigest()}
    need(len(selected_anns)==sum(len(r['class_gt']) for r in rows),
         'Cached qualification and raw Train GT counts differ')
    cat_ids={int(c['id']) for c in source_data['categories']}
    need(cat_ids==set(range(1,8)),'Official Train category map mismatch')
    sub={k:copy.deepcopy(v) for k,v in source_data.items()
         if k not in ('images','annotations','videos')}
    sub['images']=[copy.deepcopy(official_imgs[i]) for i in sorted(cached_ids)]
    sub['annotations']=copy.deepcopy(selected_anns)
    sub['videos']=[copy.deepcopy(v) for v in source_data['videos']
                   if int(v['id']) in target_videos]
    need(len(sub['images'])==len(rows) and
         set(int(a['image_id']) for a in sub['annotations']).issubset(cached_set),
         'COCO cached universe inclusion failure')
    return sub,require_metadata


def coco_evaluation(rows, choices, r9,r12,r8,d,s0a,source_data,m):
    from pycocotools.coco import COCO
    sub,meta=select_cached_universe(rows,source_data)
    print(f"[PASS] COCO cached Train qualification frame contract: "
          f"cached={meta['certified_cached_image_count']} "
          f"raw={meta['raw_train_video_image_count']} "
          f"omitted={meta['uncached_raw_train_image_count']} "
          f"omitted_GT={meta['uncached_raw_train_gt_count']}",flush=True)
    coco=COCO();coco.dataset=sub
    with contextlib.redirect_stdout(io.StringIO()):coco.createIndex()
    coco_ca=copy.deepcopy(coco)
    coco_ca.dataset['categories']=[{'id':1,'name':'object'}]
    for a in coco_ca.dataset['annotations']:a['category_id']=1
    with contextlib.redirect_stdout(io.StringIO()):coco_ca.createIndex()
    evaluators={v:r8.make_evaluators(coco,coco_ca) for v in ('Base','Selected')}
    lookup={int(r['image_id']):(r,int(a)) for r,a in zip(rows,choices)}
    need(len(lookup)==len(rows),'Qualification lookup has duplicate IDs')
    image_data={int(i['id']):i for i in sub['images']}
    need(set(lookup)==set(image_data),'COCO cached GT / action-lookup coverage mismatch')
    runtime,_,CO,FX=r9.initialize_runtime()
    state=torch.load(m.FINAL,map_location='cpu',weights_only=False)
    r12.load_experts(runtime,state);r12.load_downstream(runtime,state);runtime.eval()
    for p in runtime.parameters():p.requires_grad_(False)
    cats=torch.arange(1,8,dtype=torch.long)
    observed=set();computed_counts={a:collections.Counter() for a in ('Base','Selected')}
    with torch.no_grad():
        for vid in sorted({int(r['video']) for r in rows}):
            can,spa,tmp=s0a.load_video_triplet(vid)
            expected_ids={int(r['image_id']) for r in rows if int(r['video'])==vid}
            need(len(can)==len(spa)==len(tmp)==len(expected_ids),
                 f'Cached triplet/qualifier count mismatch: video {vid}')
            predictions={v:{} for v in evaluators};ca_predictions={v:{} for v in evaluators}
            for i,fr in enumerate(can):
                iid=int(fr['image_id']);need(iid in expected_ids and iid not in observed,
                                            f'Unexpected image ID in replay: {iid}')
                observed.add(iid)
                row,chosen=lookup[iid];need(int(row['video'])==vid,'Replay video provenance drift')
                labels=fr['gt_labels'].detach().cpu().long().tolist()
                need(sorted(labels)==sorted(row['class_gt']),f'Replay GT class drift iid {iid}')
                O,X=s0a.make_O_X(fr,spa[i],tmp[i],CO,FX)
                z,b,_=d.production_forward(runtime,O,X)
                cf=m.variants(O,z,b)
                # Independent per-frame operational parity: original certified Train
                # ledger MUST reproduce Base/Selected counts, not just aggregate totals.
                for out_name,action in (('Base','Base'),('Selected',m.ACTIONS[chosen])):
                    zz,bb=cf[action]
                    pred=r8.extract_top100_predictions(zz.detach().float().cpu(),
                                                        bb.detach().float().cpu())
                    mm=r8.match_operational_detections_to_gt(
                        pred['op_scores'],pred['op_c_idx'],pred['op_boxes'],pred['op_q_idx'],
                        fr['gt_boxes_norm'].detach().cpu().float(),fr['gt_labels'].detach().cpu().long())
                    got={k:int(mm[k]) for k in ('tp','fp','fn')}
                    need(got==row['counts'][action],
                         f'Cached outcome/replayed model drift: image={iid} action={action} '
                         f'cached={row["counts"][action]} observed={got}')
                    computed_counts[out_name].update(got)
                    r8.add_coco_prediction(predictions[out_name],ca_predictions[out_name],iid,pred,
                        float(image_data[iid]['width']),float(image_data[iid]['height']),cats)
            for variant in evaluators:
                ev,ev_ca=evaluators[variant]
                ev.update(predictions[variant]);ev_ca.update(ca_predictions[variant])
            print(f'    [COCO] cached Train qualifier video={vid} frames={len(can)}',flush=True)
            del can,spa,tmp,predictions,ca_predictions
            import gc
            gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
    need(observed==set(lookup),'Full cached image-ID replay coverage mismatch')
    summary={variant:d.finalize_variant_evaluator(*evaluators[variant],list(m.CLASSES))
             for variant in evaluators}
    for variant in summary:
        need(math.isfinite(summary[variant]['map30']) and 0<=summary[variant]['map30']<=1,
             f'Nonfinite/out of range COCO mAP: {variant}')
    return summary,meta,computed_counts


def preflight(*,report=False):
    m=load_original()
    require_paths={
        'source':ORIGINAL,
        'guard_checkpoint':ORIG_OUT/'run15a_preservation_guard.pth',
        'guard_certificate':ORIG_OUT/'run15a_guard_training_certificate.json',
        'calibration':ORIG_OUT/'run15a_guard_calibration.json',
        'contract':ORIG_OUT/'run15a_contract.json',
        'gap_register':ORIG_OUT/'run15a_concept_gap_register.json',
    }
    for name,p in require_paths.items():need(p.is_file(),f'{name} missing: {p}')
    need(not (ORIG_OUT/'run15a_trainonly_qualification.json').exists(),
         'Original qualification already exists; refusing second qualification')
    need(not (ORIG_OUT/'run15a_provisional_deployment_package.pth').exists(),
         'Original provisional model already exists; refusing duplicate qualification')
    need(not (OUT/'run15a_patch02_trainonly_qualification.json').exists(),
         'PATCH02 qualification already exists; refusing duplicate qualification')
    contract=read_json(require_paths['contract'])
    need(contract['source_sha256']==PINNED_PATCH01 and
         contract['parent_checkpoint_sha256']==m.R12_SHA and
         contract['train_json_sha256']==sha(m.TRAIN_JSON),
         'Source/parent/Train annotation contract drift')
    need(contract['actions']==list(m.ACTIONS) and contract['features']==list(m.FEATURES),
         'Action/feature contract drift')
    cert=read_json(require_paths['guard_certificate'])
    cksha=sha(require_paths['guard_checkpoint'])
    need(cert['sha256']==cksha and cert['source_sha256']==PINNED_PATCH01,
         'Certified trained checkpoint is inconsistent with original run')
    calibration=read_json(require_paths['calibration'])
    need(calibration['schema']=='run15a_guard_calibration_v1' and
         calibration['guard_checkpoint_sha256']==cksha and
         calibration['status']=='CALIBRATED',
         'Original calibration is not locked and successful')
    threshold=calibration['selected_threshold']
    need(isinstance(threshold,(int,float)) and threshold in m.CAL_THRESHOLDS,
         'Original selected threshold missing or not in frozen grid')
    need(sum(row.get('threshold')==threshold and row['gate']['status']=='PASS'
             for row in calibration['grid'])==1,
         'Original selected threshold not uniquely qualified on CAL partition')
    need(sha(m.FINAL)==m.R12_SHA,'R12 frozen parent drift')
    need(sha(m.TRAIN_JSON)==contract['train_json_sha256'],'Train annotation drift')
    need(cert['fit_videos']==sorted(contract['partitions']['fit']),
         'Training checkpoint FIT split drift')
    need(cert['calibration_videos']==sorted(contract['partitions']['calibration']) and
         cert['qualification_videos']==sorted(contract['partitions']['qualification']),
         'Training certificate CAL/QUAL split drift')
    need(set(cert['fit_videos']).isdisjoint(cert['calibration_videos']) and
         set(cert['fit_videos']).isdisjoint(cert['qualification_videos']) and
         set(cert['calibration_videos']).isdisjoint(cert['qualification_videos']),
         'Train partitions not disjoint')
    info={'patch02_source_sha256':sha(Path(__file__)),
          'original_run15a_patch01_sha256':PINNED_PATCH01,
          'guard_checkpoint_sha256':cksha,
          'calibration_sha256':sha(require_paths['calibration']),
          'frozen_r12_sha256':m.R12_SHA,
          'official_train_json_sha256':sha(m.TRAIN_JSON),
          'fixed_threshold':threshold,
          'fit_videos':cert['fit_videos'],
          'calibration_videos':cert['calibration_videos'],
          'qualification_videos':cert['qualification_videos'],
          'original_qualification_created':False,
          'validation_or_test_accessed':False}
    if report: print('[PASS] Frozen RUN15A PATCH01 source, Train ledger contract, guard checkpoint, '
                     'CAL threshold and partition independence verified',flush=True)
    return m,info


def selftest():
    base={'images':[{'id':1,'video_id':5,'width':100,'height':100},
                    {'id':2,'video_id':5,'width':100,'height':100},
                    {'id':3,'video_id':8,'width':100,'height':100}],
          'videos':[{'id':5},{'id':8}],
          'categories':[{'id':j,'name':str(j)} for j in range(1,8)],
          'annotations':[{'id':10,'image_id':1,'category_id':1,'bbox':[0,0,10,10]},
                         {'id':11,'image_id':2,'category_id':2,'bbox':[0,0,12,12]},
                         {'id':12,'image_id':3,'category_id':3,'bbox':[0,0,14,14]}]}
    r={'video':5,'image_id':1,'class_gt':[0],'class_gt_area':[.01]}
    sub,stats=select_cached_universe([r],base)
    need([im['id'] for im in sub['images']]==[1] and len(sub['annotations'])==1,
         'Cached ID universe subset fixture failed')
    need(stats['uncached_raw_train_image_count']==1 and stats['uncached_raw_train_gt_count']==1,
         'Raw Train exclusions not accounted for')
    try:
        select_cached_universe([dict(r,class_gt=[1])],base)
    except AssertionError:pass
    else:raise AssertionError('Category mismatch was not rejected')
    try:
        select_cached_universe([dict(r,image_id=4)],base)
    except AssertionError:pass
    else:raise AssertionError('Out-of-universe image ID was not rejected')
    try:
        select_cached_universe([r,r],base)
    except AssertionError:pass
    else:raise AssertionError('Duplicate image ID was not rejected')
    try:
        select_cached_universe([dict(r,class_gt_area=[.20])],base)
    except AssertionError:pass
    else:raise AssertionError('Raw/cached box-area drift was not rejected')
    print('[PASS] PATCH02 adversarial cached-universe tests (subset, omitted GT, category, image ID, duplicates, box area); no SeaClips accessed')


def finalize():
    m,roots=preflight(report=True)
    r9,r12,r8,d,s0a,folds,partitions,source_data=m.phase_preflight()
    need(partitions==read_json(ORIG_OUT/'run15a_contract.json')['partitions'],
         'Split assignment drift')
    rows=m.read_guard_data()
    groups={key:[row for row in rows if row['video'] in set(ids)]
            for key,ids in partitions.items()}
    need(all(groups.values()) and sum(map(len,groups.values()))==20075,'Certified Train partition census drift')
    # train_guard() is called only after verifying that both checkpoint and
    # training certificate already exist; it will LOAD, never fit.
    ck=ORIG_OUT/'run15a_preservation_guard.pth'
    need(ck.is_file() and (ORIG_OUT/'run15a_guard_training_certificate.json').is_file(),
         'Trained guard absent; recovery MUST NOT initiate fitting')
    model,mu,sigma=m.train_guard(groups['fit'],groups['calibration'],groups['qualification'])
    cal=read_json(ORIG_OUT/'run15a_guard_calibration.json')
    threshold=cal['selected_threshold']
    kval=groups['qualification']
    probs=m.predict_proba(model,kval,mu,sigma)
    choices=m.choose_from_probability(probs,threshold)
    ops=m.evaluate_choices(kval,choices)
    print('[PASS] Previously trained guard and locked calibration loaded; '
          'operational qualification computed without fitting or selection',flush=True)
    coco,universe,live_counts=coco_evaluation(kval,choices,r9,r12,r8,d,s0a,source_data,m)
    ops['coco']=coco
    for variant in ('Base','Selected'):
        need(dict(live_counts[variant])==
             {k:int(ops['metrics'][variant][k]) for k in ('tp','fp','fn')},
             f'Full inference and cached ledger operational aggregates differ: {variant}')
    ops['schema']='run15a_patch02_qualification_recovery_v1'
    ops['stage']='RESUMED_UNFINISHED_ORIGINAL_QUALIFICATION'
    ops['train_only']=True
    ops['partition']='TRAIN-video-heldout-for-GUARD-only_NOT_upstream_R0_or_R12'
    ops['original_certification']={**roots,'recovery_reason':
        'Original COCO evaluator incorrectly used all official Train image IDs '
        'instead of the exact 20075-frame cached evidence universe',
        'no_weights_or_thresholds_modified':True,
        'no_new_model_training_or_calibration':True}
    ops['coco_subset_integrity']=universe
    ops['selected_action_threshold']=threshold
    ops['decision']=m.prospective_gate(ops)
    ops['critical_limitations']=[
        'Upstream R0/R12 core was already fitted on all 47 Train videos; holdout pertains only to guard.',
        'COCO AP is measured on the CERTIFIED CACHED Train frames; uncached official Train frames '
        'are excluded and their number/GT counts are shown in coco_subset_integrity.',
        'External guard does NOT fix internal RUN09 U/H target, soft/hard routing or OOF-to-Full transfer.',
        'Observed Validation remains previously exposed; official Test must remain sealed.',
        'No guarantee of R0-or-better performance on unseen inputs or small LEISURE_VEHICLE cases.']
    path=OUT/'run15a_patch02_trainonly_qualification.json'
    atomic_json_new(path,ops)
    print(f"[RESULT] TRAIN-GUARD-ONLY {ops['decision']['status']} "
          f"Base={ops['metrics']['Base']} Selected={ops['metrics']['Selected']}",flush=True)
    print(f"[COCO] R0={coco['Base']['map30']:.6f} Selected={coco['Selected']['map30']:.6f}")
    print('[REPORT]',path,'SHA-256',sha(path),flush=True)
    if ops['decision']['status']!='PASS':
        print('[HOLD] RUN15A guard qualification did not meet prospective gate. '
              'No provisional deployment package will be created.',flush=True)
        return 2
    deployment=OUT/'run15a_patch02_provisional_deployment.pth'
    need(not deployment.exists(),'Provisional package already exists')
    torch.save({'schema':'run15a_patch02_guarded_bfcef_f_provisional_v1',
                'parent_R12_sha256':m.R12_SHA,'guard_checkpoint_sha256':roots['guard_checkpoint_sha256'],
                'guard_state':model.state_dict(),'feature_names':m.FEATURES,'actions':m.ACTIONS,
                'normalizer_mean':mu,'normalizer_std':sigma,'threshold':float(threshold),
                'qualification_sha256':sha(path),
                'scope':'TRAIN-GUARD-ONLY provisional, not end-to-end holdout certified'},deployment)
    cert={'schema':'run15a_patch02_provisional_certificate_v1',
          'sha256':sha(deployment),'qualification_sha256':sha(path),
          'parent_R12_sha256':m.R12_SHA,
          'official_test_accessed':False,
          'certification_level':'TRAIN_GUARD_ONLY_NOT_INDEPENDENT_FULL_MODEL'}
    atomic_json_new(OUT/'run15a_patch02_provisional_certificate.json',cert)
    print('[PASS] PROVISIONAL Train guard only. Validation/Test NOT authorized.',flush=True)
    return 0


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--phase',required=True,choices=('selftest','preflight','finalize'))
    args=ap.parse_args()
    selftest()
    if args.phase=='selftest':return 0
    if args.phase=='preflight':preflight(report=True);return 0
    return finalize()


if __name__=='__main__':sys.exit(main())
