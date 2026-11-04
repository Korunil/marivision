from __future__ import annotations
"""RUN_11: read-only five-fold scientific audit for B-FCEF-F.

Requires RUN09D fold0 report, RUN10B fold1..4 reports, and Train COCO JSON.
Does not import PyTorch, read model weights, tune thresholds, or use Val/Test.
All summary AP values are MACRO means of fold APs, never pooled COCO AP.
"""
import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('D:/MARIVISION')
EVAL = ROOT / 'H0_workbench' / 'eval_tools'
S0 = EVAL / 'bfcef_f_s0'
F0 = S0 / 'run09d_calibrated_qualification_v1' / 'run09d_calibrated_integrated_qualification.json'
FOLDER = S0 / 'run10b_fold_excluded_calibration_v1'
OUT = S0 / 'run11_five_fold_scientific_audit_v1'
TRAIN = ROOT / 'seaclips-train.json'
EXPERTS = ('S', 'L', 'C', 'T')
VARIANTS = ('Base', 'Full', 'ClsOnly', 'BoxOnly', 'MinusS', 'MinusL', 'MinusC', 'MinusT')
GT_EXPECTED = 64670
VIDEOS_EXPECTED = set(range(1, 48))
REPORT_SHA = {
    0: '629365fa4e4cf45312dfad714146aafb46a2808c9f259a298eab6b040d2cdd5f',
    1: 'b03aabe2403468c9859a5f44261d61f589b97274b0babb55499c6d8f8e393fe1',
    2: '61e0c80cb20287ce1006fbf5ba1ba997cb788553fdc794b56346276954754634',
    3: '6d41d737a2559ac420538c2536b922ce5f79d81f4cd9136f96842c00eecfad23',
    4: '5d448126007be5dea490c9bbda7ff379c3c0e1fd177e0da8b25d930ca1ca0c0f',
}
CALIBRATED_SHA = {
    0: 'a1debfaadcc7f754319c63386ac93d993dc2a036cefbdf4e1ee70a567b517deb',
    1: '16de02429df4a33b60760b7222d6e6045e8c69a7818e20f9939a2c2fa83078c5',
    2: 'ef748d5dc766450b0ab6cadde79f328fac7c36378e26d3ee356213558a99c25c',
    3: 'c5298290a05cd703eac0e45958dd850e307cba59d287f48275aebd74be6b76a4',
    4: '48a0051948dd79703081d6f9f74268eac1bcd1dc8b885e49b9eb32078d0cc2b8',
}
KNOWN_COUNTS = {
    0: ((8918, 17137, 6177), (8861, 13509, 6234)),
    1: ((9874, 16013, 4626), (9960, 15415, 4540)),
    2: ((10686, 5722, 1003), (10827, 4466, 862)),
    3: ((10919, 7528, 1029), (10741, 3530, 1207)),
    4: ((10344, 6401, 1094), (10374, 4651, 1064)),
}


def require(ok, msg):
    if not ok:
        raise AssertionError('[RUN11 FATAL] ' + msg)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def readjson(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rate(a, b):
    return a / b if b else 0.0


def outcome(tp, fp, fn):
    return {'tp': int(tp), 'fp': int(fp), 'fn': int(fn),
            'precision': rate(tp, tp+fp), 'recall': rate(tp, tp+fn),
            'f1': rate(2*tp, 2*tp+fp+fn)}


def safe_ap(x):
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) and x >= 0 else None


def macro(values):
    a = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(a)/len(a) if a else None


def csvwrite(path, records):
    if not records:
        return
    cols = list(dict.fromkeys(k for item in records for k in item))
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for item in records:
            w.writerow(item)


def fold_path(i):
    return F0 if i == 0 else FOLDER / f'meta_fold{i}_qualification.json'


def load_reports(strict_sha=True):
    reports = {}
    for f in range(5):
        p = fold_path(f)
        require(p.is_file(), f'Missing fold {f} report: {p}')
        actual = sha(p)
        if strict_sha:
            require(actual == REPORT_SHA[f], f'Fold {f} report hash mismatch: {actual} != {REPORT_SHA[f]}')
        reports[f] = readjson(p)
        meta = reports[f]['_meta']
        require(int(meta['meta_fold']) == f, f'Fold {f} meta index mismatch')
        require(set(reports[f]['variants']) == set(VARIANTS), f'Fold {f} variant roster drift')
        require(meta['calibrated_checkpoint_sha256'] == CALIBRATED_SHA[f],
                f'Fold {f} calibrated checkpoint reference drift')
        if f == 0:
            require(meta['no_val_test'] is True and meta['no_training'] is True,
                    'Fold0 source embargo/training mismatch')
            held = [int(v) for v in meta['heldout_videos']]
        else:
            require(meta['official_test_accessed'] is False and
                    meta['official_validation_accessed'] is False and
                    meta['calibration_uses_heldout_labels'] is False and
                    meta['only_gate_thresholds_changed'] is True,
                    f'Fold {f} embargo/calibration flag failed')
            cal = reports[f]['calibration']
            held = [int(v) for v in meta['heldout_videos']]
            train = [int(v) for v in meta['train_videos']]
            require(sorted(cal['calibration_videos']) == sorted(train),
                    f'Fold {f} calibration training inventory mismatch')
            require(sorted(cal['heldout_videos']) == sorted(held),
                    f'Fold {f} calibration heldout inventory mismatch')
            require(set(train).isdisjoint(held) and len(train)+len(held)==47,
                    f'Fold {f} leakage/coverage mismatch')
            for task in ('cls', 'box'):
                for expert in EXPERTS:
                    item = cal[task][expert]
                    require(0 <= item['threshold'] <= 1 and
                            item['status'] in ('CALIBRATED_OPEN','CLOSED_NO_POSITIVE_TRAIN_UTILITY'),
                            f'Fold{f} {task}/{expert} threshold/status invalid')
        require(len(held)==len(set(held)) and len(held) in (9,10), f'Fold {f} heldout malformed')
        for name in VARIANTS:
            m = reports[f]['variants'][name]
            n = outcome(m['tp'], m['fp'], m['fn'])
            for metric in ('precision', 'recall', 'f1'):
                require(abs(n[metric] - float(m[metric])) < 1e-8,
                        f'Fold{f}/{name} {metric} inconsistency')
            by = m['by_video_operational']
            require(set(map(int, by)) == set(held), f'Fold{f}/{name} per-video coverage drift')
            for k in ('tp', 'fp', 'fn'):
                require(sum(int(v[k]) for v in by.values()) == int(m[k]),
                        f'Fold{f}/{name} per-video {k} sum drift')
            require(int(m['tp']) + int(m['fn']) ==
                    int(reports[f]['variants']['Base']['tp'] + reports[f]['variants']['Base']['fn']),
                    f'Fold{f}/{name} ground-truth conservation failed')
        for name in VARIANTS:
            if name == 'Base':
                continue
            tr = reports[f]['transitions'][name]
            b, q = reports[f]['variants']['Base'], reports[f]['variants'][name]
            require(tr['rescued'] - tr['lost'] == q['tp'] - b['tp'],
                    f'Fold{f}/{name} GT consequence conservation failed')
            require(tr['fp_added'] - tr['fp_removed'] == q['fp'] - b['fp'],
                    f'Fold{f}/{name} FP consequence conservation failed')
            require(tr['retained'] + tr['lost'] == b['tp'] and
                    tr['retained'] + tr['rescued'] == q['tp'],
                    f'Fold{f}/{name} GT intersection conservation failed')
        for j, name in enumerate(('Base','Full')):
            m = reports[f]['variants'][name]
            require((m['tp'], m['fp'], m['fn']) == KNOWN_COUNTS[f][j],
                    f'Fold{f}/{name} historical counts drift')
        require(sum((reports[f].get('class_support_fold0') if f == 0
                     else reports[f].get('class_support', {})).values()) ==
                reports[f]['variants']['Base']['tp']+reports[f]['variants']['Base']['fn'],
                f'Fold {f} class GT support drift')
        diag = reports[f]['full_mechanism_diagnostics']
        require(diag['frames'] > 0 and diag['candidate_queries'] > 0, f'Fold{f} diagnostics empty')
        require(0 <= diag['hard_bypass_frames'] <= diag['frames'], f'Fold{f} bypass malformed')
        print(f'[PASS] Fold {f} report integrity and exact paired-video conservation; SHA={actual[:16]}')
    return reports


def camera_metadata():
    if not TRAIN.is_file():
        return None, 'Train COCO metadata missing; no camera/domain subgroup audit'
    d = readjson(TRAIN)
    videos = d.get('videos')
    require(isinstance(videos, list), 'Train JSON lacks videos metadata')
    result = {}
    for v in videos:
        vid = int(v.get('id', v.get('video_id')))
        require('camera_sensor' in v, f'Camera sensor missing for {vid}')
        result[vid] = str(v['camera_sensor']).strip()
    require(set(result) == VIDEOS_EXPECTED, 'Train JSON camera inventory is not video IDs 1..47')
    return result, None


def fold_video_rows(reports, camera):
    folds, videos = [], []
    all_held = []
    for f, r in reports.items():
        b, q = r['variants']['Base'], r['variants']['Full']
        folds.append({'fold':f,'videos':len(b['by_video_operational']),
                      'gt':int(b['tp']+b['fn']),
                      'base_map30':b['map30'],'full_map30':q['map30'],'delta_map30':q['map30']-b['map30'],
                      'base_mapca30':b['mapca30'],'full_mapca30':q['mapca30'],
                      'delta_mapca30':q['mapca30']-b['mapca30'],
                      'base_f1':b['f1'],'full_f1':q['f1'],'delta_f1':q['f1']-b['f1'],
                      'delta_tp':q['tp']-b['tp'],'delta_fp':q['fp']-b['fp'],
                      'delta_fn':q['fn']-b['fn'],
                      'fp_reduction_fraction':rate(b['fp']-q['fp'],b['fp'])})
        for v, old in b['by_video_operational'].items():
            vid = int(v)
            new = q['by_video_operational'][v]
            a = outcome(**{k:int(old[k]) for k in ('tp','fp','fn')})
            z = outcome(**{k:int(new[k]) for k in ('tp','fp','fn')})
            require(a['tp']+a['fn'] == z['tp']+z['fn'],f'Video {vid} GT mismatch')
            videos.append({'fold':f,'video':vid,'camera':camera[vid] if camera else 'UNKNOWN',
                           'gt':a['tp']+a['fn'],'base_tp':a['tp'],'base_fp':a['fp'],'base_fn':a['fn'],
                           'full_tp':z['tp'],'full_fp':z['fp'],'full_fn':z['fn'],
                           'base_f1':a['f1'],'full_f1':z['f1'],'delta_f1':z['f1']-a['f1'],
                           'delta_tp':z['tp']-a['tp'],'delta_fp':z['fp']-a['fp'],
                           'delta_fn':z['fn']-a['fn']})
            all_held.append(vid)
    require(len(videos)==47 and set(all_held)==VIDEOS_EXPECTED and len(set(all_held))==47,
            '47-video paired coverage failed')
    require(sum(v['gt'] for v in videos)==GT_EXPECTED,'All-fold cached GT count mismatch')
    return folds, videos


def pool(videos):
    b = outcome(*(sum(v['base_'+k] for v in videos) for k in ('tp','fp','fn')))
    q = outcome(*(sum(v['full_'+k] for v in videos) for k in ('tp','fp','fn')))
    return {'base':b,'full':q,'delta_f1':q['f1']-b['f1'],
            'delta_precision':q['precision']-b['precision'],
            'delta_recall':q['recall']-b['recall'],
            'delta_tp':q['tp']-b['tp'],'delta_fp':q['fp']-b['fp'],
            'delta_fn':q['fn']-b['fn'],
            'fp_reduction_fraction':rate(b['fp']-q['fp'],b['fp'])}


def bootstrap_video(videos, reps=5000, seed=20261009):
    rng = random.Random(seed)
    vals = {'delta_f1':[], 'delta_precision':[], 'delta_recall':[],
            'fp_reduction_fraction':[]}
    for _ in range(reps):
        sample = rng.choices(videos,k=len(videos))
        m = pool(sample)
        for k in vals:
            vals[k].append(m[k])
    def percentile(a, p):
        s = sorted(a)
        q=(len(s)-1)*p
        lo=int(q); hi=min(lo+1,len(s)-1)
        if lo == hi: return s[lo]
        return s[lo]*(hi-q)+s[hi]*(q-lo)
    return {'method':'video-cluster nonparametric bootstrap; 47 videos resampled with replacement',
            'repetitions':reps,'seed':seed,'confidence_intervals_95':{
                k:[percentile(a,.025),percentile(a,.975)] for k,a in vals.items()}}


def aggregate_subgroups(videos, key):
    groups = defaultdict(list)
    for v in videos:
        groups[v[key]].append(v)
    return [{'group':name,'video_count':len(g),**pool(g)} for name,g in sorted(groups.items(),key=lambda t:str(t[0]))]


def audit_class_size(reports):
    names = sorted(reports[0]['variants']['Base']['per_class_ap30'])
    classes=[]
    for name in names:
        points=[]
        for fold,r in reports.items():
            support=r['class_support_fold0'] if fold==0 else r['class_support']
            require(name in support,f'Class name {name} not found in fold{fold}')
            b=safe_ap(r['variants']['Base']['per_class_ap30'][name])
            f=safe_ap(r['variants']['Full']['per_class_ap30'][name])
            if int(support[name])>0 and b is not None and f is not None:
                points.append({'fold':fold,'support':int(support[name]),'base_ap':b,'full_ap':f,'delta_ap':f-b})
        classes.append({'class':name,'gt_support_all_folds':sum(
            (r['class_support_fold0'] if fold==0 else r['class_support'])[name]
            for fold,r in reports.items()),'evaluable_folds':len(points),
            'mean_delta_ap_over_evaluable_folds':macro([x['delta_ap'] for x in points]),
            'negative_fold_count':sum(x['delta_ap'] < -1e-9 for x in points),
            'per_fold':points})
    sizes=[]
    for label,metric in (('small','ap_s30'),('medium','ap_m30'),('large','ap_l30')):
        rows=[]
        for i,r in reports.items():
            b=safe_ap(r['variants']['Base'][metric]); f=safe_ap(r['variants']['Full'][metric])
            if b is not None and f is not None:
                rows.append({'fold':i,'base_ap':b,'full_ap':f,'delta_ap':f-b})
        sizes.append({'size':label,'evaluable_folds':len(rows),'mean_delta_ap':macro([r['delta_ap'] for r in rows]),'per_fold':rows})
    return classes,sizes


def audit_ablation(reports):
    rows=[]
    for i,r in reports.items():
        full=r['variants']['Full']
        for e in EXPERTS:
            m=r['variants']['Minus'+e]
            rows.append({'fold':i,'expert':e,'full_map30':full['map30'],
                         'without_map30':m['map30'],
                         'contribution_to_map30':full['map30']-m['map30'],
                         'contribution_to_f1':full['f1']-m['f1'],
                         'delta_fp_when_expert_removed':m['fp']-full['fp'],
                         'delta_tp_when_expert_removed':m['tp']-full['tp']})
    return rows


def audit_routing(reports):
    rows=[]
    for fold,r in reports.items():
        d=r['full_mechanism_diagnostics']
        for task in ('cls','box'):
            info=d['routes'][task]
            oc=d['output_change']
            for e in EXPERTS:
                cal=None if fold==0 else r['calibration'][task][e]
                rows.append({'fold':fold,'task':task,'expert':e,
                             'threshold':cal['threshold'] if cal else None,
                             'candidate_gate_active_fraction':d['hard_mask_active_fraction_of_candidates'][task][e],
                             'mean_pi_expert':info['pi_'+e]['mean'],
                             'mean_pi0':info['pi_0']['mean'],
                             'mean_total_expert_mass':info['expert_mass_1_minus_pi0']['mean'],
                             'changed_query_fraction_at_1e_5':oc[task+'_query_fraction_above_tol'].get('1e-05'),
                             'hard_bypass_fraction':d['hard_bypass_fraction'],
                             'top100_pair_changed_frame_fraction':d['top100_pair_changed_frame_fraction'],
                             'operational_pair_changed_frame_fraction':d['operational_pair_changed_frame_fraction']})
    return rows


def audit_transitions_and_variants(reports):
    transitions=[]
    for fold,report in reports.items():
        for v in VARIANTS:
            m=report['variants'][v]
            tr=report['transitions'].get(v,{})
            transitions.append({'fold':fold,'variant':v,
                'rescued':tr.get('rescued',0),'lost':tr.get('lost',0),
                'retained':tr.get('retained',0),
                'fp_added':tr.get('fp_added',0),'fp_removed':tr.get('fp_removed',0),
                'tp':m['tp'],'fp':m['fp'],'fn':m['fn'],
                'map30':m['map30'],'mapca30':m['mapca30'],'f1':m['f1']})
    summary={}
    for v in VARIANTS:
        rows=[a for a in transitions if a['variant']==v]
        o=outcome(*(sum(x[k] for x in rows) for k in ('tp','fp','fn')))
        summary[v]={**o,
            'fold_macro_map30':macro([x['map30'] for x in rows]),
            'fold_macro_mapca30':macro([x['mapca30'] for x in rows]),
            'rescued_gt':sum(x['rescued'] for x in rows),
            'lost_gt':sum(x['lost'] for x in rows),
            'fp_added_identity_pairs':sum(x['fp_added'] for x in rows),
            'fp_removed_identity_pairs':sum(x['fp_removed'] for x in rows)}
    base=summary['Base']; full=summary['Full']
    require(full['rescued_gt']-full['lost_gt']==full['tp']-base['tp'],
            'Aggregate GT transition conservation failed')
    require(full['fp_added_identity_pairs']-full['fp_removed_identity_pairs']==full['fp']-base['fp'],
            'Aggregate FP transition conservation failed')
    return transitions,summary


def diagnostics(videos,folds,classes,ablation, camera_available):
    warnings=[]
    if any(v['delta_f1'] < -0.001 for v in videos):
        warnings.append('Some videos regress in F1; inspect per-video CSV before generalization claims')
    if any(v['delta_tp'] < 0 for v in videos):
        warnings.append('Some videos lose true positives; quantify class and video concentration')
    if any(x['mean_delta_ap_over_evaluable_folds'] is not None and
           x['mean_delta_ap_over_evaluable_folds'] < -0.02 for x in classes):
        warnings.append('At least one class has macro mean AP regression below -0.02')
    if any(a['contribution_to_map30'] < -0.001 for a in ablation):
        warnings.append('At least one MinusExpert ablation increases AP; avoid universal complementarity claims')
    if not camera_available:
        warnings.append('Camera metadata unavailable: domain-level audit incomplete')
    warnings.append('Per-video category AP and category-specific TP/FP/FN cannot be recovered from aggregate reports')
    warnings.append('Macro fold AP is not pooled COCO AP; pooled AP requires a unified prediction/evaluator run')
    warnings.append('Common R50 was trained on all Train videos: conditional downstream OOF, not end-to-end OOF')
    return warnings


def textnum(x, d=4):
    return f'{x:.{d}f}' if x is not None else 'NA'


def markdown_report(a):
    pooled=a['pooled']; boots=a['video_bootstrap']['confidence_intervals_95']; lines=[]
    lines += ['# RUN_11 Five-Fold Integrated Scientific Audit','',
              '**Status:** COMPLETE — READ ONLY; scientific review required before official Validation.', '',
              '## Five-fold pooled operational metrics','',
              '| Measure | R50 baseline | B-FCEF-F Full |','|---|---:|---:|']
    for key in ('tp','fp','fn','precision','recall','f1'):
        d=0 if key in ('tp','fp','fn') else 5
        lines.append(f'| {key} | {textnum(pooled["base"][key],d)} | {textnum(pooled["full"][key],d)} |')
    lines += ['',f'- Δ pooled F1: **{pooled["delta_f1"]:+.5f}**',
              f'- Δ FP: **{pooled["delta_fp"]:+d}** ({pooled["fp_reduction_fraction"]:.2%} reduction)',
              f'- Full vs Base GT identities rescued/lost: {a["variant_summary"]["Full"]["rescued_gt"]} / {a["variant_summary"]["Full"]["lost_gt"]}',
              f'- Full vs Base FP pairs added/removed: {a["variant_summary"]["Full"]["fp_added_identity_pairs"]} / {a["variant_summary"]["Full"]["fp_removed_identity_pairs"]}',
              f'- Video-cluster 95% bootstrap CI for Δ pooled F1: **[{boots["delta_f1"][0]:+.5f}, {boots["delta_f1"][1]:+.5f}]**',
              '', '## Fold-level paired metrics','',
              '| Fold | Δ mAP@.30 | Δ mAPca | Δ F1 | Δ TP | Δ FP |',
              '|---|---:|---:|---:|---:|---:|']
    for f in a['folds']:
        lines.append(f'| {f["fold"]} | {f["delta_map30"]:+.5f} | {f["delta_mapca30"]:+.5f} | {f["delta_f1"]:+.5f} | {f["delta_tp"]:+d} | {f["delta_fp"]:+d} |')
    lines += ['', '## Interpretive caveats','']
    lines.extend('- '+w for w in a['warnings'])
    lines += ['', '## Statistical scope','',
              'Unit of resampling is the **video**, not the frame/detection. Bootstrap intervals describe variability among these Train videos; they are not proof of unseen-domain performance.',
              'Per-class, size, and fold COCO AP values are macro-described, not recomputed from pooled predictions.',
              'Thresholds and model weights were not changed by this audit.',
              '', '## Follow-up decision','',
              'Review `run11_video_paired.csv`, `run11_class_ap.json`, `run11_ablation.csv`, `run11_routing.csv` and domain findings. Make a documented Validation GO / HOLD decision after reviewing risks, not automatically from aggregate F1.']
    return '\n'.join(lines)+'\n'


def audit(strict_sha=True, bootstrap_reps=5000, write=True):
    reports=load_reports(strict_sha)
    camera, camera_warning=camera_metadata()
    folds,videos=fold_video_rows(reports,camera)
    pooled=pool(videos)
    cls,size=audit_class_size(reports)
    ablation=audit_ablation(reports)
    routing=audit_routing(reports)
    transitions,variant_summary=audit_transitions_and_variants(reports)
    warnings=diagnostics(videos,folds,cls,ablation,camera is not None)
    if camera_warning: warnings.append(camera_warning)
    res={'_meta':{'schema':'bfcef_f_run11_read_only_v1','source_sha256':sha(Path(__file__)),
                  'reports_sha256':{str(f):sha(fold_path(f)) for f in range(5)},
                  'train_only':True,'no_retraining':True,'no_recalibration':True,
                  'no_validation_or_test_access':True,
                  'review_status':'AWAITING_SCIENTIFIC_GO_HOLD_DECISION'},
         'pooled':pooled,'folds':folds,
         'macro_fold_map30':{
             'base':macro([r['variants']['Base']['map30'] for r in reports.values()]),
             'full':macro([r['variants']['Full']['map30'] for r in reports.values()])},
         'macro_fold_mapca30':{
             'base':macro([r['variants']['Base']['mapca30'] for r in reports.values()]),
             'full':macro([r['variants']['Full']['mapca30'] for r in reports.values()])},
         'video_bootstrap':bootstrap_video(videos,bootstrap_reps),
         'video_signs':{'f1_positive':sum(v['delta_f1']>1e-12 for v in videos),
                        'f1_negative':sum(v['delta_f1'] < -1e-12 for v in videos),
                        'f1_equal':sum(abs(v['delta_f1']) <= 1e-12 for v in videos)},
         'leave_one_fold_out_pooled':[
             {'omitted_fold':f,'pooled':pool([v for v in videos if v['fold']!=f])}
             for f in range(5)],
         'leave_one_video_out_pooled_delta_f1':[
             {'omitted_video':v['video'],'delta_f1':pool([w for w in videos if w['video']!=v['video']])['delta_f1']}
             for v in videos],
         'camera':aggregate_subgroups(videos,'camera') if camera else None,
         'classes':cls,'sizes':size,'ablations':ablation,'routing':routing,
         'transitions':transitions,'variant_summary':variant_summary,
         'warnings':warnings}
    if write:
        OUT.mkdir(parents=True,exist_ok=True)
        (OUT/'run11_five_fold_audit.json').write_text(json.dumps(res,indent=2,allow_nan=False),encoding='utf-8')
        (OUT/'run11_scientific_report.md').write_text(markdown_report(res),encoding='utf-8')
        (OUT/'run11_class_ap.json').write_text(json.dumps(cls,indent=2,allow_nan=False),encoding='utf-8')
        (OUT/'run11_size_ap.json').write_text(json.dumps(size,indent=2,allow_nan=False),encoding='utf-8')
        csvwrite(OUT/'run11_fold_metrics.csv',folds)
        csvwrite(OUT/'run11_video_paired.csv',videos)
        csvwrite(OUT/'run11_ablation.csv',ablation)
        csvwrite(OUT/'run11_transitions.csv',transitions)
        csvwrite(OUT/'run11_routing.csv',routing)
        csvwrite(OUT/'run11_camera.csv',[{'camera':a['group'],'video_count':a['video_count'],
             'base_tp':a['base']['tp'],'base_fp':a['base']['fp'],
             'full_tp':a['full']['tp'],'full_fp':a['full']['fp'],
             'delta_f1':a['delta_f1'],'delta_tp':a['delta_tp'],'delta_fp':a['delta_fp']}
            for a in (res['camera'] or [])])
    print('='*90)
    print('[PASS] RUN11 five-fold report/census/consequence integrity — no training or calibration')
    print('Pooled Base:',pooled['base'])
    print('Pooled Full:',pooled['full'])
    print(f"Δ F1={pooled['delta_f1']:+.6f}, Δ TP={pooled['delta_tp']:+d}, Δ FP={pooled['delta_fp']:+d}")
    print('47-video Δ F1 signs:',res['video_signs'])
    print('Video-bootstrap Δ F1 95% CI:',res['video_bootstrap']['confidence_intervals_95']['delta_f1'])
    print(f"Warnings requiring review: {len(warnings)}")
    for w in warnings: print('  -',w)
    if write: print('Report directory:',OUT)
    print('STOP: independent Validation/official Test not accessed')
    return res


def selftest():
    assert abs(outcome(5,3,2)['f1'] - (10/15))<1e-12
    a={'tp':1,'fp':2,'fn':2}
    v={'video':1,'fold':0,'camera':'A','gt':3,
       **{'base_'+k:a[k] for k in a},
       'full_tp':2,'full_fp':1,'full_fn':1}
    x=pool([v]); assert x['delta_tp']==1 and x['delta_fp']==-1 and x['delta_f1']>0
    b=bootstrap_video([v],20)
    assert abs(b['confidence_intervals_95']['delta_f1'][0]-x['delta_f1'])<1e-12
    assert safe_ap(-1) is None and safe_ap(.2)==.2
    print('[PASS] RUN11 synthetic paired-count, bootstrap and AP-missing-value tests')


def main():
    global ROOT,EVAL,S0,F0,FOLDER,OUT,TRAIN
    ap=argparse.ArgumentParser()
    ap.add_argument('--self-test-only',action='store_true')
    ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--bootstrap-reps',type=int,default=5000)
    ap.add_argument('--accept-report-hash-drift',action='store_true',
                    help='Diagnostic only: compare content but skip exact report SHA gate. Not certified.')
    args=ap.parse_args()
    selftest()
    if args.self_test_only: return
    require(100<=args.bootstrap_reps<=50000,'Bootstrap repetitions outside 100..50000')
    ROOT=args.root
    EVAL=ROOT/'H0_workbench'/'eval_tools'
    S0=EVAL/'bfcef_f_s0'
    F0=S0/'run09d_calibrated_qualification_v1'/'run09d_calibrated_integrated_qualification.json'
    FOLDER=S0/'run10b_fold_excluded_calibration_v1'
    OUT=S0/'run11_five_fold_scientific_audit_v1'
    TRAIN=ROOT/'seaclips-train.json'
    if OUT.exists():
        require(not any(OUT.iterdir()),'Output already exists: refuses overwrite')
    audit(strict_sha=not args.accept_report_hash_drift,bootstrap_reps=args.bootstrap_reps)


if __name__=='__main__':
    main()
