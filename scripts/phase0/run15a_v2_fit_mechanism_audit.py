from __future__ import annotations
"""RUN15A V2: read-only FIT-video mechanism attribution.

Uses the immutable fitted fixed-threshold downstream and the 29 original FIT
videos only. No training, no threshold adjustment, no CAL/QUAL/Validation/Test
reading. One-at-a-time expert availability knockouts are *conditional ablations*,
not isolated causal identification or retrained alternatives.

Run from D:\\MARIVISION\\H0_workbench\\eval_tools with 311env.
"""
import argparse
import collections
import contextlib
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED_DRIVER_SHA = '6fa33f1128cbc22266c580e7be85a4b69fd1bac2651ed358b64140236cd733e2'
EXPECTED_MECHANISM_SHA = '42404d765097c65b23ec318ed51b8c4fc7b2acdd9eb34a331a76c09313bcb637'
EXPECTED_FIT_SHA = 'b265aa6a12987de61afaa0010071717b853b01ea0844d1502cc8321029cb3f8d'
EXPECTED_FIT_CERT_SHA = 'd44191b7c8894854b8cc8770fe12e6e3c6ba18b0aa0fe8aba28812283329b1a4'
CLASSES = ('ANIMAL','BOAT','MARINE_MARKER','OBJECT','LEISURE_VEHICLE','SAILING_VESSEL','SHIP')
EXPERTS = ('S','L','C','T')
VARIANTS = ('Base','Full','ClsOnly','BoxOnly','NoS','NoL','NoC','NoT')
STAGES = ('score_below_0p20','top100_pair_evicted','same_pair_box_iou_below_0p30','greedy_competition_or_other')
VERSION = 'run15a_v2_fit_mechanism_audit_v1'


def need(ok, msg):
    if not ok: raise AssertionError('[RUN15A FIT AUDIT HOLD] ' + str(msg))


def sha(path):
    p = Path(path)
    need(p.is_file(), f'Missing required source/weight/certificate: {p}')
    digest = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def json_read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def json_new(path, payload):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    need(not p.exists(), 'Refuse overwrite: ' + str(p))
    temp = p.with_name(p.name + '.tmp')
    need(not temp.exists(), 'Abandoned temporary file, inspect before proceeding: ' + str(temp))
    with temp.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, allow_nan=False)
    temp.rename(p)


def csv_new(path, rows, fieldnames):
    p = Path(path)
    need(not p.exists(), 'Refuse overwrite: ' + str(p))
    temp = p.with_name(p.name + '.tmp')
    need(not temp.exists(), 'Abandoned temporary CSV: ' + str(temp))
    with temp.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp.rename(p)


def metric(counter):
    t, f, n = (int(counter.get(k, 0)) for k in ('tp','fp','fn'))
    return {'tp': t, 'fp': f, 'fn': n,
            'precision': t/(t+f) if t+f else 0.,
            'recall': t/(t+n) if t+n else 0.,
            'f1': 2*t/(2*t+f+n) if 2*t+f+n else 0.}


def loss_stage(q, c, box, gt_box, logits, top100_pairs, iou_fn, op_score=0.20, op_iou=0.30):
    """Mutually exclusive diagnostic stage for a once-matched baseline pair.

    Score first because a pair under operating threshold could *also* be evicted
    from the Top100; then Top100; then current box IoU; then global competition.
    It does NOT assert that this stage is a model-level counterfactual cause.
    """
    score = float(logits[q, c].sigmoid().item())
    pair = (int(q), int(c))
    iou = float(iou_fn(box[q], gt_box).item())
    if score < op_score:
        return STAGES[0], score, iou
    if pair not in top100_pairs:
        return STAGES[1], score, iou
    if iou < op_iou:
        return STAGES[2], score, iou
    return STAGES[3], score, iou


def top_pairs(logits, n_classes=7):
    score = logits.sigmoid().flatten()
    indices = score.topk(min(100, score.numel())).indices.tolist()
    return {(int(i // n_classes), int(i % n_classes)) for i in indices}


@contextlib.contextmanager
def availability_knockout(model, expert):
    """In-memory task-symmetric expert removal. Always restore _availability.

    This recomputes gating+coordinator+router with A[:,expert]=0 and holds all
    trained weights and the fixed numeric gate thresholds unchanged.
    """
    need(expert in EXPERTS, 'Unknown expert knockout')
    instance_dict = vars(model)
    had_override = '_availability' in instance_dict
    previous_override = instance_dict.get('_availability')
    original = model._availability
    k = EXPERTS.index(expert)

    def ablated(O, candidate_query_mask, task):
        avail = original(O, candidate_query_mask, task).clone()
        need(avail.ndim == 2 and avail.shape[1] == len(EXPERTS), 'Availability layout drift')
        avail[:, k] = 0
        return avail

    # Attribute assignment is temporary; it neither touches nn.Parameter nor weights.
    model._availability = ablated
    try:
        yield
    finally:
        if had_override:
            model._availability = previous_override
        else:
            del model.__dict__['_availability']
        need(getattr(model, '_availability') == original,
             'Expert knockout failed to restore original availability method')


def assert_model_shapes(out, original):
    import torch
    q = int(original.h.shape[0])
    for name in ('A_cls','A_box','hard_cls','hard_box','g_cls','g_box'):
        t=out[name]
        need(tuple(t.shape)==(q,4),f'Route tensor {name} shape {tuple(t.shape)} != {(q,4)}')
        need(bool(torch.isfinite(t).all()),name+' nonfinite')
    for name in ('pi_cls','pi_box'):
        p=out[name]
        need(tuple(p.shape)==(q,5), f'{name} shape drift')
        need(bool(torch.isfinite(p).all()), f'{name} nonfinite')
        need(bool((p >= -1e-6).all()),name+' negative routing mass')
        need(bool(torch.allclose(p.sum(dim=1),torch.ones(q,device=p.device,dtype=p.dtype),atol=1e-4,rtol=0)),
             name+' simplex violation')
    for task in ('cls','box'):
        A=out['A_'+task];h=out['hard_'+task]
        need(bool(((h<=A+1e-6)&(h>=-1e-6)).all()),task+' masked route was active despite unavailability')
        inactive = (h.sum(dim=1)==0)
        current = out['z'] if task=='cls' else out['b']
        base = original.z if task=='cls' else original.b
        need(torch.equal(current[inactive],base[inactive]),'No-action per-query R0 identity violated: '+task)


def accounting(stats, classes, scale, base_matches, full_matches, labels, areas):
    """Per-frame truth accounting, requires matching GT identities not TP totals."""
    lost=base_matches-full_matches
    rescued=full_matches-base_matches
    stats['lost_gt']+=len(lost)
    stats['rescued_gt']+=len(rescued)
    for j in lost:
        classes['lost'][CLASSES[int(labels[j])]]+=1
        scale['lost'][areas[j]]+=1
    for j in rescued:
        classes['rescued'][CLASSES[int(labels[j])]]+=1
        scale['rescued'][areas[j]]+=1
    return lost, rescued


def verify_inputs(project):
    project=Path(project).resolve()
    driver=project/'run15a_internal_rebuild_PATCH03.py'
    mechanism=project/'run15a_internal_mechanism.py'
    out=project/'bfcef_f_s0'/'run15a_internal_rebuild_v2_fixed_gate'
    ckpt=out/'downstream_internal_fitted.pth'
    cert=out/'internal_fit_certificate.json'
    hashes={'driver':sha(driver),'mechanism':sha(mechanism),
            'fit_checkpoint':sha(ckpt),'fit_certificate':sha(cert)}
    need(hashes == {'driver':EXPECTED_DRIVER_SHA,'mechanism':EXPECTED_MECHANISM_SHA,
                   'fit_checkpoint':EXPECTED_FIT_SHA,'fit_certificate':EXPECTED_FIT_CERT_SHA},
         'Immutable fitted source/certificate SHA mismatch: '+str(hashes))
    certification=json_read(cert)
    need(certification['checkpoint_sha256']==EXPECTED_FIT_SHA,'Fit certificate checkpoint mismatch')
    need(certification['source_sha256']==EXPECTED_DRIVER_SHA,'Fit certificate source mismatch')
    need(certification.get('no_val_test') is True,'FIT certificate seal mismatch')
    sys.path.insert(0,str(project))
    import run15a_internal_rebuild_PATCH03 as original
    import run15a_internal_mechanism as mechanism_mod
    need(Path(original.__file__).resolve()==driver,'Loaded driver from unexpected location')
    need(Path(mechanism_mod.__file__).resolve()==mechanism,'Loaded mechanism from unexpected location')
    need(tuple(original.FIT)==tuple(certification['train_videos']) and len(original.FIT)==29,
         'Fit split drift')
    need(tuple(original.CAL)==(11,12,15,21,23,30,33,39,46), 'CAL split changed')
    need(set(original.FIT).isdisjoint(original.CAL) and
         set(original.FIT).isdisjoint(original.QUAL), 'FIT population overlaps CAL/QUAL')
    need(original.FIXED_GATE_THRESHOLD==0.02,'Fixed fitted threshold drift')
    need(tuple(original.CLASSES)==CLASSES,'Class logits mapping drift')
    return original,mechanism_mod,hashes,out


def update_gate_stats(gate, out):
    import torch
    q=int(out['z'].shape[0]);gate['all_queries']+=q
    hcl=out['hard_cls'];hbx=out['hard_box']
    for task,hard in (('cls',hcl),('box',hbx)):
        avail=out['A_'+task];routes=out['pi_'+task]
        gate[task]['candidate_queries']+=int((avail.sum(1)>0).sum().item())
        gate[task]['active_queries']+=int((hard.sum(1)>0).sum().item())
        gate[task]['closed_queries']+=int((hard.sum(1)==0).sum().item())
        gate[task]['base_mass_sum']+=float(routes[:,0].sum().item())
        gate[task]['corrective_mass_sum']+=float(routes[:,1:].sum().item())
        for i,e in enumerate(EXPERTS):
            gate[task]['experts'][e]['eligible']+=int((avail[:,i]>0).sum())
            gate[task]['experts'][e]['opened']+=int((hard[:,i]>0).sum())
            gate[task]['experts'][e]['route_mass']+=float(routes[:,i+1].sum())
    gate['both_active']+=int(((hcl.sum(1)>0)&(hbx.sum(1)>0)).sum())
    gate['neither_active']+=int(((hcl.sum(1)==0)&(hbx.sum(1)==0)).sum())


def new_stats():
    return {'n_frames':0,'gt':0,'lost_gt':0,'rescued_gt':0,
            'counts':{k:collections.Counter() for k in VARIANTS},
            'lost_against_base':{k:0 for k in VARIANTS},
            'rescue_against_base':{k:0 for k in VARIANTS},
            'knockout_rescues_full':{e:0 for e in EXPERTS},
            'knockout_losses_full':{e:0 for e in EXPERTS},
            'lost_stage':collections.Counter(),
            'classes':{'lost':collections.Counter(),'rescued':collections.Counter()},
            'scale':{'lost':collections.Counter(),'rescued':collections.Counter()},
            'class_scale':{'support':collections.Counter(), 'Base_tp':collections.Counter(), 'Full_tp':collections.Counter(),
                           'lost':collections.Counter(),'rescued':collections.Counter()},
            'gate':{'all_queries':0,'both_active':0,'neither_active':0,
                    **{task:{'candidate_queries':0,'active_queries':0,'closed_queries':0,
                             'base_mass_sum':0.,'corrective_mass_sum':0.,
                             'experts':{e:{'eligible':0,'opened':0,'route_mass':0.} for e in EXPERTS}}
                       for task in ('cls','box')}}}


def output_stats(s):
    gate=s['gate']
    ret={'frames':s['n_frames'],'gt':s['gt'],'variants':{name:metric(cnt) for name,cnt in s['counts'].items()},
         'lost_gt_against_base':s['lost_against_base'],
         'rescued_gt_against_base':s['rescue_against_base'],
         'knockout_rescues_full':s['knockout_rescues_full'],
         'knockout_losses_full':s['knockout_losses_full'],
         'loss_stages':dict(s['lost_stage']),
         'classes':{x:dict(y) for x,y in s['classes'].items()},
         'scale_proxy':{x:dict(y) for x,y in s['scale'].items()},
         'class_scale':{x:dict(y) for x,y in s['class_scale'].items()},
         'gate':gate}
    for task in ('cls','box'):
        x=gate[task]
        x['active_over_candidates']=x['active_queries']/x['candidate_queries'] if x['candidate_queries'] else None
        x['base_mass_mean_all_queries']=x['base_mass_sum']/gate['all_queries'] if gate['all_queries'] else None
        for e,y in x['experts'].items():
            y['open_over_eligible']=y['opened']/y['eligible'] if y['eligible'] else None
            y['route_mass_per_eligible']=y['route_mass']/y['eligible'] if y['eligible'] else None
    return ret


def process_video(vid, raw, original, mechanism_mod, model, co, fx, s0a, r9, *, stride):
    import torch
    can,spa,tmp=s0a.load_video_triplet(vid)
    need(len(can)>0, 'Empty FIT video')
    by_id={int(im['id']):im for im in raw['images']}
    cams={int(v['id']):str(v['camera_sensor']) for v in raw['videos']}
    annotations=collections.Counter(int(a['image_id']) for a in raw['annotations'])
    categories_by_image=collections.defaultdict(collections.Counter)
    for ann in raw['annotations']:
        if int(ann['image_id']) in by_id and int(by_id[int(ann['image_id'])]['video_id'])==vid:
            categories_by_image[int(ann['image_id'])][int(ann['category_id'])-1]+=1
    stats=new_stats();rows=[];seen=set();camera=cams[vid]
    loss_indices=[]
    need(camera in ('e-CAM','FLIR','Axis'),'Unknown camera label')
    with torch.no_grad():
        for index in range(0,len(can),stride):
            fr=can[index]
            iid=int(fr['image_id']);need(iid not in seen,'Duplicate cached FIT image ID')
            seen.add(iid)
            need(iid in by_id and int(by_id[iid]['video_id'])==vid,'FIT image belongs to another video')
            need(annotations[iid]==len(fr['gt_labels']),'FIT GT annotation count mismatch')
            need(categories_by_image[iid] == collections.Counter(int(v) for v in fr['gt_labels'].tolist()),
                 'FIT GT category multiset mismatch')
            O,X=s0a.make_O_X(fr,spa[index],tmp[index],co,fx)
            bundle=r9.expert_bundle(model,O,X)
            result=mechanism_mod.mechanism_forward(model,O,X,bundle,hard=True)
            assert_model_shapes(result,O)
            if stats['n_frames']==0:
                trainpath=mechanism_mod.mechanism_forward(model,O,X,bundle,hard=False)
                need(torch.allclose(trainpath['z'],result['z'],atol=1e-6,rtol=0) and
                     torch.allclose(trainpath['b'],result['b'],atol=1e-6,rtol=0),
                     'Real FIT hard-forward vs STE-forward mismatch')
                native_z,native_b,_=model(O,X,m_ext=None,m_hat_vis=None,m_avail=None,context_mode='auto')
                need(torch.allclose(native_z,result['z'],atol=1e-6,rtol=0) and
                     torch.allclose(native_b,result['b'],atol=1e-6,rtol=0),
                     'Real FIT production vs mechanism mismatch')
            update_gate_stats(stats['gate'],result)
            outputs={'Base':(O.z,O.b), 'Full':(result['z'],result['b']),
                     'ClsOnly':(result['z'],O.b), 'BoxOnly':(O.z,result['b'])}
            for e in EXPERTS:
                with availability_knockout(model,e):
                    ablated=mechanism_mod.mechanism_forward(model,O,X,bundle,hard=True)
                    if stats['n_frames']==0:
                        native_az,native_ab,_=model(O,X,m_ext=None,m_hat_vis=None,m_avail=None,context_mode='auto')
                        need(torch.allclose(native_az,ablated['z'],atol=1e-6,rtol=0) and
                             torch.allclose(native_ab,ablated['b'],atol=1e-6,rtol=0),
                             'Availability-knockout native vs mechanism parity failed for '+e)
                assert_model_shapes(ablated,O)
                outputs['No'+e]=(ablated['z'],ablated['b'])
            gt=fr['gt_boxes_norm'].detach().float().cpu()
            labels=fr['gt_labels'].detach().long().cpu()
            matches={}
            for name,(z,b) in outputs.items():
                m,fp,count=original.capture_match(z,b,fr['gt_boxes_norm'],fr['gt_labels'])
                matches[name] = {int(j) for q,c,j in m}
                if name=='Base': baseline_pair={int(j):(int(q),int(c)) for q,c,j in m}
                stats['counts'][name].update(count)
                need(len(matches[name])==count['tp'],'Matching GT identity duplication')
            base=matches['Base'];full=matches['Full']
            for name in VARIANTS:
                stats['lost_against_base'][name]+=len(base-matches[name])
                stats['rescue_against_base'][name]+=len(matches[name]-base)
            for e in EXPERTS:
                stats['knockout_rescues_full'][e]+=len(matches['No'+e]-full)
                stats['knockout_losses_full'][e]+=len(full-matches['No'+e])
            stats['n_frames']+=1;stats['gt']+=len(labels)
            sizes=[]
            for a in gt:
                area=float(a[2]*a[3]);sizes.append('small' if area<=32**2/640**2 else ('medium' if area<=96**2/640**2 else 'large'))
            lost,rescued=accounting(stats,stats['classes'],stats['scale'],base,full,labels,sizes)
            if lost:loss_indices.append(index)
            for j in range(len(labels)):
                key=CLASSES[int(labels[j])]+'|'+sizes[j]
                stats['class_scale']['support'][key]+=1
                if j in base: stats['class_scale']['Base_tp'][key]+=1
                if j in full: stats['class_scale']['Full_tp'][key]+=1
                if j in lost: stats['class_scale']['lost'][key]+=1
                if j in rescued: stats['class_scale']['rescued'][key]+=1
            current_top=top_pairs(result['z'])
            for j in sorted(lost):
                q,c=baseline_pair[j]
                reason,sf,iouf=loss_stage(q,c,result['b'],gt[j].to(result['b'].device),result['z'],
                                          current_top,original.single_iou)
                stats['lost_stage'][reason]+=1
                sb=float(O.z[q,c].sigmoid().item())
                ib=float(original.single_iou(O.b[q],gt[j].to(O.b.device)).item())
                rows.append({'video':vid,'image_id':iid,'frame_cache_idx':index,'camera':camera,
                             'gt_cache_idx':j,'gt_class':CLASSES[int(labels[j])],
                             'scale_proxy':sizes[j], 'baseline_query':q,'baseline_class_idx':c,
                             'stage':reason,'base_score':f'{sb:.9f}', 'full_score':f'{sf:.9f}',
                             'base_pair_iou':f'{ib:.9f}', 'full_pair_iou':f'{iouf:.9f}',
                             'cls_only_preserves_gt':int(j in matches['ClsOnly']),
                             'box_only_preserves_gt':int(j in matches['BoxOnly']),
                             **{'without_'+e+'_preserves_gt':int(j in matches['No'+e]) for e in EXPERTS}})
    original.verify_gate_thresholds(model,f'FIT audit video {vid} post-eval')
    runs=[]
    for fidx in loss_indices:
        if not runs or fidx != runs[-1][-1]+stride:runs.append([fidx])
        else:runs[-1].append(fidx)
    stats['lost_frame_bursts']={'frames_with_any_lost_gt':len(loss_indices),
          'consecutive_frame_burst_count':len(runs),
          'longest_consecutive_sampled_frames':max(map(len,runs),default=0),
          'note':'Contiguous sampled frames with a loss, NOT a tracked-object persistence statistic'}
    need(sum(s['tp']+s['fn'] for s in stats['counts'].values())==stats['gt']*len(VARIANTS),
         'GT census conservation failed')
    need(stats['lost_against_base']['Full']==sum(stats['lost_stage'].values()),
         'Lost-GT attribution rows incomplete')
    ret=output_stats(stats);ret['lost_frame_bursts']=stats['lost_frame_bursts']
    return ret,rows,len(can),camera


def merged_report(video_reports, manifest, *, stride):
    total_counts={v:collections.Counter() for v in VARIANTS}
    sums=collections.Counter()
    gate=new_stats()['gate']
    stage=collections.Counter();class_lost=collections.Counter();class_rescued=collections.Counter()
    scale_lost=collections.Counter();scale_rescued=collections.Counter()
    class_scale={k:collections.Counter() for k in ('support','Base_tp','Full_tp','lost','rescued')}
    agg_lost={v:0 for v in VARIANTS};agg_rescued={v:0 for v in VARIANTS}
    knockout_rescues=collections.Counter();knockout_losses=collections.Counter()
    by_cam={}
    for v,row in video_reports.items():
        for variant,m in row['variants'].items():
            total_counts[variant].update({k:m[k] for k in ('tp','fp','fn')})
        for k in ('frames','gt'):sums[k]+=row[k]
        for k in agg_lost:agg_lost[k]+=row['lost_gt_against_base'][k];agg_rescued[k]+=row['rescued_gt_against_base'][k]
        knockout_rescues.update(row['knockout_rescues_full']);knockout_losses.update(row['knockout_losses_full'])
        stage.update(row['loss_stages']);class_lost.update(row['classes']['lost']);class_rescued.update(row['classes']['rescued'])
        scale_lost.update(row['scale_proxy']['lost']);scale_rescued.update(row['scale_proxy']['rescued'])
        for k in class_scale:class_scale[k].update(row['class_scale'][k])
        g=row['gate']
        for k in ('all_queries','both_active','neither_active'):gate[k]+=g[k]
        for task in ('cls','box'):
            for k in ('candidate_queries','active_queries','closed_queries','base_mass_sum','corrective_mass_sum'):
                gate[task][k]+=g[task][k]
            for expert in EXPERTS:
                for k in ('eligible','opened','route_mass'):
                    gate[task]['experts'][expert][k]+=g[task]['experts'][expert][k]
        camera=row['camera'];cs=by_cam.setdefault(camera,{name:collections.Counter() for name in VARIANTS})
        for name in VARIANTS:cs[name].update({k:row['variants'][name][k] for k in ('tp','fp','fn')})
    gate_s=new_stats();gate_s['gate']=gate
    global_gate=output_stats(gate_s)['gate']
    result={'schema':VERSION,'stage':'FIT_ONLY_POSTHOC_REPLAY','stride':stride,
            'source_pins':manifest['hashes'],'FIT_videos':sorted(map(int,video_reports)),
            'FIT_video_count':len(video_reports),'FIT_cached_frames_sampled':sums['frames'],
            'FIT_gt_sampled':sums['gt'],
            'metrics':{k:metric(v) for k,v in total_counts.items()},
            'lost_gt_against_base':agg_lost,'rescued_gt_against_base':agg_rescued,
            'knockout_rescues_full':dict(knockout_rescues),'knockout_losses_full':dict(knockout_losses),
            'lost_gt_stage_counts':dict(stage),'lost_by_class':dict(class_lost),
            'rescued_by_class':dict(class_rescued),
            'lost_by_scale_cached640_proxy':dict(scale_lost),
            'rescued_by_scale_cached640_proxy':dict(scale_rescued),
            'class_x_scale_cached640_proxy':{k:dict(v) for k,v in class_scale.items()},
            'gate':global_gate,
            'camera_metrics':{cam:{name:metric(c) for name,c in variants.items()} for cam,variants in by_cam.items()},
            'per_video':video_reports,
            'limitations':[
                'FIT videos already seen by frozen R0 and full-Train experts: NOT independent full detector evaluation',
                'FIT labels were used to fit this downstream: read-only diagnostic attribution, not qualification',
                'Expert knockouts recompute gates/coordinator/router but do not retrain: conditional interventions, non-additive effects',
                'Loss-stage labels are decision-chain diagnostics, not unique causal explanations',
                'Cached image/GT universe only; original uncached frames are excluded',
                'Scale proxies use cached640 normalized area, not native-image COCO area',
                'Consecutive-frame loss bursts do not establish tracked-object temporal persistence',
                'No changes to frozen V2 weights, thresholds, previous CAL HOLD or unopened QUAL',
                'No official Validation or Test data accessed',
            ]}
    need(agg_lost['Full']==sum(stage.values()),'Global GT-stage census mismatch')
    need(result['metrics']['Full']['tp']-result['metrics']['Base']['tp']==
         agg_rescued['Full']-agg_lost['Full'],'Global identity/TP conservation mismatch')
    return result


def downstream_state_digest(original, model):
    """Pure read-only digest of the exact trainable downstream state tensors."""
    import torch
    state=original.state_downstream(model)
    h=hashlib.sha256()
    for module in sorted(state):
        for key in sorted(state[module]):
            leaf=state[module][key]
            if isinstance(leaf, dict):
                for name,tensor in sorted(leaf.items()):
                    need(isinstance(tensor,torch.Tensor),'Non-tensor downstream state member')
                    h.update(f'{module}/{key}/{name}/{tuple(tensor.shape)}/{str(tensor.dtype)}'.encode())
                    h.update(tensor.detach().contiguous().cpu().numpy().tobytes())
            else:
                need(isinstance(leaf,torch.Tensor),'Non-tensor downstream state')
                h.update(f'{module}/{key}/{tuple(leaf.shape)}/{str(leaf.dtype)}'.encode())
                h.update(leaf.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def run(project, stride):
    need(isinstance(stride,int) and stride>=1,'Stride must be integer >=1')
    original,mechanism_mod,hashes,fit_dir=verify_inputs(project)
    import torch
    contract={'schema':VERSION,'hashes':hashes,'stride':stride,'FIT_videos':list(original.FIT),
              'no_train_fit':True,'no_recalibration':True,'CAL_QUAL_official_val_test_unopened':True}
    output=fit_dir / ('fit_mechanism_audit_stride'+str(stride))
    output.mkdir(parents=True,exist_ok=True)
    contract_file=output/'contract.json'
    if contract_file.exists():need(json_read(contract_file)==contract,'Resume contract mismatch')
    else:json_new(contract_file,contract)
    # Deliberately only initialize runtime and FIT cached sources.
    r9,r12,r8,d,s0a,raw=original.imports_and_verify()
    model,criterion,co,fx=original.load_fitted(r9,r12)
    original.verify_gate_thresholds(model,'FIT read-only mechanism audit')
    model.eval()
    immutable_in_memory_digest=downstream_state_digest(original,model)
    video_summaries={}
    fields=('video','image_id','frame_cache_idx','camera','gt_cache_idx','gt_class','scale_proxy',
            'baseline_query','baseline_class_idx','stage','base_score','full_score',
            'base_pair_iou','full_pair_iou','cls_only_preserves_gt','box_only_preserves_gt',
            'without_S_preserves_gt','without_L_preserves_gt','without_C_preserves_gt','without_T_preserves_gt')
    for vid in sorted(original.FIT):
        vdir=output/f'video_{vid:02d}'
        tmpdir=output/f'video_{vid:02d}.tmp'
        p=vdir/'summary.json'; c=vdir/'lost_gt.csv'
        need(not tmpdir.exists(), 'Incomplete per-video transaction, inspect before proceeding: '+str(tmpdir))
        if vdir.exists():
            need(p.is_file() and c.is_file(),'Partial committed video directory')
            saved=json_read(p)
            need(saved['contract_sha256']==sha(contract_file) and saved['video']==vid and
                 saved['loss_csv_sha256']==sha(c),'Resume video provenance mismatch')
            summary=saved['summary']
            need(summary['lost_gt_against_base']['Full']==saved['lost_csv_rows'],
                 'Saved lost GT-row count mismatch')
            print(f'[RESUME] Verified existing FIT {vid:02d}: {summary["frames"]} frames',flush=True)
        else:
            summary,rows,cached_count,camera=process_video(vid,raw,original,mechanism_mod,model,co,fx,s0a,r9,stride=stride)
            summary['camera']=camera;summary['raw_cached_frame_count']=cached_count
            tmpdir.mkdir(parents=True,exist_ok=False)
            tc=tmpdir/'lost_gt.csv';tp=tmpdir/'summary.json'
            csv_new(tc,rows,fields)
            json_new(tp,{'schema':VERSION,'video':vid,'contract_sha256':sha(contract_file),
                         'summary':summary,'lost_csv_rows':len(rows),'loss_csv_sha256':sha(tc)})
            tmpdir.rename(vdir)
            print(f'[FIT] video {vid:02d} processed={summary["frames"]}/{cached_count}, '
                  f'lost={summary["lost_gt_against_base"]["Full"]}, '
                  f'knockout rescue={summary["knockout_rescues_full"]}',flush=True)
        video_summaries[str(vid)]=summary
    original.verify_gate_thresholds(model,'FIT audit completed')
    need(downstream_state_digest(original,model)==immutable_in_memory_digest,
         'Downstream tensor state mutated during read-only FIT audit')
    need(sha(fit_dir/'downstream_internal_fitted.pth')==EXPECTED_FIT_SHA,
         'Fitted checkpoint mutated during read-only FIT audit')
    result=merged_report(video_summaries,contract,stride=stride)
    result['in_memory_downstream_digest_unchanged']=immutable_in_memory_digest
    result['video_files']={str(v):{'json_sha256':sha(output/f'video_{v:02d}'/'summary.json'),
                                    'csv_sha256':sha(output/f'video_{v:02d}'/'lost_gt.csv')}
                           for v in sorted(original.FIT)}
    json_new(output/'fit_mechanism_audit_report.json',result)
    print('[FIT-ONLY READ-ONLY AUDIT COMPLETE]',output/'fit_mechanism_audit_report.json',flush=True)
    print('[Summary] Base',result['metrics']['Base'],'Full',result['metrics']['Full'],flush=True)
    print('[Summary] Lost',result['lost_gt_against_base']['Full'],
          'rescued',result['rescued_gt_against_base']['Full'],
          'stages',result['lost_gt_stage_counts'],flush=True)
    print('[LIMIT] Diagnostic FIT evidence only; CAL HOLD and QUAL embargo unchanged.',flush=True)


def selftest():
    import torch
    from types import SimpleNamespace
    q=5
    class Fake:
        def __init__(self):self.i=0
        def _availability(self,O,mask,task):return torch.ones(q,4)
    fake=Fake();before=fake._availability
    with availability_knockout(fake,'C'):
        t=fake._availability(None,None,'cls')
        need(bool((t[:,2]==0).all()) and bool((t[:,0]==1).all()),'Expert knockout column incorrect')
    need(fake._availability==before and '_availability' not in vars(fake),'Method not restored')
    try:
        with availability_knockout(fake,'L'):raise RuntimeError('expected test injection')
    except RuntimeError:pass
    need(fake._availability==before,'Knockout restoration after exception failed')
    logits=torch.full((300,7),-15.);boxes=torch.zeros((300,4));gt=torch.tensor([.5,.5,.2,.2]);
    logits[3,1]=torch.logit(torch.tensor(.201));boxes[3]=gt
    fn=lambda a,b: torch.tensor(1.)
    stage,s,iou=loss_stage(3,1,boxes,gt,logits,{(3,1)},fn)
    need(stage==STAGES[3],'Valid score/box should reach competition-stage')
    logits[3,1]=torch.logit(torch.tensor(.19465))
    stage,s,iou=loss_stage(3,1,boxes,gt,logits,{(3,1)},fn)
    need(stage==STAGES[0],'Below-threshold score misclassified')
    logits[3,1]=torch.logit(torch.tensor(.3))
    stage,s,iou=loss_stage(3,1,boxes,gt,logits,set(),fn)
    need(stage==STAGES[1],'Top100 eviction not recorded')
    stage,s,iou=loss_stage(3,1,boxes,gt,logits,{(3,1)},lambda a,b: torch.tensor(.29))
    need(stage==STAGES[2],'Below-IoU mechanism not recorded')
    # Identity swap can have equal TP totals yet lose a baseline GT.
    s=new_stats();base={0,1};full={1,2};classes={'lost':collections.Counter(),'rescued':collections.Counter()};
    scales={'lost':collections.Counter(),'rescued':collections.Counter()}
    lost,rescued=accounting(s,classes,scales,base,full,torch.tensor([0,1,2]),['small']*3)
    need(lost=={0} and rescued=={2} and s['lost_gt']==1,
         'GT identity swap escaped preservation accounting')
    # Synthetic output parity and no-action query-level identity.
    p=torch.zeros(5,5);p[:,0]=1
    synthetic={'z':torch.zeros(5,7),'b':torch.zeros(5,4),
               'pi_cls':p,'pi_box':p}
    for task in ('cls','box'):
        synthetic['A_'+task]=torch.ones(5,4)
        synthetic['hard_'+task]=torch.zeros(5,4)
        synthetic['g_'+task]=torch.zeros(5,4)
    o=SimpleNamespace(h=torch.zeros(5,8),z=torch.zeros(5,7),b=torch.zeros(5,4))
    assert_model_shapes(synthetic,o)
    synthetic['hard_cls'][0,0]=1
    synthetic['A_cls'][0,0]=0
    try:assert_model_shapes(synthetic,o)
    except AssertionError:pass
    else:raise AssertionError('Unavailable active expert was not rejected')
    print('[PASS] FIT-audit adversarial fixtures: expert restoration, stage precedence, '
          'identity swap, soft/no-action shape, unavailable expert route; no SeaClips opened')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('selftest','preflight','run'),default='run')
    p.add_argument('--project',default='.',help='D:/MARIVISION/H0_workbench/eval_tools')
    p.add_argument('--stride',type=int,default=1,help='1=every FIT cached frame; >1=systematic diagnostic subsample')
    a=p.parse_args()
    selftest()
    if a.phase=='preflight':
        _, _, hashes, _ = verify_inputs(a.project)
        print('[PASS] Immutable V2 FIT source/weight/certificate identity, exact FIT split; no cached video or CAL/QUAL opened')
    if a.phase=='run':run(a.project,a.stride)
