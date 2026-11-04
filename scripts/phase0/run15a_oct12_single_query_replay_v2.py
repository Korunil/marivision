"""RUN15A OCT12 V2: FIT-only native single-query action fidelity, READ ONLY.

This is a diagnostic of one classification expert residual at a time, NOT an
oracle upper bound, AP result, training procedure, or inference policy.

Built for the certified PATCH04 candidate and Pilot32 checkpoint. GT is used
to select offline counterfactuals and score outcomes; it never becomes an
inference input to the trained model.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
from PIL import Image  # preserve verified Windows import order
for _key,_value in {'R0_MSDA_BACKEND':'pytorch','OMP_NUM_THREADS':'1',
                    'MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}.items():
    os.environ.setdefault(_key,_value)
import torch
from run15a_final_windows import FIT, load_runtime, require, sha, write_new
from run15a_final_mechanism import mechanism_forward
from run15a_final_risk import RiskPolicy
from run15a_final_supervision import isolated_native_targets, pair_states, joint_transition
from run15a_final_preservation import identity_ledger

EXPERTS=('S','L','C','T')
# Production source pins from the actual PATCH04 / PATCH05 repository manifest.
# This is a new read-only diagnostic, not a modification of model source.
PINS={
 'run15a_final_mechanism.py':'fa35e32419c37af92b323c48273ff2154e7913358c5f0567b6498f811bad7a3b',
 'run15a_final_risk.py':'6f72ff8a995e181f84d1ba55ee59994930834e36d4f19ee4684750333c48e2d3',
 'run15a_final_supervision.py':'fb854a002c4675017cb548211e8167ad99a264946869924bba5a5bf80b70a3d6',
 'run15a_final_preservation.py':'325e27481a7cf40b8b12b77bfdfb8187914124f0c595d11eb81a48bfa52b3ae3',
 'run15a_final_windows.py':'cbf847e2ef594a51b6bf6e70883fb105da0ac17aa9b61d883546f0969891667f',
 'run15a_final_pilot_fit.py':'edd5086f147dd23d5c6b0f2bfe012bd1b39559676ce35046c287095df0a74182',
}
CKPT_SHA='8ffe141883aa9d7768f0e5bdbcd90a00cc2bf5386e614f3178ebcbab6fa411e9'
BENEFICIAL={3,5,6}
HARMFUL={1,2,7}


def digest_fraction(*parts):
    """Deterministic neutral subsampling independent of Python hash randomization."""
    raw=':'.join(map(str,parts)).encode('ascii')
    return int.from_bytes(hashlib.sha256(raw).digest()[:8],'big')/2**64


def mark_class_action(transition,relevant,available):
    if transition.ndim!=2 or relevant.shape!=transition.shape or available.ndim!=1 or available.shape[0]!=transition.shape[0]:
        raise ValueError('Target shapes differ')
    valid=relevant.any(-1)&available
    good=((transition==3)|(transition==5)|(transition==6))&relevant
    harm=((transition==1)|(transition==2)|(transition==7))&relevant
    u=good.any(-1)&valid
    h=harm.any(-1)&valid
    return (u & ~h),(h & valid),(valid & ~u & ~h)


def transition_lookup(matched,fps,shape,device):
    return pair_states(matched,fps,shape[0],shape[1],device)


def fp_extra(base_fp,after_fp):
    # Treat false-positive hypotheses as a set of (query,class) pairs.
    # Also return lost old FPs: suppressing an FP is not itself a GT rescue.
    old={(int(q),int(c)) for q,c in base_fp}
    new={(int(q),int(c)) for q,c in after_fp}
    return len(new-old),len(old-new)


def positive_ranking_proxy(base_z,new_z,query,transition,relevant):
    """Descriptive score/rank movement for full-action positive-class labels.

    Uses within-class count of strictly higher scores, rather than TopK's
    undefined tie breaking. A rank improvement is positive when a target TP score moves UP
    (types 3/6) or a target FP score moves DOWN (type 5).
    Does not calculate AP.
    """
    old=base_z.sigmoid();new=new_z.sigmoid()
    qo=int(query);results=[]
    for c in range(old.shape[1]):
        if not bool(relevant[qo,c]):continue
        t=int(transition[qo,c]);
        if t not in BENEFICIAL:continue
        sign=-1. if t==5 else 1.
        old_s=old[qo,c];new_s=new[qo,c]
        rank_old=int((old[:,c]>old_s).sum())
        rank_new=int((new[:,c]>new_s).sum())
        results.append({'class':c,'type':t,
            'signed_score_gain':sign*float((new_s-old_s).item()),
            'signed_rank_gain':sign*float(rank_old-rank_new)})
    return results


def single_native(model,O,delta,query,alpha,task='cls',sham=None):
    """Only ONE query can change; all other queries retain exact R50 values.

    alpha=0 bypasses native decode for strict bitwise fallback. This is NOT a
    full deployed gate simulation; alpha is an explicit diagnostic strength.
    """
    if task!='cls':raise ValueError('OCT12 first-stage replay is classification only')
    q=int(query)
    if q<0 or q>=len(O.h):raise ValueError('Query index out of range')
    if not (0<=float(alpha)<=1 and math.isfinite(float(alpha))):raise ValueError('Invalid residual amplitude')
    z=O.z.clone();b=O.b.clone()
    if float(alpha):
        d=delta if sham is None else sham
        assert d.shape==O.h[q].shape
        z[q:q+1]=model.adapter.decode_cls(O.h[q:q+1]+float(alpha)*d.reshape(1,-1))
    if not torch.equal(b,O.b):raise AssertionError('Class replay changed boxes')
    if q:
        if not torch.equal(z[:q],O.z[:q]):raise AssertionError('Other queries changed')
    if q+1<len(z):
        if not torch.equal(z[q+1:],O.z[q+1:]):raise AssertionError('Other queries changed')
    if float(alpha)==0 and not torch.equal(z,O.z):raise AssertionError('alpha=0 not bitwise R50')
    return z,b


def outcome_of_replay(matcher,O,zz,bb,gt_boxes,gt_labels,base_matches,base_fp,base_state,query,transition,relevant):
    matched,fps,_=matcher(zz,bb,gt_boxes,gt_labels)
    current=transition_lookup(matched,fps,tuple(O.z.shape),O.z.device)
    t=joint_transition(base_state,current)
    rel=relevant[query].bool()
    # Reproduce at least one source-labeled positive class transition, AND
    # do not introduce another query/class harm even at the same query.
    predicted=transition[query]
    matching_positive=bool(((predicted==t[query]) &
                            ((predicted==3)|(predicted==5)|(predicted==6)) & rel).any())
    # Parenthesize relevance explicitly; operator precedence matters.
    newly_relevant=(O.z.sigmoid()>=.20)|(zz.sigmoid()>=.20)
    actual_harm=bool(((((t==1)|(t==2)|(t==7))&newly_relevant)).any())
    harm_target_reproduced=bool(((((t[query]==predicted) &
                       ((predicted==1)|(predicted==2)|(predicted==7)) & rel)).any()))
    ledger=identity_ledger(base_matches,matched)
    new_fp,removed_fp=fp_extra(base_fp,fps)
    survived=(matching_positive and not ledger['lost'] and new_fp==0 and not actual_harm)
    return {'matching_positive':matching_positive,
            'harm_target_reproduced':harm_target_reproduced,
            'any_harmful_transition':actual_harm,
            'survived':bool(survived),
            'any_safe_benefit':bool((len(ledger['rescued'])>0 or removed_fp>0) and
                                    not ledger['lost'] and new_fp==0),
            'new_fp':new_fp,'removed_fp':removed_fp,
            'lost':len(ledger['lost']),'rescued':len(ledger['rescued']),
            'tp':len(matched),'fp':len(fps)}


def representative_type(t,rel,query):
    codes=sorted({int(v) for v in t[query][rel[query]].detach().cpu().tolist()})
    return [v for v in codes if v in BENEFICIAL],[v for v in codes if v in HARMFUL]


def snapshot_coordinator_inputs(captured_coords):
    """Snapshot BEFORE recomputation: forward hooks otherwise clobber source Q rows.

    A coordinator recomputation with Q=1 invokes the same forward hook, and
    overwrites the mutable captured dictionary. Never reuse that live object.
    """
    snap={}
    for key,triple in captured_coords.items():
        if len(triple)!=3:raise AssertionError('Coordinator hook tuple drift')
        snap[key]=tuple(t.detach().clone() for t in triple)
    return snap


def verify_tsk_output(value):
    """Check the actual gate contract, not an assumed g_hard==g_soft."""
    for name in ('g_soft','g_hard','mask_hard'):
        if name not in value:raise AssertionError('Missing '+name+' from native gate')
    gs,gh,mh=(value[name] for name in ('g_soft','g_hard','mask_hard'))
    if gs.shape!=gh.shape or gs.shape!=mh.shape or gs.ndim!=2 or gs.shape[-1]!=1:
        raise AssertionError('Native gate output shape drift')
    if not bool(torch.isfinite(gs).all()):raise AssertionError('Nonfinite TSK amplitude')
    if not bool(torch.all((mh==0)|(mh==1))):raise AssertionError('Nonbinary native TSK mask')
    if not torch.equal(gh,mh*gs):raise AssertionError('g_hard != mask_hard*g_soft')
    return {'g_soft':gs.detach().clone(),'g_hard':gh.detach().clone(),
            'mask_hard':mh.detach().clone()}


def permutation_donor(pool,query,video,frame,expert_index,draw):
    """Deterministic cross-query same-expert control (no self donor)."""
    choices=[int(i) for i in pool if int(i)!=int(query)]
    if not choices:return None
    offset=int(digest_fraction('perm',video,frame,expert_index,query)*len(choices))
    return choices[(offset+int(draw))%len(choices)]


def norm_match(source,reference):
    """Match the original residual norm; zero vectors stay zero."""
    a=reference.norm();b=source.norm()
    if float(a)==0 or float(b)==0:return torch.zeros_like(reference)
    return source*(a/b)


def score_to_threshold(logits,query,classes,threshold=.20):
    """Minimum distance of a selected class probability to operating threshold."""
    if not classes:return None
    probs=logits[int(query),classes].sigmoid()
    return float((probs-float(threshold)).abs().min().item())


def summarize_events(outcomes):
    """Every denominator is a distinct expert-query ITEM, not sham draws.

    Transition-type bins can overlap if one item has >1 useful class.
    """
    result={}; keys=sorted({(x['label'],x['mode'],x['expert']) for x in outcomes})
    # Any full-amplitude single-vs-candidate-batch outcome discrepancy makes
    # the WHOLE item ambiguous, including its paired permuted controls.
    ambiguous_ids={r['item_id'] for r in outcomes if r.get('rounding_ambiguous',False)}
    for label,mode,e in keys:
        rows=[x for x in outcomes if (x['label'],x['mode'],x['expert'])==(label,mode,e)]
        by_item=collections.defaultdict(list)
        for r in rows:by_item[r['item_id']].append(r)
        items=list(by_item.values())
        clean=[rs for rs in items if rs[0]['item_id'] not in ambiguous_ids]
        def per_item(rs,field):return sum(float(r[field]) for r in rs)/len(rs)
        per_item_survival=[per_item(rs,'survived') for rs in items]
        per_item_benefit=[per_item(rs,'any_safe_benefit') for rs in items]
        result[f'{label}/{mode}/{e}']={
            'unique_items':len(items),'draw_records':len(rows),
            'clean_unique_items':len(clean),
            'clean_mean_item_survival':sum(per_item(rs,'survived') for rs in clean)/max(1,len(clean)),
            'clean_mean_safe_benefit':sum(per_item(rs,'any_safe_benefit') for rs in clean)/max(1,len(clean)),
            'mean_survival_per_item_all_descriptive':sum(per_item_survival)/max(1,len(items)),
            'mean_safe_benefit_per_item_all_descriptive':sum(per_item_benefit)/max(1,len(items)),
            'rounding_ambiguous_items_excluded':len(items)-len(clean),
            'clean_harm_reproduction_rate':sum(per_item(rs,'harm_target_reproduced') for rs in clean)/max(1,len(clean)),
            'clean_lost_per_item_mean':sum(per_item(rs,'lost') for rs in clean)/max(1,len(clean)),
            'clean_new_fp_per_item_mean':sum(per_item(rs,'new_fp') for rs in clean)/max(1,len(clean)),
            'alpha_median':sorted(r['alpha'] for r in rows)[len(rows)//2],
        }
        if label=='positive':
            by_type={}
            for t in (3,5,6):
                elig=[rs for rs in items if t in rs[0]['positive_types']]
                elig_clean=[rs for rs in elig if rs[0]['item_id'] not in ambiguous_ids]
                by_type[str(t)]={
                    'unique_items':len(elig),'clean_unique_items':len(elig_clean),
                    'clean_mean_item_survival':sum(per_item(rs,'survived') for rs in elig_clean)/max(1,len(elig_clean)),
                    'rounding_ambiguous_items_excluded':len(elig)-len(elig_clean),
                }
            result[f'{label}/{mode}/{e}']['positive_type_strata']=by_type
            result[f'{label}/{mode}/{e}']['multiple_positive_type_items']=sum(len(rs[0]['positive_types'])>1 for rs in items)
    return result


def coord_forced_alpha(model,captured_coord,task,query,k,gsoft,availability):
    """PILOT32 INITIALIZATION ONLY: counterfactual one-expert route.

    Not the actual deployment amplitude; coordinator was never fitted on
    hard-open contexts. A zero benefit here does not refute label validity.

    Recompute the native coordination head with gate-amplitude and hard-mask
    slots changed for that one query. All other experts remain hard closed.
    This intentionally overrides the existing risk/TSK decision ONLY for an
    offline feasibility diagnostic; it is NOT deployed pi or admissibility.
    """
    if task!='cls':raise ValueError('Only classification supported')
    inputs=captured_coord.get('cls')
    if inputs is None:raise AssertionError('Coordinator input was not captured')
    ci,delta,logged=inputs
    expected=4*int(model.gate_dim)+12
    if ci.ndim!=2 or ci.shape[-1]!=expected or delta.ndim!=3 or logged.shape!=(len(ci),4):
        raise AssertionError('Coordinator interface/feature layout drift')
    q=int(query);K=4;gstart=4*int(model.gate_dim)
    if q<0 or q>=len(ci):raise AssertionError('Coordinator snapshot corrupted or query out of bounds')
    if not bool(availability[q,k]):raise AssertionError('Unavailable expert selected')
    if not 0<float(gsoft)<1:raise AssertionError('TSK amplitude not in (0,1)')
    # The original coord head output must be reproducible before intervention.
    one=ci[q:q+1].clone()
    x=one.clone()
    x[:,gstart:gstart+K]=0
    x[:,gstart+K:gstart+2*K]=0
    x[:,gstart+k]=gsoft
    x[:,gstart+K+k]=1
    coord=model.coord_cls(x,delta[q:q+1])
    g=torch.zeros((1,K),device=coord.device,dtype=coord.dtype)
    g[0,k]=gsoft
    pi=model.router(g,availability[q:q+1].to(dtype=g.dtype),coord)
    if tuple(pi.shape)!=(1,5):raise AssertionError('Router shape drift')
    if not bool(torch.isfinite(pi).all()):raise AssertionError('Nonfinite route')
    if not torch.allclose(pi.sum(-1),torch.ones(1,device=pi.device),atol=1e-6):
        raise AssertionError('Simplex mass not conserved')
    for other in range(K):
        if other!=k and float(pi[0,other+1])!=0:raise AssertionError('Non-selected expert routed')
    val=float(pi[0,k+1].item())
    if not 0<val<1:raise AssertionError('Forced-alpha is not a strict convex weight')
    return val


def validate_whole_action(model,O,bundle,gt_boxes,gt_labels,matcher,targets,baseline_state):
    """Regression check: independent replay of the same global action.

    Do not mistake this for evidence that the isolated labels are causal.
    """
    available=bundle['candidate_mask'].reshape(-1).bool()
    whole_decoded={}
    for k,e in enumerate(EXPERTS):
        z=O.z.clone();b=O.b.clone()
        z[available]=model.adapter.decode_cls(O.h[available]+bundle['delta_cls'][available,k])
        mat,fp,_=matcher(z,b,gt_boxes,gt_labels)
        state=transition_lookup(mat,fp,tuple(O.z.shape),O.z.device)
        expected=targets[e]['cls']['transition']
        if not torch.equal(joint_transition(baseline_state,state),expected):
            raise AssertionError(f'Whole-candidate target replay mismatch for {e}')
        rel=((O.z.sigmoid()>=.20)|(z.sigmoid()>=.20))&available[:,None]
        if not torch.equal(rel,targets[e]['cls']['relevant']):
            raise AssertionError(f'Whole-candidate relevant-mask mismatch for {e}')
        whole_decoded[e]=z.detach().clone()
    return whole_decoded


def replay(project,videos,frames,neutral_fraction,sham_draws):
    project=Path(project).resolve()
    require(tuple(videos)==(2,3,4) and frames==16,
            'Replay scope locked to census FIT videos 2,3,4 first 16 frames')
    for name,digest in PINS.items():require(sha(project/name)==digest,'Source drift: '+name)
    ckpt=project/'bfcef_f_s0'/'run15a_final_candidate_sandbox'/'pilot_fit'/'pilot_videos_2_3_frames16.pth'
    receipt=ckpt.with_suffix('.receipt.json')
    require(ckpt.exists() and receipt.exists() and sha(ckpt)==CKPT_SHA,'Pilot32 checkpoint mismatch')
    rec=json.loads(receipt.read_text(encoding='utf-8'))
    require(rec.get('checkpoint_sha256')==CKPT_SHA and rec.get('frames')==32,'Pilot receipt mismatch')
    old,r9,r8,d,s0a,raw,_,_,_=load_runtime(project,downstream='none')
    r12=old.imports_and_verify()[1]
    model,criterion,CO,FX=old.make_model(r9,r12)
    payload=torch.load(ckpt,map_location='cpu',weights_only=False)
    require(payload.get('checkpoint_type')=='PILOT_ONLY_NOT_CALIBRATED','Incorrect pilot checkpoint type')
    old.restore_downstream(model,payload['downstream_state'])
    old.lock_gate_thresholds(model)
    old.verify_gate_thresholds(model,'OCT12 replay')
    model.eval()
    counters=collections.Counter()
    outcomes=[]
    # These hooks expose existing native g_soft and coordinator inputs. They
    # do not change forward values or any parameters.
    captured_gates={};captured_coords={};hooks=[]
    def gate_hook(expert):
        def fn(module,inputs,output):
            captured_gates[expert]=verify_tsk_output(output)
        return fn
    def coord_hook(task):
        def fn(module,inputs,output):
            captured_coords[task]=(inputs[0].detach(),inputs[1].detach(),output.detach())
        return fn
    for e in EXPERTS:hooks.append(model.gate_cls[e].register_forward_hook(gate_hook(e)))
    hooks.append(model.coord_cls.register_forward_hook(coord_hook('cls')))
    # Controls: positive/harmful ALL; neutral deterministic hash-stratified.
    try:
        with torch.no_grad():
            for vid in videos:
                can,spa,tmp=s0a.load_video_triplet(vid)
                for j in range(min(frames,len(can))):
                    captured_gates.clear();captured_coords.clear()
                    fr=can[j]
                    O,X=s0a.make_O_X(fr,spa[j],tmp[j],CO,FX)
                    bundle=r9.expert_bundle(model,O,X)
                    out=mechanism_forward(model,O,X,bundle,hard=True,policy=RiskPolicy())
                    base_mat,base_fp,_=old.capture_match(O.z,O.b,fr['gt_boxes_norm'],fr['gt_labels'])
                    bs=transition_lookup(base_mat,base_fp,tuple(O.z.shape),O.z.device)
                    targets=isolated_native_targets(model,O,bundle,fr['gt_boxes_norm'],fr['gt_labels'],old.capture_match)
                    # S0: original global labels reproduce bitwise at transition level.
                    whole_decoded=validate_whole_action(model,O,bundle,fr['gt_boxes_norm'],fr['gt_labels'],old.capture_match,targets,bs)
                    if not (torch.equal(out['z'],O.z) and torch.equal(out['b'],O.b)):
                        raise AssertionError('Pilot32 checkpoint unexpectedly makes native intervention')
                    coord_snapshot=snapshot_coordinator_inputs(captured_coords)
                    ci,delta,logged=coord_snapshot['cls']
                    # Verify coordinator hook records match recomputed output.
                    if not torch.allclose(model.coord_cls(ci,delta),logged,atol=1e-7,rtol=0):
                        raise AssertionError('Closed-context coordinator hook not repeatable')
                    for k,e in enumerate(EXPERTS):
                        tr=targets[e]['cls']['transition'];rel=targets[e]['cls']['relevant']
                        available=out['A_cls'][:,k].bool()
                        good,bad,neutral=mark_class_action(tr,rel,available)
                        counters[f'{e}_safe_useful']+=int(good.sum())
                        counters[f'{e}_harmful']+=int(bad.sum())
                        counters[f'{e}_valid']+=int((good|bad|neutral).sum())
                        selected=(good|bad).nonzero().reshape(-1).tolist()
                        selected+= [int(q) for q in neutral.nonzero().reshape(-1).tolist()
                                    if digest_fraction(20261012,vid,j,k,int(q))<neutral_fraction]
                        counters[f'{e}_neutral_sampled']+=len(selected)-int((good|bad).sum())
                        for q in selected:
                            kind='positive' if bool(good[q]) else 'harmful' if bool(bad[q]) else 'neutral'
                            gate_value=captured_gates[e]
                            gp=float(gate_value['g_soft'][q,0].item())
                            gm=float(gate_value['g_hard'][q,0].item())
                            tsk_open=bool(gate_value['mask_hard'][q,0].item())
                            alpha_router=coord_forced_alpha(model,coord_snapshot,'cls',q,k,gp,out['A_cls'])
                            ref_pos,ref_harm=representative_type(tr,rel,q)
                            start=float(O.z[q].sigmoid().max())
                            item_id=f'{vid}:{j}:{e}:{q}'
                            rel_classes=[c for c in range(O.z.shape[1]) if bool(rel[q,c]) and int(tr[q,c]) in BENEFICIAL]
                            threshold_distance=score_to_threshold(O.z,q,rel_classes)
                            full_threshold_distance=score_to_threshold(whole_decoded[e],q,rel_classes)
                            donor_pool=available.nonzero().flatten().tolist()
                            for mode,a in [('zero',0.),('quarter',.25),('half',.5),('full',1.),('pilot32_forced_init',alpha_router)]:
                                zz,bb=single_native(model,O,bundle['delta_cls'][q,k],q,a)
                                if mode=='zero' and not torch.equal(zz,O.z):raise AssertionError('Zero not exact')
                                result=outcome_of_replay(old.capture_match,O,zz,bb,fr['gt_boxes_norm'],fr['gt_labels'],
                                        base_mat,base_fp,bs,q,tr,rel)
                                if mode=='zero' and (result['lost'] or result['rescued'] or
                                                     result['new_fp'] or result['removed_fp'] or
                                                     result['tp']!=len(base_mat) or result['fp']!=len(base_fp)):
                                    raise AssertionError('Zero-action matcher/identity ledger != R50')
                                rounding_ambiguous=False
                                max_batch_diff=0.0
                                if mode=='full':
                                    zbatch=O.z.clone();zbatch[q]=whole_decoded[e][q]
                                    max_batch_diff=float((zz[q]-zbatch[q]).abs().max().item())
                                    batch_result=outcome_of_replay(old.capture_match,O,zbatch,bb,fr['gt_boxes_norm'],fr['gt_labels'],
                                        base_mat,base_fp,bs,q,tr,rel)
                                    rounding_ambiguous=any(result[key]!=batch_result[key] for key in
                                        ('matching_positive','harm_target_reproduced','lost','rescued','new_fp','removed_fp','survived'))
                                row={'video':vid,'frame':j,'image_id':int(fr['image_id']),
                                     'expert':e,'query':int(q),'item_id':item_id,'label':kind,'positive_types':ref_pos,
                                     'harmful_types':ref_harm,'mode':mode,'alpha':a,'g_soft':gp,
                                     'tsk_hard_open':tsk_open,'g_hard':gm,
                                     'min_base_distance_to_020_on_labeled_classes':threshold_distance,
                                     'min_full_batch_distance_to_020_on_labeled_classes':full_threshold_distance,
                                     'max_single_vs_candidate_batch_logit_diff':max_batch_diff,
                                     'rounding_ambiguous':rounding_ambiguous,
                                     'baseline_max_class_score':start,
                                     'max_score_delta':float((zz[q].sigmoid()-O.z[q].sigmoid()).abs().max().item()),
                                     'positive_class_score_rank_proxies':positive_ranking_proxy(O.z,zz,q,tr,rel),
                                     **result}
                                outcomes.append(row)
                            if kind=='positive':
                                orig=bundle['delta_cls'][q,k]
                                controls=[('sign_flip_full',None,-orig)]
                                for draw in range(sham_draws):
                                    donor=permutation_donor(donor_pool,q,vid,j,k,draw)
                                    if donor is not None:
                                        # A different query's SAME-expert residual,
                                        # norm matched to the actual target residual.
                                        controls.append(('permute_full',donor,
                                            norm_match(bundle['delta_cls'][donor,k],orig)))
                                for mode,donor,ctrl in controls:
                                    zz,bb=single_native(model,O,orig,q,1.,sham=ctrl)
                                    result=outcome_of_replay(old.capture_match,O,zz,bb,fr['gt_boxes_norm'],fr['gt_labels'],
                                            base_mat,base_fp,bs,q,tr,rel)
                                    outcomes.append({'video':vid,'frame':j,'image_id':int(fr['image_id']),
                                        'expert':e,'query':int(q),'item_id':item_id,
                                        'label':kind,'positive_types':ref_pos,'harmful_types':ref_harm,
                                        'mode':mode,'donor_query':donor,'alpha':1.,'g_soft':gp,
                                        'tsk_hard_open':tsk_open,'g_hard':gm,
                                        'min_base_distance_to_020_on_labeled_classes':threshold_distance,
                                        'min_full_batch_distance_to_020_on_labeled_classes':full_threshold_distance,
                                        'max_single_vs_candidate_batch_logit_diff':0.,'rounding_ambiguous':False,
                                        'positive_class_score_rank_proxies':positive_ranking_proxy(O.z,zz,q,tr,rel),
                                        **result})
                print(f'[REPLAY FIT] video={vid} frames={min(frames,len(can))}',flush=True)
                del can,spa,tmp
    finally:
        for hook in hooks:hook.remove()
    for e,expected in {'S':12,'L':13,'C':13,'T':16}.items():
        require(counters[f'{e}_safe_useful']==expected,
                f'Compared with known census {e}, safe-useful drift: {counters[f"{e}_safe_useful"]} != {expected}')
    group=summarize_events(outcomes)
    return {
        'schema':'RUN15A_OCT12_SINGLE_QUERY_REPLAY_v2',
        'status':'FIT_READONLY_CAUSAL_ACTION_DIAGNOSTIC_NOT_QUALIFIED',
        'videos':list(videos),'frames_each':frames,'pilot32_checkpoint_sha256':CKPT_SHA,
        'source_pins':PINS,'new_script_sha256':sha(Path(__file__)),
        'class_label_census':dict(counters),'summary':group,'events':outcomes,
        'no_model_parameter_changes':True,'no_optimizer_no_training':True,
        'CAL_QUAL_official_Val_Test_unopened':True,
        'limitations':[
            'GT explicitly selects oracle diagnostic actions and scores outcomes; it is not a deployable selection policy',
            'Only cls single-query actions from 32-update pilot; NO box or multi-expert replay',
            'One-query forced-router alpha overrides hard TSK/risk admission for offline feasibility only',
            'The forced one-expert alpha is a Pilot32 initialization diagnostic ONLY (coordinator never trained open)',
            'TSK g_hard is zero for TSK-closed experts even when g_soft is positive',
            'Batch-size-sensitive classifications are logged as rounding-ambiguous and must be excluded from strict conclusions',
            'Permuted controls are item-aggregated; repeated draws are not independent observations',
            'Exact real model actions beyond sampled queries cannot be inferred from these labels',
            'No mAP/mAPca is calculated; score delta and identity counts are not AP',
            'All FIT frames were previously seen by upstream Train models; no generalization claim',
            'Sampling is deterministic and highly correlated across frames: do not assume IID confidence bounds',
        ]}


def selftest():
    tr=torch.tensor([[3,0,8],[6,1,8],[5,8,8],[8,8,8],[1,8,8],[6,7,8]],dtype=torch.long)
    rel=torch.ones(tr.shape,dtype=torch.bool)
    a=torch.ones(len(tr),dtype=torch.bool)
    g,h,n=mark_class_action(tr,rel,a)
    assert g.tolist()==[True,False,True,False,False,False]
    assert h.tolist()==[False,True,False,False,True,True]
    assert n.tolist()==[False,False,False,True,False,False]
    # Synthetic strict native fallback, unrelated query unchanged, score shifts.
    class Adapter:
        def decode_cls(self,x):return x[:,0:2]
    class Model:adapter=Adapter()
    class O:
        h=torch.tensor([[.1,.2],[.3,.4]],dtype=torch.float32)
        z=torch.tensor([[.1,.2],[.3,.4]],dtype=torch.float32)
        b=torch.zeros(2,4,dtype=torch.float32)
    o=O();m=Model();delta=torch.tensor([.2,-.1],dtype=torch.float32)
    z0,b0=single_native(m,o,delta,1,0)
    z1,b1=single_native(m,o,delta,1,1.)
    assert torch.equal(z0,o.z) and torch.equal(b0,o.b)
    assert torch.equal(z1[0],o.z[0]) and torch.equal(b1,o.b)
    assert torch.allclose(z1[1],o.z[1]+delta)
    assert fp_extra([(1,0)],[(1,0),(0,1)])==(1,0)
    r=positive_ranking_proxy(o.z,z1,1,torch.tensor([[8,8],[6,5]]),
                             torch.ones((2,2),dtype=torch.bool))
    assert len(r)==2 and r[0]['signed_score_gain']>0 and r[1]['signed_score_gain']>0
    assert 0<=digest_fraction(20261012,2,3,0,33)<1
    # Reproduce an exact one-hot baseline-simplex router counterfactual.
    class Coord:
        def __call__(self,x,delta):
            assert x.shape==(1,16) and delta.shape==(1,4,2)
            return torch.zeros((1,4))
    class Router:
        def __call__(self,g,A,c):
            mass=A*g*torch.exp(c)
            return torch.cat((1/(1+mass.sum(-1,keepdim=True)),mass/(1+mass.sum(-1,keepdim=True))),dim=-1)
    class CounterfactualModel:
        gate_dim=1
        coord_cls=Coord()
        router=Router()
    ci=torch.zeros((2,16));delta=torch.zeros((2,4,2))
    alpha=coord_forced_alpha(CounterfactualModel(),{'cls':(ci,delta,torch.zeros(2,4))},
                'cls',0,2,.02,torch.ones((2,4)))
    assert abs(alpha-(.02/1.02))<1e-7
    # No available route is allowed, even under a forced-admission diagnostic.
    try:
        coord_forced_alpha(CounterfactualModel(),{'cls':(ci,delta,torch.zeros(2,4))},
                'cls',0,2,.02,torch.tensor([[1,1,0,1],[1,1,1,1]],dtype=torch.float32))
        raise AssertionError('Unavailable expert was force-routed')
    except AssertionError as ex:
        assert 'Unavailable expert selected' in str(ex)
    gate=verify_tsk_output({'g_soft':torch.tensor([[.02],[.03]]),
        'mask_hard':torch.tensor([[0.],[1.]]),'g_hard':torch.tensor([[0.],[.03]])})
    assert gate['g_hard'][0,0]==0 and gate['g_hard'][1,0]==gate['g_soft'][1,0]
    assert permutation_donor([0,1,2],1,2,3,0,0) in (0,2)
    assert torch.isclose(norm_match(torch.tensor([3.,4.]),torch.tensor([0.,2.])).norm(),torch.tensor(2.))
    snap=snapshot_coordinator_inputs({'cls':(ci,delta,torch.zeros(2,4))})
    ci[0,0]=99
    assert float(snap['cls'][0][0,0])==0.,'Coordinator snapshot modified by live tensor'
    fake=[{'label':'positive','mode':'permute_full','expert':'S','item_id':'x',
           'survived':v,'any_safe_benefit':v,'harm_target_reproduced':False,
           'rounding_ambiguous':False,'lost':0,'new_fp':0,'alpha':1.,'positive_types':[5]}
          for v in (True,False,False)]
    group=summarize_events(fake)['positive/permute_full/S']
    assert group['unique_items']==1 and group['draw_records']==3
    assert abs(group['clean_mean_item_survival']-1/3)<1e-9
    print('[PASS] OCT12 V2: snapshot, gate contract, item-level shams, exact fallback, one-query isolation')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project',default=r'D:\MARIVISION\H0_workbench\eval_tools')
    ap.add_argument('--videos',default='2,3,4')
    ap.add_argument('--frames',type=int,default=16)
    ap.add_argument('--neutral-fraction',type=float,default=.25)
    ap.add_argument('--sham-draws',type=int,default=3)
    ap.add_argument('--selftest',action='store_true')
    args=ap.parse_args()
    if args.selftest:return selftest()
    vids=tuple(int(v.strip()) for v in args.videos.split(','))
    require(0<=args.neutral_fraction<=1,'Invalid neutral-fraction')
    require(0<=args.sham_draws<=5,'Sham draw limit exceeded')
    report=replay(Path(args.project),vids,args.frames,args.neutral_fraction,args.sham_draws)
    p=Path(args.project)/'bfcef_f_s0'/'run15a_final_candidate_sandbox'/'action_fidelity_oct12_v2'/f'single_query_cls_videos_2_3_4_frames16_neutral{round(args.neutral_fraction*100):02d}.json'
    write_new(p,report)
    print('[REPLAY COMPLETE]',p)
    print('[CLASS LABELS]',json.dumps(report['class_label_census'],sort_keys=True))
    for e in EXPERTS:
        for mode in ('full','half','pilot32_forced_init','sign_flip_full','permute_full'):
            x=report['summary'].get(f'positive/{mode}/{e}')
            if x:print(f'[POSITIVE] {e} {mode}: mean survival {x["clean_mean_item_survival"]:.3f} ' +
                        f'over {x["unique_items"]} items ({x["draw_records"]} draws), ' +
                        f'rounding ambiguous={x["rounding_ambiguous_items_excluded"]}, median-alpha {x["alpha_median"]:.5f}')
    print('[SCOPE] Read-only FIT single-query classification action fidelity. NO AP or generalization claim.')

if __name__=='__main__':main()
