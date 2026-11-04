"""TRAIN-only, read-only OOF native loss forensic. NOT a fitted safety policy.

Phase `forensic` regenerates the prior 17 TRAIN-video, four-fold OOF Expert
policy. A full, per-video parity certificate gates any counterfactual work.
All counterfactuals operate on the *fixed deployed actions*: no second policy
selection, re-sorting, fallback admission, model update, or threshold sweep.

No TUNE, previously-opened EVAL, CAL/QUAL, Validation or Test frames are loaded.
GT is used exclusively by offline forensic replay, never to propose actions.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, math
from pathlib import Path
import numpy as np
import torch
import run15a_oct12_fp_recovery_ablation as a
import run15a_oct12_preservation_first as p

SCHEMA='oct12_oof_fixed_action_harm_forensic_v1'
SOURCE_A='45ca917eec61a8571660ffbec1d0a6d8063f2c2d1b96fcdeaee4c4a422821b70'
SOURCE_P='5b339d611d8653c86def5c9f6c3ac1c955dda0480f79044a71e5d1d628a3a38c'
RECEIPT_SHA='a4d191c39c0c6a04224f1d78998dae10d6cddb5d842db1df63e2827faf1d5cf1'
# Explicitly predetermined GT-FREE collateral-footprint descriptor.
FOOTPRINT_NON_TARGET_ABS=.50
SEED=20261012
MAX_SUBSET_EVALS_PER_FRAME=1600
# Mechanism decision thresholds are diagnostic rules, not significance claims.
COVERAGE_REQUIRED=11
DOMINANT_REQUIRED=10
BENIGN_DESCRIPTOR_MAX=.15
RETAIN_FP_FRACTION=.75


def check(cond,msg):
    if not cond:raise AssertionError('[OCT12 FORENSIC HOLD] '+msg)


def hashfile(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def read_scope(project):
    project=Path(project)
    check(hashfile(a.__file__)==SOURCE_A,'Frozen ablation source drift')
    check(hashfile(p.__file__)==SOURCE_P,'Preservation OOF source drift')
    check(tuple(a.TRAIN)==(2,3,4,5,7,8,13,14,16,17,18,19,20,24,25,31,35),'TRAIN partition drift')
    check(tuple(map(tuple,p.FOLDS))==((2,7,16,20,35),(3,8,17,24),(4,13,18,25),(5,14,19,31)),'OOF folds drift')
    check(set(a.TRAIN).isdisjoint(a.TUNE) and set(a.TRAIN).isdisjoint(a.EVAL),'Video embargo violated')
    check(a.ALPHA==.25 and p.ORIGINAL_THRESHOLD==.75,'Expert deployment drift')
    path=project/'bfcef_f_s0'/'run15a_final_candidate_sandbox'/'oct12_preservation_first'/'train_oof_tune_receipt.json'
    check(path.is_file(),'Missing completed preservation OOF receipt')
    check(hashfile(path)==RECEIPT_SHA,'Immutable completed OOF receipt drift')
    r=json.loads(path.read_text(encoding='utf8'))
    check(r['status']=='HOLD_TRAIN_OOF_NO_SAFE_IMPROVEMENT','Unexpected OOF status')
    check(r['this_source_sha256']==SOURCE_P and r['base_source_sha256']==SOURCE_A,'Receipt source mismatch')
    check(r['train_videos']==list(a.TRAIN) and r['train_oof_folds']==[list(f) for f in p.FOLDS],'Receipt partition mismatch')
    check(r['excluded_previous_eval_videos']==list(a.EVAL),'Receipt EVAL embargo mismatch')
    root=path.parent
    out=root/'oct12_harm_forensic_train_oof.json'
    check(not out.exists() and not Path(str(out)+'.tmp').exists(),'Forensic output already exists; refuse overwrite')
    a.verification(project)  # model/source pins; DOES NOT open frame caches
    return r,path,out


def fixed_actions(frame,fp_model,guard_model):
    """GT-free. Mirror p.apply_guard with cutoff=1.0; return exact selected action identities."""
    pf=a.predict(fp_model,frame['expert_x'])
    risk=a.predict(guard_model,frame['expert_x'])
    props=[]
    for i,(q,c) in enumerate(frame['pairs']):
        if float(pf[i]) < p.ORIGINAL_THRESHOLD or float(risk[i]) > 1.:continue
        dz=(frame['expert_z'][:,q,c]-frame['z'][q,c]).detach().cpu().numpy()
        k=int(np.argmin(dz))
        if dz[k]>=-1e-4:continue
        props.append((float(pf[i]),int(q),int(c),k,int(i),float(risk[i])))
    # The original sorts a 4-tuple (p, q, c, k). The extra diagnostic fields
    # must not affect tie order. A stable order is enough because (p,q,c,k)
    # uniquely identifies any one pair in the original operational universe.
    props.sort(key=lambda t:t[:4],reverse=True)
    used=set();selected=[]
    for conf,q,c,k,i,risk_score in props:
        if q in used:continue
        used.add(q)
        z=frame['z'];ez=frame['expert_z'][k]
        nontarget=(ez[q]-z[q]).detach().abs().clone();nontarget[c]=0
        max_other=float(nontarget.max().item())
        selected.append({'q':q,'c':c,'k':k,'pair_index':i,
                         'fp_score':conf,'guard_score':risk_score,
                         'score':float(z[q,c].sigmoid().item()),
                         'selected_drop':float((z[q,c]-ez[q,c]).item()),
                         'max_non_target_abs_change':max_other,
                         'non_target_ge_0p50':max_other>=FOOTPRINT_NON_TARGET_ABS})
    return selected


def apply_fixed(frame,acts,indices=None,localize=()):
    """Replay fixed action indices, no candidate re-admission. `localize`: one-class changes."""
    out=frame['z'].clone()
    choose=range(len(acts)) if indices is None else indices
    local=set(localize)
    for i in choose:
        ac=acts[i];q,c,k=ac['q'],ac['c'],ac['k']
        if i in local:out[q,c]=frame['expert_z'][k,q,c]
        else:out[q]=frame['expert_z'][k,q]
    check(bool(torch.isfinite(out).all()),'Nonfinite forensic logits')
    return out


def matched_info(old,fr,logits):
    matches,fps,counts=old.capture_match(logits,fr['b'],fr['gt_boxes'],fr['gt_labels'])
    # baseline matcher triples are (query,class,GT_index).
    mapping={int(g):(int(q),int(c)) for q,c,g in matches}
    check(len(mapping)==len(matches),'Duplicate GT matches in matcher output')
    return {'matches':mapping,'fp':int(counts['fp']),'tp':int(counts['tp']),
            'fn':int(counts['fn'])}


class FrameBudgetExceeded(RuntimeError):
    pass


def make_match_cache(old,fr,acts):
    """Budgeted cached evaluator for fixed action subsets and local-only variants."""
    calls=[0];cache={}
    def assess(indices=None,localize=()):
        ii=tuple(range(len(acts))) if indices is None else tuple(sorted(indices))
        ll=tuple(sorted(set(localize)))
        check(set(ll).issubset(set(ii)),'Localization outside active fixed set')
        key=(ii,ll)
        if key not in cache:
            if calls[0]>=MAX_SUBSET_EVALS_PER_FRAME:
                raise FrameBudgetExceeded('Frame-local counterfactual cap exceeded')
            zz=apply_fixed(fr,acts,ii,ll)
            val=matched_info(old,fr,zz)
            val['lost']=sorted(set(int(t[2]) for t in fr['orig_matches'])-set(val['matches']))
            val['rescued']=sorted(set(val['matches'])-set(int(t[2]) for t in fr['orig_matches']))
            cache[key]=val;calls[0]+=1
        return cache[key]
    return assess,calls


def rank_context(fr,logits,q,c):
    flat=logits.sigmoid().reshape(-1)
    idx=int(q)*7+int(c);score=float(flat[idx])
    boundary=float(flat.topk(min(100,flat.numel())).values[-1])
    # Exact rank among flattened scores, tie order can be implementation-dependent.
    n_higher=int((flat>flat[idx]).sum().item())
    n_equal=int((flat==flat[idx]).sum().item())
    return {'score':score,'above_0p20':score>=.20,
            'strictly_higher_scores':n_higher,'score_tie_count':n_equal,
            'top100_score_boundary':boundary,'definitely_beyond_top100':n_higher>=100,
            'top100_ambiguous_due_to_ties':bool(n_higher<100<=n_higher+n_equal-1)}


def minimal_harmful_subset(assess,acts,gt):
    """Deterministic 1-minimal deletion, NOT a proven globally minimal set."""
    live=list(range(len(acts)))
    if gt not in assess(live)['lost']:return None
    for i in list(reversed(live)):
        candidate=[j for j in live if j!=i]
        if gt in assess(candidate)['lost']:
            live=candidate
    return live


def classify_loss(gt,fr,acts,assess):
    base={int(g):(int(q),int(c)) for q,c,g in fr['orig_matches']}
    check(gt in base,'Lost GT was not present in baseline')
    q,c=base[gt]
    n=len(acts);full=assess(range(n))
    check(gt in full['lost'],'Expected identity not missing after full replay')
    # Evaluate each action standalone and leave-one-out from exactly the same set.
    singleton=[i for i in range(n) if gt in assess((i,))['lost']]
    necessary=[i for i in range(n) if gt not in assess(j for j in range(n) if j!=i)['lost']]
    subset=minimal_harmful_subset(assess,acts,gt)
    check(subset is not None,'Deletion attribution failed')
    # The class-only operator changes ONLY the selected (q,c) at original
    # native amplitude; compare both full-set replacement and subset outcomes.
    all_local=assess(range(n),range(n))
    subset_all_local=assess(subset,subset)
    local_only_single=[i for i in singleton if gt in assess((i,),(i,))['lost']]
    affected=[i for i,ac in enumerate(acts) if ac['q']==q]
    orig_rank=rank_context(fr,fr['z'],q,c)
    post=apply_fixed(fr,acts)
    next_rank=rank_context(fr,post,q,c)
    single_target=[i for i in local_only_single if acts[i]['q']==q and acts[i]['c']==c]
    # Determine full-row collateral necessity in a FIXED context; leave every
    # other action unchanged and localize only the nominated action.
    collateral_context=[]
    for i in range(n):
        if gt not in assess(range(n), (i,))['lost']:
            collateral_context.append(i)
    if single_target:
        mechanism='M1_DIRECT_TRIGGER_CLASS'
        explanation='One fixed trigger-class correction suffices; class-only replay also loses the baseline GT'
    elif not affected:
        mechanism='M3_UNTOUCHED_QUERY_DISPLACEMENT'
        explanation='Originally matched query unchanged; global score ranking or greedy rematching displaced its GT (see recorded rank/match trace)'
    elif collateral_context:
        mechanism='M2_FULL_ROW_COLLATERAL'
        explanation='Localizing a selected full-row action to its trigger class prevents this loss in the same full context'
    else:
        mechanism='M4_MIXED_OR_UNRESOLVED'
        explanation='No unique direct/collateral attribution under fixed-action replays'
    # Keep M3 conditional on matching/rank trace rather than treating an untouched
    # query as proof that Top100 caused the loss.
    implic=[i for i in sorted(set(singleton+necessary+subset+collateral_context))]
    return {
      'gt':int(gt),'baseline_pair':{'q':q,'c':c},
      'full_pair_after':({'q':full['matches'][gt][0],'c':full['matches'][gt][1]}
                         if gt in full['matches'] else None),
      'mechanism':mechanism,'explanation':explanation,
      'singleton_sufficient_indices':singleton,
      'contextually_necessary_indices':necessary,
      'one_minimal_subset_indices':subset,
      'single_class_local_sufficient_indices':local_only_single,
      'direct_trigger_singleton_indices':single_target,
      'full_context_row_to_local_prevents_indices':collateral_context,
      'all_class_local_full_set_loses':bool(gt in all_local['lost']),
      'class_local_minimal_subset_loses':bool(gt in subset_all_local['lost']),
      'collateral_row_localization_prevents_even_if_M3':bool(collateral_context),
      'original_match_rank':orig_rank,'after_match_rank':next_rank,
      'attributed_action_indices':implic,
      'related_action_descriptors':[{'index':i,**acts[i]} for i in implic],
      'classification_caveat':'M1/M2/M3 are operational categories; interactions may coexist'}


def parity(ref,observed):
    exp=ref['train_oof_no_veto_nonselectable']['oof']
    for key in ('actions','lost','rescued','fp_reduction','active_videos'):
        check(observed[key]==exp[key],'Global parity '+key)
    for metric in ('tp','fp','fn'):
        check(observed['base'][metric]==exp['base'][metric],'Base parity '+metric)
        check(observed['candidate'][metric]==exp['guarded'][metric],'Candidate parity '+metric)
    check(set(observed['per_video'])==set(exp['per_video']),'Video parity key set')
    for vid,r in exp['per_video'].items():
        for k in ('actions','lost','rescued','fp_reduction','tp_delta'):
            check(observed['per_video'][vid][k]==int(r.get(k,0)),
                  f'Video {vid} parity {k}')
    check(observed['actions']==701 and observed['lost']==13 and observed['fp_reduction']==378,
          'Locked aggregate OOF contract 701/13/378')
    return {'status':'EXACT_PARITY','actions':701,'lost_gt_identities':13,
            'fp_reduction':378,'videos':len(observed['per_video'])}


def extract_oof(rt):
    frames=list(a.native_frames(rt,a.TRAIN))
    check(set(fr['video'] for fr in frames)==set(a.TRAIN),'TRAIN frame coverage drift')
    contexts={}
    for fold,hold_videos in enumerate(p.FOLDS):
        held=set(hold_videos)
        ftrain=[f for f in frames if f['video'] not in held]
        fvalid=[f for f in frames if f['video'] in held]
        X,YTP,G=p.to_training_matrix(ftrain)
        fp=a.fit_linear(X,1.-YTP,G)
        guard=p.fit_tp_guard(X,YTP,G)
        for fr in fvalid:
            key=(fr['video'],fr['frame'])
            check(key not in contexts,'Duplicate frame in OOF folds')
            contexts[key]=(fr,fp,guard,fold+1)
    check(len(contexts)==len(frames),'All TRAIN frames OOF-scored exactly once')
    return contexts


def collect_parity(contexts,old):
    base=collections.Counter();cand=collections.Counter();pv=collections.defaultdict(collections.Counter)
    harms=[];all_actions=[];actions_total=lost_total=rescued_total=0
    for (vid,j),(fr,fp,guard,fold) in sorted(contexts.items()):
        acts=fixed_actions(fr,fp,guard)
        # Exact same deployed logits as original p.apply_guard(cutoff=1).
        native,an=p.apply_guard(fr,fp,guard,1.0)
        replay=apply_fixed(fr,acts)
        check(an==len(acts),'Native action count mismatch')
        check(torch.equal(native,replay),'Fixed actions fail native bitwise parity')
        full=matched_info(old,fr,replay)
        bmap={int(g):(int(q),int(c)) for q,c,g in fr['orig_matches']}
        lost=sorted(set(bmap)-set(full['matches']))
        rescued=sorted(set(full['matches'])-set(bmap))
        bm=fr['counts'];base.update(bm)
        cand.update({k:full[k] for k in ('tp','fp','fn')})
        per=pv[str(vid)]
        per.update({'actions':len(acts),'lost':len(lost),'rescued':len(rescued),
                    'fp_reduction':int(bm['fp']-full['fp']),
                    'tp_delta':int(full['tp']-bm['tp'])})
        actions_total+=len(acts);lost_total+=len(lost);rescued_total+=len(rescued)
        if lost:harms.append({'key':(vid,j),'fr':fr,'acts':acts,'lost':lost,'full':full,'fold':fold})
        for idx,ac in enumerate(acts):
            all_actions.append({'video':int(vid),'frame':int(j),'image_id':int(fr['image_id']),
                                'fold':int(fold),'action_index':idx,**ac,
                                'in_frame_with_full_loss':bool(lost)})
    vids={v:{k:int(rec.get(k,0)) for k in ('actions','lost','rescued','fp_reduction','tp_delta')}
          for v,rec in sorted(pv.items(),key=lambda kv:int(kv[0]))}
    rec=lambda c:{k:int(c.get(k,0)) for k in ('tp','fp','fn')}
    return {'observed':{'base':rec(base),'candidate':rec(cand),
                        'actions':actions_total,'lost':lost_total,'rescued':rescued_total,
                        'fp_reduction':int(base['fp']-cand['fp']),
                        'active_videos':sum(v['actions']>0 for v in vids.values()),
                        'per_video':vids},
            'harms':harms,'actions':all_actions}


def forensic_harms(harms,old):
    events=[];implicated=set();frame_counts=[]
    for group in harms:
        fr,acts=group['fr'],group['acts'];vid,j=group['key']
        assess,calls=make_match_cache(old,fr,acts)
        # Reproduce exact full loss before attributing any event.
        full=assess(range(len(acts)))
        check(full['lost']==group['lost'],'Harm-frame loss mismatch')
        local=[]
        for gt in group['lost']:
            try:
                item=classify_loss(gt,fr,acts,assess)
            except FrameBudgetExceeded:
                base={int(g):(int(q),int(c)) for q,c,g in fr['orig_matches']}
                q,c=base[gt]
                item={'gt':int(gt),'baseline_pair':{'q':q,'c':c},
                      'mechanism':'M4_MIXED_OR_UNRESOLVED',
                      'explanation':'Prespecified 1600-evaluation frame cap reached; attribution unresolved',
                      'attributed_action_indices':[],
                      'related_action_descriptors':[],
                      'full_context_row_to_local_prevents_indices':[]}
            item.update({'video':int(vid),'frame':int(j),'image_id':int(fr['image_id']),
                         'fold':group['fold'],'frame_action_count':len(acts)})
            events.append(item)
            for i in item['attributed_action_indices']:implicated.add((vid,j,i))
        frame_counts.append({'video':int(vid),'frame':int(j),'native_actions':len(acts),
                             'lost_identities':group['lost'],'counterfactual_match_calls':calls[0]})
        print(f'[FORENSIC] video={vid} frame={j} losses={group["lost"]} actions={len(acts)} match_calls={calls[0]}',flush=True)
    return events,implicated,frame_counts


def controls_and_descriptors(contexts,old,actions,implicated):
    """Controls: whole-frame loss-free AND individually GT-loss-free actions.

    Actions in harmful frames or implicated in harm are NEVER inferred benign.
    Non-harmful in a fixed full action set alone is insufficient to call action
    benign; singleton replay is an additional observable check, not proof.
    """
    selected=[]
    action_by_frame=collections.defaultdict(list)
    for row in actions:action_by_frame[(row['video'],row['frame'])].append(row)
    for key,rows in sorted(action_by_frame.items()):
        fr,fp,guard,fold=contexts[key]
        if rows[0]['in_frame_with_full_loss'] or not fr['orig_matches']:
            continue  # no baseline GT available for a meaningful benign action control
        acts=fixed_actions(fr,fp,guard)
        check(len(acts)==len(rows),'Action ledger mismatch during controls')
        for idx,row in enumerate(rows):
            # Individual-only action can be tested independently of full policy.
            info=matched_info(old,fr,apply_fixed(fr,acts,(idx,)))
            baseline_set={int(g) for q,c,g in fr['orig_matches']}
            if baseline_set-set(info['matches']):continue
            check((key[0],key[1],idx) not in implicated,'Implicated harmful action in benign controls')
            selected.append(row)
    return selected


def permutation_null(harm,controls):
    """Predeclared descriptor contrast with within-video shuffles, descriptive only."""
    # Harm-associated action may appear in multiple identity events: deduplicate.
    def key(x):return (x['video'],x['frame'],x['action_index'])
    H=list({key(x):x for x in harm}.values());C=list({key(x):x for x in controls}.values())
    videos=sorted(set(h['video'] for h in H))
    eligible_H=[h for h in H if any(c['video']==h['video'] for c in C)]
    eligible_C=[c for c in C if c['video'] in set(h['video'] for h in eligible_H)]
    if len(eligible_H)<2 or len(eligible_C)<2:
        return {'status':'INSUFFICIENT_VIDEO_MATCHED_CONTROLS','harm_actions':len(H),
                'controls':len(C),'p_value':None}
    bh=np.array([float(x['non_target_ge_0p50']) for x in eligible_H])
    bc=np.array([float(x['non_target_ge_0p50']) for x in eligible_C])
    obs=float(bh.mean()-bc.mean())
    buckets={}
    for vid in videos:
        h=[x for x in eligible_H if x['video']==vid]
        c=[x for x in eligible_C if x['video']==vid]
        if h and c:
            buckets[vid]=(np.array([float(x['non_target_ge_0p50']) for x in h+c]),len(h))
    rng=np.random.default_rng(SEED)
    diffs=[]
    for _ in range(2048):
        aa=[];bb=[]
        for v,(pool,nh) in buckets.items():
            perm=rng.permutation(len(pool))
            aa.extend(pool[perm[:nh]]);bb.extend(pool[perm[nh:]])
        diffs.append(float(np.mean(aa)-np.mean(bb)))
    return {'status':'DESCRIPTIVE_WITHIN_VIDEO_PERMUTATION_NOT_INFERENCE',
            'harm_actions':len(H),'controls':len(C),
            'video_matched_harm':len(eligible_H),'video_matched_controls':len(eligible_C),
            'descriptor_harm_fraction':float(bh.mean()),
            'descriptor_control_fraction':float(bc.mean()),
            'observed_fraction_difference':obs,
            'permutation_null_p_two_sided':float((1+sum(abs(v)>=abs(obs) for v in diffs))/(len(diffs)+1)),
            'n_shuffles':len(diffs),'seed':SEED,
            'warning':'Events/actions correlated. Not a significance test or validation of a fitted detector.'}


def decide(events,harm_actions,controls,observed):
    counts=collections.Counter(x['mechanism'].split('_')[0] for x in events)
    n=len(events)
    if n!=13:return {'status':'HOLD_PARITY_OR_UNEXPECTED_EVENT_COUNT','mechanisms':dict(counts)}
    if len([e for e in events if e['mechanism'].split('_')[0] in ('M1','M2','M3')])<COVERAGE_REQUIRED:
        return {'status':'HOLD_ATTRIBUTION_COVERAGE','mechanisms':dict(counts)}
    if counts['M1']>=DOMINANT_REQUIRED:
        # Claude's predeclared condition: direct harms do not have systematically
        # LOWER FP confidence than same-video benign controls. GT labels here
        # are offline attribution only; no model/threshold is fitted.
        direct=[ev for ev in events if ev['mechanism'].startswith('M1')]
        mh=[float(row['fp_score']) for ev in direct
            for row in ev['related_action_descriptors']
            if row['index'] in ev.get('direct_trigger_singleton_indices',[])]
        vids={ev['video'] for ev in direct}
        mc=[float(row['fp_score']) for row in controls if row['video'] in vids]
        if not mh or not mc:
            return {'status':'HOLD_DIRECT_FP_ORDERING_NO_MATCHED_CONTROLS',
                    'mechanisms':dict(counts)}
        hm=float(np.median(mh));cm=float(np.median(mc))
        return {'status':('STOP_DIRECT_HARM_DOMINANT' if hm>=cm else
                          'HOLD_DIRECT_HARM_FP_ORDERING_NOT_ESTABLISHED'),
                'mechanisms':dict(counts),'median_harm_trigger_fp_score':hm,
                'median_same_video_control_fp_score':cm,
                'warning':'Nonindependent action samples; exploratory median contrast only'}
    if counts['M2']<DOMINANT_REQUIRED:
        return {'status':'STOP_NO_DOMINANT_ACTIONABLE_COLLATERAL','mechanisms':dict(counts)}
    nvid=len(set(e['video'] for e in events if e['mechanism'].startswith('M2')))
    if nvid<3:return {'status':'STOP_COLLATERAL_NOT_DISTRIBUTED','mechanisms':dict(counts)}
    # Mechanism M2 permits considering ONE predeclared GT-free descriptor.
    # The caller supplies unique harm-associated action rows; NOT one per GT.
    attributed=harm_actions
    # Do not let prolific unrelated benign videos dominate a harm-domain control rate.
    harm_video_set={e['video'] for e in events if e['mechanism'].startswith('M2')}
    matched_controls=[x for x in controls if x['video'] in harm_video_set]
    ctrl=sum(bool(x['non_target_ge_0p50']) for x in matched_controls)
    harm_cov=sum(int(any(ev_action['non_target_ge_0p50']
                             for ev_action in ev['related_action_descriptors']
                             if ev_action['index'] in ev['full_context_row_to_local_prevents_indices']))
                 for ev in events if ev['mechanism'].startswith('M2'))
    ctrl_rate=ctrl/len(matched_controls) if matched_controls else None
    return {'status':('DESCRIPTOR_ELIGIBLE_FOR_SINGLE_FIXED_TRAIN_REPLAY' 
                      if harm_cov>=DOMINANT_REQUIRED and ctrl_rate is not None and ctrl_rate<=BENIGN_DESCRIPTOR_MAX
                      else 'STOP_COLLATERAL_DESCRIPTOR_INSUFFICIENT'),
            'mechanisms':dict(counts),'affected_video_count':nvid,
            'harm_events_with_descriptor':harm_cov,'harm_attributed_actions':len(attributed),
            'controls':len(matched_controls),'control_descriptor_fraction':ctrl_rate,
            'next_gate':'At least 75% of 378 FP reductions on fixed TRAIN replay, if descriptor is discriminative'}


def fixed_descriptor_veto(contexts,old):
    """Only the single predeclared GT-free .50 descriptor; same OOF frames/actions.

    This is a descriptive in-TRAIN veto, never a fitted/qualified policy. It is
    executed ONLY if M2 mechanism and control-frequency checks have passed.
    """
    total=collections.Counter();base=collections.Counter();loss=0;act=0;video=set()
    for (vid,j),(fr,fp,guard,fold) in sorted(contexts.items()):
        acts=fixed_actions(fr,fp,guard)
        keep=[i for i,x in enumerate(acts) if not x['non_target_ge_0p50']]
        out=apply_fixed(fr,acts,keep)
        m=matched_info(old,fr,out)
        original={int(g) for q,c,g in fr['orig_matches']}
        loss+=len(original-set(m['matches']))
        total.update({k:m[k] for k in ('tp','fp','fn')})
        base.update(fr['counts'])
        act+=len(keep)
        if keep:video.add(vid)
    removed=base['fp']-total['fp']
    return {'actions':int(act),'active_videos':len(video),
            'fp_reduction':int(removed),'lost':int(loss),
            'retention_fraction_of_378':float(removed/378),
            'passes_prespecified_fp_retention':bool(removed >= math.ceil(378*RETAIN_FP_FRACTION)),
            'note':'Same already-inspected TRAIN; descriptive, NOT qualification'}


def run(project):
    ref,refpath,dest=read_scope(project)
    print('[PASS] sources, OOF receipt, TRAIN-only scope, embargo; no forbidden data loaded',flush=True)
    rt=a.setup_native(project);old=rt[0]
    contexts=extract_oof(rt)
    col=collect_parity(contexts,old)
    certificate=parity(ref,col['observed'])
    print('[GATE 0 PASS] 701 actions / 378 FP removed / 13 GT lost, full per-video parity',flush=True)
    events,implicated,frame_counts=forensic_harms(col['harms'],old)
    check(len(events)==13,'13 identity-level forensic events expected')
    action_map={(v['video'],v['frame'],v['action_index']):v for v in col['actions']}
    harm_action_rows=[action_map[k] for k in sorted(implicated)]
    controls=controls_and_descriptors(contexts,old,col['actions'],implicated)
    descriptor=permutation_null(harm_action_rows,controls)
    decision=decide(events,harm_action_rows,controls,col['observed'])
    descriptor_veto=None
    if decision['status']=='DESCRIPTOR_ELIGIBLE_FOR_SINGLE_FIXED_TRAIN_REPLAY':
        descriptor_veto=fixed_descriptor_veto(contexts,old)
        decision['status']=('STOP_DESCRIPTOR_GT_LOSSES_REMAIN' if descriptor_veto['lost']>0 else
                            'CONTINUE_ONLY_AS_UNQUALIFIED_MECHANISM' if
                            descriptor_veto['passes_prespecified_fp_retention'] else
                            'STOP_DESCRIPTOR_FP_BENEFIT_NOT_RETAINED')
    report={'schema':SCHEMA,'status':decision['status'],'base_source_sha256':SOURCE_A,
            'preservation_source_sha256':SOURCE_P,'original_oof_receipt_sha256':hashfile(refpath),
            'forensic_source_sha256':hashfile(Path(__file__)),
            'gate0':certificate,'aggregates':col['observed'],
            'frozen_contract':{'train_videos':list(a.TRAIN),'folds':[list(x) for x in p.FOLDS],
              'fp_cutoff':p.ORIGINAL_THRESHOLD,'expert_amplitude':a.ALPHA,
              'gt_free_footprint_threshold':FOOTPRINT_NON_TARGET_ABS,
              'minimum_classified_events':COVERAGE_REQUIRED,'dominant_mechanism_events':DOMINANT_REQUIRED,
              'max_control_descriptor_frequency':BENIGN_DESCRIPTOR_MAX,
              'min_fp_reduction_retention_fraction':RETAIN_FP_FRACTION,
              'max_match_calls_per_harm_frame':MAX_SUBSET_EVALS_PER_FRAME},
            'harm_frames':frame_counts,'gt_identity_events':events,
            'harm_associated_actions':harm_action_rows,
            'benign_control_definition':'Actions in full-policy loss-free frames, also singleton GT-loss-free',
            'control_count':len(controls),
            'control_action_descriptors':controls,
            'gt_free_descriptor_contrast':descriptor,
            'one_predeclared_descriptor_veto':descriptor_veto,'decision':decision,
            'limitations':['TRAIN-only, upstream-seen by frozen detection models',
                'One-minimal deletion subsets are not guaranteed globally minimal',
                'One GT identity can be affected by multiple actions and one action by multiple GTs',
                'M1/M2/M3 are operational categories; effects may interact',
                'Control singleton non-harm does not prove universal intervention safety',
                'Permutation null descriptive only: actions and frames are not independent',
                'Do not use previously opened EVAL, TUNE, CAL, QUAL, Val or Test',
                'No inference-time training or production policy authorized']}
    a.write_new(dest,report)
    print('[FORENSIC REPORT]',dest,'status',decision['status'],flush=True)
    return report


def selftest():
    # Contract-only test; NO native frames/model loaded.
    assert set(a.TRAIN).isdisjoint(a.TUNE) and set(a.TRAIN).isdisjoint(a.EVAL)
    assert sum(map(len,p.FOLDS))==17
    f={'z':torch.zeros((3,7)),'expert_z':torch.zeros((4,3,7)),
       'pairs':[(0,2),(0,1),(1,0)],'expert_x':np.zeros((3,18))}
    f['expert_z'][0,0,2]=-1.0;f['expert_z'][1,0,1]=-2.0
    f['expert_z'][0,1,0]=-0.2
    yes={'mean':[0.]*18,'std':[1.]*18,'w':[0.]*18,'bias':20.}
    no={'mean':[0.]*18,'std':[1.]*18,'w':[0.]*18,'bias':-20.}
    acts=fixed_actions(f,yes,no)
    out=apply_fixed(f,acts)
    native,n=p.apply_guard(f,yes,no,1.)
    check(len(acts)==n==2 and torch.equal(out,native),'Selftest: fixed actions / native parity')
    check(torch.equal(apply_fixed(f,acts,[]),f['z']),'Selftest: zero actions exact fallback')
    single=apply_fixed(f,acts,(0,),(0,))
    check(sum(int(not torch.equal(single[q],f['z'][q])) for q in range(3))==1,
          'Selftest: class-only intervention scope')
    mock={'train_oof_no_veto_nonselectable':{'oof':{'actions':701,'lost':13,'rescued':0,
          'fp_reduction':378,'active_videos':1,'base':{'tp':1,'fp':1,'fn':1},
          'guarded':{'tp':1,'fp':1,'fn':1},'per_video':{'2':{'actions':701,'lost':13,'rescued':0,'fp_reduction':378,'tp_delta':0}}}}}
    ob={'actions':701,'lost':13,'rescued':0,'fp_reduction':378,'active_videos':1,
        'base':{'tp':1,'fp':1,'fn':1},'candidate':{'tp':1,'fp':1,'fn':1},
        'per_video':{'2':{'actions':701,'lost':13,'rescued':0,'fp_reduction':378,'tp_delta':0}}}
    assert parity(mock,ob)['status']=='EXACT_PARITY'
    try:parity(mock,{**ob,'lost':12})
    except AssertionError:pass
    else:raise AssertionError('Parity fail-open')
    print('[PASS] OOF frozen-action selection, bitwise native parity, class-local replay, hard parity gate')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--phase',required=True,choices=('selftest','verify','forensic'))
    ap.add_argument('--project',default=str(a.BASE))
    opt=ap.parse_args()
    if opt.phase=='selftest':selftest()
    elif opt.phase=='verify':
        ref,refpath,dest=read_scope(opt.project)
        print('[PASS] Frozen source/OOF pins and TRAIN-only embargo')
        print('[NEXT OUTPUT]',dest)
    else:run(opt.project)

if __name__=='__main__':main()
