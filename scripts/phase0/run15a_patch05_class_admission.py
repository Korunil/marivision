"""RUN15A PATCH05 -- bounded FIT-only classification-admission learning.

New downstream experiment. Leaves all certified sources and old checkpoints
untouched. Isolated GT full-amplitude action labels are TRAINING EVIDENCE only;
not deployable safety guarantees. Box pathway stays in frozen architecture.

Usage:
  python run15a_patch05_class_admission.py --phase selftest
  python run15a_patch05_class_admission.py --phase train --videos 2,3,4 --frames 16
  python run15a_patch05_class_admission.py --phase eval --videos 5,6 --stride 10
"""
from __future__ import annotations
import argparse, collections, hashlib, json, math, os, sys
from pathlib import Path
from PIL import Image  # preserve proven Windows import order
os.environ.setdefault('R0_MSDA_BACKEND', 'pytorch')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import torch
import torch.nn.functional as F
from run15a_final_risk import RiskPolicy, risk_admission
from run15a_final_mechanism import mechanism_forward
from run15a_final_supervision import isolated_native_targets, expert_deep_supervision
from run15a_final_preservation import protected_gt_loss,identity_ledger
from run15a_final_windows import load_runtime,fit_eval,require,write_new,sha,FIT

EXPERTS=('S','L','C','T')
PIN={
 'run15a_final_risk.py':'6f72ff8a995e181f84d1ba55ee59994930834e36d4f19ee4684750333c48e2d3',
 'run15a_final_mechanism.py':'fa35e32419c37af92b323c48273ff2154e7913358c5f0567b6498f811bad7a3b',
 'run15a_final_supervision.py':'fb854a002c4675017cb548211e8167ad99a264946869924bba5a5bf80b70a3d6',
 'run15a_final_preservation.py':'325e27481a7cf40b8b12b77bfdfb8187914124f0c595d11eb81a48bfa52b3ae3',
 'run15a_final_windows.py':'cbf847e2ef594a51b6bf6e70883fb105da0ac17aa9b61d883546f0969891667f',
 'run15a_final_pilot_fit.py':'edd5086f147dd23d5c6b0f2bfe012bd1b39559676ce35046c287095df0a74182'
}
PILOT_SHA='8ffe141883aa9d7768f0e5bdbcd90a00cc2bf5386e614f3178ebcbab6fa411e9'
SCHEMA='run15a_patch05_cls_admission_v1'
POLICY=RiskPolicy()


def verify(project):
    for name,expected in PIN.items():
        require(sha(project/name)==expected,'Certified candidate source drift: '+name)
    original=project/'bfcef_f_s0'/'run15a_final_candidate_sandbox'/'pilot_fit'/'pilot_videos_2_3_frames16.pth'
    receipt=original.with_suffix('.receipt.json')
    require(original.exists() and receipt.exists(),'Existing 32-update pilot is missing')
    require(sha(original)==PILOT_SHA,'Unexpected 32-update pilot checkpoint')
    r=json.loads(receipt.read_text(encoding='utf8'))
    require(r['checkpoint_sha256']==PILOT_SHA and r['frames']==32,'Existing pilot receipt drift')
    return original


def labels(transition,relevant,available):
    """Safe useful iff the *same* query has useful and no harmful transitions.

    GT supervision only. Ambiguous mixed outcomes are HARM, never positive.
    A query without relevant class transitions is neutral, not a positive.
    """
    if transition.shape!=relevant.shape or available.ndim!=1 or available.shape[0]!=transition.shape[0]:
        raise ValueError('Class-action target shape mismatch')
    if transition.ndim!=2 or relevant.dtype!=torch.bool or available.dtype!=torch.bool:
        raise ValueError('Class-action target dtype/rank mismatch')
    useful=(((transition==3)|(transition==5)|(transition==6))&relevant).any(dim=-1)
    harm=(((transition==1)|(transition==2)|(transition==7))&relevant).any(dim=-1)
    good=useful & ~harm & available
    unsafe=harm & available   # includes mixed useful-and-harmful queries
    neutral=~useful & ~harm & available
    if bool((good & unsafe).any()) or bool((neutral & (good|unsafe)).any()):
        raise AssertionError('Action label sets overlap')
    return good,unsafe,neutral


def real_cls_risk(out,O,expert,policy):
    """Exactly reproduce cls risk evidence used by mechanism_forward()."""
    posterior=out['cls_raw'][expert]
    if posterior.shape!=(*O.z.shape,9):
        raise AssertionError('Transition posterior shape drift')
    high=O.z.sigmoid()>=.20
    at_risk=((posterior[...,1]+posterior[...,2])*high).amax(-1,keepdim=True)
    introduced_fp=posterior[...,7].amax(-1,keepdim=True)
    channels=dict(out['cls_channels'][expert])
    channels['p_loss']=torch.maximum(channels['p_loss'],at_risk)
    channels['p_add']=torch.maximum(channels['p_add'],introduced_fp)
    return risk_admission(channels,'cls',policy)


def balanced_admission_loss(score,gsoft,threshold,good,unsafe,neutral,*,
                            risk_temperature=.15,tsk_temperature=.5):
    """No fixed-gate threshold changes. Equalize per-outcome loss groups.

    Risk margin >0 admits; TSK gsoft > locked threshold admits.  Positive
    labels only when the isolated full expert action is useful without harm.
    Harm or mixed cases have higher rejection cost; neutrality has weak
    abstention regularization.  Does not use GT at inference.
    """
    if score.ndim!=1 or score.shape!=gsoft.shape:
        raise ValueError('Expected per-query score and gsoft vectors')
    if not (good.shape==unsafe.shape==neutral.shape==score.shape):
        raise ValueError('Invalid per-query label shape')
    if not 0 < float(threshold) < 1:
        raise ValueError('Invalid frozen TSK threshold')
    if not (torch.isfinite(score).all() and torch.isfinite(gsoft).all()):
        raise ValueError('Nonfinite admission evidence')
    # Explicitly no loss for unavailable queries; callers construct masks.
    gsoft=gsoft.clamp(1.e-6,1-1.e-6)
    thresh=torch.as_tensor(float(threshold),device=score.device,dtype=score.dtype)
    gate_margin=torch.logit(gsoft)-torch.logit(thresh)
    zero=score.sum()*0+gsoft.sum()*0
    terms={}
    for name,mask in [('safe_useful',good),('harmful',unsafe),('neutral',neutral)]:
        if bool(mask.any()):
            if name=='safe_useful':
                r=F.softplus(-score[mask]/risk_temperature).mean()
                t=F.softplus(-gate_margin[mask]/tsk_temperature).mean()
            else:
                r=F.softplus(score[mask]/risk_temperature).mean()
                t=F.softplus(gate_margin[mask]/tsk_temperature).mean()
            terms[name]=(r+t)/2
        else:terms[name]=zero
    # Harm prioritized; neutral weak. Existence of positive is never invented.
    total=1.0*terms['safe_useful']+2.0*terms['harmful']+0.08*terms['neutral']
    return total,terms


def capture_gate_soft(model):
    """Observe exact TSK gsoft from production forward without editing it."""
    store={}
    hooks=[]
    for expert in EXPERTS:
        def hook(module,inputs,output,expert=expert):
            store[expert]=output['g_soft']
        hooks.append(model.gate_cls[expert].register_forward_hook(hook))
    return store,hooks


def create_model(project,checkpoint):
    old,r9,r8,d,s0a,raw,*_=load_runtime(project,downstream='none')
    r12=old.imports_and_verify()[1]
    model,criterion,CO,FX=old.make_model(r9,r12)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    require(payload.get('checkpoint_type')=='PILOT_ONLY_NOT_CALIBRATED','Wrong parent checkpoint')
    require(tuple(payload['video_ids'])==(2,3) and payload['max_frames_per_video']==16,'Wrong parent pilot identity')
    old.restore_downstream(model,payload['downstream_state'])
    old.lock_gate_thresholds(model)
    old.verify_gate_thresholds(model,'PATCH05 restored pilot')
    return old,r9,r8,d,s0a,raw,model,criterion,CO,FX


def training(project,videos,frames):
    require(len(videos)>0 and len(videos)<=3 and len(set(videos))==len(videos) and all(v in FIT for v in videos),
            'Train with <=3 distinct FIT videos only')
    require(1<=frames<=24,'Bounded PATCH05: 1-24 frames/video')
    # Refuse an accidental rerun BEFORE loading data or updating parameters.
    outdir=project/'bfcef_f_s0'/'run15a_final_candidate_sandbox'/'class_admission_patch05'
    outfile=outdir/('cls_fit_videos_'+'_'.join(map(str,videos))+f'_frames{frames}.pth')
    require(not outfile.exists() and not outfile.with_suffix('.receipt.json').exists(),
            'PATCH05 result already exists: preserve and evaluate it, do not retrain')
    checkpoint=verify(project)
    old,r9,r8,d,s0a,raw,model,criterion,CO,FX=create_model(project,checkpoint)
    r9.set_training_modes(model,joint=True)
    model.train()
    for m in (model.expert_s,model.expert_l,model.expert_c,model.expert_t,model.adapter,model.context_bus):m.eval()
    old.lock_gate_thresholds(model)
    trainable={name:p for name,p in model.named_parameters() if p.requires_grad}
    require(trainable,'No downstream trainable parameters')
    optimizer=torch.optim.AdamW(trainable.values(),lr=1e-4,weight_decay=1e-4)
    old.verify_gate_thresholds(model,'PATCH05 before',optimizer)
    initial={name:p.detach().clone() for name,p in trainable.items() if
             name.startswith(('gate_cls.','self_assessment.','coord_cls.'))}
    store,hooks=capture_gate_soft(model)
    step_records=[]
    counts=collections.Counter()
    try:
        for vid in videos:
            can,spa,tmp=s0a.load_video_triplet(vid)
            for j in range(min(frames,len(can))):
                fr=can[j];O,X=s0a.make_O_X(fr,spa[j],tmp[j],CO,FX)
                bundle=r9.expert_bundle(model,O,X)
                targets=isolated_native_targets(model,O,bundle,fr['gt_boxes_norm'],fr['gt_labels'],old.capture_match)
                optimizer.zero_grad(set_to_none=True)
                store.clear()
                out=mechanism_forward(model,O,X,bundle,hard=False,policy=POLICY)
                require(set(store)==set(EXPERTS),'TSK evidence hook incomplete')
                target_loss=out['z'].sum()*0
                frame_counts=collections.Counter()
                terms_debug={}
                for k,expert in enumerate(EXPERTS):
                    g,h,n=labels(targets[expert]['cls']['transition'],targets[expert]['cls']['relevant'],
                                 out['A_cls'][:,k].bool())
                    risk=real_cls_risk(out,O,expert,POLICY)
                    gsoft=store[expert].reshape(-1)
                    require(gsoft.shape==g.shape,'TSK gsoft shape mismatch')
                    thresh=float(model.gate_cls[expert].mask_thresh.detach().item())
                    l,t=balanced_admission_loss(risk['score'].reshape(-1),gsoft,thresh,g,h,n)
                    target_loss=target_loss+l/len(EXPERTS)
                    for key,mask in [('safe_useful',g),('harmful',h),('neutral',n)]:
                        frame_counts[f'{expert}_{key}']+=int(mask.sum())
                    frame_counts[f'{expert}_accepted_good']+=int((g & (out['hard_cls'][:,k]>0)).sum())
                    frame_counts[f'{expert}_accepted_harmful']+=int((h & (out['hard_cls'][:,k]>0)).sum())
                    terms_debug[expert]=float(l.detach())
                deep,details=expert_deep_supervision(out,targets)
                matched,_,_=old.capture_match(O.z,O.b,fr['gt_boxes_norm'],fr['gt_labels'])
                preservation=protected_gt_loss(O.z,O.b,out['z'],out['b'],fr['gt_boxes_norm'],matched)
                detection=r9.detection_loss(criterion,out['z'],out['b'],fr)
                route=(out['pi_cls'][:,1:].square().mean()+out['pi_box'][:,1:].square().mean())
                # Deep U/H calibration remains in the objective; admission
                # learning cannot simply replace actual consequence targets.
                loss=deep + 5.0*preservation['total'] + 0.20*detection + 0.003*route + 0.75*target_loss
                require(bool(torch.isfinite(loss)),'Nonfinite PATCH05 objective')
                loss.backward()
                grad={}
                for label,prefix in [('gate_cls','gate_cls.'),('self_assessment','self_assessment.')]:
                    grad[label]=sum(float(p.grad.detach().abs().sum()) for name,p in trainable.items()
                                    if name.startswith(prefix) and p.grad is not None)
                require(grad['gate_cls']>0 and grad['self_assessment']>0,'Classification admission gradients not received')
                torch.nn.utils.clip_grad_norm_(trainable.values(),0.1)
                optimizer.step()
                old.verify_gate_thresholds(model,f'PATCH05 video={vid} frame={j}',optimizer)
                with torch.no_grad():
                    model.eval()
                    post=mechanism_forward(model,O,X,bundle,hard=True,policy=POLICY)
                    newmatch,_,_=old.capture_match(post['z'],post['b'],fr['gt_boxes_norm'],fr['gt_labels'])
                    ledger=identity_ledger(matched,newmatch)
                    cls_open=int((post['hard_cls'].sum(-1)>0).sum())
                    box_open=int((post['hard_box'].sum(-1)>0).sum())
                    model.train()
                    for m in (model.expert_s,model.expert_l,model.expert_c,model.expert_t,model.adapter,model.context_bus):m.eval()
                counts.update(frame_counts)
                rec={'video':vid,'frame':j,'loss':float(loss.detach()),'deep':float(deep.detach()),
                     'admission_loss':float(target_loss.detach()),'detector_loss':float(detection.detach()),
                     'classification_open_after_update':cls_open,'box_open_after_update':box_open,
                     'GT_lost_after_update':len(ledger['lost']),'GT_rescued_after_update':len(ledger['rescued']),
                     'grad_l1':grad,'class_positive':sum(frame_counts[f'{e}_safe_useful'] for e in EXPERTS),
                     'class_harmful':sum(frame_counts[f'{e}_harmful'] for e in EXPERTS)}
                step_records.append(rec)
                print(f'[P05] v={vid} frame={j} L={rec["loss"]:.4f} risk={rec["admission_loss"]:.4f} '
                      f'GT+={rec["GT_rescued_after_update"]} GT-={rec["GT_lost_after_update"]} '
                      f'cls_open={cls_open} box_open={box_open}',flush=True)
            del can,spa,tmp
    finally:
        for hook in hooks:hook.remove()
    old.verify_gate_thresholds(model,'PATCH05 completed',optimizer)
    require(sum(counts[f'{e}_safe_useful'] for e in EXPERTS)>0,
            'No positive classification actions actually seen: do not save an admission-trained checkpoint')
    changes={}
    for prefix in ('gate_cls.','self_assessment.','coord_cls.'):
        keys=[k for k in initial if k.startswith(prefix)]
        changes[prefix[:-1]]={'changed_parameters':sum(not torch.equal(initial[k],trainable[k].detach()) for k in keys),
            'total_parameters':len(keys),'update_l1':sum(float((initial[k]-trainable[k].detach()).abs().sum()) for k in keys)}
    require(changes['gate_cls']['changed_parameters']>0 and changes['self_assessment']['changed_parameters']>0,
            'No actual downstream learning in class gate/UH')
    outdir=project/'bfcef_f_s0'/'run15a_final_candidate_sandbox'/'class_admission_patch05'
    outfile=outdir/('cls_fit_videos_'+'_'.join(map(str,videos))+f'_frames{frames}.pth')
    receipt=outfile.with_suffix('.receipt.json')
    require(not outfile.exists() and not receipt.exists(),'PATCH05 refuses overwriting existing result')
    outdir.mkdir(parents=True,exist_ok=True)
    torch.save({'schema':SCHEMA,'checkpoint_type':'PATCH05_CLASS_ADMISSION_FIT_ONLY',
        'parent_sha256':PILOT_SHA,'source_sha256':sha(Path(__file__)),
        'fit_videos':list(videos),'frames_per_video':frames,
        'downstream_state':old.state_downstream(model)},outfile)
    report={'schema':SCHEMA,'status':'FIT_PILOT_ONLY_NOT_QUALIFIED','parent_sha256':PILOT_SHA,
        'checkpoint_sha256':sha(outfile),'trained_videos':list(videos),'frames_per_video':frames,
        'steps':len(step_records),'source_sha256':sha(Path(__file__)),'source_pins':PIN,
        'changed_parameter_groups':changes,'target_counts':dict(counts),'steps_detail':step_records,
        'CAL_QUAL_Val_Test_unopened':True,'NO_R0_OR_EXPERT_TRAINING':True,
        'limitations':['Source FIT labels are isolated full-amplitude outcomes; no joint-safety guarantee',
          'Fitting losses on sequential different frames are not convergence metrics',
          'Rare box rescues do not support forcing box risk admission',
          'Held-out FIT videos for the new downstream do not imply end-to-end model independence',
          'No metric superiority without paired AP evaluation']}
    write_new(receipt,report)
    print('[PATCH05 FIT CHECKPOINT]',outfile,'\n[RECEIPT]',receipt,flush=True)
    return report


def evaluation(project,checkpoint_name,videos,stride):
    require(videos and len(set(videos))==len(videos) and all(v in FIT for v in videos),'FIT-only evaluation')
    require(1<=stride<=100,'Invalid stride')
    verify(project)
    cp=project/'bfcef_f_s0'/'run15a_final_candidate_sandbox'/'class_admission_patch05'/checkpoint_name
    require(cp.exists() and cp.with_suffix('.receipt.json').exists(),'PATCH05 trained checkpoint missing')
    receipt=json.loads(cp.with_suffix('.receipt.json').read_text(encoding='utf-8'))
    require(receipt.get('checkpoint_sha256')==sha(cp),'PATCH05 checkpoint SHA mismatch')
    require(receipt.get('source_sha256')==sha(Path(__file__)),'PATCH05 code drift')
    trained=set(receipt['trained_videos'])
    require(not set(videos).intersection(trained),'Use different FIT videos from those used for PATCH05 training')
    old,r9,r8,d,s0a,raw,model,criterion,CO,FX=create_model(project,verify(project))
    payload=torch.load(cp,map_location='cpu',weights_only=False)
    require(payload['checkpoint_type']=='PATCH05_CLASS_ADMISSION_FIT_ONLY' and payload['parent_sha256']==PILOT_SHA,
            'Invalid PATCH05 checkpoint payload')
    old.restore_downstream(model,payload['downstream_state'])
    old.lock_gate_thresholds(model)
    old.verify_gate_thresholds(model,'PATCH05 FIT evaluation')
    model.eval()
    runtime=(old,r9,r8,d,s0a,raw,model,CO,FX)
    result=fit_eval(runtime,videos,stride)
    result['schema']=SCHEMA+'_evaluation'
    result['status']='FIT_PAIRED_AP_DIAGNOSTIC_NOT_QUALIFIED'
    result['existing_V2_weights_warm_start']=False
    result['NO_new_training']=False
    result['trained_patch05_checkpoint_sha256']=sha(cp)
    result['trained_patch05_videos']=sorted(trained)
    result['CAL_QUAL_official_val_test_unopened']=True
    result['limitations']=['FIT-only paired AP on distinct downstream pilot videos; upstream networks saw all Train',
        'Do not compare absolute FIT mAP with SeaClips published Test mAP',
        'Global baseline GT identity safety must be assessed across prospective videos',
        'This is neither CAL nor QUAL nor official Test']
    out=cp.parent/(cp.stem+'_eval_'+'_'.join(map(str,videos))+f'_stride{stride}.json')
    write_new(out,result)
    print('[PATCH05 PAIRED FIT AP REPORT]',out)
    for name,m in result['coco_metrics'].items():
        print(name,'mAP@.30',m['map30'],'mAPca@.30',m['mapca30'])
    print('lost',result['baseline_matched_gt_lost'],'rescued',result['previously_missed_gt_rescued'],
          'gate_counts',result['gate_counts'])
    return result


def selftest():
    trans=torch.tensor([[3,0],[5,8],[1,6],[8,8],[6,7],[1,8],[3,6]])
    rel=torch.ones_like(trans,dtype=torch.bool)
    avail=torch.ones(len(trans),dtype=torch.bool)
    good,bad,neu=labels(trans,rel,avail)
    assert good.tolist()==[True,True,False,False,False,False,True]
    assert bad.tolist()==[False,False,True,False,True,True,False]
    assert neu.tolist()==[False,False,False,True,False,False,False]
    r=torch.tensor([-0.4,0.3,-0.2,0.1],requires_grad=True)
    s=torch.tensor([.018,.022,.025,.019],requires_grad=True)
    a=torch.tensor([True,False,False,False]);b=torch.tensor([False,True,False,False]);n=torch.tensor([False,False,True,True])
    loss,_=balanced_admission_loss(r,s,.020,a,b,n)
    assert bool(torch.isfinite(loss))
    loss.backward()
    assert bool(torch.isfinite(r.grad).all() and torch.isfinite(s.grad).all())
    assert r.grad[0]<0 and s.grad[0]<0  # beneficial pushes admission UP
    assert r.grad[1]>0 and s.grad[1]>0  # harmful pushes admission DOWN
    # Neither mixed harms nor non-candidate queries become positive.
    ga,ba,na=labels(trans,rel,torch.tensor([0,0,0,0,0,0,0]).bool())
    assert not (ga|ba|na).any()
    print('[PASS] PATCH05 classification labels, mixed-harm dominance, direct risk/TSK gradients')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',required=True,choices=('selftest','train','eval'))
    p.add_argument('--project',default=r'D:\MARIVISION\H0_workbench\eval_tools')
    p.add_argument('--videos',default='2,3,4')
    p.add_argument('--frames',type=int,default=16)
    p.add_argument('--stride',type=int,default=10)
    p.add_argument('--checkpoint',default='cls_fit_videos_2_3_4_frames16.pth')
    a=p.parse_args()
    if a.phase=='selftest':return selftest()
    project=Path(a.project).resolve()
    vids=tuple(int(i) for i in a.videos.split(','))
    if a.phase=='train':training(project,vids,a.frames)
    else:evaluation(project,a.checkpoint,vids,a.stride)

if __name__=='__main__':main()
