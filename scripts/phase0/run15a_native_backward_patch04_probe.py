"""PATCH04 production-R50 FIT-only backward check; no training or optimizer update.

Source-pinned to PATCH01 parity, PATCH02 preservation and PATCH03 joint STE.
The original PATCH03 probe is preserved unmodified.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch
from run15a_final_mechanism import mechanism_forward
from run15a_final_risk import RiskPolicy
from run15a_final_windows import load_runtime

EXPECTED_RISK_SHA = '6f72ff8a995e181f84d1ba55ee59994930834e36d4f19ee4684750333c48e2d3'
EXPECTED_MECHANISM_SHA = 'fa35e32419c37af92b323c48273ff2154e7913358c5f0567b6498f811bad7a3b'
EXPECTED_PRESERVATION_SHA = '325e27481a7cf40b8b12b77bfdfb8187914124f0c595d11eb81a48bfa52b3ae3'
EXPECTED_WINDOWS_SHA = 'cbf847e2ef594a51b6bf6e70883fb105da0ac17aa9b61d883546f0969891667f'
EXPECTED_PILOT_SHA = 'edd5086f147dd23d5c6b0f2bfe012bd1b39559676ce35046c287095df0a74182'


def check(cond, msg):
    if not cond:
        raise AssertionError('[PATCH04 NATIVE GRADIENT HOLD] ' + msg)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(module):
    grads=[p.grad.detach().float() for p in module.parameters() if p.grad is not None]
    return {'l1':sum(float(g.abs().sum().item()) for g in grads),
            'finite':all(bool(torch.isfinite(g).all()) for g in grads),
            'n_non_none':len(grads)}


def evaluate(project:Path):
    files={
       'run15a_final_risk.py':EXPECTED_RISK_SHA,
       'run15a_final_mechanism.py':EXPECTED_MECHANISM_SHA,
       'run15a_final_preservation.py':EXPECTED_PRESERVATION_SHA,
       'run15a_final_windows.py':EXPECTED_WINDOWS_SHA,
       'run15a_final_pilot_fit.py':EXPECTED_PILOT_SHA}
    for name,h in files.items():
        check(sha(project/name)==h,'Candidate source mismatch: '+name)
    old,r9,r8,d,s0a,raw,_,_,_=load_runtime(project,downstream='none')
    r12=old.imports_and_verify()[1]
    model,criterion,CO,FX=old.make_model(r9,r12)
    r9.set_training_modes(model,joint=True)
    old.lock_gate_thresholds(model)
    params=[p for p in model.parameters() if p.requires_grad]
    check(bool(params),'No trainable downstream params')
    model.train()
    for x in (model.expert_s,model.expert_l,model.expert_c,model.expert_t,model.adapter,model.context_bus): x.eval()
    can,spa,tmp=s0a.load_video_triplet(2)
    frame=can[0]
    O,X=s0a.make_O_X(frame,spa[0],tmp[0],CO,FX)
    bundle=r9.expert_bundle(model,O,X)
    out=mechanism_forward(model,O,X,bundle,hard=False,policy=RiskPolicy())
    check(torch.equal(out['z'],O.z) and torch.equal(out['b'],O.b),'All-closed forward differs from baseline')
    check(int(out['hard_cls'].sum()+out['hard_box'].sum())==0,'Probe requires initially closed gates')
    check(int(bundle['candidate_mask'].sum())>0,'No eligible candidate')
    # Backward diagnostic 1: exact-zero route; an independent check of PATCH03.
    for p in params:p.grad=None
    route=(out['pi_cls'][:,1:].sum()+out['pi_box'][:,1:].sum())
    check(float(route.detach().item())==0.,'Closed route has positive forward mass')
    route.backward(retain_graph=True)
    route_grad={k:stats(v) for k,v in [('gate_cls',model.gate_cls),('gate_box',model.gate_box),
                    ('self_assessment',model.self_assessment),('coord_cls',model.coord_cls),('coord_box',model.coord_box)]}
    print('[PATCH04] route-grad:',json.dumps(route_grad,sort_keys=True),flush=True)
    check(all(s['finite'] for s in route_grad.values()),'Nonfinite route gradients')
    check(route_grad['gate_cls']['l1']>0 and route_grad['gate_box']['l1']>0 and route_grad['self_assessment']['l1']>0,
          'Joint-STE route cannot train both gates and self-assessment')
    # Backward diagnostic 2: genuine detector supervision on EXACT baseline forward.
    for p in params:p.grad=None
    detection=r9.detection_loss(criterion,out['z'],out['b'],frame)
    check(bool(torch.isfinite(detection)),'Nonfinite detector loss')
    detection.backward()
    det_grad={k:stats(v) for k,v in [('gate_cls',model.gate_cls),('gate_box',model.gate_box),
                    ('self_assessment',model.self_assessment),('coord_cls',model.coord_cls),('coord_box',model.coord_box)]}
    print('[PATCH04] detector-grad:',json.dumps(det_grad,sort_keys=True),flush=True)
    check(all(s['finite'] for s in det_grad.values()),'Nonfinite detector gradients')
    check(det_grad['gate_cls']['l1']>0 or det_grad['gate_box']['l1']>0,
          'Detector loss has no gate gradients at jointly closed initialization')
    check(det_grad['self_assessment']['l1']>0,'Detector loss has no self-assessment gradients')
    old.verify_gate_thresholds(model,'PATCH04 native backward check')
    return {'schema':'run15a_native_backward_patch04_v1',
            'status':'FIT_ONEFRAME_NATIVE_DETECTOR_GRADIENT_PASS_NOT_QUALIFIED',
            'frame':{'video':2,'image_id':int(frame['image_id']),
                     'eligible_queries':int(bundle['candidate_mask'].sum().item())},
            'forward_equal_to_frozen_R0':True,'hard_cls_open':0,'hard_box_open':0,
            'route_forward_exact_zero':True,
            'route_grad_l1':route_grad,'detector_loss_grad_l1':det_grad,
            'detector_loss':float(detection.detach().item()),
            'source_sha256':{k:sha(project/k) for k in files},
            'NO_optimizer_step_or_checkpoint_write':True,
            'FIT_only':True,'CAL_QUAL_Val_Test_unopened':True,
            'limitations':['One-frame gradient smoke is not evidence of successful training',
                           'Surrogate gradients are biased and do not guarantee safe interventions',
                           'No mAP evidence or generalization claim']}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project',default=r'D:\MARIVISION\H0_workbench\eval_tools')
    project=Path(ap.parse_args().project).resolve()
    report=evaluate(project)
    output=project/'bfcef_f_s0'/'run15a_final_candidate_sandbox'/'native_backward_patch04'
    output.mkdir(parents=True,exist_ok=True)
    target=output/'oneframe_gradient.json'
    check(not target.exists(),'Refuse overwrite: '+str(target))
    tmp=target.with_suffix('.tmp')
    check(not tmp.exists(),'Incomplete prior transaction '+str(tmp))
    tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(target)
    print('[PATCH04 PASS] Real FIT-frame detector loss reaches the joint gates, exact baseline preserved',flush=True)
    print('[NEW]',target,flush=True)

if __name__=='__main__':main()
