"""RUN15A PATCH03: One real FIT frame, NO optimizer, NO checkpoint write.

Diagnoses whether *both-closed* R50 risk and TSK gates transmit training
surrogate gradient while retaining exact inference fallback.
This is a source-pinned engineering smoke test, NOT model qualification.
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
EXPECTED_MECHANISM_SHA = 'a12fc4b26a41e05a084cc81eed41e14a866ddcdf50154ebfafe159eef82da9d1'
EXPECTED_PRESERVATION_SHA = '325e27481a7cf40b8b12b77bfdfb8187914124f0c595d11eb81a48bfa52b3ae3'
EXPECTED_WINDOWS_SHA = 'cbf847e2ef594a51b6bf6e70883fb105da0ac17aa9b61d883546f0969891667f'
EXPECTED_PILOT_SHA = 'edd5086f147dd23d5c6b0f2bfe012bd1b39559676ce35046c287095df0a74182'


def check(cond, msg):
    if not cond:
        raise AssertionError('[PATCH03 GRADIENT HOLD] ' + msg)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def norm(module):
    return sum(float(p.grad.detach().float().abs().sum().item()) for p in module.parameters() if p.grad is not None)


def evaluate(project: Path):
    checks = {'run15a_final_risk.py': EXPECTED_RISK_SHA,
              'run15a_final_mechanism.py': EXPECTED_MECHANISM_SHA,
              'run15a_final_preservation.py': EXPECTED_PRESERVATION_SHA,
              'run15a_final_windows.py': EXPECTED_WINDOWS_SHA,
              'run15a_final_pilot_fit.py': EXPECTED_PILOT_SHA}
    for file, expected in checks.items():
        check(sha(project/file) == expected, 'Candidate source mismatch: ' + file)
    original, r9, r8, d, s0a, raw, _, _, _ = load_runtime(project, downstream='none')
    # Source-pinned original R0 and full-Train specialist loading; no V2 weights.
    r12 = original.imports_and_verify()[1]
    model, criterion, CO, FX = original.make_model(r9, r12)
    r9.set_training_modes(model, joint=True)
    original.lock_gate_thresholds(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    check(bool(trainable), 'No downstream parameters trainable')
    model.train()
    for expert in (model.expert_s, model.expert_l, model.expert_c, model.expert_t, model.adapter, model.context_bus):
        expert.eval()
    can, spa, tmp = s0a.load_video_triplet(2)
    frame=can[0]
    O, X = s0a.make_O_X(frame, spa[0], tmp[0], CO, FX)
    bundle = r9.expert_bundle(model, O, X)
    out = mechanism_forward(model, O, X, bundle, hard=False, policy=RiskPolicy())
    check(torch.equal(out['z'], O.z) and torch.equal(out['b'],O.b),
          'Fresh initial state is not exact baseline fallback: inspect before proceeding')
    check(int(out['hard_cls'].sum()+out['hard_box'].sum()) == 0,
          'This probe targets both-closed initialization; fresh gates unexpectedly opened')
    eligible = int(bundle['candidate_mask'].sum().item())
    check(eligible > 0, 'No eligible candidates on this FIT frame')
    # Gradient to ROUTING gate/risk from output, even when both hard masks are closed.
    for p in trainable:
        p.grad=None
    route_proxy = out['pi_cls'][:,1:].sum() + out['pi_box'][:,1:].sum()
    check(float(route_proxy.detach().item()) == 0.0, 'Closed expert route is nonzero')
    route_proxy.backward(retain_graph=True)
    signal={k: norm(v) for k,v in [
       ('gate_cls', model.gate_cls), ('gate_box',model.gate_box),
       ('self_assessment',model.self_assessment),
       ('coord_cls',model.coord_cls),('coord_box',model.coord_box)]}
    check(signal['gate_cls']>0 and signal['gate_box']>0 and signal['self_assessment']>0,
          'Jointly closed gates still suppress necessary learning gradients: '+str(signal))
    # Detector-level loss has a separate backward path through the exact-zero-forward surrogate.
    for p in trainable:
        p.grad=None
    detection = r9.detection_loss(criterion, out['z'], out['b'], frame)
    check(bool(torch.isfinite(detection)), 'Nonfinite detector loss')
    detection.backward()
    det_grad={k: norm(v) for k,v in [
       ('gate_cls',model.gate_cls),('gate_box',model.gate_box),
       ('self_assessment',model.self_assessment)]}
    check(det_grad['gate_cls']>0 or det_grad['gate_box']>0,
          'Detector loss has no gate gradients in all-closed state')
    original.verify_gate_thresholds(model,'PATCH03 one-FIT-frame gradient check')
    return {'schema':'run15a_joint_ste_patch03_gradient_v1',
       'status':'FIT_ONEFRAME_GRADIENT_SMOKE_PASS_NOT_QUALIFIED',
       'frame':{'video':2,'cached_image_id':int(frame['image_id']), 'eligible_queries':eligible},
       'hard_classification_open':0,'hard_box_open':0,
       'route_forward_exact_zero':True,'baseline_forward_bit_exact':True,
       'joint_ste_route_gradient_l1':signal,
       'detector_loss_gradient_l1':det_grad,
       'detector_loss':float(detection.detach().item()),
       'source_sha256':{k:sha(project/k) for k in checks},
       'NO_checkpoint_save_or_optimizer_step':True,
       'only_FIT_accessed':True,'NO_optimizer_step':True,'NO_CAL_QUAL_VALIDATION_TEST':True,
       'limitations':['Nonzero gradient does not imply learnable intervention policy',
                      'Coordinator gradients can be zero while hard gates are all closed',
                      'Fresh weights differ from prior four-frame pilot checkpoint']}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project',default=r'D:\MARIVISION\H0_workbench\eval_tools')
    args=ap.parse_args()
    project=Path(args.project).resolve()
    report=evaluate(project)
    out=project/'bfcef_f_s0'/'run15a_final_candidate_sandbox'/'joint_ste_patch03'
    out.mkdir(parents=True,exist_ok=True)
    dest=out/'oneframe_gradient.json'
    check(not dest.exists(), 'Refuse overwrite; preserve previous experiment: '+str(dest))
    temp=dest.with_suffix('.tmp')
    check(not temp.exists(), 'Incomplete result transaction exists')
    temp.write_text(json.dumps(report, indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(dest)
    print('[PATCH03 PASS] One FIT frame: exact fallback and gradient flow for jointly closed gates')
    print('[GRADIENTS]',json.dumps({'route':report['joint_ste_route_gradient_l1'],
                                   'detection':report['detector_loss_gradient_l1']},indent=2))
    print('[NEW]',dest)

if __name__=='__main__':main()
