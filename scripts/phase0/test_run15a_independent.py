"""Independent, data-free adversarial tests for RUN15A.
Does NOT import author implementation's test function.
No SeaClips GT/checkpoints accessed.
"""
from __future__ import annotations
import ast
import importlib.util
import inspect
import sys
from pathlib import Path
import numpy as np
import torch

SOURCE=Path(__file__).parent/'run_bfcef_f_run15a_integrated.py'
assert SOURCE.is_file(),SOURCE
spec=importlib.util.spec_from_file_location('run15a_independent',SOURCE)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
checks=0

def check(cond,msg):
    global checks
    assert cond,msg
    checks+=1

# Requirement: isolated FP reduction cannot override true-positive loss.
actions={
    'Base':{'tp':2,'fp':10,'fn':0},
    'ClsOnly':{'tp':1,'fp':0,'fn':1},
    'BoxOnly':{'tp':2,'fp':9,'fn':0},
    'Full':{'tp':1,'fp':1,'fn':1}}
check(mod.safe_frame_label(actions,{
    'Base':{0,1},'ClsOnly':{1},'BoxOnly':{0,1},'Full':{1}})==2,
    'TP-loss-tolerant action incorrectly promoted')
for i in range(4):
    case={name:{'tp':2,'fp':10} for name in mod.ACTIONS}
    case[mod.ACTIONS[i]]={'tp':2,'fp':10}
    check(mod.safe_frame_label(case,{a:{0,1} for a in mod.ACTIONS})==0,
          'Unnecessary intervention')
check(mod.choose_from_probability([[0.1,.78,.06,.06],[.52,.48,0,0],[.09,.39,.4,.12]],.70).tolist()==[1,0,0],
      'Abstention margin/calibration contract')
check(mod.counts_metric(2,3,1)['f1']==4/8,'F1 formula error')
check(set(mod.FIT_FOLDS).isdisjoint(mod.CAL_FOLDS) and
      set(mod.CAL_FOLDS).isdisjoint(mod.QUAL_FOLDS) and
      set(mod.FIT_FOLDS).isdisjoint(mod.QUAL_FOLDS) and
      set(mod.FIT_FOLDS+mod.CAL_FOLDS+mod.QUAL_FOLDS)==set(range(5)),
      'Fold-level disjointness broken')
check(mod.FEATURES and 'gt' not in ' '.join(mod.FEATURES).lower(),
      'GT leaked into predictor feature schema')

# Matching totals do not authorize an action that exchanges detected GT identities.
identity_swap={
    'Base':{'tp':2,'fp':10},'ClsOnly':{'tp':2,'fp':0},
    'BoxOnly':{'tp':2,'fp':10},'Full':{'tp':2,'fp':10}}
check(mod.safe_frame_label(identity_swap,{
    'Base':{0,1},'ClsOnly':{1,2},'BoxOnly':{0,1},'Full':{0,1}})==0,
    'Count-preserving identity swap wrongfully promoted')
# G06 must depend on actual RUN12 executable source and checkpoint provenance,
# never on r12.__doc__ (which is None because a future import precedes the string).
r12_text=Path(__file__).parent/'run_bfcef_f_run12_final_train.py'
if r12_text.is_file():
    import importlib.util
    spec2=importlib.util.spec_from_file_location('r12_static',r12_text)
    r12m=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(r12m)
    check(r12m.__doc__ is None and
          'r9.load_source_fold' in inspect.getsource(r12m.phase_calibrate),
          'Underlying RUN12 module-docstring pitfall not reproduced')
    check('r12.__doc__' not in inspect.getsource(mod.audit_legacy_conformance),
          'G06 still depends on misleading module documentation string')

# Construct two frame outcomes to independently test class/size/camera recall aggregation.
def row(video,cls,base,improved):
    return {'video':video,'camera':'e-CAM', 'gt_scale':['small'], 'class_gt':[cls],
            'matched_class':{'Base':{cls:base['tp']},'ClsOnly':{cls:improved['tp']},
                             'BoxOnly':{cls:base['tp']},'Full':{cls:improved['tp']}},
            'matched_scale':{'Base':{'small':base['tp']},'ClsOnly':{'small':improved['tp']},
                             'BoxOnly':{'small':base['tp']},'Full':{'small':improved['tp']}},
            'counts':{'Base':base,'ClsOnly':improved,'BoxOnly':base,'Full':improved}}
a={'tp':1,'fp':2,'fn':0};better={'tp':1,'fp':1,'fn':0}
b={'tp':1,'fp':5,'fn':0};worse={'tp':0,'fp':0,'fn':1}
r=mod.evaluate_choices([row(1,4,a,better),row(2,2,b,worse)],np.asarray([1,0]))
check(r['metrics']['Base']['tp']==2 and r['metrics']['Selected']['tp']==2,
      'Operational class/choice TP mapping broken')
check(r['metrics']['Selected']['fp']==6,'Selected FP accounting wrong')
check(r['per_class_recall']['LEISURE_VEHICLE']['support']==1,
      'Rare-class GT support dropped')
check(r['per_scale_recall']['small']['support']==2,
      'Scale GT support dropped')
check(r['camera_operational']['e-CAM']['Base']['tp']==2,
      'Camera aggregation wrong')
# Independent negative test: mAP regression must prevent qualification even if op metrics improve.
r['coco']={'Base':{'map30':0.5},'Selected':{'map30':0.4}}
check('COCO mAP@0.30 regressed' in mod.prospective_gate(r)['failures'],
      'COCO gate ignored')
# Qualifier must not sneak into fitting/threshold selection via callable signatures.
check('qual' not in inspect.signature(mod.calibrate).parameters,
      'Qualifier leaked into calibration signature')
check('cal' not in inspect.signature(mod.train_guard).parameters or
      'qual' in inspect.signature(mod.train_guard).parameters,
      'Review model training signature (should only validate split manifests)')

# The selected action is one coherent output pair. R0 must be bitwise exact when closed.
class Rank:
    def extract_top100_predictions(self,z,boxes):
        scores,index=z.sigmoid().flatten().topk(100)
        qi=index//7;ci=index%7;good=scores>=.2
        return {'all_scores':scores,'all_q_idx':qi,'all_c_idx':ci,'all_boxes':boxes[qi],
                'op_scores':scores[good],'op_q_idx':qi[good],'op_c_idx':ci[good],
                'op_boxes':boxes[qi][good]}
rank=Rank()
class O:
    z=torch.randn(300,7);b=torch.rand(300,4)
o=O();guard=mod.FrameActionGuard()
with torch.no_grad():
    for p in guard.parameters():p.zero_()
    guard.net[-1].bias[0]=30
z,b,info=mod.deploy_guard(guard,np.zeros(len(mod.FEATURES),dtype=np.float32),
                          np.ones(len(mod.FEATURES),dtype=np.float32),.5,
                          o,o.z*0.5,o.b+.1,rank)
check(info['action']=='Base' and torch.equal(z,o.z) and torch.equal(b,o.b),
      'Baseline preservation fails at production selector output')
# No artifacts below test/val nomenclature may be used as input path in new code.
source=SOURCE.read_text(encoding='utf8')
tree=ast.parse(source)
assigns={x.id:node.value.value for node in ast.walk(tree) if isinstance(node,ast.Assign)
         for x in node.targets if isinstance(x,ast.Name) and isinstance(node.value,ast.Constant)
         and isinstance(node.value.value,str)}
check(assigns.get('R12_SHA')=='1211292b180eaf00862fc8098ad5cda6a2f64701c0954b5f8f752f090d2ce84c',
      'Certified frozen model identity changed')
print(f'[PASS] Independent RUN15A adversarial unit suite: {checks} assertions; no SeaClips files or weights opened')
