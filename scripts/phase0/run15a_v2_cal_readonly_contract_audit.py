"""RUN_15A V2 read-only contract audit.

Checks existing immutable V2 calibration and certificate, and constructs a
counterexample against the *actual* PATCH03B preservation_objective() source.
No optimizer, weights, Train/Validation/Test cache, or checkpoint is opened.
This is forensic evidence; NOT another qualification or a proposed retune.
"""
from __future__ import annotations
import argparse, hashlib, importlib, json, math, sys
from pathlib import Path
from types import SimpleNamespace
import torch

EXPECTED_DRIVER = '6fa33f1128cbc22266c580e7be85a4b69fd1bac2651ed358b64140236cd733e2'
EXPECTED_MECHANISM = '42404d765097c65b23ec318ed51b8c4fc7b2acdd9eb34a331a76c09313bcb637'
EXPECTED_PARENT = '1211292b180eaf00862fc8098ad5cda6a2f64701c0954b5f8f752f090d2ce84c'
EXPECTED_FIT = 'b265aa6a12987de61afaa0010071717b853b01ea0844d1502cc8321029cb3f8d'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, msg):
    if not condition:
        raise AssertionError('READ-ONLY AUDIT FAILED: ' + msg)


def test_real_preservation_function(module, baseline_score):
    z = torch.full((300,7), -12.,dtype=torch.float32)
    z[17,1] = torch.tensor(baseline_score).logit()
    b = torch.tensor([[0.50,0.50,0.20,0.20]],dtype=torch.float32).repeat(300,1)
    O = SimpleNamespace(z=z,b=b)
    gt = {'gt_boxes_norm':b[:1].clone(), 'gt_labels':torch.tensor([1],dtype=torch.long)}
    new_z = z.clone()
    new_z[17,1] = z[17,1]-0.04  # exactly the free margin in PATCH03B source
    out = {'z':new_z, 'b':b.clone(),
           'pi_cls':torch.zeros((300,5),dtype=torch.float32),
           'pi_box':torch.zeros((300,5),dtype=torch.float32)}
    dummy_r9 = SimpleNamespace(detection_loss=lambda criterion,z,b,can: z.sum()*0)
    old = module.joint_marginal_uh_loss
    try:
        module.joint_marginal_uh_loss = lambda *args:out['z'].sum()*0
        _, detail, bm = module.preservation_objective(None,O,out,gt,None,dummy_r9)
    finally:
        module.joint_marginal_uh_loss = old
    matched,fps,current = module.capture_match(new_z,b,gt['gt_boxes_norm'],gt['gt_labels'])
    actual_score=float(new_z[17,1].sigmoid())
    require(bm['tp']==1 and bm['fn']==0, 'Baseline fixture must detect its GT')
    require(float(detail['tp_class'])<1e-12, 'The original source preservation hinge has changed')
    require(current['tp']==0 and current['fn']==1 and actual_score<.2,
            'Fixture did not expose zero-penalty loss of a genuine baseline match')
    return {'baseline_score':baseline_score,'new_score':actual_score,
            'tp_class_penalty':detail['tp_class'],'baseline_tp':bm['tp'],
            'new_tp':current['tp'],'lost_baseline_gt':1}


def run(root, calibration, certificate, output):
    root=Path(root);calibration=Path(calibration);certificate=Path(certificate)
    driver=root/'run15a_internal_rebuild_PATCH03.py'
    mechanism=root/'run15a_internal_mechanism.py'
    require(sha(driver)==EXPECTED_DRIVER,'Driver source SHA mismatch')
    require(sha(mechanism)==EXPECTED_MECHANISM,'Mechanism source SHA mismatch')
    cert=json.loads(certificate.read_text(encoding='utf-8'))
    cal=json.loads(calibration.read_text(encoding='utf-8'))
    require(cert['checkpoint_sha256']==EXPECTED_FIT,'FIT certificate checkpoint SHA mismatch')
    require(cert['parent_sha256']==EXPECTED_PARENT,'Upstream source contract changed')
    require(cert['source_sha256']==EXPECTED_DRIVER,'FIT was not produced by PATCH03B driver')
    require(cal['screen']['status']=='HOLD','CAL status is not HOLD')
    require(cal['gate_threshold']==.02,'Calibration threshold changed')
    require(cal['upstream_seen_all_47_train'] is True,'Upstream exposure flag changed')
    b=cal['metrics']['Base'];f=cal['metrics']['Internal']
    require(f['tp']-b['tp']==-450 and f['fp']-b['fp']==-4411,'Original calibration results changed')
    require(cal['newly_rescued_gt']-cal['baseline_matched_gt_lost']==f['tp']-b['tp'],
            'GT identity accounting inconsistency')
    require(sum(x['Base']['tp'] for x in cal['per_video'].values())==b['tp'],
            'Video TP aggregation failure')
    require(sum(x['Internal']['fp'] for x in cal['per_video'].values())==f['fp'],
            'Video FP aggregation failure')
    require(cal['gate_active_queries']['cls']==cal['gate_active_queries']['box'],
            'Gate activity is no longer identical; update hypothesis')
    sys.path.insert(0,str(root))
    mod=importlib.import_module('run15a_internal_rebuild_PATCH03')
    require(mod.FIXED_GATE_THRESHOLD==.02 and mod.LOSS_WEIGHTS['tp_class']==8.,
            'Frozen fixed threshold or loss coefficient changed')
    tests=[test_real_preservation_function(mod,s) for s in (.201,.205)]
    by_video={vid:{'delta_tp':v['Internal']['tp']-v['Base']['tp'],
                   'delta_fp':v['Internal']['fp']-v['Base']['fp']}
              for vid,v in cal['per_video'].items()}
    by_class={name:{'support':v['support'],
                    'delta_tp':round(v['support']*(v['internal']-v['base']))}
              for name,v in cal['per_class_recall'].items()}
    result={
       'scope':'READ_ONLY_STATIC_AND_SYNTHETIC_ANNOTATION_FREE_AUDIT',
       'independent_generalization_claim':False,
       'calibration_sha256':sha(calibration),
       'fit_certificate_sha256':sha(certificate),
       'source_pins':{'driver':EXPECTED_DRIVER,'mechanism':EXPECTED_MECHANISM},
       'cal_status':'HOLD',
       'delta':{'tp':f['tp']-b['tp'],'fp':f['fp']-b['fp'],
                'recall_pp':100*(f['recall']-b['recall']),
                'f1_pp':100*(f['f1']-b['f1'])},
       'matched_identity':{'lost':cal['baseline_matched_gt_lost'],'rescued':cal['newly_rescued_gt']},
       'gate_activity':{'cls':cal['gate_active_queries']['cls'],
                        'box':cal['gate_active_queries']['box'],
                        'per_frame_each':cal['gate_active_queries']['cls']/cal['frames'],
                        'total_candidate_denominator_not_in_report':True},
       'by_video':by_video,'by_class':by_class,
       'proven_contract_counterexamples':tests,
       'source_facts':{
           'free_logit_margin':.04,
           'baseline_score_operating_threshold':.2,
           'tp_class_penalty_not_hard_constraint':True,
           'baseline_unmatched_predictions_used_as_fp_targets':True,
           'per_example_hard_tp_preservation_not_enforced_in_forward':True,
           'upstream_exposed_to_all_train_videos':True
       },
       'scientific_action':'Preserve CAL HOLD and QUAL embargo; no retuning or retraining from this audit.'
    }
    Path(output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('[PASS] Source-pinned read-only audit: 2 real-objective counterexamples, metric conservation, status & lineage')
    print('Synthetic 0.201 ->',tests[0]['new_score'],'with L_TPclass=',tests[0]['tp_class_penalty'])
    print('CAL',result['delta'],'gate-active queries per frame',result['gate_activity']['per_frame_each'])
    print('No Train cache, checkpoint weight, Validation or Test opened. No fitting/calibration performed.')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source-dir',default='.')
    p.add_argument('--calibration',default='bfcef_f_s0/run15a_internal_rebuild_v2_fixed_gate/internal_calibration.json')
    p.add_argument('--certificate',default='bfcef_f_s0/run15a_internal_rebuild_v2_fixed_gate/internal_fit_certificate.json')
    p.add_argument('--output',default='bfcef_f_s0/run15a_internal_rebuild_v2_fixed_gate/RUN15A_V2_CAL_READONLY_AUDIT.json')
    a=p.parse_args()
    run(a.source_dir,a.calibration,a.certificate,a.output)
