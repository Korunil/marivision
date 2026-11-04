"""Independently targeted feasibility algorithm tests; no authoritative caches/checkpoints loaded."""
import torch
import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run15a_feasibility_proof as m


def fit_report_path():
    here=Path(__file__).resolve().parent
    authoritative=here/'bfcef_f_s0'/'run15a_internal_rebuild_v2_fixed_gate'/'fit_mechanism_audit_stride1'/'fit_mechanism_audit_report.json'
    archived=here.parent/'source_evidence'/'fit_mechanism_audit_report.json'
    return authoritative if authoritative.exists() else archived


class FeasibilityTests(unittest.TestCase):
    def test_identity_swap_cannot_be_hidden_by_same_tp_count(self):
        x={
          'Base':dict(tp=1,fp=2,fn=1,lost=0,rescued=0),
          'Full':dict(tp=1,fp=0,fn=1,lost=1,rescued=1),
          'ClsOnly':dict(tp=1,fp=0,fn=1,lost=1,rescued=1),
          'BoxOnly':dict(tp=1,fp=2,fn=1,lost=0,rescued=0)}
        self.assertEqual(m.choose_safe(x),'Base')
        x['BoxOnly']=dict(tp=2,fp=1,fn=0,lost=0,rescued=1)
        self.assertEqual(m.choose_safe(x),'BoxOnly')

    def test_metric_threshold(self):
        self.assertAlmostEqual(m.metric(2,1,1)['f1'],2/3)
        self.assertEqual(m.metric(0,0,0)['f1'],0)

    def test_existing_fit_report_conservation(self):
        p=fit_report_path()
        if not p.exists():self.skipTest('Original Library materialization only available in analysis session')
        d=m.validate_report(p)
        self.assertEqual(d['FIT_gt_sampled'],41284)
        self.assertEqual(d['lost_gt_against_base']['Full'],859)
        self.assertEqual(d['rescued_gt_against_base']['Full'],723)

    def test_existing_fit_report_mutation_detected(self):
        p=fit_report_path()
        if not p.exists():self.skipTest('Original Library materialization only available in analysis session')
        d=m.load(p)
        d['per_video']['2']['lost_gt_against_base']['Full']+=1
        with tempfile.TemporaryDirectory() as t:
            q=Path(t)/'bad.json';q.write_text(json.dumps(d))
            with self.assertRaisesRegex(RuntimeError,'Archived FIT report SHA-256 mismatch'):
                m.validate_report(q)

    def test_observable_box_coverage_is_monotonic(self):
        import torch
        O=SimpleNamespace(b=torch.tensor([[.5,.5,.2,.2],[.1,.1,.1,.1],[.9,.9,.05,.05]],dtype=torch.float32))
        gt=torch.tensor([[.1,.1,.1,.1],[.9,.9,.05,.05]])
        r=m.coverage_for_missing_gt(O,[],gt,torch.tensor([True,False,False]))
        self.assertEqual(r['missing_gt'],2)
        self.assertEqual(r['selected_bci_iou30'],0)
        self.assertEqual(r['all_R0_query_iou30'],2)

    def test_grouped_predictability_and_receipt_harm_detection(self):
        from random import Random
        rr=Random(5)
        features={k:float(i)/10 for i,k in enumerate([
           'camera_Axis','camera_FLIR','camera_eCAM','n_candidate','base_top100_score_mean',
           'base_top100_score_max','base_top100_score_at_rank100','base_operating_scores',
           'base_candidate_score_mean','base_candidate_box_area_mean',
           'base_candidate_cached640_small_fraction','cls_route_sum','box_route_sum',
           'cls_hard_fraction','box_hard_fraction','cls_delta_abs_mean','box_delta_abs_mean',
           'score_delta_abs_mean','corrected_candidate_score_mean'])}
        rows=[]
        for vid in m.FIT_VIDEOS:
            for j in range(3):
                p=(vid+j)%4
                base=dict(tp=2,fp=3,fn=1,lost=0,rescued=0)
                actions={'Base':base}
                for aidx,a in enumerate(m.CHANGES):
                    harm=int((p+aidx)%4==0)
                    gain=int((p+aidx)%3==1 and not harm)
                    x=dict(tp=2+gain-harm,fp=3-gain,fn=1-gain+harm,
                           lost=harm,rescued=gain)
                    f=dict(features);f['cls_delta_abs_mean']=float(aidx+1)*.1+rr.random()/100
                    f['camera_Axis']=float(vid%3==0);f['camera_FLIR']=float(vid%3==1)
                    x['features']=f
                    actions[a]=x
                rows.append({'schema':m.PROOF_SCHEMA,'video':vid,'image_id':vid*1000+j,
                             'frame_cache_idx':j,'stride':10,'gt_count':3,
                             'actions':actions,'coverage':{'missing_gt':1,
                             'selected_bci_iou30':0,'all_R0_query_iou30':1}})
        o,picks=m.oracle_frame(rows)
        self.assertGreaterEqual(o['selected']['tp'],o['baseline']['tp'])
        self.assertLessEqual(o['selected']['fp'],o['baseline']['fp'])
        r,chosen=m.cross_video_probe(rows)
        self.assertEqual(r['status'],'COMPLETED_DIAGNOSTIC_ONLY')
        self.assertEqual(r['n_FIT_videos'],29)
        self.assertEqual(r['n_frames'],87)
        with tempfile.TemporaryDirectory() as t:
            d=Path(t)
            c={'schema':m.PROOF_SCHEMA,'phase':'export','FIT_videos':list(m.FIT_VIDEOS),
               'stride':10,'fit_report_sha256':'a','source_pins':{},
               'all_existing_artifacts_read_only':True,'CAL_QUAL_official_val_test_unopened':True}
            m.write_new(d/'export_contract.json',c)
            ch=m.sha256(d/'export_contract.json')
            for vid in m.FIT_VIDEOS:
                dest=d/f'FIT_video_{vid:02d}.jsonl'
                mine=[x for x in rows if x['video']==vid]
                dest.write_text(''.join(json.dumps(x)+'\n' for x in mine))
                m.write_new(dest.with_suffix('.receipt.json'),{
                     'schema':m.PROOF_SCHEMA,'video':vid,'stride':10,'rows':len(mine),
                     'contract_sha256':ch,'file_sha256':m.sha256(dest)})
            _,readrows=m.all_evidence(d)
            self.assertEqual(len(readrows),87)
            tamper=d/'FIT_video_02.jsonl'
            tamper.write_text(tamper.read_text()+'\n')
            with self.assertRaisesRegex(RuntimeError,'receipt SHA'):
                m.all_evidence(d)


if __name__=='__main__':unittest.main(verbosity=2)
