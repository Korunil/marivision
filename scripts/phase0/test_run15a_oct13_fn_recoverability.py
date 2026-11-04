"""Deterministic non-native contracts for OCT13 first recoverability gate."""
import inspect, math, unittest
import numpy as np
import torch
import run15a_oct13_fn_recoverability as m


class FakeOld:
    def single_iou(self,x,y):
        b=x.detach().float().cpu().numpy().reshape(1,4)
        g=y.detach().float().cpu().numpy().reshape(1,4)
        return torch.tensor(float(m.box_ious(b,g)[0,0]))
    def capture_match(self,z,b,gtb,gtl):
        return m.capture_threshold(self,z,b,gtb,gtl,.20)


class FNContracts(unittest.TestCase):
    def test_native_geometry(self):
        b=np.array([[.5,.5,.2,.2],[.20,.20,.1,.1],[.75,.75,.10,.10]])
        v=m.box_ious(b,b)
        self.assertTrue(np.allclose(np.diag(v),1))
        self.assertAlmostEqual(v[0,1],0)
        self.assertTrue(np.allclose(v,v.T))

    def test_no_gt_and_empty_geometry(self):
        self.assertEqual(m.box_ious(np.zeros((2,4)),np.zeros((0,4))).shape,(2,0))
        self.assertNotIn("gt",inspect.getsource(m.corroborated_actions))

    def test_frame_metadata_not_image_id_guess(self):
        x={9:{'video_id':2,'frame_id':10},10:{'video_id':2,'frame_id':11}}
        self.assertTrue(m.consecutive(x,9,10,2))
        self.assertFalse(m.consecutive(x,9,10,3))
        self.assertFalse(m.consecutive({9:{'video_id':2},10:{'video_id':2}},9,10,2))
        x[10]['frame_id']=12
        self.assertFalse(m.consecutive(x,9,10,2))

    def test_native_fixed_threshold_parity(self):
        old=FakeOld();z=torch.full((3,7),-10.)
        z[0,1]=math.log(.22/.78);z[1,1]=math.log(.12/.88)
        boxes=torch.tensor([[.5,.5,.2,.2],[.20,.20,.1,.1],[.75,.75,.1,.1]])
        gt=torch.tensor([[.5,.5,.2,.2],[.20,.20,.1,.1]])
        label=torch.tensor([1,1])
        aa,_,base=m.capture_threshold(old,z,boxes,gt,label,.20)
        ab,_,low=m.capture_threshold(old,z,boxes,gt,label,.10)
        self.assertEqual(base,{'tp':1,'fp':0,'fn':1})
        self.assertEqual(low,{'tp':2,'fp':0,'fn':0})
        self.assertEqual(len(aa),1);self.assertEqual(len(ab),2)

    def test_fn_qualified_with_frozen_query(self):
        z=torch.full((3,7),-8.)
        z[0,1]=math.log(.15/.85)
        b=torch.tensor([[.5,.5,.2,.2],[.2,.2,.1,.1],[.8,.8,.1,.1]])
        fr={'z':z,'b':b,'gt_boxes':torch.tensor([[.5,.5,.2,.2]]),
            'gt_labels':torch.tensor([1]),'orig_matches':[]}
        anatomy=m.fn_anatomy(fr)
        self.assertEqual(anatomy[0]['type'],'SUBTHRESHOLD_TOP100_SAMECLASS')
        zo,boost=m.oracle_action(fr,anatomy)
        self.assertEqual(len(boost),1)
        self.assertGreater(float(zo[0,1]),float(z[0,1]))
        self.assertTrue(torch.equal(fr['z'],z))

    def test_class_confusion_anatomy(self):
        z=torch.full((101,7),-8.)
        z[:,1]=2.0  # >100 stronger wrong-class hypotheses crowd out correct class
        fr={'z':z,'b':torch.tensor([[.5,.5,.2,.2]]*101),
            'gt_boxes':torch.tensor([[.5,.5,.2,.2]]),
            'gt_labels':torch.tensor([2]),'orig_matches':[]}
        self.assertEqual(m.fn_anatomy(fr)[0]['type'],'ACCEPTED_WRONG_CLASS_OVERLAP')

    def test_no_overlapping_query_anatomy(self):
        z=torch.full((1,7),-8.)
        fr={'z':z,'b':torch.tensor([[.1,.1,.1,.1]]),
            'gt_boxes':torch.tensor([[.9,.9,.1,.1]]),
            'gt_labels':torch.tensor([2]),'orig_matches':[]}
        self.assertEqual(m.fn_anatomy(fr)[0]['type'],'NO_USABLE_OVERLAP_TOP100_ANYCLASS')

    def test_temporal_gt_free_no_motion(self):
        z=torch.full((2,7),-8.);z[0,2]=-1.5
        b=torch.tensor([[.5,.5,.2,.2],[.3,.3,.1,.1]])
        prev_z=z.clone();prev_z[0,2]=-1
        fr={'z':z,'b':b};prev={'z':prev_z,'b':b}
        actions=m.corroborated_actions(fr,prev)
        self.assertEqual([(i['q'],i['c']) for i in actions],[(0,2)])
        updated=m.promote(fr,actions)
        self.assertGreaterEqual(float(updated[0,2].sigmoid()),.209)
        self.assertTrue(torch.equal(z,fr['z']))

    def test_temporal_clutter_can_pass(self):
        z=torch.full((1,7),-8.);z[0,3]=-1.45
        p=z.clone();p[0,3]=-.1
        fr={'z':z,'b':torch.tensor([[.4,.4,.1,.1]])}
        self.assertEqual(len(m.corroborated_actions(fr,{'z':p,'b':fr['b']})),1)
        # Passing a temporal rule is not proof the candidate is a TP.
        self.assertEqual(m.classify_rule_passers([],[],m.corroborated_actions(fr,{'z':p,'b':fr['b']}))['new_gt_matched_by_passers'],0)

    def test_cutoff_decision_no_relaxation(self):
        s={'qualifying':{'count':42,'active_videos':5,'max_share':.3},
           'oracle_delta':{'map30':.03,'mapca30':.03},
           'temporal_rule':{'precision':.4,'adjacent_verified':100,'adjacent_coverage':1.0},
           'null_assessment':{'count_matched_random_ap':{'temporal_above_random_p95_both_ap':True}}}
        self.assertEqual(m.decide(s),'HOLD_NEEDS_ISOLATED_T_EXPERT_AND_OPERATIONAL_NULL_REVIEW')
        s['temporal_rule']['adjacent_coverage']=0.02
        self.assertEqual(m.decide(s),'HOLD_INSUFFICIENT_AUTHORITATIVE_ADJACENCY_COVERAGE')
        s['temporal_rule']['adjacent_coverage']=1.0
        s['qualifying']['count']=10
        self.assertEqual(m.decide(s),'STOP_INSUFFICIENT_FN_OPPORTUNITY')
        s['qualifying']['count']=42;s['oracle_delta']['map30']=.015
        self.assertEqual(m.decide(s),'STOP_INSUFFICIENT_ORACLE_AP_GAIN')
        s['oracle_delta']['map30']=.03;s['temporal_rule']['precision']=.1
        self.assertEqual(m.decide(s),'STOP_TEMPORAL_RULE_LOW_PRECISION')

    def test_dependency_preflight_requires_callable_native_iou(self):
        class Empty:pass
        with self.assertRaisesRegex(AssertionError,'single_iou'):
            m.require_native_iou(Empty())
        m.require_native_iou(FakeOld())

    def test_fn_weak_overlap_is_top100_score_supported(self):
        # A raw box overlaps, but its 7 logits are below the declared floor
        # and excluded from Top100 by 110 higher-scoring hypotheses.
        z=torch.full((111,7),-12.)
        z[:110,1]=0.
        b=torch.tensor([[.7,.7,.05,.05]]*110+[[.5,.5,.2,.2]])
        fr={'z':z,'b':b,'gt_boxes':torch.tensor([[.5,.5,.2,.2]]),
            'gt_labels':torch.tensor([2]),'orig_matches':[]}
        result=m.fn_anatomy(fr)[0]
        self.assertEqual(result['type'],'SAMECLASS_GEOMETRY_UNAVAILABLE_AT_SCORE_OR_TOP100')
        self.assertLess(result['max_box_iou_usable_top100'],.1)
        self.assertGreater(result['max_box_iou_raw300'],.99)

    def test_fn_weak_overlap_from_actual_top100(self):
        z=torch.full((2,7),-9.);z[0,2]=-2.7
        b=torch.tensor([[.5,.5,.14,.14],[.1,.1,.1,.1]])
        fr={'z':z,'b':b,'gt_boxes':torch.tensor([[.5,.5,.2,.2]]),
            'gt_labels':torch.tensor([2]),'orig_matches':[]}
        # IoU 0.49: this is qualified for class-correct oracle; instead
        # offset the geometry to move IoU below 0.30 but above 0.10.
        b[0]=torch.tensor([.565,.565,.14,.14])
        self.assertEqual(m.fn_anatomy(fr)[0]['type'],'WEAK_OVERLAP_USABLE_TOP100_ANYCLASS')

    def test_only_novel_matched_gt_counts_as_recovery(self):
        actions=[{'q':2,'c':1},{'q':3,'c':1}]
        baseline=[(1,1,10)]
        post=[(2,1,10),(3,1,11)]
        out=m.classify_rule_passers(baseline,post,actions)
        self.assertEqual(out['new_gt_matched_by_passers'],1)
        self.assertEqual(out['promoted_tp_swaps_of_existing_gt'],1)
        # A changed q/class for an existing GT is not an extra TP.
        self.assertEqual(out['post_matched_gt_count']-out['baseline_matched_gt_count'],1)

    def test_random_promotion_exact_count_seeded_no_gt(self):
        z=torch.full((10,7),-3.)
        rng1=np.random.default_rng(20261013);rng2=np.random.default_rng(20261013)
        x=m.random_matched_actions(z,5,rng1)
        y=m.random_matched_actions(z,5,rng2)
        self.assertEqual(x,y)
        self.assertEqual(len(set((a['q'],a['c']) for a in x)),5)
        self.assertEqual(m.random_matched_actions(z,0,rng1),[])
        with self.assertRaises(AssertionError):m.random_matched_actions(z,100,rng1)

    def test_no_partial_frame_coverage_can_pass(self):
        self.assertEqual(m.adjacency_coverage(255,5),5/255)
        self.assertEqual(m.adjacency_coverage(0,0),0)
        s={'qualifying':{'count':100,'active_videos':5,'max_share':.25},
           'oracle_delta':{'map30':.2,'mapca30':.2},
           'temporal_rule':{'precision':.9,'adjacent_coverage':5/255},
           'null_assessment':{'count_matched_random_ap':{'temporal_above_random_p95_both_ap':True}}}
        self.assertEqual(m.decide(s),'HOLD_INSUFFICIENT_AUTHORITATIVE_ADJACENCY_COVERAGE')

    def test_random_null_is_decision_relevant_not_cutoff_ap(self):
        s={'qualifying':{'count':100,'active_videos':5,'max_share':.25},
           'oracle_delta':{'map30':.1,'mapca30':.1},
           'temporal_rule':{'precision':.5,'adjacent_coverage':1.0},
           'null_assessment':{'lower_cutoff_operational_advantage':True,
                              'count_matched_random_ap':{'temporal_above_random_p95_both_ap':False}}}
        self.assertEqual(m.decide(s),'STOP_NO_TEMPORAL_AP_ADVANTAGE_OVER_COUNT_MATCHED_RANDOM')

    def test_single_use_source_partition_constraints(self):
        src=inspect.getsource(m)
        self.assertIn('FORENSIC_SHA',src)
        self.assertIn('not dest.exists()',src)
        self.assertIn('set(a.TRAIN).isdisjoint(a.TUNE)',src)
        self.assertIn('set(a.TRAIN).isdisjoint(a.EVAL)',src)
        self.assertNotIn('TRAIN+TUNE',src)
        self.assertNotIn('a.native_frames',src)
        self.assertNotIn('fit_linear(',src)
        self.assertNotIn('optimizer',src.lower())

    def test_source_has_no_evaluation_leak(self):
        src=inspect.getsource(m.run)
        self.assertIn('for vid in a.TRAIN:',src)
        self.assertNotIn('for vid in a.EVAL',src)
        self.assertNotIn('for vid in a.TUNE',src)
        self.assertNotIn('load_video_triplet(42)',src)


if __name__=='__main__':unittest.main(verbosity=2)
