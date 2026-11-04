"""No-network, CPU-only contracts. Not a substitute for Windows native FIT run."""
from pathlib import Path
import unittest, types, sys, unittest.mock
import run15a_oct12_preservation_first as p
import run15a_oct12_fp_recovery_ablation as a
import numpy as np
import torch

class PreservationFirstContracts(unittest.TestCase):
    @staticmethod
    def model(dim,bias):
        return {'mean':[0.]*dim,'std':[1.]*dim,'w':[0.]*dim,'bias':bias}

    def frame(self):
        raw=torch.zeros((3,7));native=raw.repeat(4,1,1)
        native[0,0,2]=-2.5;native[1,1,3]=-1.25
        return {'z':raw,'expert_z':native,'pairs':[(0,2),(1,3)],
                'expert_x':np.zeros((2,18)),'base_x':np.zeros((2,14)),'video':2,
                'b':torch.zeros((3,4)),'gt_boxes':torch.zeros((0,4)),
                'gt_labels':torch.zeros(0),'orig_matches':[],
                'counts':{'tp':0,'fp':2,'fn':0}}

    def test_fold_disjoint_complete_and_eval_excluded(self):
        self.assertEqual(set(v for f in p.FOLDS for v in f),set(a.TRAIN))
        self.assertEqual(sum(len(f) for f in p.FOLDS),len(a.TRAIN))
        self.assertFalse(set(a.EVAL).intersection(set(a.TRAIN+a.TUNE)))

    def test_original_sources_and_receipt_pin(self):
        self.assertEqual(a.sha(Path(a.__file__)),p.SOURCE_SHA)
        self.assertEqual(p.RECEIPT_SHA,'4e827507e1f5997bcaf77358c1f3a67f5226ef22a52cdd1e1fb8e3cfad746329')

    def test_veto_hard_reject_and_unchanged_fallback(self):
        f=self.frame();orig=f['z'].clone()
        out,n=p.apply_guard(f,self.model(18,20),self.model(18,20),.5)
        self.assertEqual(n,0);self.assertTrue(torch.equal(out,orig))
        self.assertTrue(torch.equal(f['z'],orig))

    def test_veto_allows_native_response_only_for_selected_queries(self):
        f=self.frame()
        out,n=p.apply_guard(f,self.model(18,20),self.model(18,-20),.5)
        self.assertEqual(n,2)
        self.assertTrue(torch.equal(out[0],f['expert_z'][0,0]))
        self.assertTrue(torch.equal(out[1],f['expert_z'][1,1]))
        self.assertTrue(torch.equal(out[2],f['z'][2]))
        self.assertTrue(torch.equal(f['z'],torch.zeros((3,7))))

    def test_multiple_class_candidates_same_query_do_not_overwrite(self):
        f=self.frame();f['pairs']=[(0,2),(0,3)]
        f['expert_z'][2,0,3]=-3.
        out,n=p.apply_guard(f,self.model(18,20),self.model(18,-20),.5)
        self.assertEqual(n,1)
        self.assertTrue(torch.equal(out[1:],f['z'][1:]))

    def test_tp_weight_is_used_and_finite(self):
        rng=np.random.default_rng(7)
        X=rng.normal(size=(240,18));Y=np.tile([0.,1.],120)
        G=np.repeat(np.arange(6),40)
        fit=p.fit_tp_guard(X,Y,G,epochs=30)
        self.assertEqual(fit['tp_cost'],8.)
        self.assertEqual(fit['feature_dim'],18)
        self.assertTrue(np.isfinite(a.predict(fit,X)).all())
        self.assertNotIn('probability',fit['interpretation'].lower().split('not ')[0])

    def test_per_video_mass_and_class_cost(self):
        # More important than a regression of coefficient values.
        X=np.zeros((240,18));Y=np.tile([0.,1.],120);G=np.repeat(np.arange(6),40)
        fit=p.fit_tp_guard(X,Y,G,epochs=10)
        # With otherwise identical features, high TP cost shifts sigmoid toward TP.
        self.assertGreater(float(a.predict(fit,X[:1])[0]),.50)

    def test_tune_pass_requires_zero_lost_and_positive_fp(self):
        good={'lost':0,'actions':4,'delta_f1':.12,'fp_reduction':4,'rescued':0}
        self.assertTrue(p.tune_passes(good))
        self.assertFalse(p.tune_passes({**good,'lost':1}))
        self.assertFalse(p.tune_passes({**good,'fp_reduction':0}))

    def test_oof_gate_rejects_loss_concentration_and_zero_actions(self):
        ok={'lost':0,'delta_f1':.1,'fp_reduction':8,'actions':20,'active_videos':4,
            'max_video_action_fraction':.55,'per_video':{'2':{'lost':0}}}
        self.assertTrue(p.oof_eligible(ok))
        self.assertFalse(p.oof_eligible({**ok,'lost':1}))
        self.assertFalse(p.oof_eligible({**ok,'max_video_action_fraction':.95}))
        self.assertFalse(p.oof_eligible({**ok,'active_videos':1}))
        self.assertFalse(p.oof_eligible({**ok,'actions':0}))

    def test_fold_merge_pools_counts_and_f1_not_mean_f1(self):
        def one(video,tp,fp,fn,ct,cf,cn,acts):
            def f(t,f,n):return {'tp':t,'fp':f,'fn':n,'f1':2*t/(2*t+f+n)}
            return {'base':f(tp,fp,fn),'guarded':f(ct,cf,cn),'lost':0,'rescued':0,'actions':acts,
                    'per_video':{str(video):{'lost':0,'actions':acts}}}
        r=p.merge_fold_reports([one(2,3,9,1,3,3,1,4),one(3,15,1,2,15,0,2,2)])
        self.assertEqual(r['guarded']['tp'],18)
        self.assertEqual(r['guarded']['fp'],3)
        self.assertEqual(r['actions'],6)
        self.assertAlmostEqual(r['guarded']['f1'],36/(36+3+3))

    def test_nonselectable_no_veto_matches_original_native_expert(self):
        # The cutoff=1.0 control uses exactly the unguarded expert policy.
        f=self.frame()
        fp=self.model(18,20)
        guard=self.model(18,20)
        actual,n=p.apply_guard(f,fp,guard,1.0)
        expected,m=a.intervene(f,fp,'Expert',p.ORIGINAL_THRESHOLD,1.0)
        self.assertTrue(torch.equal(actual,expected))
        self.assertEqual(n,m)

    def test_preservation_requires_harm_in_reference(self):
        self.assertEqual(p.preservation_evidence({'lost':0},{'lost':0}),
                         'PRESERVATION_UNTESTED_REFERENCE_LOSS_FREE')
        self.assertEqual(p.preservation_evidence({'lost':3},{'lost':0}),
                         'OBSERVED_OOF_LOSS_REDUCTION_NOT_GENERALIZATION_PROOF')
        self.assertEqual(p.preservation_evidence({'lost':3},{'lost':2}),
                         'GUARDED_LOSSES_OBSERVED')

    def test_outcome_status_does_not_claim_unobserved_safety(self):
        good={'lost':0,'actions':10,'delta_f1':.1,'fp_reduction':9,'rescued':0}
        self.assertEqual(p.outcome_status({'lost':0},{'lost':0},good),
                         'DEV_UTILITY_ONLY_PRESERVATION_UNTESTED')
        self.assertEqual(p.outcome_status({'lost':2},{'lost':0},good),
                         'DEV_OOF_LOSS_REDUCTION_OBSERVED_TUNE_PASS_NOT_QUALIFIED')
        self.assertEqual(p.outcome_status({'lost':2},{'lost':0},{**good,'lost':1}),
                         'HOLD_TUNE_PRESERVATION_OR_UTILITY')

    def test_no_veto_is_not_a_selectable_cutoff(self):
        code=Path(p.__file__).read_text(encoding='utf8')
        self.assertNotIn('1.0)\n# Video-level folds',code)
        self.assertNotIn('1.0)',repr(p.CUTS))
        self.assertTrue(all(x < 1.0 for x in p.CUTS))
        self.assertIn('train_oof_no_veto_nonselectable',code)
        self.assertIn('tune_no_veto_reference',code)

    def test_only_allowed_phases_and_no_eval_data_consumer(self):
        code=Path(p.__file__).read_text(encoding='utf8')
        self.assertIn("choices=('selftest','verify','train-tune')",code)
        self.assertNotIn('native_frames(rt,a.EVAL)',code)
        self.assertNotIn('a.paired_coco(',code)
        self.assertNotIn('torch.save(',code)
        self.assertNotIn('optimizer.step(',code)
        self.assertNotIn('paired_eval.json',code)

if __name__=='__main__':unittest.main()
