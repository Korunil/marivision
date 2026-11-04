"""Source-independent synthetic regression checks; no native model or data loaded."""
import inspect
import unittest
from pathlib import Path
import numpy as np
import run15a_oct13_expert_information as x


def toy_groups(n=16):
    result=[]
    for v in range(n):
        y=np.asarray([(i+v)%2 for i in range(24)],dtype=np.int8)
        za=np.asarray([(.22+.025*i+.01*(v%3)) for i in range(24)],dtype=float)
        sb=np.asarray([(.2+.5*float(y[i])+.001*v) for i in range(24)])
        sc=np.asarray([(.1+.7*float(y[i])+.001*v) for i in range(24)])
        result.append({'video':v+2,'tp_y':y,'r50_score':za,
                       'scores':{'A_R50':za,'B_Simple':sb,'C_Expert':sc}})
    return result


class OCT13ScientificContracts(unittest.TestCase):
    def test_auc_perfect_and_inverse(self):
        self.assertAlmostEqual(x.roc_auc([0,1,0,1],[0.1,0.9,0.2,0.8]),1.)
        self.assertAlmostEqual(x.roc_auc([0,1,0,1],[0.9,0.1,0.8,0.2]),0.)

    def test_auc_tie_handling(self):
        self.assertAlmostEqual(x.roc_auc([0,1,0,1],[.2,.2,.2,.2]),0.5)
        self.assertAlmostEqual(x.roc_auc([0,1,0,1],[.1,.5,.5,.5]),0.75)

    def test_auc_no_class_returns_none(self):
        self.assertIsNone(x.roc_auc([0,0],[.1,.8]))
        self.assertIsNone(x.roc_auc([1,1],[.1,.8]))

    def test_auc_rejects_nonfinite(self):
        with self.assertRaises(AssertionError):x.roc_auc([0,1],[0.0,float('nan')])

    def test_paired_oof_metrics_correct_orientation(self):
        r=toy_groups();m=x.metrics(r)
        self.assertEqual((m['n'],m['TP'],m['FP']),(384,192,192))
        self.assertAlmostEqual(m['B_Simple'],1.)
        self.assertAlmostEqual(m['C_Expert'],1.)
        self.assertAlmostEqual(m['delta_C_minus_B'],0.)

    def test_bootstrap_video_pairing_and_seed(self):
        r=toy_groups();b1=x.paired_video_bootstrap(r,reps=100,seed=101)
        b2=x.paired_video_bootstrap(r,reps=100,seed=101)
        self.assertEqual(b1,b2)
        self.assertEqual(b1['replicates_valid'],100)
        self.assertEqual(b1['delta_C_minus_B_ci95'],[0.,0.])

    def test_cutoff_rules_precommitted(self):
        boot=lambda lo,hi:{'delta_C_minus_B_ci95':[lo,hi]}
        self.assertEqual(x.decide({'delta_C_minus_B':.05},boot(.021,.07)),'INCREMENTAL_EXPERT_INFORMATION_PRESENT_TRAIN_ONLY')
        self.assertEqual(x.decide({'delta_C_minus_B':.009},boot(-.02,.029)),'NO_PERSUASIVE_INCREMENTAL_EXPERT_INFORMATION')
        self.assertTrue(x.decide({'delta_C_minus_B':.012},boot(-.02,.06)).startswith('INCONCLUSIVE'))
        self.assertFalse(x.decide({'delta_C_minus_B':.08},boot(.02,.10)).startswith('INCREMENTAL'))

    def test_video_robustness_heldout_not_refitted(self):
        diag=x.secondary_video_metrics(toy_groups())
        self.assertEqual(len(diag['leave_one_video_out']),16)
        self.assertEqual(diag['valid_per_video_auc_count'],16)
        self.assertFalse(diag['any_loo_sign_reversal'])

    def test_exploratory_edge_never_qualifies(self):
        ret=x.edge_band(toy_groups())
        self.assertEqual(ret['status'],'EXPLORATORY_POST_HOC_NOT_FOR_SELECTION')
        self.assertEqual(ret['r50_score_band'],[.20,.27])

    def test_single_class_video_diagnostic_undefined(self):
        r=toy_groups();r[0]['tp_y']=np.zeros(24,dtype=np.int8)
        diag=x.secondary_video_metrics(r)
        self.assertTrue(diag['per_video'][0]['single_class_undefined'])
        self.assertEqual(diag['valid_per_video_auc_count'],15)

    def test_fold_coverage_and_video_embargo(self):
        all_vid=[v for fold in x.FOLDS for v in fold]
        self.assertEqual(len(set(all_vid)),17)
        self.assertEqual(set(all_vid),set(x.a.TRAIN))
        self.assertFalse(set(all_vid)&set(x.a.EVAL))
        self.assertFalse(set(all_vid)&set(x.a.TUNE))

    def test_source_has_one_shot_output_and_no_tune_eval_consumers(self):
        s=inspect.getsource(x.run)
        self.assertIn('a.write_new(dest,report)',s)
        self.assertIn('f.extract_oof(rt)',s)
        self.assertNotIn('native_frames(rt,a.EVAL)',s)
        self.assertNotIn('native_frames(rt,a.TUNE)',s)
        self.assertNotIn('run_eval(',s)

    def test_frozen_script_hashes(self):
        self.assertEqual(x.sha(x.a.__file__),x.SOURCE_A)
        self.assertEqual(x.sha(x.p.__file__),x.SOURCE_P)
        self.assertEqual(x.sha(x.f.__file__),x.SOURCE_F)

    def test_full_synthetic_fold_feature_parity(self):
        rng=np.random.default_rng(7);frames={}
        for v in x.a.TRAIN:
            n=30;bf=rng.normal(size=(n,14));ef=np.column_stack([bf,rng.normal(size=(n,4))]);
            lab=np.tile([0.,1.],15)
            fr={'video':v,'frame':0,'label':lab,'base_x':bf,'expert_x':ef,
                'pairs':[(i,1) for i in range(n)],'image_id':int(v*1000)}
            frames[v]=fr
        contexts={}
        for fold,held in enumerate(x.FOLDS,1):
            train=[frames[v] for v in x.a.TRAIN if v not in held]
            X,Y,G=x.p.to_training_matrix(train)
            expert=x.a.fit_linear(X,1.-Y,G,epochs=30)
            for v in held:
                contexts[(v,0)]=(frames[v],expert,None,fold)
        preds,fits=x.scores_by_video(contexts)
        self.assertEqual(len(preds),17)
        self.assertEqual(len(fits),4)
        self.assertEqual(sum(len(p['tp_y']) for p in preds),510)
        self.assertEqual(x.metrics(x.combine_per_video(preds))['n'],510)

    def test_native_cutoff_is_only_a_descriptive_grid(self):
        self.assertEqual(x.CUTOFFS[0],.20)
        self.assertTrue(all(x.CUTOFFS[i]<x.CUTOFFS[i+1] for i in range(len(x.CUTOFFS)-1)))
        self.assertNotIn('r50_cutoff_null',inspect.getsource(x.decide))

if __name__=='__main__':unittest.main()
