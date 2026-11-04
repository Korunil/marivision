import unittest, pathlib, sys, types, unittest.mock
# Import the production ablation module FIRST. Its Windows/PyTorch import
# contract loads Pillow and configures thread environment before NumPy/Torch.
# Importing NumPy/Torch here first can initialize incompatible OpenMP runtimes
# before the module's guarded order is established.
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import run15a_oct12_fp_recovery_ablation as a
import numpy as np
import torch

class RecoveryAblationTests(unittest.TestCase):
    def test_simple_gain_reach_and_actual_action_strength_audit(self):
        # A 0.99-score hypothesis needs a 5.98-logit reduction to fall below 0.20.
        self.assertEqual(len(a.GRID_GAIN),3)
        self.assertGreaterEqual(max(a.GRID_GAIN),6.0)
        z=torch.zeros((2,7));z[0,2]=float(np.log(.99/.01))
        bf=np.zeros((1,14));ex=z.repeat(4,1,1)
        frame={'z':z,'b':torch.ones((2,4)), 'expert_z':ex,
               'pairs':[(0,2)],'base_x':bf,'expert_x':np.zeros((1,18))}
        model={'mean':[0.]*14,'std':[1.]*14,'w':[0.]*14,'bias':20.}
        old,n_old=a.intervene(frame,model,'Simple',.85,2.)
        audit=[]
        new,n_new=a.intervene(frame,model,'Simple',.85,max(a.GRID_GAIN),audit=audit)
        self.assertEqual((n_old,n_new),(1,1))
        self.assertGreater(float(torch.sigmoid(old[0,2])),.2)
        self.assertLess(float(torch.sigmoid(new[0,2])),.2)
        self.assertEqual(len(audit),1)
        summ=a.summarize_action_strength(audit)
        self.assertEqual(summ['action_units'],1)
        self.assertEqual(summ['selected_class_crossed_below_0p20'],1)
        self.assertEqual(summ['n_original_score_at_least_0p65'],1)
        self.assertAlmostEqual(summ['other_class_max_abs_logit_change']['max'],0.0)
        self.assertTrue(torch.equal(frame['z'],z))

    def test_expert_action_strength_audit_seven_class_footprint(self):
        z=torch.zeros((2,7));ex=z.repeat(4,1,1)
        ex[0,0,2]=-3.;ex[0,0,4]=2.
        f={'z':z,'b':torch.ones(2,4),'expert_z':ex,'pairs':[(0,2)],
           'base_x':np.zeros((1,14)),'expert_x':np.zeros((1,18))}
        mdl={'mean':[0.]*18,'std':[1.]*18,'w':[0.]*18,'bias':20.}
        audit=[]
        out,n=a.intervene(f,mdl,'Expert',.85,1.,audit=audit)
        self.assertEqual(n,1)
        summary=a.summarize_action_strength(audit)
        self.assertEqual(summary['action_units'],1)
        self.assertAlmostEqual(summary['selected_class_logit_drop']['max'],3.0)
        self.assertAlmostEqual(summary['other_class_max_abs_logit_change']['max'],2.0)
        self.assertTrue(torch.equal(out[1],z[1]))

    def test_all_fit_partitions_disjoint_and_complete(self):
        self.assertEqual(set(a.FIT),{2,3,4,5,6,7,8,13,14,16,17,18,19,20,24,25,29,31,35,36,37,38,40,41,42,43,44,45,47})
        self.assertFalse(set(a.TRAIN)&set(a.TUNE))
        self.assertFalse(set(a.TRAIN)&set(a.EVAL))
        self.assertFalse(set(a.TUNE)&set(a.EVAL))
    def test_base_features_no_gt_and_class_onehot(self):
        z=np.zeros((3,7));z[0,3]=3
        b=np.ones((3,4))*.25;ez=np.broadcast_to(z,(4,3,7)).copy()
        bf,ef=a.feat_arrays(z,b,ez,[(0,3)],{(0,3):7})
        self.assertEqual(bf.shape,(1,14));self.assertEqual(ef.shape,(1,18));self.assertEqual(bf[0,-7:].tolist(),[0,0,0,1,0,0,0])
    def test_no_cross_query_or_box_change_expert(self):
        z=torch.zeros(3,7);ez=z.repeat(4,1,1)
        ez[0,0,1]=-2.;ez[1,1,2]=4.
        bf=np.zeros((2,14));ef=np.zeros((2,18))
        frame={'z':z,'b':torch.ones((3,4)),'expert_z':ez,'pairs':[(0,1),(1,2)],'base_x':bf,'expert_x':ef}
        model={'mean':[0.]*18,'std':[1.]*18,'w':[0.]*18,'bias':10.}
        out,n=a.intervene(frame,model,'Expert',.9)
        self.assertEqual(n,1);self.assertLess(out[0,1],0);self.assertTrue(torch.equal(out[1],z[1]));self.assertTrue(torch.equal(frame['z'],z))
    def test_deterministic_logistic_and_group_usage(self):
        rng=np.random.default_rng(2026);X=rng.normal(size=(240,14));y=np.array([0.,1.]*120);g=np.repeat(np.arange(6),40)
        fit1=a.fit_linear(X,y,g,epochs=80);fit2=a.fit_linear(X,y,g,epochs=80)
        self.assertEqual(fit1['w'],fit2['w'])
        self.assertTrue(np.isfinite(a.predict(fit1,X)).all())
    def test_training_rejects_class_degenerate(self):
        X=np.zeros((150,14));y=np.zeros(150);g=np.repeat(np.arange(5),30)
        with self.assertRaises(AssertionError):a.fit_linear(X,y,g)
    def test_simple_modifies_only_selected_class(self):
        z=torch.zeros(2,7);bf=np.zeros((1,14))
        frame={'z':z,'b':torch.ones(2,4),'expert_z':z.repeat(4,1,1),'pairs':[(0,2)],'base_x':bf,'expert_x':np.zeros((1,18))}
        model={'mean':[0.]*14,'std':[1.]*14,'w':[0.]*14,'bias':10.}
        out,n=a.intervene(frame,model,'Simple',.9,1.)
        self.assertEqual(n,1);self.assertLess(out[0,2],z[0,2]);self.assertEqual(int(torch.count_nonzero(out-z)),1)
    def test_existing_ablation_never_calls_train_criterion(self):
        # Code contract: no optimizer or torch.save in this ablation.
        code=pathlib.Path(a.__file__).read_text(encoding='utf8')
        self.assertNotIn('optimizer.step(',code)
        self.assertNotIn('torch.save(',code)
    def test_risk_tune_no_safe_actions_is_hold(self):
        class DummyOld:
            def capture_match(self,z,b,gt_boxes,gt_labels):return [],[],{'tp':0,'fp':0,'fn':0}
        # No eligible pairs -> no correction -> cannot get a positive F1 gain.
        f={'z':torch.zeros((2,7)),'b':torch.ones(2,4),'pairs':[],
           'base_x':np.empty((0,14)),'expert_x':np.empty((0,18)),
           'expert_z':torch.zeros(4,2,7),'gt_boxes':torch.empty(0,4),'gt_labels':torch.empty(0),
           'orig_matches':[],'counts':{'tp':0,'fp':0,'fn':0},'video':6}
        m={'Simple':{'mean':[0.]*14,'std':[1.]*14,'w':[0.]*14,'bias':0},
           'Expert':{'mean':[0.]*18,'std':[1.]*18,'w':[0.]*18,'bias':0}}
        fake=types.ModuleType('run15a_final_preservation')
        fake.identity_ledger=lambda old,new:{'lost':set(),'rescued':set()}
        with unittest.mock.patch.dict(sys.modules,{'run15a_final_preservation':fake}):
            ch,rows=a.choose([f],m,DummyOld())
        self.assertIsNone(ch['Simple']);self.assertIsNone(ch['Expert'])

if __name__=='__main__':unittest.main()
