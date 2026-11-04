import unittest
from collections import Counter
import numpy as np
import torch
import run15a_oct12_harm_forensic as f


class MatchByRank:
    def capture_match(self,z,b,gt_boxes,gt_labels):
        scores=z.sigmoid().flatten()
        keep=scores.topk(min(100,scores.numel())).indices.tolist()
        keep=[ix for ix in keep if scores[ix]>=.2]
        found=(0 in keep)
        matches=[(0,0,0)] if found else []
        nfp=len(keep)-(1 if found else 0)
        return matches,[(int(ix//7),int(ix%7)) for ix in keep if ix!=0],{
            'tp':int(found),'fp':nfp,'fn':int(not found)}


def frame(expert_override,extra_q=2):
    z=torch.full((extra_q,7),-7.,dtype=torch.float32)
    z[0,0]=1.5
    b=torch.ones((extra_q,4),dtype=torch.float32)
    ez=z.unsqueeze(0).repeat(4,1,1)
    for q,k,c,value in expert_override:ez[k,q,c]=value
    return {'z':z,'b':b,'expert_z':ez,'pairs':[],'expert_x':np.zeros((0,18)),
            'orig_matches':[(0,0,0)],'gt_boxes':torch.empty((0,4)),
            'gt_labels':torch.empty((0,),dtype=torch.long)}


def action(q,c,k=0,descriptor=True):
    return {'q':q,'c':c,'k':k,'non_target_ge_0p50':descriptor,'fp_score':.9,'guard_score':.2}


class ForensicContracts(unittest.TestCase):
    def test_source_partition_and_folds(self):
        self.assertEqual(len(f.a.TRAIN),17)
        self.assertEqual(set(f.a.TRAIN),set(v for fold in f.p.FOLDS for v in fold))
        self.assertFalse(set(f.a.TRAIN)&set(f.a.TUNE))
        self.assertFalse(set(f.a.TRAIN)&set(f.a.EVAL))

    def test_fixed_action_native_bitwise_invariance(self):
        fr=frame([(0,0,0,-2.),(0,0,1,-4.),(1,0,1,-6.)],extra_q=3)
        fr['pairs']=[(0,0),(0,1),(1,1)];fr['expert_x']=np.zeros((3,18))
        always={'mean':[0.]*18,'std':[1.]*18,'w':[0.]*18,'bias':20.}
        never={'mean':[0.]*18,'std':[1.]*18,'w':[0.]*18,'bias':-20.}
        acts=f.fixed_actions(fr,always,never)
        z,used=f.p.apply_guard(fr,always,never,1.)
        self.assertEqual(used,len(acts));self.assertTrue(torch.equal(z,f.apply_fixed(fr,acts)))
        self.assertEqual(len(set(x['q'] for x in acts)),len(acts))

    def test_veto_does_not_reselect(self):
        fr=frame([(0,0,0,-2.),(0,0,1,-4.)])
        acts=[action(0,0),action(1,0)]
        self.assertTrue(torch.equal(f.apply_fixed(fr,acts,()),fr['z']))
        self.assertTrue(torch.equal(f.apply_fixed(fr,acts,(1,))[0],fr['z'][0]))

    def test_class_local_no_non_target_change(self):
        fr=frame([(0,0,1,-3.),(0,0,0,-4.)])
        acts=[action(0,1)]
        local=f.apply_fixed(fr,acts,(0,),(0,))
        self.assertEqual(float(local[0,1]),-3.)
        self.assertTrue(torch.equal(local[0,0:1],fr['z'][0,0:1]))

    def test_M1_trigger_class_direct(self):
        fr=frame([(0,0,0,-4.)])
        acts=[action(0,0)]
        assess,_=f.make_match_cache(MatchByRank(),fr,acts)
        ev=f.classify_loss(0,fr,acts,assess)
        self.assertEqual(ev['mechanism'],'M1_DIRECT_TRIGGER_CLASS')
        self.assertEqual(ev['singleton_sufficient_indices'],[0])

    def test_M2_full_row_collateral(self):
        fr=frame([(0,0,0,-4.),(0,0,1,-4.)])
        acts=[action(0,1)]
        assess,_=f.make_match_cache(MatchByRank(),fr,acts)
        ev=f.classify_loss(0,fr,acts,assess)
        self.assertEqual(ev['mechanism'],'M2_FULL_ROW_COLLATERAL')
        self.assertEqual(ev['full_context_row_to_local_prevents_indices'],[0])
        self.assertFalse(ev['class_local_minimal_subset_loses'])

    def test_M3_untouched_query_top100_displacement(self):
        fr=frame([(0,1,1,-8.),(0,1,2,5.)],extra_q=20)
        fr['z'][0,0]=1.0
        # q0,c0 is just inside Top100. 99 other scores strictly above it.
        positions=[(q,c) for q in range(20) for c in range(7) if (q,c)!=(0,0) and (q,c)!=(1,2) and (q,c)!=(1,1)]
        for q,c in positions[:99]:fr['z'][q,c]=2.0
        fr['expert_z']=fr['z'].unsqueeze(0).repeat(4,1,1)
        fr['expert_z'][0,1,1]=-8.;fr['expert_z'][0,1,2]=5.
        acts=[action(1,1)]
        assess,_=f.make_match_cache(MatchByRank(),fr,acts)
        self.assertNotIn(0,assess(())['lost'])
        self.assertIn(0,assess((0,))['lost'])
        ev=f.classify_loss(0,fr,acts,assess)
        self.assertEqual(ev['mechanism'],'M3_UNTOUCHED_QUERY_DISPLACEMENT')
        self.assertTrue(ev['after_match_rank']['definitely_beyond_top100'])

    def test_one_minimal_subset_fixed_indices(self):
        fr=frame([(0,0,0,-4.),(0,1,1,-8.)])
        acts=[action(0,0),action(1,1)]
        assess,_=f.make_match_cache(MatchByRank(),fr,acts)
        self.assertEqual(f.minimal_harmful_subset(assess,acts,0),[0])

    def test_harm_unit_not_action_count(self):
        ev=[{'mechanism':'M1_DIRECT_TRIGGER_CLASS','video':3,
             'direct_trigger_singleton_indices':[0],
             'related_action_descriptors':[dict(action(0,0),index=0,fp_score=.91)]}]*13
        controls=[dict(action(0,0),video=3,fp_score=.80)]*3
        d=f.decide(ev,[],controls,{})
        self.assertEqual(d['status'],'STOP_DIRECT_HARM_DOMINANT')
        self.assertAlmostEqual(d['median_harm_trigger_fp_score'],.91)

    def test_direct_harm_lower_fp_confidence_is_not_auto_stop(self):
        ev=[{'mechanism':'M1_DIRECT_TRIGGER_CLASS','video':3,
             'direct_trigger_singleton_indices':[0],
             'related_action_descriptors':[dict(action(0,0),index=0,fp_score=.77)]}]*13
        controls=[dict(action(0,0),video=3,fp_score=.90)]*5
        d=f.decide(ev,[],controls,{})
        self.assertEqual(d['status'],'HOLD_DIRECT_HARM_FP_ORDERING_NOT_ESTABLISHED')

    def test_insufficient_coverage_holds(self):
        ev=[{'mechanism':'M4_MIXED_OR_UNRESOLVED'}]*3+[{'mechanism':'M1_DIRECT_TRIGGER_CLASS'}]*10
        d=f.decide(ev,[],[],{})
        self.assertEqual(d['status'],'HOLD_ATTRIBUTION_COVERAGE')

    def test_gate_zero_parity_fail_closed(self):
        exp={'train_oof_no_veto_nonselectable':{'oof':{
            'actions':701,'lost':13,'rescued':0,'fp_reduction':378,
            'active_videos':1,'base':{'tp':1,'fp':2,'fn':3},
            'guarded':{'tp':0,'fp':1,'fn':4},
            'per_video':{'2':{'actions':701,'lost':13,'rescued':0,'fp_reduction':378,'tp_delta':-1}}}}}
        ob={'actions':701,'lost':13,'rescued':0,'fp_reduction':378,'active_videos':1,
            'base':{'tp':1,'fp':2,'fn':3},'candidate':{'tp':0,'fp':1,'fn':4},
            'per_video':{'2':{'actions':701,'lost':13,'rescued':0,'fp_reduction':378,'tp_delta':-1}}}
        self.assertEqual(f.parity(exp,ob)['status'],'EXACT_PARITY')
        for bad in [dict(ob,lost=12),dict(ob,actions=700),dict(ob,fp_reduction=377)]:
            with self.assertRaises(AssertionError):f.parity(exp,bad)

    def test_permutation_insufficient_group_controls(self):
        null=f.permutation_null([dict(action(0,0),video=1,frame=1,action_index=0)],[])
        self.assertEqual(null['status'],'INSUFFICIENT_VIDEO_MATCHED_CONTROLS')


if __name__=='__main__':unittest.main(verbosity=2)
