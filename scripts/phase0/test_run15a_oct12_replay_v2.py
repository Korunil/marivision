"""Independent no-R50/CPU tests for read-only OCT12 replay V2.

Run together with source (production imports must remain available):
python -W ignore -m unittest -v test_run15a_oct12_replay_v2.py
"""
import unittest
import torch
import io
import contextlib
from unittest.mock import patch
import run15a_oct12_single_query_replay_v2 as replay_module
from run15a_oct12_single_query_replay_v2 import (
    mark_class_action, verify_tsk_output, snapshot_coordinator_inputs,
    coord_forced_alpha, permutation_donor, norm_match, single_native,
    positive_ranking_proxy, summarize_events, score_to_threshold)


class ReplayV2ContractTests(unittest.TestCase):
    def test_causal_single_query_zero(self):
        class Adapter:
            def decode_cls(self,x):return x[:,:2]
        class Model:adapter=Adapter()
        class O:
            h=torch.tensor([[0.,0.],[.4,.3],[.1,.2]])
            z=torch.tensor([[0.,0.],[.4,.3],[.1,.2]])
            b=torch.zeros(3,4)
        m=Model();o=O();delta=torch.tensor([.5,-.3])
        z0,b0=single_native(m,o,delta,1,0)
        z1,b1=single_native(m,o,delta,1,1)
        self.assertTrue(torch.equal(z0,o.z))
        self.assertTrue(torch.equal(b0,o.b))
        self.assertTrue(torch.equal(z1[[0,2]],o.z[[0,2]]))
        self.assertTrue(torch.equal(z1[1],o.z[1]+delta))
        self.assertTrue(torch.equal(b1,o.b))

    def test_gate_hard_closed_is_not_soft(self):
        v=verify_tsk_output({'g_soft':torch.tensor([[.018],[.023]]),
             'mask_hard':torch.tensor([[0.],[1.]]), 'g_hard':torch.tensor([[0.],[.023]])})
        self.assertGreater(float(v['g_soft'][0]),0)
        self.assertEqual(float(v['g_hard'][0]),0)
        bad={'g_soft':torch.tensor([[.018]]),'mask_hard':torch.tensor([[0.]]),'g_hard':torch.tensor([[.018]])}
        with self.assertRaises(AssertionError):verify_tsk_output(bad)

    def test_hook_snapshot_survives_reentry(self):
        class Coord(torch.nn.Module):
            def forward(self,x,d):return torch.zeros(x.shape[0],4)
        class Router(torch.nn.Module):
            def forward(self,g,A,c):
                a=g*A*torch.exp(c)
                return torch.cat((1/(1+a.sum(-1,keepdim=True)),a/(1+a.sum(-1,keepdim=True))),-1)
        class Model:
            gate_dim=1
            coord_cls=Coord()
            router=Router()
        model=Model()
        live={}
        hook=model.coord_cls.register_forward_hook(lambda mod,inp,out:live.update({'cls':(inp[0],inp[1],out)}))
        try:
            ci=torch.zeros(4,16);d=torch.zeros(4,4,2)
            model.coord_cls(ci,d)
            snap=snapshot_coordinator_inputs(live)
            for q in (1,3,2):
                alpha=coord_forced_alpha(model,snap,'cls',q,1,.02,torch.ones(4,4))
                self.assertAlmostEqual(alpha,.02/1.02,places=6)
            self.assertEqual(len(live['cls'][0]),1)  # real hook overwritten
            self.assertEqual(len(snap['cls'][0]),4)  # source preserved
        finally:hook.remove()

    def test_controls_norm_and_not_self(self):
        donors=[permutation_donor([3,8,9,12],8,2,1,0,draw) for draw in range(3)]
        self.assertEqual(len(set(donors)),3)
        self.assertNotIn(8,donors)
        self.assertAlmostEqual(float(norm_match(torch.tensor([6.,8.]),torch.tensor([2.,0.])).norm()),2.,places=6)

    def test_sham_denominator_is_unique_action_item(self):
        def item(k,s):return {'label':'positive','mode':'permute_full','expert':'C','item_id':k,
             'survived':s,'any_safe_benefit':s,'harm_target_reproduced':False,
             'rounding_ambiguous':False,'lost':0,'new_fp':0,'alpha':1.,'positive_types':[5]}
        summary=summarize_events([item('a',True),item('a',False),item('a',False),item('b',False)])
        got=summary['positive/permute_full/C']
        self.assertEqual(got['unique_items'],2)
        self.assertEqual(got['draw_records'],4)
        self.assertAlmostEqual(got['clean_mean_item_survival'],1/6,places=6)
        self.assertEqual(got['positive_type_strata']['5']['unique_items'],2)

    def test_rounding_ambiguity_excludes_all_paired_controls(self):
        def item(mode,ambig):return {'label':'positive','mode':mode,'expert':'S','item_id':'v:1:S:9',
            'survived':True,'any_safe_benefit':True,'harm_target_reproduced':False,
            'rounding_ambiguous':ambig,'lost':0,'new_fp':0,'alpha':1.,'positive_types':[3]}
        out=summarize_events([item('full',True),item('permute_full',False)])
        self.assertEqual(out['positive/full/S']['clean_unique_items'],0)
        self.assertEqual(out['positive/permute_full/S']['clean_unique_items'],0)

    def test_multitype_counts_are_labeled_nonindependent(self):
        r={'label':'positive','mode':'full','expert':'S','item_id':'y',
           'survived':True,'any_safe_benefit':True,'harm_target_reproduced':False,
           'rounding_ambiguous':False,'lost':0,'new_fp':0,'alpha':1.,'positive_types':[3,5]}
        x=summarize_events([r])['positive/full/S']
        self.assertEqual(x['unique_items'],1)
        self.assertEqual(x['multiple_positive_type_items'],1)
        self.assertEqual(x['positive_type_strata']['3']['unique_items'],1)
        self.assertEqual(x['positive_type_strata']['5']['unique_items'],1)

    def test_rank_proxy_is_within_class(self):
        baseline=torch.tensor([[1.,9.],[2.,8.],[3.,7.]])
        updated=baseline.clone();updated[0,0]=4.
        tr=torch.full((3,2),8,dtype=torch.long);tr[0,0]=6
        data=positive_ranking_proxy(baseline,updated,0,tr,torch.ones((3,2),dtype=torch.bool))
        self.assertEqual(data[0]['signed_rank_gain'],2.)

    def test_cli_print_uses_actual_summary_key(self):
        """Catch failure occurring only AFTER the read-only report has been written."""
        report={
            'class_label_census':{'S_safe_useful':12},
            'summary':{'positive/full/S':{
                'clean_mean_item_survival':0.75,
                'unique_items':4,'draw_records':4,
                'rounding_ambiguous_items_excluded':0,'alpha_median':1.0,
            }},
        }
        buf=io.StringIO()
        with patch.object(replay_module,'replay',return_value=report), \
             patch.object(replay_module,'write_new',return_value=None), \
             patch('sys.argv',['run15a_oct12_single_query_replay_v2.py']):
            with contextlib.redirect_stdout(buf):
                replay_module.main()
        self.assertIn('mean survival 0.750',buf.getvalue())

    def test_threshold_distance(self):
        at20=torch.logit(torch.tensor(.2))
        z=torch.tensor([[0.,at20]])
        self.assertLess(score_to_threshold(z,0,[1]),1e-7)

if __name__=='__main__':unittest.main()
