"""PATCH05 standalone synthetic tests. Never accesses FIT/CAL/QUAL/Val/Test."""
import unittest
import torch
from types import SimpleNamespace
from run15a_patch05_class_admission import labels,balanced_admission_loss,real_cls_risk, capture_gate_soft
from run15a_final_risk import RiskPolicy, risk_admission


class Patch05Tests(unittest.TestCase):
    def test_mixed_is_harm_never_positive(self):
        t=torch.tensor([[3,8],[3,7],[0,0],[1,8],[8,8],[6,8]])
        r=torch.ones_like(t,dtype=torch.bool)
        a=torch.ones(6,dtype=torch.bool)
        good,bad,neutral=labels(t,r,a)
        self.assertEqual(good.tolist(),[True,False,False,False,False,True])
        self.assertEqual(bad.tolist(),[False,True,False,True,False,False])
        self.assertEqual(neutral.tolist(),[False,False,True,False,True,False])

    def test_unavailable_never_labeled(self):
        t=torch.tensor([[3],[1],[8]])
        r=torch.ones_like(t,dtype=torch.bool)
        g,h,n=labels(t,r,torch.zeros(3,dtype=torch.bool))
        self.assertEqual(int(g.sum()+h.sum()+n.sum()),0)

    def test_risk_score_identical_to_production(self):
        z=torch.tensor([[1.0,-.4],[0.2,0.5]])
        O=SimpleNamespace(z=z)
        p=torch.zeros(2,2,9)
        p[...,1]=.07;p[...,2]=.03;p[...,7]=.05
        channels={'U':torch.tensor([[.8],[.2]]),'H':torch.tensor([[.1],[.4]]),
                  'sigma':torch.ones(2,1)*.1,'p_loss':torch.ones(2,1)*.05,
                  'p_add':torch.ones(2,1)*.01}
        out={'cls_raw':{'S':p},'cls_channels':{'S':channels}}
        post=real_cls_risk(out,O,'S',RiskPolicy())
        expected=dict(channels)
        expected['p_loss']=torch.maximum(channels['p_loss'],((p[...,1]+p[...,2])*(z.sigmoid()>=.2)).amax(-1,keepdim=True))
        expected['p_add']=torch.maximum(channels['p_add'],p[...,7].amax(-1,keepdim=True))
        self.assertTrue(torch.equal(post['score'],risk_admission(expected,'cls',RiskPolicy())['score']))

    def test_gradient_sign_good_vs_harm(self):
        r=torch.tensor([-.5,-.5,-.5],requires_grad=True)
        s=torch.tensor([.019,.021,.020],requires_grad=True)
        pos=torch.tensor([1,0,0],dtype=torch.bool)
        bad=torch.tensor([0,1,0],dtype=torch.bool)
        neutral=torch.tensor([0,0,1],dtype=torch.bool)
        loss,_=balanced_admission_loss(r,s,.02,pos,bad,neutral)
        loss.backward()
        self.assertLess(float(r.grad[0]),0)
        self.assertGreater(float(r.grad[1]),0)
        self.assertLess(float(s.grad[0]),0)
        self.assertGreater(float(s.grad[1]),0)

    def test_toy_direct_admission_learns_without_threshold_change(self):
        r=torch.nn.Parameter(torch.tensor([-.5,-.5,-.5])); logits=torch.nn.Parameter(torch.full((3,),-4.))
        opt=torch.optim.Adam([r,logits],lr=.05)
        pos=torch.tensor([1,0,0],dtype=torch.bool)
        bad=torch.tensor([0,1,0],dtype=torch.bool)
        neutral=torch.tensor([0,0,1],dtype=torch.bool)
        for _ in range(120):
            opt.zero_grad();loss,_=balanced_admission_loss(r,logits.sigmoid(),.02,pos,bad,neutral)
            loss.backward();opt.step()
        self.assertGreater(float(r[0]),0)
        self.assertLess(float(r[1]),0)
        self.assertGreater(float(logits[0].sigmoid()),.02)
        self.assertLess(float(logits[1].sigmoid()),.02)

    def test_no_positives_does_not_invent_positive_labels(self):
        s=torch.tensor([-.4,-.4],requires_grad=True)
        g=torch.tensor([.01,.01],requires_grad=True)
        pos=torch.zeros(2,dtype=torch.bool)
        bad=torch.tensor([True,False]);neutral=torch.tensor([False,True])
        loss,_=balanced_admission_loss(s,g,.02,pos,bad,neutral)
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(float(s.grad[0]),0)
        self.assertGreater(float(g.grad[0]),0)

if __name__=='__main__':unittest.main()
