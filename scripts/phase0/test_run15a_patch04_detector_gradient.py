"""Negative controls for the native-decoder double-zero gradient defect."""
import unittest
import torch
from test_run15a_final_architecture import fixture
from run15a_final_risk import RiskPolicy
from run15a_final_mechanism import mechanism_forward


def l1grad(module):
    return sum(float(p.grad.detach().abs().sum()) for p in module.parameters() if p.grad is not None)


class NativeDetectorGradientTests(unittest.TestCase):
    def closed(self):
        model, O, X, b = fixture()
        for e in 'SLCT':
            model.gate_cls[e].logit.data.fill_(-4.)
            model.gate_box[e].logit.data.fill_(-4.)
        policy = RiskPolicy(risk_margin_cls=3., risk_margin_box=3.)
        a = mechanism_forward(model, O, X, b, hard=True, policy=policy)
        t = mechanism_forward(model, O, X, b, hard=False, policy=policy)
        self.assertEqual(int(t['hard_cls'].sum() + t['hard_box'].sum()), 0)
        self.assertTrue(torch.equal(t['z'], a['z']))
        self.assertTrue(torch.equal(t['b'], a['b']))
        self.assertTrue(torch.equal(t['z'], O.z))
        self.assertTrue(torch.equal(t['b'], O.b))
        return model, O, X, b, t

    def test_23_all_closed_detector_loss_reaches_both_gate_branches(self):
        model, O, X, b, out = self.closed()
        # Separate native classification/box terms. Both must reach their
        # corresponding gate despite exact hard-forward abstention.
        ((out['z'] - O.z.detach() + 1).square().mean() +
         (out['b'] - O.b.detach() + 1).square().mean()).backward()
        self.assertGreater(l1grad(model.gate_cls), 0.)
        self.assertGreater(l1grad(model.gate_box), 0.)
        self.assertGreater(l1grad(model.self_assessment), 0.)

    def test_24_all_closed_finite_nonzero_gt_like_detector_gradients(self):
        model, O, X, b, out = self.closed()
        # A standard classification-plus-localization supervised objective,
        # no artificial objective on the routing tensor.
        t_cls=torch.zeros_like(out['z']);t_cls[0,0]=1.
        t_box=torch.full_like(out['b'],.35)
        det=(torch.nn.functional.binary_cross_entropy_with_logits(out['z'],t_cls)
             +torch.nn.functional.smooth_l1_loss(out['b'],t_box))
        self.assertTrue(bool(torch.isfinite(det)))
        det.backward()
        self.assertGreater(l1grad(model.gate_cls)+l1grad(model.gate_box),0.)

    def test_25_unavailable_queries_do_not_leak_gate_gradients(self):
        model, O, X, b = fixture()
        b=dict(b);b['candidate_mask']=torch.zeros_like(b['candidate_mask'])
        p=RiskPolicy(risk_margin_cls=3.,risk_margin_box=3.)
        t=mechanism_forward(model,O,X,b,hard=False,policy=p)
        self.assertTrue(torch.equal(t['z'],O.z))
        self.assertTrue(torch.equal(t['b'],O.b))
        # Outgoing detection correction gradient must be exactly zero
        # through permanently unavailable experts/candidates.
        (t['z'].square().mean()+t['b'].square().mean()).backward()
        self.assertEqual(l1grad(model.gate_cls),0.)
        self.assertEqual(l1grad(model.gate_box),0.)
        self.assertEqual(l1grad(model.self_assessment),0.)

    def test_26_exclusion_invariance_hard_and_training_forward(self):
        model,O,X,b=fixture()
        for e in 'SLCT':
            model.gate_cls[e].logit.data.fill_(-4.)
            model.gate_box[e].logit.data.fill_(-4.)
        p=RiskPolicy(risk_margin_cls=3.,risk_margin_box=3.)
        t=mechanism_forward(model,O,X,b,hard=False,policy=p)
        a=mechanism_forward(model,O,X,b,hard=True,policy=p)
        for key in ['z','b','pi_cls','pi_box','hard_cls','hard_box']:
            self.assertTrue(torch.equal(t[key],a[key]),key)
        self.assertTrue(torch.equal(a['z'],O.z))
        self.assertTrue(torch.equal(a['b'],O.b))

if __name__=='__main__':unittest.main()
