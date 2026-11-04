"""Independent adversarial tests: executable B-FCEF-F final-candidate core."""
import unittest
import torch
from types import SimpleNamespace
from run15a_final_risk import RiskPolicy, risk_admission, intersect_gate, native_exact_output
from run15a_final_preservation import protected_gt_loss,identity_ledger
from run15a_final_supervision import pair_states,joint_transition,expert_deep_supervision
from run15a_final_mechanism import mechanism_forward

class Assessment(torch.nn.Module):
    def __init__(self,num_classes=2,u=.9,h=.1):
        super().__init__();self.cls_u=torch.nn.Parameter(torch.tensor(float(u)));self.box_u=torch.nn.Parameter(torch.tensor(.9))
        self.cls_h=torch.nn.Parameter(torch.tensor(float(h)));self.num_classes=num_classes
    def forward(self,feat,h,N,R,ncls,nbox,C):
        q=h.shape[0]; device=h.device
        one=torch.ones(q,1,device=device)
        cc={'U':self.cls_u.sigmoid().expand(q,1),'H':self.cls_h.sigmoid().expand(q,1),
            'sigma':one*.04,'p_loss':one*.02, 'p_rescue':one*.2,'p_prune':one*.2,
            'p_add':one*.02,'p_preserve':one*.5}
        bc={'U':self.box_u.sigmoid().expand(q,1),'H':one*.1,'sigma':one*.04,
            'p_loss':one*.02,'p_rescue':one*.2,'p_retain':one*.7,'exp_delta_iou':one*.1}
        return (torch.zeros(q,self.num_classes,9),torch.zeros(q,5),cc,bc)

class Gate(torch.nn.Module):
    def __init__(self):super().__init__();self.logit=torch.nn.Parameter(torch.tensor(1.5))
    def forward(self,x):
        v=self.logit.sigmoid().expand(x.shape[0],1)
        hard=(v>.1).float(); soft=torch.sigmoid((v-.1)*10)
        ste=hard+soft-soft.detach()
        return {'mask_hard':hard,'mask_ste':ste,'g_hard':hard*v,'g_ste':ste*v,'g_soft':v}

class Router(torch.nn.Module):
    def forward(self,g,A,c):
        p=A*g*torch.exp(c.clamp(-5,5))
        den=1+p.sum(-1,keepdim=True)
        return torch.cat([1/den,p/den],-1)

class Adapter(torch.nn.Module):
    def decode_cls(self,h):return h[:,:2]
    def decode_box(self,h,r):
        return torch.cat([torch.sigmoid(h[:,2:4]),torch.full_like(h[:,2:4],.20)],-1)

class MockModel(torch.nn.Module):
    gate_dim=19
    def __init__(self):
        super().__init__()
        self.self_assessment=torch.nn.ModuleDict({e:Assessment(u=1.5,h=-1.5) for e in 'SLCT'})
        self.gate_cls=torch.nn.ModuleDict({e:Gate() for e in 'SLCT'})
        self.gate_box=torch.nn.ModuleDict({e:Gate() for e in 'SLCT'})
        self.coord_cls=self.coord_box=lambda x,d:torch.zeros(x.shape[0],4,device=x.device)
        self.router=Router();self.adapter=Adapter()
    def _availability(self,O,mask,task):return mask.expand(-1,4)

def fixture():
    torch.manual_seed(20)
    q=4; h=torch.randn(q,4); z=h[:,:2].clone();b=torch.cat([torch.sigmoid(h[:,2:4]),torch.full((q,2),.2)],-1)
    O=SimpleNamespace(h=h,z=z,b=b,r=torch.zeros(q,2))
    raw={e:{'N':torch.ones(q,1)*.4,'R':torch.ones(q,1)*.8} for e in 'SLCT'}
    X=SimpleNamespace(raw_proxies=raw)
    cand=SimpleNamespace(query_rank_norm=torch.linspace(0,1,q).reshape(-1,1))
    deltas=torch.randn(q,4,4)*.8
    bundle={'C_t':torch.ones(1,1)*.2,'delta_cls':deltas,'delta_box':deltas*.5,
            'features':{e:torch.randn(q,8) for e in 'SLCT'},
            'L_expected_iou':torch.ones(q,1)*.5,'candidate_set':cand,
            'candidate_mask':torch.tensor([[1.],[1.],[0.],[1.]])}
    return MockModel(),O,X,bundle

class FinalArchitectureTests(unittest.TestCase):
    def test_01_risk_rejects_harm(self):
        x={'U':torch.tensor([[.2]]),'H':torch.tensor([[.8]]),
           'sigma':torch.tensor([[.1]]),'p_loss':torch.tensor([[.3]])}
        self.assertEqual(risk_admission(x,'cls',RiskPolicy())['mask_hard'].item(),0)
    def test_02_closed_expert_has_exact_zero_routing(self):
        g=Gate()(torch.ones(2,3))
        r={'mask_hard':torch.zeros(2,1),'mask_ste':torch.zeros(2,1),'soft':torch.full((2,1),.2)}
        x=intersect_gate(g,r,torch.ones(2,1))
        self.assertTrue(torch.count_nonzero(Router()(x['g_hard'],torch.ones(2,1),torch.ones(2,1))[:,1:])==0)
    def test_03_forced_abstention_bit_exact(self):
        model,O,X,b=fixture()
        out=mechanism_forward(model,O,X,b,hard=True,policy=RiskPolicy(risk_margin_cls=3,risk_margin_box=3))
        self.assertTrue(torch.equal(O.z,out['z']))
        self.assertTrue(torch.equal(O.b,out['b']))
        self.assertTrue(torch.equal(out['pi_cls'][:,0],torch.ones(len(O.h))))
    def test_04_task_decoupled_opening(self):
        model,O,X,b=fixture()
        out=mechanism_forward(model,O,X,b,hard=True,policy=RiskPolicy(risk_margin_cls=3,risk_margin_box=-3))
        self.assertTrue(torch.equal(O.z,out['z']))
        self.assertGreater(int((out['hard_box'].sum(-1)>0).sum()),0)
    def test_05_hard_forward_train_deploy_bit_exact(self):
        model,O,X,b=fixture()
        p=RiskPolicy(risk_margin_cls=-3,risk_margin_box=-3)
        a=mechanism_forward(model,O,X,b,hard=True,policy=p)
        t=mechanism_forward(model,O,X,b,hard=False,policy=p)
        self.assertTrue(torch.equal(a['z'],t['z']))
        self.assertTrue(torch.equal(a['b'],t['b']))
        self.assertTrue(torch.equal(a['pi_cls'],t['pi_cls']))
        self.assertTrue(torch.equal(a['hard_cls'],t['hard_cls']))
        self.assertTrue(torch.equal(a['z'][2],O.z[2]))
        self.assertTrue(torch.equal(a['b'][2],O.b[2]))
    def test_06_gradient_risk_and_TSK_reaches_modules(self):
        model,O,X,b=fixture()
        out=mechanism_forward(model,O,X,b,hard=False,policy=RiskPolicy(risk_margin_cls=-3,risk_margin_box=-3))
        loss=out['z'].square().mean()+out['b'].square().mean()
        loss.backward()
        grad=model.gate_cls['S'].logit.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad))
        self.assertNotEqual(float(grad.abs()),0.)
    def test_07_score_threshold_counterexample(self):
        z=torch.tensor([[torch.logit(torch.tensor(.201)).item(),-10.]])
        zz=z.clone();zz[0,0]=torch.logit(torch.tensor(.194653))
        b=torch.tensor([[.5,.5,.2,.2]])
        x=protected_gt_loss(z,b,zz,b,b,[(0,0,0)])
        self.assertGreater(float(x['cutoff']),0.)
        self.assertGreater(float(x['score_erosion']),0.)
    def test_08_identity_swap_is_not_preservation(self):
        a=identity_ledger([(0,0,4)],[(0,0,5)])
        self.assertEqual(a['net_tp'],0)
        self.assertEqual(a['lost'],[4]);self.assertEqual(a['rescued'],[5])
    def test_09_closed_ste_has_gradient(self):
        model,O,X,b=fixture()
        p=RiskPolicy(risk_margin_cls=3,risk_margin_box=3)
        out=mechanism_forward(model,O,X,b,hard=False,policy=p)
        self.assertTrue(torch.equal(out['z'],O.z));self.assertTrue(torch.equal(out['b'],O.b))
        (out['z'].sum()+out['b'].sum()).backward()
        self.assertIsNotNone(model.gate_cls['S'].logit.grad)
        self.assertTrue(torch.isfinite(model.gate_cls['S'].logit.grad))
    def test_11_exact_baseline_preservation_penalty_zero(self):
        z=torch.tensor([[torch.logit(torch.tensor(.201)).item(),-10.]],requires_grad=True)
        b=torch.tensor([[.5,.5,.2,.2]],requires_grad=True)
        d=protected_gt_loss(z,b,z,b,b,[(0,0,0)])
        self.assertEqual(float(d['total'].detach()),0.0)
    def test_12_transition_all_nine_states(self):
        base=torch.tensor([[0,0,0],[1,1,1],[2,2,2]])
        new=torch.tensor([[0,1,2]]*3)
        self.assertEqual(joint_transition(base,new).tolist(),[[0,1,2],[3,4,5],[6,7,8]])
    def test_13_consequence_supervision_is_differentiable(self):
        out={'cls_raw':{},'box_raw':{},'cls_channels':{},'box_channels':{}}; targets={}
        for e in 'SLCT':
            post=torch.nn.functional.softmax(torch.randn(2,2,9,requires_grad=True),-1)
            box=torch.randn(2,5,requires_grad=True)
            u=torch.nn.Parameter(torch.tensor([[.7],[.7]]));h=torch.nn.Parameter(torch.tensor([[.3],[.3]]))
            out['cls_raw'][e]=post;out['box_raw'][e]=box
            out['cls_channels'][e]={'U':u,'H':h}
            out['box_channels'][e]={'U':u,'H':h}
            targets[e]={'cls':{'transition':torch.tensor([[2,0],[1,8]]),
                               'relevant':torch.tensor([[1,1],[1,0]],dtype=torch.bool)},
                        'box':{'relevant':torch.tensor([[1],[0]],dtype=torch.bool),
                               'rescue':torch.tensor([[1.],[0.]]),
                               'loss':torch.tensor([[0.],[0.]]),
                               'retain':torch.tensor([[0.],[0.]]),
                               'delta_iou':torch.tensor([[.02],[0.]]),
                               'iou_mask':torch.tensor([[1],[0]],dtype=torch.bool)}}
        loss,_=expert_deep_supervision(out,targets)
        loss.backward()
        self.assertGreater(float(out['cls_channels']['S']['U'].grad.abs().sum()),0.)
        self.assertTrue(bool(torch.isfinite(loss)))
    def test_14_single_at_risk_class_not_diluted(self):
        class UnsafeAssessment(Assessment):
            def forward(self,*args):
                post,br,cc,bc=super().forward(*args)
                post=post.clone()
                post[...,8]=1.
                # One class predicts TP->NONOP with probability 1;
                # averaging across classes would mask this hazard.
                post[:,0,:]=0;post[:,0,2]=1.
                return post,br,cc,bc
        model,O,X,b=fixture()
        O.z=O.z.clone();O.z[:,0]=3.0  # all class-0 hypotheses operationally active
        model.self_assessment['S']=UnsafeAssessment(u=1.5,h=-1.5)
        out=mechanism_forward(model,O,X,b,hard=True,
            policy=RiskPolicy(consequence_loss_weight=2.0,risk_margin_box=-3.))
        self.assertEqual(float(out['hard_cls'][0,0]),0.0)
        self.assertGreater(float(out['hard_box'][0].sum()),0.0)
    def test_15_risk_ste_exact_bitwise_at_float32_edge_cases(self):
        # Rounding defect in the original form (hard+soft)-soft.
        # This is a genuine float32 counterexample, not a tolerance check.
        soft=torch.linspace(0.0,1.0,10001,dtype=torch.float32)
        ones=torch.ones_like(soft)
        legacy=(ones+soft)-soft.detach()
        self.assertTrue(bool((legacy!=ones).any()))
        safe=ones+(soft-soft.detach())
        self.assertTrue(torch.equal(safe,ones))
        inputs={'U':torch.linspace(.01,.99,10001).reshape(-1,1),
                'H':torch.full((10001,1),.1),
                'sigma':torch.full((10001,1),.04),
                'p_loss':torch.full((10001,1),.02)}
        result=risk_admission(inputs,'box',RiskPolicy())
        self.assertTrue(torch.equal(result['mask_ste'],result['mask_hard']))


    def test_17_cpu_gt_auto_alignment_to_native_box_dtype(self):
        # Independent adversarial fixture: original function sends GT through
        # with mismatched dtype, whereas corrected IoU must receive CUDA/native
        # compatible tensors. Works on CPU-only CI without needing a GPU.
        import run15a_final_preservation as mod
        z=torch.tensor([[torch.logit(torch.tensor(.3)).item(),-10.]],dtype=torch.float32)
        b=torch.tensor([[.5,.5,.2,.2]],dtype=torch.float32)
        gt=b.double()  # GT source may not have decoder dtype.
        original=mod.box_iou_cxcywh
        calls=[]
        def assert_aligned(x,y):
            self.assertEqual(x.device,y.device)
            self.assertEqual(x.dtype,y.dtype)
            calls.append((x.device,x.dtype))
            return original(x,y)
        try:
            mod.box_iou_cxcywh=assert_aligned
            loss=mod.protected_gt_loss(z,b,z,b,gt,[(0,0,0)])
            self.assertEqual(float(loss['total']),0.0)
            self.assertEqual(len(calls),2)
            self.assertEqual(gt.dtype,torch.float64)  # caller not mutated
        finally:
            mod.box_iou_cxcywh=original

    def test_18_real_cross_device_gt_cpu_outputs_cuda(self):
        # This exact configuration caused the user's Windows traceback.
        # Local CPU-only runners skip, but Windows CUDA must execute it.
        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable; Windows GPU release must run this')
        dev=torch.device('cuda')
        z=torch.tensor([[torch.logit(torch.tensor(.201)).item(),-10.]],device=dev,requires_grad=True)
        b=torch.tensor([[.5,.5,.2,.2]],device=dev,requires_grad=True)
        gt=b.detach().cpu()
        dropped=z.detach().clone();dropped[0,0]=torch.logit(torch.tensor(.194653,device=dev))
        loss=protected_gt_loss(z,b,dropped,b,gt,[(0,0,0)])
        self.assertTrue(bool(torch.isfinite(loss['total']).all()))
        self.assertGreater(float(loss['cutoff']),0.)
        unchanged=protected_gt_loss(z,b,z,b,gt,[(0,0,0)])
        self.assertEqual(float(unchanged['total'].detach()),0.)
        unchanged['total'].backward()
        self.assertIsNotNone(z.grad)

    def test_19_mixed_output_devices_rejected_early(self):
        if not torch.cuda.is_available():
            self.skipTest('Requires CUDA')
        z=torch.zeros(1,2,device='cuda')
        b=torch.tensor([[.5,.5,.2,.2]])
        with self.assertRaisesRegex(ValueError,'share one device'):
            protected_gt_loss(z,b,z,b,b,[(0,0,0)])

    def test_16_native_batch_sensitive_decoder_forward_parity(self):
        # Simulates a decoder whose rounding depends on the number of rows,
        # as can happen for real GPU matmul kernels. Both hard and training
        # paths must perform the same active-query native calculation.
        class BatchSensitiveAdapter(Adapter):
            def decode_cls(self,h):
                return h[:,:2]+h.new_tensor(h.shape[0]*.0001)
            def decode_box(self,h,r):
                v=torch.cat([torch.sigmoid(h[:,2:4]),torch.full_like(h[:,2:4],.20)],-1)
                return v+h.new_tensor(h.shape[0]*.0001)
        model,O,X,b=fixture()
        model.adapter=BatchSensitiveAdapter()
        p=RiskPolicy(risk_margin_cls=-3.,risk_margin_box=-3.)
        hard=mechanism_forward(model,O,X,b,hard=True,policy=p)
        ste=mechanism_forward(model,O,X,b,hard=False,policy=p)
        for key in ('z','b','pi_cls','pi_box','hard_cls','hard_box'):
            self.assertTrue(torch.equal(hard[key],ste[key]),key)
        ((ste['z'].square().mean())+(ste['b'].square().mean())).backward()
        self.assertIsNotNone(model.gate_cls['S'].logit.grad)
        self.assertGreater(float(model.gate_cls['S'].logit.grad.abs().sum()),0.)

    def test_10_unavailable_expert_cannot_act(self):
        model,O,X,b=fixture()
        a=mechanism_forward(model,O,X,b,hard=True,policy=RiskPolicy(risk_margin_cls=-3,risk_margin_box=-3))
        self.assertTrue(torch.equal(a['z'][2],O.z[2]));self.assertTrue(torch.equal(a['b'][2],O.b[2]))
        self.assertEqual(float(a['pi_cls'][2,1:].sum().detach()),0)


    def test_20_both_permissions_closed_receive_nonzero_gradients(self):
        # Replicates the real initialization g_soft=sigmoid(-4)<.02,
        # together with a harmful action rejected by risk admission.
        glogit=torch.nn.Parameter(torch.tensor(-4.0))
        rlogit=torch.nn.Parameter(torch.tensor(-2.0))
        amp=glogit.sigmoid().reshape(1,1)
        gh=(amp>.02).to(amp.dtype)
        surrogate=torch.sigmoid(10*(amp-.02))
        gm=gh+(surrogate-surrogate.detach())
        gate={'mask_hard':gh,'mask_ste':gm,'g_soft':amp,
              'g_hard':gh*amp,'g_ste':gm*amp}
        ch={'U':rlogit.sigmoid().reshape(1,1),
            'H':torch.tensor([[.8]]),'sigma':torch.tensor([[.1]]),
            'p_loss':torch.tensor([[.1]])}
        risk=risk_admission(ch,'box',RiskPolicy())
        self.assertEqual(float(gh.item()),0.)
        self.assertEqual(float(risk['mask_hard'].item()),0.)
        out=intersect_gate(gate,risk,torch.ones(1,1))
        self.assertEqual(float(out['hard'].item()),0.)
        self.assertTrue(torch.equal(out['ste'],out['hard']))
        self.assertTrue(torch.equal(out['g_ste'],out['g_hard']))
        (out['ste']+out['g_ste']).sum().backward()
        self.assertGreater(float(glogit.grad.abs()),0.)
        self.assertGreater(float(rlogit.grad.abs()),0.)

    def test_21_closed_gate_ste_full_mechanism_admission_gradients(self):
        model,O,X,b=fixture()
        for e in 'SLCT':
            model.gate_cls[e].logit.data.fill_(-4.)
            model.gate_box[e].logit.data.fill_(-4.)
        p=RiskPolicy(risk_margin_cls=3.,risk_margin_box=3.)
        out=mechanism_forward(model,O,X,b,hard=False,policy=p)
        self.assertEqual(int(out['hard_cls'].sum()+out['hard_box'].sum()),0)
        self.assertTrue(torch.equal(out['z'],O.z))
        self.assertTrue(torch.equal(out['b'],O.b))
        (out['pi_cls'][:,1:].sum()+out['pi_box'][:,1:].sum()).backward()
        self.assertGreater(float(model.gate_cls['S'].logit.grad.abs()),0.)
        self.assertGreater(float(model.gate_box['S'].logit.grad.abs()),0.)
        self.assertGreater(float(model.self_assessment['S'].cls_u.grad.abs()),0.)
        self.assertGreater(float(model.self_assessment['S'].box_u.grad.abs()),0.)

    def test_22_zero_availability_stops_expert_gradients(self):
        glogit=torch.nn.Parameter(torch.tensor(-4.))
        rlogit=torch.nn.Parameter(torch.tensor(-2.))
        amp=glogit.sigmoid().reshape(1,1);z=torch.zeros_like(amp)
        gt={'mask_hard':z,'mask_ste':z+(amp-amp.detach()),
            'g_hard':z,'g_ste':z+(amp-amp.detach()),'g_soft':amp}
        risk=risk_admission({'U':rlogit.sigmoid().reshape(1,1),
              'H':torch.tensor([[.8]]),'sigma':torch.tensor([[.1]]),
              'p_loss':torch.tensor([[.1]])},'box',RiskPolicy())
        out=intersect_gate(gt,risk,z)
        self.assertEqual(float(out['g_ste'].item()),0.)
        (out['g_ste']+out['ste']).sum().backward()
        self.assertEqual(float(glogit.grad),0.)
        self.assertEqual(float(rlogit.grad),0.)

if __name__=='__main__':unittest.main(verbosity=2)
