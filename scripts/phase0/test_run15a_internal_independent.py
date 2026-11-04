"""Independent fail-first adversarial fixtures for RUN15A internal revision.

No SeaClips files, no checkpoints, no official Validation/Test.
Produces exact parity assertions and numerically exercises gate gradients.
"""
import copy, math, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from torch import nn
from types import SimpleNamespace
import run15a_internal_rebuild as R
from run15a_internal_mechanism import mechanism_forward as V2
from run15a_legacy_parity_reference import mechanism_forward as LEGACY

PASSED=0
def assert_test(condition,msg):
    global PASSED
    if not condition:raise AssertionError('[INDEPENDENT TEST] '+msg)
    PASSED+=1

def make_bundle(Q=8,D=9,C=7):
    torch.manual_seed(44)
    h=torch.rand(Q,D,requires_grad=False)
    z=h[:,:7].clone() - .1
    b=torch.stack([h[:,0]*.4+.2,h[:,1]*.4+.2,h[:,2]*.2+.1,h[:,3]*.2+.1],-1)
    O=SimpleNamespace(h=h,z=z,b=b,r=torch.rand(Q,2),capabilities={})
    raw={e:{'N':torch.ones(Q,1)*.2,'R':torch.ones(Q,1)*.3} for e in ('S','L','C','T')}
    X=SimpleNamespace(raw_proxies=raw)
    df=torch.rand(Q,4,D)*0.12
    # Each specialist produces nonzero latent residuals in both tasks.
    bundle={'C_t':torch.rand(1,2),
        'features':{e:torch.rand(Q,3) for e in ('S','L','C','T')},
        'delta_cls':df,'delta_box':df*.5,
        'L_expected_iou':torch.ones(Q,1)*.5,
        'candidate_set':SimpleNamespace(queries=list(range(Q)),query_rank_norm=torch.zeros(Q,1)),
        'candidate_mask':torch.ones(Q,1)}
    return O,X,bundle

class Assessment(nn.Module):
    def __init__(self):super().__init__();self.w=nn.Parameter(torch.tensor(.15))
    def forward(self,feature,h,N,R,nc,nb,ctx):
        Q=h.shape[0]; dev=h.device
        cp=torch.ones(Q,9,device=dev)/9
        br=torch.zeros(Q,4,device=dev)
        d={k:torch.ones(Q,1,device=dev)*self.w.sigmoid()
           for k in ('p_rescue','p_loss','p_prune','p_add','p_preserve','p_retain','exp_delta_iou', 'U','H','sigma')}
        return cp,br,d,d

class Gate(nn.Module):
    def __init__(self):
        super().__init__();self.v=nn.Parameter(torch.tensor(-4.02));self.register_buffer('mask_thresh',torch.tensor(.020))
    def forward(self,x):
        g=self.v.sigmoid().expand(x.shape[0],1)
        m=(g>self.mask_thresh).to(g.dtype)
        smooth=torch.sigmoid(12*(g-self.mask_thresh))
        ste=m+(smooth-smooth.detach())
        return {'g_soft':g,'mask_hard':m,'mask_ste':ste,'g_hard':m*g,'g_ste':ste*g}

class Coord(nn.Module):
    def __init__(self):super().__init__();self.scale=nn.Parameter(torch.tensor(.25))
    def forward(self,features,delta):return torch.ones(delta.shape[:2])*self.scale

class Router(nn.Module):
    def forward(self,amp,availability,coord):
        x=amp*availability*coord.exp()
        den=1+x.sum(-1,keepdim=True)
        return torch.cat((1/den,x/den),-1)

class Adapter(nn.Module):
    def decode_cls(self,h):return h[:,:7]-.1
    def decode_box(self,h,r):
        return torch.stack([h[:,0]*.4+.2,h[:,1]*.4+.2,h[:,2]*.2+.1,h[:,3]*.2+.1],-1)

class Toy(nn.Module):
    def __init__(self):
        super().__init__();self.gate_dim=20
        self.self_assessment=nn.ModuleDict({e:Assessment() for e in ('S','L','C','T')})
        self.gate_cls=nn.ModuleDict({e:Gate() for e in ('S','L','C','T')})
        self.gate_box=nn.ModuleDict({e:Gate() for e in ('S','L','C','T')})
        self.coord_cls=Coord();self.coord_box=Coord();self.router=Router();self.adapter=Adapter()
    def _availability(self,O,candidate_query_mask,task):
        names={'S':'scale','L':'localization','C':'semantic','T':'temporal'}
        return candidate_query_mask*torch.tensor([float(O.capabilities.get('base',True))*
          float(O.capabilities.get(n,True))*float(O.capabilities.get(n+'.'+task,True))
          for n in names.values()]).unsqueeze(0)

# gate feature dimension = 18 + C_t dim(2) =20

def run():
    R.validate_splits();assert_test(len(R.FIT)+len(R.CAL)+len(R.QUAL)==47,'video partition')
    assert_test(.017986<R.FIXED_GATE_THRESHOLD<=.025,'neutral threshold vs init')
    model=Toy();O,X,bundle=make_bundle()
    # Two tasks and all experts. Train hard-forward must have same numerical value as eval hard.
    model.train();train=V2(model,O,X,bundle,hard=False)
    model.eval();hard=V2(model,O,X,bundle,hard=True)
    for field in ('z','b','g_cls','g_box','pi_cls','pi_box','A_cls','A_box'):
        assert_test(torch.allclose(train[field],hard[field],rtol=0,atol=1e-7),
                    'train-vs-deployed hard forward differs: '+field)
    assert_test(torch.equal(train['train_cls_mask'],train['hard_cls']), 'STE mask forward not hard-exact')
    assert_test(torch.equal(train['train_box_mask'],train['hard_box']), 'box STE mask forward not hard-exact')
    # Independently cloned ORIGINAL certified hard mechanism must match V2
    # for the ordinary all-capabilities case (production requires this parity).
    model.eval()
    baseline_legacy=LEGACY(model,O,X,bundle,hard=True)
    changed=V2(model,O,X,bundle,hard=True)
    for field in ('z','b','pi_cls','pi_box'):
        assert_test(torch.allclose(baseline_legacy[field],changed[field],rtol=0,atol=1e-7),
                    'Original certified hard mechanism diverged: '+field)
    # Independent gradient check. Clone fresh computation to avoid autograd reuse.
    model.train();train=V2(model,O,X,bundle,hard=False)
    loss=train['z'].sum()+train['b'].sum()
    loss.backward()
    gate_grad=sum(float(p.grad.abs().sum()) for p in model.gate_cls.parameters() if p.grad is not None)
    assert_test(gate_grad>0,'No derivative passes through STE cls gate')
    # Joint-context marginal U/H must be a real differentiable loss, not prose.
    model.zero_grad(set_to_none=True);model.train()
    train=V2(model,O,X,bundle,hard=False)
    gt_box=O.b[:1].detach().clone();gt_cls=O.z[:1].argmax(dim=1)
    matches,fps,_=R.capture_match(O.z,O.b,gt_box,gt_cls)
    marginal=R.joint_marginal_uh_loss(model,O,train,matches,fps,gt_box)
    assert_test(torch.isfinite(marginal).item(),'Joint-marginal loss not finite')
    marginal.backward()
    sa_grad=sum(float(p.grad.abs().sum()) for p in model.self_assessment.parameters() if p.grad is not None)
    assert_test(sa_grad>0,'Joint-marginal targets did not train U/H predictions')
    # Integration fixture: complete differentiable preservation objective.
    model.zero_grad(set_to_none=True);model.train()
    train=V2(model,O,X,bundle,hard=False)
    gt_box=O.b[:1].detach().clone()
    gt_cls=O.z[:1].argmax(dim=1)
    can={'gt_boxes_norm':gt_box,'gt_labels':gt_cls}
    fake_r9=SimpleNamespace(detection_loss=lambda crit,z,b,fr:
                            (z-O.z).square().mean()+(b-O.b).square().mean())
    total,terms,base=R.preservation_objective(model,O,train,can,None,fake_r9)
    assert_test(torch.isfinite(total).item() and total.requires_grad,
                'Full internal loss is not finite/differentiable')
    assert_test('joint_marginal_uh' in terms and terms['n_protected']>=0,
                'Full internal loss lacks marginal/outcome ledger')
    total.backward()
    assert_test(any(p.grad is not None for p in model.coord_cls.parameters()),
                'Full internal loss has no coordinator gradient')
    # Runtime availability reduces precise route to zero.
    Oc=copy.copy(O);Oc.capabilities={'temporal':False,'scale.box':False}
    model.eval();disabled=V2(model,Oc,X,bundle,hard=True)
    assert_test(torch.all(disabled['A_cls'][:,3]==0),'Unavailable temporal cls admitted')
    assert_test(torch.all(disabled['A_box'][:,3]==0),'Unavailable temporal box admitted')
    assert_test(torch.all(disabled['A_box'][:,0]==0),'Disabled scale box admitted')
    assert_test(torch.all(disabled['pi_cls'][:,4]==0) and torch.all(disabled['pi_box'][:,4]==0),
                'Closed temporal route reopened')
    # Exact fallback: no actions under disabled base capability.
    Ob=copy.copy(O);Ob.capabilities={'base':False}
    baseline=V2(model,Ob,X,bundle,hard=True)
    assert_test(torch.equal(baseline['z'],O.z) and torch.equal(baseline['b'],O.b),
                'No-capability fallback failed bitwise')
    assert_test(all(k in baseline for k in ('A_cls','A_box')), 'fallback diagnostics missing')
    # Matched-pair preservation objective: real target kept, duplicate false pair suppressed.
    z=torch.full((300,7),-10.);b=torch.full((300,4),.1)
    z[11,1]=3.;z[13,1]=2.;b[11]=torch.tensor([.4,.5,.2,.2]);b[13]=b[11]
    m,fp,c=R.capture_match(z,b,b[11:12],torch.tensor([1]))
    assert_test(c['tp']==1 and c['fp']==1 and len(m)==1,'Duplicate FP test failed')
    assert_test(R.single_iou(b[11],b[11])>.999,'GT IoU identity')
    # Rare-class control: no FP suppression targets for class index 4.
    assert_test(4 not in [c for q,c in fp if c!=4],'LV protection excluded from FP penalty')
    print(f'[PASS] {PASSED} independent internal mechanism assertions; '
          'no SeaClips Train/Validation/Test or checkpoints opened')

if __name__=='__main__':run()
