"""Independent adversarial tests for RUN15A V2 FIT-only read-only mechanism audit.

No SeaClips, checkpoints, CAL, QUAL, Validation or Test files read.
"""
import collections
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
import run15a_v2_fit_mechanism_audit as a


def run():
    n=0
    def check(v,msg):
        nonlocal n
        assert v,msg
        n+=1

    class TorchFake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param=torch.nn.Parameter(torch.tensor(1.0))
        def _availability(self,O,mask,task):return torch.ones(5,4)
    m=TorchFake();ref=m._availability
    b=m.param.detach().clone()
    with a.availability_knockout(m,'T'):
        for t in ('cls','box'):
            A=m._availability(None,None,t)
            check(bool(torch.equal(A[:,3],torch.zeros(5))), 'T knockout missing')
            check(bool(torch.equal(A[:,0],torch.ones(5))), 'S modified inadvertently')
    check(m._availability==ref and '_availability' not in vars(m),'Torch module method restoration failed')
    check(torch.equal(m.param.detach(),b),'Knockout changed a model parameter')
    try:
        with a.availability_knockout(m,'S'):
            raise LookupError('injected')
    except LookupError:pass
    check(m._availability==ref,'Injected-exception restoration failed')

    with tempfile.TemporaryDirectory() as td:
        root=Path(td);file=root/'record.json'
        a.json_new(file,{'x':1})
        check(a.json_read(file)=={'x':1},'JSON serialization incorrect')
        try:a.json_new(file,{'x':2})
        except AssertionError:pass
        else:raise AssertionError('Provenance file silently overwritten')
        rows=[{'a':'3','b':'4'}]
        a.csv_new(root/'ledger.csv',rows,['a','b'])
        check('3,4' in (root/'ledger.csv').read_text(),'CSV ledger incorrect')
        check(len(a.sha(file))==64,'SHA must be SHA-256')

    q=5
    baseline=SimpleNamespace(h=torch.zeros(q,8),z=torch.zeros(q,7),b=torch.zeros(q,4))
    pi=torch.zeros(q,5);pi[:,0]=1
    d={'z':baseline.z.clone(),'b':baseline.b.clone(),'pi_cls':pi.clone(),'pi_box':pi.clone()}
    for t in ('cls','box'):
        d['A_'+t]=torch.ones(q,4)
        d['hard_'+t]=torch.zeros(q,4)
        d['g_'+t]=torch.zeros(q,4)
    a.assert_model_shapes(d,baseline);n+=1
    g=a.new_stats()['gate']
    a.update_gate_stats(g,d)
    check(g['all_queries']==5 and g['cls']['candidate_queries']==5,'Eligibility denominator wrong')
    check(g['cls']['active_queries']==0 and g['box']['active_queries']==0,'False positive open gate')
    d['hard_cls'][1,2]=1; d['pi_cls'][1,0]=0.7;d['pi_cls'][1,3]=0.3
    a.assert_model_shapes(d,baseline);n+=1
    a.update_gate_stats(g,d)
    check(g['cls']['active_queries']==1 and g['cls']['candidate_queries']==10,'Query activations confused with eligible count')
    check(g['cls']['experts']['C']['opened']==1,'Expert C activation incorrectly counted')
    check(abs(g['cls']['experts']['C']['route_mass']-0.3)<1e-5,'Expert C mass incorrect')
    check(g['neither_active']==9,'Cross-task neither-open incorrect')

    logits=torch.full((300,7),-20.)
    logits[4,2]=torch.logit(torch.tensor(.201))
    box=torch.zeros(300,4)
    gt=torch.zeros(4)
    one=lambda u,v:torch.tensor(1.)
    two=lambda u,v:torch.tensor(.2)
    stage,sc,iu=a.loss_stage(4,2,box,gt,logits,{(4,2)},one)
    check(stage=='greedy_competition_or_other','Stage default failed')
    logits[4,2]=torch.logit(torch.tensor(.19))
    stage,sc,iu=a.loss_stage(4,2,box,gt,logits,set(),two)
    check(stage=='score_below_0p20','Score-before-Top100 diagnostic priority violated')
    logits[4,2]=torch.logit(torch.tensor(.3))
    check(a.loss_stage(4,2,box,gt,logits,set(),one)[0]=='top100_pair_evicted','Top100 stage failed')
    check(a.loss_stage(4,2,box,gt,logits,{(4,2)},two)[0]=='same_pair_box_iou_below_0p30','IoU stage failed')
    check((4,2) in a.top_pairs(logits), 'Top100 rank conversion wrong')

    stats=a.new_stats()
    labels=torch.tensor([0,1,2]);scale=['small','medium','large']
    lost,rescue=a.accounting(stats,stats['classes'],stats['scale'],{0,1},{1,2},labels,scale)
    check((lost,rescue)==({0},{2}),'Identity-preservation test conflated with equal TP')
    check(stats['classes']['lost']==collections.Counter({'ANIMAL':1}),'Lost class incorrect')
    check(stats['scale']['rescued']==collections.Counter({'large':1}),'Rescued scale incorrect')

    # Merge a one-video synthetic audit. Guarantee signed TP identity conservation.
    s=a.new_stats();s['n_frames']=1;s['gt']=3
    s['counts']['Base'].update({'tp':2,'fp':1,'fn':1});s['counts']['Full'].update({'tp':2,'fp':0,'fn':1})
    for v in a.VARIANTS:
        if v not in ('Base','Full'):s['counts'][v].update({'tp':2,'fp':1,'fn':1})
    s['lost_against_base']['Full']=1;s['rescue_against_base']['Full']=1
    s['lost_stage']['score_below_0p20']=1
    summary=a.output_stats(s);summary['camera']='e-CAM'
    manifest={'hashes':{'driver':'a'}}
    report=a.merged_report({'3':summary},manifest,stride=1)
    check(report['metrics']['Base']['tp']==2 and report['metrics']['Full']['tp']==2,
          'Aggregated identity-swapped TP count wrong')
    check(report['lost_gt_against_base']['Full']==1 and report['rescued_gt_against_base']['Full']==1,
          'Aggregate matched identity conservation failed')
    check(report['lost_gt_stage_counts']=={'score_below_0p20':1},'Stage aggregation wrong')
    check(report['FIT_cached_frames_sampled']==1,'FIT frame aggregation wrong')

    # End-to-end miniature FIT replay through the real audit aggregation and
    # attribution code with a production-shaped, deterministic synthetic source.
    class FakeModel2:
        def _availability(self, O, candidate_query_mask, task):
            out=torch.zeros(300,4)
            out[3,0]=1
            return out
        def __call__(self,O,X,**kw):
            z,boxes,info=FakeMechanism.mechanism_forward(self,O,X,{},True)['z'],None,None
            out=FakeMechanism.mechanism_forward(self,O,X,{},True)
            return z,out['b'],{}
    class FakeMechanism:
        @staticmethod
        def mechanism_forward(model,O,X,bundle,hard):
            avail=model._availability(O,None,'cls')
            opened=avail.clone()
            z=O.z.clone();b=O.b.clone()
            if bool(opened[3,0]>0):
                z[3,1]=torch.logit(torch.tensor(.19465))
            route=torch.zeros(300,5)
            route[:,0]=1
            if bool(opened[3,0]>0):
                route[3,0]=.5;route[3,1]=.5
            return dict(z=z,b=b,A_cls=avail,A_box=avail.clone(),
                hard_cls=opened,hard_box=opened.clone(),
                g_cls=opened,g_box=opened.clone(),pi_cls=route,pi_box=route.clone())
    class FakeSource:
        @staticmethod
        def load_video_triplet(vid):
            f={'image_id':101,'gt_labels':torch.tensor([1]),
               'gt_boxes_norm':torch.tensor([[.5,.5,.2,.2]])}
            return [f],[{}],[{}]
        @staticmethod
        def make_O_X(fr,spa,tmp,co,fx):
            z=torch.full((300,7),-20.)
            z[3,1]=torch.logit(torch.tensor(.201))
            b=torch.zeros(300,4);b[3]=torch.tensor([.5,.5,.2,.2])
            return SimpleNamespace(h=torch.zeros(300,8),z=z,b=b),object()
    class FakeR9:
        @staticmethod
        def expert_bundle(model,O,X):return {}
    class FakeOriginal:
        @staticmethod
        def verify_gate_thresholds(model,stage):return True
        @staticmethod
        def single_iou(x,y):
            if torch.allclose(x,y,atol=1e-6):return torch.tensor(1.)
            return torch.tensor(0.)
        @staticmethod
        def capture_match(z,b,gt,labels):
            scores=z.sigmoid().flatten()
            idx=scores.topk(100).indices
            used=False;mm=[];fp=[]
            for flat in idx.tolist():
                score=float(scores[flat])
                if score < .2:continue
                qi,ci=divmod(flat,7)
                if not used and qi==3 and ci==1 and torch.allclose(b[3],gt[0]):
                    mm.append((3,1,0));used=True
                else:fp.append((qi,ci))
            return mm,fp,dict(tp=len(mm),fp=len(fp),fn=1-len(mm))
    fake_raw={'images':[{'id':101,'video_id':3}],
              'videos':[{'id':3,'camera_sensor':'e-CAM'}],
              'annotations':[{'image_id':101,'category_id':2}]}
    report,ledger,cache_size,camera=a.process_video(
       3,fake_raw,FakeOriginal,FakeMechanism,FakeModel2(),None,None,
       FakeSource,FakeR9,stride=1)
    check(report['frames']==1 and report['gt']==1 and cache_size==1,'FIT mini replay coverage')
    check(report['lost_gt_against_base']['Full']==1,'FIT mini replay lost identity')
    check(report['lost_gt_against_base']['NoS']==0,'Expert S knockout did not rescue')
    check(report['loss_stages']=={'score_below_0p20':1},'Loss stage incorrect in replay')
    check(len(ledger)==1 and ledger[0]['without_S_preserves_gt']==1,'Lost GT ledger expert attribution')
    check(report['class_scale']['support']=={'BOAT|large':1},'FIT class-scale cross-tab incorrect')
    report['camera']=camera
    global_report=a.merged_report({'3':report},{'hashes':{'driver':'fixture'}},stride=1)
    check(global_report['lost_gt_against_base']['Full']==1,'FIT global replay merge incorrect')

    print(f'[PASS] {n} independent FIT-audit assertions; no SeaClips, model weights or CAL/QUAL opened')

if __name__=='__main__':run()
