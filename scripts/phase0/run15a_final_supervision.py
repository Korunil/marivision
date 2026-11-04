"""GT-based FIT-only deep consequence supervision for permanent B-FCEF-F experts.

Each isolated expert/task proposal is replayed through the SAME native
heads and Top100 matcher as deployment, instead of score-direction proxies.
Isolated action targets are NOT a proof of joint-model outcome causality;
full joint GT identities are checked separately in the preservation objective.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

EXPERTS=('S','L','C','T')
TP,FP,NONOP=0,1,2

def pair_states(matched, false_positives, q_count, classes,device):
    x=torch.full((q_count,classes),NONOP,dtype=torch.long,device=device)
    for q,c in false_positives:x[int(q),int(c)]=FP
    for q,c,g in matched:x[int(q),int(c)]=TP
    return x

def joint_transition(base,local):
    if base.shape!=local.shape:raise ValueError('State shape mismatch')
    if not bool(torch.all((base>=0)&(base<3)&(local>=0)&(local<3))):
        raise ValueError('Unknown TP/FP/NONOP state')
    return 3*base+local

def isolated_native_targets(model,O,bundle,gt_boxes,gt_labels,matcher):
    """8 exact isolated expert/task actions at native output level.

    Always applies one expert's whole-candidate residual at full amplitude.
    Does not use trained gate/router and cannot read CAL/QUAL/Test by itself.
    """
    q,c=O.z.shape
    device=O.h.device; candidate=bundle['candidate_mask'].reshape(-1).bool()
    base_m,base_fp,_=matcher(O.z,O.b,gt_boxes,gt_labels)
    base_state=pair_states(base_m,base_fp,q,c,device)
    targets={}
    with torch.no_grad():
        for k,expert in enumerate(EXPERTS):
            per_task={}
            for task in ('cls','box'):
                zz=O.z.clone();bb=O.b.clone()
                delta=bundle['delta_cls' if task=='cls' else 'delta_box'][:,k]
                if bool(candidate.any()):
                    h=O.h[candidate]+delta[candidate]
                    if task=='cls':zz[candidate]=model.adapter.decode_cls(h)
                    else:bb[candidate]=model.adapter.decode_box(h,O.r[candidate])
                local_m,local_fp,_=matcher(zz,bb,gt_boxes,gt_labels)
                state=pair_states(local_m,local_fp,q,c,device)
                if task=='cls':
                    relevant=(O.z.sigmoid()>=.20)|(zz.sigmoid()>=.20)
                    relevant &=candidate[:,None]
                    t=joint_transition(base_state,state)
                    per_task[task]={'transition':t,'relevant':relevant}
                else:
                    before={int(qi):int(gi) for qi,ci,gi in base_m}
                    after={int(qi):int(gi) for qi,ci,gi in local_m}
                    rescue=torch.zeros((q,1),device=device)
                    loss=torch.zeros((q,1),device=device)
                    retain=torch.zeros((q,1),device=device)
                    diou=torch.zeros((q,1),device=device)
                    imask=torch.zeros((q,1),dtype=torch.bool,device=device)
                    for qi in range(q):
                        if not bool(candidate[qi]):continue
                        old=before.get(qi);new=after.get(qi)
                        loss[qi,0]=float(old is not None and old!=new)
                        rescue[qi,0]=float(new is not None and new!=old)
                        retain[qi,0]=float(old is not None and old==new)
                        if old is not None:
                            from run15a_final_preservation import box_iou_cxcywh
                            xy=gt_boxes[old:old+1].to(device)
                            delta_iou=box_iou_cxcywh(bb[qi:qi+1],xy)-box_iou_cxcywh(O.b[qi:qi+1],xy)
                            diou[qi,0]=delta_iou.item();imask[qi,0]=True
                    per_task[task]={'rescue':rescue,'loss':loss,'retain':retain,
                                    'delta_iou':diou,'iou_mask':imask,'relevant':candidate[:,None]}
            targets[expert]=per_task
    return targets

def expert_deep_supervision(out,targets):
    """Weighted classwise 3x3 NLL and localization consequences.

    Consequence/value heads get explicit gradient. No GT is allowed at
    inference; targets exist only in FIT optimization.
    """
    total=None;detail={}
    class_weights=torch.tensor([2,16,16,8,1,4,10,8,.1],dtype=torch.float32)
    for expert in EXPERTS:
        cls=out['cls_raw'][expert]
        raw=out['box_raw'][expert]
        t=targets[expert]
        relevant=t['cls']['relevant']
        if bool(relevant.any()):
            y=t['cls']['transition'][relevant]
            p=cls[relevant].clamp_min(1e-8)
            w=class_weights.to(device=p.device)[y]
            v=-(p.gather(-1,y[:,None]).squeeze(-1).log()*w).sum()/w.sum().clamp_min(1)
        else:v=cls.sum()*0
        # Train the learnable U/H aggregator on explicit GT transition
        # consequences, not solely on an indirect detector loss.
        uc=out['cls_channels'][expert]['U'].reshape(-1)
        hc=out['cls_channels'][expert]['H'].reshape(-1)
        uq=torch.zeros(uc.shape,device=uc.device);hq=torch.zeros_like(uq)
        if bool(relevant.any()):
            target=t['cls']['transition']
            util=((target==3)|(target==5)|(target==6)) & relevant
            harm=((target==1)|(target==2)|(target==7)) & relevant
            uq=util.any(-1).float();hq=harm.any(-1).float()
            valid=relevant.any(-1)
            risk_cls=F.binary_cross_entropy(uc[valid].clamp(1e-6,1-1e-6),uq[valid])+F.binary_cross_entropy(hc[valid].clamp(1e-6,1-1e-6),hq[valid])
        else:risk_cls=cls.sum()*0
        b=t['box'];mask=b['relevant'].reshape(-1)
        if bool(mask.any()):
            targets_box=torch.cat([b[k] for k in ('rescue','loss','retain')],dim=-1)
            bce=F.binary_cross_entropy_with_logits(raw[mask,:3],targets_box[mask],reduction='mean')
        else:bce=raw.sum()*0
        imask=b['iou_mask'].reshape(-1)
        iou=F.mse_loss(raw[imask,3:4].tanh(),b['delta_iou'][imask]) if bool(imask.any()) else raw.sum()*0
        if bool(mask.any()):
            box_u=out['box_channels'][expert]['U'].reshape(-1)[mask].clamp(1e-6,1-1e-6)
            box_h=out['box_channels'][expert]['H'].reshape(-1)[mask].clamp(1e-6,1-1e-6)
            risk_box=(F.binary_cross_entropy(box_u,b['rescue'].reshape(-1)[mask])+
                      F.binary_cross_entropy(box_h,b['loss'].reshape(-1)[mask]))
        else:risk_box=raw.sum()*0
        detail[expert]={'cls':v,'box':bce,'box_iou':iou,'risk_cls':risk_cls,'risk_box':risk_box}
        this=v+.5*bce+.25*iou+.5*risk_cls+.5*risk_box
        total=this if total is None else total+this
    return total/len(EXPERTS),detail
