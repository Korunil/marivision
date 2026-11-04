"""Operationally anchored FIT-only preservation surrogate for B-FCEF-F.

Targets are derived from actual baseline TP query/class/GT anchors obtained
with the source-pinned global Top100, score>=.20, IoU>=.30 matcher.
This is not a GT-free inference rule and cannot prove unseen TP preservation.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F

@dataclass(frozen=True)
class PreservationPolicy:
    score_threshold: float = .20
    iou_threshold: float = .30
    safety_epsilon: float = .002
    rank_epsilon: float = .001


def box_iou_cxcywh(x:torch.Tensor,y:torch.Tensor)->torch.Tensor:
    if x.shape!=y.shape or x.ndim!=2 or x.shape[1]!=4:
        raise ValueError('cxcywh boxes expected as [N,4]')
    xa=x[:,:2]-x[:,2:]/2;xb=x[:,:2]+x[:,2:]/2
    ya=y[:,:2]-y[:,2:]/2;yb=y[:,:2]+y[:,2:]/2
    wh=(torch.minimum(xb,yb)-torch.maximum(xa,ya)).clamp_min(0)
    inter=wh.prod(-1)
    area=x[:,2:].clamp_min(0).prod(-1)+y[:,2:].clamp_min(0).prod(-1)-inter
    return inter/area.clamp_min(1e-9)


def protected_gt_loss(base_z,base_b,new_z,new_b,gt_boxes,anchors,policy:PreservationPolicy=PreservationPolicy()):
    """Zero on an exact fallback and positive on score cutoff / IoU erosion.

    anchors: iterable (query_idx,class_idx,gt_idx) matched by baseline.
    Includes continuous score erosion, threshold barrier, box erosion,
    IoU threshold barrier, and native Top100 boundary surrogate.
    Does not guarantee exact global greedy GT matching; verify separately.
    """
    if base_z.shape!=new_z.shape or base_b.shape!=new_b.shape:
        raise ValueError('Baseline/new output shapes mismatch')
    if base_z.ndim!=2 or base_b.shape!=(base_z.shape[0],4) or gt_boxes.ndim!=2 or gt_boxes.shape[1]!=4:
        raise ValueError('Malformed detector tensors')
    # The cached FIT GT is stored on CPU, while R0/native outputs live on CUDA.
    # Align GT with the box computation ONCE before any tensor indexing. Merely
    # moving the GT indices to CPU is insufficient: box IoU still requires
    # both operands to share a device (and a floating-point dtype).
    if not (base_z.device==new_z.device==base_b.device==new_b.device):
        raise ValueError('Baseline/new detector outputs must share one device')
    if not (torch.is_floating_point(gt_boxes) and torch.is_floating_point(base_b)):
        raise ValueError('GT and predicted boxes must be floating point')
    gt_boxes=gt_boxes.to(device=new_b.device,dtype=new_b.dtype)
    anchors=list(anchors)
    zero=new_z.sum()*0+new_b.sum()*0
    if not anchors:
        return {'score_erosion':zero,'cutoff':zero,'box_erosion':zero,'iou_cutoff':zero,'rank':zero,'total':zero}
    q=torch.tensor([int(a[0]) for a in anchors],dtype=torch.long,device=new_z.device)
    c=torch.tensor([int(a[1]) for a in anchors],dtype=torch.long,device=new_z.device)
    g=torch.tensor([int(a[2]) for a in anchors],dtype=torch.long,device=new_z.device)
    if int(q.min())<0 or int(q.max())>=len(base_z) or int(c.min())<0 or int(c.max())>=base_z.shape[1] or int(g.min())<0 or int(g.max())>=len(gt_boxes):
        raise ValueError('Anchor index out of bounds')
    bs=base_z[q,c].detach().sigmoid(); ns=new_z[q,c].sigmoid()
    score_erosion=F.relu(bs-ns).square().mean()
    cutoff_anchor=torch.minimum(bs,torch.full_like(ns,policy.score_threshold+policy.safety_epsilon))
    cutoff=F.relu(cutoff_anchor-ns).square().mean()
    old_iou=box_iou_cxcywh(base_b[q].detach(),gt_boxes[g]).detach()
    new_iou=box_iou_cxcywh(new_b[q],gt_boxes[g])
    box_erosion=F.relu(old_iou-new_iou).square().mean()
    iou_anchor=torch.minimum(old_iou,torch.full_like(new_iou,policy.iou_threshold+policy.safety_epsilon))
    iou_cutoff=F.relu(iou_anchor-new_iou).square().mean()
    flat=new_z.sigmoid().flatten()
    boundary=flat.topk(min(100,flat.numel())).values[-1].detach()
    # Only an R0-matched GT pair can be protected, but its ranking can be
    # altered by OTHER queries; this soft term catches that effect.
    rank_anchor=torch.minimum(bs,boundary+policy.rank_epsilon)
    rank=F.relu(rank_anchor-ns).square().mean()
    return {'score_erosion':score_erosion,'cutoff':cutoff,
            'box_erosion':box_erosion,'iou_cutoff':iou_cutoff,'rank':rank,
            'total':score_erosion+cutoff+box_erosion+iou_cutoff+rank}


def identity_ledger(base_matches,new_matches):
    """Exact frame-level identity accounting; no net-TP substitution."""
    b={int(x[2]) for x in base_matches};n={int(x[2]) for x in new_matches}
    return {'lost':sorted(b-n),'rescued':sorted(n-b),
            'baseline_matched':len(b),'new_matched':len(n),
            'net_tp':len(n)-len(b)}
