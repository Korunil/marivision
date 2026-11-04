"""B-FCEF-F task-specific detector-consequence admission primitives.

This module is an execution component, not a trained model or a calibration.
It uses only inference-observable predicted U/H/uncertainty/loss.
No ground truth is used to decide at inference.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch

EXPERTS = ('S', 'L', 'C', 'T')

@dataclass(frozen=True)
class RiskPolicy:
    risk_margin_cls: float = 0.0
    risk_margin_box: float = 0.0
    uncertainty_weight: float = 0.5
    consequence_loss_weight: float = 0.5
    temperature: float = 0.15

    def validate(self) -> None:
        if self.temperature <= 0 or self.uncertainty_weight < 0 or self.consequence_loss_weight < 0:
            raise ValueError('Invalid risk-policy parameters')


def risk_admission(channels: dict, task: str, policy: RiskPolicy) -> dict:
    """Produce exact hard mask and STE mask, both [Q,1].

    The learned U/H/sigma are predictions, not verified actual harm.
    This is a fixed policy until a separately locked FIT calibration exists.
    """
    policy.validate()
    if task not in ('cls', 'box'):
        raise ValueError('Unknown task')
    for key in ('U','H','sigma','p_loss'):
        if key not in channels:raise ValueError('Missing consequence channel: '+key)
    U,H,sigma,loss=[channels[k] for k in ('U','H','sigma','p_loss')]
    if not (U.shape==H.shape==sigma.shape==loss.shape and U.ndim==2 and U.shape[1]==1):
        raise ValueError('Consequence channels must be [Q,1]')
    if not all(bool(torch.isfinite(a).all()) for a in (U,H,sigma,loss)):
        raise ValueError('Nonfinite risk evidence')
    margin=policy.risk_margin_cls if task=='cls' else policy.risk_margin_box
    if task=='cls' and 'p_add' in channels:
        introduced_fp=channels['p_add']
        if introduced_fp.shape!=loss.shape:raise ValueError('p_add shape drift')
        loss=torch.maximum(loss,introduced_fp)
    score=U-H-policy.uncertainty_weight*sigma-policy.consequence_loss_weight*loss-margin
    hard=(score>0).to(score.dtype)
    soft=torch.sigmoid(score/policy.temperature)
    # Parenthesize the surrogate zero *before* adding the hard value.
    # (hard + soft) - soft.detach() can move hard by one ULP in float32.
    ste=hard+(soft-soft.detach())
    return {'score':score,'mask_hard':hard,'mask_ste':ste,'soft':soft}


def intersect_gate(gate: dict, risk: dict, availability: torch.Tensor) -> dict:
    """Exact conjunction at inference; jointly differentiable training surrogate.

    Independent hard-forward STE masks MUST NOT be multiplied for training:
    when both TSK and risk masks are closed, the product-rule gives exactly
    zero gradient to both, trapping the model in universal abstention.

    Instead, use a continuous relaxed *joint* surrogate to supply gradients
    while preserving bit-exact hard-forward values and availability masking.
    The TSK `g_soft` is the existing native adaptive-rule amplitude; risk
    `soft` is the existing smooth risk decision. Neither is an extra model.
    """
    for key in ('mask_hard', 'mask_ste', 'g_hard', 'g_ste', 'g_soft'):
        if key not in gate:
            raise ValueError('Missing TSK component '+key)
    for key in ('mask_hard', 'mask_ste', 'soft'):
        if key not in risk:
            raise ValueError('Missing risk component '+key)
    if availability.ndim != 2 or availability.shape[1] != 1:
        raise ValueError('availability must be [Q,1]')
    for x in (gate['mask_hard'], gate['g_soft'], risk['mask_hard'], risk['soft']):
        if x.shape != availability.shape:
            raise ValueError('Joint gate/risk/availability shape mismatch')
    hard = gate['mask_hard'] * risk['mask_hard'] * availability
    g_hard = gate['g_hard'] * risk['mask_hard'] * availability
    # Inference ALWAYS receives `hard` and `g_hard`.  Training receives
    # their exact forward values plus a smooth zero-forward surrogate.
    # This creates gradients even when BOTH permission masks are closed.
    relaxed = gate['g_soft'] * risk['soft'] * availability
    joint_ste = hard + (relaxed - relaxed.detach())
    g_ste = g_hard + (relaxed - relaxed.detach())
    if torch.any(g_hard[hard == 0] != 0):
        raise AssertionError('Closed gate carries corrective mass')
    if not torch.equal(joint_ste, hard) or not torch.equal(g_ste, g_hard):
        raise AssertionError('Joint admission surrogate changed forward values')
    return {'hard': hard, 'ste': joint_ste, 'g_hard': g_hard, 'g_ste': g_ste}


def native_exact_output(original:torch.Tensor, decoded:torch.Tensor, hard_query:torch.Tensor,
                        ste_query:torch.Tensor, training: bool) -> torch.Tensor:
    """Exact forward bypass and native active tensors, with STE backward.

    When training=True, the value is IDENTICAL to hard inference, but the
    backward graph includes both the gate and the native corrected head.
    Unlike soft candidate-wide bypass, closed queries have exact R0 outputs.
    """
    if original.shape!=decoded.shape:raise ValueError('Native output shape mismatch')
    if hard_query.shape!=(original.shape[0],):raise ValueError('Bad hard query mask')
    if ste_query.shape!=(original.shape[0],):raise ValueError('Bad STE query mask')
    hard_query=hard_query.bool()
    hard_output=original.clone()
    if bool(hard_query.any()):
        hard_output[hard_query]=decoded[hard_query] if not training else decoded[hard_query].detach()
    if not training:
        return hard_output
    smooth = original + ste_query.unsqueeze(-1)*(decoded-original)
    # STE: hard-forward bitwise equal, differentiable proxy only backward.
    return hard_output + (smooth-smooth.detach())
