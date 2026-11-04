from __future__ import annotations
"""RUN_15A final candidate: task-decoupled risk-aware sparse TSK fusion.
Frozen upstream R0 and specialists; native exact hard-forward STE.
This is NEW implementation; not a replacement of the certified V2 source.
"""
import math
import torch
from run15a_final_risk import RiskPolicy, risk_admission, intersect_gate
EXPERTS=("S","L","C","T")
LEGACY_SHA256="d51399ed4eaea068dbcfd0a330ad87b01fe277ce907df9e3121af460a514976e"

def mechanism_forward(model, O, X, bundle, hard: bool, policy: RiskPolicy | None = None):
    policy=policy or RiskPolicy()
    C_t = bundle["C_t"]
    delta_cls = bundle["delta_cls"]
    delta_box = bundle["delta_box"]
    features = bundle["features"]
    L_expected_iou = bundle["L_expected_iou"]
    candidate_set = bundle["candidate_set"]
    candidate_mask = bundle["candidate_mask"]

    q_count = O.h.shape[0]
    dtype = O.h.dtype
    device = O.h.device

    # Availability is candidate membership for all permanent task-capable experts.
    A_cls = model._availability(O, candidate_mask, task="cls")
    A_box = model._availability(O, candidate_mask, task="box")

    cls_raw = {}
    box_raw = {}
    cls_channels = {}
    box_channels = {}

    for k, expert in enumerate(EXPERTS):
        N = X.raw_proxies[expert]["N"].reshape(q_count, 1)
        R = X.raw_proxies[expert]["R"].reshape(q_count, 1)
        ncls = delta_cls[:, k].norm(dim=-1, keepdim=True)
        nbox = delta_box[:, k].norm(dim=-1, keepdim=True)

        out = model.self_assessment[expert](
            features[expert],
            O.h,
            N,
            R,
            ncls,
            nbox,
            C_t,
        )

        if not isinstance(out, tuple) or len(out) != 4:
            raise AssertionError(
                "[FATAL] Corrected self-assessment four-output contract drift."
            )

        cp, br, cc, bc = out
        cls_raw[expert] = cp
        box_raw[expert] = br
        cls_channels[expert] = cc
        box_channels[expert] = bc

    probability = O.z.sigmoid().clamp(1e-6, 1 - 1e-6)
    sorted_probability, _ = torch.sort(probability, dim=-1, descending=True)
    score = sorted_probability[:, 0:1]
    margin = sorted_probability[:, 0:1] - sorted_probability[:, 1:2]
    bern = -(
        probability * probability.log()
        + (1 - probability) * (1 - probability).log()
    ).mean(dim=-1, keepdim=True) / math.log(2.0)
    area = (O.b[:, 2:3] * O.b[:, 3:4]).clamp_min(1e-8)
    log_area = area.log()
    aspect = O.b[:, 2:3] / O.b[:, 3:4].clamp_min(1e-4)
    candidate_rank = candidate_set.query_rank_norm
    Cq = C_t.expand(q_count, -1)

    gate_cls_features = []
    gate_box_features = []
    cls_gate_out = []
    box_gate_out = []

    for k, expert in enumerate(EXPERTS):
        N = X.raw_proxies[expert]["N"].reshape(q_count, 1)
        R = X.raw_proxies[expert]["R"].reshape(q_count, 1)
        ncls = delta_cls[:, k].norm(dim=-1, keepdim=True)
        nbox = delta_box[:, k].norm(dim=-1, keepdim=True)
        cc = cls_channels[expert]
        bc = box_channels[expert]

        cls_quality = torch.zeros(q_count, 1, device=device, dtype=dtype)
        box_quality = (
            L_expected_iou
            if expert == "L"
            else torch.zeros(q_count, 1, device=device, dtype=dtype)
        )

        gf_cls = torch.cat([
            score, margin, bern, log_area, aspect, candidate_rank,
            N, R,
            cc["p_rescue"], cc["p_loss"], cc["p_prune"], cc["p_add"], cc["p_preserve"],
            cc["U"], cc["H"], cc["sigma"],
            ncls, cls_quality, Cq,
        ], dim=-1)

        gf_box = torch.cat([
            score, margin, bern, log_area, aspect, candidate_rank,
            N, R,
            bc["p_rescue"], bc["p_loss"], bc["p_retain"], bc["exp_delta_iou"],
            bc["U"], bc["H"], bc["sigma"],
            nbox, box_quality,
            torch.zeros(q_count, 1, device=device, dtype=dtype),
            Cq,
        ], dim=-1)

        if gf_cls.shape[-1] != model.gate_dim or gf_box.shape[-1] != model.gate_dim:
            raise AssertionError("[FATAL] Gate feature dimension drift.")

        gate_cls_features.append(gf_cls)
        gate_box_features.append(gf_box)
        cls_gate_out.append(model.gate_cls[expert](gf_cls))
        box_gate_out.append(model.gate_box[expert](gf_box))

    # Joint TSK and predicted consequence-risk admission, separate per task.
    cls_inter=[]
    box_inter=[]
    for k, expert in enumerate(EXPERTS):
        # Class-wise TP-loss evidence must not be diluted by averaging seven
        # classes: one rare-class baseline detection can be the only harmed
        # hypothesis for a query. The expert's 3x3 transition posterior is
        # evaluated for every original class above operational threshold.
        # The original head predicts these probabilities without GT at runtime.
        posterior=cls_raw[expert]
        if posterior.shape!=(q_count,O.z.shape[1],9):
            raise AssertionError('Classwise 3x3 posterior contract changed')
        high=O.z.sigmoid()>=.20
        at_risk=((posterior[...,1]+posterior[...,2])*high).amax(-1,keepdim=True)
        introduced_fp=posterior[...,7].amax(-1,keepdim=True)
        cls_risk_channels=dict(cls_channels[expert])
        cls_risk_channels['p_loss']=torch.maximum(cls_channels[expert]['p_loss'],at_risk)
        cls_risk_channels['p_add']=torch.maximum(cls_channels[expert]['p_add'],introduced_fp)
        cls_risk=risk_admission(cls_risk_channels, 'cls', policy)
        box_risk=risk_admission(box_channels[expert], 'box', policy)
        cls_inter.append(intersect_gate(cls_gate_out[k],cls_risk,A_cls[:,k:k+1]))
        box_inter.append(intersect_gate(box_gate_out[k],box_risk,A_box[:,k:k+1]))
    hard_cls=torch.cat([v['hard'] for v in cls_inter],dim=-1)
    hard_box=torch.cat([v['hard'] for v in box_inter],dim=-1)
    train_cls_mask=torch.cat([v['ste'] for v in cls_inter],dim=-1)
    train_box_mask=torch.cat([v['ste'] for v in box_inter],dim=-1)
    g_cls=torch.cat([v['g_hard' if hard else 'g_ste'] for v in cls_inter],dim=-1)
    g_box=torch.cat([v['g_hard' if hard else 'g_ste'] for v in box_inter],dim=-1)

    coord_in_cls = torch.cat(gate_cls_features + [g_cls, hard_cls if hard else train_cls_mask, A_cls], dim=-1)
    coord_in_box = torch.cat(gate_box_features + [g_box, hard_box if hard else train_box_mask, A_box], dim=-1)

    coord_cls = model.coord_cls(coord_in_cls, delta_cls)
    coord_box = model.coord_box(coord_in_box, delta_box)

    pi_cls = model.router(g_cls, A_cls, coord_cls)
    pi_box = model.router(g_box, A_box, coord_box)

    h_cls = O.h + torch.sum(pi_cls[:, 1:].unsqueeze(-1) * delta_cls, dim=1)
    h_box = O.h + torch.sum(pi_box[:, 1:].unsqueeze(-1) * delta_box, dim=1)

    # Native heads are decoded for the entire query population only during
    # training (to allow STE gradients through closed gates). Both training
    # and inference have EXACT hard-forward output semantics.
    active_cls=hard_cls.sum(dim=-1)>0
    active_box=hard_box.sum(dim=-1)>0
    # Both execution modes compute the authoritative active-query native
    # forward values with the *same indexing and decoder batch sizes*.
    # Full-population decode is used ONLY as a zero-valued backward proxy.
    # This prevents CUDA GEMM/batch-size rounding differences between
    # hard inference (active slice) and training (all 300 queries).
    z=O.z.clone(); b=O.b.clone()
    if bool(active_cls.any()):
        native_cls=model.adapter.decode_cls(h_cls[active_cls])
        z[active_cls]=native_cls if hard else native_cls.detach()
    if bool(active_box.any()):
        native_box=model.adapter.decode_box(h_box[active_box],O.r[active_box])
        b[active_box]=native_box if hard else native_box.detach()
    if not hard:
        # Native inference values are authoritative, including exact zero
        # change on every hard-closed query. Backward uses a single gate-
        # controlled decoder path. h_cls/h_box are already gated by the
        # joint-STE baseline simplex router. Multiplying the decoded residual
        # by an additional STE OR mask causes a DOUBLE-ZERO derivative at
        # all-closed initialization: gate=0 and decoded-baseline=0.
        #
        # The zero-forward graft below provides the detector loss a gradient
        # through the original gate/risk -> router -> native decoder chain,
        # without altering a single inference output value. Availability
        # masking remains inside the gate-constrained router.
        d_cls=model.adapter.decode_cls(h_cls)
        d_box=model.adapter.decode_box(h_box,O.r)
        z=z+(d_cls-d_cls.detach())
        b=b+(d_box-d_box.detach())

    return {
        "z": z,
        "b": b,
        "cls_raw": cls_raw,
        "box_raw": box_raw,
        "cls_channels": cls_channels,
        "box_channels": box_channels,
        "g_cls": g_cls,
        "g_box": g_box,
        "hard_cls": hard_cls,
        "hard_box": hard_box,
        "pi_cls": pi_cls,
        "pi_box": pi_box,
        "A_cls": A_cls,
        "A_box": A_box,
        "delta_cls": delta_cls,
        "delta_box": delta_box,
        "risk_cls_admission": hard_cls,
        "risk_box_admission": hard_box,
        "train_cls_mask": train_cls_mask,
        "train_box_mask": train_box_mask,
    }
