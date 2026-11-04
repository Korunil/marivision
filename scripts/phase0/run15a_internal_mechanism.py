from __future__ import annotations
"""RUN_15A internal v2 mechanism, derived from certified RUN09B function.
Changes: capability-aware A, straight-through hard-forward during training,
identical hard inference path. This does NOT mutate original model/weights.
"""
import math
import torch
EXPERTS=("S","L","C","T")
LEGACY_SHA256="d51399ed4eaea068dbcfd0a330ad87b01fe277ce907df9e3121af460a514976e"

def mechanism_forward(model, O, X, bundle, hard: bool):
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

    hard_cls = torch.cat([x["mask_hard"] for x in cls_gate_out], dim=-1) * A_cls
    hard_box = torch.cat([x["mask_hard"] for x in box_gate_out], dim=-1) * A_box
    # Straight-through forward mask is exact hard mask; backward is differentiable.
    train_cls_mask = torch.cat([x["mask_ste"] for x in cls_gate_out], dim=-1) * A_cls
    train_box_mask = torch.cat([x["mask_ste"] for x in box_gate_out], dim=-1) * A_box

    if hard:
        g_cls = torch.cat([x["g_hard"] for x in cls_gate_out], dim=-1) * A_cls
        g_box = torch.cat([x["g_hard"] for x in box_gate_out], dim=-1) * A_box
    else:
        g_cls = torch.cat([x["g_ste"] for x in cls_gate_out], dim=-1) * A_cls
        g_box = torch.cat([x["g_ste"] for x in box_gate_out], dim=-1) * A_box

    coord_in_cls = torch.cat(gate_cls_features + [g_cls, hard_cls if hard else train_cls_mask, A_cls], dim=-1)
    coord_in_box = torch.cat(gate_box_features + [g_box, hard_box if hard else train_box_mask, A_box], dim=-1)

    coord_cls = model.coord_cls(coord_in_cls, delta_cls)
    coord_box = model.coord_box(coord_in_box, delta_box)

    pi_cls = model.router(g_cls, A_cls, coord_cls)
    pi_box = model.router(g_box, A_box, coord_box)

    h_cls = O.h + torch.sum(pi_cls[:, 1:].unsqueeze(-1) * delta_cls, dim=1)
    h_box = O.h + torch.sum(pi_box[:, 1:].unsqueeze(-1) * delta_box, dim=1)

    z = O.z.clone()
    b = O.b.clone()

    if hard:
        active_cls = hard_cls.sum(dim=-1) > 0
        active_box = hard_box.sum(dim=-1) > 0
    else:
        active_cls = A_cls.sum(dim=-1) > 0
        active_box = A_box.sum(dim=-1) > 0

    if bool(active_cls.any()):
        z[active_cls] = model.adapter.decode_cls(h_cls[active_cls])
    if bool(active_box.any()):
        b[active_box] = model.adapter.decode_box(h_box[active_box], O.r[active_box])

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
        "train_cls_mask": train_cls_mask,
        "train_box_mask": train_box_mask,
    }
