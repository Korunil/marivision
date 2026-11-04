"""
D:/MARIVISION/H0_workbench/eval_tools/run_bfcef_e0_smoke.py
MARSEN B-FCEF: Stage E0 - Pre-Training Micro-Smoke Verification Suite
(D-4478..D-4491, P-2298..P-2303)

Verification Suite:
  1. Module Resolution & Provenance: Resolves Deformable-DETR repo and verifies R0 checkpoint.
  2. Deterministic Frame Selection: Dynamically finds the first Train frame satisfying
     N_GT > 0 and N_canonical_TP > 0.
  3. Live Feature Tap: Runs forward pass to capture authentic q_native, reference_points,
     and multiscale pyramid (P3, P4, P5) using certified decoder-level indexing.
  4. Explicit R0 Freezing: Enforces requires_grad_(False) on all R0 parameters.
  5. Normalized GT Target Grounding: Uses transformed normalized cxcywh boxes from SeaClipsDetection.
  6. Test 1 (Zero-Start Identity): Asserts ||delta_q|| == 0.000000 exact machine zero;
     reconstruction parity within certified 1e-5 tolerance.
  7. Test 2 (Two-Stage Zero-Start Gradient Flow):
     - Step 1 (weight_decay=0.0): Asserts projector W2 gradient > 0, upstream encoder gradient == 0.
     - Step 2: Asserts upstream encoder gradient becomes finite and > 0 once W2 opens.
  8. Test 3 (Backbone Freezing): Asserts all R0 heads and backbone tensors have grad is None.
  9. Test 4 (Mini-Optimization): 8 steps at lr=1e-4 verifying loss reduction and ||delta_q|| > 0.
  10. Test 5 (Exact Parameter Reset): Verifies parameter tensors modified by training restore
      bitwise equal to pristine_zero_state (zero inter-fold leakage).
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
from pathlib import Path

ROOT_DIR = Path("D:/MARIVISION")
OUT_DIR = Path("D:/MARIVISION/H0_workbench/eval_tools")
SHARD_DIR = OUT_DIR / "full_split_shards"

CENSUS_JSON = OUT_DIR / "marsen_b2_production_census_56shards.json"
TRAIN_JSON = ROOT_DIR / "seaclips-train.json"
ZIP_PATH = ROOT_DIR / "SeaClips.zip"
R0_CHECKPOINT_PATH = ROOT_DIR / "checkpoints_r0" / "checkpoint_best.pth"
FROZEN_R0_CKPT_SHA256 = "28e6288c3f6769c2d5275aa77ddb248735b55a09cd46a6b570e3f117890b268b"

REPO_CANDIDATES = [ROOT_DIR / "Deformable-DETR", ROOT_DIR / "deformable_detr"]
DEFORMABLE_DETR_ROOT = None
for candidate in REPO_CANDIDATES:
    if (
        candidate.exists()
        and (candidate / "main.py").exists()
        and (candidate / "models" / "deformable_detr.py").exists()
        and (candidate / "datasets" / "seaclips_detection.py").exists()
    ):
        DEFORMABLE_DETR_ROOT = candidate
        break

assert DEFORMABLE_DETR_ROOT is not None, "[FATAL] Cannot locate authoritative Deformable-DETR repository."

for p in reversed([OUT_DIR, DEFORMABLE_DETR_ROOT, ROOT_DIR]):
    p_str = str(p)
    if p_str in sys.path:
        sys.path.remove(p_str)
    sys.path.insert(0, p_str)

import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from run_bfcef_e0_substrate import (
    BFCEFSubstrate,
    AuthenticDeformableDETRAdapter,
    CanonicalObjectState,
    CanonicalMultiscaleFeatures,
    load_authoritative_r0_args,
    compute_sha256,
    CANONICAL_EMB_DIM,
    NUM_CLASSES
)

sys.stdout.reconfigure(line_buffering=True)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def box_iou_norm_cxcywh(b1, b2):
    """Pairwise IoU between two normalized cxcywh boxes in [0, 1]."""
    b1_x1 = b1[0] - b1[2] / 2.0
    b1_y1 = b1[1] - b1[3] / 2.0
    b1_x2 = b1[0] + b1[2] / 2.0
    b1_y2 = b1[1] + b1[3] / 2.0

    b2_x1 = b2[0] - b2[2] / 2.0
    b2_y1 = b2[1] - b2[3] / 2.0
    b2_x2 = b2[0] + b2[2] / 2.0
    b2_y2 = b2[1] + b2[3] / 2.0

    ix = max(0.0, min(b1_x2, b2_x2) - max(b1_x1, b2_x1))
    iy = max(0.0, min(b1_y2, b2_y2) - max(b1_y1, b2_y1))
    inter = ix * iy
    union = b1[2] * b1[3] + b2[2] * b2[3] - inter
    return inter / union if union > 0.0 else 0.0


def compute_canonical_baseline_tp_mask(
    eval_logits: torch.Tensor,
    eval_boxes: torch.Tensor,
    gt_boxes_norm: torch.Tensor,
    gt_labels: torch.Tensor
) -> torch.Tensor:
    """Computes exact canonical baseline True Positive query mask."""
    num_q = eval_boxes.shape[0]
    probs = eval_logits.sigmoid()
    flat_probs = probs.flatten()
    topk_scores, topk_indices = torch.topk(flat_probs, min(100, flat_probs.numel()))

    q_indices = (topk_indices // NUM_CLASSES).tolist()
    c_indices = (topk_indices % NUM_CLASSES).tolist()
    scores = topk_scores.tolist()

    matched_gt_indices = set()
    canonical_tp_mask = torch.zeros(num_q, dtype=torch.bool, device=eval_boxes.device)

    num_gt = gt_boxes_norm.shape[0]
    if num_gt == 0:
        return canonical_tp_mask

    gt_boxes_list = gt_boxes_norm.tolist()
    gt_labels_list = gt_labels.tolist()

    for sc, q_idx, c_idx in zip(scores, q_indices, c_indices):
        if sc < 0.20:
            continue

        det_box = eval_boxes[q_idx].tolist()
        best_iou = -1.0
        best_gt_idx = None

        for g_idx in range(num_gt):
            if g_idx in matched_gt_indices:
                continue
            if gt_labels_list[g_idx] != c_idx:
                continue
            iou = box_iou_norm_cxcywh(det_box, gt_boxes_list[g_idx])
            if iou >= 0.30 and iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx

        if best_gt_idx is not None:
            matched_gt_indices.add(best_gt_idx)
            canonical_tp_mask[q_idx] = True

    return canonical_tp_mask


def main():
    t_start = time.time()
    driver_sha = compute_sha256(Path(__file__))

    print("=" * 105)
    print("MARSEN B-FCEF: STAGE E0 - PRE-TRAINING MICRO-SMOKE VERIFICATION SUITE")
    print(f"Driver Script SHA-256: {driver_sha}")
    print(f"Target Device:         {DEVICE}")
    print("Verification Gate:     Zero-Start Identity | Two-Stage Gradients | R0 Freezing | Pristine Reset")
    print("=" * 105, flush=True)

    # 1. Checkpoint & Provenance Verification
    assert compute_sha256(R0_CHECKPOINT_PATH) == FROZEN_R0_CKPT_SHA256, "[FATAL] Checkpoint SHA mismatch!"
    print("[*] Verified certified R0 checkpoint SHA-256.")

    # 2. Build Authentic R0 Model & Official Criterion
    args = load_authoritative_r0_args()
    args.device = str(DEVICE)
    from models.deformable_detr import build as build_deformable_detr

    r0_model, official_criterion, _ = build_deformable_detr(args)
    r0_ckpt = torch.load(R0_CHECKPOINT_PATH, map_location="cpu")
    sd = r0_ckpt.get("model", r0_ckpt)
    cleaned_sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    r0_model.load_state_dict(cleaned_sd, strict=True)
    r0_model.to(DEVICE)
    r0_model.eval()

    # Explicit global R0 parameter freeze (D-4490)
    for p in r0_model.parameters():
        p.requires_grad_(False)
    assert not any(p.requires_grad for p in r0_model.parameters()), "[FATAL] One or more R0 parameters remain trainable!"
    print("[*] Formally froze all R0 parameters (requires_grad=False).")

    official_criterion.to(DEVICE)

    # 3. Bind Sovereign Adapter & Instantiate Substrate
    adapter = AuthenticDeformableDETRAdapter(r0_model.class_embed[-1], r0_model.bbox_embed[-1]).to(DEVICE)
    substrate = BFCEFSubstrate(adapter=adapter).to(DEVICE)

    # 4. Ingest Dataset & Find Deterministic Frame with GT > 0 and Canonical TP > 0 (D-4488)
    from datasets.seaclips_detection import SeaClipsDetection
    from util.misc import collate_fn

    dataset = SeaClipsDetection(ZIP_PATH, TRAIN_JSON, is_train=False, random_flip=False)
    print(f"[*] Ingested dataset ({len(dataset)} items). Searching for deterministic frame with GT > 0 and TP > 0...")

    chosen_idx = None
    chosen_target = None
    chosen_samples = None
    chosen_out_live = None
    chosen_canonical_tp_mask = None
    hooked_taps_chosen = None

    for idx in range(len(dataset)):
        item = dataset[idx]
        image_tensor, target = item
        if len(target["boxes"]) == 0:
            continue

        samples, _ = collate_fn([item])
        samples = samples.to(DEVICE)

        hooked_taps = {}
        def tap_hook(module, inp, outp):
            srcs = inp[0]
            hooked_taps["p3"] = srcs[0].detach()
            hooked_taps["p4"] = srcs[1].detach()
            hooked_taps["p5"] = srcs[2].detach()
            hs = outp[0]
            init_ref = outp[1]
            inter_refs = outp[2]
            last_lvl = hs.shape[0] - 1
            hooked_taps["q_native"] = hs[last_lvl].detach()
            if last_lvl == 0:
                hooked_taps["ref_pts"] = init_ref.detach()
            else:
                hooked_taps["ref_pts"] = inter_refs[last_lvl - 1].detach()

        handle = r0_model.transformer.register_forward_hook(tap_hook)
        with torch.no_grad():
            out_live = r0_model(samples)
        handle.remove()

        d_logits = out_live["pred_logits"][0].detach()
        d_boxes = out_live["pred_boxes"][0].detach()

        gt_boxes_norm = target["boxes"].to(DEVICE)
        gt_labels = target["labels"].to(DEVICE)

        tp_mask = compute_canonical_baseline_tp_mask(
            d_logits, d_boxes, gt_boxes_norm, gt_labels
        )

        if tp_mask.sum().item() > 0:
            chosen_idx = idx
            chosen_target = target
            chosen_samples = samples
            chosen_out_live = out_live
            chosen_canonical_tp_mask = tp_mask
            hooked_taps_chosen = hooked_taps
            break

    assert chosen_idx is not None, "[FATAL] Could not find any frame with GT > 0 and canonical TP > 0 in dataset!"

    img_id = chosen_target["image_id"]
    if isinstance(img_id, torch.Tensor):
        img_id = int(img_id.item())
    else:
        img_id = int(img_id)

    print(f"[+] Selected deterministic frame index {chosen_idx} (Image ID {img_id}): {len(chosen_target['boxes'])} GT boxes, {chosen_canonical_tp_mask.sum().item()} canonical baseline TPs.")

    # 5. Build Authentic Tapped Frame Record
    ms_feats = CanonicalMultiscaleFeatures(
        p3=hooked_taps_chosen["p3"],
        p4=hooked_taps_chosen["p4"],
        p5=hooked_taps_chosen["p5"]
    )
    tapped_fr = {
        "image_id": img_id,
        "img_w": 640.0,
        "img_h": 640.0,
        "d_logits": chosen_out_live["pred_logits"][0].detach(),
        "d_boxes": chosen_out_live["pred_boxes"][0].detach(),
        "d_queries": hooked_taps_chosen["q_native"][0],
        "d_reference_points": hooked_taps_chosen["ref_pts"][0],
        "multiscale_features": ms_feats,
        "temporal_history": None
    }

    gt_boxes_norm = chosen_target["boxes"].to(DEVICE)
    gt_labels = chosen_target["labels"].to(DEVICE)
    canonical_tp_mask = chosen_canonical_tp_mask
    d_logits = tapped_fr["d_logits"]
    d_boxes = tapped_fr["d_boxes"]

    # 6. Initialize Zero Projectors & Snapshot Pristine Zero-Start State (D-4487, P-2303)
    substrate.init_zero_projectors()
    pristine_zero_state = copy.deepcopy({
        k: v.detach().clone() for k, v in substrate.state_dict().items()
        if not k.startswith("adapter.")
    })

    # -------------------------------------------------------------------------
    # TEST 1: Zero-Start Identity Verification
    # -------------------------------------------------------------------------
    print("\n[*] [Test 1] Verifying Zero-Start Identity Invariant (D_{E0, initial} == D_B)...")
    with torch.no_grad():
        state_init = substrate.adapter.extract_canonical_state(tapped_fr)
        e_l, a_l, _, _ = substrate.expert_l(state_init)
        delta_q_box_0 = a_l.unsqueeze(-1) * substrate.expert_l.proj_box(e_l)
        delta_q_cls_0 = a_l.unsqueeze(-1) * substrate.expert_l.proj_cls(e_l)

        z_mod_0, b_mod_0, _, _ = substrate(
            tapped_fr, active_expert_mask=[False, True, False, False]
        )

    max_dq_box_0 = float(delta_q_box_0.abs().max().item())
    max_dq_cls_0 = float(delta_q_cls_0.abs().max().item())
    max_diff_z_0 = float(torch.abs(z_mod_0 - d_logits).max().item())
    max_diff_b_0 = float(torch.abs(b_mod_0 - d_boxes).max().item())

    assert max_dq_box_0 == 0.0, f"[FATAL] Initial delta_q_box non-zero: {max_dq_box_0}"
    assert max_dq_cls_0 == 0.0, f"[FATAL] Initial delta_q_cls non-zero: {max_dq_cls_0}"
    assert max_diff_z_0 < 1e-5, f"[FATAL] Initial native-head logit reconstruction drift: {max_diff_z_0:.2e}"
    assert max_diff_b_0 < 1e-5, f"[FATAL] Initial native-head box reconstruction drift: {max_diff_b_0:.2e}"
    print(f"    [+] PASS: ||delta_q|| == 0.000000 exact machine zero; baseline predictions preserved within certified native-head reconstruction tolerance.")

    # -------------------------------------------------------------------------
    # TEST 2: Two-Stage Zero-Start Gradient Flow (D-4486, P-2302)
    # -------------------------------------------------------------------------
    print("\n[*] [Test 2] Verifying Two-Stage Zero-Start Gradient Flow...")
    exp_l = substrate.expert_l
    optimizer = torch.optim.AdamW(exp_l.parameters(), lr=1e-4, weight_decay=0.0)

    targets = [{
        "boxes": gt_boxes_norm,
        "labels": gt_labels
    }]

    # Stage 1: First backward pass (W2 == 0)
    optimizer.zero_grad()
    z_mod_1, b_mod_1, _, _ = substrate(tapped_fr, active_expert_mask=[False, True, False, False])
    out_dict_1 = {"pred_logits": z_mod_1.unsqueeze(0), "pred_boxes": b_mod_1.unsqueeze(0)}
    loss_dict_1 = official_criterion(out_dict_1, targets)
    loss_det_1 = sum(loss_dict_1[k] * official_criterion.weight_dict[k] for k in loss_dict_1 if k in official_criterion.weight_dict)

    loss_prot_1 = torch.tensor(0.0, device=DEVICE)
    if canonical_tp_mask.any():
        loss_prot_1 = F.l1_loss(b_mod_1[canonical_tp_mask], d_boxes[canonical_tp_mask]) + \
                      F.mse_loss(z_mod_1[canonical_tp_mask], d_logits[canonical_tp_mask])

    state_act_1 = substrate.adapter.extract_canonical_state(tapped_fr)
    e_l_act_1, a_l_act_1, _, _ = exp_l(state_act_1)
    dq_b_1 = a_l_act_1.unsqueeze(-1) * exp_l.proj_box(e_l_act_1)
    dq_c_1 = a_l_act_1.unsqueeze(-1) * exp_l.proj_cls(e_l_act_1)
    loss_reg_1 = (dq_b_1 ** 2).mean() + (dq_c_1 ** 2).mean()

    total_loss_step1 = loss_det_1 + 2.0 * loss_prot_1 + 0.1 * loss_reg_1
    total_loss_step1.backward()

    # Verify Stage 1 gradients: W2 receives gradient, upstream encoder gradient == 0
    proj_box_grad = exp_l.proj_box.net[-1].weight.grad
    assert proj_box_grad is not None and torch.isfinite(proj_box_grad).all(), "[FATAL] Projector W2 gradient missing or non-finite!"
    assert proj_box_grad.abs().sum().item() > 0.0, "[FATAL] Projector W2 received zero gradient on step 1!"

    enc_weight_grad_1 = exp_l.encoder.net[0].weight.grad
    assert enc_weight_grad_1 is None or enc_weight_grad_1.abs().sum().item() == 0.0, (
        f"[FATAL] Upstream encoder gradient non-zero on step 1 before W2 opened! "
        f"Sum: {enc_weight_grad_1.abs().sum().item() if enc_weight_grad_1 is not None else 0.0}"
    )
    print("    [+] PASS (Stage 1): Projector W2 gradient > 0; upstream encoder gradient == 0 as required by W2 == 0.")

    # -------------------------------------------------------------------------
    # TEST 3: Backbone & Prediction Head Freezing
    # -------------------------------------------------------------------------
    print("\n[*] [Test 3] Verifying R0 Backbone & Native Prediction Head Freezing...")
    assert r0_model.class_embed[-1].weight.grad is None, "[FATAL] Baseline class_embed accumulated gradients!"
    assert r0_model.bbox_embed[-1].layers[0].weight.grad is None, "[FATAL] Baseline bbox_embed accumulated gradients!"
    for p in r0_model.backbone.parameters():
        assert p.grad is None, "[FATAL] Baseline backbone accumulated gradients!"
    print("    [+] PASS: R0 backbone and native prediction heads strictly frozen (grad is None).")

    # Take optimization step 1 to open W2
    optimizer.step()

    # Stage 2: Second forward/backward pass (W2 != 0)
    optimizer.zero_grad()
    z_mod_2, b_mod_2, _, _ = substrate(tapped_fr, active_expert_mask=[False, True, False, False])
    out_dict_2 = {"pred_logits": z_mod_2.unsqueeze(0), "pred_boxes": b_mod_2.unsqueeze(0)}
    loss_dict_2 = official_criterion(out_dict_2, targets)
    loss_det_2 = sum(loss_dict_2[k] * official_criterion.weight_dict[k] for k in loss_dict_2 if k in official_criterion.weight_dict)

    loss_prot_2 = torch.tensor(0.0, device=DEVICE)
    if canonical_tp_mask.any():
        loss_prot_2 = F.l1_loss(b_mod_2[canonical_tp_mask], d_boxes[canonical_tp_mask]) + \
                      F.mse_loss(z_mod_2[canonical_tp_mask], d_logits[canonical_tp_mask])

    state_act_2 = substrate.adapter.extract_canonical_state(tapped_fr)
    e_l_act_2, a_l_act_2, _, _ = exp_l(state_act_2)
    dq_b_2 = a_l_act_2.unsqueeze(-1) * exp_l.proj_box(e_l_act_2)
    dq_c_2 = a_l_act_2.unsqueeze(-1) * exp_l.proj_cls(e_l_act_2)
    loss_reg_2 = (dq_b_2 ** 2).mean() + (dq_c_2 ** 2).mean()

    total_loss_step2 = loss_det_2 + 2.0 * loss_prot_2 + 0.1 * loss_reg_2
    total_loss_step2.backward()

    enc_weight_grad_2 = exp_l.encoder.net[0].weight.grad
    assert enc_weight_grad_2 is not None and torch.isfinite(enc_weight_grad_2).all(), "[FATAL] Encoder gradient missing on step 2!"
    assert enc_weight_grad_2.abs().sum().item() > 0.0, "[FATAL] Encoder gradient is zero on step 2 after W2 opened!"
    print(f"    [+] PASS (Stage 2): Projector opened -> upstream encoder receives finite non-zero gradient ({enc_weight_grad_2.abs().sum().item():.3e}).")

    # -------------------------------------------------------------------------
    # TEST 4: Mini-Optimization & Loss Reduction (CHL-2139)
    # -------------------------------------------------------------------------
    print("\n[*] [Test 4] Executing 8-Step Mini-Optimization & Loss Reduction...")
    loss_history = [float(total_loss_step1.item()), float(total_loss_step2.item())]
    optimizer.step()

    for step in range(3, 9):
        optimizer.zero_grad()
        z_m, b_m, _, _ = substrate(tapped_fr, active_expert_mask=[False, True, False, False])
        out_d = {"pred_logits": z_m.unsqueeze(0), "pred_boxes": b_m.unsqueeze(0)}
        l_dict = official_criterion(out_d, targets)
        l_det = sum(l_dict[k] * official_criterion.weight_dict[k] for k in l_dict if k in official_criterion.weight_dict)

        l_prot = torch.tensor(0.0, device=DEVICE)
        if canonical_tp_mask.any():
            l_prot = F.l1_loss(b_m[canonical_tp_mask], d_boxes[canonical_tp_mask]) + \
                     F.mse_loss(z_m[canonical_tp_mask], d_logits[canonical_tp_mask])

        st = substrate.adapter.extract_canonical_state(tapped_fr)
        el, al, _, _ = exp_l(st)
        dq_b_step = al.unsqueeze(-1) * exp_l.proj_box(el)
        dq_c_step = al.unsqueeze(-1) * exp_l.proj_cls(el)
        l_reg = (dq_b_step ** 2).mean() + (dq_c_step ** 2).mean()

        tot_l = l_det + 2.0 * l_prot + 0.1 * l_reg
        tot_l.backward()
        optimizer.step()
        loss_history.append(float(tot_l.item()))

    assert all(np.isfinite(loss_history)), f"[FATAL] Non-finite loss in history: {loss_history}"
    assert min(loss_history[1:]) < loss_history[0], f"[FATAL] Optimization failed to reduce loss: {loss_history}"

    with torch.no_grad():
        st_post = substrate.adapter.extract_canonical_state(tapped_fr)
        el_post, al_post, _, _ = exp_l(st_post)
        dq_b_post = al_post.unsqueeze(-1) * exp_l.proj_box(el_post)
        norm_dq = float(torch.norm(dq_b_post).item())

    assert norm_dq > 0.0, "[FATAL] Post-optimization ||delta_q|| is zero!"
    print(f"    [+] PASS: 8-step mini-optimization converged (Loss: {loss_history[0]:.4f} -> {min(loss_history):.4f}); Active ||delta_q|| = {norm_dq:.4f} > 0.")

    # -------------------------------------------------------------------------
    # TEST 5: Exact Parameter Reset Equality (D-4487, P-2303)
    # -------------------------------------------------------------------------
    print("\n[*] [Test 5] Verifying Cross-Fold Exact Parameter Reset (P-2303, Zero Leakage)...")
    trained_encoder_w = substrate.expert_l.encoder.net[0].weight
    ref_encoder_w = pristine_zero_state["expert_l.encoder.net.0.weight"]
    assert not torch.equal(trained_encoder_w, ref_encoder_w), "[FATAL] Expert encoder weight was not modified by training!"

    substrate.load_state_dict(pristine_zero_state, strict=False)
    current_state = substrate.state_dict()

    for k, v_ref in pristine_zero_state.items():
        assert torch.equal(current_state[k], v_ref), f"[FATAL] Parameter reset mismatch on {k}!"

    with torch.no_grad():
        st_reset = substrate.adapter.extract_canonical_state(tapped_fr)
        el_r, al_r, _, _ = substrate.expert_l(st_reset)
        dq_b_reset = al_r.unsqueeze(-1) * substrate.expert_l.proj_box(el_r)
        max_dq_reset = float(dq_b_reset.abs().max().item())

    assert max_dq_reset == 0.0, f"[FATAL] Post-reset ||delta_q|| non-zero: {max_dq_reset}"
    print("    [+] PASS: All expert parameter tensors bitwise equal to pristine zero-start snapshot (||delta_q|| == 0.000000).")

    print("\n" + "=" * 105)
    print(f"[+] ALL 5 MICRO-SMOKE TESTS PASSED IN {time.time() - t_start:.2f}s")
    print("[+] Optimization and isolation verified: Substrate ready for full Stage E0 cross-fitted training.")
    print("=" * 105 + "\n", flush=True)


if __name__ == "__main__":
    main()