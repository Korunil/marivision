from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# PATH RESOLUTION
# ---------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent

# During manual staging:
#
# D:\MARIVISION\H0_workbench\eval_tools\
#     _BFCEF_F_CHATGPT_STAGING\
#
# Once promoted, THIS_DIR itself will be eval_tools.
if THIS_DIR.name == "_BFCEF_F_CHATGPT_STAGING":
    EVAL_TOOLS = THIS_DIR.parent
else:
    EVAL_TOOLS = THIS_DIR

ROOT_DIR = Path(
    "D:/MARIVISION"
)

for path_entry in (
    THIS_DIR,
    EVAL_TOOLS,
):
    if str(path_entry) not in sys.path:
        sys.path.insert(
            0,
            str(path_entry),
        )


from bfcef_f.adapters.deformable_detr import (
    DeformableDETRNativeAdapter,
)
from bfcef_f.config import (
    BFCEFFScreeningConfig,
)
from bfcef_f.contracts import (
    CanonicalObjectState,
    FrameExpertInputs,
    TemporalState,
)
from bfcef_f.model import (
    BFCEFFModel,
)
from bfcef_f.spatial_evidence import (
    extract_localization_spatial_evidence,
    extract_scale_spatial_evidence,
)


R0_CHECKPOINT_SHA256 = (
    "28e6288c3f6769c2d5275aa77ddb248735b55a09cd46a6b570e3f117890b268b"
)


def sha256(
    path: Path,
) -> str:
    h = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:
        for chunk in iter(
            lambda: f.read(
                1 << 20
            ),
            b"",
        ):
            h.update(
                chunk
            )

    return h.hexdigest()


def grad_l1(
    parameter: torch.nn.Parameter,
) -> float:
    """
    Absolute gradient mass.

    Used to prove that a particular trainable residual projector receives
    a non-zero training signal.
    """

    if parameter.grad is None:
        return 0.0

    return float(
        parameter.grad
        .detach()
        .abs()
        .sum()
        .item()
    )


class TestBFCEFFACProd(
    unittest.TestCase
):
    """
    B-FCEF-F production architecture conformance.

    IMPORTANT TRAIN/EVAL CONTRACT
    -----------------------------

    Evaluation / inference:
        Uses exact hard intervention masks.

        If all expert/task intervention masks are closed, BFCEFFModel must
        return the literal cached baseline logits and boxes.

    Training:
        Must NOT enter the inference hard-bypass branch merely because the
        hard masks are initially closed.

        Training instead uses differentiable soft gate amplitudes so that
        gradients can reach the zero-initialized expert residual projectors.

    Frozen R0/native heads:
        May transmit gradients through their mathematical operations to
        expert residuals, but their own parameters must remain
        requires_grad=False and must never update.
    """

    def setUp(
        self,
    ):
        torch.manual_seed(
            4688
        )

        self.Q = 300
        self.D = 256
        self.C = 7

        self.config = (
            BFCEFFScreeningConfig()
        )

        self.K = (
            self.config.temporal_memory_K
        )

        # --------------------------------------------------------------
        # Synthetic native baseline.
        #
        # This is used only for architecture, gradient and routing tests.
        # The real R0 checkpoint/cache parity is tested separately.
        # --------------------------------------------------------------

        self.native_cls = nn.Linear(
            self.D,
            self.C,
        )

        self.native_box = nn.Sequential(
            nn.Linear(
                self.D,
                self.D,
            ),
            nn.ReLU(),
            nn.Linear(
                self.D,
                self.D,
            ),
            nn.ReLU(),
            nn.Linear(
                self.D,
                4,
            ),
        )

        h = torch.randn(
            self.Q,
            self.D,
        )

        r = torch.rand(
            self.Q,
            2,
        ).clamp(
            1e-3,
            1.0 - 1e-3,
        )

        native_adapter = (
            DeformableDETRNativeAdapter(
                self.native_cls,
                self.native_box,
            )
        )

        with torch.no_grad():
            z = (
                native_adapter
                .decode_cls(
                    h
                )
                .detach()
            )

            b = (
                native_adapter
                .decode_box(
                    h,
                    r,
                )
                .detach()
            )

        s = (
            z.sigmoid()
            .max(
                dim=-1
            )
            .values
        )

        f_maps = {
            "P3": torch.randn(
                1,
                256,
                20,
                20,
            ),
            "P4": torch.randn(
                1,
                256,
                10,
                10,
            ),
            "P5": torch.randn(
                1,
                256,
                5,
                5,
            ),
        }

        capabilities = {
            "base": True
        }

        for name in (
            "scale",
            "localization",
            "semantic",
            "temporal",
        ):
            capabilities[name] = True

            capabilities[
                f"{name}.cls"
            ] = True

            capabilities[
                f"{name}.box"
            ] = True

        self.O = CanonicalObjectState(
            h=h,
            z=z,
            b=b,
            r=r,
            s=s,
            F_maps=f_maps,
            native_handle={
                "synthetic": True
            },
            capabilities=capabilities,
        )

        scale_spatial = (
            extract_scale_spatial_evidence(
                b,
                f_maps,
                ring_expansion=(
                    self.config
                    .scale_ring_expansion
                ),
            )
        )

        loc_spatial = (
            extract_localization_spatial_evidence(
                b,
                f_maps,
                side_samples=(
                    self.config
                    .localization_side_samples
                ),
                offset_frac=(
                    self.config
                    .localization_boundary_offset
                ),
            )
        )

        temporal = TemporalState(
            legacy_cues=torch.zeros(
                self.Q,
                8,
            ),

            disp_history=torch.zeros(
                self.Q,
                self.K,
                2,
            ),

            score_persistence=torch.zeros(
                self.Q,
                self.K,
            ),

            class_consistency=torch.zeros(
                self.Q,
                self.K,
            ),

            appearance_similarity=torch.zeros(
                self.Q,
                self.K,
            ),

            association_confidence=torch.zeros(
                self.Q,
                self.K,
            ),

            same_slot=torch.zeros(
                self.Q,
                self.K,
            ),

            valid=torch.zeros(
                self.Q,
                self.K,
            ),

            track_age=torch.zeros(
                self.Q,
                1,
            ),

            velocity=torch.zeros(
                self.Q,
                2,
            ),

            acceleration=torch.zeros(
                self.Q,
                2,
            ),

            global_motion=torch.zeros(
                self.Q,
                2,
            ),
        )

        raw = {
            expert: {
                "N": torch.rand(
                    self.Q,
                    1,
                ),
                "R": torch.rand(
                    self.Q,
                    1,
                ),
            }
            for expert in (
                "S",
                "L",
                "C",
                "T",
            )
        }

        self.X = FrameExpertInputs(
            cues_s=torch.randn(
                self.Q,
                10,
            ),

            cues_l=torch.randn(
                self.Q,
                10,
            ),

            cues_c=torch.randn(
                self.Q,
                6,
            ),

            scale_spatial=(
                scale_spatial
            ),

            loc_spatial=(
                loc_spatial
            ),

            temporal=(
                temporal
            ),

            raw_proxies=(
                raw
            ),

            scene_q=torch.rand(
                1,
                5,
            ),
        )

        self.model = BFCEFFModel(
            native_class_embed=(
                self.native_cls
            ),

            native_bbox_embed=(
                self.native_box
            ),

            candidate_topk=100,

            config=(
                self.config
            ),

            num_classes=(
                self.C
            ),
        )

    # ================================================================
    # A. REAL DATA CONTRACTS
    # ================================================================

    def test_01_real_legacy_cue_schema(
        self,
    ):
        self.assertEqual(
            self.X.cues_s.shape[-1],
            10,
        )

        self.assertEqual(
            self.X.cues_l.shape[-1],
            10,
        )

        self.assertEqual(
            self.X.cues_c.shape[-1],
            6,
        )

        self.assertEqual(
            self.X.temporal
            .legacy_cues
            .shape[-1],
            8,
        )

    def test_02_spatial_descriptor_dimensions(
        self,
    ):
        self.assertEqual(
            self.X
            .scale_spatial
            .shape,
            (
                self.Q,
                21,
            ),
        )

        self.assertEqual(
            self.X
            .loc_spatial
            .shape,
            (
                self.Q,
                24,
            ),
        )

    # ================================================================
    # B. SPECIALIST EVIDENCE EXISTENCE
    # ================================================================

    def test_03_scale_spatial_responds_to_each_pyramid_level(
        self,
    ):
        base = (
            extract_scale_spatial_evidence(
                self.O.b,
                self.O.F_maps,
                ring_expansion=(
                    self.config
                    .scale_ring_expansion
                ),
            )
        )

        for level in (
            "P3",
            "P4",
            "P5",
        ):
            perturbed = {
                key: value.clone()
                for key, value
                in self.O.F_maps.items()
            }

            perturbed[level] = (
                perturbed[level]
                + 0.25
                * torch.randn_like(
                    perturbed[level]
                )
            )

            changed = (
                extract_scale_spatial_evidence(
                    self.O.b,
                    perturbed,
                    ring_expansion=(
                        self.config
                        .scale_ring_expansion
                    ),
                )
            )

            self.assertFalse(
                torch.equal(
                    base,
                    changed,
                ),
                f"Scale evidence ignored {level}.",
            )

    def test_04_localization_spatial_responds_to_p3_and_p4(
        self,
    ):
        base = (
            extract_localization_spatial_evidence(
                self.O.b,
                self.O.F_maps,
                side_samples=(
                    self.config
                    .localization_side_samples
                ),
                offset_frac=(
                    self.config
                    .localization_boundary_offset
                ),
            )
        )

        for level in (
            "P3",
            "P4",
        ):
            perturbed = {
                key: value.clone()
                for key, value
                in self.O.F_maps.items()
            }

            perturbed[level] = (
                perturbed[level]
                + 0.25
                * torch.randn_like(
                    perturbed[level]
                )
            )

            changed = (
                extract_localization_spatial_evidence(
                    self.O.b,
                    perturbed,
                    side_samples=(
                        self.config
                        .localization_side_samples
                    ),
                    offset_frac=(
                        self.config
                        .localization_boundary_offset
                    ),
                )
            )

            self.assertFalse(
                torch.equal(
                    base,
                    changed,
                ),
                (
                    "Localization evidence "
                    f"ignored {level}."
                ),
            )

    def test_05_temporal_k_step_state_influences_t_expert(
        self,
    ):
        self.model.eval()

        C_t = (
            self.model
            .context_bus(
                None,
                None,
                self.X.scene_q,
                torch.zeros(3),
                "unknown",
            )
        )

        _, _, base_feat = (
            self.model
            .expert_t(
                self.O.h,
                self.X.temporal,
                C_t,
            )
        )

        t2 = TemporalState(
            legacy_cues=(
                self.X.temporal
                .legacy_cues
                .clone()
            ),

            disp_history=(
                self.X.temporal
                .disp_history
                .clone()
            ),

            score_persistence=(
                self.X.temporal
                .score_persistence
                .clone()
            ),

            class_consistency=(
                self.X.temporal
                .class_consistency
                .clone()
            ),

            appearance_similarity=(
                self.X.temporal
                .appearance_similarity
                .clone()
            ),

            association_confidence=(
                self.X.temporal
                .association_confidence
                .clone()
            ),

            same_slot=(
                self.X.temporal
                .same_slot
                .clone()
            ),

            valid=(
                self.X.temporal
                .valid
                .clone()
            ),

            track_age=(
                self.X.temporal
                .track_age
                .clone()
            ),

            velocity=(
                self.X.temporal
                .velocity
                .clone()
            ),

            acceleration=(
                self.X.temporal
                .acceleration
                .clone()
            ),

            global_motion=(
                self.X.temporal
                .global_motion
                .clone()
            ),
        )

        # Alter an older causal history entry,
        # not just the most recent step.
        t2.disp_history[
            :,
            -1,
            :,
        ] = 1.0

        t2.association_confidence[
            :,
            -1,
        ] = 1.0

        t2.valid[
            :,
            -1,
        ] = 1.0

        _, _, changed_feat = (
            self.model
            .expert_t(
                self.O.h,
                t2,
                C_t,
            )
        )

        self.assertFalse(
            torch.equal(
                base_feat,
                changed_feat,
            )
        )

    # ================================================================
    # C. CONTEXT + COUNTERFACTUAL SELF-ASSESSMENT
    # ================================================================

    def test_06_context_modes_are_semantically_distinct(
        self,
    ):
        bus = (
            self.model
            .context_bus
            .eval()
        )

        q = torch.rand(
            1,
            5,
        )

        ext = bus(
            {
                "weather": 0,
                "reflections": 1,
                "sea_state": 0,
            },
            None,
            q,
            torch.ones(3),
            "ext",
        )

        vis = bus(
            None,
            None,
            q,
            torch.zeros(3),
            "vis",
        )

        # Architecture-manifest canonical names must be exact aliases.
        ext_canonical = bus(
            {
                "weather": 0,
                "reflections": 1,
                "sea_state": 0,
            },
            None,
            q,
            torch.ones(3),
            "external",
        )

        vis_canonical = bus(
            None,
            None,
            q,
            torch.zeros(3),
            "visual_inference",
        )

        self.assertTrue(torch.equal(ext, ext_canonical))
        self.assertTrue(torch.equal(vis, vis_canonical))

        fake_visual = {
            "weather_logits": (
                torch.randn(
                    1,
                    4,
                )
            ),
            "reflection_logits": (
                torch.randn(
                    1,
                    2,
                )
            ),
            "sea_logits": (
                torch.randn(
                    1,
                    3,
                )
            ),
        }

        unknown_from_ext = bus(
            {
                "weather": 0,
                "reflections": 1,
                "sea_state": 0,
            },
            None,
            q,
            torch.ones(3),
            "unknown",
        )

        unknown_from_visual = bus(
            None,
            fake_visual,
            q,
            torch.ones(3),
            "unknown",
        )

        self.assertFalse(
            torch.equal(
                ext,
                unknown_from_ext,
            )
        )

        self.assertFalse(
            torch.equal(
                vis,
                unknown_from_ext,
            )
        )

        # UNKNOWN must ignore both external
        # and visually inferred metadata.
        self.assertTrue(
            torch.equal(
                unknown_from_ext,
                unknown_from_visual,
            )
        )

    def test_07_raw_need_reliability_change_learned_consequences(
        self,
    ):
        head = (
            self.model
            .self_assessment["C"]
            .eval()
        )

        q = 16

        feat = torch.randn(
            q,
            self.config
            .expert_widths["C"],
        )

        h = torch.randn(
            q,
            self.D,
        )

        norm_cls = torch.rand(
            q,
            1,
        )

        norm_box = torch.rand(
            q,
            1,
        )

        C_t = torch.randn(
            1,
            self.config.d_ctx,
        )

        zeros = torch.zeros(
            q,
            1,
        )

        ones = torch.ones(
            q,
            1,
        )

        _, _, cls0, box0 = head(
            feat,
            h,
            zeros,
            zeros,
            norm_cls,
            norm_box,
            C_t,
        )

        _, _, cls1, box1 = head(
            feat,
            h,
            ones,
            ones,
            norm_cls,
            norm_box,
            C_t,
        )

        self.assertFalse(
            torch.equal(
                cls0["U"],
                cls1["U"],
            )
        )

        self.assertFalse(
            torch.equal(
                cls0["H"],
                cls1["H"],
            )
        )

        self.assertFalse(
            torch.equal(
                box0["U"],
                box1["U"],
            )
        )

        self.assertFalse(
            torch.equal(
                box0["H"],
                box1["H"],
            )
        )

    def test_08_classification_uncertainty_uniform_vs_peaked(
        self,
    ):
        head = (
            self.model
            .self_assessment["S"]
        )

        q = 2

        feat = torch.randn(
            q,
            self.config
            .expert_widths["S"],
        )

        h = torch.randn(
            q,
            self.D,
        )

        n = torch.zeros(
            q,
            1,
        )

        r = torch.zeros(
            q,
            1,
        )

        norm_cls = torch.zeros(
            q,
            1,
        )

        norm_box = torch.zeros(
            q,
            1,
        )

        C_t = torch.randn(
            1,
            self.config.d_ctx,
        )

        last = (
            head.cls_head[-1]
        )

        weight_backup = (
            last.weight
            .detach()
            .clone()
        )

        bias_backup = (
            last.bias
            .detach()
            .clone()
        )

        try:
            # --------------------------------------------
            # Uniform 9-state posterior for ALL classes.
            # --------------------------------------------
            last.weight.data.zero_()
            last.bias.data.zero_()

            _, _, uniform, _ = head(
                feat,
                h,
                n,
                r,
                norm_cls,
                norm_box,
                C_t,
            )

            self.assertTrue(
                torch.allclose(
                    uniform["sigma"],
                    torch.ones_like(
                        uniform["sigma"]
                    ),
                    atol=1e-3,
                    rtol=0.0,
                )
            )

            # --------------------------------------------
            # One highly peaked state for EVERY class.
            # --------------------------------------------
            bias = (
                last.bias.data
                .view(
                    self.C,
                    9,
                )
            )

            bias.fill_(
                -20.0
            )

            bias[
                :,
                0,
            ] = 20.0

            _, _, peaked, _ = head(
                feat,
                h,
                n,
                r,
                norm_cls,
                norm_box,
                C_t,
            )

            self.assertTrue(
                torch.all(
                    peaked["sigma"]
                    < 1e-3
                )
            )

        finally:
            last.weight.data.copy_(
                weight_backup
            )

            last.bias.data.copy_(
                bias_backup
            )

    # ================================================================
    # D. CRITICAL TRAINING-vs-EVALUATION CONTRACT
    # ================================================================

    def test_09_eval_uses_exact_hard_bypass_at_neutral_initialization(
        self,
    ):
        """
        Evaluation/inference must use exact intervention masks.

        At baseline-neutral initialization all masks are closed, therefore
        BFCEFFModel must return the literal cached baseline tensors.
        """

        self.model.eval()

        z, b, info = self.model(
            self.O,
            self.X,
            context_mode="unknown",
        )

        self.assertTrue(
            info["bypassed"]
        )

        # Stronger than numerical equality:
        # the hard global bypass returns the literal cached tensors.
        self.assertIs(
            z,
            self.O.z,
        )

        self.assertIs(
            b,
            self.O.b,
        )

        self.assertTrue(
            torch.equal(
                z,
                self.O.z,
            )
        )

        self.assertTrue(
            torch.equal(
                b,
                self.O.b,
            )
        )

    def test_10_training_uses_soft_route_even_when_hard_masks_are_closed(
        self,
    ):
        """
        Training MUST NOT take the inference hard-bypass branch.

        The exact hard masks remain closed at initialization, but soft gate
        amplitudes and expert route mass must remain non-zero for available
        candidate queries.

        This is what prevents the zero-residual + closed-gate deadlock.
        """

        self.model.train()

        z, b, info = self.model(
            self.O,
            self.X,
            context_mode="unknown",
        )

        # Training path must remain live.
        self.assertFalse(
            info["bypassed"]
        )

        # Deep-supervision contract: retain raw classwise 3x3 posterior
        # predictions and raw box consequence predictions for every expert.
        for expert in ("S", "L", "C", "T"):
            self.assertEqual(
                tuple(info["cls_posteriors"][expert].shape),
                (self.Q, self.C, 9),
            )
            self.assertEqual(
                tuple(info["box_raw"][expert].shape),
                (self.Q, 5),
            )

        # Deployment masks still begin exactly closed.
        self.assertEqual(
            float(
                info["m_cls"]
                .sum()
                .item()
            ),
            0.0,
        )

        self.assertEqual(
            float(
                info["m_box"]
                .sum()
                .item()
            ),
            0.0,
        )

        # Differentiable training amplitudes remain positive.
        self.assertGreater(
            float(
                info["g_cls"]
                .sum()
                .item()
            ),
            0.0,
        )

        self.assertGreater(
            float(
                info["g_box"]
                .sum()
                .item()
            ),
            0.0,
        )

        # Non-baseline router mass exists during training.
        self.assertGreater(
            float(
                info["pi_cls"][
                    :,
                    1:
                ]
                .sum()
                .item()
            ),
            0.0,
        )

        self.assertGreater(
            float(
                info["pi_box"][
                    :,
                    1:
                ]
                .sum()
                .item()
            ),
            0.0,
        )

        # Initial residuals are exactly zero, so the numeric output may
        # still equal Base. The crucial invariant is a live autograd path.
        self.assertTrue(
            z.requires_grad
        )

        self.assertTrue(
            b.requires_grad
        )

    def test_11_gradients_reach_every_expert_residual_projector(
        self,
    ):
        """
        Explicitly prove that every S/L/C/T classification AND box residual
        projector receives a non-zero gradient at neutral initialization.

        At the same time, the frozen native/R0 heads must receive none.
        """

        self.model.train()

        self.model.zero_grad(
            set_to_none=True
        )

        z, b, _ = self.model(
            self.O,
            self.X,
            context_mode="unknown",
        )

        # Deliberately offset targets ensure a non-zero derivative at the
        # exact-zero residual initialization.
        target_z = (
            self.O.z
            + 0.10
        )

        target_b = (
            self.O.b
            + 0.05
        ).clamp(
            0.0,
            1.0,
        )

        loss = (
            (
                z
                - target_z
            )
            .pow(2)
            .mean()
            +
            (
                b
                - target_b
            )
            .pow(2)
            .mean()
        )

        loss.backward()

        experts = {
            "S": self.model.expert_s,
            "L": self.model.expert_l,
            "C": self.model.expert_c,
            "T": self.model.expert_t,
        }

        for (
            name,
            expert,
        ) in experts.items():

            cls_grad = grad_l1(
                expert
                .proj_cls
                .weight
            )

            box_grad = grad_l1(
                expert
                .proj_box
                .weight
            )

            self.assertGreater(
                cls_grad,
                0.0,
                (
                    "No classification "
                    "residual-projector gradient "
                    f"for Expert {name}."
                ),
            )

            self.assertGreater(
                box_grad,
                0.0,
                (
                    "No box "
                    "residual-projector gradient "
                    f"for Expert {name}."
                ),
            )

        # -------------------------------------------------------------
        # Frozen native heads:
        # gradients may pass THROUGH their operations into h*, but no
        # parameter inside the native adapter may itself receive a grad.
        # -------------------------------------------------------------
        for (
            name,
            parameter,
        ) in self.model.adapter.named_parameters():

            self.assertFalse(
                parameter.requires_grad,
                (
                    "Frozen native parameter "
                    "unexpectedly trainable: "
                    f"{name}"
                ),
            )

            self.assertIsNone(
                parameter.grad,
                (
                    "Frozen native parameter "
                    "unexpectedly received gradient: "
                    f"{name}"
                ),
            )

    def test_12_optimizer_updates_experts_but_not_frozen_native_heads(
        self,
    ):
        """
        Strong end-to-end optimization assertion:

        1. Run training forward.
        2. Backpropagate.
        3. Execute AdamW step.
        4. Every expert residual projector must change.
        5. Every frozen native/R0 tensor must remain bitwise identical.
        """

        self.model.train()

        frozen_before = {
            key: value
            .detach()
            .clone()
            for (
                key,
                value,
            ) in (
                self.model
                .adapter
                .state_dict()
                .items()
            )
        }

        expert_before = {
            "S_cls": (
                self.model
                .expert_s
                .proj_cls
                .weight
                .detach()
                .clone()
            ),
            "L_cls": (
                self.model
                .expert_l
                .proj_cls
                .weight
                .detach()
                .clone()
            ),
            "C_cls": (
                self.model
                .expert_c
                .proj_cls
                .weight
                .detach()
                .clone()
            ),
            "T_cls": (
                self.model
                .expert_t
                .proj_cls
                .weight
                .detach()
                .clone()
            ),
            "S_box": (
                self.model
                .expert_s
                .proj_box
                .weight
                .detach()
                .clone()
            ),
            "L_box": (
                self.model
                .expert_l
                .proj_box
                .weight
                .detach()
                .clone()
            ),
            "C_box": (
                self.model
                .expert_c
                .proj_box
                .weight
                .detach()
                .clone()
            ),
            "T_box": (
                self.model
                .expert_t
                .proj_box
                .weight
                .detach()
                .clone()
            ),
        }

        optimizer = (
            torch.optim.AdamW(
                [
                    parameter
                    for parameter
                    in self.model.parameters()
                    if parameter.requires_grad
                ],
                lr=1e-3,
                weight_decay=0.0,
            )
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        z, b, _ = self.model(
            self.O,
            self.X,
            context_mode="unknown",
        )

        target_z = (
            self.O.z
            + 0.10
        )

        target_b = (
            self.O.b
            + 0.05
        ).clamp(
            0.0,
            1.0,
        )

        loss = (
            (
                z
                - target_z
            )
            .pow(2)
            .mean()
            +
            (
                b
                - target_b
            )
            .pow(2)
            .mean()
        )

        loss.backward()

        optimizer.step()

        # Native baseline heads must remain EXACTLY unchanged.
        for (
            key,
            value,
        ) in (
            self.model
            .adapter
            .state_dict()
            .items()
        ):
            self.assertTrue(
                torch.equal(
                    value,
                    frozen_before[key],
                ),
                (
                    "Frozen native/R0 tensor "
                    "changed after optimizer step: "
                    f"{key}"
                ),
            )

        expert_after = {
            "S_cls": (
                self.model
                .expert_s
                .proj_cls
                .weight
                .detach()
            ),
            "L_cls": (
                self.model
                .expert_l
                .proj_cls
                .weight
                .detach()
            ),
            "C_cls": (
                self.model
                .expert_c
                .proj_cls
                .weight
                .detach()
            ),
            "T_cls": (
                self.model
                .expert_t
                .proj_cls
                .weight
                .detach()
            ),
            "S_box": (
                self.model
                .expert_s
                .proj_box
                .weight
                .detach()
            ),
            "L_box": (
                self.model
                .expert_l
                .proj_box
                .weight
                .detach()
            ),
            "C_box": (
                self.model
                .expert_c
                .proj_box
                .weight
                .detach()
            ),
            "T_box": (
                self.model
                .expert_t
                .proj_box
                .weight
                .detach()
            ),
        }

        for key in expert_before:
            self.assertFalse(
                torch.equal(
                    expert_before[key],
                    expert_after[key],
                ),
                (
                    "Trainable residual projector "
                    f"failed to update: {key}"
                ),
            )

    # ================================================================
    # E. TASK-SPECIFIC AVAILABILITY
    # ================================================================

    def test_13_task_specific_capability_mask(
        self,
    ):
        self.model.eval()

        # Classification unavailable for Scale.
        # Localization remains available.
        self.O.capabilities[
            "scale.cls"
        ] = False

        self.O.capabilities[
            "scale.box"
        ] = True

        # Force all local gates open so capability A is the deciding
        # factor, rather than gate calibration.
        for expert in (
            "S",
            "L",
            "C",
            "T",
        ):
            (
                self.model
                .gate_cls[expert]
                .consequents
                .bias
                .data
                .fill_(10.0)
            )

            (
                self.model
                .gate_box[expert]
                .consequents
                .bias
                .data
                .fill_(10.0)
            )

        _, _, info = self.model(
            self.O,
            self.X,
            context_mode="unknown",
        )

        # S classification availability must be exactly zero.
        self.assertEqual(
            float(
                info["A_cls"][
                    :,
                    0
                ]
                .sum()
                .item()
            ),
            0.0,
        )

        # S box capability remains available.
        self.assertGreater(
            float(
                info["A_box"][
                    :,
                    0
                ]
                .sum()
                .item()
            ),
            0.0,
        )

        # S classification route mass can therefore never be restored
        # by the Router.
        self.assertTrue(
            torch.equal(
                info["pi_cls"][
                    :,
                    1
                ],
                torch.zeros_like(
                    info["pi_cls"][
                        :,
                        1
                    ]
                ),
            )
        )

    # ================================================================
    # F. MANDATORY REAL R0 / CANONICAL-CACHE PARITY
    # ================================================================

    def test_14_real_r0_native_parity_and_cache_schema(
        self,
    ):
        """
        This test is intentionally mandatory on the user's machine.

        It must FAIL if any required certified R0 artifact is unavailable.
        There is no silent skip.
        """

        cache_path = (
            EVAL_TOOLS
            / "e0_cache"
            / "bfcef_e0_cache_vid1.pth"
        )

        checkpoint_path = (
            ROOT_DIR
            / "checkpoints_r0"
            / "checkpoint_best.pth"
        )

        p0_driver = (
            EVAL_TOOLS
            / "run_bfcef_p0_a0_certification.py"
        )

        self.assertTrue(
            cache_path.exists(),
            (
                "Missing certified canonical "
                f"cache shard: {cache_path}"
            ),
        )

        self.assertTrue(
            checkpoint_path.exists(),
            (
                "Missing certified R0 "
                f"checkpoint: {checkpoint_path}"
            ),
        )

        self.assertTrue(
            p0_driver.exists(),
            (
                "Missing certified P0/A0 "
                f"driver: {p0_driver}"
            ),
        )

        self.assertEqual(
            sha256(
                checkpoint_path
            ),
            R0_CHECKPOINT_SHA256,
        )

        records = torch.load(
            cache_path,
            map_location="cpu",
        )

        self.assertGreater(
            len(records),
            0,
        )

        rec = records[0]

        # First certify the authoritative cache schema.
        self.assertEqual(
            rec["cues_s"]
            .shape[-1],
            10,
        )

        self.assertEqual(
            rec["cues_l"]
            .shape[-1],
            10,
        )

        self.assertEqual(
            rec["cues_c"]
            .shape[-1],
            6,
        )

        # Reuse the already-certified R0 constructor.
        # We do NOT reimplement R0 architecture here.
        from run_bfcef_p0_a0_certification import (
            load_authoritative_r0_model,
        )

        _, native_cls, native_box = (
            load_authoritative_r0_model()
        )

        adapter = (
            DeformableDETRNativeAdapter(
                native_cls.cpu(),
                native_box.cpu(),
            )
            .eval()
        )

        q_native = (
            rec["q_native"]
            .float()
        )

        ref_pts = (
            rec["ref_pts"]
            .float()
        )

        cached_logits = (
            rec["d_logits"]
            .float()
        )

        cached_boxes = (
            rec["d_boxes"]
            .float()
        )

        decoded_logits = (
            adapter
            .decode_cls(
                q_native
            )
        )

        decoded_boxes = (
            adapter
            .decode_box(
                q_native,
                ref_pts,
            )
        )

        max_logit_error = float(
            (
                decoded_logits
                - cached_logits
            )
            .abs()
            .max()
            .item()
        )

        max_box_error = float(
            (
                decoded_boxes
                - cached_boxes
            )
            .abs()
            .max()
            .item()
        )

        self.assertLessEqual(
            max_logit_error,
            1e-4,
        )

        self.assertLessEqual(
            max_box_error,
            1e-4,
        )

        # Real native heads are frozen in the B-FCEF adapter.
        for parameter in adapter.parameters():
            self.assertFalse(
                parameter.requires_grad
            )

        print(
            "\n[AC-PROD] REAL R0 PARITY PASS"
            f"\n  max_logit_error = {max_logit_error:.3e}"
            f"\n  max_box_error   = {max_box_error:.3e}"
        )


if __name__ == "__main__":
    unittest.main(
        verbosity=2
    )