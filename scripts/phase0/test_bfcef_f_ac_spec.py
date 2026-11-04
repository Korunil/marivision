from __future__ import annotations

import unittest

import torch

from bfcef_f.neuro_fuzzy_gate import AdaptiveSparseTSKGate
from bfcef_f.router import GateConstrainedSimplexRouter


class TestBFCEFFACSpec(unittest.TestCase):
    """
    Mathematical/specification-level invariants for frozen B-FCEF-F.

    These tests do not replace AC-PROD. They certify the mathematical
    behavior expected from the permanent Gate/Router architecture.
    """

    def setUp(self):
        torch.manual_seed(4606)
        self.router = GateConstrainedSimplexRouter()

    def test_01_router_simplex(self):
        g = torch.rand(32, 4)
        A = torch.ones_like(g)
        c = torch.randn_like(g)

        pi = self.router(g, A, c)

        self.assertEqual(tuple(pi.shape), (32, 5))
        self.assertTrue(torch.all(pi >= 0))

        self.assertTrue(
            torch.allclose(
                pi.sum(dim=-1),
                torch.ones(32),
                atol=1e-6,
                rtol=0.0,
            )
        )

    def test_02_closed_gate_cannot_be_reopened_by_router(self):
        g = torch.tensor(
            [[1.0, 0.0, 0.8, 0.0]]
        )

        A = torch.ones_like(g)

        # Extremely favorable coordinator logits must still be unable
        # to reopen an expert whose local gate is exactly zero.
        c = torch.full_like(g, 5.0)

        pi = self.router(g, A, c)

        # pi[:, 0] = baseline
        # pi[:, 1] = S
        # pi[:, 2] = L
        # pi[:, 3] = C
        # pi[:, 4] = T

        self.assertEqual(float(pi[0, 2]), 0.0)
        self.assertEqual(float(pi[0, 4]), 0.0)

        self.assertGreater(
            float(pi[0, 1]),
            0.0,
        )
        self.assertGreater(
            float(pi[0, 3]),
            0.0,
        )

    def test_03_unavailable_expert_cannot_receive_route_mass(self):
        g = torch.ones(2, 4)

        A = torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
            ]
        )

        c = torch.full_like(g, 5.0)

        pi = self.router(g, A, c)

        # Row 0:
        # L and T unavailable.
        self.assertEqual(
            float(pi[0, 2]),
            0.0,
        )
        self.assertEqual(
            float(pi[0, 4]),
            0.0,
        )

        # Row 1:
        # S and C unavailable.
        self.assertEqual(
            float(pi[1, 1]),
            0.0,
        )
        self.assertEqual(
            float(pi[1, 3]),
            0.0,
        )

    def test_04_all_experts_closed_yields_exact_baseline_route(self):
        g = torch.zeros(7, 4)
        A = torch.ones_like(g)
        c = torch.randn_like(g)

        pi = self.router(
            g,
            A,
            c,
        )

        self.assertTrue(
            torch.equal(
                pi[:, 0],
                torch.ones(7),
            )
        )

        self.assertTrue(
            torch.equal(
                pi[:, 1:],
                torch.zeros(7, 4),
            )
        )

    def test_05_gate_exposes_soft_and_exact_hard_semantics(self):
        gate = AdaptiveSparseTSKGate(
            in_dim=5,
            num_rules=4,
        )
    
        x = torch.randn(8, 5)
    
        gate.train()
        out = gate(x)
    
        self.assertEqual(
            set(out.keys()),
            {
                "g_soft",
                "mask_hard",
                "mask_ste",
                "g_ste",
                "g_hard",
            },
        )
    
        # Continuous gate amplitude.
        self.assertEqual(out["g_soft"].shape, (8, 1))
        self.assertTrue(
            torch.all(out["g_soft"] >= 0.0)
        )
        self.assertTrue(
            torch.all(out["g_soft"] <= 1.0)
        )
    
        # Exact hard mask must contain only binary values.
        hard_values = set(
            torch.unique(out["mask_hard"])
            .detach()
            .cpu()
            .tolist()
        )
        self.assertTrue(
            hard_values.issubset({0.0, 1.0})
        )
    
        # Exact inference intervention.
        self.assertTrue(
            torch.equal(
                out["g_hard"],
                out["mask_hard"] * out["g_soft"],
            )
        )
    
        # Straight-through training intervention.
        self.assertTrue(
            torch.allclose(
                out["g_ste"],
                out["mask_ste"] * out["g_soft"],
            )
        )
    
        # Forward values of the STE mask must equal the hard mask.
        self.assertTrue(
            torch.equal(
                out["mask_ste"].detach(),
                out["mask_hard"].detach(),
            )
        )
    
        # Therefore forward values of g_ste and g_hard must also
        # be identical; they differ only in backward behaviour.
        self.assertTrue(
            torch.equal(
                out["g_ste"].detach(),
                out["g_hard"].detach(),
            )
        )
    def test_06_soft_gate_is_differentiable_while_hard_gate_is_exact(self):
        gate = AdaptiveSparseTSKGate(
            in_dim=5,
            num_rules=4,
            initial_logit=-4.0,
            threshold=0.10,
        )

        x = torch.randn(
            12,
            5,
            requires_grad=True,
        )

        out = gate(x)

        # Training uses this differentiable amplitude.
        out["g_soft"].sum().backward()

        self.assertIsNotNone(
            x.grad
        )

        self.assertGreater(
            float(
                x.grad.abs().sum()
            ),
            0.0,
        )

        # Deployment intervention remains exactly zero.
        self.assertTrue(
            torch.equal(
                out["g_hard"],
                torch.zeros_like(
                    out["g_hard"]
                ),
            )
        )

    def test_07_baseline_route_increases_as_gate_strength_decreases(self):
        A = torch.ones(1, 4)
        c = torch.zeros(1, 4)

        weak = self.router(
            torch.full(
                (1, 4),
                0.01,
            ),
            A,
            c,
        )

        strong = self.router(
            torch.full(
                (1, 4),
                0.90,
            ),
            A,
            c,
        )

        self.assertGreater(
            float(
                weak[0, 0]
            ),
            float(
                strong[0, 0]
            ),
        )


if __name__ == "__main__":
    unittest.main(
        verbosity=2
    )
