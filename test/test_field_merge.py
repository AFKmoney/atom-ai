"""Field-aware loss + MERGE coherence tests."""
from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from src.atom_native import AtomNativeModel
from src.io.atomizer import Atomizer
from src.toroidal.aggregation import AggregationEngine
from src.toroidal.atom import ToroidalAtom, ToroidalAtomCollection


class MergeTests(unittest.TestCase):
    def test_merge_coherent_reduces_atom_count(self) -> None:
        eng = AggregationEngine(d_model=8, phase_coherence_threshold=0.5, merge_energy_floor=0.05)
        torch.manual_seed(0)
        n, d = 8, 8
        base_phi = torch.randn(1, d)
        phi = base_phi.expand(n, d).clone() + 0.02 * torch.randn(n, d)
        r = F.normalize(torch.randn(n, d), dim=-1)
        omega = torch.rand(n, d) + 0.1
        E = torch.ones(n, d) * 0.7
        kappa = torch.ones(n, d) * 0.4
        M = F.normalize(torch.randn(n, d), dim=-1)
        tau = torch.ones(n, d)
        rho = torch.zeros(n, 1)
        merged, remove, count = eng.merge_coherent(r, phi, omega, E, kappa, M, tau, rho)
        self.assertGreater(count, 0)
        self.assertEqual(len(merged), count)
        self.assertEqual(len(remove), 2 * count)
        # Heavier: rho bumped
        self.assertTrue(all(float(m["rho"].max()) >= 1.0 for m in merged))

    def test_collection_lazy_buffers_after_add(self) -> None:
        coll = ToroidalAtomCollection()
        d = 4
        for _ in range(5):
            coll.add(
                ToroidalAtom(
                    r=F.normalize(torch.randn(d), dim=-1),
                    phi=torch.randn(d),
                    omega=torch.ones(d),
                    E=torch.ones(d) * 0.5,
                    kappa=torch.ones(d),
                    M=F.normalize(torch.randn(d), dim=-1),
                    tau=torch.ones(d),
                    rho=torch.zeros(1),
                )
            )
        self.assertEqual(len(coll), 5)
        self.assertEqual(tuple(coll.r.shape), (5, d))


class FieldLossTests(unittest.TestCase):
    def test_contrast_penalty_when_surface_ignores_alpha(self) -> None:
        model = AtomNativeModel(
            d_model=8,
            n_modes=8,
            n_atoms_max=32,
            max_payload_bytes=8,
            enable_merge=True,
            field_contrast_weight=1.0,
            field_loss_weight=0.0,
            field_contrast_margin=0.2,
        )
        packets = Atomizer(max_span_bytes=8).encode("alpha beta gamma delta ")
        # Force skip_gate closed and field path weak so logits ignore α → high cos → hinge fires
        with torch.no_grad():
            model.surface.skip_gate.fill_(-8.0)
            model.surface.field_gate.fill_(-8.0)
        loss, info = model.transition_loss(packets[0], packets[1])
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("field_contrast_loss", info)
        # With field path suppressed, null/true logits are similar → contrast > 0
        self.assertGreater(info["field_contrast_loss"], 0.0)

    def test_field_loss_backward_and_merge_flag(self) -> None:
        model = AtomNativeModel(
            d_model=8,
            n_modes=8,
            n_atoms_max=64,
            max_payload_bytes=8,
            enable_merge=True,
            slow_every=2,
            field_contrast_weight=0.1,
            field_loss_weight=0.05,
        )
        packets = Atomizer(max_span_bytes=8).encode("Bonjour le monde atomique! " * 4)
        model.train()
        for i in range(min(12, len(packets) - 1)):
            loss, info = model.transition_loss(packets[i], packets[i + 1])
            loss.backward()
            model.zero_grad(set_to_none=True)
        self.assertGreaterEqual(info["merge_count_total"], 0)
        self.assertIn("slow_tick", info)


if __name__ == "__main__":
    unittest.main()
