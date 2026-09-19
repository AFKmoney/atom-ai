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
        merged, remove, count, diag = eng.merge_coherent(r, phi, omega, E, kappa, M, tau, rho)
        self.assertGreater(count, 0)
        self.assertEqual(len(merged), count)
        self.assertEqual(len(remove), 2 * count)
        # Heavier: rho bumped
        self.assertTrue(all(float(m["rho"].max()) >= 1.0 for m in merged))
        self.assertIn("max_phase_coherence", diag)
        self.assertGreaterEqual(diag["max_phase_coherence"], 0.0)
        self.assertGreaterEqual(diag["n_pairs_above_energy_floor"], 0)

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


    def test_default_thresholds_merge_synthetic_coherent(self) -> None:
        """New defaults (coherence 0.45) must merge synthetic coherent atoms."""
        eng = AggregationEngine(d_model=8)  # defaults: thr=0.45, floor=0.08
        self.assertAlmostEqual(eng.phase_coherence_threshold, 0.45, places=5)
        self.assertAlmostEqual(eng.merge_energy_floor, 0.08, places=5)
        torch.manual_seed(1)
        n, d = 6, 8
        base_phi = torch.zeros(1, d)
        phi = base_phi.expand(n, d).clone() + 0.01 * torch.randn(n, d)
        r = F.normalize(torch.randn(n, d), dim=-1)
        omega = torch.rand(n, d) + 0.1
        E = torch.ones(n, d) * 0.6
        kappa = torch.ones(n, d) * 0.4
        M = F.normalize(torch.randn(n, d), dim=-1)
        tau = torch.ones(n, d)
        rho = torch.zeros(n, 1)
        merged, remove, count, diag = eng.merge_coherent(r, phi, omega, E, kappa, M, tau, rho)
        self.assertGreater(count, 0, f"expected merges under defaults; diag={diag}")
        self.assertGreaterEqual(diag["max_phase_coherence"], eng.phase_coherence_threshold)
        self.assertGreater(diag["n_pairs_above_energy_floor"], 0)

    def test_mph_half_regime_merges_under_operational_threshold(self) -> None:
        """Logged regime: E=0.5, φ≈0 → mph=0.5; thr=0.45 merges, thr=0.55 does not."""
        torch.manual_seed(2)
        n, d = 8, 8
        phi = torch.zeros(n, d)
        r = F.normalize(torch.randn(n, d), dim=-1)
        omega = torch.ones(n, d)
        E = torch.ones(n, d) * 0.5
        kappa = torch.ones(n, d) * 0.4
        M = F.normalize(torch.randn(n, d), dim=-1)
        tau = torch.ones(n, d)
        rho = torch.zeros(n, 1)
        blocked = AggregationEngine(d_model=d, phase_coherence_threshold=0.55, merge_energy_floor=0.08)
        _, _, c_block, diag_b = blocked.merge_coherent(r, phi, omega, E, kappa, M, tau, rho)
        self.assertEqual(c_block, 0)
        self.assertAlmostEqual(diag_b["max_phase_coherence"], 0.5, places=4)
        operational = AggregationEngine(d_model=d)  # 0.45 default
        _, _, c_ok, diag_o = operational.merge_coherent(r, phi, omega, E, kappa, M, tau, rho)
        self.assertGreater(c_ok, 0, f"mph~0.5 must merge under thr=0.45; diag={diag_o}")
        self.assertAlmostEqual(diag_o["max_phase_coherence"], 0.5, places=4)

    def test_model_set_merge_thresholds_propagates(self) -> None:
        model = AtomNativeModel(d_model=8, n_modes=8, n_atoms_max=32, max_payload_bytes=8)
        model.set_merge_thresholds(coherence_threshold=0.42, energy_floor=0.05)
        self.assertAlmostEqual(model.merge_coherence_threshold, 0.42, places=5)
        self.assertAlmostEqual(model.core.aggregation.phase_coherence_threshold, 0.42, places=5)
        self.assertAlmostEqual(model.core.aggregation.merge_energy_floor, 0.05, places=5)


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


class FieldIgnoranceTests(unittest.TestCase):
    def test_l_ign_zero_when_cos_below_margin(self) -> None:
        """L_ign = ReLU(cos - m) is 0 when cos(ℓ(α), ℓ(sg[α_bar])) < m."""
        model = AtomNativeModel(
            d_model=8,
            n_modes=8,
            n_atoms_max=32,
            max_payload_bytes=8,
            enable_merge=False,
            field_contrast_weight=0.0,
            field_loss_weight=0.0,
            field_ignorance_weight=1.0,
            field_ignorance_margin=0.85,
        )
        # Seed α-bank with a distinct other-prompt field (stopgrad snap).
        with torch.no_grad():
            other = torch.randn_like(model.core.state.alpha) * 0.5
            model._alpha_bank.append(other.clone())
        packets = Atomizer(max_span_bytes=8).encode("alpha beta gamma delta ")
        # Open field path so surface can separate; if cos < m, hinge is 0.
        with torch.no_grad():
            model.surface.skip_gate.fill_(4.0)
            model.surface.field_gate.fill_(4.0)
        # Directly evaluate hinge shape/zero via aux after one forward that
        # populates bank with current ([:-1] keeps the seeded other).
        out = model.forward_packet(packets[0])
        with torch.no_grad():
            model._alpha_bank.append(out["field"].detach().clone())
        aux, info = model._field_auxiliary_losses(out, packets[0])
        self.assertTrue(torch.isfinite(aux))
        self.assertEqual(tuple(aux.shape), ())
        self.assertIn("field_ignorance_loss", info)
        cos = info["field_logit_cos_ignorance"]
        if cos == cos:  # not nan
            expected = max(0.0, cos - 0.85)
            self.assertAlmostEqual(info["field_ignorance_loss"], expected, places=5)
            if cos < 0.85:
                self.assertEqual(info["field_ignorance_loss"], 0.0)

    def test_l_ign_shape_and_backward_smoke(self) -> None:
        model = AtomNativeModel(
            d_model=8,
            n_modes=8,
            n_atoms_max=32,
            max_payload_bytes=8,
            enable_merge=False,
            field_contrast_weight=0.0,
            field_loss_weight=0.0,
            field_ignorance_weight=0.08,
            field_ignorance_margin=0.85,
        )
        packets = Atomizer(max_span_bytes=8).encode("hello world atom field ")
        model.train()
        # First step seeds bank; second applies L_ign against other α.
        loss1, _ = model.transition_loss(packets[0], packets[1])
        loss1.backward()
        model.zero_grad(set_to_none=True)
        loss2, info = model.transition_loss(packets[1], packets[2])
        self.assertTrue(torch.isfinite(loss2))
        self.assertEqual(tuple(loss2.shape), ())
        self.assertIn("field_ignorance_loss", info)
        loss2.backward()

    def test_l_ign_ablate_shared_zeros_bar_side_inputs(self) -> None:
        """Ablate flag: ℓ(α_bar) uses None atom_r/persist; ℓ(α) unchanged."""
        model = AtomNativeModel(
            d_model=8,
            n_modes=8,
            n_atoms_max=32,
            max_payload_bytes=8,
            enable_merge=False,
            field_contrast_weight=0.0,
            field_loss_weight=0.0,
            field_ignorance_weight=1.0,
            field_ignorance_margin=0.85,
            field_ignorance_ablate_shared=True,
        )
        with torch.no_grad():
            other = torch.randn_like(model.core.state.alpha) * 0.5
            model._alpha_bank.append(other.clone())
            model.surface.skip_gate.fill_(4.0)
            model.surface.field_gate.fill_(4.0)
        packets = Atomizer(max_span_bytes=8).encode("ablate shared state test ")
        out = model.forward_packet(packets[0])
        with torch.no_grad():
            model._alpha_bank.append(out["field"].detach().clone())
        calls: list[dict] = []
        orig_forward = model.surface.forward

        def wrapped_forward(*args, **kwargs):
            calls.append(
                {
                    "atom_r_is_none": kwargs.get("atom_r") is None,
                    "persist_is_none": kwargs.get("persistent_state") is None,
                }
            )
            return orig_forward(*args, **kwargs)

        model.surface.forward = wrapped_forward  # type: ignore[method-assign]
        try:
            aux, info = model._field_auxiliary_losses(out, packets[0])
        finally:
            model.surface.forward = orig_forward  # type: ignore[method-assign]
        self.assertTrue(torch.isfinite(aux))
        self.assertTrue(calls, "expected at least one surface call for α_bar")
        self.assertTrue(
            any(c["atom_r_is_none"] and c["persist_is_none"] for c in calls),
            f"ablate should pass None/None on bar call; saw {calls}",
        )
        self.assertIn("field_ignorance_loss", info)

    def test_l_ign_prompt_bank_uses_distinct_prompts(self) -> None:
        """Prompt-bank flag: α_bar from _prompt_alpha_bank, not train ticks."""
        model = AtomNativeModel(
            d_model=8,
            n_modes=8,
            n_atoms_max=32,
            max_payload_bytes=8,
            enable_merge=False,
            field_contrast_weight=0.0,
            field_loss_weight=0.0,
            field_ignorance_weight=1.0,
            field_ignorance_margin=0.85,
            field_ignorance_prompt_bank=True,
        )
        model._ignorance_prompts = ("Bonjour", "Hello world")
        packets = Atomizer(max_span_bytes=8).encode("train tick alpha beta ")
        model.train()
        with torch.no_grad():
            model._alpha_bank.append(torch.ones_like(model.core.state.alpha))
        alpha_before = model.core.state.alpha.detach().clone()
        loss, info = model.transition_loss(packets[0], packets[1])
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(len(model._prompt_alpha_bank), 1)
        self.assertGreaterEqual(len(model._alpha_bank), 1)
        # Refresh must restore live field (not leave prompt-episode wipe).
        self.assertEqual(tuple(model.core.state.alpha.shape), tuple(alpha_before.shape))
        self.assertIn("field_ignorance_loss", info)
        loss.backward()


if __name__ == "__main__":
    unittest.main()


class FieldNextPacketTests(unittest.TestCase):
    def test_next_packet_loss_backward_under_hard(self) -> None:
        model = AtomNativeModel(
            d_model=8,
            n_modes=8,
            n_atoms_max=32,
            max_payload_bytes=8,
            enable_merge=False,
            field_contrast_weight=0.0,
            field_loss_weight=0.0,
            field_ignorance_weight=0.0,
            field_next_packet_weight=0.05,
            field_obligatory_hard=True,
        )
        packets = Atomizer(max_span_bytes=8).encode("next packet alpha beta gamma ")
        model.train()
        loss, info = model.transition_loss(packets[0], packets[1])
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("field_next_packet_loss", info)
        self.assertGreaterEqual(info["field_next_packet_loss"], 0.0)
        loss.backward()
        grad_hit = any(
            p.grad is not None and float(p.grad.abs().sum()) > 0
            for p in model.field_next_probe.parameters()
        )
        self.assertTrue(grad_hit, "next-packet probe should receive gradients")

