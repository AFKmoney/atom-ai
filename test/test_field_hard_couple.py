"""Unit tests: field→hard-decode couple residual (opt-in)."""
from __future__ import annotations

import unittest

import torch

from src.atom_native import AtomNativeModel, AtomSurfaceHead
from src.io.atomizer import Atomizer


def _tiny_byte_tick(**kwargs) -> AtomNativeModel:
    args = dict(
        d_model=8,
        n_modes=8,
        n_atoms_max=32,
        max_payload_bytes=1,
        atomizer=Atomizer(max_span_bytes=1),
        field_max_rms=3.0,
        field_obligatory_hard=True,
        enable_merge=False,
        last_atom_readout=True,
    )
    args.update(kwargs)
    return AtomNativeModel(**args)


class FieldHardCoupleTests(unittest.TestCase):
    def test_default_zero_matches_off(self) -> None:
        torch.manual_seed(0)
        model = _tiny_byte_tick()
        model.eval()
        self.assertEqual(float(model.surface.field_hard_couple_scale), 0.0)
        alpha = torch.randn(8, 8)
        persist = torch.randn(8)
        atom_r = torch.randn(8)
        with torch.no_grad():
            out0 = model.surface(
                None,
                alpha=alpha,
                persistent_state=persist,
                atom_r=atom_r,
                atom_payloads=[b"a", b"b"],
                last_phi=torch.zeros(8),
            )
            model.surface.field_hard_couple_scale = 0.0
            out1 = model.surface(
                None,
                alpha=alpha,
                persistent_state=persist,
                atom_r=atom_r,
                atom_payloads=[b"a", b"b"],
                last_phi=torch.zeros(8),
            )
        self.assertTrue(torch.equal(out0["byte_logits"], out1["byte_logits"]))

    def test_scale_changes_logits_and_grads_hit_skip(self) -> None:
        torch.manual_seed(1)
        model = _tiny_byte_tick()
        model.train()
        model.surface.field_hard_couple_scale = 0.5
        packets = Atomizer(max_span_bytes=1).encode("Bon!")
        loss, info = model.transition_loss(packets[0], packets[1])
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        grad = model.surface.field_hard_couple_byte.weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all().item())
        self.assertGreater(float(grad.abs().sum()), 0.0)
        # Frozen skip stays untouched under hard.
        self.assertTrue(
            (model.surface.field_byte_skip.weight == 0).all().item()
        )

    def test_nonzero_scale_differs_from_zero(self) -> None:
        torch.manual_seed(2)
        head = AtomSurfaceHead(
            d_model=8, max_payload_bytes=1, n_modes=8, last_atom_readout=True
        )
        head.set_obligatory_hard(True)
        head.payload_enabled = False
        alpha = torch.randn(8, 8)
        persist = torch.randn(8)
        atom_r = torch.randn(8)
        kwargs = dict(
            state=None,
            alpha=alpha,
            persistent_state=persist,
            atom_r=atom_r,
            atom_payloads=[b"x", b"y"],
            last_phi=torch.zeros(8),
        )
        head.field_hard_couple_scale = 0.0
        with torch.no_grad():
            base = head(**kwargs)["byte_logits"].clone()
            head.field_hard_couple_scale = 1.0
            coupled = head(**kwargs)["byte_logits"].clone()
        self.assertFalse(torch.allclose(base, coupled))
        delta = (coupled - base).abs().mean().item()
        self.assertGreater(delta, 1e-6)


if __name__ == "__main__":
    unittest.main()
