"""Unit tests: multi-tick free-run aux (L_roll)."""
from __future__ import annotations

import unittest

import torch

from src.atom_native import AtomNativeModel
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


class FreeRunAuxTests(unittest.TestCase):
    def test_free_run_finite_grads_and_horizon(self) -> None:
        torch.manual_seed(0)
        model = _tiny_byte_tick()
        model.train()
        packets = Atomizer(max_span_bytes=1).encode("Bonjour!")
        self.assertGreaterEqual(len(packets), 5)
        loss, info = model.free_run_aux_loss(packets[0], packets[1:5], weight=1.0)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(info.get("free_run_aux"))
        self.assertEqual(info.get("free_run_horizon"), 4)
        loss.backward()
        assert model.surface.last_atom is not None
        grad = model.surface.last_atom.head.weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all().item())
        # α-MLP should also get grads under hard.
                g_mlp = next(p for p in model.surface.alpha_byte_proj.parameters() if p.grad is not None)
        self.assertIsNotNone(g_mlp)
        self.assertTrue(torch.isfinite(g_mlp).all().item())

    def test_free_run_commits_horizon_bytes(self) -> None:
        torch.manual_seed(0)
        model = _tiny_byte_tick()
        model.eval()
        packets = Atomizer(max_span_bytes=1).encode("Bonjour!!")
        with torch.no_grad():
            model.reset_state(reset_atomizer=True)
            before = model.atomizer.position
            model.free_run_aux_loss(packets[0], packets[1:5])
            after = model.atomizer.position
        # 4 golds ⇒ 4 free-run commits via packet_from_payload.
        self.assertEqual(after - before, 4)

    def test_cli_default_off_does_not_alter_transition(self) -> None:
        # Sanity: free_run path is opt-in; normal transition still works.
        torch.manual_seed(0)
        model = _tiny_byte_tick()
        model.train()
        packets = Atomizer(max_span_bytes=1).encode("Hi")
        loss, info = model.transition_loss(packets[0], packets[1])
        self.assertNotIn("free_run_aux", info)
        self.assertTrue(torch.isfinite(loss).item())


if __name__ == "__main__":
    unittest.main()
