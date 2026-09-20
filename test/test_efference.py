"""Unit tests: efference copy (bridge #9)."""
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


class EfferenceTests(unittest.TestCase):
    def test_efference_finite_and_grads(self) -> None:
        torch.manual_seed(0)
        model = _tiny_byte_tick()
        model.train()
        packets = Atomizer(max_span_bytes=1).encode("Bonjour")
        loss, info = model.efference_loss(packets[0], packets[1])
        self.assertTrue(torch.isfinite(loss).item())
        self.assertIn("byte_loss", info)
        loss.backward()
        assert model.surface.last_atom is not None
        grad = model.surface.last_atom.head.weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all().item())

    def test_efference_commits_own_byte(self) -> None:
        torch.manual_seed(0)
        model = _tiny_byte_tick()
        model.eval()
        packets = Atomizer(max_span_bytes=1).encode("Bonjour")
        with torch.no_grad():
            # Reference prediction from the same start state.
            model.reset_state(reset_atomizer=True)
            model.eval()
            expected = bytes(model.predict_packet(packets[0], deterministic=True, merge_enabled=False))
            # Efference run from the same start state.
            model.reset_state(reset_atomizer=True)
            model.eval()
            before = model.atomizer.position
            model.efference_loss(packets[0], packets[1])
            after = model.atomizer.position
        self.assertEqual(after - before, 1)
        living = list(model.core.atoms.atoms)
        self.assertGreaterEqual(len(living), 1)
        self.assertEqual(bytes(living[-1].payload or b""), expected)

    def test_normal_transition_does_not_commit(self) -> None:
        torch.manual_seed(0)
        model = _tiny_byte_tick()
        model.eval()
        packets = Atomizer(max_span_bytes=1).encode("Bonjour")
        with torch.no_grad():
            model.reset_state(reset_atomizer=True)
            model.eval()
            before = model.atomizer.position
            model.transition_loss(packets[0], packets[1])
            after = model.atomizer.position
        self.assertEqual(after, before)

    def test_efference_fallback_span_buffer(self) -> None:
        # Off byte-tick with a pending span: fall back, no raise.
        torch.manual_seed(0)
        model = AtomNativeModel(
            d_model=8,
            n_modes=8,
            n_atoms_max=32,
            max_payload_bytes=8,
            atomizer=Atomizer(max_span_bytes=8),
            field_max_rms=3.0,
            field_obligatory_hard=True,
            enable_merge=False,
        )
        model.eval()
        packets = Atomizer(max_span_bytes=8).encode("hello world")
        model.atomizer.encode_bytes(b"ab", reset=True, flush=False)
        self.assertGreater(len(model.atomizer._buffer), 0)
        with torch.no_grad():
            loss, _ = model.efference_loss(packets[0], packets[1])
        self.assertTrue(torch.isfinite(loss).item())


if __name__ == "__main__":
    unittest.main()
