"""Unit tests: --last-atom-dropout (train-only zeroing of the last_atom logit term)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.atom_native import AtomNativeModel, AtomSurfaceHead
from src.io.atomizer import Atomizer


def _hard_head() -> AtomSurfaceHead:
    torch.manual_seed(0)
    head = AtomSurfaceHead(d_model=8, max_payload_bytes=1, n_modes=8, last_atom_readout=True)
    head.field_obligatory_readout = True
    head.set_obligatory_hard(True)
    head.payload_enabled = False
    return head


def _inputs():
    g = torch.Generator().manual_seed(1)
    return dict(
        alpha=torch.randn(8, 8, generator=g),
        atom_payloads=[b"a", b"b"],
        last_phi=torch.randn(8, generator=g),
    )


def _logits(head: AtomSurfaceHead, *, train: bool, grad: bool = True) -> torch.Tensor:
    head.train(train)
    with torch.set_grad_enabled(grad):
        return head(None, **_inputs())["byte_logits"].detach().clone()


def _no_la_reference(head: AtomSurfaceHead) -> torch.Tensor:
    """Logits with the last_atom term removed (scale 0 ⇒ la_byte == 0 exactly)."""
    old = head.last_atom_scale
    head.last_atom_scale = 0.0
    try:
        return _logits(head, train=False)
    finally:
        head.last_atom_scale = old


class LastAtomDropoutTests(unittest.TestCase):
    def test_default_is_zero(self) -> None:
        head = AtomSurfaceHead(d_model=8, max_payload_bytes=1, last_atom_readout=True)
        self.assertEqual(float(head.last_atom_dropout), 0.0)

    def test_p0_identical_logits_and_no_rng_draw(self) -> None:
        head = _hard_head()
        base = _logits(head, train=True)
        head.last_atom_dropout = 0.0
        torch.manual_seed(123)
        out = _logits(head, train=True)
        after = torch.rand(())
        torch.manual_seed(123)
        ref_after = torch.rand(())
        self.assertTrue(torch.equal(base, out))
        # P=0 must not consume global RNG (keeps live-recipe RNG stream intact).
        self.assertTrue(torch.equal(after, ref_after))

    def test_p1_train_removes_last_atom(self) -> None:
        head = _hard_head()
        full = _logits(head, train=True)
        ref = _no_la_reference(head)
        self.assertGreater(float((full - ref).abs().max()), 1e-6)  # la term is live
        head.last_atom_dropout = 1.0
        dropped = _logits(head, train=True)
        self.assertTrue(torch.allclose(dropped, ref, atol=1e-6))

    def test_eval_and_no_grad_unaffected(self) -> None:
        head = _hard_head()
        full_eval = _logits(head, train=False)
        head.last_atom_dropout = 1.0
        self.assertTrue(torch.equal(_logits(head, train=False), full_eval))
        # no-grad prediction in train mode (efference own-byte pick) also ignores P
        self.assertTrue(torch.allclose(_logits(head, train=True, grad=False), full_eval))

    def test_p_half_is_per_position_bernoulli(self) -> None:
        head = _hard_head()
        head.last_atom_dropout = 0.5
        full = _logits(head, train=False)
        ref = _no_la_reference(head)
        torch.manual_seed(7)
        n_drop = 0
        n = 400
        for _ in range(n):
            out = _logits(head, train=True)
            if torch.allclose(out, ref, atol=1e-6):
                n_drop += 1
            else:
                self.assertTrue(torch.allclose(out, full, atol=1e-6))
        self.assertTrue(0.4 < n_drop / n < 0.6, n_drop)

    def test_force_off_diagnostic(self) -> None:
        head = _hard_head()
        ref = _no_la_reference(head)
        head.last_atom_force_off = True
        self.assertTrue(torch.allclose(_logits(head, train=False), ref, atol=1e-6))

    def test_checkpoint_roundtrip(self) -> None:
        def build() -> AtomNativeModel:
            return AtomNativeModel(
                d_model=8, n_modes=8, n_atoms_max=32, max_payload_bytes=1,
                atomizer=Atomizer(max_span_bytes=1), field_max_rms=3.0,
                field_obligatory_hard=True, enable_merge=False, last_atom_readout=True,
            )

        model = build()
        model.surface.last_atom_dropout = 0.5
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.pt"
            model.save(path)
            other = build()
            other.load(path)
            self.assertEqual(float(other.surface.last_atom_dropout), 0.5)


if __name__ == "__main__":
    unittest.main()
