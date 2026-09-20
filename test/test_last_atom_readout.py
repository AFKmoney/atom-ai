"""Unit tests: byte-tick substrate + last-atom readout + dentate 2-gram."""
from __future__ import annotations

import unittest

import torch

from src.atom_native import AtomNativeModel, AtomSurfaceHead, LastAtomReadout
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
    )
    args.update(kwargs)
    return AtomNativeModel(**args)


class ByteTickAtomizerTests(unittest.TestCase):
    def test_single_byte_packets_lossless(self) -> None:
        raw = "Bonjour, tu vas bien ?\nOui, merci !".encode("utf-8")
        packets = Atomizer(max_span_bytes=1).encode_bytes(raw)
        self.assertEqual(len(packets), len(raw))
        self.assertTrue(all(len(p.payload) == 1 for p in packets))
        self.assertEqual(b"".join(p.payload for p in packets), raw)

    def test_multibyte_split_mid_codepoint(self) -> None:
        # Byte-tick is a pure byte stream: e-acute is 2 ticks.
        packets = Atomizer(max_span_bytes=1).encode("caf\u00e9")
        self.assertEqual(b"".join(p.payload for p in packets), "caf\u00e9".encode("utf-8"))
        self.assertEqual(len(packets), 5)

    def test_floor_is_one(self) -> None:
        with self.assertRaises(ValueError):
            Atomizer(max_span_bytes=0)

    def test_commit_boundary_never_generated(self) -> None:
        atomizer = Atomizer(max_span_bytes=1)
        self.assertEqual(atomizer.packet_from_payload(b"a").boundary, "max_span")
        self.assertEqual(atomizer.packet_from_payload(b"\n").boundary, "newline")
        self.assertEqual(atomizer.packet_from_payload(b",").boundary, "punctuation")
        self.assertEqual(atomizer.packet_from_payload(b" ").boundary, "whitespace")


class LastAtomReadoutTests(unittest.TestCase):
    def test_requires_byte_tick(self) -> None:
        with self.assertRaises(ValueError):
            AtomSurfaceHead(d_model=8, max_payload_bytes=4, last_atom_readout=True)
        head = AtomSurfaceHead(d_model=8, max_payload_bytes=1, last_atom_readout=True)
        self.assertIsNotNone(head.last_atom)

    def test_forward_shape_finite(self) -> None:
        torch.manual_seed(0)
        readout = LastAtomReadout(d_model=8)
        out = readout(ord("a"), ord("b"), torch.randn(8))
        self.assertEqual(tuple(out.shape), (256,))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_dentate_keep_is_quarter(self) -> None:
        readout = LastAtomReadout(d_model=8)
        hidden = 4 * 2 * 8
        self.assertEqual(readout.dentate_keep(hidden), hidden // 4)

    def test_dentate_sparsity_exact(self) -> None:
        torch.manual_seed(0)
        readout = LastAtomReadout(d_model=8)
        readout.eval()
        pair = torch.randn(1, 2 * readout.embed_dim)
        with torch.no_grad():
            h = torch.relu(readout.expand(pair))
            keep = readout.dentate_keep(h.shape[-1])
            vals, idx = torch.topk(h, keep, dim=-1)
            sparse = torch.zeros_like(h).scatter(-1, idx, vals)
        # Enough positive mass that top-k keeps exactly `keep` nonzeros.
        self.assertGreater((h > 0).sum().item(), keep)
        self.assertEqual(int((sparse != 0).sum().item()), keep)

    def test_sensitive_to_last_byte(self) -> None:
        torch.manual_seed(0)
        model = _tiny_byte_tick(last_atom_readout=True)
        model.eval()
        atomizer = Atomizer(max_span_bytes=1)
        pa = atomizer.encode("a", reset=True)[0]
        pb = atomizer.encode("b", reset=True)[0]
        with torch.no_grad():
            model.reset_state(reset_atomizer=True)
            model.eval()
            la = model.forward_packet(pa)["surface"]["byte_logits"]
            model.reset_state(reset_atomizer=True)
            model.eval()
            lb = model.forward_packet(pb)["surface"]["byte_logits"]
            model.reset_state(reset_atomizer=True)
            model.eval()
            la2 = model.forward_packet(pa)["surface"]["byte_logits"]
        self.assertFalse(torch.allclose(la, lb))
        self.assertTrue(torch.allclose(la, la2))

    def test_gradients_flow_through_readout(self) -> None:
        torch.manual_seed(0)
        model = _tiny_byte_tick(last_atom_readout=True)
        model.train()
        packets = Atomizer(max_span_bytes=1).encode("Bonjour")
        loss, _ = model.transition_loss(packets[0], packets[1])
        loss.backward()
        assert model.surface.last_atom is not None
        self.assertIsNotNone(model.surface.last_atom.head.weight.grad)
        self.assertIsNotNone(model.surface.last_atom.byte_embed.weight.grad)
        self.assertTrue(torch.isfinite(model.surface.last_atom.head.weight.grad).all().item())

    def test_byte_tick_generate_with_gate(self) -> None:
        # Accumulated gate (len>=8): 6 one-byte packets must all emit.
        torch.manual_seed(0)
        model = _tiny_byte_tick(last_atom_readout=True)
        model.eval()
        raw = model.generate_packets(
            "Bonjour",
            max_packets=6,
            deterministic=True,
            payload_copy=False,
            dialogue_wrap=True,
            speech_gate=True,
            merge_enabled=False,
        )
        self.assertEqual(len(raw), 6)


if __name__ == "__main__":
    unittest.main()
