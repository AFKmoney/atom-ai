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

class LastAtomScaleTests(unittest.TestCase):
    def test_default_scale_is_one(self) -> None:
        head = AtomSurfaceHead(d_model=8, max_payload_bytes=1, last_atom_readout=True)
        self.assertEqual(float(head.last_atom_scale), 1.0)

    def test_scale_multiplies_hard_mix_la(self) -> None:
        torch.manual_seed(0)
        model = _tiny_byte_tick(last_atom_readout=True)
        model.eval()
        atomizer = Atomizer(max_span_bytes=1)
        packets = atomizer.encode("ab")
        surf = model.surface

        def logits_at(scale: float):
            surf.last_atom_scale = float(scale)
            model.reset_state(reset_atomizer=True)
            with torch.no_grad():
                model.forward_packet(packets[0])
                out = model.forward_packet(packets[1])["surface"]["byte_logits"]
            return out.clone()

        out1 = logits_at(1.0)
        out0 = logits_at(0.0)
        out_half = logits_at(0.5)
        delta_full = out1 - out0
        delta_half = out_half - out0
        self.assertTrue(torch.isfinite(delta_full).all().item())
        self.assertGreater(float(delta_full.pow(2).mean().sqrt()), 1e-6)
        self.assertTrue(torch.allclose(delta_half, 0.5 * delta_full, rtol=1e-4, atol=1e-5))




class LastAtomScaleAdaptiveTests(unittest.TestCase):
    def test_default_s_min_is_zero(self) -> None:
        head = AtomSurfaceHead(d_model=8, max_payload_bytes=1, last_atom_readout=True)
        self.assertEqual(float(head.last_atom_scale_min), 0.0)

    def test_adaptive_equalizes_la_vs_alpha_rms(self) -> None:
        """Synthetic tensors: S_eff * la_rms ≈ α_rms (true RMS match, S_min=0)."""
        torch.manual_seed(0)
        head = AtomSurfaceHead(d_model=8, max_payload_bytes=1, last_atom_readout=True)
        head.last_atom_scale_adaptive = True
        # default S_min=0.0, S_max=1.0
        self.assertEqual(float(head.last_atom_scale_min), 0.0)
        head.last_atom_scale_max = 1.0
        # α small, la large → S_eff ≈ α/la << 0.05 (former floor); effective la ≈ α
        alpha_term = torch.randn(1, 256) * 0.5
        la_raw = torch.randn(1, 256) * 20.0
        s_eff = head.effective_last_atom_scale(alpha_term, la_raw)
        alpha_rms = float(alpha_term.pow(2).mean().sqrt())
        la_raw_rms = float(la_raw.pow(2).mean().sqrt())
        la_eff_rms = float((s_eff * la_raw).pow(2).mean().sqrt())
        expected = alpha_rms / (la_raw_rms + 1e-8)
        expected = max(0.0, min(1.0, expected))
        self.assertLess(expected, 0.05)  # would have hit former floor
        self.assertAlmostEqual(float(s_eff), expected, places=5)
        self.assertLess(float(s_eff), 0.05)
        self.assertAlmostEqual(la_eff_rms / alpha_rms, 1.0, places=2)

    def test_adaptive_clamps_to_min_max(self) -> None:
        head = AtomSurfaceHead(d_model=8, max_payload_bytes=1, last_atom_readout=True)
        head.last_atom_scale_adaptive = True
        head.last_atom_scale_min = 0.0
        head.last_atom_scale_max = 1.0
        # la tiny vs α → raw ratio >> 1 → clamp to S_max
        alpha_term = torch.ones(1, 256)
        la_raw = torch.ones(1, 256) * 1e-6
        s_hi = float(head.effective_last_atom_scale(alpha_term, la_raw))
        self.assertAlmostEqual(s_hi, 1.0, places=5)
        # la huge vs α → raw ratio → 0 → clamp to S_min=0
        la_huge = torch.ones(1, 256) * 1e6
        s_lo = float(head.effective_last_atom_scale(alpha_term, la_huge))
        self.assertAlmostEqual(s_lo, 0.0, places=5)
        # configurable floor still honored when set above 0
        head.last_atom_scale_min = 0.05
        s_floor = float(head.effective_last_atom_scale(alpha_term, la_huge))
        self.assertAlmostEqual(s_floor, 0.05, places=5)

    def test_fixed_scale_still_linear_when_adaptive_off(self) -> None:
        torch.manual_seed(1)
        model = _tiny_byte_tick(last_atom_readout=True)
        model.eval()
        model.surface.last_atom_scale_adaptive = False
        atomizer = Atomizer(max_span_bytes=1)
        packets = atomizer.encode("ab")
        surf = model.surface

        def logits_at(scale: float):
            surf.last_atom_scale = float(scale)
            model.reset_state(reset_atomizer=True)
            with torch.no_grad():
                model.forward_packet(packets[0])
                out = model.forward_packet(packets[1])["surface"]["byte_logits"]
            return out.clone()

        out1 = logits_at(1.0)
        out0 = logits_at(0.0)
        out_half = logits_at(0.5)
        self.assertTrue(
            torch.allclose(out_half - out0, 0.5 * (out1 - out0), rtol=1e-4, atol=1e-5)
        )

    def test_adaptive_scale_detached_from_grad(self) -> None:
        torch.manual_seed(0)
        head = AtomSurfaceHead(d_model=8, max_payload_bytes=1, last_atom_readout=True)
        head.last_atom_scale_adaptive = True
        alpha_term = torch.randn(1, 256, requires_grad=True)
        la_raw = torch.randn(1, 256, requires_grad=True)
        s_eff = head.effective_last_atom_scale(alpha_term, la_raw)
        # Scale factor itself must not create a graph back to inputs.
        self.assertFalse(s_eff.requires_grad)
        loss = (s_eff * la_raw).sum()
        loss.backward()
        self.assertIsNotNone(la_raw.grad)
        # alpha_term only entered via detached RMS → no grad
        self.assertIsNone(alpha_term.grad)


if __name__ == "__main__":
    unittest.main()
