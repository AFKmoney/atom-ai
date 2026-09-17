"""Regression: generation stays printable and surface head is state-sensitive."""

from __future__ import annotations

import string
import unittest

import torch

from src.atom_native import AtomNativeModel
from src.io.atomizer import Atomizer


PRINTABLE = set(string.printable) | set("àâäéèêëïîôùûüçÀÂÄÉÈÊËÏÎÔÙÛÜÇ«»—–’")


class GenerationSmokeTests(unittest.TestCase):
    def _tiny_trained(self) -> AtomNativeModel:
        torch.manual_seed(7)
        atomizer = Atomizer(max_span_bytes=8)
        text = (
            "Utilisateur: Bonjour\n"
            "Assistant: Salut ! Comment ca va ?\n\n"
            "Utilisateur: Qui es-tu ?\n"
            "Assistant: Je suis un modele atom-native.\n\n"
        ) * 3
        packets = atomizer.encode(text)
        model = AtomNativeModel(
            d_model=8,
            n_modes=8,
            n_atoms_max=64,
            max_payload_bytes=8,
            atomizer=Atomizer(max_span_bytes=8),
            field_max_rms=4.0,
        )
        opt = torch.optim.AdamW(model.trainable_parameters, lr=3e-3)
        model.train()
        for step in range(200):
            idx = step % (len(packets) - 1)
            if step % 50 == 0:
                model.reset_state(reset_atomizer=True)
            opt.zero_grad(set_to_none=True)
            loss, _ = model.transition_loss(packets[idx], packets[idx + 1])
            loss.backward()
            opt.step()
            model.stabilize_dynamics_parameters()
        model.eval()
        return model

    def test_generate_printable_and_long_enough(self) -> None:
        model = self._tiny_trained()
        raw = b""
        for seed in (0, 1, 2, 3, 4):
            torch.manual_seed(seed)
            raw = model.generate_packets(
                "Utilisateur: Bonjour\nAssistant:",
                max_packets=12,
                max_length=64,
                temperature=0.5,
                top_k=5,
                deterministic=False,
                prefer_printable=True,
            )
            if len(raw) > 8:
                break
        self.assertGreater(len(raw), 8, msg=f"short generations across seeds; last={raw!r}")
        text = raw.decode("utf-8", errors="replace")
        printable_frac = sum(ch in PRINTABLE for ch in text) / max(len(text), 1)
        self.assertGreaterEqual(
            printable_frac,
            0.75,
            msg=f"printable_frac={printable_frac:.3f} text={text!r}",
        )
        self.assertTrue(text.strip())

    def test_surface_norm_amplifies_state_signal(self) -> None:
        """LayerNorm makes distinct states yield distinct logits (bias no longer ties them)."""
        model = AtomNativeModel(
            d_model=8,
            n_modes=8,
            n_atoms_max=32,
            max_payload_bytes=8,
            atomizer=Atomizer(max_span_bytes=8),
        )
        with torch.no_grad():
            bias = model.surface.byte_decoder.bias.view(8, 256)
            bias.zero_()
            bias[:, ord(" ")] = 40.0  # legacy-scale bias
            s1 = torch.zeros(8)
            s1[0] = 0.01
            s2 = torch.zeros(8)
            s2[1] = 0.01
            # Raw linear without norm would be swamped; measure post-LayerNorm delta.
            model.surface.byte_decoder.bias.mul_(0.05)  # migration damp
            out1 = model.surface(s1)["byte_logits"]
            out2 = model.surface(s2)["byte_logits"]
            delta = (out1 - out2).abs().max().item()
        self.assertGreater(delta, 0.05, msg=f"state delta too small: {delta}")

    def test_infer_boundary_on_generated_payload(self) -> None:
        az = Atomizer(max_span_bytes=8)
        az.encode("Hello ")
        packet = az.packet_from_payload(b"world ")
        self.assertEqual(packet.boundary, "whitespace")
        packet2 = az.packet_from_payload(b"end.\n")
        self.assertEqual(packet2.boundary, "newline")


if __name__ == "__main__":
    unittest.main()
