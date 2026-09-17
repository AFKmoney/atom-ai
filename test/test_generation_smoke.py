"""Regression: generation stays printable and surface head is state-sensitive."""

from __future__ import annotations

import string
from pathlib import Path
import unittest

import torch

from src.atom_native import AtomNativeModel
from src.io.atomizer import Atomizer


PRINTABLE = set(string.printable) | set("àâäéèêëïîôùûüçÀÂÄÉÈÊËÏÎÔÙÛÜÇ«»—–’")


class GenerationSmokeTests(unittest.TestCase):
    def _tiny_trained(self) -> AtomNativeModel:
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
        model.eval()
        return model

    def test_generate_printable_and_long_enough(self) -> None:
        model = self._tiny_trained()
        torch.manual_seed(0)
        raw = model.generate_packets(
            "Utilisateur: Bonjour\nAssistant:",
            max_packets=12,
            max_length=64,
            temperature=0.5,
            top_k=5,
            deterministic=False,
            prefer_printable=True,
        )
        self.assertGreater(len(raw), 8)
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


class FieldReadoutSensitivityTests(unittest.TestCase):
    """Surface logits must track living field α across distinct prompts."""

    def test_synthetic_fields_move_surface_logits(self) -> None:
        """Distinct α tensors → surface logit cosine < 0.95 via field readout."""
        torch.manual_seed(0)
        model = AtomNativeModel(
            d_model=16,
            n_modes=16,
            n_atoms_max=64,
            max_payload_bytes=8,
            atomizer=Atomizer(max_span_bytes=8),
            field_max_rms=3.0,
        )
        model.eval()
        with torch.no_grad():
            # Untrained dynamics may leave α nearly collinear; feed spectral α directly.
            alpha1 = torch.randn(16, 16)
            alpha2 = torch.randn(16, 16)
            s1 = model.surface(alpha=alpha1)
            s2 = model.surface(alpha=alpha2)
            f1 = torch.cat([s1["byte_logits"].reshape(-1), s1["length_logits"].reshape(-1)])
            f2 = torch.cat([s2["byte_logits"].reshape(-1), s2["length_logits"].reshape(-1)])
            denom = float(f1.norm().item() * f2.norm().item())
            logit_cos = float(torch.dot(f1, f2).item() / denom)
        self.assertLess(
            logit_cos,
            0.95,
            msg=f"field readout failed to separate synthetic α: {logit_cos:.6f}",
        )

    def test_checkpoint_migrates_field_readout_and_separates_logits(self) -> None:
        """Old surface weights load; new field path activates; logits separate."""
        ckpt = Path(__file__).resolve().parents[1] / (
            "checkpoints/atom_native_chat_talk/atom_native.pt"
        )
        if not ckpt.exists() or ckpt.stat().st_size == 0:
            self.skipTest(f"checkpoint missing: {ckpt}")
        model = AtomNativeModel(
            d_model=64,
            n_modes=64,
            n_atoms_max=512,
            max_payload_bytes=16,
            field_max_rms=3.0,
        )
        training = model.load(ckpt) or {}
        self.assertTrue(
            bool(training.get("field_readout_migrated")),
            msg=f"expected field_readout_migrated, got {training.get('field_readout_migrated')}",
        )
        model.eval()
        prompts = [
            "Utilisateur: Bonjour\nAssistant:",
            "Utilisateur: Qui es-tu ?\nAssistant:",
        ]
        flats: list[torch.Tensor] = []
        with torch.no_grad():
            for prompt in prompts:
                model.reset_state(reset_atomizer=True)
                packets = model.atomizer.encode(prompt, reset=True)
                out = None
                for packet in packets:
                    out = model.forward_packet(packet)
                assert out is not None
                s = out["surface"]
                flats.append(
                    torch.cat([s["byte_logits"].reshape(-1), s["length_logits"].reshape(-1)])
                )
        a, b = flats[0], flats[1]
        denom = float(a.norm().item() * b.norm().item())
        logit_cos = float(torch.dot(a, b).item() / denom)
        self.assertLess(
            logit_cos,
            0.95,
            msg=f"migrated ckpt logit cosine still too high: {logit_cos:.6f}",
        )



if __name__ == "__main__":
    unittest.main()
