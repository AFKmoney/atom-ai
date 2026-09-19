"""Speech-lock repairs: no unigram broadcast, soup not re-ingested."""

from __future__ import annotations

import unittest

import torch

from src.atom_native import (
    AtomNativeModel,
    AtomSurfaceHead,
    format_dialogue_prompt,
    speech_ok,
)
from src.io.atomizer import Atomizer


class SpeechLockUnitTests(unittest.TestCase):
    def test_dialogue_wrap(self) -> None:
        self.assertEqual(
            format_dialogue_prompt("Bonjour"),
            "Utilisateur: Bonjour\nAssistant:",
        )
        self.assertIn("Assistant:", format_dialogue_prompt("Utilisateur: Hi\nAssistant:"))

    def test_speech_ok_rejects_attractor_soup(self) -> None:
        self.assertFalse(speech_ok(b""))
        self.assertFalse(speech_ok(b"auiseateu : aernun e"))
        self.assertFalse(speech_ok(b"asssuatiur:i derouin"))
        self.assertTrue(speech_ok(b"Salut"))
        self.assertTrue(speech_ok(b"Je suis la."))

    def test_payload_copy_is_position_aligned_not_broadcast(self) -> None:
        torch.manual_seed(0)
        head = AtomSurfaceHead(d_model=8, max_payload_bytes=8, n_modes=8)
        a_only = torch.ones(16)
        a_hat = a_only / a_only.norm().clamp_min(1e-8)
        byte_logits, _ = head.payload_produce(
            [b"abc"],
            a_hat,
            a_only,
            atom_energies=[1.0],
        )
        self.assertGreater(
            float(byte_logits[0, ord("a")]),
            float(byte_logits[3, ord("a")]),
        )


class SpeechGenerateTests(unittest.TestCase):
    def test_generate_does_not_commit_soup(self) -> None:
        model = AtomNativeModel(
            d_model=8,
            n_modes=8,
            n_atoms_max=32,
            max_payload_bytes=8,
            atomizer=Atomizer(max_span_bytes=8),
        )
        model.eval()
        raw = model.generate_packets(
            "Bonjour",
            max_packets=4,
            deterministic=True,
            payload_copy=False,
            dialogue_wrap=True,
            speech_gate=True,
        )
        self.assertNotIn(b"auiseateu", raw)


if __name__ == "__main__":
    unittest.main()
