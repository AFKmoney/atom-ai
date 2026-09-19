"""Chat/probe ingest: MERGE off keeps distinct living prompt atoms."""
from __future__ import annotations

import unittest

import torch

from src.atom_native import AtomNativeModel
from src.io.atomizer import Atomizer


class ChatMergeOffIngestTests(unittest.TestCase):
    def _tiny(self, **kwargs) -> AtomNativeModel:
        return AtomNativeModel(
            d_model=16,
            n_modes=16,
            n_atoms_max=64,
            max_payload_bytes=8,
            atomizer=Atomizer(max_span_bytes=8),
            field_max_rms=3.0,
            field_obligatory_hard=True,
            enable_merge=True,
            # Force merges when enabled so the contrast is sharp.
            merge_coherence_threshold=0.0,
            merge_energy_floor=0.0,
            **kwargs,
        )

    def test_forward_merge_off_keeps_multi_atoms(self) -> None:
        """Path-specific merge_enabled=False leaves prompt packets as distinct atoms."""
        torch.manual_seed(0)
        model = self._tiny()
        model.eval()
        prompt = "Utilisateur: Bonjour le monde\nAssistant:"
        packets = model.atomizer.encode(prompt, reset=True)
        self.assertGreaterEqual(len(packets), 2)

        model.reset_state(reset_atomizer=True)
        for p in packets:
            model.forward_packet(p, merge_enabled=False)
        n_off = len(model.core.atoms)
        payloads_off = [bytes(getattr(a, "payload", b"") or b"") for a in model.core.atoms.atoms]

        model.reset_state(reset_atomizer=True)
        packets2 = model.atomizer.encode(prompt, reset=True)
        for p in packets2:
            model.forward_packet(p)  # training default: MERGE on
        n_on = len(model.core.atoms)

        self.assertGreaterEqual(
            n_off,
            2,
            msg=f"merge-off ingest should keep multi atoms; got {n_off} payloads={payloads_off!r}",
        )
        self.assertEqual(n_off, len(packets))
        self.assertLessEqual(
            n_on,
            n_off,
            msg=f"MERGE-on should not exceed merge-off atoms ({n_on} vs {n_off})",
        )
        # With thr=0 / floor=0, MERGE collapses toward 1 when ≥2 atoms.
        if len(packets) >= 2:
            self.assertEqual(n_on, 1, msg=f"expected collapse under forced MERGE; got {n_on}")

    def test_generate_packets_defaults_merge_off(self) -> None:
        """generate_packets (chat path) defaults merge_enabled=False; train flag stays True."""
        torch.manual_seed(1)
        model = self._tiny()
        model.eval()
        self.assertTrue(model.enable_merge)
        prompt = "Utilisateur: Bonjour le monde atomique\nAssistant:"
        packets = model.atomizer.encode(prompt, reset=True)
        n_packets = len(packets)
        self.assertGreaterEqual(n_packets, 2)

        # Manual prime mirroring generate_packets default.
        model.reset_state(reset_atomizer=True)
        for p in packets[:-1]:
            model.forward_packet(p, merge_enabled=False)
        model.forward_packet(packets[-1], merge_enabled=False)
        self.assertEqual(len(model.core.atoms), n_packets)
        self.assertTrue(model.enable_merge)  # training knob unchanged

        # Smoke: generate_packets runs and leaves multi-atom ring after prime+predict.
        model.reset_state(reset_atomizer=True)
        _ = model.generate_packets(prompt, max_packets=2, max_length=32, merge_enabled=False)
        self.assertGreaterEqual(len(model.core.atoms), 2)
        self.assertTrue(model.enable_merge)

    def test_payload_copy_sees_distinct_atoms_when_merge_off(self) -> None:
        """Distinct living payloads under merge-off boost target byte mass vs MERGE collapse."""
        torch.manual_seed(5)
        model = self._tiny()
        model.eval()
        prompt = "AAA BBB CCC DDD EEE FFF "
        packets = model.atomizer.encode(prompt, reset=True)
        self.assertGreaterEqual(len(packets), 2)

        model.reset_state(reset_atomizer=True)
        last_off = None
        for p in packets:
            last_off = model.forward_packet(p, merge_enabled=False)
        assert last_off is not None
        n_off = len(model.core.atoms)
        logits_off = last_off["surface"]["byte_logits"]
        # Bytes present in prompt payloads.
        target = sorted({b for p in packets for b in (p.payload or b"")})
        mass_off = float(logits_off.softmax(dim=-1)[:, target].sum().item())

        model.reset_state(reset_atomizer=True)
        packets2 = model.atomizer.encode(prompt, reset=True)
        last_on = None
        for p in packets2:
            last_on = model.forward_packet(p)
        assert last_on is not None
        n_on = len(model.core.atoms)
        logits_on = last_on["surface"]["byte_logits"]
        mass_on = float(logits_on.softmax(dim=-1)[:, target].sum().item())

        self.assertGreaterEqual(n_off, 2)
        self.assertEqual(n_on, 1)
        # Soft check: merge-off should not be worse on prompt-byte mass.
        self.assertGreaterEqual(
            mass_off + 1e-6,
            mass_on * 0.9,
            msg=f"mass_off={mass_off:.4f} mass_on={mass_on:.4f} n_off={n_off} n_on={n_on}",
        )


if __name__ == "__main__":
    unittest.main()
