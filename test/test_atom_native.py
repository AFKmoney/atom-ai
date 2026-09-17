"""Tests for the atom-native input contract and adapter."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.atom_native import AtomNativeModel, FieldAmplitudeController
from src.io.atomizer import Atomizer


class AtomizerTests(unittest.TestCase):
    def test_round_trip_is_exact_for_utf8(self) -> None:
        text = "ATOM: déjà vu — 42\nfluide."
        atomizer = Atomizer(max_span_bytes=16)
        packets = atomizer.encode(text)
        reconstructed = b"".join(packet.payload for packet in packets).decode("utf-8")
        self.assertEqual(reconstructed, text)
        self.assertGreater(len(packets), 1)
        self.assertTrue(all(packet.features.shape == (320,) for packet in packets))

    def test_same_surface_span_can_depend_on_context(self) -> None:
        first = Atomizer(max_span_bytes=16).encode("a world ")
        second = Atomizer(max_span_bytes=16).encode("b world ")
        first_world = next(packet for packet in first if packet.payload == b"world ")
        second_world = next(packet for packet in second if packet.payload == b"world ")
        self.assertFalse(torch.allclose(first_world.features, second_world.features))

    def test_chunked_stream_is_equivalent_and_lossless(self) -> None:
        text = "alpha beta — gamma\n delta"
        one_shot = Atomizer(max_span_bytes=8).encode(text)
        chunked = list(Atomizer(max_span_bytes=8).stream(["alpha ", "beta ", "— ", "gamma\n", " delta"]))
        self.assertEqual(b"".join(packet.payload for packet in one_shot), text.encode("utf-8"))
        self.assertEqual(b"".join(packet.payload for packet in chunked), text.encode("utf-8"))

    def test_state_restore_preserves_next_packet_features(self) -> None:
        source = Atomizer(max_span_bytes=16)
        source.encode("prefix ")
        saved = source.state_dict()
        expected = source.packet_from_payload(b"next ")

        restored = Atomizer(max_span_bytes=16)
        restored.load_state_dict(saved)
        actual = restored.packet_from_payload(b"next ")
        torch.testing.assert_close(expected.features, actual.features)
        self.assertEqual(expected.start, actual.start)
        self.assertEqual(expected.end, actual.end)

    def test_pending_buffer_is_checkpointable(self) -> None:
        source = Atomizer(max_span_bytes=16)
        source.encode_bytes(b"partial", reset=True, flush=False)
        saved = source.state_dict()
        expected = source.encode_bytes(b" end ", reset=False, flush=True)

        restored = Atomizer(max_span_bytes=16)
        restored.load_state_dict(saved)
        actual = restored.encode_bytes(b" end ", reset=False, flush=True)
        self.assertEqual([packet.payload for packet in expected], [packet.payload for packet in actual])


class AtomNativeModelTests(unittest.TestCase):
    def test_field_controller_caps_only_large_rms(self) -> None:
        controller = FieldAmplitudeController(max_rms=2.0)
        field, info = controller(torch.ones(8, 8) * 10.0)
        self.assertLessEqual(info["field_rms_after"], 2.001)
        self.assertLess(info["field_scale"], 1.0)
        small, small_info = controller(torch.ones(8, 8) * 0.5)
        torch.testing.assert_close(small, torch.ones(8, 8) * 0.5)
        self.assertEqual(small_info["field_scale"], 1.0)

    def test_dynamics_parameter_projection(self) -> None:
        model = AtomNativeModel(
            d_model=4,
            n_modes=4,
            n_atoms_max=32,
            max_payload_bytes=12,
            field_max_rms=2.0,
            energy_decay_bounds=(0.9, 0.999),
            atomizer=Atomizer(max_span_bytes=12),
        )
        with torch.no_grad():
            model.core.dynamics.dynamics.energy_decay.fill_(2.0)
            model.core.dynamics.dynamics.coupling_scale.fill_(4.0)
            model.core.dynamics.dynamics.phase_sync.fill_(3.0)
        state = model.stabilize_dynamics_parameters()
        self.assertAlmostEqual(state["energy_decay"], 0.999, places=5)
        self.assertAlmostEqual(state["coupling_scale"], 2.0, places=5)
        self.assertAlmostEqual(state["phase_sync"], 1.0, places=5)

    def test_forward_loss_backward_and_checkpoint_reload(self) -> None:
        atomizer = Atomizer(max_span_bytes=12)
        packets = atomizer.encode("alpha beta gamma delta epsilon.")
        self.assertGreaterEqual(len(packets), 3)

        model = AtomNativeModel(
            d_model=4,
            n_modes=4,
            n_atoms_max=32,
            max_payload_bytes=12,
            atomizer=Atomizer(max_span_bytes=12),
        )
        optimizer = torch.optim.AdamW(model.trainable_parameters, lr=1e-3)
        loss, info = model.transition_loss(packets[0], packets[1])
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(info["surface"]["byte_logits"]).all())
        loss.backward()
        optimizer.step()
        self.assertGreater(len(model.core.atoms), 0)
        self.assertTrue(all(torch.isfinite(parameter).all() for parameter in model.parameters()))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "atom_native.pt"
            model.save(path)
            reloaded = AtomNativeModel(
                d_model=4,
                n_modes=4,
                n_atoms_max=32,
                max_payload_bytes=12,
                atomizer=Atomizer(max_span_bytes=12),
            )
            reloaded.load(path)
            self.assertEqual(len(reloaded.core.atoms), len(model.core.atoms))
            self.assertEqual(reloaded.core.state.n_modes, 4)
            self.assertTrue(torch.isfinite(reloaded.core.state.alpha).all())


if __name__ == "__main__":
    unittest.main()
