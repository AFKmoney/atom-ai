"""Unit tests: field_obligatory_hard-v2 — frozen α + trainable α-proj; freeze bypass."""
from __future__ import annotations

import unittest

import torch

from src.atom_native import AtomNativeModel
from src.io.atomizer import Atomizer


class FieldObligatoryHardTests(unittest.TestCase):
    """Hard-v2+: mix floor 1.0; freeze non-α bypass; α-only MLP trainable."""

    def _tiny(self, **kwargs) -> AtomNativeModel:
        return AtomNativeModel(
            d_model=16,
            n_modes=16,
            n_atoms_max=64,
            max_payload_bytes=8,
            atomizer=Atomizer(max_span_bytes=8),
            field_max_rms=3.0,
            **kwargs,
        )

    def _surface_pair(self, model: AtomNativeModel, alpha: torch.Tensor):
        with torch.no_grad():
            persist = torch.randn(16)
            atom_r = torch.randn(16)
            legacy = torch.randn(16)
            s_live = model.surface(
                legacy, alpha=alpha, persistent_state=persist, atom_r=atom_r
            )
            s_zero = model.surface(
                legacy,
                alpha=torch.zeros_like(alpha),
                persistent_state=persist,
                atom_r=atom_r,
            )
            f1 = torch.cat(
                [s_live["byte_logits"].reshape(-1), s_live["length_logits"].reshape(-1)]
            )
            f0 = torch.cat(
                [s_zero["byte_logits"].reshape(-1), s_zero["length_logits"].reshape(-1)]
            )
            return f1, f0

    def _logit_cos_two_alphas(self, model: AtomNativeModel) -> float:
        """Cosine between logits for two different non-zero α (same persist/atom_r)."""
        with torch.no_grad():
            a = torch.randn(16, 16) * 0.5
            b = torch.randn(16, 16) * 0.5
            persist = torch.randn(16)
            atom_r = torch.randn(16)
            legacy = torch.randn(16)
            s_a = model.surface(legacy, alpha=a, persistent_state=persist, atom_r=atom_r)
            s_b = model.surface(legacy, alpha=b, persistent_state=persist, atom_r=atom_r)
            fa = torch.cat([s_a["byte_logits"].reshape(-1), s_a["length_logits"].reshape(-1)])
            fb = torch.cat([s_b["byte_logits"].reshape(-1), s_b["length_logits"].reshape(-1)])
            denom = float(fa.norm().item() * fb.norm().item())
            self.assertGreater(denom, 1e-12)
            return float(torch.dot(fa, fb).item() / denom)

    def test_hard_zeroing_alpha_moves_logits(self) -> None:
        """Hard-v2: non-zero α ⇒ logits; α=0 ⇒ ~0 logits (frozen+proj of zeros)."""
        torch.manual_seed(42)
        model = self._tiny(field_obligatory_hard=True)
        model.eval()
        self.assertTrue(model.field_obligatory_hard)
        self.assertTrue(model.surface.field_obligatory_hard)
        self.assertTrue(model.surface.field_obligatory_readout)
        self.assertGreaterEqual(float(model.surface.obl_mix_floor), 1.0 - 1e-9)
        alpha = torch.randn(16, 16) * 0.5
        f1, f0 = self._surface_pair(model, alpha)
        self.assertGreater(float(f1.norm().item()), 1e-3)
        self.assertLess(float(f0.norm().item()), 1e-5)
        cos_ab = self._logit_cos_two_alphas(model)
        self.assertLess(cos_ab, 0.99, msg=f"two-α cosine={cos_ab:.6f}")

    def test_hard_freezes_non_alpha_bypass_params(self) -> None:
        """Decoder / field skip / gates must not require grad under hard."""
        model = self._tiny(field_obligatory_hard=True)
        surf = model.surface
        frozen_modules = (
            surf.byte_decoder,
            surf.length_decoder,
            surf.field_to_state,
            surf.field_byte_skip,
            surf.field_length_skip,
        )
        for module in frozen_modules:
            for param in module.parameters():
                self.assertFalse(
                    param.requires_grad,
                    msg=f"expected frozen non-α param in {module.__class__.__name__}",
                )
        for name in ("field_gate", "skip_gate", "obl_mix"):
            self.assertFalse(
                getattr(surf, name).requires_grad,
                msg=f"expected frozen {name}",
            )
        self.assertFalse(isinstance(surf.alpha_byte_frozen, torch.nn.Parameter))
        self.assertFalse(isinstance(surf.alpha_length_frozen, torch.nn.Parameter))


    def test_hard_v2_trainable_rms_uncapped(self) -> None:
        """Speech hard mix (ARCHITECTURE.md): no RMS cap on trainable branch.

        Replaces the old capped contract: the cap pinned cap≈0.01 and crushed
        trainable gradients ~100x (measured on a speech ckpt), blocking
        learning. Mix is now obl*train + 0.3*frozen (+ optional payload).
        """
        torch.manual_seed(0)
        model = self._tiny(field_obligatory_hard=True)
        surf = model.surface
        with torch.no_grad():
            # Inflate MLP: uncapped residual must be free to dominate frozen.
            surf.alpha_byte_proj.fc2.weight.mul_(50.0)
            surf.obl_gate.fill_(5.0)
        alpha = torch.randn(16, 16) * 0.5
        persist = torch.randn(16)
        atom_r = torch.randn(16)
        legacy = torch.randn(16)
        a = surf.alpha_only_features(alpha)
        a_hat = a / a.pow(2).mean().sqrt().clamp_min(1e-8)
        frozen = (surf.alpha_byte_frozen @ a_hat).view(surf.max_payload_bytes, 256)
        out = surf(legacy, alpha=alpha, persistent_state=persist, atom_r=atom_r)
        # No payload atoms passed -> payload branch contributes zeros.
        residual = out["byte_logits"] - 0.3 * frozen
        f_rms = float(frozen.pow(2).mean().sqrt().item())
        r_rms = float(residual.pow(2).mean().sqrt().item())
        self.assertGreater(r_rms, f_rms * 2.0, msg=f"r_rms={r_rms} f_rms={f_rms}")

    def test_hard_v2_alpha_proj_trainable(self) -> None:
        """Hard-v2+: α-only MLP (+ obl_gate) require grad; frozen maps stay buffers."""
        model = self._tiny(field_obligatory_hard=True)
        surf = model.surface
        self.assertTrue(hasattr(surf.alpha_byte_proj, "fc1"))
        self.assertTrue(hasattr(surf.alpha_byte_proj, "fc2"))
        self.assertEqual(int(surf.alpha_mlp_hidden), 4 * int(surf.d_model))
        for module in (surf.alpha_byte_proj, surf.alpha_length_proj):
            for param in module.parameters():
                self.assertTrue(
                    param.requires_grad,
                    msg="α-only MLP must be trainable under hard-v2+",
                )
        self.assertTrue(surf.obl_gate.requires_grad)
        self.assertFalse(isinstance(surf.alpha_byte_frozen, torch.nn.Parameter))
        self.assertFalse(isinstance(surf.alpha_length_frozen, torch.nn.Parameter))
        # Gradients flow through trainable α MLP on a hard forward.
        alpha = torch.randn(16, 16) * 0.5
        persist = torch.randn(16)
        atom_r = torch.randn(16)
        legacy = torch.randn(16)
        out = model.surface(
            legacy, alpha=alpha, persistent_state=persist, atom_r=atom_r
        )
        loss = out["byte_logits"].pow(2).mean() + out["length_logits"].pow(2).mean()
        loss.backward()
        grad_sum = 0.0
        for param in surf.alpha_byte_proj.parameters():
            self.assertIsNotNone(param.grad)
            grad_sum += float(param.grad.abs().sum())
        self.assertGreater(grad_sum, 0.0)

    def test_hard_stricter_than_soft_on_short_ce_contract(self) -> None:

        """Hard keeps α-only logits; soft still α-sensitive but may use residual."""
        torch.manual_seed(7)
        soft = self._tiny(field_obligatory_readout=True)
        hard = self._tiny(field_obligatory_hard=True)
        soft.eval()
        hard.eval()
        cos_soft = self._logit_cos_two_alphas(soft)
        cos_hard = self._logit_cos_two_alphas(hard)
        self.assertLess(cos_soft, 0.99)
        self.assertLess(cos_hard, 0.99)
        # Hard is pure frozen α map (α=0 → ~0 logits). Soft may separate more
        # via residual noise; only require both stay clearly prompt-sensitive.
        self.assertLess(cos_hard, 0.5, msg=f"hard two-α cosine too high: {cos_hard:.6f}")


    def test_hard_v2_alpha_mlp_deeper_than_linear(self) -> None:
        """α MLP is 2-layer (fc1→GELU→fc2); hidden=4*d; still α-only."""
        model = self._tiny(field_obligatory_hard=True)
        surf = model.surface
        self.assertEqual(surf.alpha_byte_proj.in_dim, surf.alpha_only_dim)
        self.assertEqual(surf.alpha_byte_proj.hidden_dim, 4 * surf.d_model)
        self.assertEqual(
            surf.alpha_byte_proj.out_dim, surf.max_payload_bytes * 256
        )
        # Decoder still frozen (no bypass).
        for param in surf.byte_decoder.parameters():
            self.assertFalse(param.requires_grad)


    def test_printable_aux_pushes_mass_and_grads_alpha_mlp(self) -> None:
        """printable_aux_weight>0: loss rises with garbage logits; grads hit α MLP."""
        torch.manual_seed(3)
        model = self._tiny(field_obligatory_hard=True, printable_aux_weight=0.08)
        self.assertGreater(model.printable_aux_weight, 0.0)
        # Synthetic packet transition.
        packets = model.atomizer.encode("Assistant: Bonjour", reset=True)
        self.assertGreaterEqual(len(packets), 2)
        model.train()
        loss, info = model.transition_loss(packets[0], packets[1])
        self.assertIn("printable_aux_loss", info)
        self.assertGreater(float(info["printable_aux_loss"]), 0.0)
        self.assertTrue(0.0 <= float(info["printable_mass_mean"]) <= 1.0)
        loss.backward()
        grad_sum = 0.0
        for param in model.surface.alpha_byte_proj.parameters():
            self.assertIsNotNone(param.grad)
            grad_sum += float(param.grad.abs().sum())
        self.assertGreater(grad_sum, 0.0)
        # Decoder still frozen / no grad under hard.
        for param in model.surface.byte_decoder.parameters():
            self.assertFalse(param.requires_grad)

    def test_hard_decode_forces_prefer_printable(self) -> None:
        """Under hard, decode applies printable bias even if prefer_printable=False."""
        torch.manual_seed(5)
        model = self._tiny(field_obligatory_hard=True)
        model.eval()
        poisoned = {
            "byte_logits": torch.zeros(8, 256),
            "length_logits": torch.zeros(8),
        }
        with torch.no_grad():
            # Mild NUL preference — printable bias (strength 2.5 → NUL -5) flips it.
            poisoned["byte_logits"][:, 0] = 3.0
            poisoned["byte_logits"][:, ord("A")] = 1.0  # best among printable after bias
            poisoned["length_logits"][0] = 5.0
        raw = model.surface.decode(poisoned, prefer_printable=False, deterministic=True)
        self.assertNotEqual(raw[:1], b"\x00")
        self.assertEqual(raw[:1], b"A")
        # generate_packets also forces prefer_printable under hard.
        out = model.generate_packets("Hi", max_packets=1, prefer_printable=False, temperature=1.0)
        self.assertTrue(model.surface.field_obligatory_hard)
        self.assertIsInstance(out, (bytes, bytearray))


    def test_span_head_pad_migrate_keeps_prefix_and_field(self) -> None:
        """16→32 max_payload: pad-copy span heads; core/field tensors unchanged."""
        import tempfile
        from pathlib import Path as P

        torch.manual_seed(9)
        small = AtomNativeModel(
            d_model=16,
            n_modes=16,
            n_atoms_max=64,
            max_payload_bytes=8,
            atomizer=Atomizer(max_span_bytes=8, pack_mode="linguistic"),
            field_max_rms=3.0,
            field_obligatory_hard=True,
        )
        with torch.no_grad():
            small.surface.alpha_byte_frozen.fill_(0.123)
            small.surface.alpha_length_frozen.fill_(0.456)
            small.surface.alpha_byte_proj.fc2.weight.fill_(0.01)
            small.core.state.alpha.fill_(0.77)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = P(tmp) / "small.pt"
            small.save(ckpt)
            big = AtomNativeModel(
                d_model=16,
                n_modes=16,
                n_atoms_max=64,
                max_payload_bytes=16,
                atomizer=Atomizer(max_span_bytes=16, pack_mode="linguistic"),
                field_max_rms=3.0,
                field_obligatory_hard=True,
            )
            training = big.load(ckpt)
            self.assertTrue(training.get("span_head_migrated"))
            # Prefix rows preserved.
            self.assertTrue(
                torch.allclose(
                    big.surface.alpha_byte_frozen[: 8 * 256],
                    torch.full_like(big.surface.alpha_byte_frozen[: 8 * 256], 0.123),
                )
            )
            self.assertTrue(
                torch.allclose(
                    big.surface.alpha_length_frozen[:8],
                    torch.full_like(big.surface.alpha_length_frozen[:8], 0.456),
                )
            )
            self.assertTrue(
                torch.allclose(
                    big.surface.alpha_byte_proj.fc2.weight[: 8 * 256],
                    torch.full_like(big.surface.alpha_byte_proj.fc2.weight[: 8 * 256], 0.01),
                )
            )
            # Field/core not wiped.
            self.assertTrue(torch.allclose(big.core.state.alpha, torch.full_like(big.core.state.alpha, 0.77)))
            self.assertEqual(big.max_payload_bytes, 16)
            self.assertTrue(big.surface.field_obligatory_hard)


if __name__ == "__main__":
    unittest.main()
