"""Unit tests: field_obligatory_hard freezes non-α bypass; α must move logits."""
from __future__ import annotations

import unittest

import torch

from src.atom_native import AtomNativeModel
from src.io.atomizer import Atomizer


class FieldObligatoryHardTests(unittest.TestCase):
    """Hard mode: mix floor 1.0 + zero/frozen non-α residual on CE surface."""

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

    def _logit_cos_alpha_vs_zeros(self, model: AtomNativeModel) -> float:
        with torch.no_grad():
            alpha = torch.randn(16, 16) * 0.5
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
            denom = float(f1.norm().item() * f0.norm().item())
            self.assertGreater(denom, 1e-12)
            return float(torch.dot(f1, f0).item() / denom)

    def test_hard_zeroing_alpha_moves_logits(self) -> None:
        """Same atom_r/persist; α vs zeros ⇒ logit cosine ≪ 0.99 under hard."""
        torch.manual_seed(42)
        model = self._tiny(field_obligatory_hard=True)
        model.eval()
        self.assertTrue(model.field_obligatory_hard)
        self.assertTrue(model.surface.field_obligatory_hard)
        self.assertTrue(model.surface.field_obligatory_readout)
        self.assertGreaterEqual(float(model.surface.obl_mix_floor), 1.0 - 1e-9)
        logit_cos = self._logit_cos_alpha_vs_zeros(model)
        self.assertLess(
            logit_cos,
            0.99,
            msg=f"hard obligatory failed: α vs zeros cosine={logit_cos:.6f}",
        )

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
        # Frozen α→logit buffers stay buffers (not Parameters).
        self.assertFalse(isinstance(surf.alpha_byte_frozen, torch.nn.Parameter))
        self.assertFalse(isinstance(surf.alpha_length_frozen, torch.nn.Parameter))

    def test_hard_stricter_than_soft_on_short_ce_contract(self) -> None:
        """Document expected short-CE behavior: hard keeps α in the logit path.

        Soft (floor 0.35) can re-collapse under long CE (see OBLIGATORY_TRAIN_25K).
        Hard forces mix≡1 and zeros the non-α residual so short CE cannot
        route around α via decoder/skip. This unit checks the structural
        contract, not a long train.
        """
        torch.manual_seed(7)
        soft = self._tiny(field_obligatory_readout=True)
        hard = self._tiny(field_obligatory_hard=True)
        soft.eval()
        hard.eval()
        cos_soft = self._logit_cos_alpha_vs_zeros(soft)
        cos_hard = self._logit_cos_alpha_vs_zeros(hard)
        self.assertLess(cos_soft, 0.99)
        self.assertLess(cos_hard, 0.99)
        # Hard should not be *less* α-sensitive than soft on this probe.
        self.assertLessEqual(cos_hard, cos_soft + 0.05)


if __name__ == "__main__":
    unittest.main()
