"""Unit tests: field_obligatory_readout cannot bypass α on main CE path."""
from __future__ import annotations
import unittest
import torch
from src.atom_native import AtomNativeModel
from src.io.atomizer import Atomizer

class FieldObligatoryReadoutTests(unittest.TestCase):
    """Main CE surface path must not bypass α via persist/atom_r alone."""

    def test_zeroing_alpha_moves_logits_with_shared_state_fixed(self) -> None:
        """Same atom_r/persistence; α vs zeros_like(α) ⇒ logit cosine < 0.99."""
        torch.manual_seed(42)
        model = AtomNativeModel(
            d_model=16,
            n_modes=16,
            n_atoms_max=64,
            max_payload_bytes=8,
            atomizer=Atomizer(max_span_bytes=8),
            field_max_rms=3.0,
            field_obligatory_readout=True,
        )
        model.eval()
        self.assertTrue(model.surface.field_obligatory_readout)
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
            logit_cos = float(torch.dot(f1, f0).item() / denom)
        self.assertLess(
            logit_cos,
            0.99,
            msg=f"obligatory readout failed: α vs zeros cosine={logit_cos:.6f}",
        )

    def test_obligatory_on_lowers_cosine_vs_off(self) -> None:
        """Turning obligatory on must lower α-vs-zeros cosine vs flag off."""
        torch.manual_seed(42)
        model = AtomNativeModel(
            d_model=16,
            n_modes=16,
            n_atoms_max=64,
            max_payload_bytes=8,
            atomizer=Atomizer(max_span_bytes=8),
            field_max_rms=3.0,
            field_obligatory_readout=False,
        )
        model.eval()
        with torch.no_grad():
            model.surface.skip_gate.fill_(-2.0)
            alpha = torch.randn(16, 16) * 0.4
            persist = torch.randn(16) * 2.0
            atom_r = torch.randn(16) * 2.0
            legacy = torch.randn(16)

            def _cos(flag: bool) -> float:
                model.surface.field_obligatory_readout = flag
                model.field_obligatory_readout = flag
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
                return float(torch.dot(f1, f0).item() / denom)

            cos_off = _cos(False)
            cos_on = _cos(True)
        self.assertLess(
            cos_on,
            cos_off - 0.01,
            msg=f"obligatory should lower cosine: off={cos_off:.6f} on={cos_on:.6f}",
        )
        self.assertLess(cos_on, 0.99, msg=f"obligatory on still too high: {cos_on:.6f}")

