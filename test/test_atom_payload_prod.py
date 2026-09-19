"""Unit tests: atom-payload production under field_obligatory_hard."""
from __future__ import annotations

import unittest

import torch

from src.atom_native import AtomNativeModel
from src.io.atomizer import Atomizer
from src.toroidal.atom import ToroidalAtom


class AtomPayloadProdTests(unittest.TestCase):
    def _tiny(self, **kwargs) -> AtomNativeModel:
        return AtomNativeModel(
            d_model=16,
            n_modes=16,
            n_atoms_max=64,
            max_payload_bytes=8,
            atomizer=Atomizer(max_span_bytes=8),
            field_max_rms=3.0,
            field_obligatory_hard=True,
            **kwargs,
        )

    def test_payload_bytes_preferred_vs_empty_atoms(self) -> None:
        """With distinct atom payloads, logit mass prefers those bytes vs empty."""
        torch.manual_seed(11)
        model = self._tiny()
        surf = model.surface
        model.eval()
        alpha = torch.randn(16, 16) * 0.5
        persist = torch.zeros(16)
        atom_r = torch.zeros(16)
        legacy = torch.zeros(16)
        # Distinct printable payloads (not overlapping the empty baseline).
        payloads = [b"ABC", b"xyz"]
        target_bytes = sorted({b for p in payloads for b in p})
        other_bytes = [b for b in range(32, 127) if b not in target_bytes][:6]

        with torch.no_grad():
            empty = surf(
                legacy,
                alpha=alpha,
                persistent_state=persist,
                atom_r=atom_r,
                atom_payloads=[],
            )
            filled = surf(
                legacy,
                alpha=alpha,
                persistent_state=persist,
                atom_r=atom_r,
                atom_payloads=payloads,
            )
            pe = empty["byte_logits"].softmax(dim=-1)
            pf = filled["byte_logits"].softmax(dim=-1)
            mass_empty = float(pe[:, target_bytes].sum().item())
            mass_filled = float(pf[:, target_bytes].sum().item())
            mass_other_f = float(pf[:, other_bytes].sum().item())

        self.assertGreater(
            mass_filled,
            mass_empty * 1.5 + 1e-6,
            msg=f"payload mass filled={mass_filled:.6f} empty={mass_empty:.6f}",
        )
        self.assertGreater(
            mass_filled,
            mass_other_f,
            msg=f"target mass {mass_filled:.6f} should beat other {mass_other_f:.6f}",
        )

    def test_payload_vanishes_when_alpha_zero(self) -> None:
        """Hard contract: α=0 ⇒ ~0 logits even with rich atom payloads."""
        torch.manual_seed(3)
        model = self._tiny()
        model.eval()
        with torch.no_grad():
            out = model.surface(
                torch.randn(16),
                alpha=torch.zeros(16, 16),
                persistent_state=torch.randn(16),
                atom_r=torch.randn(16),
                atom_payloads=[b"Hello!!", b"Bonjour"],
            )
            flat = torch.cat(
                [out["byte_logits"].reshape(-1), out["length_logits"].reshape(-1)]
            )
        self.assertLess(float(flat.norm().item()), 1e-5)

    def test_merge_concatenates_payloads(self) -> None:
        """MERGE of two living atoms concatenates their payloads."""
        model = self._tiny(enable_merge=True, merge_coherence_threshold=0.0, merge_energy_floor=0.0)
        d = 16
        a = ToroidalAtom(
            r=torch.ones(1, d),
            phi=torch.zeros(1, d),
            omega=torch.ones(1, d),
            E=torch.ones(1, d) * 0.5,
            kappa=torch.ones(1, d),
            M=torch.ones(1, d),
            tau=torch.ones(1, d),
            rho=torch.zeros(1, 1),
            payload=b"AB",
        )
        b = ToroidalAtom(
            r=torch.ones(1, d),
            phi=torch.zeros(1, d),
            omega=torch.ones(1, d),
            E=torch.ones(1, d) * 0.5,
            kappa=torch.ones(1, d),
            M=torch.ones(1, d),
            tau=torch.ones(1, d),
            rho=torch.zeros(1, 1),
            payload=b"CD",
        )
        model.core.atoms.extend([a, b])
        n_before = len(model.core.atoms)
        merges, _ = model._maybe_merge_atoms()
        self.assertGreaterEqual(merges, 1)
        self.assertLess(len(model.core.atoms), n_before)
        payloads = [getattr(x, "payload", b"") for x in model.core.atoms.atoms]
        self.assertTrue(any(p == b"ABCD" or b"AB" in p and b"CD" in p for p in payloads), payloads)

    def test_payload_modules_trainable_decoder_still_frozen(self) -> None:
        model = self._tiny()
        surf = model.surface
        for p in surf.byte_decoder.parameters():
            self.assertFalse(p.requires_grad)
        self.assertTrue(surf.byte_embed.weight.requires_grad)
        self.assertTrue(surf.payload_to_byte.weight.requires_grad)
        self.assertTrue(surf.payload_copy_scale.requires_grad)
        # Gradients hit payload path on a hard forward with payloads.
        alpha = torch.randn(16, 16) * 0.5
        out = surf(
            torch.zeros(16),
            alpha=alpha,
            persistent_state=torch.zeros(16),
            atom_r=torch.zeros(16),
            atom_payloads=[b"xy"],
        )
        out["byte_logits"].pow(2).mean().backward()
        self.assertIsNotNone(surf.byte_embed.weight.grad)
        self.assertGreater(float(surf.byte_embed.weight.grad.abs().sum()), 0.0)


    def test_per_atom_alpha_local_diversifies_logits(self) -> None:
        """Distinct living payload sets → surface logits cos clearly < 0.99.

        Replaces mean-pool/global-hist homogenization: α-local per-atom copy-bias
        must let different prompt atoms diversify decode without train.
        """
        torch.manual_seed(42)
        model = self._tiny()
        surf = model.surface
        model.eval()
        alpha = torch.randn(16, 16) * 0.5
        persist = torch.zeros(16)
        atom_r = torch.zeros(16)
        legacy = torch.zeros(16)
        payloads_a = [b"Bonjour!", b"salut.."]
        payloads_b = [b"zzzzzzzz", b"yyyyyyyy"]
        energies = [1.0, 1.0]
        with torch.no_grad():
            out_a = surf(
                legacy,
                alpha=alpha,
                persistent_state=persist,
                atom_r=atom_r,
                atom_payloads=payloads_a,
                atom_energies=energies,
            )
            out_b = surf(
                legacy,
                alpha=alpha,
                persistent_state=persist,
                atom_r=atom_r,
                atom_payloads=payloads_b,
                atom_energies=energies,
            )
            la = torch.cat(
                [out_a["byte_logits"].reshape(-1), out_a["length_logits"].reshape(-1)]
            )
            lb = torch.cat(
                [out_b["byte_logits"].reshape(-1), out_b["length_logits"].reshape(-1)]
            )
            cos = float(
                torch.nn.functional.cosine_similarity(
                    la.unsqueeze(0), lb.unsqueeze(0)
                ).item()
            )
        self.assertLess(
            cos,
            0.99,
            msg=f"per-atom α-local should diversify logits; cos={cos:.6f}",
        )

    def test_alpha_local_weights_prefer_aligned_atom(self) -> None:
        """Higher energy on one distinct atom boosts that atom's bytes vs the other."""
        torch.manual_seed(7)
        model = self._tiny()
        surf = model.surface
        model.eval()
        alpha = torch.randn(16, 16) * 0.5
        payloads = [b"AAAA", b"zzzz"]
        with torch.no_grad():
            favor_a = surf(
                torch.zeros(16),
                alpha=alpha,
                persistent_state=torch.zeros(16),
                atom_r=torch.zeros(16),
                atom_payloads=payloads,
                atom_energies=[5.0, 0.05],
            )
            favor_z = surf(
                torch.zeros(16),
                alpha=alpha,
                persistent_state=torch.zeros(16),
                atom_r=torch.zeros(16),
                atom_payloads=payloads,
                atom_energies=[0.05, 5.0],
            )
            pa = favor_a["byte_logits"].softmax(dim=-1)
            pz = favor_z["byte_logits"].softmax(dim=-1)
            mass_A_when_A = float(pa[:, ord("A")].sum().item())
            mass_A_when_Z = float(pz[:, ord("A")].sum().item())
            mass_z_when_Z = float(pz[:, ord("z")].sum().item())
            mass_z_when_A = float(pa[:, ord("z")].sum().item())
        self.assertGreater(
            mass_A_when_A,
            mass_A_when_Z,
            msg=f"A-mass favor_A={mass_A_when_A:.6f} favor_Z={mass_A_when_Z:.6f}",
        )
        self.assertGreater(
            mass_z_when_Z,
            mass_z_when_A,
            msg=f"z-mass favor_Z={mass_z_when_Z:.6f} favor_A={mass_z_when_A:.6f}",
        )


if __name__ == "__main__":
    unittest.main()
