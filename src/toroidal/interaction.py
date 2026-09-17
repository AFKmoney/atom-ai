"""Explicit toroidal interactions between computational atoms and field modes."""
from __future__ import annotations
import torch
import torch.nn as nn


class ToroidalInteraction(nn.Module):
    """Interaction engine based on phase/frequency/coupling, not Transformer attention."""

    def __init__(self, d_model: int = 256, n_modes: int = 256) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_modes = n_modes

    def field_interaction(self, alpha: torch.Tensor) -> torch.Tensor:
        """Toroidal nearest-neighbour interaction energy over the field."""
        if alpha.numel() == 0:
            return alpha.new_zeros(0)
        prev = torch.roll(alpha, 1, dims=0)
        nxt = torch.roll(alpha, -1, dims=0)
        coupling = torch.cosine_similarity(alpha, prev, dim=-1)
        forward = torch.cosine_similarity(alpha, nxt, dim=-1)
        local_gradient = (alpha - prev).pow(2).mean(dim=-1).sqrt()
        return 0.5 * (coupling + forward) - local_gradient

    def pairwise_interaction(
        self,
        atoms_r: torch.Tensor,
        atoms_phi: torch.Tensor,
        atoms_kappa: torch.Tensor,
    ) -> torch.Tensor:
        """K_ij = <kappa_i*kappa_j*cos(phi_i-phi_j)> with optional radial affinity."""
        n = atoms_r.shape[0]
        if n == 0:
            return atoms_r.new_zeros((0, 0))
        dphi = atoms_phi[:, None, :] - atoms_phi[None, :, :]
        phase = torch.cos(dphi)
        coupling = atoms_kappa[:, None, :] * atoms_kappa[None, :, :]
        radial = torch.exp(-0.5 * (atoms_r[:, None, :] - atoms_r[None, :, :]).pow(2).mean(dim=-1))
        return (phase * coupling).mean(dim=-1) * radial

    def get_interaction_matrix(self, atoms_phi: torch.Tensor, atoms_kappa: torch.Tensor) -> torch.Tensor:
        return self.pairwise_interaction(torch.zeros_like(atoms_phi), atoms_phi, atoms_kappa)
