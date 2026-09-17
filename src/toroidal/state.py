"""Compact spectral/toroidal superposition field.

Atoms are projected into a fixed periodic set of modes.  The field is state,
not a Transformer activation stack: atoms remain the structural memory while
``alpha`` is their compact superposition used by the continuous dynamics.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FractalSuperpositionState(nn.Module):
    def __init__(self, n_modes: int = 256, d_model: int = 256) -> None:
        super().__init__()
        self.n_modes = n_modes
        self.d_model = d_model
        self.alpha = nn.Parameter(torch.zeros(n_modes, d_model))
        self.t = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.register_buffer(
            "mode_phase",
            torch.arange(n_modes, dtype=torch.float32) * (2.0 * torch.pi / n_modes),
        )

    def _projection(
        self,
        r: torch.Tensor,
        phi: torch.Tensor,
        omega: torch.Tensor,
        E: torch.Tensor,
        kappa: torch.Tensor,
    ) -> torch.Tensor:
        """Project atoms continuously across toroidal modes.

        Frequency selects the centre of the mode distribution, phase controls
        its periodic modulation, and ``r`` carries the vector-valued content.
        No hard ``.item()`` indexing is used, so the projection remains smooth.
        """
        if r.ndim == 1:
            r, phi, omega, E, kappa = [x.unsqueeze(0) for x in (r, phi, omega, E, kappa)]

        mode_phase = self.mode_phase.to(device=r.device, dtype=r.dtype).view(1, self.n_modes, 1)
        phase = phi.mean(dim=-1, keepdim=True).unsqueeze(1)
        freq = omega.mean(dim=-1, keepdim=True).unsqueeze(1).abs()
        freq_angle = 2.0 * torch.pi * freq / (1.0 + freq)
        centre = (freq_angle / (2.0 * torch.pi)) * self.n_modes
        mode_index = torch.arange(self.n_modes, device=r.device, dtype=r.dtype).view(1, self.n_modes, 1)

        # Wrapped circular distance keeps the projection toroidal.
        delta = torch.remainder(mode_index - centre + self.n_modes / 2, self.n_modes) - self.n_modes / 2
        width = 1.0 + 0.25 * torch.sigmoid(freq)
        kernel = torch.exp(-(delta / width) ** 2)
        modulation = 0.5 + 0.5 * torch.cos(mode_phase - phase)
        strength = (E * kappa).mean(dim=-1, keepdim=True).unsqueeze(1)
        vectors = r.unsqueeze(1) * strength
        contribution = vectors * kernel * modulation
        return contribution.sum(dim=0)

    @torch.no_grad()
    def add_atom_contribution(self, r, phi, omega, E, kappa) -> None:
        self.alpha.add_(self._projection(r, phi, omega, E, kappa))

    @torch.no_grad()
    def remove_atom_contribution(self, r, phi, omega, E, kappa) -> None:
        self.alpha.sub_(self._projection(r, phi, omega, E, kappa))

    def get_field(self) -> torch.Tensor:
        return self.alpha

    def get_state_dict(self) -> dict:
        return {"alpha": self.alpha.detach().cpu().clone(), "t": self.t.detach().cpu().clone()}

    def load_state_dict(self, state: dict, strict: bool = True):
        with torch.no_grad():
            self.alpha.copy_(state["alpha"].to(self.alpha.device))
            self.t.copy_(state["t"].to(self.t.device))

    def to(self, device):
        super().to(device)
        return self
