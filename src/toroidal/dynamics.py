"""Explicit toroidal field dynamics and RK4 integration.

The dynamics layer deliberately contains no attention/Transformer block and no
learned feature-mixing MLP.  The state evolves through shared physical-style
rules: toroidal neighbour coupling, phase rotation, input driving, and energy
decay.  RK4 is only the numerical integrator for this continuous system.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ToroidalDynamics(nn.Module):
    """Shared-rule dynamics for the spectral toroidal field.

    ``alpha[m]`` is the amplitude/state carried by toroidal mode ``m``.
    Mode phase is fixed by its position on the torus; the incoming atom is
    injected as an external drive.  All trainable dynamics parameters are
    shared globally rather than being per-mode feature transforms.
    """

    def __init__(self, d_model: int = 256, n_modes: int = 256) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_modes = n_modes
        self.coupling_scale = nn.Parameter(torch.tensor(1.0))
        self.energy_decay = nn.Parameter(torch.tensor(0.99))
        self.phase_sync = nn.Parameter(torch.tensor(0.5))
        self.register_buffer(
            "mode_phase",
            torch.arange(n_modes, dtype=torch.float32) * (2.0 * torch.pi / n_modes),
        )

    def forward(
        self,
        alpha: torch.Tensor,
        input_token: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if alpha.shape[-2:] != (self.n_modes, self.d_model):
            raise ValueError(
                f"expected alpha [..., {self.n_modes}, {self.d_model}], got {tuple(alpha.shape)}"
            )

        # Toroidal nearest-neighbour coupling.  Periodic roll closes the ring.
        left = torch.roll(alpha, shifts=1, dims=-2)
        right = torch.roll(alpha, shifts=-1, dims=-2)
        laplacian = left + right - 2.0 * alpha
        coupling = self.phase_sync * laplacian

        # Continuous phase rotation of each mode.  This is the field analogue
        # of oscillator evolution; it is not token attention.
        phase = self.mode_phase.to(device=alpha.device, dtype=alpha.dtype)
        phase = phase.view(*([1] * (alpha.ndim - 2)), self.n_modes, 1)
        rotated = torch.sin(phase) * alpha

        # External token/context drive.  It is additive and shared across the
        # field, rather than constructing pairwise token-token attention.
        drive = torch.zeros_like(alpha)
        if input_token is not None:
            token = input_token
            while token.ndim > 1:
                token = token.mean(dim=0)
            drive = token.reshape(*([1] * (alpha.ndim - 2)), 1, self.d_model)
            drive = drive.expand(*alpha.shape[:-2], self.n_modes, self.d_model)
            drive = self.coupling_scale * drive

        if context is not None:
            ctx = context
            while ctx.ndim > 1:
                ctx = ctx.mean(dim=0)
            ctx = ctx.reshape(*([1] * (alpha.ndim - 2)), 1, self.d_model)
            ctx = ctx.expand(*alpha.shape[:-2], self.n_modes, self.d_model)
            drive = drive + 0.5 * self.coupling_scale * ctx

        decay = (self.energy_decay - 1.0) * alpha
        return coupling + rotated + drive + decay

    def _mode_freqs(self, device: torch.device) -> torch.Tensor:
        return torch.arange(self.n_modes, device=device, dtype=torch.float32) / self.n_modes * 2 * torch.pi

    def _compute_phase_laplacian(self, alpha: torch.Tensor) -> torch.Tensor:
        left = torch.roll(alpha, shifts=1, dims=-2)
        right = torch.roll(alpha, shifts=-1, dims=-2)
        return torch.sin(left - alpha) + torch.sin(right - alpha)

    def get_shared_params(self) -> dict:
        return {
            "coupling_scale": self.coupling_scale.detach().cpu(),
            "energy_decay": self.energy_decay.detach().cpu(),
            "phase_sync": self.phase_sync.detach().cpu(),
        }

    def load_shared_params(self, params: dict) -> None:
        for key in ("coupling_scale", "energy_decay", "phase_sync"):
            if key in params:
                getattr(self, key).data.copy_(params[key].to(getattr(self, key).device))


class RK4DynamicsEngine(nn.Module):
    """RK4 numerical integration of the toroidal dynamics."""

    def __init__(self, dynamics: ToroidalDynamics, dt: float = 0.01, n_steps: int = 4) -> None:
        super().__init__()
        self.dynamics = dynamics
        self.dt = dt
        self.n_steps = n_steps

    def evolve(
        self,
        alpha: torch.Tensor,
        input_token: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        def f(state: torch.Tensor) -> torch.Tensor:
            return self.dynamics(state, input_token, context)

        k1 = f(alpha)
        k2 = f(alpha + self.dt * 0.5 * k1)
        k3 = f(alpha + self.dt * 0.5 * k2)
        k4 = f(alpha + self.dt * k3)
        return alpha + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def evolve_multi(
        self,
        alpha: torch.Tensor,
        input_token: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        n_steps: int | None = None,
    ) -> torch.Tensor:
        current = alpha
        for _ in range(n_steps or self.n_steps):
            current = self.evolve(current, input_token, context)
        return current
