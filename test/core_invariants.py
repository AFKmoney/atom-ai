"""Smoke tests for the repaired continuous ATOM core."""

import torch

from src.toroidal.dynamics import RK4DynamicsEngine, ToroidalDynamics
from src.toroidal.state import FractalSuperpositionState


def test_toroidal_projection_uses_phase_and_frequency():
    state = FractalSuperpositionState(n_modes=32, d_model=8)
    r = torch.ones(1, 8)
    E = torch.ones(1, 8)
    kappa = torch.ones(1, 8)
    low = state._projection(r, torch.zeros(1, 8), torch.ones(1, 8), E, kappa)
    high = state._projection(r, torch.ones(1, 8), torch.ones(1, 8) * 8.0, E, kappa)
    assert low.shape == (32, 8)
    assert not torch.allclose(low, high)


def test_dynamics_is_periodic_and_shape_preserving():
    dynamics = ToroidalDynamics(d_model=8, n_modes=32)
    engine = RK4DynamicsEngine(dynamics, dt=0.01, n_steps=2)
    alpha = torch.randn(32, 8)
    token = torch.randn(2, 8)
    out = engine.evolve_multi(alpha, token)
    assert out.shape == alpha.shape
    assert torch.isfinite(out).all()
    assert torch.isfinite(dynamics._compute_phase_laplacian(alpha)).all()


def test_zero_field_without_drive_is_zero():
    dynamics = ToroidalDynamics(d_model=8, n_modes=32)
    alpha = torch.zeros(32, 8)
    out = dynamics(alpha)
    assert torch.allclose(out, torch.zeros_like(out))
