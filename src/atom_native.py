"""ATOM-native text model built around ``AtomPacket`` observations.

The canonical toroidal modules are intentionally not modified here.  This
adapter replaces the GPT-2 ID/embedding input path with:

    AtomPacket.features -> eight toroidal atom properties

and replaces the 50k-class surface target with a bounded byte packet head.
The recurrent field, RK4 dynamics, interactions, aggregation, abstraction and
consolidation are the existing ATOM components.

This is the primary text path. Legacy GPT-2 tokenizer wrappers live under
``_quarantine_transformer_slop/``. The toroidal core remains in ``src.toroidal``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .io.atomizer import AtomPacket, Atomizer
from .toroidal.atom import ToroidalAtom
from .toroidal.model import ToroidalFractalIntelligence

# Sane band for shared energy_decay (always applied).  Values near 0 turn the
# decay term into near-total wipe; values >1 flip it into growth.
DEFAULT_ENERGY_DECAY_BOUNDS: tuple[float, float] = (0.3, 0.95)


class AtomCompiler(nn.Module):
    """Compile a packet observation into the eight atom properties.

    There is no vocabulary lookup here.  The only learned input map receives
    the fixed-size physical observation produced by ``Atomizer``.
    """

    def __init__(self, feature_dim: int, d_model: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or max(32, 2 * d_model)
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.mapping = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 8 * d_model),
        )
        self.operation_gate = nn.Linear(feature_dim, 7)

    def forward(self, features: torch.Tensor, atom_count: int = 0) -> tuple[ToroidalAtom, torch.Tensor, torch.Tensor]:
        if features.ndim == 1:
            features = features.unsqueeze(0)
        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"expected packet features with {self.feature_dim} values, got {features.shape[-1]}"
            )
        raw = self.mapping(features).view(features.shape[0], 8, self.d_model)
        r = F.normalize(raw[:, 0], dim=-1)
        phi = raw[:, 1]
        omega = F.softplus(raw[:, 2]) + 1e-3
        energy = torch.sigmoid(raw[:, 3])
        kappa = F.softplus(raw[:, 4])
        memory = F.normalize(raw[:, 5], dim=-1)
        tau = F.softplus(raw[:, 6]) + 1e-6
        rho = torch.full(
            (features.shape[0], 1),
            min(atom_count // 64, 4),
            dtype=features.dtype,
            device=features.device,
        )
        atom = ToroidalAtom(
            r=r,
            phi=phi,
            omega=omega,
            E=energy,
            kappa=kappa,
            M=memory,
            tau=tau,
            rho=rho,
        )
        operation_logits = self.operation_gate(features)
        operation = operation_logits.argmax(dim=-1)
        confidence = operation_logits.softmax(dim=-1).max(dim=-1).values
        return atom, operation, confidence


class AtomSurfaceHead(nn.Module):
    """Decode one variable-length surface packet from the living toroidal field.

    The output alphabet is raw bytes (256 classes), not GPT-2's 50,257 token
    vocabulary.  A bounded packet contains 1..``max_payload_bytes`` bytes and
    has a separate length distribution.

    Readout contract (field-first):
      surface_input = f(spectral field α, optional consolidation, atom.r)
    A LayerNorm + legacy linear path is retained for checkpoint compatibility,
    but every decode also mixes an explicit spectral summary of α (mean‖std)
    and a bias-free field→logit skip so prompt-conditioned α is not drowned by
    a shared decoder bias (the failure mode where inter-prompt logit cosine
    stayed ≈ 0.998 while α cosine was ≈ 0.74).
    """

    # Printable UTF-8 / Latin-1 friendly prior used when ``prefer_printable``.
    _PRINTABLE = set(range(0x20, 0x7F)) | {0x09, 0x0A, 0x0D} | set(range(0xC2, 0xF5)) | set(range(0x80, 0xC0))

    def __init__(
        self,
        d_model: int,
        max_payload_bytes: int = 32,
        n_modes: int | None = None,
    ) -> None:
        super().__init__()
        if max_payload_bytes < 1:
            raise ValueError("max_payload_bytes must be positive")
        self.d_model = d_model
        self.n_modes = int(n_modes) if n_modes is not None else d_model
        self.max_payload_bytes = max_payload_bytes
        # Spectral living-field features: [mean(α) ‖ std(α) ‖ persist ‖ atom.r]
        self.field_feat_dim = 4 * d_model
        self.field_feat_norm = nn.LayerNorm(self.field_feat_dim)
        self.field_to_state = nn.Linear(self.field_feat_dim, d_model)
        # sigmoid(field_gate)≈0.88 at init → field-conditioned state dominates.
        self.field_gate = nn.Parameter(torch.tensor(2.0))
        # Bias-free skip so α differences reach logits even when |b| is large.
        self.field_byte_skip = nn.Linear(self.field_feat_dim, max_payload_bytes * 256, bias=False)
        self.field_length_skip = nn.Linear(self.field_feat_dim, max_payload_bytes, bias=False)
        # softplus(4)≈4.0 — enough for inter-prompt logit cosine << 0.998 at load.
        self.skip_gate = nn.Parameter(torch.tensor(1.0))
        self.state_norm = nn.LayerNorm(d_model)
        self.byte_decoder = nn.Linear(d_model, max_payload_bytes * 256)
        self.length_decoder = nn.Linear(d_model, max_payload_bytes)
        self._init_field_readout()

    def _init_field_readout(self) -> None:
        """Identity-on-mean + std residual; seeded Xavier skips. Safe for fresh + migrate."""
        with torch.no_grad():
            self.field_to_state.weight.zero_()
            eye = torch.eye(self.d_model, device=self.field_to_state.weight.device)
            self.field_to_state.weight[:, : self.d_model].copy_(eye)
            self.field_to_state.weight[:, self.d_model : 2 * self.d_model].copy_(0.5 * eye)
            self.field_to_state.bias.zero_()
            # Deterministic skip init so migrated checkpoints probe reproducibly.
            gen = torch.Generator(device="cpu")
            gen.manual_seed(20260917)
            for module in (self.field_byte_skip, self.field_length_skip):
                w = module.weight
                fan_in = w.shape[1]
                bound = (6.0 / (fan_in + w.shape[0])) ** 0.5
                w.copy_(
                    torch.empty_like(w, device="cpu")
                    .uniform_(-bound, bound, generator=gen)
                    .to(device=w.device, dtype=w.dtype)
                )
            self.field_gate.fill_(2.0)
            self.skip_gate.fill_(1.0)

    def field_features(
        self,
        alpha: torch.Tensor,
        persistent_state: torch.Tensor | None = None,
        atom_r: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build spectral living-field features from α (+ consolidation, atom)."""
        if alpha.ndim == 1:
            alpha = alpha.unsqueeze(0)
        mean = alpha.mean(dim=0)
        std = alpha.std(dim=0, unbiased=False)
        if persistent_state is None:
            persist = torch.zeros_like(mean)
        else:
            persist = (
                persistent_state.mean(dim=0)
                if persistent_state.ndim > 1
                else persistent_state
            )
            persist = persist.to(device=mean.device, dtype=mean.dtype).reshape(-1)
            if persist.numel() != mean.numel():
                persist = persist[: mean.numel()]
                if persist.numel() < mean.numel():
                    persist = F.pad(persist, (0, mean.numel() - persist.numel()))
        if atom_r is None:
            atom = torch.zeros_like(mean)
        else:
            atom = atom_r.mean(dim=0) if atom_r.ndim > 1 else atom_r
            atom = atom.to(device=mean.device, dtype=mean.dtype).reshape(-1)
            if atom.numel() != mean.numel():
                atom = atom[: mean.numel()]
                if atom.numel() < mean.numel():
                    atom = F.pad(atom, (0, mean.numel() - atom.numel()))
        return torch.cat([mean, std, persist, atom], dim=-1)

    def field_conditioned_state(
        self,
        alpha: torch.Tensor,
        persistent_state: torch.Tensor | None = None,
        atom_r: torch.Tensor | None = None,
        legacy_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """d_model state where spectral α dominates (gated residual w/ legacy)."""
        feats = self.field_feat_norm(
            self.field_features(alpha, persistent_state, atom_r)
        )
        field_state = self.field_to_state(feats)
        gate = torch.sigmoid(self.field_gate)
        if legacy_state is None:
            return field_state
        legacy = legacy_state.mean(dim=0) if legacy_state.ndim > 1 else legacy_state
        legacy = legacy.to(device=field_state.device, dtype=field_state.dtype)
        return gate * field_state + (1.0 - gate) * legacy

    def forward(
        self,
        state: torch.Tensor | None = None,
        *,
        alpha: torch.Tensor | None = None,
        persistent_state: torch.Tensor | None = None,
        atom_r: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Decode from living field α when provided; else legacy mean-state path."""
        if alpha is not None:
            feats = self.field_feat_norm(
                self.field_features(alpha, persistent_state, atom_r)
            )
            field_state = self.field_to_state(feats)
            gate = torch.sigmoid(self.field_gate)
            if state is None:
                combined = field_state
            else:
                legacy = state.mean(dim=0) if state.ndim > 1 else state
                legacy = legacy.to(device=field_state.device, dtype=field_state.dtype)
                combined = gate * field_state + (1.0 - gate) * legacy
            h = self.state_norm(combined)
            byte_logits = self.byte_decoder(h).view(self.max_payload_bytes, 256)
            length_logits = self.length_decoder(h)
            skip = F.softplus(self.skip_gate)
            byte_logits = byte_logits + skip * self.field_byte_skip(feats).view(
                self.max_payload_bytes, 256
            )
            length_logits = length_logits + skip * self.field_length_skip(feats)
            return {"byte_logits": byte_logits, "length_logits": length_logits}

        if state is None:
            raise ValueError("AtomSurfaceHead.forward requires state or alpha")
        if state.ndim > 1:
            state = state.mean(dim=0)
        state = self.state_norm(state)
        byte_logits = self.byte_decoder(state).view(self.max_payload_bytes, 256)
        length_logits = self.length_decoder(state)
        return {"byte_logits": byte_logits, "length_logits": length_logits}

    def _apply_printable_bias(self, byte_logits: torch.Tensor, strength: float = 2.0) -> torch.Tensor:
        """Softly discourage control/non-text bytes without hard-masking UTF-8."""
        if strength <= 0:
            return byte_logits
        penalty = torch.full(
            (256,),
            -strength,
            dtype=byte_logits.dtype,
            device=byte_logits.device,
        )
        for value in self._PRINTABLE:
            penalty[value] = 0.0
        # Keep NUL strongly suppressed; allow tab/LF/CR via printable set.
        penalty[0] = -strength * 2.0
        return byte_logits + penalty

    def decode(
        self,
        output: dict[str, torch.Tensor],
        temperature: float = 1.0,
        top_k: int | None = None,
        deterministic: bool = True,
        prefer_printable: bool = True,
    ) -> bytes:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        length_logits = output["length_logits"] / temperature
        if deterministic:
            length = int(length_logits.argmax().item()) + 1
        else:
            k = length_logits.numel()
            if top_k and top_k > 0:
                k = min(top_k, k)
            length_values, length_indices = torch.topk(length_logits, k)
            # Guard against non-finite / zero-mass softmax from extreme logits.
            probs = length_values.softmax(dim=-1)
            if not torch.isfinite(probs).all() or float(probs.sum()) <= 0:
                length = int(length_logits.argmax().item()) + 1
            else:
                length = int(length_indices[torch.multinomial(probs, 1)].item()) + 1
        length = max(1, min(length, self.max_payload_bytes))
        byte_logits = output["byte_logits"][:length] / temperature
        if prefer_printable:
            byte_logits = self._apply_printable_bias(byte_logits)
        if deterministic:
            values = byte_logits.argmax(dim=-1)
        else:
            if top_k and top_k > 0 and top_k < byte_logits.shape[-1]:
                values, indices = torch.topk(byte_logits, top_k, dim=-1)
                probs = values.softmax(dim=-1)
                if not torch.isfinite(probs).all():
                    values = byte_logits.argmax(dim=-1)
                else:
                    picks = torch.multinomial(probs, 1)
                    values = indices.gather(-1, picks).squeeze(-1)
            else:
                probs = byte_logits.softmax(dim=-1)
                if not torch.isfinite(probs).all():
                    values = byte_logits.argmax(dim=-1)
                else:
                    values = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return bytes(int(value) for value in values.tolist())


class FieldAmplitudeController:
    """Bound field amplitude without changing the toroidal core equations.

    The controller is an opt-in safety policy around the existing adapter.  It
    rescales only when the field RMS exceeds ``max_rms``; directions and small
    amplitudes are left untouched.  Dynamics, interactions and the eight atom
    properties remain the same.
    """

    def __init__(self, max_rms: float | None = None, eps: float = 1e-6) -> None:
        if max_rms is not None and max_rms <= 0:
            raise ValueError("max_rms must be positive or None")
        self.max_rms = max_rms
        self.eps = eps

    def __call__(self, field: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        rms_before = torch.sqrt(field.pow(2).mean() + self.eps)
        if self.max_rms is None:
            scale = torch.ones_like(rms_before)
            stabilized = field
        else:
            scale = torch.minimum(
                torch.ones_like(rms_before),
                field.new_tensor(self.max_rms) / (rms_before + self.eps),
            )
            stabilized = field * scale
        rms_after = torch.sqrt(stabilized.pow(2).mean() + self.eps)
        return stabilized, {
            "field_rms_before": float(rms_before.detach().item()),
            "field_rms_after": float(rms_after.detach().item()),
            "field_scale": float(scale.detach().item()),
        }


class AtomNativeModel(nn.Module):
    """Use existing ATOM dynamics with an atom-native input/output contract."""

    def __init__(
        self,
        d_model: int = 8,
        n_modes: int = 8,
        n_atoms_max: int = 128,
        max_payload_bytes: int = 32,
        atomizer: Atomizer | None = None,
        field_max_rms: float | None = None,
        energy_decay_bounds: tuple[float, float] | None = None,
        enable_merge: bool = True,
        slow_every: int = 1,
        field_loss_weight: float = 0.0,
        field_contrast_weight: float = 0.0,
        field_contrast_margin: float = 0.55,
        slow_rms_rel_tol: float = 0.15,
    ) -> None:
        super().__init__()
        self.atomizer = atomizer or Atomizer(max_span_bytes=max_payload_bytes)
        self.core = ToroidalFractalIntelligence(
            vocab_size=256,
            d_model=d_model,
            n_modes=n_modes,
            n_atoms_max=n_atoms_max,
        )
        self.compiler = AtomCompiler(self.atomizer.feature_dim, d_model)
        self.surface = AtomSurfaceHead(d_model, max_payload_bytes=max_payload_bytes, n_modes=n_modes)
        self.max_payload_bytes = max_payload_bytes
        self.field_controller = FieldAmplitudeController(field_max_rms)
        self.field_max_rms = field_max_rms
        # Always-on physical band.  The weak prior floor (1e-3) let persist
        # training drift energy_decay to ~0.001 and collapse the field.
        if energy_decay_bounds is None:
            energy_decay_bounds = DEFAULT_ENERGY_DECAY_BOUNDS
        self.energy_decay_bounds = energy_decay_bounds
        # Continuum intelligence knobs (structure + field dependence, not Θ growth).
        self.enable_merge = bool(enable_merge)
        self.slow_every = max(1, int(slow_every))
        self.field_loss_weight = float(field_loss_weight)
        self.field_contrast_weight = float(field_contrast_weight)
        self.field_contrast_margin = float(field_contrast_margin)
        self.slow_rms_rel_tol = float(slow_rms_rel_tol)
        self.field_probe = nn.Linear(self.surface.field_feat_dim, self.atomizer.feature_dim)
        self.merge_count_total = 0
        self._tick = 0
        self._prev_field_rms: float | None = None
        # Detached rolling α snapshots for harder field-contrast negatives
        # (mode-shuffle alone is too weak vs CE; inter-prompt collapse returns).
        self._alpha_bank: list[torch.Tensor] = []
        self._alpha_bank_max = 8

        # The legacy encoder and legacy 256-way production head are not used
        # by this adapter.  Keep them in the core checkpoint for compatibility,
        # but make clear that no GPT-style embedding participates in training.
        for parameter in self.core.encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.core.production.parameters():
            parameter.requires_grad_(False)
        self.core.state.alpha.requires_grad_(False)

    @property
    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.trainable_parameters)

    def stabilize_dynamics_parameters(self) -> dict[str, float]:
        """Project shared dynamics params into a physical range every step.

        ``energy_decay`` is called a decay parameter by the core.  Without a
        strong floor it can drift near 0 and wipe the field each tick
        (``decay = (energy_decay - 1) * alpha``).  Bounds are always-on
        (default ``DEFAULT_ENERGY_DECAY_BOUNDS``).  Does not alter
        ``src/toroidal/dynamics.py``.
        """
        parameter = self.core.dynamics.dynamics
        bounds = self.energy_decay_bounds or DEFAULT_ENERGY_DECAY_BOUNDS
        low, high = bounds
        if not (0.0 < low <= high):
            raise ValueError("energy_decay_bounds must satisfy 0 < low <= high")
        with torch.no_grad():
            before = float(parameter.energy_decay.detach().item())
            parameter.energy_decay.clamp_(low, high)
            parameter.coupling_scale.clamp_(0.0, 2.0)
            parameter.phase_sync.clamp_(-1.0, 1.0)
            after = float(parameter.energy_decay.detach().item())
        return {
            "coupling_scale": float(parameter.coupling_scale.detach().item()),
            "energy_decay": after,
            "phase_sync": float(parameter.phase_sync.detach().item()),
            "energy_decay_before": before,
            "energy_decay_repaired": before != after,
            "energy_decay_bounds": (float(low), float(high)),
        }

    def reset_state(self, reset_atomizer: bool = True) -> None:
        """Start a new explicit episode while preserving learned weights."""
        with torch.no_grad():
            self.core.state.alpha.zero_()
            self.core.state.t.zero_()
            self.core.consolidation.persistence.zero_()
            self.core.consolidation.persistence_strength.zero_()
            self.core.abstraction.abstraction_memory.zero_()
            self.core.abstraction.abstraction_usage.zero_()
            self.core.abstraction.abstraction_age.zero_()
        self.core.atoms = type(self.core.atoms)()
        self.core.energy_history = []
        self.core.consolidation_count = 0
        self.core.abstraction_count = 0
        self._tick = 0
        self._prev_field_rms = None
        self._alpha_bank = []
        if reset_atomizer:
            self.atomizer.reset()

    def _output_state(
        self,
        alpha: torch.Tensor,
        persistent_state: torch.Tensor | None,
        abstractions: list[dict],
        atom_r: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Field-dominated surface conditioning vector (also used by probes).

        Prefer the spectral living-field readout when available so probes and
        decode agree.  Abstractions remain a light residual on the legacy mean.
        """
        legacy = alpha.mean(dim=0) if alpha.ndim > 1 else alpha
        if persistent_state is not None:
            legacy = legacy + (
                persistent_state.mean(dim=0)
                if persistent_state.ndim > 1
                else persistent_state
            )
        if abstractions:
            patterns = [
                item["pattern"].to(device=legacy.device, dtype=legacy.dtype).reshape(-1)
                for item in abstractions
                if item["pattern"].numel() == self.core.encoder.d_model
            ]
            if patterns:
                legacy = legacy + 0.1 * torch.stack(patterns).mean(dim=0)
        # Always-on consolidation buffer when per-tick persist is absent.
        persist = persistent_state
        if persist is None:
            persist = self.core.consolidation.persistence
        return self.surface.field_conditioned_state(
            alpha,
            persistent_state=persist,
            atom_r=atom_r,
            legacy_state=legacy,
        )

    def _maybe_merge_atoms(self) -> int:
        """MERGE coherent structural atoms into heavier ones (detached memory)."""
        if not self.enable_merge or len(self.core.atoms) < 2:
            return 0
        atoms = self.core.atoms
        merged, remove_idx, merge_count = self.core.aggregation.merge_coherent(
            atoms.r,
            atoms.phi,
            atoms.omega,
            atoms.E,
            atoms.kappa,
            atoms.M,
            atoms.tau,
            atoms.rho,
        )
        if merge_count <= 0 or not remove_idx:
            return 0
        atoms.remove(remove_idx)
        new_atoms = [
            ToroidalAtom(
                r=item["r"].detach().clone(),
                phi=item["phi"].detach().clone(),
                omega=item["omega"].detach().clone(),
                E=item["E"].detach().clone(),
                kappa=item["kappa"].detach().clone(),
                M=item["M"].detach().clone(),
                tau=item["tau"].detach().clone(),
                rho=item["rho"].detach().clone(),
            )
            for item in merged
        ]
        atoms.extend(new_atoms)
        self.merge_count_total += merge_count
        return merge_count

    def _field_rms_stable(self, rms: float) -> bool:
        prev = self._prev_field_rms
        self._prev_field_rms = rms
        if prev is None:
            return False
        return abs(rms - prev) / (abs(prev) + 1e-6) <= self.slow_rms_rel_tol

    def _advance(self, atom: ToroidalAtom, operation: torch.Tensor, confidence: torch.Tensor) -> dict:
        """Advance the unchanged toroidal core by one compiled atom.

        Fast path (every tick): inject + evolve + surface.
        Slow path (``slow_every`` + RMS-stable): abstraction + consolidation.
        Optional MERGE densifies coherent structural memory without GPU farms.
        """
        self._tick += 1
        self.core.state.add_atom_contribution(atom.r, atom.phi, atom.omega, atom.E, atom.kappa)

        # Structural memory is intentionally detached from the current loss
        # graph.  The current atom still drives dynamics and receives gradients.
        memory_atom = ToroidalAtom(
            r=atom.r.detach().clone(),
            phi=atom.phi.detach().clone(),
            omega=atom.omega.detach().clone(),
            E=atom.E.detach().clone(),
            kappa=atom.kappa.detach().clone(),
            M=atom.M.detach().clone(),
            tau=atom.tau.detach().clone(),
            rho=atom.rho.detach().clone(),
        )
        self.core.atoms.add(memory_atom)
        merge_count = self._maybe_merge_atoms()

        alpha = self.core.state.get_field().detach().clone()
        alpha_new = self.core.dynamics.evolve(alpha, input_token=atom.r)
        interaction_energy = self.core.interaction.field_interaction(alpha_new)
        left = torch.roll(alpha_new, 1, 0)
        right = torch.roll(alpha_new, -1, 0)
        alpha_new = alpha_new + 0.1 * self.core.dynamics.dynamics.phase_sync * (
            left + right - 2.0 * alpha_new
        )
        alpha_new, amplitude = self.field_controller(alpha_new)

        rms_after = float(amplitude["field_rms_after"])
        do_slow = self.slow_every <= 1 or (
            self._tick % self.slow_every == 0 and self._field_rms_stable(rms_after)
        )
        if self.slow_every > 1 and self._tick % self.slow_every != 0:
            self._prev_field_rms = rms_after

        aggregates: list[dict] = []
        abstractions: list[dict] = []
        persistent_state = None
        if do_slow:
            if len(self.core.atoms) > 10:
                aggregates, _ = self.core.aggregation.aggregate(
                    self.core.atoms.r,
                    self.core.atoms.phi,
                    self.core.atoms.omega,
                    self.core.atoms.E,
                    self.core.atoms.kappa,
                    self.core.atoms.M,
                    self.core.atoms.tau,
                    self.core.atoms.rho,
                )
            for aggregate in aggregates:
                index = self.core.abstraction.create_or_update_abstraction([aggregate])
                if index >= 0:
                    abstractions.append(
                        {
                            "id": index,
                            "pattern": self.core.abstraction.abstraction_memory[index].detach(),
                        }
                    )
                    self.core.abstraction_count += 1
            self.core.abstraction.step_age()

            if len(self.core.energy_history) > 5:
                energy = self.core.atoms.E.mean(dim=0)
                stability = self.core.consolidation.compute_stability(
                    energy, self.core.energy_history
                )
                persistent_state, mask = self.core.consolidation.consolidate(
                    alpha_new,
                    energy.unsqueeze(0),
                    stability.unsqueeze(0),
                )
                if mask.any():
                    self.core.consolidation_count += int(mask.sum().item())
        self.core.energy_history.append(self.core.atoms.E.mean(dim=0).detach())
        self.core.energy_history = self.core.energy_history[-20:]

        # Living consolidation buffer (EMA) — not only the per-tick consolidate().
        persist_for_readout = persistent_state
        if persist_for_readout is None:
            persist_for_readout = self.core.consolidation.persistence
        output_state = self._output_state(
            alpha_new, persist_for_readout, abstractions, atom_r=atom.r
        )
        surface = self.surface(
            output_state,
            alpha=alpha_new,
            persistent_state=persist_for_readout,
            atom_r=atom.r,
        )
        with torch.no_grad():
            self.core.state.alpha.copy_(alpha_new.detach())
            self.core.state.t.add_(self.core.dynamics.dt)
        return {
            "surface": surface,
            "field": alpha_new,
            "confidence": confidence,
            "operation": operation,
            "interaction_energy": interaction_energy,
            "aggregates": aggregates,
            "abstractions": abstractions,
            "persistent_state": persistent_state,
            "n_atoms": len(self.core.atoms),
            "merge_count": merge_count,
            "merge_count_total": self.merge_count_total,
            "slow_tick": do_slow,
            "atom_r": atom.r,
            **amplitude,
        }

    def forward_packet(self, packet: AtomPacket) -> dict:
        features = packet.features.to(self.core.state.alpha.device)
        atom, operation, confidence = self.compiler(features, atom_count=len(self.core.atoms))
        return self._advance(atom, operation, confidence)

    def _field_auxiliary_losses(
        self,
        output: dict,
        current: AtomPacket,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Keep surface conditioned on α; probe persistence from field features.

        Contrastive hinge: logits from true α must differ from zeroed / shuffled α
        (cosine below margin). Persistence probe: reconstruct current packet
        features from spectral field feats. Weighted small vs CE so bytes still learn.
        """
        device = output["field"].device
        zero = torch.zeros((), device=device)
        info: dict[str, float] = {
            "field_contrast_loss": 0.0,
            "field_persist_loss": 0.0,
            "field_logit_cos_zero": float("nan"),
            "field_logit_cos_shuf": float("nan"),
        }
        contrast = zero
        persist = zero
        alpha = output["field"]
        atom_r = output.get("atom_r")
        persist_state = output.get("persistent_state")
        if persist_state is None:
            persist_state = self.core.consolidation.persistence
        true_flat = output["surface"]["byte_logits"].reshape(-1)

        if self.field_contrast_weight > 0:
            surf_zero = self.surface(
                None,
                alpha=torch.zeros_like(alpha),
                persistent_state=persist_state,
                atom_r=atom_r,
            )
            cos_z = F.cosine_similarity(
                true_flat.unsqueeze(0),
                surf_zero["byte_logits"].reshape(-1).unsqueeze(0),
            ).squeeze()
            if alpha.shape[0] > 1:
                shuf_alpha = alpha[torch.randperm(alpha.shape[0], device=alpha.device)]
            else:
                shuf_alpha = -alpha
            surf_shuf = self.surface(
                None, alpha=shuf_alpha, persistent_state=persist_state, atom_r=atom_r
            )
            cos_s = F.cosine_similarity(
                true_flat.unsqueeze(0),
                surf_shuf["byte_logits"].reshape(-1).unsqueeze(0),
            ).squeeze()
            # Harder negative: a prior living field from the episode bank (if any).
            cos_b = true_flat.new_zeros(())
            if self._alpha_bank:
                bank_alpha = self._alpha_bank[int(torch.randint(len(self._alpha_bank), (1,)).item())]
                bank_alpha = bank_alpha.to(device=alpha.device, dtype=alpha.dtype)
                if bank_alpha.shape == alpha.shape and not torch.allclose(bank_alpha, alpha, atol=1e-5):
                    surf_bank = self.surface(
                        None, alpha=bank_alpha, persistent_state=persist_state, atom_r=atom_r
                    )
                    cos_b = F.cosine_similarity(
                        true_flat.unsqueeze(0),
                        surf_bank["byte_logits"].reshape(-1).unsqueeze(0),
                    ).squeeze()
            margin = self.field_contrast_margin
            # Tighter margin on bank negatives so inter-prompt logits must separate.
            contrast = (
                F.relu(cos_z - margin)
                + F.relu(cos_s - margin)
                + F.relu(cos_b - (margin - 0.15))
            )
            info["field_logit_cos_zero"] = float(cos_z.detach().item())
            info["field_logit_cos_shuf"] = float(cos_s.detach().item())
            info["field_logit_cos_bank"] = float(cos_b.detach().item()) if cos_b.ndim == 0 or cos_b.numel()==1 else float("nan")
            info["field_contrast_loss"] = float(contrast.detach().item())

        if self.field_loss_weight > 0:
            feats = self.surface.field_feat_norm(
                self.surface.field_features(alpha, persist_state, atom_r)
            )
            pred = self.field_probe(feats)
            target_feat = current.features.to(device=pred.device, dtype=pred.dtype)
            cos = F.cosine_similarity(pred.unsqueeze(0), target_feat.unsqueeze(0)).squeeze()
            persist = 1.0 - cos
            info["field_persist_loss"] = float(persist.detach().item())

        total_aux = (
            self.field_contrast_weight * contrast + self.field_loss_weight * persist
        )
        return total_aux, info

    def transition_loss(self, current: AtomPacket, target: AtomPacket) -> tuple[torch.Tensor, dict]:
        """Predict the complete next atom surface packet (+ optional field aux)."""
        output = self.forward_packet(current)
        # Update α bank (detached) for cross-field contrast on later steps.
        with torch.no_grad():
            snap = output["field"].detach().clone()
            self._alpha_bank.append(snap)
            if len(self._alpha_bank) > self._alpha_bank_max:
                self._alpha_bank.pop(0)
        surface = output["surface"]
        payload = target.payload[: self.max_payload_bytes]
        target_length = torch.tensor(
            [len(payload) - 1], dtype=torch.long, device=surface["length_logits"].device
        )
        length_loss = F.cross_entropy(surface["length_logits"].unsqueeze(0), target_length)
        target_bytes = torch.tensor(list(payload), dtype=torch.long, device=surface["byte_logits"].device)
        byte_logits = surface["byte_logits"][: len(payload)]
        byte_loss = F.cross_entropy(byte_logits, target_bytes)
        ce_loss = length_loss + byte_loss
        aux_loss, aux_info = self._field_auxiliary_losses(output, current)
        loss = ce_loss + aux_loss
        return loss, {
            "loss": float(loss.detach().item()),
            "ce_loss": float(ce_loss.detach().item()),
            "length_loss": float(length_loss.detach().item()),
            "byte_loss": float(byte_loss.detach().item()),
            "n_atoms": len(self.core.atoms),
            "surface": surface,
            "field_norm": float(output["field"].detach().norm().item()),
            "field_rms_before": output["field_rms_before"],
            "field_rms_after": output["field_rms_after"],
            "field_scale": output["field_scale"],
            "merge_count": int(output.get("merge_count", 0)),
            "merge_count_total": int(output.get("merge_count_total", self.merge_count_total)),
            "slow_tick": bool(output.get("slow_tick", True)),
            **aux_info,
        }

    @torch.no_grad()
    def predict_packet(
        self,
        packet: AtomPacket,
        temperature: float = 1.0,
        top_k: int | None = None,
        deterministic: bool = True,
        prefer_printable: bool = True,
    ) -> bytes:
        output = self.forward_packet(packet)
        return self.surface.decode(
            output["surface"],
            temperature=temperature,
            top_k=top_k,
            deterministic=deterministic,
            prefer_printable=prefer_printable,
        )

    def generate_packets(
        self,
        prompt: str,
        max_packets: int = 8,
        temperature: float = 1.0,
        top_k: int | None = None,
        deterministic: bool = True,
        max_length: int | None = None,
        prefer_printable: bool = True,
        reset: bool = True,
    ) -> bytes:
        """Prime on a prompt and generate bounded atom surface packets.

        Priming forwards every prompt packet into the toroidal field so the
        surface head is conditioned on the full prompt, not only the final
        span.  Each generated payload is committed back into the atomizer with
        an inferred structural boundary (not a fixed ``generated`` tag) so
        feature context stays closer to the training distribution.
        """
        self.eval()
        if reset:
            self.reset_state(reset_atomizer=True)
        prompt_packets = self.atomizer.encode(prompt, reset=reset)
        if not prompt_packets:
            raise ValueError("prompt must produce at least one atom packet")
        # Consume the full prompt into the field; the last packet is the
        # transition source for the first generated surface packet.
        for packet in prompt_packets[:-1]:
            self.forward_packet(packet)
        current = prompt_packets[-1]
        generated = bytearray()
        for _ in range(max_packets):
            payload = self.predict_packet(
                current,
                temperature=temperature,
                top_k=top_k,
                deterministic=deterministic,
                prefer_printable=prefer_printable,
            )
            if not payload:
                break
            generated.extend(payload)
            current = self.atomizer.packet_from_payload(payload)
            if max_length is not None and len(generated) >= max_length:
                break
            # Soft stop on blank double-newline turns (dialogue corpora).
            if generated.endswith(b"\n\n") and len(generated) > 2:
                break
        if max_length is not None:
            return bytes(generated[:max_length])
        return bytes(generated)

    def _checkpoint(self) -> dict:
        core = self.core
        return {
            "compiler": self.compiler.state_dict(),
            "surface": self.surface.state_dict(),
            "core": {
                "encoder": core.encoder.state_dict(),
                "state": core.state.get_state_dict(),
                "dynamics": core.dynamics.state_dict(),
                "interaction": core.interaction.state_dict(),
                "aggregation": core.aggregation.state_dict(),
                "abstraction": core.abstraction.state_dict(),
                "consolidation": core.consolidation.state_dict(),
                "production": core.production.state_dict(),
                "atoms": core.atoms.state_dict(),
                "energy_history": core.energy_history,
                "training_loss": core.training_loss,
                "consolidation_count": core.consolidation_count,
                "abstraction_count": core.abstraction_count,
            },
            "atomizer": self.atomizer.state_dict(),
            "config": {
                "d_model": core.encoder.d_model,
                "n_modes": core.state.n_modes,
                "n_atoms_max": core.encoder.n_atoms_max,
                "max_payload_bytes": self.max_payload_bytes,
                "atomizer_version": self.atomizer.VERSION,
                "field_max_rms": self.field_max_rms,
                "energy_decay_bounds": self.energy_decay_bounds,
                "enable_merge": self.enable_merge,
                "slow_every": self.slow_every,
                "field_loss_weight": self.field_loss_weight,
                "field_contrast_weight": self.field_contrast_weight,
                "field_contrast_margin": self.field_contrast_margin,
                "merge_count_total": self.merge_count_total,
            },
            "field_probe": self.field_probe.state_dict(),
        }

    def save(self, path: str | Path, extra_state: dict | None = None) -> str:
        checkpoint = self._checkpoint()
        if extra_state is not None:
            checkpoint["training"] = extra_state
        path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        return path

    def load(self, path: str | Path) -> dict | None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.compiler.load_state_dict(checkpoint["compiler"])
        surface_state = checkpoint["surface"]
        # Legacy checkpoints predate ``state_norm`` and/or field-α readout.
        # Load matching tensors; activate new field path; damp biases that drown α.
        legacy_surface = "state_norm.weight" not in surface_state
        field_readout_missing = "field_to_state.weight" not in surface_state
        current = self.surface.state_dict()
        filtered = {key: value for key, value in surface_state.items() if key in current and current[key].shape == value.shape}
        missing = [key for key in current if key not in filtered]
        self.surface.load_state_dict(filtered, strict=False)
        if legacy_surface or missing or field_readout_missing:
            with torch.no_grad():
                if legacy_surface or "state_norm.weight" in missing:
                    self.surface.state_norm.reset_parameters()
                # Shrink decoder bias so field-conditioned Wx / skip can compete.
                self.surface.byte_decoder.bias.mul_(0.05)
                self.surface.length_decoder.bias.mul_(0.05)
                if field_readout_missing or any(
                    key.startswith("field_") or key in {"skip_gate"} for key in missing
                ):
                    self.surface._init_field_readout()
                    self.surface.field_feat_norm.reset_parameters()
                    # Open skip strongly so migrated ckpts separate logits before retrain.
                    self.surface.skip_gate.fill_(4.0)
        core_state = checkpoint["core"]
        core = self.core
        core.encoder.load_state_dict(core_state["encoder"])
        core.state.load_state_dict(core_state["state"])
        core.dynamics.load_state_dict(core_state["dynamics"])
        core.interaction.load_state_dict(core_state["interaction"])
        core.aggregation.load_state_dict(core_state["aggregation"])
        core.abstraction.load_state_dict(core_state["abstraction"])
        core.consolidation.load_state_dict(core_state["consolidation"])
        core.production.load_state_dict(core_state["production"])
        core.atoms.load_state_dict(core_state["atoms"])
        core.energy_history = core_state.get("energy_history", [])
        core.training_loss = core_state.get("training_loss", 0.0)
        core.consolidation_count = core_state.get("consolidation_count", 0)
        core.abstraction_count = core_state.get("abstraction_count", 0)
        self.atomizer.load_state_dict(checkpoint["atomizer"])
        if "field_probe" in checkpoint:
            try:
                self.field_probe.load_state_dict(checkpoint["field_probe"])
            except Exception:
                pass  # feature_dim / d_model drift — keep fresh probe
        cfg = checkpoint.get("config") or {}
        if "enable_merge" in cfg:
            self.enable_merge = bool(cfg["enable_merge"])
        if "slow_every" in cfg:
            self.slow_every = max(1, int(cfg["slow_every"]))
        if "field_loss_weight" in cfg:
            self.field_loss_weight = float(cfg["field_loss_weight"])
        if "field_contrast_weight" in cfg:
            self.field_contrast_weight = float(cfg["field_contrast_weight"])
        if "field_contrast_margin" in cfg:
            self.field_contrast_margin = float(cfg["field_contrast_margin"])
        self.merge_count_total = int(cfg.get("merge_count_total", 0) or 0)
        # Always repair drifted dynamics (e.g. energy_decay~0.001 from 1.05M persist).
        dynamics_state = self.stabilize_dynamics_parameters()
        training = checkpoint.get("training")
        if training is None:
            training = {}
        training = dict(training)
        training["legacy_surface_migrated"] = bool(legacy_surface or missing or field_readout_missing)
        training["field_readout_migrated"] = bool(field_readout_missing)
        training["dynamics_on_load"] = dynamics_state
        return training
