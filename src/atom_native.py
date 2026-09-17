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
    """Decode one variable-length surface packet from the toroidal state.

    The output alphabet is raw bytes (256 classes), not GPT-2's 50,257 token
    vocabulary.  A bounded packet contains 1..``max_payload_bytes`` bytes and
    has a separate length distribution.  This keeps exact byte reconstruction
    possible while allowing one recurrent ATOM tick to carry a span.

    A LayerNorm sits in front of the linear heads so small toroidal field
    amplitudes are not drowned by decoder bias (a failure mode observed when
    ``|W x| << |b|`` made every prompt decode to the same packet).
    """

    # Printable UTF-8 / Latin-1 friendly prior used when ``prefer_printable``.
    _PRINTABLE = set(range(0x20, 0x7F)) | {0x09, 0x0A, 0x0D} | set(range(0xC2, 0xF5)) | set(range(0x80, 0xC0))

    def __init__(self, d_model: int, max_payload_bytes: int = 32) -> None:
        super().__init__()
        if max_payload_bytes < 1:
            raise ValueError("max_payload_bytes must be positive")
        self.d_model = d_model
        self.max_payload_bytes = max_payload_bytes
        self.state_norm = nn.LayerNorm(d_model)
        self.byte_decoder = nn.Linear(d_model, max_payload_bytes * 256)
        self.length_decoder = nn.Linear(d_model, max_payload_bytes)

    def forward(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
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
        self.surface = AtomSurfaceHead(d_model, max_payload_bytes=max_payload_bytes)
        self.max_payload_bytes = max_payload_bytes
        self.field_controller = FieldAmplitudeController(field_max_rms)
        self.field_max_rms = field_max_rms
        # Always-on physical band.  The weak prior floor (1e-3) let persist
        # training drift energy_decay to ~0.001 and collapse the field.
        if energy_decay_bounds is None:
            energy_decay_bounds = DEFAULT_ENERGY_DECAY_BOUNDS
        self.energy_decay_bounds = energy_decay_bounds

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
        if reset_atomizer:
            self.atomizer.reset()

    def _output_state(
        self,
        alpha: torch.Tensor,
        persistent_state: torch.Tensor | None,
        abstractions: list[dict],
    ) -> torch.Tensor:
        state = alpha.mean(dim=0) if alpha.ndim > 1 else alpha
        if persistent_state is not None:
            state = state + (
                persistent_state.mean(dim=0)
                if persistent_state.ndim > 1
                else persistent_state
            )
        if abstractions:
            patterns = [
                item["pattern"].to(device=state.device, dtype=state.dtype).reshape(-1)
                for item in abstractions
                if item["pattern"].numel() == self.core.encoder.d_model
            ]
            if patterns:
                state = state + 0.1 * torch.stack(patterns).mean(dim=0)
        return state

    def _advance(self, atom: ToroidalAtom, operation: torch.Tensor, confidence: torch.Tensor) -> dict:
        """Advance the unchanged toroidal core by one compiled atom."""
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

        alpha = self.core.state.get_field().detach().clone()
        alpha_new = self.core.dynamics.evolve(alpha, input_token=atom.r)
        interaction_energy = self.core.interaction.field_interaction(alpha_new)
        left = torch.roll(alpha_new, 1, 0)
        right = torch.roll(alpha_new, -1, 0)
        alpha_new = alpha_new + 0.1 * self.core.dynamics.dynamics.phase_sync * (
            left + right - 2.0 * alpha_new
        )
        alpha_new, amplitude = self.field_controller(alpha_new)

        aggregates: list[dict] = []
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
        abstractions: list[dict] = []
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

        persistent_state = None
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

        output_state = self._output_state(alpha_new, persistent_state, abstractions)
        surface = self.surface(output_state)
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
            **amplitude,
        }

    def forward_packet(self, packet: AtomPacket) -> dict:
        features = packet.features.to(self.core.state.alpha.device)
        atom, operation, confidence = self.compiler(features, atom_count=len(self.core.atoms))
        return self._advance(atom, operation, confidence)

    def transition_loss(self, current: AtomPacket, target: AtomPacket) -> tuple[torch.Tensor, dict]:
        """Predict the complete next atom surface packet."""
        output = self.forward_packet(current)
        surface = output["surface"]
        payload = target.payload[: self.max_payload_bytes]
        target_length = torch.tensor(
            [len(payload) - 1], dtype=torch.long, device=surface["length_logits"].device
        )
        length_loss = F.cross_entropy(surface["length_logits"].unsqueeze(0), target_length)
        target_bytes = torch.tensor(list(payload), dtype=torch.long, device=surface["byte_logits"].device)
        byte_logits = surface["byte_logits"][: len(payload)]
        byte_loss = F.cross_entropy(byte_logits, target_bytes)
        loss = length_loss + byte_loss
        return loss, {
            "loss": float(loss.detach().item()),
            "length_loss": float(length_loss.detach().item()),
            "byte_loss": float(byte_loss.detach().item()),
            "n_atoms": len(self.core.atoms),
            "surface": surface,
            "field_norm": float(output["field"].detach().norm().item()),
            "field_rms_before": output["field_rms_before"],
            "field_rms_after": output["field_rms_after"],
            "field_scale": output["field_scale"],
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
            },
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
        # Legacy checkpoints predate ``state_norm``.  Load matching tensors
        # strictly, keep fresh LayerNorm init, and damp oversized biases that
        # previously drowned the field signal (|W x| / |b| ~ 0.005).
        legacy_surface = "state_norm.weight" not in surface_state
        current = self.surface.state_dict()
        filtered = {key: value for key, value in surface_state.items() if key in current and current[key].shape == value.shape}
        missing = [key for key in current if key not in filtered]
        self.surface.load_state_dict(filtered, strict=False)
        if legacy_surface or missing:
            with torch.no_grad():
                # Fresh LayerNorm around the trained operating point.
                self.surface.state_norm.reset_parameters()
                # Shrink decoder bias so normalized state can compete; continued
                # training re-fits the bias under LayerNorm.
                self.surface.byte_decoder.bias.mul_(0.05)
                self.surface.length_decoder.bias.mul_(0.05)
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
        # Always repair drifted dynamics (e.g. energy_decay~0.001 from 1.05M persist).
        dynamics_state = self.stabilize_dynamics_parameters()
        training = checkpoint.get("training")
        if training is None:
            training = {}
        training = dict(training)
        training["legacy_surface_migrated"] = bool(legacy_surface or missing)
        training["dynamics_on_load"] = dynamics_state
        return training
