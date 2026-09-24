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
from .speech_lock import format_dialogue_prompt, speech_ok
from .toroidal.atom import ToroidalAtom
from .toroidal.model import ToroidalFractalIntelligence

# Sane band for shared energy_decay (always applied).  Values near 0 turn the
# decay term into near-total wipe; values >1 flip it into growth.
DEFAULT_ENERGY_DECAY_BOUNDS: tuple[float, float] = (0.3, 0.95)

# Fixed distinct prompts for L_ign α_bar when --field-ignorance-prompt-bank is on.
# Probe-style set (not adjacent train ticks). Keep short for refresh cost.
DEFAULT_IGNORANCE_PROMPTS: tuple[str, ...] = (
    "Bonjour",
    "Qui es-tu ?",
    "Il était une fois",
    "Utilisateur: Bonjour\nAssistant:",
    "The future of AI is",
    "Valkyria Chronicles III is",
    "Hello, how are you?",
    "Comment ça va aujourd'hui ?",
)


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


class AlphaOnlyMLP(nn.Module):
    """α-conditioned 2-layer map: Linear → GELU → Linear (no byte_decoder bypass).

    Used under ``field_obligatory_hard`` as the *only* trainable α→logit adapter.
    Input is the JL fingerprint (or any α-only feature); hidden defaults to 4*d.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        *,
        bias_out: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.fc1 = nn.Linear(self.in_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.out_dim, bias=bias_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class LastAtomReadout(nn.Module):
    """Last-atom readout + dentate 2-gram (ARCHITECTURE.md, bridges #20/#1).

    Next-byte logits from the last committed byte + previous byte + last atom
    phase. NOT payload-copy (no hist/winner broadcast).

    1. embed last byte, embed previous byte (2-gram)
    2. dentate: expand 4x -> ReLU -> keep top 25% -> project back
    3. add phase(cos phi, sin phi) projection, GELU
    4. linear to 256 logits
    """

    def __init__(self, d_model: int, embed_dim: int | None = None, expand: int = 4) -> None:
        super().__init__()
        embed_dim = int(embed_dim or d_model)
        self.embed_dim = embed_dim
        self.d_model = int(d_model)
        self.byte_embed = nn.Embedding(256, embed_dim)
        hidden = expand * 2 * embed_dim
        self.expand = nn.Linear(2 * embed_dim, hidden)
        self.contract = nn.Linear(hidden, embed_dim)
        self.phase_proj = nn.Linear(2 * d_model, embed_dim)
        self.head = nn.Linear(embed_dim, 256)

    def dentate_keep(self, hidden_dim: int) -> int:
        return max(1, hidden_dim // 4)

    def forward(self, last_byte: int, prev_byte: int, phi: torch.Tensor) -> torch.Tensor:
        """Return 256 next-byte logits. ``phi`` is (d_model,) live phase."""
        device = self.head.weight.device
        last = torch.tensor([int(last_byte) % 256], dtype=torch.long, device=device)
        prev = torch.tensor([int(prev_byte) % 256], dtype=torch.long, device=device)
        pair = torch.cat([self.byte_embed(last), self.byte_embed(prev)], dim=-1)
        h = F.relu(self.expand(pair))
        keep = self.dentate_keep(h.shape[-1])
        top_values, top_index = torch.topk(h, keep, dim=-1)
        sparse = torch.zeros_like(h).scatter(-1, top_index, top_values)
        z = self.contract(sparse)
        flat = phi.reshape(-1).to(device=device, dtype=z.dtype)
        if flat.numel() != self.d_model:
            raise ValueError(f"last-atom phi needs {self.d_model} elems, got {flat.numel()}")
        phase = torch.cat([torch.cos(flat), torch.sin(flat)], dim=-1)
        out = F.gelu(z + self.phase_proj(phase).unsqueeze(0))
        return self.head(out).reshape(256)


def _byte_cycle(buf: bytearray | bytes) -> bool:
    """True if the tail of ``buf`` closes a 2-8 byte repeat (anti-repeat).

    ARCHITECTURE.md says 2-6; extended to 8 because byte-tick outputs loop
    on 7-byte attractors (e.g. "Tu ont: "). Decode only, weights untouched.
    """
    n_buf = len(buf)
    for n in (2, 3, 4, 5, 6, 7, 8):
        if n_buf >= 2 * n and bytes(buf[-n:]) == bytes(buf[-2 * n : -n]):
            return True
    return False


def _tail_bytes(atom_payloads: list[bytes] | None) -> tuple[int, int]:
    """(last byte, previous byte) of living payloads; prev falls back to last."""
    tails = [bytes(p or b"") for p in (atom_payloads or [])]
    tails = [p for p in tails if p]
    last = tails[-1][-1] if tails else 0
    prev = tails[-2][-1] if len(tails) > 1 else last
    return last, prev


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
        last_atom_readout: bool = False,
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
        # Obligatory α readout (main CE path): full-α JL fingerprint → logits.
        # Frozen random map cannot be CE-collapsed; trainable residual can adapt.
        # zeros_like(α) ⇒ exact zero branch. Persist/atom_r cannot bypass.
        self.alpha_only_dim = 2 * d_model
        self._alpha_flat_dim = int(self.n_modes) * int(d_model)
        # Fixed JL projection of flattened α (buffer — not trained).
        jl = torch.empty(self.alpha_only_dim, self._alpha_flat_dim)
        torch.manual_seed(20260918)
        nn.init.normal_(jl, mean=0.0, std=(1.0 / self._alpha_flat_dim) ** 0.5)
        self.register_buffer("alpha_jl", jl, persistent=True)
        # Frozen random α→logit map (buffer) + trainable adapter.
        fr_b = torch.empty(max_payload_bytes * 256, self.alpha_only_dim)
        fr_l = torch.empty(max_payload_bytes, self.alpha_only_dim)
        nn.init.normal_(fr_b, mean=0.0, std=(1.0 / self.alpha_only_dim) ** 0.5)
        nn.init.normal_(fr_l, mean=0.0, std=(1.0 / self.alpha_only_dim) ** 0.5)
        self.register_buffer("alpha_byte_frozen", fr_b, persistent=True)
        self.register_buffer("alpha_length_frozen", fr_l, persistent=True)
        # Deeper α-only decode (hard-v2+): α_feat → hidden 4*d → logits.
        # Still conditioned ONLY on α (JL fingerprint); no byte_decoder/skip bypass.
        self.alpha_mlp_hidden = 4 * d_model
        self.alpha_byte_proj = AlphaOnlyMLP(
            self.alpha_only_dim,
            max_payload_bytes * 256,
            self.alpha_mlp_hidden,
            bias_out=False,
        )
        self.alpha_length_proj = AlphaOnlyMLP(
            self.alpha_only_dim,
            max_payload_bytes,
            self.alpha_mlp_hidden,
            bias_out=False,
        )
        # mix_w ∈ [obl_mix_floor, 1] applied to FROZEN branch — CE cannot shut off.
        self.obl_mix = nn.Parameter(torch.tensor(0.0))
        self.obl_mix_floor = 0.35
        self.obl_gate = nn.Parameter(torch.tensor(1.0))  # trainable additive scale
        self.obl_floor = 0.5
        self.field_obligatory_readout = False  # enabled via AtomNativeModel flag
        self.field_obligatory_hard = False  # hard-v2: floor=1.0 + freeze bypass + α-proj
        self._soft_obl_mix_floor = 0.35
        self.state_norm = nn.LayerNorm(d_model)
        self.byte_decoder = nn.Linear(d_model, max_payload_bytes * 256)
        self.length_decoder = nn.Linear(d_model, max_payload_bytes)
        self._init_field_readout()
        self._init_obligatory_readout()
        # Binary mask over byte alphabet (1=printable UTF-8 / Latin-1 friendly).
        # Used by decode bias and by printable_aux train loss on hard logits.
        _pm = torch.zeros(256)
        for value in self._PRINTABLE:
            _pm[value] = 1.0
        self.register_buffer("printable_mask", _pm, persistent=False)
        # Atom-payload production (hard path): per-atom α-local embed + copy-bias
        # (no mean-pool / global-hist). NOT byte_decoder bypass.
        self.byte_embed = nn.Embedding(256, d_model)
        self.payload_alpha_gate = nn.Linear(self.alpha_only_dim, d_model, bias=False)
        self.payload_to_byte = nn.Linear(d_model, max_payload_bytes * 256, bias=False)
        self.payload_to_length = nn.Linear(d_model, max_payload_bytes, bias=False)
        self.payload_copy_scale = nn.Parameter(torch.tensor(1.0))
        self.payload_enabled = True
        self._init_payload_production()
        self.last_atom_readout = bool(last_atom_readout)
        self.last_atom = None
        if self.last_atom_readout:
            if max_payload_bytes != 1:
                raise ValueError("last_atom_readout requires max_payload_bytes=1 (byte-tick)")
            self.last_atom = LastAtomReadout(d_model)

    def _init_payload_production(self) -> None:
        """Small init so payload branch is audible but does not swamp frozen α."""
        with torch.no_grad():
            nn.init.normal_(self.byte_embed.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.payload_alpha_gate.weight)
            # Slight identity-ish open: gate starts near 0.5 after sigmoid(0).
            nn.init.normal_(self.payload_to_byte.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.payload_to_length.weight, mean=0.0, std=0.02)
            self.payload_copy_scale.fill_(1.0)

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

    def _init_obligatory_readout(self) -> None:
        """Seed JL/frozen buffers + trainable adapter (deterministic)."""
        with torch.no_grad():
            gen = torch.Generator(device="cpu")
            gen.manual_seed(20260918)
            jl = torch.empty_like(self.alpha_jl, device="cpu")
            jl.normal_(generator=gen, mean=0.0, std=(1.0 / self._alpha_flat_dim) ** 0.5)
            self.alpha_jl.copy_(jl.to(device=self.alpha_jl.device, dtype=self.alpha_jl.dtype))
            for buf_name, fan_out in (
                ("alpha_byte_frozen", self.alpha_byte_frozen.shape[0]),
                ("alpha_length_frozen", self.alpha_length_frozen.shape[0]),
            ):
                buf = getattr(self, buf_name)
                tmp = torch.empty_like(buf, device="cpu")
                tmp.normal_(
                    generator=gen,
                    mean=0.0,
                    std=(1.0 / self.alpha_only_dim) ** 0.5,
                )
                buf.copy_(tmp.to(device=buf.device, dtype=buf.dtype))
            self._init_alpha_trainable_mlp(generator=gen)
            self.obl_gate.fill_(1.0)
            self.obl_mix.fill_(0.0)

    def _init_alpha_trainable_mlp(
        self, generator: torch.Generator | None = None
    ) -> None:
        """Xavier-uniform init for α-only MLP adapters (does not touch frozen JL/maps)."""
        if generator is None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(20260918)
        with torch.no_grad():
            for module in (self.alpha_byte_proj, self.alpha_length_proj):
                for layer in (module.fc1, module.fc2):
                    w = layer.weight
                    fan_in = w.shape[1]
                    bound = (6.0 / (fan_in + w.shape[0])) ** 0.5
                    w.copy_(
                        torch.empty_like(w, device="cpu")
                        .uniform_(-bound, bound, generator=generator)
                        .to(device=w.device, dtype=w.dtype)
                    )
                    if layer.bias is not None:
                        layer.bias.zero_()

    def alpha_only_features(self, alpha: torch.Tensor) -> torch.Tensor:
        """JL fingerprint of full flattened α — no persist/atom_r (zeros stay zeros).

        mean‖std alone collapses inter-prompt cosine (~0.93) while full α sits
        near ~0.71; a fixed JL map preserves that separation without new attention.
        """
        flat = alpha.reshape(-1)
        # Match buffer device/dtype; pad/trim if shape drifts.
        if flat.numel() < self._alpha_flat_dim:
            flat = F.pad(flat, (0, self._alpha_flat_dim - flat.numel()))
        elif flat.numel() > self._alpha_flat_dim:
            flat = flat[: self._alpha_flat_dim]
        flat = flat.to(device=self.alpha_jl.device, dtype=self.alpha_jl.dtype)
        return self.alpha_jl @ flat

    def obligatory_scale(self) -> torch.Tensor:
        """softplus(obl_gate) + floor — additive residual cannot reach zero."""
        return F.softplus(self.obl_gate) + float(self.obl_floor)

    def obligatory_mix_weight(self) -> torch.Tensor:
        """Convex mix weight ∈ [obl_mix_floor, 1] — CE cannot shut α-branch off."""
        floor = float(self.obl_mix_floor)
        return floor + (1.0 - floor) * torch.sigmoid(self.obl_mix)

    def payload_produce(
        self,
        atom_payloads: list[bytes] | None,
        a_hat: torch.Tensor,
        a_only: torch.Tensor,
        *,
        atom_energies: list[float | torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-atom α-local payload embed + copy-bias (no mean-pool / global-hist).

        Each living atom contributes from its own payload, gated by similarity of
        that atom's embed with the α-derived gate and by atom energy. Softmax over
        atoms keeps the mixture local to α so distinct prompt atoms diversify
        surface logits. Returns (byte_logits [L,256], length_logits [L]).
        Vanishes when α is exactly zero or no non-empty payloads.
        No Transformer QKV; no byte_decoder reopen.
        """
        device = a_hat.device
        dtype = a_hat.dtype
        L = self.max_payload_bytes
        zero_b = torch.zeros(L, 256, device=device, dtype=dtype)
        zero_l = torch.zeros(L, device=device, dtype=dtype)
        raw = list(atom_payloads or [])
        energy_raw = list(atom_energies) if atom_energies is not None else [1.0] * len(raw)
        if len(energy_raw) < len(raw):
            energy_raw = energy_raw + [1.0] * (len(raw) - len(energy_raw))

        embeds: list[torch.Tensor] = []
        lengths: list[int] = []
        local_hists: list[torch.Tensor] = []
        energies: list[torch.Tensor] = []
        kept_payloads: list[bytes] = []
        for payload, energy in zip(raw, energy_raw):
            payload_b = bytes(payload) if payload else b""
            ids = list(payload_b[:L])
            if not ids:
                continue
            kept_payloads.append(payload_b)
            lengths.append(len(ids))
            idx = torch.tensor(ids, device=device, dtype=torch.long)
            embeds.append(self.byte_embed(idx).mean(dim=0))
            hist_i = torch.zeros(256, device=device, dtype=dtype)
            hist_i.scatter_add_(
                0,
                idx,
                torch.ones(idx.numel(), device=device, dtype=dtype),
            )
            local_hists.append(hist_i)
            if torch.is_tensor(energy):
                e_val = energy.detach().to(device=device, dtype=dtype).reshape(-1).mean()
            else:
                e_val = torch.tensor(float(energy), device=device, dtype=dtype)
            energies.append(F.softplus(e_val).clamp_min(1e-6))
        if not embeds:
            return zero_b, zero_l

        gate = torch.sigmoid(self.payload_alpha_gate(a_hat))
        # α-local scores: payload-embed alignment with α-gate × energy.
        scores = []
        for embed, energy in zip(embeds, energies):
            sim = (embed * gate).sum()
            scores.append(sim + torch.log(energy))
        score_t = torch.stack(scores, dim=0)
        # Soft pool for embed MLP path; hard α-local winner for copy-bias so one
        # living atom's payload owns surface bytes (no mean-pool / global-hist).
        pool_w = F.softmax(score_t, dim=0)
        copy_w = torch.zeros_like(score_t)
        copy_w[int(score_t.argmax().item())] = 1.0

        pooled = torch.stack(
            [pool_w[i] * embeds[i] for i in range(len(embeds))], dim=0
        ).sum(dim=0)
        gated = pooled * gate
        byte_logits = self.payload_to_byte(gated).view(L, 256)
        length_logits = self.payload_to_length(gated)

        # Architectural gain: single-atom copy must stay audible vs frozen α RMS (~1).
        copy = F.softplus(self.payload_copy_scale) * 4.0
        # Position-aligned copy only. No unigram broadcast across span slots.
        for i, payload in enumerate(kept_payloads):
            w = copy_w[i]
            for pos, b in enumerate(payload[:L]):
                byte_logits[pos, int(b)] = byte_logits[pos, int(b)] + copy * w
        if lengths:
            # Length prior from sharp α-local winner length (not uniform mean).
            mean_len = float(
                sum(copy_w[i].detach() * float(lengths[i]) for i in range(len(lengths)))
                / max(float(copy_w.detach().sum().item()), 1e-8)
            )
            li = max(0, min(L - 1, int(round(mean_len)) - 1))
            length_logits = length_logits.clone()
            length_logits[li] = length_logits[li] + copy

        # Presence gate: exact α=0 ⇒ off (hard zeroing); nonzero α ⇒ full-strength
        # payload branch (do not crush copy-bias by tiny JL RMS).
        alive = (a_only.detach().pow(2).sum() > 0).to(dtype=dtype)
        return alive * byte_logits, alive * length_logits

    def set_obligatory_hard(self, enabled: bool) -> None:
        """Hard-v2 obligatory: mix floor 1.0; freeze non-α bypass; α-proj + payload.

        Forward: logits = frozen_α_map(α) + cap * (scale*α_MLP(α) + payload_prod)
        with cap≤1 so trainable RMS cannot exceed frozen RMS.
        Decoder / field skip / gates stay zeroed+frozen; CE may train
        alpha_*_proj MLP (+ obl_gate) and atom-payload production. Field dynamics
        still train. Payload path is atom-native (not byte_decoder reopen).
        """
        enabled = bool(enabled)
        self.field_obligatory_hard = enabled
        if enabled:
            self.field_obligatory_readout = True
            self.obl_mix_floor = 1.0
            self._freeze_non_alpha_bypass(True)
        else:
            self.obl_mix_floor = float(getattr(self, "_soft_obl_mix_floor", 0.35))
            self._freeze_non_alpha_bypass(False)

    def _freeze_non_alpha_bypass(self, freeze: bool) -> None:
        """Zero + freeze decoder/skip/gates that can bypass α on the CE path."""
        modules = (
            self.field_to_state,
            self.field_byte_skip,
            self.field_length_skip,
            self.byte_decoder,
            self.length_decoder,
        )
        scalars = (self.field_gate, self.skip_gate, self.obl_mix)
        with torch.no_grad():
            if freeze:
                for module in modules:
                    for param in module.parameters():
                        param.zero_()
                        param.requires_grad_(False)
                for param in scalars:
                    param.zero_()
                    param.requires_grad_(False)
            else:
                for module in modules:
                    for param in module.parameters():
                        param.requires_grad_(True)
                for param in scalars:
                    param.requires_grad_(True)

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
        atom_payloads: list[bytes] | None = None,
        atom_energies: list[float | torch.Tensor] | None = None,
        last_phi: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Decode from living field α when provided; else legacy mean-state path.

        When ``field_obligatory_readout`` is on, an α-only residual (mean‖std)
        is always added into the main CE logits with a non-zero floor scale so
        persist/atom_r / decoder bias cannot fully bypass living α.

        Under ``field_obligatory_hard``, living ``atom_payloads`` drive an
        α-local per-atom payload branch (no mean-pool / global-hist) alongside α-MLP.
        """
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
            # When obligatory: skip uses α-only feats (padded) so shared persist/
            # atom_r cannot substitute for α inside the skip path either.
            if self.field_obligatory_readout:
                a_only = self.alpha_only_features(alpha)
                a_rms = a_only.pow(2).mean().sqrt().clamp_min(1e-8)
                a_hat = a_only / a_rms
                skip_in = F.pad(a_hat, (0, self.field_feat_dim - self.alpha_only_dim))
                skip_in = self.field_feat_norm(skip_in)
            else:
                skip_in = feats
            byte_logits = byte_logits + skip * self.field_byte_skip(skip_in).view(
                self.max_payload_bytes, 256
            )
            length_logits = length_logits + skip * self.field_length_skip(skip_in)
            if self.field_obligatory_readout:
                a_only = self.alpha_only_features(alpha)
                a_rms = a_only.pow(2).mean().sqrt().clamp_min(1e-8)
                a_hat = a_only / a_rms
                # Frozen branch (CE cannot collapse) + trainable adapter.
                frozen_byte = (self.alpha_byte_frozen @ a_hat).view(
                    self.max_payload_bytes, 256
                )
                frozen_len = self.alpha_length_frozen @ a_hat
                if getattr(self, "field_obligatory_hard", False):
                    # Hard speech mix (ARCHITECTURE.md): trainable α-MLP +
                    # 0.3 frozen JL + optional payload branch. No RMS cap on
                    # the trainable branch: the cap pinned cap≈0.01 and crushed
                    # trainable gradients ~100x (measured), blocking learning.
                    # Payload stays outside any cap; α=0 still zeros payload
                    # via unclamped a_scale. No decoder/skip reopen.
                    train_byte = self.alpha_byte_proj(a_hat).view(
                        self.max_payload_bytes, 256
                    )
                    train_len = self.alpha_length_proj(a_hat)
                    if getattr(self, "payload_enabled", True):
                        pay_byte, pay_len = self.payload_produce(
                            atom_payloads,
                            a_hat,
                            a_only,
                            atom_energies=atom_energies,
                        )
                    else:
                        pay_byte = torch.zeros_like(frozen_byte)
                        pay_len = torch.zeros_like(frozen_len)
                    obl = self.obligatory_scale()
                    la_byte = torch.zeros_like(frozen_byte)
                    if getattr(self, "last_atom", None) is not None:
                        last_b, prev_b = _tail_bytes(atom_payloads)
                        phi_vec = last_phi
                        if phi_vec is None:
                            phi_vec = torch.zeros(
                                self.d_model,
                                device=frozen_byte.device,
                                dtype=frozen_byte.dtype,
                            )
                        la_byte = self.last_atom(last_b, prev_b, phi_vec).unsqueeze(0)
                    return {
                        "byte_logits": obl * train_byte + 0.3 * frozen_byte + pay_byte + la_byte,
                        "length_logits": obl * train_len + 0.3 * frozen_len + pay_len,
                    }
                train_byte = self.alpha_byte_proj(a_hat).view(self.max_payload_bytes, 256)
                train_len = self.alpha_length_proj(a_hat)
                mix = self.obligatory_mix_weight()
                # Hard mix uses FROZEN map so inter-prompt α structure reaches logits.
                byte_logits = (1.0 - mix) * byte_logits + mix * frozen_byte
                length_logits = (1.0 - mix) * length_logits + mix * frozen_len
                obl = self.obligatory_scale()
                byte_logits = byte_logits + obl * train_byte
                length_logits = length_logits + obl * train_len
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
        mask = getattr(self, "printable_mask", None)
        if mask is None:
            penalty = torch.full(
                (256,),
                -strength,
                dtype=byte_logits.dtype,
                device=byte_logits.device,
            )
            for value in self._PRINTABLE:
                penalty[value] = 0.0
        else:
            # printable→0, non-printable→-strength (vectorized)
            penalty = (1.0 - mask.to(device=byte_logits.device, dtype=byte_logits.dtype)) * (-strength)
        # Keep NUL strongly suppressed; allow tab/LF/CR via printable set.
        penalty = penalty.clone()
        penalty[0] = -strength * 2.0
        return byte_logits + penalty

    def printable_mass(self, byte_logits: torch.Tensor) -> torch.Tensor:
        """Per-position softmax mass on printable UTF-8 / Latin-1 friendly bytes."""
        probs = byte_logits.softmax(dim=-1)
        mask = self.printable_mask.to(device=probs.device, dtype=probs.dtype)
        return (probs * mask).sum(dim=-1)

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
        # Hard α-MLP path: always prefer printable (chat_atom_native contract).
        if prefer_printable or getattr(self, "field_obligatory_hard", False):
            strength = 2.5 if getattr(self, "field_obligatory_hard", False) else 2.0
            byte_logits = self._apply_printable_bias(byte_logits, strength=strength)
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
        merge_coherence_threshold: float = 0.45,
        merge_energy_floor: float = 0.08,
        slow_every: int = 1,
        field_loss_weight: float = 0.0,
        field_contrast_weight: float = 0.0,
        field_contrast_margin: float = 0.55,
        field_ignorance_weight: float = 0.0,
        field_ignorance_margin: float = 0.85,
        field_ignorance_ablate_shared: bool = False,
        field_ignorance_prompt_bank: bool = False,
        field_obligatory_readout: bool = False,
        field_obligatory_hard: bool = False,
        last_atom_readout: bool = False,
        printable_aux_weight: float = 0.0,
        field_next_packet_weight: float = 0.0,
        slow_rms_rel_tol: float = 0.15,
    ) -> None:
        super().__init__()
        self.atomizer = atomizer or Atomizer(max_span_bytes=max_payload_bytes, pack_mode="linguistic")
        self.core = ToroidalFractalIntelligence(
            vocab_size=256,
            d_model=d_model,
            n_modes=n_modes,
            n_atoms_max=n_atoms_max,
        )
        self.compiler = AtomCompiler(self.atomizer.feature_dim, d_model)
        self.surface = AtomSurfaceHead(
            d_model,
            max_payload_bytes=max_payload_bytes,
            n_modes=n_modes,
            last_atom_readout=bool(last_atom_readout),
        )
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
        self.merge_coherence_threshold = float(merge_coherence_threshold)
        self.merge_energy_floor = float(merge_energy_floor)
        self.core.aggregation.phase_coherence_threshold = self.merge_coherence_threshold
        self.core.aggregation.merge_energy_floor = self.merge_energy_floor
        self.slow_every = max(1, int(slow_every))
        self.field_loss_weight = float(field_loss_weight)
        self.field_contrast_weight = float(field_contrast_weight)
        self.field_contrast_margin = float(field_contrast_margin)
        self.field_ignorance_weight = float(field_ignorance_weight)
        self.field_ignorance_margin = float(field_ignorance_margin)
        self.field_ignorance_ablate_shared = bool(field_ignorance_ablate_shared)
        self.field_ignorance_prompt_bank = bool(field_ignorance_prompt_bank)
        self.field_obligatory_hard = bool(field_obligatory_hard)
        self.last_atom_readout = bool(last_atom_readout)
        self.field_obligatory_readout = bool(field_obligatory_readout) or self.field_obligatory_hard
        # Light aux: push hard α-MLP logits toward printable UTF-8 mass (no decoder bypass).
        self.printable_aux_weight = float(printable_aux_weight)
        # Predict *next* packet features from α (+ atom_r in field_features).
        self.field_next_packet_weight = float(field_next_packet_weight)
        self.surface.field_obligatory_readout = self.field_obligatory_readout
        if self.field_obligatory_hard:
            self.surface.set_obligatory_hard(True)
        else:
            self.surface.field_obligatory_hard = False
        self.slow_rms_rel_tol = float(slow_rms_rel_tol)
        self.field_probe = nn.Linear(self.surface.field_feat_dim, self.atomizer.feature_dim)
        self.field_next_probe = nn.Linear(self.surface.field_feat_dim, self.atomizer.feature_dim)
        self.merge_count_total = 0
        self._tick = 0
        self._efference_every = 0  # set by trainer CLI (--efference-every)
        self._prev_field_rms: float | None = None
        # Detached rolling α snapshots for harder field-contrast negatives
        # (mode-shuffle alone is too weak vs CE; inter-prompt collapse returns).
        self._alpha_bank: list[torch.Tensor] = []
        self._alpha_bank_max = 8
        # Distinct-prompt α bank for L_ign (filled by _refresh_prompt_alpha_bank).
        self._prompt_alpha_bank: list[torch.Tensor] = []
        self._prompt_bank_refresh_every = 256
        self._ignorance_prompts: tuple[str, ...] = DEFAULT_IGNORANCE_PROMPTS

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

    def set_merge_thresholds(
        self,
        coherence_threshold: float | None = None,
        energy_floor: float | None = None,
    ) -> None:
        """Apply MERGE gates onto AggregationEngine (CLI / resume overrides)."""
        if coherence_threshold is not None:
            self.merge_coherence_threshold = float(coherence_threshold)
            self.core.aggregation.phase_coherence_threshold = self.merge_coherence_threshold
        if energy_floor is not None:
            self.merge_energy_floor = float(energy_floor)
            self.core.aggregation.merge_energy_floor = self.merge_energy_floor

    @staticmethod
    def _atom_prop_batch1(t: torch.Tensor, *, is_rho: bool = False) -> torch.Tensor:
        """Normalize MERGE outputs to AtomCompiler shapes: (1, d) / rho (1, 1)."""
        x = t.detach().clone().reshape(-1)
        if is_rho:
            return x[:1].reshape(1, 1)
        return x.reshape(1, -1)

    def _maybe_merge_atoms(self, *, enabled: bool | None = None) -> tuple[int, dict]:
        """MERGE coherent structural atoms into heavier ones (detached memory).

        ``enabled`` overrides ``self.enable_merge`` for one call. Chat/probe ingest
        passes ``enabled=False`` so prompt packets stay distinct living atoms;
        training leaves it ``None`` and keeps MERGE thr=0.45.
        """
        empty_diag = {"max_phase_coherence": 0.0, "n_pairs_above_energy_floor": 0}
        use_merge = self.enable_merge if enabled is None else bool(enabled)
        if not use_merge or len(self.core.atoms) < 2:
            return 0, empty_diag
        atoms = self.core.atoms
        merged, remove_idx, merge_count, diag = self.core.aggregation.merge_coherent(
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
            return 0, diag
        # Capture payloads before remove (MERGE concatenates constituent bytes).
        pre_payloads = [bytes(getattr(a, "payload", b"") or b"") for a in atoms.atoms]
        atoms.remove(remove_idx)
        # Aggregation squeezes the compiler's leading batch-1 dim for math; restore
        # (1, d) / (1, 1) so ToroidalAtomCollection can stack with live atoms.
        max_keep = max(64, int(self.max_payload_bytes) * 4)
        new_atoms = []
        for item in merged:
            constituents = item.get("constituents") or []
            merged_payload = b"".join(
                pre_payloads[i] for i in constituents if 0 <= i < len(pre_payloads)
            )
            if len(merged_payload) > max_keep:
                merged_payload = merged_payload[-max_keep:]
            new_atoms.append(
                ToroidalAtom(
                    r=self._atom_prop_batch1(item["r"]),
                    phi=self._atom_prop_batch1(item["phi"]),
                    omega=self._atom_prop_batch1(item["omega"]),
                    E=self._atom_prop_batch1(item["E"]),
                    kappa=self._atom_prop_batch1(item["kappa"]),
                    M=self._atom_prop_batch1(item["M"]),
                    tau=self._atom_prop_batch1(item["tau"]),
                    rho=self._atom_prop_batch1(item["rho"], is_rho=True),
                    payload=merged_payload,
                )
            )
        atoms.extend(new_atoms)
        self.merge_count_total += merge_count
        return merge_count, diag

    def _field_rms_stable(self, rms: float) -> bool:
        prev = self._prev_field_rms
        self._prev_field_rms = rms
        if prev is None:
            return False
        return abs(rms - prev) / (abs(prev) + 1e-6) <= self.slow_rms_rel_tol

    def _advance(
        self,
        atom: ToroidalAtom,
        operation: torch.Tensor,
        confidence: torch.Tensor,
        *,
        merge_enabled: bool | None = None,
    ) -> dict:
        """Advance the unchanged toroidal core by one compiled atom.

        Fast path (every tick): inject + evolve + surface.
        Slow path (``slow_every`` + RMS-stable): abstraction + consolidation.
        Optional MERGE densifies coherent structural memory without GPU farms.
        ``merge_enabled`` overrides training MERGE for chat/probe ingest only.
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
            payload=bytes(getattr(atom, "payload", b"") or b""),
        )
        self.core.atoms.add(memory_atom)
        merge_count, merge_diag = self._maybe_merge_atoms(enabled=merge_enabled)

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
        living_atoms = list(self.core.atoms.atoms)
        living_payloads = [
            bytes(getattr(a, "payload", b"") or b"") for a in living_atoms
        ]
        living_energies = [
            a.E.detach().float().mean() for a in living_atoms
        ]
        la_last_b, la_prev_b = _tail_bytes(living_payloads)
        surface = self.surface(
            output_state,
            alpha=alpha_new,
            persistent_state=persist_for_readout,
            atom_r=atom.r,
            atom_payloads=living_payloads,
            atom_energies=living_energies,
            last_phi=atom.phi.reshape(-1),
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
            "max_phase_coherence": float(merge_diag.get("max_phase_coherence", 0.0)),
            "n_pairs_above_energy_floor": int(merge_diag.get("n_pairs_above_energy_floor", 0)),
            "slow_tick": do_slow,
            "atom_r": atom.r,
            "la_bytes": (la_last_b, la_prev_b),
            "la_phi": atom.phi.reshape(-1).detach(),
            **amplitude,
        }

    def forward_packet(
        self, packet: AtomPacket, *, merge_enabled: bool | None = None
    ) -> dict:
        features = packet.features.to(self.core.state.alpha.device)
        atom, operation, confidence = self.compiler(features, atom_count=len(self.core.atoms))
        # Living atom carries the packet's surface bytes for payload production.
        atom.payload = bytes(packet.payload or b"")
        return self._advance(atom, operation, confidence, merge_enabled=merge_enabled)

    @torch.no_grad()
    def _refresh_prompt_alpha_bank(self) -> None:
        """Fill ``_prompt_alpha_bank`` from fixed distinct prompts without lasting
        mutation of the live train field / atoms / atomizer / train α-bank.

        Snapshots core + atomizer + tick bookkeeping, runs each prompt from a
        clean episode, stores detached α, then restores. Used only when
        ``field_ignorance_prompt_bank`` is on.
        """
        device = self.core.state.alpha.device
        snap = {
            "state": {k: v.detach().clone() if torch.is_tensor(v) else v
                      for k, v in self.core.state.state_dict().items()},
            "atoms": self.core.atoms.state_dict(),
            "consolidation": {
                k: v.detach().clone() if torch.is_tensor(v) else v
                for k, v in self.core.consolidation.state_dict().items()
            },
            "abstraction": {
                k: v.detach().clone() if torch.is_tensor(v) else v
                for k, v in self.core.abstraction.state_dict().items()
            },
            "atomizer": self.atomizer.state_dict(),
            "energy_history": list(self.core.energy_history),
            "consolidation_count": self.core.consolidation_count,
            "abstraction_count": self.core.abstraction_count,
            "tick": self._tick,
            "prev_field_rms": self._prev_field_rms,
            "alpha_bank": [a.detach().clone() for a in self._alpha_bank],
            "training": self.training,
        }
        was_training = self.training
        self.eval()
        bank: list[torch.Tensor] = []
        try:
            for text in self._ignorance_prompts:
                self.reset_state(reset_atomizer=True)
                # restore train α-bank after reset_state cleared it (temp)
                packets = self.atomizer.encode(text, reset=True)
                if not packets:
                    continue
                out = None
                for packet in packets:
                    out = self.forward_packet(packet)
                if out is not None:
                    bank.append(out["field"].detach().clone().to(device=device))
        finally:
            self.core.state.load_state_dict(snap["state"], strict=False)
            self.core.atoms.load_state_dict(snap["atoms"])
            self.core.consolidation.load_state_dict(snap["consolidation"])
            self.core.abstraction.load_state_dict(snap["abstraction"])
            self.atomizer.load_state_dict(snap["atomizer"])
            self.core.energy_history = snap["energy_history"]
            self.core.consolidation_count = snap["consolidation_count"]
            self.core.abstraction_count = snap["abstraction_count"]
            self._tick = snap["tick"]
            self._prev_field_rms = snap["prev_field_rms"]
            self._alpha_bank = snap["alpha_bank"]
            if was_training:
                self.train()
            else:
                self.eval()
        self._prompt_alpha_bank = bank

    def _field_auxiliary_losses(
        self,
        output: dict,
        current: AtomPacket,
        target: AtomPacket | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Keep surface conditioned on α; probe persistence + next-packet from field.

        Contrastive hinge: logits from true α must differ from zeroed / shuffled α
        (cosine below margin). Field-ignorance hinge: logits from true α must differ
        from logits under stopgrad mean/bank of *other* prompts' α (not zero).
        Persistence probe: reconstruct *current* packet features from spectral field
        feats. Next-packet production: predict *target* packet features from the same
        α (+ atom_r) features — small weight under hard. Weighted small vs CE.
        """
        device = output["field"].device
        zero = torch.zeros((), device=device)
        info: dict[str, float] = {
            "field_contrast_loss": 0.0,
            "field_persist_loss": 0.0,
            "field_next_packet_loss": 0.0,
            "field_ignorance_loss": 0.0,
            "printable_aux_loss": 0.0,
            "printable_mass_mean": float("nan"),
            "field_logit_cos_zero": float("nan"),
            "field_logit_cos_shuf": float("nan"),
            "field_logit_cos_ignorance": float("nan"),
        }
        contrast = zero
        persist = zero
        next_pkt = zero
        ignorance = zero
        printable = zero
        alpha = output["field"]
        atom_r = output.get("atom_r")
        persist_state = output.get("persistent_state")
        if persist_state is None:
            persist_state = self.core.consolidation.persistence
        true_flat = output["surface"]["byte_logits"].reshape(-1)
        # Last-atom readout: aux hinges compare same-bytes/different-alpha so
        # they keep isolating alpha (bytes held constant, no shape change).
        la_extra: dict = {}
        if getattr(self.surface, "last_atom", None) is not None:
            la_pair = output.get("la_bytes") or (0, 0)
            la_extra = {
                "atom_payloads": [bytes([la_pair[1]]), bytes([la_pair[0]])],
                "last_phi": output.get("la_phi"),
            }

        if self.field_contrast_weight > 0:
            surf_zero = self.surface(
                None,
                alpha=torch.zeros_like(alpha),
                persistent_state=persist_state,
                atom_r=atom_r,
                **la_extra,
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
                None, alpha=shuf_alpha, persistent_state=persist_state, atom_r=atom_r,
                **la_extra,
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
                        None, alpha=bank_alpha, persistent_state=persist_state, atom_r=atom_r,
                        **la_extra,
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

        feats = None
        if self.field_loss_weight > 0 or self.field_next_packet_weight > 0:
            feats = self.surface.field_feat_norm(
                self.surface.field_features(alpha, persist_state, atom_r)
            )
        if self.field_loss_weight > 0:
            pred = self.field_probe(feats)
            cur_feat = current.features.to(device=pred.device, dtype=pred.dtype)
            cos = F.cosine_similarity(pred.unsqueeze(0), cur_feat.unsqueeze(0)).squeeze()
            persist = 1.0 - cos
            info["field_persist_loss"] = float(persist.detach().item())

        # Field → next-packet production: predict *upcoming* packet features from α+atoms.
        if self.field_next_packet_weight > 0 and target is not None:
            pred_next = self.field_next_probe(feats)
            next_feat = target.features.to(device=pred_next.device, dtype=pred_next.dtype)
            cos_n = F.cosine_similarity(pred_next.unsqueeze(0), next_feat.unsqueeze(0)).squeeze()
            next_pkt = 1.0 - cos_n
            info["field_next_packet_loss"] = float(next_pkt.detach().item())

        # Field-ignorance: surface must not map any non-zero α to one shared
        # logit template. Compare ℓ(α) vs ℓ(sg[α_bar]) where α_bar is the
        # sliding mean (or bank) of *other* prompts — not zero, not same-prompt.
        if self.field_ignorance_weight > 0:
            if self.field_ignorance_prompt_bank:
                if not self._prompt_alpha_bank:
                    self._refresh_prompt_alpha_bank()
                source = self._prompt_alpha_bank
            else:
                # Bank already includes current snap (appended in transition_loss);
                # use other entries only.
                source = self._alpha_bank[:-1]
            others = [
                a for a in source
                if a.shape == alpha.shape and not torch.allclose(a, alpha.detach(), atol=1e-5)
            ]
            if others:
                alpha_bar = torch.stack(
                    [a.to(device=alpha.device, dtype=alpha.dtype) for a in others]
                ).mean(dim=0)
                alpha_bar = alpha_bar.detach()  # sg[α_bar]
                # Ablate shared atom_r / persistence on the α_bar call only so
                # the hinge isolates α-driven readout. ℓ(α) keeps live state.
                if self.field_ignorance_ablate_shared:
                    bar_persist = None
                    bar_atom_r = None
                else:
                    bar_persist = persist_state
                    bar_atom_r = atom_r
                surf_bar = self.surface(
                    None, alpha=alpha_bar, persistent_state=bar_persist, atom_r=bar_atom_r,
                    **la_extra,
                )
                cos_ign = F.cosine_similarity(
                    true_flat.unsqueeze(0),
                    surf_bar["byte_logits"].reshape(-1).unsqueeze(0),
                ).squeeze()
                ignorance = F.relu(cos_ign - self.field_ignorance_margin)
                info["field_logit_cos_ignorance"] = float(cos_ign.detach().item())
                info["field_ignorance_loss"] = float(ignorance.detach().item())

        # Printable UTF-8 aux on hard (or any) surface logits: reward mass on
        # printable bytes so CE under frozen+MLP stops favoring binary garbage.
        # Gradients flow into α-MLP only under hard (decoder frozen).
        if self.printable_aux_weight > 0:
            byte_logits = output["surface"]["byte_logits"]
            mass = self.surface.printable_mass(byte_logits)
            # 1 - mass ∈ [0,1]; also lightly penalize via -log(mass) for sharp push.
            printable = (1.0 - mass).mean() + 0.25 * (-mass.clamp_min(1e-6).log()).mean()
            info["printable_aux_loss"] = float(printable.detach().item())
            info["printable_mass_mean"] = float(mass.detach().mean().item())

        total_aux = (
            self.field_contrast_weight * contrast
            + self.field_loss_weight * persist
            + self.field_next_packet_weight * next_pkt
            + self.field_ignorance_weight * ignorance
            + self.printable_aux_weight * printable
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
            # Refresh distinct-prompt bank for L_ign (does not alter train bank).
            if (
                self.field_ignorance_prompt_bank
                and self.field_ignorance_weight > 0
                and (
                    not self._prompt_alpha_bank
                    or (self._tick > 0 and self._tick % self._prompt_bank_refresh_every == 0)
                )
            ):
                self._refresh_prompt_alpha_bank()
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
        aux_loss, aux_info = self._field_auxiliary_losses(output, current, target)
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
            "max_phase_coherence": float(output.get("max_phase_coherence", 0.0)),
            "n_pairs_above_energy_floor": int(output.get("n_pairs_above_energy_floor", 0)),
            "slow_tick": bool(output.get("slow_tick", True)),
            **aux_info,
        }

    def efference_loss(self, current: AtomPacket, target: AtomPacket) -> tuple[torch.Tensor, dict]:
        """One efference tick (bridge #9): predict from gold state, commit the
        model's OWN byte (generate-like), then CE on the gold target from that
        state. Trains recovery from free-run drift (train~=generate).

        Falls back to a normal transition when the atomizer holds a pending
        span (only possible off byte-tick; byte-tick buffer is always empty).
        """
        if len(self.atomizer._buffer) == 0:
            with torch.no_grad():
                pred_bytes = self.predict_packet(current, deterministic=True, merge_enabled=False)
            pred_packet = self.atomizer.packet_from_payload(bytes(pred_bytes))
            return self.transition_loss(pred_packet, target)
        return self.transition_loss(current, target)

    def free_run_aux_loss(
        self,
        start: AtomPacket,
        golds: list[AtomPacket],
        *,
        weight: float = 1.0,
    ) -> tuple[torch.Tensor, dict]:
        """Multi-tick free-run CE using the SAME commit path as generate_packets.

        Math (L_roll): for horizon H = len(golds),
          L_roll = (1/H) Σ_h CE(logits(s_h), gold_h)
        where s_0 = start, and s_{h+1} = packet_from_payload(argmax(decode(s_h)))
        with the discrete commit stop-grad (scheduled sampling). Gradients flow
        through each step's surface logits (α-MLP + last_atom + …) under the
        free-run field trajectory — closing teacher-forced ↔ chat mismatch.

        Mutates living field like generate (no restore): continuum after this
        step is free-run-drifted. Falls back to transition_loss when golds is
        empty or atomizer has a pending span.
        """
        if not golds:
            raise ValueError("free_run_aux_loss requires at least one gold packet")
        if len(self.atomizer._buffer) != 0:
            return self.transition_loss(start, golds[0])

        device = self.core.state.alpha.device
        total_ce = torch.zeros((), device=device)
        byte_losses: list[float] = []
        length_losses: list[float] = []
        current = start
        n_steps = 0
        last_output: dict | None = None
        for gold in golds:
            output = self.forward_packet(current, merge_enabled=False)
            last_output = output
            surface = output["surface"]
            payload = gold.payload[: self.max_payload_bytes]
            if not payload:
                # Skip empty gold; still need a commit to advance free-run.
                with torch.no_grad():
                    pred_bytes = self.surface.decode(
                        surface, deterministic=True, prefer_printable=True
                    )
                current = self.atomizer.packet_from_payload(bytes(pred_bytes))
                continue
            target_length = torch.tensor(
                [len(payload) - 1], dtype=torch.long, device=surface["length_logits"].device
            )
            length_loss = F.cross_entropy(
                surface["length_logits"].unsqueeze(0), target_length
            )
            target_bytes = torch.tensor(
                list(payload), dtype=torch.long, device=surface["byte_logits"].device
            )
            byte_logits = surface["byte_logits"][: len(payload)]
            byte_loss = F.cross_entropy(byte_logits, target_bytes)
            step_ce = length_loss + byte_loss
            total_ce = total_ce + step_ce
            byte_losses.append(float(byte_loss.detach().item()))
            length_losses.append(float(length_loss.detach().item()))
            n_steps += 1
            # Discrete free-run commit (stop-grad) — same decode as chat.
            with torch.no_grad():
                pred_bytes = self.surface.decode(
                    surface, deterministic=True, prefer_printable=True
                )
            current = self.atomizer.packet_from_payload(bytes(pred_bytes))

        if n_steps == 0:
            return self.transition_loss(start, golds[0])

        mean_ce = total_ce / float(n_steps)
        w = float(weight)
        loss = w * mean_ce
        mean_byte = sum(byte_losses) / len(byte_losses)
        mean_len = sum(length_losses) / len(length_losses)
        out = last_output or {}
        return loss, {
            "loss": float(loss.detach().item()),
            "ce_loss": float(mean_ce.detach().item()),
            "byte_loss": mean_byte,
            "length_loss": mean_len,
            "n_atoms": len(self.core.atoms),
            "field_norm": float(out.get("field", self.core.state.alpha).detach().norm().item())
                if "field" in out else float(self.core.state.alpha.detach().norm().item()),
            "field_rms_before": float(out.get("field_rms_before", 0.0) or 0.0),
            "field_rms_after": float(out.get("field_rms_after", 0.0) or 0.0),
            "field_scale": float(out.get("field_scale", 1.0) or 1.0),
            "merge_count": int(out.get("merge_count", 0) or 0),
            "merge_count_total": int(out.get("merge_count_total", self.merge_count_total) or self.merge_count_total),
            "max_phase_coherence": float(out.get("max_phase_coherence", 0.0) or 0.0),
            "n_pairs_above_energy_floor": int(out.get("n_pairs_above_energy_floor", 0) or 0),
            "slow_tick": bool(out.get("slow_tick", True)),
            "free_run_aux": True,
            "free_run_horizon": int(n_steps),
            "free_run_aux_weight": w,
            "field_contrast_loss": 0.0,
            "field_persist_loss": 0.0,
            "field_ignorance_loss": 0.0,
            "printable_aux_loss": 0.0,
            "printable_mass_mean": float("nan"),
            "field_next_packet_loss": 0.0,
        }

    @torch.no_grad()
    def predict_packet(
        self,
        packet: AtomPacket,
        temperature: float = 1.0,
        top_k: int | None = None,
        deterministic: bool = True,
        prefer_printable: bool = True,
        *,
        merge_enabled: bool | None = None,
    ) -> bytes:
        output = self.forward_packet(packet, merge_enabled=merge_enabled)
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
        merge_enabled: bool = False,
        payload_copy: bool = False,
        dialogue_wrap: bool = True,
        speech_gate: bool = True,
        anti_repeat: bool = True,
    ) -> bytes:
        """Prime on a prompt and generate bounded atom surface packets."""
        self.eval()
        if self.field_obligatory_hard:
            self.surface.set_obligatory_hard(True)
            prefer_printable = True
        prev_payload_flag = bool(getattr(self.surface, "payload_enabled", True))
        self.surface.payload_enabled = bool(payload_copy)
        try:
            if reset:
                self.reset_state(reset_atomizer=True)
            if dialogue_wrap:
                prompt = format_dialogue_prompt(prompt)
            prompt_packets = self.atomizer.encode(prompt, reset=reset)
            if not prompt_packets:
                raise ValueError("prompt must produce at least one atom packet")
            for packet in prompt_packets[:-1]:
                self.forward_packet(packet, merge_enabled=merge_enabled)
            current = prompt_packets[-1]
            generated = bytearray()
            for _ in range(max_packets):
                payload = self.predict_packet(
                    current,
                    temperature=temperature,
                    top_k=top_k,
                    deterministic=deterministic,
                    prefer_printable=prefer_printable,
                    merge_enabled=merge_enabled,
                )
                if anti_repeat and payload and _byte_cycle(generated + payload):
                    # N-gram anti-repeat (ARCHITECTURE.md): the new bytes close
                    # a 2-6 byte cycle -> draw once more. RNG state is
                    # saved/restored and the redraw is seeded by position so
                    # deterministic probes stay reproducible.
                    rng_state = torch.get_rng_state()
                    try:
                        torch.manual_seed(0xA700 + len(generated))
                        payload = self.predict_packet(
                            current,
                            temperature=0.7,
                            top_k=8,
                            deterministic=False,
                            prefer_printable=prefer_printable,
                            merge_enabled=merge_enabled,
                        )
                    finally:
                        torch.set_rng_state(rng_state)
                if not payload:
                    break
                # Byte-tick contract (ARCHITECTURE.md): the speech gate judges
                # the accumulated phrase, never a single (1-byte) packet.
                if (
                    speech_gate
                    and len(generated) >= 8
                    and not speech_ok(bytes(generated))
                ):
                    break
                generated.extend(payload)
                current = self.atomizer.packet_from_payload(payload)
                if max_length is not None and len(generated) >= max_length:
                    break
                if generated.endswith(b"\n\n") and len(generated) > 2:
                    break
            if max_length is not None:
                return bytes(generated[:max_length])
            return bytes(generated)
        finally:
            self.surface.payload_enabled = prev_payload_flag


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
                "merge_coherence_threshold": float(self.merge_coherence_threshold),
                "merge_energy_floor": float(self.merge_energy_floor),
                "slow_every": self.slow_every,
                "field_loss_weight": self.field_loss_weight,
                "field_contrast_weight": self.field_contrast_weight,
                "field_contrast_margin": self.field_contrast_margin,
                "field_ignorance_weight": self.field_ignorance_weight,
                "field_ignorance_margin": self.field_ignorance_margin,
                "field_ignorance_ablate_shared": self.field_ignorance_ablate_shared,
                "field_ignorance_prompt_bank": self.field_ignorance_prompt_bank,
                "field_obligatory_readout": bool(self.field_obligatory_readout),
                "field_obligatory_hard": bool(self.field_obligatory_hard),
                "payload_copy": bool(getattr(self.surface, "payload_enabled", True)),
                "last_atom_readout": bool(self.last_atom_readout),
                "printable_aux_weight": float(self.printable_aux_weight),
                "field_next_packet_weight": float(self.field_next_packet_weight),
                "merge_count_total": self.merge_count_total,
            },
            "field_probe": self.field_probe.state_dict(),
            "field_next_probe": self.field_next_probe.state_dict(),
        }

    def save(self, path: str | Path, extra_state: dict | None = None) -> str:
        checkpoint = self._checkpoint()
        if extra_state is not None:
            checkpoint["training"] = extra_state
        path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        return path


    @staticmethod
    def _pad_copy_leading(dst: torch.Tensor, src: torch.Tensor) -> int:
        """Copy leading rows/elems from src into dst; return rows copied on dim0."""
        if dst.ndim == 0 or src.ndim == 0:
            return 0
        n = min(int(dst.shape[0]), int(src.shape[0]))
        if n <= 0:
            return 0
        if dst.ndim == 1:
            dst[:n].copy_(src[:n].to(device=dst.device, dtype=dst.dtype))
        else:
            # Match trailing dims; truncate if src wider (should not happen on grow).
            if dst.shape[1:] != src.shape[1:]:
                # Best-effort: copy overlapping trailing slice.
                slices = [slice(0, n)]
                for d_dim, s_dim in zip(dst.shape[1:], src.shape[1:]):
                    slices.append(slice(0, min(d_dim, s_dim)))
                dst[tuple(slices)].copy_(src[tuple(slices)].to(device=dst.device, dtype=dst.dtype))
            else:
                dst[:n].copy_(src[:n].to(device=dst.device, dtype=dst.dtype))
        return n

    def _migrate_span_head_weights(self, surface_state: dict) -> dict:
        """Grow max_payload surface heads: keep prefix weights; init only new rows.

        Does **not** touch field/core. Used when resuming 16-byte hard ckpt into
        a larger max_payload (e.g. 32) under hard-v2 α-MLP.
        """
        info = {
            "span_head_migrated": False,
            "padded_keys": [],
            "old_max_payload": None,
            "new_max_payload": int(self.max_payload_bytes),
        }
        new_max = int(self.max_payload_bytes)
        # Infer old max from length_decoder bias/weight if present.
        old_len = surface_state.get("length_decoder.bias")
        if old_len is None:
            old_len = surface_state.get("length_decoder.weight")
        if old_len is None:
            old_len = surface_state.get("alpha_length_frozen")
        if old_len is None or not hasattr(old_len, "shape"):
            return info
        old_max = int(old_len.shape[0])
        info["old_max_payload"] = old_max
        if old_max >= new_max:
            return info  # shrink or equal — filtered load handles equal; shrink rare

        current = self.surface.state_dict()
        padded = []
        with torch.no_grad():
            for key, dst in current.items():
                if key not in surface_state:
                    continue
                src = surface_state[key]
                if not hasattr(src, "shape") or not hasattr(dst, "shape"):
                    continue
                if tuple(src.shape) == tuple(dst.shape):
                    continue
                # Only pad along output (dim0) growth for known span heads.
                if src.shape[0] >= dst.shape[0]:
                    continue
                if key.endswith(".fc1.weight") or key.endswith(".fc1.bias"):
                    continue  # MLP input layer not payload-sized
                n = self._pad_copy_leading(dst, src)
                if n > 0:
                    padded.append(key)
            # Re-apply padded tensors onto the module.
            if padded:
                patch = {k: current[k] for k in padded}
                self.surface.load_state_dict(patch, strict=False)
        info["span_head_migrated"] = bool(padded)
        info["padded_keys"] = padded
        return info


    def load(self, path: str | Path) -> dict | None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.compiler.load_state_dict(checkpoint["compiler"])
        surface_state = checkpoint["surface"]
        # Legacy checkpoints predate ``state_norm`` and/or field-α readout.
        # Load matching tensors; activate new field path; damp biases that drown α.
        legacy_surface = "state_norm.weight" not in surface_state
        field_readout_missing = "field_to_state.weight" not in surface_state
        has_alpha_proj = any(
            key.startswith("alpha_byte_proj") for key in surface_state
        )
        obligatory_missing = (not has_alpha_proj) or ("alpha_jl" not in surface_state)
        current = self.surface.state_dict()
        filtered = {key: value for key, value in surface_state.items() if key in current and current[key].shape == value.shape}
        missing = [key for key in current if key not in filtered]
        has_la_keys = any(key.startswith("last_atom") for key in surface_state)
        want_la = getattr(self.surface, "last_atom", None) is not None
        if has_la_keys and not want_la:
            print("warning: checkpoint has last-atom weights but model built without last_atom_readout (keys dropped)")
        if want_la and not has_la_keys:
            print("info: last_atom_readout on but checkpoint lacks last-atom weights (fresh init kept)")
        self.surface.load_state_dict(filtered, strict=False)
        # If max_payload grew (e.g. 16→32), pad-copy span heads before any reinit wipe.
        span_migrate = self._migrate_span_head_weights(surface_state)
        if span_migrate.get("span_head_migrated"):
            # Recompute missing after pad: only truly-absent keys remain "missing".
            current = self.surface.state_dict()
            filtered_keys = set(filtered) | set(span_migrate.get("padded_keys") or [])
            missing = [key for key in current if key not in filtered_keys]
        mlp_missing = any(
            key.startswith("alpha_byte_proj") or key.startswith("alpha_length_proj")
            for key in missing
        )
        def _is_payload_key(key: str) -> bool:
            return (
                key.startswith("byte_embed")
                or key.startswith("payload_")
            )
        payload_missing = any(_is_payload_key(key) for key in missing)
        non_mlp_missing = [
            key
            for key in missing
            if not (
                key.startswith("alpha_byte_proj")
                or key.startswith("alpha_length_proj")
                or key.startswith("last_atom")
                or _is_payload_key(key)
            )
        ]
        if payload_missing:
            with torch.no_grad():
                self.surface._init_payload_production()
        if legacy_surface or non_mlp_missing or field_readout_missing:
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

        frozen_alpha_missing = any(
            key in missing
            for key in (
                "alpha_jl",
                "alpha_byte_frozen",
                "alpha_length_frozen",
                "obl_gate",
                "obl_mix",
            )
        )
        if obligatory_missing or frozen_alpha_missing:
            with torch.no_grad():
                self.surface._init_obligatory_readout()
                # Slight bias damp so fresh α-only residual is audible on load.
                if not (legacy_surface or field_readout_missing):
                    self.surface.byte_decoder.bias.mul_(0.5)
                    self.surface.length_decoder.bias.mul_(0.5)
        elif mlp_missing:
            # Linear→MLP deepen: keep loaded frozen JL / maps; reinit trainable MLP only.
            with torch.no_grad():
                self.surface._init_alpha_trainable_mlp()
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
        if "field_next_probe" in checkpoint:
            try:
                self.field_next_probe.load_state_dict(checkpoint["field_next_probe"])
            except Exception:
                pass
        cfg = checkpoint.get("config") or {}
        if "enable_merge" in cfg:
            self.enable_merge = bool(cfg["enable_merge"])
        if "merge_coherence_threshold" in cfg:
            self.merge_coherence_threshold = float(cfg["merge_coherence_threshold"])
            self.core.aggregation.phase_coherence_threshold = self.merge_coherence_threshold
        if "merge_energy_floor" in cfg:
            self.merge_energy_floor = float(cfg["merge_energy_floor"])
            self.core.aggregation.merge_energy_floor = self.merge_energy_floor
        if "slow_every" in cfg:
            self.slow_every = max(1, int(cfg["slow_every"]))
        if "field_loss_weight" in cfg:
            self.field_loss_weight = float(cfg["field_loss_weight"])
        if "field_contrast_weight" in cfg:
            self.field_contrast_weight = float(cfg["field_contrast_weight"])
        if "field_contrast_margin" in cfg:
            self.field_contrast_margin = float(cfg["field_contrast_margin"])
        if "field_ignorance_weight" in cfg:
            self.field_ignorance_weight = float(cfg["field_ignorance_weight"])
        if "field_ignorance_margin" in cfg:
            self.field_ignorance_margin = float(cfg["field_ignorance_margin"])
        if "field_ignorance_ablate_shared" in cfg:
            self.field_ignorance_ablate_shared = bool(cfg["field_ignorance_ablate_shared"])
        if "field_ignorance_prompt_bank" in cfg:
            self.field_ignorance_prompt_bank = bool(cfg["field_ignorance_prompt_bank"])
        if "field_obligatory_readout" in cfg:
            self.field_obligatory_readout = bool(cfg["field_obligatory_readout"])
        if "printable_aux_weight" in cfg:
            self.printable_aux_weight = float(cfg["printable_aux_weight"])
        if "field_next_packet_weight" in cfg:
            self.field_next_packet_weight = float(cfg["field_next_packet_weight"])
        if "field_obligatory_hard" in cfg and bool(cfg["field_obligatory_hard"]):
            self.field_obligatory_hard = True
            self.field_obligatory_readout = True
            self.surface.set_obligatory_hard(True)
        else:
            self.field_obligatory_hard = False
            self.surface.field_obligatory_hard = False
            # Constructor / CLI may set obligatory after load; always sync surface.
            self.surface.field_obligatory_readout = bool(self.field_obligatory_readout)
        self.merge_count_total = int(cfg.get("merge_count_total", 0) or 0)
        if "payload_copy" in cfg:
            self.surface.payload_enabled = bool(cfg["payload_copy"])
        # Always repair drifted dynamics (e.g. energy_decay~0.001 from 1.05M persist).
        dynamics_state = self.stabilize_dynamics_parameters()
        training = checkpoint.get("training")
        if training is None:
            training = {}
        training = dict(training)
        training["legacy_surface_migrated"] = bool(
            legacy_surface or bool(non_mlp_missing) or field_readout_missing
            or bool(span_migrate.get("span_head_migrated"))
        )
        training["alpha_mlp_migrated"] = bool(mlp_missing and not obligatory_missing)
        training["field_readout_migrated"] = bool(field_readout_missing)
        training["field_obligatory_migrated"] = bool(obligatory_missing)
        training["payload_prod_migrated"] = bool(payload_missing)
        training["span_head_migrated"] = bool(span_migrate.get("span_head_migrated"))
        training["span_head_migrate_info"] = span_migrate
        training["dynamics_on_load"] = dynamics_state
        return training
