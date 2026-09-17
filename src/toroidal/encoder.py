"""Token + context -> structured toroidal atom."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .atom import ToroidalAtom


class ToroidalEncoder(nn.Module):
    """Shared encoder that creates/modifies computational matter."""
    def __init__(self, vocab_size=32000, d_model=256, d_hidden=512, n_atoms_max=1024, tau_eps=1e-6):
        super().__init__()
        self.vocab_size, self.d_model, self.n_atoms_max, self.tau_eps = vocab_size, d_model, n_atoms_max, tau_eps
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.mapping = nn.Sequential(nn.Linear(d_model, d_hidden), nn.GELU(), nn.Linear(d_hidden, 8 * d_model))
        self.op_gate = nn.Linear(d_model, 7)
        self.theta_shared = nn.ParameterDict({
            "coupling_scale": nn.Parameter(torch.tensor(1.0)),
            "energy_decay": nn.Parameter(torch.tensor(0.99)),
            "phase_sync_strength": nn.Parameter(torch.tensor(0.5)),
        })

    def forward(self, token_ids, context_ids=None, existing_atoms=None):
        tok = self.token_embedding(token_ids)
        if context_ids is not None:
            ctx = self.token_embedding(context_ids).mean(dim=-2) if context_ids.dim() > 1 else self.token_embedding(context_ids)
            tok = tok + 0.25 * ctx
        raw = self.mapping(tok).view(tok.shape[0], 8, self.d_model)
        r = F.normalize(raw[:, 0], dim=-1)
        phi = raw[:, 1]
        omega = F.softplus(raw[:, 2]) + 1e-3
        E = torch.sigmoid(raw[:, 3])
        kappa = F.softplus(raw[:, 4])
        M = F.normalize(raw[:, 5], dim=-1)
        tau = F.softplus(raw[:, 6]) + self.tau_eps
        # Hierarchical level is structural metadata, not random noise.
        base_rho = len(existing_atoms) if existing_atoms is not None else 0
        rho = torch.full((tok.shape[0], 1), min(base_rho // 64, 4), dtype=tok.dtype, device=tok.device)
        new_atom = ToroidalAtom(r=r, phi=phi, omega=omega, E=E, kappa=kappa, M=M, tau=tau, rho=rho)
        op_logits = self.op_gate(tok)
        operation = op_logits.argmax(dim=-1)
        confidence = op_logits.softmax(dim=-1).max(dim=-1).values
        return new_atom, operation, confidence

    def get_shared_params(self):
        return {k: v.detach().clone() for k, v in self.theta_shared.items()}

    def load_shared_params(self, params):
        with torch.no_grad():
            for k, v in params.items():
                if k in self.theta_shared:
                    self.theta_shared[k].copy_(v)
