"""Complete Toroidal Fractal Intelligence pipeline."""
from __future__ import annotations
import torch
import torch.nn as nn
from .encoder import ToroidalEncoder
from .state import FractalSuperpositionState
from .dynamics import RK4DynamicsEngine, ToroidalDynamics
from .interaction import ToroidalInteraction
from .aggregation import AggregationEngine
from .abstraction import AbstractionEngine
from .consolidation import ConsolidationEngine
from .production import ProductionHead
from .atom import ToroidalAtomCollection

class ToroidalFractalIntelligence(nn.Module):
    def __init__(self, vocab_size=32000, d_model=256, n_modes=256, n_atoms_max=1024):
        super().__init__()
        self.encoder = ToroidalEncoder(vocab_size, d_model, n_atoms_max=n_atoms_max)
        self.state = FractalSuperpositionState(n_modes, d_model)
        self.dynamics = RK4DynamicsEngine(ToroidalDynamics(d_model, n_modes), dt=0.01, n_steps=4)
        self.interaction = ToroidalInteraction(d_model, n_modes)
        self.aggregation = AggregationEngine(d_model)
        self.abstraction = AbstractionEngine(d_model)
        self.consolidation = ConsolidationEngine(d_model)
        self.production = ProductionHead(d_model, vocab_size)
        self.atoms = ToroidalAtomCollection()
        self.energy_history = []
        self.training_loss = 0.0
        self.consolidation_count = 0
        self.abstraction_count = 0

    def forward(self, token_ids, context_ids=None):
        if token_ids.dim() > 1:
            token_ids = token_ids[:, -1]
        new_atom, operation, confidence = self.encoder(token_ids, context_ids, self.atoms)
        self.state.add_atom_contribution(new_atom.r, new_atom.phi, new_atom.omega, new_atom.E, new_atom.kappa)
        self.atoms.add(new_atom)
        # The committed state is recurrent memory; use a snapshot so committing
        # the next state cannot invalidate the autograd graph for this tick.
        alpha = self.state.get_field().detach().clone()
        alpha_new = self.dynamics.evolve(alpha, input_token=new_atom.r)
        interaction_energy = self.interaction.field_interaction(alpha_new)
        # Explicit toroidal interaction feeds the resulting state.
        left, right = torch.roll(alpha_new, 1, 0), torch.roll(alpha_new, -1, 0)
        alpha_new = alpha_new + 0.1 * self.dynamics.dynamics.phase_sync * (left + right - 2 * alpha_new)
        aggregates = []
        if len(self.atoms) > 10:
            aggregates, _ = self.aggregation.aggregate(self.atoms.r, self.atoms.phi, self.atoms.omega, self.atoms.E, self.atoms.kappa, self.atoms.M, self.atoms.tau, self.atoms.rho)
        abstractions = []
        for agg in aggregates:
            idx = self.abstraction.create_or_update_abstraction([agg])
            if idx >= 0:
                abstractions.append({"id": idx, "pattern": self.abstraction.abstraction_memory[idx].detach()})
                self.abstraction_count += 1
        self.abstraction.step_age()
        persistent_state = None
        if len(self.energy_history) > 5:
            E = self.atoms.E.mean(dim=0)
            stability = self.consolidation.compute_stability(E, self.energy_history)
            persistent_state, mask = self.consolidation.consolidate(alpha_new, E.unsqueeze(0), stability.unsqueeze(0))
            if mask.any(): self.consolidation_count += int(mask.sum().item())
        self.energy_history.append(self.atoms.E.mean(dim=0).detach())
        self.energy_history = self.energy_history[-20:]
        logits, conf = self.production.produce(alpha_new, persistent_state, abstractions)
        logits = logits.expand(token_ids.shape[0], -1)
        with torch.no_grad():
            self.state.alpha.copy_(alpha_new.detach())
            self.state.t.add_(self.dynamics.dt)
        return {"logits": logits, "field": alpha_new, "confidence": conf, "interaction_energy": interaction_energy, "aggregates": aggregates, "abstractions": abstractions, "persistent_state": persistent_state, "operation": operation}

    def train_step(self, token_ids, target_ids, optimizer, context_ids=None):
        optimizer.zero_grad(set_to_none=True)
        output = self.forward(token_ids, context_ids)
        if target_ids.dim() > 1: target_ids = target_ids[:, -1]
        loss = nn.CrossEntropyLoss()(output["logits"], target_ids)
        alpha = output["field"]
        total_loss = loss + 0.01 * alpha.abs().mean()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()
        self.training_loss = loss.item()
        return {"loss": loss.item(), "total_loss": total_loss.item(), "n_atoms": len(self.atoms), "n_aggregates": len(output["aggregates"]), "n_abstractions": len(output["abstractions"]), "consolidation_count": self.consolidation_count}

    def generate(self, prompt_ids, max_length=100, temperature=1.0, top_k=50):
        generated = prompt_ids.clone()
        for _ in range(max_length):
            out = self.forward(generated[-1:].unsqueeze(0))
            token_id, _, _ = self.production.sample(self.state.get_field(), out["persistent_state"], out["abstractions"], temperature, top_k)
            generated = torch.cat([generated, torch.tensor([token_id], device=generated.device)])
            if token_id == 2: break
        return generated

    def save(self, path, extra_state=None):
        checkpoint = {"encoder": self.encoder.state_dict(), "state": self.state.get_state_dict(), "dynamics": self.dynamics.state_dict(), "interaction": self.interaction.state_dict(), "aggregation": self.aggregation.state_dict(), "abstraction": self.abstraction.state_dict(), "consolidation": self.consolidation.state_dict(), "production": self.production.state_dict(), "atoms": self.atoms.state_dict(), "energy_history": self.energy_history, "training_loss": self.training_loss, "consolidation_count": self.consolidation_count, "abstraction_count": self.abstraction_count, "config": {"vocab_size": self.encoder.vocab_size, "d_model": self.encoder.d_model, "n_modes": self.state.n_modes, "n_atoms_max": self.encoder.n_atoms_max}}
        if extra_state is not None:
            checkpoint["training"] = extra_state
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        for name in ("encoder", "dynamics", "interaction", "aggregation", "abstraction", "consolidation", "production"): getattr(self, name).load_state_dict(checkpoint[name])
        self.state.load_state_dict(checkpoint["state"]); self.atoms.load_state_dict(checkpoint["atoms"])
        self.energy_history = checkpoint.get("energy_history", []); self.training_loss = checkpoint.get("training_loss", 0.0); self.consolidation_count = checkpoint.get("consolidation_count", 0); self.abstraction_count = checkpoint.get("abstraction_count", 0)
        return checkpoint.get("training")

    def get_shared_params_summary(self):
        return {"coupling_scale": self.dynamics.dynamics.coupling_scale.item(), "energy_decay": self.dynamics.dynamics.energy_decay.item(), "phase_sync": self.dynamics.dynamics.phase_sync.item(), "n_atoms": len(self.atoms), "n_modes": self.state.n_modes, "d_model": self.encoder.d_model}

def create_model(vocab_size=32000, d_model=256, n_modes=256, n_atoms_max=1024):
    return ToroidalFractalIntelligence(vocab_size, d_model, n_modes, n_atoms_max)
