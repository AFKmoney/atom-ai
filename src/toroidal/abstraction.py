"""Extract reusable higher-level patterns from toroidal aggregates."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class AbstractionEngine(nn.Module):
    def __init__(self, d_model: int = 256, abstraction_capacity: int = 64) -> None:
        super().__init__()
        self.d_model = d_model
        self.abstraction_capacity = abstraction_capacity
        self.register_buffer("abstraction_memory", torch.zeros(abstraction_capacity, d_model))
        self.register_buffer("abstraction_usage", torch.zeros(abstraction_capacity))
        self.register_buffer("abstraction_age", torch.zeros(abstraction_capacity, dtype=torch.long))

    def compute_pattern(self, aggregates: list[dict]) -> torch.Tensor | None:
        if not aggregates:
            return None
        # Structural signature: retain primitive atom information without a
        # learned feature projection. Normalize each property before fusion.
        parts = []
        for key in ("r", "phi", "omega", "E", "kappa", "M", "tau"):
            x = torch.stack([a[key] for a in aggregates]).mean(dim=0)
            parts.append(F.normalize(x, dim=-1) if key in ("r", "M") else x)
        pattern = sum(parts) / len(parts)
        return F.normalize(pattern, dim=-1)

    def match_abstraction(self, pattern: torch.Tensor) -> tuple[int, float]:
        if self.abstraction_capacity == 0:
            return -1, -1.0
        p = F.normalize(pattern.reshape(1, -1), dim=-1)
        mem = F.normalize(self.abstraction_memory, dim=-1)
        similarities = (p @ mem.T).squeeze(0)
        # Empty slots are never selected as existing abstractions.
        similarities = similarities.masked_fill(self.abstraction_usage <= 0, -1.0)
        idx = similarities.argmax().item()
        return idx, similarities[idx].item()

    def create_or_update_abstraction(self, aggregates: list[dict], threshold: float = 0.8) -> int:
        pattern = self.compute_pattern(aggregates)
        if pattern is None:
            return -1
        best_idx, best_sim = self.match_abstraction(pattern)
        with torch.no_grad():
            if best_idx >= 0 and best_sim >= threshold:
                self.abstraction_memory[best_idx].mul_(0.9).add_(0.1 * pattern)
                self.abstraction_usage[best_idx].add_(1)
                self.abstraction_age[best_idx].zero_()
                return best_idx
            free = torch.nonzero(self.abstraction_usage <= 0, as_tuple=False).flatten()
            if free.numel():
                idx = free[0].item()
            else:
                # Replace the least-used/oldest representation, never overwrite all memory.
                score = self.abstraction_usage + 0.01 * self.abstraction_age.float()
                idx = score.argmin().item()
            self.abstraction_memory[idx].copy_(pattern)
            self.abstraction_usage[idx] = 1
            self.abstraction_age[idx] = 0
            return idx

    def step_age(self) -> None:
        with torch.no_grad():
            self.abstraction_age.add_(1)

    def get_active_abstractions(self, min_similarity: float = 0.5) -> list[dict]:
        active = torch.nonzero(self.abstraction_usage > 0, as_tuple=False).flatten().tolist()
        return [{"id": i, "pattern": self.abstraction_memory[i].detach().cpu()} for i in active]
