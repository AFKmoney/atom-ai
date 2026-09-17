"""Hierarchical aggregation driven by toroidal coherence."""
from __future__ import annotations
import torch
import torch.nn as nn


class AggregationEngine(nn.Module):
    def __init__(self, d_model: int = 256, phase_coherence_threshold: float = 0.7,
                 energy_threshold: float = 0.3, max_depth: int = 5) -> None:
        super().__init__()
        self.d_model = d_model
        self.phase_coherence_threshold = phase_coherence_threshold
        self.energy_threshold = energy_threshold
        self.max_depth = max_depth

    def compute_coherence(self, phi: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        dphi = phi[:, None, :] - phi[None, :, :]
        phase = torch.cos(dphi).mean(dim=-1)
        energy = torch.sqrt((E[:, None, :] * E[None, :, :]).clamp_min(0))
        return phase * energy.mean(dim=-1)

    def find_clusters(self, coherence: torch.Tensor, min_size: int = 2) -> list[list[int]]:
        n = coherence.shape[0]
        if n == 0:
            return []
        mask = coherence > self.phase_coherence_threshold
        mask.fill_diagonal_(False)
        visited = torch.zeros(n, dtype=torch.bool, device=coherence.device)
        clusters: list[list[int]] = []
        for i in range(n):
            if visited[i]:
                continue
            queue = [i]
            visited[i] = True
            cluster: list[int] = []
            while queue:
                node = queue.pop()
                cluster.append(node)
                neighbours = torch.nonzero(mask[node] & ~visited, as_tuple=False).flatten().tolist()
                for j in neighbours:
                    visited[j] = True
                    queue.append(j)
            if len(cluster) >= min_size:
                clusters.append(cluster)
        return clusters

    @staticmethod
    def _circular_mean(phi: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(phi).mean(dim=0), torch.cos(phi).mean(dim=0))

    def aggregate(self, r, phi, omega, E, kappa, M, tau, rho) -> tuple[list[dict], list[list[int]]]:
        tensors = [x.unsqueeze(0) if x.dim() == 1 else x for x in (r, phi, omega, E, kappa, M, tau, rho)]
        if tensors[0].dim() == 3 and tensors[0].shape[-2] == 1:
            tensors = [x.squeeze(-2) for x in tensors]
        r, phi, omega, E, kappa, M, tau, rho = tensors
        if r.shape[0] < 2:
            return [], []
        coherence = self.compute_coherence(phi, E)
        clusters = self.find_clusters(coherence)
        aggregates = []
        for cluster in clusters:
            idx = torch.tensor(cluster, device=r.device)
            weights = E[idx].mean(dim=-1).clamp_min(1e-6)
            weights = weights / weights.sum()
            aggregates.append({
                "r": (r[idx] * weights[:, None]).sum(dim=0),
                "phi": self._circular_mean(phi[idx]),
                "omega": (omega[idx] * weights[:, None]).sum(dim=0),
                "E": E[idx].mean(dim=0).clamp(0, 1),
                "kappa": kappa[idx].mean(dim=0),
                "M": M[idx].mean(dim=0),
                "tau": tau[idx].mean(dim=0),
                "rho": rho[idx].max(dim=0).values + 1,
            })
        return aggregates, clusters

    def hierarchical_aggregate(self, r, phi, omega, E, kappa, M, tau, rho, depth: int = 3):
        depth = min(depth, self.max_depth)
        hierarchy = []
        current = [x.clone() for x in (r, phi, omega, E, kappa, M, tau, rho)]
        for _ in range(depth):
            aggregates, _ = self.aggregate(*current)
            if not aggregates:
                break
            hierarchy.append(aggregates)
            if len(aggregates) < 2:
                break
            current = [torch.stack([a[key] for a in aggregates]) for key in ("r", "phi", "omega", "E", "kappa", "M", "tau", "rho")]
        return hierarchy
