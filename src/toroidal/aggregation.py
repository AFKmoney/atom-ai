"""Hierarchical aggregation and coherent MERGE driven by toroidal phase/energy."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AggregationEngine(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        phase_coherence_threshold: float = 0.55,
        energy_threshold: float = 0.3,
        max_depth: int = 5,
        merge_energy_floor: float = 0.08,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.phase_coherence_threshold = phase_coherence_threshold
        self.energy_threshold = energy_threshold
        self.max_depth = max_depth
        self.merge_energy_floor = merge_energy_floor

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
    def _circular_mean(phi: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        if weights is None:
            return torch.atan2(torch.sin(phi).mean(dim=0), torch.cos(phi).mean(dim=0))
        w = weights.reshape(-1, *([1] * (phi.ndim - 1)))
        return torch.atan2((torch.sin(phi) * w).sum(dim=0), (torch.cos(phi) * w).sum(dim=0))

    def _energy_weights(self, E: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        weights = E[idx].mean(dim=-1).clamp_min(1e-6)
        return weights / weights.sum()

    def _combine_cluster(
        self, cluster, r, phi, omega, E, kappa, M, tau, rho
    ) -> dict:
        """Energy-weighted honest combine of r/φ/ω/E/κ/M/τ/ρ into one heavier atom."""
        idx = torch.tensor(cluster, device=r.device, dtype=torch.long)
        weights = self._energy_weights(E, idx)
        r_m = F.normalize((r[idx] * weights[:, None]).sum(dim=0), dim=-1)
        phi_m = self._circular_mean(phi[idx], weights)
        omega_m = (omega[idx] * weights[:, None]).sum(dim=0)
        e_sum = (E[idx] * weights[:, None]).sum(dim=0)
        # Structural mass boost while staying in [0, 1].
        e_boost = (1.0 - (1.0 - e_sum).clamp(0, 1) ** len(cluster)).clamp(0, 1)
        kappa_m = (kappa[idx] * weights[:, None]).sum(dim=0)
        M_m = F.normalize((M[idx] * weights[:, None]).sum(dim=0), dim=-1)
        tau_m = (tau[idx] * weights[:, None]).sum(dim=0)
        rho_m = rho[idx].max(dim=0).values + 1
        return {
            "r": r_m,
            "phi": phi_m,
            "omega": omega_m,
            "E": e_boost,
            "kappa": kappa_m,
            "M": M_m,
            "tau": tau_m,
            "rho": rho_m,
            "constituents": list(cluster),
        }

    def aggregate(self, r, phi, omega, E, kappa, M, tau, rho) -> tuple[list[dict], list[list[int]]]:
        tensors = [x.unsqueeze(0) if x.dim() == 1 else x for x in (r, phi, omega, E, kappa, M, tau, rho)]
        if tensors[0].dim() == 3 and tensors[0].shape[-2] == 1:
            tensors = [x.squeeze(-2) for x in tensors]
        r, phi, omega, E, kappa, M, tau, rho = tensors
        if r.shape[0] < 2:
            return [], []
        coherence = self.compute_coherence(phi, E)
        clusters = self.find_clusters(coherence)
        aggregates = [
            self._combine_cluster(c, r, phi, omega, E, kappa, M, tau, rho) for c in clusters
        ]
        return aggregates, clusters

    def merge_coherent(
        self,
        r, phi, omega, E, kappa, M, tau, rho,
        *,
        max_merges: int = 8,
    ) -> tuple[list[dict], list[int], int]:
        """MERGE phase-coherent, energetic atoms into heavier structure.

        Returns (merged_atom_dicts, remove_indices, merge_count).
        """
        tensors = [x.unsqueeze(0) if x.dim() == 1 else x for x in (r, phi, omega, E, kappa, M, tau, rho)]
        if tensors[0].dim() == 3 and tensors[0].shape[-2] == 1:
            tensors = [x.squeeze(-2) for x in tensors]
        r, phi, omega, E, kappa, M, tau, rho = tensors
        n = int(r.shape[0])
        if n < 2:
            return [], [], 0

        coherence = self.compute_coherence(phi, E)
        available = torch.ones(n, dtype=torch.bool, device=r.device)
        mean_E = E.mean(dim=-1)
        merged: list[dict] = []
        remove: list[int] = []
        merge_count = 0

        tri = torch.triu(coherence, diagonal=1)
        flat = tri.reshape(-1)
        k = min(max(n * 2, 8), flat.numel())
        top_vals, top_idx = torch.topk(flat, k=k)
        for score, flat_i in zip(top_vals.tolist(), top_idx.tolist()):
            if merge_count >= max_merges:
                break
            if score < self.phase_coherence_threshold:
                break
            i = flat_i // n
            j = flat_i % n
            if i == j or not bool(available[i] and available[j]):
                continue
            if float(mean_E[i]) < self.merge_energy_floor and float(mean_E[j]) < self.merge_energy_floor:
                continue
            if float(torch.sqrt(mean_E[i] * mean_E[j])) < self.merge_energy_floor * 0.5:
                continue
            cluster = [i, j]
            merged.append(self._combine_cluster(cluster, r, phi, omega, E, kappa, M, tau, rho))
            remove.extend(cluster)
            available[i] = False
            available[j] = False
            merge_count += 1
        return merged, sorted(set(remove)), merge_count

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
            current = [
                torch.stack([a[key] for a in aggregates])
                for key in ("r", "phi", "omega", "E", "kappa", "M", "tau", "rho")
            ]
        return hierarchy
