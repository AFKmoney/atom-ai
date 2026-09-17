"""
ToroidalAtom: a single computational primitive with 8 properties.

Design choice: atoms are represented as *tensors* (not Python objects)
so that the entire collection can be batched and differentiable.
Each property is a torch.Tensor of shape [d_model] (or scalar where appropriate).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ToroidalAtom(nn.Module):
    """
    A single toroidal atom.

    Properties (all differentiable tensors):
        r      [d_model]  — position / radius direction in latent space
        phi    [d_model]  — phase (angle latent)
        omega  [d_model]  — natural frequency (positive)
        E      [d_model]  — energy / activation / importance (0..1)
        kappa  [d_model]  — coupling strength (positive)
        M      [d_model]  — local memory direction
        tau    [d_model]  — temporal scale (positive)
        rho    [1]       — hierarchical level (integer)
    """

    def __init__(
        self,
        r: torch.Tensor,
        phi: torch.Tensor,
        omega: torch.Tensor,
        E: torch.Tensor,
        kappa: torch.Tensor,
        M: torch.Tensor,
        tau: torch.Tensor,
        rho: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("r", r)
        self.register_buffer("phi", phi)
        self.register_buffer("omega", omega)
        self.register_buffer("E", E)
        self.register_buffer("kappa", kappa)
        self.register_buffer("M", M)
        self.register_buffer("tau", tau)
        self.register_buffer("rho", rho)

    def energy(self) -> torch.Tensor:
        """Scalar energy of this atom."""
        return self.E.sum()

    def phase_distance(self, other: "ToroidalAtom") -> torch.Tensor:
        """Phase difference (wrapped) with another atom."""
        return torch.remainder(self.phi - other.phi, 2 * torch.pi)

    def coupling_to(self, other: "ToroidalAtom") -> torch.Tensor:
        """Pairwise coupling K_ij = kappa_ij * cos(phi_i - phi_j)."""
        dphi = self.phase_distance(other)
        return self.kappa * other.kappa * torch.cos(dphi)

    def clone(self) -> "ToroidalAtom":
        return ToroidalAtom(
            r=self.r.clone(),
            phi=self.phi.clone(),
            omega=self.omega.clone(),
            E=self.E.clone(),
            kappa=self.kappa.clone(),
            M=self.M.clone(),
            tau=self.tau.clone(),
            rho=self.rho.clone(),
        )

    def to(self, device):
        return ToroidalAtom(
            r=self.r.to(device),
            phi=self.phi.to(device),
            omega=self.omega.to(device),
            E=self.E.to(device),
            kappa=self.kappa.to(device),
            M=self.M.to(device),
            tau=self.tau.to(device),
            rho=self.rho.to(device),
        )


class ToroidalAtomCollection(nn.Module):
    """
    A batch/collection of toroidal atoms.
    Stores atoms as stacked tensors for vectorized operations.
    """

    def __init__(self, atoms: list[ToroidalAtom] | None = None) -> None:
        super().__init__()
        self.atoms: list[ToroidalAtom] = atoms or []
        self._buffers: dict[str, torch.Tensor] = {}
        self._dirty = True
        if atoms:
            self._rebuild_buffers()

    def _rebuild_buffers(self) -> None:
        if not self.atoms:
            self._buffers = {}
            self._dirty = False
            return
        self._buffers = {
            "r": torch.stack([a.r for a in self.atoms]),
            "phi": torch.stack([a.phi for a in self.atoms]),
            "omega": torch.stack([a.omega for a in self.atoms]),
            "E": torch.stack([a.E for a in self.atoms]),
            "kappa": torch.stack([a.kappa for a in self.atoms]),
            "M": torch.stack([a.M for a in self.atoms]),
            "tau": torch.stack([a.tau for a in self.atoms]),
            "rho": torch.stack([a.rho for a in self.atoms]),
        }
        self._dirty = False

    def _ensure_buffers(self) -> None:
        if getattr(self, "_dirty", True):
            self._rebuild_buffers()

    @property
    def r(self) -> torch.Tensor:
        self._ensure_buffers()
        return self._buffers.get("r", torch.zeros(0))

    @property
    def phi(self) -> torch.Tensor:
        self._ensure_buffers()
        return self._buffers.get("phi", torch.zeros(0))

    @property
    def omega(self) -> torch.Tensor:
        self._ensure_buffers()
        return self._buffers.get("omega", torch.zeros(0))

    @property
    def E(self) -> torch.Tensor:
        self._ensure_buffers()
        return self._buffers.get("E", torch.zeros(0))

    @property
    def kappa(self) -> torch.Tensor:
        self._ensure_buffers()
        return self._buffers.get("kappa", torch.zeros(0))

    @property
    def M(self) -> torch.Tensor:
        self._ensure_buffers()
        return self._buffers.get("M", torch.zeros(0))

    @property
    def tau(self) -> torch.Tensor:
        self._ensure_buffers()
        return self._buffers.get("tau", torch.zeros(0))

    @property
    def rho(self) -> torch.Tensor:
        self._ensure_buffers()
        return self._buffers.get("rho", torch.zeros(0))

    def __len__(self) -> int:
        return len(self.atoms)

    def add(self, atom: ToroidalAtom) -> None:
        self.atoms.append(atom)
        self._dirty = True

    def extend(self, atoms: list[ToroidalAtom]) -> None:
        if atoms:
            self.atoms.extend(atoms)
            self._dirty = True

    def remove(self, indices: list[int]) -> None:
        # Remove in reverse order to preserve indices
        for i in sorted(indices, reverse=True):
            self.atoms.pop(i)
        self._dirty = True

    def top_by_energy(self, k: int) -> list[ToroidalAtom]:
        energies = self.E.sum(dim=-1) if self.E.numel() else torch.zeros(len(self.atoms))
        top_k = energies.topk(min(k, len(self.atoms)), dim=0).indices
        return [self.atoms[i.item()] for i in top_k]

    def to(self, device):
        new_atoms = [a.to(device) for a in self.atoms]
        collection = ToroidalAtomCollection(new_atoms)
        return collection

    def state_dict(self):
        return {
            "atoms": [
                {
                    "r": a.r.cpu(), "phi": a.phi.cpu(), "omega": a.omega.cpu(),
                    "E": a.E.cpu(), "kappa": a.kappa.cpu(), "M": a.M.cpu(),
                    "tau": a.tau.cpu(), "rho": a.rho.cpu(),
                }
                for a in self.atoms
            ]
        }

    def load_state_dict(self, state: dict) -> None:
        self.atoms = []
        for sd in state["atoms"]:
            self.atoms.append(ToroidalAtom(
                r=sd["r"], phi=sd["phi"], omega=sd["omega"],
                E=sd["E"], kappa=sd["kappa"], M=sd["M"],
                tau=sd["tau"], rho=sd["rho"],
            ))
        self._rebuild_buffers()
