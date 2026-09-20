"""Grow a checkpoint to a wider model (Net2Net-style tile + noise).

Usage:
    PYTHONPATH=. .venv/bin/python tools/grow_checkpoint.py \\
      --src checkpoints/byte_tick/atom_native_step_120000_epr4.pt \\
      --dst checkpoints/byte_tick/atom_native_step_120000_d32.pt \\
      --d-new 32 --n-new 32

Width doubling is function-preserving (new rows tiled, new cols zeroed,
embeddings tiled+noise); frozen JL buffers are kept fresh from the constructor
(same seed sequence); everything else is copied. Checkpoints stay
trainable: resume the destination with matching --d-model/--n-modes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.atom_native import DEFAULT_ENERGY_DECAY_BOUNDS, AtomNativeModel

def grow_jl(jl: torch.Tensor, n_old: int, d_old: int, noise: float,
            gen: torch.Generator) -> torch.Tensor:
    """Interleaved JL growth preserving the fingerprint on tiled α.

    New α (2n×2d) flat row-major holds old elem (i,j) at i*2d+j.
    Fingerprint block 2 = block 1 + noise, so rms (hence a_hat) is kept.
    """
    n_new, d_new = 2 * n_old, 2 * d_old
    assert tuple(jl.shape) == (2 * d_old, n_old * d_old)
    out = torch.zeros(2 * d_new, n_new * d_new, dtype=jl.dtype)
    for a in range(2 * d_old):
        for i in range(n_old):
            for j in range(d_old):
                out[a, i * d_new + j] = jl[a, i * d_old + j]
    scale = float(jl.float().pow(2).mean().sqrt()) * noise
    out[2 * d_old:, :] = out[: 2 * d_old, :].clone()
    if scale > 0:
        out[2 * d_old:, :] += torch.randn((2 * d_old, n_new * d_new), generator=gen) * scale
    return out


# (2d, nd) -> (4d, 4nd) handled by grow_jl; (X, 2d) frozen maps zero-pad cols.


def grow_tensor(src: torch.Tensor, dst_shape: tuple[int, ...], noise: float,
                gen: torch.Generator, key: str = "") -> tuple[torch.Tensor, str]:
    """Function-preserving width growth (each dim equal or exactly doubled).

    - dim0 (outputs/rows) doubled: tile rows, noise on the new rows.
    - other dims (inputs/cols) doubled: zero-pad, so old outputs are exact.
    - embeddings/field states: tiled (dims must stay alive for gradients).
    - 1-D norms/biases/phases: tiled exact.
    Embeddings are the symmetry breakers (noise); zero-cols learn from 0.
    """
    if tuple(src.shape) == tuple(dst_shape):
        return src.clone(), "copy"
    if src.dim() != len(dst_shape):
        raise ValueError(f"rank change unsupported: {tuple(src.shape)} -> {dst_shape}")
    for s, d in zip(src.shape, dst_shape):
        if d != s and d != 2 * s:
            raise ValueError(f"non-doubling dim: {tuple(src.shape)} -> {dst_shape}")
    is_embed = ("embed" in key) or key.endswith("/alpha") or key.endswith("mode_phase")
    out = torch.zeros(dst_shape, dtype=src.dtype)
    if src.dim() == 1:
        s = src.shape[0]
        out[:s].copy_(src)
        out[s:].copy_(src)
        return out, "tile1d"
    if src.dim() == 2:
        r, c = src.shape
        R, C = dst_shape
        out[:r, :c].copy_(src)
        with torch.no_grad():
            scale = float(src.float().pow(2).mean().sqrt()) * noise
            if R == 2 * r:  # new output rows = copies + noise
                out[r:, :c].copy_(src)
                if scale > 0:
                    out[r:, :c] += torch.randn((r, c), generator=gen) * scale
            if C == 2 * c:
                if is_embed:  # keep dims alive: tile + noise
                    out[:R, c:].copy_(out[:R, :c])
                    if scale > 0:
                        out[:R, c:] += torch.randn((R, c), generator=gen) * scale
                # else: zero-pad (old outputs exact)
        return out, "preserve2d"
    raise ValueError(f"rank {src.dim()} unsupported for {key}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Grow checkpoint width d/n")
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--d-new", type=int, default=32)
    parser.add_argument("--n-new", type=int, default=32)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()

    gen = torch.Generator().manual_seed(args.seed)
    blob = torch.load(args.src, map_location="cpu", weights_only=False)
    config = blob.get("config") or {}
    bounds = config.get("energy_decay_bounds")
    energy_decay_bounds = tuple(bounds) if bounds is not None else DEFAULT_ENERGY_DECAY_BOUNDS
    model = AtomNativeModel(
        d_model=args.d_new,
        n_modes=args.n_new,
        n_atoms_max=int(config.get("n_atoms_max", 64)),
        max_payload_bytes=int(config.get("max_payload_bytes", 1)),
        field_max_rms=float(config.get("field_max_rms", 3.0)),
        energy_decay_bounds=energy_decay_bounds,
        field_obligatory_hard=True,
        field_obligatory_readout=True,
        last_atom_readout=bool(config.get("last_atom_readout", True)),
    )

    def leaves(o, prefix=""):
        if torch.is_tensor(o):
            yield prefix, o
        elif isinstance(o, dict):
            for k, v in o.items():
                yield from leaves(v, f"{prefix}/{k}" if prefix else str(k))

    fresh = dict(leaves(model._checkpoint()))
    src_leaves = dict(leaves(blob))
    grown = dict(leaves(model._checkpoint()))  # placeholder, replaced below
    stats: dict[str, int] = {}
    grown_state: dict[str, torch.Tensor] = {}
    for key, dst in fresh.items():
        if key == "surface/alpha_jl":
            d_old = blob["config"].get("d_model", 16)
            n_old = blob["config"].get("n_modes", 16)
            grown_state[key] = grow_jl(src_leaves[key], n_old, d_old, args.noise, gen)
            stats["jl-interleaved"] = stats.get("jl-interleaved", 0) + 1
            continue
        if key in ("surface/alpha_byte_frozen", "surface/alpha_length_frozen"):
            src = src_leaves[key]
            new = torch.zeros(tuple(dst.shape), dtype=src.dtype)
            new[:, : src.shape[1]].copy_(src)  # zero-pad cols: frozen branch exact
            grown_state[key] = new
            stats["frozen-grown"] = stats.get("frozen-grown", 0) + 1
            continue
        if key not in src_leaves:
            grown_state[key] = dst.clone()
            stats["missing-kept-fresh"] = stats.get("missing-kept-fresh", 0) + 1
            print(f"  WARN kept-fresh (absent in src): {key} {tuple(dst.shape)}")
            continue
        new, action = grow_tensor(src_leaves[key], tuple(dst.shape), args.noise, gen, key)
        grown_state[key] = new
        stats[action] = stats.get(action, 0) + 1

    # Write grown tensors back into the live modules, then save.
    with torch.no_grad():
        model.compiler.load_state_dict(
            {k.split("/", 1)[1]: v for k, v in grown_state.items() if k.startswith("compiler/")})
        model.surface.load_state_dict(
            {k.split("/", 1)[1]: v for k, v in grown_state.items() if k.startswith("surface/")})
        for sub in ("encoder", "dynamics", "interaction", "aggregation",
                    "abstraction", "consolidation", "production", "atoms"):
            part = {k.split("/", 2)[2]: v for k, v in grown_state.items()
                    if k.startswith(f"core/{sub}/")}
            if part:
                mod = getattr(model.core, sub)
                if hasattr(mod, "load_state_dict"):
                    mod.load_state_dict(part, strict=False)
                elif isinstance(mod, dict):
                    mod.update(part)
        # Core state (alpha/t) and atomizer pieces.
        state_part = {k.split("/", 2)[2]: v for k, v in grown_state.items()
                      if k.startswith("core/state/")}
        if state_part and hasattr(model.core.state, "alpha") and "alpha" in state_part:
            model.core.state.alpha.copy_(state_part["alpha"])
        if state_part and hasattr(model.core.state, "t") and "t" in state_part:
            model.core.state.t.copy_(state_part["t"])
        atom_part = {k.split("/", 1)[1]: v for k, v in grown_state.items()
                     if k.startswith("atomizer/")}
        for k, v in atom_part.items():
            if hasattr(model.atomizer, k) and torch.is_tensor(getattr(model.atomizer, k)):
                getattr(model.atomizer, k).copy_(v)
        for probe_name in ("field_probe", "field_next_probe"):
            part = {k.split("/", 1)[1]: v for k, v in grown_state.items()
                    if k.startswith(probe_name + "/")}
            if part and hasattr(model, probe_name):
                getattr(model, probe_name).load_state_dict(part)

    out = Path(args.dst)
    model.save(out)
    print(f"grew {args.src} -> {out} (d={args.d_new}, n={args.n_new})")
    print("actions:", stats)


if __name__ == "__main__":
    main()
