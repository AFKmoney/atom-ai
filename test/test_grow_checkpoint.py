"""Grow checkpoint d/n (tools/grow_checkpoint.py): shapes, load, forward smoke."""

import os
import subprocess
import sys

import torch

from src.atom_native import AtomNativeModel


def _tiny(d, n, tmp_path, name):
    model = AtomNativeModel(
        d_model=d,
        n_modes=n,
        n_atoms_max=16,
        max_payload_bytes=1,
        field_max_rms=3.0,
        field_obligatory_hard=True,
        field_obligatory_readout=True,
        last_atom_readout=True,
    )
    path = tmp_path / name
    model.save(path)
    return path


def test_grow_doubles_width_and_loads_strict(tmp_path):
    src = _tiny(8, 8, tmp_path, "src.pt")
    dst = tmp_path / "grown.pt"
    proc = subprocess.run(
        [sys.executable, "tools/grow_checkpoint.py", "--src", str(src),
         "--dst", str(dst), "--d-new", "16", "--n-new", "16"],
        capture_output=True, text=True, cwd=".",
        env={**os.environ, "PYTHONPATH": "."},
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    blob = torch.load(dst, map_location="cpu", weights_only=False)
    assert blob["config"]["d_model"] == 16
    assert blob["config"]["n_modes"] == 16
    surf = blob["surface"]
    assert tuple(surf["state_norm.weight"].shape) == (16,)
    assert tuple(surf["alpha_byte_proj.fc1.weight"].shape) == (64, 32)
    assert tuple(surf["alpha_jl"].shape) == (32, 256)
    assert tuple(surf["alpha_byte_frozen"].shape) == (256, 32)
    assert tuple(surf["byte_decoder.weight"].shape) == (256, 16)
    # Frozen branch zero-padded: old block exact.
    src_blob = torch.load(src, map_location="cpu", weights_only=False)
    old_frozen = src_blob["surface"]["alpha_byte_frozen"]
    assert torch.equal(surf["alpha_byte_frozen"][:, :16], old_frozen)
    assert surf["alpha_byte_frozen"][:, 16:].abs().max().item() == 0.0
    # Strict-ish load into a fresh d16 model: no missing keys.
    model = AtomNativeModel(
        d_model=16, n_modes=16, n_atoms_max=16, max_payload_bytes=1,
        field_max_rms=3.0, field_obligatory_hard=True,
        field_obligatory_readout=True, last_atom_readout=True,
    )
    model.load(dst)
    missing_surface = set(model.surface.state_dict()) - set(surf)
    assert not missing_surface


def test_grown_model_forward_smoke(tmp_path):
    src = _tiny(8, 8, tmp_path, "src.pt")
    dst = tmp_path / "grown.pt"
    proc = subprocess.run(
        [sys.executable, "tools/grow_checkpoint.py", "--src", str(src),
         "--dst", str(dst), "--d-new", "16", "--n-new", "16"],
        capture_output=True, text=True, cwd=".",
        env={**os.environ, "PYTHONPATH": "."},
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    model = AtomNativeModel(
        d_model=16, n_modes=16, n_atoms_max=16, max_payload_bytes=1,
        field_max_rms=3.0, field_obligatory_hard=True,
        field_obligatory_readout=True, last_atom_readout=True,
    )
    model.load(dst)
    model.eval()
    packets = model.atomizer.encode("Bonjour", reset=True)
    assert packets
    with torch.no_grad():
        out = model.forward_packet(packets[0], merge_enabled=False)["surface"]
    assert tuple(out["byte_logits"].shape) == (1, 256)
    assert torch.isfinite(out["byte_logits"]).all()
