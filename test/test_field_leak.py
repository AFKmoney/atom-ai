"""Unit tests: --field-leak λ (opt-in per-tick field leak; default 0 = legacy).

* default 0.0 is bit-identical to the pre-lever code (commit 90f7462) on a
  multi-tick train+eval run (field α, logits, loss, grads, params after AdamW);
* λ>0 is exactly "α_prev ← (1-λ)·α_prev, then the unchanged tick";
* λ>0 keeps α off the RMS cap where λ=0 saturates it;
* checkpoint roundtrip restores λ; old configs (no key) load as 0.0;
* trainer CLI: default 0.0, and --field-leak wins after --resume.
"""
from __future__ import annotations

import importlib
import importlib.util
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from src.atom_native import AtomNativeModel
from src.io.atomizer import Atomizer

ROOT = Path(__file__).resolve().parents[1]
LEGACY_COMMIT = "90f7462"
TEXT = "Bonjour, comment ça va ? Il était une fois un petit atome. Qui es-tu ? " * 3


def _build(cls=AtomNativeModel, **kw):
    torch.manual_seed(0)
    return cls(
        d_model=8, n_modes=8, n_atoms_max=32, max_payload_bytes=1,
        atomizer=Atomizer(max_span_bytes=1), field_max_rms=3.0,
        field_obligatory_hard=True, enable_merge=False, last_atom_readout=True,
        field_loss_weight=0.05, **kw,
    )


def _legacy_cls():
    """AtomNativeModel from the pre-lever commit, imported as a sibling package."""
    try:
        code = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{LEGACY_COMMIT}:src/atom_native.py"],
            capture_output=True, text=True, check=True,
        ).stdout
    except Exception:  # pragma: no cover - no git / shallow clone
        return None
    tmp = Path(tempfile.mkdtemp(prefix="legacy_src_"))
    pkg = tmp / "legacy_src_fieldleak"
    shutil.copytree(ROOT / "src", pkg, ignore=shutil.ignore_patterns("__pycache__"))
    (pkg / "atom_native.py").write_text(code, encoding="utf-8")
    sys.path.insert(0, str(tmp))
    try:
        mod = importlib.import_module("legacy_src_fieldleak.atom_native")
    finally:
        sys.path.remove(str(tmp))
    return mod.AtomNativeModel


def _run(model, n_train=40, n_eval=20):
    """Train n_train transitions (backward + AdamW), then eval n_eval. Returns trace."""
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    packets = model.atomizer.encode(TEXT, reset=True)
    model.reset_state(reset_atomizer=False)
    model.train()
    trace = []
    torch.manual_seed(5)
    for cur, tgt in list(zip(packets[:-1], packets[1:]))[:n_train]:
        opt.zero_grad(set_to_none=True)
        loss, _ = model.transition_loss(cur, tgt)
        loss.backward()
        opt.step()
        model.stabilize_dynamics_parameters()
        trace.append(loss.detach().clone())
        trace.append(model.core.state.alpha.detach().clone())
    model.eval()
    with torch.no_grad():
        for cur, tgt in list(zip(packets[:-1], packets[1:]))[n_train:n_train + n_eval]:
            loss, _ = model.transition_loss(cur, tgt)
            trace.append(loss.detach().clone())
            trace.append(model.core.state.alpha.detach().clone())
    trace.extend(p.detach().clone() for p in model.parameters())
    return trace


class FieldLeakTests(unittest.TestCase):
    def test_default_is_zero(self) -> None:
        self.assertEqual(_build().field_leak, 0.0)

    def test_invalid_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _build(field_leak=1.0)
        with self.assertRaises(ValueError):
            _build(field_leak=-0.1)

    def test_default_bit_identical_to_legacy_commit(self) -> None:
        legacy = _legacy_cls()
        if legacy is None:
            self.skipTest("legacy commit not available")
        new_trace = _run(_build())
        old_trace = _run(_build(cls=legacy))
        self.assertEqual(len(new_trace), len(old_trace))
        for a, b in zip(new_trace, old_trace):
            self.assertTrue(torch.equal(a, b))

    def test_explicit_zero_equals_default(self) -> None:
        a = _run(_build())
        b = _run(_build(field_leak=0.0))
        for x, y in zip(a, b):
            self.assertTrue(torch.equal(x, y))

    def test_leak_is_pre_injection_scaling(self) -> None:
        lam = 0.2
        m_leak, m_ref = _build(field_leak=lam), _build()  # same seed -> same weights
        m_leak.eval(); m_ref.eval()
        g = torch.Generator().manual_seed(3)
        prev = torch.randn(8, 8, generator=g) * 0.5
        pkt = m_ref.atomizer.encode("a", reset=True)[0]
        with torch.no_grad():
            m_leak.core.state.alpha.copy_(prev)
            m_ref.core.state.alpha.copy_((1.0 - lam) * prev)
            out_l = m_leak.forward_packet(pkt)
            out_r = m_ref.forward_packet(pkt)
        self.assertTrue(torch.allclose(out_l["field"], out_r["field"], atol=1e-6))
        self.assertTrue(torch.allclose(out_l["surface"]["byte_logits"],
                                       out_r["surface"]["byte_logits"], atol=1e-5))

    def test_leak_keeps_field_off_cap(self) -> None:
        def frac_at_cap(model):
            model.eval()
            model.reset_state()
            # Unstable-mode regime of the trained ckpts: energy_decay at its floor.
            with torch.no_grad():
                model.core.dynamics.dynamics.energy_decay.fill_(0.3)
            model.field_controller.max_rms = 0.05
            model.field_max_rms = 0.05
            hits = 0
            pk = model.atomizer.encode(TEXT * 4, reset=True)
            with torch.no_grad():
                for p in pk:
                    hits += model.forward_packet(p)["field_scale"] < 0.999999
            return hits / len(pk)
        f0 = frac_at_cap(_build())
        f1 = frac_at_cap(_build(field_leak=0.2))
        self.assertGreater(f0, 0.3)
        self.assertLess(f1, f0 * 0.5)

    def test_checkpoint_roundtrip_and_old_config(self) -> None:
        model = _build(field_leak=0.05)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.pt"
            model.save(path)
            other = _build()
            other.load(path)
            self.assertAlmostEqual(other.field_leak, 0.05)
            blob = torch.load(path, map_location="cpu", weights_only=False)
            del blob["config"]["field_leak"]
            torch.save(blob, path)
            old = _build()
            old.load(path)
            self.assertEqual(old.field_leak, 0.0)


class FieldLeakCliTests(unittest.TestCase):
    def _train(self, tmp: Path, name: str, *extra: str) -> dict:
        data = tmp / "corpus.txt"
        if not data.exists():
            data.write_text(TEXT * 20, encoding="utf-8")
        cmd = [
            sys.executable, str(ROOT / "tools" / "run_atom_native.py"), "--data", str(data),
            "--output-dir", str(tmp), "--checkpoint-name", name, "--steps", "3",
            "--d-model", "8", "--n-modes", "8", "--n-atoms-max", "32", "--max-span-bytes", "1",
            "--field-obligatory-hard", "--no-enable-merge", "--no-payload-copy",
            "--last-atom-readout", "--allow-parallel-train", "--seed", "1", *extra,
        ]
        env = {"PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "PATH": "/usr/bin:/bin"}
        subprocess.run(cmd, cwd=ROOT, env=env, check=True, capture_output=True, text=True)
        return torch.load(tmp / name, map_location="cpu", weights_only=False)["config"]

    def test_cli_default_and_wins_after_resume(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            self.assertEqual(self._train(tmp, "a.pt")["field_leak"], 0.0)
            self.assertAlmostEqual(self._train(tmp, "b.pt", "--field-leak", "0.05")["field_leak"], 0.05)
            # resume a λ=0.05 ckpt without the flag -> CLI default 0 wins
            cfg = self._train(tmp, "c.pt", "--resume", str(tmp / "b.pt"))
            self.assertEqual(cfg["field_leak"], 0.0)
            # resume a λ=0 ckpt with the flag -> CLI value wins
            cfg = self._train(tmp, "d.pt", "--resume", str(tmp / "a.pt"), "--field-leak", "0.1")
            self.assertAlmostEqual(cfg["field_leak"], 0.1)


if __name__ == "__main__":
    unittest.main()
