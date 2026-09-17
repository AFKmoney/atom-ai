"""Light tests for advisory train lock (skip if flock unavailable)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Import helpers from the runner module without executing main.
import importlib.util


def _load_runner():
    path = Path(__file__).resolve().parents[1] / "tools" / "run_atom_native.py"
    spec = importlib.util.spec_from_file_location("run_atom_native_under_test", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Avoid running side effects; module only defines helpers + main.
    spec.loader.exec_module(mod)
    return mod


class TrainLockTests(unittest.TestCase):
    def test_second_acquire_fails_fast(self) -> None:
        try:
            import fcntl  # noqa: F401
        except ImportError:
            self.skipTest("fcntl unavailable")
        mod = _load_runner()
        with tempfile.TemporaryDirectory() as tmp:
            lock_path = Path(tmp) / "atom-ai-train.lock"
            first = mod.acquire_train_lock(lock_path)
            def _close():
                try:
                    if not first.closed:
                        first.close()
                except Exception:
                    pass
            self.addCleanup(_close)
            with self.assertRaises(SystemExit) as ctx:
                mod.acquire_train_lock(lock_path)
            msg = str(ctx.exception)
            self.assertIn("advisory lock", msg)
            self.assertIn(str(lock_path.resolve()), msg)
            self.assertIn(f"pid={os.getpid()}", lock_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
