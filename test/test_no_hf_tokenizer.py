"""Fail if the live atom-native path imports Hugging Face / GPT-2 tokenizers."""

from __future__ import annotations

import ast
import importlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIVE_MODULES = (
    "src.atom_native",
    "tools.run_atom_native",
    "tools.chat_atom_native",
)
LIVE_PATHS = (
    ROOT / "src" / "atom_native.py",
    ROOT / "tools" / "run_atom_native.py",
    ROOT / "tools" / "chat_atom_native.py",
    ROOT / "src" / "io" / "atomizer.py",
    ROOT / "src" / "io" / "stream_corpus.py",
)
FORBIDDEN_MODULES = {"transformers", "tokenizers"}
FORBIDDEN_NAMES = {
    "AutoTokenizer",
    "GPT2Tokenizer",
    "GPT2TokenizerFast",
    "ToroidalTokenizer",
}


def _ast_forbidden(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in FORBIDDEN_MODULES or "gpt2" in alias.name.lower():
                    hits.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".", 1)[0] if module else ""
            if root in FORBIDDEN_MODULES or "gpt2" in module.lower():
                hits.append(f"from {module} import ...")
            for alias in node.names:
                if alias.name in FORBIDDEN_NAMES:
                    hits.append(f"from {module} import {alias.name}")
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "transformers":
                hits.append(f"transformers.{node.attr}")
    return hits


class NoHfTokenizerOnLivePathTests(unittest.TestCase):
    def test_live_sources_have_no_hf_tokenizer_imports(self) -> None:
        all_hits: dict[str, list[str]] = {}
        for path in LIVE_PATHS:
            self.assertTrue(path.is_file(), f"missing live path {path}")
            hits = _ast_forbidden(path)
            if hits:
                all_hits[str(path.relative_to(ROOT))] = hits
        self.assertEqual(all_hits, {}, f"HF/GPT-2 tokenizer imports on live path: {all_hits}")

    def test_importing_live_modules_does_not_load_transformers(self) -> None:
        # Drop any prior accidental load so the check is meaningful.
        for name in list(sys.modules):
            if name == "transformers" or name.startswith("transformers."):
                del sys.modules[name]
        for mod in LIVE_MODULES:
            importlib.import_module(mod)
        loaded = [name for name in sys.modules if name == "transformers" or name.startswith("transformers.")]
        self.assertEqual(loaded, [], f"live import pulled transformers: {loaded}")

    def test_no_live_tokenizer_py_hf_module(self) -> None:
        """src/io/tokenizer.py must not exist on the live path (HF/GPT-2)."""
        live = ROOT / "src" / "io" / "tokenizer.py"
        self.assertFalse(
            live.exists(),
            "src/io/tokenizer.py must stay quarantined under legacy/; not on live path",
        )


if __name__ == "__main__":
    unittest.main()
