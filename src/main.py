"""Command-line entry point — atom-native train / chat / interactive.

Default modes use Atomizer + AtomNativeModel (no GPT-2 / Hugging Face tokenizer).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(script: str, extra: list[str]) -> int:
    cmd = [sys.executable, str(REPO_ROOT / script), *extra]
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        (":" + env["PYTHONPATH"]) if env.get("PYTHONPATH") else ""
    )
    return subprocess.call(cmd, cwd=str(REPO_ROOT), env=env)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ATOM / atom-ai (atom-native toroidal field model)"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "chat", "interactive"],
        default="train",
        help="train via tools/run_atom_native.py; chat/interactive via tools/chat_atom_native.py",
    )
    parser.add_argument("--data", default="data/corpus_mixed_fr_en.txt")
    parser.add_argument("--output-dir", default="checkpoints/atom_native_chat")
    parser.add_argument("--checkpoint", default="checkpoints/atom_native_chat/atom_native.pt")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-modes", type=int, default=64)
    parser.add_argument("--n-atoms-max", type=int, default=512)
    parser.add_argument("--max-span-bytes", type=int, default=16)
    parser.add_argument("--episode-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--prompt", default="Bonjour, comment ça va")
    parser.add_argument("--max-packets", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--field-max-rms", type=float, default=4.0)
    args, unknown = parser.parse_known_args()

    if args.mode == "train":
        extra = [
            "--data", args.data,
            "--output-dir", args.output_dir,
            "--steps", str(args.steps),
            "--d-model", str(args.d_model),
            "--n-modes", str(args.n_modes),
            "--n-atoms-max", str(args.n_atoms_max),
            "--max-span-bytes", str(args.max_span_bytes),
            "--episode-length", str(args.episode_length),
            "--learning-rate", str(args.learning_rate),
            "--seed", str(args.seed),
            "--field-max-rms", str(args.field_max_rms),
            *unknown,
        ]
        raise SystemExit(_run("tools/run_atom_native.py", extra))

    extra = [
        "--checkpoint", args.checkpoint,
        "--prompt", args.prompt,
        "--max-packets", str(args.max_packets),
        "--temperature", str(args.temperature),
        "--top-k", str(args.top_k),
        *unknown,
    ]
    if args.mode == "interactive":
        extra.append("--interactive")
    raise SystemExit(_run("tools/chat_atom_native.py", extra))


if __name__ == "__main__":
    main()
