#!/usr/bin/env python3
"""Honest speech probe for an ATOM checkpoint (ligne A).

Loads a checkpoint with its own config, generates with:
  payload_copy=False, dialogue_wrap=True, speech_gate=False, merge off.

Pass rule (docs/MEASURE_LOG.md): two prompts emit >=4 Latin letters
AND the strings differ.

Usage:
  PYTHONPATH=. .venv/bin/python tools/probe_speech.py --checkpoint checkpoints/byte_tick/atom_native_step_8000_medium.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.atom_native import DEFAULT_ENERGY_DECAY_BOUNDS, AtomNativeModel

PROMPTS = ["Bonjour", "Qui es-tu ?", "Merci", "Comment vas-tu ?"]


def load(checkpoint: Path) -> AtomNativeModel:
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = blob.get("config") or {}
    bounds = cfg.get("energy_decay_bounds")
    model = AtomNativeModel(
        d_model=int(cfg.get("d_model", 16)),
        n_modes=int(cfg.get("n_modes", 16)),
        n_atoms_max=int(cfg.get("n_atoms_max", 64)),
        max_payload_bytes=int(cfg.get("max_payload_bytes", 4)),
        field_max_rms=float(cfg.get("field_max_rms", 3.0)),
        energy_decay_bounds=tuple(bounds) if bounds else DEFAULT_ENERGY_DECAY_BOUNDS,
        field_obligatory_hard=bool(cfg.get("field_obligatory_hard", False)),
        field_obligatory_readout=bool(cfg.get("field_obligatory_readout", False)),
        last_atom_readout=bool(cfg.get("last_atom_readout", False)),
        enable_merge=False,
    )
    model.load(checkpoint)
    if "last_atom_scale" in cfg:
        model.surface.last_atom_scale = float(cfg["last_atom_scale"])
    model.eval()
    return model


def latin_letters(text: str) -> int:
    return sum(1 for ch in text if ch.isalpha() and ch.isascii())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-packets", type=int, default=24)
    parser.add_argument("--max-length", type=int, default=120)
    args = parser.parse_args()

    model = load(Path(args.checkpoint))
    print(f"[probe] {args.checkpoint} hard={model.field_obligatory_hard}")
    results: list[str] = []
    for prompt in PROMPTS:
        raw = model.generate_packets(
            prompt,
            max_packets=args.max_packets,
            max_length=args.max_length,
            deterministic=True,
            payload_copy=False,
            dialogue_wrap=True,
            speech_gate=False,
            merge_enabled=False,
        )
        text = raw.decode("utf-8", errors="replace")
        results.append(text)
        print(f"  {prompt!r} -> {text!r} (latin={latin_letters(text)})")
    rich = [t for t in results if latin_letters(t) >= 4]
    distinct = len(set(results)) > 1
    passed = len(rich) >= 2 and distinct
    print(f"[probe] PASS={passed} ({len(rich)}/{len(results)} with >=4 latin, distinct={distinct})")


if __name__ == "__main__":
    main()
