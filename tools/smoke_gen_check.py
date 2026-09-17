#!/usr/bin/env python3
"""Quick generation smoke check for a checkpoint (printable UTF-8, length, diversity)."""
from __future__ import annotations

import argparse
import string
import sys
from pathlib import Path

import torch

from src.atom_native import AtomNativeModel


def load(checkpoint: str) -> AtomNativeModel:
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = blob.get("config") or {}
    model = AtomNativeModel(
        d_model=int(cfg.get("d_model", 64)),
        n_modes=int(cfg.get("n_modes", 64)),
        n_atoms_max=int(cfg.get("n_atoms_max", 512)),
        max_payload_bytes=int(cfg.get("max_payload_bytes", 16)),
        field_max_rms=cfg.get("field_max_rms"),
    )
    training = model.load(checkpoint)
    model.eval()
    return model, training


def printable_frac(text: str) -> float:
    allowed = set(string.printable) | set("àâäéèêëïîôùûüçÀÂÄÉÈÊËÏÎÔÙÛÜÇ«»—–’°")
    return sum(ch in allowed for ch in text) / max(len(text), 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--min-chars", type=int, default=16)
    parser.add_argument("--min-printable", type=float, default=0.8)
    args = parser.parse_args()

    model, training = load(args.checkpoint)
    print(f"loaded {args.checkpoint} migrated={training.get('legacy_surface_migrated')}")
    prompts = [
        "Bonjour",
        "Qui es-tu ?",
        "Utilisateur: Bonjour\nAssistant:",
        "The future of AI is",
    ]
    texts = []
    ok = True
    for prompt in prompts:
        torch.manual_seed(0)
        raw = model.generate_packets(
            prompt if "Assistant:" in prompt else f"Utilisateur: {prompt}\nAssistant:",
            max_packets=16,
            max_length=128,
            temperature=0.6,
            top_k=6,
            deterministic=False,
            prefer_printable=True,
        )
        text = raw.decode("utf-8", errors="replace")
        texts.append(text)
        frac = printable_frac(text)
        status = "OK" if len(text) >= args.min_chars and frac >= args.min_printable else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"--- {status} prompt={prompt!r} chars={len(text)} printable={frac:.2f}")
        print(repr(text[:200]))
    # Diversity: not all identical
    if len(set(texts)) < 2:
        print("FAIL: all prompts produced identical text")
        ok = False
    else:
        print("OK: prompts produced diverse text")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
