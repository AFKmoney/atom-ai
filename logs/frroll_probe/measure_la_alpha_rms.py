#!/usr/bin/env python3
"""Measure last_atom vs α-MLP (obl*train_byte) logit RMS — read-only."""
from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path

import torch

from src.atom_native import _tail_bytes
from tools.probe_speech import load

PROMPTS = ["Bonjour", "Qui es-tu ?", "Merci", "Comment vas-tu ?"]


def living_payloads(model) -> list[bytes]:
    out: list[bytes] = []
    col = getattr(model.core, "atoms", None)
    atoms = list(getattr(col, "atoms", []) or []) if col is not None else []
    for atom in atoms:
        p = getattr(atom, "payload", None)
        if p is None:
            continue
        if isinstance(p, (bytes, bytearray)):
            out.append(bytes(p))
        else:
            out.append(bytes(int(x) % 256 for x in p))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()
    model = load(Path(args.checkpoint))
    surf = model.surface
    assert surf.last_atom is not None, "ckpt lacks last_atom"
    ratios: list[float] = []
    print(f"[la_alpha_rms] {args.checkpoint}")
    for prompt in PROMPTS:
        with torch.no_grad():
            model.generate_packets(
                prompt,
                max_packets=8,
                max_length=64,
                deterministic=True,
                payload_copy=False,
                dialogue_wrap=True,
                speech_gate=False,
                merge_enabled=False,
            )
            alpha = model.core.state.alpha
            a_only = surf.alpha_only_features(alpha)
            a_rms = a_only.pow(2).mean().sqrt().clamp_min(1e-8)
            a_hat = a_only / a_rms
            train_byte = surf.alpha_byte_proj(a_hat).view(surf.max_payload_bytes, 256)
            alpha_term = surf.obligatory_scale() * train_byte
            payloads = living_payloads(model)
            last_b, prev_b = _tail_bytes(payloads)
            phi = getattr(model.core.state, "phi", None)
            if phi is None:
                phi = torch.zeros(surf.d_model, device=alpha.device, dtype=alpha.dtype)
            la_byte = surf.last_atom(last_b, prev_b, phi).unsqueeze(0)
            a_term_rms = float(alpha_term.pow(2).mean().sqrt())
            la_rms = float(la_byte.pow(2).mean().sqrt())
            ratio = la_rms / max(a_term_rms, 1e-12)
            ratios.append(ratio)
            print(
                f"  {prompt!r}: alpha_rms={a_term_rms:.4f} "
                f"la_rms={la_rms:.4f} la/alpha={ratio:.4f}"
            )
    if ratios:
        print(
            f"[la_alpha_rms] mean_ratio={st.mean(ratios):.4f} "
            f"min={min(ratios):.4f} max={max(ratios):.4f}"
        )


if __name__ == "__main__":
    main()
