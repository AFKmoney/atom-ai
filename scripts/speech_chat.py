#!/usr/bin/env python3
"""Chat ATOM on the last non-collapsed hard ckpt. No Grok Bot. No extra train.

Preferred checkpoints (first that exists wins):
  1. checkpoints/atom_native_field_next/atom_native.pt     @ 3,458,000  logits 0.838
  2. checkpoints/atom_native_linguistic/ or spans @ 3,425,000           logits 0.866
  3. checkpoints/atom_native_chat_talk/atom_native.pt
NOT the alpha_local / payload ckpt @ 3,508,000 (logits 0.999 collapse).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.atom_native import AtomNativeModel


CANDIDATES = (
    Path("checkpoints/atom_native_field_next/atom_native.pt"),
    Path("checkpoints/atom_native_next_packet/atom_native.pt"),
    Path("checkpoints/atom_native_linguistic_spans/atom_native.pt"),
    Path("checkpoints/atom_native_linguistic/atom_native.pt"),
    Path("checkpoints/atom_native_chat_talk/atom_native.pt"),
    Path("checkpoints/atom_native_obligatory_hard/atom_native.pt"),
)


def resolve_ckpt(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise SystemExit(f"checkpoint missing: {path}")
        return path
    for path in CANDIDATES:
        if path.is_file() and path.stat().st_size > 0:
            return path
    raise SystemExit(
        "no non-collapsed ckpt found. Pass --checkpoint. "
        "Do not use checkpoints/atom_native_alpha_local/"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ATOM speech chat (payload-copy OFF)")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--prompt", default="Bonjour")
    parser.add_argument("--max-packets", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-modes", type=int, default=64)
    parser.add_argument("--max-payload-bytes", type=int, default=32)
    args = parser.parse_args()

    ckpt = resolve_ckpt(args.checkpoint)
    print(f"[speech] loading {ckpt}")
    model = AtomNativeModel(
        d_model=args.d_model,
        n_modes=args.n_modes,
        n_atoms_max=512,
        max_payload_bytes=args.max_payload_bytes,
        field_max_rms=3.0,
        field_obligatory_hard=True,
        enable_merge=True,
    )
    extra = model.load(ckpt) or {}
    step = extra.get("step") or extra.get("global_step")
    print(f"[speech] extra_keys={sorted(extra.keys())[:12]} step={step}")
    model.eval()
    raw = model.generate_packets(
        args.prompt,
        max_packets=args.max_packets,
        deterministic=True,
        prefer_printable=True,
        payload_copy=False,
        dialogue_wrap=True,
        speech_gate=True,
        merge_enabled=False,
    )
    text = raw.decode("utf-8", errors="replace")
    print(f"[speech] prompt={args.prompt!r}")
    print(f"[speech] response={text!r}")
    if not text.strip():
        print("[speech] empty — speech_ok refused soup. Need the 3.458M ckpt, not 3.508M.")


if __name__ == "__main__":
    main()
