#!/usr/bin/env python3
"""Teacher-forcing diagnostic: argmax accuracy by gold byte class.

Tells whether the model *knows* vowels/spaces in teacher-forcing while
free-run generate cycles on consonants (exposure bias / decode dynamics).

Usage:
  PYTHONPATH=. .venv/bin/python tools/diagnose_teacher.py --checkpoint ... --data ... --transitions 400
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

import torch

from src.io.atomizer import Atomizer
from tools.probe_speech import load


def byte_class(b: int) -> str:
    ch = chr(b) if b < 128 else "?"
    if ch in "aeiouyAEIOUYàâäéèêëïîôùûü":
        return "vowel"
    if ch == " ":
        return "space"
    if ch == "\n":
        return "newline"
    if ch.isalpha():
        return "consonant"
    if chr(b) in ".,;:!?'-":
        return "punct"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--transitions", type=int, default=400)
    parser.add_argument("--max-span-bytes", type=int, default=4)
    args = parser.parse_args()

    model = load(Path(args.checkpoint))
    model.eval()
    raw = Path(args.data).read_bytes()
    atomizer = Atomizer(max_span_bytes=args.max_span_bytes, pack_mode="linguistic")
    packets = atomizer.encode_bytes(raw[:200_000])
    pairs = list(zip(packets[:-1], packets[1:]))[: args.transitions]

    model.reset_state(reset_atomizer=True)
    model.eval()
    correct: collections.Counter = collections.Counter()
    total: collections.Counter = collections.Counter()
    len_ok = 0
    with torch.no_grad():
        for current, target in pairs:
            out = model.forward_packet(current, merge_enabled=False)
            surf = out["surface"]
            gold = target.payload[: model.max_payload_bytes]
            pred_len = int(surf["length_logits"].argmax().item()) + 1
            if pred_len == len(gold):
                len_ok += 1
            for pos, g in enumerate(gold):
                pred = int(surf["byte_logits"][pos].argmax().item())
                cls = byte_class(g)
                total[cls] += 1
                if pred == g:
                    correct[cls] += 1
    n = sum(total.values())
    print(f"[diag] {args.checkpoint} transitions={len(pairs)} bytes={n} len_acc={len_ok/len(pairs):.3f}")
    for cls in ("vowel", "consonant", "space", "newline", "punct", "other"):
        t = total[cls]
        if t:
            print(f"  {cls:10s} acc={correct[cls]/t:.3f} ({correct[cls]}/{t})")
    print(f"  overall byte acc={sum(correct.values())/n:.3f}")
    # Top predicted bytes (argmax histogram) vs gold histogram.
    print("  (argmax histogram omitted; see probe for free-run behavior)")


if __name__ == "__main__":
    main()
