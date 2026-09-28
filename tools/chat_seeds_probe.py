"""Seeded chat probe: same prompts/params as train_until_coherent.sh, fixed torch seeds.

PYTHONPATH=. python tools/chat_seeds_probe.py --checkpoint X.pt [--seeds 1 2 3] [--temps 0.8 0.6 1.0]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from chat_atom_native import generate_reply, load_model  # noqa: E402

PROMPTS = ["Bonjour, comment ça va ?", "Qui es-tu ?", "Il était une fois"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--temps", type=float, nargs="+", default=[0.8])
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--max-packets", type=int, default=96)
    ap.add_argument("--max-length", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    model = load_model(args.checkpoint)
    rows = []
    for temp in args.temps:
        for seed in args.seeds:
            for prompt in PROMPTS:
                torch.manual_seed(seed)
                resp = generate_reply(
                    model, prompt, max_packets=args.max_packets, temperature=temp,
                    top_k=args.top_k, max_length=args.max_length, role_prime=False,
                )
                rows.append({"temp": temp, "seed": seed, "prompt": prompt, "resp": resp})
                print(f"[T={temp} seed={seed}] {prompt!r} -> {resp!r}")
    if args.out:
        Path(args.out).write_text(json.dumps({"checkpoint": args.checkpoint, "rows": rows},
                                             ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
