"""Val CE with and without the last_atom readout term (field-path-alone diagnostic).

Rebuilds the exact stream-ring validation used by run_atom_native at the end of a
chunk (same shards/chunk-bytes, skip=END-base packets, last 256 ring transitions),
then evaluates each checkpoint twice under eval/no_grad:
  normal      : logits as trained / as chat sees them
  la_zeroed   : surface.last_atom_force_off=True (la_byte := 0) — what the field
                path (α-MLP + frozen JL [+ payload]) predicts alone.

Usage:
  PYTHONPATH=. python tools/eval_last_atom_ablation.py --end-steps 4868000 4876000 \
      --checkpoints A.pt B.pt --out docs/artifacts/last_atom_dropout/val_ablation.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chat_atom_native import load_model  # noqa: E402
from src.io.atomizer import Atomizer  # noqa: E402
from src.io.stream_corpus import StreamingPacketSource, resolve_shard_paths  # noqa: E402


def build_val_pairs_multi(end_steps, base: int, glob: str, chunk_bytes: int, n_val: int = 256):
    """One stream pass; snapshot the ring's last ``n_val`` transitions at each end step."""
    src = StreamingPacketSource(
        Atomizer(max_span_bytes=1),
        resolve_shard_paths(None, glob),
        chunk_bytes=chunk_bytes,
        loop=True,
        reset_atomizer=True,
    )
    out = {}
    consumed = 0
    for end in sorted(end_steps):
        for _ in range(max(0, end - base) - consumed):
            next(src)
        consumed = max(consumed, end - base)
        pairs = src.validation_transitions()
        out[end] = pairs[-n_val:] if len(pairs) > n_val else pairs
    return out


@torch.no_grad()
def evaluate(model, pairs, force_off: bool) -> dict:
    model.surface.last_atom_force_off = bool(force_off)
    model.reset_state(reset_atomizer=True)
    model.eval()
    losses, byte_losses = [], []
    for cur, tgt in pairs:
        loss, info = model.transition_loss(cur, tgt)
        losses.append(float(loss.item()))
        byte_losses.append(float(info["byte_loss"]))
    model.surface.last_atom_force_off = False
    return {
        "loss": float(np.mean(losses)),
        "byte_loss": float(np.mean(byte_losses)),
        "n": len(losses),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--end-steps", type=int, nargs="+", required=True)
    ap.add_argument("--base", type=int, default=496000)
    ap.add_argument("--data-glob", default="data/corpus_fr_medium*.txt")
    ap.add_argument("--chunk-bytes", type=int, default=65536)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--ring-cache", default=None,
        help="optional .pt cache of the built rings (keyed by end steps/base/glob/chunk); "
             "loaded if present and matching, else built and saved atomically",
    )
    args = ap.parse_args()
    t0 = time.time()
    key = {"end_steps": sorted(args.end_steps), "base": args.base,
           "data_glob": args.data_glob, "chunk_bytes": args.chunk_bytes}
    rings = None
    if args.ring_cache and Path(args.ring_cache).exists():
        blob = torch.load(args.ring_cache, map_location="cpu", weights_only=False)
        if blob.get("key") == key:
            rings = blob["rings"]
            print(f"val rings loaded from cache {args.ring_cache}", flush=True)
    if rings is None:
        rings = build_val_pairs_multi(args.end_steps, args.base, args.data_glob, args.chunk_bytes)
        print(f"val rings built for {sorted(rings)} in {time.time()-t0:.1f}s", flush=True)
        if args.ring_cache:
            tmp = Path(str(args.ring_cache) + f".tmp{time.time_ns()}")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"key": key, "rings": rings}, tmp)
            tmp.replace(args.ring_cache)
    results = {"val_source": "stream_ring_last256 (same as run_atom_native stream val)", "rings": {}}
    models = {ck: load_model(ck) for ck in args.checkpoints}
    for end, pairs in sorted(rings.items()):
        results["rings"][str(end)] = {"n_pairs": len(pairs), "checkpoints": {}}
        for ck, model in models.items():
            normal = evaluate(model, pairs, force_off=False)
            zeroed = evaluate(model, pairs, force_off=True)
            results["rings"][str(end)]["checkpoints"][ck] = {"normal": normal, "la_zeroed": zeroed}
            print(
                f"[ring@{end}] {ck}\n  normal   loss={normal['loss']:.4f} byte_ce={normal['byte_loss']:.4f}\n"
                f"  la_zero  loss={zeroed['loss']:.4f} byte_ce={zeroed['byte_loss']:.4f}",
                flush=True,
            )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
