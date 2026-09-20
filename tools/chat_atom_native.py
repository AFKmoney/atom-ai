"""Chat with an atom-native checkpoint (Atomizer encode → generate → UTF-8).

Usage:
    PYTHONPATH=. python tools/chat_atom_native.py \
      --checkpoint checkpoints/atom_native_chat/atom_native.pt \
      --prompt "Bonjour"

    PYTHONPATH=. python tools/chat_atom_native.py \
      --checkpoint checkpoints/atom_native_chat/atom_native.pt \
      --interactive
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from src.atom_native import DEFAULT_ENERGY_DECAY_BOUNDS, AtomNativeModel


def load_model(checkpoint: str | Path, device: str = "cpu") -> AtomNativeModel:
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    config = blob.get("config") or {}
    bounds = config.get("energy_decay_bounds")
    energy_decay_bounds = tuple(bounds) if bounds is not None else DEFAULT_ENERGY_DECAY_BOUNDS
    # Prefer checkpoint field_max_rms; fall back to current chat default (3.0).
    field_max_rms = config.get("field_max_rms")
    if field_max_rms is None:
        field_max_rms = 3.0
    hard = bool(config.get("field_obligatory_hard", False))
    model = AtomNativeModel(
        d_model=int(config.get("d_model", 64)),
        n_modes=int(config.get("n_modes", 64)),
        n_atoms_max=int(config.get("n_atoms_max", 512)),
        max_payload_bytes=int(config.get("max_payload_bytes", 16)),
        field_max_rms=field_max_rms,
        energy_decay_bounds=energy_decay_bounds,
        field_obligatory_hard=hard,
        field_obligatory_readout=bool(config.get("field_obligatory_readout", False)) or hard,
        last_atom_readout=bool(config.get("last_atom_readout", False)),
    )
    training = model.load(path)
    if hard or bool(config.get("field_obligatory_hard", False)):
        model.field_obligatory_hard = True
        model.field_obligatory_readout = True
        model.surface.set_obligatory_hard(True)
        # Stay on hard α-MLP; prefer_printable is forced in generate_packets.
        assert model.surface.field_obligatory_hard, "chat must stay on hard α-MLP path"
    dyn = (training or {}).get("dynamics_on_load") or {}
    if dyn.get("energy_decay_repaired"):
        print(
            f"[load] repaired energy_decay "
            f"{dyn.get('energy_decay_before')} -> {dyn.get('energy_decay')} "
            f"bounds={dyn.get('energy_decay_bounds')}"
        )
    model.to(device)
    model.eval()
    return model


def generate_reply(
    model: AtomNativeModel,
    prompt: str,
    max_packets: int = 24,
    temperature: float = 0.7,
    top_k: int = 8,
    max_length: int | None = 256,
    deterministic: bool = False,
    role_prime: bool = True,
    merge_enabled: bool = False,
) -> str:
    # Dialogue-style priming: if the user did not already include a role tag,
    # wrap as Utilisateur/Assistant so the field sees the training pattern.
    # MERGE stays off on chat ingest (train still uses enable_merge thr=0.45).
    primed = prompt
    if role_prime and "Assistant:" not in prompt and "Utilisateur:" not in prompt:
        primed = f"Utilisateur: {prompt}\nAssistant:"
    raw = model.generate_packets(
        primed,
        max_packets=max_packets,
        temperature=temperature,
        top_k=top_k,
        deterministic=deterministic,
        max_length=max_length,
        prefer_printable=True,
        reset=True,
        merge_enabled=merge_enabled,
    )
    text = raw.decode("utf-8", errors="replace")
    # Drop leading whitespace-only noise common before content stabilizes.
    return text.lstrip("\n\r ")


def interactive_loop(model: AtomNativeModel, args: argparse.Namespace) -> None:
    print("ATOM-native chat (Atomizer + toroidal field). Commands: quit | status | save")
    while True:
        try:
            user = input("\nVous: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        lower = user.lower()
        if lower in {"quit", "exit", "q"}:
            break
        if lower == "status":
            print(
                f"atoms={len(model.core.atoms)} "
                f"d_model={model.core.encoder.d_model} "
                f"n_modes={model.core.state.n_modes} "
                f"max_payload={model.max_payload_bytes}"
            )
            continue
        if lower == "save":
            out = Path(args.checkpoint).with_name("interactive_atom_native.pt")
            model.save(out)
            print(f"saved {out}")
            continue
        reply = generate_reply(
            model,
            user,
            max_packets=args.max_packets,
            temperature=args.temperature,
            top_k=args.top_k,
            max_length=args.max_length,
            deterministic=args.deterministic,
            role_prime=not args.no_role_prime,
            merge_enabled=bool(args.merge_ingest),
        )
        print(f"ATOM: {reply}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with atom-native checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="Bonjour")
    parser.add_argument("--max-packets", type=int, default=24)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-role-prime", action="store_true",
                        help="do not wrap prompt as Utilisateur/Assistant")
    parser.add_argument(
        "--merge-ingest",
        action="store_true",
        help="enable MERGE during chat priming (default: off; train MERGE unchanged)",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    model = load_model(args.checkpoint, device=args.device)
    if args.interactive:
        interactive_loop(model, args)
        return

    reply = generate_reply(
        model,
        args.prompt,
        max_packets=args.max_packets,
        temperature=args.temperature,
        top_k=args.top_k,
        max_length=args.max_length,
        deterministic=args.deterministic,
        role_prime=not args.no_role_prime,
        merge_enabled=bool(args.merge_ingest),
    )
    print(f"Prompt: {args.prompt}")
    print(f"Response: {reply}")
    print(
        f"[ingest] merge_enabled={bool(args.merge_ingest)} "
        f"atoms={len(model.core.atoms)} train_enable_merge={model.enable_merge}"
    )


if __name__ == "__main__":
    main()
