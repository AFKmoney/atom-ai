"""Probe a checkpoint in both field regimes: cold (prompt only) and primed."""

from __future__ import annotations

import argparse

import torch

from src.atom_native import format_dialogue_prompt

from tools.chat_atom_native import load_model


def generate_raw(model, prompt_packets, max_packets=24, max_length=60):
    for packet in prompt_packets[:-1]:
        model.forward_packet(packet, merge_enabled=False)
    current = prompt_packets[-1]
    generated = bytearray()
    for _ in range(max_packets):
        payload = model.predict_packet(current, deterministic=True, prefer_printable=True, merge_enabled=False)
        if not payload:
            break
        generated.extend(payload)
        current = model.atomizer.packet_from_payload(payload)
        if len(generated) >= max_length or generated.endswith(b"\n\n"):
            break
    return bytes(generated)


def probe(checkpoint, prompt, primer_packets, primer_file, max_packets, max_length):
    primed_prompt = format_dialogue_prompt(prompt)
    primer = open(primer_file, "rb").read(4000).decode("utf-8", errors="ignore")
    for use_primer, tag in ((False, "cold"), (True, "PRIMED")):
        model = load_model(checkpoint)
        model.eval()
        model.reset_state(reset_atomizer=True)
        with torch.no_grad():
            if use_primer:
                packets = model.atomizer.encode(primer, reset=True)[:primer_packets]
                for i, packet in enumerate(packets):
                    model.forward_packet(packet, merge_enabled=False)
                    if (i + 1) % 64 == 0:
                        model.core.atoms = type(model.core.atoms)()
                        model.core.energy_history = []
            packets = model.atomizer.encode(primed_prompt, reset=False)
            out = generate_raw(model, packets, max_packets, max_length)
            rms = float(model.core.state.alpha.pow(2).mean().sqrt())
        print(f"{tag:6s} (field_rms={rms:.2f}): {out!r}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", default="Utilisateur: Tu penses quoi ?\nAssistant:")
    p.add_argument("--primer-packets", type=int, default=1500)
    p.add_argument("--primer-file", default="data/corpus_fr_medium.txt")
    p.add_argument("--max-packets", type=int, default=24)
    p.add_argument("--max-length", type=int, default=60)
    a = p.parse_args()
    probe(a.checkpoint, a.prompt, a.primer_packets, a.primer_file, a.max_packets, a.max_length)
