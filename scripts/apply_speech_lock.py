#!/usr/bin/env python3
"""Idempotent speech-lock patcher for src/atom_native.py.

    python3 scripts/apply_speech_lock.py
    python3 -m pytest test/test_speech_lock.py -q
    python3 scripts/speech_chat.py --prompt Bonjour
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "src" / "atom_native.py"

IMPORT = "from .speech_lock import format_dialogue_prompt, speech_ok\n"

GENERATE = '''    def generate_packets(
        self,
        prompt: str,
        max_packets: int = 8,
        temperature: float = 1.0,
        top_k: int | None = None,
        deterministic: bool = True,
        max_length: int | None = None,
        prefer_printable: bool = True,
        reset: bool = True,
        merge_enabled: bool = False,
        payload_copy: bool = False,
        dialogue_wrap: bool = True,
        speech_gate: bool = True,
    ) -> bytes:
        """Prime on a prompt and generate bounded atom surface packets."""
        self.eval()
        if self.field_obligatory_hard:
            self.surface.set_obligatory_hard(True)
            prefer_printable = True
        prev_payload_flag = bool(getattr(self.surface, "payload_enabled", True))
        self.surface.payload_enabled = bool(payload_copy)
        try:
            if reset:
                self.reset_state(reset_atomizer=True)
            if dialogue_wrap:
                prompt = format_dialogue_prompt(prompt)
            prompt_packets = self.atomizer.encode(prompt, reset=reset)
            if not prompt_packets:
                raise ValueError("prompt must produce at least one atom packet")
            for packet in prompt_packets[:-1]:
                self.forward_packet(packet, merge_enabled=merge_enabled)
            current = prompt_packets[-1]
            generated = bytearray()
            for _ in range(max_packets):
                payload = self.predict_packet(
                    current,
                    temperature=temperature,
                    top_k=top_k,
                    deterministic=deterministic,
                    prefer_printable=prefer_printable,
                    merge_enabled=merge_enabled,
                )
                if not payload:
                    break
                if speech_gate and not speech_ok(payload):
                    break
                generated.extend(payload)
                current = self.atomizer.packet_from_payload(payload)
                if max_length is not None and len(generated) >= max_length:
                    break
                if generated.endswith(b"\\n\\n") and len(generated) > 2:
                    break
            if max_length is not None:
                return bytes(generated[:max_length])
            return bytes(generated)
        finally:
            self.surface.payload_enabled = prev_payload_flag

'''


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")
    changed = False
    if "from .speech_lock import" not in text:
        needle = "from .io.atomizer import AtomPacket, Atomizer\n"
        if needle not in text:
            raise SystemExit("atomizer import not found")
        text = text.replace(needle, needle + IMPORT, 1)
        changed = True
        print("[apply] import speech_lock")

    old_copy = (
        "        copy = F.softplus(self.payload_copy_scale) * 4.0\n"
        "        # Per-atom copy-bias: winner atom's own hist / positions only.\n"
        "        for i, (payload, hist_i) in enumerate(zip(kept_payloads, local_hists)):\n"
        "            w = copy_w[i]\n"
        "            mass_i = hist_i.sum().clamp_min(1.0)\n"
        "            byte_logits = byte_logits + copy * w * (hist_i / mass_i).unsqueeze(0)\n"
        "            for pos, b in enumerate(payload[:L]):\n"
        "                byte_logits[pos, int(b)] = byte_logits[pos, int(b)] + copy * w\n"
    )
    new_copy = (
        "        copy = F.softplus(self.payload_copy_scale) * 4.0\n"
        "        # Position-aligned copy only. No unigram broadcast across span slots.\n"
        "        for i, payload in enumerate(kept_payloads):\n"
        "            w = copy_w[i]\n"
        "            for pos, b in enumerate(payload[:L]):\n"
        "                byte_logits[pos, int(b)] = byte_logits[pos, int(b)] + copy * w\n"
    )
    if old_copy in text:
        text = text.replace(old_copy, new_copy, 1)
        changed = True
        print("[apply] position-aligned copy")
    elif "No unigram broadcast" in text:
        print("[apply] copy already aligned")

    if "self.payload_enabled = True" not in text:
        old_init = (
            "        self.payload_copy_scale = nn.Parameter(torch.tensor(1.0))\n"
            "        self._init_payload_production()\n"
        )
        new_init = (
            "        self.payload_copy_scale = nn.Parameter(torch.tensor(1.0))\n"
            "        self.payload_enabled = True\n"
            "        self._init_payload_production()\n"
        )
        if old_init in text:
            text = text.replace(old_init, new_init, 1)
            changed = True
            print("[apply] payload_enabled flag")

    old_pay = (
        "                    pay_byte, pay_len = self.payload_produce(\n"
        "                        atom_payloads,\n"
        "                        a_hat,\n"
        "                        a_only,\n"
        "                        atom_energies=atom_energies,\n"
        "                    )\n"
    )
    new_pay = (
        "                    if getattr(self, \"payload_enabled\", True):\n"
        "                        pay_byte, pay_len = self.payload_produce(\n"
        "                            atom_payloads,\n"
        "                            a_hat,\n"
        "                            a_only,\n"
        "                            atom_energies=atom_energies,\n"
        "                        )\n"
        "                    else:\n"
        "                        pay_byte = torch.zeros_like(frozen_byte)\n"
        "                        pay_len = torch.zeros_like(frozen_len)\n"
    )
    if old_pay in text:
        text = text.replace(old_pay, new_pay, 1)
        changed = True
        print("[apply] gated payload_produce")

    start = text.find("    def generate_packets(")
    end = text.find("    def _checkpoint(self)")
    if start < 0 or end < 0:
        raise SystemExit("generate_packets / _checkpoint anchors not found")
    if text[start:end] != GENERATE:
        text = text[:start] + GENERATE + "\n" + text[end:]
        changed = True
        print("[apply] generate_packets rewritten")

    ast.parse(text)
    if changed:
        TARGET.write_text(text, encoding="utf-8")
        print(f"[apply] wrote {TARGET}")
    else:
        print("[apply] already applied")


if __name__ == "__main__":
    main()
