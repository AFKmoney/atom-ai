#!/usr/bin/env python3
"""Idempotent speech-lock patcher for src/atom_native.py.

    python3 scripts/apply_speech_lock.py
    python3 scripts/speech_chat.py --prompt Bonjour
"""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "src" / "atom_native.py"

IMPORT = "from .speech_lock import format_dialogue_prompt, speech_ok\n"


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
    else:
        print("[apply] WARN copy block not found")

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

    if "payload_copy: bool = False" not in text:
        old_sig = (
            "        reset: bool = True,\n"
            "        merge_enabled: bool = False,\n"
            "    ) -> bytes:\n"
        )
        new_sig = (
            "        reset: bool = True,\n"
            "        merge_enabled: bool = False,\n"
            "        payload_copy: bool = False,\n"
            "        dialogue_wrap: bool = True,\n"
            "        speech_gate: bool = True,\n"
            "    ) -> bytes:\n"
        )
        if old_sig not in text:
            raise SystemExit("generate_packets signature not found")
        text = text.replace(old_sig, new_sig, 1)
        changed = True
        print("[apply] generate_packets kwargs")

    if "speech_gate and not speech_ok" not in text:
        old_loop = (
            "        if reset:\n"
            "            self.reset_state(reset_atomizer=True)\n"
            "        prompt_packets = self.atomizer.encode(prompt, reset=reset)\n"
        )
        new_loop = (
            "        prev_payload_flag = bool(getattr(self.surface, \"payload_enabled\", True))\n"
            "        self.surface.payload_enabled = bool(payload_copy)\n"
            "        try:\n"
            "            if reset:\n"
            "                self.reset_state(reset_atomizer=True)\n"
            "            if dialogue_wrap:\n"
            "                prompt = format_dialogue_prompt(prompt)\n"
            "            prompt_packets = self.atomizer.encode(prompt, reset=reset)\n"
        )
        if old_loop not in text:
            raise SystemExit("generate reset/encode block not found")
        text = text.replace(old_loop, new_loop, 1)
        old_ok = (
            "            if not payload:\n"
            "                break\n"
            "            generated.extend(payload)\n"
        )
        new_ok = (
            "            if not payload:\n"
            "                break\n"
            "            if speech_gate and not speech_ok(payload):\n"
            "                break\n"
            "            generated.extend(payload)\n"
        )
        if old_ok not in text:
            raise SystemExit("payload extend block not found")
        text = text.replace(old_ok, new_ok, 1)
        closer = (
            "        if max_length is not None:\n"
            "            return bytes(generated[:max_length])\n"
            "        return bytes(generated)\n"
        )
        closed = (
            "            if max_length is not None:\n"
            "                return bytes(generated[:max_length])\n"
            "            return bytes(generated)\n"
            "        finally:\n"
            "            self.surface.payload_enabled = prev_payload_flag\n"
        )
        if closer not in text:
            raise SystemExit("generate return block not found")
        text = text.replace(closer, closed, 1)
        changed = True
        print("[apply] soup-gate + dialogue wrap")

    if changed:
        TARGET.write_text(text, encoding="utf-8")
        print(f"[apply] wrote {TARGET}")
    else:
        print("[apply] already applied")


if __name__ == "__main__":
    main()
