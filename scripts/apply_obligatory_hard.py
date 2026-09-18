#!/usr/bin/env python3
"""Apply --field-obligatory-hard to src/atom_native.py (idempotent)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "src" / "atom_native.py"

INIT_OLD = (
    "        self.field_obligatory_readout = False  # enabled via AtomNativeModel flag\n"
)
INIT_NEW = (
    "        self.field_obligatory_readout = False  # enabled via AtomNativeModel flag\n"
    "        self.field_obligatory_hard = False  # mix floor=1.0 + freeze non-α bypass\n"
    "        self._soft_obl_mix_floor = 0.35\n"
)

METHODS = '''
    def set_obligatory_hard(self, enabled: bool) -> None:
        """Hard obligatory: mix floor 1.0; zero/freeze trainable non-α bypass."""
        enabled = bool(enabled)
        self.field_obligatory_hard = enabled
        if enabled:
            self.field_obligatory_readout = True
            self.obl_mix_floor = 1.0
            self._freeze_non_alpha_bypass(True)
        else:
            self.obl_mix_floor = float(getattr(self, "_soft_obl_mix_floor", 0.35))
            self._freeze_non_alpha_bypass(False)

    def _freeze_non_alpha_bypass(self, freeze: bool) -> None:
        """Zero + freeze decoder/skip/gates that can bypass α on the CE path."""
        modules = (
            self.field_to_state,
            self.field_byte_skip,
            self.field_length_skip,
            self.byte_decoder,
            self.length_decoder,
        )
        scalars = (self.field_gate, self.skip_gate, self.obl_mix)
        with torch.no_grad():
            if freeze:
                for module in modules:
                    for param in module.parameters():
                        param.zero_()
                        param.requires_grad_(False)
                for param in scalars:
                    param.zero_()
                    param.requires_grad_(False)
            else:
                for module in modules:
                    for param in module.parameters():
                        param.requires_grad_(True)
                for param in scalars:
                    param.requires_grad_(True)

'''

FORWARD_NEEDLE = (
    "                train_byte = self.alpha_byte_proj(a_hat).view(self.max_payload_bytes, 256)\n"
    "                train_len = self.alpha_length_proj(a_hat)\n"
    "                mix = self.obligatory_mix_weight()\n"
)

FORWARD_INSERT = (
    "                if getattr(self, \"field_obligatory_hard\", False):\n"
    "                    # Hard: mix≡1 and non-α bypass residual forced off.\n"
    "                    return {\"byte_logits\": frozen_byte, \"length_logits\": frozen_len}\n"
    "                train_byte = self.alpha_byte_proj(a_hat).view(self.max_payload_bytes, 256)\n"
    "                train_len = self.alpha_length_proj(a_hat)\n"
    "                mix = self.obligatory_mix_weight()\n"
)

MODEL_INIT_OLD = (
    "        field_obligatory_readout: bool = False,\n"
    "        slow_rms_rel_tol: float = 0.15,\n"
)
MODEL_INIT_NEW = (
    "        field_obligatory_readout: bool = False,\n"
    "        field_obligatory_hard: bool = False,\n"
    "        slow_rms_rel_tol: float = 0.15,\n"
)

MODEL_ASSIGN_OLD = (
    "        self.field_obligatory_readout = bool(field_obligatory_readout)\n"
    "        self.surface.field_obligatory_readout = self.field_obligatory_readout\n"
)
MODEL_ASSIGN_NEW = (
    "        self.field_obligatory_hard = bool(field_obligatory_hard)\n"
    "        self.field_obligatory_readout = bool(field_obligatory_readout) or self.field_obligatory_hard\n"
    "        self.surface.field_obligatory_readout = self.field_obligatory_readout\n"
    "        if self.field_obligatory_hard:\n"
    "            self.surface.set_obligatory_hard(True)\n"
    "        else:\n"
    "            self.surface.field_obligatory_hard = False\n"
)


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")
    if "def set_obligatory_hard" in text and "field_obligatory_hard: bool" in text:
        print(f"already applied: {TARGET}")
        return
    if INIT_OLD not in text:
        raise SystemExit("INIT_OLD anchor not found")
    text = text.replace(INIT_OLD, INIT_NEW, 1)
    if "def set_obligatory_hard" not in text:
        anchor = "    def field_features("
        if anchor not in text:
            raise SystemExit("field_features anchor not found")
        text = text.replace(anchor, METHODS + anchor, 1)
    if "field_obligatory_hard" in text and "return {\"byte_logits\": frozen_byte" in text:
        pass
    elif FORWARD_NEEDLE not in text:
        raise SystemExit("FORWARD_NEEDLE not found")
    else:
        text = text.replace(FORWARD_NEEDLE, FORWARD_INSERT, 1)
    if MODEL_INIT_OLD not in text:
        raise SystemExit("MODEL_INIT_OLD not found")
    text = text.replace(MODEL_INIT_OLD, MODEL_INIT_NEW, 1)
    if MODEL_ASSIGN_OLD not in text:
        raise SystemExit("MODEL_ASSIGN_OLD not found")
    text = text.replace(MODEL_ASSIGN_OLD, MODEL_ASSIGN_NEW, 1)
    TARGET.write_text(text, encoding="utf-8")
    print(f"applied hard obligatory to {TARGET}")


if __name__ == "__main__":
    main()
