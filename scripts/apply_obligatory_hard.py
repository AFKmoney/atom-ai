#!/usr/bin/env python3
"""Apply --field-obligatory-hard mechanism to src/atom_native.py (idempotent)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "src" / "atom_native.py"

MARKER = "field_obligatory_hard"

INIT_FLAG = """        self.field_obligatory_readout = False  # enabled via AtomNativeModel flag
"""

INIT_FLAG_NEW = """        self.field_obligatory_readout = False  # enabled via AtomNativeModel flag
        self.field_obligatory_hard = False  # mix floor=1.0 + freeze non-α bypass
        self._soft_obl_mix_floor = 0.35
"""

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

FORWARD_SOFT_BLOCK = """            if self.field_obligatory_readout:
                a_only = self.alpha_only_features(alpha)
                a_rms = a_only.pow(2).mean().sqrt().clamp_min(1e-8)
                a_hat = a_only / a_rms
                # Frozen branch (CE cannot collapse) + trainable adapter.
                frozen_byte = (self.alpha_byte_frozen @ a_hat).view(
                    self.max_payload_bytes, 256
                )
                frozen_len = self.alpha_length_frozen @ a_hat
                train_byte = self.alpha_byte_proj(a_hat).view(self.max_payload_bytes, 256)
                train_len = self.alpha_length_proj(a_hat)
                mix = self.obligatory_mix_weight()
                # Hard mix uses FROZEN map so inter-prompt α structure reaches logits.
                byte_logits = (1.0 - mix) * byte_logits + mix * frozen_byte
                length_logits = (1.0 - mix) * length_logits + mix * frozen_len
                obl = self.obligatory_scale()
                byte_logits = byte_logits + obl * train_byte
                length_logits = length_logits + obl * train_len
            return {"byte_logits": byte_logits, "length_logits": length_logits}
"""

FORWARD_HARD_BLOCK = """            if self.field_obligatory_readout:
                a_only = self.alpha_only_features(alpha)
                a_rms = a_only.pow(2).mean().sqrt().clamp_min(1e-8)
                a_hat = a_only / a_rms
                # Frozen branch (CE cannot collapse) + trainable adapter.
                frozen_byte = (self.alpha_byte_frozen @ a_hat).view(
                    self.max_payload_bytes, 256
                )
                frozen_len = self.alpha_length_frozen @ a_hat
                if getattr(self, "field_obligatory_hard", False):
                    # Hard: mix≡1 and non-α bypass residual forced off.
                    return {"byte_logits": frozen_byte, "length_logits": frozen_len}
                train_byte = self.alpha_byte_proj(a_hat).view(self.max_payload_bytes, 256)
                train_len = self.alpha_length_proj(a_hat)
                mix = self.obligatory_mix_weight()
                # Soft mix uses FROZEN map so inter-prompt α structure reaches logits.
                byte_logits = (1.0 - mix) * byte_logits + mix * frozen_byte
                length_logits = (1.0 - mix) * length_logits + mix * frozen_len
                obl = self.obligatory_scale()
                byte_logits = byte_logits + obl * train_byte
                length_logits = length_logits + obl * train_len
            return {"byte_logits": byte_logits, "length_logits": length_logits}
"""

MODEL_INIT_OLD = """        field_obligatory_readout: bool = False,
        slow_rms_rel_tol: float = 0.15,
    ) -> None:
"""

MODEL_INIT_NEW = """        field_obligatory_readout: bool = False,
        field_obligatory_hard: bool = False,
        slow_rms_rel_tol: float = 0.15,
    ) -> None:
"""

MODEL_ASSIGN_OLD = """        self.field_obligatory_readout = bool(field_obligatory_readout)
        self.surface.field_obligatory_readout = self.field_obligatory_readout
        self.slow_rms_rel_tol = float(slow_rms_rel_tol)
"""

MODEL_ASSIGN_NEW = """        self.field_obligatory_hard = bool(field_obligatory_hard)
        self.field_obligatory_readout = bool(field_obligatory_readout) or self.field_obligatory_hard
        self.surface.field_obligatory_readout = self.field_obligatory_readout
        if self.field_obligatory_hard:
            self.surface.set_obligatory_hard(True)
        else:
            self.surface.field_obligatory_hard = False
        self.slow_rms_rel_tol = float(slow_rms_rel_tol)
"""


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")
    if "def set_obligatory_hard" in text and "field_obligatory_hard: bool" in text:
        print(f"already applied: {TARGET}")
        return
    if INIT_FLAG not in text:
        raise SystemExit("anchor INIT_FLAG not found")
    text = text.replace(INIT_FLAG, INIT_FLAG_NEW, 1)
    # Insert methods before field_features
    anchor = "    def field_features("
    if METHODS.strip() not in text:
        if anchor not in text:
            raise SystemExit("anchor field_features not found")
        text = text.replace(anchor, METHODS + "    def field_features(", 1)
    if FORWARD_SOFT_BLOCK in text:
        text = text.replace(FORWARD_SOFT_BLOCK, FORWARD_HARD_BLOCK, 1)
    elif "if getattr(self, \"field_obligatory_hard\"" in text:
        pass
    else:
        raise SystemExit("forward soft block not found — check atom_native.py")
    if MODEL_INIT_OLD not in text:
        raise SystemExit("model init anchor not found")
    text = text.replace(MODEL_INIT_OLD, MODEL_INIT_NEW, 1)
    if MODEL_ASSIGN_OLD not in text:
        raise SystemExit("model assign anchor not found")
    text = text.replace(MODEL_ASSIGN_OLD, MODEL_ASSIGN_NEW, 1)
    TARGET.write_text(text, encoding="utf-8")
    print(f"applied hard obligatory to {TARGET}")


if __name__ == "__main__":
    main()
