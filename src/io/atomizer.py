"""ATOM-native streaming atomization.

This module deliberately does not build a vocabulary and does not assign token
IDs.  Raw UTF-8 bytes are treated as observations.  The atomizer groups the
observations into reversible spans and emits an ``AtomPacket`` containing the
surface bytes plus deterministic local/contextual features.  A downstream
ATOM compiler turns that packet into the eight toroidal atom properties.

The important distinction from the legacy GPT-2 path is:

    raw stream + local buffer + current context -> packet -> toroidal atom

The packet is an event, not a permanent vocabulary entry.  Two equal surface
spans can have different features when their preceding stream context differs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import string
from typing import Iterable, Iterator

import torch


@dataclass(frozen=True)
class AtomPacket:
    """One reversible, context-conditioned input event for ATOM.

    ``payload`` is kept as bytes so that round-tripping does not depend on a
    lossy Unicode replacement.  ``features`` is a fixed-size observation used
    by the token-to-atom compiler; it is not a token ID or a vocabulary row.
    """

    payload: bytes
    start: int
    end: int
    boundary: str
    level: int
    phase: float
    duration: float
    features: torch.Tensor

    @property
    def length(self) -> int:
        return len(self.payload)

    @property
    def text(self) -> str:
        return self.payload.decode("utf-8", errors="replace")

    def to(self, device: str | torch.device) -> "AtomPacket":
        return AtomPacket(
            payload=self.payload,
            start=self.start,
            end=self.end,
            boundary=self.boundary,
            level=self.level,
            phase=self.phase,
            duration=self.duration,
            features=self.features.to(device),
        )


class Atomizer:
    """A small stateful compiler from a byte stream to atom packets.

    Segmentation is online and reversible.  It uses structural boundaries
    (whitespace, punctuation, line breaks and bounded span duration), not a
    learned BPE vocabulary.  The packet features contain a byte observation,
    span statistics and a rolling context sketch.  The latter makes the same
    surface span context-sensitive without introducing token IDs.

    ``max_span_bytes`` is a safety bound, not a vocabulary merge size.  The
    boundary is always emitted at the end of the stream, so every byte is
    represented exactly once.
    """

    VERSION = "atomizer-v1-byte-span"
    FEATURE_DIM = 320
    _PUNCTUATION = set(bytes(string.punctuation, "ascii"))
    _BOUNDARY_CODES = {
        "whitespace": 1,
        "punctuation": 2,
        "newline": 3,
        "class_transition": 4,
        "max_span": 5,
        "eos": 6,
        "generated": 7,
    }

    def __init__(self, max_span_bytes: int = 32, phase_period: float = 256.0) -> None:
        if max_span_bytes < 2:
            raise ValueError("max_span_bytes must be at least 2")
        if phase_period <= 0:
            raise ValueError("phase_period must be positive")
        self.max_span_bytes = int(max_span_bytes)
        self.phase_period = float(phase_period)
        self.reset()

    @property
    def feature_dim(self) -> int:
        return self.FEATURE_DIM

    def reset(self) -> None:
        """Reset only the streaming context, not the configuration."""
        self.position = 0
        self.previous_length = 0
        self.context_group = torch.zeros(16, dtype=torch.float32)
        self.packet_count = 0
        self._buffer = bytearray()
        self._buffer_start = 0

    @staticmethod
    def _is_continuation_byte(value: int) -> bool:
        return 0x80 <= value <= 0xBF

    @classmethod
    def _ends_at_utf8_boundary(cls, buffer: bytearray) -> bool:
        """Return whether a max-span cut would preserve UTF-8 code points."""
        if not buffer:
            return True
        last = buffer[-1]
        if last < 0x80:
            return True
        if cls._is_continuation_byte(last):
            continuation_count = 0
            for value in reversed(buffer):
                if not cls._is_continuation_byte(value):
                    break
                continuation_count += 1
            if continuation_count >= len(buffer):
                return False
            lead = buffer[-continuation_count - 1]
            if 0xC2 <= lead <= 0xDF:
                expected = 1
            elif 0xE0 <= lead <= 0xEF:
                expected = 2
            elif 0xF0 <= lead <= 0xF4:
                expected = 3
            else:
                return True
            return continuation_count >= expected
        # A leading multi-byte byte at the end is incomplete.
        return not (0xC2 <= last <= 0xF4)

    @staticmethod
    def _class(value: int) -> str:
        if value in (9, 10, 13, 32):
            return "space"
        if 48 <= value <= 57:
            return "digit"
        if 65 <= value <= 90 or 97 <= value <= 122:
            return "alpha"
        if value < 32:
            return "control"
        if value in Atomizer._PUNCTUATION:
            return "punctuation"
        if Atomizer._is_continuation_byte(value):
            return "continuation"
        return "other"

    @classmethod
    def _boundary_reason(cls, buffer: bytearray) -> str | None:
        if not buffer:
            return None
        current = buffer[-1]
        if current == 10 or current == 13:
            return "newline"
        if current in cls._PUNCTUATION:
            return "punctuation"
        if current in (9, 32) and any(byte not in (9, 10, 13, 32) for byte in buffer[:-1]):
            return "whitespace"
        # A structural switch is useful for mixed identifiers/numbers, but do
        # not split UTF-8 continuation bytes in the middle of a code point.
        if len(buffer) >= 2 and not cls._is_continuation_byte(current):
            previous = buffer[-2]
            previous_class = cls._class(previous)
            current_class = cls._class(current)
            if previous_class != current_class and {previous_class, current_class} <= {"alpha", "digit"}:
                return "class_transition"
        return None

    @staticmethod
    def _entropy(hist: torch.Tensor) -> float:
        positive = hist[hist > 0]
        if positive.numel() == 0:
            return 0.0
        return float(-(positive * positive.log2()).sum().item() / 8.0)

    @staticmethod
    def _safe_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
        denom = float(left.norm().item() * right.norm().item())
        if denom <= 1e-12:
            return 0.0
        return float(torch.dot(left, right).item() / denom)

    def _make_features(self, payload: bytes, start: int, boundary: str, level: int) -> torch.Tensor:
        values = torch.tensor(list(payload), dtype=torch.long)
        hist = torch.bincount(values, minlength=256).float()
        hist = hist / max(float(len(payload)), 1.0)
        # Keep the low-nibble groups for the rolling context sketch.  Unlike
        # a high-nibble grouping, this distinguishes nearby ASCII symbols such
        # as ``a`` and ``b`` while remaining only 16 values wide.
        grouped = hist.reshape(16, 16).sum(dim=0)

        first = torch.zeros(16, dtype=torch.float32)
        last = torch.zeros(16, dtype=torch.float32)
        first[payload[0] >> 4] = 1.0
        last[payload[-1] >> 4] = 1.0

        whitespace = sum(value in (9, 10, 13, 32) for value in payload) / len(payload)
        digits = sum(48 <= value <= 57 for value in payload) / len(payload)
        alpha = sum(65 <= value <= 90 or 97 <= value <= 122 for value in payload) / len(payload)
        punctuation = sum(value in self._PUNCTUATION for value in payload) / len(payload)
        control = sum(value < 32 for value in payload) / len(payload)
        transitions = sum(
            self._class(left) != self._class(right)
            for left, right in zip(payload, payload[1:])
        ) / max(len(payload) - 1, 1)
        phase = 2.0 * math.pi * (start % self.phase_period) / self.phase_period
        context_similarity = self._safe_cosine(grouped, self.context_group)
        boundary_code = self._BOUNDARY_CODES.get(boundary, 0)
        stats = torch.tensor(
            [
                len(payload) / self.max_span_bytes,
                whitespace,
                digits,
                alpha,
                punctuation,
                control,
                self._entropy(hist),
                transitions,
                payload[0] / 255.0,
                payload[-1] / 255.0,
                math.sin(phase),
                math.cos(phase),
                self.previous_length / self.max_span_bytes,
                (context_similarity + 1.0) / 2.0,
                boundary_code / max(self._BOUNDARY_CODES.values()),
                level / 3.0,
            ],
            dtype=torch.float32,
        )
        # 256 byte observations + first/last nibbles + 16 scalar observables +
        # 16 rolling context values = 320 features.
        return torch.cat((hist, first, last, stats, self.context_group), dim=0)

    def _level_for(self, boundary: str, payload: bytes) -> int:
        if boundary == "newline" or b"\n" in payload:
            return 3
        if boundary == "punctuation" or any(byte in self._PUNCTUATION for byte in payload):
            return 2
        return 1

    def _emit(self, payload: bytes, start: int, boundary: str) -> AtomPacket:
        level = self._level_for(boundary, payload)
        phase = 2.0 * math.pi * (start % self.phase_period) / self.phase_period
        features = self._make_features(payload, start, boundary, level)
        packet = AtomPacket(
            payload=payload,
            start=start,
            end=start + len(payload),
            boundary=boundary,
            level=level,
            phase=phase,
            duration=float(len(payload)),
            features=features,
        )
        grouped = features[:256].reshape(16, 16).sum(dim=0)
        self.context_group.mul_(0.85).add_(0.15 * grouped)
        self.previous_length = len(payload)
        self.position = packet.end
        self.packet_count += 1
        return packet

    def _flush_pending(self, boundary: str = "eos") -> list[AtomPacket]:
        if not self._buffer:
            return []
        packet = self._emit(bytes(self._buffer), self._buffer_start, boundary)
        self._buffer.clear()
        self._buffer_start = self.position
        return [packet]

    def encode_bytes(
        self,
        raw: bytes,
        reset: bool = True,
        flush: bool = True,
    ) -> list[AtomPacket]:
        """Atomize bytes and guarantee exact byte reconstruction.

        ``flush=False`` keeps a partial UTF-8/span buffer for the next call;
        this is what makes ``stream`` genuinely incremental.
        """
        if reset:
            self.reset()
        packets: list[AtomPacket] = []
        for value in raw:
            if not self._buffer:
                self._buffer_start = self.position
            self._buffer.append(value)
            reason = self._boundary_reason(self._buffer)
            if reason is not None:
                packets.extend(self._flush_pending(reason))
            elif len(self._buffer) >= self.max_span_bytes and self._ends_at_utf8_boundary(self._buffer):
                packets.extend(self._flush_pending("max_span"))
        if flush:
            packets.extend(self._flush_pending("eos"))
        return packets

    def encode(self, text: str, reset: bool = True) -> list[AtomPacket]:
        return self.encode_bytes(text.encode("utf-8"), reset=reset, flush=True)

    def packet_from_payload(self, payload: bytes, boundary: str = "generated") -> AtomPacket:
        """Commit a predicted surface packet to the atomizer context."""
        if not payload:
            raise ValueError("generated atom payload cannot be empty")
        if self._buffer:
            raise ValueError("cannot inject a generated packet while a raw span is pending")
        return self._emit(bytes(payload), self.position, boundary)

    def state_dict(self) -> dict:
        return {
            "version": self.VERSION,
            "max_span_bytes": self.max_span_bytes,
            "phase_period": self.phase_period,
            "position": self.position,
            "previous_length": self.previous_length,
            "context_group": self.context_group.clone(),
            "packet_count": self.packet_count,
            "buffer": bytes(self._buffer),
            "buffer_start": self._buffer_start,
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("version", self.VERSION) != self.VERSION:
            raise ValueError(f"unsupported atomizer version: {state.get('version')}")
        self.position = int(state.get("position", 0))
        self.previous_length = int(state.get("previous_length", 0))
        self.context_group = state.get("context_group", torch.zeros(16)).clone().float()
        self.packet_count = int(state.get("packet_count", 0))
        self._buffer = bytearray(state.get("buffer", b""))
        self._buffer_start = int(state.get("buffer_start", self.position))

    def __call__(self, text: str) -> list[AtomPacket]:
        return self.encode(text)

    def stream(self, chunks: Iterable[str]) -> Iterator[AtomPacket]:
        """Atomize consecutive chunks without resetting the stream context."""
        first = True
        for chunk in chunks:
            for packet in self.encode_bytes(
                chunk.encode("utf-8"),
                reset=first,
                flush=False,
            ):
                first = False
                yield packet
        for packet in self._flush_pending("eos"):
            yield packet
