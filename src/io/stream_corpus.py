"""Streaming full-dataset ingest for ATOM-native training.

Resolve one or more shard files, read them in UTF-8-safe byte chunks, atomize
online with ``Atomizer.encode_bytes(..., flush=False)`` (flush at shard ends),
and yield consecutive ``(current, target)`` packet pairs without ever materializing
the full packet list in RAM.

Capacity thesis: ATOM targets GPT-3 *capability* via persistent structured matter
while streaming complete corpora on a single box — not by matching 175B params.
"""

from __future__ import annotations

from collections import deque
from glob import glob
from pathlib import Path
from typing import Deque, Iterator, Sequence

from src.io.atomizer import AtomPacket, Atomizer


def resolve_shard_paths(
    data: str | Path | None = None,
    data_glob: str | None = None,
    *,
    root: str | Path | None = None,
) -> list[Path]:
    """Resolve ``--data`` and/or ``--data-glob`` into an ordered list of shards.

    Explicit ``data`` paths come first (deduped), then glob matches sorted
    lexicographically for stable epoch order.
    """
    base = Path(root) if root is not None else Path.cwd()
    paths: list[Path] = []
    seen: set[Path] = set()

    def _add(path: Path) -> None:
        resolved = path if path.is_absolute() else (base / path)
        resolved = resolved.resolve()
        if resolved in seen:
            return
        if not resolved.is_file():
            raise FileNotFoundError(f"shard not found: {resolved}")
        seen.add(resolved)
        paths.append(resolved)

    if data:
        _add(Path(data))
    if data_glob:
        pattern = data_glob
        matches = sorted(Path(p).resolve() for p in glob(pattern, root_dir=str(base)))
        if not matches:
            # Also try the pattern as absolute / cwd-relative without root_dir
            matches = sorted(Path(p).resolve() for p in glob(pattern))
        if not matches:
            raise FileNotFoundError(f"no shards matched glob: {data_glob!r}")
        for match in matches:
            if match in seen:
                continue
            if not match.is_file():
                continue
            seen.add(match)
            paths.append(match)

    if not paths:
        raise ValueError("provide --data and/or --data-glob resolving to at least one file")
    return paths


def _utf8_safe_split(buffer: bytes) -> tuple[bytes, bytes]:
    """Split ``buffer`` into (emit, hold) so hold has no incomplete UTF-8 tail."""
    if not buffer:
        return b"", b""
    # Walk back over continuation bytes / incomplete lead.
    i = len(buffer)
    while i > 0 and 0x80 <= buffer[i - 1] <= 0xBF:
        i -= 1
    if i == 0:
        # Entire buffer is continuations — hold everything until more arrives.
        return b"", buffer
    lead = buffer[i - 1]
    if lead < 0x80:
        return buffer, b""
    if 0xC2 <= lead <= 0xDF:
        need = 2
    elif 0xE0 <= lead <= 0xEF:
        need = 3
    elif 0xF0 <= lead <= 0xF4:
        need = 4
    else:
        # Invalid lead; emit as-is (atomizer is byte-level).
        return buffer, b""
    have = len(buffer) - (i - 1)
    if have < need:
        return buffer[: i - 1], buffer[i - 1 :]
    return buffer, b""


def iter_byte_chunks(
    paths: Sequence[str | Path],
    chunk_bytes: int = 1 << 20,
) -> Iterator[tuple[int, bytes, bool]]:
    """Yield ``(shard_index, chunk, shard_ended)`` UTF-8-safe byte chunks.

    Chunks never split mid-codepoint.  When a shard ends, the final yield for
    that shard has ``shard_ended=True`` (possibly with empty ``chunk`` if only
    a flush is needed after a prior partial).
    """
    if chunk_bytes < 1:
        raise ValueError("chunk_bytes must be >= 1")
    path_list = [Path(p) for p in paths]
    for shard_index, path in enumerate(path_list):
        hold = b""
        emitted_any = False
        with path.open("rb") as handle:
            while True:
                block = handle.read(chunk_bytes)
                if not block:
                    break
                combined = hold + block
                emit, hold = _utf8_safe_split(combined)
                if emit:
                    emitted_any = True
                    yield shard_index, emit, False
        if hold:
            emitted_any = True
            yield shard_index, hold, True
        elif emitted_any:
            yield shard_index, b"", True
        else:
            # Empty shard: still signal end so callers can flush atomizer state.
            yield shard_index, b"", True


class StreamingPacketSource:
    """Online ``(current, target)`` transitions from corpus shards.

    Parameters
    ----------
    atomizer:
        Stateful ``Atomizer`` used for incremental encode (owned by caller;
        this source does not deep-copy it).
    paths:
        Ordered shard paths.
    chunk_bytes:
        Read size before UTF-8-safe trimming (default 1 MiB).
    loop:
        If True, restart from the first shard forever (infinite continuum).
    buffer_size:
        Bounded ring of recent packets for validation samples (default 2048).
    """

    def __init__(
        self,
        atomizer: Atomizer,
        paths: Sequence[str | Path],
        *,
        chunk_bytes: int = 1 << 20,
        loop: bool = True,
        buffer_size: int = 2048,
        reset_atomizer: bool = True,
    ) -> None:
        if not paths:
            raise ValueError("StreamingPacketSource requires at least one shard path")
        if buffer_size < 2:
            raise ValueError("buffer_size must be >= 2 to hold a transition pair")
        self.atomizer = atomizer
        self.paths = [Path(p) for p in paths]
        self.chunk_bytes = int(chunk_bytes)
        self.loop = bool(loop)
        self.buffer_size = int(buffer_size)
        self.bytes_seen = 0
        self.packets_seen = 0
        self.shard_index = 0
        self.epoch = 0
        self._ring: Deque[AtomPacket] = deque(maxlen=self.buffer_size)
        self._prev: AtomPacket | None = None
        self._pending: Deque[tuple[AtomPacket, AtomPacket]] = deque()
        self._exhausted = False
        if reset_atomizer:
            self.atomizer.reset()

    def stats(self) -> dict:
        return {
            "bytes_seen": self.bytes_seen,
            "packets_seen": self.packets_seen,
            "shard_index": self.shard_index,
            "epoch": self.epoch,
            "ring_size": len(self._ring),
            "n_shards": len(self.paths),
            "loop": self.loop,
            "chunk_bytes": self.chunk_bytes,
            "paths": [str(p) for p in self.paths],
        }

    def validation_packets(self) -> list[AtomPacket]:
        """Snapshot of the ring buffer (oldest → newest)."""
        return list(self._ring)

    def validation_transitions(self) -> list[tuple[AtomPacket, AtomPacket]]:
        packets = self.validation_packets()
        if len(packets) < 2:
            return []
        return list(zip(packets[:-1], packets[1:]))

    def _ingest_packets(self, packets: list[AtomPacket]) -> None:
        for packet in packets:
            self.packets_seen += 1
            self._ring.append(packet)
            if self._prev is not None:
                self._pending.append((self._prev, packet))
            self._prev = packet

    def _feed_chunk(self, chunk: bytes, *, flush: bool) -> None:
        if chunk:
            self.bytes_seen += len(chunk)
            packets = self.atomizer.encode_bytes(chunk, reset=False, flush=False)
            self._ingest_packets(packets)
        if flush:
            # Empty encode with flush=True drains the atomizer pending span.
            flushed = self.atomizer.encode_bytes(b"", reset=False, flush=True)
            self._ingest_packets(flushed)

    def _chunk_stream(self) -> Iterator[tuple[int, bytes, bool]]:
        while True:
            for shard_index, chunk, shard_ended in iter_byte_chunks(
                self.paths, chunk_bytes=self.chunk_bytes
            ):
                yield shard_index, chunk, shard_ended
            if not self.loop:
                return
            self.epoch += 1

    def __iter__(self) -> Iterator[tuple[AtomPacket, AtomPacket]]:
        return self

    def __next__(self) -> tuple[AtomPacket, AtomPacket]:
        if self._pending:
            return self._pending.popleft()
        if self._exhausted:
            raise StopIteration
        if not hasattr(self, "_chunk_iter"):
            self._chunk_iter = self._chunk_stream()
        while not self._pending:
            try:
                shard_index, chunk, shard_ended = next(self._chunk_iter)
            except StopIteration:
                self._exhausted = True
                raise
            self.shard_index = shard_index
            self._feed_chunk(chunk, flush=shard_ended)
        return self._pending.popleft()

    def next_transition(self) -> tuple[AtomPacket, AtomPacket]:
        """Pull the next ``(current, target)`` pair (same as ``next(self)``)."""
        return next(self)
