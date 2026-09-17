"""Input adapters for atom-native byte streams and streaming corpus ingest."""

from .atomizer import AtomPacket, Atomizer
from .stream_corpus import StreamingPacketSource, iter_byte_chunks, resolve_shard_paths

__all__ = [
    "AtomPacket",
    "Atomizer",
    "StreamingPacketSource",
    "iter_byte_chunks",
    "resolve_shard_paths",
]
