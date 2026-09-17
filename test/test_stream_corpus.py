"""Tests for streaming full-dataset ingest."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.io.atomizer import Atomizer
from src.io.stream_corpus import (
    StreamingPacketSource,
    iter_byte_chunks,
    resolve_shard_paths,
)


class StreamCorpusTests(unittest.TestCase):
    def _write_shards(self, root: Path) -> list[Path]:
        shards = [
            root / "a.txt",
            root / "b.txt",
            root / "c.txt",
        ]
        shards[0].write_text("alpha β\n", encoding="utf-8")
        shards[1].write_text("gamma — delta ", encoding="utf-8")
        shards[2].write_text("epsilon\nfin.", encoding="utf-8")
        return shards

    def test_resolve_data_and_glob(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shards = self._write_shards(root)
            paths = resolve_shard_paths(None, str(root / "*.txt"), root=root)
            self.assertEqual(paths, sorted(shards))
            single = resolve_shard_paths(shards[1], None, root=root)
            self.assertEqual(single, [shards[1].resolve()])

    def test_iter_byte_chunks_utf8_safe_and_concat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shards = self._write_shards(root)
            source = b"".join(p.read_bytes() for p in shards)
            assembled = bytearray()
            for _idx, chunk, _ended in iter_byte_chunks(shards, chunk_bytes=3):
                assembled.extend(chunk)
            self.assertEqual(bytes(assembled), source)
            # Multi-byte char β (C3 B2) must not be split mid-codepoint across yields
            # when chunk_bytes is small: every non-empty chunk must be valid UTF-8
            # or empty (flush markers).
            for _idx, chunk, _ended in iter_byte_chunks(shards, chunk_bytes=1):
                if chunk:
                    chunk.decode("utf-8")  # raises if mid-codepoint split leaked

    def test_streaming_pairs_lossless_payload_concat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shards = self._write_shards(root)
            source = b"".join(p.read_bytes() for p in shards)
            atomizer = Atomizer(max_span_bytes=8)
            stream = StreamingPacketSource(
                atomizer,
                shards,
                chunk_bytes=4,
                loop=False,
                buffer_size=64,
            )
            payloads = []
            pairs = list(stream)
            self.assertGreater(len(pairs), 0)
            # First current + all targets reconstruct the packet stream order.
            payloads.append(pairs[0][0].payload)
            for _cur, tgt in pairs:
                payloads.append(tgt.payload)
            self.assertEqual(b"".join(payloads), source)
            self.assertEqual(stream.bytes_seen, len(source))
            self.assertGreater(stream.packets_seen, 1)

    def test_stream_source_does_not_require_full_encode_list_api(self) -> None:
        """Stream path must work via encode_bytes(flush=False), not a giant list."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shards = self._write_shards(root)
            atomizer = Atomizer(max_span_bytes=8)
            calls: list[tuple[int, bool]] = []
            original = atomizer.encode_bytes

            def tracked(raw, reset=True, flush=True):
                calls.append((len(raw), bool(flush)))
                return original(raw, reset=reset, flush=flush)

            atomizer.encode_bytes = tracked  # type: ignore[method-assign]
            stream = StreamingPacketSource(
                atomizer,
                shards,
                chunk_bytes=5,
                loop=False,
                buffer_size=32,
            )
            _ = list(stream)
            # At least one call used flush=False (incremental), and we never
            # passed the entire corpus as a single encode_bytes argument.
            self.assertTrue(any(not flush for _n, flush in calls))
            total_source = sum(p.stat().st_size for p in shards)
            self.assertTrue(all(n < total_source or flush for n, flush in calls))

    def test_loop_shards_repeats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shards = self._write_shards(root)
            stream = StreamingPacketSource(
                Atomizer(max_span_bytes=8),
                shards,
                chunk_bytes=8,
                loop=True,
                buffer_size=128,
            )
            # Pull more transitions than one epoch can provide.
            pairs = [stream.next_transition() for _ in range(80)]
            self.assertEqual(len(pairs), 80)
            self.assertGreaterEqual(stream.epoch, 1)
            self.assertGreater(stream.bytes_seen, sum(p.stat().st_size for p in shards))

    def test_ring_buffer_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shards = self._write_shards(root)
            stream = StreamingPacketSource(
                Atomizer(max_span_bytes=4),
                shards,
                chunk_bytes=8,
                loop=True,
                buffer_size=8,
            )
            for _ in range(40):
                stream.next_transition()
            self.assertLessEqual(len(stream.validation_packets()), 8)


if __name__ == "__main__":
    unittest.main()
